import os
import re
import random
import itertools
import threading
import time
from queue import Queue
import logging

import torch
import numpy as np
import torch.distributed as dist

from Utils.Quaternion import quat_to_angvel
from Utils.Protobuf.ProtobufLoader import load_protobuf_data


class DatasetManager:
    def __init__(self, batch_size, device, shuffle=True, drop_last=True):
        import os
        import re
        import random
        import itertools
        import threading
        import time
        from queue import Queue
        import logging

        import torch
        import numpy as np
        import torch.distributed as dist

        from Utils.Quaternion import quat_to_angvel
        from Utils.Protobuf.ProtobufLoader import load_protobuf_data

        class DatasetManager:
            def __init__(self, batch_size, device, shuffle=True, drop_last=True):
                self.train_data_pool = None
                self.data_window_indices_for_train = None
                self.data_window_indices_for_validation = None
                self.__iter_mode = 'train'

                self.is_distributed_train = False
                if device == 'cuda-dist':
                    self.is_distributed_train = True
                    self.world_size = dist.get_world_size()
                    self.rank = dist.get_rank()
                    device = f'cuda:{self.rank}'
                    if self.rank == 0:
                        logging.info(f"|device: 'cuda-dist'| Distributed training mode of {self.world_size} devices set.")
                        self.seed = int(time.time())
                        random.seed(self.seed)
                        dist.broadcast(torch.tensor(self.seed, device=device, dtype=torch.int), src=0)
                    else:
                        seed = torch.empty(1, dtype=torch.int).to(device)
                        dist.broadcast(seed, src=0)
                        self.seed = seed.item()
                        random.seed(self.seed)

                    if batch_size % self.world_size != 0:
                        raise ValueError("Batch size should be divisible by "
                                         "the number of gpus(processes) for distributed training.")
                    if not drop_last:
                        raise ValueError("drop_last should be True for distributed training.")

                    logging.info(f"Rank {self.rank} | World size: {self.world_size} | Device: {device} | Seed: {self.seed}")

                self.device = device

                self.batch_size = batch_size
                self.shuffle = shuffle
                self.drop_last = drop_last
                self.current_index = 0

                self.loading_animation = itertools.cycle(['-', '/', '|', '\\'])

            def __len__(self):
                self.check_indices_made()

                if self.__iter_mode == 'train':
                    if self.drop_last:
                        return len(self.data_window_indices_for_train) // self.batch_size
                    else:
                        return int(np.ceil(len(self.data_window_indices_for_train) / self.batch_size))
                elif self.__iter_mode == 'validation':
                    if self.drop_last:
                        return len(self.data_window_indices_for_validation) // self.batch_size
                    else:
                        return int(np.ceil(len(self.data_window_indices_for_validation) / self.batch_size))
                else:
                    raise ValueError(f"Invalid iter mode: {self.__iter_mode}")

            def load_protobuf_data_threaded(self, candidate_data_queue, loaded_data_queue, total_files,
                                            seq_len=None, pb_frame_skip_step=None,
                                            forecast_frame_len=None, forecast_frame_step=None):
                def prepend_values_recursive(data_dict, prepend_length):
                    for key in data_dict.keys():
                        if isinstance(data_dict[key], dict):  # Nested dictionary
                            prepend_values_recursive(data_dict[key], prepend_length)
                        elif isinstance(data_dict[key], torch.Tensor):
                            prepend_shape = [1] * len(data_dict[key].shape)
                            prepend_shape[0] = prepend_length
                            prepend_value = data_dict[key][0].unsqueeze(0).repeat(*prepend_shape)
                            data_dict[key] = torch.cat((prepend_value, data_dict[key]), dim=0)
                        elif (isinstance(data_dict[key], list) and len(data_dict[key]) > 0
                              and isinstance(data_dict[key][0], torch.Tensor)):
                            for i in range(len(data_dict[key])):
                                prepend_shape = [1] * len(data_dict[key][i].shape)
                                prepend_shape[0] = prepend_length
                                prepend_value = data_dict[key][i][0].unsqueeze(0).repeat(*prepend_shape)
                                data_dict[key][i] = torch.cat((prepend_value, data_dict[key][i]), dim=0)

                while not candidate_data_queue.empty():
                    item = candidate_data_queue.get()
                    data_path = item['data_path']
                    data_file_name = item['data_file_name']

                    loading_char = next(self.loading_animation)
                    print(f"\r{loading_char} {total_files - candidate_data_queue.qsize()}/{total_files} "
                          f"| Loading file : {data_file_name}", end="")

                    # Check frame length exceeds the required sequence length based on file name.
                    is_frame_num_match = re.search(r"_(\d+)_(\d+).pb$", data_file_name)
                    if is_frame_num_match:
                        start_frame, end_frame = map(int, is_frame_num_match.groups())
                        frame_count = end_frame - start_frame + 1
                        if (frame_count - 1) // pb_frame_skip_step < forecast_frame_len * forecast_frame_step:
                            loaded_data_queue.put(None)
                            logging.warning(f"\r{loaded_data_queue.qsize()}/{total_files} "
                                            f"| Skip loading. {data_file_name} has less frames than the required sequence length."
                                            f"                                                   ")
                            continue
                    loaded_data = load_protobuf_data(data_path)

                    # Check once more with real data length.
                    if (loaded_data['nFrames'] - 1) // pb_frame_skip_step < forecast_frame_len * forecast_frame_step:
                        loaded_data_queue.put(None)
                        logging.warning(f"\r{loaded_data_queue.qsize()}/{total_files} "
                                        f"| Skip loading. {data_file_name} has less frames than the required sequence length."
                                        f"                                                   ")
                        continue

                    train_data = {
                        "file_name": data_file_name,
                        "nFrames": (loaded_data['nFrames'] - 1) // pb_frame_skip_step,
                        "nJoints": loaded_data['nJoints'],
                        "scale": loaded_data['scale'],
                        "nVertices": loaded_data['nVertices'],
                        "wrist": {
                            "position": torch.from_numpy(
                                loaded_data['wristPosition'][pb_frame_skip_step::pb_frame_skip_step]).float().to(self.device),
                            "orientation": torch.from_numpy(
                                loaded_data['wristRotation'][pb_frame_skip_step::pb_frame_skip_step]).float().to(self.device),
                            "linear_velocity": torch.from_numpy(
                                np.diff(loaded_data['wristPosition'][::pb_frame_skip_step], axis=0)).float().to(self.device),
                            "angular_velocity":
                                torch.from_numpy(quat_to_angvel(loaded_data['wristRotation'][::pb_frame_skip_step])).float().to(
                                    self.device),
                        },
                        "joints": {
                            "position": torch.from_numpy(
                                loaded_data['handJointPosition'][pb_frame_skip_step::pb_frame_skip_step]).float().to(
                                self.device),  # Exclude wrist position
                            "dof": torch.from_numpy(loaded_data['handDOF'][pb_frame_skip_step::pb_frame_skip_step]).float().to(
                                self.device),
                            "linear_velocity": torch.from_numpy(
                                np.diff(loaded_data['handJointPosition'][::pb_frame_skip_step], axis=0)).float().to(
                                self.device),
                            "dof_velocity": torch.from_numpy(
                                np.diff(loaded_data['handDOF'][::pb_frame_skip_step], axis=0)).float().to(self.device)
                        },
                        "gvs": torch.from_numpy(loaded_data['gvs'][pb_frame_skip_step::pb_frame_skip_step]).float().to(
                            self.device),
                        "contactLabel": torch.from_numpy(
                            loaded_data['contactLabel'][pb_frame_skip_step::pb_frame_skip_step]).float().to(self.device)
                    }

                    lvs_list = []
                    for ldx in range(len(loaded_data['lvs'][0])):
                        lvs_list.append(
                            torch.stack([
                                torch.from_numpy(frame[ldx]).float().to(self.device) for frame in
                                loaded_data['lvs'][pb_frame_skip_step::pb_frame_skip_step]
                            ]))
                    train_data["lvs"] = lvs_list

                    prepend_values_recursive(train_data, seq_len - 1)
                    train_data['nFrames'] += seq_len - 1

                    loaded_data_queue.put({'loaded_data': train_data, 'data_file_name': data_file_name})
                    logging.info(f"\r{loaded_data_queue.qsize()}/{total_files} | {data_file_name} loaded."
                                 f"                                                   ")

            def load_dataset(self, dir_path, load_data_recursive=False,
                             seq_len=None, seq_split_window_step=None, pb_frame_skip_step=None,
                             forecast_frame_len=None, forecast_frame_step=None,
                             make_indices_for_train=True, validation_split_ratio=0):
                if not self.is_distributed_train or self.rank == 0:
                    logging.info(f"Loading dataset from {dir_path}")
                if make_indices_for_train and (seq_len is None or
                                               seq_split_window_step is None or
                                               pb_frame_skip_step is None or
                                               forecast_frame_len is None or
                                               forecast_frame_step is None):
                    raise ValueError("seq_len or step pb_frame_skip_step or forecast_frame_len or forecast_step should be "
                                     "given. If you want to load only, set load_only=True.")
                self.train_data_pool = []

                if not self.is_distributed_train or self.rank == 0:
                    if load_data_recursive:
                        data_name_list = []
                        for root, dirs, files in os.walk(dir_path):
                            for file in files:
                                data_name_list.append(os.path.join(root, file))
                    else:
                        data_name_list = os.listdir(dir_path)

                    candidate_data_queue = Queue()
                    loaded_data_queue = Queue()
                    for idx, data_file_name in enumerate(data_name_list):
                        if not load_data_recursive:
                            data_path = os.path.join(dir_path, data_file_name)
                        else:
                            data_path = data_file_name
                        if os.path.isfile(data_path) is False or data_file_name.split('.')[-1] != 'pb':
                            logging.warning(f"\r{data_path} is not a protobuf file. Skip loading."
                                            f"                                                   ")
                            continue
                        else:
                            candidate_data_queue.put({'data_path': data_path, 'data_file_name': data_file_name})
                    total_files = candidate_data_queue.qsize()

                    data_loading_threads = []
                    for _ in range(6):
                        data_loading_thread = threading.Thread(target=self.load_protobuf_data_threaded,
                                                               args=(candidate_data_queue, loaded_data_queue, total_files,
                                                                     seq_len, pb_frame_skip_step, forecast_frame_len,
                                                                     forecast_frame_step))
                        data_loading_threads.append(data_loading_thread)
                    logging.info(
                        f"Total {total_files} files found. Loading protobuf data with {len(data_loading_threads)} threads.")
                    for thread in data_loading_threads:
                        thread.start()
                    for thread in data_loading_threads:
                        thread.join()

                    while not loaded_data_queue.empty():
                        loaded_data_queue_item = loaded_data_queue.get()
                        if loaded_data_queue_item is None:  # None is put when the data is skipped.
                            continue
                        loaded_data = loaded_data_queue_item['loaded_data']
                        data_file_name = loaded_data_queue_item['data_file_name']
                        if self.is_distributed_train:
                            logging.info(f"Device: {self.device} | "
                                         f"Broadcasting data "
                                         f"{len(self.train_data_pool) + 1}/{loaded_data_queue.qsize()} to other devices.")

                            self.broadcast_train_data_to_others_rank0(loaded_data_queue.qsize(), loaded_data)

                            logging.info(f"Device: {self.device} | "
                                         f"Data {len(self.train_data_pool) + 1}/{total_files} "
                                         f"broadcasting to other devices is done.")

                        self.train_data_pool.append(loaded_data)
                    logging.info(f"Total {total_files} files loaded.")
                else:  # for sharing datas to rank != 0 devices on distributed training
                    self.wait_for_broadcast_train_data_from_rank0()

                if not make_indices_for_train:
                    return self.train_data_pool

                return self.make_indices_for_train(seq_len, seq_split_window_step,
                                                   forecast_frame_len, forecast_frame_step,
                                                   validation_split_ratio)

            def broadcast_train_data_to_others_rank0(self, qsize, loaded_data):
                dist.broadcast(torch.tensor(qsize, dtype=torch.int, device=self.device), src=0)

                wrist_position_shape = (
                    torch.tensor(loaded_data['wrist']['position'].shape, dtype=torch.int, device=self.device))
                dist.broadcast(wrist_position_shape, src=0)
                dist.broadcast(loaded_data['wrist']['position'], src=0)

                wrist_orientation_shape = (
                    torch.tensor(loaded_data['wrist']['orientation'].shape, dtype=torch.int, device=self.device))
                dist.broadcast(wrist_orientation_shape, src=0)
                dist.broadcast(loaded_data['wrist']['orientation'], src=0)

                wrist_linear_velocity_shape = (
                    torch.tensor(loaded_data['wrist']['linear_velocity'].shape, dtype=torch.int, device=self.device))
                dist.broadcast(wrist_linear_velocity_shape, src=0)
                dist.broadcast(loaded_data['wrist']['linear_velocity'], src=0)

                wrist_angular_velocity_shape = (
                    torch.tensor(loaded_data['wrist']['angular_velocity'].shape, dtype=torch.int, device=self.device))
                dist.broadcast(wrist_angular_velocity_shape, src=0)
                dist.broadcast(loaded_data['wrist']['angular_velocity'], src=0)

                joint_position_shape = (
                    torch.tensor(loaded_data['joints']['position'].shape, dtype=torch.int, device=self.device))
                dist.broadcast(joint_position_shape, src=0)
                dist.broadcast(loaded_data['joints']['position'], src=0)

                joint_dof_shape = (
                    torch.tensor(loaded_data['joints']['dof'].shape, dtype=torch.int, device=self.device))
                dist.broadcast(joint_dof_shape, src=0)
                dist.broadcast(loaded_data['joints']['dof'], src=0)

                joint_linear_velocity_shape = (
                    torch.tensor(loaded_data['joints']['linear_velocity'].shape, dtype=torch.int, device=self.device))
                dist.broadcast(joint_linear_velocity_shape, src=0)
                dist.broadcast(loaded_data['joints']['linear_velocity'], src=0)

                joint_dof_velocity_shape = (
                    torch.tensor(loaded_data['joints']['dof_velocity'].shape, dtype=torch.int, device=self.device))
                dist.broadcast(joint_dof_velocity_shape, src=0)
                dist.broadcast(loaded_data['joints']['dof_velocity'], src=0)

                gvs_shape = torch.tensor(loaded_data['gvs'].shape, dtype=torch.int, device=self.device)
                dist.broadcast(gvs_shape, src=0)
                dist.broadcast(loaded_data['gvs'], src=0)

                contact_label_shape = (
                    torch.tensor(loaded_data['contactLabel'].shape, dtype=torch.int, device=self.device))
                dist.broadcast(contact_label_shape, src=0)
                dist.broadcast(loaded_data['contactLabel'], src=0)

                dist.broadcast(torch.tensor(len(loaded_data['lvs']), dtype=torch.int, device=self.device), src=0)
                for lvs in loaded_data['lvs']:
                    lvs_shape = torch.tensor(lvs.shape, dtype=torch.int, device=self.device)
                    dist.broadcast(lvs_shape, src=0)
                    dist.broadcast(lvs, src=0)

            def wait_for_broadcast_train_data_from_rank0(self):
                while_broadcast = True
                while while_broadcast:
                    qsize = torch.empty(1, dtype=torch.int).to(self.device)
                    dist.broadcast(qsize, src=0)
                    qsize = qsize.item()
                    if qsize == 0:
                        while_broadcast = False
                    loaded_data = {}
                    logging.info(f"Device: {self.device} | "
                                 f"Waiting for broadcasting data "
                                 f"{len(self.train_data_pool) + 1}/{qsize + len(self.train_data_pool) + 1} from rank 0.")

                    # Wrists
                    wrist_position_shape = torch.empty(2, dtype=torch.int, device=self.device)
                    dist.broadcast(wrist_position_shape, src=0)
                    wrist_position = torch.empty([si.item() for si in wrist_position_shape], device=self.device)
                    dist.broadcast(wrist_position, src=0)

                    wrist_orientation_shape = torch.empty(2, dtype=torch.int, device=self.device)
                    dist.broadcast(wrist_orientation_shape, src=0)
                    wrist_orientation = torch.empty([si.item() for si in wrist_orientation_shape], device=self.device)
                    dist.broadcast(wrist_orientation, src=0)

                    wrist_linear_velocity_shape = torch.empty(2, dtype=torch.int, device=self.device)
                    dist.broadcast(wrist_linear_velocity_shape, src=0)
                    wrist_linear_velocity = torch.empty([si.item() for si in wrist_linear_velocity_shape], device=self.device)
                    dist.broadcast(wrist_linear_velocity, src=0)

                    wrist_angular_velocity_shape = torch.empty(2, dtype=torch.int, device=self.device)
                    dist.broadcast(wrist_angular_velocity_shape, src=0)
                    wrist_angular_velocity = torch.empty([si.item() for si in wrist_angular_velocity_shape], device=self.device)
                    dist.broadcast(wrist_angular_velocity, src=0)

                    loaded_data['nFrames'] = wrist_position.shape[0]
                    loaded_data['wrist'] = {
                        "position": wrist_position,
                        "orientation": wrist_orientation,
                        "linear_velocity": wrist_linear_velocity,
                        "angular_velocity": wrist_angular_velocity
                    }

                    # Joints
                    joints_position_shape = torch.empty(3, dtype=torch.int, device=self.device)
                    dist.broadcast(joints_position_shape, src=0)
                    joints_position = torch.empty([si.item() for si in joints_position_shape], device=self.device)
                    dist.broadcast(joints_position, src=0)

                    joints_dof_shape = torch.empty(2, dtype=torch.int, device=self.device)
                    dist.broadcast(joints_dof_shape, src=0)
                    joints_dof = torch.empty([si.item() for si in joints_dof_shape], device=self.device)
                    dist.broadcast(joints_dof, src=0)

                    joints_linear_velocity_shape = torch.empty(3, dtype=torch.int, device=self.device)
                    dist.broadcast(joints_linear_velocity_shape, src=0)
                    joints_linear_velocity = torch.empty([si.item() for si in joints_linear_velocity_shape], device=self.device)
                    dist.broadcast(joints_linear_velocity, src=0)

                    joints_dof_velocity_shape = torch.empty(2, dtype=torch.int, device=self.device)
                    dist.broadcast(joints_dof_velocity_shape, src=0)
                    joints_dof_velocity = torch.empty([si.item() for si in joints_dof_velocity_shape], device=self.device)
                    dist.broadcast(joints_dof_velocity, src=0)

                    loaded_data['joints'] = {
                        "position": joints_position,
                        "dof": joints_dof,
                        "linear_velocity": joints_linear_velocity,
                        "dof_velocity": joints_dof_velocity
                    }

                    # GVS
                    gvs_shape = torch.empty(4, dtype=torch.int, device=self.device)
                    dist.broadcast(gvs_shape, src=0)
                    gvs = torch.empty([si.item() for si in gvs_shape], device=self.device)
                    dist.broadcast(gvs, src=0)
                    loaded_data['gvs'] = gvs

                    # Contact label
                    contact_label_shape = torch.empty(2, dtype=torch.int, device=self.device)
                    dist.broadcast(contact_label_shape, src=0)
                    contact_label = torch.empty([si.item() for si in contact_label_shape], device=self.device)
                    dist.broadcast(contact_label, src=0)
                    loaded_data['contactLabel'] = contact_label

                    lvs_list = []
                    lvs_len = torch.empty(1, dtype=torch.int, device=self.device)
                    dist.broadcast(lvs_len, src=0)
                    lvs_len = lvs_len.item()
                    for lvs in range(lvs_len):
                        lvs_shape = torch.empty(4, dtype=torch.int, device=self.device)
                        dist.broadcast(lvs_shape, src=0)
                        lvs = torch.empty([si.item() for si in lvs_shape], device=self.device)
                        dist.broadcast(lvs, src=0)
                        lvs_list.append(lvs)
                    loaded_data['lvs'] = lvs_list

                    self.train_data_pool.append(loaded_data)

                    logging.info(f"Device: {self.device} | Data broadcasting from rank 0 is done.")

            def make_indices_for_train(self, seq_len, seq_split_window_step, forecast_frame_len, forecast_step,
                                       validation_split_ratio=0.2):
                self.check_dataset_loaded()

                self.data_window_indices_for_train = []
                total_data = len(self.train_data_pool)
                for ddx, data in enumerate(self.train_data_pool):
                    if not self.is_distributed_train or self.rank == 0:
                        loading_char = next(self.loading_animation)
                        print(f"\r{loading_char} {ddx + 1}/{total_data} | Making data window indices for train", end="")
                    # round() is for floating point error. It's not (data['nFrames'] - seq_len - forecast_frame_len "- 1"),
                    # since the input_end_frame is a first frame of forecast_frame.
                    candidate_frames = int(np.ceil(
                        round((data['nFrames'] - seq_len - forecast_frame_len * forecast_step) / seq_split_window_step, 10)))
                    for fdx in range(candidate_frames):
                        self.data_window_indices_for_train.append({
                            "data_index": ddx,
                            "input_start_frame": fdx * seq_split_window_step,

                            # Exclusive.
                            "input_end_frame": fdx * seq_split_window_step + seq_len,

                            # -2 since the first input of the decoder query is dof t-1.
                            "forecast_start_frame": fdx * seq_split_window_step + seq_len - 2,

                            # Exclusive, the target frame to predict per forecast_step. (seq_len - 2 + 1) for exclusive.
                            "forecast_end_frame": fdx * seq_split_window_step + seq_len - 1 +
                                                  forecast_frame_len * forecast_step,
                            "forecast_step": forecast_step
                        })
                    if not self.is_distributed_train or self.rank == 0:
                        logging.info(f"\r{ddx + 1}/{total_data} | {data['file_name']} window indices made."
                                     f"                                                   ")
                total_frames = sum([data['nFrames'] for data in self.train_data_pool])

                if not self.is_distributed_train or self.rank == 0:
                    logging.info(f"Total {len(self.data_window_indices_for_train)} train indices "
                                 f"for total {total_frames} frames "
                                 f"from total {total_data} data made.")

                # Split the indices for train and validation
                total_indices = len(self.data_window_indices_for_train)
                split_idx = int(np.floor(total_indices * (1 - validation_split_ratio)))
                if self.shuffle:
                    random.shuffle(self.data_window_indices_for_train)
                self.data_window_indices_for_validation = self.data_window_indices_for_train[split_idx:]
                self.data_window_indices_for_train = self.data_window_indices_for_train[:split_idx]

                if not self.is_distributed_train or self.rank == 0:
                    logging.info(f"Train indices: {len(self.data_window_indices_for_train)}, "
                                 f"Validation indices: {len(self.data_window_indices_for_validation)}, "
                                 f"validation split: {validation_split_ratio * 100}%")
                return self.data_window_indices_for_train, self.data_window_indices_for_validation

            def get_input_data_from_index(self, index):
                self.check_dataset_loaded()
                self.check_indices_made()

                if self.__iter_mode == 'train':
                    target_indices = self.data_window_indices_for_train
                elif self.__iter_mode == 'validation':
                    target_indices = self.data_window_indices_for_validation
                else:
                    raise ValueError(f"Invalid iter mode: {self.__iter_mode}")

                data_index = target_indices[index]["data_index"]
                input_start_frame = target_indices[index]["input_start_frame"]
                input_end_frame = target_indices[index]["input_end_frame"]
                forecast_start_frame = target_indices[index]["forecast_start_frame"]
                forecast_end_frame = target_indices[index]["forecast_end_frame"]
                forecast_step = target_indices[index]["forecast_step"]

                input_frame_len = input_end_frame - input_start_frame
                forecast_frame_len = (forecast_end_frame - forecast_start_frame) // forecast_step + 1

                data = self.train_data_pool[data_index]

                # Wrist information
                w_t = torch.concat(
                    [
                        # wrist orientation, linear velocity, angular velocity
                        data['wrist']['orientation'][input_start_frame:input_end_frame].reshape(input_frame_len, -1),
                        data['wrist']['linear_velocity'][input_start_frame:input_end_frame].reshape(input_frame_len, -1),
                        data['wrist']['angular_velocity'][input_start_frame:input_end_frame].reshape(input_frame_len, -1)
                    ], dim=-1)

                # Hand information
                h_t = torch.concat(
                    [
                        # joint & fingertip position, dof
                        data['joints']['position'][input_start_frame:input_end_frame].reshape(input_frame_len, -1),
                        data['joints']['dof'][input_start_frame:input_end_frame].reshape(input_frame_len, -1),

                        # joint & fingertip linear velocity, dof velocity
                        data['joints']['linear_velocity'][input_start_frame:input_end_frame].reshape(input_frame_len, -1),
                        data['joints']['dof_velocity'][input_start_frame:input_end_frame].reshape(input_frame_len, -1),
                    ], dim=-1)
                h_t[-1] = 0

                # Contact label
                c_t = data['contactLabel'][input_start_frame:input_end_frame].reshape(input_frame_len, -1)
                c_t[-1] = 0

                # GVS
                g_t = data['gvs'][input_start_frame:input_end_frame]
                g_t[-1] = 0

                # LVS
                l_t = [data['lvs'][ldx][input_start_frame:input_end_frame] for ldx in range(len(data['lvs']))]
                for ldx, lvs in enumerate(l_t):
                    l_t[ldx][-1] = 0

                # Target DOFs
                # tgt_dofs[:-1] for decoder input, tgt_dofs[1:] for ground truth.
                tgt_dofs = \
                    data['joints']['dof'][forecast_start_frame:forecast_end_frame:forecast_step].reshape(forecast_frame_len, -1)

                tgt_joint_poses = \
                    data['joints']['position'][forecast_start_frame:forecast_end_frame:forecast_step][1:]

                return w_t, h_t, c_t, g_t, l_t, tgt_dofs, tgt_joint_poses, target_indices[index]

            def train(self):
                if self.__iter_mode != 'train':
                    self.current_index = 0
                self.__iter_mode = 'train'

            def validation(self):
                if self.__iter_mode != 'validation':
                    self.current_index = 0
                self.__iter_mode = 'validation'

            def __iter__(self):
                self.check_dataset_loaded()
                self.check_indices_made()

                self.current_index = 0
                return self

            def __next__(self):
                self.check_dataset_loaded()
                self.check_indices_made()

                if self.__iter_mode == 'train':
                    target_indices = self.data_window_indices_for_train
                elif self.__iter_mode == 'validation':
                    target_indices = self.data_window_indices_for_validation
                else:
                    raise ValueError(f"Invalid iter mode: {self.__iter_mode}")

                if self.shuffle and self.current_index == 0:
                    random.shuffle(target_indices)

                if self.drop_last and self.current_index + self.batch_size > len(target_indices):
                    if self.current_index == 0:
                        raise ValueError(f"Batch size is too large for the dataset. "
                                         f"Current batch size: {self.batch_size}, "
                                         f"Target data indices: {len(target_indices)}.")
                    self.current_index = 0
                    raise StopIteration
                elif self.current_index >= len(target_indices):
                    self.current_index = 0
                    raise StopIteration

                batch_data_w_t = []
                batch_data_h_t = []
                batch_data_c_t = []
                batch_data_g_t = []
                batch_data_l_t = None
                batch_data_tgt_dofs = []
                batch_data_tgt_joint_poses = []
                batch_data_data_indices = []

                batch_cnt = 0
                while batch_cnt < self.batch_size:
                    if self.current_index >= len(target_indices):
                        break
                    target_index = self.current_index + self.rank if self.is_distributed_train else self.current_index
                    w_t, h_t, c_t, g_t, l_t, tgt_dofs, tgt_joint_poses, idx_info = (
                        self.get_input_data_from_index(target_index))
                    batch_data_w_t.append(w_t)
                    batch_data_h_t.append(h_t)
                    batch_data_c_t.append(c_t)
                    batch_data_g_t.append(g_t)
                    if batch_data_l_t is None:
                        batch_data_l_t = [[l_t[ldx]] for ldx in range(len(l_t))]
                    else:
                        for ldx, lvs in enumerate(l_t):
                            batch_data_l_t[ldx].append(l_t[ldx])
                    batch_data_tgt_dofs.append(tgt_dofs)
                    batch_data_tgt_joint_poses.append(tgt_joint_poses)

                    batch_data_data_indices.append(idx_info)
                    if self.is_distributed_train:
                        self.current_index += self.world_size
                        batch_cnt += self.world_size
                    else:
                        self.current_index += 1
                        batch_cnt += 1

                batch_data_w_t = torch.stack(batch_data_w_t)
                batch_data_h_t = torch.stack(batch_data_h_t)
                batch_data_c_t = torch.stack(batch_data_c_t)
                batch_data_g_t = torch.stack(batch_data_g_t).unsqueeze(2)
                batch_data_l_t = [torch.stack(batch_data_l_t[ldx]).unsqueeze(2) for ldx in range(len(batch_data_l_t))]
                batch_data_tgt_dofs = torch.stack(batch_data_tgt_dofs)
                batch_data_tgt_joint_poses = torch.stack(batch_data_tgt_joint_poses)

                return (batch_data_w_t, batch_data_h_t, batch_data_c_t, batch_data_g_t, batch_data_l_t,
                        batch_data_tgt_dofs, batch_data_tgt_joint_poses,
                        batch_data_data_indices)

            def check_dataset_loaded(self):
                if self.train_data_pool is None:
                    raise ValueError("Dataset is not loaded yet. Call load_dataset() first.")

            def check_indices_made(self):
                if self.data_window_indices_for_train is None or self.data_window_indices_for_validation is None:
                    raise ValueError("Indices for training data is not made yet. Call make_indices_for_train() first.")

        def __main_dataset_manager(process_num, motion_planner_train_option, world_size=None):
            from Utils.Logger import setup_logging
            setup_logging()
            if motion_planner_train_option.device == 'cuda-dist':
                from Utils.DistributedTrainUtils import setup_devices
                rank = process_num
                logging.info(f"Spawning process for device {rank}.")
                setup_devices(rank, world_size)
            # Load and make the dataset
            dataset_manager = DatasetManager(batch_size=motion_planner_train_option.batch_size,
                                             device=motion_planner_train_option.device,
                                             shuffle=True,
                                             drop_last=True)
            dataset_manager.load_dataset(
                dir_path='./Dataset/Test',
                load_data_recursive=motion_planner_train_option.load_data_recursive,
                seq_len=motion_planner_train_option.seq_len,
                seq_split_window_step=motion_planner_train_option.seq_split_window_step,
                pb_frame_skip_step=motion_planner_train_option.pb_frame_skip_step,
                forecast_frame_len=motion_planner_train_option.forecast_frame_len,
                forecast_frame_step=motion_planner_train_option.forecast_frame_step,
                make_indices_for_train=True)

            for batch_idx, (w_t, h_t, c_t, g_t, l_t, tgt_dofs, tgt_joint_poses, data_indices) in enumerate(dataset_manager):
                if motion_planner_train_option.device == 'cuda-dist':
                    logging.info(f"Rank {rank} | Batch {batch_idx + 1} "
                                 f"| w_t: {w_t.shape}, h_t: {h_t.shape}, c_t: {c_t.shape}, "
                                 f"g_t: {g_t.shape}, l_t: {l_t[0].shape}, "
                                 f"tgt_dofs: {tgt_dofs.shape}, tgt_joint_poses: {tgt_joint_poses.shape}, "
                                 f"data_indices: {len(data_indices)}")
                    logging.info(f"Train_data: {dataset_manager.train_data_pool[dataset_manager.data_window_indices_for_train[0]['data_index']]['nFrames']}")
                    logging.info(f"Data indices: {dataset_manager.data_window_indices_for_train[0:2]}")
                else:
                    print(f"Batch {batch_idx + 1} "
                          f"| w_t: {w_t.shape}, h_t: {h_t.shape}, c_t: {c_t.shape}, "
                          f"g_t: {g_t.shape}, l_t: {l_t[0].shape}, "
                          f"tgt_dofs: {tgt_dofs.shape}, tgt_joint_poses: {tgt_joint_poses.shape}, "
                          f"data_indices: {len(data_indices)}")

                if batch_idx == 10:
                    break

            for batch_idx, (w_t, h_t, c_t, g_t, l_t, tgt_dofs, tgt_joint_poses, data_indices) in enumerate(dataset_manager):
                if motion_planner_train_option.device == 'cuda-dist':
                    logging.info(f"Rank {rank} | Batch {batch_idx + 1} "
                                 f"| w_t: {w_t.shape}, h_t: {h_t.shape}, c_t: {c_t.shape}, "
                                 f"g_t: {g_t.shape}, l_t: {l_t[0].shape}, "
                                 f"tgt_dofs: {tgt_dofs.shape}, tgt_joint_poses: {tgt_joint_poses.shape}, "
                                 f"data_indices: {len(data_indices)}")
                    logging.info(f"Train_data: {dataset_manager.train_data_pool[dataset_manager.data_window_indices_for_train[0]['data_index']]['nFrames']}")
                    logging.info(f"Data indices: {dataset_manager.data_window_indices_for_train[0:2]}")
                else:
                    print(f"Batch {batch_idx + 1} "
                          f"| w_t: {w_t.shape}, h_t: {h_t.shape}, c_t: {c_t.shape}, "
                          f"g_t: {g_t.shape}, l_t: {l_t[0].shape}, "
                          f"tgt_dofs: {tgt_dofs.shape}, tgt_joint_poses: {tgt_joint_poses.shape}, "
                          f"data_indices: {len(data_indices)}")

                if batch_idx == 10:
                    break

        if __name__ == "__main__":
            from MotionPlannerTrainOption import load_motion_planner_train_options_from_json

            motion_planner_train_option = load_motion_planner_train_options_from_json('motion_planner_train_option.json')
            # Note. motion_planner_train_option.dataset_dir_path = './Dataset/Test' in __main_dataset_manager

            if motion_planner_train_option.device == 'cuda-dist':
                device_cnt = torch.cuda.device_count()
                logging.info(f'Start distributed training with {device_cnt} devices.')
                torch.multiprocessing.spawn(__main_dataset_manager,
                                            args=(motion_planner_train_option, device_cnt),
                                            nprocs=device_cnt,
                                            join=True)
            else:
                __main_dataset_manager(0, motion_planner_train_option)

        self.data_window_indices_for_train = None
        self.data_window_indices_for_validation = None
        self.__iter_mode = 'train'

        self.is_distributed_train = False
        if device == 'cuda-dist':
            self.is_distributed_train = True
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            device = f'cuda:{self.rank}'
            if self.rank == 0:
                logging.info(f"|device: 'cuda-dist'| Distributed training mode of {self.world_size} devices set.")
                self.seed = int(time.time())
                random.seed(self.seed)
                dist.broadcast(torch.tensor(self.seed, device=device, dtype=torch.int), src=0)
            else:
                seed = torch.empty(1, dtype=torch.int).to(device)
                dist.broadcast(seed, src=0)
                self.seed = seed.item()
                random.seed(self.seed)

            if batch_size % self.world_size!= 0:
                raise ValueError("Batch size should be divisible by "
                                 "the number of gpus(processes) for distributed training.")
            if not drop_last:
                raise ValueError("drop_last should be True for distributed training.")

            logging.info(f"Rank {self.rank} | World size: {self.world_size} | Device: {device} | Seed: {self.seed}")

        self.device = device

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.current_index = 0

        self.loading_animation = itertools.cycle(['-', '/', '|', '\\'])

    def __len__(self):
        self.check_indices_made()

        if self.__iter_mode == 'train':
            if self.drop_last:
                return len(self.data_window_indices_for_train) // self.batch_size
            else:
                return int(np.ceil(len(self.data_window_indices_for_train) / self.batch_size))
        elif self.__iter_mode == 'validation':
            if self.drop_last:
                return len(self.data_window_indices_for_validation) // self.batch_size
            else:
                return int(np.ceil(len(self.data_window_indices_for_validation) / self.batch_size))
        else:
            raise ValueError(f"Invalid iter mode: {self.__iter_mode}")

    def load_protobuf_data_threaded(self, candidate_data_queue, loaded_data_queue, total_files,
                                    seq_len=None, pb_frame_skip_step=None,
                                    forecast_frame_len=None, forecast_frame_step=None):
        def prepend_values_recursive(data_dict, prepend_length):
            for key in data_dict.keys():
                if isinstance(data_dict[key], dict):  # Nested dictionary
                    prepend_values_recursive(data_dict[key], prepend_length)
                elif isinstance(data_dict[key], torch.Tensor):
                    prepend_shape = [1] * len(data_dict[key].shape)
                    prepend_shape[0] = prepend_length
                    prepend_value = data_dict[key][0].unsqueeze(0).repeat(*prepend_shape)
                    data_dict[key] = torch.cat((prepend_value, data_dict[key]), dim=0)
                elif (isinstance(data_dict[key], list) and len(data_dict[key]) > 0
                      and isinstance(data_dict[key][0], torch.Tensor)):
                    for i in range(len(data_dict[key])):
                        prepend_shape = [1] * len(data_dict[key][i].shape)
                        prepend_shape[0] = prepend_length
                        prepend_value = data_dict[key][i][0].unsqueeze(0).repeat(*prepend_shape)
                        data_dict[key][i] = torch.cat((prepend_value, data_dict[key][i]), dim=0)
        while not candidate_data_queue.empty():
            item = candidate_data_queue.get()
            data_path = item['data_path']
            data_file_name = item['data_file_name']

            loading_char = next(self.loading_animation)
            print(f"\r{loading_char} {total_files - candidate_data_queue.qsize()}/{total_files} "
                  f"| Loading file : {data_file_name}", end="")

            # Check frame length exceeds the required sequence length based on file name.
            is_frame_num_match = re.search(r"_(\d+)_(\d+).pb$", data_file_name)
            if is_frame_num_match:
                start_frame, end_frame = map(int, is_frame_num_match.groups())
                frame_count = end_frame - start_frame + 1
                if (frame_count - 1) // pb_frame_skip_step < forecast_frame_len * forecast_frame_step:
                    loaded_data_queue.put(None)
                    logging.warning(f"\r{loaded_data_queue.qsize()}/{total_files} "
                                    f"| Skip loading. {data_file_name} has less frames than the required sequence length."
                                    f"                                                   ")
                    continue
            loaded_data = load_protobuf_data(data_path)

            # Check once more with real data length.
            if (loaded_data['nFrames'] - 1) // pb_frame_skip_step < forecast_frame_len * forecast_frame_step:
                loaded_data_queue.put(None)
                logging.warning(f"\r{loaded_data_queue.qsize()}/{total_files} "
                                f"| Skip loading. {data_file_name} has less frames than the required sequence length."
                                f"                                                   ")
                continue

            train_data = {
                "file_name": data_file_name,
                "nFrames": (loaded_data['nFrames'] - 1) // pb_frame_skip_step,
                "nJoints": loaded_data['nJoints'],
                "scale": loaded_data['scale'],
                "nVertices": loaded_data['nVertices'],
                "wrist": {
                    "position": torch.from_numpy(
                        loaded_data['wristPosition'][pb_frame_skip_step::pb_frame_skip_step]).float().to(self.device),
                    "orientation": torch.from_numpy(
                        loaded_data['wristRotation'][pb_frame_skip_step::pb_frame_skip_step]).float().to(self.device),
                    "linear_velocity": torch.from_numpy(
                        np.diff(loaded_data['wristPosition'][::pb_frame_skip_step], axis=0)).float().to(self.device),
                    "angular_velocity":
                        torch.from_numpy(quat_to_angvel(loaded_data['wristRotation'][::pb_frame_skip_step])).float().to(
                            self.device),
                },
                "joints": {
                    "position": torch.from_numpy(
                        loaded_data['handJointPosition'][pb_frame_skip_step::pb_frame_skip_step]).float().to(
                        self.device),  # Exclude wrist position
                    "dof": torch.from_numpy(loaded_data['handDOF'][pb_frame_skip_step::pb_frame_skip_step]).float().to(
                        self.device),
                    "linear_velocity": torch.from_numpy(
                        np.diff(loaded_data['handJointPosition'][::pb_frame_skip_step], axis=0)).float().to(
                        self.device),
                    "dof_velocity": torch.from_numpy(
                        np.diff(loaded_data['handDOF'][::pb_frame_skip_step], axis=0)).float().to(self.device)
                },
                "gvs": torch.from_numpy(loaded_data['gvs'][pb_frame_skip_step::pb_frame_skip_step]).float().to(
                    self.device),
                "contactLabel": torch.from_numpy(
                    loaded_data['contactLabel'][pb_frame_skip_step::pb_frame_skip_step]).float().to(self.device)
            }

            lvs_list = []
            for ldx in range(len(loaded_data['lvs'][0])):
                lvs_list.append(
                    torch.stack([
                        torch.from_numpy(frame[ldx]).float().to(self.device) for frame in
                        loaded_data['lvs'][pb_frame_skip_step::pb_frame_skip_step]
                    ]))
            train_data["lvs"] = lvs_list

            prepend_values_recursive(train_data, seq_len-1)
            train_data['nFrames'] += seq_len-1

            loaded_data_queue.put({'loaded_data': train_data, 'data_file_name': data_file_name})
            logging.info(f"\r{loaded_data_queue.qsize()}/{total_files} | {data_file_name} loaded."
                         f"                                                   ")

    def load_dataset(self, dir_path, load_data_recursive=False,
                     seq_len=None, seq_split_window_step=None, pb_frame_skip_step=None,
                     forecast_frame_len=None, forecast_frame_step=None,
                     make_indices_for_train=True, validation_split_ratio=0):
        if not self.is_distributed_train or self.rank == 0:
            logging.info(f"Loading dataset from {dir_path}")
        if make_indices_for_train and (seq_len is None or
                                       seq_split_window_step is None or
                                       pb_frame_skip_step is None or
                                       forecast_frame_len is None or
                                       forecast_frame_step is None):
            raise ValueError("seq_len or step pb_frame_skip_step or forecast_frame_len or forecast_step should be "
                             "given. If you want to load only, set load_only=True.")
        self.train_data_pool = []

        if not self.is_distributed_train or self.rank == 0:
            if load_data_recursive:
                data_name_list = []
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        data_name_list.append(os.path.join(root, file))
            else:
                data_name_list = os.listdir(dir_path)

            candidate_data_queue = Queue()
            loaded_data_queue = Queue()
            for idx, data_file_name in enumerate(data_name_list):
                if not load_data_recursive:
                    data_path = os.path.join(dir_path, data_file_name)
                else:
                    data_path = data_file_name
                if os.path.isfile(data_path) is False or data_file_name.split('.')[-1] != 'pb':
                    logging.warning(f"\r{data_path} is not a protobuf file. Skip loading."
                                    f"                                                   ")
                    continue
                else:
                    candidate_data_queue.put({'data_path': data_path, 'data_file_name': data_file_name})
            total_files = candidate_data_queue.qsize()

            data_loading_threads = []
            for _ in range(6):
                data_loading_thread = threading.Thread(target=self.load_protobuf_data_threaded,
                                                       args=(candidate_data_queue, loaded_data_queue, total_files,
                                                             seq_len, pb_frame_skip_step, forecast_frame_len,
                                                             forecast_frame_step))
                data_loading_threads.append(data_loading_thread)
            logging.info(
                f"Total {total_files} files found. Loading protobuf data with {len(data_loading_threads)} threads.")
            for thread in data_loading_threads:
                thread.start()
            for thread in data_loading_threads:
                thread.join()

            while not loaded_data_queue.empty():
                loaded_data_queue_item = loaded_data_queue.get()
                if loaded_data_queue_item is None:  # None is put when the data is skipped.
                    continue
                loaded_data = loaded_data_queue_item['loaded_data']
                data_file_name = loaded_data_queue_item['data_file_name']
                if self.is_distributed_train:
                    logging.info(f"Device: {self.device} | "
                                 f"Broadcasting data "
                                 f"{len(self.train_data_pool)+1}/{loaded_data_queue.qsize()} to other devices.")

                    self.broadcast_train_data_to_others_rank0(loaded_data_queue.qsize(), loaded_data)

                    logging.info(f"Device: {self.device} | "
                                 f"Data {len(self.train_data_pool)+1}/{total_files} "
                                 f"broadcasting to other devices is done.")

                self.train_data_pool.append(loaded_data)
            logging.info(f"Total {total_files} files loaded.")
        else:  # for sharing datas to rank != 0 devices on distributed training
            self.wait_for_broadcast_train_data_from_rank0()

        if not make_indices_for_train:
            return self.train_data_pool

        return self.make_indices_for_train(seq_len, seq_split_window_step,
                                           forecast_frame_len, forecast_frame_step,
                                           validation_split_ratio)

    def broadcast_train_data_to_others_rank0(self, qsize, loaded_data):
        dist.broadcast(torch.tensor(qsize, dtype=torch.int, device=self.device), src=0)

        wrist_position_shape = (
            torch.tensor(loaded_data['wrist']['position'].shape, dtype=torch.int, device=self.device))
        dist.broadcast(wrist_position_shape, src=0)
        dist.broadcast(loaded_data['wrist']['position'], src=0)

        wrist_orientation_shape = (
            torch.tensor(loaded_data['wrist']['orientation'].shape, dtype=torch.int, device=self.device))
        dist.broadcast(wrist_orientation_shape, src=0)
        dist.broadcast(loaded_data['wrist']['orientation'], src=0)

        wrist_linear_velocity_shape = (
            torch.tensor(loaded_data['wrist']['linear_velocity'].shape, dtype=torch.int, device=self.device))
        dist.broadcast(wrist_linear_velocity_shape, src=0)
        dist.broadcast(loaded_data['wrist']['linear_velocity'], src=0)

        wrist_angular_velocity_shape = (
            torch.tensor(loaded_data['wrist']['angular_velocity'].shape, dtype=torch.int, device=self.device))
        dist.broadcast(wrist_angular_velocity_shape, src=0)
        dist.broadcast(loaded_data['wrist']['angular_velocity'], src=0)

        joint_position_shape = (
            torch.tensor(loaded_data['joints']['position'].shape, dtype=torch.int, device=self.device))
        dist.broadcast(joint_position_shape, src=0)
        dist.broadcast(loaded_data['joints']['position'], src=0)

        joint_dof_shape = (
            torch.tensor(loaded_data['joints']['dof'].shape, dtype=torch.int, device=self.device))
        dist.broadcast(joint_dof_shape, src=0)
        dist.broadcast(loaded_data['joints']['dof'], src=0)

        joint_linear_velocity_shape = (
            torch.tensor(loaded_data['joints']['linear_velocity'].shape, dtype=torch.int, device=self.device))
        dist.broadcast(joint_linear_velocity_shape, src=0)
        dist.broadcast(loaded_data['joints']['linear_velocity'], src=0)

        joint_dof_velocity_shape = (
            torch.tensor(loaded_data['joints']['dof_velocity'].shape, dtype=torch.int, device=self.device))
        dist.broadcast(joint_dof_velocity_shape, src=0)
        dist.broadcast(loaded_data['joints']['dof_velocity'], src=0)

        gvs_shape = torch.tensor(loaded_data['gvs'].shape, dtype=torch.int, device=self.device)
        dist.broadcast(gvs_shape, src=0)
        dist.broadcast(loaded_data['gvs'], src=0)

        contact_label_shape = (
            torch.tensor(loaded_data['contactLabel'].shape, dtype=torch.int, device=self.device))
        dist.broadcast(contact_label_shape, src=0)
        dist.broadcast(loaded_data['contactLabel'], src=0)

        dist.broadcast(torch.tensor(len(loaded_data['lvs']), dtype=torch.int, device=self.device), src=0)
        for lvs in loaded_data['lvs']:
            lvs_shape = torch.tensor(lvs.shape, dtype=torch.int, device=self.device)
            dist.broadcast(lvs_shape, src=0)
            dist.broadcast(lvs, src=0)

    def wait_for_broadcast_train_data_from_rank0(self):
        while_broadcast = True
        while while_broadcast:
            qsize = torch.empty(1, dtype=torch.int).to(self.device)
            dist.broadcast(qsize, src=0)
            qsize = qsize.item()
            if qsize == 0:
                while_broadcast = False
            loaded_data = {}
            logging.info(f"Device: {self.device} | "
                         f"Waiting for broadcasting data "
                         f"{len(self.train_data_pool)+1}/{qsize+len(self.train_data_pool)+1} from rank 0.")

            # Wrists
            wrist_position_shape = torch.empty(2, dtype=torch.int, device=self.device)
            dist.broadcast(wrist_position_shape, src=0)
            wrist_position = torch.empty([si.item() for si in wrist_position_shape], device=self.device)
            dist.broadcast(wrist_position, src=0)

            wrist_orientation_shape = torch.empty(2, dtype=torch.int, device=self.device)
            dist.broadcast(wrist_orientation_shape, src=0)
            wrist_orientation = torch.empty([si.item() for si in wrist_orientation_shape], device=self.device)
            dist.broadcast(wrist_orientation, src=0)

            wrist_linear_velocity_shape = torch.empty(2, dtype=torch.int, device=self.device)
            dist.broadcast(wrist_linear_velocity_shape, src=0)
            wrist_linear_velocity = torch.empty([si.item() for si in wrist_linear_velocity_shape], device=self.device)
            dist.broadcast(wrist_linear_velocity, src=0)

            wrist_angular_velocity_shape = torch.empty(2, dtype=torch.int, device=self.device)
            dist.broadcast(wrist_angular_velocity_shape, src=0)
            wrist_angular_velocity =torch.empty([si.item() for si in wrist_angular_velocity_shape], device=self.device)
            dist.broadcast(wrist_angular_velocity, src=0)

            loaded_data['nFrames'] = wrist_position.shape[0]
            loaded_data['wrist'] = {
                "position": wrist_position,
                "orientation": wrist_orientation,
                "linear_velocity": wrist_linear_velocity,
                "angular_velocity": wrist_angular_velocity
            }

            # Joints
            joints_position_shape = torch.empty(3, dtype=torch.int, device=self.device)
            dist.broadcast(joints_position_shape, src=0)
            joints_position = torch.empty([si.item() for si in joints_position_shape], device=self.device)
            dist.broadcast(joints_position, src=0)

            joints_dof_shape = torch.empty(2, dtype=torch.int, device=self.device)
            dist.broadcast(joints_dof_shape, src=0)
            joints_dof = torch.empty([si.item() for si in joints_dof_shape], device=self.device)
            dist.broadcast(joints_dof, src=0)

            joints_linear_velocity_shape = torch.empty(3, dtype=torch.int, device=self.device)
            dist.broadcast(joints_linear_velocity_shape, src=0)
            joints_linear_velocity = torch.empty([si.item() for si in joints_linear_velocity_shape], device=self.device)
            dist.broadcast(joints_linear_velocity, src=0)

            joints_dof_velocity_shape = torch.empty(2, dtype=torch.int, device=self.device)
            dist.broadcast(joints_dof_velocity_shape, src=0)
            joints_dof_velocity = torch.empty([si.item() for si in joints_dof_velocity_shape], device=self.device)
            dist.broadcast(joints_dof_velocity, src=0)

            loaded_data['joints'] = {
                "position": joints_position,
                "dof": joints_dof,
                "linear_velocity": joints_linear_velocity,
                "dof_velocity": joints_dof_velocity
            }

            # GVS
            gvs_shape = torch.empty(4, dtype=torch.int, device=self.device)
            dist.broadcast(gvs_shape, src=0)
            gvs = torch.empty([si.item() for si in gvs_shape], device=self.device)
            dist.broadcast(gvs, src=0)
            loaded_data['gvs'] = gvs

            # Contact label
            contact_label_shape = torch.empty(2, dtype=torch.int, device=self.device)
            dist.broadcast(contact_label_shape, src=0)
            contact_label = torch.empty([si.item() for si in contact_label_shape], device=self.device)
            dist.broadcast(contact_label, src=0)
            loaded_data['contactLabel'] = contact_label

            lvs_list = []
            lvs_len = torch.empty(1, dtype=torch.int, device=self.device)
            dist.broadcast(lvs_len, src=0)
            lvs_len = lvs_len.item()
            for lvs in range(lvs_len):
                lvs_shape = torch.empty(4, dtype=torch.int, device=self.device)
                dist.broadcast(lvs_shape, src=0)
                lvs = torch.empty([si.item() for si in lvs_shape], device=self.device)
                dist.broadcast(lvs, src=0)
                lvs_list.append(lvs)
            loaded_data['lvs'] = lvs_list

            self.train_data_pool.append(loaded_data)

            logging.info(f"Device: {self.device} | Data broadcasting from rank 0 is done.")

    def make_indices_for_train(self, seq_len, seq_split_window_step, forecast_frame_len, forecast_step,
                               validation_split_ratio=0.2):
        self.check_dataset_loaded()

        self.data_window_indices_for_train = []
        total_data = len(self.train_data_pool)
        for ddx, data in enumerate(self.train_data_pool):
            if not self.is_distributed_train or self.rank == 0:
                loading_char = next(self.loading_animation)
                print(f"\r{loading_char} {ddx + 1}/{total_data} | Making data window indices for train", end="")
            # round() is for floating point error. It's not (data['nFrames'] - seq_len - forecast_frame_len "- 1"),
            # since the input_end_frame is a first frame of forecast_frame.
            candidate_frames = int(np.ceil(
                round((data['nFrames'] - seq_len - forecast_frame_len * forecast_step) / seq_split_window_step, 10)))
            for fdx in range(candidate_frames):
                self.data_window_indices_for_train.append({
                    "data_index": ddx,
                    "input_start_frame": fdx * seq_split_window_step,

                    # Exclusive.
                    "input_end_frame": fdx * seq_split_window_step + seq_len,

                    # -2 since the first input of the decoder query is dof t-1.
                    "forecast_start_frame": fdx * seq_split_window_step + seq_len - 2,

                    # Exclusive, the target frame to predict per forecast_step. (seq_len - 2 + 1) for exclusive.
                    "forecast_end_frame": fdx * seq_split_window_step + seq_len - 1 +
                                          forecast_frame_len * forecast_step,
                    "forecast_step": forecast_step
                })
            if not self.is_distributed_train or self.rank == 0:
                logging.info(f"\r{ddx + 1}/{total_data} | {data['file_name']} window indices made."
                             f"                                                   ")
        total_frames = sum([data['nFrames'] for data in self.train_data_pool])

        if not self.is_distributed_train or self.rank == 0:
            logging.info(f"Total {len(self.data_window_indices_for_train)} train indices "
                         f"for total {total_frames} frames "
                         f"from total {total_data} data made.")

        # Split the indices for train and validation
        total_indices = len(self.data_window_indices_for_train)
        split_idx = int(np.floor(total_indices * (1 - validation_split_ratio)))
        if self.shuffle:
            random.shuffle(self.data_window_indices_for_train)
        self.data_window_indices_for_validation = self.data_window_indices_for_train[split_idx:]
        self.data_window_indices_for_train = self.data_window_indices_for_train[:split_idx]

        if not self.is_distributed_train or self.rank == 0:
            logging.info(f"Train indices: {len(self.data_window_indices_for_train)}, "
                         f"Validation indices: {len(self.data_window_indices_for_validation)}, "
                         f"validation split: {validation_split_ratio * 100}%")
        return self.data_window_indices_for_train, self.data_window_indices_for_validation

    def get_input_data_from_index(self, index):
        self.check_dataset_loaded()
        self.check_indices_made()

        if self.__iter_mode == 'train':
            target_indices = self.data_window_indices_for_train
        elif self.__iter_mode == 'validation':
            target_indices = self.data_window_indices_for_validation
        else:
            raise ValueError(f"Invalid iter mode: {self.__iter_mode}")

        data_index = target_indices[index]["data_index"]
        input_start_frame = target_indices[index]["input_start_frame"]
        input_end_frame = target_indices[index]["input_end_frame"]
        forecast_start_frame = target_indices[index]["forecast_start_frame"]
        forecast_end_frame = target_indices[index]["forecast_end_frame"]
        forecast_step = target_indices[index]["forecast_step"]

        input_frame_len = input_end_frame - input_start_frame
        forecast_frame_len = (forecast_end_frame - forecast_start_frame) // forecast_step + 1

        data = self.train_data_pool[data_index]

        # Wrist information
        w_t = torch.concat(
            [
                # wrist orientation, linear velocity, angular velocity
                data['wrist']['orientation'][input_start_frame:input_end_frame].reshape(input_frame_len, -1),
                data['wrist']['linear_velocity'][input_start_frame:input_end_frame].reshape(input_frame_len, -1),
                data['wrist']['angular_velocity'][input_start_frame:input_end_frame].reshape(input_frame_len, -1)
            ], dim=-1)

        # Hand information
        h_t = torch.concat(
            [
                # joint & fingertip position, dof
                data['joints']['position'][input_start_frame:input_end_frame].reshape(input_frame_len, -1),
                data['joints']['dof'][input_start_frame:input_end_frame].reshape(input_frame_len, -1),

                # joint & fingertip linear velocity, dof velocity
                data['joints']['linear_velocity'][input_start_frame:input_end_frame].reshape(input_frame_len, -1),
                data['joints']['dof_velocity'][input_start_frame:input_end_frame].reshape(input_frame_len, -1),
            ], dim=-1)
        h_t[-1] = 0

        # Contact label
        c_t = data['contactLabel'][input_start_frame:input_end_frame].reshape(input_frame_len, -1)
        c_t[-1] = 0

        # GVS
        g_t = data['gvs'][input_start_frame:input_end_frame]
        g_t[-1] = 0

        # LVS
        l_t = [data['lvs'][ldx][input_start_frame:input_end_frame] for ldx in range(len(data['lvs']))]
        for ldx, lvs in enumerate(l_t):
            l_t[ldx][-1] = 0

        # Target DOFs
        # tgt_dofs[:-1] for decoder input, tgt_dofs[1:] for ground truth.
        tgt_dofs = \
            data['joints']['dof'][forecast_start_frame:forecast_end_frame:forecast_step].reshape(forecast_frame_len, -1)

        tgt_joint_poses = \
            data['joints']['position'][forecast_start_frame:forecast_end_frame:forecast_step][1:]

        return w_t, h_t, c_t, g_t, l_t, tgt_dofs, tgt_joint_poses, target_indices[index]

    def train(self):
        if self.__iter_mode != 'train':
            self.current_index = 0
        self.__iter_mode = 'train'

    def validation(self):
        if self.__iter_mode != 'validation':
            self.current_index = 0
        self.__iter_mode = 'validation'

    def __iter__(self):
        self.check_dataset_loaded()
        self.check_indices_made()

        self.current_index = 0
        return self

    def __next__(self):
        self.check_dataset_loaded()
        self.check_indices_made()

        if self.__iter_mode == 'train':
            target_indices = self.data_window_indices_for_train
        elif self.__iter_mode == 'validation':
            target_indices = self.data_window_indices_for_validation
        else:
            raise ValueError(f"Invalid iter mode: {self.__iter_mode}")

        if self.shuffle and self.current_index == 0:
            random.shuffle(target_indices)

        if self.drop_last and self.current_index + self.batch_size > len(target_indices):
            if self.current_index == 0:
                raise ValueError(f"Batch size is too large for the dataset. "
                                 f"Current batch size: {self.batch_size}, "
                                 f"Target data indices: {len(target_indices)}.")
            self.current_index = 0
            raise StopIteration
        elif self.current_index >= len(target_indices):
            self.current_index = 0
            raise StopIteration

        batch_data_w_t = []
        batch_data_h_t = []
        batch_data_c_t = []
        batch_data_g_t = []
        batch_data_l_t = None
        batch_data_tgt_dofs = []
        batch_data_tgt_joint_poses = []
        batch_data_data_indices = []

        batch_cnt = 0
        while batch_cnt < self.batch_size:
            if self.current_index >= len(target_indices):
                break
            target_index = self.current_index + self.rank if self.is_distributed_train else self.current_index
            w_t, h_t, c_t, g_t, l_t, tgt_dofs, tgt_joint_poses, idx_info =(
                self.get_input_data_from_index(target_index))
            batch_data_w_t.append(w_t)
            batch_data_h_t.append(h_t)
            batch_data_c_t.append(c_t)
            batch_data_g_t.append(g_t)
            if batch_data_l_t is None:
                batch_data_l_t = [[l_t[ldx]] for ldx in range(len(l_t))]
            else:
                for ldx, lvs in enumerate(l_t):
                    batch_data_l_t[ldx].append(l_t[ldx])
            batch_data_tgt_dofs.append(tgt_dofs)
            batch_data_tgt_joint_poses.append(tgt_joint_poses)

            batch_data_data_indices.append(idx_info)
            if self.is_distributed_train:
                self.current_index += self.world_size
                batch_cnt += self.world_size
            else:
                self.current_index += 1
                batch_cnt += 1

        batch_data_w_t = torch.stack(batch_data_w_t)
        batch_data_h_t = torch.stack(batch_data_h_t)
        batch_data_c_t = torch.stack(batch_data_c_t)
        batch_data_g_t = torch.stack(batch_data_g_t).unsqueeze(2)
        batch_data_l_t = [torch.stack(batch_data_l_t[ldx]).unsqueeze(2) for ldx in range(len(batch_data_l_t))]
        batch_data_tgt_dofs = torch.stack(batch_data_tgt_dofs)
        batch_data_tgt_joint_poses = torch.stack(batch_data_tgt_joint_poses)

        return (batch_data_w_t, batch_data_h_t, batch_data_c_t, batch_data_g_t, batch_data_l_t,
                batch_data_tgt_dofs, batch_data_tgt_joint_poses,
                batch_data_data_indices)

    def check_dataset_loaded(self):
        if self.train_data_pool is None:
            raise ValueError("Dataset is not loaded yet. Call load_dataset() first.")

    def check_indices_made(self):
        if self.data_window_indices_for_train is None or self.data_window_indices_for_validation is None:
            raise ValueError("Indices for training data is not made yet. Call make_indices_for_train() first.")


def __main_dataset_manager(process_num, motion_planner_train_option, world_size=None):
    from Utils.Logger import setup_logging
    setup_logging()
    if motion_planner_train_option.device == 'cuda-dist':
        from Utils.DistributedTrainUtils import setup_devices
        rank = process_num
        logging.info(f"Spawning process for device {rank}.")
        setup_devices(rank, world_size)
    # Load and make the dataset
    dataset_manager = DatasetManager(batch_size=motion_planner_train_option.batch_size,
                                     device=motion_planner_train_option.device,
                                     shuffle=True,
                                     drop_last=True)
    dataset_manager.load_dataset(
        dir_path='./Dataset/Test',
        load_data_recursive=motion_planner_train_option.load_data_recursive,
        seq_len=motion_planner_train_option.seq_len,
        seq_split_window_step=motion_planner_train_option.seq_split_window_step,
        pb_frame_skip_step=motion_planner_train_option.pb_frame_skip_step,
        forecast_frame_len=motion_planner_train_option.forecast_frame_len,
        forecast_frame_step=motion_planner_train_option.forecast_frame_step,
        make_indices_for_train=True)

    for batch_idx, (w_t, h_t, c_t, g_t, l_t, tgt_dofs, tgt_joint_poses, data_indices) in enumerate(dataset_manager):
        if motion_planner_train_option.device == 'cuda-dist':
            logging.info(f"Rank {rank} | Batch {batch_idx + 1} "
                         f"| w_t: {w_t.shape}, h_t: {h_t.shape}, c_t: {c_t.shape}, "
                         f"g_t: {g_t.shape}, l_t: {l_t[0].shape}, "
                         f"tgt_dofs: {tgt_dofs.shape}, tgt_joint_poses: {tgt_joint_poses.shape}, "
                         f"data_indices: {len(data_indices)}")
            logging.info(f"Train_data: {dataset_manager.train_data_pool[dataset_manager.data_window_indices_for_train[0]['data_index']]['nFrames']}")
            logging.info(f"Data indices: {dataset_manager.data_window_indices_for_train[0:2]}")
        else:
            print(f"Batch {batch_idx + 1} "
                  f"| w_t: {w_t.shape}, h_t: {h_t.shape}, c_t: {c_t.shape}, "
                  f"g_t: {g_t.shape}, l_t: {l_t[0].shape}, "
                  f"tgt_dofs: {tgt_dofs.shape}, tgt_joint_poses: {tgt_joint_poses.shape}, "
                  f"data_indices: {len(data_indices)}")

        if batch_idx == 10:
            break

    for batch_idx, (w_t, h_t, c_t, g_t, l_t, tgt_dofs, tgt_joint_poses, data_indices) in enumerate(dataset_manager):
        if motion_planner_train_option.device == 'cuda-dist':
            logging.info(f"Rank {rank} | Batch {batch_idx + 1} "
                         f"| w_t: {w_t.shape}, h_t: {h_t.shape}, c_t: {c_t.shape}, "
                         f"g_t: {g_t.shape}, l_t: {l_t[0].shape}, "
                         f"tgt_dofs: {tgt_dofs.shape}, tgt_joint_poses: {tgt_joint_poses.shape}, "
                         f"data_indices: {len(data_indices)}")
            logging.info(f"Train_data: {dataset_manager.train_data_pool[dataset_manager.data_window_indices_for_train[0]['data_index']]['nFrames']}")
            logging.info(f"Data indices: {dataset_manager.data_window_indices_for_train[0:2]}")
        else:
            print(f"Batch {batch_idx + 1} "
                  f"| w_t: {w_t.shape}, h_t: {h_t.shape}, c_t: {c_t.shape}, "
                  f"g_t: {g_t.shape}, l_t: {l_t[0].shape}, "
                  f"tgt_dofs: {tgt_dofs.shape}, tgt_joint_poses: {tgt_joint_poses.shape}, "
                  f"data_indices: {len(data_indices)}")

        if batch_idx == 10:
            break


if __name__ == "__main__":
    from MotionPlannerTrainOption import load_motion_planner_train_options_from_json

    motion_planner_train_option = load_motion_planner_train_options_from_json('motion_planner_train_option.json')
    # Note. motion_planner_train_option.dataset_dir_path = './Dataset/Test' in __main_dataset_manager

    if motion_planner_train_option.device == 'cuda-dist':
        device_cnt = torch.cuda.device_count()
        logging.info(f'Start distributed training with {device_cnt} devices.')
        torch.multiprocessing.spawn(__main_dataset_manager,
                                    args=(motion_planner_train_option, device_cnt),
                                    nprocs=device_cnt,
                                    join=True)
    else:
        __main_dataset_manager(0, motion_planner_train_option)
