import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import pickle
import random
import copy
import math


class DamoDataset(Dataset):

    def __init__(
            self,
            data_path,
            n_max_markers,
            seq_len,
            marker_shuffle=False,
            marker_occlusion=False,
            marker_ghost=False,
            marker_jittering=False,
            distance_from_skin=0.0095,
            distance_augmentation=False,
            test=False,
            superset_only=False
    ):
        super().__init__()

        self.n_max_markers = n_max_markers  # 100
        self.seq_len = seq_len  # 7
        self.marker_shuffle = marker_shuffle
        self.marker_occlusion = marker_occlusion
        self.marker_ghost = marker_ghost
        self.marker_jittering = marker_jittering
        self.distance_from_skin = distance_from_skin
        self.distance_augmentation = distance_augmentation
        self.test = test
        self.superset_only=superset_only

        # The length of the sequence must be odd.
        assert self.seq_len % 2 == 1, ValueError(
            f"seq_len ({self.seq_len}) must be odd number.")

        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.frames = self.data['frames']
        self.stacked_frames = np.cumsum(self.frames)
        self.total_frame = sum(self.frames)
        self.length = len(self.frames)

        print(f'INFO | Dataset | Total Motion num: {self.length}')
        print(f'INFO | Dataset | Total frame length: {self.total_frame}')

        self.superset_variation = self.data['superset_variation']

        # The set maximum number of markers and the actual maximum number of markers in the dataset.
        print(f'INFO | Dataset | \n'
              f'\tData max markers num: {len(self.superset_variation)}'
              f'\tConfig max markers num: {self.n_max_markers}')

        self.j3_indices = self.data['j3_indices']
        self.j3_weights = self.data['j3_weights']
        self.j3_offsets = self.data['j3_offsets']
        self.init_vn = self.data['init_vertices_normal']
        self.init_j_l_p = self.data['init_joint_local_position']
        self.j_marker_indices = self.data['j_marker_indices']
        self.n_joints = len(self.j_marker_indices)
        self.jpi = self.data['joint_hierarchy']
        self.root_position = self.data['root_position']
        self.joint_global_rotation = self.data['joint_global_rotation']
        self.vertex_joint_weights = self.data['vertex_joint_weights']

    def __len__(self):
        if self.test:
            return self.total_frame // 10
        else:
            return self.length * 100

    def __getitem__(self, item_idx):
        if self.test:
            real_idx = item_idx * 10
            motion_idx = np.searchsorted(self.stacked_frames - real_idx, 0, side='right')
            n_frame = self.frames[motion_idx]

            if motion_idx == 0:
                frame_idx = real_idx
            else:
                frame_idx = real_idx - self.stacked_frames[motion_idx - 1]
        else:
            motion_idx = item_idx % self.length
            n_frame = self.frames[motion_idx]
            frame_idx = random.randint(0, n_frame - 1)

        half_seq = self.seq_len // 2
        front_count = 0
        front_idx = -1
        end_count = 0
        end_idx = -1
        n_joints = self.n_joints

        if frame_idx < half_seq:
            front_count = half_seq - frame_idx
            front_idx = 0
        else:
            front_idx = frame_idx - half_seq
        if frame_idx >= n_frame - half_seq:
            end_count = half_seq - n_frame + frame_idx + 1
            end_idx = n_frame
        else:
            end_idx = frame_idx + half_seq + 1

        use_superset = True

        if not self.superset_only:
            if torch.rand(1) < 0.5:
                use_superset = False

        if use_superset:
            n_selected_superset = np.random.randint(22, len(self.superset_variation))
            selected_superset_indices = np.random.choice(
                np.arange(len(self.superset_variation)),
                n_selected_superset,
                replace=False
            )
            marker_indices = np.empty(n_selected_superset, dtype=int)
            for i in range(n_selected_superset):
                variation = self.superset_variation[selected_superset_indices[i]]
                random_idx = np.random.randint(0, len(variation))
                marker_indices[i] = variation[random_idx]
        else:
            marker_indices = []
            values = [2, 3, 4]
            prob = [0.7, 0.2, 0.1]
            indices_num_list = np.random.choice(values, size=n_joints, p=prob)
            for i in range(n_joints):
                marker_indices.extend(np.random.choice(
                    self.j_marker_indices[i],
                    size=indices_num_list[i],
                    replace=False
                ))
            marker_indices = np.array(marker_indices)

        n_body_type_num = self.j3_offsets.shape[0]
        body_type = random.randint(0, n_body_type_num - 1)

        # j3_indices = self.data['j3_indices']  # (6890, 3)
        # j3_weights = self.data['j3_weights']  # (6890, 3)
        j3_offsets = self.j3_offsets[body_type]  # (6890, 3, 3)
        init_vn = self.init_vn[body_type]  # (6890, 3)
        init_j_l_p = self.init_j_l_p[body_type]  # (22, 3)
        j_g_p = np.empty((end_idx - front_idx, n_joints, 3))  # (7, 22, 3)

        theta = math.radians(random.uniform(0, 360))
        aug_rot_mat = torch.tensor([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ])

        for i, pi in enumerate(self.jpi):
            if pi == -1:
                assert i == 0
                j_g_p[:, i, :] = self.root_position[motion_idx][front_idx:end_idx, :]\
                                 + init_j_l_p[np.newaxis, i, :]
                continue

            j_g_p[:, i, :] = (self.joint_global_rotation[motion_idx][front_idx:end_idx, pi, :]\
                              @ init_j_l_p[np.newaxis, i, :, np.newaxis]).squeeze()
            j_g_p[:, i, :] += j_g_p[:, pi, :]

        points_seq = []
        points_mask = []
        n_clean_markers = -1
        n_ghost_markers = 0
        n_total_markers = -1
        marker_joint_weights = None
        # t_j3_indices = None
        t_j3_weights = None
        t_j3_offsets = None
        shuffled_indices = None

        output_j_g_r = None
        output_j_g_p = None

        if self.distance_augmentation:
            aug_range = 0.005
            aug_offset = (torch.rand(1) - 0.5) * aug_range
            distance_from_skin = self.distance_from_skin + aug_offset
        else:
            distance_from_skin = self.distance_from_skin

        for _ in range(front_count):
            points_seq.append(torch.zeros(1, self.n_max_markers, 3))
            points_mask.append(torch.zeros(1, self.n_max_markers, requires_grad=False))

        for idx in range(front_idx, end_idx):
            temp_marker_indices = copy.deepcopy(marker_indices)
            if self.marker_occlusion:
                # Randomly select the number of markers to delete.
                n_occlusion = random.randint(0, 5)
                # Select markers to delete amounting to n_occlusion from marker_indices.
                occlusion_indices = np.random.choice(len(temp_marker_indices), size=n_occlusion, replace=False)
                # Delete the selected markers.
                temp_marker_indices = np.delete(temp_marker_indices, occlusion_indices)

            j_g_r = self.joint_global_rotation[motion_idx][idx]  # (J, 3, 3)
            j3_g_r = np.repeat(j_g_r[np.newaxis, :, :, :], repeats=len(temp_marker_indices), axis=0)  # (M, J, 3, 3)
            j3_index = self.j3_indices[temp_marker_indices, :]  # (M, j3)
            j3_weight = self.j3_weights[temp_marker_indices, :]  # (M, j3)
            j3_offset = j3_offsets[temp_marker_indices, :, :]  # (M, j3, 3)

            j3_g_r = j3_g_r[np.arange(j3_g_r.shape[0])[:, None], j3_index]  # (M, j3, 3, 3)

            j3_g_p = np.repeat(j_g_p[idx - front_idx][np.newaxis, :, :], repeats=len(temp_marker_indices), axis=0)  # (M, J, 3)
            j3_g_p = j3_g_p[np.arange(j3_g_p.shape[0])[:, None], j3_index]  # (M, j3, 3)

            points = j3_g_r @ j3_offset[:, :, :, np.newaxis]  # (M, j3, 3, 1)
            # (M, j3, 3, 3) @ (M, j3, 3, 1) -> (M, j3, 3, 1)
            points = points.squeeze()
            points += j3_g_p
            points = np.sum(points * j3_weight[:, :, np.newaxis], axis=1)

            points_norm = j3_g_r @ init_vn[temp_marker_indices, np.newaxis, :, np.newaxis]
            # (M, j3, 3, 3) @ (M, 1, 3, 1) -> (M, j3, 3, 1)
            points_norm = np.sum(points_norm.squeeze() * j3_weight[:, :, np.newaxis], axis=1)

            points = torch.from_numpy(points).to(torch.float32)
            points_norm = torch.from_numpy(points_norm).to(torch.float32)

            points += (points_norm * distance_from_skin)

            n_clean_markers = len(temp_marker_indices)

            if self.marker_jittering:
                jittering_range = 0.01
                jittering = (torch.rand_like(points) - 0.5) * jittering_range
                points = points + jittering

            if self.marker_ghost:
                max_ghost_markers = 2
                n_ghost_markers = torch.randint(-max_ghost_markers, max_ghost_markers + 1, (1,)).item()
                n_ghost_markers = max(0, n_ghost_markers)
                points_center = points.mean(axis=0)
                ghost_markers_range = 1
                ghost_markers = (torch.rand(n_ghost_markers, points.shape[-1]) - 0.5) * ghost_markers_range
                ghost_markers = ghost_markers + points_center
                points = torch.cat((points, ghost_markers), dim=0)

                n_total_markers = n_clean_markers + n_ghost_markers
            else:
                n_total_markers = n_clean_markers

            assert(self.n_max_markers >= n_total_markers)

            points = F.pad(
                input=points,
                pad=[0, 0, 0, self.n_max_markers - n_total_markers],
                mode='constant',
                value=0
            )

            if self.marker_shuffle:
                shuffled_indices = torch.randperm(n_total_markers)
                points[:n_total_markers, :] = points[shuffled_indices, :]

            points_seq.append(points.unsqueeze(0))
            mask = torch.zeros(1, self.n_max_markers, requires_grad=False)

            mask[:, :n_total_markers] = 1
            points_mask.append(mask)

            if idx == frame_idx:

                assert marker_joint_weights is None, ValueError(
                    f"Center frame is already set.")

                ghost_indices = torch.ones((self.n_max_markers, 1))
                ghost_indices[:n_clean_markers, 0] = 0

                w = self.vertex_joint_weights[temp_marker_indices, :]
                marker_joint_weights = torch.from_numpy(w).to(torch.float32)
                marker_joint_weights = F.pad(
                    input=marker_joint_weights,
                    pad=[0, 0, 0, self.n_max_markers - n_clean_markers],
                    mode='constant',
                    value=0
                )
                marker_joint_weights = torch.cat((marker_joint_weights, ghost_indices), dim=-1)

                t_j3_weights = torch.from_numpy(j3_weight).to(torch.float32)
                t_j3_weights = F.pad(
                    input=t_j3_weights,
                    pad=[0, 0, 0, self.n_max_markers - n_clean_markers],
                    mode='constant',
                    value=0
                )
                t_j3_weights[n_clean_markers:, 0] = 1

                t_j3_offsets = torch.from_numpy(j3_offset).to(torch.float32)
                t_j3_offsets = F.pad(
                    input=t_j3_offsets,
                    pad=[0, 0, 0, 0, 0, self.n_max_markers - n_clean_markers],
                    mode='constant',
                    value=0
                )

                if self.marker_shuffle:
                    marker_joint_weights[:n_total_markers, :] = marker_joint_weights[shuffled_indices, :]
                    t_j3_weights[:n_total_markers, :] = t_j3_weights[shuffled_indices, :]
                    t_j3_offsets[:n_total_markers, :, :] = t_j3_offsets[shuffled_indices, :, :]

                output_j_g_r = torch.from_numpy(j_g_r).detach().to(torch.float32)
                output_j_g_p = torch.from_numpy(j_g_p[idx - front_idx]).detach().to(torch.float32)
                output_j_g_p = torch.matmul(output_j_g_p, aug_rot_mat)

        for _ in range(end_count):
            points_seq.append(torch.zeros(1, self.n_max_markers, 3))
            points_mask.append(torch.zeros(1, self.n_max_markers, requires_grad=False))

        points_seq = torch.cat(points_seq, dim=0)
        points_mask = torch.cat(points_mask, dim=0)

        points_seq = torch.matmul(points_seq, aug_rot_mat)

        items = {
            'points_seq': points_seq,
            'points_mask': points_mask,
            'marker_joint_weights': marker_joint_weights,
            'j3_weights': t_j3_weights,
            'j3_offsets': t_j3_offsets,
            'joint_position': output_j_g_p,
            'joint_rotation': output_j_g_r
        }

        return items