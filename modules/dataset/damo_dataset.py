import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import pickle
import random
import copy
import math

from human_body_prior.body_model.lbs import batch_rodrigues


class DamoDataset(Dataset):
    def __init__(
            self,
            common_dataset_path,
            dataset_paths,
            n_max_markers,
            seq_len,
            noise_jitter=False,
            noise_ghost=False,
            noise_occlusion=False,
            noise_shuffle=False,
            dist_from_skin=0.01,
            dist_augmentation=False,
            test=False
    ):
        super().__init__()

        self.n_max_markers = n_max_markers
        self.seq_len = seq_len
        self.noise_jitter = noise_jitter
        self.noise_ghost = noise_ghost
        self.noise_occlusion = noise_occlusion
        self.noise_shuffle = noise_shuffle
        self.dist_from_skin = dist_from_skin
        self.dist_augmentation = dist_augmentation
        self.test = test

        assert self.seq_len % 2 == 1, ValueError(
            f"seq_len ({self.seq_len}) must be odd number.")

        with open(common_dataset_path, 'rb') as f:
            common_data = pickle.load(f)

        self.topology = common_data['topology']
        self.weights = common_data['weights']
        self.j_v_idx = common_data['j_v_idx']
        self.n_joints = common_data['n_joints']
        self.superset_variant = common_data['soma_superset_variant']
        self.caesar_bind_v = common_data['caesar_bind_v']
        self.caesar_bind_vn = common_data['caesar_bind_vn']
        self.caesar_bind_jgp = common_data['caesar_bind_jgp']
        self.caesar_bind_jlp = common_data['caesar_bind_jlp']
        self.caesar_v_j3_offsets = common_data['caesar_v_j3_offsets']
        self.v_j3_indices = common_data['v_j3_indices']
        self.v_j3_weights = common_data['v_j3_weights']

        self.n_frames = []
        self.m_v_idx = []
        self.m_j3_indices = []
        self.m_j3_weights = []
        self.m_j3_offsets = []
        self.bind_v_shaped = []
        self.bind_vn = []
        self.bind_jlp = []
        self.bind_jgp = []
        self.ghost_marker_mask = []
        self.markers = []
        self.poses = []
        self.jgp = []

        for dataset_path in dataset_paths:
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)

            self.n_frames.extend(data['n_frames'])
            self.m_v_idx.extend(data['m_v_idx'])
            self.m_j3_indices.extend(data['m_j3_indices'])
            self.m_j3_weights.extend(data['m_j3_weights'])
            self.m_j3_offsets.extend(data['m_j3_offsets'])
            self.bind_v_shaped.extend(data['bind_v_shaped'])
            self.bind_vn.extend(data['bind_vn'])
            self.bind_jlp.extend(data['bind_jlp'])
            self.bind_jgp.extend(data['bind_jgp'])
            self.ghost_marker_mask.extend(data['ghost_marker_mask'])
            self.markers.extend(data['markers'])
            self.poses.extend(data['poses'])
            self.jgp.extend(data['jgp'])

        self.stacked_num_frames = np.cumsum(self.n_frames)
        self.total_num_frames = sum(self.n_frames)
        self.length = len(self.n_frames)

        print(f'INFO | Dataset | Total Motion num: {self.length}')
        print(f'INFO | Dataset | Total frame length: {self.total_num_frames}')

    def __len__(self):
        if self.test:
            return self.total_num_frames // 10
        else:
            return self.length * 100

    def __getitem__(self, item_idx):
        if self.test:
            actual_idx = item_idx * 10
            motion_idx = np.searchsorted(self.stacked_num_frames - actual_idx, 0, side='right')
            n_frame = self.n_frames[motion_idx]

            if motion_idx == 0:
                frame_idx = actual_idx
            else:
                frame_idx = actual_idx - self.stacked_num_frames[motion_idx - 1]
        else:
            motion_idx = item_idx % self.length
            n_frame = self.n_frames[motion_idx]
            frame_idx = random.randint(0, n_frame - 1)

        half_seq = self.seq_len // 2
        start_count = 0
        start_idx = -1
        end_count = 0
        end_idx = -1

        if frame_idx < half_seq:
            start_count = half_seq - frame_idx
            start_idx = 0
        else:
            start_idx = frame_idx - half_seq
        if frame_idx >= n_frame - half_seq:
            end_count = half_seq - n_frame + frame_idx + 1
            end_idx = n_frame
        else:
            end_idx = frame_idx + half_seq + 1

        if item_idx % 2 == 0:
            return self.getitem_real(
                motion_idx,
                frame_idx,
                start_idx,
                end_idx,
                start_count,
                end_count
            )
        else:
            return self.getitem_synthetic(
                motion_idx,
                frame_idx,
                start_idx,
                end_idx,
                start_count,
                end_count
            )

    def getitem_real(self, mi, fi, si, ei, sc, ec):
        n_joints = self.n_joints
        n_markers = self.markers[mi].shape[1]

        points_seq = []
        points_mask = []
        padded_m_j_weights = None
        padded_m_j3_weights = None
        padded_m_j3_offsets = None

        for _ in range(sc):
            points_seq.append(torch.zeros(1, self.n_max_markers, 3))
            points_mask.append(torch.zeros(1, self.n_max_markers, requires_grad=False))

        for i in range(si, ei):
            points = torch.from_numpy(self.markers[mi][i]).to(torch.float32)
            mag = torch.norm(points, dim=-1)
            n_vm = torch.nonzero(mag > 0.001).shape[0]
            n_vm = min(n_vm, self.n_max_markers)

            padded_points = torch.zeros(self.n_max_markers, 3)
            padded_points[:n_vm, :] = points[:n_vm, :]
            points_seq.append(padded_points.unsqueeze(0))

            mask = torch.zeros(1, self.n_max_markers, requires_grad=False)
            mask[:, :n_vm] = 1
            points_mask.append(mask)

            if i == fi:
                ghost_marker_mask = torch.Tensor(self.ghost_marker_mask[mi][i, :n_vm]).to(torch.float32)

                m_j_weights = torch.Tensor(self.weights[self.m_v_idx[mi][i, :n_vm], :]).to(torch.float32)
                m_j_weights = m_j_weights * ghost_marker_mask[:, None]

                ghost_weights = torch.ones((self.n_max_markers, 1))
                ghost_weights[:n_vm, 0] -= ghost_marker_mask

                padded_m_j_weights = F.pad(
                    input=m_j_weights,
                    pad=[0, 0, 0, self.n_max_markers - n_vm],
                    mode='constant',
                    value=0
                )
                padded_m_j_weights = torch.cat((padded_m_j_weights, ghost_weights), dim=-1)

                m_j3_weights = torch.Tensor(self.m_j3_weights[mi][i, :n_vm]).to(torch.float32)
                m_j3_weights = m_j3_weights * ghost_marker_mask[:, None]

                padded_m_j3_weights = F.pad(
                    input=m_j3_weights,
                    pad=[0, 0, 0, self.n_max_markers - n_vm],
                    mode='constant',
                    value=0
                )
                padded_m_j3_weights[:n_vm, 0] += (1 - ghost_marker_mask)

                m_j3_offsets = torch.Tensor(self.m_j3_offsets[mi][i, :n_vm]).to(torch.float32)
                m_j3_offsets = m_j3_offsets * ghost_marker_mask[:, None, None]

                padded_m_j3_offsets = F.pad(
                    input=m_j3_offsets,
                    pad=[0, 0, 0, 0, 0, self.n_max_markers - n_vm],
                    mode='constant',
                    value=0
                )

        for _ in range(ec):
            points_seq.append(torch.zeros(1, self.n_max_markers, 3))
            points_mask.append(torch.zeros(1, self.n_max_markers, requires_grad=False))

        points_seq = torch.cat(points_seq, dim=0)
        points_mask = torch.cat(points_mask, dim=0)

        items = {
            'points_seq': points_seq,
            'points_mask': points_mask,
            'm_j_weights': padded_m_j_weights,
            'm_j3_weights': padded_m_j3_weights,
            'm_j3_offsets': padded_m_j3_offsets,
            'motion_idx': mi,
            'frame_idx': fi,
            'real': True
        }

        return items

    def getitem_synthetic(self, mi, fi, si, ei, sc, ec):
        n_joints = self.n_joints

        if torch.rand(1) < 0.5:
            use_superset = False
        else:
            use_superset = True

        if use_superset:
            n_selected_superset = np.random.randint(22, len(self.superset_variant))
            selected_superset_indices = np.random.choice(
                np.arange(len(self.superset_variant)),
                n_selected_superset,
                replace=False
            )
            marker_indices = np.empty(n_selected_superset, dtype=int)
            for i in range(n_selected_superset):
                variation = self.superset_variant[selected_superset_indices[i]]
                random_idx = np.random.randint(0, len(variation))
                marker_indices[i] = variation[random_idx]
        else:
            marker_indices = []
            values = [2, 3, 4]
            prob = [0.7, 0.2, 0.1]
            indices_num_list = np.random.choice(values, size=n_joints, p=prob)
            for i in range(n_joints):
                marker_indices.extend(np.random.choice(
                    self.j_v_idx[i],
                    size=indices_num_list[i],
                    replace=False
                ))
            marker_indices = np.array(marker_indices)

        n_body = self.caesar_bind_v.shape[0]
        body_idx = random.randint(0, n_body - 1)

        j3_offsets = self.caesar_v_j3_offsets[body_idx]
        bind_vn = torch.from_numpy(self.caesar_bind_vn[body_idx]).to(torch.float32)
        jgp_seq = torch.from_numpy(self.jgp[mi][si:ei]).to(torch.float32)
        poses = torch.from_numpy(self.poses[mi][si:ei]).to(torch.float32)
        jgr_seq = batch_rodrigues(poses.view(-1, 3)).view(ei-si, self.n_joints, 3, 3)

        for i, pi in enumerate(self.topology):
            if i == 0:
                continue
            jgr_seq[:, i] = jgr_seq[:, pi] @ jgr_seq[:, i]

        theta = math.radians(random.uniform(0, 360))
        aug_rot_mat = torch.tensor([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ]).to(torch.float32)

        points_seq = []
        points_mask = []
        padded_m_j_weights = None
        padded_m_j3_weights = None
        padded_m_j3_offsets = None

        if self.dist_augmentation:
            aug_range = 0.005
            aug_offset = (torch.rand(1) - 0.5) * aug_range
            dist_from_skin = self.dist_from_skin + aug_offset
        else:
            dist_from_skin = self.dist_from_skin

        for _ in range(sc):
            points_seq.append(torch.zeros(1, self.n_max_markers, 3))
            points_mask.append(torch.zeros(1, self.n_max_markers, requires_grad=False))

        for i in range(si, ei):
            temp_marker_indices = copy.deepcopy(marker_indices)
            if self.noise_occlusion:
                # Randomly select the number of markers to delete.
                n_occlusion = random.randint(0, 5)
                # Select markers to delete amounting to n_occlusion from marker_indices.
                occlusion_indices = np.random.choice(len(temp_marker_indices), size=n_occlusion, replace=False)
                # Delete the selected markers.
                temp_marker_indices = np.delete(temp_marker_indices, occlusion_indices)

            temp_marker_indices = torch.from_numpy(temp_marker_indices).to(torch.int32)

            m_j3_indices = torch.from_numpy(self.v_j3_indices[temp_marker_indices, :]).to(torch.int32)
            m_j3_weights = torch.from_numpy(self.v_j3_weights[temp_marker_indices, :]).to(torch.float32)
            m_j3_offsets = torch.from_numpy(j3_offsets[temp_marker_indices, :, :]).to(torch.float32)

            jgr = jgr_seq[i - si]
            j3gr = jgr[m_j3_indices, :, :]
            jgp = jgp_seq[i - si]
            j3gp = jgp[m_j3_indices, :]

            points = j3gr @ m_j3_offsets[:, :, :, None]
            points = torch.squeeze(points)
            points += j3gp
            points = torch.sum(points * m_j3_weights[:, :, None], dim=1)

            points_norm = j3gr @ bind_vn[temp_marker_indices, None, :, None]
            points_norm = torch.sum(torch.squeeze(points_norm) * m_j3_weights[:, :, None], dim=1)

            points += (points_norm * dist_from_skin)

            n_clean_markers = len(temp_marker_indices)

            if self.noise_jitter:
                jitter_range = 0.01
                jitter = (torch.rand_like(points) - 0.5) * jitter_range
                points += jitter

            if self.noise_ghost:
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

            if n_total_markers > self.n_max_markers:
                points = points[:self.n_max_markers]
                n_total_markers = self.n_max_markers

            points = F.pad(
                input=points,
                pad=[0, 0, 0, self.n_max_markers - n_total_markers],
                mode='constant',
                value=0
            )

            if self.noise_shuffle:
                shuffled_indices = torch.randperm(n_total_markers)
                points[:n_total_markers, :] = points[shuffled_indices, :]

            points_seq.append(points.unsqueeze(0))
            mask = torch.zeros(1, self.n_max_markers, requires_grad=False)

            mask[:, :n_total_markers] = 1
            points_mask.append(mask)

            if i == fi:
                assert padded_m_j_weights is None, ValueError(
                    f"Center frame is already set.")

                ghost_weights = torch.ones((self.n_max_markers, 1))
                ghost_weights[:n_clean_markers, 0] = 0

                padded_m_j_weights = torch.from_numpy(self.weights[temp_marker_indices, :]).to(torch.float32)
                padded_m_j_weights = F.pad(
                    input=padded_m_j_weights,
                    pad=[0, 0, 0, self.n_max_markers - n_clean_markers],
                    mode='constant',
                    value=0
                )
                padded_m_j_weights = torch.cat((padded_m_j_weights, ghost_weights), dim=-1)

                padded_m_j3_weights = F.pad(
                    input=m_j3_weights,
                    pad=[0, 0, 0, self.n_max_markers - n_clean_markers],
                    mode='constant',
                    value=0
                )
                padded_m_j3_weights[n_clean_markers:, 0] = 1

                padded_m_j3_offsets = F.pad(
                    input=m_j3_offsets,
                    pad=[0, 0, 0, 0, 0, self.n_max_markers - n_clean_markers],
                    mode='constant',
                    value=0
                )

                if self.noise_shuffle:
                    padded_m_j_weights[:n_total_markers, :] = padded_m_j_weights[shuffled_indices, :]
                    padded_m_j3_weights[:n_total_markers, :] = padded_m_j3_weights[shuffled_indices, :]
                    padded_m_j3_offsets[:n_total_markers, :, :] = padded_m_j3_offsets[shuffled_indices, :, :]

        for _ in range(ec):
            points_seq.append(torch.zeros(1, self.n_max_markers, 3))
            points_mask.append(torch.zeros(1, self.n_max_markers, requires_grad=False))

        points_seq = torch.cat(points_seq, dim=0)
        points_mask = torch.cat(points_mask, dim=0)

        points_seq = torch.matmul(points_seq, aug_rot_mat)

        items = {
            'points_seq': points_seq,
            'points_mask': points_mask,
            'm_j_weights': padded_m_j_weights,
            'm_j3_weights': padded_m_j3_weights,
            'm_j3_offsets': padded_m_j3_offsets,
            'motion_idx': mi,
            'frame_idx': fi,
            'real': False
        }

        return items
