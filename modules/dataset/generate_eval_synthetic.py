import torch
import torch.nn.functional as F

import numpy as np
import pickle
import random
import copy
import math
from tqdm import tqdm

from human_body_prior.body_model.lbs import batch_rodrigues
from modules.utils.paths import Paths


def main():
    data_names = [
        "HDM05",
        "SFU",
        "MoSh",
        "SOMA"
    ]

    noises = [
        'j',
        'jg',
        'jgo',
        'jgos'
    ]

    data_date = '20240329'

    for data_name in data_names:

        eval_dataset = {}

        for noise in noises:
            eval_data = generate_eval_synthetic(data_date, data_name, noise)
            eval_dataset[noise] = eval_data

        save_dir = Paths.datasets / 'eval'
        save_path = save_dir / f'damo_eval_synthetic_{data_name}.pkl'

        with open(save_path, 'wb') as f:
            pickle.dump(eval_dataset, f)


def generate_eval_synthetic(data_date, data_name, noise):
    data_dir = Paths.datasets / 'batch' / data_date / data_name
    data_paths = list(data_dir.glob('*.pkl'))

    common_data_path = Paths.datasets / 'common' / f'damo_common_{data_date}.pkl'

    with open(common_data_path, 'rb') as f:
        common_data = pickle.load(f)

    topology = common_data['topology']
    weights = common_data['weights']
    j_v_idx = common_data['j_v_idx']
    n_joints = common_data['n_joints']
    superset_variant = common_data['soma_superset_variant']
    caesar_bind_v = common_data['caesar_bind_v']
    caesar_bind_vn = common_data['caesar_bind_vn']
    caesar_bind_jgp = common_data['caesar_bind_jgp']
    caesar_bind_jlp = common_data['caesar_bind_jlp']
    v_j3_indices = common_data['v_j3_indices']
    v_j3_weights = common_data['v_j3_weights']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu = torch.device('cpu')

    dist_augmentation = True
    dist_from_skin = 0.01
    noise_jitter = True if 'j' in noise else False
    noise_ghost = True if 'g' in noise else False
    noise_occlusion = True if 'o' in noise else False
    noise_shuffle = True if 's' in noise else False

    eval_data = {
        'points_seq': [],
        'points_mask': [],
        'm_j_weights': [],
        'm_j3_weights': [],
        'm_j3_offsets': [],
        'poses': [],
        'jgp': [],
        'bind_jgp': [],
        'bind_jlp': [],
    }

    for file_idx, data_path in enumerate(data_paths):
        print(f'\n{data_name} | {noise} | [{file_idx + 1}/{len(data_paths)}] | file name: {data_path.stem}')

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        n_frames = data['n_frames'][0]
        n_joints = 24
        n_max_markers = 90

        if torch.rand(1) < 0.5:
            use_superset = True
        else:
            use_superset = False

        if use_superset:
            n_selected_superset = np.random.randint(22, len(superset_variant))
            selected_superset_indices = np.random.choice(
                np.arange(len(superset_variant)),
                n_selected_superset,
                replace=False
            )
            marker_indices = np.empty(n_selected_superset, dtype=int)
            for i in range(n_selected_superset):
                variation = superset_variant[selected_superset_indices[i]]
                random_idx = np.random.randint(0, len(variation))
                marker_indices[i] = variation[random_idx]
        else:
            marker_indices = []
            values = [2, 3, 4]
            prob = [0.7, 0.2, 0.1]
            indices_num_list = np.random.choice(values, size=n_joints, p=prob)
            for i in range(n_joints):
                marker_indices.extend(np.random.choice(
                    j_v_idx[i],
                    size=indices_num_list[i],
                    replace=False
                ))
            marker_indices = np.array(marker_indices)

        n_body = caesar_bind_v.shape[0]
        body_idx = random.randint(0, n_body - 1)

        bind_jlp = torch.from_numpy(caesar_bind_jlp[body_idx]).to(torch.float32)
        bind_jgp = torch.from_numpy(caesar_bind_jgp[body_idx]).to(torch.float32)
        trans = torch.from_numpy(data['jgp'][0][:, 0]).to(torch.float32)
        poses = torch.from_numpy(data['poses'][0]).to(torch.float32)
        jlr_seq = batch_rodrigues(poses.view(-1, 3)).view(n_frames, n_joints, 3, 3)

        jgt_seq = torch.eye(4).repeat(n_frames, n_joints, 1, 1)
        jgt_seq[:, :, :3, :3] = jlr_seq
        jgt_seq[:, :, :3, 3] = bind_jlp[None, :, :]
        jgt_seq[:, 0, :3, 3] = trans

        for i, pi in enumerate(topology):
            if i == 0:
                continue
            jgt_seq[:, i] = jgt_seq[:, pi] @ jgt_seq[:, i]

        jgp_seq = jgt_seq[:, :, :3, 3]
        jgr_seq = jgt_seq[:, :, :3, :3]

        eval_data['poses'].append(poses.numpy())
        eval_data['jgp'].append(jgp_seq.numpy())
        eval_data['bind_jgp'].append(bind_jgp.numpy())
        eval_data['bind_jlp'].append(bind_jlp.numpy())

        points_seq_list = []
        points_mask_list = []
        m_j_weights_list = []
        m_j3_weights_list = []
        m_j3_offsets_list = []

        for fi in tqdm(range(n_frames)):
            half_seq = 7 // 2
            sc = 0
            ec = 0

            if fi < half_seq:
                sc = half_seq - fi
                si = 0
            else:
                si = fi - half_seq
            if fi >= n_frames - half_seq:
                sc = half_seq - n_frames + fi + 1
                ei = n_frames
            else:
                ei = fi + half_seq + 1

            points_seq = []
            points_mask = []
            padded_m_j_weights = None
            padded_m_j3_weights = None
            padded_m_j3_offsets = None

            if dist_augmentation:
                aug_range = 0.005
                aug_offset = (torch.rand(1) - 0.5) * aug_range
                dist_from_skin = dist_from_skin + aug_offset
            else:
                dist_from_skin = dist_from_skin

            for _ in range(sc):
                points_seq.append(torch.zeros(1, n_max_markers, 3))
                points_mask.append(torch.zeros(1, n_max_markers, requires_grad=False))

            for i in range(si, ei):
                temp_marker_indices = copy.deepcopy(marker_indices)
                if noise_occlusion:
                    # Randomly select the number of markers to delete.
                    n_occlusion = random.randint(0, 5)
                    # Select markers to delete amounting to n_occlusion from marker_indices.
                    occlusion_indices = np.random.choice(len(temp_marker_indices), size=n_occlusion, replace=False)
                    # Delete the selected markers.
                    temp_marker_indices = np.delete(temp_marker_indices, occlusion_indices)

                temp_marker_indices = torch.from_numpy(temp_marker_indices).to(torch.int32)

                bind_m = torch.from_numpy(caesar_bind_v[body_idx][temp_marker_indices]).to(torch.float32)
                bind_mn = torch.from_numpy(caesar_bind_vn[body_idx][temp_marker_indices]).to(torch.float32)
                bind_disp_m = bind_m + bind_mn * dist_from_skin  # [m, 3]

                if noise_jitter:
                    jitter_range = 0.01
                    jitter = (torch.rand_like(bind_disp_m) - 0.5) * jitter_range
                    bind_disp_m += jitter

                m_j3_indices = torch.from_numpy(v_j3_indices[temp_marker_indices, :]).to(torch.int32)
                # [m, 3]

                m_j3_weights = torch.from_numpy(v_j3_weights[temp_marker_indices, :]).to(torch.float32)
                # [m, 3]

                m_j3_offsets = bind_disp_m[:, None, :] - bind_jgp[m_j3_indices]
                # [m, 3, 3]

                jgr = jgr_seq[i]  # [j, 3, 3]
                j3gr = jgr[m_j3_indices, :, :]  # [m, 3, 3, 3]
                jgp = jgp_seq[i]  # [j, 3]
                j3gp = jgp[m_j3_indices, :]  # [m, 3, 3]

                points = j3gr @ m_j3_offsets[:, :, :, None]  # [m, 3, 3, 1]
                points = torch.squeeze(points)  # [m, 3, 3]
                points += j3gp  # [m, 3, 3]
                points = torch.sum(points * m_j3_weights[:, :, None], dim=1)  # [m, 3]

                n_clean_markers = len(temp_marker_indices)

                if noise_ghost:
                    max_ghost_markers = 2
                    n_ghost_markers = torch.randint(-max_ghost_markers, max_ghost_markers + 1, (1,)).item()
                    n_ghost_markers = max(0, n_ghost_markers)
                    points_center = points.mean(dim=0)
                    ghost_markers_range = 1
                    ghost_markers = (torch.rand(n_ghost_markers, points.shape[-1]) - 0.5) * ghost_markers_range
                    ghost_markers = ghost_markers + points_center
                    points = torch.cat((points, ghost_markers), dim=0)

                    n_total_markers = n_clean_markers + n_ghost_markers
                else:
                    n_total_markers = n_clean_markers

                if n_total_markers > n_max_markers:
                    points = points[:n_max_markers]
                    n_total_markers = n_max_markers

                points = F.pad(
                    input=points,
                    pad=[0, 0, 0, n_max_markers - n_total_markers],
                    mode='constant',
                    value=0
                )

                if noise_shuffle:
                    shuffled_indices = torch.randperm(n_total_markers)
                    points[:n_total_markers, :] = points[shuffled_indices, :]

                points_seq.append(points.unsqueeze(0))
                mask = torch.zeros(1, n_max_markers, requires_grad=False)

                mask[:, :n_total_markers] = 1
                points_mask.append(mask)

                if i == fi:
                    assert padded_m_j_weights is None, ValueError(
                        f"Center frame is already set.")

                    ghost_weights = torch.ones((n_max_markers, 1))
                    ghost_weights[:n_clean_markers, 0] = 0

                    padded_m_j_weights = torch.from_numpy(weights[temp_marker_indices, :]).to(torch.float32)
                    padded_m_j_weights = F.pad(
                        input=padded_m_j_weights,
                        pad=[0, 0, 0, n_max_markers - n_clean_markers],
                        mode='constant',
                        value=0
                    )
                    padded_m_j_weights = torch.cat((padded_m_j_weights, ghost_weights), dim=-1)

                    padded_m_j3_weights = F.pad(
                        input=m_j3_weights,
                        pad=[0, 0, 0, n_max_markers - n_clean_markers],
                        mode='constant',
                        value=0
                    )
                    padded_m_j3_weights[n_clean_markers:, 0] = 1

                    padded_m_j3_offsets = F.pad(
                        input=m_j3_offsets,
                        pad=[0, 0, 0, 0, 0, n_max_markers - n_clean_markers],
                        mode='constant',
                        value=0
                    )

                    if noise_shuffle:
                        padded_m_j_weights[:n_total_markers, :] = padded_m_j_weights[shuffled_indices, :]
                        padded_m_j3_weights[:n_total_markers, :] = padded_m_j3_weights[shuffled_indices, :]
                        padded_m_j3_offsets[:n_total_markers, :, :] = padded_m_j3_offsets[shuffled_indices, :, :]

            for _ in range(ec):
                points_seq.append(torch.zeros(1, n_max_markers, 3))
                points_mask.append(torch.zeros(1, n_max_markers, requires_grad=False))

            points_seq = torch.cat(points_seq, dim=0)
            points_mask = torch.cat(points_mask, dim=0)

            points_seq_list.append(points_seq.unsqueeze(0))
            points_mask_list.append(points_mask.unsqueeze(0))
            m_j_weights_list.append(padded_m_j_weights.unsqueeze(0))
            m_j3_weights_list.append(padded_m_j3_weights.unsqueeze(0))
            m_j3_offsets_list.append(padded_m_j3_offsets.unsqueeze(0))

        eval_data['points_seq'].append(torch.cat(points_seq_list, dim=0).numpy())
        eval_data['points_mask'].append(torch.cat(points_mask_list, dim=0).numpy())
        eval_data['m_j_weights'].append(torch.cat(m_j_weights_list, dim=0).numpy())
        eval_data['m_j3_weights'].append(torch.cat(m_j3_weights_list, dim=0).numpy())
        eval_data['m_j3_offsets'].append(torch.cat(m_j3_offsets_list, dim=0).numpy())

    return eval_data



if __name__ == '__main__':
    main()