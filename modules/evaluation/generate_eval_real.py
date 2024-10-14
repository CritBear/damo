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

    data_date = '20240329'

    for data_name in data_names:
        eval_dataset = {}
        eval_data = generate_eval_real(data_date, data_name)

        eval_dataset['real'] = eval_data

        save_dir = Paths.datasets / 'eval'
        save_path = save_dir / f'damo_eval_real_{data_name}.pkl'

        with open(save_path, 'wb') as f:
            pickle.dump(eval_dataset, f)


def generate_eval_real(data_date, data_name):
    data_dir = Paths.datasets / 'batch' / data_date / data_name
    data_paths = list(data_dir.glob('*.pkl'))

    common_data_path = Paths.datasets / 'common' / f'damo_common_{data_date}.pkl'

    with open(common_data_path, 'rb') as f:
        common_data = pickle.load(f)

    topology = common_data['topology']
    weights = common_data['weights']

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
        print(f'\n{data_name} | real | [{file_idx + 1}/{len(data_paths)}] | file name: {data_path.stem}')

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        n_frames = data['n_frames'][0]
        n_joints = 24
        n_max_markers = 90

        points_seq_list = []
        points_mask_list = []
        m_j_weights_list = []
        m_j3_weights_list = []
        m_j3_offsets_list = []

        # eval_data['poses'].append(data['poses'][0][::2])
        # eval_data['jgp'].append(data['jgp'][0][::2])
        eval_data['poses'].append(data['poses'][0])
        eval_data['jgp'].append(data['jgp'][0])
        eval_data['bind_jgp'].append(data['bind_jgp'][0])
        eval_data['bind_jlp'].append(data['bind_jlp'][0])

        # for fi in tqdm(range(0, n_frames, 2)):
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

            for _ in range(sc):
                points_seq.append(torch.zeros(1, n_max_markers, 3))
                points_mask.append(torch.zeros(1, n_max_markers, requires_grad=False))

            for i in range(si, ei):
                points = torch.from_numpy(data['markers'][0][i]).to(torch.float32)
                mag = torch.norm(points, dim=-1)
                n_vm = torch.nonzero(mag > 0.001).shape[0]
                n_vm = min(n_vm, n_max_markers)

                padded_points = torch.zeros(n_max_markers, 3)
                padded_points[:n_vm, :] = points[:n_vm, :]
                points_seq.append(padded_points.unsqueeze(0))

                mask = torch.zeros(1, n_max_markers, requires_grad=False)
                mask[:, :n_vm] = 1
                points_mask.append(mask)

                if i == fi:
                    ghost_marker_mask = torch.Tensor(data['ghost_marker_mask'][0][i, :n_vm]).to(torch.float32)

                    m_j_weights = torch.Tensor(weights[data['m_v_idx'][0][i, :n_vm], :]).to(torch.float32)
                    m_j_weights = m_j_weights * ghost_marker_mask[:, None]

                    ghost_weights = torch.ones((n_max_markers, 1))
                    ghost_weights[:n_vm, 0] -= ghost_marker_mask

                    padded_m_j_weights = F.pad(
                        input=m_j_weights,
                        pad=[0, 0, 0, n_max_markers - n_vm],
                        mode='constant',
                        value=0
                    )
                    padded_m_j_weights = torch.cat((padded_m_j_weights, ghost_weights), dim=-1)

                    m_j3_weights = torch.Tensor(data['m_j3_weights'][0][i, :n_vm]).to(torch.float32)
                    m_j3_weights = m_j3_weights * ghost_marker_mask[:, None]

                    padded_m_j3_weights = F.pad(
                        input=m_j3_weights,
                        pad=[0, 0, 0, n_max_markers - n_vm],
                        mode='constant',
                        value=0
                    )
                    padded_m_j3_weights[:n_vm, 0] += (1 - ghost_marker_mask)

                    m_j3_offsets = torch.Tensor(data['m_j3_offsets'][0][i, :n_vm]).to(torch.float32)
                    m_j3_offsets = m_j3_offsets * ghost_marker_mask[:, None, None]

                    padded_m_j3_offsets = F.pad(
                        input=m_j3_offsets,
                        pad=[0, 0, 0, 0, 0, n_max_markers - n_vm],
                        mode='constant',
                        value=0
                    )

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