import numpy as np
import torch
import pickle
import c3d
import random

from modules.utils.paths import Paths


def generate_eval_dataset():
    raw_data_dirs = [
        "HDM05",
        "SFU",
        "MoSh",
        "SOMA"
    ]

    for dataset_name in raw_data_dirs:
        generate_eval_dataset_entry(dataset_name)


def generate_eval_dataset_entry(dataset_name):
    data_dir = Paths.datasets / 'eval' / 'raw' / dataset_name
    c3d_files = list(data_dir.rglob('*.c3d'))
    npz_files = list(data_dir.rglob('*.npz'))

    target_npz_files = []
    for file_idx, c3d_file in enumerate(c3d_files):
        target_npz_file_path = [
            f'{c3d_file.parts[-2]}/{(c3d_file.stem + suffix)}'
            for suffix in ['_poses', '_stageii', '_C3D_poses']
        ]

        found = False
        for i, npz_file in enumerate(npz_files):
            if f'{npz_file.parts[-2]}/{(npz_file.stem)}' in target_npz_file_path:
                target_npz_files.append(npz_file)
                found = True
                break

        assert found is True, f'Index: {file_idx} | {c3d_file}'

    for file_idx, c3d_file in enumerate(c3d_files):
        pass


if __name__ == '__main__':
    torch.set_printoptions(precision=8, sci_mode=False)
    np.set_printoptions(precision=8, suppress=True)
    generate_eval_dataset()