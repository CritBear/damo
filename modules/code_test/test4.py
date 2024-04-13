import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

from modules.utils.paths import Paths


raw_data_dirs = ['PosePrior', 'HDM05', 'SOMA', 'DanceDB', 'CMU', 'SFU']


joint = []

for dataset_name in raw_data_dirs:
    data_dir = Paths.datasets / 'raw' / dataset_name

    npz_files = list(data_dir.rglob('*.npz'))

    for i in tqdm(range(len(npz_files)), file=sys.stdout):
        amass = np.load(npz_files[i], allow_pickle=True)

        if 'poses' not in amass.keys():
            continue

        poses = amass['poses']

        n_frames = poses.shape[0]

        euler = R.from_rotvec(poses.reshape((n_frames, -1, 3)).reshape((-1, 3)))\
            .as_euler('xyz', degrees=True).reshape((n_frames, -1, 3))

        joint.append(euler[:, :, :])

joint = np.concatenate(joint, axis=0)

for i in range(joint.shape[1]):
    # x, y, z 축별로 히스토그램 생성
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1행 3열의 subplot 생성

    # x축 데이터 히스토그램
    axs[0].hist(joint[:, i, 0], bins=40, color='r', alpha=0.7)
    axs[0].set_title('X Axis Distribution')

    # y축 데이터 히스토그램
    axs[1].hist(joint[:, i, 1], bins=40, color='g', alpha=0.7)
    axs[1].set_title('Y Axis Distribution')

    # z축 데이터 히스토그램
    axs[2].hist(joint[:, i, 2], bins=40, color='b', alpha=0.7)
    axs[2].set_title('Z Axis Distribution')

    plt.tight_layout()  # subplot 간 간격 자동 조정
    plt.suptitle(f'Joint {i}', fontsize=16, fontweight='bold')
    plt.show()