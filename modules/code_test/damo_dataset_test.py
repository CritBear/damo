import numpy as np
import pickle
import keyboard

import vpython.no_notebook
from vpython import *
import time

from modules.utils.paths import Paths
from modules.dataset.damo_dataset_legacy import DAMO_Dataset


frame_idx = 0
dataset = DAMO_Dataset(
    data_path=Paths.datasets / 'damo0704_dataset_train.pkl',
    n_max_markers=90,
    seq_len=7,
    marker_shuffle=True,
    marker_occlusion=True,
    marker_ghost=True,
    marker_jittering=True,
    distance_from_skin=0.0095,
    distance_augmentation=True,
    test=True,
    superset_only=False
)

window = canvas(x=0, y=0, width=1200, height=1200, center=vector(0, 0, 0), background=vector(0, 0, 0))
axis_x = arrow(pos=vector(0, 0, 0), axis=vector(1, 0, 0), shaftwidth=0.05, color=vpython.color.red)
axis_y = arrow(pos=vector(0, 0, 0), axis=vector(0, 1, 0), shaftwidth=0.05, color=vpython.color.green)
axis_z = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 1), shaftwidth=0.05, color=vpython.color.blue)

data = dataset[frame_idx]

points = data['points_seq'][7 // 2, :, :]  # 7 // 2

pil = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

n_max_markers = 90
n_joints = 22

ja_weight = np.array(data['marker_joint_weights'])
j3_indices = np.argsort(ja_weight[:, :])[:, -1:-4:-1]
j3_weight = np.array(data['j3_weights'])

v_markers = [
    sphere(pos=vector(*points[i]), radius=0.01, color=color.cyan)
    for i in range(90)
]
v_joints = [
    sphere(pos=vector(*data['joint_position'][i]), radius=0.02, color=color.white)
    for i in range(22)
]
v_bones = [
    cylinder(pos=v_joints[i].pos, axis=(v_joints[pil[i]].pos - v_joints[i].pos), radius=0.01,
             color=color.white)
    for i in range(1, 22)
]
gt_joints = [
    sphere(radius=0.02, color=color.orange)
    for i in range(22)
]
gt_bones = [
    cylinder(radius=0.01,
             color=color.orange)
    for i in range(1, 22)
]

v_weights = [
    cylinder(radius=0.005, color=color.white)
    for _ in range(n_max_markers * 3)
]

for m in range(n_max_markers):
    if v_markers[m].pos.mag < 0.01:
        continue

    for i in range(3):
        if j3_indices[m, i] == 22:
            v_weights[m * 3 + i].opacity = 0
        else:
            v_weights[m * 3 + i].pos = v_markers[m].pos
            v_weights[m * 3 + i].axis = v_joints[j3_indices[m, i]].pos - v_markers[m].pos
            v_weights[m * 3 + i].color = vector(1, 1, 1) * j3_weight[m, i]
            v_weights[m * 3 + i].opacity = j3_weight[m, i]

# v_offsets = [
#     cylinder(radius=0.005, color=color.orange)
#     for i in range(90 * 3)
# ]
#
# for i in range(90):
#     for j in range(3):
#         v_offsets[i * 3 + j].pos = v_markers[i].pos
#         v_offsets[i * 3 + j].axis = vector(*(-data['j3_offsets'][i][j]))

data = dataset[0]
while True:
    print(f'{frame_idx} / {len(dataset)}')
    points = data['points_seq'][7 // 2, :, :]
    ja_weight = np.array(data['marker_joint_weights'])
    j3_indices = np.argsort(ja_weight[:, :])[:, -1:-4:-1]
    j3_weight = np.array(data['j3_weights'])

    for i in range(90):
        v_markers[i].pos = vector(*points[i])

    for i in range(22):
        v_joints[i].pos = vector(*data['joint_position'][i])

    for i in range(1, 22):
        v_bones[i-1].pos = v_joints[i].pos
        v_bones[i-1].axis = (v_joints[pil[i]].pos - v_joints[i].pos)

    for m in range(n_max_markers):
        if v_markers[m].pos.mag < 0.01:
            v_weights[m * 3 + 0].opacity = 0
            v_weights[m * 3 + 1].opacity = 0
            v_weights[m * 3 + 2].opacity = 0
            continue

        for i in range(3):
            if j3_indices[m, i] == 22:
                v_weights[m * 3 + i].opacity = 0
            else:
                v_weights[m * 3 + i].pos = v_markers[m].pos
                v_weights[m * 3 + i].axis = v_joints[j3_indices[m, i]].pos - v_markers[m].pos
                v_weights[m * 3 + i].color = vector(1, 1, 1) * j3_weight[m, i]
                v_weights[m * 3 + i].opacity = j3_weight[m, i]

    if keyboard.is_pressed('right'):
        frame_idx = (frame_idx + 1) % len(dataset)
        data = dataset[frame_idx]

    if keyboard.is_pressed("left"):
        frame_idx = (frame_idx - 1) % len(dataset)
        data = dataset[frame_idx]

    time.sleep(1/60)