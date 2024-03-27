import numpy as np
import pickle

import torch
from scipy.spatial.transform import Rotation as R
from smplx.lbs import batch_rodrigues, transform_mat

from modules.utils.paths import Paths

import vpython.no_notebook
from vpython import *
import time
import keyboard
import copy


data_path = Paths.datasets / 'raw/mocap_npz/SFU/0005/0005_Jogging001_poses.npz'
data = np.load(data_path)

# root_pos	shape: (1377, 3)
# root_rot	shape: (1377, 3)
# joint_positions	shape: (1377, 52, 3)
# joint_rotations	shape: (1377, 52, 3, 3)
# init_vertices	shape: (6890, 3)
# init_vertex_normals	shape: (6890, 3)
# init_joint	shape: (52, 3)

raw_data_path = Paths.datasets / 'raw/SFU/SFU/0005/0005_Jogging001_poses.npz'
raw_data = np.load(raw_data_path)

# trans	shape: (1377, 3)
# gender	shape: ()
# mocap_framerate	shape: ()
# betas	shape: (16,)
# dmpls	shape: (1377, 8)
# poses	shape: (1377, 156)

smplh_data = np.load(Paths.support_data / 'smplh' / 'neutral' / 'model.npz')
topology = smplh_data['kintree_table'][0]
topology[0] = -1

n_joints = len(topology)

window = canvas(x=0, y=0, width=1200, height=1200, center=vector(0, 0, 0), background=vector(0, 0, 0))
axis_x = arrow(pos=vector(0, 0, 0), axis=vector(1, 0, 0), shaftwidth=0.05, color=vpython.color.red)
axis_y = arrow(pos=vector(0, 0, 0), axis=vector(0, 1, 0), shaftwidth=0.05, color=vpython.color.green)
axis_z = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 1), shaftwidth=0.05, color=vpython.color.blue)

window.forward = vector(-1, 0, 0)
window.up = vector(0, 0, 1)


rest_jgp = data['init_joint']
rest_jlp = np.zeros_like(rest_jgp)
rest_jlp[1:] = rest_jgp[1:] - rest_jgp[topology[1:]]

jlr = data['joint_rotations']
n_frames = jlr.shape[0]

# poses_torch = torch.from_numpy(raw_data['poses'])
# rot_mat = batch_rodrigues(poses_torch.view(-1, 3)).view([-1, 3, 3])
# rest_jlp = np.tile(rest_jlp, (n_frames, 1, 1))
# trans_torch = torch.from_numpy(rest_jlp).view([-1, 3, 1])
# transform = transform_mat(rot_mat, trans_torch).view(-1, n_joints, 4, 4)
#
# jlr = rot_mat.view(n_frames, n_joints, 3, 3).detach().cpu().numpy()
# jt = transform.detach().cpu().numpy()

poses = raw_data['poses'].reshape(-1, n_joints, 3)
n_frames = poses.shape[0]
jlr = R.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(n_frames, n_joints, 3, 3)

jt = np.tile(np.eye(4), (n_frames, n_joints, 1, 1))
jt[:, :, :3, 3] = rest_jlp[np.newaxis, :, :]
jt[:, :, :3, :3] = copy.deepcopy(jlr)
# jt[:, 0, :3, 3] = data['root_pos']

for i, pi in enumerate(topology):
    if i == 0:
        assert pi == -1
        continue

    jt[:, i] = jt[:, pi] @ jt[:, i]

jgp = jt[:, :, :3, 3]

v_joints = [
    sphere(pos=vector(*rest_jgp[i]), radius=0.02, color=color.white)
    for i in range(n_joints)
]
v_bones = [
    cylinder(pos=v_joints[i].pos, axis=(v_joints[topology[i]].pos - v_joints[i].pos), radius=0.01,
             color=color.white)
    for i in range(1, n_joints)
]

v_joints_axis_x = [
    arrow(shaftwidth=0.005, color=vector(1, 0, 0))
    for _ in range(n_joints)
]
v_joints_axis_y = [
    arrow(shaftwidth=0.005, color=vector(0, 1, 0))
    for _ in range(n_joints)
]
v_joints_axis_z = [
    arrow(shaftwidth=0.005, color=vector(0, 0, 1))
    for _ in range(n_joints)
]

f = 0
while True:
    for j in range(n_joints):
        v_joints[j].pos = vector(*jgp[f, j])
        v_joints_axis_x[j].pos = v_joints[j].pos
        v_joints_axis_y[j].pos = v_joints[j].pos
        v_joints_axis_z[j].pos = v_joints[j].pos
        v_joints_axis_x[j].axis = vector(*jlr[f, j, :3, 0]) * 0.1
        v_joints_axis_y[j].axis = vector(*jlr[f, j, :3, 1]) * 0.1
        v_joints_axis_z[j].axis = vector(*jlr[f, j, :3, 2]) * 0.1

    for i in range(1, n_joints):
        v_bones[i-1].pos = v_joints[i].pos
        v_bones[i-1].axis = (v_joints[topology[i]].pos - v_joints[i].pos)

    if keyboard.is_pressed('right'):
        f = (f + 1) % n_frames

    if keyboard.is_pressed("left"):
        f = (f - 1) % n_frames

    time.sleep(1/120)

# for i in data:
#     print(i, end='\t')
#     if hasattr(data[i], 'shape'):
#         print('shape: ' + str(data[i].shape))
#     elif hasattr(data[i], '__len__'):
#         print('len: ' + str(len(data[i])), end='')
#         if hasattr(data[i][0], 'shape'):
#             print(f', shape: {data[i][0].shape}', end='')
#         print('')
#     else:
#         print(data[i])