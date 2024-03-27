import numpy as np
import pickle

from modules.utils.paths import Paths

import vpython.no_notebook
from vpython import *
import time
import keyboard


with open(Paths.datasets / 'eval' / 'eval_data_for_damo' / 'eval_data_damo-soma_HDM05_jgos.pkl', 'rb') as f:
    data = pickle.load(f)

n_motions = len(data['joint_local_positions'])
m = 0

jlp = data['joint_local_positions'][m]
markers = data['synthetic_marker'][m]
jlr = data['joint_local_rotations'][m]
topology = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
n_joints = len(topology)
n_frames = jlr.shape[0]
n_max_markers = markers.shape[1]

jt = np.tile(np.eye(4), (n_frames, n_joints, 1, 1))
jt[:, :, :3, 3] = jlp[np.newaxis, :, :]
jt[:, :, :3, :3] = jlr.copy()

rest_jgp = jlp.copy()
for i, pi in enumerate(topology):
    if i == 0:
        assert pi == -1
        continue

    jt[:, i] = jt[:, pi] @ jt[:, i]
    rest_jgp[i] += rest_jgp[pi]

# jgp = data['joint_global_positions'][m]
jgp = jt[:, :, :3, 3]
jgr = jt[:, :, :3, :3]

window = canvas(x=0, y=0, width=1200, height=1200, center=vector(0, 0, 0), background=vector(0, 0, 0))
axis_x = arrow(pos=vector(0, 0, 0), axis=vector(1, 0, 0), shaftwidth=0.05, color=vpython.color.red)
axis_y = arrow(pos=vector(0, 0, 0), axis=vector(0, 1, 0), shaftwidth=0.05, color=vpython.color.green)
axis_z = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 1), shaftwidth=0.05, color=vpython.color.blue)

window.forward = vector(-1, 0, 0)
window.up = vector(0, 0, 1)

v_markers = [
    sphere(radius=0.01, color=color.cyan)
    for i in range(n_max_markers)
]

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

    # time.sleep(1/60)
    # continue

    for i in range(n_max_markers):
        v_markers[i].pos = vector(*markers[f, i])

    for j in range(n_joints):
        v_joints[j].pos = vector(*jgp[f, j])
        v_joints_axis_x[j].pos = v_joints[j].pos
        v_joints_axis_y[j].pos = v_joints[j].pos
        v_joints_axis_z[j].pos = v_joints[j].pos
        v_joints_axis_x[j].axis = vector(*jgr[f, j, :3, 0]) * 0.1
        v_joints_axis_y[j].axis = vector(*jgr[f, j, :3, 1]) * 0.1
        v_joints_axis_z[j].axis = vector(*jgr[f, j, :3, 2]) * 0.1

    for i in range(1, n_joints):
        v_bones[i-1].pos = v_joints[i].pos
        v_bones[i-1].axis = (v_joints[topology[i]].pos - v_joints[i].pos)

    if keyboard.is_pressed('right'):
        f = (f + 1) % n_frames

    if keyboard.is_pressed("left"):
        f = (f - 1) % n_frames

    time.sleep(1/60)
