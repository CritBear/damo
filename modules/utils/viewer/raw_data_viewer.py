import numpy as np
import torch
import c3d
import keyboard

from modules.utils.paths import Paths
from modules.utils.functions import dict2class
from human_body_prior.body_model.body_model_prior import BodyModel

import time
import vpython.no_notebook
from vpython import *


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


def to_gpu(ndarray):
    return torch.Tensor(ndarray).to(device)


def read_c3d_markers(file_path):
    reader = c3d.Reader(open(file_path, 'rb'))
    marker_data = []

    for i, points, analog in reader.read_frames():
        marker_data.append(points)

    marker_data = np.array(marker_data)

    return marker_data[:, :, :3]


window = canvas(x=0, y=0, width=1200, height=800, center=vector(0, 0, 0), background=vector(0, 0, 0))
axis_x = arrow(pos=vector(0, 0, 0), axis=vector(1, 0, 0), shaftwidth=0.05, color=vpython.color.red)
axis_y = arrow(pos=vector(0, 0, 0), axis=vector(0, 1, 0), shaftwidth=0.05, color=vpython.color.green)
axis_z = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 1), shaftwidth=0.05, color=vpython.color.blue)

window.forward = vector(-1, 0, 0)
window.up = vector(0, 0, 1)

dataset_name = 'CMU'
data_dir = Paths.datasets / 'raw' / dataset_name
c3d_files = list(data_dir.rglob('*.c3d'))
npz_files = list(data_dir.rglob('*.npz'))

c3d_file = c3d_files[22]
npz_file = npz_files[22]

num_betas = 10
num_dmpls = 8
topology = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
)
amass = np.load(npz_file, allow_pickle=True)
amass = dict2class(amass)
smplh_model_path = Paths.support_data / 'body_models' / 'smplh' / amass.gender / 'model.npz'
dmpl_model_path = Paths.support_data / 'body_models' / 'dmpls' / amass.gender / 'model.npz'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bm = BodyModel(
    bm_fname=smplh_model_path.as_posix(),
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=dmpl_model_path.as_posix(),
    model_type='smpl'
).to(device)
n_frames = len(amass.trans)
bp = {
    'root_orient': to_gpu(amass.poses[:, :3]),
    'pose_body': to_gpu(amass.poses[:, 3:66]),
    'pose_hand': to_gpu(amass.poses[:, 66:]),
    'trans': to_gpu(amass.trans),
    'betas': to_gpu(np.repeat(amass.betas[:num_betas][np.newaxis], repeats=n_frames, axis=0)),
    # 'dmpls': to_gpu(amass.dmpls[:, :num_dmpls])
}
motion = bm(**{k: v for k, v in bp.items() if k in ['pose_body', 'betas',
                                                            'trans', 'root_orient']})
n_joints = 24
markers = read_c3d_markers(c3d_file)
markers /= 1000

# markers = markers[:, :, [0, 2, 1]]
# markers[:, :, 1] *= -1

assert n_frames == markers.shape[0]
n_max_markers = markers.shape[1]

v_markers = [
    sphere(radius=0.01, color=color.cyan)
    for i in range(n_max_markers)
]

v_joints = [
    sphere(radius=0.02, color=color.orange)
    for i in range(n_joints)
]
v_bones = [
    cylinder(radius=0.01,
             color=color.orange)
    for i in range(1, n_joints)
]

f = 0
while True:

    for i in range(n_joints):
        v_joints[i].pos = vector(*motion.Jtr[f, i])

    for i in range(1, n_joints):
        v_bones[i - 1].pos = v_joints[i].pos
        v_bones[i - 1].axis = (v_joints[topology[i]].pos - v_joints[i].pos)

    for i in range(n_max_markers):
        v_markers[i].pos = vector(*markers[f, i])


    if keyboard.is_pressed('right'):
        f = (f + 1) % n_frames

    if keyboard.is_pressed("left"):
        f = (f - 1) % n_frames

    # f += 1
    time.sleep(1 / 60)