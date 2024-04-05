import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from modules.training.damo_trainer import DamoTrainer
from modules.training.training_options import load_options_from_json
from modules.utils.paths import Paths
from modules.dataset.damo_dataset import DamoDataset
from modules.network.damo import Damo
from modules.utils.functions import dict2class
from modules.solver.pose_solver import find_bind_pose, find_pose, test_lbs
from human_body_prior.body_model.lbs import batch_rodrigues

import time
import keyboard
import vpython.no_notebook
from vpython import canvas, sphere, vector, color, cylinder, arrow, text


json_path = Paths.support_data / "train_options.json"
options = load_options_from_json(json_path)

if options.seed is not None:
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

data = DamoDataset(
    common_dataset_path=options.common_dataset_path,
    dataset_paths=options.test_dataset_paths,
    n_max_markers=options.n_max_markers,
    seq_len=options.seq_len,
    r_ss_ds_ratio=[0, 1, 0],
    noise_jitter=True,
    noise_ghost=True,
    noise_occlusion=True,
    noise_shuffle=True,
    dist_from_skin=0.01,
    dist_augmentation=True,
    test=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_printoptions(precision=6, sci_mode=False)

window = canvas(x=0, y=0, width=1200, height=1200, center=vector(0, 0, 0), background=vector(0, 0, 0))
axis_x = arrow(pos=vector(0, 0, 0), axis=vector(1, 0, 0), shaftwidth=0.05, color=vpython.color.red)
axis_y = arrow(pos=vector(0, 0, 0), axis=vector(0, 1, 0), shaftwidth=0.05, color=vpython.color.green)
axis_z = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 1), shaftwidth=0.05, color=vpython.color.blue)

window.forward = vector(-1, 0, 0)
window.up = vector(0, 0, 1)

n_max_markers = 90
n_joints = 24

v_markers = [
    sphere(radius=0.01, color=color.cyan)
    for i in range(n_max_markers)
]

v_vmarkers = [
    sphere(radius=0.01, color=color.blue)
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

v_bind_joints = [
    sphere(radius=0.02, color=color.orange)
    for i in range(n_joints)
]
v_bind_bones = [
    cylinder(radius=0.01,
             color=color.orange)
    for i in range(1, n_joints)
]


for i in range(len(data)):
    items = data[i]
    items = dict2class(items)
    points = items.points_seq[3].to(device)
    indices = items.m_j_weights.to(device)
    j3_weights = items.m_j3_weights.to(device)
    j3_offsets = items.m_j3_offsets.to(device)
    bind_jgp = items.bind_jgp.to(device)
    poses = items.poses.to(device)
    trans = items.trans.to(device)

    n_max_markers = points.shape[0]
    n_joints = 24
    topology = torch.Tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).to(torch.int32)
    topology = topology.to(device)

    j3_indices = torch.argsort(indices, dim=-1, descending=True)[..., :3]
    j_weights = torch.zeros(n_max_markers, n_joints + 1).to(device)
    j_offsets = torch.zeros(n_max_markers, n_joints + 1, 3).to(device)

    j_weights[
        torch.arange(n_max_markers)[:, None],
        j3_indices
    ] = j3_weights
    j_weights = j_weights[:, :n_joints]

    j_offsets[
        torch.arange(n_max_markers)[:, None],
        j3_indices
    ] = j3_offsets
    j_offsets = j_offsets[:, :n_joints]

    bind_jlp = torch.zeros(n_joints, 3).to(device)
    bind_jlp[1:] = bind_jgp[1:] - bind_jgp[topology[1:]]

    jgt = torch.eye(4).repeat(n_joints, 1, 1).to(device)
    jlr = batch_rodrigues(poses.view(-1, 3)).view(-1, 3, 3)
    jgt[:, :3, 3] = bind_jlp
    jgt[0, :3, 3] = trans
    jgt[:, :3, :3] = jlr

    for j, pi in enumerate(topology):
        if j == 0:
            assert pi == -1
            continue

        jgt[j] = torch.matmul(jgt[pi], jgt[j])
    jgp = jgt[:, :3, 3]

    # virtual_points = test_lbs(jgt, j_weights, j_offsets)

    params, jgt, virtual_points = find_pose(topology, bind_jlp, points, j_weights, j_offsets)
    jgr = jgt[:, :3, :3]
    jgp = jgt[:, :3, 3]

    while True:
        for j in range(n_joints):
            v_joints[j].pos = vector(*jgp[j])

        for j in range(1, n_joints):
            v_bones[j - 1].pos = v_joints[j].pos
            v_bones[j - 1].axis = (v_joints[topology[j]].pos - v_joints[j].pos)

        for j in range(n_max_markers):
            v_markers[j].pos = vector(*points[j])
            v_vmarkers[j].pos = vector(*virtual_points[j])

        if keyboard.is_pressed('right'):
            break

        # f += 1
        time.sleep(1 / 60)