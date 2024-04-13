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
from modules.solver.scipy_pose_solver import find_pose as scipy_find_pose
from human_body_prior.body_model.lbs import batch_rodrigues
from modules.solver.quat_pose_solver_numpy import PoseSolver

import time
import keyboard
import vpython.no_notebook
from vpython import canvas, sphere, vector, color, cylinder, arrow, text, label


seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# "PosePrior", "CMU"
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
    dataset_paths=options.train_dataset_paths,
    n_max_markers=options.n_max_markers,
    seq_len=options.seq_len,
    r_ss_ds_ratio=options.train_data_ratio,
    noise_jitter=True,
    noise_ghost=True,
    noise_occlusion=True,
    noise_shuffle=True,
    dist_from_skin=0.01,
    dist_augmentation=True,
    test=True,
    debug=True,
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

v_ss_markers = [
    sphere(radius=0.01, color=color.orange)
    for i in range(n_max_markers)
]

v_sa_markers = [
    sphere(radius=0.01, color=color.magenta)
    for i in range(n_max_markers)
]


v_bind_markers = [
    sphere(radius=0.01, color=color.green)
    for i in range(n_max_markers)
]

v_weight_labels = [
    label()
    for i in range(n_max_markers)
]

v_ss_weight_labels = [
    label()
    for i in range(n_max_markers)
]

v_sa_weight_labels = [
    label()
    for i in range(n_max_markers)
]

v_vmarkers = [
    sphere(radius=0.01, color=color.blue)
    for i in range(n_max_markers)
]

v_joints = [
    sphere(radius=0.02, color=color.white)
    for i in range(n_joints)
]

v_gt_joints = [
    sphere(radius=0.02, color=color.orange)
    for i in range(n_joints)
]

v_joints_label = [
    label()
    for i in range(n_joints)
]

v_bones = [
    cylinder(radius=0.01,
             color=color.white)
    for i in range(1, n_joints)
]

v_gt_bones = [
    cylinder(radius=0.01,
             color=color.orange)
    for i in range(1, n_joints)
]

v_bind_joints = [
    sphere(radius=0.02, color=color.white)
    for i in range(n_joints)
]
v_bind_bones = [
    cylinder(radius=0.01,
             color=color.white)
    for i in range(1, n_joints)
]


for i in range(len(data)):
    items = data[i]['real']
    items = dict2class(items)
    points = items.points_seq[3].to(device)
    indices = items.m_j_weights.to(device)
    j3_weights = items.m_j3_weights.to(device)
    j3_offsets = items.m_j3_offsets.to(device)
    bind_jgp = items.bind_jgp.to(device)
    poses = items.poses.to(device)
    trans = items.trans.to(device)

    items_ss = dict2class(data[i]['synthetic_superset'])
    items_sa = dict2class(data[i]['synthetic_arbitrary'])
    points_ss = items_ss.points_seq[3].to(device)
    points_sa = items_sa.points_seq[3].to(device)
    indices_ss = items_ss.m_j_weights.to(device)
    j3_weights_ss = items_ss.m_j3_weights.to(device)
    j3_offsets_ss = items_ss.m_j3_offsets.to(device)
    indices_sa = items_sa.m_j_weights.to(device)
    j3_weights_sa = items_sa.m_j3_weights.to(device)
    j3_offsets_sa = items_sa.m_j3_offsets.to(device)

    n_max_markers = points.shape[0]
    n_joints = 24
    topology = torch.Tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).to(torch.int32)
    topology = topology.to(device)

    j3_indices = torch.argsort(indices, dim=-1, descending=True)[..., :3]  # [m, 3]
    j_weights = torch.zeros(n_max_markers, n_joints + 1).to(device)  # [m, j+1]
    j_offsets = torch.zeros(n_max_markers, n_joints + 1, 3).to(device)  # [m, j+1, 3]

    j3_indices_ss = torch.argsort(indices_ss, dim=-1, descending=True)[..., :3]  # [m, 3]
    j_weights_ss = torch.zeros(n_max_markers, n_joints + 1).to(device)  # [m, j+1]
    j_offsets_ss = torch.zeros(n_max_markers, n_joints + 1, 3).to(device)  # [m, j+1, 3]

    j3_indices_sa = torch.argsort(indices_sa, dim=-1, descending=True)[..., :3]  # [m, 3]
    j_weights_sa = torch.zeros(n_max_markers, n_joints + 1).to(device)  # [m, j+1]
    j_offsets_sa = torch.zeros(n_max_markers, n_joints + 1, 3).to(device)  # [m, j+1, 3]


    j_weights[
        torch.arange(n_max_markers)[:, None],
        j3_indices
    ] = j3_weights
    j_weights = j_weights[:, :n_joints]  # [m, j]

    j_offsets[
        torch.arange(n_max_markers)[:, None],
        j3_indices
    ] = j3_offsets
    j_offsets = j_offsets[:, :n_joints]  # [m, j, 3]

    j_weights_ss[
        torch.arange(n_max_markers)[:, None],
        j3_indices_ss
    ] = j3_weights_ss
    j_weights_ss = j_weights_ss[:, :n_joints]  # [m, j]

    j_offsets_ss[
        torch.arange(n_max_markers)[:, None],
        j3_indices_ss
    ] = j3_offsets_ss
    j_offsets_ss = j_offsets_ss[:, :n_joints]  # [m, j, 3]

    j_weights_sa[
        torch.arange(n_max_markers)[:, None],
        j3_indices_sa
    ] = j3_weights_sa
    j_weights_sa = j_weights_sa[:, :n_joints]  # [m, j]

    j_offsets_sa[
        torch.arange(n_max_markers)[:, None],
        j3_indices_sa
    ] = j3_offsets_sa
    j_offsets_sa = j_offsets_sa[:, :n_joints]  # [m, j, 3]

    # for j in range(n_max_markers):
    #     if j3_indices[j][0] == n_joints:
    #         j3_indices[j][0] = 0
    #         j_weights[j] = 0
    #         j_offsets[j] = 0
    #
    #     if j3_indices[j][1] == n_joints:
    #         j3_indices[j][1] = 0
    #
    #     if j3_indices[j][2] == n_joints:
    #         j3_indices[j][2] = 0

    # j3_indices = np.argsort(indices.cpu().numpy())[:, -1:-4:-1]
    # ja_weight = np.zeros((n_max_markers, n_joints + 1))
    # ja_offset = np.zeros((n_max_markers, n_joints + 1, 3))
    # ja_weight[
    #     np.arange(n_max_markers)[:, np.newaxis],
    #     j3_indices.astype(np.int32)
    # ] = j3_weights.cpu().numpy()
    #
    # ja_offset[
    #     np.arange(n_max_markers)[:, np.newaxis],
    #     j3_indices.astype(np.int32)
    # ] = j3_offsets.cpu().numpy()

    bind_jlp = torch.zeros(n_joints, 3).to(device)
    bind_jlp[1:] = bind_jgp[1:] - bind_jgp[topology[1:]]

    bind_mgp = j_offsets + bind_jgp[None, :, :]
    bind_mgp = torch.sum(bind_mgp * j_weights[:, :, None], dim=1)

    gt_jgt = torch.eye(4).repeat(n_joints, 1, 1).to(device)
    gt_jlr = batch_rodrigues(poses.view(-1, 3)).view(-1, 3, 3)
    gt_jgt[:, :3, 3] = bind_jlp
    gt_jgt[0, :3, 3] = trans
    gt_jgt[:, :3, :3] = gt_jlr

    print(poses.view(-1, 3)[0])

    for j, pi in enumerate(topology):
        if j == 0:
            assert pi == -1
            continue

        gt_jgt[j] = torch.matmul(gt_jgt[pi], gt_jgt[j])
    gt_jgp = gt_jgt[:, :3, 3]
    #
    #
    # virtual_points = test_lbs(jgt, j_weights, j_offsets)

    # solver = PoseSolver(topology=topology.cpu().numpy(), gt_skeleton_template=bind_jlp.cpu().numpy())
    # param, jgt = solver.solve(points.cpu().numpy(), ja_weight[:, :n_joints], ja_offset[:, :n_joints])
    # virtual_points = solver.lbs(jgt, ja_weight[:, :n_joints], ja_offset[:, :n_joints])

    params, jgt, virtual_points = scipy_find_pose(topology.cpu(), bind_jlp.cpu(), points.cpu(), j_weights.cpu(), j_offsets.cpu())
    jgr = jgt[:, :3, :3]
    jgp = jgt[:, :3, 3]

    while True:
        for j in range(n_joints):
            v_joints[j].pos = vector(*jgp[j])
            v_gt_joints[j].pos = vector(*gt_jgp[j])
            v_bind_joints[j].pos = vector(*bind_jgp[j])
            v_joints_label[j].pos = v_joints[j].pos
            v_joints_label[j].text = f'{j}'

        for j in range(1, n_joints):
            v_bones[j - 1].pos = v_joints[j].pos
            v_bones[j - 1].axis = (v_joints[topology[j]].pos - v_joints[j].pos)
            v_gt_bones[j - 1].pos = v_gt_joints[j].pos
            v_gt_bones[j - 1].axis = (v_gt_joints[topology[j]].pos - v_gt_joints[j].pos)
            v_bind_bones[j - 1].pos = v_bind_joints[j].pos
            v_bind_bones[j - 1].axis = (v_bind_joints[topology[j]].pos - v_bind_joints[j].pos)

        for j in range(n_max_markers):
            v_markers[j].pos = vector(*points[j])
            v_ss_markers[j].pos = vector(*points_ss[j])
            v_sa_markers[j].pos = vector(*points_sa[j])
            v_vmarkers[j].pos = vector(*virtual_points[j])
            v_bind_markers[j].pos = vector(*bind_mgp[j])
            v_weight_labels[j].pos = v_markers[j].pos
            v_weight_labels[j].text = f'r {j3_indices[j][0]},{j3_indices[j][1]},{j3_indices[j][2]}'
                                       # f'{j3_weights[j][0]:.2f},{j3_weights[j][1]:.2f},{j3_weights[j][2]:.2f}')
            v_ss_weight_labels[j].pos = v_ss_markers[j].pos
            v_ss_weight_labels[j].text = f'ss {j3_indices_ss[j][0]},{j3_indices_ss[j][1]},{j3_indices_ss[j][2]}'
            v_sa_weight_labels[j].pos = v_sa_markers[j].pos
            v_sa_weight_labels[j].text = f'sa {j3_indices_sa[j][0]},{j3_indices_sa[j][1]},{j3_indices_sa[j][2]}'

        if keyboard.is_pressed('right'):
            break

        # f += 1
        time.sleep(1 / 60)