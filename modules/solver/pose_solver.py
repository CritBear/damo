import torch
from dataclasses import dataclass
from typing import Callable
from scipy.spatial.transform import Rotation as R

from human_body_prior.body_model.lbs import batch_rodrigues
from modules.utils.paths import Paths
from modules.solver.svd_solver import SVD_Solver
from modules.utils.functions import x_rot_mat, y_rot_mat, z_rot_mat


@dataclass
class ParamInfo:
    joint_idx: int
    get_rot_mat: Callable[[float], torch.Tensor]
    lower_limit: float
    upper_limit: float


def find_bind_pose(topology, base_bind_jgp, points, weights, offsets, gp=False):
    assert points.shape[0] == weights.shape[0] == offsets.shape[0]
    print("finding skeleton...", end='')
    n_frames = points.shape[0]
    n_joints = topology.shape[0]
    svd_solver = SVD_Solver(n_joints)

    # print("base bind jlp")
    # base_bind_jlp = torch.zeros(n_joints, 3)
    # base_bind_jlp[1:] = base_bind_jgp[1:] - base_bind_jgp[topology[1:]]
    # base_bone_lengths = torch.linalg.norm(base_bind_jlp, dim=-1)
    # bind_jlp = base_bind_jlp.clone()
    # return base_bind_jlp

    try:
        # Only use translation
        jgp = svd_solver(points, weights, offsets)[..., 3]

        bone_lengths = [torch.zeros(n_frames) for _ in range(n_joints)]
        for i, pi in enumerate(topology):
            if i == 0:
                assert pi == -1
                continue

            bone_lengths[i] = torch.linalg.norm(jgp[:, i] - jgp[:, pi], dim=-1)

        bone_lengths_mean = torch.zeros(n_joints)
        bone_lengths_std = torch.zeros(n_joints)
        bone_lengths_n_nan = torch.zeros(n_joints)

        for i in range(n_joints):
            bone_lengths[i] = bone_lengths[i][~bone_lengths[i].isnan()]
            bone_lengths_n_nan[i] = n_frames - bone_lengths[i].shape[0]
            bone_lengths[i] = filter_outliers_with_iqr(bone_lengths[i])

            bone_lengths_mean[i] = torch.mean(bone_lengths[i])
            bone_lengths_std[i] = torch.std(bone_lengths[i])

        print('nan count: ', end='')
        print(bone_lengths_n_nan)
        print('bone lengths mean: ', end='')
        print(bone_lengths_mean)
        print('bone lengths std:  ', end='')
        print(bone_lengths_std)

        pair_joints = [
            (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)
        ]
        for pair in pair_joints:
            length_mean = (bone_lengths_mean[pair[0]] + bone_lengths_mean[pair[1]]) / 2
            bone_lengths_mean[pair[0]] = bone_lengths_mean[pair[1]] = length_mean

        base_lengths_sum = 0
        svd_lengths_sum = 0
        target_joints = [4, 5, 7, 8, 20, 21]

        base_bind_jlp = torch.zeros(n_joints, 3)
        base_bind_jlp[1:] = base_bind_jgp[1:] - base_bind_jgp[topology[1:]]
        base_bone_lengths = torch.linalg.norm(base_bind_jlp, dim=-1)
        bind_jlp = base_bind_jlp.clone()

        for i in target_joints:
            if bone_lengths_n_nan[i] < n_frames * 0.1:
                base_lengths_sum += base_bone_lengths[i]
                svd_lengths_sum += bone_lengths_mean[i]
            else:
                print(f"too many nan: {i}, {bone_lengths_n_nan[i]}")

        scale = svd_lengths_sum / base_lengths_sum

        for i in range(1, n_joints):
            if i in target_joints:
                bone_lengths_mean[i] = torch.clamp(
                    bone_lengths_mean[i],
                    base_bone_lengths[i] * scale * 0.7,
                    base_bone_lengths[i] * scale * 1.3
                )
            else:
                bone_lengths_mean[i] = base_bone_lengths[i] * scale

            bind_jlp[i] *= bone_lengths_mean[i]

        bind_jlp = base_bind_jlp

        if gp is False:
            return bind_jlp
    except:
        print("base bind jlp")
        base_bind_jlp = torch.zeros(n_joints, 3)
        base_bind_jlp[1:] = base_bind_jgp[1:] - base_bind_jgp[topology[1:]]
        base_bone_lengths = torch.linalg.norm(base_bind_jlp, dim=-1)
        bind_jlp = base_bind_jlp.clone()
        return base_bind_jlp


def filter_outliers_with_iqr(data, outlier_factor=1.5):
    q1 = torch.quantile(data, 0.25)
    q3 = torch.quantile(data, 0.75)

    iqr = q3 - q1

    lower = q1 - outlier_factor * iqr
    upper = q3 + outlier_factor * iqr

    filtered_data = data[(data > lower) & (data < upper)]

    return filtered_data


def find_pose(
        topology, bind_jlp, points, weights, offsets, init_params=None, verbose=False, verbose_arg='',
        eps=1e-5, max_iter=70, mse_threshold=1e-5, u=1e-3, v=1.5
):
    device = points.device

    bounds = torch.Tensor([
        [-999, 999], [-999, 999], [-999, 999],  # pos
        [-360, 360], [-360, 360], [-360, 360],  # 0
        [-100, 40], [-30, 40], [-25, 50],  # 1
        [-100, 40], [-40, 30], [-50, 25],  # 2
        [-25, 100], [-20, 20], [-20, 20],  # 3
        [-25, 150], [-45, 45], [-20, 20],  # 4
        [-25, 150], [-45, 45], [-20, 20],  # 5
        [-40, 40], [-20, 15], [-20, 20],  # 6
        [-60, 60], [-20, 40], [-25, 25],  # 7
        [-60, 60], [-40, 20], [-25, 25],  # 8
        [-20, 20], [-15, 15], [-15, 15],  # 9
        [-40, 20], [-20, 20], [-50, 50],  # 10
        [-40, 20], [-20, 20], [-50, 50],  # 11
        [-50, 50], [-50, 50], [-40, 40],  # 12
        [-50, 25], [-60, 35], [-50, 50],  # 13
        [-50, 25], [-35, 60], [-50, 50],  # 14
        [-60, 40], [-45, 45], [-35, 35],  # 15
        [-50, 50], [-75, 40], [-100, 50],  # 16
        [-50, 50], [-40, 75], [-50, 100],  # 17
        [-180, 180], [-100, 40], [-180, 180],  # 18
        [-180, 180], [-40, 100], [-180, 180],  # 19
        [-100, 50], [-50, 35], [-70, 100],  # 20
        [-100, 50], [-35, 50], [-100, 70],  # 21
        [-180, 180], [-180, 180], [-180, 180],  # 22
        [-180, 180], [-180, 180], [-180, 180],  # 23
    ]).to(torch.float32).to(device)

    n_markers = points.shape[0]
    n_joints = topology.shape[0]
    n_params = n_joints * 3 + 3

    bind_jgp = bind_jlp.clone()
    for i, pi in enumerate(topology):
        if i == 0:
            assert pi == -1
            continue

        bind_jgp[i] = bind_jgp[pi] + bind_jgp[i]
    root_height = bind_jgp[0, 1] - min(bind_jgp[:, 1])

    if init_params is not None:
        assert len(init_params) == n_params
        params = init_params.to(device)
    else:
        params = torch.zeros(n_params).to(device)
        params[2] = root_height
        # params[3:6] = torch.Tensor([90, 0, 0]).to(device)  # torch.pi / 2

    out_n = n_markers * 3
    jacobian = torch.zeros([out_n, n_params]).to(device)

    last_update = 0
    last_mse = 0

    jlt = torch.eye(4).repeat(n_joints, 1, 1).to(device)
    jlt[:, :3, 3] = bind_jlp

    # jgt, virtual_points = lbs(topology, jlt, params, weights, offsets)
    # return params, jgt, virtual_points

    for i in range(max_iter):
        residual, mse = get_residual(topology, jlt, params, points, weights, offsets, bounds, mse=True)

        if abs(mse - last_mse) < mse_threshold:
            jgt, virtual_points = lbs(topology, jlt, params, weights, offsets, bounds)

            if verbose:
                print(verbose_arg + f'iter: {i}, mse: {mse}')
            return params, jgt, virtual_points

        for k in range(params.shape[0]):
            jacobian[:, k] = get_derivative(topology, jlt, params, points, weights, offsets, bounds, k, eps)

        jtj = torch.matmul(jacobian.T, jacobian)
        jtj = jtj + u * torch.eye(jtj.shape[0]).to(device)

        update = last_mse - mse
        delta = torch.matmul(
            torch.matmul(torch.linalg.inv(jtj), jacobian.T), residual
        ).ravel()
        params -= delta

        if update > last_update and update > 0:
            u /= v
        else:
            u *= v

        last_update = update
        last_mse = mse

    jgt, virtual_points = lbs(topology, jlt, params, weights, offsets, bounds)

    if verbose:
        print(verbose_arg + f'iter: {max_iter}, mse: {last_mse}')
    return params, jgt, virtual_points


def get_residual(topology, jlt, params, points, weights, offsets, bounds, mse=False):
    jgt, virtual_points = lbs(topology, jlt, params, weights, offsets, bounds)

    residual = virtual_points - points

    if mse:
        residual_mse = torch.mean(torch.linalg.norm(residual, dim=-1)).item()
        residual = residual.view(-1, 1)
        return residual, residual_mse
    else:
        residual = residual.view(-1, 1)
        return residual


def get_derivative(topology, jlt, params, points, weights, offsets, bounds, k, eps):
    params1 = params.clone()
    params2 = params.clone()

    params1[k] += eps
    params2[k] -= eps

    res1 = get_residual(topology, jlt, params1, points, weights, offsets, bounds)
    res2 = get_residual(topology, jlt, params2, points, weights, offsets, bounds)

    d = (res1 - res2) / (2 * eps)

    return d.ravel()


def fk(topology, jlt, params, bounds):

    jt = jlt.clone()
    jt[0, :3, 3] = params[:3]

    rot_mat = batch_rodrigues(params[3:].view(-1, 3))

    # torch.max(torch.min(params, bounds[:, 1]), bounds[:, 0])
    # euler = torch.deg2rad(params[3:].view(-1, 3))
    # x = euler[:, 0]
    # y = euler[:, 1]
    # z = euler[:, 2]
    # jt = jlt.clone()
    # jt[0, :3, 3] = params[:3]
    #
    # cos_x, sin_x = torch.cos(x), torch.sin(x)
    # cos_y, sin_y = torch.cos(y), torch.sin(y)
    # cos_z, sin_z = torch.cos(z), torch.sin(z)
    #
    # x_mat = torch.stack([
    #     torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x),
    #     torch.zeros_like(x), cos_x, -sin_x,
    #     torch.zeros_like(x), sin_x, cos_x
    # ], dim=1).view(-1, 3, 3)
    #
    # y_mat = torch.stack([
    #     cos_y, torch.zeros_like(y), sin_y,
    #     torch.zeros_like(y), torch.ones_like(y), torch.zeros_like(y),
    #     -sin_y, torch.zeros_like(y), cos_y
    # ], dim=1).view(-1, 3, 3)
    #
    # z_mat = torch.stack([
    #     cos_z, -sin_z, torch.zeros_like(z),
    #     sin_z, cos_z, torch.zeros_like(z),
    #     torch.zeros_like(z), torch.zeros_like(z), torch.ones_like(z)
    # ], dim=1).view(-1, 3, 3)
    #
    # rot_mat = torch.matmul(torch.matmul(z_mat, y_mat), x_mat)

    jt[:, :3, :3] = rot_mat

    for i, pi in enumerate(topology):
        if i == 0:
            assert pi == -1
            continue

        jt[i] = jt[pi] @ jt[i]

    return jt


def lbs(topology, jlt, params, weights, offsets, bounds):
    jgt = fk(topology, jlt, params, bounds)

    points = jgt[None, :, :3, :3] @ offsets[:, :, :, None]
    points = points.squeeze() + jgt[None, :, :3, 3]
    points = torch.sum(points * weights[:, :, None], dim=1)
    return jgt, points


def test_lbs(jgt, weights, offsets):
    points = jgt[None, :, :3, :3] @ offsets[:, :, :, None]
    points = points.squeeze() + jgt[None, :, :3, 3]
    points = torch.sum(points * weights[:, :, None], dim=1)
    return points

