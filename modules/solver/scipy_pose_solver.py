from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch


def find_pose(topology, bind_jlp, points, weights, offsets, verbose=False, verbose_arg=''):
    bind_jlp = bind_jlp.numpy()
    points = points.numpy()
    weights = weights.numpy()
    offsets = offsets.numpy()

    n_markers = points.shape[0]
    n_joints = topology.shape[0]
    n_params = n_joints * 3 + 3

    jlt = np.tile(np.eye(4), (n_joints, 1, 1))
    jlt[:, :3, 3] = bind_jlp

    bind_jgp = bind_jlp.copy()
    for i, pi in enumerate(topology):
        if i == 0:
            assert pi == -1
            continue

        bind_jgp[i] = bind_jgp[pi] + bind_jgp[i]
    root_height = bind_jgp[0, 1] - min(bind_jgp[:, 1])

    params = np.zeros(n_params)
    params[2] = root_height
    params[3:6] = [np.pi / 2, 0, 0]

    bounds = [
        (-999, 999), (-999, 999), (-999, 999),  # pos
        (-360, 360), (-360, 360), (-360, 360),  # 0
        (-100, 40), (-30, 40), (-25, 50),  # 1
        (-100, 40), (-40, 30), (-50, 25),  # 2
        (-25, 100), (-20, 20), (-20, 20),  # 3
        (-25, 150), (-45, 45), (-20, 20),  # 4
        (-25, 150), (-45, 45), (-20, 20),  # 5
        (-40, 40), (-20, 15), (-20, 20),  # 6
        (-60, 60), (-20, 40), (-25, 25),  # 7
        (-60, 60), (-40, 20), (-25, 25),  # 8
        (-20, 20), (-15, 15), (-15, 15),  # 9
        (-40, 20), (-20, 20), (-50, 50),  # 10
        (-40, 20), (-20, 20), (-50, 50),  # 11
        (-50, 50), (-50, 50), (-40, 40),  # 12
        (-50, 25), (-60, 35), (-50, 50),  # 13
        (-50, 25), (-35, 60), (-50, 50),  # 14
        (-60, 40), (-45, 45), (-35, 35),  # 15
        (-50, 50), (-75, 40), (-100, 50),  # 16
        (-50, 50), (-40, 75), (-50, 100),  # 17
        (-180, 180), (-100, 40), (-180, 180),  # 18
        (-180, 180), (-40, 100), (-180, 180),  # 19
        (-100, 50), (-50, 35), (-70, 100),  # 20
        (-100, 50), (-35, 50), (-100, 70),  # 21
        (-180, 180), (-180, 180), (-180, 180),  # 22
        (-180, 180), (-180, 180), (-180, 180),  # 23
    ]

    result = minimize(get_residual, params, args=(topology, jlt, points, weights, offsets))

    jgt, virtual_points = lbs(result.x, topology, jlt, weights, offsets)

    return result.x, jgt, virtual_points


def get_residual(params, topology, jlt, points, weights, offsets):
    jgt, virtual_points = lbs(params, topology, jlt, weights, offsets)

    residual = virtual_points - points
    residual_mse = np.mean(np.linalg.norm(residual, axis=-1))

    return residual_mse


def lbs(params, topology, jlt, weights, offsets):
    jgt = fk(params, topology, jlt)

    points = jgt[None, :, :3, :3] @ offsets[:, :, :, None]
    points = points.squeeze() + jgt[None, :, :3, 3]
    points = np.sum(points * weights[:, :, None], axis=1)
    return jgt, points


def fk(params, topology, jlt):
    jt = jlt.copy()
    jt[0, :3, 3] = params[:3]

    # rot_mat = R.from_euler('xyz', params[3:].reshape(-1, 3), degrees=True).as_matrix()
    rot_mat = R.from_rotvec(params[3:].reshape(-1, 3)).as_matrix()

    jt[:, :3, :3] = rot_mat

    for i, pi in enumerate(topology):
        if i == 0:
            assert pi == -1
            continue

        jt[i] = jt[pi] @ jt[i]

    return jt
