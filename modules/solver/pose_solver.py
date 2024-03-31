import torch
from dataclasses import dataclass
from typing import Callable

from modules.utils.paths import Paths
from modules.solver.svd_solver import SVD_Solver
from modules.utils.functions import x_rot_mat, y_rot_mat, z_rot_mat


@dataclass
class ParamInfo:
    joint_idx: int
    get_rot_mat: Callable[[float], torch.Tensor]
    lower_limit: float
    upper_limit: float


def find_bind_jgp():
    pass


def find_pose(topology, eps=1e-5, max_iter=10, mse_threshold=1e-4, u=1e-3, v=1.5):
    params_infos = [
        ParamInfo(1, x_rot_mat, -180, 180), ParamInfo(1, y_rot_mat, -180, 180), ParamInfo(1, z_rot_mat, -180, 180),
        ParamInfo(2, x_rot_mat, -180, 180), ParamInfo(2, y_rot_mat, -180, 180), ParamInfo(2, z_rot_mat, -180, 180),
        ParamInfo(3, x_rot_mat, -180, 180), ParamInfo(3, y_rot_mat, -180, 180), ParamInfo(3, z_rot_mat, -180, 180),
        ParamInfo(4, x_rot_mat, -180, 180), ParamInfo(4, y_rot_mat, -180, 180), ParamInfo(4, z_rot_mat, -180, 180),
        ParamInfo(5, x_rot_mat, -180, 180), ParamInfo(5, y_rot_mat, -180, 180), ParamInfo(5, z_rot_mat, -180, 180),
        ParamInfo(6, x_rot_mat, -180, 180), ParamInfo(6, y_rot_mat, -180, 180), ParamInfo(6, z_rot_mat, -180, 180),
        ParamInfo(7, x_rot_mat, -180, 180), ParamInfo(7, y_rot_mat, -180, 180), ParamInfo(7, z_rot_mat, -180, 180),
        ParamInfo(8, x_rot_mat, -180, 180), ParamInfo(8, y_rot_mat, -180, 180), ParamInfo(8, z_rot_mat, -180, 180),
        ParamInfo(9, x_rot_mat, -180, 180), ParamInfo(9, y_rot_mat, -180, 180), ParamInfo(9, z_rot_mat, -180, 180),
        ParamInfo(10, x_rot_mat, -180, 180), ParamInfo(10, y_rot_mat, -180, 180), ParamInfo(10, z_rot_mat, -180, 180),
        ParamInfo(11, x_rot_mat, -180, 180), ParamInfo(11, y_rot_mat, -180, 180), ParamInfo(11, z_rot_mat, -180, 180),
        ParamInfo(12, x_rot_mat, -180, 180), ParamInfo(12, y_rot_mat, -180, 180), ParamInfo(12, z_rot_mat, -180, 180),
        ParamInfo(13, x_rot_mat, -180, 180), ParamInfo(13, y_rot_mat, -180, 180), ParamInfo(13, z_rot_mat, -180, 180),
        ParamInfo(14, x_rot_mat, -180, 180), ParamInfo(14, y_rot_mat, -180, 180), ParamInfo(14, z_rot_mat, -180, 180),
        ParamInfo(15, x_rot_mat, -180, 180), ParamInfo(15, y_rot_mat, -180, 180), ParamInfo(15, z_rot_mat, -180, 180),
        ParamInfo(16, x_rot_mat, -180, 180), ParamInfo(16, y_rot_mat, -180, 180), ParamInfo(16, z_rot_mat, -180, 180),
        ParamInfo(17, x_rot_mat, -180, 180), ParamInfo(17, y_rot_mat, -180, 180), ParamInfo(17, z_rot_mat, -180, 180),
        ParamInfo(18, x_rot_mat, -180, 180), ParamInfo(18, y_rot_mat, -180, 180), ParamInfo(18, z_rot_mat, -180, 180),
        ParamInfo(19, x_rot_mat, -180, 180), ParamInfo(19, y_rot_mat, -180, 180), ParamInfo(19, z_rot_mat, -180, 180),
        ParamInfo(20, x_rot_mat, -180, 180), ParamInfo(20, y_rot_mat, -180, 180), ParamInfo(20, z_rot_mat, -180, 180),
        ParamInfo(21, x_rot_mat, -180, 180), ParamInfo(21, y_rot_mat, -180, 180), ParamInfo(21, z_rot_mat, -180, 180),
    ]