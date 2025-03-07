import torch
import numpy as np
import c3d


def dict2class(res):
    class result_meta(object):
        pass

    res_class = result_meta()
    for k, v in res.items():
        res_class.__setattr__(k, v)

    return res_class


def read_c3d_markers(file_path):
    reader = c3d.Reader(open(file_path, 'rb'))
    marker_data = []

    for i, points, analog in reader.read_frames():
        marker_data.append(points)

    marker_data = np.array(marker_data)

    return marker_data[:, :, :3]


def x_rot_mat(deg, use_torch=True):
    if use_torch:
        rad = torch.deg2rad(deg)
        cos = torch.cos(rad)
        sin = torch.sin(rad)

        rot_mat = torch.Tensor([
            [1, 0, 0],
            [0, cos, -sin],
            [0, sin, cos]
        ])

    else:
        rad = np.deg2rad(deg)
        cos = np.cos(rad)
        sin = np.sin(rad)

        rot_mat = np.array([
            [1, 0, 0],
            [0, cos, -sin],
            [0, sin, cos]
        ])

    return rot_mat


def y_rot_mat(deg, use_torch=True):
    if use_torch:
        rad = torch.deg2rad(deg)
        cos = torch.cos(rad)
        sin = torch.sin(rad)

        rot_mat = torch.Tensor([
            [cos, 0, sin],
            [0, 1, 0],
            [-sin, 0, cos]
        ])

    else:
        rad = np.deg2rad(deg)
        cos = np.cos(rad)
        sin = np.sin(rad)

        rot_mat = np.array([
            [cos, 0, sin],
            [0, 1, 0],
            [-sin, 0, cos]
        ])

    return rot_mat


def z_rot_mat(deg, use_torch=True):
    if use_torch:
        rad = torch.deg2rad(deg)
        cos = torch.cos(rad)
        sin = torch.sin(rad)

        rot_mat = torch.Tensor([
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1]
        ])
    else:
        rad = np.deg2rad(deg)
        cos = np.cos(rad)
        sin = np.sin(rad)

        rot_mat = np.array([
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1]
        ])

    return rot_mat
