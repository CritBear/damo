from pathlib import Path
import json
import pickle
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .base import load_native_model, sha256
from .data import load_common
from .geometry import forward_kinematics

def up_rotation(axis):
    if axis not in ('y', 'z'):
        raise ValueError('Explicit y or z up-axis is required')
    return Rotation.from_rotvec([np.pi / 2, 0, 0]).as_matrix() if axis == 'y' else np.eye(3)

class NativeMotion:

    def __init__(self, model_path, motion, *, up_axis='z', device='cpu'):
        from smplx.lbs import lbs
        self.reference_lbs = lbs
        self.native = native = load_native_model(model_path)
        self.motion, self.device = (motion, device)
        self.world = up_rotation(up_axis)
        if motion['poses'].shape[1] != len(native['parents']) * 3:
            raise ValueError('AMASS pose dimension does not match native model')
        nb = len(motion['betas'])
        if nb > native['shapedirs'].shape[-1]:
            raise ValueError('Native model lacks supplied shape coefficients')
        with np.load(model_path, allow_pickle=False) as raw:
            posedirs = raw['posedirs']
        expected = (len(native['vertices']), 3, (len(native['parents']) - 1) * 9)
        if posedirs.shape != expected or not np.isfinite(posedirs).all():
            raise ValueError('Invalid native pose correctives')
        self.bind_vertices = native['vertices'] + np.einsum('vck,k->vc', native['shapedirs'][:, :, :nb], motion['betas'])
        self.bind_joints = native['regressor'] @ self.bind_vertices
        self.bind_local = np.zeros((22, 3))
        self.bind_local[1:] = self.bind_joints[1:22] - self.bind_joints[native['parents'][1:22]]
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
        self.parameters = {'v_template': tensor(native['vertices'][None]), 'shapedirs': tensor(native['shapedirs'][:, :, :nb]), 'posedirs': tensor(posedirs.reshape(-1, expected[-1]).T), 'J_regressor': tensor(native['regressor']), 'parents': torch.as_tensor(native['parents'], dtype=torch.long, device=device), 'lbs_weights': tensor(native['weights'])}
        self.betas = tensor(motion['betas'][None])

    def geometry(self, frames):
        frames = np.asarray(frames, dtype=int)
        pose = torch.as_tensor(self.motion['poses'][frames], dtype=torch.float64, device=self.device)
        with torch.inference_mode():
            vertices, joints = self.reference_lbs(betas=self.betas.expand(len(frames), -1), pose=pose, **self.parameters)
        translation = self.motion['trans'][frames, None]
        return ((vertices.cpu().numpy() + translation) @ self.world.T, (joints.cpu().numpy() + translation) @ self.world.T)

    def body_targets(self):
        poses = self.motion['poses'][:, :66].copy()
        poses[:, :3] = Rotation.from_matrix(self.world @ Rotation.from_rotvec(poses[:, :3]).as_matrix()).as_rotvec()
        rotations = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
        roots = (self.motion['trans'] + self.bind_joints[0]) @ self.world.T
        transforms = forward_kinematics(rotations, roots, self.bind_local, self.native['parents'][:22])
        return (poses, transforms)
