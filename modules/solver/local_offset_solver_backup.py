import torch
import numpy as np

from human_body_prior.body_model.lbs import batch_rodrigues


def solve_for_multiprocessing(
        jgr, jgp, markers, j3_indices, j3_weights, init_params=None,
        eps=1e-5, max_iter=10, mse_threshold=1e-4, u=1e-3, v=1.5
):
    n_markers, _ = markers.shape
    if init_params is not None:
        params = init_params.flatten().clone()
    else:
        params = torch.zeros(n_markers * 3 * 3).to(markers.device)

    out_n = n_markers * 3
    jacobian = torch.zeros([out_n, params.shape[0]]).to(markers.device)

    last_update = 0
    last_mse = 0

    for i in range(max_iter):
        params = torch.clamp(params, min=-2, max=2)

        residual, mse = get_residual(jgr, jgp, params, markers, j3_indices, j3_weights, mse=True)

        if abs(mse - last_mse) < mse_threshold:
            return params, lbs(jgr, jgp, params, j3_indices, j3_weights), i, mse

        for k in range(params.shape[0]):
            jacobian[:, k] = get_derivative(k, jgr, jgp, params, j3_indices, j3_weights, eps)

        jtj = torch.matmul(jacobian.T, jacobian)
        jtj = jtj + u * torch.eye(jtj.shape[0]).to(markers.device)

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

    return params, lbs(jgr, jgp, params, j3_indices, j3_weights), max_iter, last_mse


def get_residual(jgr, jgp, params, markers, j3_indices, j3_weights, mse=False):
    virtual_markers = lbs(jgr, jgp, params, j3_indices, j3_weights)

    residual = virtual_markers - markers

    if mse:
        residual_mse = torch.mean(torch.linalg.norm(residual, dim=-1)).item()
        residual = residual.view(-1, 1)
        return residual, residual_mse
    else:
        residual = residual.view(-1, 1)
        return residual


def get_derivative(k, jgr, jgp, params, j3_indices, j3_weights, eps):
    params1 = params.clone()
    params2 = params.clone()

    params1[k] += eps
    params2[k] -= eps

    res1 = lbs(jgr, jgp, params1, j3_indices, j3_weights)
    res2 = lbs(jgr, jgp, params2, j3_indices, j3_weights)

    d = (res1 - res2) / (2 * eps)

    return d.ravel()


def lbs(jgr, jgp, params, j3_indices, j3_weights):
    j3gr = jgr[j3_indices, :, :]  # [m, j3, 3, 3]
    j3gp = jgp[j3_indices, :]  # [m, j3, 3]

    # clamped_param = torch.clamp(params.view(-1, 3, 3), min=-2, max=2)

    points = j3gr @ params.view(-1, 3, 3)[:, :, :, None]  # [m, j3, 3, 1]
    points = points.squeeze()
    points += j3gp  # [m, j3, 3]
    points = torch.sum(points * j3_weights[:, :, None], dim=1)  # [m, 3]

    return points



class LocalOffsetSolver:
    def __init__(self, topology, eps=1e-5, max_iter=15, mse_threshold=1e-4):
        self.topology = topology
        self.jgr = None
        self.jgp = None
        self.n_joints = len(topology)
        self.eps = eps
        self.max_iter = max_iter
        self.mse_threshold = mse_threshold

    def set_transform(self, poses, jgp, device=None):

        if device is None:
            device = poses.device

        self.jgp = jgp.to(device)

        poses_c = poses.clone().view(-1, self.n_joints, 3)
        jgr = batch_rodrigues(poses_c.view(-1, 3)).view(-1, self.n_joints, 3, 3)
        for i, pi in enumerate(self.topology):
            if i == 0:
                assert pi == -1
                continue

            jgr[:, i] = jgr[:, pi] @ jgr[:, i]

        self.jgr = jgr.to(device)

    def batch_find_local_offsets(self, markers, j3_indices, j3_weights, init_params=None, u=1e-3, v=1.5):
        n_frames, n_markers, _ = markers.shape

        if init_params is not None:
            params = init_params.flatten().clone()
        else:
            params = torch.zeros(n_frames * n_markers * 3 * 3).to(markers.device)

        out_n = n_frames * n_markers * 3
        jacobian = torch.zeros([out_n, params.shape[0]]).to(markers.device)

        last_update = 0
        last_mse = 0

        for i in range(self.max_iter):
            params = torch.clamp(params, min=-2, max=2)

            residual, mse = self.batch_get_residual(params, markers, j3_indices, j3_weights, mse=True)

            if abs(mse - last_mse) < self.mse_threshold:
                print(f' | iter: {i} | mse: {mse}')
                return params, self.batch_lbs(params, j3_indices, j3_weights)

            jacobian = self.get_jacobian(params, i, j3_indices, j3_weights)
            # for k in range(params.shape[0]):
            #     jacobian[:, k] = self.batch_get_derivative(k, params, j3_indices, j3_weights)

            jtj = torch.matmul(jacobian.T, jacobian)
            jtj = jtj + u * torch.eye(jtj.shape[0]).to(markers.device)

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

        print(f' | solving complete. iter: {self.max_iter} | ', end='')
        return params, self.batch_lbs(params, j3_indices, j3_weights)

    def get_jacobian(self, params, i, j3_indices, j3_weights):
        num_params = params.shape[0]
        params_repeat_add = params.repeat(num_params, 1)
        params_repeat_sub = params.repeat(num_params, 1)

        eps_matrix = torch.eye(num_params) * self.eps
        params_repeat_add += eps_matrix
        params_repeat_sub -= eps_matrix

        res_add = self.lbs(params_repeat_add, i, j3_indices, j3_weights)
        res_sub = self.lbs(params_repeat_sub, i, j3_indices, j3_weights)

        jacobian = (res_add - res_sub) / (2 * self.eps)

        return jacobian.view(num_params, -1).T

    def find_local_offsets(self, i, markers, j3_indices, j3_weights, init_params=None, u=1e-3, v=1.5):
        n_markers, _ = markers.shape

        if init_params is not None:
            params = init_params.flatten().clone()
        else:
            params = torch.zeros(n_markers * 3 * 3).to(markers.device)

        out_n = n_markers * 3
        jacobian = torch.zeros([out_n, params.shape[0]]).to(markers.device)

        last_update = 0
        last_mse = 0

        # return params, self.lbs(params, markers, m_v_idx, jgr, jgp, ghost_marker_mask)

        for i in range(self.max_iter):
            params = torch.clamp(params, min=-2, max=2)

            residual, mse = self.get_residual(params, i, markers, j3_indices, j3_weights, mse=True)

            if abs(mse - last_mse) < self.mse_threshold:
                # print(f' | iter: {i} | mse: {mse}', end='')
                return params, self.lbs(params, i, j3_indices, j3_weights)

            for k in range(params.shape[0]):
                jacobian[:, k] = self.get_derivative(k, params, i, j3_indices, j3_weights)

            jtj = torch.matmul(jacobian.T, jacobian)
            jtj = jtj + u * torch.eye(jtj.shape[0]).to(markers.device)

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

        print(f' | solving complete. iter: {self.max_iter} | ', end='')
        return params, self.lbs(params, i, j3_indices, j3_weights)

    def batch_get_derivative(self, k, params, j3_indices, j3_weights):
        params1 = params.clone()
        params2 = params.clone()

        params1[k] += self.eps
        params2[k] -= self.eps

        res1 = self.batch_lbs(params1, j3_indices, j3_weights)
        res2 = self.batch_lbs(params2, j3_indices, j3_weights)

        d = (res1 - res2) / (2 * self.eps)

        return d.ravel()

    def get_derivative(self, k, params, i, j3_indices, j3_weights):
        params1 = params.clone()
        params2 = params.clone()

        params1[k] += self.eps
        params2[k] -= self.eps

        res1 = self.lbs(params1, i, j3_indices, j3_weights)
        res2 = self.lbs(params2, i, j3_indices, j3_weights)

        d = (res1 - res2) / (2 * self.eps)

        return d.ravel()

    def batch_get_residual(self, params, markers, j3_indices, j3_weights, mse=False):
        virtual_markers = self.batch_lbs(params, j3_indices, j3_weights)

        residual = virtual_markers - markers

        if mse:
            residual_mse = torch.mean(torch.linalg.norm(residual, dim=-1)).item()
            residual = residual.view(-1, 1)
            return residual, residual_mse
        else:
            residual = residual.view(-1, 1)
            return residual

    def get_residual(self, params, i, markers, j3_indices, j3_weights, mse=False):
        virtual_markers = self.lbs(params, i, j3_indices, j3_weights)

        residual = virtual_markers - markers
        # residual = torch.mean(residual, dim=0).view(-1, 1)

        if mse:
            residual_mse = torch.mean(torch.linalg.norm(residual, dim=-1)).item()
            residual = residual.view(-1, 1)
            return residual, residual_mse
        else:
            residual = residual.view(-1, 1)
            return residual

    def lbs(self, params, i, j3_indices, j3_weights):
        j3gr = self.jgr[i, j3_indices, :, :]  # [m, j3, 3, 3]
        j3gp = self.jgp[i, j3_indices, :]  # [m, j3, 3]

        # clamped_param = torch.clamp(params.view(-1, 3, 3), min=-2, max=2)

        points = j3gr @ params.view(-1, 3, 3)[:, :, :, None]  # [m, j3, 3, 1]
        points = points.squeeze()
        points += j3gp  # [m, j3, 3]
        points = torch.sum(points * j3_weights[:, :, None], dim=1)  # [m, 3]

        return points

    def batch_lbs(self, params, j3_indices, j3_weights):
        n_frames, n_mvm, _ = j3_indices.shape

        j3gr = self.jgr[torch.arange(n_frames)[:, None, None], j3_indices, :, :]
        j3gp = self.jgp[torch.arange(n_frames)[:, None, None], j3_indices, :]

        points = j3gr @ params.view(n_frames, n_mvm, 3, 3)[:, :, :, :, None]  # [f, m, j3, 3, 1]
        points = points.squeeze()
        points += j3gp  # [f, m, j3, 3]
        points = torch.sum(points * j3_weights[:, :, :, None], dim=2)  # [f, m, 3]

        return points
