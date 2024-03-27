import torch
import numpy as np

from human_body_prior.body_model.lbs import batch_rodrigues


class LocalOffsetSolver:
    def __init__(self, topology, j3_indices, j3_weights, eps=1e-5, max_iter=15, mse_threshold=1e-4):
        self.topology = topology
        self.j3_indices = j3_indices
        self.j3_weights = j3_weights
        self.n_joints = len(topology)
        self.eps = eps
        self.max_iter = max_iter
        self.mse_threshold = mse_threshold

    def find_marker_local_offsets(
            self, markers, m_v_idx, jgp, poses, ghost_marker_mask,
            init_params=None, u=1e-3, v=1.5
    ):
        n_frames, n_markers, _ = markers.shape

        if init_params is not None:
            params = init_params.flatten().clone()
        else:
            params = torch.zeros(n_frames, n_markers * 3 * 3).to(markers.device)
        out_n = n_frames * n_markers * 3
        jacobian = torch.zeros([out_n, params.shape[0]]).to(markers.device)

        poses = poses.view(n_frames, -1, 3)
        jgr = batch_rodrigues(poses.view(-1, 3)).view(n_frames, self.n_joints, 3, 3)

        for i, pi in enumerate(self.topology):
            if i == 0:
                assert pi == -1
                continue

            jgr[:, i] = jgr[:, pi] @ jgr[:, i]

        last_update = 0
        last_mse = 0

        # return params, self.lbs(params, markers, m_v_idx, jgr, jgp, ghost_marker_mask)

        for i in range(self.max_iter):
            params = torch.clamp(params, min=-2, max=2)

            residual = self.get_residual(params, markers, m_v_idx, jgr, jgp, ghost_marker_mask)
            mse = torch.mean(torch.sqrt(torch.sum(torch.square(residual), dim=-1))).item()

            if abs(mse - last_mse) < self.mse_threshold:
                print(f' | solving complete. iter: {i} | ', end='')
                return params, self.lbs(params, markers, m_v_idx, jgr, jgp, ghost_marker_mask)

            for k in range(params.shape[0]):
                jacobian[:, k] = self.get_derivative(k, params, markers, m_v_idx, jgr, jgp, ghost_marker_mask)

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
        return params, self.lbs(params, markers, m_v_idx, jgr, jgp, ghost_marker_mask)

    def get_derivative(self, k, params, markers, m_v_idx, jgr, jgp, ghost_marker_mask):
        params1 = params.clone()
        params2 = params.clone()

        params1[k] += self.eps
        params2[k] -= self.eps

        # res1 = self.get_residual(params1, markers, m_v_idx, jgr, jgp, ghost_marker_mask)
        # res2 = self.get_residual(params2, markers, m_v_idx, jgr, jgp, ghost_marker_mask)

        res1 = self.get_residual(params1, markers, m_v_idx, jgr, jgp, ghost_marker_mask)
        res2 = self.get_residual(params2, markers, m_v_idx, jgr, jgp, ghost_marker_mask)

        d = (res1 - res2) / (2 * self.eps)

        return d.ravel()

    def get_residual(self, params, markers, m_v_idx, jgr, jgp, ghost_marker_mask, mse=False):
        virtual_markers = self.lbs(params, markers, m_v_idx, jgr, jgp, ghost_marker_mask)

        masked_marker = markers * ghost_marker_mask.unsqueeze(-1)

        residual = virtual_markers - masked_marker
        # residual = torch.mean(residual, dim=0).view(-1, 1)
        residual = residual.view(-1, 1)

        if mse:
            residual = torch.mean(torch.sqrt(torch.sum(torch.square(residual), dim=-1))).item()

        return residual

    def lbs(self, params, markers, m_v_idx, jgr, jgp, ghost_marker_mask):
        n_markers = markers.shape[1]
        j3_indices = self.j3_indices[m_v_idx]
        j3_weights = self.j3_weights[m_v_idx]
        j3gr = jgr[:, j3_indices, :, :]  # [f, m, j3, 3, 3]
        j3gp = jgp[:, j3_indices, :]  # [f, m, j3, 3]

        # clamped_param = torch.clamp(params.view(-1, 3, 3), min=-2, max=2)

        points = j3gr @ params.view(-1, 3, 3)[None, :, :, :, None]  # [f, m, j3, 3, 1]
        points = points.squeeze()
        points += j3gp  # [f, m, j3, 3]
        points = torch.sum(points * j3_weights[None, :, :, None], dim=2)  # [f, m, 3]

        masked_points = points * ghost_marker_mask.unsqueeze(-1)

        return masked_points
