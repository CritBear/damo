import numpy as np
import pickle
import torch
from scipy.spatial.transform import Rotation as R

from modules.solver.svd_solver import SVD_Solver
from modules.utils.paths import Paths


class PoseSolver:
    def __init__(
            self, topology, gt_skeleton_template=None,
            eps=1e-5, max_iter=15, mse_threshold=1e-6
    ):
        self.topology = topology
        self.n_joints = topology.shape[0]
        self.joint_transform = np.repeat(np.identity(4)[np.newaxis, :, :], self.n_joints, axis=0)
        self.gt_skeleton = False
        self.skeleton_template = None
        self.rest_pose = None

        if gt_skeleton_template is not None:
            assert self.n_joints == gt_skeleton_template.shape[0]

            self.skeleton_template = gt_skeleton_template
            for j in range(self.n_joints):
                self.joint_transform[j, :3, 3] = self.skeleton_template[j]
        else:
            self.svd_solver = SVD_Solver(self.n_joints)
            self.joint_pair_indices = [
                (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)
            ]
            if self.rest_pose is None:
                additional_data = np.load(Paths.support_data / 'additional_file.npz')
                self.rest_pose = additional_data['init_joint_list'][0, :22, :]
                # smplh_data = np.load(Paths.support_data / 'smplh' / 'neutral' / 'model.npz')
                # self.rest_pose = smplh_data['J'][:22, :]

            self.joint_norm = np.zeros((self.n_joints, 3))
            joint_length_sum = 0
            self.joint_length_ratio = np.zeros(self.n_joints)
            self.base_joint_length = np.zeros(self.n_joints)
            self.base_skeleton = np.zeros((self.n_joints, 3))

            for i, pi in enumerate(self.topology):
                if i == 0:
                    assert pi == -1
                    continue

                offset = self.rest_pose[i] - self.rest_pose[pi]
                length = np.linalg.norm(offset)
                joint_length_sum += length
                self.base_skeleton[i] = offset
                self.joint_norm[i] = offset / length
                self.joint_length_ratio[i] = length
                self.base_joint_length[i] = length

            self.joint_length_ratio /= joint_length_sum

        self.eps = eps
        self.max_iter = max_iter
        self.mse_threshold = mse_threshold

    def solve(self, points, weight, offset, init_params=None, u=1e-3, v=1.5):
        assert self.joint_transform is not None

        if init_params is not None:
            params = init_params
        else:
            params = np.zeros(3 + self.n_joints * 4)
            params[6::4] = 1

        out_n = points.shape[0] * 3
        jacobian = np.zeros([out_n, params.shape[0]])

        last_update = 0
        last_mse = 0
        for i in range(self.max_iter):
            residual = self.get_residual(params, weight, offset, points)
            mse = np.mean(np.square(residual))

            if abs(mse - last_mse) < self.mse_threshold:
                return params, self.fk(params)

            for k in range(params.shape[0]):
                jacobian[:, k] = self.get_derivative(params, k, weight, offset, points)

            jtj = np.matmul(jacobian.T, jacobian)
            jtj = jtj + u * np.eye(jtj.shape[0])

            update = last_mse - mse
            delta = np.matmul(
                np.matmul(np.linalg.inv(jtj), jacobian.T), residual
            ).ravel()
            params -= delta

            if update > last_update and update > 0:
                u /= v
            else:
                u *= v

            last_update = update
            last_mse = mse

        return params, self.fk(params)

    def get_derivative(self, params, k, weight, offset, target):
        params1 = params.copy()
        params2 = params.copy()

        params1[k] += self.eps
        params2[k] -= self.eps

        res1 = self.get_residual(params1, weight, offset, target)
        res2 = self.get_residual(params2, weight, offset, target)

        d = (res1 - res2) / (2 * self.eps)

        return d.ravel()

    def get_residual(self, params, weight, offset, target, mse=False):
        jgt = self.fk(params)
        lbs_points = self.lbs(jgt, weight, offset)
        residual = (lbs_points - target).reshape(-1, 1)

        if mse:
            residual = np.mean(np.square(residual))

        return residual

    def fk(self, params):
        jt = self.joint_transform.copy()
        jt[0, :3, 3] = params[:3]

        for i in range(self.n_joints):
            jt[i, :3, :3] = R.from_quat(params[3+i*4:7+i*4]).as_matrix()

        for i, pi in enumerate(self.topology):
            if i == 0:
                assert pi == -1
                continue

            jt[i] = jt[pi] @ jt[i]

        return jt

    def lbs(self, jgt, weight, offset):
        offset = np.concatenate((offset, np.ones((offset.shape[0], offset.shape[1], 1))), axis=-1)
        lbs_points = jgt[np.newaxis, :, :3, :] @ offset[:, :, :, np.newaxis]
        lbs_points = lbs_points.squeeze()
        lbs_points = np.sum(lbs_points * weight[:, :, np.newaxis], axis=1)

        return lbs_points

    def build_skeleton_template(self, points, weight, offset):
        # Build skeleton template
        # Step 1 : SVD solving : joint global position
        # Step 2 : Compute joint length
        # Step 3 : Normalize joint length
        assert points.shape[0] == weight.shape[0] == offset.shape[0]
        n_frames = points.shape[0]

        print("Building skeleton template...", end='')

        p = torch.from_numpy(points)
        w = torch.from_numpy(weight)

        o = torch.from_numpy(offset)
        jgp = self.svd_solver(p, w, o).numpy()[:, :, :, 3]  # (n_frames, n_joints, 3)

        joint_length = np.zeros((n_frames, self.n_joints))
        for i, pi in enumerate(self.topology):
            if i == 0:
                assert pi == -1
                continue

            joint_length[:, i] = np.sqrt(np.sum((jgp[:, i, :] - jgp[:, pi, :])**2, axis=-1))

        joint_mean_length = np.zeros(22)
        joint_nan_count = np.zeros(22)

        for i in range(1, 22):
            joint_nan_count[i] = np.sum(np.isnan(joint_length[:, i]))
            joint_mean_length[i] = PoseSolver.get_mean_without_outliers(joint_length[:, i])

        for (li, ri) in self.joint_pair_indices:
            pair_mean_length = (joint_mean_length[li] + joint_mean_length[ri]) / 2
            joint_mean_length[li] = pair_mean_length
            joint_mean_length[ri] = pair_mean_length

        # print('')
        # print('paired mean length')
        # print(joint_mean_length)

        base_length_sum = 0
        svd_length_sum = 0

        # leg: [4, 5, 7, 8]
        # arm: [18, 19, 20, 21]
        scale_basis_joint_indices = [4, 5, 7, 8, 20, 21]

        for i in scale_basis_joint_indices:
            if joint_nan_count[i] < n_frames * 0.1:
                base_length_sum += self.base_joint_length[i]
                svd_length_sum += joint_mean_length[i]
            else:
                print(f"NaN: {i}, {joint_nan_count[i]}")

        scale = svd_length_sum / base_length_sum

        for i in range(1, 22):
            if i in scale_basis_joint_indices:
                joint_mean_length[i] = np.clip(
                    joint_mean_length[i],
                    self.base_joint_length[i] * scale * 0.7,
                    self.base_joint_length[i] * scale * 1.3
                )
            else:
                joint_mean_length[i] = self.base_joint_length[i] * scale

            self.joint_transform[i, :3, 3] = self.joint_norm[i] * joint_mean_length[i]

        # self.joint_transform[:, :3, 3] = self.joint_transform[:, [2, 0, 1], 3]
        # self.joint_transform[:, 0, 3] *= -1
        # self.joint_transform[:, 1, 3] *= -1

        # print('final joint length')
        # print(joint_mean_length)

        return self.joint_transform[:, :3, 3].copy()

    @staticmethod
    def get_mean_without_outliers(data):
        data = data[~np.isnan(data)]
        data = np.sort(data)
        # interquartile_range

        # calculate interquartile range
        q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
        iqr = q75 - q25

        # calculate the outlier cutoff
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off

        # identify outliers
        outliers = [x for x in data if x < lower or x > upper]

        # remove outliers
        outlier_removed_data = [x for x in data if x >= lower and x <= upper]

        # calculate mean
        mean = np.mean(outlier_removed_data)

        return mean

