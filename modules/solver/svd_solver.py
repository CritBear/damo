import torch
import torch.nn as nn


class SVD_Solver(nn.Module):
    def __init__(self, n_joints):
        super().__init__()
        self.n_joints = n_joints

    def forward(self, X, w, Z):
        """
        :param points: (batch_size, n_max_markers, 3)
        :param weight: (batch_size, n_max_markers, n_joints)
        :param offset: (batch_size, n_max_markers, n_joints, 3)
        :return: (batch_size, n_joints, 3, 4)
        """
        bn = Z.shape[0]
        n_joints = Z.shape[-2]
        Y_hat = torch.empty(bn, n_joints, 3, 4).to(torch.float32).to(X.device)
        for bi in range(bn):
            wb = w[bi]
            for i in range(n_joints):
                markers = (wb[:, i] > 0.01).nonzero(as_tuple=False).view((-1))
                Z_ = Z[bi, markers, i].unsqueeze(0).permute(0, 2, 1)  # bn, 3, m
                X_ = X[bi, markers].unsqueeze(0).permute(0, 2, 1)  # bn, 3, m
                R, t = SVD_Solver.svd_rot(Z_, X_, wb[markers, i])
                R_t = torch.cat((R[0], t[0]), -1)
                Y_hat[bi, i] = R_t
        return Y_hat

    @staticmethod
    def svd_rot(P, Q, w):
        '''
            Implementation of "Least-Squares Rigid Motion Using SVD"
            https://igl.ethz.ch/projects/ARAP/svd_rot.pdf

            Problem:
                finding rotation R and translation t matrices
                that minimizes \sum (w_i ||(Rp_i + t) - q_i||^2)
                (least square error)

            Solution:
                t = q_mean - R*p_mean
                R = V * D * U.T
            '''
        assert P.shape[-2:] == Q.shape[-2:]
        d, n = P.shape[-2:]

        # X,Y are d x n
        P_ = torch.sum(P * w, dim=-1) / torch.sum(w)
        Q_ = torch.sum(Q * w, dim=-1) / torch.sum(w)
        X = P - P_[..., None]
        Y = Q - Q_[..., None]
        Yt = Y.permute(0, 2, 1)

        # S is d x d
        S = torch.matmul(X, Yt)

        # U, V are d x d
        U, _, V = torch.svd(S)
        # V = V_t.permute(0, 2, 1)
        Ut = U.permute(0, 2, 1)

        det = torch.det(torch.matmul(V, Ut))
        Ut[:, -1, :] *= det.view(-1, 1)

        # R is d x d
        R = torch.matmul(V, Ut)

        # t is d x n
        t = Q_.view(-1, d, 1) - torch.matmul(R, P_.view(-1, d, 1))

        return R, t

