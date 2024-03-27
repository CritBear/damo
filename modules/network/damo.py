import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Damo(nn.Module):

    def __init__(self, options):
        super().__init__()

        self.options = options

        self.d_model = self.options.d_model
        self.d_hidden = self.options.d_hidden
        self.n_layers = self.options.n_layers
        self.n_heads = self.options.n_heads
        self.n_max_markers = self.options.n_max_markers
        self.n_joints = self.options.n_joints
        self.seq_len = self.options.seq_len

        assert self.seq_len % 2 == 1, ValueError(
            f"seq_len ({self.seq_len}) must be odd number.")

        self.embedding = nn.Sequential(
            Transpose(-3, -1),
            ResConv2DBlock(3, self.d_model, self.d_hidden),
            nn.ReLU()
        )

        self.attention_layers = LayeredMixedAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            seq_len=self.seq_len
        )

        self.post_attention_layer = nn.Sequential(
            Transpose(-3, -1),
            nn.Conv2d(self.seq_len, 1, 1, 1),
            nn.ReLU(),
            Transpose(-2, -1)
        )

        self.joint_feats_layer = nn.Sequential(
            nn.ReLU(),
            ResConv1DBlock(self.d_model, self.d_model, self.d_hidden),
            nn.ReLU(),
            # SDivide(self.d_model ** .5),
            nn.Conv1d(self.d_model, self.n_joints * 4, 1, 1)
        )

        self.joint_indices_predictor = nn.Sequential(
            nn.ReLU(),
            ResConv1DBlock(self.n_joints * 4, self.n_joints + 1, self.d_hidden * 2)
        )

        self.weight_predictor = nn.Sequential(
            nn.ReLU(),
            ResConv1DBlock(self.n_joints * 2 + 1, 3, self.d_hidden),
            nn.Softmax(dim=1),
            Transpose(-2, -1)
        )

        self.offset_predictor = nn.Sequential(
            nn.ReLU(),
            ResConv1DBlock(self.n_joints * 4 + 1, 9, self.d_hidden * 2),
            Transpose(-2, -1)
        )

        # self.weight_predictor = nn.Sequential(
        #     nn.ReLU(),
        #     ResConv1DBlock(self.n_joints, self.n_joints, self.d_hidden),
        #     Transpose(-2, -1)
        # )
        #
        # self.offset_predictor = nn.Sequential(
        #     nn.ReLU(),
        #     ResConv1DBlock(self.n_joints * 3, self.n_joints * 3, self.d_hidden * 3),
        #     Transpose(-2, -1)
        # )

        # self.svd_solver = SVD_Solver(self.n_joints)

    def forward(self, points_seq, points_mask):
        """
        :param points_seq: (batch_size, seq_len, max_markers, 3) tensor
        :param points_mask: (batch_size, seq_len, max_markers) tensor
        :return: marker configuration
            marker-joint weight: (batch_size, max_markers, n_joints) tensor
            marker-joint offset: (batch_size, max_markers, n_joints, 3) tensor
        """

        points_mask.require_grad = False
        points_seq = points_seq * points_mask.unsqueeze(-1)
        # (batch_size, seq_len, max_markers, 3)

        points_offset_seq = Damo.compute_offsets(points_seq)
        points_centered_seq = points_seq - points_offset_seq
        # (batch_size, seq_len, max_markers, 3)

        points_feats_seq = self.embedding(points_centered_seq)
        # (batch_size, d_model, max_markers, seq_len)

        points_attention_seq = self.attention_layers(points_feats_seq, points_mask)
        # (batch_size, d_model, max_markers, seq_len)

        center_seq_feats = self.post_attention_layer(points_attention_seq).squeeze()
        # (batch_size, d_model, max_markers)

        marker_configuration_feats = self.joint_feats_layer(center_seq_feats)
        # (batch_size, n_joints * 4, max_markers)

        joint_indices = self.joint_indices_predictor(marker_configuration_feats)
        # (batch_size, n_joints + 1, max_markers)

        weight_feats = torch.cat((marker_configuration_feats[:, :self.n_joints, :], joint_indices), dim=1)
        weight = self.weight_predictor(weight_feats)
        # (batch_size, max_markers, 3)

        offset_feats = torch.cat((marker_configuration_feats[:, self.n_joints:, :], joint_indices), dim=1)
        offset = self.offset_predictor(offset_feats)
        # (batch_size, max_markers, 3 * 3)

        # weight = self.weight_predictor(marker_configuration_feats[:, :self.n_joints, :])
        # # (batch_size, max_markers, n_joints)
        #
        # offset = self.offset_predictor(marker_configuration_feats[:, self.n_joints:, :])
        # # (batch_size, max_markers, n_joints * 3)

        center_seq_idx = self.seq_len // 2
        center_seq_mask = points_mask[:, center_seq_idx, :].unsqueeze(-1)
        # (batch_size, max_markers, 1)

        joint_indices = joint_indices.permute(0, 2, 1) * center_seq_mask
        weight = weight * center_seq_mask
        offset = offset * center_seq_mask

        batch_size, _, _ = offset.shape
        offset = offset.view(batch_size, self.n_max_markers, 3, 3)

        # joint_global_rot, joint_global_tr = self.svd_solver(points_centered_seq[:, center_seq_idx, :, :], weight, offset)
        # # (batch_size, n_joints, 3, 3), (batch_size, n_joints, 3, 1)
        # joint_global_tr = joint_global_tr.squeeze() + points_offset_seq
        # #  (batch_size, n_joints, 3)

        return joint_indices, weight, offset

    @staticmethod
    def compute_offsets(points_seq):
        batch_size, seq_len, _, _ = points_seq.shape

        nonzero_mask = ((points_seq == 0.0).sum(-1) != 3)
        batch_offsets = []

        for batch_idx in range(batch_size):
            seq_offsets = []
            for seq_idx in range(seq_len):
                if nonzero_mask[batch_idx][seq_idx].sum() == 0:
                    seq_offsets.append(points_seq.new(np.zeros([1, 3])))
                    continue
                seq_offsets.append(torch.median(points_seq[batch_idx, seq_idx, nonzero_mask[batch_idx, seq_idx]], dim=0, keepdim=True).values)
            batch_offsets.append(torch.cat(seq_offsets, dim=0).view(seq_len, 1, 3))

        return torch.cat(batch_offsets, dim=0).view(batch_size, seq_len, 1, 3)


class LayeredMixedAttention(nn.Module):

    def __init__(self, d_model, n_heads, n_layers, seq_len):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len

        self.center_seq_idx = seq_len // 2

        self.total_attention_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.total_attention_layers.append(nn.ModuleList([
                MultiHeadAttention(self.d_model, self.n_heads)
                for _ in range(self.seq_len)
            ]))

    def make_score_mask(self, query_mask, key_mask):
        """
        :param query_mask: (batch_size, query_len)
        :param key_mask: (batch_size, key_len)
        :return: score_mask: (batch_size, 1, query_len, key_len)
        """
        query_len, key_len = query_mask.size(1), key_mask.size(1)

        key_mask = key_mask.ne(0).unsqueeze(1).unsqueeze(2).repeat(1, 1, query_len, 1)
        # (batch_size, 1, query_len, key_len)
        query_mask = query_mask.ne(0).unsqueeze(1).unsqueeze(3).repeat(1, 1, 1, key_len)
        # (batch_size, 1, query_len, key_len)

        score_mask = key_mask & query_mask
        score_mask.required_grad = False
        return score_mask


    def forward(self, points_attention_seq, points_mask):
        """
        :param points_attention_seq: (batch_size, d_model, max_markers, seq_len) tensor
        :param points_mask: (batch_size, seq_len, max_markers) tensor
        :return: (batch_size, d_model, max_markers, seq_len) tensor
        """

        for attention_layers in self.total_attention_layers:
            attention_seq = []
            for idx, attention_layer in enumerate(attention_layers):
                query_idx = idx
                key_value_idx = self.center_seq_idx
                # query_idx = self.center_seq_idx
                # key_value_idx = idx

                score_mask = self.make_score_mask(
                    query_mask=points_mask[:, query_idx, :],
                    key_mask=points_mask[:, key_value_idx, :]
                )

                attention = attention_layer(
                    points_attention_seq[:, :, :, query_idx],
                    points_attention_seq[:, :, :, key_value_idx],
                    points_attention_seq[:, :, :, key_value_idx],
                    mask=score_mask
                )
                attention_seq.append(attention.unsqueeze(dim=3))

            points_attention_seq = torch.cat(attention_seq, dim=3)

        return points_attention_seq


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0, ValueError(
            f"d_model ({d_model}) % n_heads ({n_heads}) is not 0 ({d_model % n_heads})")

        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.proj = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, kernel_size=1) for _ in range(3)]
        )
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.post_merge = nn.Sequential(
            nn.Conv1d(2 * d_model, 2 * d_model, kernel_size=1),
            nn.BatchNorm1d(2 * d_model),
            nn.ReLU(),
            nn.Conv1d(2 * d_model, d_model, kernel_size=1),
        )
        nn.init.constant_(self.post_merge[-1].bias, 0.0)

    def forward(self, init_query, key, value, mask):
        """
        :param init_query: (batch_size, d_model, n_max_markers) tensor
        :param key: (batch_size, d_model, n_max_markers) tensor
        :param value: (batch_size, d_model, n_max_markers) tensor
        :param mask: (batch_size, 1, src_len, tgt_len) tensor
        :return: (batch_size, d_model, n_max_markers) tensor
        """
        batch_size = init_query.size(0)

        # (batch_size, d_model, n_max_markers)

        query, key, value = [l(x).view(batch_size, self.d_k, self.n_heads, -1) for l, x in
                             zip(self.proj, (init_query, key, value))]

        # (batch_size, d_k, n_heads, n_max_markers)

        x = MultiHeadAttention.scaled_dot_product_attention(query, key, value, mask)
        x = self.merge(x.contiguous().view(batch_size, self.d_k * self.n_heads, -1))
        x = self.post_merge(torch.cat([x, init_query], dim=1))

        return x

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask):
        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5

        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        attention_weight = torch.nn.functional.softmax(scores, dim=-1)
        return torch.einsum('bhnm,bdhm->bdhn', attention_weight, value)


class ResLinearBlock(nn.Module):

    def __init__(self, d_in, d_out, d_hidden):
        super().__init__()

        self.res_linear = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
            nn.BatchNorm1d(d_out)
        )

        self.res_linear_short = nn.Sequential(
            *(
                [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out)]
                if d_in != d_out else
                [nn.Identity()]
            )
        )

    def forward(self, x):
        return self.res_linear(x) + self.res_linear_short(x)


class ResConv2DBlock(nn.Module):

    def __init__(self, d_in, d_out, d_hidden):
        super().__init__()

        self.res_conv2d = nn.Sequential(
            nn.Conv2d(d_in, d_hidden, 1, 1),
            nn.BatchNorm2d(d_hidden),
            nn.ReLU(),
            nn.Conv2d(d_hidden, d_out, 1, 1),
            nn.BatchNorm2d(d_out)
        )

        self.res_conv2d_short = nn.Sequential(
            *(
                [nn.Conv2d(d_in, d_out, 1, 1), nn.BatchNorm2d(d_out)]
                if d_in != d_out else
                [nn.Identity()]
            )
        )

    def forward(self, x):
        return self.res_conv2d(x) + self.res_conv2d_short(x)


class ResConv1DBlock(nn.Module):

    def __init__(self, d_in, d_out, d_hidden):
        super().__init__()

        self.res_conv1d = nn.Sequential(
            nn.Conv1d(d_in, d_hidden, 1, 1),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Conv1d(d_hidden, d_out, 1, 1),
            nn.BatchNorm1d(d_out)
        )

        self.res_conv1d_short = nn.Sequential(
            *(
                [nn.Conv1d(d_in, d_out, 1, 1), nn.BatchNorm1d(d_out)]
                if d_in != d_out else
                [nn.Identity()]
            )
        )

    def forward(self, x):
        return self.res_conv1d(x) + self.res_conv1d_short(x)


class Transpose(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
        self._name = 'transpose'

    def forward(self, x):
        return x.transpose(*self.shape)


class SDivide(nn.Module):
    def __init__(self, scale):
        super(SDivide, self).__init__()
        self.scale = scale
        self._name = 'scalar_divide'

    def forward(self, x):
        return x / self.scale


class SVD_Solver(nn.Module):

    def __init__(self, n_joints):
        super().__init__()
        self.n_joints = n_joints

    def forward(self, points, weight, offset):
        """
        :param points: (batch_size, n_max_markers, 3)
        :param weight: (batch_size, n_max_markers, n_joints)
        :param offset: (batch_size, n_max_markers, n_joints, 3)
        :return: (batch_size, n_joints, 3, 4)
        """

        output_size = (points.shape[0], self.n_joints, 3, 4)
        Y = torch.empty(output_size).to(torch.float32).to(points.device)
        # (batch_size, n_joints, 3, 4)

        X = points.permute(0, 2, 1).unsqueeze(1)  # (batch_size, 1, 3, n_max_markers)
        Z = offset.permute(0, 2, 3, 1)  # (batch_size, n_joints, 3, n_max_markers)
        w = weight.permute(0, 2, 1).unsqueeze(2)  # (batch_size, n_joints, 1, n_max_markers)
        R, t = SVD_Solver.svd_rot(Z, X, w)

        return R, t

        # for i in range(self.n_joints):
        #     X = points.permute(0, 2, 1)  # (batch_size, 3, n_max_markers)
        #     Z = offset[:, :, i].permute(0, 2, 1)  # (batch_size, 3, n_max_markers)
        #     w = weight[:, :, i].unsqueeze(1)  # (batch_size, 1, n_max_markers)
        #     R, t = SVD_Solver.svd_rot(Z, X, w)
        #     R_t = torch.cat((R, t), -1)
        #     Y[:, i] = R_t
        #
        # return Y

    @staticmethod
    def svd_rot(P, Q, w):
        d, n = P.shape[-2:]

        # X,Y are d x n
        P_ = torch.sum(P * w, dim=-1) / torch.sum(w, dim=-1)
        Q_ = torch.sum(Q * w, dim=-1) / torch.sum(w, dim=-1)
        X = P - P_[..., None]
        Y = Q - Q_[..., None]
        Yt = Y.permute(0, 1, 3, 2)

        # S is d x d
        S = torch.matmul(X, Yt)

        # U, V are d x d
        U, _, V = torch.svd(S)
        # V = V_t.permute(0, 2, 1)
        Ut = U.permute(0, 1, 3, 2)

        det = torch.det(torch.matmul(V, Ut))
        bn, jn = det.shape[:2]
        Ut_u = Ut[:, :, :-1, :]
        Ut_d = Ut[:, :, -1, :]
        Ut_d = Ut_d * det.view(bn, jn, 1)

        Ut = torch.cat((Ut_u, Ut_d.unsqueeze(2)), dim=2)

        # R is d x d
        R = torch.matmul(V, Ut)

        # t is d x n
        t = Q_.view(bn, jn, d, 1) - torch.matmul(R, P_.view(bn, jn, d, 1))

        return R, t


