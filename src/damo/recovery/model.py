import torch
import torch.nn as nn

class ChannelLayerNorm(nn.LayerNorm):

    def forward(self, x):
        return super().forward(x.movedim(1, -1)).movedim(-1, 1)

def _replace_batch_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            replacement = ChannelLayerNorm(child.num_features, eps=child.eps)
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            _replace_batch_norm(child)

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
        assert self.seq_len % 2 == 1, ValueError(f'seq_len ({self.seq_len}) must be odd number.')
        self.embedding = nn.Sequential(Transpose(-3, -1), ResConv2DBlock(3, self.d_model, self.d_hidden), nn.ReLU())
        attention_type = CenterAlignedAttention if getattr(options, 'attention_mode', 'legacy') == 'center' else LayeredMixedAttention
        self.attention_layers = attention_type(d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers, seq_len=self.seq_len)
        self.post_attention_layer = nn.Sequential(Transpose(-3, -1), nn.Conv2d(self.seq_len, 1, 1, 1), nn.ReLU(), Transpose(-2, -1))
        self.joint_feats_layer = nn.Sequential(nn.ReLU(), ResConv1DBlock(self.d_model, self.d_model, self.d_hidden), nn.ReLU(), nn.Conv1d(self.d_model, self.n_joints * 4, 1, 1))
        self.joint_indices_predictor = nn.Sequential(nn.ReLU(), ResConv1DBlock(self.n_joints * 4, self.n_joints + 1, self.d_hidden * 2))
        self.weight_predictor = nn.Sequential(nn.ReLU(), ResConv1DBlock(self.n_joints * 2 + 1, 3, self.d_hidden), nn.Softmax(dim=1), Transpose(-2, -1))
        self.offset_predictor = nn.Sequential(nn.ReLU(), ResConv1DBlock(self.n_joints * 4 + 1, 9, self.d_hidden * 2), Transpose(-2, -1))
        if getattr(options, 'normalization', 'batch') == 'layer':
            _replace_batch_norm(self)

    def forward(self, points_seq, points_mask):
        if points_seq.ndim != 4 or points_seq.shape[1] != self.seq_len or points_seq.shape[-1] != 3:
            raise ValueError('Expected points (batch, configured sequence length, markers, 3)')
        if points_mask.shape != points_seq.shape[:-1]:
            raise ValueError('Point mask shape does not match points')
        points_mask = points_mask.detach().to(dtype=points_seq.dtype)
        points_seq = points_seq.masked_fill(~points_mask.bool().unsqueeze(-1), 0.0)
        points_offset_seq = Damo.compute_offsets(points_seq, points_mask)
        points_centered_seq = points_seq - points_offset_seq
        if getattr(self.options, 'mask_centered_inputs', False):
            points_centered_seq = points_centered_seq.masked_fill(~points_mask.bool().unsqueeze(-1), 0.0)
        points_feats_seq = self.embedding(points_centered_seq)
        points_attention_seq = self.attention_layers(points_feats_seq, points_mask)
        center_seq_feats = self.post_attention_layer(points_attention_seq).squeeze(1)
        marker_configuration_feats = self.joint_feats_layer(center_seq_feats)
        joint_indices = self.joint_indices_predictor(marker_configuration_feats)
        if self.options.joint_distribution == 'softmax':
            joint_indices = joint_indices.softmax(dim=1)
        weight_feats = torch.cat((marker_configuration_feats[:, :self.n_joints, :], joint_indices), dim=1)
        weight = self.weight_predictor(weight_feats)
        offset_feats = torch.cat((marker_configuration_feats[:, self.n_joints:, :], joint_indices), dim=1)
        offset = self.offset_predictor(offset_feats)
        center_seq_idx = self.seq_len // 2
        center_seq_mask = points_mask[:, center_seq_idx, :].unsqueeze(-1)
        joint_indices = joint_indices.permute(0, 2, 1) * center_seq_mask
        weight = weight * center_seq_mask
        offset = offset * center_seq_mask
        batch_size, _, _ = offset.shape
        offset = offset.reshape(batch_size, points_seq.shape[2], 3, 3)
        return (joint_indices, weight, offset)

    @staticmethod
    def compute_offsets(points_seq, points_mask=None):
        nonzero_mask = (points_seq == 0.0).sum(-1) != 3 if points_mask is None else points_mask.bool()
        masked = points_seq.masked_fill(~nonzero_mask[..., None], float('nan'))
        return torch.nan_to_num(torch.nanmedian(masked, dim=2, keepdim=True).values, nan=0.0)

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
            self.total_attention_layers.append(nn.ModuleList([MultiHeadAttention(self.d_model, self.n_heads) for _ in range(self.seq_len)]))

    def make_score_mask(self, query_mask, key_mask):
        key_mask = key_mask.ne(0).unsqueeze(1).unsqueeze(2)
        query_mask = query_mask.ne(0).unsqueeze(1).unsqueeze(3)
        score_mask = key_mask & query_mask
        return score_mask

    def forward(self, points_attention_seq, points_mask):
        for attention_layers in self.total_attention_layers:
            attention_seq = []
            for idx, attention_layer in enumerate(attention_layers):
                query_idx = idx
                key_value_idx = self.center_seq_idx
                score_mask = self.make_score_mask(query_mask=points_mask[:, query_idx, :], key_mask=points_mask[:, key_value_idx, :])
                attention = attention_layer(points_attention_seq[:, :, :, query_idx], points_attention_seq[:, :, :, key_value_idx], points_attention_seq[:, :, :, key_value_idx], mask=score_mask)
                attention_seq.append(attention.unsqueeze(dim=3))
            points_attention_seq = torch.cat(attention_seq, dim=3)
        return points_attention_seq

class CenterAlignedAttention(LayeredMixedAttention):

    def __init__(self, d_model, n_heads, n_layers, seq_len):
        super().__init__(d_model, n_heads, n_layers, seq_len)
        self.fusions = nn.ModuleList([nn.Conv2d(seq_len, 1, 1) for _ in range(n_layers - 1)])

    def forward(self, features, points_mask):
        center_mask = points_mask[:, self.center_seq_idx]
        query = features[..., self.center_seq_idx]
        allowed_masks = [self.make_score_mask(center_mask, points_mask[:, frame]) for frame in range(self.seq_len)]
        for layer_index, branches in enumerate(self.total_attention_layers):
            outputs = []
            for frame, attention in enumerate(branches):
                memory = features[..., frame]
                allowed = allowed_masks[frame]
                value = attention(query, memory, memory, allowed)
                value = value * center_mask[:, None]
                outputs.append(value)
            aligned = torch.stack(outputs, dim=-1)
            if layer_index < len(self.fusions):
                update = self.fusions[layer_index](aligned.permute(0, 3, 1, 2)).squeeze(1)
                query = (query + torch.relu(update)) * center_mask[:, None]
        return aligned

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, ValueError(f'd_model ({d_model}) % n_heads ({n_heads}) is not 0 ({d_model % n_heads})')
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.proj = nn.ModuleList([nn.Conv1d(d_model, d_model, kernel_size=1) for _ in range(3)])
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.post_merge = nn.Sequential(nn.Conv1d(2 * d_model, 2 * d_model, kernel_size=1), nn.BatchNorm1d(2 * d_model), nn.ReLU(), nn.Conv1d(2 * d_model, d_model, kernel_size=1))
        nn.init.constant_(self.post_merge[-1].bias, 0.0)

    def forward(self, init_query, key, value, mask):
        batch_size = init_query.size(0)
        query, key, value = [l(x).view(batch_size, self.d_k, self.n_heads, -1) for l, x in zip(self.proj, (init_query, key, value))]
        x = MultiHeadAttention.scaled_dot_product_attention(query, key, value, mask)
        x = self.merge(x.contiguous().view(batch_size, self.d_k * self.n_heads, -1))
        x = self.post_merge(torch.cat([x, init_query], dim=1))
        return x

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask):
        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attention_weight = torch.nn.functional.softmax(scores, dim=-1)
        if mask is not None:
            attention_weight = attention_weight.masked_fill(~mask.bool(), 0.0)
        return torch.einsum('bhnm,bdhm->bdhn', attention_weight, value)

class ResConv2DBlock(nn.Module):

    def __init__(self, d_in, d_out, d_hidden):
        super().__init__()
        self.res_conv2d = nn.Sequential(nn.Conv2d(d_in, d_hidden, 1, 1), nn.BatchNorm2d(d_hidden), nn.ReLU(), nn.Conv2d(d_hidden, d_out, 1, 1), nn.BatchNorm2d(d_out))
        self.res_conv2d_short = nn.Sequential(*([nn.Conv2d(d_in, d_out, 1, 1), nn.BatchNorm2d(d_out)] if d_in != d_out else [nn.Identity()]))

    def forward(self, x):
        return self.res_conv2d(x) + self.res_conv2d_short(x)

class ResConv1DBlock(nn.Module):

    def __init__(self, d_in, d_out, d_hidden):
        super().__init__()
        self.res_conv1d = nn.Sequential(nn.Conv1d(d_in, d_hidden, 1, 1), nn.BatchNorm1d(d_hidden), nn.ReLU(), nn.Conv1d(d_hidden, d_out, 1, 1), nn.BatchNorm1d(d_out))
        self.res_conv1d_short = nn.Sequential(*([nn.Conv1d(d_in, d_out, 1, 1), nn.BatchNorm1d(d_out)] if d_in != d_out else [nn.Identity()]))

    def forward(self, x):
        return self.res_conv1d(x) + self.res_conv1d_short(x)

class Transpose(nn.Module):

    def __init__(self, *args):
        super().__init__()
        self.shape = args
        self._name = 'transpose'

    def forward(self, x):
        return x.transpose(*self.shape)
