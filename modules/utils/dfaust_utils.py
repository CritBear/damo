import torch
import numpy as np


def compute_vertex_normal(vertices, indices):
    # code obtained from
    # https://github.com/nghorbani/amass/blob/master/src/amass/data/dfaust_synthetic_mocap.py

    # code obtained from https://github.com/BachiLi/redner
    # redner/pyredner/shape.py
    def dot(v1, v2):
        # v1 := 13776 x 3
        # v1 := 13776 x 3
        # return := 13776

        return torch.sum(v1 * v2, dim=1)

    def squared_length(v):
        # v = 13776 x 3
        return torch.sum(v * v, dim=1)

    def length(v):
        # v = 13776 x 3
        # 13776
        return torch.sqrt(squared_length(v))

    # Nelson Max, "Weights for Computing Vertex Normals from Facet Vectors", 1999
    # vertices := 6890 x 3
    # indices := 13776 x 3
    normals = torch.zeros(vertices.shape, dtype=torch.float32, device=vertices.device)
    v = [vertices[indices[:, 0].long(), :],
         vertices[indices[:, 1].long(), :],
         vertices[indices[:, 2].long(), :]]

    for i in range(3):
        v0 = v[i]
        v1 = v[(i + 1) % 3]
        v2 = v[(i + 2) % 3]
        e1 = v1 - v0
        e2 = v2 - v0
        e1_len = length(e1)
        e2_len = length(e2)
        side_a = e1 / torch.reshape(e1_len, [-1, 1])  # 13776, 3
        side_b = e2 / torch.reshape(e2_len, [-1, 1])  # 13776, 3
        if i == 0:
            n = torch.cross(side_a, side_b, dim=-1)  # 13776, 3
            n = n / torch.reshape(length(n), [-1, 1])
        angle = torch.where(dot(side_a, side_b) < 0,
                            np.pi - 2.0 * torch.asin(0.5 * length(side_a + side_b)),
                            2.0 * torch.asin(0.5 * length(side_b - side_a)))
        sin_angle = torch.sin(angle)  # 13776

        # XXX: Inefficient but it's PyTorch's limitation
        contrib = n * (sin_angle / (e1_len * e2_len)).reshape(-1, 1).expand(-1, 3)  # 13776, 3
        index = indices[:, i].long().reshape(-1, 1).expand([-1, 3])  # torch.Size([13776, 3])
        normals.scatter_add_(0, index, contrib)

    normals = normals / torch.reshape(length(normals), [-1, 1])
    return normals.contiguous()