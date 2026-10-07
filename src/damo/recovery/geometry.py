import numpy as np
from scipy.spatial.transform import Rotation

def validate_topology(topology):
    parents = np.asarray(topology, dtype=np.int64)
    if parents.ndim != 1 or len(parents) == 0 or parents[0] != -1:
        raise ValueError('Topology needs one root at index zero, with parent -1')
    if any((p < 0 or p >= j for j, p in enumerate(parents[1:], 1))):
        raise ValueError('Parents must precede their children')
    return parents

def forward_kinematics(rotations, root_positions, bind_local, topology):
    parents = validate_topology(topology)
    rotations = np.asarray(rotations)
    root_positions = np.asarray(root_positions)
    bind_local = np.asarray(bind_local)
    if rotations.shape[-3:] != (len(parents), 3, 3):
        raise ValueError('Rotations must have shape (..., joints, 3, 3)')
    transforms = np.broadcast_to(np.eye(4), rotations.shape[:-2] + (4, 4)).copy()
    transforms[..., :3, :3] = rotations
    transforms[..., :3, 3] = bind_local
    transforms[..., 0, :3, 3] = root_positions
    for j, p in enumerate(parents[1:], 1):
        transforms[..., j, :, :] = transforms[..., p, :, :] @ transforms[..., j, :, :]
    return transforms

def lbs(transforms, weights, offsets):
    rotated = np.einsum('...jab,...mjb->...mja', transforms[..., :3, :3], offsets)
    positioned = rotated + transforms[..., None, :, :3, 3]
    return np.sum(positioned * weights[..., None], axis=-2)

def fit_bind_markers(points, transforms, weights, bind_global):
    a = np.einsum('mj,jab->mab', weights, transforms[:, :3, :3])
    shift = transforms[:, :3, 3] - np.einsum('jab,jb->ja', transforms[:, :3, :3], bind_global)
    b = np.asarray(points) - weights @ shift
    return np.stack([np.linalg.lstsq(ai, bi, rcond=None)[0] for ai, bi in zip(a, b)])

def weighted_rigid_alignment(local, world, weights):
    local, world, weights = map(np.asarray, (local, world, weights))
    valid = (weights > 0) & np.isfinite(weights) & np.isfinite(local).all(-1) & np.isfinite(world).all(-1)
    if valid.sum() < 3:
        return (np.eye(3), np.full(3, np.nan), False)
    p, q, w = (local[valid], world[valid], weights[valid])
    w = w / w.sum()
    pc, qc = (np.sum(p * w[:, None], axis=0), np.sum(q * w[:, None], axis=0))
    h = (p - pc).T @ ((q - qc) * w[:, None])
    u, s, vt = np.linalg.svd(h)
    if s[1] <= max(s[0] * 1e-08, 1e-14):
        return (np.eye(3), np.full(3, np.nan), False)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    return (rotation, qc - rotation @ pc, True)

def estimate_joint_transforms(points, weights, offsets, threshold=0.01):
    f, _, j = weights.shape
    transforms = np.broadcast_to(np.eye(4), (f, j, 4, 4)).copy()
    valid = np.zeros((f, j), dtype=bool)
    for frame in range(f):
        for joint in range(j):
            w = np.where(weights[frame, :, joint] > threshold, weights[frame, :, joint], 0)
            r, t, ok = weighted_rigid_alignment(offsets[frame, :, joint], points[frame], w)
            transforms[frame, joint, :3, :3] = r
            transforms[frame, joint, :3, 3] = t
            valid[frame, joint] = ok
    return (transforms, valid)

def iqr_mean(values, factor=1.5):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    q1, q3 = np.quantile(values, [0.25, 0.75])
    keep = (values >= q1 - factor * (q3 - q1)) & (values <= q3 + factor * (q3 - q1))
    return float(values[keep].mean()) if keep.any() else np.nan

def estimate_skeleton(joint_positions, base_global, topology):
    parents = validate_topology(topology)
    base_global = np.asarray(base_global)
    base_local = np.zeros((len(parents), 3))
    base_local[1:] = base_global[1:] - base_global[parents[1:]]
    base_lengths = np.linalg.norm(base_local, axis=-1)
    observed = np.full(len(parents), np.nan)
    for j, p in enumerate(parents[1:], 1):
        observed[j] = iqr_mean(np.linalg.norm(joint_positions[:, j] - joint_positions[:, p], axis=-1))
    valid = np.isfinite(observed) & (observed > 0) & (base_lengths > 1e-08)
    scale = float(np.dot(base_lengths[valid], observed[valid]) / np.dot(base_lengths[valid], base_lengths[valid])) if valid.any() else 1.0
    lengths = np.where(valid, observed, base_lengths * scale)
    result = base_local * (lengths / np.maximum(base_lengths, 1e-12))[:, None]
    return (result, {'scale': scale, 'observed_bones': valid.tolist()})

def decode_configuration(indices, rep_weights, rep_offsets, mask):
    indices, rep_weights, rep_offsets = map(np.asarray, (indices, rep_weights, rep_offsets))
    n_joints = indices.shape[-1] - 1
    ranks = np.argsort(-indices, axis=-1, kind='stable')[..., :3]
    valid = np.asarray(mask, dtype=bool) & (ranks[..., 0] != n_joints)
    shape = indices.shape[:-1]
    weights = np.zeros(shape + (n_joints + 1,), dtype=rep_weights.dtype)
    offsets = np.zeros(shape + (n_joints + 1, 3), dtype=rep_offsets.dtype)
    np.put_along_axis(weights, ranks, rep_weights, axis=-1)
    np.put_along_axis(offsets, np.broadcast_to(ranks[..., None], ranks.shape + (3,)), rep_offsets, axis=-2)
    weights = weights[..., :n_joints] * valid[..., None]
    offsets = offsets[..., :n_joints, :] * valid[..., None, None]
    total = weights.sum(-1, keepdims=True)
    valid &= total[..., 0] > 1e-08
    weights = weights / np.maximum(total, 1e-08)
    return (weights, offsets, valid)
