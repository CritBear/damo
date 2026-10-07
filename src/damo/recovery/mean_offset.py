import numpy as np

def mean_inverse_bind_markers(points, valid, transforms, bind_joints, ranks, weights, *, block_size=256, max_condition=10000.0):
    points = np.asarray(points, float)
    valid = np.asarray(valid, bool)
    transforms = np.asarray(transforms, float)
    bind_joints = np.asarray(bind_joints, float)
    ranks = np.asarray(ranks)
    weights = np.asarray(weights, float)
    n, m = valid.shape
    if points.shape != (n, m, 3) or transforms.shape != (n, 22, 4, 4) or bind_joints.shape != (22, 3) or (ranks.shape != (m, 3)) or (weights.shape != (m, 3)) or (ranks.dtype.kind not in 'iu') or ((ranks < 0) | (ranks >= 22)).any() or (weights < 0).any() or (not np.allclose(weights.sum(-1), 1, atol=1e-06)):
        raise ValueError('Invalid inverse-mean dimensions/weights')
    if not all((np.isfinite(x).all() for x in (points[valid], transforms, bind_joints, weights))):
        raise ValueError('Nonfinite inverse-mean input')
    count = valid.sum(0)
    if (count == 0).any():
        raise ValueError('A marker has no valid frames; no mean can be defined')
    sums = np.zeros((m, 3))
    squares = np.zeros((m, 3))
    worst = 0.0
    roundtrip = 0.0
    for start in range(0, n, block_size):
        frame, marker = np.nonzero(valid[start:start + block_size])
        frame += start
        if not len(frame):
            continue
        w = weights[marker]
        jt = transforms[frame[:, None], ranks[marker]]
        j0 = bind_joints[ranks[marker]]
        a = (w[..., None, None] * jt[..., :3, :3]).sum(1)
        cond = np.linalg.cond(a)
        if not np.isfinite(cond).all() or (cond >= max_condition).any():
            raise ValueError('Ill-conditioned inverse frame; do not silently drop or reweight it')
        worst = max(worst, float(cond.max()))
        shift = jt[..., :3, 3] - np.einsum('mkab,mkb->mka', jt[..., :3, :3], j0)
        q = np.linalg.solve(a, (points[frame, marker] - (w[..., None] * shift).sum(1))[..., None])[..., 0]
        check = np.einsum('mab,mb->ma', a, q) + (w[..., None] * shift).sum(1)
        roundtrip = max(roundtrip, float(np.linalg.norm(check - points[frame, marker], axis=-1).max()))
        np.add.at(sums, marker, q)
        np.add.at(squares, marker, q * q)
    mean = sums / count[:, None]
    variance = np.maximum(0, squares / count[:, None] - mean * mean)
    return (mean, dict(valid_frames_per_marker=count.tolist(), arithmetic_mean=True, missing_excluded=True, outlier_trimming=False, temporal_weighting=False, pooled_across_motions=False, per_marker_canonical_std_xyz_mm=(np.sqrt(variance) * 1000).tolist(), per_marker_canonical_rms_deviation_mm=(np.sqrt(variance.sum(-1)) * 1000).tolist(), inverse_lbs_max_condition=worst, inverse_roundtrip_max_mm=roundtrip * 1000))
