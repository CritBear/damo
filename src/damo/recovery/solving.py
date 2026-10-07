import itertools
import numpy as np
from .geometry import decode_configuration, validate_topology
from .solver_variants import offset_skeleton, solve_frame_variant
from .fast_pose import solve_fast
from .pose import smooth_motion
DEFAULT_BACKEND = 'hybrid'
BACKENDS = ('quaternion', 'quaternion_exact', 'manifold', 'hybrid')

def estimate_configuration_skeleton(distribution, rep_weights, rep_offsets, mask, topology):
    parents = validate_topology(topology)
    n = len(parents)
    distribution = np.asarray(distribution)
    rep_weights = np.asarray(rep_weights)
    rep_offsets = np.asarray(rep_offsets)
    mask = np.asarray(mask, bool)
    if distribution.shape != mask.shape + (n + 1,) or rep_weights.shape != mask.shape + (3,) or rep_offsets.shape != mask.shape + (3, 3):
        raise ValueError('Configuration shapes do not match topology')
    if not all((np.isfinite(a[mask]).all() for a in (distribution, rep_weights, rep_offsets))):
        raise ValueError('Nonfinite observed configuration')
    if (rep_weights[mask] < 0).any():
        raise ValueError('Negative representative weights')
    ids = np.argsort(-distribution, axis=-1, kind='stable')[..., :3]
    valid = mask & (ids[..., 0] != n)
    bind, info = offset_skeleton(ids, rep_weights, rep_offsets, valid, parents)
    joints = np.zeros_like(bind)
    for j, p in enumerate(parents[1:], 1):
        joints[j] = joints[p] + bind[j]
    square = 0.0
    mass = 0.0
    for a, b in itertools.combinations(range(3), 2):
        use = valid & (ids[..., a] < n) & (ids[..., b] < n) & (rep_weights[..., a] > 0) & (rep_weights[..., b] > 0)
        ja = ids[..., a][use]
        jb = ids[..., b][use]
        weight = np.minimum(rep_weights[..., a][use], rep_weights[..., b][use])
        error = rep_offsets[..., a, :][use] - rep_offsets[..., b, :][use] - (joints[jb] - joints[ja])
        square += float((weight * np.sum(error ** 2, axis=-1)).sum())
        mass += float(weight.sum())
    info.update(method='offset_difference', constraint_rms_mm=1000 * np.sqrt(square / max(mass, 1e-30)), bone_lengths_m=np.linalg.norm(bind[1:], axis=-1).tolist(), scope='entire input sequence')
    return (bind, info)

def solve_sequence(points, weights, offsets, bind_local, topology, *, max_nfev=100, smoothing=True, backend=DEFAULT_BACKEND, retry_rmse=1e-05):
    if backend not in BACKENDS:
        raise ValueError('Unknown solver backend')
    params = []
    raw = []
    diagnostics = []
    for f in range(len(points)):
        observed = (weights[f].sum(-1) > 1e-08) & np.isfinite(points[f]).all(-1)
        if not observed.any():
            params.append(None if not params or params[-1] is None else params[-1].copy())
            raw.append(None if not raw or raw[-1] is None else raw[-1].copy())
            diagnostics.append({'frame': f, 'success': False, 'marker_fit_pass': False, 'message': 'No usable markers; held nearest pose'})
            continue
        if backend == 'quaternion':
            result = solve_frame_variant(points[f], weights[f], offsets[f], bind_local, topology, max_nfev=max_nfev)
            result['quality_pass'] = bool(result['success'] and result['marker_rmse'] <= retry_rmse)
        else:
            result = solve_fast(points[f], weights[f], offsets[f], bind_local, topology, backend=backend, max_nfev=max_nfev, retry_rmse=retry_rmse)
        params.append(result['params'].copy())
        raw.append(result['transforms'].copy())
        result['marker_fit_pass'] = result.pop('quality_pass')
        diagnostics.append({'frame': f, **{k: v for k, v in result.items() if k not in ('params', 'transforms')}})
    first = next((f for f, p in enumerate(params) if p is not None), None)
    if first is None:
        raise ValueError('Entire sequence has no usable markers')
    for f in range(first):
        params[f] = params[first].copy()
        raw[f] = raw[first].copy()
    params = np.stack(params)
    raw = np.stack(raw)
    roots = params[:, :3]
    quats = params[:, 3:].reshape(len(params), len(topology), 4)
    final, roots, quats = smooth_motion(roots, quats, bind_local, topology, window=31 if smoothing else 1)
    return {'transforms': final, 'raw_transforms': raw, 'root_positions': roots, 'local_quaternions': quats, 'diagnostics': diagnostics}
