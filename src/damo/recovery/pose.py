import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .geometry import forward_kinematics, lbs, validate_topology

def solve_frame(points, weights, offsets, bind_local, topology, *, initial=None, max_nfev=100, rotation_parameterization='quaternion'):
    parents = validate_topology(topology)
    points, weights, offsets = map(np.asarray, (points, weights, offsets))
    active = (weights.sum(-1) > 1e-08) & np.isfinite(points).all(-1)
    if not active.any():
        raise ValueError('No observed non-ghost markers to solve')
    points, weights, offsets = (points[active], weights[active], offsets[active])
    j = len(parents)
    width = 4 if rotation_parameterization == 'quaternion' else 3
    if rotation_parameterization not in ('quaternion', 'rotvec'):
        raise ValueError('Unknown rotation parameterization')
    if initial is None:
        initial = np.zeros(3 + width * j)
        if width == 4:
            initial[6::4] = 1
        rest = forward_kinematics(np.broadcast_to(np.eye(3), (j, 3, 3)), np.zeros(3), bind_local, parents)
        initial[:3] = np.median(points - lbs(rest, weights, offsets), axis=0)
    else:
        initial = np.asarray(initial, dtype=float).copy()
        if initial.shape != (3 + width * j,):
            raise ValueError('Initial pose has incorrect size')

    def transforms(params):
        raw = params[3:].reshape(j, width)
        if width == 4:
            norm = np.linalg.norm(raw, axis=-1, keepdims=True)
            q = np.divide(raw, norm, out=np.tile([0.0, 0.0, 0.0, 1.0], (j, 1)), where=norm > 1e-12)
            rot = Rotation.from_quat(q).as_matrix()
        else:
            rot = Rotation.from_rotvec(raw).as_matrix()
        return forward_kinematics(rot, params[:3], bind_local, parents)

    def residual(params):
        error = (lbs(transforms(params), weights, offsets) - points).ravel()
        if width == 4:
            error = np.concatenate([error, np.linalg.norm(params[3:].reshape(j, 4), axis=-1) - 1])
        return error
    method = 'lm' if len(residual(initial)) >= len(initial) else 'trf'
    result = least_squares(residual, initial, method=method, max_nfev=max_nfev, ftol=1e-07, xtol=1e-07, gtol=1e-07)
    jt = transforms(result.x)
    return {'params': result.x, 'transforms': jt, 'success': bool(result.success), 'message': str(result.message), 'nfev': result.nfev, 'method': method, 'marker_rmse': float(np.sqrt(np.mean((lbs(jt, weights, offsets) - points) ** 2)))}

def smooth_motion(root_positions, local_quaternions, bind_local, topology, window=31, order=3):
    from .postprocess import savgol_pose
    result = savgol_pose(root_positions, local_quaternions, bind_local, topology, window=window, order=order)
    return (result['transforms'], result['root_positions'], result['local_quaternions'])

def solve_sequence(points, weights, offsets, bind_local, topology, *, max_nfev=100, smoothing=True):
    params, transforms, diagnostics = ([], [], [])
    initial = None
    for frame in range(len(points)):
        if not (weights[frame].sum(-1) > 1e-08).any():
            if initial is None:
                params.append(None)
                transforms.append(None)
            else:
                params.append(initial.copy())
                transforms.append(transforms[-1].copy())
            diagnostics.append({'frame': frame, 'success': False, 'message': 'No usable markers; held nearest pose'})
            continue
        solved = solve_frame(points[frame], weights[frame], offsets[frame], bind_local, topology, initial=initial, max_nfev=max_nfev)
        initial = solved['params']
        params.append(initial.copy())
        transforms.append(solved['transforms'])
        diagnostics.append({'frame': frame, **{k: v for k, v in solved.items() if k not in ('params', 'transforms')}})
    first = next((i for i, p in enumerate(params) if p is not None), None)
    if first is None:
        raise ValueError('Entire sequence has no usable markers')
    for i in range(first):
        params[i] = params[first].copy()
        transforms[i] = transforms[first].copy()
    params = np.stack(params)
    raw = np.stack(transforms)
    roots = params[:, :3]
    quats = params[:, 3:].reshape(len(params), len(topology), 4)
    final, roots, quats = smooth_motion(roots, quats, bind_local, topology, window=31 if smoothing else 1)
    return {'transforms': final, 'raw_transforms': raw, 'root_positions': roots, 'local_quaternions': quats, 'diagnostics': diagnostics}
