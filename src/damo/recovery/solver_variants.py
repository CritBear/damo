import itertools
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .geometry import validate_topology, forward_kinematics, lbs, estimate_joint_transforms, weighted_rigid_alignment
from .pose import solve_frame

def offset_skeleton(joint_ids, rep_weights, rep_offsets, valid, topology, *, min_weight=0.0, max_condition=10000000000.0):
    parents = validate_topology(topology)
    n = len(parents)
    ids = np.asarray(joint_ids)
    w = np.asarray(rep_weights, float)
    z = np.asarray(rep_offsets, float)
    valid = np.asarray(valid, bool)
    if ids.shape != w.shape or z.shape != w.shape + (3,) or valid.shape != w.shape[:-1]:
        raise ValueError('Mismatched representative configuration shapes')
    normal = np.zeros((n, n))
    rhs = np.zeros((n, 3))
    adjacency = np.zeros((n, n), bool)
    pairs = 0
    for a, b in itertools.combinations(range(w.shape[-1]), 2):
        use = valid & np.isfinite(w[..., a]) & np.isfinite(w[..., b]) & (w[..., a] > min_weight) & (w[..., b] > min_weight) & np.isfinite(z[..., a, :]).all(-1) & np.isfinite(z[..., b, :]).all(-1) & (ids[..., a] >= 0) & (ids[..., a] < n) & (ids[..., b] >= 0) & (ids[..., b] < n) & (ids[..., a] != ids[..., b])
        ja = ids[..., a][use]
        jb = ids[..., b][use]
        weight = np.minimum(w[..., a][use], w[..., b][use])
        delta = z[..., a, :][use] - z[..., b, :][use]
        np.add.at(normal, (ja, ja), weight)
        np.add.at(normal, (jb, jb), weight)
        np.add.at(normal, (ja, jb), -weight)
        np.add.at(normal, (jb, ja), -weight)
        np.add.at(rhs, ja, -weight[:, None] * delta)
        np.add.at(rhs, jb, weight[:, None] * delta)
        adjacency[ja, jb] = True
        adjacency[jb, ja] = True
        pairs += len(ja)
    reached = {0}
    todo = [0]
    while todo:
        new = set(map(int, np.flatnonzero(adjacency[todo.pop()]))) - reached
        reached.update(new)
        todo.extend(new)
    if len(reached) != n:
        raise ValueError(f'Offset graph disconnected: root component {len(reached)}/{n}')
    condition = float(np.linalg.cond(normal[1:, 1:]))
    if not np.isfinite(condition) or condition > max_condition:
        raise ValueError(f'Offset graph ill-conditioned: {condition}')
    joints = np.zeros((n, 3))
    joints[1:] = np.linalg.solve(normal[1:, 1:], rhs[1:])
    bind = np.zeros_like(joints)
    bind[1:] = joints[1:] - joints[parents[1:]]
    return (bind, {'condition': condition, 'pair_observations': pairs, 'min_weight': min_weight, 'root_anchored': True})

def make_initial(rotations, points, weights, offsets, bind, parents):
    transforms = forward_kinematics(rotations, np.zeros(3), bind, parents)
    active = (weights.sum(-1) > 1e-08) & np.isfinite(points).all(-1)
    if not active.any():
        raise ValueError('No usable markers')
    root = np.median((points - lbs(transforms, weights, offsets))[active], axis=0)
    return np.r_[root, Rotation.from_matrix(rotations).as_quat().ravel()]

def initial_candidates(points, weights, offsets, bind, topology):
    parents = validate_topology(topology)
    n = len(parents)
    identity = np.tile(np.eye(3), (n, 1, 1))
    rest = forward_kinematics(identity, np.zeros(3), bind, parents)
    active = (weights.sum(-1) > 1e-08) & np.isfinite(points).all(-1)
    root_r, _, ok = weighted_rigid_alignment(lbs(rest, weights, offsets), points, active.astype(float))
    aligned = identity.copy()
    if ok:
        aligned[0] = root_r
    transforms, observable = estimate_joint_transforms(points[None], weights[None], offsets[None])
    global_r = identity.copy()
    local_r = identity.copy()
    for j, p in enumerate(parents):
        global_r[j] = transforms[0, j, :3, :3] if observable[0, j] else aligned[0] if j == 0 else global_r[p]
        local_r[j] = global_r[j] if j == 0 else global_r[p].T @ global_r[j]
    return {'joint_svd': make_initial(local_r, points, weights, offsets, bind, parents), 'root_rigid': make_initial(aligned, points, weights, offsets, bind, parents), 'rest': make_initial(identity, points, weights, offsets, bind, parents)}

def skew(v):
    v = np.asarray(v)
    out = np.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype)
    x, y, z = np.moveaxis(v, -1, 0)
    out[..., 0, 1] = -z
    out[..., 0, 2] = y
    out[..., 1, 0] = z
    out[..., 1, 2] = -x
    out[..., 2, 0] = -y
    out[..., 2, 1] = x
    return out

def right_jacobian(v):
    theta2 = np.sum(v * v, axis=-1)
    small = theta2 < 1e-08
    theta = np.sqrt(np.maximum(theta2, 1e-30))
    a = np.where(small, 0.5 - theta2 / 24 + theta2 ** 2 / 720, (1 - np.cos(theta)) / np.maximum(theta2, 1e-30))
    b = np.where(small, 1 / 6 - theta2 / 120 + theta2 ** 2 / 5040, (theta - np.sin(theta)) / theta ** 3)
    k = skew(v)
    return np.eye(3) - a[..., None, None] * k + b[..., None, None] * (k @ k)

class PoseResidual:

    def __init__(self, points, weights, offsets, bind, topology, base_rotations):
        self.parents = validate_topology(topology)
        n = len(self.parents)
        self.points = np.asarray(points, float)
        self.w = np.asarray(weights, float)
        self.z = np.asarray(offsets, float)
        self.bind = np.asarray(bind, float)
        self.base = np.asarray(base_rotations, float)
        descendants = np.zeros((n, n))
        for j in range(n):
            k = j
            while k >= 0:
                descendants[k, j] = 1
                k = self.parents[k]
        self.descendants = descendants
        self.support = self.w @ descendants.T
        self._x = None

    def evaluate(self, x):
        if self._x is not None and np.array_equal(x, self._x):
            return
        self._x = np.array(x, copy=True)
        v = x[3:].reshape(-1, 3)
        self.local = self.base @ Rotation.from_rotvec(v).as_matrix()
        self.transforms = forward_kinematics(self.local, x[:3], self.bind, self.parents)
        r = self.transforms[:, :3, :3]
        t = self.transforms[:, :3, 3]
        y = np.einsum('jab,mjb->mja', r, self.z) + t[None]
        weighted = y * self.w[..., None]
        self.error = (weighted.sum(1) - self.points).ravel()
        lever = np.einsum('kj,mja->mka', self.descendants, weighted) - self.support[..., None] * t[None]
        rotation_jac = -skew(lever) @ (r @ right_jacobian(v))[None]
        jac = np.empty((len(self.points), 3, 3 + 3 * len(r)))
        jac[:, :, :3] = self.w.sum(-1)[:, None, None] * np.eye(3)
        jac[:, :, 3:] = rotation_jac.transpose(0, 2, 1, 3).reshape(len(self.points), 3, -1)
        self.jacobian = jac.reshape(-1, jac.shape[-1])

    def fun(self, x):
        self.evaluate(x)
        return self.error

    def jac(self, x):
        self.evaluate(x)
        return self.jacobian

def solve_frame_exact(points, weights, offsets, bind, topology, *, initial, max_nfev=100):
    active = (weights.sum(-1) > 1e-08) & np.isfinite(points).all(-1)
    if not active.any():
        raise ValueError('No usable markers')
    base = Rotation.from_quat(initial[3:].reshape(-1, 4)).as_matrix()
    residual = PoseResidual(points[active], weights[active], offsets[active], bind, topology, base)
    x0 = np.r_[initial[:3], np.zeros(3 * len(topology))]
    method = 'lm' if 3 * active.sum() >= len(x0) else 'trf'
    result = least_squares(residual.fun, x0, jac=residual.jac, method=method, max_nfev=max_nfev, ftol=1e-07, xtol=1e-07, gtol=1e-07)
    residual.evaluate(result.x)
    return {'params': np.r_[result.x[:3], Rotation.from_matrix(residual.local).as_quat().ravel()], 'transforms': residual.transforms.copy(), 'success': bool(result.success), 'message': str(result.message), 'nfev': result.nfev, 'method': method, 'marker_rmse': float(np.sqrt(np.mean(residual.error ** 2)))}

def solve_frame_variant(points, weights, offsets, bind, topology, *, initialization='joint_svd', exact=False, max_nfev=100, retry_rmse=0.0001):
    seeds = initial_candidates(points, weights, offsets, bind, topology)
    if initialization not in seeds:
        raise ValueError('Unknown initialization')

    def run(key):
        solver = solve_frame_exact if exact else solve_frame
        return solver(points, weights, offsets, bind, topology, initial=seeds[key], max_nfev=max_nfev)
    results = {initialization: run(initialization)}
    first = results[initialization]
    if exact and (not first['success'] or first['marker_rmse'] > retry_rmse):
        for key in ('root_rigid', 'rest'):
            if key not in results:
                results[key] = run(key)
    selected = min(results, key=lambda key: results[key]['marker_rmse'])
    answer = dict(results[selected])
    answer.update(selected_seed=selected, attempts=len(results), total_nfev=sum((r['nfev'] for r in results.values())))
    return answer
