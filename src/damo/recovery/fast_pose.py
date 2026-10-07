import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .geometry import forward_kinematics, lbs, validate_topology
from .solver_variants import skew, initial_candidates

class TangentResidual:

    def __init__(self, points, weights, offsets, bind, topology):
        self.parents = validate_topology(topology)
        n = len(self.parents)
        active = (weights.sum(-1) > 1e-08) & np.isfinite(points).all(-1)
        if not active.any():
            raise ValueError('No usable observed markers')
        self.points = np.asarray(points[active], float)
        self.w = np.asarray(weights[active], float)
        self.z = np.asarray(offsets[active], float)
        self.bind = np.asarray(bind, float)
        if not np.isfinite(self.w).all() or not np.isfinite(self.z).all():
            raise ValueError('Nonfinite marker configuration')
        self.descendants = np.zeros((n, n))
        for j in range(n):
            k = j
            while k >= 0:
                self.descendants[k, j] = 1
                k = self.parents[k]
        self.support = self.w @ self.descendants.T

    def evaluate(self, local, root, jacobian=True):
        transforms = forward_kinematics(local, root, self.bind, self.parents)
        r = transforms[:, :3, :3]
        t = transforms[:, :3, 3]
        y = np.einsum('jab,mjb->mja', r, self.z) + t[None]
        weighted = y * self.w[..., None]
        error = (weighted.sum(1) - self.points).ravel()
        if not jacobian:
            return (error, transforms)
        lever = np.einsum('kj,mja->mka', self.descendants, weighted) - self.support[..., None] * t[None]
        rotation_jac = -skew(lever) @ r[None]
        jac = np.empty((len(self.points), 3, 3 + 3 * len(r)))
        jac[:, :, :3] = self.w.sum(-1)[:, None, None] * np.eye(3)
        jac[:, :, 3:] = rotation_jac.transpose(0, 2, 1, 3).reshape(len(self.points), 3, -1)
        return (error, transforms, jac.reshape(-1, jac.shape[-1]))

class QuaternionResidual:

    def __init__(self, *args):
        self.tangent = TangentResidual(*args)
        self.n = len(self.tangent.parents)
        self._x = None

    def evaluate(self, x):
        if self._x is not None and np.array_equal(x, self._x):
            return
        self._x = np.array(x, copy=True)
        raw = x[3:].reshape(self.n, 4)
        norm = np.linalg.norm(raw, axis=-1, keepdims=True)
        if (norm < 1e-12).any():
            raise ValueError('Degenerate quaternion in analytic solver')
        unit = raw / norm
        local = Rotation.from_quat(unit).as_matrix()
        error, self.transforms, tangent = self.tangent.evaluate(local, x[:3])
        self.local = local
        self.error = np.r_[error, norm.ravel() - 1]
        q_to_tangent = 2 * np.concatenate([unit[:, 3, None, None] * np.eye(3) - skew(unit[:, :3]), -unit[:, :3, None]], axis=-1) / norm[..., None]
        rotation = np.einsum('rjk,jkq->rjq', tangent[:, 3:].reshape(-1, self.n, 3), q_to_tangent)
        self.jacobian = np.zeros((len(error) + self.n, len(x)))
        self.jacobian[:len(error), :3] = tangent[:, :3]
        self.jacobian[:len(error), 3:] = rotation.reshape(len(error), -1)
        for j in range(self.n):
            self.jacobian[len(error) + j, 3 + 4 * j:7 + 4 * j] = unit[j]

    def fun(self, x):
        self.evaluate(x)
        return self.error

    def jac(self, x):
        self.evaluate(x)
        return self.jacobian

def solve_quaternion_exact(points, weights, offsets, bind, topology, *, initial, max_nfev=100):
    objective = QuaternionResidual(points, weights, offsets, bind, topology)
    method = 'lm' if len(objective.fun(initial)) >= len(initial) else 'trf'
    result = least_squares(objective.fun, initial, jac=objective.jac, method=method, max_nfev=max_nfev, ftol=1e-07, xtol=1e-07, gtol=1e-07)
    objective.evaluate(result.x)
    error = objective.error[:-len(topology)]
    return {'params': result.x, 'transforms': objective.transforms.copy(), 'success': bool(result.success), 'nfev': result.nfev, 'message': str(result.message), 'method': 'quaternion_exact_' + method, 'marker_rmse': float(np.sqrt(np.mean(error ** 2)))}

def solve_manifold(points, weights, offsets, bind, topology, *, initial, max_nfev=100):
    if max_nfev < 1:
        raise ValueError('max_nfev must be positive')
    objective = TangentResidual(points, weights, offsets, bind, topology)
    local = Rotation.from_quat(initial[3:].reshape(-1, 4)).as_matrix()
    root = np.array(initial[:3], float)
    error, transforms, jac = objective.evaluate(local, root)
    cost = 0.5 * error @ error
    damping = 0.001
    nu = 2.0
    nfev = 1
    accepted = 0
    success = False
    trace = [float(cost)]
    message = 'Maximum function evaluations reached'
    max_step = 0.0
    while nfev < max_nfev:
        scale = np.maximum(np.linalg.norm(jac, axis=0), 1e-10)
        scaled = jac / scale
        normal = scaled.T @ scaled
        gradient = scaled.T @ error
        if np.max(abs(gradient)) <= 1e-07 * max(np.linalg.norm(error), 1e-15):
            success = True
            message = 'Scaled gradient tolerance'
            break
        try:
            step_scaled = cho_solve(cho_factor(normal + damping * np.eye(len(gradient)), lower=True, check_finite=False), -gradient, check_finite=False)
        except np.linalg.LinAlgError:
            damping *= nu
            nu = min(nu * 2, 100000000.0)
            if not np.isfinite(damping) or damping > 1e+20:
                message = 'Singular damped system'
                break
            continue
        step = step_scaled / scale
        rotation_step = np.linalg.norm(step[3:].reshape(-1, 3), axis=-1).max()
        factor = min(1.0, 0.5 / max(rotation_step, 1e-30), 0.5 / max(np.linalg.norm(step[:3]), 1e-30))
        step *= factor
        step_scaled *= factor
        predicted = -gradient @ step_scaled - 0.5 * step_scaled @ normal @ step_scaled
        if not np.isfinite(predicted) or predicted <= 0:
            message = 'No positive predicted reduction'
            break
        trial_local = local @ Rotation.from_rotvec(step[3:].reshape(-1, 3)).as_matrix()
        trial_root = root + step[:3]
        trial_error, trial_transforms = objective.evaluate(trial_local, trial_root, jacobian=False)
        nfev += 1
        trial_cost = 0.5 * trial_error @ trial_error
        gain = (cost - trial_cost) / predicted
        if np.isfinite(trial_cost) and trial_cost < cost:
            previous = cost
            local = trial_local
            root = trial_root
            cost = trial_cost
            error = trial_error
            transforms = trial_transforms
            accepted += 1
            trace.append(float(cost))
            max_step = max(max_step, float(rotation_step * factor))
            damping = max(1e-12, damping * (1 / 3 if gain >= 1 else max(1 / 3, 1 - (2 * gain - 1) ** 3)))
            nu = 2.0
            if gain > 0.25 and previous - cost < 1e-07 * previous:
                success = True
                message = 'Relative cost tolerance'
                break
            if np.linalg.norm(step[:3]) < 1e-07 and rotation_step * factor < 1e-07:
                success = True
                message = 'Tangent step tolerance'
                break
            error, transforms, jac = objective.evaluate(local, root)
        else:
            damping *= nu
            nu = min(nu * 2, 100000000.0)
            if not np.isfinite(damping) or damping > 1e+20:
                message = 'Damping overflow'
                break
    return {'params': np.r_[root, Rotation.from_matrix(local).as_quat().ravel()], 'transforms': transforms, 'success': success, 'nfev': nfev, 'message': message, 'method': 'retracted_lm', 'marker_rmse': float(np.sqrt(np.mean(error ** 2))), 'accepted_steps': accepted, 'max_accepted_rotation_step': max_step, 'accepted_costs': trace}

def solve_fast(points, weights, offsets, bind, topology, *, backend='hybrid', max_nfev=100, retry_rmse=1e-05):
    if backend not in ('quaternion_exact', 'manifold', 'hybrid'):
        raise ValueError('Unknown fast backend')
    seeds = initial_candidates(points, weights, offsets, bind, topology)
    fn = solve_manifold if backend == 'manifold' else solve_quaternion_exact
    first = fn(points, weights, offsets, bind, topology, initial=seeds['joint_svd'], max_nfev=max_nfev)
    candidates = {'joint_svd': first}
    if backend == 'hybrid' and (not first['success'] or first['marker_rmse'] > retry_rmse):
        candidates['manifold_svd'] = solve_manifold(points, weights, offsets, bind, topology, initial=seeds['joint_svd'], max_nfev=max_nfev)
        for key in ('root_rigid', 'rest'):
            candidates[key] = solve_quaternion_exact(points, weights, offsets, bind, topology, initial=seeds[key], max_nfev=max_nfev)
    selected = min(candidates, key=lambda k: candidates[k]['marker_rmse'])
    result = {k: v for k, v in candidates[selected].items() if k != 'accepted_costs'}
    result.update(selected_seed=selected, attempts=len(candidates), total_nfev=sum((v['nfev'] for v in candidates.values())), quality_pass=bool(candidates[selected]['success'] and candidates[selected]['marker_rmse'] <= retry_rmse), retry_rmse=retry_rmse)
    return result
