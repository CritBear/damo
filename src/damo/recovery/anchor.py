import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from damo.recovery.geometry import weighted_rigid_alignment, decode_configuration, forward_kinematics
from damo.recovery.solver_variants import initial_candidates
from damo.recovery.base import sha256
from .skeleton_prior import SkeletonProblem
CONFIG = dict(train_weighting='equal source group, equal original motion within group', pca_variance=0.99, pca_max_components=16, coefficient_bound_std=3.0, scale_bounds='0.9 train p01 to 1.1 train p99', candidate_frames=64, fit_poses=24, frame_selection='observed joint-anchor normalized geometry farthest-first from middle candidate; evenly spaced candidate pool', attachment_threshold=0.01, robust_rigid_iterations=3, robust_rigid_sigma_m=0.03, anchor_sigma_m=0.04, marker_sigma_m=0.03, confidence='effective support / 4, second-axis spread / (spread+.02), mean attachment, residual and leave-one-marker-out translation stability', calibrated_confidence=False, anchor_min_quality=0.03, offset_weight=0.5, anchor_weight=1.0, marker_weight=1.0, shape_prior_weight=0.2, symmetry_weight=0.2, rotation_prior_weight=0.05, rotation_sigma_rad=0.35, hard_length_limits=False, length_limit_weight=4.0, direction_limit_weight=4.0, direction_allowance_rad=0.1, initial_shape_max_nfev=100, joint_fit_max_iter=180, joint_fit_max_eval=260, robust_loss='pseudo-Huber on 3D vector norms; unit dimensionless transition', ground_truth_in_fit=False, pose_smoothing=False, one_skeleton_per_motion_window=True, final_pose_solver='unchanged hybrid solver, 100 evaluations per attempt; shape fitting has additional budget')
SYMMETRY = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)]

def quantile(x, w, q):
    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]
    return np.interp(q, (np.cumsum(w) - 0.5 * w) / w.sum(), x)

def shape_numpy(x, prior):
    j = np.zeros((22, 3))
    j[1:] = (prior['mean'] + prior['basis'] @ x[1:]).reshape(21, 3) * np.exp(x[0])
    b = np.zeros_like(j)
    b[1:] = j[1:] - j[prior['parents'][1:]]
    return (j, b)

def shape_penalty_numpy(x, prior):
    j, b = shape_numpy(x, prior)
    length = np.maximum(np.linalg.norm(b[1:], axis=1), 1e-09)
    pairs = np.array(SYMMETRY) - 1
    ratio = np.log(length[pairs[:, 0]] / length[pairs[:, 1]])
    cosine = np.sum(b[1:] / length[:, None] * prior['mean_direction'], axis=1)
    return np.r_[np.sqrt(0.2) * x[1:] / np.sqrt(len(x) - 1), np.sqrt(0.2) * (x[0] - prior['log_scale_mean']) / prior['log_scale_std'], np.sqrt(0.2 / len(pairs)) * (ratio - prior['ratio_mean']) / prior['ratio_std'], np.sqrt(4 / 21) * np.minimum(np.log(length / prior['length_low']), 0) / 0.05, np.sqrt(4 / 21) * np.maximum(np.log(length / prior['length_high']), 0) / 0.05, np.sqrt(4 / 21) * np.minimum(cosine - np.cos(prior['direction_limit']), 0) / 0.02]

def robust_anchor(points, weights, offsets):
    keep = weights > CONFIG['attachment_threshold']
    p = offsets[keep]
    x = points[keep]
    a = weights[keep].astype(float)
    if len(a) < 3:
        return (np.eye(3), np.zeros(3), 0.0, dict(reason='fewer_than_three_markers'))
    effective = a.copy()
    ok = False
    for _ in range(CONFIG['robust_rigid_iterations']):
        r, t, ok = weighted_rigid_alignment(p, x, effective)
        if not ok:
            return (np.eye(3), np.zeros(3), 0.0, dict(reason='degenerate_marker_geometry'))
        residual = np.linalg.norm(p @ r.T + t - x, axis=1)
        effective = a / np.sqrt(1 + (residual / 0.03) ** 2)
    wn = effective / effective.sum()
    neff = 1 / np.sum(wn ** 2)
    center = p - np.sum(p * wn[:, None], axis=0)
    eig = np.linalg.eigvalsh((center * wn[:, None]).T @ center)
    spread = float(np.sqrt(max(eig[-2], 0)))
    rmse = float(np.sqrt(np.sum(wn * residual ** 2)))
    omitted = []
    for m in range(len(a)):
        ww = effective.copy()
        ww[m] = 0.0
        rr, tt, valid = weighted_rigid_alignment(p, x, ww)
        if valid:
            omitted.append(np.linalg.norm(tt - t))
    stability = float(np.quantile(omitted, 0.8)) if omitted else 1.0
    support = min(neff / 4.0, 1.0) * (spread / (spread + 0.02)) * float(np.sum(wn * a))
    quality = support / (1 + (rmse / 0.03) ** 2 + (stability / 0.03) ** 2)
    if quality < CONFIG['anchor_min_quality']:
        quality = 0.0
    return (r, t, quality, dict(effective_markers=float(neff), spread_m=spread, rigid_rmse_m=rmse, leave_one_out_p80_m=stability, quality=quality, markers=len(a)))

def make_anchors(points, w, z):
    f = len(points)
    rot = np.broadcast_to(np.eye(3), (f, 22, 3, 3)).copy()
    pos = np.zeros((f, 22, 3))
    q = np.zeros((f, 22))
    details = []
    for k in range(f):
        row = []
        for j in range(22):
            rot[k, j], pos[k, j], q[k, j], info = robust_anchor(points[k], w[k, :, j], z[k, :, j])
            row.append(info)
        details.append(row)
    return (rot, pos, q, details)

def fit(points, full, rep, offsets, mask, marker_index, parents, prior, prior_dir):
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    begin = time.perf_counter()
    w, z, use = decode_configuration(full, rep, offsets, mask)
    problem = SkeletonProblem(full, rep, offsets, mask, marker_index, parents, prior_dir)
    h, b, mass = problem.equations()
    vals, vecs = np.linalg.eigh(h)
    assert vals.min() > max(vals.max() * 1e-12, 1e-12)
    factor = np.sqrt(vals)[:, None] * vecs.T / (np.sqrt(mass) * 0.03)
    target = vecs.T @ b / np.sqrt(vals)[:, None] / (np.sqrt(mass) * 0.03)
    x0 = np.r_[prior['log_scale_mean'], np.zeros(prior['basis'].shape[1])]
    low = np.r_[prior['log_scale_bounds'][0], np.full(len(x0) - 1, -3.0)]
    high = np.r_[prior['log_scale_bounds'][1], np.full(len(x0) - 1, 3.0)]

    def initial_residual(x):
        j, _ = shape_numpy(x, prior)
        return np.r_[np.sqrt(0.5) * (factor @ j[1:] - target).ravel(), shape_penalty_numpy(x, prior)]
    initial = least_squares(initial_residual, x0, bounds=(low, high), max_nfev=100, ftol=1e-08, xtol=1e-08, gtol=1e-08)
    _, initial_bind = shape_numpy(initial.x, prior)
    pool = np.unique(np.linspace(0, len(points) - 1, min(len(points), CONFIG['candidate_frames'])).astype(int))
    ar, ap, aq, details = make_anchors(points[pool], w[pool], z[pool])
    cent = np.stack([np.median(points[f][use[f]], axis=0) for f in pool])
    relative = ap - cent[:, None, :]
    features = relative * np.sqrt(aq)[..., None]
    features = features.reshape(len(pool), -1)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-09)
    chosen = [len(pool) // 2]
    distance = np.full(len(pool), np.inf)
    for _ in range(min(CONFIG['fit_poses'], len(pool)) - 1):
        distance = np.minimum(distance, np.sum((features - features[chosen[-1]]) ** 2, axis=1))
        distance[chosen] = -1
        chosen.append(int(np.argmax(distance)))
    chosen = np.sort(chosen)
    frames = pool[chosen]
    ar, ap, aq = (ar[chosen], ap[chosen], aq[chosen])
    selected_details = [details[i] for i in chosen]
    assert (aq > 0).sum() >= 6, 'Too few observable joint anchors'
    init = np.stack([initial_candidates(points[f], w[f], z[f], initial_bind, parents)['joint_svd'] for f in frames])

    def tensor(x):
        return torch.as_tensor(np.array(x, copy=True), dtype=torch.float64)
    P = {k: tensor(v) for k, v in prior.items() if k not in ('weights', 'parents')}
    fct = tensor(factor)
    tgt = tensor(target)
    W = tensor(w[frames])
    Z = tensor(z[frames])
    X = tensor(points[frames])
    U = tensor(use[frames])
    A = tensor(ap)
    Q = tensor(aq)
    AR = tensor(ar)
    loglo, loghi = prior['log_scale_bounds']
    scaled = np.r_[(initial.x[0] - (loglo + loghi) / 2) / ((loghi - loglo) / 2), initial.x[1:] / 3]
    shape = torch.nn.Parameter(tensor(np.arctanh(np.clip(scaled, -0.999, 0.999))))
    root = torch.nn.Parameter(tensor(init[:, :3]))
    quat = torch.nn.Parameter(tensor(init[:, 3:].reshape(-1, 22, 4)))

    def skeleton():
        logscale = (loglo + loghi) / 2 + (loghi - loglo) / 2 * torch.tanh(shape[0])
        coef = 3 * torch.tanh(shape[1:])
        joints = torch.cat([torch.zeros((1, 3)), (P['mean'] + P['basis'] @ coef).reshape(21, 3) * torch.exp(logscale)], dim=0)
        bind = torch.cat([torch.zeros((1, 3)), joints[1:] - joints[parents[1:]]], dim=0)
        return (joints, bind, logscale, coef)

    def rotations(q):
        q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
        x, y, zv, ww = q.unbind(-1)
        return torch.stack([1 - 2 * (y * y + zv * zv), 2 * (x * y - zv * ww), 2 * (x * zv + y * ww), 2 * (x * y + zv * ww), 1 - 2 * (x * x + zv * zv), 2 * (y * zv - x * ww), 2 * (x * zv - y * ww), 2 * (y * zv + x * ww), 1 - 2 * (x * x + y * y)], dim=-1).reshape(*q.shape[:-1], 3, 3)

    def fk(local, bind):
        rr = [local[:, 0]]
        tt = [root]
        for j, p in enumerate(parents[1:], 1):
            rr.append(rr[p] @ local[:, j])
            tt.append(tt[p] + (rr[p] @ bind[j, :, None]).squeeze(-1))
        return (torch.stack(rr, 1), torch.stack(tt, 1))

    def robust(vector):
        return 2 * (torch.sqrt(1 + (vector * vector).sum(-1)) - 1)
    pairs = np.array(SYMMETRY) - 1

    def objective(return_details=False):
        joints, bind, logscale, coef = skeleton()
        rr, tt = fk(rotations(quat), bind)
        length = torch.linalg.vector_norm(bind[1:], dim=-1).clamp_min(1e-09)
        direction = bind[1:] / length[:, None]
        pred = (W[..., None] * (torch.einsum('fjab,fmjb->fmja', rr, Z) + tt[:, None])).sum(2)
        terms = {'offset': 0.5 * ((fct @ joints[1:] - tgt) ** 2).sum(), 'anchors': (Q * robust((tt - A) / 0.04)).sum() / Q.sum().clamp_min(1), 'markers': (U * robust((pred - X) / 0.03)).sum() / U.sum().clamp_min(1), 'shape_prior': 0.2 * (coef.square().mean() + ((logscale - P['log_scale_mean']) / P['log_scale_std']) ** 2), 'symmetry': 0.2 * (((torch.log(length[pairs[:, 0]] / length[pairs[:, 1]]) - P['ratio_mean']) / P['ratio_std']) ** 2).mean(), 'length_bounds': 4 * ((torch.minimum(torch.log(length / P['length_low']), torch.zeros(21)) / 0.05) ** 2 + (torch.maximum(torch.log(length / P['length_high']), torch.zeros(21)) / 0.05) ** 2).mean(), 'direction_bounds': 4 * (torch.minimum((direction * P['mean_direction']).sum(1) - torch.cos(P['direction_limit']), torch.zeros(21)) / 0.02).square().mean(), 'rotation_prior': 0.05 * (Q * ((rr - AR) ** 2).sum((-1, -2)) / (2 * 0.35 ** 2)).sum() / Q.sum().clamp_min(1)}
        loss = sum(terms.values())
        return (loss, terms, rr, tt, bind, logscale, coef) if return_details else loss
    check_loss = objective()
    check_loss.backward()
    gradient_errors = []
    for param, indices in [(shape, [0, len(shape) - 1]), (root, [(0, 0), (len(root) - 1, 2)]), (quat, [(0, 0, 0), (len(quat) - 1, 20, 3)])]:
        for index in indices:
            analytic = float(param.grad[index])
            original_value = float(param[index].detach())
            eps = 1e-06
            with torch.no_grad():
                param[index] = original_value + eps
                plus = float(objective())
                param[index] = original_value - eps
                minus = float(objective())
                param[index] = original_value
            numeric = (plus - minus) / (2 * eps)
            gradient_errors.append(abs(numeric - analytic) / max(1, abs(numeric), abs(analytic)))
    assert max(gradient_errors) < 0.0002, gradient_errors
    optimizer = torch.optim.LBFGS([shape, root, quat], lr=1.0, max_iter=CONFIG['joint_fit_max_iter'], max_eval=CONFIG['joint_fit_max_eval'], history_size=30, line_search_fn='strong_wolfe', tolerance_grad=1e-07, tolerance_change=1e-10)
    closures = 0
    trace = []

    def closure():
        nonlocal closures
        optimizer.zero_grad()
        loss = objective()
        assert torch.isfinite(loss)
        loss.backward()
        closures += 1
        if closures == 1 or closures % 10 == 0:
            trace.append(dict(evaluation=closures, objective=float(loss.detach())))
        return loss
    startloss = float(objective().detach())
    optimizer.step(closure)
    with torch.no_grad():
        loss, terms, rr, tt, bind, logscale, coef = objective(True)
    result = bind.numpy()
    qu = quat.detach().numpy()
    qu /= np.linalg.norm(qu, axis=-1, keepdims=True)
    fkcheck = forward_kinematics(Rotation.from_quat(qu.reshape(-1, 4)).as_matrix().reshape(-1, 22, 3, 3), root.detach().numpy(), result, parents)
    np.testing.assert_allclose(fkcheck[:, :, :3, 3], tt.numpy(), atol=1e-10)
    np.testing.assert_allclose(fkcheck[:, :, :3, :3], rr.numpy(), atol=1e-10)
    assert np.isfinite(result).all() and (np.linalg.norm(result[1:], axis=1) > 0.0001).all()
    assert float(loss) <= startloss + 1e-06
    niter = optimizer.state[shape]['n_iter']
    grad = max((float(v.grad.abs().max()) for v in (shape, root, quat)))
    detail = dict(method='train size/proportion/direction manifold + robust multi-pose observed joint anchors + LBS', selected_local_frames=frames.tolist(), candidate_local_frames=pool.tolist(), initial_shape_success=bool(initial.success), initial_shape_nfev=int(initial.nfev), initial_shape_message=initial.message, objective_initial=startloss, objective_final=float(loss), objective_terms={k: float(v) for k, v in terms.items()}, shared_scale_rms_m=float(torch.exp(logscale)), shape_coefficients_std=coef.tolist(), iterations=int(niter), function_evaluations=closures, iteration_cap=int(niter) >= CONFIG['joint_fit_max_iter'], evaluation_cap=closures >= CONFIG['joint_fit_max_eval'], max_gradient=grad, convergence_claim=False, seconds=time.perf_counter() - begin, gradient_check_max_relative_error=max(gradient_errors), anchor_observable_per_joint=(aq > 0).sum(0).tolist(), anchor_mean_quality_per_joint=aq.mean(0).tolist(), anchors=selected_details, trace=trace, same_skeleton_all_fitted_poses=True, gt_used=False, parameters=CONFIG)
    diagnostic = dict(frame_ids=frames, anchor_positions=ap, anchor_rotations=ar, anchor_quality=aq, fitted_positions=tt.numpy(), fitted_rotations=rr.numpy(), root_positions=root.detach().numpy(), local_quaternions=qu, initial_bind=initial_bind)
    return (result, detail, diagnostic)
