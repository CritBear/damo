import os
import json
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')

@threadpool_limits.wrap(limits=1)
def solve_block(job):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from .recovery.anchor_solver import AnchorSolver
    from .recovery.geometry import forward_kinematics, decode_configuration, lbs
    folder, start, end, recipe = job
    with np.load(Path(folder) / 'predictions.npz', allow_pickle=False) as v:
        a = {k: v[k] for k in v.files}
    with np.load(Path(folder) / 'skeleton.npz', allow_pickle=False) as v:
        bind, confidence = (v['bind'], v['confidence'])
    solver = AnchorSolver(recipe)
    maximum_fk_error = 0.0
    from .recovery import weighted_pose as weighted
    w, z, use = decode_configuration(a['full'][start:end], a['rep'][start:end], a['offsets'][start:end], a['mask'][start:end])
    fields = {k: [] for k in ('params', 'transforms', 'marker_rmse_mm', 'failure', 'cap', 'success', 'nfev', 'total_nfev')}
    for i in range(start, end):
        try:
            policy = solver.recipe['pose_policy']
            f = i - start
            r = weighted.solve_fast(a['points'][i], w[f], z[f], bind, a['parents'], marker_confidence=confidence[i], backend=policy['backend'], max_nfev=policy['max_nfev_per_attempt'], retry_rmse=policy['retry_rmse_m'])
            assert np.isfinite(r['params']).all() and np.isfinite(r['transforms']).all()
            local = Rotation.from_quat(r['params'][3:].reshape(22, 4)).as_matrix()
            fk = forward_kinematics(local, r['params'][:3], bind, a['parents'])
            np.testing.assert_allclose(fk, r['transforms'], atol=1e-07, rtol=1e-06)
            maximum_fk_error = max(maximum_fk_error, float(np.max(np.abs(fk - r['transforms']))))
            error = lbs(r['transforms'], w[f], z[f]) - a['points'][i]
            row = {k: r[k] for k in ('params', 'transforms', 'success', 'nfev', 'total_nfev')}
            row.update(marker_rmse_mm=float(np.sqrt(np.mean(error[use[f]] ** 2)) * 1000), failure='', cap='maximum' in r['message'].lower() and 'evaluation' in r['message'].lower())
        except (ValueError, AssertionError, np.linalg.LinAlgError, FloatingPointError) as e:
            row = dict(params=np.full(91, np.nan), transforms=np.full((22, 4, 4), np.nan), marker_rmse_mm=np.nan, failure=str(e), cap=False, success=False, nfev=0, total_nfev=0)
        for k in fields:
            fields[k].append(row[k])
    path = Path(folder) / 'blocks' / f'{start:06d}_{end:06d}.npz'
    np.savez_compressed(path, frame_ids=np.arange(start, end), **{k: np.asarray(v) for k, v in fields.items()})
    return (str(path), maximum_fk_error)

def infer(checkpoint, recipe, input_path, output, device='cpu', unit='m', workers=8, stable_marker_slots=False):
    import numpy as np
    import torch
    from .recovery.base import sha256
    from .recovery.checkpoint import load_model
    from .recovery.inference import read_markers, predict
    from .recovery.anchor_solver import AnchorSolver
    checkpoint, recipe = (Path(checkpoint).resolve(), Path(recipe).resolve())
    p = dict(checkpoint_sha256=sha256(checkpoint), solver_recipe_sha256=sha256(recipe))
    input_path, output = (Path(input_path).resolve(), Path(output).resolve())
    if output.exists():
        raise FileExistsError('Use a new output directory; existing results are never overwritten')
    if not 1 <= workers <= 64:
        raise ValueError('workers must be 1..64')
    points, mask = read_markers(input_path, unit=unit)
    ids = None
    if input_path.suffix.lower() == '.npz':
        with np.load(input_path, allow_pickle=False) as v:
            for key in ('marker_index', 'marker_ids'):
                if key in v:
                    ids = np.asarray(v[key])
                    break
    if ids is None:
        if not stable_marker_slots:
            raise ValueError('Provide stable marker_index/marker_ids, or --stable-marker-slots for fixed tracks')
        ids = np.broadcast_to(np.arange(points.shape[1]), mask.shape).copy()
    if ids.ndim == 1:
        ids = np.broadcast_to(ids, mask.shape).copy()
    if ids.shape != mask.shape or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('Marker IDs must be integer F x M or M')
    if any(((ids[i, mask[i]] < 0).any() or len(set(ids[i, mask[i]])) != int(mask[i].sum()) for i in range(len(points)))):
        raise ValueError('Observed marker IDs must be unique and nonnegative per frame')
    solver = AnchorSolver(recipe)
    parents = solver.prior['parents']
    torch.set_num_threads(2)
    model, mc, payload = load_model(checkpoint, device=device)
    if mc.n_joints != 22:
        raise ValueError('Anchor solving requires a body22 checkpoint')
    p.update(epoch=payload.get('epoch'), model_config=payload.get('model_config'))
    output.mkdir(parents=True)
    (output / 'blocks').mkdir()
    write_json(output / 'provenance.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), profile=p, input=str(input_path), input_sha256=sha256(input_path), unit=unit, frames=len(points), ground_truth_in_fit=False))
    write_json(output / 'status.json', dict(state='predicting', pid=os.getpid()))
    full, rep, offsets = predict(model, points, mask, batch_size=32, device=device)
    del model
    a = dict(points=points, mask=mask, full=full, rep=rep, offsets=offsets, marker_index=ids, parents=parents)
    np.savez_compressed(output / 'predictions.npz', **a)
    write_json(output / 'status.json', dict(state='fitting_shared_skeleton', pid=os.getpid()))
    bind, confidence, detail, diagnostics = solver.fit(**a)
    np.savez_compressed(output / 'skeleton.npz', bind=bind, confidence=confidence)
    np.savez_compressed(output / 'anchor_diagnostics.npz', **diagnostics)
    write_json(output / 'skeleton.json', detail)
    jobs = [(str(output), i, min(i + 32, len(points)), str(recipe)) for i in range(0, len(points), 32)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        blocks = []
        for i, result in enumerate(pool.map(solve_block, jobs)):
            blocks.append(result)
            write_json(output / 'status.json', dict(state='solving', pid=os.getpid(), done=min((i + 1) * 32, len(points)), total=len(points)))
    results = {}
    for path, _ in blocks:
        with np.load(path, allow_pickle=False) as v:
            for key in v.files:
                results.setdefault(key, []).append(v[key])
    results = {k: np.concatenate(v) for k, v in results.items()}
    np.testing.assert_array_equal(results['frame_ids'], np.arange(len(points)))
    np.savez_compressed(output / 'result.npz', **results, bind=bind, parents=parents)
    failures = int(np.count_nonzero(results['failure'] != ''))
    finite = np.isfinite(results['params']).all(axis=1)
    summary = dict(state='complete_with_failures' if failures or not results['success'].all() else 'complete', frames=len(points), failures=failures, unsuccessful_frames=int((~results['success']).sum()), finite_frames=int(finite.sum()), selected_attempt_cap=int(results['cap'].sum()), marker_rmse_mm=float(results['marker_rmse_mm'][finite].mean()) if finite.any() else None, fk_max_error=max((v for _, v in blocks)), checkpoint_sha256=p['checkpoint_sha256'], recipe_sha256=p['solver_recipe_sha256'], result_sha256=sha256(output / 'result.npz'), blocks={Path(path).name: sha256(path) for path, _ in blocks}, completed_utc=datetime.now(timezone.utc).isoformat(), convergence_not_proven=True)
    write_json(output / 'verification.json', summary)
    write_json(output / 'status.json', summary)
    return summary
