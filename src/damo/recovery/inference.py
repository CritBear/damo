from pathlib import Path
import numpy as np
import torch
from .checkpoint import load_model
from .data import load_common, body22_common
from .geometry import decode_configuration
from .solving import solve_sequence, estimate_configuration_skeleton, DEFAULT_BACKEND

def read_markers(path, *, unit='m'):
    path = Path(path)
    if path.suffix.lower() == '.c3d':
        import c3d
        with path.open('rb') as f:
            reader = c3d.Reader(f)
            frames, masks = ([], [])
            for _, points, _ in reader.read_frames():
                frames.append(points[:, :3].copy())
                masks.append((points[:, 3] >= 0) & np.isfinite(points[:, :3]).all(-1))
            result = np.asarray(frames, dtype=np.float32)
            mask = np.asarray(masks, dtype=bool)
    elif path.suffix.lower() == '.npy':
        result = np.load(path, allow_pickle=False)
        mask = np.isfinite(result).all(-1) & (result != 0).any(-1)
    else:
        with np.load(path, allow_pickle=False) as data:
            result = data['points'] if 'points' in data else data['markers']
            mask = data['mask'].astype(bool) if 'mask' in data else np.isfinite(result).all(-1) & (result != 0).any(-1)
    if result.ndim != 3 or result.shape[-1] != 3 or len(result) == 0 or (mask.shape != result.shape[:-1]):
        raise ValueError('Input must be (frames, markers, 3), with optional (frames, markers) mask')
    if unit not in ('m', 'cm', 'mm'):
        raise ValueError('Unknown point unit')
    mask &= np.isfinite(result).all(-1)
    result = np.where(mask[..., None], result, 0).astype(np.float32)
    result *= {'m': 1.0, 'cm': 0.01, 'mm': 0.001}[unit]
    return (result, mask)

@torch.inference_mode()
def predict(model, points, mask, *, batch_size=64, device='cpu'):
    model.eval()
    cfg = model.options
    f, original_m, _ = points.shape
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if original_m > cfg.n_max_markers:
        raise ValueError(f'Input has {original_m} marker slots; maximum is {cfg.n_max_markers}. Explicitly select/compact markers first.')
    padded = np.zeros((f, cfg.n_max_markers, 3), dtype=np.float32)
    padded_mask = np.zeros((f, cfg.n_max_markers), dtype=bool)
    padded[:, :original_m] = points
    padded_mask[:, :original_m] = mask
    half = cfg.seq_len // 2
    padded = np.pad(padded, ((half, half), (0, 0), (0, 0)))
    padded_mask = np.pad(padded_mask, ((half, half), (0, 0)))
    collected = [[], [], []]
    for start in range(0, f, batch_size):
        centers = range(start, min(f, start + batch_size))
        x = torch.from_numpy(np.stack([padded[i:i + cfg.seq_len] for i in centers])).to(device)
        masks = torch.from_numpy(np.stack([padded_mask[i:i + cfg.seq_len] for i in centers])).to(device)
        for storage, value in zip(collected, model(x, masks)):
            storage.append(value[:, :original_m].cpu().numpy())
    return tuple((np.concatenate(v) for v in collected))

def infer(checkpoint, input_path, output_path, *, device='cpu', unit='m', batch_size=64, common_path=None, skeleton_path=None, max_nfev=100, smoothing=True, joint_distribution=None, solver_backend=DEFAULT_BACKEND):
    model, cfg, payload = load_model(checkpoint, device, joint_distribution=joint_distribution)
    points, mask = read_markers(input_path, unit=unit)
    indices, rep_weights, rep_offsets = predict(model, points, mask, batch_size=batch_size, device=device)
    result = {'points': points, 'mask': mask, 'indices': indices, 'rep_weights': rep_weights, 'rep_offsets': rep_offsets, 'unit': np.array('m'), 'joint_distribution': np.array(cfg.joint_distribution), 'checkpoint_joint_distribution': np.array(payload.get('model_config', {}).get('joint_distribution', 'linear'))}
    if common_path is not None or skeleton_path is not None:
        weights, offsets, valid = decode_configuration(indices, rep_weights, rep_offsets, mask)
        result.update(weights=weights, offsets=offsets, usable_mask=valid)
        if skeleton_path is not None:
            with np.load(skeleton_path, allow_pickle=False) as data:
                topology, bind_local = (data['topology'], data['bind_local'])
        else:
            common = load_common(common_path)
            if cfg.n_joints == 22 and common['n_joints'] == 24:
                common = body22_common(common)
            if common['n_joints'] != cfg.n_joints:
                raise ValueError('Skeleton and model joint counts differ')
            topology = common['topology']
            bind_local, info = estimate_configuration_skeleton(indices, rep_weights, rep_offsets, mask, topology)
            result.update(skeleton_graph_condition=np.array(info['condition']), skeleton_constraint_rms_mm=np.array(info['constraint_rms_mm']))
        if len(topology) != cfg.n_joints:
            raise ValueError('Skeleton and model joint counts differ')
        solved = solve_sequence(points, weights, offsets, bind_local, topology, max_nfev=max_nfev, smoothing=smoothing, backend=solver_backend)
        diagnostics = solved.pop('diagnostics')
        result.update(solved, topology=topology, bind_local=bind_local, solver_success=np.array([d['success'] for d in diagnostics]), solver_marker_fit_pass=np.array([d['marker_fit_pass'] for d in diagnostics]), solver_backend=np.array(solver_backend), pose_initialization=np.array('joint_svd'), skeleton_method=np.array('known' if skeleton_path is not None else 'offset_difference'), smoothing_applied=np.array(smoothing))
        import json
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).with_suffix('.solver.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)
    return {'frames': len(points), 'markers': points.shape[1], 'output': str(output_path), 'pose_solved': 'transforms' in result, 'joint_distribution': cfg.joint_distribution}
