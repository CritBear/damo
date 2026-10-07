from dataclasses import dataclass, asdict
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from .geometry import forward_kinematics, validate_topology

def _pose(roots, quaternions, bind, parents):
    parents = validate_topology(parents)
    roots = np.array(roots, dtype=np.float64, copy=True)
    q = np.array(quaternions, dtype=np.float64, copy=True)
    bind = np.asarray(bind, dtype=np.float64)
    if roots.ndim != 2 or roots.shape[1] != 3 or (not len(roots)):
        raise ValueError('Expected a nonempty F x 3 root trajectory')
    if q.shape != (len(roots), len(parents), 4) or bind.shape != (len(parents), 3):
        raise ValueError('Inconsistent quaternion, skeleton or topology dimensions')
    if not all((np.isfinite(a).all() for a in (roots, q, bind))):
        raise ValueError('Split at failed frames; do not smooth across nonfinite poses')
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if (norm < 1e-12).any():
        raise ValueError('Zero quaternion is not a rotation')
    q /= norm
    for t in range(1, len(q)):
        q[t, np.sum(q[t] * q[t - 1], axis=-1) < 0] *= -1
    return (roots, q, bind, parents)

def _transforms(roots, q, bind, parents):
    local = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))
    return forward_kinematics(local, roots, bind, parents)

def savgol_pose(roots, quaternions, bind, parents, *, window=31, order=3, fps=None):
    if not isinstance(window, (int, np.integer)) or window < 1 or window % 2 != 1:
        raise ValueError('window must be a positive odd integer')
    if not isinstance(order, (int, np.integer)) or order < 0 or (window > 1 and order >= window):
        raise ValueError('order must be nonnegative and smaller than window')
    if fps is not None and (not np.isfinite(fps) or fps <= 0):
        raise ValueError('fps must be positive')
    roots, q, bind, parents = _pose(roots, quaternions, bind, parents)
    effective = min(window, len(roots) if len(roots) % 2 else len(roots) - 1)
    fallback = 0
    if effective > order and effective >= 3:
        roots = savgol_filter(roots, effective, order, axis=0, mode='interp')
        filtered = savgol_filter(q, effective, order, axis=0, mode='interp')
        norm = np.linalg.norm(filtered, axis=-1, keepdims=True)
        fallback = int((norm < 1e-08).sum())
        q = np.where(norm >= 1e-08, filtered / np.maximum(norm, 1e-08), q)
    else:
        effective = 1
    return dict(root_positions=roots, local_quaternions=q, transforms=_transforms(roots, q, bind, parents), metadata=dict(method='savgol_local_quaternion', window_requested=window, window_effective=effective, order=order, fps=fps, support_span_seconds=(effective - 1) / fps if fps else None, boundary='interp', offline=True, quaternion_fallbacks=fallback, ground_truth_in_fit=False))

@dataclass(frozen=True)
class FootLockConfig:
    speed_m_s: float = 0.15
    height_band_m: float = 0.03
    min_contact_seconds: float = 0.1
    blend_seconds: float = 0.05
    max_correction_m: float = 0.05
    max_rotation_degrees: float = 25.0
    min_confidence: float = 0.2

def _segments(mask):
    cuts = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(cuts == 1), np.flatnonzero(cuts == -1))

def _between(a, b):
    a, b = (a / np.linalg.norm(a), b / np.linalg.norm(b))
    cross = np.cross(a, b)
    sine = np.linalg.norm(cross)
    cosine = np.clip(a @ b, -1, 1)
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        basis = np.eye(3)[np.argmin(abs(a))]
        axis = np.cross(a, basis)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    return Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine)).as_matrix()

def foot_lock(roots, quaternions, bind, parents, *, fps, confidence=None, config=FootLockConfig()):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive')
    if any((not np.isfinite(v) or v <= 0 for v in asdict(config).values())):
        raise ValueError('Foot-lock thresholds must be positive and finite')
    roots, q, bind, parents = _pose(roots, quaternions, bind, parents)
    chains = [(1, 4, 7, 10), (2, 5, 8, 11)]
    if len(parents) < 12 or any((tuple(parents[[b, c, d]]) != (a, b, c) for a, b, c, d in chains)):
        raise ValueError('Foot lock requires the SMPL body topology')
    n = len(roots)
    confidence = np.ones((n, 2)) if confidence is None else np.asarray(confidence)
    if confidence.shape != (n, 2) or not np.isfinite(confidence).all():
        raise ValueError('Expected finite F x 2 foot confidence')
    raw = _transforms(roots, q, bind, parents)
    local = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(n, len(parents), 3, 3)
    contacts = np.zeros((n, 2), bool)
    applied = np.zeros((n, 2), bool)
    targets = raw[:, :, :3, 3][:, [10, 11]].copy()
    segments = []
    skipped = 0
    for side, (hip, knee, ankle, toe) in enumerate(chains):
        feet = raw[:, toe, :3, 3]
        speed = np.linalg.norm(np.gradient(feet, axis=0), axis=-1) * fps if n > 1 else np.full(n, np.inf)
        low = float(np.quantile(feet[:, 2], 0.05))
        mask = (speed < config.speed_m_s) & (feet[:, 2] < low + config.height_band_m) & (confidence[:, side] >= config.min_confidence)
        for start, stop in _segments(mask):
            if stop - start < max(3, int(np.ceil(config.min_contact_seconds * fps))):
                continue
            contacts[start:stop, side] = True
            anchor = np.median(feet[start:stop], axis=0)
            segments.append(dict(side=side, start=int(start), stop=int(stop), anchor=anchor.tolist(), height_reference_m=low))
            for t in range(start, stop):
                ramp = min(1.0, (t - start) / max(1.0, config.blend_seconds * fps), (stop - 1 - t) / max(1.0, config.blend_seconds * fps))
                blend = ramp * ramp * (3 - 2 * ramp)
                desired_toe = feet[t] + blend * (anchor - feet[t])
                targets[t, side] = desired_toe
                delta = desired_toe - feet[t]
                if np.linalg.norm(delta) < 1e-10:
                    continue
                h, k, a = raw[t, [hip, knee, ankle], :3, 3]
                dest = a + delta
                direction = dest - h
                distance = np.linalg.norm(direction)
                l1, l2 = (np.linalg.norm(k - h), np.linalg.norm(a - k))
                if np.linalg.norm(delta) > config.max_correction_m or not abs(l1 - l2) + 1e-06 < distance < l1 + l2 - 1e-06:
                    skipped += 1
                    continue
                axis = direction / distance
                bend = k - h - axis * np.dot(k - h, axis)
                if np.linalg.norm(bend) < 1e-08:
                    skipped += 1
                    continue
                bend /= np.linalg.norm(bend)
                x = (l1 * l1 - l2 * l2 + distance * distance) / (2 * distance)
                new_k = h + x * axis + np.sqrt(max(0.0, l1 * l1 - x * x)) * bend
                hip_global = _between(k - h, new_k - h) @ raw[t, hip, :3, :3]
                knee_global = _between(a - k, dest - new_k) @ raw[t, knee, :3, :3]
                proposed = np.array([raw[t, parents[hip], :3, :3].T @ hip_global, hip_global.T @ knee_global, knee_global.T @ raw[t, ankle, :3, :3]])
                change = Rotation.from_matrix(proposed @ local[t, [hip, knee, ankle]].swapaxes(-1, -2)).magnitude()
                if np.rad2deg(change).max() > config.max_rotation_degrees:
                    skipped += 1
                    continue
                local[t, [hip, knee, ankle]] = proposed
                applied[t, side] = True
    qout = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_quat().reshape(q.shape)
    final = forward_kinematics(local, roots, bind, parents)
    return dict(root_positions=roots, local_quaternions=qout, transforms=final, contacts=contacts, applied=applied, targets=targets, metadata=dict(method='experimental_contact_two_bone_ik', config=asdict(config), segments=segments, contact_foot_frames=int(contacts.sum()), applied_foot_frames=int(applied.sum()), skipped_corrections=skipped, calibrated_floor=False, mesh_penetration_constrained=False, ground_truth_in_fit=False))
