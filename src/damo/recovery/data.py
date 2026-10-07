from collections import OrderedDict
from pathlib import Path
import pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from .geometry import forward_kinematics, lbs, validate_topology

def body22_common(common):
    if common['n_joints'] == 22:
        return common
    if common['n_joints'] != 24 or list(common['topology'][-2:]) != [20, 21]:
        raise ValueError('body22 conversion requires SMPL hand leaves 22->20, 23->21')
    out = dict(common)
    weights = np.array(common['weights'][:, :22], copy=True)
    weights[:, 20] += common['weights'][:, 22]
    weights[:, 21] += common['weights'][:, 23]
    ranks = np.argsort(-weights, axis=-1, kind='stable')[:, :3]
    rep_weights = np.take_along_axis(weights, ranks, axis=-1)
    rep_weights /= rep_weights.sum(-1, keepdims=True)
    out.update(n_joints=22, topology=np.array(common['topology'][:22]), weights=weights, v_j3_indices=ranks, v_j3_weights=rep_weights, j_v_idx=[np.flatnonzero(weights.argmax(-1) == j).tolist() for j in range(22)])
    for key in ('caesar_bind_jgp', 'caesar_bind_jlp'):
        out[key] = common[key][:, :22]
    if 'J_regressor' in common:
        out['J_regressor'] = common['J_regressor'][:22]
    return out

def body22_clip(clip, common):
    n = int(clip['n_frames'])
    if np.asarray(clip['poses']).shape[-1] == 66:
        return clip
    out = dict(clip)
    out['poses'] = np.asarray(clip['poses']).reshape(n, -1, 3)[:, :22].reshape(n, 66)
    for key in ('jgp',):
        out[key] = clip[key][:, :22]
    for key in ('bind_jgp', 'bind_jlp'):
        if key in clip:
            out[key] = clip[key][:22]
    if 'markers' not in clip:
        return out
    clean = np.asarray(clip['ghost_marker_mask'], dtype=bool)
    vertex = np.where(clean, clip['m_v_idx'], 0).astype(int)
    ranks = common['v_j3_indices'][vertex]
    w = common['v_j3_weights'][vertex]
    local = Rotation.from_rotvec(out['poses'].reshape(-1, 3)).as_matrix().reshape(n, 22, 3, 3)
    jt = forward_kinematics(local, out['jgp'][:, 0], out['bind_jlp'], common['topology'])
    selected = jt[np.arange(n)[:, None, None], ranks]
    bind = out['bind_jgp'][ranks]
    a = (w[..., None, None] * selected[..., :3, :3]).sum(axis=2)
    shift = selected[..., :3, 3] - np.einsum('fmkab,fmkb->fmka', selected[..., :3, :3], bind)
    rhs = clip['markers'] - (shift * w[..., None]).sum(axis=2)
    bind_markers = np.einsum('fmab,fmb->fma', np.linalg.pinv(a), rhs)
    out['m_j3_indices'] = ranks
    out['m_j3_weights'] = w * clean[..., None]
    out['m_j3_offsets'] = (bind_markers[..., None, :] - bind) * clean[..., None, None]
    return out

def load_common(path):
    if Path(path).suffix.lower() == '.npz':
        from .base import load_base
        common = load_base(path)
    else:
        with Path(path).open('rb') as f:
            common = pickle.load(f)
    required = ('topology', 'weights', 'caesar_bind_v', 'caesar_bind_vn', 'caesar_bind_jgp', 'caesar_bind_jlp', 'soma_superset_variant', 'j_v_idx', 'v_j3_indices', 'v_j3_weights')
    missing = set(required) - set(common)
    if missing:
        raise ValueError(f'Common cache lacks {sorted(missing)}')
    validate_topology(common['topology'])
    common['n_joints'] = len(common['topology'])
    if common['weights'].shape[1] != common['n_joints']:
        raise ValueError('Common weights and topology disagree')
    return common

def discover_clips(data_root, names, date='20240329'):
    paths = []
    for name in names:
        found = sorted((Path(data_root) / 'batch' / date / name).glob('*.pkl'))
        if not found:
            raise FileNotFoundError(f"No cached clips for {name}: {Path(data_root) / 'batch' / date / name}")
        paths.extend(found)
    if not paths:
        raise ValueError('At least one dataset is required')
    return paths

def load_clip(path):
    with Path(path).open('rb') as f:
        raw = pickle.load(f)
    if isinstance(raw.get('n_frames'), (list, tuple)):
        if len(raw['n_frames']) != 1:
            raise ValueError(f'{path}: expected one motion per cache; split merged files first')
        raw = {k: v[0] for k, v in raw.items() if isinstance(v, (list, tuple)) and len(v) == 1}
    if int(raw['n_frames']) < 1:
        raise ValueError(f'Empty motion: {path}')
    return raw

def validate_base_binding(common, clip):
    if common.get('base_id') and clip.get('base_id') != common['base_id']:
        raise ValueError('Motion cache was not prepared for this independent base; rebuild it from source motion data')

def pose_bins(clip):
    joints = np.asarray(clip['jgp'])
    if joints.shape[1] < 22:
        raise ValueError('Pose-stratified sampling requires the SMPL body topology')
    local = np.asarray(clip['bind_jlp'])
    leg_length = np.mean([np.linalg.norm(local[4]) + np.linalg.norm(local[7]), np.linalg.norm(local[5]) + np.linalg.norm(local[8])])
    if not np.isfinite(leg_length) or leg_length <= 1e-06:
        raise ValueError('Invalid leg length for pose sampling')
    height = (joints[:, [7, 8, 20, 21], 2] - joints[:, :1, 2]) / leg_length
    bins = np.digitize(height, [-0.75, -0.25, 0.25, 0.75])
    torso = joints[:, 12] - joints[:, 0]
    up = torso[:, 2] / np.maximum(np.linalg.norm(torso, axis=-1), 1e-08)
    return (bins @ np.array([1, 5, 25, 125]) + 625 * np.digitize(up, [-0.5, 0.5])).astype(np.int32)

def mirror_smpl_motion(rotations, root_positions):
    n_joints = rotations.shape[-3]
    if n_joints not in (22, 24):
        raise ValueError('Mirroring requires the SMPL 22/24 joint convention')
    pairs = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
    reflection = np.diag([-1.0, 1.0, 1.0])
    return (reflection @ rotations[..., pairs[:n_joints], :, :] @ reflection, root_positions @ reflection)

class MarkerDataset(Dataset):

    def __init__(self, common, paths, model_config, *, seed=2024, samples_per_clip=100, ratios=(0.5, 0.25, 0.25), noise=True, style='legacy', cache_size=2, joint_policy='native', sampling='clip_uniform', mirror_probability=0.0, pose_targets=False):
        self.common = load_common(common) if not isinstance(common, dict) else common
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError('Dataset has no clips')
        self.config = model_config
        self.pose_targets = bool(pose_targets)
        self.joint_policy = joint_policy
        if joint_policy == 'body22':
            self.common = body22_common(self.common)
        elif joint_policy != 'native':
            raise ValueError('joint_policy must be native or body22')
        if model_config.n_joints != self.common['n_joints']:
            raise ValueError(f"Model has {model_config.n_joints} joints but cache has {self.common['n_joints']}. Select matching data or explicitly prepare a converted cache.")
        self.ratios = np.asarray(ratios, dtype=float)
        if self.ratios.shape != (3,) or not np.isfinite(self.ratios).all() or (self.ratios < 0).any() or (self.ratios.sum() <= 0):
            raise ValueError('ratios must be three nonnegative real/superset/arbitrary probabilities')
        self.ratios /= self.ratios.sum()
        if self.ratios[1] > 0 and (not self.common['soma_superset_variant']):
            raise ValueError('This base has no verified superset layout; supply one or set the superset ratio to zero')
        if samples_per_clip < 1 or style not in ('legacy', 'paper'):
            raise ValueError('Invalid samples_per_clip or synthesis style')
        self.samples_per_clip = samples_per_clip
        self.seed = seed
        self.epoch = 0
        self.noise = noise
        self.style = style
        if not 0 <= mirror_probability <= 1:
            raise ValueError('mirror_probability must be in [0, 1]')
        self.mirror_probability = float(mirror_probability)
        self.cache_size = cache_size
        self._cache = OrderedDict()
        if sampling not in ('clip_uniform', 'dataset_uniform', 'pose_uniform'):
            raise ValueError('sampling must be clip_uniform, dataset_uniform or pose_uniform')
        self.sampling = sampling
        self.dataset_names = sorted({p.parent.name for p in self.paths})
        self.dataset_indices = [self.dataset_names.index(p.parent.name) for p in self.paths]
        self.dataset_groups = [np.flatnonzero(np.asarray(self.dataset_indices) == i) for i in range(len(self.dataset_names))]
        self.pose_groups, self.pose_group_probabilities = ([], [])
        if sampling == 'pose_uniform':
            grouped = [{} for _ in self.dataset_names]
            for clip_index, path in enumerate(self.paths):
                clip = load_clip(path)
                validate_base_binding(self.common, clip)
                codes = pose_bins(clip)
                groups = grouped[self.dataset_indices[clip_index]]
                for code in np.unique(codes):
                    frames = np.flatnonzero(codes == code).astype(np.int32)
                    members = np.column_stack((np.full(len(frames), clip_index, dtype=np.int32), frames))
                    groups.setdefault(int(code), []).append(members)
            for groups in grouped:
                members = [np.concatenate(groups[code]) for code in sorted(groups)]
                probability = np.minimum(np.array([len(group) for group in members]) / 32.0, 1.0)
                self.pose_groups.append(members)
                self.pose_group_probabilities.append(probability / probability.sum())

    def __len__(self):
        return len(self.paths) * self.samples_per_clip

    def set_epoch(self, epoch):
        self.epoch = epoch

    def clip(self, index):
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        result = load_clip(self.paths[index])
        validate_base_binding(self.common, result)
        if self.joint_policy == 'body22':
            result = body22_clip(result, self.common)
        if self.cache_size > 0:
            self._cache[index] = result
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        clip_index = index % len(self.paths)
        chosen_frame = None
        if self.sampling == 'dataset_uniform':
            group = self.dataset_groups[index % len(self.dataset_groups)]
            clip_index = int(rng.choice(group))
        elif self.sampling == 'pose_uniform':
            group = index % len(self.dataset_names)
            poses = self.pose_groups[group]
            members = poses[int(rng.choice(len(poses), p=self.pose_group_probabilities[group]))]
            clip_index, chosen_frame = map(int, members[int(rng.integers(len(members)))])
        clip = self.clip(clip_index)
        frame = int(rng.integers(clip['n_frames'])) if chosen_frame is None else chosen_frame
        probabilities = self.ratios.copy()
        if 'markers' not in clip:
            probabilities[0] = 0
            if probabilities.sum() == 0:
                raise ValueError('Real-only sampling requested from synthetic-only AMASS data')
            probabilities /= probabilities.sum()
        mode = int(rng.choice(3, p=probabilities))
        return {**self.sample(clip, frame, mode, rng), 'dataset_idx': self.dataset_indices[clip_index]}

    def sample(self, clip, frame, mode, rng):
        validate_base_binding(self.common, clip)
        s, m, j = (self.config.seq_len, self.config.n_max_markers, self.config.n_joints)
        result = {'points_seq': np.zeros((s, m, 3), dtype=np.float32), 'points_mask': np.zeros((s, m), dtype=np.float32), 'm_j_weights': np.zeros((m, j + 1), dtype=np.float32), 'm_j3_weights': np.zeros((m, 3), dtype=np.float32), 'm_j3_offsets': np.zeros((m, 3, 3), dtype=np.float32)}
        if mode == 0:
            self._real(result, clip, frame)
        else:
            self._synthetic(result, clip, frame, mode, rng)
        return {**{k: torch.from_numpy(v) for k, v in result.items()}, 'frame_idx': frame, 'real': mode == 0}

    def _real(self, result, clip, frame):
        if 'markers' not in clip:
            raise ValueError('Real-marker sampling requested from a synthetic-only clip; set ratios[0]=0')
        half = self.config.seq_len // 2
        if self.pose_targets:
            rotations = Rotation.from_rotvec(np.asarray(clip['poses'][frame]).reshape(-1, 3)).as_matrix()
            result['joint_transforms'] = forward_kinematics(rotations, clip['jgp'][frame, 0], clip['bind_jlp'], self.common['topology']).astype(np.float32)
        for slot, f in enumerate(range(frame - half, frame + half + 1)):
            if not 0 <= f < clip['n_frames']:
                continue
            points = np.asarray(clip['markers'][f])
            finite = np.isfinite(points).all(-1)
            if 'marker_valid' in clip:
                raw_valid = np.asarray(clip['marker_valid'][f], dtype=bool)
                if raw_valid.shape != finite.shape:
                    raise ValueError('Marker validity and point arrays disagree')
                if (raw_valid & ~finite).any():
                    raise ValueError('Observed marker has nonfinite coordinates')
                valid = raw_valid
            else:
                valid = finite & (np.linalg.norm(points, axis=-1) > 0.001)
            selected = np.flatnonzero(valid)[:self.config.n_max_markers]
            n = len(selected)
            result['points_seq'][slot, :n] = points[selected]
            result['points_mask'][slot, :n] = 1
            if f != frame:
                continue
            clean = np.asarray(clip['ghost_marker_mask'][f, selected], dtype=bool)
            vertex = np.asarray(clip['m_v_idx'][f, selected], dtype=int)
            if ((vertex[clean] < 0) | (vertex[clean] >= len(self.common['weights']))).any():
                raise ValueError('Real marker has an invalid guiding vertex')
            vertex = np.where(clean, vertex, 0)
            if 'marker_native_weights' in clip:
                native = np.asarray(clip['marker_native_weights'])
                if native.shape != (clip['markers'].shape[1], self.config.n_joints) or not np.isfinite(native).all() or (native < 0).any() or (not np.allclose(native.sum(-1), 1, atol=1e-06)):
                    raise ValueError('Invalid source-native marker weights')
                target_weights = native[selected]
            else:
                target_weights = self.common['weights'][vertex]
            result['m_j_weights'][:n, :-1] = target_weights * clean[:, None]
            result['m_j_weights'][:n, -1] = ~clean
            result['m_j3_weights'][:n] = clip['m_j3_weights'][f, selected] * clean[:, None]
            result['m_j3_weights'][:n, 0] += ~clean
            result['m_j3_offsets'][:n] = clip['m_j3_offsets'][f, selected] * clean[:, None, None]

    def _select_vertices(self, mode, rng):
        if mode == 1:
            variants = self.common['soma_superset_variant']
            n = len(variants) if self.style == 'paper' or len(variants) <= 22 else int(rng.integers(22, len(variants)))
            selected = rng.choice(len(variants), n, replace=False)
            return np.array([rng.choice(variants[i]) for i in selected], dtype=int)
        groups = self.common['j_v_idx']
        counts = rng.choice([2, 3, 4], len(groups), p=[0.7, 0.2, 0.1])
        if any((len(group) < count for group, count in zip(groups, counts))):
            raise ValueError('A joint has too few guiding vertices for arbitrary sampling')
        return np.concatenate([rng.choice(group, count, replace=False) for group, count in zip(groups, counts)])

    def _synthetic(self, result, clip, frame, mode, rng):
        c, cfg = (self.common, self.config)
        marker_vertices = self._select_vertices(mode, rng)
        body = int(rng.integers(len(c['caesar_bind_v'])))
        bind_global = np.asarray(c['caesar_bind_jgp'][body])
        half = cfg.seq_len // 2
        start, end = (max(0, frame - half), min(clip['n_frames'], frame + half + 1))
        pose = np.asarray(clip['poses'][start:end]).reshape(end - start, cfg.n_joints, 3)
        rotations = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(end - start, cfg.n_joints, 3, 3)
        root_positions = clip['jgp'][start:end, 0]
        if self.mirror_probability > 0 and rng.random() < self.mirror_probability:
            rotations, root_positions = mirror_smpl_motion(rotations, root_positions)
        transforms = forward_kinematics(rotations, root_positions, c['caesar_bind_jlp'][body], c['topology'])
        if self.pose_targets:
            result['joint_transforms'] = transforms[frame - start].astype(np.float32)
        skin_distance = 0.01 + (rng.uniform(-0.0025, 0.0025) if self.noise and self.style == 'legacy' else 0)
        for f in range(start, end):
            slot = f - frame + half
            vertices = marker_vertices.copy()
            if self.noise:
                n_remove = int(rng.integers(0, min(5, len(vertices)) + 1))
                vertices = np.delete(vertices, rng.choice(len(vertices), n_remove, replace=False))
            vertices = vertices[:cfg.n_max_markers]
            bind_markers = c['caesar_bind_v'][body, vertices] + c['caesar_bind_vn'][body, vertices] * skin_distance
            if self.noise and self.style == 'legacy':
                bind_markers = bind_markers + rng.uniform(-0.005, 0.005, bind_markers.shape)
            ranks = c['v_j3_indices'][vertices]
            rep_weights = c['v_j3_weights'][vertices]
            rep_offsets = bind_markers[:, None] - bind_global[ranks]
            jt = transforms[f - start]
            points = np.einsum('mkab,mkb->mka', jt[ranks, :3, :3], rep_offsets) + jt[ranks, :3, 3]
            points = (points * rep_weights[..., None]).sum(axis=1)
            n_clean = len(vertices)
            n_ghost = 0
            if self.noise:
                n_ghost = int(rng.integers(0, 4)) if self.style == 'paper' else max(0, int(rng.integers(-2, 3)))
                n_ghost = min(n_ghost, cfg.n_max_markers - n_clean)
                if self.style == 'paper':
                    gv = rng.integers(0, len(c['weights']), size=n_ghost)
                    gz = c['caesar_bind_v'][body, gv, None] - bind_global[None]
                    ghost = lbs(jt, c['weights'][gv], gz)
                else:
                    ghost = rng.uniform(-0.5, 0.5, (n_ghost, 3)) + points.mean(axis=0)
                points = np.concatenate([points, ghost])
                if self.style == 'paper':
                    points += rng.uniform(-0.005, 0.005, points.shape)
            n = n_clean + n_ghost
            order = rng.permutation(n) if self.noise else np.arange(n)
            result['points_seq'][slot, :n] = points[order]
            result['points_mask'][slot, :n] = 1
            if f == frame:
                full_w = np.zeros((n, cfg.n_joints + 1), dtype=np.float32)
                full_w[:n_clean, :-1] = c['weights'][vertices]
                full_w[n_clean:, -1] = 1
                w = np.zeros((n, 3), dtype=np.float32)
                w[:n_clean] = rep_weights
                w[n_clean:, 0] = 1
                z = np.zeros((n, 3, 3), dtype=np.float32)
                z[:n_clean] = rep_offsets
                result['m_j_weights'][:n] = full_w[order]
                result['m_j3_weights'][:n] = w[order]
                result['m_j3_offsets'][:n] = z[order]
        if self.noise:
            angle = rng.uniform(0, 2 * np.pi)
            r = Rotation.from_rotvec([0, 0, angle]).as_matrix()
            result['points_seq'] = (result['points_seq'] @ r).astype(np.float32)
            if self.pose_targets:
                result['joint_transforms'][:, :3, :] = r.T @ result['joint_transforms'][:, :3, :]
