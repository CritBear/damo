from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .data import MarkerDataset
from .geometry import forward_kinematics
from .base import sha256
FORMAT = 'damo-smplx-observed-store-v1'

class ObservedMarkerDataset(MarkerDataset):

    def __init__(self, common, paths, model_config, *, store, source_root=None, **kwargs):
        sampling = kwargs.pop('sampling', 'clip_uniform')
        super().__init__(common, paths, model_config, sampling='clip_uniform', **kwargs)
        if self.common['n_joints'] != 22 or not np.array_equal(self.ratios, [1.0, 0.0, 0.0]):
            raise ValueError('Observed store requires native body22 and ratios [1,0,0]')
        self.store = Path(store)
        self._arrays = None
        self.manifest = json.loads((self.store / 'manifest.json').read_text())
        if self.manifest['format'] != FORMAT or not self.manifest['complete'] or self.manifest['base_id'] != self.common.get('base_id') or (self.manifest['n_max_markers'] != model_config.n_max_markers):
            raise ValueError('Observed store/model/base mismatch; rebuild the acceleration store')
        if source_root is None:
            if isinstance(common, dict):
                raise ValueError('source_root required with an in-memory common base')
            source_root = Path(common).resolve().parent.parent
        source_root = Path(source_root).resolve()
        if sha256(source_root / 'cache_manifest.json') != self.manifest['source_manifest_sha256']:
            raise ValueError('Source cache manifest changed; rebuild the acceleration store')
        for name, entry in self.manifest['arrays'].items():
            if sha256(self.store / (name + '.npy')) != entry['sha256']:
                raise ValueError(f'Acceleration array hash changed: {name}')
        self.records = [self.manifest['clips'][p.resolve().relative_to(source_root).as_posix()] for p in self.paths]
        if sampling not in ('clip_uniform', 'dataset_uniform', 'pose_uniform'):
            raise ValueError('Invalid sampling')
        self.sampling = sampling
        if sampling == 'pose_uniform':
            grouped = [{} for _ in self.dataset_names]
            codes = self.arrays['pose_bins']
            for index, info in enumerate(self.records):
                values = codes[info['start']:info['start'] + info['frames']]
                group = grouped[self.dataset_indices[index]]
                for code in np.unique(values):
                    frames = np.flatnonzero(values == code).astype(np.int32)
                    members = np.column_stack((np.full(len(frames), index, dtype=np.int32), frames))
                    group.setdefault(int(code), []).append(members)
            for groups in grouped:
                members = [np.concatenate(groups[code]) for code in sorted(groups)]
                probability = np.minimum(np.array([len(group) for group in members]) / 32.0, 1.0)
                self.pose_groups.append(members)
                self.pose_group_probabilities.append(probability / probability.sum())

    @property
    def arrays(self):
        if self._arrays is None:
            self._arrays = {name: np.load(self.store / (name + '.npy'), mmap_mode='r', allow_pickle=False) for name in self.manifest['arrays']}
            for name, array in self._arrays.items():
                entry = self.manifest['arrays'][name]
                if list(array.shape) != entry['shape'] or str(array.dtype) != entry['dtype']:
                    raise ValueError(f'Acceleration array layout changed: {name}')
        return self._arrays

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_arrays'] = None
        return state

    def clip(self, index):
        info = self.records[index]
        return dict(n_frames=info['frames'], markers=True, base_id=self.common['base_id'], _store=info)

    def _real(self, result, clip, frame):
        info = clip['_store']
        a = self.arrays
        half = self.config.seq_len // 2
        first = max(0, frame - half)
        last = min(info['frames'], frame + half + 1)
        slot = first - frame + half
        start = info['start'] + first
        end = info['start'] + last
        center = info['start'] + frame
        result['points_seq'][slot:slot + last - first] = a['points'][start:end]
        result['points_mask'][slot:slot + last - first] = a['mask'][start:end]
        ids = a['marker_index'][center]
        valid = ids >= 0
        ids = ids[valid]
        n = len(ids)
        for field, target in [('full_weights', 'm_j_weights'), ('top3_weights', 'm_j3_weights'), ('offsets', 'm_j3_offsets')]:
            result[target][:n] = a[field][ids]
        if self.pose_targets:
            rotations = Rotation.from_rotvec(a['poses'][center].reshape(-1, 3)).as_matrix()
            result['joint_transforms'] = forward_kinematics(rotations, a['roots'][center], np.asarray(info['bind_jlp'], dtype=np.float32), self.common['topology']).astype(np.float32)

    def _synthetic(self, *args, **kwargs):
        raise RuntimeError('Observed store cannot synthesize markers')
