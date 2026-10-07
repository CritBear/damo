import numpy as np
from .data import MarkerDataset
from .observed_store import ObservedMarkerDataset

class FastMarkerDataset(MarkerDataset):

    def __init__(self, common, paths, model_config, *, store=None, source_root=None, **kwargs):
        sampling = kwargs.get('sampling', 'clip_uniform')
        super().__init__(common, paths, model_config, **{**kwargs, 'sampling': 'clip_uniform' if store else sampling})
        self.common = dict(self.common)
        self.common['j_v_idx'] = [np.asarray(g, dtype=np.int64) for g in self.common['j_v_idx']]
        self.common['soma_superset_variant'] = [np.asarray(g, dtype=np.int64) for g in self.common['soma_superset_variant']]
        self.observed = None
        if store is not None:
            self.observed = ObservedMarkerDataset(common, paths, model_config, store=store, source_root=source_root, **{**kwargs, 'ratios': [1, 0, 0]})
            self.sampling = sampling
            self.pose_groups = self.observed.pose_groups
            self.pose_group_probabilities = self.observed.pose_group_probabilities

    def clip(self, index):
        if self.observed is None:
            return super().clip(index)
        clip = self.observed.clip(index)
        info = clip['_store']
        start = info['start']
        end = start + info['frames']
        arrays = self.observed.arrays
        clip.update(poses=arrays['poses'][start:end], jgp=arrays['roots'][start:end, None, :])
        return clip

    def _real(self, result, clip, frame):
        if self.observed is None:
            return super()._real(result, clip, frame)
        return self.observed._real(result, clip, frame)
