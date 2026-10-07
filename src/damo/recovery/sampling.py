import numpy as np
from .synthetic_fast import FastMarkerDataset


class FrameDataset(FastMarkerDataset):
    def __init__(self, *args, sampling='clip_uniform', draws=None, **kwargs):
        self.frame_sampling = sampling == 'frame_uniform'
        self.draws = draws
        if draws is not None and (not isinstance(draws, int) or draws < 1 or not self.frame_sampling):
            raise ValueError('draws must be a positive integer used with frame_uniform')
        super().__init__(*args, sampling='clip_uniform' if self.frame_sampling else sampling, **kwargs)
        if self.frame_sampling:
            self.sampling = 'frame_uniform'
            lengths = [r['frames'] for r in self.observed.records] if self.observed else [self.clip(i)['n_frames'] for i in range(len(self.paths))]
            if min(lengths) < 1:
                raise ValueError('Empty motion segment')
            self.cumulative = [np.cumsum([lengths[int(i)] for i in group], dtype=np.int64) for group in self.dataset_groups]

    def __len__(self):
        return self.draws if self.draws is not None else super().__len__()

    def __getitem__(self, index):
        if not self.frame_sampling:
            return super().__getitem__(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        group = index % len(self.dataset_groups)
        cumulative = self.cumulative[group]
        position = int(rng.integers(cumulative[-1]))
        member = int(np.searchsorted(cumulative, position, side='right'))
        frame = position - (int(cumulative[member - 1]) if member else 0)
        clip_index = int(self.dataset_groups[group][member])
        clip = self.clip(clip_index)
        mode = int(rng.choice(3, p=self.ratios))
        return {**self.sample(clip, frame, mode, rng), 'dataset_idx': group}
