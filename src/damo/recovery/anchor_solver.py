import copy
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from . import anchor
from .base import sha256
from .skeleton_prior import SkeletonProblem
from .solving import estimate_configuration_skeleton


class AnchorSolver:
    def __init__(self, recipe_path):
        self.path = Path(recipe_path).resolve()
        self.recipe = json.loads(self.path.read_text(encoding='utf-8'))
        if self.recipe['state'] != 'frozen' or self.recipe['ground_truth_in_fit'] is not False:
            raise ValueError('Expected a frozen recipe without evaluation truth in fitting')
        root = self.path.parent
        for rel, expected in self.recipe['prior_sha256'].items():
            target = (root / rel).resolve()
            if root not in target.parents or sha256(target) != expected:
                raise ValueError(f'Prior file mismatch: {rel}')
        for rel, expected in self.recipe['source_sha256'].items():
            target = (Path(__file__).parent / rel).resolve()
            if target.parent != Path(__file__).parent or sha256(target) != expected:
                raise ValueError(f'Solver source mismatch: {rel}')
        self.prior_dir = root
        with np.load(root / self.recipe['shape_prior'], allow_pickle=False) as archive:
            self.prior = {k: archive[k].copy(order='K') for k in archive.files}
        self.recipe_sha256 = sha256(self.path)

    def fit(self, *, points, full, rep, offsets, mask, marker_index, parents):
        if not np.array_equal(parents, self.prior['parents']):
            raise ValueError('Prior topology differs from checkpoint')
        if len({len(x) for x in (points, full, rep, offsets, mask, marker_index)}) != 1:
            raise ValueError('Inconsistent input lengths')
        original = dict(anchor.CONFIG)
        dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
        try:
            anchor.CONFIG.update(self.recipe['skeleton_parameters'])
            with threadpool_limits(limits=1):
                bind, detail, diagnostic = anchor.fit(points, full, rep, offsets, mask, marker_index,
                                                     parents, self.prior, self.prior_dir)
                detail = copy.deepcopy(detail)
                base, base_info = estimate_configuration_skeleton(full, rep, offsets, mask, parents)
                problem = SkeletonProblem(full, rep, offsets, mask, marker_index, parents, self.prior_dir)
                prior_bind, prior_info = problem.fit(base)
                confidence, confidence_info = problem.confidence(prior_bind)
        finally:
            anchor.CONFIG.clear()
            anchor.CONFIG.update(original)
            torch.set_default_dtype(dtype)
            torch.set_num_threads(threads)
        if not np.isfinite(bind).all() or not np.isfinite(confidence).all():
            raise ValueError('Nonfinite skeleton or confidence')
        return bind, confidence, dict(recipe_sha256=self.recipe_sha256, skeleton=detail,
            confidence=confidence_info, confidence_initialization=dict(base=base_info, prior=prior_info),
            one_skeleton_per_window=True, ground_truth_in_fit=False), diagnostic
