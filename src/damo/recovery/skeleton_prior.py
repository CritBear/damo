import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve_triangular

def joints_from_bind(bind, parents):
    j = np.zeros_like(bind, dtype=float)
    for k, p in enumerate(parents[1:], 1):
        j[k] = j[p] + bind[k]
    return j

class SkeletonProblem:

    def __init__(self, full, rep, offsets, mask, marker_ids, parents, prior_dir):
        prior_dir = Path(prior_dir)
        self.parents = np.asarray(parents)
        self.n = len(parents)
        self.shape = mask.shape
        ids = np.argsort(-full, axis=-1, kind='stable')[..., :3]
        self.valid = np.asarray(mask, bool) & (ids[..., 0] < self.n)
        ja = []
        jb = []
        delta = []
        weight = []
        flat = []
        track = []
        fm = np.arange(mask.size).reshape(mask.shape)
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            use = self.valid & (ids[..., a] < self.n) & (ids[..., b] < self.n) & (rep[..., a] > 0) & (rep[..., b] > 0)
            ja.extend(ids[..., a][use])
            jb.extend(ids[..., b][use])
            delta.extend(offsets[..., a, :][use] - offsets[..., b, :][use])
            weight.extend(np.minimum(rep[..., a][use], rep[..., b][use]))
            flat.extend(fm[use])
            track.extend(marker_ids[use])
        self.ja = np.array(ja, int)
        self.jb = np.array(jb, int)
        self.delta = np.array(delta, float)
        self.weight = np.array(weight, float)
        self.flat = np.array(flat, int)
        self.track = np.array(track, int)
        assert len(self.weight) > 0 and (self.track >= 0).all()
        self.design = np.zeros((self.n - 1, self.n))
        self.design[np.arange(self.n - 1), np.arange(1, self.n)] = 1
        self.design[np.arange(self.n - 1), self.parents[1:]] -= 1
        self.design = self.design[:, 1:]
        with np.load(prior_dir / 'train_log_length_statistics.npz') as p:
            self.mu = p['mean'].copy()
            cov = p['covariance'].copy()
        cov = 0.5 * cov + 0.5 * np.diag(np.diag(cov)) + 0.05 ** 2 * np.eye(21)
        self.whiten = solve_triangular(np.linalg.cholesky(cov), np.eye(21), lower=True) / np.sqrt(21)
        stats = json.loads((prior_dir / 'summary.json').read_text())['train_bones']
        self.low = np.log(np.array([r['p01'] for r in stats]) * 0.9)
        self.high = np.log(np.array([r['p99'] for r in stats]) * 1.1)
        pairs = json.loads((prior_dir / 'train_joint_pair_prior.json').read_text())['pairs']
        med = np.ones((22, 22))
        sig = np.ones((22, 22))
        for r in pairs:
            a, b = (r['joint_a'], r['joint_b'])
            med[a, b] = med[b, a] = r['median']
            sig[a, b] = sig[b, a] = max(0.1, (np.log(r['p99']) - np.log(r['p01'])) / 4.6527)
        self.compatibility_z = np.abs(np.log(np.maximum(np.linalg.norm(self.delta, axis=1), 1e-09) / med[self.ja, self.jb])) / sig[self.ja, self.jb]

    def equations(self, confidence=None, excluded=None):
        w = self.weight.copy()
        if confidence is not None:
            w *= confidence.ravel()[self.flat]
        if excluded is not None:
            w[self.track == excluded] = 0
        h = np.zeros((22, 22))
        b = np.zeros((22, 3))
        np.add.at(h, (self.ja, self.ja), w)
        np.add.at(h, (self.jb, self.jb), w)
        np.add.at(h, (self.ja, self.jb), -w)
        np.add.at(h, (self.jb, self.ja), -w)
        np.add.at(b, self.ja, -w[:, None] * self.delta)
        np.add.at(b, self.jb, w[:, None] * self.delta)
        return (h[1:, 1:], b[1:], float(w.sum()))

    def objective(self, h, b, mass):
        vals, vecs = np.linalg.eigh(h)
        assert vals.min() > max(vals.max() * 1e-12, 1e-12), 'Disconnected or singular marker graph'
        scale = np.sqrt(mass) * 0.03
        factor = np.sqrt(vals)[:, None] * vecs.T / scale
        target = vecs.T @ b / np.sqrt(vals)[:, None] / scale
        data_jac = np.kron(factor, np.eye(3))
        self.last_condition = float(vals.max() / vals.min())

        def evaluate(x, jac=False):
            points = x.reshape(21, 3)
            bones = self.design @ points
            length = np.maximum(np.linalg.norm(bones, axis=1), 1e-10)
            log = np.log(length)
            low = np.minimum(log - self.low, 0) / (0.05 * np.sqrt(21))
            high = np.maximum(log - self.high, 0) / (0.05 * np.sqrt(21))
            if not jac:
                return np.r_[(factor @ points - target).ravel(), self.whiten @ (log - self.mu), low, high]
            derivative = (self.design[:, :, None] * bones[:, None, :] / length[:, None, None] ** 2).reshape(21, 63)
            return np.vstack([data_jac, self.whiten @ derivative, (log < self.low)[:, None] * derivative / (0.05 * np.sqrt(21)), (log > self.high)[:, None] * derivative / (0.05 * np.sqrt(21))])
        return (lambda x: evaluate(x), lambda x: evaluate(x, True))

    def fit(self, initial, confidence=None, excluded=None):
        h, b, mass = self.equations(confidence, excluded)
        fun, jac = self.objective(h, b, mass)
        x = joints_from_bind(initial, self.parents)[1:].ravel()
        result = least_squares(fun, x, jac=jac, method='trf', max_nfev=100, ftol=1e-08, xtol=1e-08, gtol=1e-08)
        joints = np.vstack([np.zeros((1, 3)), result.x.reshape(21, 3)])
        bind = np.zeros((22, 3))
        bind[1:] = joints[1:] - joints[self.parents[1:]]
        assert np.isfinite(bind).all() and (np.linalg.norm(bind[1:], axis=1) > 1e-07).all()
        return (bind, dict(success=bool(result.success), nfev=int(result.nfev), cost=float(result.cost), condition=self.last_condition, message=result.message))

    def confidence(self, prior_bind):
        disagreement = np.zeros(len(self.weight))
        loo = []
        for marker in np.unique(self.track):
            chosen = self.track == marker
            try:
                bind, info = self.fit(prior_bind, excluded=marker)
                j = joints_from_bind(bind, self.parents)
                disagreement[chosen] = np.linalg.norm(self.delta[chosen] - (j[self.jb[chosen]] - j[self.ja[chosen]]), axis=1) / 0.03
                loo.append(dict(marker_id=int(marker), supported=True, **info))
            except (AssertionError, ValueError, np.linalg.LinAlgError) as e:
                loo.append(dict(marker_id=int(marker), supported=False, reason=str(e)))
        mass = np.bincount(self.flat, weights=self.weight, minlength=np.prod(self.shape))
        score = np.maximum(self.compatibility_z - 2, 0) ** 2 / 4 + disagreement ** 2 / 9
        s = np.bincount(self.flat, weights=self.weight * score, minlength=len(mass)) / np.maximum(mass, 1e-20)
        c = np.clip(1 / (1 + s), 0.05, 1)
        ids = np.full(len(c), -1, int)
        ids[self.flat] = self.track
        for marker in np.unique(self.track):
            selected = ids == marker
            c[selected] = np.sqrt(c[selected] * np.median(c[selected]))
        c = c.reshape(self.shape)
        c[~self.valid] = 0
        return (c, dict(method='train pair-distance compatibility + leave-one-marker-track-out agreement + track median', minimum=0.05, mean=float(c[self.valid].mean()), p10=float(np.quantile(c[self.valid], 0.1)), below_half=int((c[self.valid] < 0.5).sum()), valid=int(self.valid.sum()), loo=loo, calibrated_probability=False, attachment_weights_unchanged=True))
