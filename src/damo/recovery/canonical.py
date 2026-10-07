import numpy as np

def fixed_bind_labels(bind_markers, bind_joints, native_weights, vertex_ids):
    markers = np.asarray(bind_markers, dtype=np.float64)
    joints = np.asarray(bind_joints, dtype=np.float64)
    weights = np.asarray(native_weights, dtype=np.float32)
    ids = np.asarray(vertex_ids)
    if markers.ndim != 2 or markers.shape[1] != 3 or joints.shape != (22, 3) or (weights.ndim != 2) or (weights.shape[1] != 22) or (ids.shape != (len(markers),)) or (ids.dtype.kind not in 'iu') or (ids < 0).any() or (ids >= len(weights)).any():
        raise ValueError('Invalid canonical marker/joint/weight dimensions')
    if not all((np.isfinite(x).all() for x in (markers, joints, weights))) or (weights < 0).any() or (not np.allclose(weights.sum(-1), 1, atol=1e-06)):
        raise ValueError('Invalid canonical values or weight normalization')
    full = weights[ids].copy()
    ranks = np.argsort(-full, axis=-1, kind='stable')[:, :3]
    top3 = np.take_along_axis(full, ranks, axis=-1)
    top3 /= top3.sum(-1, keepdims=True)
    offsets = (markers[:, None, :] - joints[ranks]).astype(np.float32)
    return (full, ranks.astype(np.int32), top3, offsets)

def align_latent_markers(observed_labels, latent_labels, latent_points):
    points = np.asarray(latent_points, dtype=np.float64)
    if len(set(latent_labels)) != len(latent_labels) or len(set(observed_labels)) != len(observed_labels) or points.shape != (len(latent_labels), 3) or (not np.isfinite(points).all()):
        raise ValueError('Invalid latent marker identities/positions')
    lookup = {label: i for i, label in enumerate(latent_labels)}
    if not set(observed_labels).issubset(lookup):
        raise ValueError('Observed label absent from canonical marker set')
    return points[[lookup[label] for label in observed_labels]].copy()
