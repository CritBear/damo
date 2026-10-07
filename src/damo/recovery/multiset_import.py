from pathlib import Path
from collections import Counter
import tempfile
import numpy as np
from .amass_markers import read_marker_archive, object_field, _labels
from .base import sha256

def select_marker_columns(labels, valid, vids):
    value = np.asarray(labels)
    if value.ndim != 1 or value.dtype.kind not in 'US':
        raise ValueError('labels must be a string vector')
    labels = [v.decode() if isinstance(v, bytes) else str(v) for v in value]
    counts = Counter(labels)
    keep = np.ones(len(labels), bool)
    removed = []
    for i, label in enumerate(labels):
        reason = None
        if not valid[:, i].any():
            reason = 'no_valid_observations'
        elif not label or label != label.strip():
            reason = 'invalid_label'
        elif counts[label] > 1:
            reason = 'ambiguous_duplicate_label'
        elif label not in vids:
            reason = 'no_official_attachment'
        if reason:
            keep[i] = False
            removed.append(dict(column=i, label=label, reason=reason, valid_observations=int(valid[:, i].sum())))
    return (keep, removed)

def read_native_markers(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as a:
        points = a['markers']
        labels = a['labels']
        if points.ndim != 3 or points.shape[1:] != (len(labels), 3):
            raise ValueError('Marker dimensions')
        valid = np.asarray(a['marker_valid'], bool) if 'marker_valid' in a else np.isfinite(points).all(-1) & np.any(points != 0, axis=-1)
        if valid.shape != points.shape[:-1] or (valid & ~np.isfinite(points).all(-1)).any():
            raise ValueError('Marker validity')
        vids = object_field(path, 'markers_latent_vids')
        if not isinstance(vids, dict):
            raise ValueError('Invalid official attachment dictionary')
        keep, removed = select_marker_columns(labels, valid, vids)
        if keep.all():
            return {**read_marker_archive(path), 'empty_marker_labels_removed': [], 'removed_marker_columns': []}
        if not keep.any():
            raise ValueError('No observed marker columns')
        fields = {key: a[key] for key in ['surface_model_type', 'latent_labels', 'markers_latent', 'poses', 'trans', 'mocap_frame_rate', 'betas', 'gender']}
        fields.update(markers=points[:, keep], labels=a['labels'][keep], marker_valid=valid[:, keep], markers_latent_vids=np.array(vids, dtype=object))
    with tempfile.TemporaryDirectory(prefix='damo-native-import-') as temp:
        view = Path(temp) / 'view.npz'
        np.savez(view, **fields)
        result = read_marker_archive(view)
    result.update(path=path, source_sha256=sha256(path), empty_marker_labels_removed=[r['label'] for r in removed if r['reason'] == 'no_valid_observations'], removed_marker_columns=removed, validity_source='AMASS finite/nonzero or explicit validity preserved; owner-approved unique official attachment subset')
    return result
