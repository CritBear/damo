from collections import OrderedDict
from pathlib import Path
import io
import json
import pickle
import zipfile
import numpy as np
from scipy.spatial import cKDTree
from .base import sha256
from .data import load_common
from .paired import NativeMotion

class _NumpyDataUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            return OrderedDict
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'scalar'):
            return getattr(np._core.multiarray, name)
        raise ValueError(f'Unsupported object in AMASS metadata: {module}.{name}')

def object_field(path, key):
    with zipfile.ZipFile(path) as archive:
        with archive.open(key + '.npy') as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f'Unsupported metadata NPY version: {version}')
            if not dtype.hasobject:
                raise ValueError(f'{key} is not an object field')
            value = _NumpyDataUnpickler(io.BytesIO(stream.read())).load()
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f'{key}: object metadata shape mismatch')
    return value.item() if value.shape == () else value

def _labels(value, name):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in 'US':
        raise ValueError(f'{name} must be a string vector')
    labels = [v.decode() if isinstance(v, bytes) else str(v) for v in value]
    if any((not v or v != v.strip() for v in labels)) or len(set(labels)) != len(labels):
        raise ValueError(f'{name} contains empty, padded or duplicate labels')
    return labels

def read_marker_archive(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as a:
        if 'markers' not in a or 'labels' not in a:
            raise ValueError('Observed markers/labels absent; vertex indices alone are not observed trajectories')
        model = str(a['surface_model_type'].item())
        if model != 'smplx':
            raise ValueError(f'Expected SMPL-X metadata, got {model}')
        labels = _labels(a['labels'], 'labels')
        latent = _labels(a['latent_labels'], 'latent_labels')
        points = np.asarray(a['markers'], dtype=np.float64)
        poses = np.asarray(a['poses'], dtype=np.float64)
        trans = np.asarray(a['trans'], dtype=np.float64)
        fps = float(a['mocap_frame_rate'])
        betas = np.asarray(a['betas'], dtype=np.float64)
        gender = str(a['gender'].item())
        latent_points = np.asarray(a['markers_latent'], dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 165 or len(poses) < 7 or (points.shape != (len(poses), len(labels), 3)) or (trans.shape != (len(poses), 3)) or (betas.ndim != 1) or (latent_points.shape != (len(latent), 3)) or (gender not in ('male', 'female', 'neutral')) or (not np.isfinite(fps)) or (fps <= 0):
            raise ValueError('Invalid AMASS SMPL-X motion/marker dimensions')
        if not all((np.isfinite(v).all() for v in (poses, trans, betas, latent_points))):
            raise ValueError('Nonfinite AMASS body parameters')
        if 'marker_valid' in a:
            valid = np.asarray(a['marker_valid'], dtype=bool)
            if valid.shape != points.shape[:2]:
                raise ValueError('Invalid marker_valid dimensions')
            validity = 'explicit marker_valid'
        else:
            valid = np.isfinite(points).all(-1) & np.any(points != 0, axis=-1)
            validity = 'AMASS exact-zero missing convention; finite nonzero observations'
        if np.any(valid & ~np.isfinite(points).all(-1)) or not valid.any():
            raise ValueError('Invalid observed marker coordinates')
    vids = object_field(path, 'markers_latent_vids')
    if not isinstance(vids, dict) or set(vids) != set(latent):
        raise ValueError('latent_labels and optimized vertex dictionary disagree')
    if any((not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) or (not 0 <= v < 10475) for v in vids.values())):
        raise ValueError('Invalid SMPL-X marker vertex index')
    missing = set(labels) - set(vids)
    if missing:
        raise ValueError(f'Observed labels have no optimized attachment: {sorted(missing)}')
    points = points.copy()
    points[~valid] = 0
    return dict(points=points, valid=valid, labels=labels, latent_labels=latent, latent_points=latent_points, x_vids=np.array([vids[v] for v in labels], dtype=np.int64), poses=poses, trans=trans, betas=betas, gender=gender, fps=fps, validity_source=validity, path=path, source_sha256=sha256(path), dmpls=None)
