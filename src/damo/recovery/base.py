from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
from scipy.stats import truncnorm
from .geometry import validate_topology
FORMAT = 'damo-native-base-v1'
BODY_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def _dense(value):
    if hasattr(value, 'toarray'):
        value = value.toarray()
    if hasattr(value, 'r'):
        value = value.r
    return np.asarray(value)

def load_native_model(path):
    path = Path(path)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            values = {k: archive[k] for k in ('v_template', 'shapedirs', 'J_regressor', 'weights', 'kintree_table', 'f')}
    elif path.suffix.lower() == '.pkl':
        with path.open('rb') as stream:
            raw = pickle.load(stream, encoding='latin1')
        values = {k: _dense(raw[k]) for k in ('v_template', 'shapedirs', 'J_regressor', 'weights', 'kintree_table', 'f')}
    else:
        raise ValueError('Native model must be .npz or an explicitly trusted .pkl')
    v = np.asarray(values['v_template'], dtype=np.float64)
    directions = np.asarray(values['shapedirs'], dtype=np.float64)
    regressor = np.asarray(values['J_regressor'], dtype=np.float64)
    weights = np.asarray(values['weights'], dtype=np.float64)
    table = np.asarray(values['kintree_table'])
    faces = np.asarray(values['f'])
    if v.ndim != 2 or v.shape[1] != 3 or directions.shape[:2] != v.shape or (directions.ndim != 3):
        raise ValueError('Invalid template or shape directions')
    if regressor.ndim != 2 or regressor.shape[0] not in (24, 52, 55) or regressor.shape[1] != len(v):
        raise ValueError('Expected native SMPL (24), SMPL-H (52), or SMPL-X (55) joints')
    if weights.shape != (len(v), len(regressor)) or table.shape != (2, len(regressor)) or table.dtype.kind not in 'iu':
        raise ValueError('Native weights, regressor and hierarchy disagree')
    if not all((np.isfinite(x).all() for x in (v, directions, regressor, weights))):
        raise ValueError('Nonfinite native model parameters')
    if (weights < 0).any() or not np.allclose(weights.sum(-1), 1, atol=1e-06, rtol=0):
        raise ValueError('Skinning weights must be nonnegative and sum to one')
    if not np.allclose(regressor.sum(-1), 1, atol=0.0001, rtol=0):
        raise ValueError('Joint regressor must preserve translation')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0 or (faces.dtype.kind not in 'iu') or (faces.min() < 0) or (faces.max() >= len(v)):
        raise ValueError('Invalid mesh faces')
    ids = [int(x) for x in table[1]]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate joint IDs')
    lookup = {joint: i for i, joint in enumerate(ids)}
    try:
        parents = np.array([-1] + [lookup[int(p)] for p in table[0, 1:]])
    except KeyError as error:
        raise ValueError('Unknown parent joint ID') from error
    validate_topology(parents)
    if not np.array_equal(parents[:22], BODY_PARENTS):
        raise ValueError('Native model does not use the expected SMPL body joint order')
    return {'vertices': v, 'shapedirs': directions, 'regressor': regressor, 'weights': weights, 'parents': parents, 'faces': faces.astype(np.int64)}

def collapse_body_weights(weights, parents):
    out = np.zeros((len(weights), 22), dtype=np.float64)
    for joint in range(len(parents)):
        ancestor = joint
        while ancestor >= 22:
            ancestor = int(parents[ancestor])
        out[:, ancestor] += weights[:, joint]
    return out

def vertex_normals(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    face_normals = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    normals = np.zeros_like(v)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    length = np.linalg.norm(normals, axis=-1)
    if not np.isfinite(length).all() or (length <= 1e-12).any():
        raise ValueError('Mesh contains vertices with undefined normals')
    return normals / length[:, None]

def _pack_groups(groups):
    sizes = np.array([len(g) for g in groups], dtype=np.int64)
    values = np.concatenate(groups).astype(np.int64) if len(groups) else np.empty(0, np.int64)
    return (values, np.r_[0, np.cumsum(sizes)])

def _unpack_groups(values, offsets):
    if offsets.ndim != 1 or len(offsets) == 0 or offsets[0] != 0 or (offsets[-1] != len(values)) or (np.diff(offsets) < 0).any():
        raise ValueError('Invalid packed candidate groups')
    return [values[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]

def read_marker_layout(path, n_vertices):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('model_family') not in ('smpl', 'smplh') or data.get('vertex_count') != n_vertices:
        raise ValueError('Marker layout must name a matching SMPL/SMPL-H mesh')
    groups = data.get('groups')
    if not isinstance(groups, dict) or not groups:
        raise ValueError('Marker layout needs named nonempty candidate groups')
    result = []
    for name, group in groups.items():
        ids = np.asarray(group)
        if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu' or (ids.min() < 0) or (ids.max() >= n_vertices) or (len(np.unique(ids)) != len(ids)):
            raise ValueError(f'Invalid candidate group: {name}')
        result.append(ids.astype(np.int64))
    return (list(groups), result)

def build_base(model_path, output_path, *, source_description, bodies=256, num_betas=10, beta_std=1.0, beta_limit=2.5, seed=2024, marker_layout=None):
    output = Path(output_path)
    if output.suffix.lower() != '.npz' or output.exists():
        raise ValueError('Choose a new .npz output; existing assets are never replaced')
    if not source_description.strip():
        raise ValueError('Record where this native model came from')
    if bodies < 1 or num_betas < 1 or (not np.isfinite([beta_std, beta_limit]).all()) or (beta_std <= 0) or (beta_limit <= 0):
        raise ValueError('Invalid shape sampling parameters')
    native = load_native_model(model_path)
    if num_betas > native['shapedirs'].shape[-1]:
        raise ValueError('Requested more shape coefficients than the native model provides')
    rng = np.random.default_rng(seed)
    betas = truncnorm.rvs(-beta_limit / beta_std, beta_limit / beta_std, scale=beta_std, size=(bodies, num_betas), random_state=rng)
    betas[0] = 0
    vertices = native['vertices'][None] + np.einsum('bk,vck->bvc', betas, native['shapedirs'][:, :, :num_betas])
    joints = np.einsum('jv,bvc->bjc', native['regressor'][:22], vertices)
    local = np.zeros_like(joints)
    local[:, 1:] = joints[:, 1:] - joints[:, BODY_PARENTS[1:]]
    normals = np.stack([vertex_normals(v, native['faces']) for v in vertices])
    weights = collapse_body_weights(native['weights'], native['parents'])
    ranks = np.argsort(-weights, axis=-1, kind='stable')[:, :3]
    representative = np.take_along_axis(weights, ranks, axis=-1)
    representative /= representative.sum(-1, keepdims=True)
    joint_groups = [np.flatnonzero(weights.argmax(-1) == j) for j in range(22)]
    if any((len(g) < 4 for g in joint_groups)):
        raise ValueError('Every body joint needs at least four arbitrary-marker candidates')
    names, groups = read_marker_layout(marker_layout, len(weights)) if marker_layout else ([], [])
    joint_values, joint_offsets = _pack_groups(joint_groups)
    layout_values, layout_offsets = _pack_groups(groups)
    metadata = {'format': FORMAT, 'source_kind': 'smpl_shape_samples', 'source_model': {'path': str(Path(model_path).resolve()), 'sha256': sha256(model_path), 'description': source_description, 'native_joints': len(native['parents'])}, 'sampling': {'bodies': bodies, 'num_betas': num_betas, 'beta_std': beta_std, 'beta_limit': beta_limit, 'seed': seed, 'first_body': 'zero_betas'}, 'bind_coordinates': 'native_smpl', 'length_unit': 'm', 'marker_layout': {'path': str(Path(marker_layout).resolve()), 'sha256': sha256(marker_layout)} if marker_layout else None, 'marker_labels': names, 'legacy_common_or_additional_used': False, 'note': 'Model provenance must be confirmed by the owner; numerical checks do not authenticate a file.'}
    if len(native['parents']) == 55:
        metadata.update(model_family='smplx', source_kind='smplx_shape_samples', bind_coordinates='native_smplx')
    arrays = {'vertices': vertices.astype(np.float32), 'normals': normals.astype(np.float32), 'joints_global': joints.astype(np.float32), 'joints_local': local.astype(np.float32), 'weights': weights, 'joint_regressor': native['regressor'][:22], 'topology': BODY_PARENTS, 'faces': native['faces'], 'betas': betas, 'top3_indices': ranks, 'top3_weights': representative, 'joint_candidate_values': joint_values, 'joint_candidate_offsets': joint_offsets, 'layout_candidate_values': layout_values, 'layout_candidate_offsets': layout_offsets}
    array_hashes = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items()}
    metadata['array_sha256'] = array_hashes
    metadata['base_id'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    arrays['metadata'] = np.array(json.dumps(metadata, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    load_base(temporary)
    temporary.replace(output)
    return {'output': str(output.resolve()), 'sha256': sha256(output), 'base_id': metadata['base_id'], 'bodies': bodies, 'joints': 22, 'superset_groups': len(groups), 'body_source': 'SMPL shape samples, not CAESAR'}

def load_base(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata.get('format') != FORMAT:
            raise ValueError('Unknown independent base format')
        arrays = {k: archive[k] for k in metadata['array_sha256']}
    identity = metadata['base_id']
    unsigned = {k: v for k, v in metadata.items() if k != 'base_id'}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() != identity:
        raise ValueError('Base metadata identity mismatch')
    for name, value in arrays.items():
        if hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != metadata['array_sha256'][name]:
            raise ValueError(f'Base array hash mismatch: {name}')
        if value.dtype.kind in 'fc' and (not np.isfinite(value).all()):
            raise ValueError(f'Nonfinite base array: {name}')
    joint_groups = _unpack_groups(arrays['joint_candidate_values'], arrays['joint_candidate_offsets'])
    layout_groups = _unpack_groups(arrays['layout_candidate_values'], arrays['layout_candidate_offsets'])
    return {'n_joints': 22, 'n_vertices': len(arrays['weights']), 'topology': arrays['topology'], 'weights': arrays['weights'], 'J_regressor': arrays['joint_regressor'], 'caesar_bind_v': arrays['vertices'], 'caesar_bind_vn': arrays['normals'], 'caesar_bind_jgp': arrays['joints_global'], 'caesar_bind_jlp': arrays['joints_local'], 'v_j3_indices': arrays['top3_indices'], 'v_j3_weights': arrays['top3_weights'], 'j_v_idx': joint_groups, 'soma_superset_variant': layout_groups, 'base_manifest': metadata, 'base_id': identity, 'faces': arrays['faces'], 'body_betas': arrays['betas']}
