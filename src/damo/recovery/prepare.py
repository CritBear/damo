import json
import pickle
import shutil
from pathlib import Path
import numpy as np
from .base import sha256, collapse_body_weights
from .canonical import fixed_bind_labels
from .data import load_common
from .mean_offset import mean_inverse_bind_markers
from .multiset_import import read_native_markers
from .paired import NativeMotion
from .splits import digest


def prepare_native(raw_root, models, common_path, output, *, datasets, chunk_frames=600, device='cpu'):
    raw_root, models, output = Path(raw_root).resolve(), Path(models).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new cache directory')
    if chunk_frames < 7 or not datasets or len(set(datasets)) != len(datasets):
        raise ValueError('Invalid chunk length or dataset list')
    common = load_common(common_path)
    if common.get('n_joints') != 22 or not common.get('base_id'):
        raise ValueError('Build a native body22 base first')
    paths = []
    for name in datasets:
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Dataset name must be a directory name')
        found = sorted((raw_root / name).rglob('*_stageii.npz'))
        if not found:
            raise FileNotFoundError(f'No SMPL-X stageii recordings for {name}')
        paths.extend((name, p) for p in found)
    output.mkdir(parents=True)
    (output / 'common').mkdir()
    shutil.copyfile(common_path, output / 'common/native_base.npz')
    manifest = dict(format='damo-body22-cache-v1', complete=False, date='fixed_bind',
                    common_path='common/native_base.npz', common_sha256=sha256(common_path),
                    model_family='smplx', offset_policy='fixed_mean_inverse', files={}, motions={},
                    structural_exclusions=[], duplicate_source_exclusions=[], compact_motion_indices=False,
                    requires_observed_store=False, source_kind='observed_markers_in_amass_smplx')
    seen = {}
    def save():
        (output / 'cache_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save()
    for name, path in paths:
        source_hash = sha256(path)
        rel = path.relative_to(raw_root).as_posix()
        if source_hash in seen:
            manifest['duplicate_source_exclusions'].append(dict(excluded=rel, same_as=seen[source_hash]))
            continue
        seen[source_hash] = rel
        try:
            motion = read_native_markers(path)
        except ValueError as exc:
            manifest['structural_exclusions'].append(dict(source=rel, source_sha256=source_hash, reason=str(exc)))
            save()
            continue
        model_path = models / f'SMPLX_{motion["gender"].upper()}.npz'
        if not model_path.is_file():
            model_path = models / motion['gender'] / 'model.npz'
        body = NativeMotion(model_path, motion, up_axis='z', device=device)
        if not np.array_equal(body.native['parents'][:22], common['topology']):
            raise ValueError('Source and common topology differ')
        if body.native['vertices'].shape[0] != common['weights'].shape[0]:
            raise ValueError('Source and common surface models differ')
        points, valid, ids = motion['points'], motion['valid'], motion['x_vids']
        if int(valid.sum(axis=1).max()) > 90:
            raise ValueError(f'{rel}: more than 90 observed markers; select a documented subset explicitly')
        poses, transforms = body.body_targets()
        native_weights = collapse_body_weights(body.native['weights'], body.native['parents'])
        ranks = np.argsort(-native_weights[ids], axis=-1, kind='stable')[:, :3]
        weights = np.take_along_axis(native_weights[ids], ranks, axis=-1)
        weights /= weights.sum(axis=-1, keepdims=True)
        mean, diagnostic = mean_inverse_bind_markers(points, valid, transforms, body.bind_joints[:22], ranks, weights)
        full, ranks, weights, offsets = fixed_bind_labels(mean, body.bind_joints[:22], native_weights, ids)
        local_rel = path.relative_to(raw_root / name)
        motion_name = '__'.join(local_rel.with_suffix('').parts).removesuffix('_stageii')
        key = name + '__' + motion_name
        if key in manifest['motions']:
            raise ValueError('Ambiguous source motion name')
        subject = local_rel.parent.as_posix()
        records = []
        n = len(points)
        for start in range(0, n, chunk_frames):
            end = min(n, start + chunk_frames)
            count = end - start
            if count == 0:
                continue
            tile = lambda a: np.broadcast_to(a, (count,) + a.shape).copy()
            clip = dict(n_frames=count, base_id=common['base_id'], poses=poses[start:end].astype(np.float32),
                        jgp=transforms[start:end, :, :3, 3].astype(np.float32),
                        bind_jgp=body.bind_joints[:22].astype(np.float32), bind_jlp=body.bind_local.astype(np.float32),
                        markers=points[start:end].astype(np.float32), marker_valid=valid[start:end],
                        m_v_idx=tile(ids), m_j3_indices=tile(ranks), m_j3_weights=tile(weights).astype(np.float32),
                        m_j3_offsets=tile(offsets).astype(np.float32),
                        ghost_marker_mask=np.ones((count, len(ids)), dtype=bool),
                        marker_native_weights=full)
            filename = f'batch/fixed_bind/{name}/{digest(key)[:20]}_{start:06d}_{end:06d}.pkl'
            target = output / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('wb') as stream:
                pickle.dump(clip, stream, protocol=pickle.HIGHEST_PROTOCOL)
            manifest['files'][filename] = dict(sha256=sha256(target), frames=count, bytes=target.stat().st_size,
                                                motion=motion_name, dataset=name, subject=subject, source_sha256=source_hash)
            records.append(filename)
        manifest['motions'][key] = dict(dataset=name, motion=motion_name, key=key, subject=subject, clips=records,
                                        source_relative=rel, source_sha256=source_hash, source_model_sha256=sha256(model_path),
                                        frames=n, markers=len(ids), fps=motion['fps'], gender=motion['gender'],
                                        bind_jgp=body.bind_joints[:22].tolist(), bind_jlp=body.bind_local.tolist(),
                                        label_diagnostics=diagnostic, removed_marker_columns=motion['removed_marker_columns'])
        print(json.dumps(dict(event='prepared_motion', motion=key, frames=n)), flush=True)
        save()
    present = {m['dataset'] for m in manifest['motions'].values()}
    if present != set(datasets):
        raise ValueError('At least one requested dataset has no valid recordings; inspect the incomplete manifest')
    manifest.update(complete=True, total_motions=len(manifest['motions']),
                    total_frames=sum(m['frames'] for m in manifest['motions'].values()))
    save()
    return dict(cache=str(output), motions=manifest['total_motions'], frames=manifest['total_frames'],
                excluded=len(manifest['structural_exclusions']), duplicates=len(manifest['duplicate_source_exclusions']))
