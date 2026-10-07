import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .recovery.base import sha256
from .recovery.checkpoint import load_model


def export_checkpoint(source, output, *, tag, weights_license):
    output = Path(output)
    if output.suffix != '.pt' or output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Choose a new .pt output and metadata path')
    model, config, payload = load_model(source)
    run = payload.get('run_config', {})
    training = run.get('training', {})
    if training.get('loss') != 'legacy' or training.get('geometry_loss') or training.get('orientation_loss'):
        raise ValueError('This is not a verified baseline training checkpoint; do not relabel auxiliary-loss weights')
    if not tag.strip() or not weights_license.strip():
        raise ValueError('Provide a release tag and an explicit weights license or terms URL')
    state = payload.get('training_state', {})
    metadata = dict(format='damo-baseline-inference-v1', tag=tag, weights_license=weights_license,
                    epoch=int(payload['epoch']), split_id=run.get('split_id'),
                    loss='legacy_configuration_only', source_checkpoint_sha256=sha256(source),
                    selection=state.get('selection'), best_epoch=state.get('best_epoch'),
                    best_validation_score=state.get('best_score'),
                    checkpoint_validation_score=state.get('best_score') if payload['epoch'] == state.get('best_epoch') else None,
                    model_config=asdict(config), unit='m', length_axis='XYZ; world-up supplied by input data',
                    diagnostic_only=bool(run.get('diagnostic_only', False) or state.get('limited_history', False)),
                    best_score_is_not_last_score=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    inference = dict(format_version=3, model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                     model_config=asdict(config), epoch=int(payload['epoch']), metadata=metadata)
    temporary = output.with_suffix('.pt.tmp')
    torch.save(inference, temporary)
    reloaded, _, _ = load_model(temporary)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), reloaded.state_dict()[key].cpu(), rtol=0, atol=0)
    temporary.replace(output)
    metadata.update(checkpoint_sha256=sha256(output), bytes=output.stat().st_size)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    output.with_suffix('.sha256').write_text(metadata['checkpoint_sha256'] + '  ' + output.name + '\n', encoding='utf-8')
    return metadata


def smooth_result(source, output, *, window=31, fps=None):
    from .recovery.postprocess import savgol_pose
    output = Path(output)
    if output.suffix != '.npz' or output.exists():
        raise ValueError('Choose a new .npz output')
    with np.load(source, allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    if not np.isfinite(values['params']).all():
        raise ValueError('Failed frames are present; do not smooth across failed reconstruction')
    parents = values['parents']
    processed = savgol_pose(values['params'][:, :3], values['params'][:, 3:].reshape(-1, len(parents), 4),
                            values['bind'], parents, window=window, fps=fps)
    metadata = processed.pop('metadata')
    values.update(raw_params=values['params'], raw_transforms=values['transforms'], **processed)
    values['params'] = np.concatenate([values['root_positions'], values['local_quaternions'].reshape(len(values['params']), -1)], axis=1)
    values['processing'] = np.array(json.dumps(metadata))
    for key in ('success', 'cap', 'nfev', 'total_nfev', 'marker_rmse_mm'):
        if key in values:
            values['raw_solver_' + key] = values.pop(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **values)
    return dict(output=str(output), processing=metadata, input_sha256=sha256(source), sha256=sha256(output))
