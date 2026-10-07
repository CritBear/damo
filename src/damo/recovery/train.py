from dataclasses import asdict
from pathlib import Path
import json
import hashlib
import platform
import shutil
import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from .checkpoint import load_model, save_checkpoint
from .config import ModelConfig
from .data import MarkerDataset, discover_clips, load_common
from .loss import configuration_loss
from .model import Damo
from .splits import resolve_split, digest

class EvaluationShard(Sampler):

    def __init__(self, dataset, rank, world):
        self.indices = range(rank, len(dataset), world)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def update_early_stopping(score, epoch, settings, state=None):
    state = dict(state or {})
    if score < state.get('reference_score', float('inf')) - settings.get('min_delta', 0.0):
        state.update(reference_score=float(score), reference_epoch=int(epoch))
    state['stale_epochs'] = int(epoch - state['reference_epoch'])
    state['stop'] = bool(epoch >= settings.get('min_epochs', 0) and state['stale_epochs'] >= settings['patience'])
    return state

def _move(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

def _epoch(model, loader, device, loss_mode, optimizer=None, max_steps=None, grad_clip=None, dataset_breakdown=False):
    model.train(optimizer is not None)
    started = time.monotonic()
    keys = ('total', 'indices', 'weights', 'offsets')
    count_index = len(keys)
    totals = torch.zeros(count_index + 1, dtype=torch.float64, device=device)
    group_totals = torch.zeros((len(loader.dataset.dataset_names), count_index + 1), dtype=torch.float64, device=device) if dataset_breakdown else None
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            batch = _move(batch, device)
            predictions = model(batch['points_seq'], batch['points_mask'])
            losses = configuration_loss(predictions, batch, mode=loss_mode)
            objective = losses.get('objective', losses['total'])
            if not torch.isfinite(objective):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                else:
                    finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                    if finite and (not torch.stack(finite).all()):
                        raise FloatingPointError('Nonfinite gradient')
                optimizer.step()
            count = batch['points_mask'][:, batch['points_mask'].shape[1] // 2].sum()
            totals[:count_index] += torch.stack([losses[k].detach().double() for k in keys]) * count
            totals[count_index] += count
            if group_totals is not None:
                for group in range(len(group_totals)):
                    select = batch['dataset_idx'] == group
                    if not select.any():
                        continue
                    subset = {key: value[select] for key, value in batch.items() if torch.is_tensor(value)}
                    metrics = configuration_loss(tuple((p[select] for p in predictions)), subset, mode=loss_mode)
                    markers = subset['points_mask'][:, subset['points_mask'].shape[1] // 2].sum()
                    group_totals[group, :count_index] += torch.stack([metrics[k].detach().double() for k in keys]) * markers
                    group_totals[group, count_index] += markers
            if optimizer is not None and int(os.getenv('RANK', 0)) == 0 and (step == 0 or (step + 1) % 100 == 0 or step + 1 == len(loader)):
                print(json.dumps({'event': 'train_progress', 'step': step + 1, 'steps': min(len(loader), max_steps) if max_steps is not None else len(loader), 'seconds': time.monotonic() - started, 'mean_loss': (totals[0] / totals[count_index]).item()}), flush=True)
    if dist.is_initialized():
        dist.all_reduce(totals)
        if group_totals is not None:
            dist.all_reduce(group_totals)
    if totals[count_index] == 0:
        raise ValueError('No observed markers processed; check data and step limits')
    result = {**dict(zip(keys, (totals[:count_index] / totals[count_index]).cpu().tolist())), 'markers': int(totals[count_index])}
    if group_totals is not None:
        result['datasets'] = {name: {**dict(zip(keys, (row[:count_index] / row[count_index]).cpu().tolist())), 'markers': int(row[count_index])} for name, row in zip(loader.dataset.dataset_names, group_totals) if row[count_index] > 0}
    return result

def train(cfg, *, device='cpu', resume=None, epochs=None, max_steps=None, max_val_steps=None):
    rank, world, local_rank = (int(os.getenv('RANK', 0)), int(os.getenv('WORLD_SIZE', 1)), int(os.getenv('LOCAL_RANK', 0)))
    if world > 1:
        if device.startswith('cuda'):
            torch.cuda.set_device(local_rank)
            device = f'cuda:{local_rank}'
        dist.init_process_group('nccl' if device.startswith('cuda') and os.name != 'nt' else 'gloo')
    try:
        return _train(cfg, device, resume, epochs, max_steps, max_val_steps, rank, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def _train(cfg, device, resume, epochs, max_steps, max_val_steps, rank, world):
    seed = int(cfg.get('seed', 2024))
    seed_everything(seed)
    torch.set_num_threads(int(cfg.get('cpu_threads', 4)))
    model_cfg = ModelConfig(**cfg['model'])
    cfg['implementation_id'] = digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))})
    cfg['runtime'] = {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': str(np.__version__), 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else 'cpu'}
    training = cfg.get('training', {})
    if training.get('loss', 'legacy') != 'legacy' or training.get('geometry_loss') or training.get('orientation_loss'):
        raise ValueError('Use baseline legacy loss without auxiliary objectives')
    early_settings = training.get('early_stopping')
    if early_settings and (early_settings.get('patience', 0) < 1 or early_settings.get('min_delta', 0) < 0 or early_settings.get('min_epochs', 0) < 0):
        raise ValueError('Invalid early stopping configuration')
    selection = training.get('selection', 'mean_validation_total')
    if selection not in ('mean_validation_total', 'dataset_macro_validation_total'):
        raise ValueError('Unknown checkpoint selection criterion')
    data_cfg = cfg.get('dataset', {})
    splits, split_id = resolve_split(cfg)
    if split_id:
        cfg['split_id'] = split_id
    common_path = cfg.get('common') or str(Path(cfg['data_root']) / 'common' / f"damo_common_{data_cfg.get('date', '20240329')}.pkl")
    common = load_common(common_path)
    cache_manifest = Path(cfg['data_root']) / 'cache_manifest.json'
    if cache_manifest.exists() and json.loads(cache_manifest.read_text()).get('requires_observed_store'):
        if not (data_cfg.get('observed_store') or (data_cfg.get('fast_synthetic') and data_cfg.get('synthesis_store'))):
            raise ValueError('Compact native cache requires its verified memory-mapped store')
    if data_cfg.get('val_modes', ['real']) != ['real']:
        raise ValueError('Checkpoint selection in this baseline release uses real validation only')
    train_paths, val_paths = (splits['train'], splits['val'])
    dataset_args = {'common': common, 'model_config': model_cfg, 'seed': seed, 'style': data_cfg.get('style', 'legacy'), 'cache_size': data_cfg.get('cache_size', 2), 'joint_policy': data_cfg.get('joint_policy', 'native'), 'pose_targets': False}
    from .sampling import FrameDataset
    dataset_type = FrameDataset
    if data_cfg.get('synthesis_store'):
        store = Path(data_cfg['synthesis_store'])
        dataset_args.update(store=store if store.is_absolute() else Path(cfg['data_root']) / store, source_root=cfg['data_root'])
    train_set = dataset_type(paths=train_paths, draws=data_cfg.get('train_samples_per_epoch'), samples_per_clip=data_cfg.get('samples_per_clip', 100), ratios=data_cfg.get('ratios', [0.5, 0.25, 0.25]), sampling=data_cfg.get('sampling', 'clip_uniform'), mirror_probability=data_cfg.get('mirror_probability', 0.0), **dataset_args)
    cfg['sampling_summary'] = {'method': train_set.sampling, 'datasets': {name: {'clips': len(group), 'pose_strata': len(train_set.pose_groups[i]) if train_set.pose_groups else None} for i, (name, group) in enumerate(zip(train_set.dataset_names, train_set.dataset_groups))}}
    train_sampler = DistributedSampler(train_set, world, rank, seed=seed) if world > 1 else None
    workers = int(training.get('num_workers', 0))
    loader_args = dict(batch_size=int(training.get('batch_size', 64)), num_workers=workers, pin_memory=device.startswith('cuda'), persistent_workers=False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, sampler=train_sampler, shuffle=train_sampler is None, generator=generator, drop_last=False, **loader_args)
    val_loaders = {}
    for name in data_cfg.get('val_modes', ['real', 'synthetic']):
        ratios = {'real': [1, 0, 0], 'synthetic': data_cfg.get('val_synthetic_ratios', [0, 0.5, 0.5])}[name]
        if name == 'synthetic' and (len(ratios) != 3 or ratios[0] != 0):
            raise ValueError('Synthetic validation ratios must be [0, superset, arbitrary]')
        ds = dataset_type(paths=val_paths, samples_per_clip=data_cfg.get('val_samples_per_clip', 10), ratios=ratios, **dataset_args)
        val_loaders[name] = DataLoader(ds, sampler=EvaluationShard(ds, rank, world) if world > 1 else None, shuffle=False, **loader_args)
    if resume:
        model, saved_cfg, payload = load_model(resume, device=device, config=model_cfg)
        if 'optimizer_state' not in payload or 'epoch' not in payload:
            raise ValueError('Resume requires a recovery training checkpoint with optimizer state')
        if payload['model_config'] != asdict(model_cfg):
            raise ValueError('Resume model configuration differs')
        for key in ('seed', 'dataset', 'training', 'split_id'):
            if payload['run_config'].get(key) != cfg.get(key):
                raise ValueError(f'Resume configuration differs in {key}; use the original configuration')
        if payload['run_config'].get('implementation_id', cfg['implementation_id']) != cfg['implementation_id']:
            raise ValueError('Recovery source code changed since checkpoint; resume with the recorded implementation')
        start_epoch = int(payload['epoch'])
    else:
        model = Damo(model_cfg).to(device)
        start_epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training.get('lr', 0.001)), weight_decay=float(training.get('weight_decay', 0)))
    if resume:
        optimizer.load_state_dict(payload['optimizer_state'])
        rng = payload['rng_state']
        torch.set_rng_state(rng['torch'].cpu())
        generator.set_state(rng['loader'].cpu())
        if rng.get('cuda') is not None and device.startswith('cuda'):
            torch.cuda.set_rng_state_all([value.cpu() for value in rng['cuda']])
    scheduler = None
    if training.get('scheduler') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training['epochs']), eta_min=float(training.get('min_lr', 1e-05)))
    elif training.get('scheduler') == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=float(training.get('plateau_factor', 0.5)), patience=int(training.get('plateau_patience', 8)), threshold_mode='abs', threshold=float(training.get('plateau_threshold', 0.0001)), min_lr=float(training.get('min_lr', 1e-05)))
    elif training.get('scheduler') not in (None, 'none'):
        raise ValueError('Unknown learning-rate scheduler')
    if scheduler is not None and resume:
        if 'scheduler_state' not in payload.get('training_state', {}):
            raise ValueError('Resume checkpoint lacks scheduler state')
        scheduler.load_state_dict(payload['training_state']['scheduler_state'])
    state = payload.get('training_state', {}) if resume else {}
    early_state = state.get('early_stopping')
    if early_state and early_state.get('stop'):
        raise ValueError('This run already met its validation stopping rule; preserve it')
    best_score, best_epoch = (state.get('best_score', float('inf')), state.get('best_epoch', 0))
    limited_history = state.get('limited_history', bool(resume and (not state))) or max_steps is not None or max_val_steps is not None
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[int(os.getenv('LOCAL_RANK', 0))] if device.startswith('cuda') else None)
    output = Path(cfg.get('output_dir', 'outputs/recovery'))
    if (output / 'last.pt').exists() and resume is None:
        raise FileExistsError(f'{output}/last.pt exists; choose a new output directory or --resume')
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        source_snapshot = output / 'implementation'
        source_snapshot.mkdir(exist_ok=True)
        for path in Path(__file__).parent.glob('*.py'):
            shutil.copyfile(path, source_snapshot / path.name)
        if split_id:
            source_manifest = Path(data_cfg['split_manifest'])
            if not source_manifest.is_absolute():
                source_manifest = Path(cfg['data_root']) / source_manifest
            shutil.copyfile(source_manifest, output / 'split_manifest.json')
        (output / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    n_epochs = int(epochs if epochs is not None else training.get('epochs', 300))
    if n_epochs <= start_epoch:
        raise ValueError('Requested final epoch must exceed the resumed epoch')
    summary = None
    for epoch in range(start_epoch, n_epochs):
        started = time.monotonic()
        train_set.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = _epoch(model, train_loader, device, 'legacy', optimizer, max_steps, training.get('grad_clip'))
        bare = model.module if hasattr(model, 'module') else model
        if world > 1:
            for buffer in bare.buffers():
                dist.broadcast(buffer, src=0)
        val_metrics = {name: _epoch(bare, loader, device, 'legacy', max_steps=max_val_steps, dataset_breakdown=selection == 'dataset_macro_validation_total') for name, loader in val_loaders.items()}
        summary = {'epoch': epoch + 1, 'train': train_metrics, 'validation': val_metrics, 'seconds': time.monotonic() - started, 'lr': optimizer.param_groups[0]['lr'], 'world_size': world, 'limited_steps': max_steps is not None or max_val_steps is not None}
        if selection == 'dataset_macro_validation_total':
            score = float(np.mean([group['total'] for metric in val_metrics.values() for group in metric['datasets'].values()]))
        else:
            score = float(np.mean([metric['total'] for metric in val_metrics.values()]))
        if not np.isfinite(score):
            raise FloatingPointError('Nonfinite validation selection score')
        improved = score < best_score
        if improved:
            best_score, best_epoch = (score, epoch + 1)
        summary.update(selection_score=score, best_score=best_score, best_epoch=best_epoch)
        if scheduler is not None:
            if training.get('scheduler') == 'plateau':
                scheduler.step(score)
            else:
                scheduler.step()
        if early_settings:
            early_state = update_early_stopping(score, epoch + 1, early_settings, early_state)
            summary['early_stopping'] = early_state.copy()
            summary['next_lr'] = optimizer.param_groups[0]['lr']
        state = {'best_score': best_score, 'best_epoch': best_epoch, 'selection': selection, 'limited_history': limited_history, 'release_eligible': bool(split_id) and (not limited_history) and (not cfg.get('diagnostic_only', False))}
        if scheduler is not None:
            state['scheduler_state'] = scheduler.state_dict()
        if early_settings:
            state['early_stopping'] = early_state.copy()
        if rank == 0:
            with (output / 'metrics.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(summary) + '\n')
            print(json.dumps(summary), flush=True)
            rng = {'torch': torch.get_rng_state(), 'loader': generator.get_state(), 'cuda': torch.cuda.get_rng_state_all() if device.startswith('cuda') else None}
            if improved:
                save_checkpoint(output / 'best.pt', model, optimizer, epoch + 1, model_cfg, cfg, rng, state)
            save_checkpoint(output / 'last.pt', model, optimizer, epoch + 1, model_cfg, cfg, rng, state)
            if (epoch + 1) % training.get('save_every', 10) == 0:
                save_checkpoint(output / f'epoch_{epoch + 1:04d}.pt', model, optimizer, epoch + 1, model_cfg, cfg, rng, state)
        if early_settings and early_state['stop']:
            break
    if early_settings and rank == 0:
        reason = 'validation_plateau' if early_state['stop'] else 'epoch_cap' if summary['epoch'] >= int(training['epochs']) else 'requested_epoch_limit'
        (output / 'training_complete.json').write_text(json.dumps({'stop_reason': reason, 'completed_epochs': summary['epoch'], 'best_epoch': best_epoch, 'best_score': best_score, 'early_stopping': early_state, 'convergence_not_proven_at_cap': reason == 'epoch_cap'}, indent=2), encoding='utf-8')
    return summary
