import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .recovery.base import sha256
from .recovery.config import ModelConfig
from .recovery.checkpoint import load_model
from .recovery.data import load_common
from .recovery.sampling import FrameDataset
from .recovery.splits import resolve_split
from .recovery.train import _epoch


def evaluate(cfg, checkpoint, output, *, device='cpu', condition='real'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(cfg.get('cpu_threads', 4))
    partitions, split_id = resolve_split(cfg)
    model, config, payload = load_model(checkpoint, device=device)
    if config != ModelConfig(**cfg['model']):
        raise ValueError('Checkpoint architecture differs from evaluation configuration')
    data = cfg['dataset']
    options = {}
    if data.get('synthesis_store'):
        store = Path(data['synthesis_store'])
        options.update(store=store if store.is_absolute() else Path(cfg['data_root']) / store,
                       source_root=cfg['data_root'])
    ds = FrameDataset(common=load_common(cfg['common']), paths=partitions['val'], model_config=config,
                      seed=cfg.get('seed', 2024), samples_per_clip=data.get('val_samples_per_clip', 32),
                      ratios=[1, 0, 0] if condition == 'real' else [0, 0, 1], noise=condition != 'clean',
                      style=data.get('style', 'legacy'), **options)
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], num_workers=cfg['training'].get('num_workers', 0))
    result = _epoch(model, loader, device, 'legacy', dataset_breakdown=True)
    result.update(macro=sum(m['total'] for m in result['datasets'].values()) / len(result['datasets']),
                  checkpoint_sha256=sha256(checkpoint), split_id=split_id, condition=condition,
                  epoch=payload.get('epoch'), selection_condition='real', metric='baseline_configuration_loss')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result
