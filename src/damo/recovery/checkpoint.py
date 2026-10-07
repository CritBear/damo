from dataclasses import asdict, replace
from pathlib import Path
import torch
from .config import ModelConfig
from .model import Damo

def load_model(path, device='cpu', config=None, *, joint_distribution=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    state = payload.get('model_state', payload)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    if config is None:
        if 'model_config' in payload:
            saved = {'joint_distribution': 'linear', **payload['model_config']}
            config = ModelConfig(**saved)
        else:
            n_joints = state['joint_indices_predictor.1.res_conv1d.3.weight'].shape[0] - 1
            layers = {int(k.split('.')[2]) for k in state if k.startswith('attention_layers.total_attention_layers.')}
            config = ModelConfig(n_joints=n_joints, seq_len=state['post_attention_layer.1.weight'].shape[1], d_model=state['embedding.1.res_conv2d.3.weight'].shape[0], d_hidden=state['embedding.1.res_conv2d.0.weight'].shape[0], n_layers=len(layers), joint_distribution='linear')
    if joint_distribution is not None:
        config = replace(config, joint_distribution=joint_distribution)
    model = Damo(config)
    model.load_state_dict(state, strict=True)
    return (model.to(device), config, payload)

def save_checkpoint(path, model, optimizer, epoch, config, run_config, rng_state, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bare = model.module if hasattr(model, 'module') else model
    payload = {'format_version': 2, 'model_state': bare.state_dict(), 'model_config': asdict(config), 'optimizer_state': optimizer.state_dict(), 'epoch': epoch, 'run_config': run_config, 'rng_state': rng_state, 'training_state': training_state or {}}
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
