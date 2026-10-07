from dataclasses import asdict, dataclass, fields
from pathlib import Path
import os
import yaml

@dataclass
class ModelConfig:
    n_joints: int = 24
    n_max_markers: int = 90
    seq_len: int = 7
    d_model: int = 125
    d_hidden: int = 128
    n_layers: int = 4
    n_heads: int = 5
    attention_mode: str = 'legacy'
    mask_centered_inputs: bool = False
    joint_distribution: str = 'softmax'
    normalization: str = 'batch'

    def __post_init__(self):
        if self.seq_len < 1 or self.seq_len % 2 != 1:
            raise ValueError('seq_len must be positive and odd')
        if self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        if min(self.n_joints, self.n_max_markers, self.d_hidden, self.n_layers) < 1:
            raise ValueError('Model dimensions must be positive')
        if self.attention_mode not in ('legacy', 'center'):
            raise ValueError('attention_mode must be legacy or center')
        if self.joint_distribution not in ('linear', 'softmax'):
            raise ValueError('joint_distribution must be linear or softmax')
        if self.normalization not in ('batch', 'layer'):
            raise ValueError('normalization must be batch or layer')

    @classmethod
    def from_dict(cls, values):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})

def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a mapping')
    for key in ('data_root', 'common', 'output_dir'):
        if key in cfg and cfg[key] is not None:
            value = os.path.expandvars(str(cfg[key]))
            if '$' in value:
                raise ValueError(f'Unresolved environment variable in {key}: {value}')
            p = Path(value).expanduser()
            cfg[key] = str(p if p.is_absolute() else (path.parent / p).resolve())
    cfg['model'] = asdict(ModelConfig(**cfg.get('model', {})))
    cfg['config_path'] = str(path)
    return cfg
