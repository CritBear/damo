# DAMO — baseline release

A recovered implementation of DAMO for marker configuration prediction and motion reconstruction. 

## Install

Use Python 3.11 or newer. Install a PyTorch build suitable for your GPU before installing this package if you intend to train on CUDA.

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS or `.venv/Scripts/Activate.ps1` in Windows PowerShell, then install:

```bash
python -m pip install -e .
python -m damo --help
```

For preparation from licensed SMPL-X recordings and C3D input:

```bash
python -m pip install -e ".[raw]"
```

The core training and inference dependencies are PyTorch, NumPy, SciPy, PyYAML, and threadpoolctl. The `raw` extra adds the official `smplx` LBS implementation and a C3D reader. Body-model parameters must be obtained separately.

## Data preparation

Obtain the datasets and body models directly under their applicable licenses. 
[AMASS](https://amass.is.tue.mpg.de/)

Expected local layout:

```text
data/raw/ACCAD/.../*_stageii.npz
data/raw/CMU/.../*_stageii.npz
data/raw/DanceDB/.../*_stageii.npz
data/raw/HDM05/.../*_stageii.npz
data/raw/PosePrior/.../*_stageii.npz
data/raw/SFU/.../*_stageii.npz
data/raw/SOMA/.../*_stageii.npz
data/body_models/SMPLX_NEUTRAL.npz
data/body_models/SMPLX_MALE.npz
data/body_models/SMPLX_FEMALE.npz
```

```bash
python -m damo build-base --model data/body_models/SMPLX_NEUTRAL.npz --output data/native_base.npz --source-description "Locally licensed SMPL-X model" --bodies 128 --num-betas 16
python -m damo prepare --raw-root data/raw --models data/body_models --common data/native_base.npz --output data/cache
python -m damo split --data-root data/cache --output data/cache/split.json --val-fraction 0.2 --seed 2024
```

## Train and evaluate

Run from the repository root. Top-level paths in the YAML are relative to the configuration file; `split_manifest` and `synthesis_store` are relative to `data_root`.

```bash
python -m damo train --config configs/baseline.yaml --device cuda
python -m damo train --config configs/baseline.yaml --device cuda --resume runs/baseline/last.pt
python -m damo evaluate --config configs/baseline.yaml --checkpoint runs/baseline/best.pt --condition real --device cuda --output runs/baseline/validation_real.json
python -m damo evaluate --config configs/baseline.yaml --checkpoint runs/baseline/best.pt --condition clean --device cuda --output runs/baseline/validation_clean.json
python -m damo evaluate --config configs/baseline.yaml --checkpoint runs/baseline/best.pt --condition noisy --device cuda --output runs/baseline/validation_noisy.json
```

## Predict and reconstruct

Input is an NPZ containing `points` or `markers` of shape `(frames, markers, 3)`, plus an explicit `(frames, markers)` `mask`. NPY and C3D are also accepted. Without an explicit mask, finite nonzero points are treated as observed; supply a mask to represent a valid point at the origin. Units are specified with `--unit m`, `cm`, or `mm`. Maximum input marker slots: 90. Predictions preserve the input column order and use zero-padded context at sequence boundaries.

```bash
python -m damo predict --checkpoint checkpoints/baseline-best.pt --input data/sample.npz --output outputs/sample_configuration.npz --device cuda
python -m damo build-prior --config configs/baseline.yaml --output priors/baseline
python -m damo infer --checkpoint checkpoints/baseline-best.pt --recipe priors/baseline/recipe.json --input data/sample.npz --output outputs/sample --device cuda --workers 8 --stable-marker-slots
python -m damo smooth --input outputs/sample/result.npz --output outputs/sample/sg31.npz --fps 120
```
