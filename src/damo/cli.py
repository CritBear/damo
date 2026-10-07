import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog='damo', description='DAMO baseline training and reconstruction')
    commands = parser.add_subparsers(dest='command', required=True)
    base = commands.add_parser('build-base', help='Build a local body22 base from a licensed SMPL-X model')
    base.add_argument('--model', required=True)
    base.add_argument('--output', required=True)
    base.add_argument('--source-description', required=True)
    base.add_argument('--bodies', type=int, default=128)
    base.add_argument('--num-betas', type=int, default=16)
    base.add_argument('--seed', type=int, default=2024)
    prepare = commands.add_parser('prepare', help='Prepare licensed SMPL-X stageii motion/marker recordings')
    prepare.add_argument('--raw-root', required=True)
    prepare.add_argument('--models', required=True)
    prepare.add_argument('--common', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--datasets', nargs='+', default=['ACCAD','CMU','DanceDB','HDM05','PosePrior','SFU','SOMA'])
    prepare.add_argument('--chunk-frames', type=int, default=600)
    prepare.add_argument('--device', default='cpu')
    split = commands.add_parser('split', help='Split original motions inside every dataset')
    split.add_argument('--data-root', required=True)
    split.add_argument('--output', required=True)
    split.add_argument('--val-fraction', type=float, default=.2)
    split.add_argument('--seed', type=int, default=2024)
    train = commands.add_parser('train', help='Train using baseline configuration loss only')
    train.add_argument('--config', required=True)
    train.add_argument('--device', default='cpu')
    train.add_argument('--resume')
    train.add_argument('--epochs', type=int)
    train.add_argument('--max-steps', type=int)
    train.add_argument('--max-val-steps', type=int)
    prior = commands.add_parser('build-prior', help='Fit shape and confidence priors using train motions only')
    prior.add_argument('--config', required=True)
    prior.add_argument('--output', required=True)
    predict = commands.add_parser('predict', help='Predict marker configurations without pose solving')
    predict.add_argument('--checkpoint', required=True)
    predict.add_argument('--input', required=True)
    predict.add_argument('--output', required=True)
    predict.add_argument('--device', default='cpu')
    predict.add_argument('--unit', choices=['m','cm','mm'], default='m')
    predict.add_argument('--batch-size', type=int, default=64)
    infer = commands.add_parser('infer', help='Predict and reconstruct a motion with a train-only solver recipe')
    infer.add_argument('--checkpoint', required=True)
    infer.add_argument('--recipe', required=True)
    infer.add_argument('--input', required=True)
    infer.add_argument('--output', required=True)
    infer.add_argument('--device', default='cpu')
    infer.add_argument('--unit', choices=['m','cm','mm'], default='m')
    infer.add_argument('--workers', type=int, default=8)
    infer.add_argument('--stable-marker-slots', action='store_true')
    smooth = commands.add_parser('smooth', help='Apply optional offline SG31 smoothing to a solved motion')
    smooth.add_argument('--input', required=True)
    smooth.add_argument('--output', required=True)
    smooth.add_argument('--window', type=int, default=31)
    smooth.add_argument('--fps', type=float)
    evaluate = commands.add_parser('evaluate', help='Measure configuration loss on validation, without pose solving')
    evaluate.add_argument('--config', required=True)
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--device', default='cpu')
    evaluate.add_argument('--condition', choices=['real','clean','noisy'], default='real')
    evaluate.add_argument('--output', required=True)
    export = commands.add_parser('export-checkpoint', help='Export baseline weights with public metadata and SHA256')
    export.add_argument('--checkpoint', required=True)
    export.add_argument('--output', required=True)
    export.add_argument('--tag', required=True)
    export.add_argument('--weights-license', required=True)
    args = parser.parse_args()
    if args.command == 'build-base':
        from .recovery.base import build_base
        result = build_base(args.model, args.output, source_description=args.source_description, bodies=args.bodies,
                            num_betas=args.num_betas, seed=args.seed)
    elif args.command == 'prepare':
        from .recovery.prepare import prepare_native
        result = prepare_native(args.raw_root, args.models, args.common, args.output, datasets=args.datasets,
                                chunk_frames=args.chunk_frames, device=args.device)
    elif args.command == 'split':
        from .recovery.splits import create_split
        result = create_split(args.data_root, args.output, val_fraction=args.val_fraction, seed=args.seed)
    elif args.command in ('train', 'build-prior'):
        from .recovery.config import load_config
        cfg = load_config(args.config)
        if args.command == 'train':
            from .recovery.train import train
            result = train(cfg, device=args.device, resume=args.resume, epochs=args.epochs,
                           max_steps=args.max_steps, max_val_steps=args.max_val_steps)
        else:
            from .recovery.prior import build_prior
            result = build_prior(cfg, args.output)
    elif args.command == 'predict':
        from .recovery.inference import infer
        if Path(args.output).exists() or Path(args.output).suffix != '.npz':
            raise ValueError('Choose a new .npz output')
        result = infer(args.checkpoint, args.input, args.output, device=args.device, unit=args.unit, batch_size=args.batch_size)
    elif args.command == 'infer':
        from .reconstruct import infer, write_json
        existed = Path(args.output).exists()
        try:
            result = infer(args.checkpoint, args.recipe, args.input, args.output, device=args.device,
                           unit=args.unit, workers=args.workers, stable_marker_slots=args.stable_marker_slots)
        except Exception as exc:
            if not existed and Path(args.output).is_dir():
                write_json(Path(args.output) / 'status.json', dict(state='failed', error=str(exc)))
            raise
    elif args.command == 'smooth':
        from .artifacts import smooth_result
        result = smooth_result(args.input, args.output, window=args.window, fps=args.fps)
    elif args.command == 'evaluate':
        from .evaluation import evaluate
        from .recovery.config import load_config
        result = evaluate(load_config(args.config), args.checkpoint, args.output, device=args.device, condition=args.condition)
    else:
        from .artifacts import export_checkpoint
        result = export_checkpoint(args.checkpoint, args.output, tag=args.tag, weights_license=args.weights_license)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
