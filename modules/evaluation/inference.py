import torch
import numpy as np
import pickle

from modules.training.training_options import load_options_from_json_for_inference, TrainingOptions
from modules.utils.paths import Paths
from modules.dataset.damo_dataset import DamoDataset
from modules.network.damo import Damo


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inference():
    eval_dataset = {
        'HDM05': True,
        'SFU': False,
        'Mosh': False,
        'SOMA': False
    }

    eval_noise = {
        'j': False,
        'jg': False,
        'jgo': False,
        'jgos': False,
        'real': True
    }

    model_name = 'damo_240330034650'
    epoch = '100'
    model_dir = Paths.trained_models / model_name

    model_path = model_dir / f'{model_name}_epc{epoch}.pt'
    options_path = model_dir / 'options.json'
    # options = load_options_from_json_for_inference(options_path)
    options = TrainingOptions()

    model = Damo(options).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    output_dir = Paths.test_results / 'model_outputs' / f'{model_name}_epc{epoch}'
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, b_dataset in eval_dataset.items():
        if not b_dataset:
            continue

        for noise, b_noise in eval_noise.items():
            if not b_noise:
                continue

            inference_entry(model, dataset, noise, output_dir)


def inference_entry(model, dataset, noise, output_dir):
    is_real_marker = True if noise == 'real' else False
    marker_type = 'real' if is_real_marker else 'synthetic'

    print('')
    print(f'eval_data_{dataset}_{noise}')

    eval_data_path = Paths.datasets / 'evaluate' / 'legacy' / f'eval_data_damo-soma_{dataset}_{noise}.pkl'
    with open(eval_data_path, 'rb') as f:
        eval_data = pickle.load(f)

    total_sample = []
    with torch.no_grad():
        n_files = len(eval_data[f'{marker_type}_marker'])
        for fi in range(n_files):
            sample = {
                'points': [],
                'indices': [],
                'weights': [],
                'offsets': [],
                'jgp': [],
                'bind_jlp': [],
                'jlr': []
            }
            sample['jgp'].append(eval_data['joint_global_positions'][fi])
            # sample['bind_jlp'].append(eval_data['joint_local_positions'][fi])
            sample['jlr'].append(eval_data['joint_local_rotations'][fi])

            n_frames = eval_data[f'{marker_type}_marker'][fi].shape[0]

            batch_size = model.options.batch_size

            for bi in range(n_frames // batch_size):
                print(f"\rInfer... [{fi + 1}/{n_files}][{bi + 1}/{n_frames // batch_size}]", end='')
                padded_points = np.zeros((batch_size, 7, 90, 3))
                mask = np.zeros((batch_size, 7, 90))

                for b in range(batch_size):
                    for s in range(-3, 4):
                        f = bi * batch_size + b + s
                        if f >= 0 and f < n_frames:
                            if is_real_marker:
                                points = eval_data['real_marker'][fi][f, :, :]
                                mag = np.linalg.norm(points, axis=1)
                                filtered_points = points[mag > 0.001]

                                # number of real valid markers
                                n_rvm = min(filtered_points.shape[0], 90)

                                padded_points[b, s, :n_rvm, :] = filtered_points[:n_rvm, :]
                                mask[b, s, :n_rvm] = 1
                            else:
                                n_synthetic_markers = int(eval_data['synthetic_marker_num'][fi][f])
                                points = eval_data['synthetic_marker'][fi][f, :n_synthetic_markers, :]
                                mag = np.linalg.norm(points, axis=1)
                                filtered_points = points[mag > 0.001]

                                n_rvm = min(filtered_points.shape[0], 90)

                                padded_points[b, s, :n_rvm, :] = filtered_points[:n_rvm, :]
                                mask[b, s, :n_rvm] = 1

                if is_real_marker:
                    padded_points /= 1000

                padded_points = torch.from_numpy(padded_points).to(device).to(torch.float32)
                mask = torch.from_numpy(mask).to(device).to(torch.float32)
                indices_pred, weight_pred, offset_pred = model(padded_points, mask)

                sample['points'].append(padded_points.cpu().numpy()[:, 3, :, :])
                sample['indices'].append(indices_pred.cpu().numpy())
                sample['weights'].append(weight_pred.cpu().numpy())
                sample['offsets'].append(offset_pred.cpu().numpy())

            if len(sample['points']) > 0:
                sample['points'] = np.vstack(sample['points'])
                sample['indices'] = np.vstack(sample['indices'])
                sample['weights'] = np.vstack(sample['weights'])
                sample['offsets'] = np.vstack(sample['offsets'])

                total_sample.append(sample)

    output_path = output_dir / f'model_output_{eval_data_path.stem}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(total_sample, f)


if __name__ == '__main__':
    inference()
