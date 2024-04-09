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
        'SFU': True,
        'Mosh': True,
        'SOMA': True
    }

    eval_noise = {
        'j': True,
        'jg': True,
        'jgo': True,
        'jgos': True,
        'real': True
    }

    model_name = 'damo_240404153143'
    epoch = '120'
    eval_data_date = '20240408'
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

        eval_data_path = Paths.datasets / 'eval' / f'damo_eval_{dataset}_{eval_data_date}.pkl'
        with open(eval_data_path, 'rb') as f:
            eval_data = pickle.load(f)

        for noise, b_noise in eval_noise.items():
            if not b_noise:
                continue

            inference_entry(model, eval_data, dataset, noise)

        output_path = output_dir / f'model_output_{eval_data_path.stem}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(eval_data, f)


def inference_entry(model, eval_data, dataset, noise):
    is_real = True if noise == 'real' else False

    if is_real:
        eval_data['real_indices_pred'] = []
        eval_data['real_weights_pred'] = []
        eval_data['real_offsets_pred'] = []
        points_seq = eval_data['real']
    else:
        eval_data[f'synthetic_{noise}_indices_pred'] = []
        eval_data[f'synthetic_{noise}_weights_pred'] = []
        eval_data[f'synthetic_{noise}_offsets_pred'] = []
        points_seq = eval_data[f'synthetic_{noise}']

    print('')
    print(f'eval_data_{dataset}_{noise}')

    batch_size = model.options.batch_size
    n_max_markers = model.options.n_max_markers
    seq_len = model.options.seq_len

    with torch.no_grad():
        n_files = len(points_seq)
        for fi in range(n_files):
            n_frames = points_seq[fi].shape[0]

            indices = []
            weights = []
            offsets = []

            for bi in range(n_frames // batch_size):
                print(f"\rInfer... [{fi + 1}/{n_files}][{bi + 1}/{n_frames // batch_size}]", end='')
                padded_points = np.zeros((batch_size, seq_len, n_max_markers, 3))
                mask = np.zeros((batch_size, seq_len, n_max_markers))

                for b in range(batch_size):
                    for s in range(-(seq_len//2), (seq_len//2)+1):
                        f = bi * batch_size + b + s
                        if 0 <= f < n_frames:
                            points = points_seq[fi][f, :, :]
                            mag = np.linalg.norm(points, axis=1)
                            filtered_points = points[mag > 0.001]

                            # number of real valid markers
                            n_rvm = min(filtered_points.shape[0], n_max_markers)

                            padded_points[b, s, :n_rvm, :] = filtered_points[:n_rvm, :]
                            mask[b, s, :n_rvm] = 1

                padded_points = torch.from_numpy(padded_points).to(device).to(torch.float32)
                mask = torch.from_numpy(mask).to(device).to(torch.float32)
                indices_pred, weights_pred, offsets_pred = model(padded_points, mask)

                indices.append(indices_pred)
                weights.append(weights_pred)
                offsets.append(offsets_pred)

            indices = torch.cat(indices, dim=0).cpu().numpy()
            weights = torch.cat(weights, dim=0).cpu().numpy()
            offsets = torch.cat(offsets, dim=0).cpu().numpy()

            assert indices.shape[0] == (n_frames // batch_size) * batch_size, \
                f'{indices.shape[0]} | {(n_frames // batch_size) * batch_size}'

            if is_real:
                eval_data['real_indices_pred'].append(indices)
                eval_data['real_weights_pred'].append(weights)
                eval_data['real_offsets_pred'].append(offsets)
            else:
                eval_data[f'synthetic_{noise}_indices_pred'].append(indices)
                eval_data[f'synthetic_{noise}_weights_pred'].append(weights)
                eval_data[f'synthetic_{noise}_offsets_pred'].append(offsets)


if __name__ == '__main__':
    inference()
