import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
import pickle
from tqdm import tqdm
import time

from modules.training.training_options import load_options_from_json_for_inference
from modules.utils.paths import Paths
from modules.network.damo import Damo
from modules.dataset.damo_dataset import DamoDataset

from modules.solver.pose_solver import find_bind_pose, find_pose
from modules.solver.scipy_pose_solver import find_pose as scipy_find_pose
from modules.utils.viewer.vpython_viewer import VpythonViewer as Viewer
from modules.evaluation.score_manager import ScoreManager
from human_body_prior.body_model.lbs import batch_rodrigues
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R


def main():

    data_names = [
        "HDM05",
        # "SFU",
        # "MoSh",
        # "SOMA"
    ]

    noises = [
        'j',
        'jg',
        'jgo',
        'jgos'
    ]
    model_name = 'damo_240412022202'
    epoch = '100'

    for data_name in data_names:
        for noise in noises:
            inference_synthetic(data_name, noise, model_name, epoch)



def inference_synthetic(data_name, noise, model_name, epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu = torch.device('cpu')

    # model_dir = Paths.trained_models / 'seraph' / model_name
    model_dir = Paths.trained_models / model_name
    model_path = model_dir / f'{model_name}_epc{epoch}.pt'
    options_path = model_dir / 'options.json'
    options = load_options_from_json_for_inference(options_path)

    print(f'{model_name}_{epoch} | {data_name} | {noise}')

    eval_data_path = Paths.datasets / 'eval' / 'legacy' / 'eval_data_for_ms' / 'all_ref_idx' / f'eval_data_ms_{data_name}_{noise}_ARI.pkl'
    with open(eval_data_path, 'rb') as f:
        eval_data = pickle.load(f)

    # print(eval_data['synthetic_marker'][0].shape)
    # print(eval_data['synthetic_marker_num'][0].shape)
    # print(eval_data['joint_local_rotations'][0].shape)
    # print(eval_data['joint_global_rotations'][0].shape)
    # print(eval_data['joint_local_positions'][0].shape)
    # print(eval_data['joint_global_positions'][0].shape)

    results = {
        'points_seq': [],
        'indices': [],
        'weights': [],
        'offsets': [],
        'jlr': [],
        'jgp': [],
        'bind_jlp': [],
        'body_idx': []
    }

    model = Damo(options).to(device)
    model.load_state_dict(torch.load(model_path))
    # ddp_state_dict = {key.replace('module.', ''): value for key, value in torch.load(model_path).items()}
    # model.load_state_dict(ddp_state_dict)
    model.eval()
    output_dir = Paths.test_results / 'model_outputs' / f'{model_name}_epc{epoch}_lsynthetic'
    output_dir.mkdir(parents=True, exist_ok=True)

    n_files = len(eval_data['synthetic_marker'])

    batch_size = options.batch_size
    n_max_markers = options.n_max_markers
    seq_len = options.seq_len

    loss_count = 0

    with torch.no_grad():
        for fi in range(n_files):
            file_points_seq = torch.from_numpy(eval_data['synthetic_marker'][fi]).to(device)

            n_frames = file_points_seq.shape[0]

            points_list = []
            indices_list = []
            weights_list = []
            offsets_list = []

            results['jgp'].append(eval_data['joint_global_positions'][fi])
            results['jlr'].append(eval_data['joint_local_rotations'][fi])
            results['bind_jlp'].append(eval_data['joint_local_positions'][fi])
            results['body_idx'].append(eval_data['body_shape_index'][fi])

            for bi in range(n_frames // batch_size):
                print(f"\rInfer... [{fi + 1}/{n_files}][{bi + 1}/{n_frames // batch_size}]", end='')
                padded_points = np.zeros((batch_size, 7, 90, 3))
                mask = np.zeros((batch_size, 7, 90))

                for b in range(batch_size):
                    for s in range(-3, 4):
                        f = bi * batch_size + b + s
                        if f >= 0 and f < n_frames:
                            n_synthetic_markers = int(eval_data['synthetic_marker_num'][fi][f])
                            points = eval_data['synthetic_marker'][fi][f, :n_synthetic_markers, :]
                            mag = np.linalg.norm(points, axis=1)
                            filtered_points = points[mag > 0.001]

                            n_rvm = min(filtered_points.shape[0], 90)

                            padded_points[b, s, :n_rvm, :] = filtered_points[:n_rvm, :]
                            mask[b, s, :n_rvm] = 1

                padded_points = torch.from_numpy(padded_points).to(options.device).to(torch.float32)
                mask = torch.from_numpy(mask).to(options.device).to(torch.float32)
                indices_pred, weights_pred, offsets_pred = model(padded_points, mask)

                indices_list.append(indices_pred)
                weights_list.append(weights_pred)
                offsets_list.append(offsets_pred)
                points_list.append(padded_points)

            results['indices'].append(torch.cat(indices_list, dim=0).cpu().numpy())
            results['weights'].append(torch.cat(weights_list, dim=0).cpu().numpy())
            results['offsets'].append(torch.cat(offsets_list, dim=0).cpu().numpy())
            results['points_seq'].append(torch.cat(points_list, dim=0).cpu().numpy())

    output_path = output_dir / f'{model_name}_epc{epoch}_{data_name}_{noise}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)


def inference_real(data_name, eval_data, model_name, epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu = torch.device('cpu')

    model_dir = Paths.trained_models / 'seraph' / model_name
    # model_dir = Paths.trained_models / model_name
    model_path = model_dir / f'{model_name}_epc{epoch}.pt'
    options_path = model_dir / 'options.json'
    options = load_options_from_json_for_inference(options_path)

    print(f'{model_name}_{epoch} | {data_name} | real')

    # total_params = sum(p.numel() for p in torch.load(model_path).values())
    # print(f"Total parameters: {total_params}")
    # exit()

    model = Damo(options).to(device)
    ddp_state_dict = {key.replace('module.', ''): value for key, value in torch.load(model_path).items()}
    model.load_state_dict(ddp_state_dict)
    # model.load_state_dict(torch.load(model_path))
    model.eval()

    output_dir = Paths.test_results / 'model_outputs' / f'{model_name}_epc{epoch}_synthetic'
    output_dir.mkdir(parents=True, exist_ok=True)

    n_files = len(eval_data['real']['points_seq'])

    batch_size = options.batch_size
    n_max_markers = options.n_max_markers
    seq_len = options.seq_len

    eval_data['real']['indices'] = []
    eval_data['real']['weights'] = []
    eval_data['real']['offsets'] = []

    loss = LossMemory()
    loss_count = 0

    with torch.no_grad():
        for fi in range(n_files):
            batch_loss = LossMemory()
            file_points_seq = torch.from_numpy(eval_data['real']['points_seq'][fi]).to(device)
            file_points_mask = torch.from_numpy(eval_data['real']['points_mask'][fi]).to(device)
            file_indices = torch.from_numpy(eval_data['real']['m_j_weights'][fi]).to(device)
            file_weights = torch.from_numpy(eval_data['real']['m_j3_weights'][fi]).to(device)
            file_offsets = torch.from_numpy(eval_data['real']['m_j3_offsets'][fi]).to(device)

            n_frames = file_points_seq.shape[0]

            indices_list = []
            weights_list = []
            offsets_list = []

            for bi in range(n_frames // batch_size):
                print(f"\rInfer... [{fi + 1}/{n_files}][{bi + 1}/{n_frames // batch_size}]", end='')

                points_seq = file_points_seq[bi * batch_size:(bi + 1) * batch_size]
                points_mask = file_points_mask[bi * batch_size:(bi + 1) * batch_size]
                indices = file_indices[bi * batch_size:(bi + 1) * batch_size]
                weights = file_weights[bi * batch_size:(bi + 1) * batch_size]
                offsets = file_offsets[bi * batch_size:(bi + 1) * batch_size]

                indices_pred, weights_pred, offsets_pred = model(points_seq, points_mask)

                output_mask = points_mask[:, options.seq_len // 2, :]

                loss_dict = loss_type_1(
                    indices=indices,
                    indices_pred=indices_pred,
                    weights=weights,
                    weights_pred=weights_pred,
                    offsets=offsets,
                    offsets_pred=offsets_pred,
                    mask=output_mask
                )
                loss.add_loss_dict(loss_dict)
                batch_loss.add_loss_dict(loss_dict)

                indices_list.append(indices_pred)
                weights_list.append(weights_pred)
                offsets_list.append(offsets_pred)

            if n_frames // batch_size > 0:
                loss_count += n_frames // batch_size

                eval_data['real']['indices'].append(torch.cat(indices_list, dim=0).cpu().numpy())
                eval_data['real']['weights'].append(torch.cat(weights_list, dim=0).cpu().numpy())
                eval_data['real']['offsets'].append(torch.cat(offsets_list, dim=0).cpu().numpy())

                batch_loss.divide(n_frames // batch_size)
                batch_loss.print()

        print('\nTotal_____________________')
        loss.divide(loss_count)
        loss.print()
        print('\n')
    output_path = output_dir / f'{model_name}_epc{epoch}_{data_name}_real.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(eval_data['real'], f)


def loss_type_1(indices, indices_pred, weights, weights_pred, offsets, offsets_pred, mask):
    mse = torch.nn.MSELoss(reduction='none')

    indices_loss = mse(indices, indices_pred)
    indices_loss = indices_loss * mask.unsqueeze(2)
    indices_loss = indices_loss.sum() / mask.sum()

    weights_loss = mse(weights, weights_pred)
    weights_loss = weights_loss * mask.unsqueeze(2)
    weights_loss = weights_loss.sum() / mask.sum()

    offsets_loss = mse(offsets, offsets_pred)
    offsets_loss = offsets_loss * weights.unsqueeze(-1)
    offsets_loss = offsets_loss * mask.unsqueeze(2).unsqueeze(3)
    offsets_loss = offsets_loss.sum() / mask.sum()

    total_loss = indices_loss + weights_loss + offsets_loss

    loss = {
        'total': total_loss,
        'indices': indices_loss,
        'weights': weights_loss,
        'offsets': offsets_loss
    }

    return loss


if __name__ == '__main__':
    main()