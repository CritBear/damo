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
        "SFU",
        "MoSh",
        "SOMA"
    ]

    noises = [
        'j',
        'jg',
        'jgo',
        'jgos'
    ]

    for data_name in data_names:
        eval_data_path = Paths.datasets / 'eval' / f'damo_eval_synthetic_{data_name}.pkl'
        with open(eval_data_path, 'rb') as f:
            eval_data = pickle.load(f)

        for noise in noises:
            inference_synthetic(data_name, noise, eval_data)




def inference_synthetic(data_name, noise, eval_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu = torch.device('cpu')

    model_name = 'damo_240408222808'
    epoch = '200'
    model_dir = Paths.trained_models / model_name
    model_path = model_dir / f'{model_name}_epc{epoch}.pt'
    options_path = model_dir / 'options.json'
    options = load_options_from_json_for_inference(options_path)

    print(f'{model_name}_{epoch} | {data_name} | {noise}')

    model = Damo(options).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    output_dir = Paths.test_results / 'model_outputs' / f'{model_name}_epc{epoch}_synthetic'
    output_dir.mkdir(parents=True, exist_ok=True)

    n_files = len(eval_data[noise]['points_seq'])

    batch_size = model.options.batch_size
    n_max_markers = model.options.n_max_markers
    seq_len = model.options.seq_len

    eval_data[noise]['indices'] = []
    eval_data[noise]['weights'] = []
    eval_data[noise]['offsets'] = []
    with torch.no_grad():
        for fi in range(n_files):
            file_points_seq = torch.from_numpy(eval_data[noise]['points_seq'][fi]).to(device)
            file_points_mask = torch.from_numpy(eval_data[noise]['points_mask'][fi]).to(device)

            n_frames = file_points_seq.shape[0]

            indices_list = []
            weights_list = []
            offsets_list = []

            for bi in range(n_frames // batch_size):
                print(f"\rInfer... [{fi + 1}/{n_files}][{bi + 1}/{n_frames // batch_size}]", end='')

                points_seq = file_points_seq[bi*batch_size:(bi+1)*batch_size]
                points_mask = file_points_mask[bi*batch_size:(bi+1)*batch_size]

                indices, weights, offsets = model(points_seq, points_mask)

                indices_list.append(indices)
                weights_list.append(weights)
                offsets_list.append(offsets)

            print('')

            eval_data[noise]['indices'].append(torch.cat(indices_list, dim=0).cpu().numpy())
            eval_data[noise]['weights'].append(torch.cat(weights_list, dim=0).cpu().numpy())
            eval_data[noise]['offsets'].append(torch.cat(offsets_list, dim=0).cpu().numpy())

    output_path = output_dir / f'{model_name}_epc{epoch}_synthetic_{data_name}_{noise}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(eval_data[noise], f)





if __name__ == '__main__':
    main()