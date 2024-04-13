import sys
import numpy as np
import pickle
import time
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

from modules.utils.paths import Paths
from modules.evaluation.score_manager import ScoreManager
from human_body_prior.body_model.lbs import batch_rodrigues
from modules.solver.quat_pose_solver_numpy import PoseSolver
from modules.utils.viewer.vpython_viewer import VpythonViewer as Viewer
from modules.solver.pose_solver import find_bind_pose, find_pose, test_lbs
from modules.solver.scipy_pose_solver import find_pose as scipy_find_pose
from modules.utils.functions import dict2class


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


def to_gpu(ndarray):
    return torch.Tensor(ndarray).to(device)


def evaluate():
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
        'jgos': True
    }

    model_name = 'damo_240408222808'
    epoch = '200'

    for dataset, do_dataset in eval_dataset.items():
        if not do_dataset:
            continue

        for noise, do_noise in eval_noise.items():
            if not do_noise:
                continue

            eval_data_dir = Paths.test_results / 'model_outputs' / f'{model_name}_epc{epoch}_synthetic'
            eval_data_path = eval_data_dir / f'{model_name}_epc{epoch}_synthetic_{dataset}_{noise}.pkl'

            with open(eval_data_path, 'rb') as f:
                eval_data = pickle.load(f)

            evaluate_entry(eval_data, dataset, noise, model_name, epoch)


def evaluate_entry(eval_data, dataset, noise, model_name, epoch):
    visualize = False
    generate_figure_data = False
    position_smoothing = True
    rotation_smoothing = True
    pre_motion_idx = -1
    pre_n_frames = -1

    common_data_path = Paths.datasets / 'common' / f'damo_common_20240329.pkl'

    with open(common_data_path, 'rb') as f:
        common_data = pickle.load(f)

    topology = common_data['topology']
    n_joints = topology.shape[0]
    n_max_markers = 90

    score_manager = ScoreManager()
    viewer = Viewer(
        n_max_markers=n_max_markers,
        topology=topology,
        b_vertex=True
    ) if visualize else None

    data_type = 'real' if noise == 'real' else 'synthetic'

    smplh_neutral_path = Paths.support_data / 'body_models' / 'smplh' / 'neutral' / 'model.npz'
    smplh_neutral = np.load(smplh_neutral_path)
    smplh_neutral = dict2class(smplh_neutral)
    base_bind_jgp = to_gpu(smplh_neutral.J)

    n_motions = len(eval_data['points_seq'])
    for motion_idx in range(n_motions):
        if pre_motion_idx != -1:
            motion_idx = pre_motion_idx

        print(f"Solving... motion clip:[{motion_idx + 1}/{n_motions}]")

        seq_len = eval_data['points_seq'][motion_idx].shape[1]

        points = to_gpu(eval_data['points_seq'][motion_idx][:, seq_len//2, :, :])
        indices = to_gpu(eval_data['indices'][motion_idx])
        weights = to_gpu(eval_data['weights'][motion_idx])
        offsets = to_gpu(eval_data['offsets'][motion_idx])

        n_frames = indices.shape[0]

        points = points[:n_frames]

        gt_jgp = to_gpu(eval_data['jgp'][motion_idx][:n_frames])
        poses = to_gpu(eval_data['poses'][motion_idx])
        gt_jlr = batch_rodrigues(poses.view(-1, n_joints, 3)[:n_frames].view(-1, 3)).view(n_frames, n_joints, 3, 3)
        gt_bind_jgp = to_gpu(eval_data['bind_jgp'][motion_idx])
        gt_bind_jlp = to_gpu(eval_data['bind_jlp'][motion_idx])

        j3_indices = torch.argsort(indices, dim=-1, descending=True)[..., :3]  # [f, m, 3]
        j_weights = torch.zeros(n_frames, n_max_markers, n_joints + 1).to(device)  # [f, m, j+1]
        j_offsets = torch.zeros(n_frames, n_max_markers, n_joints + 1, 3).to(device)  # [f, m, j+1, 3]

        j_weights[
            torch.arange(n_frames)[:, None, None],
            torch.arange(n_max_markers)[:, None],
            j3_indices
        ] = weights
        j_weights = j_weights[:, :, :n_joints]  # [f, m, j]

        j_offsets[
            torch.arange(n_frames)[:, None, None],
            torch.arange(n_max_markers)[:, None],
            j3_indices
        ] = offsets
        j_offsets = j_offsets[:, :, :n_joints]  # [f, m, j, 3]

        # for f in range(n_frames):
        #     for j in range(n_max_markers):
        #         if j3_indices[f][j][0] == n_joints:
        #             j3_indices[f][j][0] = 0
        #             j_weights[f][j] = 0
        #             j_offsets[f][j] = 0
        #
        #         if j3_indices[f][j][1] == n_joints:
        #             j3_indices[f][j][1] = 0
        #
        #         if j3_indices[f][j][2] == n_joints:
        #             j3_indices[f][j][2] = 0

        # Debug__________________________________
        start_frame = 0
        if pre_n_frames > 0:
            n_frames = min(n_frames, pre_n_frames)
        # _______________________________________

        # bind_jlp = find_bind_pose(topology, base_bind_jgp, points, j_weights, j_offsets)
        bind_jlp = gt_bind_jlp.to(cpu)

        gt_bind_jlp = gt_bind_jlp.to(cpu)
        gt_bind_jlp.share_memory_()
        points = points.to(cpu)
        points.share_memory_()
        j_weights = j_weights.to(cpu)
        j_weights.share_memory_()
        j_offsets = j_offsets.to(cpu)
        j_offsets.share_memory_()

        # params_seq = torch.zeros(n_frames, n_joints, 3).to(cpu)
        # jgt_seq = torch.zeros(n_frames, n_joints, 4, 4).to(cpu)
        # params_seq.share_memory_()
        # jgt_seq.share_memory_()

        params_seq = np.zeros((n_frames, n_joints, 3))
        jgt_seq = np.zeros((n_frames, n_joints, 4, 4))

        task_queue = mp.Queue()
        result_queue = mp.Queue()

        start_time = time.time()
        processes = []
        n_processes = 8

        print(f' | num processes: {n_processes}')
        for rank in range(n_processes):
            p = mp.Process(
                target=worker,
                args=(
                    task_queue, result_queue,
                    topology, bind_jlp, points, j_weights, j_offsets
                )
            )
            p.start()
            processes.append(p)

        for i in range(n_frames):
            task_queue.put(i)

        for _ in range(len(processes)):
            task_queue.put(None)

        for _ in range(n_frames):
            i, params, jgt = result_queue.get()
            params_seq[i] = params
            jgt_seq[i] = jgt

        for p in processes:
            p.join()

        end_time = time.time()
        print(f'\ttime: {end_time - start_time:.1f}')

        # jgp_seq = to_cpu(jgt_seq[:, :, :3, 3])
        jgp_seq = jgt_seq[:, :, :3, 3]
        # params_seq = to_cpu(params_seq)
        bind_jlp = to_cpu(bind_jlp)
        gt_jgp = to_cpu(gt_jgp)
        gt_jlr = to_cpu(gt_jlr)

        # params_seq = R.from_euler('xyz', params_seq.reshape(-1, 3), degrees=True).as_quat().reshape(-1, n_joints, 4)
        params_seq = R.from_rotvec(params_seq.reshape(-1, 3)).as_quat().reshape(-1, n_joints, 4)

        filtered_jgp_seq = jgp_seq.copy()
        filtered_params_seq = np.empty_like(params_seq)

        for j in range(n_joints):
            if position_smoothing:
                for t_idx in range(3):
                    filtered_jgp_seq[:, j, t_idx] = savgol_filter(jgp_seq[:, j, t_idx], window_length=31,
                                                                     polyorder=3)
            if rotation_smoothing:
                for p_idx in range(4):
                    filtered_params_seq[:, j, p_idx] = savgol_filter(params_seq[:, j, p_idx], window_length=31,
                                                                     polyorder=3)

        if position_smoothing:
            jgp_seq = filtered_jgp_seq
        if rotation_smoothing:
            params_seq = filtered_params_seq

        jlr_seq = R.from_quat(params_seq.reshape(n_frames * n_joints, 4)).as_matrix().reshape(
            (n_frames, n_joints, 3, 3))

        score_manager.calc_error(
            jgp_seq,
            gt_jgp[start_frame:start_frame + n_frames],
            jlr_seq,
            gt_jlr[start_frame:start_frame + n_frames]
        )
        jpe = score_manager.memory['jpe'][-1]
        joe = score_manager.memory['joe'][-1]

        if visualize:
            viewer.visualize(
                markers_seq=points[start_frame:start_frame + n_frames],
                # joints_seq=jgt_seq[:, :, :3, 3],
                # gt_joints_seq=gt_jgp[start_frame:start_frame + n_frames],
                joints_seq=jlr_seq,
                gt_joints_seq=gt_jlr[start_frame:start_frame + n_frames],
                b_position=False,
                skeleton_template=bind_jlp,
                root_pos_seq=jgp_seq[:, 0, :],
                gt_root_pos_seq=gt_jgp[start_frame:start_frame + n_frames, 0],
                fps=60,
                jpe=jpe,
                joe=joe,
                view_local_space=True,
                vertices_seq=None,
                # j3_indices=j3_indices,
                # ja_weight=ja_weight[:, :, :n_joints],
                # ja_offset=ja_offset[:, :, :n_joints, :],
                # init_joint=init_joint
            )

        # Debug_________________________
        if pre_motion_idx != -1:
            break
        # ______________________________

    score_dir = Paths.test_results / 'evaluation_scores' / model_name
    score_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / f'{dataset}_{noise}_score.pkl'
    score_manager.save_score(score_path)


def worker(
        task_queue, result_queue,
        topology, bind_jlp, points, weights, offsets
):
    while True:
        i = task_queue.get()
        if i is None:
            break

        params, jgt, virtual_points = scipy_find_pose(
            topology, bind_jlp, points[i], weights[i], offsets[i], verbose=True, verbose_arg=f'index: {i}, '
        )
        if i % 10 == 0:
            print(f'{i}/{points.shape[0]}')
        result_queue.put((i, params[3:].reshape(-1, 3), jgt))


    #     if generate_figure_data:
    #         figure_data['marker_positions'].append(points)
    #         figure_data['joint_global_transform'].append(jgt_seq)
    #         figure_data['body_shape_index'].append(fixed_synthetic['body_type'][motion_idx])
    #         figure_data['init_vertices'].append(dataset_info['init_vertices'][fixed_synthetic['body_type'][motion_idx]])
    #         figure_data['marker_indices'].append(fixed_synthetic['marker_indices'][motion_idx])
    #         figure_data['weight'].append(j_weights)
    #         figure_data['offset'].append(j_offsets)
    #
    # if generate_figure_data:
    #     figure_dir = Paths.test_results / 'figure_data' / model_name
    #     figure_dir.mkdir(parents=True, exist_ok=True)
    #     figure_path = figure_dir / f'{file_name}_figure_{pre_motion_idx}.pkl'
    #     with open(figure_path, 'wb') as f:
    #         pickle.dump(figure_data, f)


if __name__ == '__main__':
    np.set_printoptions(precision=5, suppress=True)
    evaluate()