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
        'SFU': True,
        'Mosh': True,
        'SOMA': True
    }

    eval_noise = {
        'j': False,
        'jg': False,
        'jgo': False,
        'jgos': False,
        'real': True
    }

    model_name = 'damo_240404153143_epc120'
    eval_data_date = '20240408'
    test_data_dir = Paths.test_results / 'model_outputs'

    for dataset, do_dataset in eval_dataset.items():
        if not do_dataset:
            continue

        eval_data_path = Paths.test_results / 'model_outputs' / model_name
        eval_data_path = eval_data_path / f'model_output_damo_eval_{dataset}_{eval_data_date}.pkl'

        with open(eval_data_path, 'rb') as f:
            eval_data = pickle.load(f)

        for noise, do_noise in eval_noise.items():
            if not do_noise:
                continue

            evaluate_entry(eval_data, noise, model_name)


def evaluate_entry(eval_data, noise, model_name):
    visualize = True
    generate_figure_data = False
    position_smoothing = False
    rotation_smoothing = False
    pre_motion_idx = 0
    pre_n_frames = 400

    topology = eval_data['topology']
    n_joints = topology.shape[0]
    n_max_markers = 90

    score_manager = ScoreManager()
    viewer = Viewer(
        n_max_markers=n_max_markers,
        topology=topology,
        b_vertex=True
    ) if visualize else None

    if noise == 'real':
        points_list = eval_data['real']
        indices_list = eval_data['real_indices_pred']
        weights_list = eval_data['real_weights_pred']
        offsets_list = eval_data['real_offsets_pred']
        gt_jgp_list = eval_data['real_jgp']
        gt_bind_jgp_list = eval_data['real_bind_jgp']
        gt_bind_jlp_list = eval_data['real_bind_jlp']
    else:
        points_list = eval_data[f'synthetic_{noise}']
        indices_list = eval_data[f'synthetic_{noise}_indices_pred']
        weights_list = eval_data[f'synthetic_{noise}_weights_pred']
        offsets_list = eval_data[f'synthetic_{noise}_offsets_pred']
        gt_jgp_list = eval_data['synthetic_jgp']
        gt_bind_jgp_list = eval_data['synthetic_bind_jgp']
        gt_bind_jlp_list = eval_data['synthetic_bind_jlp']

    n_motions = len(indices_list)
    for motion_idx in range(n_motions):
        if pre_motion_idx != -1:
            motion_idx = pre_motion_idx

        print(f"Solving... motion clip:[{motion_idx + 1}/{n_motions}]")

        points = to_gpu(points_list[motion_idx])
        indices = to_gpu(indices_list[motion_idx])
        weights = to_gpu(weights_list[motion_idx])
        offsets = to_gpu(offsets_list[motion_idx])

        n_frames = indices.shape[0]

        point_pad = torch.zeros(n_frames, n_max_markers, 3).to(device)
        point_pad[:, :points.shape[1]] = points[:n_frames]
        points = point_pad

        gt_jgp = to_gpu(gt_jgp_list[motion_idx][:n_frames])
        poses = to_gpu(eval_data['poses'][motion_idx])
        gt_jlr = batch_rodrigues(poses.view(-1, n_joints, 3)[:n_frames].view(-1, 3)).view(n_frames, n_joints, 3, 3)
        gt_bind_jgp = to_gpu(gt_bind_jgp_list[motion_idx])
        gt_bind_jlp = to_gpu(gt_bind_jlp_list[motion_idx])

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

        # Debug__________________________________
        start_frame = 0
        if pre_n_frames > 0:
            n_frames = min(n_frames, pre_n_frames)
        # _______________________________________

        gt_bind_jlp = gt_bind_jlp.to(cpu)
        gt_bind_jlp.share_memory_()
        points = points.to(cpu)
        points.share_memory_()
        j_weights = j_weights.to(cpu)
        j_weights.share_memory_()
        j_offsets = j_offsets.to(cpu)
        j_offsets.share_memory_()

        params_seq = torch.zeros(n_frames, n_joints, 3).to(cpu)
        jgt_seq = torch.zeros(n_frames, n_joints, 4, 4).to(cpu)
        params_seq.share_memory_()
        jgt_seq.share_memory_()

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
                    topology, gt_bind_jlp, points, j_weights, j_offsets
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

        jgt_seq = to_cpu(jgt_seq)
        params_seq = to_cpu(params_seq)
        bind_jlp = to_cpu(gt_bind_jlp)
        gt_jgp = to_cpu(gt_jgp)
        gt_jlr = to_cpu(gt_jlr)

        filtered_jgt_seq = jgt_seq.copy()
        filtered_params_seq = np.empty_like(params_seq)

        for j in range(n_joints):
            if position_smoothing:
                for t_idx in range(3):
                    filtered_jgt_seq[:, j, t_idx, 3] = savgol_filter(jgt_seq[:, j, t_idx, 3],
                                                                     window_length=31 if n_frames > 30 else 5,
                                                                     polyorder=2)
            if rotation_smoothing:
                for p_idx in range(3):
                    filtered_params_seq[:, j, p_idx] = savgol_filter(params_seq[:, j, p_idx], window_length=31,
                                                                     polyorder=2)

        if position_smoothing:
            jgt_seq = filtered_jgt_seq
        if rotation_smoothing:
            params_seq = filtered_params_seq

        jlr_seq = R.from_rotvec(params_seq.reshape(n_frames * n_joints, 3)).as_matrix().reshape(
            (n_frames, n_joints, 3, 3))

        score_manager.calc_error(
            jgt_seq[:, :, :3, 3],
            gt_jgp[start_frame:start_frame + n_frames],
            jlr_seq,
            gt_jlr[start_frame:start_frame + n_frames]
        )

        if visualize:
            viewer.visualize(
                markers_seq=points[start_frame:start_frame + n_frames],
                # joints_seq=jgt_seq[:, :, :3, 3],
                # gt_joints_seq=gt_jgp[start_frame:start_frame + n_frames],
                joints_seq=jlr_seq,
                gt_joints_seq=gt_jlr[start_frame:start_frame + n_frames],
                b_position=False,
                skeleton_template=bind_jlp,
                root_pos_seq=jgt_seq[:, 0, :3, 3],
                gt_root_pos_seq=gt_jgp[start_frame:start_frame + n_frames, 0],
                fps=60,
                # jpe=jpe,
                # joe=joe,
                view_local_space=False,
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


def worker(
        task_queue, result_queue,
        topology, bind_jlp, points, weights, offsets
):
    while True:
        i = task_queue.get()
        if i is None:
            break

        params, jgt, virtual_points = find_pose(
            topology, bind_jlp, points[i], weights[i], offsets[i]
        )
        print(i)
        result_queue.put((i, params[3:].view(-1, 3).clone(), jgt.clone()))

def evaluate_entry_legacy(test_data_path, model_name, body_shape_indices):
    visualize = True
    generate_figure_data = False
    position_smoothing = False
    rotation_smoothing = False
    pre_motion_idx = 0
    pre_n_frames = 400
    file_name = test_data_path.stem

    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    topology = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
    n_max_markers = 90
    n_joints = 22

    score_manager = ScoreManager()
    viewer = Viewer(
        n_max_markers=n_max_markers,
        topology=topology,
        b_vertex=True
    ) if visualize else None

    with open(Paths.support_data / 'dataset_gen_info.pkl', 'rb') as f:
        dataset_info = pickle.load(f)

    vertex_joint_weight = dataset_info['vertex_joint_weights']  # (6890, 20)

    if generate_figure_data:
        assert pre_motion_idx != -1
        # with open(Paths.support_data / 'dataset_gen_info.pkl', 'rb') as f:
        #     dataset_info = pickle.load(f)

        with open(Paths.support_data / 'fixed_synthetic_generation.pkl', 'rb') as f:
            fixed_synthetic = pickle.load(f)

        figure_data = {
            'marker_positions': [],  # [ (n_frames, n_max_markers, 3) ]
            'body_shape_index': [],  # [ int ]
            'init_vertices': [],  # [ (n_vertices, 3) ]
            'joint_global_transform': [],  # [ (n_frames, n_joints, 4, 4) ]
            'smooth_joint_global_transform': [],  # [ (n_frames, n_joints, 4, 4) ]
            'marker_indices': [],  # [ [n_markers] ]
            'weight': [],  # [ (n_frames, n_max_markers, n_joints + 1) ]
            'offset': [],  # [ (n_frames, n_max_markers, n_joints + 1, 3) ]
        }

    for motion_idx in range(len(test_data)):
        if body_shape_indices is None:
            body_shape_idx = 0
        else:
            body_shape_idx = body_shape_indices[motion_idx]

        # Debug_________________________
        if pre_motion_idx != -1:
            motion_idx = pre_motion_idx
        # ______________________________

        print(f"Solving... motion clip:[{motion_idx + 1}/{len(test_data)}]")

        motion = test_data[motion_idx]
        points = to_gpu(motion['points'])  # (n_frames, n_max_markers, 3)
        indices = to_gpu(motion['indices'])  # (n_frames, n_max_markers, n_joints + 1)
        j3_weights = to_gpu(motion['weights'])  # (n_frames, n_max_markers, 3)
        j3_offsets = to_gpu(motion['offsets'])  # (n_frames, n_max_markers, 3, 3)

        n_frames = points.shape[0]

        gt_jgp = to_gpu(motion['jgp'][0][:n_frames])  # (n_frames, n_joints, 3)
        # gt_jgr = clip['joint_global_rotations'][0][:n_frames]  # (n_frames, n_joints, 3, 3)
        # gt_jlp = motion['joint_local_positions'][0]# [-n_joints:]  # (n_joints, 3)
        gt_jlr = to_gpu(motion['jlr'][0][:n_frames])  # (n_frames, n_joints, 3, 3)
        gt_jgp = to_cpu(gt_jgp)
        gt_jlr = to_cpu(gt_jlr)

        j3_indices = torch.argsort(indices, dim=-1, descending=True)[..., :3]  # [f, m, 3]
        j_weights = torch.zeros(n_frames, n_max_markers, n_joints + 3).to(device)  # [f, m, j+1]
        j_offsets = torch.zeros(n_frames, n_max_markers, n_joints + 3, 3).to(device)  # [f, m, j+1, 3]

        j_weights[
            torch.arange(n_frames)[:, None, None],
            torch.arange(n_max_markers)[:, None],
            j3_indices
        ] = j3_weights
        j_weights = j_weights[:, :, :n_joints]  # [f, m, j]

        j_offsets[
            torch.arange(n_frames)[:, None, None],
            torch.arange(n_max_markers)[:, None],
            j3_indices
        ] = j3_offsets
        j_offsets = j_offsets[:, :, :n_joints]  # [f, m, j, 3]

        # solver = PoseSolver(topology=topology, gt_skeleton_template=None, max_iter=1)  # gt_jlp

        # if solver.skeleton_template is None:
        #     skeleton_template = solver.build_skeleton_template(points=points.cpu().numpy(), weight=j_weights.cpu().numpy(), offset=j_offsets.cpu().numpy())
        # bind_jlp = to_gpu(skeleton_template)

        smplh_neutral_path = Paths.support_data / 'body_models' / 'smplh' / 'neutral' / 'model.npz'
        smplh_neutral = np.load(smplh_neutral_path)
        smplh_neutral = dict2class(smplh_neutral)
        base_bind_jgp = smplh_neutral.J

        base_bind_jgp = to_gpu(base_bind_jgp)

        bind_jlp = find_bind_pose(topology, base_bind_jgp[:n_joints], points, j_weights, j_offsets)

        jgt_seq = []
        params_seq = []

        # Debug__________________________________
        start_frame = 0
        if pre_n_frames > 0:
            n_frames = min(n_frames, pre_n_frames)
        # _______________________________________


        prev_params = None
        for f_idx in tqdm(range(start_frame, start_frame + n_frames), file=sys.stdout):
            params, jgt, virtual_points = find_pose(
                topology, bind_jlp, points[f_idx], j_weights[f_idx], j_offsets[f_idx]
            )

            prev_params = params
            jgt_seq.append(jgt[None, :, :, :].clone())
            params_seq.append(params[3:].reshape(-1, 3)[None, :, :].clone())

        jgt_seq = torch.cat(jgt_seq, dim=0)  # (n_frames, n_joints, 4, 4)
        params_seq = torch.cat(params_seq, dim=0)  # (n_frames, n_joints, 4)

        jgt_seq = to_cpu(jgt_seq)
        params_seq = to_cpu(params_seq)
        bind_jlp = to_cpu(bind_jlp)

        filtered_jgt_seq = jgt_seq.copy()
        filtered_params_seq = np.empty_like(params_seq)

        for j in range(n_joints):
            if position_smoothing:
                for t_idx in range(3):
                    filtered_jgt_seq[:, j, t_idx, 3] = savgol_filter(jgt_seq[:, j, t_idx, 3],
                                                                     window_length=31 if n_frames > 30 else 5,
                                                                     polyorder=2)
            if rotation_smoothing:
                for p_idx in range(3):
                    filtered_params_seq[:, j, p_idx] = savgol_filter(params_seq[:, j, p_idx], window_length=31,
                                                                     polyorder=2)

        if position_smoothing:
            jgt_seq = filtered_jgt_seq
        if rotation_smoothing:
            params_seq = filtered_params_seq

        jlr_seq = R.from_rotvec(params_seq.reshape(n_frames * n_joints, 3)).as_matrix().reshape(
            (n_frames, n_joints, 3, 3))

        print(jlr_seq.shape)

        init_joint = dataset_info['init_joint_global_position'][body_shape_idx]
        vertex_template = dataset_info['init_vertices'][body_shape_idx]  # (6890, 3)
        vertex_joint_offset = vertex_template[:, np.newaxis, :] - init_joint[np.newaxis, :, :]  # (6890, 22, 3)

        vertex_global = jgt_seq[:, np.newaxis, :, :3, :3] @ vertex_joint_offset[np.newaxis, :, :, :, np.newaxis]
        # (F, M, J, 3, 3) @ (F, M, J, 3, 1) -> (F, M, J, 3, 1)
        vertex_global = vertex_global.squeeze()  # (F, M, J, 3)
        vertex_global += jgt_seq[:, np.newaxis, :, :3, 3]
        vertex_global = np.sum(vertex_global * vertex_joint_weight[:, :, np.newaxis], axis=2)

        score_manager.calc_error(
            jgt_seq[:, :, :3, 3],
            gt_jgp[start_frame:start_frame + n_frames],
            jlr_seq,
            gt_jlr[start_frame:start_frame + n_frames]
        )

        jpe = score_manager.memory['jpe'][-1]
        joe = score_manager.memory['joe'][-1]

        print(joe.shape)

        if generate_figure_data:
            figure_data['marker_positions'].append(points)
            figure_data['joint_global_transform'].append(jgt_seq)
            figure_data['body_shape_index'].append(fixed_synthetic['body_type'][motion_idx])
            figure_data['init_vertices'].append(dataset_info['init_vertices'][fixed_synthetic['body_type'][motion_idx]])
            figure_data['marker_indices'].append(fixed_synthetic['marker_indices'][motion_idx])
            figure_data['weight'].append(j_weights)
            figure_data['offset'].append(j_offsets)

        # jlr_seq[:, 0, :, :] = np.identity(3)
        # gt_jlr[:, 0, :, :] = np.identity(3)

        if visualize:
            viewer.visualize(
                markers_seq=points[start_frame:start_frame + n_frames],
                # joints_seq=jgt_seq[:, :, :3, 3],
                # gt_joints_seq=gt_jgp[start_frame:start_frame + n_frames],
                joints_seq=jlr_seq,
                gt_joints_seq=gt_jlr[start_frame:start_frame + n_frames],
                b_position=False,
                skeleton_template=bind_jlp,
                root_pos_seq=jgt_seq[:, 0, :3, 3],
                gt_root_pos_seq=gt_jgp[start_frame:start_frame + n_frames, 0],
                fps=60,
                # jpe=jpe,
                # joe=joe,
                view_local_space=False,
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

    print(file_name)
    score_dir = Paths.test_results / 'evaluation_scores' / model_name
    score_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / f'{file_name}_score.pkl'
    score_manager.save_score(score_path)

    if generate_figure_data:
        figure_dir = Paths.test_results / 'figure_data' / model_name
        figure_dir.mkdir(parents=True, exist_ok=True)
        figure_path = figure_dir / f'{file_name}_figure_{pre_motion_idx}.pkl'
        with open(figure_path, 'wb') as f:
            pickle.dump(figure_data, f)


if __name__ == '__main__':
    np.set_printoptions(precision=5, suppress=True)
    evaluate()