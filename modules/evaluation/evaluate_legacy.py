import sys
import numpy as np
import pickle
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

from modules.utils.paths import Paths
from modules.evaluation.score_manager import ScoreManager
from modules.solver.quat_pose_solver_numpy import PoseSolver
from modules.utils.viewer.vpython_viewer import VpythonViewer as Viewer


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

    model_name = 'damo_240330034650_epc100'

    for dataset, b_dataset in eval_dataset.items():
        if not b_dataset:
            continue

        for noise, b_noise in eval_noise.items():
            if not b_noise:
                continue

            eval_data_path = Paths.datasets / 'evaluate' / 'legacy' / f'eval_data_damo-soma_{dataset}_{noise}.pkl'
            with open(eval_data_path, 'rb') as f:
                eval_data = pickle.load(f)

            body_shape_indices = eval_data['body_shape_index'] if noise != 'real' else None

            test_data_path = Paths.test_results / 'model_outputs' / model_name / f'model_output_{eval_data_path.stem}.pkl'  # _model_output
            evaluate_entry(test_data_path, model_name, body_shape_indices)


def evaluate_entry(test_data_path, model_name, body_shape_indices):
    visualize = True
    generate_figure_data = False
    position_smoothing = False
    rotation_smoothing = False
    pre_motion_idx = 0
    pre_n_frames = 400

    file_name = test_data_path.stem

    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    topology = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])
    n_max_markers = 90
    n_joints = 24

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
            'weights': [],  # [ (n_frames, n_max_markers, n_joints + 1) ]
            'offsets': [],  # [ (n_frames, n_max_markers, n_joints + 1, 3) ]
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
        points = motion['points']  # (n_frames, n_max_markers, 3)
        indices = motion['indices']  # (n_frames, n_max_markers, n_joints + 1)
        j3_weight = motion['weights']  # (n_frames, n_max_markers, 3)
        j3_offset = motion['offsets']  # (n_frames, n_max_markers, 3, 3)

        n_frames = points.shape[0]

        gt_jgp = motion['jgp'][0][:n_frames]  # (n_frames, n_joints, 3)
        # gt_jgr = clip['joint_global_rotations'][0][:n_frames]  # (n_frames, n_joints, 3, 3)
        # gt_jlp = motion['joint_local_positions'][0]# [-n_joints:]  # (n_joints, 3)
        # print(gt_jlp.shape)
        # return
        gt_jlr = motion['jlr'][0][:n_frames]  # (n_frames, n_joints, 3, 3)

        j3_indices = np.argsort(indices)[:, :, -1:-4:-1]
        ja_weight = np.zeros((n_frames, n_max_markers, n_joints + 1))
        ja_offset = np.zeros((n_frames, n_max_markers, n_joints + 1, 3))

        ja_weight[
            np.arange(n_frames)[:, np.newaxis, np.newaxis],
            np.arange(n_max_markers)[:, np.newaxis],
            j3_indices.astype(np.int32)
        ] = j3_weight

        ja_offset[
            np.arange(n_frames)[:, np.newaxis, np.newaxis],
            np.arange(n_max_markers)[:, np.newaxis],
            j3_indices.astype(np.int32)
        ] = j3_offset

        solver = PoseSolver(topology=topology, gt_skeleton_template=None, max_iter=1)  # gt_jlp
        if solver.skeleton_template is None:
            skeleton_template = solver.build_skeleton_template(points=points, weight=ja_weight, offset=ja_offset)

        jgt_seq = []
        params_seq = []

        # Debug__________________________________
        start_frame = 0
        if pre_n_frames > 0:
            n_frames = min(n_frames, pre_n_frames)
        # _______________________________________

        prev_params = None
        for f_idx in tqdm(range(start_frame, start_frame + n_frames), file=sys.stdout):
            p = points[f_idx]
            w = ja_weight[f_idx, :, :n_joints]
            o = ja_offset[f_idx, :, :n_joints, :]
            params, jgt = solver.solve(p, w, o, init_params=prev_params)
            prev_params = params
            jgt_seq.append(jgt[np.newaxis, :, :, :])

            # params = np.zeros(3 + n_joints * 4)
            # params[6::4] = 1

            params_seq.append(params[3:].reshape(-1, 4)[np.newaxis, :, :].copy())

        jgt_seq = np.vstack(jgt_seq)  # (n_frames, n_joints, 4, 4)
        params_seq = np.vstack(params_seq)  # (n_frames, n_joints, 4)

        filtered_jgt_seq = jgt_seq.copy()
        filtered_params_seq = np.empty_like(params_seq)

        for j in range(n_joints):
            if position_smoothing:
                for t_idx in range(3):
                    filtered_jgt_seq[:, j, t_idx, 3] = savgol_filter(jgt_seq[:, j, t_idx, 3], window_length=31 if n_frames > 30 else 5, polyorder=2)
            if rotation_smoothing:
                for p_idx in range(4):
                    filtered_params_seq[:, j, p_idx] = savgol_filter(params_seq[:, j, p_idx], window_length=31, polyorder=2)

        if position_smoothing:
            jgt_seq = filtered_jgt_seq
        if rotation_smoothing:
            params_seq = filtered_params_seq

        jlr_seq = R.from_quat(params_seq.reshape(n_frames * n_joints, 4)).as_matrix().reshape((n_frames, n_joints, 3, 3))

        print(jlr_seq.shape)

        # init_joint = dataset_info['init_joint_global_position'][body_shape_idx]
        # vertex_template = dataset_info['init_vertices'][body_shape_idx]  # (6890, 3)
        # vertex_joint_offset = vertex_template[:, np.newaxis, :] - init_joint[np.newaxis, :, :]  # (6890, 22, 3)
        #
        # vertex_global = jgt_seq[:, np.newaxis, :, :3, :3] @ vertex_joint_offset[np.newaxis, :, :, :, np.newaxis]
        # # (F, M, J, 3, 3) @ (F, M, J, 3, 1) -> (F, M, J, 3, 1)
        # vertex_global = vertex_global.squeeze()  # (F, M, J, 3)
        # vertex_global += jgt_seq[:, np.newaxis, :, :3, 3]
        # vertex_global = np.sum(vertex_global * vertex_joint_weight[:, :, np.newaxis], axis=2)

        score_manager.calc_error(
            jgt_seq[:, :22, :3, 3],
            gt_jgp[start_frame:start_frame + n_frames, :22],
            jlr_seq[:, :22],
            gt_jlr[start_frame:start_frame + n_frames, :22]
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
            figure_data['weight'].append(ja_weight)
            figure_data['offset'].append(ja_offset)

        # jlr_seq[:, 0, :, :] = np.identity(3)
        # gt_jlr[:, 0, :, :] = np.identity(3)

        if visualize:
            viewer.visualize(
                markers_seq=points[start_frame:start_frame + n_frames],
                # joints_seq=jgt_seq[:, :, :3, 3],
                # gt_joints_seq=gt_jgp[start_frame:start_frame + n_frames],
                joints_seq=jlr_seq,
                gt_joints_seq=gt_jlr[start_frame:start_frame+n_frames],
                b_position=False,
                skeleton_template=skeleton_template,
                root_pos_seq=jgt_seq[:, 0, :3, 3],
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