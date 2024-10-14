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

from multiprocessing import Process, Queue


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            angle_between((1, 0, 0), (1, 0, 0))
            0.0
            angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


jpi = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
front_vector = np.array([1, 0, 0])


def get_joe_from_joint_global_positions(joint_positions_1, joint_positions_2):
    '''
    Args:
        joint_positions_1 (numpy.ndarray): Numpy array (n_joints[22], 3[xyz])
        joint_positions_2 (numpy.ndarray): Numpy array (n_joints[22], 3[xyz])

    Returns:
        numpy.ndarray: Numpy array containing the Joe values for each joint. (deg) 22개. 0번째는 0.
    '''
    result_joint_joes = [0]
    for jdx in range(1, len(jpi)):
        parent_idx = jpi[jdx]
        if parent_idx == 0:
            child_bone_center_1 = joint_positions_1[jdx] - joint_positions_1[parent_idx]
            child_bone_center_1 /= np.linalg.norm(child_bone_center_1)
            child_bone_center_2 = joint_positions_2[jdx] - joint_positions_2[parent_idx]
            child_bone_center_2 /= np.linalg.norm(child_bone_center_2)
        else:
            parent_bone_1 = joint_positions_1[parent_idx] - joint_positions_1[jpi[parent_idx]]
            parent_bone_1 /= np.linalg.norm(parent_bone_1)
            child_bone_1 = joint_positions_1[jdx] - joint_positions_1[parent_idx]
            child_bone_1 /= np.linalg.norm(child_bone_1)

            parent_rotation_mat_1 = rotation_matrix_from_vectors(parent_bone_1, front_vector)
            child_bone_center_1 = np.dot(parent_rotation_mat_1, child_bone_1)

            parent_bone_2 = joint_positions_2[parent_idx] - joint_positions_2[jpi[parent_idx]]
            parent_bone_2 /= np.linalg.norm(parent_bone_2)
            child_bone_2 = joint_positions_2[jdx] - joint_positions_2[parent_idx]
            child_bone_2 /= np.linalg.norm(child_bone_2)

            parent_rotation_mat_2 = rotation_matrix_from_vectors(parent_bone_2, front_vector)
            child_bone_center_2 = np.dot(parent_rotation_mat_2, child_bone_2)

        result_joint_joes.append(angle_between(child_bone_center_1, child_bone_center_2))
    return np.rad2deg(np.array(result_joint_joes))


def joe_worker(shared_queue, shared_result):
    while True:
        while not shared_queue.empty():
            joint_position_1, joint_position_2 = shared_queue.get()
            result = get_joe_from_joint_global_positions(joint_position_1, joint_position_2)
            shared_result.put(result)
        time.sleep(0.01)


def get_mean_joe_from_frames(joint_positions_frames_1,
                             joint_positions_frames_2,
                             num_processes=10, x_90_rotate_to_frame_1=False,
                             joe_process_list=None, shared_queue=None, shared_result=None):
    """
    joint_positions_frames: (n_frames, n_joints[22까지만 씀 ], 3[xyz])
    두 parameters의 shape은 무조건 동일해야 함.
    joint_positions_frames_1을 90도 돌려서 맞추려면 x_90_rotate_to_frame_1 = True
    Return: Mean JOE scalar 반환. (deg)
    """
    theta = np.pi / 2  # -90도 회전에 해당하는 라디안 값
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])

    if isinstance(joint_positions_frames_1, list):
        joint_positions_frames_1 = np.array(joint_positions_frames_1)
    if isinstance(joint_positions_frames_2, list):
        joint_positions_frames_2 = np.array(joint_positions_frames_2)

    # print(joint_positions_frames_1.shape[0])
    # print(joint_positions_frames_2.shape[0])
    assert joint_positions_frames_1.shape[0] == joint_positions_frames_2.shape[0]
    assert joint_positions_frames_1.shape[1] == joint_positions_frames_2.shape[1] == 22
    assert joint_positions_frames_1.shape[2] == joint_positions_frames_2.shape[2] == 3
    n_frames = joint_positions_frames_1.shape[0]
    if x_90_rotate_to_frame_1:
        joint_positions_frames_1 = np.dot(joint_positions_frames_1, R_x)

    st = time.time()
    joe_results = []

    return_joe_process_list = False
    if joe_process_list is None:
        joe_process_list = []
        shared_queue = Queue()
        shared_result = Queue()
    else:
        if shared_queue is None:
            shared_queue = Queue()
        if shared_result is None:
            shared_result = Queue()
        return_joe_process_list = True
        for pdx in range(0, len(joe_process_list) - num_processes):
            joe_process_list[pdx].kill()
        joe_process_list = joe_process_list[len(joe_process_list) - num_processes:]

    for fdx in range(n_frames):
        shared_queue.put((joint_positions_frames_1[fdx], joint_positions_frames_2[fdx]))
    for pdx in range(0, num_processes - len(joe_process_list)):
        process = Process(target=joe_worker,
                          args=(shared_queue, shared_result))
        joe_process_list.append(process)
        process.start()

    while len(joe_results) < n_frames:
        joe_results.append(shared_result.get())
        # print(
        #     f"\r| JOE Calculating... {len(joe_results)}/{n_frames} |"
        #     f" {len(joe_results) / (time.time() - st):.2f} iter/sec |"
        #     f" {(n_frames - len(joe_results)) / (len(joe_results) / (time.time() - st)):.2f} sec left", end="     ")

    result = np.array(joe_results).mean(axis=1)
    return (result, joe_process_list, shared_queue, shared_result) if return_joe_process_list else result


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
        'j': True,
        'jg': True,
        'jgo': True,
        'jgos': True,
        'real': False
    }

    model_name = 'damo_240412022202'
    epoch = '100'

    for dataset, do_dataset in eval_dataset.items():
        if not do_dataset:
            continue

        for noise, do_noise in eval_noise.items():
            if not do_noise:
                continue

            eval_data_dir = Paths.test_results / 'model_outputs' / f'{model_name}_epc{epoch}_lsynthetic'
            eval_data_path = eval_data_dir / f'{model_name}_epc{epoch}_{dataset}_{noise}.pkl'

            print(f'{dataset} | {noise}')
            with open(eval_data_path, 'rb') as f:
                eval_data = pickle.load(f)

            evaluate_entry(eval_data, dataset, noise, model_name, epoch)


def evaluate_entry(eval_data, dataset, noise, model_name, epoch):
    visualize = True
    generate_figure_data = True
    position_smoothing = True
    rotation_smoothing = True
    pre_motion_idx = 8
    pre_n_frames = None
    # pre_n_frames = [0, 50]

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

    if generate_figure_data:
        visualization_data_path = Paths.test_results / 'figure_data' / f'{model_name}_epc{epoch}_lsynthetic_{dataset}_{noise}_{pre_motion_idx}.pkl'
        visualization_data = {
            'marker_positions': [],  # [ (n_frames, n_max_markers, 3) ]
            'body_shape_index': [],  # [ int ]
            'init_vertices': [],  # [ (n_vertices, 3) ]
            'joint_global_transform': [],  # [ (n_frames, n_joints, 4, 4) ]
            'smooth_joint_global_transform': [],  # [ (n_frames, n_joints, 4, 4) ]
            'marker_indices': [],  # [ [n_markers] ]
            'weight': [],  # [ (n_frames, n_max_markers, n_joints + 1) ]
            'offset': [],  # [ (n_frames, n_max_markers, n_joints + 1, 3) ]
        }

    n_motions = len(eval_data['indices'])
    for motion_idx in range(pre_motion_idx, n_motions):
        # if pre_motion_idx != -1:
        #     motion_idx = pre_motion_idx

        print(f"Solving... motion clip:[{motion_idx + 1}/{n_motions}]")

        seq_len = eval_data['points_seq'][motion_idx].shape[1]

        points = to_gpu(eval_data['points_seq'][motion_idx][:, seq_len//2, :, :])
        indices = to_gpu(eval_data['indices'][motion_idx])
        weights = to_gpu(eval_data['weights'][motion_idx])
        offsets = to_gpu(eval_data['offsets'][motion_idx])

        n_frames = indices.shape[0]

        # print('_________________')
        # print(points.shape)
        # print(indices.shape)

        points = points[:n_frames]

        gt_jgp = np.zeros((n_frames, n_joints, 3))
        gt_jgp[:, :22] = eval_data['jgp'][motion_idx][:n_frames]
        gt_jgp = to_gpu(gt_jgp)
        # poses = to_gpu(eval_data['poses'][motion_idx])

        gt_jlr = np.zeros((n_frames, n_joints, 3, 3))
        gt_jlr[:, :, 0, 0] = 1
        gt_jlr[:, :, 1, 1] = 1
        gt_jlr[:, :, 2, 2] = 1
        gt_jlr[:, :22] = eval_data['jlr'][motion_idx][:n_frames]
        gt_jlr = to_gpu(gt_jlr)
        # gt_jlr = to_gpu(eval_data['jlr'][motion_idx][:n_frames])
        # gt_jlr = batch_rodrigues(poses.view(-1, n_joints, 3)[:n_frames].view(-1, 3)).view(n_frames, n_joints, 3, 3)
        # gt_bind_jgp = to_gpu(eval_data['bind_jgp'][motion_idx])

        gt_bind_jlp = np.zeros((n_joints, 3))
        gt_bind_jlp[:22] = eval_data['bind_jlp'][motion_idx]
        gt_bind_jlp = to_gpu(gt_bind_jlp)
        # gt_bind_jlp = to_gpu(eval_data['bind_jlp'][motion_idx])

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
        if pre_n_frames is not None:
            start_frame = int(n_frames * pre_n_frames[0])
            end_frame = min(n_frames, start_frame + pre_n_frames[1])
            n_frames = end_frame - start_frame
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
                    task_queue, result_queue, start_frame,
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

        num_processes = 10
        joe_process_list = []
        shared_queue = None
        shared_result = None

        joe_result, joe_process_list, shared_queue, shared_result = \
            get_mean_joe_from_frames(gt_jgp[start_frame:start_frame + n_frames, :22],
                                     jgp_seq[:, :22],
                                     num_processes=num_processes,
                                     joe_process_list=joe_process_list,
                                     shared_queue=shared_queue,
                                     shared_result=shared_result)

        score_manager.calc_error_use_outer(
            jgp_seq,
            gt_jgp[start_frame:start_frame + n_frames],
            joe_result=joe_result
        )

        # score_manager.calc_error(
        #     jgp_seq,
        #     gt_jgp[start_frame:start_frame + n_frames],
        #     jlr_seq,
        #     gt_jlr[start_frame:start_frame + n_frames]
        # )
        jpe = score_manager.memory['jpe'][-1]
        joe = score_manager.memory['joe'][-1]

        if generate_figure_data:
            visualization_data['marker_positions'].append(points)
            visualization_data['joint_global_transform'].append(jgt_seq)
            visualization_data['body_shape_index'].append(eval_data['body_idx'][motion_idx])
            visualization_data['init_vertices'].append(common_data['caesar_bind_v'][eval_data['body_idx'][motion_idx]])
            visualization_data['weight'].append(j_weights)
            visualization_data['offset'].append(j_offsets)

            with open(visualization_data_path, 'wb') as f:
                pickle.dump(visualization_data, f)

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

    score_dir = Paths.test_results / 'evaluation_scores' / f'{model_name}_epc{epoch}_synthetic'
    score_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / f'{dataset}_{noise}_score.pkl'
    score_manager.save_score(score_path)


def worker(
        task_queue, result_queue, si,
        topology, bind_jlp, points, weights, offsets
):
    while True:
        i = task_queue.get()
        if i is None:
            break

        params, jgt, virtual_points = scipy_find_pose(
            topology, bind_jlp, points[i+si], weights[i+si], offsets[i+si], verbose=True, verbose_arg=f'index: {i}, '
        )
        if i % 100 == 0:
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