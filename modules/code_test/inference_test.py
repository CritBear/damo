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


def worker(
        task_queue, result_queue,
        topology, bind_jlp, points, weights, offsets
):
    while True:
        i = task_queue.get()
        if i is None:
            break

        params, jgt, virtual_points = scipy_find_pose(
            topology, bind_jlp[i], points[i], weights[i], offsets[i], verbose=True, verbose_arg=f'index: {i}, '
        )
        print(f'{i}/{points.shape[0]}')
        result_queue.put((i, params[3:].reshape(-1, 3), jgt))


def solve(points_seq, indices, weights, offsets, bind_jlp, gt_jgp, poses, topology):
    n_frames = points_seq.shape[0]
    n_joints = 24
    n_max_markers = 90

    points = points_seq[:, 3]
    gt_jlr = batch_rodrigues(poses.view(-1, n_joints, 3)[:n_frames].view(-1, 3)).view(n_frames, n_joints, 3, 3)

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

    for f in range(n_frames):
        for j in range(n_max_markers):
            if j3_indices[f][j][0] == n_joints:
                j3_indices[f][j][0] = 0
                j_weights[f][j] = 0
                j_offsets[f][j] = 0

            if j3_indices[f][j][1] == n_joints:
                j3_indices[f][j][1] = 0

            if j3_indices[f][j][2] == n_joints:
                j3_indices[f][j][2] = 0

    points = points.to(cpu)
    points.share_memory_()
    j_weights = j_weights.to(cpu)
    j_weights.share_memory_()
    j_offsets = j_offsets.to(cpu)
    j_offsets.share_memory_()
    bind_jlp = bind_jlp.to(cpu)
    bind_jlp.share_memory_()

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

    params_seq = R.from_rotvec(params_seq.reshape(-1, 3)).as_quat().reshape(-1, n_joints, 4)
    jlr_seq = R.from_quat(params_seq.reshape(n_frames * n_joints, 4)).as_matrix().reshape(
        (n_frames, n_joints, 3, 3))

    jgp_seq = jgt_seq[:, :, :3, 3]

    gt_jgp = gt_jgp.to(cpu).numpy()
    gt_jlr = gt_jlr.to(cpu).numpy()
    bind_jlp = bind_jlp.numpy()

    score_manager.calc_error(
        jgp_seq,
        gt_jgp,
        jlr_seq,
        gt_jlr
    )

    jpe = score_manager.memory['jpe'][-1]
    joe = score_manager.memory['joe'][-1]

    print(viewer.n_joints)
    print(bind_jlp.shape)

    viewer.visualize(
        markers_seq=points,
        # joints_seq=jgt_seq[:, :, :3, 3],
        # gt_joints_seq=gt_jgp[start_frame:start_frame + n_frames],
        joints_seq=jlr_seq,
        gt_joints_seq=gt_jlr,
        b_position=False,
        skeleton_template=bind_jlp[0],
        root_pos_seq=jgp_seq[:, 0, :],
        gt_root_pos_seq=gt_jgp[:, 0, :],
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




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu = torch.device('cpu')

    # model_name = 'damo_240412022202'
    model_name = 'damo_240408222808'
    # epoch = '290'
    epoch = '200'
    eval_data_date = '20240408'
    model_dir = Paths.trained_models / model_name

    model_path = model_dir / f'{model_name}_epc{epoch}.pt'
    options_path = model_dir / 'options.json'
    options = load_options_from_json_for_inference(options_path)
    options.test_dataset_names = ['SOMA']
    options.process_options()

    state_dict = torch.load(model_path)
    model = Damo(options).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = DamoDataset(
        common_dataset_path=options.common_dataset_path,
        dataset_paths=options.test_dataset_paths,
        n_max_markers=options.n_max_markers,
        seq_len=options.seq_len,
        r_ss_ds_ratio=[1, 0, 0],
        noise_jitter=True,
        noise_ghost=True,
        noise_occlusion=True,
        noise_shuffle=True,
        dist_from_skin=0.01,
        dist_augmentation=True,
        z_rot_augmentation=False,
        test=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=6,
        pin_memory=True
    )

    test_iterator = tqdm(
        enumerate(test_dataloader), total=len(test_dataloader), desc="test"
    )

    n_joints = 24
    n_max_markers = 90

    topology = test_dataset.topology

    score_manager = ScoreManager()
    viewer = Viewer(
        n_max_markers=n_max_markers,
        topology=topology,
        b_vertex=True
    )


    p = i = w = o = tposes = tgt_jgp = tbind_jlp = None

    with torch.no_grad():
        for idx, batch in test_iterator:
            points_seq = batch['points_seq'].to(device)
            points_mask = batch['points_mask'].to(device)
            indices = batch['m_j_weights'].to(device)
            weights = batch['m_j3_weights'].to(device)
            offsets = batch['m_j3_offsets'].to(device)

            poses = batch['poses'].to(device)
            gt_jgp = batch['jgp'].to(device)
            bind_jlp = batch['bind_jlp'].to(device)

            indices_pred, weights_pred, offsets_pred \
                = model(points_seq, points_mask)

            p = points_seq
            i = indices_pred
            w = weights_pred
            o = offsets_pred
            tposes = poses
            tgt_jgp = gt_jgp
            tbind_jlp = bind_jlp

            break

    solve(p, i, w, o, tbind_jlp, tgt_jgp, tposes, topology)