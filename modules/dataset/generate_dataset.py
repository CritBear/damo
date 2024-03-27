import os
import numpy as np
import random
import copy
import pickle
import gc
import torch
import c3d
from scipy.spatial.transform import Rotation as R
from datetime import datetime

from modules.solver.local_offset_solver import LocalOffsetSolver
from modules.utils.paths import Paths
from modules.utils.dfaust_utils import compute_vertex_normal
from modules.utils.functions import dict2class
from human_body_prior.body_model.body_model_prior import BodyModel

import time
import keyboard

# AMASS npz file format
# ____________________________
# trans	shape: (1377, 3)
# gender	shape: ()
# mocap_framerate	shape: ()
# betas	shape: (16,)
# dmpls	shape: (1377, 8)
# poses	shape: (1377, 156)
# ____________________________


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


def to_gpu(ndarray):
    return torch.Tensor(ndarray).to(device)


def generate_dataset():
    print(device)
    # raw_data_dirs = ['CMU', 'SFU']
    raw_data_dirs = ['CMU', 'PosePrior', 'HDM05', 'SOMA']  # 'PosePrior', 'HDM05', 'SOMA', 'DanceDB',

    topology = np.array(
        [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    )

    n_joints = 24
    n_vertices = 6890

    smplh_neutral_path = Paths.support_data / 'body_models' / 'smplh' / 'neutral' / 'model.npz'
    smplh_neutral = np.load(smplh_neutral_path)
    smplh_neutral = dict2class(smplh_neutral)

    weights = smplh_neutral.weights_prior
    j_regressor = smplh_neutral.J_regressor_prior
    faces = smplh_neutral.f

    j3_weights = np.zeros((n_vertices, 3))
    j3_indices = np.zeros_like(j3_weights, dtype=np.int32)
    j_m_idx = [[] for _ in range(n_joints)]
    for i in range(n_vertices):
        top3_indices = weights[i].argsort()[-3:][::-1]
        j3_weights[i] = weights[i][top3_indices] / np.sum(weights[i][top3_indices])
        j3_indices[i] = top3_indices
        j_m_idx[j3_indices[i][0]].append(i)

    supp_data_path = Paths.support_data / 'additional_file.npz'
    supp_data = np.load(supp_data_path)
    supp_data = dict2class(supp_data)

    caesar_bind_v = supp_data.init_vertices_list
    n_caesar_model = caesar_bind_v.shape[0]

    caesar_bind_vn = []
    v_tensor = to_gpu(caesar_bind_v)
    f_tensor = to_gpu(faces.astype(np.int32))

    caesar_bind_jgp = torch.einsum('bik,ji->bjk', [v_tensor, to_gpu(j_regressor)])
    caesar_bind_jlp = torch.zeros_like(caesar_bind_jgp)
    topology_tensor = to_gpu(topology).to(torch.int32)
    caesar_bind_jlp[:, 1:] = caesar_bind_jgp[:, 1:] - caesar_bind_jgp[:, topology_tensor[1:]]

    for i in range(n_caesar_model):
        caesar_bind_vn.append(compute_vertex_normal(v_tensor[i], f_tensor))

    j3_indices_tensor = to_gpu(j3_indices).to(torch.int32)
    caesar_j3_offsets = v_tensor[:, :, None, :] - caesar_bind_jgp[:, j3_indices_tensor, :]

    caesar_bind_vn = torch.stack(caesar_bind_vn, dim=0)
    caesar_bind_vn = to_cpu(caesar_bind_vn)
    caesar_bind_jgp = to_cpu(caesar_bind_jgp)
    caesar_bind_jlp = to_cpu(caesar_bind_jlp)
    caesar_j3_offsets = to_cpu(caesar_j3_offsets)

    with open(Paths.support_data / 'soma_superset_variation.pkl', 'rb') as f:
        soma_superset_variant = pickle.load(f)

    common_data = {
        'topology': topology,
        'weights': weights,
        'J_regressor': j_regressor,
        'n_joints': n_joints,
        'n_vertices': n_vertices,
        'caesar_bind_v': caesar_bind_v,
        'caesar_bind_vn': caesar_bind_vn,
        'caesar_bind_jgp': caesar_bind_jgp,
        'caesar_bind_jlp': caesar_bind_jlp,
        'caesar_v_j3_offsets': caesar_j3_offsets,
        'soma_superset_variant': soma_superset_variant,
        'j_v_idx': j_m_idx,
        'v_j3_indices': j3_indices,
        'v_j3_weights': j3_weights
    }
    common_data = dict2class(common_data)

    now = datetime.now()
    date = now.strftime("%Y%m%d")

    common_dataset_path = Paths.datasets / 'common' / f'damo_common_{date}.pkl'
    with open(common_dataset_path, 'wb') as f:
        pickle.dump(common_data.__dict__, f)

    for dataset_name in raw_data_dirs:
        generate_dataset_entry(dataset_name, common_data, date)


def generate_dataset_entry(dataset_name, common_data, date):
    data_dir = Paths.datasets / 'raw' / dataset_name
    c3d_files = list(data_dir.rglob('*.c3d'))
    npz_files = list(data_dir.rglob('*.npz'))

    num_betas = 10
    num_dmpls = 8
    n_joints = common_data.n_joints
    n_vertices = common_data.n_vertices
    topology = common_data.topology

    j3_indices = common_data.v_j3_indices
    j3_weights = common_data.v_j3_weights

    dataset = {
        # Each files___________________
        'file_names': [],
        'frame_idx': [],
        'n_frames': [],
        'm_v_idx': [],
        'm_j3_indices': [],
        'm_j3_weights': [],
        'm_j3_offsets': [],
        'betas': [],
        'bind_v_shaped': [],
        'bind_vn': [],
        'bind_jlp': [],
        'bind_jgp': [],
        # Each frames__________________
        'ghost_marker_mask': [],
        'markers': [],
        'poses': [],
        'jgp': [],
    }
    dataset = dict2class(dataset)

    topology = to_gpu(topology).type(torch.int32)  # [j]
    j3_indices = to_gpu(j3_indices).type(torch.int32)  # [v, 3]
    j3_weights = to_gpu(j3_weights)  # [v, 3]

    local_offset_solver = LocalOffsetSolver(
        topology=topology,
        j3_indices=j3_indices,
        j3_weights=j3_weights
    )

    # check files
    target_npz_files = []
    for file_idx, c3d_file in enumerate(c3d_files):
        target_npz_file_path = [
            c3d_file.parent / (c3d_file.stem + suffix)
            for suffix in ['_poses.npz', '_stageii.npz']
        ]

        found = False
        for i, npz_file in enumerate(npz_files):
            if npz_file in target_npz_file_path:
                target_npz_files.append(npz_file)
                found = True
                break

        assert found is True, f'Index: {file_idx} | {c3d_file}'

    for file_idx, c3d_file in enumerate(c3d_files):
        gc.collect()
        torch.cuda.empty_cache()

        print('')
        # gc.collect()
        # torch.cuda.empty_cache()
        # cur_vram_memory = torch.cuda.memory_allocated()
        # max_vram_memory = torch.cuda.max_memory_allocated()
        # print(f'VRAM memory: {cur_vram_memory / (1024 ** 3):.2f} GB / {max_vram_memory / (1024 ** 3):.2f} GB')

        # assert npz_file.stem == c3d_file.stem + '_poses' or npz_file.stem == c3d_file.stem + '_stageii',\
        #     f'Index: {file_idx} | {npz_file.name}, {c3d_file.name}'

        print(f'dataset name: {dataset_name} | [{file_idx+1}/{len(c3d_files)}] | file name: {c3d_file.stem}')

        npz_file = target_npz_files[file_idx]
        amass = np.load(npz_file, allow_pickle=True)
        amass = dict2class(amass)

        smplh_model_path = Paths.support_data / 'body_models' / 'smplh' / amass.gender / 'model.npz'
        dmpl_model_path = Paths.support_data / 'body_models' / 'dmpls' / amass.gender / 'model.npz'

        bm = BodyModel(
            bm_fname=smplh_model_path.as_posix(),
            num_betas=num_betas,
            num_dmpls=num_dmpls,
            dmpl_fname=dmpl_model_path.as_posix(),
            model_type='smpl'
        ).to(device)

        n_frames = len(amass.trans)

        bp = {
            'root_orient': to_gpu(amass.poses[:, :3]),
            'pose_body': to_gpu(amass.poses[:, 3:66]),
            'pose_hand': to_gpu(amass.poses[:, 66:]),
            'trans': to_gpu(amass.trans),
            'betas': to_gpu(np.repeat(amass.betas[:num_betas][np.newaxis], repeats=n_frames, axis=0))
        }

        body_key_list = ['pose_body', 'betas', 'trans', 'root_orient']
        if hasattr(amass, 'dmpls'):
            bp['dmpls'] = to_gpu(amass.dmpls[:, :num_dmpls])
            body_key_list.append('dmpls')

        motion = bm(**{k: v for k, v in bp.items() if k in body_key_list})

        bind_vn = compute_vertex_normal(motion.v_shaped, motion.f)  # [v, 3]

        vgp = motion.v
        faces = motion.f
        bind_vgp = motion.v_shaped  # [v, 3]
        bind_jgp = motion.J  # [j, 3]
        bind_jlp = torch.zeros_like(bind_jgp)
        bind_jlp[1:] = bind_jgp[1:] - bind_jgp[topology[1:]]  # [j, 3]
        jgp = motion.Jtr  # [f, j, 3]
        poses = motion.full_pose  # [f, j*3]

        # Read c3d file________________________________
        markers = read_c3d_markers(c3d_file)  # [f, m ,3]

        if dataset_name in ['SFU', 'PosePrior']:
            markers = markers[:, :, [0, 2, 1]]
            markers[:, :, 1] *= -1

        markers = to_gpu(markers) / 1000
        assert n_frames == markers.shape[0],\
            f'c3d frames: {markers.shape[0]}, {c3d_file}\nnpz frames: {n_frames}, {npz_file}'

        n_markers = markers.shape[1]
        print(f'num frames: {n_frames} | num markers: {n_markers}')

        batch_size = 1000
        n_batches = n_frames // batch_size

        # test(
        #     topology=topology,
        #     jgp=jgp,
        #     bind_jgp=bind_jgp,
        #     markers=markers,
        # )

        def frame_mini_batch(fsi, fei):
            print(f'batch: {fsi}-{fei} / {n_frames}', end='')

            dataset.file_names.append(c3d_file.stem)
            dataset.frame_idx.append([fsi, fei])

            generate_dataset_entry_batch(
                dataset=dataset,
                n_frames=fei-fsi,
                markers=markers[fsi:fei],
                betas=amass.betas,
                faces=faces,
                vgp=vgp[fsi:fei],
                bind_vgp=bind_vgp,
                bind_vn=bind_vn,
                bind_jgp=bind_jgp,
                bind_jlp=bind_jlp,
                j3_indices=j3_indices,
                j3_weights=j3_weights,
                local_offset_solver=local_offset_solver,
                jgp=jgp[fsi:fei],
                poses=poses[fsi:fei]
            )

        for i in range(n_batches):
            frame_start_idx = i * batch_size
            frame_end_idx = frame_start_idx + batch_size
            frame_mini_batch(frame_start_idx, frame_end_idx)

        extra_frames = n_frames % batch_size
        if extra_frames > 600:
            frame_start_idx = n_batches * batch_size
            frame_end_idx = n_frames
            frame_mini_batch(frame_start_idx, frame_end_idx)



    # Split train test dataset
    train_ratio = 0.7
    file_len = len(dataset.n_frames)
    split_idx = int(file_len * train_ratio)
    all_idx = list(range(file_len))
    train_idx = set(random.sample(all_idx, split_idx))
    test_idx = set(all_idx) - train_idx

    train_dataset = {k: [] for k in vars(dataset)}
    test_dataset = {k: [] for k in vars(dataset)}

    for key, values in vars(dataset).items():
        train_dataset[key] = [value for i, value in enumerate(values) if i in train_idx]
        test_dataset[key] = [value for i, value in enumerate(values) if i in test_idx]

    train_dataset_path = Paths.datasets / 'train' / f'damo_train_{date}_{dataset_name}.pkl'
    test_dataset_path = Paths.datasets / 'test' / f'damo_test_{date}_{dataset_name}.pkl'

    with open(train_dataset_path, 'wb') as f:
        pickle.dump(train_dataset, f)

    with open(test_dataset_path, 'wb') as f:
        pickle.dump(test_dataset, f)


def generate_dataset_entry_batch(
        dataset,
        n_frames,
        markers,
        betas,
        faces,
        vgp,
        bind_vgp,
        bind_vn,
        bind_jgp,
        bind_jlp,
        j3_indices,
        j3_weights,
        local_offset_solver,
        jgp,
        poses
):
    # print('')
    # _________________________________
    # Get vertex normals, valid markers

    vn = []  # vertex normals
    n_vm_list = []  # number of valid markers
    vm_list = []  # valid markers
    mag = torch.norm(markers, dim=-1)  # [f, m]
    for i in range(n_frames):
        vn.append(compute_vertex_normal(vgp[i], faces))

        vm_f_idx = torch.nonzero(mag[i] > 0.001).squeeze()
        vm_f = markers[i][vm_f_idx]
        vm_list.append(vm_f)
        n_vm_list.append(vm_f.shape[0])

    vn = torch.stack(vn, dim=0)  # [f, v, 3]
    n_mvm = max(n_vm_list)  # number of max valid markers
    vm = torch.zeros(n_frames, n_mvm, 3).to(markers.device)  # [f, vm, 3]
    for i in range(n_frames):
        vm[i, :n_vm_list[i]] = vm_list[i]

    del mag
    torch.cuda.empty_cache()
    gc.collect()

    if n_mvm > 90:
        print('')
        return

    print(f' | vm {n_mvm}')

    # ________________________
    # Select matching vertices

    vec_m_v = vm[:, :, None, :] - vgp[:, None, :, :]  # [f, vm, v, 3]
    norm_m_v = torch.linalg.norm(vec_m_v, dim=-1)  # [f, vm, v]
    norm_vn = torch.linalg.norm(vn)  # [f, v]

    dist_m_closest_v = torch.min(norm_m_v, dim=-1).values  # [f, vm]
    ghost_marker_mask = torch.where(dist_m_closest_v > 0.15, torch.tensor(0), torch.tensor(1))

    dot_mv_vn = torch.sum(vec_m_v * vn[:, None, :, :], dim=-1)

    del vec_m_v
    torch.cuda.empty_cache()
    gc.collect()

    cos_sim = dot_mv_vn / (norm_m_v * norm_vn)
    angle_diff = torch.acos(torch.clamp(cos_sim, -1, 1))  # [f, vm, v]

    angle_diff = 0.1 * (1 - torch.cos(angle_diff)) / 2  # 0 ~ 180 -> 0 ~ 0.1, non-linear scaling
    selection_criteria = norm_m_v + (angle_diff / 100)  # [f, vm, v]
    m_v_idx = torch.argmin(selection_criteria, dim=-1)  # [f, vm]

    del norm_vn, dot_mv_vn, cos_sim, angle_diff, selection_criteria, dist_m_closest_v
    torch.cuda.empty_cache()
    gc.collect()

    disp_length = norm_m_v[torch.arange(n_frames)[:, None], torch.arange(n_mvm), m_v_idx]  # [f, vm]
    disp_bind_mgp = bind_vgp[m_v_idx] + bind_vn[m_v_idx] * disp_length[:, :, None]  # [f, vm, 3]

    init_m_j3_offsets = disp_bind_mgp[:, :, None, :] - bind_jgp[None, None, :, :]  # [f, vm, j, 3]

    m_j3_indices = j3_indices[m_v_idx]  # [f, vm, j3]
    m_j3_weights = j3_weights[m_v_idx]  # [f, vm, j3]

    init_m_j3_offsets = init_m_j3_offsets[
        torch.arange(n_frames)[:, None, None], torch.arange(n_mvm)[:, None], m_j3_indices
    ]  # [f, vm, j3, 3]

    for i in range(n_frames):
        m_j3_weights[i, n_vm_list[i]:, :] = 0
        init_m_j3_offsets[i, n_vm_list[i]:] = 0
        ghost_marker_mask[i, n_vm_list[i]:] = 0

    local_offset_solver.set_transform(poses, jgp)

    params = torch.zeros(n_frames, n_mvm, 3, 3).to(vm.device)
    virtual_markers = torch.zeros(n_frames, n_mvm, 3).to(vm.device)
    for i in range(n_frames):
        print(f'\r\tsolving... [{i+1}/{n_frames}]', end='')
        params_f, virtual_markers_f = local_offset_solver.find_local_offsets(
            i=i,
            markers=vm[i, :n_vm_list[i]],
            j3_indices=m_j3_indices[i, :n_vm_list[i]],
            j3_weights=m_j3_weights[i, :n_vm_list[i]],
            init_params=init_m_j3_offsets[i, :n_vm_list[i]]
        )
        params[i, :n_vm_list[i], :, :] = params_f.view(-1, 3, 3)
        virtual_markers[i, :n_vm_list[i], :] = virtual_markers_f.view(-1, 3)

    virtual_vm = local_offset_solver.batch_lbs(params, m_j3_indices, m_j3_weights)
    error = torch.linalg.norm(virtual_vm - vm, dim=-1)
    mask = error != 0
    mean_error = torch.masked_select(error, mask).mean()
    print(f'\n\tbatch error: {mean_error}')

    # local_offset_solver.set_transform(poses, jgp)
    #
    # params = torch.zeros(n_frames, n_mvm, 3, 3).to(vm.device)
    # virtual_markers = torch.zeros(n_frames, n_mvm, 3).to(vm.device)
    # for i in range(n_frames):
    #     print(f'\r\tsolving... [{i+1}/{n_frames}]', end='')
    #
    #     m_j3_weights[i, n_vm_list[i]:, :] = 0
    #     params_f, virtual_markers_f = local_offset_solver.find_marker_local_offsets(
    #         i=i,
    #         markers=vm[i, :n_vm_list[i]],
    #         j3_indices=m_j3_indices[i, :n_vm_list[i]],
    #         j3_weights=m_j3_weights[i, :n_vm_list[i]],
    #         init_params=init_marker_local_offsets[i, :n_vm_list[i]]
    #     )
    #     params[i, :n_vm_list[i], :, :] = params_f.view(-1, 3, 3)
    #     virtual_markers[i, :n_vm_list[i], :] = virtual_markers_f.view(-1, 3)

    # topology = np.array(
    #     [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    # )
    # test(
    #     topology=topology,
    #     jgp=jgp,
    #     bind_jgp=bind_jgp,
    #     markers=markers,
    #     virtual_markers=virtual_markers,
    #     vgp_m=vgp[torch.arange(n_frames)[:, None], m_v_idx],
    #     vgp=vgp,
    #     vn=vn[torch.arange(n_frames)[:, None], m_v_idx]
    # )

    dataset.n_frames.append(n_frames)
    dataset.betas.append(betas)
    dataset.m_v_idx.append(to_cpu(m_v_idx))
    dataset.m_j3_indices.append(to_cpu(m_j3_indices))
    dataset.m_j3_weights.append(to_cpu(m_j3_weights))
    dataset.m_j3_offsets.append(to_cpu(init_m_j3_offsets))
    dataset.bind_v_shaped.append(to_cpu(bind_vgp))
    dataset.bind_vn.append(to_cpu(bind_vn))
    dataset.bind_jlp.append(to_cpu(bind_jlp))
    dataset.bind_jgp.append(to_cpu(bind_jgp))
    dataset.ghost_marker_mask.append(to_cpu(ghost_marker_mask))
    dataset.markers.append(to_cpu(vm))
    dataset.poses.append(to_cpu(poses))
    dataset.jgp.append(to_cpu(jgp))


def generate_dataset_entry_batch_legacy(
        dataset,
        n_frames,
        markers,
        betas,
        vgp,
        bind_vgp,
        bind_vn,
        bind_jgp,
        bind_jlp,
        j3_indices,
        j3_weights,
        local_offset_solver,
        jgp,
        poses
):
    # Select closest vertex indices
    markers_temp = markers.clone()
    markers_temp = torch.where((markers_temp > -0.001) & (markers_temp < 0.001), torch.tensor(9999), markers_temp)

    n_markers = markers.shape[1]
    marker_batch_size = 50
    n_marker_batches = n_markers // marker_batch_size

    params_batch = []
    virtual_markers_batch = []
    m_v_idx_batch = []
    ghost_marker_mask_batch = []

    def marker_mini_batch(msi, mei):
        print(f'\n\tmarker batch: {msi}-{mei} / {n_markers}', end='')

        vec_markers_verts = markers_temp[:, msi:mei, None, :] - vgp[:, None, :, :]  # [f, m, v, 3]
        dist_markers_verts = torch.sqrt(torch.sum(vec_markers_verts ** 2, dim=-1))  # [f, m, v]

        del vec_markers_verts
        torch.cuda.empty_cache()
        gc.collect()

        dist_markers_verts[dist_markers_verts > 1000] = 9999

        dist_markers_closest_verts = torch.min(dist_markers_verts, dim=-1).values  # [f, m]

        ghost_marker_mask = torch.where(dist_markers_closest_verts > 0.1, torch.tensor(0), torch.tensor(1))

        filtered_dist_markers_verts = torch.where(
            dist_markers_verts > 1000,
            torch.tensor(float('nan')),
            dist_markers_verts
        )

        dist_avg = torch.nanmean(filtered_dist_markers_verts, dim=0)  # [m, v]
        m_v_idx = torch.argmin(dist_avg, dim=-1)

        del dist_markers_verts
        del filtered_dist_markers_verts
        torch.cuda.empty_cache()
        gc.collect()

        m_idx = torch.arange(dist_avg.shape[0], device=device)
        disp_bind_mgp = bind_vgp[m_v_idx, :] + bind_vn[m_v_idx, :] * dist_avg[m_idx, m_v_idx][:, None]  # [m, 3]

        init_marker_local_offset = disp_bind_mgp[:, None, :] - bind_jgp[None, :, :]  # [m, j, 3]
        init_marker_local_offset = init_marker_local_offset[m_idx[:, None], j3_indices[m_v_idx], :]  # [m, j3, 3]

        init_marker_local_offset[torch.isnan(init_marker_local_offset)] = 0

        params, virtual_markers = local_offset_solver.find_markers_local_offsets(
            markers=markers[:, msi:mei, :],
            m_v_idx=m_v_idx,
            jgp=jgp,
            poses=poses,
            ghost_marker_mask=ghost_marker_mask,
            init_params=init_marker_local_offset
        )
        params_batch.append(params.view(mei - msi, 3, 3))
        virtual_markers_batch.append(virtual_markers)
        m_v_idx_batch.append(m_v_idx)
        ghost_marker_mask_batch.append(ghost_marker_mask)

    for i in range(n_marker_batches):
        marker_start_idx = i * marker_batch_size
        marker_end_idx = marker_start_idx + marker_batch_size
        marker_mini_batch(marker_start_idx, marker_end_idx)

    extra_markers = n_markers % marker_batch_size
    if extra_markers > 0:
        marker_start_idx = n_marker_batches * marker_batch_size
        marker_end_idx = n_markers
        marker_mini_batch(marker_start_idx, marker_end_idx)

    j3_offsets = torch.cat(params_batch, dim=0)
    virtual_markers = torch.cat(virtual_markers_batch, dim=1)
    m_v_idx = torch.cat(m_v_idx_batch, dim=0)
    ghost_marker_mask = torch.cat(ghost_marker_mask_batch, dim=1)

    dataset.n_frames.append(n_frames)
    dataset.betas.append(betas)
    dataset.m_v_idx.append(to_cpu(m_v_idx))
    dataset.m_j3_indices.append(to_cpu(j3_indices[m_v_idx]))
    dataset.m_j3_weights.append(to_cpu(j3_weights[m_v_idx]))
    dataset.m_j3_offsets.append(to_cpu(j3_offsets))
    dataset.bind_v_shaped.append(to_cpu(bind_vgp))
    dataset.bind_vn.append(to_cpu(bind_vn))
    dataset.bind_jlp.append(to_cpu(bind_jlp))
    dataset.bind_jgp.append(to_cpu(bind_jgp))
    dataset.ghost_marker_mask.append(to_cpu(ghost_marker_mask))
    dataset.markers.append(to_cpu(markers))
    dataset.poses.append(to_cpu(poses))
    dataset.jgp.append(to_cpu(jgp))

    masked_markers = markers * ghost_marker_mask.unsqueeze(-1)
    print('solving error: ', end='')
    print(torch.mean(torch.sqrt(torch.sum(torch.square(masked_markers - virtual_markers), dim=-1))).item())  # No optim: 0.0788 <-> Yes optim: 6.2435e-05



def test(**kwargs):
    import vpython.no_notebook
    from vpython import canvas, sphere, vector, color, cylinder, arrow, text

    kwargs = dict2class(kwargs)
    '''
    topology=topology,
    jgp=jgp,
    bind_jgp=bind_jgp,
    markers=markers,
    virtual_markers=virtual_markers
    '''

    # topology = to_cpu(kwargs.topology)
    topology = kwargs.topology
    n_joints = len(topology)
    jgp = to_cpu(kwargs.jgp)
    bind_jgp = to_cpu(kwargs.bind_jgp)
    markers = to_cpu(kwargs.markers)

    window = canvas(x=0, y=0, width=1200, height=1200, center=vector(0, 0, 0), background=vector(0, 0, 0))
    axis_x = arrow(pos=vector(0, 0, 0), axis=vector(1, 0, 0), shaftwidth=0.05, color=vpython.color.red)
    axis_y = arrow(pos=vector(0, 0, 0), axis=vector(0, 1, 0), shaftwidth=0.05, color=vpython.color.green)
    axis_z = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 1), shaftwidth=0.05, color=vpython.color.blue)

    window.forward = vector(-1, 0, 0)
    window.up = vector(0, 0, 1)

    n_max_markers = markers.shape[1]
    n_frames = markers.shape[0]

    v_markers = [
        sphere(radius=0.01, color=color.cyan)
        for i in range(n_max_markers)
    ]

    v_vmarkers = [
        sphere(radius=0.01, color=color.blue)
        for i in range(n_max_markers)
    ]

    v_joints = [
        sphere(pos=vector(*bind_jgp[i]), radius=0.02, color=color.orange)
        for i in range(n_joints)
    ]
    v_bones = [
        cylinder(pos=v_joints[i].pos, axis=(v_joints[topology[i]].pos - v_joints[i].pos), radius=0.01,
                 color=color.orange)
        for i in range(1, n_joints)
    ]

    # v_weights = [
    #     cylinder(radius=0.003, color=color.white)
    #     for i in range(n_vertices * 3)
    # ]

    n_verts = kwargs.vgp.shape[1]
    v_verts = [
        sphere(radius=0.005, color=color.white)
        for _ in range(n_verts)
    ]
    v_norms = [
        arrow(shaftwidth=0.005, color=color.magenta)
        for _ in range(n_max_markers)
    ]

    print('render')

    f = 0
    while True:

        # time.sleep(1/60)
        # continue

        for i in range(n_joints):
            v_joints[i].pos = vector(*jgp[f, i])

        for i in range(1, n_joints):
            v_bones[i - 1].pos = v_joints[i].pos
            v_bones[i - 1].axis = (v_joints[topology[i]].pos - v_joints[i].pos)

        for i in range(n_max_markers):
            v_markers[i].pos = vector(*markers[f, i])
            v_vmarkers[i].pos = vector(*kwargs.virtual_markers[f, i])

            # v_norms[i].pos = v_verts[markers_verts_idx[i]].pos
            # v_norms[i].axis = vector(*vert_norms[f, markers_verts_idx[i]]) * 0.1

        # for i in range(n_vertices):
        # for j in range(3):
        #     v_weights[i * 3 + j].pos = vector(*verts[f, i])
        #     v_weights[i * 3 + j].axis = v_joints[j3_indices[i, j]].pos - v_weights[i * 3 + j].pos
        #     v_weights[i * 3 + j].color = vector(1, 1, 1) * j3_weights[i, j]
        #     v_weights[i * 3 + j].visible = True if j3_weights[i, j] > 0.5 else False

            v_norms[i].pos = vector(*kwargs.vgp_m[f, i])
            v_norms[i].axis = vector(*kwargs.vn[f, i]) * 0.1

        for i in range(n_verts):
            v_verts[i].pos = vector(*kwargs.vgp[f, i])

        # if i in markers_verts_idx:
        #     v_verts[i].color = color.magenta
        # else:
        #     v_verts[i].color = color.white

        if keyboard.is_pressed('right'):
            f = (f + 1) % n_frames

        if keyboard.is_pressed("left"):
            f = (f - 1) % n_frames

        # f += 1
        time.sleep(1 / 60)

def read_c3d_markers(file_path):
    reader = c3d.Reader(open(file_path, 'rb'))
    marker_data = []

    for i, points, analog in reader.read_frames():
        marker_data.append(points)

    marker_data = np.array(marker_data)

    return marker_data[:, :, :3]


if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=False)
    np.set_printoptions(precision=4, suppress=True)
    generate_dataset()