import sys

import numpy as np
import torch
import torch.nn.functional as F
import pickle
import c3d
import random
import copy
from datetime import datetime
from tqdm import tqdm

from modules.solver.bind_pose_marker_solver import find_bind_mgp
from modules.utils.paths import Paths
from modules.utils.functions import dict2class, read_c3d_markers
from modules.utils.dfaust_utils import compute_vertex_normal
from human_body_prior.body_model.body_model_prior import BodyModel
from human_body_prior.body_model.lbs import batch_rodrigues


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


def to_gpu(ndarray):
    return torch.Tensor(ndarray).to(device)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_eval_dataset():
    # set_random_seed(0)

    raw_data_dirs = [
        "HDM05",
        "SFU",
        "MoSh",
        "SOMA"
    ]

    for dataset_name in raw_data_dirs:
        generate_eval_dataset_entry(dataset_name)


def generate_eval_dataset_entry(dataset_name):
    data_dir = Paths.datasets / 'eval' / 'raw' / dataset_name
    c3d_files = list(data_dir.rglob('*.c3d'))
    npz_files = list(data_dir.rglob('*.npz'))

    common_data_path = Paths.datasets / 'common' / 'damo_common_20240329.pkl'
    with open(common_data_path, 'rb') as f:
        common_data = pickle.load(f)
    common_data = dict2class(common_data)

    num_betas = 10
    num_dmpls = 8
    topology = common_data.topology
    j3_indices = common_data.v_j3_indices
    j3_weights = common_data.v_j3_weights

    eval_data = {
        'topology': common_data.topology,
        'J_regressor': common_data.J_regressor,
        'weights': common_data.weights,
        'file_names': [],  #
        'n_frames': [],  #

        'legacy_j': [],  #
        'legacy_jg': [],  #
        'legacy_jgo': [],  #
        'legacy_jgos': [],  #
        'legacy_real': [],  #

        'j': [],  #
        'jg': [],  #
        'jgo': [],  #
        'jgos': [],  #
        'real': [],  #
        'betas': [],  #
        'synthetic_bind_jlp': [],  #
        'synthetic_bind_jgp': [],  #
        'real_bind_jlp': [],  #
        'real_bind_jgp': [],  #
        'synthetic_bind_v': [],  #
        'real_bind_v': [],  #
        'poses': [],  #
        'synthetic_jgp': [],  #
        'real_jgp': []  #
    }

    target_npz_files = []
    for file_idx, c3d_file in enumerate(c3d_files):
        target_npz_file_path = [
            f'{c3d_file.parts[-2]}/{(c3d_file.stem + suffix)}'
            for suffix in ['_poses', '_stageii', '_C3D_poses']
        ]

        found = False
        for i, npz_file in enumerate(npz_files):
            if f'{npz_file.parts[-2]}/{(npz_file.stem)}' in target_npz_file_path:
                target_npz_files.append(npz_file)
                found = True
                break

        assert found is True, f'Index: {file_idx} | {c3d_file}'

    for file_idx, c3d_file in enumerate(c3d_files):
        print(f'\ndataset name: {dataset_name} | [{file_idx + 1}/{len(c3d_files)}] | file name: {c3d_file.stem}')

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
        bind_vgp = motion.v_shaped  # [v, 3]
        bind_jgp = motion.J  # [j, 3]
        bind_jlp = torch.zeros_like(bind_jgp)
        bind_jlp[1:] = bind_jgp[1:] - bind_jgp[topology[1:]]  # [j, 3]
        jgp = motion.Jtr  # [f, j, 3]
        poses = motion.full_pose  # [f, j*3]

        eval_data['n_frames'].append(n_frames)
        eval_data['file_names'].append(c3d_file.stem)
        eval_data['real_bind_v'].append(to_cpu(bind_vgp))
        eval_data['real_bind_jgp'].append(to_cpu(bind_jgp))
        eval_data['real_bind_jlp'].append(to_cpu(bind_jlp))
        eval_data['real_jgp'].append(to_cpu(jgp))
        eval_data['poses'].append(to_cpu(poses))
        eval_data['betas'].append(amass.betas[:num_betas])

        real_marker = read_c3d_markers(c3d_file)
        real_marker /= 1000
        if dataset_name in ['SFU', 'PosePrior']:
            real_marker = real_marker[:, :, [0, 2, 1]]
            real_marker[:, :, 1] *= -1

        eval_data['real'].append(real_marker)

        # generate_synthetic_data(common_data, eval_data, poses, jgp[:, 0])

    # load_legacy_eval_data(eval_data, dataset_name)

    now = datetime.now()
    date = now.strftime("%Y%m%d")

    save_dir = Paths.datasets / 'eval'
    save_path = save_dir / f'damo_eval_{dataset_name}_{date}.pkl'

    file_count = len(c3d_files)
    # for k, v in eval_data.items():
    #     if isinstance(v, list):
    #         assert len(v) == file_count, f'{k}: {len(v)}'

    with open(save_path, 'wb') as f:
        pickle.dump(eval_data, f)


def generate_synthetic_data(common_data, eval_data, poses_ref, trans_ref):
    if torch.rand(1) < 0.5:
        use_superset = True
    else:
        use_superset = False

    superset_variant = common_data.soma_superset_variant
    n_joints = common_data.n_joints
    n_max_markers = 90

    if use_superset:
        n_selected_superset = np.random.randint(22, len(superset_variant))
        selected_superset_indices = np.random.choice(
            np.arange(len(superset_variant)),
            n_selected_superset,
            replace=False
        )
        marker_indices = np.empty(n_selected_superset, dtype=int)
        for i in range(n_selected_superset):
            variation = superset_variant[selected_superset_indices[i]]
            random_idx = np.random.randint(0, len(variation))
            marker_indices[i] = variation[random_idx]
    else:
        marker_indices = []
        values = [2, 3, 4]
        prob = [0.7, 0.2, 0.1]
        indices_num_list = np.random.choice(values, size=n_joints, p=prob)
        for i in range(n_joints):
            marker_indices.extend(np.random.choice(
                common_data.j_v_idx[i],
                size=indices_num_list[i],
                replace=False
            ))
        marker_indices = np.array(marker_indices)

    n_body = common_data.caesar_bind_v.shape[0]
    body_idx = random.randint(0, n_body - 1)

    bind_jlp = to_gpu(common_data.caesar_bind_jlp[body_idx]).to(torch.float32)
    bind_jgp = to_gpu(common_data.caesar_bind_jgp[body_idx]).to(torch.float32)
    poses = to_gpu(poses_ref).to(torch.float32)
    jlr = batch_rodrigues(poses.view(-1, n_joints, 3).view(-1, 3)).view(-1, n_joints, 3, 3)
    n_frames = jlr.shape[0]
    jgt = torch.eye(4).repeat(n_frames, n_joints, 1, 1).to(device)
    jgt[:, :, :3, :3] = jlr
    jgt[:, :, :3, 3] = bind_jlp[None, :, :]  # Error fix
    jgt[:, 0, :3, 3] = trans_ref

    for i, pi in enumerate(common_data.topology):
        if i == 0:
            continue
        jgt[:, i] = jgt[:, pi] @ jgt[:, i]
    jgp = jgt[:, :, :3, 3]
    jgr = jgt[:, :, :3, :3]

    eval_data['synthetic_bind_jlp'].append(to_cpu(bind_jlp))
    eval_data['synthetic_bind_jgp'].append(to_cpu(bind_jgp))
    eval_data['synthetic_bind_v'].append(common_data.caesar_bind_v[body_idx])
    eval_data['synthetic_jgp'].append(to_cpu(jgp))

    for noise in ['j', 'jg', 'jgo', 'jgos']:
        points_seq = []
        points_mask = []
        for f in tqdm(range(n_frames), file=sys.stdout, desc=noise):
            aug_range = 0.005
            aug_offset = (torch.rand(1).to(device) - 0.5) * aug_range
            dist_from_skin = 0.01 + aug_offset

            temp_marker_indices = copy.deepcopy(marker_indices)

            if 'o' in noise:
                n_occlusion = random.randint(0, 5)
                occlusion_indices = np.random.choice(len(temp_marker_indices), size=n_occlusion, replace=False)
                temp_marker_indices = np.delete(temp_marker_indices, occlusion_indices)
            temp_marker_indices = torch.from_numpy(temp_marker_indices).to(torch.int32).to(device)
            bind_m = torch.from_numpy(common_data.caesar_bind_v[body_idx]).to(device)[temp_marker_indices].to(torch.float32)
            bind_mn = torch.from_numpy(common_data.caesar_bind_vn[body_idx]).to(device)[temp_marker_indices].to(torch.float32)
            bind_disp_m = bind_m + bind_mn * dist_from_skin  # [m, 3]

            if 'j' in noise:
                jitter_range = 0.01
                jitter = (torch.rand_like(bind_disp_m) - 0.5) * jitter_range
                bind_disp_m += jitter

            m_j3_indices = torch.from_numpy(common_data.v_j3_indices).to(device)[temp_marker_indices, :].to(torch.int32)
            m_j3_weights = torch.from_numpy(common_data.v_j3_weights).to(device)[temp_marker_indices, :].to(torch.float32)
            m_j3_offsets = bind_disp_m[:, None, :] - bind_jgp[m_j3_indices]

            j3gr = jgr[f, m_j3_indices, :, :]
            j3gp = jgp[f, m_j3_indices, :]

            points = j3gr @ m_j3_offsets[:, :, :, None]  # [m, 3, 3, 1]
            points = torch.squeeze(points)  # [m, 3, 3]
            points += j3gp  # [m, 3, 3]
            points = torch.sum(points * m_j3_weights[:, :, None], dim=1)  # [m, 3]
            n_clean_markers = len(temp_marker_indices)

            if 'g' in noise:
                max_ghost_markers = 2
                n_ghost_markers = torch.randint(-max_ghost_markers, max_ghost_markers + 1, (1,)).item()
                n_ghost_markers = max(0, n_ghost_markers)
                points_center = points.mean(dim=0)
                ghost_markers_range = 1
                ghost_markers = (torch.rand(n_ghost_markers, points.shape[-1]).to(device) - 0.5) * ghost_markers_range
                ghost_markers = ghost_markers + points_center
                points = torch.cat((points, ghost_markers), dim=0)

                n_total_markers = n_clean_markers + n_ghost_markers
            else:
                n_total_markers = n_clean_markers

            if n_total_markers > n_max_markers:
                points = points[:n_max_markers]
                n_total_markers = n_max_markers

            points = F.pad(
                input=points,
                pad=[0, 0, 0, n_max_markers - n_total_markers],
                mode='constant',
                value=0
            )

            ghost_weights = torch.ones((n_max_markers, 1))
            ghost_weights[:n_clean_markers, 0] = 0

            if 's' in noise:
                shuffled_indices = torch.randperm(n_total_markers)
                points[:n_total_markers, :] = points[shuffled_indices, :]

            points_seq.append(points.unsqueeze(0))
            mask = torch.zeros(1, n_max_markers, requires_grad=False)
            mask[:, :n_total_markers] = 1
            points_mask.append(mask)

        points_seq = torch.cat(points_seq, dim=0)
        points_mask = torch.cat(points_mask, dim=0)
        eval_data[f'{noise}'].append(to_cpu(points_seq))


def load_legacy_eval_data(eval_data, dataset_name):
    for noise in ['j', 'jg', 'jgo', 'jgos', 'real']:
        legacy_file_name = f'eval_data_damo-soma_{dataset_name}_{noise}.pkl'
        legacy_file_path = Paths.datasets / 'eval' / 'legacy' / legacy_file_name

        with open(legacy_file_path, 'rb') as f:
            legacy_data = pickle.load(f)

        legacy_markers = legacy_data['real_marker'] if noise == 'real' else legacy_data['synthetic_marker']
        legacy_m_idx = 0

        for m_idx in range(len(eval_data['n_frames'])):
            res_n_frames = eval_data['n_frames'][m_idx]

            n_max_markers = 0
            for markers in legacy_markers:
                if markers.shape[1] > n_max_markers:
                    n_max_markers = markers.shape[0]

            markers = np.zeros((res_n_frames, n_max_markers, 3))
            f_idx = 0

            while res_n_frames > 0:
                n_part_frames, n_part_markers, _ = legacy_markers[legacy_m_idx].shape

                markers[f_idx:f_idx+n_part_frames, :n_part_markers] = legacy_markers[legacy_m_idx]

                f_idx += n_part_frames
                res_n_frames -= n_part_frames
                assert res_n_frames >= 0
                legacy_m_idx += 1

            if noise == 'real':
                eval_data['legacy_real'].append(markers)
            else:
                eval_data[f'legacy_{noise}'].append(markers)



if __name__ == '__main__':
    torch.set_printoptions(precision=8, sci_mode=False)
    np.set_printoptions(precision=8, suppress=True)
    generate_eval_dataset()