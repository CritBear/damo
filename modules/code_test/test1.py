import numpy as np
import pickle
from modules.utils.paths import Paths


# with open(Paths.datasets / 'eval' / 'eval_data_for_damo' / 'eval_data_damo-soma_HDM05_jgos.pkl', 'rb') as f:
#     d = pickle.load(f)

# d = np.load(Paths.datasets / 'raw/SFU/SFU/0005/0005_2FeetJump001_poses.npz')
# d = np.load(Paths.support_data / 'smpl_default_base_pose.npz')
# d = np.load(Paths.support_data / 'body_models' / 'smplh' / 'male' / 'model.npz')
# J_regressor_prior	shape: (24, 6890)
# f	shape: (13776, 3)
# J_regressor	shape: (52, 6890)
# kintree_table	shape: (2, 52)
# J	shape: (24, 3)
# weights_prior	shape: (6890, 24)
# weights	shape: (6890, 52)
# posedirs	shape: (6890, 3, 459)
# bs_style	shape: ()
# v_template	shape: (6890, 3)
# shapedirs	shape: (6890, 3, 16)
# bs_type	shape: ()


# vert_norms = []
        # for f in range(n_frames):
        #     vert_norms.append(compute_vertex_normal(motion.v[f], motion.f))
        # vert_norms = torch.stack(vert_norms, dim=0)  # [2751, 6890, 3]

# print(motion.v.shape)  # [2751, 6890, 3]
# print(motion.f.shape)  # [13776, 3]
# print(motion.Jtr.shape)  # [2751, 24, 3]
# print(motion.full_pose.shape)  # [2751, 72]
# print(motion.v_shaped.shape)  # [6890, 3]
# print(motion.J.shape)  # [24, 3]

# filtered_dist_closest_verts = torch.where(
#             dist_markers_closest_verts > 1000,
#             torch.tensor(float('nan')).to(device),
#             dist_markers_closest_verts
#         )
#         dist_avg_closest_verts = torch.nanmean(filtered_dist_closest_verts, dim=0)  # [m]
#         nan_indices = torch.where(torch.isnan(dist_avg_closest_verts))[0]
#         far_indices = torch.where(dist_avg_closest_verts > 0.1)[0]
#         ghost_marker_indices = torch.cat((nan_indices, far_indices)).unique()

# d_path = Paths.datasets / 'Stefanos_1os_antrikos_karsilamas_C3D_poses.npz'
d_path = r'D:\Projects\DAMO\damo\datasets\raw\SOMA\soma_subject1\clap_001_stageii.npz'
d = np.load(d_path, allow_pickle=True)

# with open(d_path, 'rb') as f:
#     d = pickle.load(f)

for i in d:
    print(i, end='\t')
    if hasattr(d[i], 'shape'):
        print('shape: ' + str(d[i].shape))
    elif hasattr(d[i], '__len__'):
        print('len: ' + str(len(d[i])), end='')
        if hasattr(d[i][0], 'shape'):
            print(f', shape: {d[i][0].shape}', end='')
        print('')
    else:
        print(d[i])


# # Compute distance (markers <-> vertex normals)
#         markers_tensor = torch.Tensor(markers).to(device)
#         vec_markers_verts = markers_tensor[:, :, None, :] - body.v[:, None, :, :]  # [f, m, v, 3]
#         dot_vec = torch.sum(vec_markers_verts * vert_norms[:, None, :, :], dim=-1)  # [f, m, v]
#         projection = body.v[:, None, :, :] + dot_vec[..., None] * vert_norms[:, None, :, :]  # [f, m, v, 3]
#         print(projection.shape)
#         dist_markers_vert_norms = (markers_tensor[:, :, None, :] - projection).norm(dim=-1)  # [f, m, v]
#         markers_verts_idx = torch.argmin(dist_markers_vert_norms, dim=-1)
#
#         # dist_markers_vert_norms = torch.abs(dot_vec)
#         # # Select vertex index of min distance
#         # dist_avg_markers_vert_norms = torch.mean(dist_markers_vert_norms, dim=0)
#         # print(dist_avg_markers_vert_norms.shape)
#         # markers_verts_idx = torch.argmin(dist_markers_vert_norms, dim=-1)
#
#         # print(markers_verts_idx.shape)  # [2751, 42]


# memory_usage = {}
            #
            # # 현재 로컬 환경의 모든 변수를 순회하면서, 텐서인 경우 그 메모리 사용량을 계산합니다.
            # for var_name, var in locals().items():
            #     if torch.is_tensor(var) and var.device.type == 'cuda':
            #         memory_usage[var_name] = var.element_size() * var.nelement() / (1024 ** 2)  # MB 단위로 변환
            #
            # # GPU에서 사용중인 전체 메모리
            # total_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            #
            # # 메모리 사용량을 내림차순으로 정렬합니다.
            # sorted_memory_usage = sorted(memory_usage.items(), key=lambda item: item[1], reverse=True)
            #
            # # 메모리 사용량을 퍼센트로 변환하여 출력합니다.
            # count = 0
            # for var_name, var_memory in sorted_memory_usage:
            #     count += 1
            #     percentage = (var_memory / total_allocated) * 100
            #     print(f'{var_name}: {var_memory:.2f} MB ({percentage:.2f}%)')
            #     if count >= 5:
            #         break


    # # Calc distance between markers and vertex_normals
    # dot_mv_vn = torch.sum(vec_m_v * vn[:, None, :, :], dim=-1)  # [f, vm, v]
    # proj_m_vn = vgp[:, None, :, :] + dot_mv_vn[:, :, :, None] * vn[:, None, :, :]  # [f, vm, v, 3]
    # dist_m_vn = torch.norm(vm[:, :, None, :] - proj_m_vn, dim=-1)  # [f, vm, v]


# mini batch solver
    # _______________________________________________________________________________
    # batch_size = 100
    # n_batches = n_frames // batch_size
    #
    # params = torch.zeros(n_frames, n_mvm, 3, 3).to(vm.device)
    # virtual_markers = torch.zeros(n_frames, n_mvm, 3).to(vm.device)
    #
    # def frame_mini_batch(si, ei):
    #     print(f'batch: {si}-{ei} / {n_frames}')
    #
    #     params_f, virtual_markers_f = local_offset_solver.batch_find_local_offsets(
    #         markers=vm[si:ei],
    #         j3_indices=m_j3_indices[si:ei],
    #         j3_weights=m_j3_weights[si:ei],
    #         init_params=init_m_j3_offsets[si:ei]
    #     )
    #     params[si:ei] = params_f.view(ei-si, -1, 3, 3)
    #     virtual_markers[si:ei] = virtual_markers_f.view(ei-si, -1, 3)
    #
    # for i in range(n_batches):
    #     frame_start_idx = i * batch_size
    #     frame_end_idx = frame_start_idx + batch_size
    #     frame_mini_batch(frame_start_idx, frame_end_idx)
    #
    # extra_frames = n_frames % batch_size
    # if extra_frames > 600:
    #     frame_start_idx = n_batches * batch_size
    #     frame_end_idx = n_frames
    #     frame_mini_batch(frame_start_idx, frame_end_idx)
    # _______________________________________________________________________________


    # def getitem_real(self, mi, fi, si, ei, sc, ec):
    #     n_joints = self.n_joints
    #     n_markers = self.markers[mi].shape[1]
    #
    #     points_seq = []
    #     points_mask = []
    #     padded_m_j_weights = None
    #     padded_m_j3_weights = None
    #     padded_m_j3_offsets = None
    #
    #     for _ in range(sc):
    #         points_seq.append(torch.zeros(1, self.n_max_markers, 3))
    #         points_mask.append(torch.zeros(1, self.n_max_markers, requires_grad=False))
    #
    #     for i in range(si, ei):
    #         ghost_marker_mask = torch.from_numpy(self.ghost_marker_mask[mi][fi])
    #
    #         points = torch.from_numpy(self.markers[mi][i]).to(torch.float32)
    #         mag = torch.norm(points, dim=1)
    #         filter_idx = torch.nonzero(mag > 0.001).squeeze()[:self.n_max_markers]
    #         filtered_points = points[filter_idx]
    #         filtered_ghost_marker_mask = ghost_marker_mask[filter_idx]
    #
    #         n_valid_markers = min(filtered_points.shape[0], self.n_max_markers)
    #         padded_points = torch.zeros(self.n_max_markers, 3)
    #         padded_points[:n_valid_markers, :] = filtered_points[:n_valid_markers, :]
    #         points_seq.append(padded_points.unsqueeze(0))
    #
    #         mask = torch.zeros(1, self.n_max_markers, requires_grad=False)
    #         mask[:, :n_valid_markers] = 1
    #         points_mask.append(mask)
    #
    #         if i == fi:
    #             m_j_weights = torch.Tensor(self.weights[self.m_v_idx[mi], :]).to(torch.float32)
    #             m_j_weights = m_j_weights[filter_idx, :] * filtered_ghost_marker_mask[:, None]
    #
    #             ghost_weights = torch.ones((self.n_max_markers, 1))
    #             ghost_weights[:n_valid_markers, 0] -= filtered_ghost_marker_mask
    #
    #             padded_m_j_weights = F.pad(
    #                 input=m_j_weights,
    #                 pad=[0, 0, 0, self.n_max_markers - n_valid_markers],
    #                 mode='constant',
    #                 value=0
    #             )
    #             padded_m_j_weights = torch.cat((padded_m_j_weights, ghost_weights), dim=-1)
    #
    #             m_j3_weights = torch.Tensor(self.m_j3_weights[mi]).to(torch.float32)
    #             m_j3_weights = m_j3_weights[filter_idx, :] * filtered_ghost_marker_mask[:, None]
    #
    #             padded_m_j3_weights = F.pad(
    #                 input=m_j3_weights,
    #                 pad=[0, 0, 0, self.n_max_markers - n_valid_markers],
    #                 mode='constant',
    #                 value=0
    #             )
    #             padded_m_j3_weights[:n_valid_markers, 0] += (1 - filtered_ghost_marker_mask) * -1
    #
    #             m_j3_offsets = torch.Tensor(self.m_j3_offsets[mi]).to(torch.float32)
    #             m_j3_offsets = m_j3_offsets[filter_idx, :, :] * filtered_ghost_marker_mask[:, None, None]
    #
    #             padded_m_j3_offsets = F.pad(
    #                 input=m_j3_offsets,
    #                 pad=[0, 0, 0, 0, 0, self.n_max_markers - n_valid_markers],
    #                 mode='constant',
    #                 value=0
    #             )
    #
    #     for _ in range(ec):
    #         points_seq.append(torch.zeros(1, self.n_max_markers, 3))
    #         points_mask.append(torch.zeros(1, self.n_max_markers, requires_grad=False))
    #
    #     points_seq = torch.cat(points_seq, dim=0)
    #     points_mask = torch.cat(points_mask, dim=0)
    #
    #     items = {
    #         'points_seq': points_seq,
    #         'points_mask': points_mask,
    #         'm_j_weights': padded_m_j_weights,
    #         'm_j3_weights': padded_m_j3_weights,
    #         'm_j3_offsets': padded_m_j3_offsets,
    #         'motion_idx': mi,
    #         'frame_idx': fi,
    #         'real': True
    #     }
    #
    #     return items

# gc.collect()
# torch.cuda.empty_cache()
# cur_vram_memory = torch.cuda.memory_allocated()
# max_vram_memory = torch.cuda.max_memory_allocated()
# print(f'VRAM memory: {cur_vram_memory / (1024 ** 3):.2f} GB / {max_vram_memory / (1024 ** 3):.2f} GB')

# assert npz_file.stem == c3d_file.stem + '_poses' or npz_file.stem == c3d_file.stem + '_stageii',\
#     f'Index: {file_idx} | {npz_file.name}, {c3d_file.name}'