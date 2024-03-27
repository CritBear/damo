import numpy as np
import time
import keyboard
from typing import Optional

from modules.utils.viewer.damo_viewer import DamoViewer


class VpythonViewer(DamoViewer):
    def __init__(self, n_max_markers, topology, b_vertex=False, **kwargs) -> None:
        super().__init__(n_max_markers, topology, **kwargs)

        import vpython.no_notebook
        import vpython as v

        self.v = v

        self.window = v.canvas(x=0, y=0, width=1200, height=800, center=v.vector(0, 0, 0), background=v.vector(0, 0, 0))
        # self.window.bind('mousedown', self.on_mouse_down)
        # self.window.bind('mouseup', self.on_mouse_up)
        # self.window.bind('keydown', self.on_key_down)
        # self.window.bind('keyup', self.on_key_up)
        self.window.forward = v.vector(-1, 0, 0)
        self.window.up = v.vector(0, 0, 1)
        # self.key_dict = {
        #     'w': False, 's': False, 'a': False, 'd': False, 'q': False, 'e': False,
        #     'right': False, 'left': False, ' ': False, 'escape': False
        # }

        axis_x = v.arrow(pos=v.vector(0, 0, 0), axis=v.vector(1, 0, 0), shaftwidth=0.05, color=vpython.color.red)
        axis_y = v.arrow(pos=v.vector(0, 0, 0), axis=v.vector(0, 1, 0), shaftwidth=0.05, color=vpython.color.green)
        axis_z = v.arrow(pos=v.vector(0, 0, 0), axis=v.vector(0, 0, 1), shaftwidth=0.05, color=vpython.color.blue)

        self.n_joints = topology.shape[0]

        self.v_markers = [
            v.sphere(radius=0.01, color=v.color.cyan)
            for _ in range(n_max_markers)
        ]

        self.v_joints = [
            v.sphere(radius=0.02, color=v.color.white)
            for _ in range(self.n_joints)
        ]
        self.v_bones = [
            v.cylinder(radius=0.01, color=v.color.white)
            for _ in range(self.n_joints - 1)
        ]

        self.v_joints_space = [
            LocalSpaceArrow(v, radius=0.005, length=0.1, brightness=1)
            for _ in range(self.n_joints)
        ]

        self.v_gt_joints = [
            v.sphere(radius=0.02, color=v.color.orange)
            for _ in range(self.n_joints)
        ]
        self.v_gt_bones = [
            v.cylinder(radius=0.01, color=v.color.orange)
            for _ in range(self.n_joints - 1)
        ]

        self.v_gt_joints_space = [
            LocalSpaceArrow(v, radius=0.005, length=0.1, brightness=0.7)
            for _ in range(self.n_joints)
        ]

        self.jpe_label = v.label(pos=v.vector(-0.3, 0, 0))
        self.joe_label = v.label(pos=v.vector(0.3, 0, 0))

        self.joe_each_label = [
            v.label()
            for _ in range(self.n_joints)
        ]

        self.v_offset = [
            v.arrow(shaftwidth=0.005)
            for _ in range(self.n_max_markers * 3)
        ]
        # self.v_offset = [
        #     v.cylinder(radius=0.005)
        #     for _ in range(self.n_max_markers * 3)
        # ]
        self.v_weight = [
            v.cylinder(radius=0.005)
            for _ in range(self.n_max_markers * 3)
        ]

        self.v_init_joints = [
            v.sphere(radius=0.02, color=v.color.cyan)
            for _ in range(self.n_joints)
        ]
        self.v_init_bones = [
            v.cylinder(radius=0.01, color=v.color.cyan)
            for _ in range(self.n_joints - 1)
        ]

        if b_vertex:
            self.v_vertices = [
                v.sphere(radius=0.01, color=v.vector(0.7, 0.7, 0.7))
                for _ in range(6890)
            ]

        # self.jpe_label = v.wtext(text='')
        # self.joe_label = v.wtext(text='')

    # def on_mouse_down(self, evt):
    #     pass
    #
    # def on_mouse_up(self, evt):
    #     pass
    #
    # def on_key_down(self, evt):
    #     pass
    #
    # def on_key_up(self, evt):
    #     pass

    def visualize(
            self, markers_seq,
            joints_seq, gt_joints_seq=None, b_position=True,
            root_pos_seq=None, gt_root_pos_seq=None,
            skeleton_template=None,
            fps=60, jpe=None, joe=None,
            view_local_space=False,
            vertices_seq=None,
            j3_indices=None,
            ja_weight=None,
            ja_offset=None,
            init_joint=None
    ):

        assert markers_seq.shape[0] == joints_seq.shape[0]
        n_frames = joints_seq.shape[0]

        if vertices_seq is not None:
            assert self.v_vertices is not None

        if init_joint is not None:
            for j in range(self.n_joints):
                self.v_init_joints[j].pos = self.v.vector(*init_joint[j]) - self.v.vector(0, 0, 1)

            for j in range(self.n_joints - 1):
                self.v_init_bones[j].pos = self.v_init_joints[j + 1].pos
                self.v_init_bones[j].axis = self.v_init_joints[self.topology[j + 1]].pos - self.v_init_joints[j + 1].pos

        if gt_joints_seq is not None:
            assert joints_seq.shape[0] == gt_joints_seq.shape[0]
            gt = True
        else:
            gt = False

        if jpe is not None:
            assert jpe.shape[0] == n_frames
        if joe is not None:
            assert joe.shape[0] == n_frames

        if b_position:
            assert len(joints_seq.shape) == 3
            assert view_local_space is False

            jgp_seq = joints_seq
            gt_jgp_seq = gt_joints_seq
        else:
            assert skeleton_template is not None
            assert skeleton_template.shape[0] == self.n_joints
            assert root_pos_seq is not None

            jlp = skeleton_template
            jlr_seq = joints_seq
            jgr_seq = np.empty_like(jlr_seq)
            jgp_seq = np.zeros((n_frames, self.n_joints, 3))
            jgp_seq[:, 0, :] = root_pos_seq

            if gt:
                assert gt_root_pos_seq is not None

                gt_jlr_seq = gt_joints_seq
                gt_jgr_seq = np.empty_like(gt_jlr_seq)
                gt_jgp_seq = np.zeros((n_frames, self.n_joints, 3))
                gt_jgp_seq[:, 0, :] = gt_root_pos_seq

            for i, pi in enumerate(self.topology):
                if i == 0:
                    assert pi == -1
                    jgr_seq[:, i, :, :] = jlr_seq[:, i, :, :]

                    if gt:
                        gt_jgr_seq[:, i, :, :] = gt_jlr_seq[:, i, :, :]

                    continue

                jgr_seq[:, i, :, :] = jgr_seq[:, pi, :, :] @ jlr_seq[:, i, :, :]
                jgp_seq[:, i, :] = (jgr_seq[:, pi, :, :] @ jlp[np.newaxis, i, :, np.newaxis]).squeeze()
                jgp_seq[:, i, :] += jgp_seq[:, pi, :]

                if gt:
                    gt_jgr_seq[:, i, :, :] = gt_jgr_seq[:, pi, :, :] @ gt_jlr_seq[:, i, :, :]
                    gt_jgp_seq[:, i, :] = (gt_jgr_seq[:, pi, :, :] @ jlp[np.newaxis, i, :, np.newaxis]).squeeze()
                    gt_jgp_seq[:, i, :] += gt_jgp_seq[:, pi, :]

        f = 0
        while True:
            if gt:
                self.update_pose(markers_seq[f], jgp_seq[f], gt_jgp_seq[f])
            else:
                self.update_pose(markers_seq[f], jgp_seq[f])

            if jpe is not None:
                self.jpe_label.text = 'jpe: ' + str(round(np.mean(jpe[f]), 2))
            if joe is not None:
                self.joe_label.text = 'joe: ' + str(round(np.mean(joe[f]), 2))
                for j in range(self.n_joints):
                    self.joe_each_label[j].pos = self.v_joints[j].pos
                    self.joe_each_label[j].text = str(round(joe[f, j], 2))

            if view_local_space:
                for j in range(self.n_joints):
                    self.v_joints_space[j].update(self.v_joints[j].pos, jgr_seq[f, j])

                    if gt:
                        self.v_gt_joints_space[j].update(self.v_gt_joints[j].pos, gt_jgr_seq[f, j])

            if vertices_seq is not None:
                for i in range(6890):
                    self.v_vertices[i].pos = self.v.vector(*vertices_seq[f, i])

            if j3_indices is not None:
                for i in range(self.n_max_markers):
                    j3 = j3_indices[f, i]
                    for ii in range(3):
                        if j3[ii] >= self.n_joints or ja_weight[f, i, j3[ii]] < 0.7:
                            self.v_weight[i * 3 + ii].visible = False
                            self.v_offset[i * 3 + ii].visible = False
                            continue

                        self.v_weight[i * 3 + ii].visible = True
                        self.v_offset[i * 3 + ii].visible = True
                        self.v_weight[i * 3 + ii].color = self.v.vector(1, 1, 0) * ja_weight[f, i, j3[ii]]
                        self.v_offset[i * 3 + ii].color = self.v.vector(1, 1, 1) * ja_weight[f, i, j3[ii]]

                        self.v_weight[i * 3 + ii].pos = self.v_joints[j3[ii]].pos
                        self.v_weight[i * 3 + ii].axis = self.v_markers[i].pos - self.v_joints[j3[ii]].pos
                        self.v_offset[i * 3 + ii].pos = self.v_init_joints[j3[ii]].pos
                        # self.v_offset[i * 3 + ii].pos = self.v_markers[i].pos
                        self.v_offset[i * 3 + ii].axis = self.v.vector(*ja_offset[f, i, j3[ii]])



            if keyboard.is_pressed('right'):
                f = (f + 1) % n_frames

            if keyboard.is_pressed("left"):
                f = (f - 1) % n_frames

            time.sleep(1 / fps)

    def update_pose(self, marker_position, joint_global_position, gt_joint_global_position=None):
        for m in range(marker_position.shape[0]):
            self.v_markers[m].pos = self.v.vector(*marker_position[m])

        for j in range(self.n_joints):
            self.v_joints[j].pos = self.v.vector(*joint_global_position[j])

        for j in range(self.n_joints - 1):
            self.v_bones[j].pos = self.v_joints[j + 1].pos
            self.v_bones[j].axis = self.v_joints[self.topology[j + 1]].pos - self.v_joints[j + 1].pos

        if gt_joint_global_position is not None:
            for j in range(self.n_joints):
                self.v_gt_joints[j].pos = self.v.vector(*gt_joint_global_position[j])

            for j in range(self.n_joints - 1):
                self.v_gt_bones[j].pos = self.v_gt_joints[j + 1].pos
                self.v_gt_bones[j].axis = self.v_gt_joints[self.topology[j + 1]].pos - self.v_gt_joints[j + 1].pos


class LocalSpaceArrow:
    def __init__(self, v, radius, length, brightness=1.0):
        self.v = v
        self.length = length
        self.x = v.arrow(shaftwidth=radius, color=v.vector(1, 0, 0)*brightness)
        self.y = v.arrow(shaftwidth=radius, color=v.vector(0, 1, 0)*brightness)
        self.z = v.arrow(shaftwidth=radius, color=v.vector(0, 0, 1)*brightness)

    def update(self, pos, rot):
        if not isinstance(pos, self.v.vector):
            pos = self.v.vector(*pos)
        self.x.pos = self.y.pos = self.z.pos = pos

        self.x.axis = self.v.vector(*rot[:3, 0]) * self.length
        self.y.axis = self.v.vector(*rot[:3, 1]) * self.length
        self.z.axis = self.v.vector(*rot[:3, 2]) * self.length
        # self.x.axis = self.v.vector(*rot[0, :3]) * self.length
        # self.y.axis = self.v.vector(*rot[1, :3]) * self.length
        # self.z.axis = self.v.vector(*rot[2, :3]) * self.length

