import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle


class ScoreManager:
    def __init__(self):
        self.memory = {}
        self.clear_memory()

    def clear_memory(self):
        self.memory = {
            'motion_mean_std': [],
            'jpe': [],
            'joe': [],
            'vpe': []
        }

    def calc_error(self, jgp, gt_jgp, jlr, gt_jlr, vgp=None, gt_vgp=None):
        jpe_mean, jpe_std = self.calc_jpe(jgp, gt_jgp)
        print(f'JPE: {jpe_mean:.2f} ± {jpe_std:.2f}')
        joe_mean, joe_std = self.calc_joe(jlr[:, :, :, :], gt_jlr[:, :, :, :])
        print(f'JOE: {joe_mean:.2f} ± {joe_std:.2f}')

        vpe_mean = None
        vpe_std = None
        if vgp is not None:
            assert gt_vgp is not None
            vpe_mean, vpe_std = self.calc_vpe(vgp, gt_vgp)
            print(f'VPE: {vpe_mean:.2f} ± {vpe_std:.2f}')


        self.memory['motion_mean_std'].append(
            {
                'jpe_mean': jpe_mean,
                'jpe_std': jpe_std,
                'joe_mean': joe_mean,
                'joe_std': joe_std,
                'vpe_mean': vpe_mean,
                'vpe_std': vpe_std
            }
        )

    def calc_jpe(self, jgp, gt_jgp):
        dist = np.sqrt(np.sum((jgp - gt_jgp) ** 2, axis=-1))
        dist *= 1000
        self.memory['jpe'].append(dist)

        return np.mean(dist), np.std(dist)

    def calc_joe(self, jlr, gt_jlr):
        f, j, _, _ = jlr.shape
        jlr = R.from_matrix(jlr.reshape(-1, 3, 3))
        gt_jlr = R.from_matrix(gt_jlr.reshape(-1, 3, 3))
        rel_rot = gt_jlr.inv() * jlr
        rot_diff = np.degrees(rel_rot.magnitude())
        rot_diff = rot_diff.reshape(f, j)
        self.memory['joe'].append(rot_diff)

        return np.mean(rot_diff), np.std(rot_diff)

    def calc_vpe(self, vgp, gt_vgp):
        dist = np.sqrt(np.sum((vgp - gt_vgp) ** 2, axis=-1))
        dist *= 1000
        self.memory['vpe'].append(dist)

        return np.mean(dist), np.std(dist)

    def save_score(self, path):
        self.memory['jpe'] = np.hstack(self.memory['jpe'])
        self.memory['joe'] = np.hstack(self.memory['joe'])

        jpe_mean = np.mean(self.memory['jpe'])
        jpe_std = np.std(self.memory['jpe'])
        joe_mean = np.mean(self.memory['joe'])
        joe_std = np.std(self.memory['joe'])

        print('________________Total________________')
        print(f'JPE: {jpe_mean:.2f} ± {jpe_std:.2f}')
        print(f'JOE: {joe_mean:.2f} ± {joe_std:.2f}')

        if len(self.memory['vpe']) > 0:
            self.memory['vpe'] = np.hstack(self.memory['vpe'])
            vpe_mean = np.mean(self.memory['vpe'])
            vpe_std = np.std(self.memory['vpe'])
            print(f'VPE: {vpe_mean:.2f} ± {vpe_std:.2f}')

        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)

