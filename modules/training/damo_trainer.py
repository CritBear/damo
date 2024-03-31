import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm
import pickle
import random

from modules.dataset.damo_dataset import DamoDataset
from modules.network.damo import Damo
from modules.utils.paths import Paths
from modules.utils.functions import dict2class


class LossMemory:
    def __init__(self):
        self.total = 0
        self.indices = 0
        self.weights = 0
        self.offsets = 0

    def add_loss_dict(self, loss_dict):
        self.total += loss_dict['total'].item()
        self.indices += loss_dict['indices'].item()
        self.weights += loss_dict['weights'].item()
        self.offsets += loss_dict['offsets'].item()

    def write(self, writer, category, epoch):
        writer.add_scalar(f"{category}/total", self.total, epoch)
        writer.add_scalar(f"{category}/indices", self.indices, epoch)
        writer.add_scalar(f"{category}/weights", self.weights, epoch)
        writer.add_scalar(f"{category}/offsets", self.offsets, epoch)

    def divide(self, value):
        assert isinstance(value, (int, float))
        self.total /= value
        self.indices /= value
        self.weights /= value
        self.offsets /= value


class DamoTrainer:
    def __init__(self, options):
        self.options = options

    def train(self):
        options = self.options

        if options.seed is not None:
            random.seed(options.seed)
            np.random.seed(options.seed)
            torch.manual_seed(options.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        train_dataset = DamoDataset(
            common_dataset_path=options.common_dataset_path,
            dataset_paths=options.train_dataset_paths,
            n_max_markers=options.n_max_markers,
            seq_len=options.seq_len,
            r_ss_ds_ratio=[0.5, 0.25, 0.25],
            noise_jitter=True,
            noise_ghost=True,
            noise_occlusion=True,
            noise_shuffle=True,
            dist_from_skin=0.01,
            dist_augmentation=True,
            test=False
        )

        test_synthetic_dataset = DamoDataset(
            common_dataset_path=options.common_dataset_path,
            dataset_paths=options.test_dataset_paths,
            n_max_markers=options.n_max_markers,
            seq_len=options.seq_len,
            r_ss_ds_ratio=[0, 0.5, 0.5],
            noise_jitter=True,
            noise_ghost=True,
            noise_occlusion=True,
            noise_shuffle=True,
            dist_from_skin=0.01,
            dist_augmentation=True,
            test=True
        )

        test_real_dataset = DamoDataset(
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
            test=True
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=options.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
            pin_memory=True,
            worker_init_fn=DamoTrainer.seed_worker
        )

        test_synthetic_dataloader = DataLoader(
            test_synthetic_dataset,
            batch_size=options.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=6,
            pin_memory=True,
            worker_init_fn=DamoTrainer.seed_worker
        )

        test_real_dataloader = DataLoader(
            test_real_dataset,
            batch_size=options.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=6,
            pin_memory=True,
            worker_init_fn=DamoTrainer.seed_worker
        )

        model = Damo(options).to(options.device)
        optimizer = torch.optim.Adam(model.parameters())

        writer = SummaryWriter(options.log_dir / options.model_name)
        print(f"INFO | Train | Tensorboard command: tensorboard --logdir={Paths.trained_models}")

        for epoch in range(options.n_epochs):
            model.train()
            train_iterator = tqdm(
                enumerate(train_dataloader), total=len(train_dataloader), desc="train"
            )

            train_loss = LossMemory()
            for idx, batch in train_iterator:
                optimizer.zero_grad()

                points_seq = batch['points_seq'].to(options.device)
                points_mask = batch['points_mask'].to(options.device)
                indices = batch['m_j_weights'].to(options.device)
                weights = batch['m_j3_weights'].to(options.device)
                offsets = batch['m_j3_offsets'].to(options.device)

                joint_indices_pred, weights_pred, offsets_pred \
                    = model(points_seq, points_mask)
                # (batch_size, max_markers, n_joints + 1)
                # (batch_size, max_markers, 3)
                # (batch_size, max_markers, 3, 3)

                output_mask = points_mask[:, options.seq_len // 2, :]
                # (batch_size, max_markers)

                loss_dict = DamoTrainer.loss_type_1(
                    indices=indices,
                    indices_pred=joint_indices_pred,
                    weights=weights,
                    weights_pred=weights_pred,
                    offsets=offsets,
                    offsets_pred=offsets_pred,
                    mask=output_mask
                )

                loss_dict['total'].backward()
                optimizer.step()

                train_loss.add_loss_dict(loss_dict)

                train_iterator.set_postfix({
                    "train_loss": float(loss_dict['total'].item()),
                    "i": float(loss_dict['indices'].item()),
                    "w": float(loss_dict['weights'].item()),
                    "o": float(loss_dict['offsets'].item())
                })

            train_loss.divide(len(train_dataloader))
            train_loss.write(writer, 'train', epoch)

            if epoch == 0 or (epoch + 1) % options.test_epoch_step == 0:
                model.eval()

                test_iterator = tqdm(
                    enumerate(test_synthetic_dataloader), total=len(test_synthetic_dataloader), desc="test_synthetic"
                )

                test_loss = LossMemory()
                with torch.no_grad():
                    for idx, batch in test_iterator:
                        points_seq = batch['points_seq'].to(options.device)
                        points_mask = batch['points_mask'].to(options.device)
                        indices = batch['m_j_weights'].to(options.device)
                        weights = batch['m_j3_weights'].to(options.device)
                        offsets = batch['m_j3_offsets'].to(options.device)

                        joint_indices_pred, weights_pred, offsets_pred \
                            = model(points_seq, points_mask)
                        # (batch_size, max_markers, n_joints)
                        # (batch_size, max_markers, n_joints, 3)

                        output_mask = points_mask[:, options.seq_len // 2, :]
                        # (batch_size, max_markers)

                        loss_dict = DamoTrainer.loss_type_1(
                            indices=indices,
                            indices_pred=joint_indices_pred,
                            weights=weights,
                            weights_pred=weights_pred,
                            offsets=offsets,
                            offsets_pred=offsets_pred,
                            mask=output_mask
                        )

                        test_loss.add_loss_dict(loss_dict)

                        test_iterator.set_postfix({
                            "test_loss": float(loss_dict['total'].item()),
                            "i": float(loss_dict['indices'].item()),
                            "w": float(loss_dict['weights'].item()),
                            "o": float(loss_dict['offsets'].item())
                        })

                test_loss.divide(len(test_synthetic_dataloader))
                test_loss.write(writer, 'test_synthetic', epoch)

                test_iterator = tqdm(
                    enumerate(test_real_dataloader), total=len(test_real_dataloader), desc="test_real"
                )

                test_loss = LossMemory()
                with torch.no_grad():
                    for idx, batch in test_iterator:
                        points_seq = batch['points_seq'].to(options.device)
                        points_mask = batch['points_mask'].to(options.device)
                        indices = batch['m_j_weights'].to(options.device)
                        weights = batch['m_j3_weights'].to(options.device)
                        offsets = batch['m_j3_offsets'].to(options.device)

                        joint_indices_pred, weights_pred, offsets_pred \
                            = model(points_seq, points_mask)
                        # (batch_size, max_markers, n_joints)
                        # (batch_size, max_markers, n_joints, 3)

                        output_mask = points_mask[:, options.seq_len // 2, :]
                        # (batch_size, max_markers)

                        loss_dict = DamoTrainer.loss_type_1(
                            indices=indices,
                            indices_pred=joint_indices_pred,
                            weights=weights,
                            weights_pred=weights_pred,
                            offsets=offsets,
                            offsets_pred=offsets_pred,
                            mask=output_mask
                        )

                        test_loss.add_loss_dict(loss_dict)

                        test_iterator.set_postfix({
                            "test_loss": float(loss_dict['total'].item()),
                            "i": float(loss_dict['indices'].item()),
                            "w": float(loss_dict['weights'].item()),
                            "o": float(loss_dict['offsets'].item())
                        })

                test_loss.divide(len(test_real_dataloader))
                test_loss.write(writer, 'test_real', epoch)

            print(f'Epoch: {epoch + 1}')

            writer.flush()
            if (epoch + 1) % 10 == 0 or (epoch + 1) == options.n_epochs:
                model_name = f'{options.model_name}_epc{epoch + 1}.pt'
                torch.save(model.state_dict(), options.model_dir / model_name)
                print(f'Trained model saved. EPOCH: {epoch + 1}')

        writer.close()

    @staticmethod
    def loss_type_1(indices, indices_pred, weights, weights_pred, offsets, offsets_pred, mask):
        mse = torch.nn.MSELoss(reduction='none')

        indices_loss = mse(indices, indices_pred)
        indices_loss = indices_loss * mask.unsqueeze(2)
        indices_loss = indices_loss.sum() / mask.sum()

        weights_loss = mse(weights, weights_pred)
        weights_loss = weights_loss * mask.unsqueeze(2)
        weights_loss = weights_loss.sum() / mask.sum()

        offsets_loss = mse(offsets, offsets_pred)
        offsets_loss = offsets_loss * mask.unsqueeze(2).unsqueeze(3)
        offsets_loss = offsets_loss.sum() / mask.sum()

        total_loss = indices_loss + weights_loss + offsets_loss

        loss = {
            'total': total_loss,
            'indices': indices_loss,
            'weights': weights_loss,
            'offsets': offsets_loss
        }

        return loss

    # for dataloader multi-processing
    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)