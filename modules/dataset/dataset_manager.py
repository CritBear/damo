import torch
import torch.distributed as dist
import logging
import random
import itertools
from queue import Queue

from modules.utils.paths import Paths


class DatasetManager:
    def __init__(self, options, shuffle=True, drop_last=True):
        self.train_data_pool = None
        self.__iter_mode = 'train'

        batch_size = options.batch_size
        device = options.device

        self.is_distributed_train = False
        if options.device == 'cuda-dist':
            self.is_distributed_train = True
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            device = f'cuda:{self.rank}'
            if self.rank == 0:
                logging.info(f"|device: 'cuda-dist'| Distributed training mode of {self.world_size} devices set.")

            self.seed = options.seed
            random.seed(self.seed)

            if batch_size % self.world_size != 0:
                raise ValueError("Batch size should be divisible by "
                                 "the number of gpus(processes) for distributed training.")
            if not drop_last:
                raise ValueError("drop_last should be True for distributed training.")

            logging.info(f"Rank {self.rank} | World size: {self.world_size} | Device: {device} | Seed: {self.seed}")

        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.loading_animation = itertools.cycle(['-', '/', '|', '\\'])

    def __len__(self):
        if self.__iter_mode == 'train':
            if self.drop_last:
                return 0
            else:
                return 0
        elif self.__iter_mode == 'test':
            if self.drop_last:
                return 0
            else:
                return 0
        else:
            raise ValueError(f"Invalid iter mode: {self.__iter_mode}")

    def load_dataset(self, options):
        if not self.is_distributed_train or self.rank == 0:
            logging.info('Loading dataset from')
            for path in options.train_dataset_paths:
                logging.info(f'    {path}')

        self.train_data_pool = []

        if not self.is_distributed_train or self.rank == 0:
            candidate_data_queue = Queue()
            loaded_data_queue = Queue()

