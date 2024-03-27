import time
import torch
import json
import argparse

from modules.utils.paths import Paths

class TrainingOptions:
    def __init__(self):
        self.model_name = 'damo'
        self.model_comment = ''
        self.dataset_names = ['SFU']
        self.dataset_date = '20240326'

        # model setting
        self.n_joints = 24
        self.n_max_markers = 90
        self.seq_len = 7
        self.d_model = 125
        self.d_hidden = 128
        self.n_layers = 4
        self.n_heads = 5

        # training setting
        self.use_model_load = False
        self.loading_model_name = ""
        self.loading_model_epoch = 0

        self.seed = 2024  # or None
        self.n_epochs = 200
        self.lr = 1e-5
        self.batch_size = 64
        self.test_epoch_step = 10
        self.test_sample_step = 10

        # wandb setting
        self.use_wandb = False
        self.wandb_login_key = ""
        self.wandb_note = ""
        self.wand_project_name = ""

        self.device = None  # 'cuda-dist'
        self.common_dataset_path = None
        self.train_dataset_paths = None
        self.test_dataset_paths = None
        self.start_time = None
        self.model_dir = None
        self.log_dir = None

    def process_options(self):
        if self.device != "cuda-dist":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"INFO | TrainOptions | Device: {self.device}")

        # processing
        self.common_dataset_path = Paths.datasets / 'common' / f'damo_common_{self.dataset_date}.pkl'
        self.train_dataset_paths = [
            Paths.datasets / 'train' / f'damo_train_{self.dataset_date}_{dataset_name}.pkl'
            for dataset_name in self.dataset_names
        ]
        self.test_dataset_paths = [
            Paths.datasets / 'test' / f'damo_test_{self.dataset_date}_{dataset_name}.pkl'
            for dataset_name in self.dataset_names
        ]

        print(f"INFO | TrainOptions | Training datasets")
        for path in self.train_dataset_paths:
            print(f'\t{path}')

        self.start_time = f"{time.strftime('%y%m%d%H%M%S')}"

        if self.model_comment != '':
            self.model_name += f'_{self.model_comment}'

        self.model_name += f'_{self.start_time}'
        self.model_dir = Paths.trained_models / self.model_name
        self.log_dir = self.model_dir / 'logs'

        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

        print(f"INFO | TrainOptions | Model directory: {self.model_dir}")

    def save_training_info(self):
        self.save_options()

        # if self.use_wandb:
        #     wandb.config.update(self.loss_weight_dict_for_motion)
        #     wandb.config.update(self.options_to_save)
        #     print('---Current Setting---')
        #     print(self.loss_weight_dict_for_motion)
        #     print(self.options_to_save)

    def save_options(self, ext='json'):
        options = {}
        for k, v in self.__dict__.items():
            if isinstance(v, list):
                options[k] = [str(item) for item in v]
            else:
                try:
                    str_v = str(v)
                    options[k] = str_v
                except (TypeError, ValueError):
                    pass

        if ext == 'json':
            with open(self.model_dir / 'options.json', 'w') as f:
                options_json = json.dumps(options, indent=4)
                f.write(options_json)
        elif ext == 'txt':
            with open(self.model_dir / 'options.txt', 'w') as f:
                for k, v in options.items():
                    f.write(f'{k}: ')
                    if isinstance(v, list):
                        f.write('\n')
                        for item in v:
                            f.write(f'    {item}\n')
                    else:
                        f.write(f'{v}\n')
        else:
            ValueError(f'ERROR: Invalid options ext: {ext}')


def load_options_from_json(path):
    with open(path, 'r') as f:
        options_dict = json.load(f)

    options = TrainingOptions()

    for k, v in options_dict.items():
        if hasattr(options, k):
            setattr(options, k, v)
        else:
            ValueError(f'ERROR: Json has invalid key: {k}')

    options.process_options()
    options.save_options(ext='json')

    return options
