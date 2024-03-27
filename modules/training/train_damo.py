import torch
import torch.distributed as dist
import logging

from modules.dataset.damo_dataset import DamoDataset
from modules.network.damo import Damo
from modules.training.training_options import load_options_from_json
from modules.dataset.dataset_manager import DatasetManager
from modules.utils.paths import Paths
from modules.utils.logger import setup_logging, get_gpu_util_mem_usage_str


def train_start():
    json_path = Paths.support_data / "train_options.json"
    options = load_options_from_json(json_path)

    if options.device == 'cuda-dist':
        device_cnt = torch.cuda.device_count()
        logging.info(f'Start distributed training with {device_cnt} devices.')
        shared_dict = torch.multiprocessing.Manager().dict()
        torch.multiprocessing.spawn(
            train,
            args=(options, device_cnt, shared_dict),
            nprocs=device_cnt,
            join=True
        )
    else:
        train(0, options)


def train(process_num, options, world_size=1, shared_dict=None):
    if options.device == 'cuda-dist':
        from modules.utils.distributed_train_utils import setup_devices
        rank = process_num
        print(f"Spawning process for device {rank}.")
        setup_devices(rank, world_size)
        device = f'cuda:{rank}'
    else:
        device = options.device

    # Make the model, loss function, and optimizer
    model = Damo(options).to(device)
    if options.device == 'cuda-dist':
        model = torch.nn.parallel.DistributedDataParallel(model)

    loss_fn = torch.nn.MSELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)

    # Load saved model parameters if the option is set
    if options.use_model_load:
        loading_model_dir = Paths.trained_models / options.loading_model_name
        raise NotImplementedError
    else:
        setup_logging(options.log_dir / "training_logs.txt")

    dataset_manager = DatasetManager(options=options, shuffle=True, drop_last=True)
    dataset_manager.load_dataset(

    )

if __name__ == '__main__':
    train_start()
