import os
import torch
from datetime import timedelta
import torch.distributed as dist


def setup_devices(local_rank, world_size, port='34129'):
    if not dist.is_nccl_available():
        raise RuntimeError("NCCL is not available.")
    if world_size <= 1:
        raise RuntimeError(f"World size {world_size} is less than or equal to 1. You should not use distributed training for a single device.")
    if world_size > torch.cuda.device_count():
        raise RuntimeError(f"World size {world_size} is greater than the number of available GPUs {torch.cuda.device_count()}.")

    print(f"Setup for device rank {local_rank} of total {world_size} devices.")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    # Set the device to the GPU
    torch.cuda.set_device(local_rank)
    # Initialize the process group
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size, timeout=timedelta(hours=1))
