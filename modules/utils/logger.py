import sys
import logging
import subprocess


def setup_logging(save_path=None):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    if save_path:
        logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
            logging.FileHandler(save_path),
            logging.StreamHandler(sys.stdout)
        ])
        logging.info(f"New logging session started : {save_path}")
    else:
        logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
            logging.StreamHandler(sys.stdout)
        ])


def get_gpu_util_mem_usage_str():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory',
                                 '--format=csv,nounits,noheader'], stdout=subprocess.PIPE, text=True)
        utilization = result.stdout.strip().split('\n')
        gpu_utils = []
        gpu_mems = []
        for gpu_idx in range(len(utilization)):
            utilization[gpu_idx] = utilization[gpu_idx].split(',')
            gpu_utils.append(utilization[gpu_idx][0].strip())
            gpu_mems.append(utilization[gpu_idx][1].strip())

        gpu_util_str =(
                f'0:{gpu_utils[0]}%' + ''.join(
            [f" {rdx}:{gpu_utils[rdx]}%" for rdx in range(1, len(gpu_utils))]))
        gpu_mem_str =(
                f'0:{gpu_mems[0]}%' + ''.join(
            [f" {rdx}:{gpu_mems[rdx]}%" for rdx in range(1, len(gpu_mems))]))

        return gpu_util_str, gpu_mem_str

    except Exception as e:
        logging.error(f"Failed to get GPU utilization: {e}")
        return "N/A", "N/A"
