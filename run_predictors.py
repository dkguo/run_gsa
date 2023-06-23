import subprocess
import sys
import time
import torch

if __name__ == '__main__':
    num_gpu = torch.cuda.device_count()
    num_process = int(sys.argv[1])

    processes = []
    for i in range(num_process):
        process = subprocess.Popen(f'CUDA_VISIBLE_DEVICES={i % num_gpu} python predictor.py', shell=True)
        processes.append(process)
        time.sleep(2)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        for process in processes:
            process.kill()
