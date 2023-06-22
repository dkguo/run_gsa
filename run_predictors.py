import subprocess
import sys

if __name__ == '__main__':
    num_gpus = int(sys.argv[1])

    processes = []
    for i in range(num_gpus):
        process = subprocess.Popen(f'CUDA_VISIBLE_DEVICES={i} python predictor.py')
        processes.append(process)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        for process in processes:
            process.kill()
