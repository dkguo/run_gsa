import subprocess
import sys
import time

if __name__ == '__main__':
    num_gpus = int(sys.argv[1])

    processes = []
    for i in range(num_gpus):
        process = subprocess.Popen(f'CUDA_VISIBLE_DEVICES={i} python predictor.py', shell=True)
        processes.append(process)
        time.sleep(2)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        for process in processes:
            process.kill()
