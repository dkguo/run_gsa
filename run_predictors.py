import subprocess

processes = []
for i in range(8):
    process = subprocess.Popen(f'CUDA_VISIBLE_DEVICES={i} python predictor.py')
    processes.append(process)

try:
    while True:
        pass
except KeyboardInterrupt:
    for process in processes:
        process.kill()
