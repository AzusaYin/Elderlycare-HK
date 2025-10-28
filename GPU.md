申请GPU
srun --account=mscbdtsuperpod --partition=normal --gpus-per-node=1 --time=02:00:00 --pty bash

for more information, please check https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/hpc4/slurm