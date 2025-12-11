#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:20:00
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -G 16
#SBATCH -J blip2_deepspeed
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module load python/3.10
module load pytorch/2.8.0

source /pscratch/sd/r/rs2531/Course_Project/env/bin/activate

cd /pscratch/sd/r/rs2531/Course_Project/DeepSpeed/ds_ws16/

srun -N "$SLURM_NNODES" \
     --ntasks-per-node=4 \
     --gpus-per-node=4 \
     python -u train_deepspeed.py \
     --config deepspeed_blip2_roco.json \
     --deepspeed
