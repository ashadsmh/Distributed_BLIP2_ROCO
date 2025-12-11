#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -G 4
#SBATCH -J blip2_ds
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module load python/3.10
module load pytorch/2.8.0

source /pscratch/sd/r/rs2531/Course_Project/env/bin/activate

cd /pscratch/sd/r/rs2531/Course_Project/DeepSpeed/ds_ws4/

srun deepspeed train_ds.py \
     --config ds_blip2_roco.json \
     --deepspeed ds_blip2_roco.json
