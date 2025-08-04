#!/bin/bash
#SBATCH --account=project_2006362
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:1

export PIP_CACHE_DIR=/scratch/project_2006362/v/cache  
export TMPDIR=/scratch/project_2006362/v/cache
export HF_HOME=/scratch/project_2006362/v/cache
export TRANSFORMERS_CACHE=/scratch/project_2006362/v/cache
export TORCH_HOME=/scratch/project_2006362/v/cache

source ./venv/bin/activate
# srun python3 run.py
srun python3 test_score.py