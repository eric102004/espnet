#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=logs/train_global_%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH --account=bbjs-delta-gpu
#SBATCH --gres=gpu:2

# set -euo pipefail

source ~/.bashrc
source path.sh

# Make W&B reliable on HPC
export WANDB_DISABLE_SERVICE=true

# We can also create submit.sh for the actual command.
python run.py \
    --train_config conf/owsm_finetune_gpu2.yaml \
    --eval_config inference_beam1.yaml \
    --stage train
    # >logs/owsm_finetune_lr0.00001_48hr.log 2>&1