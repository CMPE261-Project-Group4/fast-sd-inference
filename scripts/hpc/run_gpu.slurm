#!/bin/bash
# Usage: sbatch scripts/hpc/run_gpu.slurm configs/experiments/01_steps.yaml [extra args...]
# Run from the repo root. Adjust partition / modules to match SJSU HPC docs.

#SBATCH --job-name=eldm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

# module load python3 cuda   # TODO: confirm module names on SJSU HPC
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python scripts/run_experiments.py "$@"
