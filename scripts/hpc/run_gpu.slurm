#!/bin/bash
# Run an experiment file on one A100. Submit from the root of your clone on g17:
#
#     sbatch scripts/hpc/run_gpu.slurm configs/experiments/01_steps.yaml --num-prompts 4
#
# Extra arguments go to scripts/run_experiments.py. Results go to ${ELDM_RESULTS}
# (/scratch/cmpe261-fa26/group4/results) unless you pass --output-root. The default
# --time fits a tiny run of 2-4 prompts; give a full run its own limit:
#
#     sbatch --time=08:00:00 scripts/hpc/run_gpu.slurm configs/experiments/01_steps.yaml
#
#SBATCH --job-name=eldm
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1          # never a plain gpu:1: it can land on a P100 or an H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
CLONE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
source "${CLONE}/scripts/hpc/job_prelude.sh"

# Experiment files name configs/base.yaml and data/prompts/... relative to the repo root.
cd "${CLONE}"
# --device cuda, not the configs' "auto": a GPU that CUDA can't use must fail the job, not
# move it to the CPU (job 99031 ran at 108 s/image on cs001's CPUs). A --device in "$@" still wins.
python scripts/run_experiments.py --output-root "${ELDM_RESULTS}" --device cuda "$@"
