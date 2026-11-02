#!/bin/bash
# One run per array task, on one A100. Don't submit it yourself: scripts/hpc/submit_experiment.sh
# submits it with --array set to the runs not done yet, a time limit, and an evaluation job that
# waits for every task. Arguments go to scripts/run_experiments.py; each task runs the run at
# position $SLURM_ARRAY_TASK_ID of `run_experiments.py <same arguments> --list`.
#
#SBATCH --job-name=eldm-array
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1          # never a plain gpu:1: it can land on a P100 or an H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
CLONE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
source "${CLONE}/scripts/hpc/job_prelude.sh"

# Experiment files name configs/base.yaml and data/prompts/... relative to the repo root.
cd "${CLONE}"
# --device cuda as in run_gpu.slurm. --skip-existing: a task that Slurm requeues after a node
# failure, or that two submissions share, skips its run if it finished. The run is written to the
# node-local SSD (the prelude's STAGE) and copied to the results at its end, perf.json last.
# --index goes after "$@", whose experiment files would otherwise be read as more indices.
python scripts/run_experiments.py --output-root "${ELDM_RESULTS}" --device cuda --skip-existing \
    --stage-dir "${STAGE}" "$@" --index "${SLURM_ARRAY_TASK_ID:?submit with scripts/hpc/submit_experiment.sh}"
