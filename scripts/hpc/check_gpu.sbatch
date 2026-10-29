#!/bin/bash
# Gate G1 (Phase 3): the real pipeline makes images on one A100 for four variants:
# baseline PNDM-50 FP32, FP16, DPM-Solver++ 20 steps and DeepCache N=3, 2 prompts each.
# Then scripts/check_run.py verifies every run directory. Submit from the root of your
# clone on g17:
#
#     sbatch scripts/hpc/check_gpu.sbatch
#
# Outputs go to ${ELDM_RESULTS}/_checks/<jobid>/: one run directory per variant, plus
# prompts.jsonl and summary.json.
#
#SBATCH --job-name=eldm-check-gpu
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
CLONE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
source "${CLONE}/scripts/hpc/job_prelude.sh"

OUT="${ELDM_RESULTS}/_checks/${SLURM_JOB_ID:-local}"
status=0
python "${CLONE}/scripts/hpc/gpu_check.py" --stage "${STAGE}" --out "${OUT}" || status=$?
echo
# Check whatever was produced, even if a variant failed.
shopt -s nullglob
runs=("${OUT}"/*/)
if (( ${#runs[@]} )); then
    python "${CLONE}/scripts/check_run.py" --require-gpu A100 "${runs[@]}" || status=$?
else
    echo "FAIL: no run directories in ${OUT}"
    status=1
fi
exit "${status}"
