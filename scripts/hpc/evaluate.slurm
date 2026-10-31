#!/bin/bash
# Evaluate the quality of every run under a results directory, check them, and summarize them,
# on one A100. Submit from the root of your clone on g17:
#
#     sbatch scripts/hpc/evaluate.slurm <results_dir> <baseline_run_dir>
#
# A run is any directory under <results_dir> with a config.yaml. For each one,
# scripts/evaluate_quality.py writes quality.json and quality_per_image.csv: CLIP Score
# (ViT-g/14 and ViT-L/14), LPIPS against <baseline_run_dir>, KID against the COCO photos (FID
# from 5,000 images). Extra arguments go to evaluate_quality.py. Then scripts/check_run.py
# checks every run, and scripts/summarize.py writes <results_dir>/summary/ (summary.csv, plots),
# with speedups relative to <baseline_run_dir>. The default --time fits a few hundred images;
# give a sweep its own limit.
#
#SBATCH --job-name=eldm-eval
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1          # never a plain gpu:1: it can land on a P100 or an H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
CLONE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
source "${CLONE}/scripts/hpc/job_prelude.sh"

usage="usage: sbatch scripts/hpc/evaluate.slurm <results_dir> <baseline_run_dir> [evaluate_quality.py args]"
RESULTS="$(realpath -e "${1:?${usage}}")"
BASELINE="$(realpath -e "${2:?${usage}}")"
shift 2
[[ -f "${BASELINE}/config.yaml" ]] || { echo "error: ${BASELINE} is not a run directory" >&2; exit 1; }

mapfile -t RUNS < <(find "${RESULTS}" -name config.yaml -not -path "${RESULTS}/summary/*" -printf '%h\n' | sort)
(( ${#RUNS[@]} )) || { echo "error: no run directories under ${RESULTS}" >&2; exit 1; }
echo "evaluating ${#RUNS[@]} run(s) under ${RESULTS}, baseline ${BASELINE}:"
printf '  %s\n' "${RUNS[@]}"

cd "${CLONE}"
python scripts/evaluate_quality.py --runs "${RUNS[@]}" --baseline "${BASELINE}" "$@"
echo
# Summarize even if a check fails, so the table shows what's wrong; the job still fails.
status=0
python scripts/check_run.py --expect-quality --require-gpu A100 "${RUNS[@]}" || status=$?
echo
python scripts/summarize.py --runs "${RUNS[@]}" --baseline "${BASELINE}" \
    --csv "${RESULTS}/summary/summary.csv" --out "${RESULTS}/summary"
exit "${status}"
