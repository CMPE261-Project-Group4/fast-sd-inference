#!/bin/bash
# Phase 2 gate: every model, metric weight and dataset file loads on an HPC3 node offline.
#
# On COE-HPC3, from the root of your clone on g17 (the log lands in its logs/):
#     sbatch scripts/hpc/check_assets.sbatch
# The quick variant (files, hashes, counts; no model loads) runs on coe-hpc1:
#     source env/site.sh && python scripts/hpc/check_assets.py --quick
#
# Memory: open_clip builds ViT-g/14 (5.5 GB fp32) and then loads its 5.5 GB state dict,
# so the peak is about 11 GB.
#
#SBATCH --job-name=eldm-check-assets
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
# The clone to check: the one the job was submitted from (Slurm runs a spooled copy of
# this script), or this script's own clone when run with bash.
CLONE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
if [[ ! -f "${CLONE}/env/site.sh" ]]; then
    echo "error: ${CLONE} is not a fast-sd-inference clone; submit from the repo root" >&2
    exit 1
fi

echo "job=${SLURM_JOB_ID:-none} node=$(hostname) partition=${SLURM_JOB_PARTITION:-none}" \
     "date=$(date -Is) commit=$(cd "${CLONE}" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Clean environment: only env/site.sh, plus thread counts that match the allocation.
# Slurm does not confine CPUs here, so torch would otherwise use every core of the node.
THREADS="${SLURM_CPUS_PER_TASK:-2}"
env -i HOME="${HOME}" USER="${USER}" PATH=/usr/bin:/bin \
    OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
    bash -c 'set -euo pipefail; source "$1/env/site.sh"; exec python "$1/scripts/hpc/check_assets.py" --full' \
    check_assets "${CLONE}"
