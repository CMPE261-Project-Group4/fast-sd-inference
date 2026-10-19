# Run-time environment for fast-sd-inference, valid on both coe-hpc1 and COE-HPC3.
#
# Source this file; do not execute it:
#     source env/site.sh
#
# It loads no modules: jobs on COE-HPC3 must not depend on coe-hpc1's module tree.
# Everything lives on shared storage (/data, /scratch), which both clusters mount at
# the same paths. Safe under `set -euo pipefail`.

umask 002   # teammates in group cmpe261-fa26 can edit what we create

export ELDM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Shared group areas. /home is ~99% full, so nothing large or cached goes there.
export ELDM_SHARED="${ELDM_SHARED:-/data/cmpe261-fa26/group4}"
export ELDM_PYTHON="${ELDM_SHARED}/env/python"     # portable CPython (env/python.pin)
export ELDM_VENV="${ELDM_SHARED}/env/venv"         # built from requirements.lock
export ELDM_DATASETS="${ELDM_SHARED}/datasets"
export ELDM_RESULTS="${ELDM_RESULTS:-/scratch/cmpe261-fa26/group4/results}"

# Tool caches.
export HF_HOME="${ELDM_SHARED}/hf_cache"
export TORCH_HOME="${ELDM_SHARED}/cache/torch"
export PIP_CACHE_DIR="${ELDM_SHARED}/cache/pip"
export XDG_CACHE_HOME="${ELDM_SHARED}/cache/xdg"
export MPLCONFIGDIR="${ELDM_SHARED}/cache/matplotlib"

# Jobs run offline (COE-HPC3 has no internet): fail fast instead of hanging on the
# network. Download scripts on coe-hpc1 override these for their own process only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Use only the venv: no ~/.local packages, and this clone's src/ on the path, so each
# teammate's clone runs its own code against the one shared venv.
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ELDM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export VIRTUAL_ENV="${ELDM_VENV}"
export PATH="${ELDM_VENV}/bin:${PATH}"
