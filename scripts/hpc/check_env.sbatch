#!/bin/bash
# Check that the shared venv works in a clean environment: no modules, nothing
# inherited from the submitting shell, only env/site.sh.
#
# On COE-HPC3, from the root of your clone on g17 (the log lands in its logs/):
#     sbatch scripts/hpc/check_env.sbatch
# On coe-hpc1 the same check runs directly (the #SBATCH lines are only comments):
#     bash scripts/hpc/check_env.sbatch
#
#SBATCH --job-name=eldm-check-env
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
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

# The payload runs under `env -i`, so it sees only what env/site.sh sets.
PAYLOAD=$(cat <<'EOF'
set -euo pipefail
CLONE="$1"
source "${CLONE}/env/site.sh"
source "${CLONE}/env/python.pin"

failures=0
pass() { printf '  ok    %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; failures=$((failures + 1)); }
check() {
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then pass "${desc}"; else fail "${desc}"; fi
}

echo "Clean environment on $(hostname) ($(. /etc/os-release && echo "${PRETTY_NAME}")," \
     "$(ldd --version | head -1 | awk '{print "glibc", $NF}'), umask $(umask)):"

# Resolve both sides: /data is a symlink on coe-hpc1 but the mount itself on HPC3.
real="$(readlink -f "${ELDM_VENV}/bin/python")"
pbs="$(readlink -f "${ELDM_PYTHON}")"
check "venv python is the portable CPython (${real})" test "${real#"${pbs}"/}" != "${real}"
check "python is ${PYTHON_VERSION} (env/python.pin)" \
    test "$(python -c 'import platform; print(platform.python_version())')" = "${PYTHON_VERSION}"
check "pip check: no broken requirements" python -m pip check
check "venv matches requirements.lock" \
    diff <(python -m pip freeze --exclude-editable) <(grep -vE '^[[:space:]]*(#|$)' "${CLONE}/requirements.lock")

python - <<'PY' || failures=$((failures + 1))
import importlib
import os
import sys
import time

failed = []

def attempt(desc, fn):
    t0 = time.perf_counter()
    try:
        info = fn()
    except Exception as e:  # report every failure, not just the first
        failed.append(desc)
        print(f"  FAIL  {desc}: {type(e).__name__}: {e}")
    else:
        print(f"  ok    {desc}{' ' + info if info else ''} ({time.perf_counter() - t0:.1f}s)")

def module(name):
    return lambda: getattr(importlib.import_module(name), "__version__", "")

def attr(mod, name):
    return lambda: getattr(importlib.import_module(mod), name) and ""

for name in ["torch", "torchvision", "diffusers", "transformers", "accelerate",
             "huggingface_hub", "safetensors", "DeepCache", "torchmetrics",
             "torch_fidelity", "lpips", "open_clip", "numpy", "scipy", "PIL",
             "pandas", "matplotlib", "yaml"]:
    attempt(f"import {name}", module(name))

# The classes the project uses (imports only: weights are Phase 2).
for mod, name in [
    ("diffusers", "StableDiffusionPipeline"),
    ("diffusers", "PNDMScheduler"),
    ("diffusers", "DDIMScheduler"),
    ("diffusers", "DPMSolverMultistepScheduler"),
    ("diffusers", "EulerDiscreteScheduler"),
    ("DeepCache", "DeepCacheSDHelper"),
    ("torchmetrics.multimodal.clip_score", "CLIPScore"),
    ("torchmetrics.image.fid", "FrechetInceptionDistance"),
    ("torchmetrics.image.kid", "KernelInceptionDistance"),
    ("torchmetrics.image.lpip", "LearnedPerceptualImagePatchSimilarity"),
    ("open_clip", "create_model_and_transforms"),
    ("eldm.config", "load_experiment"),
    ("eldm.prompts", "load_prompts"),
    ("eldm.pipeline", "build_pipeline"),
    ("eldm.benchmark", "run"),
    ("eldm.metrics", "ClipScorer"),
]:
    attempt(f"{mod}.{name}", attr(mod, name))

def torch_info():
    import torch
    x = torch.randn(256, 256)
    assert torch.allclose(x @ torch.eye(256), x)
    return (f"cuda build {torch.version.cuda}, cudnn {torch.backends.cudnn.version()},"
            f" cuda available {torch.cuda.is_available()}, {torch.get_num_threads()} threads")
attempt("torch CPU matmul", torch_info)

def offline():
    import huggingface_hub.constants as c
    assert c.HF_HUB_OFFLINE, "HF_HUB_OFFLINE is not set"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1", "TRANSFORMERS_OFFLINE is not set"
    return f"HF_HOME={c.HF_HOME}"
attempt("offline flags", offline)

sys.exit(1 if failed else 0)
PY

if (( failures )); then
    echo "${failures} check(s) failed."
    exit 1
fi
echo "All checks passed."
EOF
)

env -i HOME="${HOME}" USER="${USER}" PATH=/usr/bin:/bin bash -c "${PAYLOAD}" check_env "${CLONE}"
