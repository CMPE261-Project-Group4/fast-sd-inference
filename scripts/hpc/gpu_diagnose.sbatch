#!/bin/bash
# Diagnose a job that can't use its GPU, e.g. "CUDA-capable device(s) is/are busy or
# unavailable": what Slurm allocated and set, which device files the job can open, the
# GPU's health counters, and whether CUDA can allocate on it under Slurm's raw environment
# and under job_prelude.sh's cleaned one. Submit from the root of your clone on g17:
#
#     sbatch scripts/hpc/gpu_diagnose.sbatch                    # or add --nodelist=<node>
#
# Reading section "CUDA allocation":
#   A fails too   -> the GPU or the node, not our code: resubmit with --exclude=<node> and
#                    report the node, GPU index and UUID to the HPC admins.
#   A ok, C fails -> the prelude's env -i drops something CUDA needs.
#   C ok, D fails -> CUDA_DEVICE_ORDER=PCI_BUS_ID.
# CUDA only ever runs on the GPU allocated to this job. (Job 98023 found GPU 0 on cs002
# broken: A–D all failed, while GPUs 1–3 allocated fine.)
#
# It deliberately does NOT source job_prelude.sh: it compares Slurm's raw environment
# with the prelude's cleaned one.
#
#SBATCH --job-name=eldm-gpu-diagnose
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

# No `set -e`: every probe runs and reports, even after a failure.
set -uo pipefail
PY=/data/cmpe261-fa26/group4/env/venv/bin/python

section() { printf '\n========== %s\n' "$1"; }
probe() {
    printf -- '--- $ %s\n' "$1"
    bash -c "$1" 2>&1
    printf -- '(exit %s)\n' "$?"
}

# One fresh Python process: what CUDA enumerates, and whether a real allocation works.
TORCH_PROBE=$(cat <<'EOF'
import json, os
import torch
r = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
try:
    r["device_count"] = torch.cuda.device_count()
    r["devices"] = [f"{torch.cuda.get_device_name(i)} {getattr(torch.cuda.get_device_properties(i), 'uuid', '?')}"
                    for i in range(r["device_count"])]
except Exception as e:
    r["enumerate"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
try:
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    r["alloc"] = "ok"
except Exception as e:
    r["alloc"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
print(json.dumps(r))
EOF
)
export TORCH_PROBE PY

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "FAIL: CUDA_VISIBLE_DEVICES is not set; this must run as a GPU job (sbatch)"
    exit 1
fi

# The prelude's environment: env -i, keeping only SLURM_*, CUDA_VISIBLE_DEVICES and TMPDIR.
mapfile -t KEEP < <(env | grep -E '^(SLURM_[A-Za-z0-9_]*|CUDA_VISIBLE_DEVICES|TMPDIR)=' || true)
clean() { env -i HOME="${HOME}" USER="${USER}" PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
    ${KEEP[@]+"${KEEP[@]}"} "$@"; }

section "Job"
probe 'echo "job=${SLURM_JOB_ID:-none} node=$(hostname) date=$(date -Is)"; uname -r'
probe 'scontrol show job -d "${SLURM_JOB_ID}" | grep -iE "gres|tres|Nodes="'

section "Slurm's GPU-related environment (raw)"
probe 'env | grep -E "^(CUDA|GPU|NVIDIA|ROCR|SLURM_(JOB_GPUS|STEP_GPUS|GPUS|GPU_)|LD_LIBRARY_PATH)" | sort'
probe 'echo "variables the prelude drops: $(compgen -e | grep -vE "^(SLURM_.*|CUDA_VISIBLE_DEVICES|TMPDIR|TORCH_PROBE|PY)$" | sort | tr "\n" " ")"'

section "Devices and cgroup"
probe 'nvidia-smi --query-gpu=index,pci.bus_id,uuid,compute_mode,memory.used,utilization.gpu --format=csv'
probe 'nvidia-smi -q -i "${CUDA_VISIBLE_DEVICES}" -d ECC,PAGE_RETIREMENT,ROW_REMAPPER'
probe 'ls -l /dev/nvidia*'
probe 'cat /proc/self/cgroup'
# Opening a device file creates no CUDA context. If every GPU opens, cgroups don't confine
# GPUs and CUDA_VISIBLE_DEVICES alone keeps a job on its own GPU.
probe 'for d in /dev/nvidia[0-9]*; do "$PY" -c "import os, sys; os.close(os.open(sys.argv[1], os.O_RDWR)); print(sys.argv[1], \"opens\")" "$d" 2>&1 | tail -1; done'

section "CUDA allocation on the allocated GPU, under each environment (one fresh process each)"
echo "A. Slurm's raw environment:"
"${PY}" -c "${TORCH_PROBE}" 2>&1 | tail -1
echo "B. raw + CUDA_DEVICE_ORDER=PCI_BUS_ID:"
CUDA_DEVICE_ORDER=PCI_BUS_ID "${PY}" -c "${TORCH_PROBE}" 2>&1 | tail -1
echo "C. prelude's clean environment (env -i, keep list):"
clean "${PY}" -c "${TORCH_PROBE}" 2>&1 | tail -1
echo "D. clean + CUDA_DEVICE_ORDER=PCI_BUS_ID (what job_prelude.sh sets up):"
clean CUDA_DEVICE_ORDER=PCI_BUS_ID "${PY}" -c "${TORCH_PROBE}" 2>&1 | tail -1

section "Done"
