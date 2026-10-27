# Common start of every fast-sd-inference Slurm job. Source it from the job script, right
# after finding the clone (job scripts only; it re-runs "$0"):
#
#     CLONE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
#     source "${CLONE}/scripts/hpc/job_prelude.sh"
#
# It then:
#   1. Re-runs the job script once under `env -i`, keeping only Slurm's variables,
#      CUDA_VISIBLE_DEVICES and TMPDIR. Nothing from the submitting shell (modules,
#      LD_LIBRARY_PATH, PYTHONPATH) reaches the job. CUDA_VISIBLE_DEVICES must survive:
#      cgroups don't isolate GPUs here, so it alone keeps the job on its own GPU.
#   2. Sources env/site.sh: paths, caches, offline flags, the venv.
#   3. Numbers GPUs in PCI order, as nvidia-smi and Slurm do, and matches thread counts
#      to the allocation (Slurm doesn't confine CPUs here either).
#   4. Creates STAGE, a per-job directory on the node-local SSD, removed on exit: /tmp is
#      shared by every job on the node and isn't cleaned per job. Job scripts must not
#      set their own EXIT trap.
#   5. Logs the job, node, commit, GPU and the other jobs on the node.

if [[ -z "${ELDM_CLEAN_ENV:-}" ]]; then
    mapfile -t _eldm_keep < <(env | grep -E '^(SLURM_[A-Za-z0-9_]*|CUDA_VISIBLE_DEVICES|TMPDIR)=' || true)
    # ${a[@]+"${a[@]}"}: bash < 4.4 (coe-hpc1 has 4.2) calls an empty array unbound under set -u.
    exec env -i HOME="${HOME:-}" USER="${USER:-$(id -un)}" PATH=/usr/bin:/bin ELDM_CLEAN_ENV=1 \
        ${_eldm_keep[@]+"${_eldm_keep[@]}"} bash "$0" "$@"
fi

source "${CLONE}/env/site.sh"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
fi

STAGE="${TMPDIR:-/tmp}/eldm-${SLURM_JOB_ID:-local-$$}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
export STAGE
mkdir -p "${STAGE}"
trap 'rm -rf "${STAGE}"' EXIT

echo "job=${SLURM_JOB_ID:-none} node=$(hostname) partition=${SLURM_JOB_PARTITION:-none}" \
     "cpus=${SLURM_CPUS_PER_TASK:-?} gpus=${CUDA_VISIBLE_DEVICES:-none} date=$(date -Is)" \
     "commit=$(cd "${CLONE}" && git rev-parse --short HEAD 2>/dev/null || echo unknown)$(
         cd "${CLONE}" && git diff --quiet HEAD 2>/dev/null || echo +uncommitted)"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null; then
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" \
        --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv || true
    echo "all GPUs on the node (index, utilization, memory used); other jobs' load shows here:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader || true
fi
if [[ -n "${SLURMD_NODENAME:-}" ]] && command -v squeue >/dev/null; then
    echo "jobs on ${SLURMD_NODENAME}:"
    squeue -h -w "${SLURMD_NODENAME}" -o "  %i %u %P cpus=%C %b" || true
fi
