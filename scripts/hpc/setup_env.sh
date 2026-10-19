#!/usr/bin/env bash
# Build the shared, module-free Python environment. Run on coe-hpc1 (needs internet).
#
#   scripts/hpc/setup_env.sh          install env/python.pin + requirements.lock if changed
#   scripts/hpc/setup_env.sh --lock   re-resolve requirements.in into requirements.lock
#
# Idempotent: the portable CPython and the venv each carry a stamp of what they were
# built from, and are rebuilt only when env/python.pin or requirements.lock changes.
# Paths come from env/site.sh: ${ELDM_SHARED}/env/{python,venv}, caches under
# ${ELDM_SHARED}/cache. The venv is built from the portable CPython, never from a
# module Python, so it also runs in COE-HPC3 jobs.
#
# The venv is shared by the team: do not rebuild it while jobs are using it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/env/site.sh"
source "${ROOT}/env/python.pin"

PIP_FLAGS=(--disable-pip-version-check --progress-bar off --only-binary=:all:)
LOCK="${ROOT}/requirements.lock"
LOCK_HEADER="# pip freeze --exclude-editable of a venv built from requirements.in with --only-binary=:all: on coe-hpc1. Do not edit."

log() { printf '[setup_env] %s\n' "$*"; }

make_dirs() {
    mkdir -p "${ELDM_SHARED}/env" "${HF_HOME}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" \
        "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${ELDM_DATASETS}/coco" \
        "${ELDM_DATASETS}/partiprompts" "${ELDM_RESULTS}"
}

install_python() {
    local stamp="${ELDM_PYTHON}/.pbs-sha256"
    if [[ "$(cat "${stamp}" 2>/dev/null)" == "${PBS_SHA256}" ]]; then
        log "python ${PYTHON_VERSION} up to date (${ELDM_PYTHON})"
        return
    fi
    log "installing python ${PYTHON_VERSION} (${PBS_RELEASE}) into ${ELDM_PYTHON}"
    local tmp="${ELDM_PYTHON}.new"
    rm -rf "${tmp}"
    mkdir -p "${tmp}"
    curl -fsSL --retry 3 -o "${tmp}/python.tar.gz" "${PBS_URL}"
    echo "${PBS_SHA256}  ${tmp}/python.tar.gz" | sha256sum -c --quiet -
    tar -xzf "${tmp}/python.tar.gz" -C "${tmp}" --strip-components=1
    rm "${tmp}/python.tar.gz"
    chmod -R g+w "${tmp}"
    echo "${PBS_SHA256}" > "${tmp}/.pbs-sha256"
    rm -rf "${ELDM_PYTHON}"
    mv "${tmp}" "${ELDM_PYTHON}"
}

# What the venv must be built from: the interpreter and the exact lock contents.
venv_key() {
    { echo "${PBS_SHA256}"; cat "${LOCK}"; } | sha256sum | cut -d' ' -f1
}

# The venv must contain exactly the locked packages, nothing more or less.
check_venv() {
    "${ELDM_VENV}/bin/python" -m pip check
    diff <("${ELDM_VENV}/bin/python" -m pip freeze --exclude-editable) \
         <(grep -vE '^[[:space:]]*(#|$)' "${LOCK}")
}

install_venv() {
    local stamp="${ELDM_VENV}/.lock-sha256" key
    key="$(venv_key)"
    if [[ "$(cat "${stamp}" 2>/dev/null)" == "${key}" ]]; then
        log "venv up to date (${ELDM_VENV})"
        return
    fi
    log "building venv ${ELDM_VENV} from requirements.lock"
    "${ELDM_PYTHON}/bin/python3" -m venv --clear "${ELDM_VENV}"
    "${ELDM_VENV}/bin/python" -m pip install "${PIP_FLAGS[@]}" --quiet -r "${LOCK}"
    check_venv
    echo "${key}" > "${stamp}"
    log "venv built: $("${ELDM_VENV}/bin/python" -m pip freeze --exclude-editable | wc -l) packages"
}

# Resolve requirements.in in a throwaway venv and freeze it into requirements.lock.
relock() {
    local tmp
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/eldm-lock.XXXXXX")"
    trap 'rm -rf "${tmp}"' EXIT
    log "resolving requirements.in in ${tmp}"
    "${ELDM_PYTHON}/bin/python3" -m venv "${tmp}/venv"
    "${tmp}/venv/bin/python" -m pip install "${PIP_FLAGS[@]}" --quiet -r "${ROOT}/requirements.in"
    "${tmp}/venv/bin/python" -m pip check
    { echo "${LOCK_HEADER}"; "${tmp}/venv/bin/python" -m pip freeze --exclude-editable; } > "${LOCK}"
    log "wrote ${LOCK}; run without --lock to rebuild the venv"
}

make_dirs
install_python
case "${1:-}" in
    "")     install_venv ;;
    --lock) relock ;;
    *)      echo "usage: scripts/hpc/setup_env.sh [--lock]" >&2; exit 2 ;;
esac
