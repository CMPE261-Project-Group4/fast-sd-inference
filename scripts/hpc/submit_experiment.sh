#!/bin/bash
# Run experiment files as a Slurm job array, one run per task, then evaluate them. Run it on g17
# from the root of your clone; it submits the jobs and returns:
#
#     scripts/hpc/submit_experiment.sh --output-root /scratch/cmpe261-fa26/group4/results/p7/check \
#         --time 00:10:00 configs/experiments/pilot.yaml -- --num-prompts 4
#
# It lists the runs (scripts/run_experiments.py --list) and submits scripts/hpc/run_array.sbatch
# as an array of the runs not done yet, at most --max-parallel at once. Then it submits
# scripts/hpc/evaluate.slurm, which starts only if every task succeeds; Slurm cancels it if one
# fails. To resume after a failed or cancelled task, run the same command again: finished runs
# are skipped, and with every run done it submits only the evaluation. Several experiment files
# make one array. Arguments after -- go to run_experiments.py.
#
# Options:
#   --output-root DIR     where the runs go, one directory each (required)
#   --time HH:MM:SS       each task's time limit: the longest run's, with headroom (required)
#   --baseline RUN_DIR    the run LPIPS and speedups compare to (default DIR/baseline, if it's
#                         listed or already there)
#   --eval-time HH:MM:SS  the evaluation's time limit (default 00:30:00)
#   --max-parallel N      tasks running at once (default 4)
#   --exclude NODES       nodes to avoid (default cs001,cs002, whose GPUs fail: docs/hpc/PROGRESS.md
#                         open issues); --exclude '' avoids none
#   --dry-run             list the runs and print the sbatch commands; submit nothing

set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }
usage() { sed -n '2,/^$/s/^# \{0,1\}//p' "$0" >&2; exit 2; }
show() { local IFS=' '; echo "  $*"; }
# Consecutive indices as ranges, as --array takes them: 0 1 2 5 7 8 -> 0-2,5,7-8.
ranges() {
    local out="" start="" prev="" i
    for i in "$@" ""; do
        if [[ -n "${i}" && -n "${prev}" && "${i}" -eq $((prev + 1)) ]]; then
            prev="${i}"
            continue
        fi
        if [[ -n "${start}" ]]; then
            out+="${out:+,}${start}"
            if [[ "${prev}" != "${start}" ]]; then out+="-${prev}"; fi
        fi
        start="${i}" prev="${i}"
    done
    echo "${out}"
}
absolute() { if [[ "$1" == /* ]]; then echo "${1%/}"; else echo "${PWD}/${1%/}"; fi; }

OUT="" TIME="" BASELINE="" EVAL_TIME="00:30:00" MAX_PARALLEL=4 EXCLUDE="cs001,cs002" DRY_RUN=0
EXPERIMENTS=() RUNNER_ARGS=()
while (( $# )); do
    case "$1" in
        --output-root|--time|--baseline|--eval-time|--max-parallel|--exclude)
            (( $# >= 2 )) || die "$1 needs a value"
            case "$1" in
                --output-root) OUT="$2" ;;
                --time) TIME="$2" ;;
                --baseline) BASELINE="$2" ;;
                --eval-time) EVAL_TIME="$2" ;;
                --max-parallel) MAX_PARALLEL="$2" ;;
                --exclude) EXCLUDE="$2" ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --) shift; RUNNER_ARGS=("$@"); break ;;
        -h|--help) usage ;;
        -*) echo "error: unknown option $1" >&2; usage ;;
        *) EXPERIMENTS+=("$1"); shift ;;
    esac
done
[[ -n "${OUT}" && -n "${TIME}" ]] && (( ${#EXPERIMENTS[@]} )) || usage
time_re='^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$'
[[ "${TIME}" =~ ${time_re} && "${EVAL_TIME}" =~ ${time_re} ]] || die "times are [D-]HH:MM:SS"
[[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]] || die "--max-parallel must be a positive number"
[[ -f scripts/hpc/run_array.sbatch ]] || die "run it from the root of your clone"

source env/site.sh
OUT="$(absolute "${OUT}")"
TAG="$(basename "${OUT}")"
EXCLUDE_ARGS=()
if [[ -n "${EXCLUDE}" ]]; then EXCLUDE_ARGS=(--exclude="${EXCLUDE}"); fi

# Every run records the commit it ran; uncommitted changes would make that a lie.
if [[ -n "$(git status --porcelain)" ]]; then
    (( DRY_RUN )) || die "the clone has uncommitted changes (git status); commit or stash them first"
    echo "warning: the clone has uncommitted changes; a real submission would stop here" >&2
fi
# Two arrays on one output root would generate the same runs twice. coe-hpc1's squeue sees
# another cluster, so a dry run there can't check this.
if (( ! DRY_RUN )); then
    queued="$(squeue -h -u "${USER}" -n "eldm-${TAG},eldm-eval-${TAG}" -o '%i %j %T')"
    [[ -z "${queued}" ]] || die "jobs for ${TAG} are already queued or running; wait for them or scancel them:
${queued}"
fi

# The overrides the tasks get from run_array.sbatch, so the states are theirs.
status=0
LIST="$(python scripts/run_experiments.py "${EXPERIMENTS[@]}" --output-root "${OUT}" --device cuda \
    ${RUNNER_ARGS[@]+"${RUNNER_ARGS[@]}"} --list)" || status=$?
echo "runs in ${OUT}:"
sed 's/^/  /' <<< "${LIST}"
(( status == 0 )) || die "run_experiments.py --list failed; nothing submitted"
mapfile -t TODO < <(awk -F'\t' '$3 == "todo" {print $1}' <<< "${LIST}")

if [[ -z "${BASELINE}" ]]; then
    [[ -f "${OUT}/baseline/config.yaml" ]] || awk -F'\t' '$2 == "baseline" {found = 1} END {exit !found}' <<< "${LIST}" \
        || die "no run named baseline in these experiment files or in ${OUT}; pass --baseline RUN_DIR"
    BASELINE="${OUT}/baseline"
fi
BASELINE="$(absolute "${BASELINE}")"

ARRAY_CMD=()
if (( ${#TODO[@]} )); then
    ARRAY_CMD=(sbatch --parsable --job-name="eldm-${TAG}" --array="$(ranges "${TODO[@]}")%${MAX_PARALLEL}"
               --time="${TIME}" ${EXCLUDE_ARGS[@]+"${EXCLUDE_ARGS[@]}"} scripts/hpc/run_array.sbatch
               "${EXPERIMENTS[@]}" --output-root "${OUT}" ${RUNNER_ARGS[@]+"${RUNNER_ARGS[@]}"})
fi
eval_cmd() {  # $1: the array's job ID, or nothing when every run is done
    EVAL_CMD=(sbatch --parsable --job-name="eldm-eval-${TAG}")
    if [[ -n "$1" ]]; then EVAL_CMD+=(--dependency="afterok:$1" --kill-on-invalid-dep=yes); fi
    EVAL_CMD+=(--time="${EVAL_TIME}" ${EXCLUDE_ARGS[@]+"${EXCLUDE_ARGS[@]}"} scripts/hpc/evaluate.slurm
               "${OUT}" "${BASELINE}")
}

if (( DRY_RUN )); then
    echo "would submit:"
    if (( ${#ARRAY_CMD[@]} )); then
        show "${ARRAY_CMD[@]}"
        eval_cmd "<ARRAY_ID>"
    else
        eval_cmd ""
    fi
    show "${EVAL_CMD[@]}"
    exit 0
fi

ARRAY_ID=""
if (( ${#ARRAY_CMD[@]} )); then
    ARRAY_ID="$("${ARRAY_CMD[@]}")"
    ARRAY_ID="${ARRAY_ID%%;*}"
    echo "array ${ARRAY_ID}: ${#TODO[@]} run(s), at most ${MAX_PARALLEL} at once, ${TIME} each;" \
         "logs/eldm-${TAG}-${ARRAY_ID}_<index>.out"
else
    echo "every run is done; submitting only the evaluation"
fi
eval_cmd "${ARRAY_ID}"
EVAL_ID="$("${EVAL_CMD[@]}")" \
    || die "the evaluation wasn't submitted; once array ${ARRAY_ID} finishes, run this command again"
EVAL_ID="${EVAL_ID%%;*}"
echo "evaluation ${EVAL_ID}:${ARRAY_ID:+ starts when every task of ${ARRAY_ID} succeeds;}" \
     "logs/eldm-eval-${TAG}-${EVAL_ID}.out"
echo "watch: squeue -u ${USER} -r    results: ${OUT}"
echo "to resume after a failed or cancelled task, run this same command again"
