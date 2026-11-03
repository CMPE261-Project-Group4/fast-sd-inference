import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def python_call(job: str, script: str) -> str:
    """The job's `python scripts/<script> ...` command, with its continuation lines joined."""
    text = (ROOT / "scripts/hpc" / job).read_text().replace("\\\n", " ")
    [call] = [line for line in text.splitlines() if re.match(rf"python scripts/{re.escape(script)} ", line)]
    return " ".join(call.split())


@pytest.mark.parametrize("job, script", [("run_gpu.slurm", "run_experiments.py"),
                                         ("run_array.sbatch", "run_experiments.py"),
                                         ("evaluate.slurm", "evaluate_quality.py")])
def test_gpu_jobs_require_cuda(job, script):
    """Device "auto" ran job 99031 on the CPU when CUDA couldn't use its GPU; GPU jobs ask for cuda."""
    assert "--device cuda" in python_call(job, script)


@pytest.mark.parametrize("job", ["run_gpu.slurm", "run_array.sbatch"])
def test_runs_are_staged_on_the_node(job):
    assert '--stage-dir "${STAGE}"' in python_call(job, "run_experiments.py")


def test_array_task_runs_its_index_unless_done():
    call = python_call("run_array.sbatch", "run_experiments.py")
    assert "--skip-existing" in call
    # After "$@": its experiment files would otherwise be read as more indices.
    assert re.search(r'"\$@" --index "\$\{SLURM_ARRAY_TASK_ID:\?[^}]*\}"$', call)
