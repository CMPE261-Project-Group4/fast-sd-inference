import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("job, script", [("run_gpu.slurm", "run_experiments.py"),
                                         ("evaluate.slurm", "evaluate_quality.py")])
def test_gpu_jobs_require_cuda(job, script):
    """Device "auto" ran job 99031 on the CPU when CUDA couldn't use its GPU; GPU jobs ask for cuda."""
    lines = (ROOT / "scripts/hpc" / job).read_text().splitlines()
    [call] = [line for line in lines if re.match(rf"python scripts/{re.escape(script)} ", line)]
    assert "--device cuda" in call
