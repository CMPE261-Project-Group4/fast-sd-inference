import os
import subprocess
from pathlib import Path

import yaml

from eldm.config import load_experiment

ROOT = Path(__file__).resolve().parents[1]
PILOT = "configs/experiments/pilot.yaml"


def submit(*args: str) -> subprocess.CompletedProcess:
    """submit_experiment.sh --dry-run from the clone, in a login shell's bare environment."""
    env = {"HOME": os.environ.get("HOME", "/tmp"), "USER": "tester", "PATH": "/usr/bin:/bin"}
    return subprocess.run(["bash", "scripts/hpc/submit_experiment.sh", "--dry-run", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True)


def sbatch_lines(out: str) -> list[list[str]]:
    return [line.split() for line in out.splitlines() if line.strip().startswith("sbatch ")]


def finish(out_root: Path, index: int, **changes) -> None:
    """Make a pilot run look finished in out_root, as a task with --num-prompts 4 leaves it."""
    cfg = load_experiment(ROOT / PILOT, root=ROOT)[index]
    cfg.update(output_root=str(out_root), device="cuda", num_prompts=4)
    run_dir = out_root / cfg["name"]
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, **changes}, sort_keys=False))
    (run_dir / "perf.json").write_text("{}")


def test_array_holds_only_the_runs_not_done(tmp_path):
    out = tmp_path / "check"
    finish(out, 0)  # baseline
    finish(out, 4)  # fp16
    result = submit("--output-root", str(out), "--time", "00:10:00", PILOT, "--", "--num-prompts", "4")
    assert result.returncode == 0, result.stderr
    assert "0\tbaseline\tdone" in result.stdout and "1\tddim_20\ttodo" in result.stdout
    array, evaluation = sbatch_lines(result.stdout)
    assert array == ["sbatch", "--parsable", "--job-name=eldm-check", "--array=1-3%4", "--time=00:10:00",
                     "--exclude=cs001,cs002", "scripts/hpc/run_array.sbatch", PILOT, "--output-root", str(out),
                     "--num-prompts", "4"]
    assert evaluation == ["sbatch", "--parsable", "--job-name=eldm-eval-check", "--dependency=afterok:<ARRAY_ID>",
                          "--kill-on-invalid-dep=yes", "--time=00:30:00", "--exclude=cs001,cs002",
                          "scripts/hpc/evaluate.slurm", str(out), str(out / "baseline")]


def test_every_run_done_submits_only_the_evaluation(tmp_path):
    out = tmp_path / "check"
    for i in range(5):
        finish(out, i)
    result = submit("--output-root", str(out), "--time", "00:10:00", "--eval-time", "00:15:00",
                    "--exclude", "", PILOT, "--", "--num-prompts", "4")
    assert result.returncode == 0, result.stderr
    assert sbatch_lines(result.stdout) == [
        ["sbatch", "--parsable", "--job-name=eldm-eval-check", "--time=00:15:00", "scripts/hpc/evaluate.slurm",
         str(out), str(out / "baseline")]]


def test_a_conflict_submits_nothing(tmp_path):
    out = tmp_path / "check"
    finish(out, 4, dtype="fp32")  # fp16's directory holds a run with another config
    result = submit("--output-root", str(out), "--time", "00:10:00", PILOT, "--", "--num-prompts", "4")
    assert result.returncode != 0
    assert "4\tfp16\tconflict" in result.stdout and sbatch_lines(result.stdout) == []


def test_runner_arguments_decide_what_is_done(tmp_path):
    # baseline finished with 4 prompts; without --num-prompts 4 it's another config.
    out = tmp_path / "check"
    finish(out, 0)
    result = submit("--output-root", str(out), "--time", "02:00:00", "--max-parallel", "2", PILOT)
    assert result.returncode != 0 and "0\tbaseline\tconflict" in result.stdout
    result = submit("--output-root", str(out), "--time", "02:00:00", "--max-parallel", "2", PILOT, "--", "--only",
                    "ddim_20", "fp16")
    assert result.returncode == 0, result.stderr
    array, evaluation = sbatch_lines(result.stdout)
    assert "--array=1,4%2" in array
    assert evaluation[-1] == str(out / "baseline")  # not listed, but already in the output root


def test_usage_errors(tmp_path):
    out = str(tmp_path / "x")
    for args in ([PILOT, "--output-root", out],                      # no --time
                 ["--time", "00:10:00", PILOT],                      # no --output-root
                 ["--output-root", out, "--time", "00:10:00"],       # no experiment file
                 ["--output-root", out, "--time", "10 minutes", PILOT],
                 ["--output-root", out, "--time", "00:10:00", "--max-parallel", "0", PILOT],
                 ["--output-root", out, "--time", "00:10:00", "--bogus", PILOT]):
        result = submit(*args)
        assert result.returncode != 0 and sbatch_lines(result.stdout) == [], args
    # No run named baseline in 03_precision: LPIPS needs one named.
    result = submit("--output-root", out, "--time", "00:10:00", "configs/experiments/03_precision.yaml")
    assert result.returncode != 0 and "--baseline" in result.stderr
    result = submit("--output-root", out, "--time", "00:10:00", "--baseline", "/r/p8/sweep/baseline",
                    "configs/experiments/03_precision.yaml")
    assert result.returncode == 0 and sbatch_lines(result.stdout)[1][-1] == "/r/p8/sweep/baseline"
