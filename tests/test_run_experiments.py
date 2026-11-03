import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from eldm.prompts import load_prompts

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_experiments", ROOT / "scripts/run_experiments.py")
run_experiments = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_experiments)


@pytest.fixture
def runs(monkeypatch):
    """Replace benchmark.run with a recorder; the experiment files name paths from the repo root."""
    monkeypatch.chdir(ROOT)
    seen = []
    monkeypatch.setattr(run_experiments, "run", lambda cfg, stage_dir=None: seen.append(cfg) or {"name": cfg["name"]})
    return seen


def test_prompts_file_overrides_every_run(runs):
    debug = "data/prompts/subsets/debug.jsonl"
    assert run_experiments.main(["configs/experiments/00_baseline.yaml", "--prompts-file", debug,
                                 "--output-root", "/scratch/p5/debug_a"]) == 0
    assert [(c["name"], c["batch_size"]) for c in runs] == [("baseline", 1), ("baseline_bs4", 4)]
    for cfg in runs:
        assert cfg["prompts_file"] == debug and cfg["output_root"] == "/scratch/p5/debug_a"
        assert cfg["num_prompts"] is None and len(load_prompts(cfg["prompts_file"])) == 50


def test_defaults_and_only(runs):
    run_experiments.main(["configs/experiments/03_precision.yaml", "--only", "fp16", "--num-prompts", "4"])
    assert len(runs) == 1
    cfg = runs[0]
    assert (cfg["name"], cfg["prompts_file"], cfg["num_prompts"]) == ("fp16", "data/prompts/eval_prompts.jsonl", 4)
    with pytest.raises(SystemExit):
        run_experiments.main(["configs/experiments/03_precision.yaml", "--only", "fp61"])
    assert len(runs) == 1  # a typo in --only runs nothing


PILOT = "configs/experiments/pilot.yaml"
SWEEP = [f"configs/experiments/{f}.yaml" for f in
         ("00_baseline", "01_steps", "02_schedulers", "03_precision", "04_deepcache")]


def finish(cfg, config=None, perf=True):
    """Make cfg's output directory look like a run that wrote config.yaml (and perf.json, last)."""
    out = Path(cfg["output_root"]) / cfg["name"]
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config or cfg, sort_keys=False))
    if perf:
        (out / "perf.json").write_text("{}")


@pytest.fixture
def pilot(runs, tmp_path):
    """The pilot's five resolved configs under a fresh output root, and the arguments that made them."""
    args = [PILOT, "--output-root", str(tmp_path / "out"), "--num-prompts", "4", "--device", "cuda"]
    run_experiments.main(args)
    cfgs = {cfg["name"]: cfg for cfg in runs}
    runs.clear()
    return args, cfgs


def test_list_shows_each_runs_state(pilot, runs, capsys):
    args, cfgs = pilot
    finish(cfgs["baseline"])
    finish(cfgs["ddim_20"], perf=False)  # stopped before perf.json: not finished
    capsys.readouterr()
    assert run_experiments.main(args + ["--list"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "0\tbaseline\tdone", "1\tddim_20\ttodo", "2\tdpmpp_20\ttodo", "3\tdeepcache_i3\ttodo", "4\tfp16\ttodo"]
    # Another config finished there: listed as a conflict, and --list fails.
    finish(cfgs["fp16"], config={**cfgs["fp16"], "num_prompts": 8})
    assert run_experiments.main(args + ["--list"]) == 1
    out = capsys.readouterr()
    assert out.out.splitlines()[-1] == "4\tfp16\tconflict" and "fp16" in out.err
    assert runs == []  # --list runs nothing


def test_skip_existing(pilot, runs):
    args, cfgs = pilot
    finish(cfgs["baseline"])
    assert run_experiments.main(args + ["--index", "0", "1", "--skip-existing"]) == 0
    assert [cfg["name"] for cfg in runs] == ["ddim_20"]
    # Without --skip-existing a finished run is redone, as before.
    runs.clear()
    run_experiments.main(args + ["--index", "0"])
    assert [cfg["name"] for cfg in runs] == ["baseline"]
    # A run finished with another config stops the job before it runs anything.
    runs.clear()
    finish(cfgs["fp16"], config={**cfgs["fp16"], "dtype": "fp32"})
    assert run_experiments.main(args + ["--index", "4", "--skip-existing"]) == 1
    assert runs == []


def test_index_and_only(pilot, runs, capsys):
    args, _ = pilot
    run_experiments.main(args + ["--index", "4", "2"])
    assert [cfg["name"] for cfg in runs] == ["dpmpp_20", "fp16"]  # in list order
    capsys.readouterr()
    # Indices are positions in the whole list, also under --only.
    run_experiments.main(args + ["--only", "fp16", "ddim_20", "--list"])
    assert capsys.readouterr().out.splitlines() == ["1\tddim_20\ttodo", "4\tfp16\ttodo"]
    for bad in (["--index", "5"], ["--index", "-1"], ["--index", "0", "--only", "fp16"]):
        with pytest.raises(SystemExit):
            run_experiments.main(args + bad)


def test_several_experiment_files_make_one_list(runs, tmp_path, capsys):
    out = ["--output-root", str(tmp_path)]
    run_experiments.main(["configs/experiments/00_baseline.yaml", "configs/experiments/03_precision.yaml", "--list"] + out)
    assert capsys.readouterr().out.splitlines() == [
        "0\tbaseline\ttodo", "1\tbaseline_bs4\ttodo", "2\tfp16\ttodo", "3\tfp16_bs4\ttodo"]
    # Phase 8's sweep: one list of 17 runs.
    run_experiments.main(SWEEP + ["--list"] + out)
    assert len(capsys.readouterr().out.splitlines()) == 17
    with pytest.raises(SystemExit):  # baseline, ddim_20, ... are in both
        run_experiments.main([PILOT, "configs/experiments/00_baseline.yaml", "--list"] + out)
    assert runs == []


def test_list_does_not_import_torch(tmp_path):
    """submit_experiment.sh runs --list on the head node, where loading torch takes seconds."""
    out = subprocess.run(
        [sys.executable, "-X", "importtime", "scripts/run_experiments.py", PILOT, "--list", "--output-root", str(tmp_path)],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True, check=True)
    imported = {line.rsplit("|", 1)[-1].strip() for line in out.stderr.splitlines() if line.startswith("import time:")}
    assert "eldm.config" in imported and "torch" not in imported
    assert out.stdout.splitlines()[0] == "0\tbaseline\ttodo"


def test_stage_dir_reaches_every_run(monkeypatch):
    monkeypatch.chdir(ROOT)
    calls = []
    monkeypatch.setattr(run_experiments, "run", lambda cfg, stage_dir=None: calls.append((cfg["name"], stage_dir)) or {})
    run_experiments.main(["configs/experiments/03_precision.yaml", "--stage-dir", "/tmp/eldm-1_2"])
    assert calls == [("fp16", "/tmp/eldm-1_2"), ("fp16_bs4", "/tmp/eldm-1_2")]
