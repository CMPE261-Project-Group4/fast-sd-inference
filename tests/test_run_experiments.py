import importlib.util
from pathlib import Path

import pytest

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
    monkeypatch.setattr(run_experiments, "run", lambda cfg: seen.append(cfg) or {"name": cfg["name"]})
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
