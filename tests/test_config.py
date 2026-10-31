from pathlib import Path

import pytest

from eldm.config import deep_merge, load_experiment
from eldm.prompts import Prompt, load_prompts, save_prompts

ROOT = Path(__file__).resolve().parents[1]


def test_deep_merge_nested():
    base = {"a": 1, "deepcache": {"enabled": False, "cache_interval": 3}}
    out = deep_merge(base, {"deepcache": {"enabled": True}})
    assert out["deepcache"] == {"enabled": True, "cache_interval": 3}
    assert base["deepcache"]["enabled"] is False  # base untouched


@pytest.mark.parametrize("path", sorted((ROOT / "configs/experiments").glob("*.yaml")), ids=lambda p: p.name)
def test_experiment_files_resolve(path):
    runs = load_experiment(path, root=ROOT)
    assert runs
    for cfg in runs:
        assert cfg["scheduler"] in {"pndm", "ddim", "dpmpp_2m", "euler", "euler_a"}
        assert cfg["dtype"] in {"fp32", "fp16"}
        assert "cache_branch_id" in cfg["deepcache"]
        assert cfg["tf32"] == {"matmul": False, "cudnn": False}


def test_run_names_unique_across_files():
    names = [r["name"] for p in (ROOT / "configs/experiments").glob("*.yaml") for r in load_experiment(p, ROOT)]
    assert len(names) == len(set(names))


def test_prompts_roundtrip(tmp_path):
    prompts = [Prompt(0, "a cat", "manual", "animals", 1000), Prompt(1, "a dog", "coco", "coco", 1001, 42)]
    save_prompts(prompts, tmp_path / "p.jsonl")
    assert load_prompts(tmp_path / "p.jsonl") == prompts
    assert len(load_prompts(tmp_path / "p.jsonl", limit=1)) == 1
