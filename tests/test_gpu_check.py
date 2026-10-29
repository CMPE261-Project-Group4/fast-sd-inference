import importlib.util
import json
from pathlib import Path

import pytest

from eldm.config import load_yaml
from eldm.prompts import load_prompts

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("gpu_check", ROOT / "scripts/hpc/gpu_check.py")
gpu_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpu_check)


def test_smoke_prompts_are_fixed():
    prompts = gpu_check.smoke_prompts()
    assert [(p.id, p.seed, p.source) for p in prompts] == [(0, 1000, "manual"), (1, 1001, "manual")]
    assert prompts[0].prompt == "a red vintage bicycle leaning against a brick wall"


def test_variant_configs(tmp_path):
    base = load_yaml(ROOT / "configs/base.yaml")
    cfgs = gpu_check.variant_configs(base, tmp_path / "p.jsonl", tmp_path / "stage")
    assert list(cfgs) == ["baseline", "fp16", "dpmpp_20", "deepcache_i3"]
    for name, cfg in cfgs.items():
        # Same prompts, seeds, model and output place for every variant; only the overrides differ.
        assert cfg["name"] == name and cfg["device"] == "cuda" and cfg["num_prompts"] == 2
        assert cfg["prompts_file"] == str(tmp_path / "p.jsonl") and cfg["output_root"] == str(tmp_path / "stage")
        assert cfg["model_revision"] == base["model_revision"]
    assert (cfgs["baseline"]["dtype"], cfgs["baseline"]["scheduler"], cfgs["baseline"]["num_inference_steps"]) == \
        ("fp32", "pndm", 50)
    assert cfgs["fp16"]["dtype"] == "fp16"
    assert (cfgs["dpmpp_20"]["scheduler"], cfgs["dpmpp_20"]["num_inference_steps"]) == ("dpmpp_2m", 20)
    assert cfgs["deepcache_i3"]["deepcache"] == {"enabled": True, "cache_interval": 3, "cache_branch_id": 0}
    assert base["deepcache"]["enabled"] is False  # base untouched


@pytest.fixture
def fake_gpu(monkeypatch):
    """Pretend an A100 is present, and replace benchmark.run with a fast fake."""
    monkeypatch.setattr(gpu_check.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_check.torch.cuda, "get_device_name", lambda i=0: "NVIDIA A100-PCIE-40GB")
    latencies = {"baseline": 4.0, "fp16": 1.5, "dpmpp_20": 1.6, "deepcache_i3": 2.2}
    failing = set()

    def fake_run(cfg):
        out = Path(cfg["output_root"]) / cfg["name"]
        (out / "images").mkdir(parents=True)
        (out / "perf.json").write_text("{}")
        if cfg["name"] in failing:
            raise RuntimeError("boom")
        assert load_prompts(cfg["prompts_file"], cfg["num_prompts"])  # prompts were written first
        return {"name": cfg["name"], "latency_per_image_s": latencies[cfg["name"]], "peak_mem_gb": 3.0}

    monkeypatch.setattr(gpu_check.benchmark, "run", fake_run)
    return latencies, failing


def test_main_passes_gate_and_copies_outputs(tmp_path, fake_gpu):
    out, stage = tmp_path / "out", tmp_path / "stage"
    assert gpu_check.main(["--out", str(out), "--stage", str(stage)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["gate_g1"] and summary["errors"] == {} and summary["gpu"] == "NVIDIA A100-PCIE-40GB"
    assert summary["variants"]["fp16"]["speedup"] == pytest.approx(4.0 / 1.5)
    for name in gpu_check.VARIANTS:
        assert (out / name / "perf.json").exists()


def test_main_fails_gate_when_not_faster(tmp_path, fake_gpu):
    latencies, _ = fake_gpu
    latencies["deepcache_i3"] = 4.5
    assert gpu_check.main(["--out", str(tmp_path / "out"), "--stage", str(tmp_path / "stage")]) == 1
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["not_faster_than_baseline"] == ["deepcache_i3"] and not summary["gate_g1"]


def test_main_keeps_going_after_a_failing_variant(tmp_path, fake_gpu):
    _, failing = fake_gpu
    failing.add("deepcache_i3")
    assert gpu_check.main(["--out", str(tmp_path / "out"), "--stage", str(tmp_path / "stage")]) == 1
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["errors"] == {"deepcache_i3": "RuntimeError: boom"}
    assert set(summary["variants"]) == {"baseline", "fp16", "dpmpp_20"}
    assert (tmp_path / "out" / "deepcache_i3" / "perf.json").exists()  # partial output kept for diagnosis


def test_main_refuses_other_gpus(tmp_path, fake_gpu, monkeypatch):
    monkeypatch.setattr(gpu_check.torch.cuda, "get_device_name", lambda i=0: "Tesla P100-PCIE-16GB")
    assert gpu_check.main(["--out", str(tmp_path / "out"), "--stage", str(tmp_path / "stage")]) == 1
    assert not (tmp_path / "out").exists()
