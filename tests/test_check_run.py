import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from eldm.prompts import Prompt, save_prompts

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_run", ROOT / "scripts/check_run.py")
check_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_run)


def make_run(tmp_path: Path, n: int = 2, gpu: str = "NVIDIA A100-PCIE-40GB") -> Path:
    """A run directory shaped like eldm.benchmark.run's output, which passes every check."""
    prompts = [Prompt(i, f"prompt {i}", "manual", "objects", 1000 + i) for i in range(n)]
    save_prompts(prompts, tmp_path / "prompts.jsonl")
    run = tmp_path / "baseline"
    (run / "images").mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(
        {"name": "baseline", "prompts_file": str(tmp_path / "prompts.jsonl"), "num_prompts": n}))
    rng = np.random.default_rng(0)
    for p in prompts:
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(run / "images" / f"{p.id:04d}.png")
    (run / "perf.json").write_text(json.dumps({
        "name": "baseline", "num_images": n, "latency_per_image_s": 2.5, "throughput_img_per_s": 0.4,
        "peak_mem_gb": 3.2, "env": {"device": "cuda", "gpu": gpu},
    }))
    return run


def update_json(path: Path, **changes) -> None:
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def test_good_run_passes(tmp_path):
    run = make_run(tmp_path)
    assert check_run.check_run(run, require_gpu="A100") == []
    assert check_run.main([str(run), "--require-gpu", "A100"]) == 0


def test_num_prompts_limits_expected_images(tmp_path):
    run = make_run(tmp_path, n=3)
    (run / "images" / "0002.png").unlink()
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    (run / "config.yaml").write_text(yaml.safe_dump({**cfg, "num_prompts": 2}))
    update_json(run / "perf.json", num_images=2)
    assert check_run.check_run(run) == []


@pytest.mark.parametrize("defect, expected", [
    ("missing", "1 of 2 images missing"),
    ("extra", "1 unexpected images"),
    ("blank", "0001.png is blank"),
    ("corrupt", "0001.png doesn't decode"),
])
def test_image_defects_fail(tmp_path, defect, expected):
    run = make_run(tmp_path)
    images = run / "images"
    if defect == "missing":
        (images / "0001.png").unlink()
    elif defect == "extra":
        Image.fromarray(np.full((64, 64, 3), 128, np.uint8)).save(images / "0099.png")
    elif defect == "blank":
        Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(images / "0001.png")
    elif defect == "corrupt":
        (images / "0001.png").write_bytes(b"not a png")
    problems = check_run.check_run(run)
    assert any(expected in p for p in problems), problems
    assert check_run.main([str(run)]) == 1


@pytest.mark.parametrize("changes, expected", [
    ({"latency_per_image_s": 0}, "latency_per_image_s is 0"),
    ({"throughput_img_per_s": None}, "throughput_img_per_s is None"),
    ({"num_images": 1}, "num_images 1, expected 2"),
    ({"peak_mem_gb": None}, "peak_mem_gb is None"),
    ({"env": {"device": "cuda"}}, "env.gpu is missing"),
    ({"name": "fp16"}, "name 'fp16' != config name 'baseline'"),
])
def test_perf_defects_fail(tmp_path, changes, expected):
    run = make_run(tmp_path)
    update_json(run / "perf.json", **changes)
    problems = check_run.check_run(run)
    assert any(expected in p for p in problems), problems


def test_missing_perf_and_wrong_gpu_fail(tmp_path):
    run = make_run(tmp_path, gpu="Tesla P100-PCIE-16GB")
    assert any("not A100" in p for p in check_run.check_run(run, require_gpu="A100"))
    (run / "perf.json").unlink()
    assert any("perf.json unreadable" in p for p in check_run.check_run(run))


def test_quality_only_when_expected(tmp_path):
    run = make_run(tmp_path)
    assert check_run.check_run(run) == []
    assert any("quality.json unreadable" in p for p in check_run.check_run(run, expect_quality=True))
    (run / "quality.json").write_text(json.dumps({"fid": 12.0}))
    assert check_run.check_run(run, expect_quality=True) == ["quality.json has no clip_score"]
    (run / "quality.json").write_text(json.dumps({"clip_score": 30.1}))
    assert check_run.check_run(run, expect_quality=True) == []


def test_unreadable_config_fails(tmp_path):
    run = make_run(tmp_path)
    (run / "config.yaml").unlink()
    problems = check_run.check_run(run)
    assert len(problems) == 1 and "config.yaml" in problems[0]
