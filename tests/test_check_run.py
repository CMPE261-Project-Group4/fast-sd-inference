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


ENV = {
    "device": "cuda", "node": "cs001", "torch": "2.6.0+cu124", "diffusers": "0.41.0", "git_commit": "abc123",
    "attention": {"unet": ["AttnProcessor2_0"], "vae": ["AttnProcessor2_0"]},
    "tf32_matmul": False, "tf32_cudnn": False, "gpu_uuid": "GPU-0000",
}


def make_run(tmp_path: Path, n: int = 2, gpu: str = "NVIDIA A100-PCIE-40GB") -> Path:
    """A run directory shaped like eldm.benchmark.run's output, which passes every check."""
    prompts = [Prompt(i, f"prompt {i}", "manual", "objects", 1000 + i) for i in range(n)]
    save_prompts(prompts, tmp_path / "prompts.jsonl")
    run = tmp_path / "baseline"
    (run / "images").mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(
        {"name": "baseline", "prompts_file": str(tmp_path / "prompts.jsonl"), "num_prompts": n, "batch_size": 1,
         "tf32": {"matmul": False, "cudnn": False}}))
    rng = np.random.default_rng(0)
    for p in prompts:
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(run / "images" / f"{p.id:04d}.png")
    (run / "perf.json").write_text(json.dumps({
        "name": "baseline", "num_images": n, "latency_per_image_s": 2.5, "throughput_img_per_s": 0.4,
        "nfe": 51, "nfe_full": 51, **STAGES, "stage_coverage": 0.98, "peak_mem_gb": 3.2, "env": {**ENV, "gpu": gpu},
    }))
    write_timings(run, [{"ids": [p.id]} for p in prompts])
    return run


# Per-batch stage times adding up to 98% of a 2.5 s batch.
STAGES = {"t_text_s": 0.02, "t_unet_s": 2.36, "t_vae_s": 0.07}
ROW = {"latency_s": 2.5, "nfe": 51, "nfe_full": 51, **STAGES}


def write_timings(run: Path, rows: list[dict]) -> None:
    """timings.jsonl with each row's fields over ROW's."""
    (run / "timings.jsonl").write_text("".join(json.dumps({**ROW, **r}) + "\n" for r in rows))


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
    write_timings(run, [{}] * 2)
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
    ({"env": {**ENV, "gpu": None}}, "env.gpu is missing"),
    ({"env": {**ENV, "gpu": "A100", "gpu_uuid": None}}, "env.gpu_uuid is missing"),
    ({"env": {"device": "cpu"}}, "env.tf32_matmul is missing"),
    ({"name": "fp16"}, "name 'fp16' != config name 'baseline'"),
    ({"nfe": None}, "nfe is None"),
    ({"nfe_full": 0}, "nfe_full is 0"),
    ({"nfe_full": 52}, "nfe_full 52 > nfe 51"),
    ({"env": {**ENV, "gpu": "A100", "tf32_cudnn": True}}, "TF32 cudnn was True in the run, config asks for False"),
    ({"t_vae_s": None}, "t_vae_s is None"),
    ({"stage_coverage": 0.85}, "stage_coverage is 0.85, outside (0.9, 1.01)"),
])
def test_perf_defects_fail(tmp_path, changes, expected):
    run = make_run(tmp_path)
    update_json(run / "perf.json", **changes)
    problems = check_run.check_run(run)
    assert any(expected in p for p in problems), problems


def test_cpu_run_needs_no_gpu_fields(tmp_path):
    run = make_run(tmp_path)
    cpu_env = {k: v for k, v in ENV.items() if k != "gpu_uuid"} | {"device": "cpu"}
    update_json(run / "perf.json", env=cpu_env, peak_mem_gb=None)
    assert check_run.check_run(run) == []


@pytest.mark.parametrize("rows, expected", [
    ([{}], "1 rows, expected 2 batches"),
    ([{}, {"nfe_full": 17}], "NFE differs between batches"),
    ([{"nfe_full": 60}] * 2, "nfe 51, nfe_full 60"),
    ([{"nfe": None, "nfe_full": None}] * 2, "nfe None, nfe_full None"),
    ([{}, {"t_text_s": None}], "1 rows lack stage times, e.g. row 1"),
    ([{}, {"t_unet_s": 2.0}], "stage coverage outside (0.9, 1.01) in 1 rows, e.g. row 1: 0.836"),
    ([{"latency_s": 2.3}, {}], "stage coverage outside (0.9, 1.01) in 1 rows, e.g. row 0: 1.065"),
])
def test_timings_defects_fail(tmp_path, rows, expected):
    run = make_run(tmp_path)
    write_timings(run, rows)
    problems = check_run.check_run(run)
    assert any(expected in p for p in problems), problems


def test_timings_rows_follow_batch_size(tmp_path):
    run = make_run(tmp_path, n=3)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    (run / "config.yaml").write_text(yaml.safe_dump({**cfg, "batch_size": 2}))
    assert any("3 rows, expected 2 batches" in p for p in check_run.check_run(run))
    write_timings(run, [{}] * 2)
    assert check_run.check_run(run) == []


def test_missing_perf_and_wrong_gpu_fail(tmp_path):
    run = make_run(tmp_path, gpu="Tesla P100-PCIE-16GB")
    assert any("not A100" in p for p in check_run.check_run(run, require_gpu="A100"))
    (run / "perf.json").unlink()
    assert any("perf.json unreadable" in p for p in check_run.check_run(run))


def test_quality_only_when_expected(tmp_path):
    run = make_run(tmp_path)
    assert check_run.check_run(run) == []
    assert any("quality.json unreadable" in p for p in check_run.check_run(run, expect_quality=True))
    quality = {"clip_vitg": 30.15, "clip_vitl": 26.05, "kid_mean": 0.01, "fid_skipped": "2 COCO images"}
    (run / "quality.json").write_text(json.dumps({**quality, "clip_vitg": None}))
    (run / "quality_per_image.csv").write_text("id,clip_vitg,clip_vitl\n0,30.1,26.0\n1,30.2,26.1\n")
    assert check_run.check_run(run, expect_quality=True) == ["quality.json has no clip_vitg"]
    (run / "quality.json").write_text(json.dumps({k: v for k, v in quality.items() if k != "fid_skipped"}))
    assert check_run.check_run(run, expect_quality=True) == ["quality.json has neither fid nor fid_skipped"]
    (run / "quality.json").write_text(json.dumps(quality))
    assert check_run.check_run(run, expect_quality=True) == []
    (run / "quality_per_image.csv").write_text("id,clip_vitg,clip_vitl\n1,30.2,26.1\n0,30.1,26.0\n")
    assert check_run.check_run(run, expect_quality=True) == [
        "quality_per_image.csv has 2 rows, not one per image in prompt order"]
    (run / "quality_per_image.csv").unlink()
    assert any("quality_per_image.csv unreadable" in p for p in check_run.check_run(run, expect_quality=True))


def test_unreadable_config_fails(tmp_path):
    run = make_run(tmp_path)
    (run / "config.yaml").unlink()
    problems = check_run.check_run(run)
    assert len(problems) == 1 and "config.yaml" in problems[0]
