import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from eldm.config import deep_merge, load_experiment

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("estimate_sweep", ROOT / "scripts/estimate_sweep.py")
estimate_sweep = importlib.util.module_from_spec(_spec)
sys.modules["estimate_sweep"] = estimate_sweep  # dataclasses look their module up while it loads
_spec.loader.exec_module(estimate_sweep)

# SD1.5's scheduler_config.json at the pinned revision 451f4fe (test_scheduler_config_is_sd15s).
SD15_SCHEDULER = {
    "beta_end": 0.012, "beta_schedule": "scaled_linear", "beta_start": 0.00085, "num_train_timesteps": 1000,
    "set_alpha_to_one": False, "skip_prk_steps": True, "steps_offset": 1, "trained_betas": None, "clip_sample": False,
}
BASE = yaml.safe_load((ROOT / "configs/base.yaml").read_text())


def test_scheduler_config_is_sd15s():
    try:
        pinned = estimate_sweep.sd15_scheduler_config()
    except (OSError, ValueError):
        pytest.skip("needs the shared HF cache")
    assert {k: v for k, v in pinned.items() if not k.startswith("_")} == SD15_SCHEDULER


@pytest.mark.parametrize("scheduler, steps, interval, nfe", [
    ("pndm", 50, None, (51, 51)),  # PNDM repeats its second timestep: 51 calls (job 98329)
    ("pndm", 5, None, (6, 6)),
    ("ddim", 20, None, (20, 20)),
    ("dpmpp_2m", 20, None, (20, 20)),
    ("euler", 20, None, (20, 20)),
    ("pndm", 50, 2, (51, 25)),  # the counts tests/test_probes.py measures through DeepCache itself
    ("pndm", 50, 3, (51, 17)),
    ("pndm", 50, 5, (51, 11)),
    ("dpmpp_2m", 20, 3, (20, 7)),
])
def test_expected_nfe(scheduler, steps, interval, nfe):
    cfg = deep_merge(BASE, {"scheduler": scheduler, "num_inference_steps": steps,
                            "deepcache": {"enabled": interval is not None, "cache_interval": interval or 3}})
    assert estimate_sweep.expected_nfe(cfg, SD15_SCHEDULER) == nfe


# Synthetic pilot timings (s/image): fixed 0.1 (FP32) and 0.07 (FP16), a full UNet call 0.1 and
# 0.03, a cached call 0.01 (FP32). dpmpp_20 is held out and ran a little slower than the model.
PERF = {
    "baseline": dict(nfe=51, nfe_full=51, t_unet_s=5.1, latency_per_image_s=5.2),
    "fp16": dict(nfe=51, nfe_full=51, t_unet_s=1.53, latency_per_image_s=1.6),
    "deepcache_i3": dict(nfe=51, nfe_full=17, t_unet_s=2.04, latency_per_image_s=2.14),
    "ddim_20": dict(nfe=20, nfe_full=20, t_unet_s=2.0, latency_per_image_s=2.1),
    "dpmpp_20": dict(nfe=20, nfe_full=20, t_unet_s=2.1, latency_per_image_s=2.2),
}


def write_run(d: Path, cfg: dict, perf: dict) -> Path:
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    (d / "perf.json").write_text(json.dumps({"name": cfg["name"], "num_images": 200, **perf}))
    return d


@pytest.fixture
def inputs(tmp_path):
    """Pilot runs from the real pilot.yaml, a batch-1/batch-4 pair (ratio 0.8), 10 prompts."""
    for cfg in load_experiment(ROOT / "configs/experiments/pilot.yaml", ROOT):
        write_run(tmp_path / "pilot" / cfg["name"], cfg, PERF[cfg["name"]])
    bs1, bs4 = load_experiment(ROOT / "configs/experiments/00_baseline.yaml", ROOT)
    write_run(tmp_path / "p5/baseline", bs1, dict(nfe=51, nfe_full=51, t_unet_s=4.9, latency_per_image_s=5.0))
    write_run(tmp_path / "p5/baseline_bs4", bs4, dict(nfe=51, nfe_full=51, t_unet_s=3.9, latency_per_image_s=4.0))
    (tmp_path / "prompts.jsonl").write_text("{}\n" * 10)
    (tmp_path / "scheduler.json").write_text(json.dumps(SD15_SCHEDULER))
    busy = sum(p["latency_per_image_s"] for p in PERF.values()) * 202  # 200 images + 2 warm-up each
    return tmp_path, [
        "--pilot", str(tmp_path / "pilot/*"), "--pilot-elapsed-s", str(busy + 1000 * 0.05),
        "--eval-elapsed-s", "200", "--batch-pair", str(tmp_path / "p5/baseline"), str(tmp_path / "p5/baseline_bs4"),
        "--prompt-sets", str(tmp_path / "prompts.jsonl"), "--scheduler-config", str(tmp_path / "scheduler.json"),
        "--csv", str(tmp_path / "estimate.csv"),
    ]


def test_estimate(inputs, capsys):
    tmp_path, argv = inputs
    assert estimate_sweep.main(argv) == 0
    rows = pd.read_csv(tmp_path / "estimate.csv").set_index("run")
    assert len(rows) == 17 + 1 + 1 + 5 + 1  # 00-04, the second-node baseline, a total; 05, a total

    # Measured runs keep their measured time; the model reproduces the fitted ones.
    assert rows.loc["baseline", "basis"] == "measured (pilot)"
    assert rows.loc["baseline", ["s_per_image", "model_s_per_image"]].tolist() == pytest.approx([5.2, 5.2])
    assert rows.loc["dpmpp_20", ["s_per_image", "model_s_per_image"]].tolist() == pytest.approx([2.2, 2.1])
    # The rest are modeled: PNDM-10 makes 11 calls; DeepCache i2 25 full and 26 cached.
    assert rows.loc["steps_10", ["nfe", "s_per_image"]].tolist() == pytest.approx([11, 0.1 + 11 * 0.1])
    assert rows.loc["deepcache_i2", "s_per_image"] == pytest.approx(0.1 + 25 * 0.1 + 26 * 0.01)
    assert rows.loc["deepcache_i2", "basis"] == "model"
    assert rows.loc["fp16_bs4", "s_per_image"] == pytest.approx(1.6 * 0.8)
    assert rows.loc["fp16_bs4", "basis"] == "model; fp32's batch-4 ratio"
    # FP16 DPM++ 20 with DeepCache i3: 7 full calls, 13 cached ones at 0.01 * 0.03 / 0.1.
    assert rows.loc["combined_all", "s_per_image"] == pytest.approx(0.07 + 7 * 0.03 + 13 * 0.003)
    assert "fp16 cached calls scale like full ones" in rows.loc["combined_all", "basis"]

    # GPU-hours: 10 prompts + 2 warm-up images, 0.05 s/image overhead, 200 s / 1,000 images evaluation.
    assert rows.loc["baseline", "gen_gpu_h_prompts"] == pytest.approx((12 * 5.2 + 10 * 0.05) / 3600)
    assert rows.loc["baseline", "eval_gpu_h_prompts"] == pytest.approx(10 * 0.2 / 3600)
    assert rows.loc["baseline (second node)", "gen_gpu_h_prompts"] == rows.loc["baseline", "gen_gpu_h_prompts"]
    totals = rows.loc["(total)"].set_index("group")
    sweep = rows[(rows["group"] == "sweep") & (rows.index != "(total)")]
    assert totals.loc["sweep", "gen_gpu_h_prompts"] == pytest.approx(sweep["gen_gpu_h_prompts"].sum())
    assert re.search(r"dpmpp_20 +2\.200 +2\.100 +-4\.5% +\(held out\)", capsys.readouterr().out)


def test_estimate_checks_its_inputs(inputs):
    tmp_path, argv = inputs
    perf = tmp_path / "pilot/ddim_20/perf.json"
    perf.write_text(json.dumps({**json.loads(perf.read_text()), "nfe": 21}))
    with pytest.raises(SystemExit, match="ddim_20: expected NFE"):
        estimate_sweep.main(argv)
    perf.write_text(json.dumps({**json.loads(perf.read_text()), "nfe": 20}))
    with pytest.raises(SystemExit, match="no run named fp16"):
        estimate_sweep.main([a.replace("pilot/*", "pilot/[bd]*") for a in argv])
