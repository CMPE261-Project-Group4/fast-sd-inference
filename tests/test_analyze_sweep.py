import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats

from eldm.config import load_experiment
from eldm.prompts import Prompt, save_prompts

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyze_sweep", ROOT / "scripts/analyze_sweep.py")
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)

# Two COCO prompts and four PartiPrompts in two Challenges.
PROMPTS = [Prompt(0, "a", "coco", "coco", 1000, 42), Prompt(1, "b", "coco", "coco", 1001, 43),
           Prompt(2000, "c", "parti", "Animals", 3000, None, "Basic"),
           Prompt(2001, "d", "parti", "Animals", 3001, None, "Basic"),
           Prompt(2002, "e", "parti", "Food", 3002, None, "Quantity"),
           Prompt(2003, "f", "parti", "Food", 3003, None, "Quantity")]
IDS = [p.id for p in PROMPTS]
BASE_G = [30.0, 31.0, 28.0, 29.0, 27.0, 26.0]
# SD1.5's scheduler_config.json at the pinned revision (as in test_estimate_sweep.py).
SD15_SCHEDULER = {
    "beta_end": 0.012, "beta_schedule": "scaled_linear", "beta_start": 0.00085, "num_train_timesteps": 1000,
    "set_alpha_to_one": False, "skip_prk_steps": True, "steps_offset": 1, "trained_betas": None, "clip_sample": False,
}


def write_run(d: Path, cfg: dict, clip_g: list[float], lpips: list[float], ids=IDS, perf: dict | None = None) -> Path:
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    pd.DataFrame({"id": ids, "clip_vitg": clip_g, "clip_vitl": [g - 3 for g in clip_g],
                  "lpips_to_baseline": lpips}).to_csv(d / "quality_per_image.csv", index=False)
    if perf is not None:
        (d / "perf.json").write_text(json.dumps({"name": cfg["name"], **perf}))
    return d


@pytest.fixture
def prompts_file(tmp_path):
    save_prompts(PROMPTS, tmp_path / "prompts.jsonl")
    return str(tmp_path / "prompts.jsonl")


def test_paired_quality(tmp_path, prompts_file):
    cfg = {"prompts_file": prompts_file, "num_prompts": None}
    base = write_run(tmp_path / "set/baseline", {**cfg, "name": "baseline"}, BASE_G, [0.0] * 6)
    diffs = [0.5, -0.1, 0.3, 0.2, -0.4, 0.1]
    fast = write_run(tmp_path / "set/fast", {**cfg, "name": "fast"}, [b + d for b, d in zip(BASE_G, diffs)],
                     [0.1, 0.2, 0.3, 0.2, 0.1, 0.3])
    df = analyze.paired_quality([base, fast], base)

    assert set(df.name) == {"fast"}  # the baseline isn't compared with itself
    assert set(df.group_by) == {"all", "source", "challenge"}
    assert set(df[df.group_by == "challenge"].group) == {"Basic", "Quantity"}  # COCO prompts have none
    row = df[(df.group_by == "all") & (df.metric == "clip_vitg")].iloc[0]
    half = stats.t.ppf(0.975, 5) * np.std(diffs, ddof=1) / np.sqrt(6)
    assert row.n == 6 and row.mean_diff == pytest.approx(np.mean(diffs))
    assert (row.ci_low, row.ci_high) == pytest.approx((np.mean(diffs) - half, np.mean(diffs) + half))
    assert row.baseline_mean == pytest.approx(np.mean(BASE_G))
    coco = df[(df.group == "coco") & (df.metric == "clip_vitg")].iloc[0]
    assert coco.n == 2 and coco.mean_diff == pytest.approx(0.2)
    lp = df[(df.group_by == "all") & (df.metric == "lpips_to_baseline")].iloc[0]
    assert lp.mean_diff == pytest.approx(0.2) and lp.ci_low < 0.2 < lp.ci_high and np.isnan(lp.p_value)
    clip = df[df.metric != "lpips_to_baseline"]
    assert (clip.q_value >= clip.p_value - 1e-12).all() and clip.q_value.notna().all()


def test_paired_quality_needs_the_same_prompts(tmp_path, prompts_file):
    cfg = {"prompts_file": prompts_file, "num_prompts": None}
    base = write_run(tmp_path / "baseline", {**cfg, "name": "baseline"}, BASE_G, [0.0] * 6)
    short = write_run(tmp_path / "short", {**cfg, "name": "short"}, BASE_G[:5], [0.1] * 5, ids=IDS[:5])
    with pytest.raises(SystemExit, match="different prompt ids"):
        analyze.paired_quality([base, short], base)


def test_main_writes_every_table(tmp_path, prompts_file):
    """A sweep with the runs the cost model needs: FP32 fixed 0.1 s and 0.1 s per full call, FP16 0.07 and
    0.03, a cached FP32 call 0.01 s, batch 4 at 0.7 of batch 1 (FP16)."""
    perf = {"baseline": dict(nfe=51, nfe_full=51, t_unet_s=5.1, latency_per_image_s=5.2),
            "fp16": dict(nfe=51, nfe_full=51, t_unet_s=1.53, latency_per_image_s=1.6),
            "fp16_bs4": dict(nfe=51, nfe_full=51, t_unet_s=1.07, latency_per_image_s=1.12),
            "deepcache_i3": dict(nfe=51, nfe_full=17, t_unet_s=2.04, latency_per_image_s=2.14)}
    cfgs = {c["name"]: c for f in ("00_baseline", "03_precision", "04_deepcache")
            for c in load_experiment(ROOT / f"configs/experiments/{f}.yaml", ROOT)}
    sweep = tmp_path / "sweep"
    for name, p in perf.items():
        write_run(sweep / name, {**cfgs[name], "prompts_file": prompts_file}, BASE_G, [0.0] * 6, perf=p)
    (tmp_path / "scheduler.json").write_text(json.dumps(SD15_SCHEDULER))
    pd.DataFrame({"group": ["sweep", "sweep", "sweep"], "run": ["baseline", "baseline (second node)", "fp16"],
                  "s_per_image": [5.0, 5.0, 1.68], "basis": ["measured (pilot)"] * 2 + ["model"]}).to_csv(
        tmp_path / "estimate.csv", index=False)
    out = tmp_path / "out"
    assert analyze.main(["--runs", str(sweep / "*"), "--baseline", str(sweep / "baseline"), "--out-dir", str(out),
                         "--estimate", str(tmp_path / "estimate.csv"),
                         "--batch-pair", str(sweep / "fp16"), str(sweep / "fp16_bs4"),
                         "--scheduler-config", str(tmp_path / "scheduler.json")]) == 0

    assert set(pd.read_csv(out / "paired_quality.csv").name) == {"fp16", "fp16_bs4", "deepcache_i3"}
    check = pd.read_csv(out / "estimate_check.csv").set_index("run")
    assert list(check.index) == ["baseline", "fp16"]  # the second-node row isn't a run here
    assert check.loc["fp16", "error_pct"] == pytest.approx(5.0)
    model = pd.read_csv(out / "cost_model.csv").set_index(["parameter", "dtype"]).value
    assert model["c_full", "fp32"] == pytest.approx(0.1) and model["fixed", "fp16"] == pytest.approx(0.07)
    assert model["c_partial", "fp32"] == pytest.approx(0.01) and model["batch_ratio", "fp16"] == pytest.approx(0.7)
    fc = pd.read_csv(out / "forecast.csv").set_index("run")
    assert list(fc.index) == [c["name"] for c in load_experiment(ROOT / "configs/experiments/05_combined.yaml", ROOT)]
    combined = fc.loc["combined_all"]  # FP16, DPM-Solver++ 20 steps, DeepCache every 3: 7 full calls of 20
    assert (combined.nfe, combined.nfe_full) == (20, 7)
    assert combined.s_per_image == pytest.approx(0.07 + 7 * 0.03 + 13 * 0.003)
    assert combined.speedup == pytest.approx(5.2 / combined.s_per_image)
    assert fc.loc["combined_all_bs4", "s_per_image"] == pytest.approx(0.7 * combined.s_per_image)


def test_baseline_must_be_a_scored_run(tmp_path):
    with pytest.raises(SystemExit):
        analyze.main(["--runs", str(tmp_path / "*"), "--baseline", str(tmp_path / "nope"),
                      "--out-dir", str(tmp_path / "out")])
