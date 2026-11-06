"""Store a result set's key numbers for later analysis: paired quality statistics, the cost model's
estimates against measured times, and a forecast of later runs fitted on this set's times.

Example (Phase 8):
    R=/scratch/cmpe261-fa26/group4/results
    python scripts/analyze_sweep.py --runs "$R/p8/sweep/*" --baseline $R/p8/sweep/baseline \
        --out-dir report/sweeps --estimate report/pilot/sweep_estimate.csv \
        --batch-pair $R/p8/sweep/fp16 $R/p8/sweep/fp16_bs4

Run from the repo root (prompts_file paths in config.yaml are relative to it). Writes to --out-dir:

paired_quality.csv: for each run but the baseline and each prompt group (all prompts, each source,
    each PartiPrompts Challenge; COCO prompts have no Challenge), the mean per-prompt change from the
    baseline in each CLIP Score, with the paired t-test's 95 % confidence interval and p-value
    (scipy.stats.ttest_rel) and Benjamini-Hochberg q-values (scipy.stats.false_discovery_control)
    within each group type and metric; and the mean LPIPS to the baseline with a one-sample t interval.
estimate_check.csv (with --estimate): each run's estimated s/image from estimate_sweep.py's table
    against its measured s/image.
cost_model.csv and forecast.csv (with --batch-pair): estimate_sweep.py's cost model fitted on these
    runs (its "baseline", "fp16" and "deepcache_i3", and --batch-pair's batch-4 ratio), and its
    s/image for every run of --forecast (default: 05_combined.yaml), with the speedup over the baseline.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from eldm.config import load_experiment
from eldm.prompts import load_prompts

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("estimate_sweep", ROOT / "scripts/estimate_sweep.py")
estimate_sweep = importlib.util.module_from_spec(_spec)
sys.modules["estimate_sweep"] = estimate_sweep  # dataclasses look their module up while it loads
_spec.loader.exec_module(estimate_sweep)

CLIP_METRICS = ("clip_vitg", "clip_vitl")
GROUP_BY = ("all", "source", "challenge")


def per_image(run_dir: Path) -> pd.DataFrame:
    """A run's quality_per_image.csv, indexed by prompt id, with each prompt's source and challenge."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    prompts = {p.id: p for p in load_prompts(cfg["prompts_file"], cfg.get("num_prompts"))}
    df = pd.read_csv(run_dir / "quality_per_image.csv").set_index("id").sort_index()
    df["all"] = "all"
    df["source"] = [prompts[i].source for i in df.index]
    df["challenge"] = [prompts[i].challenge for i in df.index]
    return df


def _interval(result) -> tuple[float, float]:
    ci = result.confidence_interval(0.95)
    return float(ci.low), float(ci.high)


def paired_quality(run_dirs: list[Path], baseline: Path) -> pd.DataFrame:
    base = per_image(baseline)
    rows = []
    for d in run_dirs:
        if d.resolve() == baseline.resolve():
            continue
        run = per_image(d)
        if not run.index.equals(base.index):
            raise SystemExit(f"{d} and the baseline {baseline} scored different prompt ids")
        name = yaml.safe_load((d / "config.yaml").read_text())["name"]
        for key in GROUP_BY:
            for group, idx in run.groupby(key).groups.items():  # groupby drops COCO's missing challenge
                r, b = run.loc[idx], base.loc[idx]
                n = len(idx)
                for m in CLIP_METRICS:
                    res = stats.ttest_rel(r[m], b[m])
                    lo, hi = _interval(res) if n > 1 else (np.nan, np.nan)
                    rows.append(dict(run=d.name, name=name, group_by=key, group=group, metric=m, n=n,
                                     run_mean=r[m].mean(), baseline_mean=b[m].mean(),
                                     mean_diff=(r[m] - b[m]).mean(), ci_low=lo, ci_high=hi, p_value=res.pvalue))
                res = stats.ttest_1samp(r["lpips_to_baseline"], 0.0)
                lo, hi = _interval(res) if n > 1 else (np.nan, np.nan)
                mean = r["lpips_to_baseline"].mean()
                rows.append(dict(run=d.name, name=name, group_by=key, group=group, metric="lpips_to_baseline",
                                 n=n, run_mean=mean, baseline_mean=0.0, mean_diff=mean, ci_low=lo, ci_high=hi,
                                 p_value=np.nan))
    df = pd.DataFrame(rows)
    df["q_value"] = np.nan
    for _, idx in df[df.p_value.notna()].groupby(["group_by", "metric"]).groups.items():
        df.loc[idx, "q_value"] = stats.false_discovery_control(df.loc[idx, "p_value"])
    return df


def estimate_check(runs: dict, estimate_csv: Path) -> pd.DataFrame:
    est = pd.read_csv(estimate_csv)
    est = est[est.run.isin(runs.keys())]
    rows = []
    for _, e in est.drop_duplicates("run").iterrows():
        measured = runs[e.run].perf["latency_per_image_s"]
        rows.append(dict(run=e.run, estimate_s=e.s_per_image, basis=e.basis, measured_s=measured,
                         error_pct=100 * (e.s_per_image / measured - 1),
                         node=runs[e.run].perf.get("env", {}).get("node")))
    return pd.DataFrame(rows)


def cost_model(model) -> pd.DataFrame:
    rows = [dict(parameter=p, dtype=dt, value=getattr(model, p)[dt]) for p in ("fixed", "c_full", "c_partial")
            for dt in model.fixed]
    rows.append(dict(parameter="batch_ratio", dtype=model.batch_dtype, value=model.batch_ratio))
    return pd.DataFrame(rows)


def forecast(experiments: list[Path], model, scheduler_config: dict, baseline_s: float) -> pd.DataFrame:
    rows = []
    for path in experiments:
        for cfg in load_experiment(path, ROOT):
            nfe, nfe_full = estimate_sweep.expected_nfe(cfg, scheduler_config)
            t, assumes = estimate_sweep.predict(cfg, nfe, nfe_full, model)
            rows.append(dict(experiment=path.stem, run=cfg["name"], scheduler=cfg["scheduler"],
                             steps=cfg["num_inference_steps"], dtype=cfg["dtype"], batch_size=cfg["batch_size"],
                             deepcache_interval=cfg["deepcache"]["cache_interval"] if cfg["deepcache"]["enabled"]
                             else None, nfe=nfe, nfe_full=nfe_full, s_per_image=t, speedup=baseline_s / t,
                             basis="; ".join(["model", *assumes])))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories or glob patterns")
    ap.add_argument("--baseline", type=Path, required=True, help="The run the others are compared with")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--estimate", type=Path, help="estimate_sweep.py's table, to check against measured times")
    ap.add_argument("--batch-pair", nargs=2, type=Path, metavar=("BS1_RUN", "BS4_RUN"),
                    help="Fit the cost model on --runs with this pair's batch-4 ratio, and forecast --forecast")
    ap.add_argument("--forecast", nargs="+", type=Path, default=estimate_sweep.LATER,
                    help="Experiment files to forecast (default: 05_combined.yaml)")
    ap.add_argument("--scheduler-config", type=Path, help="Default: SD1.5's pinned scheduler_config.json")
    args = ap.parse_args(argv)

    run_dirs = sorted({Path(p) for pat in args.runs for p in glob.glob(pat)
                       if (Path(p) / "quality_per_image.csv").exists()})
    if args.baseline.resolve() not in {d.resolve() for d in run_dirs}:
        ap.error(f"--baseline {args.baseline} isn't one of the scored runs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = {"paired_quality.csv": paired_quality(run_dirs, args.baseline)}
    runs = estimate_sweep.read_runs(run_dirs)
    if args.estimate:
        written["estimate_check.csv"] = estimate_check(runs, args.estimate)
    if args.batch_pair:
        bs1, bs4 = (next(iter(estimate_sweep.read_runs([d]).values())) for d in args.batch_pair)
        model = estimate_sweep.fit(runs, bs1, bs4)
        scheduler_config = (json.loads(args.scheduler_config.read_text()) if args.scheduler_config
                            else estimate_sweep.sd15_scheduler_config())
        base_s = json.loads((args.baseline / "perf.json").read_text())["latency_per_image_s"]
        written["cost_model.csv"] = cost_model(model)
        written["forecast.csv"] = forecast(args.forecast, model, scheduler_config, base_s)
    for name, df in written.items():
        df.to_csv(args.out_dir / name, index=False)
        print(f"Wrote {args.out_dir / name} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
