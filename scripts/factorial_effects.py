"""Main effects and interactions of the 2^3 factorial in configs/experiments/05_combined.yaml.

The eight factorial runs switch three factors on or off against the baseline (PNDM 50, FP32, no cache):
    S  sampler    PNDM 50 steps -> DPM-Solver++ 2M 20 steps
    P  precision  FP32 -> FP16
    C  caching    off -> DeepCache interval 3

For each response the script computes the standard 2^k effects (Yates contrasts): each effect is the
mean response with the term's sign +1 minus the mean with sign -1, over the 8 runs.
    log2_time   log2 of seconds per image. An effect of -1 halves the time. A positive interaction
                means the two factors together save less time than the product of their savings.
    delta_clip  mean per-prompt change in CLIP Score (ViT-g/14) from the baseline (paired_quality.csv).
    lpips       mean LPIPS to the baseline's images.

It also checks whether speedups multiply: for each run with two or three factors on, the product of
the single-factor speedups against the measured speedup.

    python scripts/factorial_effects.py     # writes report/local4090/factorial_effects.csv and
                                            #        report/local4090/speedup_product_check.csv

The per-prompt CLIP data needed for intervals on the effects stays in results/ (not committed), so
the effects here are point estimates; paired_quality.csv holds the per-run intervals.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FACTORS = ("S", "P", "C")
FACTOR_NAMES = {"S": "DPM-Solver++ 20 (vs PNDM 50)", "P": "FP16 (vs FP32)", "C": "DeepCache i3 (vs off)"}
# Which factors each factorial run switches on.
RUNS = {
    "ref_baseline": "",
    "only_dpmpp20": "S",
    "only_fp16": "P",
    "only_deepcache_i3": "C",
    "combined_no_deepcache": "SP",
    "combined_no_fp16": "SC",
    "combined_no_scheduler": "PC",
    "combined_all": "SPC",
}
RESPONSES = ("log2_time", "delta_clip", "lpips")


def load_design(result_dir: Path, metric: str = "clip_vitg") -> pd.DataFrame:
    """One row per factorial run: +1/-1 per factor and the three responses."""
    summary = pd.read_csv(result_dir / "summary.csv").set_index("run")
    paired = pd.read_csv(result_dir / "paired_quality.csv")
    paired = paired[(paired["group_by"] == "all") & (paired["metric"] == metric)].set_index("run")
    missing = [r for r in RUNS if r not in summary.index]
    if missing:
        raise SystemExit(f"{result_dir}/summary.csv lacks factorial runs {missing}")
    rows = []
    for run, on in RUNS.items():
        row = {"run": run, **{f: (1 if f in on else -1) for f in FACTORS}}
        row["log2_time"] = float(np.log2(summary.loc[run, "latency_per_image_s"]))
        row["delta_clip"] = 0.0 if not on else float(paired.loc[run, "mean_diff"])
        row["lpips"] = float(summary.loc[run, "lpips_to_baseline"])
        row["speedup"] = float(summary.loc[run, "speedup"])
        rows.append(row)
    return pd.DataFrame(rows)


def effects(design: pd.DataFrame) -> pd.DataFrame:
    """Every main effect and interaction (S, P, C, SP, SC, PC, SPC) for each response."""
    terms = [t for k in (1, 2, 3) for t in itertools.combinations(FACTORS, k)]
    out = []
    for term in terms:
        sign = np.prod([design[f].to_numpy() for f in term], axis=0)
        row = {"term": "".join(term), "order": len(term),
               "description": " x ".join(FACTOR_NAMES[f] for f in term)}
        for resp in RESPONSES:
            y = design[resp].to_numpy()
            row[resp] = float(y[sign > 0].mean() - y[sign < 0].mean())
        # A main effect on log2 time, read as the speedup the factor gives on average.
        row["speedup_factor"] = float(2 ** -row["log2_time"]) if len(term) == 1 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def product_check(design: pd.DataFrame) -> pd.DataFrame:
    """Measured speedup of every multi-factor run against the product of its single-factor speedups."""
    single = {RUNS[r]: s for r, s in zip(design["run"], design["speedup"]) if len(RUNS[r]) == 1}
    out = []
    for run, s in zip(design["run"], design["speedup"]):
        on = RUNS[run]
        if len(on) < 2:
            continue
        predicted = float(np.prod([single[f] for f in on]))
        out.append({"run": run, "factors": on, "predicted_speedup": predicted, "measured_speedup": s,
                    "measured_over_predicted": s / predicted})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "report/local4090",
                    help="directory with summary.csv and paired_quality.csv (default: report/local4090)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: same as --results")
    args = ap.parse_args(argv)
    out_dir = args.out_dir or args.results

    design = load_design(args.results)
    eff, prod = effects(design), product_check(design)
    out_dir.mkdir(parents=True, exist_ok=True)
    eff.to_csv(out_dir / "factorial_effects.csv", index=False)
    prod.to_csv(out_dir / "speedup_product_check.csv", index=False)
    return eff, prod


if __name__ == "__main__":
    e, p = main()
    with pd.option_context("display.width", 140, "display.precision", 3):
        print(e.drop(columns="description"))
        print(p)
