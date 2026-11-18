"""Draw the report's result figures from the committed summary tables (no GPU, no result folders).

Each figure pairs every run's speedup over its baseline with its mean per-prompt change in
CLIP Score (ViT-g/14) and the paired 95 % confidence interval, as stored by analyze_sweep.py.
A run whose whole interval lies below zero is drawn in red: a significant loss in prompt alignment.

    python scripts/plot_report_figures.py            # writes report/figures/fig1_*.png, fig2_*.png

Inputs (relative to the repo root, override with --sweeps / --combined):
    report/sweeps/summary.csv, report/sweeps/paired_quality.csv        (A100, each technique alone)
    report/local4090/summary.csv, report/local4090/paired_quality.csv  (RTX 4090, combinations)

Speedups are only compared within one figure: each set is measured against a baseline on its own GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Readable names, in the order each figure lists them (top to bottom is fastest to slowest).
SINGLE_LABELS = {
    "fp16_bs4": "FP16, batch 4",
    "steps_10": "PNDM, 10 steps",
    "fp16": "FP16",
    "deepcache_i5": "DeepCache, interval 5",
    "ddim_20": "DDIM, 20 steps",
    "dpmpp_20": "DPM-Solver++, 20 steps",
    "euler_20": "Euler, 20 steps",
    "deepcache_i3": "DeepCache, interval 3",
    "steps_20": "PNDM, 20 steps",
    "deepcache_i2": "DeepCache, interval 2",
    "steps_30": "PNDM, 30 steps",
    "dpmpp_50": "DPM-Solver++, 50 steps",
    "ddim_50": "DDIM, 50 steps",
    "euler_50": "Euler, 50 steps",
}
COMBINED_LABELS = {
    "combined_all_bs4": "All three, batch 4",
    "combined_all": "All three (FP16 + DPM 20 + DC i3)",
    "combined_all_ddim": "All three, DDIM instead",
    "combined_all_i2": "All three, DeepCache interval 2",
    "combined_all_s30": "All three, 30 steps",
    "combined_no_fp16": "Leave out FP16",
    "combined_no_deepcache": "Leave out DeepCache",
    "combined_no_scheduler": "Leave out DPM-Solver++ 20",
    "only_dpmpp20": "DPM-Solver++ 20 only",
    "only_deepcache_i3": "DeepCache i3 only",
    "only_fp16": "FP16 only",
}
METRIC = "clip_vitg"


def load_rows(result_dir: Path, labels: dict[str, str]) -> pd.DataFrame:
    """One row per labelled run: speedup from summary.csv and the overall CLIP change and its
    interval from paired_quality.csv, sorted by speedup (fastest first)."""
    summary = pd.read_csv(result_dir / "summary.csv")
    paired = pd.read_csv(result_dir / "paired_quality.csv")
    paired = paired[(paired["group_by"] == "all") & (paired["metric"] == METRIC)]
    rows = summary[["run", "speedup"]].merge(paired[["run", "mean_diff", "ci_low", "ci_high"]], on="run")
    rows = rows[rows["run"].isin(labels)].copy()
    missing = set(labels) - set(rows["run"])
    if missing:
        raise SystemExit(f"{result_dir}: no summary or paired-quality row for {sorted(missing)}")
    rows["label"] = rows["run"].map(labels)
    rows["loss"] = rows["ci_high"] < 0
    return rows.sort_values("speedup", ascending=False).reset_index(drop=True)


def draw(rows: pd.DataFrame, title: str, subtitle: str, out: Path, clip_min: float = -0.5,
         highlight: tuple[str, ...] = ()) -> Path:
    """Two panels sharing the run list: speedup bars, and the CLIP change with its 95 % interval.
    Changes below clip_min are written at the panel's left edge instead of drawn off scale."""
    n = len(rows)
    fig, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(10, 0.36 * n + 1.8), sharey=True,
                                     gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.05})
    y = list(range(n))
    grey, red, blue = "#b8b8ad", "#c8372d", "#2f6fd6"

    colors = [blue if r in highlight else grey for r in rows["run"]]
    ax_s.barh(y, rows["speedup"], color=colors, height=0.6)
    for yi, s in zip(y, rows["speedup"]):
        ax_s.text(s + rows["speedup"].max() * 0.015, yi, f"{s:.2f}×", va="center", fontsize=9)
    ax_s.set_xlim(0, rows["speedup"].max() * 1.18)
    ax_s.set_yticks(y, rows["label"], fontsize=9)
    ax_s.invert_yaxis()
    ax_s.set_xlabel("Speedup over the baseline")
    ax_s.spines[["top", "right"]].set_visible(False)

    clip_max = max(0.3, rows["ci_high"].max() + 0.05)
    ax_c.axvline(0, color="#888888", lw=1)
    for yi, r in zip(y, rows.itertuples()):
        c = red if r.loss else "#4a4a4a"
        if r.mean_diff < clip_min:
            ax_c.text(clip_min + 0.01, yi, f"◂ {r.mean_diff:.2f} (off scale)", va="center", color=red, fontsize=9)
            continue
        ax_c.plot([r.ci_low, r.ci_high], [yi, yi], color=c, lw=2)
        ax_c.plot(r.mean_diff, yi, "o", color=c, ms=6)
    ax_c.set_xlim(clip_min, clip_max)
    ax_c.set_xlabel("Δ CLIP Score (ViT-g/14) vs baseline, 95 % CI")
    ax_c.spines[["top", "right", "left"]].set_visible(False)
    ax_c.tick_params(left=False)

    fig.suptitle(title, x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 1 - 0.55 / fig.get_figheight(), subtitle, ha="left", fontsize=9, color="#555555")
    fig.text(0.99, 0.01, "Red: whole 95 % CI below zero", ha="right", fontsize=8, color="#555555")
    fig.subplots_adjust(top=1 - 0.95 / fig.get_figheight(), bottom=0.9 / fig.get_figheight(), left=0.22, right=0.98)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweeps", type=Path, default=ROOT / "report/sweeps")
    ap.add_argument("--combined", type=Path, default=ROOT / "report/local4090")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "report/figures")
    args = ap.parse_args(argv)

    single = load_rows(args.sweeps, SINGLE_LABELS)
    combined = load_rows(args.combined, COMBINED_LABELS)
    return [
        draw(single, "Each technique alone: FP16 is the largest free speedup",
             "A100-PCIE-40GB, TF32 off, 3,632 prompts; baseline PNDM 50 steps FP32",
             args.out_dir / "fig1_single_techniques.png"),
        draw(combined, "Combined techniques: keeping 10 full UNet calls avoids the quality loss",
             "RTX 4090, TF32 off, 3,632 prompts; 2³ factorial plus extensions, vs a baseline on the same GPU",
             args.out_dir / "fig2_combined.png", highlight=("combined_all_i2", "combined_all_s30")),
    ]


if __name__ == "__main__":
    for path in main():
        print(path)
