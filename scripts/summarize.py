"""Aggregate per-run results into a table, trade-off plots, and side-by-side image grids.

Example:
    python scripts/summarize.py --runs "results/*" --out report/figures
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

COLUMNS = [
    "name", "batch_size", "latency_per_image_s", "speedup", "throughput_img_per_s",
    "peak_mem_gb", "clip_score", "lpips_to_baseline", "fid",
]


def collect(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        if not (d / "perf.json").exists():
            continue
        row = json.loads((d / "perf.json").read_text())
        row.pop("env", None)
        if (d / "quality.json").exists():
            row.update(json.loads((d / "quality.json").read_text()))
        rows.append(row)
    df = pd.DataFrame(rows)
    if "baseline" in set(df["name"]):
        base_lat = df.loc[df["name"] == "baseline", "latency_per_image_s"].item()
        df["speedup"] = base_lat / df["latency_per_image_s"]
    return df[[c for c in COLUMNS if c in df.columns]].sort_values("latency_per_image_s")


def plot_tradeoff(df: pd.DataFrame, out: Path) -> None:
    if "clip_score" not in df:
        return
    d = df[df["batch_size"] == 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d["latency_per_image_s"], d["clip_score"])
    for _, r in d.iterrows():
        ax.annotate(r["name"], (r["latency_per_image_s"], r["clip_score"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Latency per image (s)")
    ax.set_ylabel("CLIP Score")
    ax.set_title("Quality vs. latency (batch size 1)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tradeoff_latency_clip.png", dpi=200)
    plt.close(fig)


def plot_memory(df: pd.DataFrame, out: Path) -> None:
    if "peak_mem_gb" not in df:
        return
    d = df.dropna(subset=["peak_mem_gb"]).sort_values("peak_mem_gb")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(d))))
    ax.barh(d["name"], d["peak_mem_gb"])
    ax.set_xlabel("Peak GPU memory (GB)")
    fig.tight_layout()
    fig.savefig(out / "peak_memory.png", dpi=200)
    plt.close(fig)


def image_grid(results_root: Path, runs: list[str], ids: list[int], out_file: Path, cell: int = 256) -> None:
    """Rows = prompts, columns = runs; same seed in every cell of a row."""
    grid = Image.new("RGB", (cell * len(runs), cell * len(ids)), "white")
    for c, run in enumerate(runs):
        for r, i in enumerate(ids):
            path = results_root / run / "images" / f"{i:04d}.png"
            if path.exists():
                grid.paste(Image.open(path).resize((cell, cell)), (c * cell, r * cell))
    grid.save(out_file)
    print(f"grid columns: {runs} -> {out_file}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["results/*"])
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--grid-runs", nargs="+",
                    default=["baseline", "steps_20", "ddim_20", "fp16", "deepcache_i3", "combined_all"])
    ap.add_argument("--grid-ids", nargs="+", type=int, default=[0, 1, 2, 3])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted({Path(p) for pat in args.runs for p in glob.glob(pat) if Path(p).is_dir()})
    df = collect(run_dirs)
    if df.empty:
        print("No runs with perf.json found.")
        return

    df.to_csv("results/summary.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    plot_tradeoff(df, out)
    plot_memory(df, out)

    results_root = run_dirs[0].parent
    grid_runs = [r for r in args.grid_runs if (results_root / r).exists()]
    if grid_runs:
        image_grid(results_root, grid_runs, args.grid_ids, out / "comparison_grid.png")


if __name__ == "__main__":
    main()
