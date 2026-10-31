"""Aggregate per-run results into a table, trade-off plots, and side-by-side image grids.

Example:
    python scripts/summarize.py --runs "results/*" --out report/figures

One row per run directory with a perf.json, plus its quality.json if evaluated. ``run`` is the
directory relative to the runs' common parent, since names repeat across result sets (two
"baseline" runs, say). ``speedup`` is the --baseline run's latency over each run's; without
--baseline, the run named "baseline", if there is exactly one.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

COLUMNS = [
    "run", "name", "batch_size", "latency_per_image_s", "speedup", "throughput_img_per_s", "peak_mem_gb",
    "nfe", "nfe_full", "t_text_s", "t_unet_s", "t_vae_s",
    "clip_vitg", "clip_vitl", "kid_mean", "kid_std", "lpips_to_baseline", "fid", "node", "gpu",
]


def collect(run_dirs: list[Path], baseline: Path | None = None) -> pd.DataFrame:
    run_dirs = [d for d in run_dirs if (d / "perf.json").exists()]
    if not run_dirs:
        return pd.DataFrame()
    root = Path(os.path.commonpath([d.resolve().parent for d in run_dirs]))
    rows = []
    for d in run_dirs:
        row = json.loads((d / "perf.json").read_text())
        env = row.pop("env", None) or {}
        row.update(run=str(d.resolve().relative_to(root)), node=env.get("node"), gpu=env.get("gpu"),
                   _path=d.resolve())
        if (d / "quality.json").exists():
            row.update({k: v for k, v in json.loads((d / "quality.json").read_text()).items() if k != "name"})
        rows.append(row)
    df = pd.DataFrame(rows)
    if baseline is not None:
        base = df[df["_path"] == baseline.resolve()]
        if base.empty:
            raise SystemExit(f"--baseline {baseline} isn't one of the runs")
    else:
        base = df[df["name"] == "baseline"]
    if len(base) == 1:
        df["speedup"] = base["latency_per_image_s"].item() / df["latency_per_image_s"]
    return df[[c for c in COLUMNS if c in df.columns]].sort_values("latency_per_image_s")


def plot_tradeoff(df: pd.DataFrame, out: Path) -> None:
    if "clip_vitg" not in df:
        return
    d = df[df["batch_size"] == 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d["latency_per_image_s"], d["clip_vitg"])
    for _, r in d.iterrows():
        ax.annotate(r["run"], (r["latency_per_image_s"], r["clip_vitg"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Latency per image (s)")
    ax.set_ylabel("CLIP Score (ViT-g/14)")
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
    ax.barh(d["run"], d["peak_mem_gb"])
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["results/*"])
    ap.add_argument("--baseline", type=Path, help="Run directory the speedups are relative to")
    ap.add_argument("--csv", type=Path, default=Path("results/summary.csv"), help="Where the table goes")
    ap.add_argument("--out", default="report/figures", help="Where the plots and grid go")
    ap.add_argument("--grid-runs", nargs="+",
                    default=["baseline", "steps_20", "ddim_20", "fp16", "deepcache_i3", "combined_all"])
    ap.add_argument("--grid-ids", nargs="+", type=int, default=[0, 1, 2, 3])
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted({Path(p) for pat in args.runs for p in glob.glob(pat) if Path(p).is_dir()})
    df = collect(run_dirs, args.baseline)
    if df.empty:
        print("No runs with perf.json found.")
        return

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    print(f"Wrote {args.csv}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    plot_tradeoff(df, out)
    plot_memory(df, out)

    results_root = run_dirs[0].parent
    grid_runs = [r for r in args.grid_runs if (results_root / r).exists()]
    if grid_runs:
        image_grid(results_root, grid_runs, args.grid_ids, out / "comparison_grid.png")


if __name__ == "__main__":
    main()
