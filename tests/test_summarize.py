import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("summarize", ROOT / "scripts/summarize.py")
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)


def make_run(root: Path, rel: str, latency: float, quality: dict | None = None, batch_size: int = 1) -> Path:
    """A run directory with the perf.json fields eldm.benchmark.run writes (and a quality.json)."""
    run = root / rel
    run.mkdir(parents=True)
    (run / "perf.json").write_text(json.dumps({
        "name": run.name, "num_images": 50, "batch_size": batch_size, "latency_per_image_s": latency,
        "throughput_img_per_s": 1 / latency, "peak_mem_gb": 5.0, "nfe": 51, "nfe_full": 51,
        "t_text_s": 0.01, "t_unet_s": latency * 0.95, "t_vae_s": 0.05, "stage_coverage": 0.98,
        "env": {"node": "cs001", "gpu": "NVIDIA A100-PCIE-40GB", "device": "cuda"},
    }))
    if quality is not None:
        (run / "quality.json").write_text(json.dumps({"name": run.name, "num_images": 50, **quality}))
    return run


QUALITY = {"clip_vitg": 38.0, "clip_vitl": 31.0, "kid_mean": 0.004, "kid_std": 0.0006,
           "lpips_to_baseline": 0.0, "fid_skipped": "50 COCO images; FID needs at least 5,000"}


def test_columns_and_run_paths(tmp_path):
    runs = [make_run(tmp_path, "debug_a/baseline", 6.0, QUALITY), make_run(tmp_path, "debug_b/baseline", 6.0, QUALITY),
            make_run(tmp_path, "debug_a/baseline_bs4", 5.0, QUALITY, batch_size=4)]
    df = summarize.collect(runs, baseline=runs[0])
    assert list(df.columns) == [c for c in summarize.COLUMNS if c != "fid"]  # no run reached FID's 5,000
    assert list(df["run"]) == ["debug_a/baseline_bs4", "debug_a/baseline", "debug_b/baseline"]  # by latency
    assert list(df["name"]) == ["baseline_bs4", "baseline", "baseline"]  # names repeat; run tells them apart
    assert list(df["speedup"]) == pytest.approx([1.2, 1.0, 1.0])
    assert set(df["node"]) == {"cs001"} and set(df["gpu"]) == {"NVIDIA A100-PCIE-40GB"}
    assert df.notna().all().all()


def test_speedup_reference(tmp_path):
    runs = [make_run(tmp_path, "baseline", 6.0), make_run(tmp_path, "fp16", 2.0)]
    assert list(summarize.collect(runs)["speedup"]) == pytest.approx([3.0, 1.0])  # the run named baseline
    assert list(summarize.collect(runs, baseline=runs[1])["speedup"]) == pytest.approx([1.0, 1 / 3])
    with pytest.raises(SystemExit, match="isn't one of the runs"):
        summarize.collect(runs, baseline=tmp_path / "other")
    two = runs + [make_run(tmp_path, "again/baseline", 6.0)]
    assert "speedup" not in summarize.collect(two)  # two runs named baseline: ambiguous
    assert summarize.collect([tmp_path / "missing"]).empty


def test_main_writes_csv_and_figures(tmp_path):
    runs = [make_run(tmp_path, "p5/debug_a/baseline", 6.0, QUALITY), make_run(tmp_path, "p5/debug_a/fp16", 2.0,
                                                                            {**QUALITY, "fid": 21.0})]
    out = tmp_path / "p5" / "summary"
    summarize.main(["--runs", *map(str, runs), "--baseline", str(runs[0]), "--csv", str(out / "summary.csv"),
                    "--out", str(out), "--grid-runs", "baseline", "fp16"])
    table = pd.read_csv(out / "summary.csv")
    assert list(table["run"]) == ["fp16", "baseline"] and list(table["speedup"]) == pytest.approx([3.0, 1.0])
    assert list(table["fid"].isna()) == [False, True]
    assert {p.name for p in out.iterdir()} == {"summary.csv", "tradeoff_latency_clip.png", "peak_memory.png",
                                               "comparison_grid.png"}
