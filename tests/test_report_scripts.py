"""Tests for the report scripts: factorial_effects.py and plot_report_figures.py (CPU only, no torch)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


factorial = load_script("factorial_effects")
figures = load_script("plot_report_figures")


def write_results(d: Path, times: dict[str, float], clip: dict[str, float], extra_runs=()) -> Path:
    """summary.csv and paired_quality.csv in analyze_sweep.py's layout, for the given runs."""
    d.mkdir(parents=True, exist_ok=True)
    base = times["ref_baseline"]
    runs = list(times) + list(extra_runs)
    t = {**times, **{r: base / 2 for r in extra_runs}}
    pd.DataFrame({"run": runs, "latency_per_image_s": [t[r] for r in runs],
                  "speedup": [base / t[r] for r in runs], "lpips_to_baseline": [0.1] * len(runs)}
                 ).to_csv(d / "summary.csv", index=False)
    rows = [{"run": r, "group_by": "all", "group": "all", "metric": "clip_vitg",
             "mean_diff": clip.get(r, 0.0), "ci_low": clip.get(r, 0.0) - 0.1, "ci_high": clip.get(r, 0.0) + 0.1}
            for r in runs if r != "ref_baseline"]
    rows.append({"run": runs[1], "group_by": "source", "group": "coco", "metric": "clip_vitg",
                 "mean_diff": 9.0, "ci_low": 8.0, "ci_high": 10.0})  # must be ignored
    pd.DataFrame(rows).to_csv(d / "paired_quality.csv", index=False)
    return d


def multiplicative_times(s: float, p: float, c: float, base: float = 4.0) -> dict[str, float]:
    """Times where each factor divides the time by its speedup, independently of the others."""
    return {run: base / np.prod([{"S": s, "P": p, "C": c}[f] for f in on]) for run, on in factorial.RUNS.items()}


def test_main_effects_recover_independent_speedups(tmp_path):
    d = write_results(tmp_path, multiplicative_times(2.0, 4.0, 1.0), {})
    eff = factorial.effects(factorial.load_design(d)).set_index("term")
    assert eff.loc["S", "speedup_factor"] == pytest.approx(2.0)
    assert eff.loc["P", "speedup_factor"] == pytest.approx(4.0)
    assert eff.loc["C", "speedup_factor"] == pytest.approx(1.0)
    for term in ("SP", "SC", "PC", "SPC"):
        assert eff.loc[term, "log2_time"] == pytest.approx(0.0, abs=1e-12)


def test_interaction_on_clip_is_found(tmp_path):
    # S and C each leave CLIP unchanged, but together lose 0.4: an S x C interaction.
    clip = {"combined_no_fp16": -0.4, "combined_all": -0.4}
    d = write_results(tmp_path, multiplicative_times(2.0, 2.0, 2.0), clip)
    eff = factorial.effects(factorial.load_design(d)).set_index("term")
    assert eff.loc["SC", "delta_clip"] == pytest.approx(-0.2)
    assert eff.loc["S", "delta_clip"] == pytest.approx(-0.2)
    assert eff.loc["P", "delta_clip"] == pytest.approx(0.0)
    assert eff.loc["SP", "delta_clip"] == pytest.approx(0.0)


def test_product_check(tmp_path):
    times = multiplicative_times(2.0, 3.0, 2.0)
    times["combined_all"] *= 1.25  # 25 % slower than the product predicts
    prod = factorial.product_check(factorial.load_design(write_results(tmp_path, times, {}))).set_index("run")
    assert set(prod.index) == {"combined_no_deepcache", "combined_no_fp16", "combined_no_scheduler", "combined_all"}
    assert prod.loc["combined_all", "predicted_speedup"] == pytest.approx(12.0)
    assert prod.loc["combined_all", "measured_over_predicted"] == pytest.approx(0.8)
    assert prod.loc["combined_no_fp16", "measured_over_predicted"] == pytest.approx(1.0)


def test_missing_factorial_run_is_an_error(tmp_path):
    times = multiplicative_times(2.0, 2.0, 2.0)
    del times["combined_no_fp16"]
    with pytest.raises(SystemExit, match="combined_no_fp16"):
        factorial.load_design(write_results(tmp_path, times, {}))


def test_factorial_main_writes_csvs(tmp_path):
    d = write_results(tmp_path / "in", multiplicative_times(2.0, 2.0, 2.0), {})
    factorial.main(["--results", str(d), "--out-dir", str(tmp_path / "out")])
    assert len(pd.read_csv(tmp_path / "out/factorial_effects.csv")) == 7
    assert len(pd.read_csv(tmp_path / "out/speedup_product_check.csv")) == 4


def test_load_rows_sorts_and_flags_losses(tmp_path):
    labels = {"only_fp16": "FP16", "combined_all": "All", "only_dpmpp20": "DPM"}
    times = multiplicative_times(2.0, 3.0, 2.0)
    d = write_results(tmp_path, times, {"combined_all": -0.3, "only_dpmpp20": 0.1})
    rows = figures.load_rows(d, labels)
    assert list(rows["label"]) == ["All", "FP16", "DPM"]  # fastest first
    assert list(rows["loss"]) == [True, False, False]
    assert rows.loc[rows["run"] == "only_dpmpp20", "mean_diff"].item() == pytest.approx(0.1)  # not the coco row


def test_load_rows_reports_missing_runs(tmp_path):
    d = write_results(tmp_path, multiplicative_times(2.0, 2.0, 2.0), {})
    with pytest.raises(SystemExit, match="nope"):
        figures.load_rows(d, {"only_fp16": "FP16", "nope": "Missing"})


def test_figures_main_draws_both_figures(tmp_path):
    single = write_results(tmp_path / "sweeps", {"ref_baseline": 4.0}, {"steps_10": -2.4},
                           extra_runs=list(figures.SINGLE_LABELS))
    combined = write_results(tmp_path / "local", multiplicative_times(2.0, 2.0, 2.0), {},
                             extra_runs=[r for r in figures.COMBINED_LABELS if r not in factorial.RUNS])
    paths = figures.main(["--sweeps", str(single), "--combined", str(combined), "--out-dir", str(tmp_path / "fig")])
    assert [p.name for p in paths] == ["fig1_single_techniques.png", "fig2_combined.png"]
    assert all(p.stat().st_size > 10_000 for p in paths)
