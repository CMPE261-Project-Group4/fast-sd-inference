"""Estimate a sweep's GPU-hours from the pilot's measured runs (gate G2).

Example:
    python scripts/estimate_sweep.py --pilot "results/p6/pilot/*" \
        --pilot-elapsed-s 3000 --eval-elapsed-s 240 \
        --batch-pair results/p5/debug_a/baseline results/p5/debug_a/baseline_bs4 \
        --csv report/pilot/sweep_estimate.csv

The model, per image:  t = fixed[dtype] + nfe_full * c_full[dtype] + (nfe - nfe_full) * c_partial[dtype]

- nfe and nfe_full are exact, from the run's scheduler: SD1.5's pinned scheduler config,
  set_timesteps(steps), one UNet call per timestep. Under DeepCache a call is a full pass when
  its step index is a multiple of the interval; DeepCache 0.1.1 finds the index with
  timesteps.index(t), so PNDM's repeated timestep reuses the first one's. Every pilot run's
  measured NFE must match, or the script stops.
- fixed (text encoder, VAE, the rest of the call) and c_full (one full UNet call) come from the
  pilot's 50-step runs "baseline" (FP32) and "fp16"; c_partial (a call DeepCache serves from its
  cache) from "deepcache_i3". The other pilot runs are held out and printed as a check.
- Batch 4 multiplies the batch-1 time by --batch-pair's measured batch-4/batch-1 ratio.
- Each run also pays its warm-up batches, plus a per-image job overhead (loading, saving
  images) and evaluation, from the pilot jobs' elapsed times. Loading is spread over the
  pilot's images, which slightly overstates it for longer runs.

Runs the pilot measured use their measured time, the rest the model; ``basis`` says which and
what the estimate assumes. Rows named "(total)" sum each group's GPU-hours.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from eldm.config import load_experiment
from eldm.pipeline import SCHEDULERS

ROOT = Path(__file__).resolve().parents[1]
FIT_RUNS = {"fp32": "baseline", "fp16": "fp16"}  # dtype -> 50-step pilot run, every call a full pass
DEEPCACHE_RUN = "deepcache_i3"
# Config keys that change a run's time. A pilot run stands in for the sweep run of the same name
# only if these match.
TIMING_KEYS = ("scheduler", "num_inference_steps", "dtype", "batch_size", "deepcache", "height", "width",
               "guidance_scale", "tf32")
SWEEP = sorted((ROOT / "configs/experiments").glob("0[0-4]_*.yaml"))  # Phase 8: each optimization alone
LATER = [ROOT / "configs/experiments/05_combined.yaml"]  # Phase 9
PROMPT_SETS = [ROOT / "data/prompts/eval_prompts.jsonl", ROOT / "data/prompts/subsets/parti_550.jsonl"]


@dataclass
class Run:
    cfg: dict
    perf: dict


@dataclass
class Model:
    fixed: dict[str, float]  # dtype -> s/image outside the UNet loop
    c_full: dict[str, float]  # dtype -> s/image per full UNet call
    c_partial: dict[str, float]  # dtype -> s/image per call DeepCache serves from its cache
    partial_dtype: str  # the dtype c_partial was measured in
    batch_ratio: float  # s/image at batch 4 over s/image at batch 1
    batch_dtype: str  # the dtype the ratio was measured in


def sd15_scheduler_config() -> dict:
    from eldm import assets

    return json.loads((assets.hf_snapshot_dir("sd15") / "scheduler/scheduler_config.json").read_text())


def expected_nfe(cfg: dict, scheduler_config: dict) -> tuple[int, int]:
    """(nfe, nfe_full) of one pipeline call with this run config."""
    scheduler = SCHEDULERS[cfg["scheduler"]].from_config(scheduler_config)
    scheduler.set_timesteps(cfg["num_inference_steps"])
    timesteps = scheduler.timesteps.tolist()
    if not cfg["deepcache"]["enabled"]:
        return len(timesteps), len(timesteps)
    interval = cfg["deepcache"]["cache_interval"]
    return len(timesteps), sum(timesteps.index(t) % interval == 0 for t in timesteps)


def read_runs(run_dirs: list[Path]) -> dict[str, Run]:
    """Run directories with a perf.json, by run name."""
    runs: dict[str, Run] = {}
    for d in run_dirs:
        if not (d / "perf.json").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        if cfg["name"] in runs:
            raise SystemExit(f"two runs named {cfg['name']} among {[str(d) for d in run_dirs]}")
        runs[cfg["name"]] = Run(cfg, json.loads((d / "perf.json").read_text()))
    return runs


def timing_config(cfg: dict) -> dict:
    return {k: cfg[k] for k in TIMING_KEYS}


def fit(pilot: dict[str, Run], bs1: Run, bs4: Run) -> Model:
    if missing := sorted({*FIT_RUNS.values(), DEEPCACHE_RUN} - set(pilot)):
        raise SystemExit(f"the pilot has no run named {', '.join(missing)}")
    fixed, c_full = {}, {}
    for dtype, name in FIT_RUNS.items():
        perf = pilot[name].perf
        c_full[dtype] = perf["t_unet_s"] / perf["nfe"]
        fixed[dtype] = perf["latency_per_image_s"] - perf["t_unet_s"]
    dc = pilot[DEEPCACHE_RUN]
    dtype, perf = dc.cfg["dtype"], dc.perf
    partial = (perf["t_unet_s"] - perf["nfe_full"] * c_full[dtype]) / (perf["nfe"] - perf["nfe_full"])
    # A cached call in another dtype is assumed to scale like a full call.
    c_partial = {d: partial * c_full[d] / c_full[dtype] for d in c_full}
    if (bs1.cfg["batch_size"], bs4.cfg["batch_size"]) != (1, 4) or bs1.cfg["dtype"] != bs4.cfg["dtype"]:
        raise SystemExit("--batch-pair needs a batch-1 and a batch-4 run of the same dtype")
    ratio = bs4.perf["latency_per_image_s"] / bs1.perf["latency_per_image_s"]
    return Model(fixed, c_full, c_partial, dtype, ratio, bs1.cfg["dtype"])


def predict(cfg: dict, nfe: int, nfe_full: int, model: Model) -> tuple[float, list[str]]:
    """Seconds per image, and what the prediction assumes beyond the measured runs."""
    dtype, assumes = cfg["dtype"], []
    t = model.fixed[dtype] + nfe_full * model.c_full[dtype] + (nfe - nfe_full) * model.c_partial[dtype]
    if nfe_full < nfe and dtype != model.partial_dtype:
        assumes.append(f"{dtype} cached calls scale like full ones")
    if cfg["batch_size"] != 1:
        if cfg["batch_size"] != 4:
            raise SystemExit(f"{cfg['name']}: only batch sizes 1 and 4 are modeled")
        t *= model.batch_ratio
        if dtype != model.batch_dtype:
            assumes.append(f"{model.batch_dtype}'s batch-4 ratio")
    return t, assumes


def estimate(experiments: list[Path], group: str, reruns: list[str], pilot: dict[str, Run], model: Model,
             scheduler_config: dict, prompt_sets: dict[str, int], overhead: float, eval_per_image: float) -> list[dict]:
    """One row per run, plus the group's "(total)" row."""
    rows = []
    for path in experiments:
        for cfg in load_experiment(path, ROOT):
            nfe, nfe_full = expected_nfe(cfg, scheduler_config)
            t_model, assumes = predict(cfg, nfe, nfe_full, model)
            measured = pilot.get(cfg["name"])
            if measured and timing_config(measured.cfg) != timing_config(cfg):
                raise SystemExit(f"pilot run {cfg['name']} doesn't match the sweep run of that name")
            t_measured = measured.perf["latency_per_image_s"] if measured else None
            t = t_measured if measured else t_model
            row = {
                "group": group, "experiment": path.stem, "run": cfg["name"], "scheduler": cfg["scheduler"],
                "steps": cfg["num_inference_steps"], "dtype": cfg["dtype"], "batch_size": cfg["batch_size"],
                "deepcache_interval": cfg["deepcache"]["cache_interval"] if cfg["deepcache"]["enabled"] else None,
                "nfe": nfe, "nfe_full": nfe_full, "s_per_image": t,
                "basis": "measured (pilot)" if measured else "; ".join(["model", *assumes]),
                "model_s_per_image": t_model, "measured_s_per_image": t_measured,
            }
            warmup = cfg["warmup_batches"] * cfg["batch_size"]
            for name, n in prompt_sets.items():
                row[f"gen_gpu_h_{name}"] = ((n + warmup) * t + n * overhead) / 3600
                row[f"eval_gpu_h_{name}"] = n * eval_per_image / 3600
            rows.append(row)
            if cfg["name"] in reruns:
                rows.append({**row, "run": f"{cfg['name']} (second node)"})
    total = {"group": group, "run": "(total)"}
    for key in rows[0]:
        if key.startswith(("gen_gpu_h_", "eval_gpu_h_")):
            total[key] = sum(r[key] for r in rows)
    return rows + [total]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", nargs="+", required=True, help="Pilot run directories or glob patterns")
    ap.add_argument("--pilot-elapsed-s", type=float, required=True, help="The pilot job's elapsed time (sacct)")
    ap.add_argument("--eval-elapsed-s", type=float, required=True, help="The pilot's evaluation job's elapsed time")
    ap.add_argument("--batch-pair", nargs=2, type=Path, required=True, metavar=("BS1_RUN", "BS4_RUN"),
                    help="Run directories of the same config at batch 1 and batch 4")
    ap.add_argument("--sweep", nargs="+", type=Path, default=SWEEP, help="Experiment files of the sweep")
    ap.add_argument("--rerun", nargs="*", default=["baseline"], help="Sweep runs repeated on a second node")
    ap.add_argument("--later", nargs="*", type=Path, default=LATER, help="Experiment files totalled separately")
    ap.add_argument("--prompt-sets", nargs="+", type=Path, default=PROMPT_SETS)
    ap.add_argument("--scheduler-config", type=Path, help="Default: SD1.5's pinned scheduler_config.json")
    ap.add_argument("--gpus", type=int, default=4, help="GPUs running the sweep at once, for the wall time")
    ap.add_argument("--csv", type=Path, help="Write the rows here")
    args = ap.parse_args(argv)

    pilot = read_runs(sorted({Path(p) for pat in args.pilot for p in glob.glob(pat)}))
    if not pilot:
        raise SystemExit(f"no pilot runs (with perf.json) match {args.pilot}")
    scheduler_config = (json.loads(args.scheduler_config.read_text()) if args.scheduler_config
                        else sd15_scheduler_config())
    for name, run in pilot.items():
        if (expected := expected_nfe(run.cfg, scheduler_config)) != (run.perf["nfe"], run.perf["nfe_full"]):
            raise SystemExit(f"{name}: expected NFE {expected}, measured {(run.perf['nfe'], run.perf['nfe_full'])}")
    pair = read_runs(args.batch_pair)
    bs1, bs4 = sorted(pair.values(), key=lambda r: r.cfg["batch_size"])
    model = fit(pilot, bs1, bs4)

    images = sum(r.perf["num_images"] for r in pilot.values())
    busy = sum(r.perf["latency_per_image_s"] * (r.perf["num_images"] + r.cfg["warmup_batches"] * r.cfg["batch_size"])
               for r in pilot.values())
    overhead = (args.pilot_elapsed_s - busy) / images
    eval_per_image = args.eval_elapsed_s / images
    prompt_sets = {p.stem: sum(1 for line in p.open() if line.strip()) for p in args.prompt_sets}

    print(f"model: fixed {model.fixed}, full call {model.c_full}, cached call {model.c_partial} s/image; "
          f"batch-4 ratio {model.batch_ratio:.3f} ({model.batch_dtype}); job overhead {overhead:.3f} s/image, "
          f"evaluation {eval_per_image:.3f} s/image")
    print("pilot runs, measured vs model (s/image):")
    for name, run in pilot.items():
        t_model, _ = predict(run.cfg, run.perf["nfe"], run.perf["nfe_full"], model)
        t = run.perf["latency_per_image_s"]
        role = "fit" if name in {*FIT_RUNS.values(), DEEPCACHE_RUN} else "held out"
        print(f"  {name:14} {t:7.3f} {t_model:7.3f} {100 * (t_model - t) / t:+6.1f}%  ({role})")

    rows = estimate(args.sweep, "sweep", args.rerun, pilot, model, scheduler_config, prompt_sets, overhead,
                    eval_per_image)
    if args.later:
        rows += estimate(args.later, "later", [], pilot, model, scheduler_config, prompt_sets, overhead,
                         eval_per_image)
    for total in (r for r in rows if r["run"] == "(total)"):
        for name, n in prompt_sets.items():
            gpu_h = total[f"gen_gpu_h_{name}"] + total[f"eval_gpu_h_{name}"]
            print(f"{total['group']:5} on {name} ({n:,} prompts): {total[f'gen_gpu_h_{name}']:.1f} GPU-h generation"
                  f" + {total[f'eval_gpu_h_{name}']:.1f} evaluation = {gpu_h:.1f} GPU-h, "
                  f"{gpu_h / args.gpus:.1f} h on {args.gpus} GPUs")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
