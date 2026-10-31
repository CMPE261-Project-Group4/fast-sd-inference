"""GPU smoke check for gate G1: four variants of the real pipeline make images on an A100.

    python scripts/hpc/gpu_check.py --stage "$STAGE" --out "$OUT"     # in check_gpu.sbatch

Each variant runs through eldm.benchmark.run (the production path: pipeline, timing,
perf.json) on 2 prompts from data/prompts/manual_prompts.yaml with fixed seeds. It writes
to STAGE on the node-local SSD, and the run directories are then copied to OUT. A failing
variant doesn't stop the others. Prints a table and writes OUT/summary.json. Exits 1 if
any variant failed, or if FP16 or DeepCache isn't faster than the FP32 baseline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import torch
import yaml

from eldm import benchmark
from eldm.config import deep_merge, load_yaml
from eldm.prompts import Prompt, save_prompts

ROOT = Path(__file__).resolve().parents[2]

# Overrides on configs/base.yaml: SD1.5 at the pinned revision, PNDM, 50 steps, FP32, 512x512.
VARIANTS = {
    "baseline": {},
    "fp16": {"dtype": "fp16"},
    "dpmpp_20": {"scheduler": "dpmpp_2m", "num_inference_steps": 20},
    "deepcache_i3": {"deepcache": {"enabled": True, "cache_interval": 3}},
}
MUST_BEAT_BASELINE = ("fp16", "deepcache_i3")  # gate G1
NUM_PROMPTS = 2


def smoke_prompts(n: int = NUM_PROMPTS) -> list[Prompt]:
    """The first n manual prompts, in file order, with fixed seeds."""
    manual = yaml.safe_load((ROOT / "data/prompts/manual_prompts.yaml").read_text())
    texts = [(cat, text) for cat, prompts in manual.items() for text in prompts][:n]
    return [Prompt(id=i, prompt=text, source="manual", category=cat, seed=1000 + i)
            for i, (cat, text) in enumerate(texts)]


def variant_configs(base: dict, prompts_file: Path, stage: Path, device: str = "cuda") -> dict[str, dict]:
    shared = {"device": device, "prompts_file": str(prompts_file), "num_prompts": NUM_PROMPTS,
              "output_root": str(stage)}
    return {name: deep_merge(base, {**overrides, **shared, "name": name}) for name, overrides in VARIANTS.items()}


def summarize(perfs: dict[str, dict], errors: dict[str, str]) -> dict:
    rows = {name: {"latency_per_image_s": p["latency_per_image_s"], "peak_mem_gb": p.get("peak_mem_gb")}
            for name, p in perfs.items()}
    base = rows.get("baseline", {}).get("latency_per_image_s")
    for row in rows.values():
        row["speedup"] = base / row["latency_per_image_s"] if base else None
    slower = [n for n in MUST_BEAT_BASELINE if base is None or n not in rows or rows[n]["latency_per_image_s"] >= base]
    return {"variants": rows, "errors": errors, "not_faster_than_baseline": slower,
            "gate_g1": not errors and not slower}


def gpu_info() -> dict:
    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="Where the run directories end up")
    ap.add_argument("--stage", type=Path, required=True, help="Node-local scratch directory")
    ap.add_argument("--allow-any-gpu", action="store_true", help="Don't require an A100 (local use)")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("FAIL: no CUDA GPU visible")
        return 1
    info = gpu_info()
    if "A100" not in info["gpu"] and not args.allow_any_gpu:
        print(f"FAIL: running on {info['gpu']!r}, not an A100 (check --gres=gpu:a100:1)")
        return 1
    # Each run's TF32 flags are set from configs/base.yaml and recorded in its perf.json env.
    print(f"GPU check on {info['gpu']}, torch {info['torch']} (CUDA {info['cuda']}, cuDNN {info['cudnn']})",
          flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    prompts_file = args.out / "prompts.jsonl"
    save_prompts(smoke_prompts(), prompts_file)

    perfs, errors = {}, {}
    for name, cfg in variant_configs(load_yaml(ROOT / "configs/base.yaml"), prompts_file, args.stage).items():
        try:
            perfs[name] = benchmark.run(cfg)
            print(f"  ok    {name}: {perfs[name]['latency_per_image_s']:.2f} s/image", flush=True)
        except Exception as e:  # keep going: one job should report on every variant
            errors[name] = f"{type(e).__name__}: {e}"
            print(f"  FAIL  {name}: {errors[name]}", flush=True)
            traceback.print_exc()
        if (args.stage / name).exists():
            shutil.copytree(args.stage / name, args.out / name, dirs_exist_ok=True)

    summary = {**info, "prompts_file": str(prompts_file), **summarize(perfs, errors)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'variant':<14}{'s/image':>9}{'speedup':>9}{'peak GB':>9}")
    for name, row in summary["variants"].items():
        speedup = f"{row['speedup']:.2f}x" if row["speedup"] else "-"
        peak = f"{row['peak_mem_gb']:.2f}" if row["peak_mem_gb"] is not None else "-"
        print(f"{name:<14}{row['latency_per_image_s']:>9.2f}{speedup:>9}{peak:>9}")
    if summary["not_faster_than_baseline"]:
        print(f"FAIL: not faster than baseline: {', '.join(summary['not_faster_than_baseline'])}")
    print(f"Gate G1 {'passed' if summary['gate_g1'] else 'FAILED'}; outputs in {args.out}")
    return 0 if summary["gate_g1"] else 1


if __name__ == "__main__":
    sys.exit(main())
