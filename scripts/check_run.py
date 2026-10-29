"""Check generated run directories, so results are verified by a script, not by eye.

For each run: every expected image (from config.yaml's prompts_file and num_prompts) exists,
with no extras, and decodes and isn't blank; perf.json has the fields the summary needs,
including NFE, stage times and the run environment, with the TF32 flags the config asked for;
timings.jsonl has one row per batch with the same NFE, and stage times that add up to between
90% and 101% of the batch's end-to-end time; with --expect-quality, quality.json has both CLIP
Scores, and KID and FID or the reason each was skipped, and quality_per_image.csv has a row
per image. Every problem is listed, not just the
first, and the exit status is non-zero if any run fails.

Example:
    python scripts/check_run.py --require-gpu A100 /scratch/cmpe261-fa26/group4/results/_checks/<jobid>/*/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from eldm.prompts import load_prompts

# Pixel standard deviation (0-255 scale) below which an image counts as blank. Stable
# Diffusion images sit far above it; black frames from fp16 NaNs and flat images are ~0.
MIN_PIXEL_STD = 2.0

# perf.json env fields every run records (eldm.benchmark._env_info), plus these on CUDA.
ENV_KEYS = ("node", "torch", "diffusers", "git_commit", "attention", "tf32_matmul", "tf32_cudnn")
CUDA_ENV_KEYS = ("gpu", "gpu_uuid")
STAGE_KEYS = ("t_text_s", "t_unet_s", "t_vae_s")
# The share of a batch's end-to-end time the three stages must account for. The rest is
# tokenizing, setting up latents and converting to PIL: a few ms. Over 1 is timer noise.
COVERAGE_RANGE = (0.90, 1.01)
QUALITY_KEYS = ("clip_vitg", "clip_vitl")


def _number_ok(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _coverage_ok(value) -> bool:
    return _number_ok(value) and COVERAGE_RANGE[0] <= value <= COVERAGE_RANGE[1]


def check_images(run_dir: Path, expected: set[str]) -> list[str]:
    images = run_dir / "images"
    found = {p.name for p in images.glob("*.png")} if images.is_dir() else set()
    problems = []
    if missing := sorted(expected - found):
        problems.append(f"{len(missing)} of {len(expected)} images missing, e.g. {missing[0]}")
    if extra := sorted(found - expected):
        problems.append(f"{len(extra)} unexpected images, e.g. {extra[0]}")
    for name in sorted(expected & found):
        try:
            with Image.open(images / name) as im:
                std = float(np.asarray(im.convert("RGB"), dtype=np.float32).std())
        except Exception as e:  # any decode failure is a broken image
            problems.append(f"{name} doesn't decode: {type(e).__name__}: {e}")
            continue
        if std < MIN_PIXEL_STD:
            problems.append(f"{name} is blank (pixel std {std:.2f})")
    return problems


def check_perf(run_dir: Path, cfg: dict, n_expected: int, require_gpu: str | None) -> list[str]:
    try:
        perf = json.loads((run_dir / "perf.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"perf.json unreadable: {type(e).__name__}: {e}"]
    problems = []
    if perf.get("name") != cfg.get("name"):
        problems.append(f"perf.json name {perf.get('name')!r} != config name {cfg.get('name')!r}")
    if perf.get("num_images") != n_expected:
        problems.append(f"perf.json num_images {perf.get('num_images')!r}, expected {n_expected}")
    for key in ("latency_per_image_s", "throughput_img_per_s"):
        if not _number_ok(perf.get(key)):
            problems.append(f"perf.json {key} is {perf.get(key)!r}")
    for key in ("nfe", "nfe_full"):
        if not _number_ok(perf.get(key)):
            problems.append(f"perf.json {key} is {perf.get(key)!r}")
    if _number_ok(perf.get("nfe")) and _number_ok(perf.get("nfe_full")) and perf["nfe_full"] > perf["nfe"]:
        problems.append(f"perf.json nfe_full {perf['nfe_full']} > nfe {perf['nfe']}")
    for key in STAGE_KEYS:
        if not _number_ok(perf.get(key)):
            problems.append(f"perf.json {key} is {perf.get(key)!r}")
    if not _coverage_ok(perf.get("stage_coverage")):
        problems.append(f"perf.json stage_coverage is {perf.get('stage_coverage')!r}, outside {COVERAGE_RANGE}")
    env = perf.get("env") or {}
    cuda = env.get("device") == "cuda"
    for key in ENV_KEYS + (CUDA_ENV_KEYS if cuda else ()):
        if env.get(key) is None:
            problems.append(f"perf.json env.{key} is missing")
    if cuda and not _number_ok(perf.get("peak_mem_gb")):
        problems.append(f"perf.json peak_mem_gb is {perf.get('peak_mem_gb')!r}")
    tf32 = cfg.get("tf32") or {}
    for flag in ("matmul", "cudnn"):
        if env.get(f"tf32_{flag}") is not None and env[f"tf32_{flag}"] != tf32.get(flag):
            problems.append(f"TF32 {flag} was {env[f'tf32_{flag}']} in the run, config asks for {tf32.get(flag)}")
    if require_gpu and require_gpu not in str(env.get("gpu") or ""):
        problems.append(f"ran on {env.get('gpu')!r}, not {require_gpu}")
    return problems


def check_timings(run_dir: Path, cfg: dict, n_expected: int) -> list[str]:
    """One timings.jsonl row per batch, each with the same NFE counts and stage times that
    account for the batch's end-to-end time."""
    try:
        rows = [json.loads(line) for line in (run_dir / "timings.jsonl").read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as e:
        return [f"timings.jsonl unreadable: {type(e).__name__}: {e}"]
    problems = []
    n_batches = math.ceil(n_expected / cfg.get("batch_size", 1))
    if len(rows) != n_batches:
        problems.append(f"timings.jsonl has {len(rows)} rows, expected {n_batches} batches")
    counts = {(r.get("nfe"), r.get("nfe_full")) for r in rows}
    if len(counts) > 1:
        problems.append(f"timings.jsonl NFE differs between batches: {sorted(counts, key=str)}")
    for nfe, nfe_full in counts:
        if not (_number_ok(nfe) and _number_ok(nfe_full) and nfe_full <= nfe):
            problems.append(f"timings.jsonl has nfe {nfe!r}, nfe_full {nfe_full!r}")
    if missing := [i for i, r in enumerate(rows) if not all(_number_ok(r.get(k)) for k in STAGE_KEYS)]:
        problems.append(f"timings.jsonl: {len(missing)} rows lack stage times, e.g. row {missing[0]}")
    else:
        coverage = [sum(r[k] for k in STAGE_KEYS) / r["latency_s"] if _number_ok(r.get("latency_s"))
                    else math.nan for r in rows]
        if bad := [(i, c) for i, c in enumerate(coverage) if not _coverage_ok(c)]:
            problems.append(f"timings.jsonl: stage coverage outside {COVERAGE_RANGE} in {len(bad)} rows,"
                            f" e.g. row {bad[0][0]}: {bad[0][1]:.3f}")
    return problems


def check_quality(run_dir: Path, ids: list[int]) -> list[str]:
    """quality.json has the run's CLIP Scores; quality_per_image.csv has a row per id, in order."""
    try:
        quality = json.loads((run_dir / "quality.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"quality.json unreadable: {type(e).__name__}: {e}"]
    problems = [f"quality.json has no {key}" for key in QUALITY_KEYS
                if not isinstance(quality.get(key), (int, float)) or isinstance(quality.get(key), bool)]
    for key, value in (("kid", "kid_mean"), ("fid", "fid")):
        if value not in quality and f"{key}_skipped" not in quality:
            problems.append(f"quality.json has neither {value} nor {key}_skipped")
    try:
        with open(run_dir / "quality_per_image.csv", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        return problems + [f"quality_per_image.csv unreadable: {type(e).__name__}: {e}"]
    if [r.get("id") for r in rows] != [str(i) for i in ids]:
        problems.append(f"quality_per_image.csv has {len(rows)} rows, not one per image in prompt order")
    return problems


def check_run(run_dir: Path, expect_quality: bool = False, require_gpu: str | None = None) -> list[str]:
    """All problems found in one run directory; an empty list means it passes."""
    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
        prompts = load_prompts(cfg["prompts_file"], cfg.get("num_prompts"))
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as e:
        return [f"can't read config.yaml or its prompts: {type(e).__name__}: {e}"]
    expected = {f"{p.id:04d}.png" for p in prompts}
    problems = check_images(run_dir, expected)
    problems += check_perf(run_dir, cfg, len(expected), require_gpu)
    problems += check_timings(run_dir, cfg, len(expected))
    if expect_quality:
        problems += check_quality(run_dir, [p.id for p in prompts])
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="Run directories")
    ap.add_argument("--expect-quality", action="store_true",
                    help="Require quality.json with CLIP Scores, and quality_per_image.csv")
    ap.add_argument("--require-gpu", help="Substring the GPU name must contain, e.g. A100")
    args = ap.parse_args(argv)

    failed = 0
    for run_dir in args.runs:
        problems = check_run(run_dir, args.expect_quality, args.require_gpu)
        if problems:
            failed += 1
            print(f"  FAIL  {run_dir}")
            for p in problems:
                print(f"          {p}")
        else:
            perf = json.loads((run_dir / "perf.json").read_text())
            env = perf.get("env") or {}
            print(f"  ok    {run_dir}: {perf['num_images']} images, {perf['latency_per_image_s']:.2f} s/image,"
                  f" NFE {perf['nfe']} ({perf['nfe_full']} full), text/UNet/VAE"
                  f" {perf['t_text_s']:.3f}/{perf['t_unet_s']:.2f}/{perf['t_vae_s']:.3f} s"
                  f" (coverage {perf['stage_coverage']:.3f}) on {env.get('gpu') or env.get('device')}"
                  f" ({env.get('node')})")
    print(f"{failed} of {len(args.runs)} run(s) failed." if failed else f"All {len(args.runs)} run(s) passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
