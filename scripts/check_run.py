"""Check generated run directories, so results are verified by a script, not by eye.

For each run: every expected image (from config.yaml's prompts_file and num_prompts) exists,
with no extras, and decodes and isn't blank; perf.json has the fields the summary needs;
with --expect-quality, quality.json has a CLIP score. Every problem is listed, not just the
first, and the exit status is non-zero if any run fails.

Example:
    python scripts/check_run.py --require-gpu A100 /scratch/cmpe261-fa26/group4/results/_checks/<jobid>/*/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from eldm.prompts import load_prompts

# Pixel standard deviation (0-255 scale) below which an image counts as blank. Stable
# Diffusion images sit far above it; black frames from fp16 NaNs and flat images are ~0.
MIN_PIXEL_STD = 2.0


def _number_ok(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


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
    env = perf.get("env") or {}
    if env.get("device") == "cuda":
        if not _number_ok(perf.get("peak_mem_gb")):
            problems.append(f"perf.json peak_mem_gb is {perf.get('peak_mem_gb')!r}")
        if not env.get("gpu"):
            problems.append("perf.json env.gpu is missing")
    if require_gpu and require_gpu not in str(env.get("gpu") or ""):
        problems.append(f"ran on {env.get('gpu')!r}, not {require_gpu}")
    return problems


def check_quality(run_dir: Path) -> list[str]:
    try:
        quality = json.loads((run_dir / "quality.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"quality.json unreadable: {type(e).__name__}: {e}"]
    return [] if "clip_score" in quality else ["quality.json has no clip_score"]


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
    if expect_quality:
        problems += check_quality(run_dir)
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="Run directories")
    ap.add_argument("--expect-quality", action="store_true", help="Require quality.json with clip_score")
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
            print(f"  ok    {run_dir}: {perf['num_images']} images, {perf['latency_per_image_s']:.2f} s/image"
                  f" on {(perf.get('env') or {}).get('gpu') or (perf.get('env') or {}).get('device')}")
    print(f"{failed} of {len(args.runs)} run(s) failed." if failed else f"All {len(args.runs)} run(s) passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
