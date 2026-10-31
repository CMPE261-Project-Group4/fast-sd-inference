"""Compute quality metrics for run directories. Writes <run>/quality.json (the run's means) and
<run>/quality_per_image.csv (one row per image, by prompt id).

Example:
    python scripts/evaluate_quality.py --runs "results/*" --baseline results/baseline

- clip_vitg, clip_vitl: CLIP Score with OpenCLIP ViT-g/14 (primary) and CLIP ViT-L/14, per
  image 100 * max(cos(image, prompt), 0) (eldm.metrics.ClipScorer). --clip-models picks them.
- lpips_to_baseline, with --baseline: LPIPS (AlexNet) between each image and the baseline's
  image for the same prompt and seed. The baseline itself scores 0.
- kid_mean, kid_std: KID of the run's COCO images against the val2017 photos their captions
  describe (--coco-images), with subsets of min(1000, n) images.
- fid: FID against the same photos, only from 5,000 COCO images; below that, fid_skipped
  says why. KID and FID are per run, not per image.

Every weights file is re-hashed against eldm.assets' pins first. The photos' Inception
features are computed once per set of photos and reused across runs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import time
from pathlib import Path

import yaml

from eldm import assets
from eldm.metrics import FID_MIN_IMAGES, ClipScorer, InceptionMetrics, kid_subset_size, lpips_to_baseline
from eldm.pipeline import resolve_device
from eldm.prompts import Prompt, load_prompts

CLIP_MODELS = {"vitg": "openclip_vitg14", "vitl": "clip_vitl14"}  # name -> eldm.assets.HF key
COCO_IMAGES = Path(os.environ.get("ELDM_DATASETS", "/data/cmpe261-fa26/group4/datasets")) / "coco/val2017"


def run_images(run_dir: Path, prompts: list[Prompt]) -> list[Path]:
    images = [run_dir / "images" / f"{p.id:04d}.png" for p in prompts]
    missing = [p for p in images if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: {len(missing)} images missing, e.g. {missing[0]}")
    return images


def realism(images: list[Path], prompts: list[Prompt], coco_images: Path, device: str,
            references: dict[tuple[int, ...], InceptionMetrics]) -> dict:
    """KID (and FID) of the run's COCO images against their own photos. ``references`` caches
    one InceptionMetrics per set of photos, so runs on the same prompts share its features."""
    coco = [(img, p.coco_image_id) for img, p in zip(images, prompts) if p.coco_image_id is not None]
    n = len(coco)
    out: dict = {"kid_num_images": n}
    if n < 2:
        return out | {"kid_skipped": f"{n} COCO images; KID needs at least 2",
                      "fid_skipped": f"{n} COCO images; FID needs at least {FID_MIN_IMAGES:,}"}
    ids = tuple(i for _, i in coco)
    if ids not in references:
        references[ids] = InceptionMetrics([coco_images / f"{i:012d}.jpg" for i in ids], device)
    out |= {"kid_subset_size": kid_subset_size(n), **references[ids].score([img for img, _ in coco])}
    if "fid" not in out:
        out["fid_skipped"] = f"{n:,} COCO images; FID needs at least {FID_MIN_IMAGES:,}"
    return out


def evaluate(run_dir: Path, args, device: str, scorers: dict[str, ClipScorer],
             references: dict[tuple[int, ...], InceptionMetrics]) -> tuple[dict, list[dict]]:
    """The run's quality.json and its per-image rows."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    prompts = load_prompts(cfg["prompts_file"], cfg.get("num_prompts"))
    images = run_images(run_dir, prompts)

    per_image: dict[str, list] = {"id": [p.id for p in prompts]}
    for name, scorer in scorers.items():
        per_image[f"clip_{name}"] = scorer.scores(images, [p.prompt for p in prompts])
    if args.baseline:
        per_image["lpips_to_baseline"] = lpips_to_baseline(images, run_images(Path(args.baseline), prompts), device)

    result = {"name": cfg["name"], "num_images": len(images)}
    result.update({key: statistics.mean(values) for key, values in per_image.items() if key != "id"})
    if args.baseline:
        result["baseline"] = str(Path(args.baseline).resolve())  # what lpips_to_baseline compares with
    result.update(realism(images, prompts, Path(args.coco_images), device, references))
    rows = [dict(zip(per_image, values)) for values in zip(*per_image.values())]
    return result, rows


def write_results(run_dir: Path, result: dict, rows: list[dict]) -> None:
    (run_dir / "quality.json").write_text(json.dumps(result, indent=2))
    with open(run_dir / "quality_per_image.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_weights(clip_models: list[str]) -> None:
    """Re-hash every weights file the evaluation loads against its pin in eldm.assets."""
    for name in clip_models:
        t0 = time.perf_counter()
        n, size = assets.verify_hf(CLIP_MODELS[name])
        print(f"verified {CLIP_MODELS[name]}: {n} files, {size / 1e9:.2f} GB ({time.perf_counter() - t0:.0f} s)")
    for name in ("inception_fid", "alexnet_lpips"):
        assets.verify_torch_hub(name)
        print(f"verified {name}: {assets.torch_hub_file(name)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories or glob patterns")
    ap.add_argument("--baseline", help="Baseline run dir for LPIPS comparison")
    ap.add_argument("--coco-images", type=Path, default=COCO_IMAGES,
                    help="COCO val2017 photos for KID and FID (default: %(default)s)")
    ap.add_argument("--clip-models", nargs="+", choices=list(CLIP_MODELS), default=list(CLIP_MODELS))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = resolve_device(args.device)
    run_dirs = sorted({Path(p) for pat in args.runs for p in glob.glob(pat)})
    run_dirs = [d for d in run_dirs if (d / "config.yaml").exists()]
    if not run_dirs:
        raise SystemExit(f"no run directories (with config.yaml) match {args.runs}")
    verify_weights(args.clip_models)
    scorers = {name: ClipScorer.load(name, device) for name in args.clip_models}  # loaded once for all runs
    references: dict[tuple[int, ...], InceptionMetrics] = {}
    for run_dir in run_dirs:
        t0 = time.perf_counter()
        result, rows = evaluate(run_dir, args, device, scorers, references)
        write_results(run_dir, result, rows)
        print(f"{run_dir} ({time.perf_counter() - t0:.0f} s): {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
