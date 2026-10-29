"""Compute quality metrics for run directories. Writes <run>/quality.json (the run's means) and
<run>/quality_per_image.csv (one row per image, by prompt id).

Example:
    python scripts/evaluate_quality.py --runs "results/*" \
        --baseline results/baseline --fid-ref data/coco/val2017

- clip_vitg, clip_vitl: CLIP Score with OpenCLIP ViT-g/14 (primary) and CLIP ViT-L/14, per
  image 100 * max(cos(image, prompt), 0) (eldm.metrics.ClipScorer). --clip-models picks them.
- lpips_to_baseline, with --baseline: LPIPS (AlexNet) between each image and the baseline's
  image for the same prompt and seed. The baseline itself scores 0.
- fid, with --fid-ref: FID of the run's COCO images against their val2017 images.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

import yaml

from eldm.metrics import ClipScorer, fid, lpips_to_baseline
from eldm.pipeline import resolve_device
from eldm.prompts import Prompt, load_prompts

CLIP_MODELS = ("vitg", "vitl")


def run_images(run_dir: Path, prompts: list[Prompt]) -> list[Path]:
    images = [run_dir / "images" / f"{p.id:04d}.png" for p in prompts]
    missing = [p for p in images if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: {len(missing)} images missing, e.g. {missing[0]}")
    return images


def evaluate(run_dir: Path, args, device: str, scorers: dict[str, ClipScorer]) -> tuple[dict, list[dict]]:
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

    if args.fid_ref:
        coco = [(img, p) for img, p in zip(images, prompts) if p.coco_image_id is not None]
        if len(coco) < 50:
            print(f"[{cfg['name']}] skipping FID: only {len(coco)} COCO prompts")
        else:
            refs = [Path(args.fid_ref) / f"{p.coco_image_id:012d}.jpg" for _, p in coco]
            result["fid"] = fid([img for img, _ in coco], refs, device)
            result["fid_num_images"] = len(coco)
    rows = [dict(zip(per_image, values)) for values in zip(*per_image.values())]
    return result, rows


def write_results(run_dir: Path, result: dict, rows: list[dict]) -> None:
    (run_dir / "quality.json").write_text(json.dumps(result, indent=2))
    with open(run_dir / "quality_per_image.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories or glob patterns")
    ap.add_argument("--baseline", help="Baseline run dir for LPIPS comparison")
    ap.add_argument("--fid-ref", help="Directory of COCO val2017 images for FID")
    ap.add_argument("--clip-models", nargs="+", choices=CLIP_MODELS, default=list(CLIP_MODELS))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = resolve_device(args.device)
    run_dirs = sorted({Path(p) for pat in args.runs for p in glob.glob(pat)})
    run_dirs = [d for d in run_dirs if (d / "config.yaml").exists()]
    scorers = {name: ClipScorer.load(name, device) for name in args.clip_models}  # loaded once for all runs
    for run_dir in run_dirs:
        result, rows = evaluate(run_dir, args, device, scorers)
        write_results(run_dir, result, rows)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
