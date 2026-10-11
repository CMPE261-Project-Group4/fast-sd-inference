"""Compute quality metrics for one or more run directories; writes <run>/quality.json.

Example:
    python scripts/evaluate_quality.py --runs "results/*" \
        --baseline results/baseline --fid-ref data/coco/val2017
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import yaml

from eldm.metrics import clip_score, fid, lpips_to_baseline
from eldm.pipeline import resolve_device
from eldm.prompts import load_prompts


def evaluate(run_dir: Path, args, device: str) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    prompts = load_prompts(cfg["prompts_file"], cfg.get("num_prompts"))
    images = [run_dir / "images" / f"{p.id:04d}.png" for p in prompts]
    missing = [p for p in images if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: {len(missing)} images missing, e.g. {missing[0]}")

    result = {"name": cfg["name"], "num_images": len(images)}
    result["clip_score"] = clip_score(images, [p.prompt for p in prompts], device)

    if args.baseline and Path(args.baseline).resolve() != run_dir.resolve():
        base_imgs = [Path(args.baseline) / "images" / f"{p.id:04d}.png" for p in prompts]
        result["lpips_to_baseline"] = lpips_to_baseline(images, base_imgs, device)

    if args.fid_ref:
        coco = [(img, p) for img, p in zip(images, prompts) if p.coco_image_id is not None]
        if len(coco) < 50:
            print(f"[{cfg['name']}] skipping FID: only {len(coco)} COCO prompts")
        else:
            refs = [Path(args.fid_ref) / f"{p.coco_image_id:012d}.jpg" for _, p in coco]
            result["fid"] = fid([img for img, _ in coco], refs, device)
            result["fid_num_images"] = len(coco)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories or glob patterns")
    ap.add_argument("--baseline", help="Baseline run dir for LPIPS comparison")
    ap.add_argument("--fid-ref", help="Directory of COCO val2017 images for FID")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    run_dirs = sorted({Path(p) for pat in args.runs for p in glob.glob(pat)})
    for run_dir in run_dirs:
        if not (run_dir / "config.yaml").exists():
            continue
        result = evaluate(run_dir, args, device)
        (run_dir / "quality.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
