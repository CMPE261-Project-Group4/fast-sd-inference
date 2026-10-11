"""Build the frozen evaluation prompt set from COCO captions + manual prompts.

Example:
    python scripts/prepare_prompts.py \
        --coco-captions data/coco/annotations/captions_val2017.json --num-coco 500
"""

from __future__ import annotations

import argparse
import json
import random

import yaml

from eldm.prompts import Prompt, save_prompts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-captions", help="Path to captions_val2017.json (omit for manual prompts only)")
    ap.add_argument("--num-coco", type=int, default=500)
    ap.add_argument("--manual", default="data/prompts/manual_prompts.yaml")
    ap.add_argument("--out", default="data/prompts/eval_prompts.jsonl")
    ap.add_argument("--sample-seed", type=int, default=261, help="Seed for choosing COCO captions")
    ap.add_argument("--base-seed", type=int, default=1000, help="Generation seed = base_seed + prompt id")
    args = ap.parse_args()

    rows: list[tuple[str, str, str, int | None]] = []  # (prompt, source, category, coco_image_id)

    if args.coco_captions:
        with open(args.coco_captions) as f:
            coco = json.load(f)
        # One caption per image so FID references are not duplicated.
        by_image: dict[int, list[str]] = {}
        for ann in coco["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann["caption"].strip())
        rng = random.Random(args.sample_seed)
        image_ids = rng.sample(sorted(by_image), args.num_coco)
        for image_id in image_ids:
            rows.append((rng.choice(by_image[image_id]), "coco", "coco", image_id))

    with open(args.manual) as f:
        manual = yaml.safe_load(f)
    for category, prompts in manual.items():
        rows.extend((p, "manual", category, None) for p in prompts)

    prompts = [
        Prompt(id=i, prompt=text, source=src, category=cat, seed=args.base_seed + i, coco_image_id=img)
        for i, (text, src, cat, img) in enumerate(rows)
    ]
    save_prompts(prompts, args.out)
    n_coco = sum(p.source == "coco" for p in prompts)
    print(f"Wrote {len(prompts)} prompts ({n_coco} COCO, {len(prompts) - n_coco} manual) -> {args.out}")


if __name__ == "__main__":
    main()
