"""Build the frozen evaluation prompt set: 2,000 COCO 2017 val captions, then all of PartiPrompts.

    python scripts/prepare_prompts.py      # the shared datasets and the frozen settings

Ids are 0..N-1, COCO first, then PartiPrompts in file order. Each prompt's generation seed
is base_seed + id. Prompts over CLIP's 77 tokens are listed and kept: every config
truncates them the same way. See data/prompts/README.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from eldm.prompts import Prompt, build_eval_set, read_partiprompts, sample_coco, save_prompts

ROOT = Path(__file__).resolve().parents[1]
DATASETS = Path(os.environ.get("ELDM_DATASETS", "/data/cmpe261-fa26/group4/datasets"))
CLIP_MAX_TOKENS = 77


def over_token_limit(prompts: list[Prompt]) -> list[tuple[int, int]] | None:
    """(id, token count) of each prompt over CLIP's limit, or None without the pinned tokenizer."""
    try:
        from transformers import CLIPTokenizer
        from transformers.utils import logging

        from eldm.assets import hf_snapshot_dir

        logging.set_verbosity_error()  # silence "sequence length is longer than the maximum"
        tokenizer = CLIPTokenizer.from_pretrained(hf_snapshot_dir("sd15") / "tokenizer")
    except Exception as e:  # e.g. a laptop without the shared cache
        print(f"Token count skipped: SD1.5's pinned tokenizer isn't available ({type(e).__name__}: {e})")
        return None
    lengths = [(p.id, len(tokenizer(p.prompt).input_ids)) for p in prompts]
    return [(i, n) for i, n in lengths if n > CLIP_MAX_TOKENS]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-captions", type=Path, default=DATASETS / "coco/annotations/captions_val2017.json")
    ap.add_argument("--parti-tsv", type=Path, default=DATASETS / "partiprompts/PartiPrompts.tsv")
    ap.add_argument("--num-coco", type=int, default=2000)
    ap.add_argument("--sample-seed", type=int, default=0, help="Seed for choosing COCO images and captions")
    ap.add_argument("--base-seed", type=int, default=1000, help="Generation seed = base_seed + prompt id")
    ap.add_argument("--out", type=Path, default=ROOT / "data/prompts/eval_prompts.jsonl")
    ap.add_argument("--skip-token-count", action="store_true", help="Don't load the tokenizer (tests)")
    args = ap.parse_args(argv)

    coco = sample_coco(args.coco_captions, args.num_coco, args.sample_seed)
    prompts = build_eval_set(coco, read_partiprompts(args.parti_tsv), args.base_seed)
    save_prompts(prompts, args.out)

    print(f"Wrote {len(prompts)} prompts to {args.out}")
    for source, n in Counter(p.source for p in prompts).items():
        print(f"  {source}: {n}")
    for challenge, n in Counter(p.challenge for p in prompts if p.challenge).most_common():
        print(f"    {challenge}: {n}")
    if not args.skip_token_count:
        over = over_token_limit(prompts)
        if over is not None:
            print(f"Over {CLIP_MAX_TOKENS} CLIP tokens (kept): {len(over)}")
            for i, n in over:
                print(f"  id {i}: {n} tokens: {prompts[i].prompt[:80]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
