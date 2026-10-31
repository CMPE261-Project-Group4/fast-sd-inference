"""Prompt-set I/O, and the pieces scripts/prepare_prompts.py builds the frozen eval set from.

The eval set is a JSONL file with one prompt per line: COCO 2017 val captions first, then
PartiPrompts. See data/prompts/README.md.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

PARTI_COLUMNS = ["Prompt", "Category", "Challenge", "Note"]


@dataclass
class Prompt:
    id: int
    prompt: str
    source: str          # "coco" | "parti" | "manual"
    category: str        # PartiPrompts' Category; "coco" for COCO
    seed: int
    coco_image_id: int | None = None
    challenge: str | None = None  # PartiPrompts' Challenge; None for COCO


def load_prompts(path: str | Path, limit: int | None = None) -> list[Prompt]:
    with open(path) as f:
        prompts = [Prompt(**json.loads(line)) for line in f if line.strip()]
    return prompts[:limit] if limit else prompts


def save_prompts(prompts: list[Prompt], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps(asdict(p)) + "\n")


def normalize_text(text: str) -> str:
    """Collapse every run of whitespace, newlines included, to one space, and strip the ends."""
    return " ".join(text.split())


def read_partiprompts(path: str | Path) -> list[dict[str, str]]:
    """PartiPrompts.tsv's rows in file order, as dicts keyed by PARTI_COLUMNS.

    Quoting is off. The file quotes nothing, and csv's default quoting would strip the quotes
    from a prompt that starts with one: '"G I G G L E" painted in thick colorful lettering...'.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        if header != PARTI_COLUMNS:
            raise ValueError(f"{path}: expected columns {PARTI_COLUMNS}, got {header}")
        for line, fields in enumerate(reader, start=2):
            if len(fields) != len(PARTI_COLUMNS):
                raise ValueError(f"{path}:{line}: expected {len(PARTI_COLUMNS)} fields, got {len(fields)}")
            row = dict(zip(PARTI_COLUMNS, (normalize_text(x) for x in fields)))
            if not row["Prompt"]:
                raise ValueError(f"{path}:{line}: empty prompt")
            rows.append(row)
    return rows


def sample_coco(captions_json: str | Path, n: int, seed: int) -> list[tuple[int, str]]:
    """n (image_id, caption) pairs from a COCO captions file, one caption per image.

    Every image is put in one seeded order, then one caption is picked per image in that
    order, and the first n are returned. So a larger n only appends: growing the set later
    (Phase 9's FID extension) leaves the first n images and their captions unchanged.
    """
    with open(captions_json) as f:
        annotations = json.load(f)["annotations"]
    by_image: dict[int, list[str]] = {}
    for ann in sorted(annotations, key=lambda a: a["id"]):
        by_image.setdefault(ann["image_id"], []).append(normalize_text(ann["caption"]))
    if n > len(by_image):
        raise ValueError(f"asked for {n} COCO images, but {captions_json} has {len(by_image)}")
    rng = random.Random(seed)
    order = sorted(by_image)
    rng.shuffle(order)
    return [(image_id, rng.choice(by_image[image_id])) for image_id in order][:n]


def build_eval_set(coco: list[tuple[int, str]], parti: list[dict[str, str]], base_seed: int) -> list[Prompt]:
    """COCO first, then PartiPrompts in file order. Ids are 0..N-1, and seed = base_seed + id."""
    rows = [dict(prompt=caption, source="coco", category="coco", coco_image_id=image_id)
            for image_id, caption in coco]
    rows += [dict(prompt=r["Prompt"], source="parti", category=r["Category"], challenge=r["Challenge"])
             for r in parti]
    return [Prompt(id=i, seed=base_seed + i, **row) for i, row in enumerate(rows)]


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Split n across strata in proportion to their sizes.

    Largest-remainder rounding: each stratum gets the floor of its exact share, and the units
    left over go to the largest remainders, ties broken by stratum name.
    """
    total = sum(sizes.values())
    if not 0 <= n <= total:
        raise ValueError(f"can't take {n} from {total}")
    counts = {k: n * size // total for k, size in sizes.items()}
    by_remainder = sorted(sizes, key=lambda k: (-(n * sizes[k] % total), k))
    for k in by_remainder[: n - sum(counts.values())]:
        counts[k] += 1
    return counts


def stratified_sample(prompts: list[Prompt], n: int, seed: int) -> list[Prompt]:
    """n prompts spread across Challenges in proportion to their sizes, in id order.

    Each Challenge gets its own seeded order (from the seed and its name), and its first k are
    taken. So with one seed, a smaller sample is always inside a larger one.
    """
    by_challenge: dict[str, list[Prompt]] = {}
    for p in sorted(prompts, key=lambda p: p.id):
        if p.challenge is None:
            raise ValueError(f"prompt {p.id} has no challenge")
        by_challenge.setdefault(p.challenge, []).append(p)
    counts = allocate({k: len(v) for k, v in by_challenge.items()}, n)
    chosen = []
    for challenge, members in by_challenge.items():
        random.Random(f"{seed}:{challenge}").shuffle(members)
        chosen += members[: counts[challenge]]
    return sorted(chosen, key=lambda p: p.id)


def build_subsets(prompts: list[Prompt], seed: int) -> dict[str, list[Prompt]]:
    """The subset files, made of the full set's rows (same ids and seeds), each in id order.

    debug:     the first 50 rows.
    pilot:     the first 100 COCO prompts and 100 PartiPrompts, stratified by Challenge.
    parti_550: every COCO prompt and 550 stratified PartiPrompts: the fallback sweep set
               for gate G2 (Dan, 2026-10-28).
    With one seed, debug ⊂ pilot ⊂ parti_550.
    """
    coco = [p for p in prompts if p.source == "coco"]
    parti = [p for p in prompts if p.source == "parti"]
    return {
        "debug": prompts[:50],
        "pilot": coco[:100] + stratified_sample(parti, 100, seed),
        "parti_550": coco + stratified_sample(parti, 550, seed),
    }
