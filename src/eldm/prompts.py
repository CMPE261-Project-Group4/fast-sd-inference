"""Prompt-set I/O. The eval set is a JSONL file with one prompt per line."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Prompt:
    id: int
    prompt: str
    source: str          # "coco" | "manual"
    category: str
    seed: int
    coco_image_id: int | None = None


def load_prompts(path: str | Path, limit: int | None = None) -> list[Prompt]:
    with open(path) as f:
        prompts = [Prompt(**json.loads(line)) for line in f if line.strip()]
    return prompts[:limit] if limit else prompts


def save_prompts(prompts: list[Prompt], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps(asdict(p)) + "\n")
