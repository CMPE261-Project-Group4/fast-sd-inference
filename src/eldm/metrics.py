"""Image-quality metrics: CLIP Score, FID, and LPIPS relative to the baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

CLIP_MODEL = "openai/clip-vit-large-patch14"


def load_image(path: str | Path, size: int | None = None) -> torch.Tensor:
    """Load as uint8 CHW tensor, optionally center-cropped and resized to ``size``."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img)).permute(2, 0, 1)


def _batched(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


@torch.no_grad()
def clip_score(images: list[Path], prompts: list[str], device: str, batch_size: int = 32) -> float:
    from torchmetrics.multimodal.clip_score import CLIPScore

    metric = CLIPScore(model_name_or_path=CLIP_MODEL).to(device)
    for idx in _batched(list(range(len(images))), batch_size):
        imgs = torch.stack([load_image(images[i]) for i in idx]).to(device)
        metric.update(imgs, [prompts[i] for i in idx])
    return metric.compute().item()


@torch.no_grad()
def fid(generated: list[Path], reference: list[Path], device: str, batch_size: int = 32) -> float:
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=2048).to(device)
    for paths, real in ((reference, True), (generated, False)):
        for chunk in _batched(paths, batch_size):
            imgs = torch.stack([load_image(p, size=512) for p in chunk]).to(device)
            metric.update(imgs, real=real)
    return metric.compute().item()


@torch.no_grad()
def lpips_to_baseline(images: list[Path], baseline: list[Path], device: str, batch_size: int = 16) -> float:
    """Mean LPIPS between each image and the baseline image for the same prompt+seed.

    Lower = closer to what the unoptimized pipeline produced.
    """
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    for a_chunk, b_chunk in zip(_batched(images, batch_size), _batched(baseline, batch_size)):
        a = torch.stack([load_image(p) for p in a_chunk]).float().div(255).to(device)
        b = torch.stack([load_image(p) for p in b_chunk]).float().div(255).to(device)
        metric.update(a, b)
    return metric.compute().item()
