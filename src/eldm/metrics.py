"""Image-quality metrics: CLIP Score, FID, and LPIPS relative to the baseline.

Every model loads offline from the pinned caches (eldm.assets): COE-HPC3 has no internet.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .assets import hf_snapshot_dir

# Encoders from a list of PIL images, or of prompts, to one embedding per item.
Encoder = Callable[[list], torch.Tensor]


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


class ClipScorer:
    """CLIP Score per image: 100 * max(cos(image embedding, prompt embedding), 0), the definition
    torchmetrics' CLIPScore uses. A run's CLIP Score is the mean. Two pinned models:

    - "vitg": OpenCLIP ViT-g/14 (LAION-2B s34B b88K), the primary score;
    - "vitl": OpenAI CLIP ViT-L/14, the secondary one: the model most SD papers report with.

    Prompts over CLIP's 77 tokens are truncated, as SD's own text encoder truncates them.
    """

    def __init__(self, encode_images: Encoder, encode_texts: Encoder, batch_size: int = 32) -> None:
        self.encode_images, self.encode_texts, self.batch_size = encode_images, encode_texts, batch_size

    @classmethod
    def load(cls, name: str, device: str, batch_size: int = 32) -> ClipScorer:
        loaders = {"vitg": _openclip_vitg14, "vitl": _clip_vitl14}
        if name not in loaders:
            raise ValueError(f"unknown CLIP model {name!r}; choose from {sorted(loaders)}")
        return cls(*loaders[name](device), batch_size=batch_size)

    @torch.no_grad()
    def scores(self, images: list[Path], prompts: list[str]) -> list[float]:
        """One score per image, in order."""
        if len(images) != len(prompts):
            raise ValueError(f"{len(images)} images but {len(prompts)} prompts")
        out: list[float] = []
        for idx in _batched(list(range(len(images))), self.batch_size):
            pil = [Image.open(images[i]).convert("RGB") for i in idx]
            image_emb = self.encode_images(pil).float()
            text_emb = self.encode_texts([prompts[i] for i in idx]).float()
            cos = torch.nn.functional.cosine_similarity(image_emb, text_emb, dim=-1)
            out += (100 * cos.clamp(min=0)).tolist()
        return out


def _openclip_vitg14(device: str) -> tuple[Encoder, Encoder]:
    import open_clip

    weights = hf_snapshot_dir("openclip_vitg14") / "open_clip_model.safetensors"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-g-14", pretrained=str(weights), device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-g-14")  # 77 tokens, truncated with the end token kept

    def encode_images(images: list) -> torch.Tensor:
        return model.encode_image(torch.stack([preprocess(im) for im in images]).to(device))

    def encode_texts(prompts: list) -> torch.Tensor:
        return model.encode_text(tokenizer(prompts).to(device))

    return encode_images, encode_texts


def _clip_vitl14(device: str) -> tuple[Encoder, Encoder]:
    from transformers import CLIPModel, CLIPProcessor

    snapshot = hf_snapshot_dir("clip_vitl14")
    model = CLIPModel.from_pretrained(snapshot).to(device).eval()
    processor = CLIPProcessor.from_pretrained(snapshot)

    def projected(output) -> torch.Tensor:
        # transformers 5 returns BaseModelOutputWithPooling from get_*_features, with the
        # projected embedding in .pooler_output; transformers 4 returned that tensor.
        return output if isinstance(output, torch.Tensor) else output.pooler_output

    def encode_images(images: list) -> torch.Tensor:
        return projected(model.get_image_features(**processor(images=images, return_tensors="pt").to(device)))

    def encode_texts(prompts: list) -> torch.Tensor:
        tokens = processor(text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        return projected(model.get_text_features(**tokens.to(device)))

    return encode_images, encode_texts


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
def lpips_to_baseline(images: list[Path], baseline: list[Path], device: str, batch_size: int = 16) -> list[float]:
    """LPIPS (AlexNet) between each image and the baseline's image for the same prompt and seed,
    in order. Lower is closer to what the unoptimized pipeline produced; identical images give 0."""
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    if len(images) != len(baseline):
        raise ValueError(f"{len(images)} images but {len(baseline)} baseline images")
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True, reduction="none").to(device)
    for a_chunk, b_chunk in zip(_batched(images, batch_size), _batched(baseline, batch_size)):
        a = torch.stack([load_image(p) for p in a_chunk]).float().div(255).to(device)
        b = torch.stack([load_image(p) for p in b_chunk]).float().div(255).to(device)
        metric.update(a, b)
    return metric.compute().flatten().tolist()
