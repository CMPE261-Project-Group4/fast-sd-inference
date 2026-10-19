"""Pinned external assets: model snapshots and metric weights.

One source of truth for scripts/hpc/prefetch_assets.py (downloads on coe-hpc1), the
offline check job and the loaders. COE-HPC3 has no internet, so every asset is pinned to
an immutable version, downloaded ahead of time, and loaded here from the local cache only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HFAsset:
    """A Hugging Face repo pinned to a commit, limited to the files we use."""

    repo: str
    revision: str            # full commit hash: immutable, unlike a branch or tag
    allow: tuple[str, ...]   # allow_patterns for snapshot_download


@dataclass(frozen=True)
class FileAsset:
    """A weights file that torch.hub caches under $TORCH_HOME/hub/checkpoints."""

    url: str
    sha256: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]

    @property
    def hash_prefix(self) -> str:
        """torch.hub convention: the filename ends in the first hex digits of its sha256."""
        return Path(self.filename).stem.rsplit("-", 1)[-1]


HF: dict[str, HFAsset] = {
    # Stable Diffusion v1.5 in safetensors (fp32 EMA weights; FP16 runs cast them). The
    # safety checker is not loaded (safety_checker=None), so it is not fetched; neither
    # are the .bin, .fp16 and non_ema duplicates or the single-file checkpoints.
    "sd15": HFAsset(
        repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
        revision="451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        allow=(
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "feature_extractor/*",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    ),
    # CLIP ViT-L/14: secondary CLIP Score (torchmetrics CLIPScore).
    "clip_vitl14": HFAsset(
        repo="openai/clip-vit-large-patch14",
        revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
        allow=("*.json", "*.txt", "model.safetensors"),
    ),
    # OpenCLIP ViT-g/14 laion2b_s34b_b88k: primary CLIP Score. Its tokenizer ships with
    # open_clip.
    "openclip_vitg14": HFAsset(
        repo="laion/CLIP-ViT-g-14-laion2B-s34B-b88K",
        revision="a20bc450485fc2dd335317234582df0e69d21b5a",
        allow=("open_clip_config.json", "open_clip_model.safetensors"),
    ),
}

TORCH_HUB: dict[str, FileAsset] = {
    # Inception-v3 (TF 2015-12-05 port) used by torch-fidelity for FID and KID.
    "inception_fid": FileAsset(
        url="https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
        "weights-inception-2015-12-05-6726825d.pth",
        sha256="6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2",
    ),
    # AlexNet backbone for LPIPS (torchvision AlexNet_Weights.IMAGENET1K_V1). LPIPS'
    # linear layers ship inside torchmetrics.
    "alexnet_lpips": FileAsset(
        url="https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        sha256="7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
    ),
}


def hf_snapshot_dir(name: str) -> Path:
    """Local snapshot directory of a pinned HF asset. Never touches the network."""
    from huggingface_hub import snapshot_download

    asset = HF[name]
    return Path(snapshot_download(
        asset.repo, revision=asset.revision, allow_patterns=list(asset.allow), local_files_only=True,
    ))


def torch_hub_file(name: str) -> Path:
    """Where torch.hub caches (and looks for) a metric weights file."""
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints" / TORCH_HUB[name].filename


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
