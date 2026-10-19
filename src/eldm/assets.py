"""Pinned external assets: model snapshots and metric weights.

One source of truth for scripts/hpc/prefetch_assets.py (downloads on coe-hpc1), the
offline check job and the loaders. COE-HPC3 has no internet, so every asset is pinned to
an immutable version, downloaded ahead of time, and loaded here from the local cache only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HFAsset:
    """A Hugging Face repo pinned to a commit, limited to the files we use."""

    repo: str
    revision: str            # full commit hash: immutable, unlike a branch or tag
    allow: tuple[str, ...]   # allow_patterns for snapshot_download
    # sha256 of every fetched file, checked against the Hub's metadata when prefetched.
    # The HF cache cannot be trusted for this: blobs of Xet-backed repos are named by
    # their Xet hash, not their sha256.
    sha256: Mapping[str, str] = field(default_factory=dict)


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
        sha256={
            "feature_extractor/preprocessor_config.json": "2a1da83b5e1032aaeef397552ddb408dca0d8cd1dc58f61bf6abf38d6f33a0a2",
            "model_index.json": "e2f6f22e274374010aec30c79eb2d6e53fff4a36ac0225523a92cb2d41a83347",
            "scheduler/scheduler_config.json": "699cce92eb7c122e2eb7dfdea78e6187fda76a5ed4a8e42319b85610e620e091",
            "text_encoder/config.json": "845df614cb9327ae7bbea027316246fae917827407da6df13572e41b5f93b4cc",
            "text_encoder/model.safetensors": "d008943c017f0092921106440254dbbe00b6a285f7883ec8ba160c3faad88334",
            "tokenizer/merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
            "tokenizer/special_tokens_map.json": "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac",
            "tokenizer/tokenizer_config.json": "00439066fcba73de57644cf41e4e3b9f2dbb09d7f3fc2005898ba52399045882",
            "tokenizer/vocab.json": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
            "unet/config.json": "78f474de6bab3d893868f37be97b636ae65c0df3073ed3256ca458ff599b5f96",
            "unet/diffusion_pytorch_model.safetensors": "19da7aaa4b880e59d56843f1fcb4dd9b599c28a1d9d9af7c1143057c8ffae9f1",
            "vae/config.json": "786a7d21647ddea6a04b9675c03d3cb45e90a2f3c6da5fbda2c54ade040036de",
            "vae/diffusion_pytorch_model.safetensors": "a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c",
        },
    ),
    # CLIP ViT-L/14: secondary CLIP Score (torchmetrics CLIPScore).
    "clip_vitl14": HFAsset(
        repo="openai/clip-vit-large-patch14",
        revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
        allow=("*.json", "*.txt", "model.safetensors"),
        sha256={
            "config.json": "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a",
            "merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
            "model.safetensors": "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb",
            "preprocessor_config.json": "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
            "special_tokens_map.json": "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf",
            "tokenizer.json": "a83e0809aa4c3af7208b2df632a7a69668c6d48775b3c3fe4e1b1199d1f8b8f4",
            "tokenizer_config.json": "deef455e52fa5e8151e339add0582e4235f066009601360999d3a9cda83b1129",
            "vocab.json": "3f0c4f7d2086b61b38487075278ea9ed04edb53a03cbb045b86c27190fa8fb69",
        },
    ),
    # OpenCLIP ViT-g/14 laion2b_s34b_b88k: primary CLIP Score. Its tokenizer ships with
    # open_clip.
    "openclip_vitg14": HFAsset(
        repo="laion/CLIP-ViT-g-14-laion2B-s34B-b88K",
        revision="a20bc450485fc2dd335317234582df0e69d21b5a",
        allow=("open_clip_config.json", "open_clip_model.safetensors"),
        sha256={
            "open_clip_config.json": "0fecef79cbe42f238ff061ed5050e75f5bb46b0a01e42cff431e2fe47fee4526",
            "open_clip_model.safetensors": "0b890e533656dbadbf30b0f4f62a053b51185622ecb7e2c7e2c6232cd95d438a",
        },
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


def verify_hf(name: str) -> tuple[int, int]:
    """Re-hash every pinned file of a cached HF asset offline. Returns (files, bytes)."""
    asset = HF[name]
    if not asset.sha256:
        raise RuntimeError(f"{name}: no sha256 pins")
    snapshot = hf_snapshot_dir(name)
    size = 0
    for rel, expected in sorted(asset.sha256.items()):
        path = snapshot / rel
        if not path.exists():
            raise RuntimeError(f"{name}: {rel} is missing from {snapshot}")
        if (actual := sha256_file(path)) != expected:
            raise RuntimeError(f"{name}: {rel} has sha256 {actual}, pinned {expected}")
        size += path.stat().st_size
    return len(asset.sha256), size


def verify_torch_hub(name: str) -> int:
    """Re-hash a cached metric weights file offline. Returns its size in bytes."""
    path, expected = torch_hub_file(name), TORCH_HUB[name].sha256
    if not path.exists():
        raise RuntimeError(f"{name}: {path} is missing")
    if (actual := sha256_file(path)) != expected:
        raise RuntimeError(f"{name}: {path} has sha256 {actual}, pinned {expected}")
    return path.stat().st_size


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
