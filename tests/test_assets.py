import hashlib
import re
from fnmatch import fnmatch
from pathlib import Path

import pytest

from eldm import assets
from eldm.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")

# Files in the pinned repos that duplicate the weights we use; none may be downloaded.
UNWANTED = [
    "v1-5-pruned.ckpt",
    "v1-5-pruned-emaonly.safetensors",
    "safety_checker/model.safetensors",
    "text_encoder/pytorch_model.bin",
    "text_encoder/model.fp16.safetensors",
    "unet/diffusion_pytorch_model.bin",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "unet/diffusion_pytorch_model.non_ema.safetensors",
    "vae/diffusion_pytorch_model.bin",
    "pytorch_model.bin",
    "tf_model.h5",
    "flax_model.msgpack",
    "open_clip_pytorch_model.bin",
]

# Files the loaders need from each repo.
REQUIRED = {
    "sd15": [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/vocab.json",
        "text_encoder/model.safetensors",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ],
    "clip_vitl14": ["config.json", "model.safetensors", "preprocessor_config.json", "merges.txt"],
    "openclip_vitg14": ["open_clip_model.safetensors"],
}


@pytest.mark.parametrize("name", sorted(assets.HF))
def test_hf_assets_pinned_to_commits(name):
    assert HEX40.fullmatch(assets.HF[name].revision)


@pytest.mark.parametrize("name", sorted(assets.HF))
def test_hf_allow_lists_fetch_only_needed_files(name):
    allow = assets.HF[name].allow
    assert not [f for f in UNWANTED if any(fnmatch(f, p) for p in allow)]
    assert all(any(fnmatch(f, p) for p in allow) for f in REQUIRED[name])


@pytest.mark.parametrize("name", sorted(assets.HF))
def test_hf_files_pinned_by_sha256(name):
    asset = assets.HF[name]
    assert set(REQUIRED[name]) <= set(asset.sha256)
    for rel, digest in asset.sha256.items():
        assert any(fnmatch(rel, p) for p in asset.allow), rel
        assert HEX64.fullmatch(digest), rel


@pytest.mark.parametrize("name", sorted(assets.TORCH_HUB))
def test_torch_hub_assets_pinned(name):
    asset = assets.TORCH_HUB[name]
    assert asset.url.startswith("https://") and asset.filename.endswith(".pth")
    assert re.fullmatch(r"[0-9a-f]{8}", asset.hash_prefix)
    assert HEX64.fullmatch(asset.sha256) and asset.sha256.startswith(asset.hash_prefix)


def test_base_config_uses_pinned_sd15():
    cfg = load_yaml(ROOT / "configs/base.yaml")
    sd15 = assets.HF["sd15"]
    assert (cfg["model_id"], cfg["model_revision"]) == (sd15.repo, sd15.revision)


def test_sha256_file(tmp_path):
    data = b"x" * 3_000_000  # spans several read chunks
    path = tmp_path / "blob"
    path.write_bytes(data)
    assert assets.sha256_file(path) == hashlib.sha256(data).hexdigest()
