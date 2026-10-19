"""Check that every asset a COE-HPC3 job needs loads offline: cached files only, no network.

    source env/site.sh
    python scripts/hpc/check_assets.py --quick   # coe-hpc1: files, hashes, counts, tokenizers
    python scripts/hpc/check_assets.py --full    # HPC3 job (check_assets.sbatch): also re-hashes
                                                 # the models and loads and runs each one on CPU

Socket connections are blocked first, so anything that tries the network fails at once
instead of reaching the internet (coe-hpc1) or hanging (HPC3). Models are loaded one at a
time and freed in between. Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gc
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def block_network() -> None:
    def refuse(*args, **kwargs):
        raise RuntimeError("network access attempted; the asset must come from the local cache")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]


block_network()

import torch  # noqa: E402

from eldm import assets  # noqa: E402
from eldm.config import load_yaml  # noqa: E402

DATASETS = Path(os.environ["ELDM_DATASETS"])
failures = 0


def check(label: str, fn) -> None:
    global failures
    t0 = time.perf_counter()
    try:
        info = fn()
    except Exception as e:  # report every check, not just the first failure
        failures += 1
        print(f"  FAIL  {label}: {type(e).__name__}: {e}", flush=True)
    else:
        print(f"  ok    {label}{': ' + info if info else ''} ({time.perf_counter() - t0:.1f}s)", flush=True)
    gc.collect()


# ---- quick: files, hashes, counts --------------------------------------------------------

def offline_flags() -> str:
    import huggingface_hub.constants as hub

    assert hub.HF_HUB_OFFLINE and os.environ.get("TRANSFORMERS_OFFLINE") == "1", "offline flags not set"
    return f"HF_HOME={hub.HF_HOME}, TORCH_HOME={os.environ['TORCH_HOME']}"


def hf_files(name: str) -> str:
    snapshot = assets.hf_snapshot_dir(name)
    missing = [rel for rel in assets.HF[name].sha256 if not (snapshot / rel).is_file()]
    assert not missing, f"missing {missing}"
    size = sum((snapshot / rel).stat().st_size for rel in assets.HF[name].sha256)
    return f"{len(assets.HF[name].sha256)} files, {size / 1e9:.2f} GB at {assets.HF[name].revision[:7]}"


def hf_hashes(name: str) -> str:
    n, size = assets.verify_hf(name)
    return f"{n} files, {size / 1e9:.2f} GB, sha256 pins ok"


def dataset_checksums() -> str:
    lines = [ln.split(maxsplit=1) for ln in (ROOT / "data/CHECKSUMS.txt").read_text().splitlines() if ln.strip()]
    for digest, rel in lines:
        actual = assets.sha256_file(DATASETS / rel)
        assert actual == digest, f"{rel}: sha256 {actual}, recorded {digest}"
    return f"{len(lines)} files match data/CHECKSUMS.txt"


def dataset_counts() -> str:
    images = list((DATASETS / "coco/val2017").glob("*.jpg"))
    captions = json.loads((DATASETS / "coco/annotations/captions_val2017.json").read_text())
    with open(DATASETS / "partiprompts/PartiPrompts.tsv", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    challenges = collections.Counter(r["Challenge"] for r in rows)
    assert len(images) == 5000 and len(captions["images"]) == 5000, (len(images), len(captions["images"]))
    assert len(rows) == 1632 and challenges["Quantity"] == 90 and challenges["Writing & Symbols"] == 91
    writable = [p for d in ("coco", "partiprompts") for p in [DATASETS / d, *(DATASETS / d).rglob("*")]
                if p.stat().st_mode & 0o222]
    assert not writable, f"{len(writable)} dataset paths are writable, e.g. {writable[0]}"
    return (f"COCO {len(images)} images / {len(captions['annotations'])} captions,"
            f" PartiPrompts {len(rows)}; read-only")


def tokenizers() -> str:
    import open_clip
    from transformers import CLIPTokenizer

    sd_tok = CLIPTokenizer.from_pretrained(assets.hf_snapshot_dir("sd15") / "tokenizer")
    clip_tok = CLIPTokenizer.from_pretrained(assets.hf_snapshot_dir("clip_vitl14"))
    ids = sd_tok("a photo of a cat", padding="max_length", max_length=77).input_ids
    assert len(ids) == 77 and ids == clip_tok("a photo of a cat", padding="max_length", max_length=77).input_ids
    assert open_clip.get_tokenizer("ViT-g-14")(["a photo of a cat"]).shape == (1, 77)
    return "SD1.5 and CLIP-L CLIPTokenizer agree; open_clip ViT-g-14 tokenizer"


# ---- full: load and run every model on CPU -----------------------------------------------

def coco_sample():
    """A COCO val image with its own caption and a caption from a different image."""
    from PIL import Image

    captions = json.loads((DATASETS / "coco/annotations/captions_val2017.json").read_text())
    images = sorted(captions["images"], key=lambda im: im["id"])
    first = {a["image_id"]: a["caption"] for a in sorted(captions["annotations"], key=lambda a: a["id"])}
    image = Image.open(DATASETS / "coco/val2017" / images[0]["file_name"]).convert("RGB")
    return image, first[images[0]["id"]], first[images[-1]["id"]]


def sd15() -> str:
    from DeepCache import DeepCacheSDHelper

    from eldm.pipeline import SCHEDULERS, build_pipeline

    cfg = {**load_yaml(ROOT / "configs/base.yaml"), "device": "cpu", "dtype": "fp32"}
    pipe, _, _ = build_pipeline(cfg)
    for name, cls in SCHEDULERS.items():
        assert isinstance(cls.from_config(pipe.scheduler.config), cls), name
    with torch.no_grad():
        ids = pipe.tokenizer("a photo of a cat", padding="max_length", max_length=77,
                             return_tensors="pt").input_ids
        emb = pipe.text_encoder(ids)[0]
        latent = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(0))
        noise = pipe.unet(latent, torch.tensor(999), encoder_hidden_states=emb).sample
        image = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
    assert emb.shape == (1, 77, 768) and noise.shape == latent.shape and image.shape == (1, 3, 64, 64)
    assert all(torch.isfinite(t).all() for t in (emb, noise, image))
    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(cache_interval=3, cache_branch_id=0)
    helper.enable()
    helper.disable()
    attn = type(next(iter(pipe.unet.attn_processors.values()))).__name__
    del pipe
    return (f"pipeline @{cfg['model_revision'][:7]}, {len(SCHEDULERS)} schedulers, text encoder + UNet"
            f" + VAE forward ok, DeepCache wraps; attention {attn}")


def clip_vitl14() -> str:
    import numpy as np
    from torchmetrics.multimodal.clip_score import CLIPScore
    from transformers import CLIPModel, CLIPProcessor

    class CLIPTensorFeatures(CLIPModel):
        """transformers 5 returns BaseModelOutputWithPooling from get_*_features, while
        torchmetrics 1.9 CLIPScore expects the projected tensor (job 92223 failed on it).
        Return .pooler_output, the tensor transformers 4 returned; the score is unchanged."""

        def get_image_features(self, *args, **kwargs):
            return super().get_image_features(*args, **kwargs).pooler_output

        def get_text_features(self, *args, **kwargs):
            return super().get_text_features(*args, **kwargs).pooler_output

    # A callable loader: CLIPScore's model-name path would ask the hub for the latest
    # revision, which is not cached.
    snapshot = assets.hf_snapshot_dir("clip_vitl14")
    model, processor = CLIPTensorFeatures.from_pretrained(snapshot), CLIPProcessor.from_pretrained(snapshot)
    metric = CLIPScore(model_name_or_path=lambda: (model, processor))
    image, caption, other = coco_sample()
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    own = float(metric(pixels, caption))
    metric.reset()
    wrong = float(metric(pixels, other))

    # Cross-check against the CLIP Score definition, 100 * max(cos(image, text), 0).
    with torch.no_grad():
        img = model.get_image_features(**processor(images=[image], return_tensors="pt"))
        txt = model.get_text_features(**processor(text=[caption], return_tensors="pt", padding=True))
    manual = float(100 * torch.nn.functional.cosine_similarity(img, txt).clamp(min=0))
    del metric, model
    assert abs(own - manual) < 1e-3, (own, manual)
    assert own > wrong, (own, wrong)
    return f"CLIPScore {own:.2f} with its own caption vs {wrong:.2f} with another (by hand {manual:.2f})"


def openclip_vitg14() -> str:
    import open_clip

    weights = assets.hf_snapshot_dir("openclip_vitg14") / "open_clip_model.safetensors"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-g-14", pretrained=str(weights))
    tokenizer = open_clip.get_tokenizer("ViT-g-14")
    image, caption, other = coco_sample()
    with torch.no_grad():
        img = model.encode_image(preprocess(image).unsqueeze(0))
        txt = model.encode_text(tokenizer([caption, other]))
        sim = torch.nn.functional.cosine_similarity(img, txt)
    del model
    assert sim[0] > sim[1], sim.tolist()
    return f"cosine {100 * sim[0]:.1f} with its own caption vs {100 * sim[1]:.1f} with another"


def metrics() -> str:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    for name in assets.TORCH_HUB:
        assets.verify_torch_hub(name)
    g = torch.Generator().manual_seed(0)
    real = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8, generator=g)
    fake = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8, generator=g)
    fid, kid = FrechetInceptionDistance(feature=2048), KernelInceptionDistance(subset_size=2)
    for m in (fid, kid):
        m.update(real, real=True)
        m.update(fake, real=False)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
    return (f"FID {float(fid.compute()):.3g}, KID {float(kid.compute()[0]):.3g},"
            f" LPIPS {float(lpips(real.float() / 255, fake.float() / 255)):.3g}; weights sha256 ok")


def coco_images() -> str:
    from PIL import Image

    paths = sorted((DATASETS / "coco/val2017").glob("*.jpg"))[::1000]
    for p in paths:
        with Image.open(p) as im:
            im.load()
    return f"decoded {len(paths)} JPEGs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = ap.parse_args()

    print(f"Offline asset check ({'full' if args.full else 'quick'}) on {socket.gethostname()},"
          f" torch {torch.__version__}, {torch.get_num_threads()} threads, network blocked:", flush=True)
    check("offline flags", offline_flags)
    for name in assets.HF:
        check(f"hf:{name} files", lambda n=name: hf_files(n))
    check("dataset sha256", dataset_checksums)
    check("dataset counts", dataset_counts)
    check("tokenizers", tokenizers)
    if args.full:
        for name in assets.HF:
            check(f"hf:{name} sha256", lambda n=name: hf_hashes(n))
        check("COCO images", coco_images)
        check("SD1.5", sd15)
        check("CLIP ViT-L/14", clip_vitl14)
        check("OpenCLIP ViT-g/14", openclip_vitg14)
        check("FID, KID, LPIPS", metrics)

    print(f"{failures} check(s) failed." if failures else "All checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
