"""Timed generation loop: latency, peak GPU memory, throughput, UNet calls (NFE)."""

from __future__ import annotations

import json
import os
import platform
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from .pipeline import build_pipeline
from .probes import PipelineProbe
from .prompts import Prompt, load_prompts

ROOT = Path(__file__).resolve().parents[2]


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _generate(pipe, cfg: dict[str, Any], batch: list[Prompt], device: str):
    # One generator per prompt so every image depends only on its own seed,
    # regardless of batch size.
    gen_device = "cpu" if device == "mps" else device
    generators = [torch.Generator(gen_device).manual_seed(p.seed) for p in batch]
    return pipe(
        prompt=[p.prompt for p in batch],
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        height=cfg["height"],
        width=cfg["width"],
        generator=generators,
    ).images


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True,
                          timeout=60).stdout.strip()


def _git_state() -> tuple[str | None, bool | None]:
    """The clone's commit and whether it has uncommitted or untracked files; None outside git."""
    try:
        return _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        return None, None


def _env_info(device: str, pipe) -> dict[str, Any]:
    """Where and how the run ran: enough to tell runs on different nodes, GPUs or code apart."""
    import diffusers

    commit, dirty = _git_state()
    info = {
        "device": device,
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "python": platform.python_version(),
        "git_commit": commit,
        "git_dirty": dirty,
        "attention": {name: sorted({type(p).__name__ for p in model.attn_processors.values()})
                      for name, model in (("unet", pipe.unet), ("vae", pipe.vae))},
        # The flags in effect, read back after build_pipeline set them.
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_uuid"] = f"GPU-{props.uuid}"  # as nvidia-smi prints it
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    return info


def _per_call(counts: list[int]) -> int | float:
    """A per-pipeline-call count: the value every batch shares (they should), else the mean."""
    return counts[0] if len(set(counts)) == 1 else statistics.mean(counts)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(cfg["output_root"]) / cfg["name"]
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    prompts = load_prompts(cfg["prompts_file"], cfg.get("num_prompts"))
    bs = cfg["batch_size"]
    batches = [prompts[i : i + bs] for i in range(0, len(prompts), bs)]

    pipe, _helper, device = build_pipeline(cfg)
    cuda = device == "cuda"
    env = _env_info(device, pipe)
    probe = PipelineProbe(pipe)

    for i in range(cfg["warmup_batches"]):
        _generate(pipe, cfg, batches[i % len(batches)], device)
    _sync(device)

    if cuda:
        torch.cuda.reset_peak_memory_stats()
        model_mem = torch.cuda.memory_allocated()

    records, n_images = [], 0
    with open(out_dir / "timings.jsonl", "w") as log:
        for batch in tqdm(batches, desc=cfg["name"]):
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            probe.reset()
            _sync(device)
            t0 = time.perf_counter()
            images = _generate(pipe, cfg, batch, device)
            _sync(device)
            dt = time.perf_counter() - t0

            peak = torch.cuda.max_memory_allocated() if cuda else None
            n_images += len(images)
            record = {"ids": [p.id for p in batch], "latency_s": dt, "peak_mem_bytes": peak,
                      "nfe": probe.nfe, "nfe_full": probe.nfe_full}
            records.append(record)
            log.write(json.dumps(record) + "\n")

            if cfg["save_images"]:
                for p, img in zip(batch, images):
                    img.save(img_dir / f"{p.id:04d}.png")

    latencies = [r["latency_s"] for r in records]
    total = sum(latencies)
    perf = {
        "name": cfg["name"],
        "num_images": n_images,
        "batch_size": bs,
        "latency_per_batch_mean_s": statistics.mean(latencies),
        "latency_per_batch_std_s": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "latency_per_image_s": total / n_images,
        "throughput_img_per_s": n_images / total,
        # UNet calls per pipeline call (see eldm.probes).
        "nfe": _per_call([r["nfe"] for r in records]),
        "nfe_full": _per_call([r["nfe_full"] for r in records]),
        "env": env,
    }
    if cuda:
        perf["model_mem_gb"] = model_mem / 1024**3
        perf["peak_mem_gb"] = max(r["peak_mem_bytes"] for r in records) / 1024**3

    with open(out_dir / "perf.json", "w") as f:
        json.dump(perf, f, indent=2)

    probe.remove()
    del pipe
    if cuda:
        torch.cuda.empty_cache()
    return perf
