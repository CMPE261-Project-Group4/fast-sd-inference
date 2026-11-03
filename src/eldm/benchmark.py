"""Timed generation loop: latency, peak GPU memory, throughput, UNet calls (NFE), stage times."""

from __future__ import annotations

import gc
import json
import os
import platform
import shutil
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
from .probes import PipelineProbe, StageTimer
from .prompts import Prompt, load_prompts

ROOT = Path(__file__).resolve().parents[2]
STAGE_KEYS = ("t_text_s", "t_unet_s", "t_vae_s")
# Written into a run directory by evaluate_quality.py, from the images there.
QUALITY_FILES = ("quality.json", "quality_per_image.csv")


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _generate(pipe, cfg: dict[str, Any], batch: list[Prompt]):
    # One generator per prompt so every image depends only on its own seed,
    # regardless of batch size. On the CPU, as diffusers recommends for reproducibility:
    # a seed gives the same noise on any GPU model or on the CPU.
    generators = [torch.Generator("cpu").manual_seed(p.seed) for p in batch]
    # The starting noise is drawn in float32 and passed as `latents`, the way diffusers'
    # maintainers advise for noise of your choosing (huggingface/diffusers#11056). Left to the
    # pipeline, a CPU generator draws unrelated noise in float16, so fp16 runs started from other
    # noise than the baseline (job 100435). In float32 it's the noise the pipeline itself draws.
    shape = (1, pipe.unet.config.in_channels, cfg["height"] // pipe.vae_scale_factor, cfg["width"] // pipe.vae_scale_factor)
    latents = torch.cat([torch.randn(shape, generator=g, dtype=torch.float32) for g in generators])
    return pipe(
        prompt=[p.prompt for p in batch],
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        height=cfg["height"],
        width=cfg["width"],
        generator=generators,
        latents=latents.to(pipe.device, pipe.unet.dtype),
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


def _publish(work_dir: Path, out_dir: Path) -> None:
    """Copy a staged run to its output directory with perf.json last, so a run directory with a
    perf.json is complete (run_experiments.py --skip-existing relies on it); then remove the
    staged copy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # A run being replaced: its perf.json would claim the new run is complete before it is,
    # and its quality files score the old images.
    for name in ("perf.json", *QUALITY_FILES):
        (out_dir / name).unlink(missing_ok=True)
    shutil.copytree(work_dir / "images", out_dir / "images", dirs_exist_ok=True)
    for name in ("config.yaml", "timings.jsonl"):
        shutil.copy2(work_dir / name, out_dir / name)
    shutil.copy2(work_dir / "perf.json", out_dir / "perf.json.tmp")
    os.replace(out_dir / "perf.json.tmp", out_dir / "perf.json")  # atomic within a directory
    shutil.rmtree(work_dir)


def run(cfg: dict[str, Any], stage_dir: str | Path | None = None) -> dict[str, Any]:
    """Generate and time one run into <output_root>/<name>/. With ``stage_dir`` (job scripts
    pass the node-local SSD), the run is written to <stage_dir>/<name>/ first and copied to
    output_root at the end, so PNG saves don't go to NFS. Saves aren't timed either way."""
    out_dir = Path(cfg["output_root"]) / cfg["name"]
    work_dir = Path(stage_dir) / cfg["name"] if stage_dir else out_dir
    if stage_dir:
        shutil.rmtree(work_dir, ignore_errors=True)  # left by a killed attempt of this job
    img_dir = work_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    with open(work_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    prompts = load_prompts(cfg["prompts_file"], cfg.get("num_prompts"))
    bs = cfg["batch_size"]
    batches = [prompts[i : i + bs] for i in range(0, len(prompts), bs)]

    pipe, helper, device = build_pipeline(cfg)
    cuda = device == "cuda"
    env = _env_info(device, pipe)
    probe, timer = PipelineProbe(pipe), StageTimer(pipe, device)

    for i in range(cfg["warmup_batches"]):
        _generate(pipe, cfg, batches[i % len(batches)])
    _sync(device)

    if cuda:
        torch.cuda.reset_peak_memory_stats()
        model_mem = torch.cuda.memory_allocated()

    records, n_images = [], 0
    with open(work_dir / "timings.jsonl", "w") as log:
        for batch in tqdm(batches, desc=cfg["name"]):
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            probe.reset()
            timer.reset()
            _sync(device)
            t0 = time.perf_counter()
            images = _generate(pipe, cfg, batch)
            _sync(device)
            dt = time.perf_counter() - t0

            peak = torch.cuda.max_memory_allocated() if cuda else None
            n_images += len(images)
            record = {"ids": [p.id for p in batch], "latency_s": dt, "peak_mem_bytes": peak,
                      "nfe": probe.nfe, "nfe_full": probe.nfe_full, **timer.seconds()}
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
    }
    # Per-image stage times, like latency_per_image_s, and the share of the end-to-end time
    # they account for; the rest is tokenizing, setting up latents and converting to PIL.
    if all(r[key] is not None for r in records for key in STAGE_KEYS):
        stage_totals = {key: sum(r[key] for r in records) for key in STAGE_KEYS}
        perf.update({key: t / n_images for key, t in stage_totals.items()})
        perf["stage_coverage"] = sum(stage_totals.values()) / total
    if cuda:
        perf["model_mem_gb"] = model_mem / 1024**3
        perf["peak_mem_gb"] = max(r["peak_mem_bytes"] for r in records) / 1024**3
    perf["env"] = env

    with open(work_dir / "perf.json", "w") as f:
        json.dump(perf, f, indent=2)

    probe.remove()
    timer.remove()
    # DeepCache's wrapped forwards hold its helper, and the helper holds the pipeline: a cycle
    # that `del` doesn't free, so the pipeline and its cached features stayed on the GPU into the
    # next run's memory figures (job 99828). Unwrap, then collect whatever cycles remain.
    if helper is not None:
        helper.disable()
    del pipe, helper
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    if stage_dir:
        _publish(work_dir, out_dir)
    return perf
