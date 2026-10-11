"""Timed generation loop: latency, peak GPU memory, throughput."""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from .pipeline import build_pipeline
from .prompts import Prompt, load_prompts


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


def _env_info(device: str) -> dict[str, Any]:
    info = {"device": device, "torch": torch.__version__, "python": platform.python_version()}
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    import diffusers

    info["diffusers"] = diffusers.__version__
    return info


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

    for i in range(cfg["warmup_batches"]):
        _generate(pipe, cfg, batches[i % len(batches)], device)
    _sync(device)

    if cuda:
        torch.cuda.reset_peak_memory_stats()
        model_mem = torch.cuda.memory_allocated()

    latencies, n_images = [], 0
    with open(out_dir / "timings.jsonl", "w") as log:
        for batch in tqdm(batches, desc=cfg["name"]):
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            _sync(device)
            t0 = time.perf_counter()
            images = _generate(pipe, cfg, batch, device)
            _sync(device)
            dt = time.perf_counter() - t0

            peak = torch.cuda.max_memory_allocated() if cuda else None
            latencies.append(dt)
            n_images += len(images)
            log.write(json.dumps({"ids": [p.id for p in batch], "latency_s": dt, "peak_mem_bytes": peak}) + "\n")

            if cfg["save_images"]:
                for p, img in zip(batch, images):
                    img.save(img_dir / f"{p.id:04d}.png")

    total = sum(latencies)
    perf = {
        "name": cfg["name"],
        "num_images": n_images,
        "batch_size": bs,
        "latency_per_batch_mean_s": statistics.mean(latencies),
        "latency_per_batch_std_s": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "latency_per_image_s": total / n_images,
        "throughput_img_per_s": n_images / total,
        "env": _env_info(device),
    }
    if cuda:
        with open(out_dir / "timings.jsonl") as f:
            peaks = [json.loads(line)["peak_mem_bytes"] for line in f]
        perf["model_mem_gb"] = model_mem / 1024**3
        perf["peak_mem_gb"] = max(peaks) / 1024**3

    with open(out_dir / "perf.json", "w") as f:
        json.dump(perf, f, indent=2)

    del pipe
    if cuda:
        torch.cuda.empty_cache()
    return perf
