"""Build a Stable Diffusion pipeline with the optimizations a run config asks for."""

from __future__ import annotations

from typing import Any

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
)

SCHEDULERS = {
    "pndm": PNDMScheduler,
    "ddim": DDIMScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
}

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_pipeline(cfg: dict[str, Any]):
    """Return ``(pipe, deepcache_helper_or_None, device)``."""
    device = resolve_device(cfg["device"])
    dtype = DTYPES[cfg["dtype"]]
    if dtype == torch.float16 and device == "cpu":
        raise ValueError("fp16 is not supported on CPU")

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model_id"],
        torch_dtype=dtype,
        safety_checker=None,          # avoid black images skewing quality metrics
        requires_safety_checker=False,
    )
    pipe.scheduler = SCHEDULERS[cfg["scheduler"]].from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    helper = None
    dc = cfg.get("deepcache", {})
    if dc.get("enabled"):
        from DeepCache import DeepCacheSDHelper

        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(cache_interval=dc["cache_interval"], cache_branch_id=dc["cache_branch_id"])
        helper.enable()

    return pipe, helper, device
