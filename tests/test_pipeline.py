from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, PNDMScheduler

from eldm import pipeline
from eldm.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]


class FakePipe:
    """Stands in for StableDiffusionPipeline so no weights are needed."""

    def __init__(self):
        self.scheduler = PNDMScheduler()

    def to(self, device):
        return self

    def set_progress_bar_config(self, **kwargs):
        pass


def test_build_pipeline_loads_pinned_revision(monkeypatch):
    calls = {}

    def fake_from_pretrained(model_id, **kwargs):
        calls.update(kwargs, model_id=model_id)
        return FakePipe()

    monkeypatch.setattr(pipeline.StableDiffusionPipeline, "from_pretrained", fake_from_pretrained)
    cfg = {**load_yaml(ROOT / "configs/base.yaml"), "device": "cpu", "scheduler": "dpmpp_2m"}
    pipe, helper, device = pipeline.build_pipeline(cfg)

    assert calls["model_id"] == cfg["model_id"]
    assert calls["revision"] == cfg["model_revision"]
    assert calls["dtype"] is torch.float32 and "torch_dtype" not in calls
    assert isinstance(pipe.scheduler, DPMSolverMultistepScheduler)
    assert helper is None and device == "cpu"


def test_build_pipeline_sets_tf32_flags(monkeypatch):
    monkeypatch.setattr(pipeline.StableDiffusionPipeline, "from_pretrained", lambda *a, **k: FakePipe())
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32)
    base = {**load_yaml(ROOT / "configs/base.yaml"), "device": "cpu"}
    assert base["tf32"] == {"matmul": False, "cudnn": False}  # the FP32 baseline is true FP32
    for flag in (True, False):
        pipeline.build_pipeline({**base, "tf32": {"matmul": flag, "cudnn": not flag}})
        assert torch.backends.cuda.matmul.allow_tf32 is flag
        assert torch.backends.cudnn.allow_tf32 is (not flag)
