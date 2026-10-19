from pathlib import Path

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
    assert isinstance(pipe.scheduler, DPMSolverMultistepScheduler)
    assert helper is None and device == "cpu"
