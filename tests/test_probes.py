import gc
import importlib.util
import json
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor

from eldm import benchmark, probes
from eldm.config import deep_merge
from eldm.pipeline import build_pipeline
from eldm.probes import PipelineProbe, StageTimer
from eldm.prompts import Prompt

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_run", ROOT / "scripts/check_run.py")
check_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_run)


def generate(pipe, steps: int = 5) -> None:
    pipe("a red cube", num_inference_steps=steps, height=32, width=32)


@pytest.mark.parametrize("scheduler, steps, nfe", [
    ("pndm", 5, 6),     # PNDM runs its second timestep twice
    ("pndm", 50, 51),   # the baseline
    ("ddim", 5, 5),
    ("dpmpp_2m", 5, 5),
])
def test_nfe_counts_unet_calls(tiny_cfg, scheduler, steps, nfe):
    pipe, _, _ = build_pipeline(deep_merge(tiny_cfg, {"scheduler": scheduler}))
    probe = PipelineProbe(pipe)
    for _ in range(2):  # reset() starts each pipeline call from zero
        probe.reset()
        generate(pipe, steps)
        assert (probe.nfe, probe.nfe_full) == (nfe, nfe)


@pytest.mark.parametrize("interval, full", [(2, 25), (3, 17), (5, 11)])
def test_deepcache_counts_full_passes(tiny_cfg, interval, full):
    pipe, helper, _ = build_pipeline(deep_merge(tiny_cfg, {"deepcache": {"enabled": True, "cache_interval": interval}}))
    assert helper is not None
    probe = PipelineProbe(pipe)
    generate(pipe, 50)
    # DeepCache runs the whole UNet when the step's index in the schedule is a multiple of the
    # interval. It finds the index with timesteps.index(t), so PNDM's repeated timestep gets the
    # first one's index: 25 full passes at interval 2, not 26.
    timesteps = pipe.scheduler.timesteps.tolist()
    assert sum(timesteps.index(t) % interval == 0 for t in timesteps) == full
    assert (probe.nfe, probe.nfe_full) == (51, full)


def test_remove_detaches_the_hooks(tiny_cfg):
    pipe, _, _ = build_pipeline(tiny_cfg)
    probe = PipelineProbe(pipe)
    probe.remove()
    generate(pipe)
    assert (probe.nfe, probe.nfe_full) == (0, 0)
    assert not pipe.unet._forward_hooks and not pipe.unet.mid_block.resnets[0]._forward_hooks


class FakeEvent:
    """torch.cuda.Event on CPU: records perf_counter; elapsed_time in milliseconds, like CUDA's."""

    def __init__(self, enable_timing: bool = False) -> None:
        assert enable_timing
        self.t = None

    def record(self) -> None:
        self.t = time.perf_counter()

    def elapsed_time(self, end: "FakeEvent") -> float:
        return (end.t - self.t) * 1000


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stage_timer_times_each_stage(tiny_cfg, monkeypatch, device):
    if device == "cuda":  # the event path, without a GPU; it must never synchronize
        monkeypatch.setattr(probes.torch.cuda, "Event", FakeEvent)
        monkeypatch.setattr(probes.torch.cuda, "synchronize", lambda *a: pytest.fail("synchronized"))
    pipe, _, _ = build_pipeline(tiny_cfg)
    timer = StageTimer(pipe, device)
    for _ in range(2):  # reset() starts each pipeline call from zero
        timer.reset()
        t0 = time.perf_counter()
        generate(pipe)
        wall = time.perf_counter() - t0
        assert len(timer._text) == 2  # the prompt and the empty negative prompt
        stages = timer.seconds()
        assert set(stages) == {"t_text_s", "t_unet_s", "t_vae_s"} and all(t > 0 for t in stages.values())
        assert sum(stages.values()) < wall
    timer.remove()
    timer.reset()
    generate(pipe)
    assert timer.seconds() == {"t_text_s": None, "t_unet_s": None, "t_vae_s": None}


def test_run_records_nfe_and_environment(tiny_cfg, monkeypatch):
    monkeypatch.setenv("SLURMD_NODENAME", "cs001")
    perf = benchmark.run(deep_merge(tiny_cfg, {"deepcache": {"enabled": True}}))  # interval 3, PNDM-5
    run_dir = Path(tiny_cfg["output_root"]) / "tiny"
    rows = [json.loads(line) for line in (run_dir / "timings.jsonl").read_text().splitlines()]
    assert [(r["ids"], r["nfe"], r["nfe_full"]) for r in rows] == [([0], 6, 2), ([1], 6, 2)]
    assert (perf["nfe"], perf["nfe_full"]) == (6, 2)
    for key in ("t_text_s", "t_unet_s", "t_vae_s"):
        assert all(r[key] > 0 for r in rows)
        assert perf[key] == pytest.approx(sum(r[key] for r in rows) / 2)
    stages = sum(r[k] for r in rows for k in ("t_text_s", "t_unet_s", "t_vae_s"))
    assert perf["stage_coverage"] == pytest.approx(stages / sum(r["latency_s"] for r in rows))
    env = perf["env"]
    assert env["node"] == "cs001" and env["device"] == "cpu"
    assert (env["tf32_matmul"], env["tf32_cudnn"]) == (False, False)
    assert env["attention"]["unet"] and env["attention"]["vae"]
    assert len(env["git_commit"]) == 40 and isinstance(env["git_dirty"], bool)
    # The coverage bound is for real runs: in the tiny pipeline, tokenizing and converting to
    # PIL take about 9% of the time (job P5.1 checks it on the A100).
    problems = check_run.check_perf(run_dir, tiny_cfg, 2, None) + check_run.check_timings(run_dir, tiny_cfg, 2)
    assert [p for p in problems if "coverage" not in p] == []


@pytest.mark.parametrize("deepcache", [False, True])
def test_run_releases_its_pipeline(tiny_cfg, monkeypatch, deepcache):
    """DeepCache's wrapped forwards hold its helper, and the helper holds the pipeline: a cycle
    that outlived ``del pipe`` and showed in the next run's memory (job 99828: fp16 right after
    deepcache_i3 peaked at 6.84 GB, 2.61 on its own)."""
    built = []

    def build(cfg):
        pipe, helper, device = build_pipeline(cfg)
        built.append(weakref.ref(pipe))
        return pipe, helper, device

    monkeypatch.setattr(benchmark, "build_pipeline", build)
    gc.disable()  # only run() may collect, so a cycle left behind can't be cleared by chance
    try:
        benchmark.run(deep_merge(tiny_cfg, {"deepcache": {"enabled": deepcache}}))
        assert built[0]() is None
    finally:
        gc.enable()


class FakePipe:
    """Records the pipeline call; has what _generate reads to shape the starting noise."""

    vae_scale_factor = 8
    device = torch.device("cpu")

    def __init__(self, dtype) -> None:
        self.unet = SimpleNamespace(config=SimpleNamespace(in_channels=4), dtype=dtype)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_starting_noise_is_per_prompt_and_drawn_in_float32_on_the_cpu(tiny_cfg, dtype):
    """Each prompt's noise comes from its own CPU generator (diffusers recommends the CPU for
    reproducibility) and is drawn in float32 whatever the pipeline's dtype: drawn in float16 it's
    unrelated noise, and job 100435's fp16 run scored LPIPS 0.76 against the baseline. In float32
    it's exactly the noise diffusers itself draws, so the fp32 runs didn't change."""
    pipe = FakePipe(dtype)
    batch = [Prompt(0, "a red cube", "manual", "objects", 1000), Prompt(1, "a blue sphere", "manual", "objects", 1001)]
    benchmark._generate(pipe, tiny_cfg, batch)
    [call] = pipe.calls
    assert [(g.device.type, g.initial_seed()) for g in call["generator"]] == [("cpu", 1000), ("cpu", 1001)]
    diffusers_fp32 = randn_tensor((2, 4, 4, 4), generator=[torch.Generator("cpu").manual_seed(s) for s in (1000, 1001)],
                                  device=torch.device("cpu"), dtype=torch.float32)  # tiny_cfg is 32x32
    assert call["latents"].dtype == dtype
    assert torch.equal(call["latents"], diffusers_fp32.to(dtype))
