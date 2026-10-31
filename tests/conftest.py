from pathlib import Path

import pytest
import torch

from eldm import pipeline
from eldm.config import deep_merge, load_yaml
from eldm.prompts import Prompt, save_prompts

ROOT = Path(__file__).resolve().parents[1]


def tiny_sd_pipeline():
    """Stable Diffusion with SD1.5's structure and scheduler but tiny random weights (sizes from
    diffusers' own fast tests). It makes 32x32 images on CPU in well under a second, offline."""
    from diffusers import AutoencoderKL, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(4, 8), layers_per_block=1, sample_size=16, in_channels=4, out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32, norm_num_groups=2,
    )
    vae = AutoencoderKL(
        block_out_channels=[4, 8], in_channels=3, out_channels=3, latent_channels=4, norm_num_groups=2,
        down_block_types=["DownEncoderBlock2D"] * 2, up_block_types=["UpDecoderBlock2D"] * 2,
    )
    # A character-level vocabulary: no merges, unknown text maps to <|endoftext|>.
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for c in "abcdefghijklmnopqrstuvwxyz":
        vocab[c], vocab[c + "</w>"] = len(vocab), len(vocab) + 1
    tokenizer = CLIPTokenizer(vocab=vocab, merges=[], model_max_length=77)
    text_encoder = CLIPTextModel(CLIPTextConfig(
        vocab_size=len(vocab), bos_token_id=0, eos_token_id=1, pad_token_id=1, hidden_size=32,
        intermediate_size=37, num_attention_heads=4, num_hidden_layers=2, max_position_embeddings=77,
    ))
    # SD1.5's scheduler_config.json.
    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              skip_prk_steps=True, set_alpha_to_one=False, steps_offset=1)
    return StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                                   scheduler=scheduler, safety_checker=None, feature_extractor=None,
                                   requires_safety_checker=False)


@pytest.fixture
def tiny_cfg(tmp_path, monkeypatch):
    """A run config for configs/base.yaml's pipeline, built as the tiny one on CPU: 2 prompts,
    5 steps, 32x32, 1 warm-up batch. eldm.pipeline.build_pipeline returns a fresh tiny pipeline."""
    monkeypatch.setattr(pipeline.StableDiffusionPipeline, "from_pretrained", lambda *a, **k: tiny_sd_pipeline())
    # build_pipeline sets these process-global flags; put them back afterwards.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32)
    prompts = [Prompt(i, text, "manual", "objects", 1000 + i) for i, text in enumerate(["a red cube", "a blue sphere"])]
    save_prompts(prompts, tmp_path / "prompts.jsonl")
    return deep_merge(load_yaml(ROOT / "configs/base.yaml"), {
        "name": "tiny", "device": "cpu", "prompts_file": str(tmp_path / "prompts.jsonl"),
        "output_root": str(tmp_path / "results"), "num_inference_steps": 5, "height": 32, "width": 32,
        "warmup_batches": 1,
    })
