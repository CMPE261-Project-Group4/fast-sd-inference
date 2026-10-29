"""Forward hooks that measure what happens inside one pipeline call: UNet calls (NFE) and the
time spent in each stage.

NFE (number of function evaluations) counts UNet calls: one per denoising step, each on the
whole batch, doubled for classifier-free guidance. PNDM takes 51 calls for 50 steps: its
second timestep is run twice. ``nfe`` counts every call and ``nfe_full`` the calls that ran
the whole UNet; they differ only under DeepCache.

DeepCache replaces the ``forward`` of the UNet, its down and up blocks and its mid block with
wrappers that return cached outputs on skipped steps. Hooks fire around those wrappers, so a
hook on the mid block would fire on skipped steps too. Full passes are counted one level
inside it, at its first resnet, which runs only when the step isn't cached.

The stages are the text encoder (run twice: the prompt and the empty negative prompt), the
UNet loop (first UNet call to last, scheduler steps included) and the VAE decode
(post_quant_conv, then the decoder). On CUDA the hooks record events on the stream and never
synchronize, so timing doesn't change what it measures; read the times after the caller's
synchronize. Elsewhere they read perf_counter, which is exact on CPU.
"""

from __future__ import annotations

import time

import torch


class PipelineProbe:
    """Counts UNet calls on a pipeline until removed. Call reset() before each pipeline call."""

    def __init__(self, pipe) -> None:
        self.nfe = 0
        self.nfe_full = 0
        self._handles = [
            pipe.unet.register_forward_hook(self._on_unet),
            pipe.unet.mid_block.resnets[0].register_forward_hook(self._on_full_pass),
        ]

    def _on_unet(self, module, args, output) -> None:
        self.nfe += 1

    def _on_full_pass(self, module, args, output) -> None:
        self.nfe_full += 1

    def reset(self) -> None:
        self.nfe = self.nfe_full = 0

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class StageTimer:
    """Times the text encoder, UNet loop and VAE decode of each pipeline call until removed.
    Call reset() before each pipeline call and seconds() after it, once synchronized."""

    def __init__(self, pipe, device: str) -> None:
        self._cuda = device == "cuda"
        self._handles = [
            pipe.text_encoder.register_forward_pre_hook(self._text_start),
            pipe.text_encoder.register_forward_hook(self._text_end),
            pipe.unet.register_forward_pre_hook(self._unet_start),
            pipe.unet.register_forward_hook(self._unet_end),
            pipe.vae.post_quant_conv.register_forward_pre_hook(self._vae_start),
            pipe.vae.decoder.register_forward_hook(self._vae_end),
        ]
        self.reset()

    def _now(self):
        if self._cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def _seconds(self, start, end) -> float | None:
        if start is None or end is None:
            return None
        return start.elapsed_time(end) / 1000 if self._cuda else end - start

    def _text_start(self, module, args) -> None:
        self._text.append([self._now(), None])

    def _text_end(self, module, args, output) -> None:
        self._text[-1][1] = self._now()

    def _unet_start(self, module, args) -> None:
        if self._unet[0] is None:
            self._unet[0] = self._now()

    def _unet_end(self, module, args, output) -> None:
        self._unet[1] = self._now()

    def _vae_start(self, module, args) -> None:
        self._vae = [self._now(), None]

    def _vae_end(self, module, args, output) -> None:
        self._vae[1] = self._now()

    def reset(self) -> None:
        self._text: list[list] = []
        self._unet: list = [None, None]
        self._vae: list = [None, None]

    def seconds(self) -> dict[str, float | None]:
        """Seconds per stage in the last pipeline call; None for a stage that didn't run."""
        text = [self._seconds(start, end) for start, end in self._text]
        return {
            "t_text_s": sum(text) if text and None not in text else None,
            "t_unet_s": self._seconds(*self._unet),
            "t_vae_s": self._seconds(*self._vae),
        }

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
