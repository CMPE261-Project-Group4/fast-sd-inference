"""Forward hooks that measure what happens inside one pipeline call.

NFE (number of function evaluations) counts UNet calls: one per denoising step, each on the
whole batch, doubled for classifier-free guidance. PNDM takes 51 calls for 50 steps: its
second timestep is run twice. ``nfe`` counts every call and ``nfe_full`` the calls that ran
the whole UNet; they differ only under DeepCache.

DeepCache replaces the ``forward`` of the UNet, its down and up blocks and its mid block with
wrappers that return cached outputs on skipped steps. Hooks fire around those wrappers, so a
hook on the mid block would fire on skipped steps too. Full passes are counted one level
inside it, at its first resnet, which runs only when the step isn't cached.
"""

from __future__ import annotations


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
