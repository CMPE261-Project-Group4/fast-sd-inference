# Related work: training-free speed-ups for latent diffusion

Notes for Section 2 of the report. Each entry says what the method does, why it saves time,
and how our benchmark uses or tests it. Full references are at the end.

## The model we speed up

**Latent Diffusion / Stable Diffusion** [1]. Diffusion runs in the 4×64×64 latent space of a VAE
instead of on 512×512 pixels. One image costs: one CLIP text-encoder pass, N UNet passes (each on a
batch of 2 because of classifier-free guidance), and one VAE decode. In our baseline (PNDM, 50 steps,
FP32) the UNet is 98 % of the time, so every method below either calls the UNet fewer times or makes
each call cheaper.

## Fewer UNet calls: better samplers

| Method | Idea | What we test |
|---|---|---|
| PNDM [2] | Pseudo-numerical (linear multistep) solver; the SD v1.5 default | Baseline at 50 steps; 10, 20, 30 steps |
| DDIM [3] | Non-Markovian, deterministic sampling with the same trained model; allows skipping steps | 20 and 50 steps |
| DPM-Solver++ [4] | High-order ODE solver designed for guided sampling; accurate at 15–25 steps | 2M variant at 20 and 50 steps; in the combined runs |
| Euler (Karras et al.) [5] | First-order solver of the probability-flow ODE with tuned noise schedule | 20 and 50 steps |

What our data adds: at 20 steps the choice of sampler matters more than the step count. PNDM
loses 0.31 CLIP points at 20 steps, while DDIM, DPM-Solver++ and Euler keep or raise it.

## Cheaper UNet calls

- **Half precision (FP16).** Same calls, but the matrix work runs on tensor cores and memory halves.
  It is a pure systems change, so its speedup depends strongly on the GPU: 4.03× on our A100,
  2.22× on the RTX 4090 (TF32 off in both).
- **DeepCache** [6]. The UNet's deep features change slowly between adjacent steps. DeepCache runs
  the full UNet every N steps and, in between, reuses the cached deep features and runs only the
  shallow skip branch. Our baseline reproduces the CLIP Scores the paper reports for 50-step PLMS
  on COCO and PartiPrompts (30.33 vs 30.30; 29.73 vs 29.51, ViT-g/14).
  What our data adds: DeepCache interval 3 is free alone but interacts with a 20-step sampler,
  because only 7 full UNet calls remain (−0.21 CLIP interaction in the 2³ factorial).

## Related methods we do not run

These need retraining, distillation or a different model, so they are outside a training-free benchmark:

- **Progressive distillation** [7] and **Latent Consistency Models** [8]: distil the sampler into a
  model that needs 1–8 steps.
- **Adversarial diffusion distillation (SD-Turbo)** [9]: 1–4 step generation after distillation.
- **BK-SDM** [10]: removes UNet blocks and recovers quality by distillation.
- **Token Merging for Stable Diffusion** [11] is training-free but changes attention; a natural next
  experiment, together with `torch.compile`.

## Metrics

- **CLIP Score** [12]: cosine similarity between CLIP image and text embeddings (×100). Our primary
  quality metric; reported with ViT-g/14 (as in DeepCache) and ViT-L/14.
- **FID** [13] and **KID** [14]: distance between Inception feature distributions of generated and real
  images. KID is unbiased at small sample sizes, so we report it on 2,000 COCO images; FID needs
  more images and is planned on 5,000.
- **LPIPS** [15]: learned perceptual distance; we use it against the baseline's image for the same
  prompt and seed, to measure how far a speed-up moves the output.

## References

1. Rombach, Blattmann, Lorenz, Esser, Ommer. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
2. Liu, Ren, Lin, Zhao. Pseudo Numerical Methods for Diffusion Models on Manifolds. ICLR 2022.
3. Song, Meng, Ermon. Denoising Diffusion Implicit Models. ICLR 2021.
4. Lu, Zhou, Bao, Chen, Li, Zhu. DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models. arXiv:2211.01095, 2022.
5. Karras, Aittala, Aila, Laine. Elucidating the Design Space of Diffusion-Based Generative Models. NeurIPS 2022.
6. Ma, Fang, Wang. DeepCache: Accelerating Diffusion Models for Free. CVPR 2024.
7. Salimans, Ho. Progressive Distillation for Fast Sampling of Diffusion Models. ICLR 2022.
8. Luo, Tan, Huang, Li, Zhao. Latent Consistency Models. arXiv:2310.04378, 2023.
9. Sauer, Lorenz, Blattmann, Rombach. Adversarial Diffusion Distillation. arXiv:2311.17042, 2023.
10. Kim, Song, Castells, Choi. BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion. ECCV 2024.
11. Bolya, Hoffman. Token Merging for Fast Stable Diffusion. CVPR Workshops 2023.
12. Hessel, Holtzman, Forbes, Le Bras, Choi. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. EMNLP 2021.
13. Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. NeurIPS 2017.
14. Bińkowski, Sutherland, Arbel, Gretton. Demystifying MMD GANs. ICLR 2018.
15. Zhang, Isola, Efros, Shechtman, Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR 2018.
