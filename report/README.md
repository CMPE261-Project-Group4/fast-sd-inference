# report/: result tables behind the report

Every number in the report comes from a CSV here. They are small, committed, and written by scripts
(never edited by hand), so anyone can redraw the figures or check a number without the GPU outputs in `results/`.

| Folder | What | GPU, prompts | Written by |
|---|---|---|---|
| `pilot/` | 5 runs that tested the setup and priced the sweep | A100, 200 prompts | `summarize.py`, `estimate_sweep.py` |
| `sweeps/` | Each technique alone: 17 runs of `00`–`04` | A100 (all on cs004), 3,632 prompts | `summarize.py`, `analyze_sweep.py` |
| `local4090/` | Combined techniques: 12 runs of `05_combined.yaml` (2³ factorial + extensions) | RTX 4090, 3,632 prompts | `summarize.py`, `analyze_sweep.py`, `factorial_effects.py` |
| `figures/` | Report figures (PNG, not committed; copies on the team Drive) | — | `plot_report_figures.py` |

Speedups are against the baseline measured on the same GPU in the same folder; don't mix folders.

## Files

| File | One row per | Key columns |
|---|---|---|
| `summary.csv` | run | `latency_per_image_s`, `speedup`, `throughput_img_per_s`, `peak_mem_gb`, `nfe` / `nfe_full` (UNet calls, all / full), stage times `t_text_s` `t_unet_s` `t_vae_s`, `clip_vitg`, `clip_vitl`, `kid_mean` ± `kid_std` (COCO only), `lpips_to_baseline`, `node`, `gpu` |
| `summary_by_source.csv` | run × source (coco, parti) | mean CLIP Scores and LPIPS per source |
| `summary_by_challenge.csv` | run × PartiPrompts Challenge (COCO = `(none)`) | mean CLIP Scores and LPIPS per Challenge |
| `paired_quality.csv` | run × group × metric | `mean_diff` vs the baseline, paired t-test `ci_low` / `ci_high`, `p_value`, Benjamini–Hochberg `q_value` within each group type; `group_by` = `all`, `source` or `challenge` |
| `sweeps/baseline_nodes.csv` | node | the baseline on cs004 and cs003 (node-to-node timing check) |
| `sweeps/cost_model.csv` | parameter | fitted seconds per full / cached UNet call and per-image overhead, FP32 and FP16 |
| `sweeps/estimate_check.csv` | run | pilot cost-model estimate vs measured s/image, `error_pct` |
| `sweeps/forecast.csv` | combined run | cost-model forecast for `05_combined.yaml` (made before the 4090 runs) |
| `pilot/sweep_estimate.csv` | planned run | estimated s/image and GPU-hours used to size the sweep |
| `local4090/factorial_effects.csv` | effect (S, P, C, SP, SC, PC, SPC) | Yates effects on `log2_time`, `delta_clip`, `lpips`; `speedup_factor` for main effects |
| `local4090/speedup_product_check.csv` | multi-factor run | product of single-factor speedups vs measured |

## Regenerate

```bash
python scripts/plot_report_figures.py
python scripts/factorial_effects.py
```

The commands that rebuild the tables themselves from `results/` are in `docs/hpc/PROGRESS.md`, under each phase's
"Outputs and how to regenerate them".
