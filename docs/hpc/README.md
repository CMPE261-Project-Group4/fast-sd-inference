# HPC guide: fast-sd-inference on the SJSU CoE HPC

Start here. This folder covers everything about running the project on the SJSU College of
Engineering HPC.

| Document | Read it for |
|---|---|
| **README.md** (this page) | The big picture: how the setup works, the 10-phase plan, where we are, what comes next |
| [DEVELOPMENT.md](DEVELOPMENT.md) | How to work on it: first-time setup, the daily loop, jobs, changing packages, models or datasets, git, troubleshooting |
| [PROGRESS.md](PROGRESS.md) | The log: every gate result, job, decision, version and open issue |
| [PLAN.md](PLAN.md) | The original phase-by-phase brief, kept word for word. Where it disagrees with "Corrections to PLAN.md" in PROGRESS.md, the corrections win. |

## How the setup works

There are two clusters. They share storage (`/home`, `/data`, `/scratch`) but nothing else:

| | coe-hpc1 (login node) | COE-HPC3 (head node g17 + compute nodes) |
|---|---|---|
| Used for | editing, git, pip, downloads, unit tests, checks of a few seconds | running every job, submitted with Slurm from g17 (`ssh coe-hpc3`; since 2026-11-01 that name doesn't resolve, so `ssh g17` from coe-hpc1) |
| OS / glibc | CentOS 7 / 2.17 | Rocky 9 / 2.34 |
| Internet | yes | **no**, on g17 and every node |
| GPUs | none | 4× A100-40GB on each of cs001–cs004 (`--gres=gpu:a100:1`) |

So everything a job needs is built and downloaded on coe-hpc1, onto shared storage. Jobs only read it:

- **Python:** a portable CPython 3.11.16 and one shared venv (83 pinned packages, torch 2.6.0 for CUDA 12.4). It runs on both clusters with no `module load`.
- **Models, metric weights and datasets:** downloaded once, pinned to exact versions and sha256 hashes, and verified against their publishers. Jobs load them offline.
- **`env/site.sh`:** the one file every shell and job sources. It sets paths, caches, `umask 002`, offline mode, the venv, and your clone's `src/` on `PYTHONPATH`.

Everyone shares one environment and one copy of the models and data. Each teammate works in their own clone:

| Path (group `cmpe261-fa26`) | Contents |
|---|---|
| `/data/cmpe261-fa26/group4/<your-id>/fast-sd-inference` | your clone |
| `/data/cmpe261-fa26/group4/env/{python,venv}` | portable Python and the shared venv (built by `scripts/hpc/setup_env.sh`) |
| `/data/cmpe261-fa26/group4/hf_cache` | SD1.5, CLIP ViT-L/14 and OpenCLIP ViT-g/14, 11 GB (`scripts/hpc/prefetch_assets.py`) |
| `/data/cmpe261-fa26/group4/cache/{torch,pip,xdg,matplotlib}` | Inception and AlexNet metric weights; tool caches |
| `/data/cmpe261-fa26/group4/datasets/{coco,partiprompts}` | COCO 2017 val and PartiPrompts, 1.8 GB, **read-only** (`scripts/hpc/fetch_datasets.sh`) |
| `/scratch/cmpe261-fa26/group4/results` | run outputs (images, timings, metrics) |

`/home` is 99% full: keep nothing large there. The whole class is in group `cmpe261-fa26` and can write to `group4/`, so we lock finished data with `chmod -R a-w`.

## The plan: 10 phases

Each phase is one branch, `hpc/p<N>-<name>`, stacked on the one before. Each gets one pull request and ends at a **gate**: a check, usually an HPC3 job, that must pass before the next phase starts. Full detail is in [PLAN.md](PLAN.md).

| Phase | Goal | Gate | Status |
|---|---|---|---|
| 0 Orient | measure the real cluster | ssh to HPC3, A100 GRES, writable storage | **pass** |
| 1 Portable environment | one venv for both clusters, no modules | imports pass in a clean environment on both | **pass** (job 92197) |
| 2 Offline assets | models, metric weights and datasets on shared storage | everything loads on HPC3 with the network blocked | **pass** (job 92251) |
| 3 First GPU job (G1) | the pipeline makes images on an A100 | baseline, FP16, DPM-Solver++ and DeepCache make non-blank images; FP16 and DeepCache are faster | **pass** (job 98056) |
| 4 Frozen prompts | the final prompt set, committed once | 2,000 COCO + 1,632 PartiPrompts; Challenge counts match | **pass** (no job; [data/prompts](../../data/prompts/README.md)) |
| 5 Measurement | trustworthy numbers: TF32 off, NFE, stage timings, CLIP/KID/FID/LPIPS | every summary column filled; runs deterministic | **pass** (jobs 98329–98332) |
| 6 Pilot (G2) | first results on 200 prompts | seconds per image and full-sweep GPU-hours reported; team picks the sweep size | **pass** (jobs 100435–100548); the sweep runs on all 3,632 prompts |
| 7 Orchestration | one command runs an experiment file and resumes after failures | a cancelled task reruns alone | **pass** (arrays 100683, 100689, 100695) |
| 8 Full sweeps | each optimization alone, on the full prompt set | all runs done and checked | **pass** (arrays 101761, 101767); gate G3 moved to its own phase |
| Selection (G3) | analyze the single-technique results and choose the combinations to run | team confirms the combined settings | not started; to be planned |
| 9 Combined + FID | combined runs, leave-one-out ablation, FID | final summaries for the report | not started |

## Where we are (2026-11-06)

- **Branches:** Phases 0–7 are merged into `main` (PRs #1–#10). Phase 8 is complete on `hpc/p8-sweeps`, branched from `main` at 28b8378, waiting to be pushed and reviewed.
- **Sweep size (gate G2, Dan, 2026-11-02):** all 3,632 prompts of `eval_prompts.jsonl`, the default in `configs/base.yaml`.
- **Works now:**
  - the shared environment;
  - all models, metric weights and datasets, pinned and verified;
  - the check jobs `check_env.sbatch` and `check_assets.sbatch`, which run from any teammate's clone;
  - **GPU jobs on an A100:** every GPU job starts with `scripts/hpc/job_prelude.sh` (clean environment, thread counts, a per-job stage directory, a job log). `check_gpu.sbatch` ran baseline, FP16, DPM-Solver++ 20 and DeepCache N=3 offline in 64 s (job 98056);
  - `scripts/check_run.py`, which checks a run directory's images, `perf.json`, `timings.jsonl` and quality files, and with `--compare` whether one run reproduces another;
  - **the frozen prompt set:** `data/prompts/eval_prompts.jsonl` (2,000 COCO + 1,632 PartiPrompts, with fixed ids and seeds), plus the subsets `debug` (50), `pilot` (200) and `parti_550` (2,550). See [data/prompts/README.md](../../data/prompts/README.md);
  - **measurement (Phase 5, checked on an A100 in jobs 98329–98332):**
    - TF32 off in `configs/base.yaml`;
    - NFE (`nfe`, `nfe_full`) and per-stage times (text encoder, UNet loop, VAE decode) in every run;
    - the node, GPU UUID, commit, attention processors and TF32 flags in `perf.json`;
    - CLIP Score with ViT-g/14 and ViT-L/14 per image, KID, FID from 5,000 images, LPIPS to the baseline;
    - `run_gpu.slurm`, `evaluate.slurm` and a `summary.csv` per result set;
    - runs are deterministic: two jobs on different GPUs made pixel-identical images;
  - **the pilot (Phase 6, jobs 100435–100548):** 5 configs × 200 prompts, quality-scored, with a per-source table (`summarize.py --by source`). [report/pilot/](../../report/pilot/) has the tables;
  - **the sweep's cost:** `scripts/estimate_sweep.py` models every sweep run's time from the pilot, exact UNet call counts included, and predicts the pilot's held-out runs within 0.2 %;
  - each prompt's starting noise comes from its own CPU generator, drawn in float32, so a seed gives the same noise on any GPU and in FP16 and FP32 runs alike;
  - **one command per sweep (Phase 7, arrays 100683–100695):** `scripts/hpc/submit_experiment.sh` runs experiment files as a Slurm job array, 4 runs at once, and chains the evaluation. Run again after a failed or cancelled task, it redoes only the unfinished runs. Runs are staged on the node's SSD, so a run on `/scratch` with a `perf.json` is complete. See [DEVELOPMENT.md §3](DEVELOPMENT.md#3-running-jobs);
  - **the sweep (Phase 8, arrays 101761 and 101767):** all 17 runs of `00`–`04` on all 3,632 prompts, on cs004, quality-scored and checked, plus a baseline on cs003. [report/sweeps/](../../report/sweeps/) has the tables, per source and per Challenge, with paired confidence intervals and q-values (`scripts/analyze_sweep.py`), the cost model's check and a forecast of the combinations;
  - 159 unit tests, plus 3 `slow` ones run by hand.
- **Watch out:**
  - **GPU 0 on cs002 is broken**: every 1-GPU job that lands there fails with "busy or unavailable". Submit GPU jobs with `sbatch --exclude=cs002 …` until the admins fix it.
  - **CUDA can't use cs001's GPUs 1–3.** A job that gets one fails at model load; resubmit it. Only GPU 0 worked in Phase 6. `submit_experiment.sh` excludes cs001 and cs002 by default.
  - **Nodes differ by up to a few percent:** the same runs were 1.7–3.0 % faster on cs003 than on cs001's GPU 0 (Phase 7), and 0.4 % slower on cs003 than on cs004 (Phase 8). Compare timings from the same node.
- **Sweep results** (A100, TF32 off, all 3,632 prompts, one node; the full table is in [PROGRESS.md](PROGRESS.md#phase-8-full-sweeps-r7-and-d6-2026-11-03-to-2026-11-04)):

  | Config | s/image | Speedup | Peak GB | ΔCLIP ViT-g/14 [95 % CI] | LPIPS to baseline |
  |---|---|---|---|---|---|
  | baseline (PNDM 50, FP32) | 5.816 | 1.00× | 4.76 | 0 (30.06) | 0 |
  | `steps_10` (PNDM) | 1.329 | 4.38× | 4.76 | −2.37 [−2.53, −2.21] | 0.481 |
  | `steps_20` (PNDM) | 2.448 | 2.38× | 4.76 | −0.31 [−0.42, −0.21] | 0.300 |
  | `ddim_20` | 2.339 | 2.49× | 4.76 | +0.05 [−0.05, +0.16] | 0.336 |
  | `dpmpp_20` | 2.343 | 2.48× | 4.76 | +0.13 [+0.04, +0.22] | 0.254 |
  | `euler_20` | 2.342 | 2.48× | 4.76 | +0.11 [+0.02, +0.21] | 0.313 |
  | `fp16` | 1.442 | 4.03× | 2.61 | −0.01 [−0.04, +0.01] | 0.016 |
  | `fp16_bs4` | 0.999 | 5.82× | 4.02 | −0.00 [−0.03, +0.02] | 0.015 |
  | `deepcache_i3` | 2.359 | 2.47× | 4.97 | +0.07 [−0.03, +0.16] | 0.255 |
  | `deepcache_i5` | 1.748 | 3.33× | 4.97 | −0.04 [−0.15, +0.07] | 0.329 |

  Fewer PNDM steps lose quality; at 20 steps the other schedulers don't. FP16 changes nothing measurable, and DeepCache holds its CLIP Score up to interval 5.

## Next steps

1. **Push `hpc/p8-sweeps` and open Phase 8's PR.**
2. **Plan the selection phase (gate G3):** analyze the single-technique results to choose the combinations worth running; then the team confirms the settings for `05_combined.yaml`, now FP16 + DPM-Solver++ 2M at 20 steps + DeepCache interval 3. [PROGRESS.md](PROGRESS.md#what-the-sweep-says-for-gate-g3) has what the sweep says about them and the alternatives.

**After that:**
- **Phase 9:** the combined runs (`05_combined` adds about 4.1 GPU-hours) and the final summaries.

**Decisions the team still owes** (each one is recorded in PROGRESS.md when made):

| When | Decision |
|---|---|
| Gate G3 (its own phase, before Phase 9) | The combined settings for `05_combined.yaml` |
| Gate G4 | Whether to add 3,000 COCO prompts so FID can use 5,000 images |
