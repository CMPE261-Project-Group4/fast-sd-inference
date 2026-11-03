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
| 7 Orchestration | one command runs an experiment file and resumes after failures | a cancelled task reruns alone | not started |
| 8 Full sweeps (G3) | each optimization alone, on the full prompt set | team confirms the combined settings | not started |
| 9 Combined + FID | combined runs, leave-one-out ablation, FID | final summaries for the report | not started |

## Where we are (2026-11-02)

- **Branches:** Phases 0–5 are merged into `main` (PRs #1–#7). Phase 6 is on `hpc/p6-pilot`, branched from `main`, waiting to be pushed and reviewed.
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
  - 138 unit tests, plus 3 `slow` ones run by hand.
- **Watch out:**
  - **GPU 0 on cs002 is broken**: every 1-GPU job that lands there fails with "busy or unavailable". Submit GPU jobs with `sbatch --exclude=cs002 …` until the admins fix it.
  - **CUDA can't use cs001's GPUs 1–3.** A job that gets one fails at model load; resubmit it. Only GPU 0 worked in Phase 6.
- **Pilot results** (A100, FP32 with TF32 off, batch 1):

  | Config | s/image | Speedup | Peak GB | CLIP ViT-g/14 | LPIPS to baseline |
  |---|---|---|---|---|---|
  | baseline (PNDM 50) | 5.775 | 1.00× | 4.76 | 30.46 | 0 |
  | `ddim_20` | 2.340 | 2.47× | 4.76 | 30.47 | 0.346 |
  | `dpmpp_20` | 2.344 | 2.46× | 4.76 | 30.57 | 0.255 |
  | `deepcache_i3` | 2.357 | 2.45× | 4.97 | 30.56 | 0.259 |
  | `fp16` | 1.468 | 3.93× | 2.61 | 30.45 | 0.012 |

  The CLIP Scores are all within 0.12 of each other, which is noise at 200 prompts. KID is read only from the sweeps.

## Next steps

1. **Phase 7:** job arrays that resume after failures.
2. **Phase 8:** the sweep on all 3,632 prompts: the 17 runs of `00`–`04` plus a second-node baseline, about 65.5 GPU-hours, or 16.4 h on 4 GPUs. [PROGRESS.md](PROGRESS.md#phase-6-pilot-r5-and-d5-gate-g2-2026-10-30-to-2026-11-02) has the estimate.

**After that:**
- **Phase 9:** the combined runs (`05_combined` adds about 4.1 GPU-hours) and the final summaries.

**Decisions the team still owes** (each one is recorded in PROGRESS.md when made):

| When | Decision |
|---|---|
| Gate G3 | The combined settings for `05_combined.yaml` |
| Gate G4 | Whether to add 3,000 COCO prompts so FID can use 5,000 images |
