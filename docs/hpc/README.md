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
| Used for | editing, git, pip, downloads, unit tests, checks of a few seconds | running every job, submitted with Slurm from g17 (`ssh coe-hpc3`) |
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
| 5 Measurement | trustworthy numbers: TF32 off, NFE, stage timings, CLIP/KID/FID/LPIPS | every summary column filled; runs deterministic | **next** |
| 6 Pilot (G2) | first results on 200 prompts | seconds per image and full-sweep GPU-hours reported; team picks the sweep size | not started |
| 7 Orchestration | one command runs an experiment file and resumes after failures | a cancelled task reruns alone | not started |
| 8 Full sweeps (G3) | each optimization alone, on the full prompt set | team confirms the combined settings | not started |
| 9 Combined + FID | combined runs, leave-one-out ablation, FID | final summaries for the report | not started |

## Where we are (2026-10-28)

- **Branches:** Phases 0–2 are merged into `main` (PRs #1–#4). Phase 3 is on `hpc/p3-gpu`, branched from `main`, and Phase 4 on `hpc/p4-prompts`, stacked on it. Both are waiting to be pushed and reviewed.
- **Works now:**
  - the shared environment;
  - all models, metric weights and datasets, pinned and verified;
  - the check jobs `check_env.sbatch` and `check_assets.sbatch`, which run from any teammate's clone;
  - **GPU jobs on an A100:** every GPU job starts with `scripts/hpc/job_prelude.sh` (clean environment, thread counts, a per-job stage directory, a job log). `check_gpu.sbatch` ran baseline, FP16, DPM-Solver++ 20 and DeepCache N=3 offline in 64 s (job 98056);
  - `scripts/check_run.py`, which checks a run directory's images and `perf.json`;
  - **the frozen prompt set:** `data/prompts/eval_prompts.jsonl` (2,000 COCO + 1,632 PartiPrompts, with fixed ids and seeds), plus the subsets `debug` (50), `pilot` (200) and `parti_550` (2,550). See [data/prompts/README.md](../../data/prompts/README.md);
  - 67 unit tests.
- **Watch out:** **GPU 0 on cs002 is broken**: every 1-GPU job that lands there fails with "busy or unavailable". Submit GPU jobs with `sbatch --exclude=cs002 …` until the admins fix it.
- **Not done yet:**
  - **`run_gpu.slurm` is reworked but hasn't run.** Its prompt files now exist; its first job is one of Phase 5's debug runs.
  - **The metric code needs work in Phase 5.** `src/eldm/metrics.py` still asks for CLIP by model name, which fails offline and with transformers 5.
  - **No real timings yet.** Phase 3's numbers are smoke numbers: 2 images, a co-tenant on the node, TF32 at torch's defaults.

## Next steps

**Phase 5, measurement correctness** (`hpc/p5-measure`). The goal: every number in `summary.csv` can be trusted before the pilot.
1. Pin the TF32 flags in the config, off by default, so FP32 really is FP32. Log the attention processor, the node and the GPU, and count NFE with a UNet hook.
2. Time the text encoder, the UNet loop and the VAE decode with CUDA events. End-to-end latency stays the headline number.
3. CLIP Score with OpenCLIP ViT-g/14 (primary) and ViT-L/14, with per-image scores in `quality_per_image.csv`.
4. KID (`subset_size=min(1000, n)`), FID only at ≥ 5,000 images, and every weight loaded offline.
5. Two debug runs of `00_baseline` on `subsets/debug.jsonl`, the first GPU jobs to use `run_gpu.slurm`. Same seeds must give the same images. The gate: every summary column is filled, runs are deterministic, and the stage times add up to about the end-to-end latency.

**After that:**
- **Phase 6:** a 200-prompt pilot.
- **Phase 7:** job arrays that resume after failures.
- **Phases 8–9:** the sweeps and the final runs.

**Decisions the team still owes** (each one is recorded in PROGRESS.md when made):

| When | Decision |
|---|---|
| Phase 5 | How CLIP Score is computed: the transformers 5 adapter in torchmetrics' CLIPScore, or computing the per-image score by hand |
| Phase 5 | How NFE is counted under DeepCache: full vs partial UNet passes |
| Gate G2 | Sweep size: all of `eval_prompts.jsonl`, or `subsets/parti_550.jsonl` (all 2,000 COCO + 550 stratified PartiPrompts) |
| Gate G3 | The combined settings for `05_combined.yaml` |
| Gate G4 | Whether to add 3,000 COCO prompts so FID can use 5,000 images |
