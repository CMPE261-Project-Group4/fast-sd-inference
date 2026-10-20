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
| 3 First GPU job (G1) | the pipeline makes images on an A100 | baseline, FP16, DPM-Solver++ and DeepCache make non-blank images; FP16 and DeepCache are faster | **next** |
| 4 Frozen prompts | the final prompt set, committed once | 2,000 COCO + 1,632 PartiPrompts; Challenge counts match | not started |
| 5 Measurement | trustworthy numbers: TF32 off, NFE, stage timings, CLIP/KID/FID/LPIPS | every summary column filled; runs deterministic | not started |
| 6 Pilot (G2) | first results on 200 prompts | seconds per image and full-sweep GPU-hours reported; team picks the sweep size | not started |
| 7 Orchestration | one command runs an experiment file and resumes after failures | a cancelled task reruns alone | not started |
| 8 Full sweeps (G3) | each optimization alone, on the full prompt set | team confirms the combined settings | not started |
| 9 Combined + FID | combined runs, leave-one-out ablation, FID | final summaries for the report | not started |

## Where we are (2026-10-19)

- **Branches:** `hpc/p0-orient`, `hpc/p1-env` and `hpc/p2-assets` are pushed, one PR each. `main` is still the scaffold until those merge, so work from `hpc/p2-assets` for now.
- **Works now:**
  - the shared environment;
  - all models, metric weights and datasets, pinned and verified;
  - the check jobs `scripts/hpc/check_env.sbatch` and `scripts/hpc/check_assets.sbatch`, which run from any teammate's clone;
  - 23 unit tests.
- **Not done yet:**
  - **No GPU job has run.** `scripts/hpc/run_gpu.slurm` is still the original scaffold, with the wrong partition and a plain `gpu:1`. Don't submit it until Phase 3 reworks it.
  - **The metric code needs work in Phase 5.** `src/eldm/metrics.py` still asks for CLIP by model name, which fails offline and with transformers 5.
  - **No final prompt set yet.** It comes in Phase 4.

## Next steps

**Phase 3, the first GPU job** (`hpc/p3-gpu`):
1. Rework `scripts/hpc/run_gpu.slurm` into the template in [DEVELOPMENT.md](DEVELOPMENT.md#writing-a-job-script):
   - `-p gpuqs --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G`;
   - find the clone from `$SLURM_SUBMIT_DIR` and source `env/site.sh`;
   - set thread counts from the allocation;
   - send results to scratch.
2. Add `scripts/check_run.py` and its tests. It checks that the image count matches the prompts, that the PNGs are readable and not blank, and that `perf.json` has the required fields.
3. Add `scripts/hpc/gpu_check.py` and `check_gpu.sbatch`. They make 2 images each for baseline PNDM-50 FP32, FP16, DPM-Solver++ 20 and DeepCache N=3.
   - Images go to a per-job folder under `$TMPDIR`, removed on exit, then are copied to `results/_checks/<jobid>/`.
   - The GPU is logged with `nvidia-smi -i "$CUDA_VISIBLE_DEVICES"`.

**After that:**
- **Phase 4:** parse PartiPrompts, sample stratified by Challenge, and freeze the prompts and subsets.
- **Phase 5:** measurement correctness.
- **Phase 6:** a 200-prompt pilot.
- **Phase 7:** job arrays that resume after failures.
- **Phases 8–9:** the sweeps and the final runs.

**Decisions the team still owes** (each one is recorded in PROGRESS.md when made):

| When | Decision |
|---|---|
| Phase 5 | How CLIP Score is computed: the transformers 5 adapter in torchmetrics' CLIPScore, or computing the per-image score by hand |
| Phase 5 | How NFE is counted under DeepCache: full vs partial UNet passes |
| Gate G2 | Sweep size: full PartiPrompts, or the 550-prompt stratified subset |
| Gate G3 | The combined settings for `05_combined.yaml` |
| Gate G4 | Whether to add 3,000 COCO prompts so FID can use 5,000 images |
