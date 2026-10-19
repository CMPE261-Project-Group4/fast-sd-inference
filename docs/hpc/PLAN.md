# HPC brief: fast-sd-inference on SJSU CoE HPC

## Context

You're working for Dan Lam (HPC user `011383814`) on a CMPE 261 team project with Ruixiang Huang and Haoran Jiang.

**The project** is a controlled benchmark of training-free inference optimizations for pretrained Stable Diffusion v1.5. We train nothing.
- **Optimizations:**
  - fewer denoising steps;
  - schedulers: PNDM (default), DDIM, DPM-Solver++ 2M, Euler;
  - FP16;
  - DeepCache feature caching.
- **Order of experiments:** each optimization alone, then combined, then a leave-one-out ablation.
- **Quality metrics:** CLIP Score (OpenCLIP ViT-g/14 primary, ViT-L/14 secondary), KID (primary at 2,000 images), FID (only at ≥5,000 images), LPIPS vs. baseline.
- **Performance metrics:** latency, peak memory, throughput, NFE (UNet calls).
- **Data:** COCO 2017 val (2,000 images, one caption each, sample seed 0) plus PartiPrompts (1,632 prompts with Category and Challenge labels). Every config uses the same prompts and seeds.

**The repo** is `github.com/CMPE261-Project-Group4/fast-sd-inference`. It's a scaffold that has never run on a GPU:
- package code: `src/eldm/{config,prompts,pipeline,benchmark,metrics}.py`;
- scripts: `scripts/{prepare_prompts,run_experiments,evaluate_quality,summarize}.py`;
- configs: `configs/base.yaml` and `configs/experiments/00–05`;
- `scripts/hpc/run_gpu.slurm` and `tests/test_config.py`.

Read `README.md` first and follow its reproducibility rules. Task IDs below (R1, D3, …) match the team's task tracker.

**Your job:** take the HPC side from zero to the full sweeps in 10 phases and about 25 commits.
- Developing here is slow, so every phase uses the same inner loop and ends at a gate.
- Each phase is one branch, and each commit is one meaningful check-in the team can review on GitHub.
- Don't start the next phase until the gate passes and you've reported to Dan.

**You never submit Slurm jobs.** Dan submits every job himself, so a person supervises all cluster use.
- You write and test the job scripts.
- You hand Dan an exact job request (template below).
- After he gives you the job ID, you read the results.

## Cluster facts (measured 2026-10-15; re-check with `sinfo` and `df -h`)

- **Two clusters.**
  - **coe-hpc1**, where you run: CentOS 7, glibc 2.17, has internet, Slurm unusable.
  - **COE-HPC3**, where jobs run: head node g17 (Rocky 9, glibc 2.34). It has **no internet**, on g17 or on any compute node.
- **Job commands:** jobs are submitted from g17 (`ssh coe-hpc3`), and only Dan does that. Job logs and outputs live on shared storage, so you can read them directly from coe-hpc1.
- **Shared storage:** only `/home`, `/data` and `/scratch` are shared between the clusters. `/home` is 99% full, so put nothing big there.
- **GPU request:** always `-p gpuqs --gres=gpu:a100:1` (A100-PCIE-40GB, nodes cs001–cs004). A plain `gpu:1` lands on a P100, so never use it.
  - `gpuqs` has a 2-day limit and allows at most 2 nodes per user.
  - Each node has a local SSD at `$TMPDIR`.
- **Modules are per cluster.** Jobs must never `module load`. For git on coe-hpc1, use the Spack git module (2.39.1); the system git (1.8.3) is too old.
- **Reuse:** if Dan's AFAR_HPC repo (CMPE 295) is on the cluster, copy its patterns: `env/site.sh`, `env/check.sh --runtime`, `slurm/check.sbatch`, and the portable CPython pinned in `env/python.pin`.

## Paths

All of these belong to group `cmpe261-fa26` and are already group-writable.

| What | Path |
|---|---|
| Dan's clone (create it if missing) | `/data/cmpe261-fa26/group4/011383814/fast-sd-inference` |
| Python + venv + wheels | `/data/cmpe261-fa26/group4/env/` |
| `HF_HOME` | `/data/cmpe261-fa26/group4/hf_cache` |
| `TORCH_HOME`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME` | `/data/cmpe261-fa26/group4/cache/` |
| Datasets | `/data/cmpe261-fa26/group4/datasets/{coco,partiprompts}` |
| Run outputs (`output_root`) | `/scratch/cmpe261-fa26/group4/results` |
| Job logs | `<clone>/logs/` (commit `logs/.gitkeep`) |

The parent folders are shared with the whole class. Never touch anything outside `group4/`.

## Rules

1. **Never run Slurm commands that start, change or stop jobs.** That means `sbatch` (including `--test-only`), `srun`, `salloc`, `scancel`, `scontrol update/hold/release`, and any wrapper script that calls them. The one exception is `submit_experiment.sh --dry-run` (Phase 7), which only prints commands.
   - Read-only queries are fine: `ssh coe-hpc3` with `squeue -u $USER`, `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode`, `sinfo`, `scontrol show job <id>`.
2. **Login node:** editing, pip, downloads, CPU unit tests and tiny CPU checks of a few seconds only. Nothing GPU, and nothing long.
3. **Every job script:**
   - starts with `set -euo pipefail` and uses absolute paths;
   - runs `source <clone>/env/site.sh`, which sets paths, `umask 002`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `PYTHONNOUSERSITE=1`, and loads no modules;
   - writes to a log directory that already exists;
   - sets an explicit `--time` no larger than the job needs.
4. **Everything a job needs is downloaded first on coe-hpc1.**
5. **`umask 002` everywhere,** so teammates can edit what you create.
6. **Ask Dan first** before any of these:
   - deleting anything you didn't create;
   - any change to the experiment design (configs 01–05, metrics, prompt sets).
7. **Lock finished data** with `chmod -R a-w`: datasets once downloaded, run directories once evaluated.

## Job request template

When a step needs the cluster, stop and give Dan this block. Wait for his reply with the job ID or the outcome before going on.

```
JOB REQUEST P<phase>.<n>: <one-line purpose>
Run on g17 (ssh coe-hpc3), from <clone>:
  sbatch --test-only <script> <args>     # validates the request, runs nothing
  sbatch <script> <args>
Resources: <partition>, <GPU or CPU>, --time <limit>, estimate <N> GPU-hours
Reads: <inputs>    Writes: <output paths>
Success looks like: <checker command and its expected result>
Reply with the job ID once submitted.
```

- Ask for one job at a time unless Dan says otherwise.
- Always ask for the smallest job that answers the question.
- Say plainly when a job is large (over about 4 GPU-hours or more than one GPU), so Dan can decide.

## Git and commit rules

- **Local commits only.** Never push, never commit to `main`, never rewrite history. Dan reviews, pushes and opens one PR per phase.
- **Branches:** one per phase, named in the phase heading below. Branch each one from the previous phase's branch so they stack in order.
- **One logical change per commit,** exactly as listed in each phase. If you find you need an extra change, make it its own commit with a clear message rather than folding it into an unrelated one.
- **Before every commit:** `pytest -q` passes, plus the commit's own check listed below. Never commit a broken state.
- **Message:** first line `type(scope): summary` (types: `build`, `feat`, `fix`, `test`, `data`, `exp`, `results`, `docs`). The body says why and how it was checked, including the job ID when a job ran.
- **Identity:** commit as Dan (check `git config user.name`; ask if it's unset).
- **Commit:** code, configs, prompt files, checksums, summary CSVs, figures, and `docs/hpc/*`.
- **Never commit:** images, datasets, model weights, venvs, logs.
- **Progress log:** the last commit of every phase also updates `docs/hpc/PROGRESS.md` with:
  - the gate result;
  - the job table: date · job ID · purpose · node · result;
  - decisions and versions (torch, diffusers, driver);
  - open issues.

## The inner loop (every commit that needs the cluster)

1. Edit on coe-hpc1, then run `pytest -q` in the venv.
2. Write or update the job script. Read it back for paths, resources and `--time`.
3. **Request a tiny job first**: 2–4 prompts and `--time=00:20:00`.
4. While it runs, follow it read-only with `squeue`/`sacct` and its log file. Don't loop on checks every few seconds; check at sensible intervals.
5. When it finishes, check the outputs with `scripts/check_run.py` (added in Phase 3), not by eye.
6. If it failed, read the log, fix the script, and request a new tiny job.
7. Request the full-size job only after the tiny one passes. Then commit.
8. **At the end of each phase, report to Dan:**
   - the commits made;
   - tests and job IDs with results;
   - problems;
   - gate status;
   - the next phase.

## Phases and commits

### Phase 0: Orient (branch `hpc/p0-orient`)

Goal: confirm the facts above before changing anything.

1. Run these checks:
   - `id`, to confirm `cmpe261-fa26` is listed;
   - `df -h /home /data /scratch`;
   - `ls -la /data/cmpe261-fa26/group4 /scratch/cmpe261-fa26/group4`;
   - `ssh coe-hpc3 "sinfo -o '%P %N %G %l'"`.
2. Make one commit:
   - **`docs(hpc): add HPC plan and progress log`**
   - Files: `docs/hpc/PLAN.md` (this brief, saved as-is) and `docs/hpc/PROGRESS.md` (what you found).

**Gate:**
- `ssh coe-hpc3 hostname` works without a prompt;
- the `a100` gres name is confirmed;
- `group4/` is writable.

### Phase 1: Portable environment, R1 (branch `hpc/p1-env`)

Goal: one Python environment on shared storage that works on both clusters with no modules.

**Commits:**

1. **`build(env): pin portable Python and Python dependencies`**
   - Files: `env/python.pin`, `requirements.in`, `requirements.lock`.
   - **Python:** CPython 3.11 from python-build-standalone (install_only, x86_64 linux-gnu), pinned by URL and sha256.
   - **Torch:** `torch==2.6.0` and `torchvision==0.21.0` from the cu124 index. These are the last builds that install on glibc 2.17; torch 2.7+ needs glibc 2.28.
   - **Other pins:** add `open_clip_torch`, and pin diffusers, transformers, DeepCache, torchmetrics, torch-fidelity and lpips.
   - **Lock file:** `requirements.lock` comes from `pip freeze`.
   - **Check:** `pip install --only-binary=:all: -r requirements.lock` succeeds on coe-hpc1.
   - **Fallback** if pip refuses on glibc 2.17:
     - on coe-hpc1, `pip download --platform manylinux_2_28_x86_64 --python-version 3.11 --only-binary=:all:` into `env/wheels`;
     - then build the venv with `--no-index --find-links` in a CPU job on HPC3, which you write and Dan submits.
2. **`build(hpc): add idempotent setup_env.sh and env/site.sh`**
   - Files: `scripts/hpc/setup_env.sh`, `env/site.sh`.
   - `setup_env.sh` installs the Python into `env/python` and builds `env/venv` from it, never from a module Python.
   - `site.sh` sets paths, caches, `umask 002` and the offline flags.
   - **Check:** running `setup_env.sh` a second time changes nothing.
3. **`test(hpc): add clean-environment import check job`**
   - Files: `scripts/hpc/check_env.sbatch`.
   - It's a CPU job (`-p defq -c 2`) that imports torch, diffusers, transformers, DeepCache, torchmetrics, open_clip and lpips inside `env -i HOME=$HOME USER=$USER PATH=/usr/bin:/bin bash …`.
   - Also update PROGRESS.md.
   - **Check:** request the job; it passes on HPC3.

**Gate:** imports pass on both clusters in a clean environment.

### Phase 2: Offline assets, R2 and part of D2 (branch `hpc/p2-assets`)

Goal: everything a job needs is on shared storage, and jobs never touch the network.

**Commits:**

1. **`feat(hpc): prefetch model and metric weights for offline jobs`**
   - Files: `scripts/hpc/prefetch_assets.py`, `configs/base.yaml` (adds `model_revision`).
   - **SD1.5** (`stable-diffusion-v1-5/stable-diffusion-v1-5`), pinned by commit hash. Download only `model_index.json` and the component folders: scheduler, text_encoder, tokenizer, unet, vae, feature_extractor. Skip the large single-file checkpoints at the top level.
   - **CLIP:** `openai/clip-vit-large-patch14` and OpenCLIP ViT-g/14 `laion2b_s34b_b88k`.
   - **Metric weights** (FID/KID Inception, LPIPS AlexNet): warm them by running each metric once on CPU with 2 tiny images.
2. **`data: add dataset fetch script and checksums`**
   - Files: `scripts/hpc/fetch_datasets.sh`, `data/CHECKSUMS.txt`.
   - Downloads COCO 2017 `val2017.zip` and `annotations_trainval2017.zip` into `datasets/coco/`, and `PartiPrompts.tsv` (google-research/parti) into `datasets/partiprompts/`.
   - Records sha256 sums, then runs `chmod -R a-w` on the datasets.
3. **`test(hpc): add offline asset load check job`**
   - Files: `scripts/hpc/check_assets.sbatch`.
   - It's a CPU job with the offline flags set that loads every model, metric and dataset file. No generation.
   - Also update PROGRESS.md.
   - **Check:** request the job; it passes.

**Gate:** everything loads on HPC3 with no network.

### Phase 3: First GPU job, R3 and gate G1 (branch `hpc/p3-gpu`)

Goal: the existing pipeline generates images on an A100.

**Commits:**

1. **`fix(hpc): rework run_gpu.slurm for HPC3`**
   - Files: `scripts/hpc/run_gpu.slurm`, `logs/.gitkeep`, `.gitignore` (adds `logs/*` and `!logs/.gitkeep`).
   - Request: `-p gpuqs --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G`.
   - Absolute paths, `source env/site.sh`, no module lines, `output_root` set to scratch.
2. **`test: add run-directory checker`**
   - Files: `scripts/check_run.py`, `tests/test_check_run.py`.
   - Checks: the image count matches the prompts, every PNG is readable and not blank, required `perf.json` fields are present, `quality.json` exists if expected.
   - The inner loop uses this from now on.
3. **`test(hpc): add GPU smoke check for four variants`**
   - Files: `scripts/hpc/gpu_check.py`, `scripts/hpc/check_gpu.sbatch`.
   - Prints `nvidia-smi` output and the GPU name.
   - Uses `eldm.pipeline.build_pipeline` to make 2 images each for: baseline PNDM-50 FP32, FP16, DPM-Solver++ 20, and DeepCache N=3.
   - Images go to `$TMPDIR`, then are copied to `results/_checks/<jobid>/`.
   - **Check:** request the job; `check_run.py` passes on its outputs.
   - **If DeepCache fails** with the pinned diffusers, step diffusers down until it works. Re-pin it in `requirements.in`/`requirements.lock` as a separate `build(env)` commit before this one.
   - Also update PROGRESS.md with the timings.

**Gate G1:**
- all four variants make non-blank images on an A100, offline, in a clean environment;
- FP16 and DeepCache are faster than the baseline.

### Phase 4: Frozen prompts, D1 and D2 (branch `hpc/p4-prompts`)

Goal: the final prompt set every run will use, committed once.

**Commits:**

1. **`feat(data): parse PartiPrompts and add the challenge field`**
   - Files: `src/eldm/prompts.py`, `scripts/prepare_prompts.py`, `tests/test_prompts.py`.
   - Parse the TSV columns Prompt, Category, Challenge and Note.
   - Add an optional `challenge` field to `Prompt`. Loading existing JSONL files must still work.
   - Default `--sample-seed 0`, matching the data doc.
   - Remove the manual prompts from the evaluation set. Keep `manual_prompts.yaml` as a smoke-test set.
   - Log how many prompts exceed 77 CLIP tokens, but keep them, since every config truncates them the same way.
2. **`feat(data): add Challenge-stratified sampling and subset files`**
   - Files: `scripts/prepare_prompts.py`, `tests/test_prompts.py`.
   - Subsets keep the full set's ids and seeds.
   - Tests cover stratification proportions and id/seed stability.
3. **`data: freeze evaluation prompts and subsets`**
   - Files: `data/prompts/eval_prompts.jsonl`, `data/prompts/subsets/{debug,pilot,parti_550}.jsonl`, `data/prompts/README.md`.
   - Contents:
     - the full set: COCO 2,000 + PartiPrompts 1,632;
     - debug = the first 50;
     - pilot = 100 COCO + 100 PartiPrompts, stratified;
     - `parti_550` = the stratified fallback.
   - The README documents the seeds and the command used.
   - Also update PROGRESS.md.
   - **Check:** counts match: 2,000 COCO, 1,632 PartiPrompts, 90 Quantity, 91 Writing & Symbols.

**Gate:** the prompt files are committed and their counts match the data doc.

### Phase 5: Measurement correctness, R4 and D3 (branch `hpc/p5-measure`)

Goal: every number in `summary.csv` is trustworthy before the pilot.

**Commits:**

1. **`feat(perf): pin TF32 flags, log run environment, count NFE`**
   - Files: `configs/base.yaml`, `src/eldm/pipeline.py`, `src/eldm/benchmark.py`, `tests/test_config.py`.
   - Config keys for `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32`, default off so FP32 really is FP32.
   - Log the attention processor, `SLURMD_NODENAME` and the GPU name.
   - Count NFE with a UNet forward hook.
2. **`feat(perf): add per-stage timing with CUDA events`**
   - Files: `src/eldm/benchmark.py`.
   - Times the text encoder, UNet loop and VAE decode with events recorded in hooks, without per-call syncs.
   - End-to-end latency stays the headline number.
3. **`feat(eval): OpenCLIP ViT-g/14 CLIP Score and per-image scores`**
   - Files: `src/eldm/metrics.py`, `scripts/evaluate_quality.py`, `tests/test_metrics.py`.
   - ViT-g/14 is primary and ViT-L/14 secondary.
   - Writes `quality_per_image.csv` with columns id, clip_vitg, clip_vitl, lpips_to_baseline.
4. **`feat(eval): add KID, gate FID at 5,000 images, load weights offline`**
   - Files: `src/eldm/metrics.py`, `scripts/evaluate_quality.py`, `scripts/summarize.py`.
   - KID with `subset_size=min(1000, n)`.
   - New columns in the summary.
5. **`test(hpc): debug-run determinism check`**
   - Files: `scripts/check_run.py` (adds a compare mode).
   - Request two runs of `00_baseline` on `debug.jsonl`, then an evaluate-and-summarize job.
   - Same seeds must give identical images, or LPIPS < 0.01.
   - Also update PROGRESS.md.

**Gate:**
- every `summary.csv` column is filled;
- runs are deterministic;
- stage times add up to roughly the end-to-end latency.

### Phase 6: Pilot, R5 and D5, gate G2 (branch `hpc/p6-pilot`)

Goal: the first real results, which also feed the initial-draft check-in.

**Commits:**

1. **`exp: add pilot experiment`**
   - Files: `configs/experiments/pilot.yaml`.
   - 5 runs on `pilot.jsonl`: baseline, `ddim_20`, `dpmpp_20`, `deepcache_i3`, `fp16`.
2. **`feat(viz): configurable summary output path`**
   - Files: `scripts/summarize.py`.
   - Adds `--summary-out`. The script currently always writes `results/summary.csv`, which is git-ignored.
3. **`results: pilot summary, plots and image grid`**
   - Files: `report/pilot/summary.csv`, `report/figures/pilot/*`.
   - Skip KID at 200 prompts.
   - Also update PROGRESS.md with the sanity checks:
     - The baseline ViT-g/14 CLIP should be near DeepCache's published 50-step PLMS numbers: 29.51 (PartiPrompts) and 30.30 (COCO). Expect noise at 200 prompts.
     - The 20-step runs should be the fastest. DeepCache N=3 should be about 1.7× faster (the paper's figure on an RTX 2080).
     - FP16 should be faster than FP32 and use less memory.

**Gate G2:** report seconds per image for each config and the GPU-hours the full sweep would take. Dan then chooses the sweep size: full PartiPrompts or the 550-prompt subset.

### Phase 7: Orchestration, R6 (branch `hpc/p7-orchestration`)

Goal: one command runs a whole experiment file, and can resume after failures.

**Commits:**

1. **`feat(runner): add --list, --index, --skip-existing and --prompts-file`**
   - Files: `scripts/run_experiments.py`, `tests/test_run_experiments.py`.
2. **`feat(runner): stage images on the node-local SSD`**
   - Files: `src/eldm/benchmark.py`.
   - Writes images to `$TMPDIR`, then copies them to `output_root` at the end. Timing is unchanged.
3. **`feat(hpc): job arrays with chained evaluate and summarize jobs`**
   - Files: `scripts/hpc/run_array.sbatch`, `scripts/hpc/submit_experiment.sh`, `README.md` (workflow section).
   - `submit_experiment.sh` is a tool for Dan to run on g17. You write it and test its logic with a `--dry-run` flag that prints the sbatch commands without running them. You never run it for real.
   - It submits an array capped at `%4`, then `afterok` evaluate and summarize jobs, and prints the job IDs.
   - Also update PROGRESS.md.
   - **Check:** ask Dan to submit a tiny experiment (2 runs × 4 prompts) with the script, cancel one task mid-run, then resubmit. Only the missing run should redo.

**Gate:** with one command, Dan can run a whole experiment file, and it can resume.

### Phase 8: Full sweeps, R7 and D6 (branch `hpc/p8-sweeps`; Dan submits each sweep after reviewing its GPU-hour estimate)

Goal: results for each optimization alone, on the full prompt set.

**Commits:**

1. **`exp: set sweep prompt set from gate G2`**
   - Files: `configs/base.yaml` or the 01–04 experiment files.
   - Only the `prompts_file` change Dan approved.
2. **`results: sweep summaries and per-Challenge tables`**
   - Files: `report/sweeps/summary.csv`, `report/sweeps/per_challenge.csv`, `report/figures/sweeps/*`.
   - Includes a baseline rerun on a second node to measure timing noise.
   - Lock the run directories read-only afterwards.
   - Also update PROGRESS.md.

**Gate G3** (team decision): Dan confirms the combined settings.

### Phase 9: Combined, ablation and FID, R8 (branch `hpc/p9-combined`)

Goal: the final runs for the report.

**Commits:**

1. **`exp: set combined settings from gate G3`**
   - Files: `configs/experiments/05_combined.yaml`.
2. **`data: append 3,000 COCO prompts for FID`**
   - Only if Dan approves gate G4.
   - Files: `data/prompts/eval_prompts.jsonl` (new ids appended; existing ids untouched), `data/prompts/README.md`.
3. **`results: combined, leave-one-out and FID summaries`**
   - Files: `report/final/summary.csv`, `report/figures/final/*`.
   - Regenerated with `scripts/summarize.py`.
   - Also update PROGRESS.md.

## Out of scope here

The stats/analysis module (H3), the report sections, and the literature survey happen off-cluster.

## Verification (how Dan checks your work)

- Each phase's gate result, with job IDs, is in `docs/hpc/PROGRESS.md`.
- Every job in that table came from a job request that Dan submitted. No job was submitted, changed or cancelled by Claude.
- `git log --oneline hpc/p<N>-*` shows the commits listed for that phase, each passing `pytest -q`.
- Dan pushes each phase branch and opens one PR per phase. Its commits are the check-ins.
