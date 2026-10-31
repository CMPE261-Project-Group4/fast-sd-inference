# HPC progress log

New here? Start with [README.md](README.md) (overview, plan, current state, next steps) and
[DEVELOPMENT.md](DEVELOPMENT.md) (how to work on it). This file is the detailed log.

The plan is [PLAN.md](PLAN.md), the brief saved as-is. Where this log's
[Corrections to PLAN.md](#corrections-to-planmd-measured-2026-10-18) differ from PLAN.md, follow the corrections.

## Status

| Phase | Branch | Gate | Result |
|---|---|---|---|
| 0 Orient | `hpc/p0-orient` | ssh without a prompt, `a100` GRES, `group4/` writable | **pass** (2026-10-18) |
| 1 Portable environment | `hpc/p1-env` | imports pass on both clusters, clean env | **pass** (2026-10-19, job 92197) |
| 2 Offline assets | `hpc/p2-assets` | everything loads on HPC3 offline | **pass** (2026-10-19, job 92251; job 92223 failed first) |
| 3 First GPU job (G1) | `hpc/p3-gpu` | 4 variants make images on an A100 | **pass** (2026-10-28, job 98056; job 97329 failed first on a broken GPU) |
| 4 Frozen prompts | `hpc/p4-prompts` | counts match the data doc | not started |
| 5 Measurement | `hpc/p5-measure` | summary filled, deterministic | not started |
| 6 Pilot (G2) | `hpc/p6-pilot` | s/image + sweep GPU-hours reported | not started |
| 7 Orchestration | `hpc/p7-orchestration` | one command, resumable | not started |
| 8 Sweeps (G3) | `hpc/p8-sweeps` | Dan confirms combined settings | not started |
| 9 Combined + FID | `hpc/p9-combined` | final summaries | not started |

## Jobs

| Date | Job ID | Purpose | Node | Result |
|---|---|---|---|---|
| 2026-10-19 | 92197 | P1.1 clean-environment import check (`check_env.sbatch`, defq, 2 CPUs) | c19 | COMPLETED 0:0 in 36 s, MaxRSS 651 MB; all checks passed |
| 2026-10-19 | 92223 | P2.1 offline asset check (`check_assets.sbatch --full`, defq, 2 CPUs, 24G) | c19 | FAILED 1:0 in 1m54s, MaxRSS 2.6 GB; 16 of 17 checks passed. CLIP ViT-L/14 failed: torchmetrics CLIPScore doesn't work with transformers 5 (see open issues) |
| 2026-10-19 | 92251 | P2.2 offline asset check, rerun with the CLIPScore adapter | c19 | COMPLETED 0:0 in 1m25s, MaxRSS 2.35 GB (sampled); all 17 checks passed |
| 2026-10-19 | 92300 | P2.3 confirm check jobs use the submitting clone (a4ffae1), `check_env.sbatch` | c19 | COMPLETED 0:0 in 20 s; the log landed in the submitting clone's `logs/`, shows commit 447ac76, and has 40/40 checks ok |
| 2026-10-27 | 97329 | P3.1 GPU smoke check (`check_gpu.sbatch`, gpuqs, 1× A100, 8 CPUs, 32G) | cs002, GPU 0 | FAILED 1:0 in 20 s. Every variant failed at `pipe.to("cuda")` with "CUDA-capable device(s) is/are busy or unavailable" on an idle GPU |
| 2026-10-28 | 98023 | P3.2 GPU access diagnostic (`gpu_diagnose.sbatch`, 1× A100, 2 CPUs, 8G) | cs002, GPU 0 | COMPLETED 0:0 in 19 s. GPU 0 can't allocate in any environment, Slurm's raw one included; GPUs 1–3 can. Cause of 97329: a broken GPU, not our code |
| 2026-10-28 | 98056 | P3.3 GPU smoke check, rerun with `--exclude=cs002` | cs001, GPU 1 | COMPLETED 0:0 in 1m04s, MaxRSS 3.4 GB; all 4 runs pass `check_run.py`; **gate G1 passed** |

## Phase 3: First GPU job, R3 and G1 (2026-10-27 to 2026-10-28)

### Gate G1: pass

Job 98056 on cs001 (GPU 1) ran `scripts/hpc/check_gpu.sbatch` from commit 883514d, with the GPU check files not yet committed (`+uncommitted` in the log). It ran in the prelude's clean environment, offline. SD1.5 loaded at `451f4fe`, and 4 variants each made 2 images from 2 smoke prompts in `data/prompts/manual_prompts.yaml` (seeds 1000 and 1001), through `eldm.benchmark.run`. Then `scripts/check_run.py --require-gpu A100` passed all 4 run directories: 2 images each, none missing, extra or blank; `perf.json` complete; GPU recorded as an A100.

| Variant | s/image | Speedup | Peak GB |
|---|---|---|---|
| baseline: PNDM 50 steps, FP32 | 4.13 | 1.00× | 4.95 |
| `fp16` | 1.46 | 2.84× | 2.61 |
| `dpmpp_20`: DPM-Solver++ 2M, 20 steps | 1.67 | 2.48× | 4.95 |
| `deepcache_i3`: DeepCache, interval 3 | 1.74 | 2.38× | 5.16 |

**These are smoke numbers, not results.** Each is 2 images after the base config's 2 warm-up batches, on a node where another user's job (97996) kept GPU 0 at 100 %. TF32 was at torch 2.6's defaults: off for matmuls, **on for cuDNN convolutions**, so the "FP32" baseline's convolutions ran in TF32. Phase 5 pins the flags; real timings come from Phase 6 onward. Image-to-image spread was small: the latency std over the 2 images was at most 6 ms.

Outputs: `/scratch/cmpe261-fa26/group4/results/_checks/98056/`: one run directory per variant, plus `prompts.jsonl` and `summary.json`.

**The images, looked at as a sanity check** (the checker decides the gate): all 8 are coherent and match their prompts. FP16 is almost identical to the baseline (mean absolute pixel difference 4.2 and 0.7 out of 255). DPM-Solver++ keeps the baseline's composition (15). DeepCache is on-prompt but changes the composition more (21 and 31): its lemon image puts a large bowl of lemons and dark fruit where the baseline has a small bowl and loose lemons. Phase 5's LPIPS against the baseline will measure this properly.

### Commits

- `6ab1ebb fix(hpc): rework run_gpu.slurm for HPC3`. It adds `scripts/hpc/job_prelude.sh` (the clean re-exec, `site.sh`, thread counts, `$STAGE`, the job log) and its test. PLAN.md's commit 1 also listed `logs/.gitkeep` and the `.gitignore` change, which `a4ffae1` already made in Phase 2.
- `8a7a383 test: add run-directory checker`: `scripts/check_run.py`.
- `518cd87 fix(hpc): count untracked files as uncommitted in the job log`, an extra commit.
- `883514d chore(hpc): add GPU access diagnostic job`, an extra commit, after job 97329: `scripts/hpc/gpu_diagnose.sbatch`, plus a troubleshooting entry.
- `df41188 fix(hpc): keep SLURMD_NODENAME in the job's clean environment`, an extra commit, after job 98056.
- `test(hpc): add GPU smoke check for four variants` (this commit): `scripts/hpc/gpu_check.py`, `scripts/hpc/check_gpu.sbatch`, their test, and these docs.

### Findings

- **GPU 0 on cs002 is broken.** Job 97329 failed there, and diagnostic job 98023 showed why:
  - GPU 0 (PCI `00000000:17:00.0`, UUID `GPU-caaea0d3-c49b-d599-41f0-f8448e269d0b`) fails every CUDA allocation with "CUDA-capable device(s) is/are busy or unavailable": under Slurm's raw environment, under the prelude's clean one, and with or without `CUDA_DEVICE_ORDER`. It's idle, at 0 MiB, in Default compute mode. GPUs 1–3 on the same node allocate fine.
  - Slurm hands out the lowest free index, and only Dan's jobs have run on cs002 since 2026-10-20. So every 1-GPU job there gets GPU 0. Dan's `llm-ping` job 93353 also failed on cs002, on 2026-10-21; its cause wasn't checked.
  - Until the admins fix it, submit GPU jobs with `--exclude=cs002`. Dan has a drafted report for the admins.
- **GPUs really aren't isolated** (correction 2, confirmed). `cgroup.conf` has `ConstrainDevices=yes`, but job 98023 opened all four `/dev/nvidia*` files and allocated on GPUs it hadn't been given (all idle at the time, with no other jobs on the node). `CUDA_VISIBLE_DEVICES` is the only thing that keeps a job on its own GPU, which is why the prelude keeps it. The committed diagnostic only allocates on its own GPU.
- **The prelude dropped `SLURMD_NODENAME`**, which doesn't match `SLURM_*`. So job 98056 didn't log its co-tenants; `squeue` afterwards showed job 97996 on cs001. Fixed in `df41188`. The GPU utilization line in the log did show GPU 0 at 100 %.
- **diffusers 0.41 tries the Hub first.** With a pinned revision and `HF_HUB_OFFLINE=1`, it prints "Couldn't connect to the Hub … offline mode is enabled" and then "Will try to load from local cache", and it loads. It's harmless; DEVELOPMENT.md now lists it.
- **DeepCache 0.1.1 works with diffusers 0.41** on the GPU, as Phase 3's plan expected. No version change was needed.
- **Not recorded yet:** the attention processor. Phase 2 found `AttnProcessor2_0` (SDPA) on CPU; Phase 5 adds it to `perf.json`.
- **`run_gpu.slurm` hasn't run.** It needs Phase 4's prompt files. Its first job is in Phase 5.

### Decisions

- **Phase 3 branched from `main`,** since PRs #1–#4 are merged, so its PR shows only Phase 3.
- **One shared job prelude,** `scripts/hpc/job_prelude.sh`, for every GPU job: `run_gpu.slurm` now, the job arrays in Phase 7. [DEVELOPMENT.md](DEVELOPMENT.md#writing-a-job-script) explains each step.
- **The gate check is decided by `check_run.py`,** not by looking at the images.

## Phase 2: Offline assets, R2 and part of D2 (2026-10-19)

### Gate: pass

Job 92251 on c19 ran `scripts/hpc/check_assets.py --full`. It ran in a clean environment (`env -i` plus `env/site.sh`), on 2 threads, with socket connections blocked and the offline flags set. Every model, metric and dataset file loaded:

| Check | Result |
|---|---|
| Hash pins | Every pinned file was re-hashed and matches: SD1.5 13 files, CLIP-L 8 files, ViT-g 2 files, 11.45 GB in total. `data/CHECKSUMS.txt` matches too. |
| Datasets | COCO has 5,000 images and 25,014 captions, PartiPrompts has 1,632 prompts, and both are read-only. 5 JPEGs decode. |
| SD1.5 | It loads through `eldm.pipeline.build_pipeline` at `451f4fe` on CPU. All 5 schedulers construct. The text encoder, UNet and VAE forward passes are finite and correctly shaped, and DeepCache attaches. |
| CLIP ViT-L/14 | Through the adapter, CLIPScore gives 20.45 with an image's own caption and 7.66 with a mismatched one. The by-hand value is also 20.45. |
| OpenCLIP ViT-g/14 | Loaded from the pinned safetensors, its cosine score is 17.8 with the own caption and −6.0 with a mismatched one. |
| FID, KID, LPIPS | 67.5, 0.0331 and 0.116 on fixed tiny inputs, the same values as on coe-hpc1. |

The same inputs gave the same numbers on coe-hpc1 and on HPC3, so both clusters see the same assets and the same environment.

Job 92223, the first attempt, failed only on CLIP-L, because torchmetrics' CLIPScore doesn't work with transformers 5. The fix is an adapter in the check (Dan's decision, 2026-10-19); see the open issues.

### Commits

- `155bfd4 feat(assets): pin external assets in one manifest`, an extra commit: `src/eldm/assets.py` is the single source of the pins.
- `8e94224 feat(hpc): prefetch model and metric weights for offline jobs`
- `4c7e66c feat(pipeline): load SD1.5 at the pinned revision`, an extra commit: without it, Phase 3 can't load offline.
- `dca6d18 data: add dataset fetch script and checksums`
- `9619f8a test(hpc): add offline asset load check job`. It adds `scripts/hpc/check_assets.py`, the payload, which PLAN.md doesn't list, and `scripts/hpc/check_assets.sbatch`.

After the gate, two commits make the work usable by teammates:
- `a4ffae1 fix(hpc): run check jobs from the submitting clone`. It also commits `logs/.gitkeep` and the `.gitignore` change, which PLAN.md had placed in Phase 3's first commit; Phase 3 keeps the `run_gpu.slurm` rework.
  - Tested on coe-hpc1: under `bash`, as a simulated Slurm spool copy, and from a fresh separate clone. That clone's own `src/` was used with the shared venv.
  - Confirmed under real Slurm by job 92300. Slurm's working directory was the submitting clone, and the log landed in that clone's `logs/` through the relative `--output`.
- `docs(hpc): add HPC overview and development guide`: [README.md](README.md) and [DEVELOPMENT.md](DEVELOPMENT.md), plus pointers from the root README.

Two commit messages state the wrong test count, because I wrote them before reading pytest's output. History isn't rewritten:
- `155bfd4` says "15 passed"; the actual count was 18.
- `8e94224` says "21 passed"; the actual count was 22.

### Assets (all under `/data/cmpe261-fa26/group4`, verified on coe-hpc1 2026-10-19)

| Asset | Pinned at | Files | Size | Verified against |
|---|---|---|---|---|
| SD1.5 `stable-diffusion-v1-5/stable-diffusion-v1-5` | `451f4fe` | 13 (safetensors + configs) | 4.27 GB | the Hub's LFS sha256 or git blob id, and per-file sha256 pins in `eldm.assets` |
| CLIP ViT-L/14 `openai/clip-vit-large-patch14` | `32bd642` | 8 | 1.71 GB | the same |
| OpenCLIP ViT-g/14 `laion/CLIP-ViT-g-14-laion2B-s34B-b88K` | `a20bc45` | 2 | 5.47 GB | the same |
| Inception (FID/KID) `weights-inception-2015-12-05-6726825d.pth` | torch-fidelity v0.2.0 URL | 1 | 0.10 GB | sha256 `6726825d…b2`, matching the filename prefix |
| AlexNet (LPIPS) `alexnet-owt-7be5be79.pth` | download.pytorch.org URL | 1 | 0.24 GB | sha256 `7be5be79…02`, matching the filename prefix |
| COCO 2017 `val2017.zip`, images extracted | images.cocodataset.org (HTTP only) | 5,000 images | 0.82 GB zip | S3 multipart ETag `d366be60…-98`, plus `data/CHECKSUMS.txt` |
| COCO `annotations_trainval2017.zip`, `captions_val2017.json` extracted | the same | 25,014 captions | 0.25 GB zip | ETag (MD5) `f4bbac64…`, plus `data/CHECKSUMS.txt` |
| PartiPrompts `PartiPrompts.tsv` | google-research/parti `5a65797` | 1,632 prompts | 0.1 MB | git blob `f96e7125…`, plus `data/CHECKSUMS.txt` |

Disk use: `hf_cache` 11 GB, `cache/torch` 0.3 GB, `datasets` 1.8 GB (read-only).

How long each script takes on coe-hpc1:

| Script | First run | Rerun |
|---|---|---|
| `prefetch_assets.py` | about 3 min (download) | 56 s (verify only) |
| `prefetch_assets.py --verify-only` | | 53 s |
| `fetch_datasets.sh` | 1m39s | 14 s |
| `check_assets.py --quick` | | 11 s |

### Findings

- **Hugging Face cache blob names aren't sha256.** Xet-backed repos name their blobs by Xet hash; for example, SD1.5's `text_encoder/model.safetensors` is in blob `4e7380d2…`, while its sha256 is `d008943c…`. So files are verified against the Hub's published LFS sha256 and git blob ids, and pinned per file in `eldm.assets`. The cache itself isn't trusted.
- **hf_xet's leftover cache** in `$HF_HOME/xet` is empty (0.00 GB). Deleting it while the downloading process is still running fails on NFS, because that process holds its log files open. The prefetch script now only reports its size.
- **COCO serves no HTTPS.** Integrity comes from the pinned ETags, including the multipart ETag reproduced from 8 MiB parts.
- **The `-c 2` thread fix works.** Job 92223 ran torch on 2 threads, because the sbatch sets `OMP_NUM_THREADS` from `SLURM_CPUS_PER_TASK`.
- **SD1.5 loads offline on HPC3 at the pinned revision.**
  - The attention processor is `AttnProcessor2_0` (SDPA).
  - diffusers warns that `torch_dtype` is deprecated in favour of `dtype` (removal in 1.0). That's for Phase 5's `pipeline.py` changes.
- **Memory.** The check job requests `--mem=24G`. sacct reported MaxRSS of 2.6 GB (job 92223) and 2.35 GB (job 92251). sacct samples memory periodically, so it can miss the short peak while open_clip loads ViT-g (model plus state dict, about 11 GB by size). For a 2-minute CPU job the 24G headroom is kept.

## Phase 1: Portable environment, R1 (2026-10-19)

### Gate: pass

The same check, `scripts/hpc/check_env.sbatch`, ran in a clean environment (`env -i`, no modules, only `env/site.sh`) on both clusters:
- **coe-hpc1:** `bash scripts/hpc/check_env.sbatch` passed all checks (CentOS 7.9, glibc 2.17, 48 s).
- **COE-HPC3:** job 92197 on c19 passed all checks (Rocky 9.8, glibc 2.34, 36 s). It ran at commit 996f2da plus this commit's `check_env.sbatch`.

The checks:
- the venv's interpreter is the portable CPython 3.11.16;
- `pip check` is clean, and `pip freeze` equals `requirements.lock`;
- torch, torchvision, diffusers, transformers, accelerate, huggingface_hub, safetensors, DeepCache, torchmetrics, torch_fidelity, lpips, open_clip, numpy, scipy, PIL, pandas, matplotlib and yaml import;
- the project's own classes import: the SD pipeline, the four schedulers, `DeepCacheSDHelper`, CLIPScore, FID, KID, LPIPS, open_clip, and `eldm.*`;
- a CPU matmul runs;
- the offline flags are set.

### Commits

- `922065f build(env): pin portable Python and Python dependencies`
- `3934e0b build(hpc): add idempotent setup_env.sh and env/site.sh`
- `996f2da fix(hpc): reset PYTHONPATH and PYTHONHOME in env/site.sh`, an extra commit (see the decisions below)
- `test(hpc): add clean-environment import check job`: this commit

### What was built

- `/data/cmpe261-fa26/group4/env/python`: CPython 3.11.16 (python-build-standalone 20260924, sha256-checked), 112 MB, needs only glibc 2.17.
- `/data/cmpe261-fa26/group4/env/venv`: 83 packages from `requirements.lock`, 4.5 GB. The pip cache in `cache/pip` takes 2.9 GB.
- `setup_env.sh` built both in 5m42s under `env -i`. A second run took 0.03 s and changed 0 files.
- The lock check (`pip install --only-binary=:all: -r requirements.lock` into a fresh venv) took 3 min on coe-hpc1, and its freeze equals the lock.

### Decisions

- **Torch comes from PyPI** (correction 9). It's the same CUDA 12.4 build as the cu124 index, so the lock has no `+cu124` local versions and no extra index. The manylinux_2_28 fallback in PLAN.md wasn't needed: every wheel installs on glibc 2.17.
- **We keep the portable CPython.** Dan asked for a fact check, and no module Python works on both clusters (measured 2026-10-19):
  - **coe-hpc1 module `python3/3.11.5`** is on coe-hpc1's local `/opt/ohpc/pub`. On HPC3 that path is a different NFS tree, and the directory is `Permission denied`.
  - **HPC3 module `python3/3.11.7`** actually runs the system Python 3.11.13 (`/lib64/libpython3.11.so`). It also puts a shared site-packages on `PYTHONPATH` (accelerate, aiohttp, albumentations, …).
  - **HPC3 module `python3/3.12.12`** and the system `/usr/bin/python3.11` (RPM 3.11.13-10.el9_8) don't exist on coe-hpc1 and need glibc 2.34.
  - **Building on HPC3** would mean staging wheels for its platform, building the venv offline in a job, using `module load` in jobs, and keeping a second venv on coe-hpc1 for pytest and downloads.
- **`site.sh` now sets `PYTHONPATH` instead of appending to it, and unsets `PYTHONHOME`** (996f2da). sbatch exports the submitting shell's environment. After a `module load python3/3.11.7` on g17, that module's packages would shadow the venv in every job. Tested: with a fake `accelerate` on the inherited `PYTHONPATH`, the old `site.sh` imported the fake, and the new one imports 1.15.0 from the venv. g17 login shells load no modules today.

## Phase 0: Orient (2026-10-18)

### Gate: pass

- `ssh -o BatchMode=yes coe-hpc3 hostname` prints `g17.hpc.coe` without a prompt.
- `sinfo` on g17 lists `gpu:a100:4` on `cs[001-004]` in gpuqs, gpuqm and gpuql.
- `group4/` is owned by `011383814:cmpe261-fa26` with mode `drwxrwsr-x`, so it's writable. The same holds for `datasets/`, `env/`, `hf_cache/` and `/scratch/.../group4/results/`.

### Measurements

All of these are read-only: commands on coe-hpc1, `ssh coe-hpc3` queries (`sinfo`, `scontrol show partition|node|config`, `sacctmgr show qos`), and the log of AFAR_HPC GPU job 90582 (cs002, 2026-10-15). No job was submitted.

| Item | Brief | Measured |
|---|---|---|
| Groups | `cmpe261-fa26` listed | `users, cmpe295-bond, coe, cmpe261-fa26` |
| coe-hpc1 | CentOS 7, glibc 2.17, internet | CentOS 7.9.2009, glibc 2.17, kernel 3.10; huggingface.co, download.pytorch.org and images.cocodataset.org all return 200 |
| g17 | Rocky 9, glibc 2.34, offline | Rocky 9.8, glibc 2.34; curl to huggingface.co times out |
| Compute nodes | offline | cs002: curl to huggingface.co times out (job 90582) |
| A100 nodes | A100-PCIE-40GB on cs001–cs004 | `Gres=gpu:a100:4` each, 64 CPUs, 250 GB RAM; A100-PCIE-40GB, CC 8.0, driver 590.44.01 (CUDA ≤ 13.1) |
| gpuqs | 2-day limit, at most 2 nodes per user | `MaxTime=2-01:00:00`, QOS `gpu3nodes` with `MaxTRESPU node=2`, `MaxCPUsPerNode=56`, `OverSubscribe=FORCE:4`, `DefMemPerCPU=1024` |
| defq | used for CPU check jobs | `MaxTime=06:00:00`, `MaxCPUsPerNode=2`, `DefMemPerCPU=1024` |
| Slurm (HPC3) | | 23.11.10; `MaxArraySize=1001`; `TmpFS=/tmp`; `JobContainerType=(null)`; no prolog |
| Disk | /home 99% full | /home 99% (2.1 T free); /data 95% (5.2 T free); /scratch 89% (13 T free) |
| git | Spack 2.39.1; system 1.8.3 | `module load git-2.39.1-gcc-12.2.0-b274tuh`; system git 1.8.3.1 |
| AFAR_HPC | reuse if present | `~/AFAR_HPC`; `env/python.pin` = CPython 3.11.16, PBS release 20260924, sha256 `49a52eb189878431a36efd137e7cd08f6429e59dc049b9cfb030bf706eb33fd4` |
| Repo | scaffold | `main` @ `c1ceb72` = `origin/main`; layout matches PLAN.md |
| PartiPrompts | 1,632; Quantity 90; Writing & Symbols 91 | 1,632 rows; columns Prompt, Category, Challenge, Note; 11 Challenges, 12 Categories; Quantity 90, Writing & Symbols 91 |
| COCO 2017 | val2017 + annotations | `val2017.zip` 815,585,330 B; `annotations_trainval2017.zip` 252,907,541 B |

### Changes made in Phase 0

- Set the repo-local git identity: `Dan Lam <dan.lam@sjsu.edu>`. No global config exists or was added.
- Set `core.sharedRepository=group` and ran `chmod -R g+w` on the clone and `group4/011383814/`. They had been created with umask 022.

## Corrections to PLAN.md (measured 2026-10-18)

1. **`$TMPDIR` is not per-job** (Phases 3, 7).
   - Slurm has `TmpFS=/tmp`, `JobContainerType=(null)` and `TmpDisk=0`.
   - In jobs, `TMPDIR=/tmp`. That's a 371 GB local SSD (seen on cs002) that every job on the node shares, and it isn't cleaned per job.
   - Stage to `${TMPDIR:-/tmp}/eldm-${SLURM_JOB_ID}` (add the array task ID for arrays), and remove it with `trap … EXIT`.
2. **GPUs aren't isolated, and CPUs are oversubscribed** (Phases 3, 5, 8).
   - In job 90582, `nvidia-smi` showed all 4 A100s while `CUDA_VISIBLE_DEVICES=0`, so the device cgroup isn't enforced.
   - gpuqs has `OverSubscribe=FORCE:4`, and other users' jobs share the cs nodes.
   - So: log `CUDA_VISIBLE_DEVICES`, use `nvidia-smi -i "$CUDA_VISIBLE_DEVICES"`, and record co-tenant jobs at job start. Keep the second-node baseline rerun.
   - Confirmed by job 98023 (2026-10-28): `cgroup.conf` has `ConstrainDevices=yes`, yet a job can open and allocate on GPUs it wasn't given. See Phase 3's findings.
3. **A plain `gpu:1` can land on a P100 or an H100** (g16, g18 and g19 are in gpuqs). Always use `--gres=gpu:a100:1`.
4. **Path spelling.** On coe-hpc1, `/data` is a symlink to `/fs/atipa/data`. On HPC3, `/data` is the NFS mount, and `/fs/atipa/data` is a symlink to `/data`. Both spellings work on both clusters. Scripts and the venv use the `/data/...` spelling.
5. **Missing directories.** `group4/cache/` and `datasets/{coco,partiprompts}` don't exist yet. `setup_env.sh` and `fetch_datasets.sh` create them.
6. **umask.** Interactive shells here default to umask 0022. Every command that creates files must run `umask 002` first, or source `env/site.sh`.
7. **Git.** No global identity exists, so commits use the repo-local config. SSH to GitHub works, so only the rule stops a push.
8. **coe-hpc1 has its own Slurm 18.08 client** (`/usr/bin/sbatch`, `sinfo`). It doesn't see HPC3. All queries go through `ssh coe-hpc3`.
9. **Phase 1 pins.**
   - I dry-ran `pip install --dry-run --only-binary=:all:` with CPython 3.11.16 on coe-hpc1, and the whole stack resolves on glibc 2.17. The manylinux_2_28 fallback isn't needed.
   - The wheels are all `manylinux2014` or older: torch 2.6.0 and its nvidia-*-cu12 12.4.x dependencies, and triton 3.2.0. torch 2.7.0 is `manylinux_2_28` only, which confirms the brief.
   - Unpinned, pip resolves: diffusers 0.41.0, transformers 5.20.1, huggingface_hub 1.33.0, DeepCache 0.1.1, torchmetrics 1.9.0, torch-fidelity 0.4.0, lpips 0.1.4, open_clip_torch 3.3.0, timm 1.0.30, numpy 2.2.6.
   - **Pin exact versions in `requirements.in`.** diffusers 0.41 and transformers 5.20 already require `torch>=2.6`. A later release could require torch 2.7, which can't install on coe-hpc1.
   - **Torch source.** The PyPI `torch==2.6.0` wheel is the CUDA 12.4 build, so the lock needs no `+cu124` local version and no extra index. If the cu124 index is used instead, add `--extra-index-url` to the lock. Use `pip freeze --exclude-editable`.
   - **DeepCache 0.1.1** (Dec 2023): `import DeepCache` loads only `DeepCacheSDHelper`, which monkeypatches `forward` methods and imports nothing from diffusers. It still has to work at run time (Phase 3).
   - **transformers 5.x is a new major version.** Phase 2's CPU load job checks torchmetrics' CLIPScore with it. The fallback is `transformers<5`.
10. **Phase 2 assets.** Pin each by revision, and download files, not whole folders.
    - **SD1.5:** `stable-diffusion-v1-5/stable-diffusion-v1-5` @ `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, not gated.
      - The component folders total about 21 GB, because each one holds `.bin`, `.fp16.*` and `non_ema` copies.
      - Allow only these: `model_index.json`, `*/config.json`, `scheduler/*`, `tokenizer/*`, `feature_extractor/*`, `text_encoder/model.safetensors`, `unet/diffusion_pytorch_model.safetensors`, `vae/diffusion_pytorch_model.safetensors`. That's about 4.3 GB.
    - **CLIP ViT-L/14:** `openai/clip-vit-large-patch14` @ `32bd64288804d66eefd0ccbe215aa642df71cc41`. Get `model.safetensors` plus the json/txt files: 1.7 GB instead of 6.8 GB.
    - **OpenCLIP ViT-g/14:** `laion/CLIP-ViT-g-14-laion2B-s34B-b88K` @ `a20bc450485fc2dd335317234582df0e69d21b5a`. Get `open_clip_model.safetensors` plus the configs (5.5 GB), and skip the `.bin` twin.
    - **Offline lookup.** A revision-pinned download doesn't write `refs/main`, so an offline `from_pretrained(repo)` with no revision fails.
      - diffusers and transformers: pass `revision=`.
      - torchmetrics' CLIPScore and open_clip take no revision argument, so load them from local snapshot paths.
    - **PartiPrompts:** use the raw URL at `google-research/parti` commit `5a657978134374ce28973948331b319adef164bd`, not `main`.
    - **Caches off /home.** `~/.cache` already holds 3 GB (pip, huggingface) on /home. Set `PIP_CACHE_DIR`, `HF_HOME`, `TORCH_HOME` and `XDG_CACHE_HOME` before any pip or download.
11. **Small script fixes.**
    - `.gitignore` already ignores `logs/`. Replace that line with `logs/*` and `!logs/.gitkeep`.
    - ~~`#SBATCH --output` can't expand variables, so hard-code `/data/cmpe261-fa26/group4/011383814/fast-sd-inference/logs/%x-%j.out`.~~ Superseded by a4ffae1. Hard-coding Dan's clone sent teammates' jobs to his code and logs. Job scripts now use `--output=logs/%x-%j.out`, relative to the submit directory, and find their clone from `$SLURM_SUBMIT_DIR`. `logs/.gitkeep` is committed, so every clone has `logs/`.
    - `env/python` and `env/venv` in Phase 1 mean `/data/cmpe261-fa26/group4/env/{python,venv}`, not repo paths.
    - defq defaults to 1 GB per CPU, with at most 2 CPUs. `check_assets.sbatch` needs an explicit `--mem`, about 24G.

## Decisions and versions

- Git identity: `Dan Lam <dan.lam@sjsu.edu>`, repo-local.
- PLAN.md is kept verbatim; corrections live in this log.
- **Environment** (Phase 1, `requirements.lock`):
  - CPython 3.11.16 (PBS 20260924);
  - torch 2.6.0+cu124 (CUDA 12.4, cuDNN 9.1.0), torchvision 0.21.0;
  - diffusers 0.41.0, transformers 5.20.1, accelerate 1.15.0, huggingface_hub 1.33.0;
  - DeepCache 0.1.1;
  - torchmetrics 1.9.0, torch-fidelity 0.4.0, lpips 0.1.4, open_clip_torch 3.3.0;
  - numpy 2.2.6.
- **Portable CPython over module Pythons** (see Phase 1 decisions).
- **Assets** (Phase 2): every model and metric weight is pinned in `src/eldm/assets.py`: repo, commit, allow-list and per-file sha256. Dataset checksums are in `data/CHECKSUMS.txt`.
  - Loaders must use the pinned snapshot, because no `refs/main` is cached:
    - diffusers takes `revision=model_revision`;
    - CLIP-L uses a callable loader with the transformers 5 adapter;
    - open_clip takes the local safetensors path.
  - Scripts: `prefetch_assets.py` and `fetch_datasets.sh` run on coe-hpc1; `check_assets.sbatch` is the HPC3 gate.
- **Versions are unchanged** despite the CLIPScore incompatibility (Dan, 2026-10-19). An adapter is used instead of `transformers<5`.
- **A100 driver:** 590.44.01 (CUDA ≤ 13.1), which supports the CUDA 12.4 wheels.

## Open issues

- **The whole class is in the group.** `cmpe261-fa26` has 19 members, so every classmate can write to and delete from `group4/`. There's no team-only group. Locking finished data with `chmod -R a-w` is the only guard. Dan could ask the admins for a team group.
- **GPU 0 on cs002 is broken (found 2026-10-28, job 98023).** Submit GPU jobs with `--exclude=cs002` until the admins fix it; Dan reports it to them. Details are in Phase 3's findings.
- **CPU threads aren't limited to the allocation. Handled since Phase 3.** In job 92197 (`-c 2`), torch picked 28 threads, all of c19's physical cores. This fits correction 2: cgroups don't confine the job. `job_prelude.sh` sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` from `SLURM_CPUS_PER_TASK`.
- **torchmetrics' CLIPScore doesn't work with transformers 5 (found by job 92223; needs Dan's OK in Phase 5).**
  - **The cause:** in transformers 5.20.1, `CLIPModel.get_image_features` and `get_text_features` return a `BaseModelOutputWithPooling`, with the projected embedding in `.pooler_output`. transformers 4.x returned the tensor itself. torchmetrics 1.9.0 calls `.norm()` on the result and fails with `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'`.
  - **No fix exists upstream.** 1.9.0 (2026-03-09) is the latest torchmetrics release.
  - **Downgrading is costly.** The last transformers 4.x release (4.57.6) needs `huggingface_hub<1.0`, but diffusers 0.41 needs `>=1.32`. So `transformers<5` would also downgrade diffusers, mean rebuilding the venv, and redo the Phase 1 gate.
  - **Decision (Dan, 2026-10-19): keep the package versions and use an adapter.**
    - The adapter is a `CLIPModel` subclass whose `get_image_features` and `get_text_features` return `.pooler_output`. It's passed to CLIPScore through its callable loader, with the model loaded from the pinned snapshot.
    - The CLIP-L check in `check_assets.py` uses it. On coe-hpc1, offline, it gives 20.45 with an image's own caption versus 7.66 with another caption.
    - A score computed by hand, `100 * max(cos, 0)`, also gives 20.45, so the metric's definition doesn't change.
  - **Proposed for Phase 5:** use the same adapter in `src/eldm/metrics.py`, or compute the per-image score by hand, which `quality_per_image.csv` needs anyway. Dan confirms which in Phase 5.
- **coe-hpc1 has an old NVIDIA driver (CUDA 10.1).** torch prints a harmless warning there when it probes CUDA. We never use a GPU on coe-hpc1.
- **NFE with DeepCache (Phase 5, needs Dan's decision).** DeepCache's cached steps still call `unet.forward` and skip only the inner blocks. A UNet forward hook would therefore count the same NFE as the baseline. Proposal: count full and partial passes separately, for example with a hook on `unet.mid_block`.
