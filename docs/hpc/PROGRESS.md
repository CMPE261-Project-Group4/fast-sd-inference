# HPC progress log

The plan is [PLAN.md](PLAN.md), the brief saved as-is. Where this log's
[Corrections to PLAN.md](#corrections-to-planmd-measured-2026-10-18) differ from PLAN.md, follow the corrections.

## Status

| Phase | Branch | Gate | Result |
|---|---|---|---|
| 0 Orient | `hpc/p0-orient` | ssh without a prompt, `a100` GRES, `group4/` writable | **pass** (2026-10-18) |
| 1 Portable environment | `hpc/p1-env` | imports pass on both clusters, clean env | **pass** (2026-10-19, job 92197) |
| 2 Offline assets | `hpc/p2-assets` | everything loads on HPC3 offline | not started |
| 3 First GPU job (G1) | `hpc/p3-gpu` | 4 variants make images on an A100 | not started |
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
    - `#SBATCH --output` can't expand variables, so hard-code `/data/cmpe261-fa26/group4/011383814/fast-sd-inference/logs/%x-%j.out`.
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
- **A100 driver:** 590.44.01 (CUDA ≤ 13.1), which supports the CUDA 12.4 wheels.

## Open issues

- **The whole class is in the group.** `cmpe261-fa26` has 19 members, so every classmate can write to and delete from `group4/`. There's no team-only group. Locking finished data with `chmod -R a-w` is the only guard. Dan could ask the admins for a team group.
- **CPU threads aren't limited to the allocation (Phase 3 onward).** In job 92197 (`-c 2`), torch picked 28 threads, all of c19's physical cores. This fits correction 2: cgroups don't confine the job. Job scripts should set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` from `SLURM_CPUS_PER_TASK`, so we don't oversubscribe a shared node and add noise to the timings.
- **transformers 5.x with torchmetrics' CLIPScore.** It imports fine; whether it actually runs is checked in Phase 2 (fallback: `transformers<5`).
- **coe-hpc1 has an old NVIDIA driver (CUDA 10.1).** torch prints a harmless warning there when it probes CUDA. We never use a GPU on coe-hpc1.
- **NFE with DeepCache (Phase 5, needs Dan's decision).** DeepCache's cached steps still call `unet.forward` and skip only the inner blocks. A UNet forward hook would therefore count the same NFE as the baseline. Proposal: count full and partial passes separately, for example with a hook on `unet.mid_block`.
