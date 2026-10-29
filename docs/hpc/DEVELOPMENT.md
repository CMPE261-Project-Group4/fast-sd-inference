# Developing on the HPC

How to work on fast-sd-inference on the SJSU CoE HPC. For the big picture (the two clusters, the plan, the current state) read [README.md](README.md) first.

## 1. First-time setup (once per person)

1. **Log in** from the campus network or VPN: `ssh <SJSU-ID>@coe-hpc1.sjsu.edu`. Check that `id` lists `cmpe261-fa26`; if it doesn't, ask the instructor.
2. **Reach HPC3 without a password.** `ssh coe-hpc3 hostname` must print `g17.hpc.coe`. If it asks for a password, run this once on coe-hpc1 (`/home` is shared, so the key works on both):
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -q -N "" && cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
   ```
3. **Use a recent git.** The system git (1.8.3) is too old:
   ```bash
   module load git-2.39.1-gcc-12.2.0-b274tuh
   ```
   To push, add `~/.ssh/id_ed25519.pub` to your GitHub account.
4. **Clone into your own folder on `/data`.** Not `/home`, which is 99% full:
   ```bash
   umask 002
   mkdir -p /data/cmpe261-fa26/group4/$USER && cd /data/cmpe261-fa26/group4/$USER
   git clone git@github.com:CMPE261-Project-Group4/fast-sd-inference.git && cd fast-sd-inference
   git switch hpc/p2-assets          # until the phase PRs are merged into main
   git config user.name "Your Name" && git config user.email "you@sjsu.edu"
   ```
5. **Check that your clone works with the shared setup.** Each check takes seconds:
   ```bash
   source env/site.sh
   python -m pytest -q                          # 23 passed
   bash scripts/hpc/check_env.sbatch            # "All checks passed."
   python scripts/hpc/check_assets.py --quick   # "All checks passed."
   ```

You do **not** run `setup_env.sh`, `prefetch_assets.py` or `fetch_datasets.sh`. The environment, models and datasets already exist, and they're shared. Rebuilding them affects everyone (see §4).

## 2. The daily loop

Start every shell with:
```bash
cd /data/cmpe261-fa26/group4/$USER/fast-sd-inference && source env/site.sh
```
This puts the shared venv's `python` first on your PATH, makes your clone's `src/` importable, sets `umask 002`, keeps caches off `/home` and turns on offline mode.

The loop, the same for every change:
1. Edit on coe-hpc1, then run `python -m pytest -q`.
2. Write or update the job script, then read it back: paths, resources, `--time`.
3. Run a **tiny job** first: 2–4 prompts, `--time=00:20:00`.
4. Follow it with `squeue` and `sacct` at sensible intervals, not every few seconds, and read its log.
5. Check the outputs with a checker script, not by eye. `scripts/check_run.py` arrives in Phase 3.
6. If it failed, read the log, fix the script, and run a new tiny job.
7. Run the full-size job only after the tiny one passes. Then commit.

On coe-hpc1 and g17, run only editing, git, pip, downloads, unit tests and CPU checks of a few seconds. Anything on a GPU, and anything long, goes in a job.

## 3. Running jobs

Submit from **g17**, from the **root of your clone**. Jobs find the clone they were submitted from, and write their log to `logs/<job-name>-<jobid>.out` in it, which you can also read from coe-hpc1:
```bash
ssh coe-hpc3
cd /data/cmpe261-fa26/group4/$USER/fast-sd-inference
sbatch --test-only scripts/hpc/check_env.sbatch   # validates the request only; it does NOT submit
sbatch scripts/hpc/check_env.sbatch               # prints "Submitted batch job <id>"
squeue -u $USER
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

The jobs we have:

| Script | What it checks | Resources | Takes |
|---|---|---|---|
| `scripts/hpc/check_env.sbatch` | the shared venv imports in a clean environment | defq, 2 CPUs, 4G, 10 min | ~40 s |
| `scripts/hpc/check_assets.sbatch` | every model, metric weight and dataset loads with the network blocked | defq, 2 CPUs, 24G, 30 min | ~1.5 min |

Experiments run with `scripts/hpc/run_gpu.slurm <experiment.yaml> [args]`: one A100, 20 minutes by default. For a full run, pass `sbatch --time=…`.

Limits to know (measured; re-check with `sinfo` and `scontrol show partition <name>`):

| Partition | Time limit | Nodes | Notes |
|---|---|---|---|
| `gpuqs` | 2 days | cs001–cs004 (4× A100 each), plus P100 and H100 nodes | QOS `gpu3nodes`: at most **2 nodes per user** |
| `defq` | 6 h | CPU nodes | at most **2 CPUs per node**; 1 GB per CPU unless you pass `--mem` |

Record every job that backs a gate or a result in the jobs table in [PROGRESS.md](PROGRESS.md).

### Writing a job script

Start from this template. `scripts/hpc/run_gpu.slurm` is a complete example.

```bash
#!/bin/bash
# <What it does>. Submit from the repo root on g17:  sbatch scripts/hpc/<name>.sbatch
#SBATCH --job-name=eldm-<name>
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1          # never a plain gpu:1: it can land on a P100 or an H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00            # no more than the job needs
#SBATCH --output=logs/%x-%j.out    # relative to the submit directory, i.e. your clone

set -euo pipefail
CLONE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
source "${CLONE}/scripts/hpc/job_prelude.sh"   # clean env, site.sh, threads, $STAGE, log

python "${CLONE}/scripts/<script>.py" --stage "${STAGE}" ...
```

`scripts/hpc/job_prelude.sh` handles everything this cluster gets wrong, so a job script doesn't have to:

| What the prelude does | Why |
|---|---|
| Re-runs the job script under `env -i`, keeping only `SLURM_*`, `SLURMD_NODENAME`, `CUDA_VISIBLE_DEVICES` and `TMPDIR` | `sbatch` copies your g17 shell into the job, so a loaded module's `PYTHONPATH` or `LD_LIBRARY_PATH` would leak in. `CUDA_VISIBLE_DEVICES` is kept because nothing else stops the job from using another job's GPU: all 4 A100s are visible. |
| Sources `env/site.sh` | paths, caches, offline flags and the venv; never `module load`, since modules differ between the clusters |
| Sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` | CUDA then numbers GPUs the way `nvidia-smi` and Slurm do |
| Sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` from `SLURM_CPUS_PER_TASK` | Slurm doesn't confine CPUs: torch used 28 threads under `-c 2` in job 92197 |
| Creates `$STAGE` = `$TMPDIR/eldm-<jobid>` and removes it on exit | the node-local SSD at `/tmp` is shared by every job on the node and isn't cleaned per job. So don't set your own `trap … EXIT`. |
| Logs the node, commit (`+uncommitted` if your tree is dirty), the GPU, the load on the node's other GPUs, and the jobs sharing the node | timings on shared nodes need this context |

The clone comes from `$SLURM_SUBMIT_DIR` because Slurm runs a copy of the script from its spool directory, so the script's own path isn't your clone.

## 4. Changing the shared environment, models or datasets

These are shared by the whole team **and by any job that is running**. Before changing them:
- tell the team;
- check that nobody's job is using them: `squeue -u 011383814,013483639,018321927`.

### Add or change a Python package

1. Edit `requirements.in`. Use exact pins. **Keep `torch==2.6.0`:** torch 2.7+ only ships wheels for glibc ≥ 2.28 and won't install on coe-hpc1.
2. Run `scripts/hpc/setup_env.sh --lock`. It re-resolves `requirements.in` into `requirements.lock` in a throwaway venv (needs internet; coe-hpc1).
3. Run `scripts/hpc/setup_env.sh`. It rebuilds the shared venv from the lock, in about 6 min, and only when the lock changed. Do this only while no jobs are running.
4. Run `bash scripts/hpc/check_env.sbatch` on coe-hpc1, then submit the same script on HPC3.
5. Commit `requirements.in` and `requirements.lock` together as `build(env): …`.

Never `pip install` into the shared venv by hand. It no longer matches the lock: `check_env` fails, and teammates get a different environment.

### Add or change a model or metric weights

All pins live in `src/eldm/assets.py`.

1. Add or edit an `HFAsset` there: the repo, the full commit hash, and a file allow-list. Leave `sha256={}` for a new one. Fetch only the files you load: no `.bin` duplicates or `fp16` or `non_ema` copies.
   - To get the current commit hash: `curl -s https://huggingface.co/api/models/<repo> | python -c "import json,sys; print(json.load(sys.stdin)['sha'])"`.
2. Run `nice -n 19 python scripts/hpc/prefetch_assets.py`. It downloads, checks every file against the hashes the Hub publishes, and then stops with the exact `sha256={...}` block to paste in.
3. Paste the block in and rerun. It must end with "All assets present and verified."
4. Add the files your code needs to `REQUIRED` in `tests/test_assets.py`, then run `python -m pytest -q`.
5. Load the asset **at its pin**:
   - diffusers: `from_pretrained(repo, revision=...)`;
   - otherwise: a path from `assets.hf_snapshot_dir(name)`.

   A bare repo name fails offline: we cache pinned commits, never `refs/main`.
6. Run `python scripts/hpc/check_assets.py --quick`. Then run `check_assets.sbatch` on HPC3 if you changed how something loads.

For a single weights file that torch.hub downloads, add a `FileAsset` with its URL and sha256. Check that the sha256 starts with the hash in the filename (the torch.hub convention).

### Datasets

They're read-only on purpose. To verify them any time:
```bash
cd /data/cmpe261-fa26/group4/datasets && sha256sum -c /data/cmpe261-fa26/group4/$USER/fast-sd-inference/data/CHECKSUMS.txt
```
`scripts/hpc/fetch_datasets.sh` re-verifies too, and downloads nothing that's already there.

To add a dataset, extend `fetch_datasets.sh` with:
- a pinned URL;
- a server-side hash, such as an ETag or a git blob id;
- count checks.

Then add the new files' lines to `data/CHECKSUMS.txt`; existing lines must not change. The dataset folders belong to Dan, so ask him to make them writable while you add files.

## 5. Git workflow

- **Branches:**
  - each phase is one branch, `hpc/p<N>-<name>`, branched from the previous phase's branch, with one PR;
  - for your own feature, branch off the latest phase branch (`feat/...`, `eval/...`) and open a PR into it.
- **Commit messages:** `type(scope): summary`. The types are `build`, `feat`, `fix`, `test`, `data`, `exp`, `results` and `docs`. The body says *why* and *how it was checked*, including job IDs.
- **Before every commit:** `python -m pytest -q` must pass. Copy the real count from its output into the message.
- **Commit:** code, configs, prompt files, checksums, summary CSVs, figures and `docs/hpc/*`.
- **Never commit:** images, datasets, model weights, venvs or logs. `.gitignore` covers these, apart from `logs/.gitkeep`.
- **End of a phase:** update [PROGRESS.md](PROGRESS.md) with the gate result, the jobs table, decisions and versions, and open issues.

If you use an AI coding assistant, give it [PLAN.md](PLAN.md) and [PROGRESS.md](PROGRESS.md), and tell it that only people submit Slurm jobs. Phases 0–2 were built that way, with Claude Code under Dan's direction.

## 6. Where things are in the repo

| File | Role |
|---|---|
| `env/python.pin` | the portable CPython: version, URL and sha256 |
| `requirements.in`, `requirements.lock` | top-level pins, and the full freeze the venv is built from |
| `env/site.sh` | sourced by every shell and job: paths, caches, offline flags, the venv |
| `src/eldm/assets.py` | pinned models and metric weights, plus offline loaders and verifiers |
| `data/CHECKSUMS.txt` | sha256 of the dataset files |
| `scripts/hpc/setup_env.sh` | builds the Python and the venv (`--lock` re-resolves) |
| `scripts/hpc/prefetch_assets.py` | downloads and verifies models and metric weights (`--verify-only`) |
| `scripts/hpc/fetch_datasets.sh` | downloads, verifies and locks the datasets |
| `scripts/hpc/check_env.sbatch` | import check in a clean environment (CPU job, or `bash` on coe-hpc1) |
| `scripts/hpc/check_assets.py`, `check_assets.sbatch` | offline load check: `--quick` on coe-hpc1, `--full` as a CPU job |
| `scripts/hpc/job_prelude.sh` | the common start of every job: clean env, `site.sh`, threads, `$STAGE`, logging |
| `scripts/hpc/gpu_diagnose.sbatch` | when a job can't use its GPU: tells a broken GPU apart from our environment handling (§7) |
| `scripts/hpc/run_gpu.slurm` | runs an experiment file on one A100. Its first real use is Phase 5, once Phase 4's prompt files exist. |

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `network access attempted`, `LocalEntryNotFoundError`, or "offline mode is enabled" | the asset isn't cached, or was loaded without its pin | load it at its pin (§4); on coe-hpc1, run `prefetch_assets.py` |
| `GLIBC_2.28 not found`, or pip finds no wheel | that version needs a newer glibc than coe-hpc1 has | keep `torch==2.6.0`; install with `--only-binary=:all:` |
| The job finished but there's no log | submitted from outside the repo root | `cd` to your clone's root before `sbatch` |
| `error: … is not a fast-sd-inference clone` | the same | the same |
| The job landed on a P100 or an H100 | a plain `--gres=gpu:1` | `--gres=gpu:a100:1` |
| `CUDA error: CUDA-capable device(s) is/are busy or unavailable` on an idle GPU | a broken GPU: GPU 0 on cs002 since at least 2026-10-27 (job 98023). Slurm hands out the lowest free index, so 1-GPU jobs on an otherwise idle cs002 always get it | resubmit with `sbatch --exclude=cs002 …`. On another node, run `scripts/hpc/gpu_diagnose.sbatch` with `--nodelist=<node>` to confirm, then exclude that node and tell the HPC admins the node, GPU index and UUID |
| `Requested node configuration is not available` | more than 2 CPUs on `defq` | use `-c 2`, or `cpuqs` |
| `check_env` fails "venv matches requirements.lock" | someone ran pip in the shared venv | when no jobs run, rerun `scripts/hpc/setup_env.sh`, which rebuilds the venv |
| `'BaseModelOutputWithPooling' object has no attribute 'norm'` | torchmetrics CLIPScore with transformers 5 | use the `CLIPTensorFeatures` adapter in `scripts/hpc/check_assets.py`; Phase 5 settles this |
| Timings vary between runs | shared nodes: other jobs use the CPUs; GPUs aren't isolated | set thread counts; repeat runs; compare runs on the same node |
| `/home` quota or "No space left" errors | caches went to `~/.cache` | `source env/site.sh` before running anything |
| Harmless warnings | "NVIDIA driver too old (10010)" on coe-hpc1; "unauthenticated requests to the HF Hub"; `torch_dtype is deprecated` | nothing to do |
