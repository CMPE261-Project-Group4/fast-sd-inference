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
| 4 Frozen prompts | `hpc/p4-prompts` | counts match the data doc | **pass** (2026-10-28; no job: built and checked on coe-hpc1) |
| 5 Measurement | `hpc/p5-measure` | summary filled, deterministic | **pass** (2026-10-29, jobs 98329–98332); approved by Dan (2026-10-30) |
| 6 Pilot (G2) | `hpc/p6-pilot` | s/image + sweep GPU-hours reported | **pass** (2026-11-02, jobs 100435, 100436, 100547, 100548); the sweep runs on all 3,632 prompts (Dan, 2026-11-02) |
| 7 Orchestration | `hpc/p7-orchestration` | one command, resumable | **pass** (2026-11-02, arrays 100683, 100689 and 100695; evaluations 100684 and 100696) |
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
| 2026-10-29 | 98329 | P5.1 GPU check with the Phase 5 measurements (`check_gpu.sbatch`, gpuqs, 1× A100, 8 CPUs, 32G) | cs001, GPU 0 | COMPLETED 0:0 in 2m07s, MaxRSS 3.3 GB; all 4 runs pass `check_run.py`, now including NFE, stage times and the run environment. No other job on the node |
| 2026-10-29 | 98330 | P5.2 debug run A: `run_gpu.slurm 00_baseline.yaml --prompts-file data/prompts/subsets/debug.jsonl`, into `p5/debug_a` (1× A100, 8 CPUs, 32G, 30 min) | cs001, GPU 0 | COMPLETED 0:0 in 10m00s, MaxRSS 1.7 GB; `baseline` and `baseline_bs4`, 50 images each. Its only neighbour on the node was 98331 |
| 2026-10-29 | 98331 | P5.3 debug run B: the same, into `p5/debug_b` | cs001, GPU 1 | COMPLETED 0:0 in 9m59s, MaxRSS 1.4 GB; ran alongside 98330 |
| 2026-10-29 | 98332 | P5.4 `evaluate.slurm p5 p5/debug_a/baseline`, after `afterok:98330:98331` (1× A100, 8 CPUs, 32G, 30 min) | cs001, GPU 0 | COMPLETED 0:0 in 1m13s, MaxRSS 5.7 GB; quality files for all 4 runs, `check_run.py --expect-quality` passes all 4, `p5/summary/` written; **Phase 5 gate passed** |
| 2026-10-30 | 99002 | GPU diagnostic on cs002's GPU 0, to report it (`gpu_diagnose.sbatch --nodelist=cs002`) | cs002, GPU 0 | COMPLETED 0:0 in 8 s. Still broken: 5,562 uncorrectable DRAM ECC errors, "Remapping Failure Occurred: Yes"; every allocation fails (Phase 6's findings) |
| 2026-10-31 | 99031 | P6.1 tiny pilot: `run_gpu.slurm pilot.yaml --num-prompts 2` into `p6/tiny` (1× A100, 8 CPUs, 32G, 10 min), from dac29a5 | cs001, GPU 1 | TIMEOUT in 10m05s, MaxRSS 6.6 GB. CUDA couldn't use GPU 1, and the configs' `device: auto` fell back to the CPU: 108 s/image. Fixed by 1e8652a. Its dependents 99032 (P6.2) and 99034 (P6.3) never started (`DependencyNeverSatisfied`); Dan cancelled them. Co-tenant: 99052 (another user, 1× A100) |
| 2026-11-01 | 99828 | P6.1 again, from d68934a | cs001, GPU 0 | COMPLETED 0:0 in 1m38s, MaxRSS 3.6 GB; 5 runs × 2 images on CUDA, `ddim_20`'s first GPU run, every NFE as predicted. fp16 peaked at 6.84 GB: the DeepCache run's pipeline was never freed (fixed by 541ce83) |
| 2026-11-01 | 99829 | P6.2 pilot, from d68934a (still leaking) | cs001, GPU 0 | Cancelled by Dan after 19m36s, with the leak found; its evaluation 99830 never started |
| 2026-11-02 | 100435 | P6.2 pilot: `run_gpu.slurm pilot.yaml` into `p6/pilot` (1× A100, 8 CPUs, 32G, 1h15), from 2ce96f7 | cs001, GPU 0 | COMPLETED 0:0 in 49m57s, MaxRSS 3.7 GB; 5 runs × 200 images. fp16 peaked at 2.61 GB, so the leak is fixed; its images started from other noise than the baseline's (fixed by fe478dc), so fp16 was rerun. Alone on the node |
| 2026-11-02 | 100436 | P6.3 `evaluate.slurm p6/pilot p6/pilot/baseline`, after `afterok:100435` (20 min) | cs001, GPU 0 | COMPLETED 0:0 in 2m30s, MaxRSS 6.1 GB; `check_run.py --expect-quality` passes all 5; fp16's LPIPS to the baseline 0.76 showed the noise problem |
| 2026-11-02 | 100547 | P6.2b: `run_gpu.slurm pilot.yaml --only fp16` into `p6/pilot` (15 min), from fe478dc; job 100435's fp16 moved to `p6/fp16_job100435` | cs001, GPU 0 | COMPLETED 0:0 in 5m26s, MaxRSS 2.8 GB; 1.468 s/image (1.453 in 100435), 2.61 GB peak. Alone on the node |
| 2026-11-02 | 100548 | P6.3b `evaluate.slurm p6/pilot p6/pilot/baseline`, after `afterok:100547` (15 min) | cs001, GPU 0 | COMPLETED 0:0 in 2m17s, MaxRSS 2.0 GB; all 5 pass, fp16's LPIPS to the baseline 0.012; **gate G2's numbers** |
| 2026-11-02 | 100683 | P7.1 `submit_experiment.sh … pilot.yaml -- --num-prompts 4` into `p7/check`: array `0-4%4` of `run_array.sbatch` (1× A100, 8 CPUs, 32G, 10 min each), from 0294c1b | cs003, GPUs 0–3 | 5 tasks COMPLETED 0:0 in 18–61 s, MaxRSS 1.7 GB; tasks 0–3 (jobs 100685–100688) ran at once, task 4 (job 100683) waited for the cap (`JobArrayTaskLimit`). The cancel meant for task 4 came after it ended |
| 2026-11-02 | 100684 | P7.2 its evaluation: `evaluate.slurm p7/check p7/check/baseline`, `afterok:100683`, `--kill-on-invalid-dep=yes` (15 min) | cs003, GPU 0 | COMPLETED 0:0 in 43 s, MaxRSS 6.1 GB; started as the last task ended; `check_run.py --expect-quality` passes all 5 |
| 2026-11-02 | 100689 | P7.3 the same with `--num-prompts 16` into `p7/resume`, array `0-4%4` | cs003, GPUs 0–3 | Tasks 1–4 (jobs 100692–100694, 100689) COMPLETED 0:0 in 35–52 s. Dan cancelled task 0, the baseline (job 100691), after 1m05s at image 7 of 16; nothing of it reached `/scratch` |
| 2026-11-02 | 100690 | P7.4 its evaluation | none | CANCELLED by Slurm before starting: `DependencyNeverSatisfied`, `afterok:100689_*(failed)` |
| 2026-11-02 | 100695 | P7.5 the same `submit_experiment.sh` command again: it listed 4 runs done and the baseline todo, and submitted array `0%4` | cs003, GPU 0 | COMPLETED 0:0 in 1m52s, MaxRSS 1.7 GB; only the baseline ran, alone on the node: 5.677 s/image |
| 2026-11-02 | 100696 | P7.6 its evaluation, `afterok:100695` | cs003, GPU 0 | COMPLETED 0:0 in 38 s, MaxRSS 1.9 GB; `check_run.py --expect-quality` passes all 5; **Phase 7 gate passed** |

## Phase 7: Orchestration, R6 (2026-11-02)

### Gate: pass (arrays 100683, 100689 and 100695; evaluations 100684 and 100696)

One command runs whole experiment files and resumes after a failure:

```bash
scripts/hpc/submit_experiment.sh --output-root <dir> --time <each run's limit> <experiment.yaml>... [-- <run_experiments.py arguments>]
```

It submits the runs that aren't done as one Slurm job array (`run_array.sbatch`, one run per task, at most 4 at once), then an evaluation (`evaluate.slurm`) that starts only if every task succeeds. Run again, it submits only the unfinished runs. Both checks ran `pilot.yaml`'s 5 runs on cs003, from 0294c1b with a clean tree; Dan submitted them.

1. **One command** (`p7/check`, 4 prompts per run). Array 100683 ran tasks 0–3 at once on GPUs 0–3. Task 4 waited for the cap (`JobArrayTaskLimit`) and started when task 1 ended. Evaluation 100684 started as the last task ended, and `check_run.py --expect-quality` passed all 5. This didn't test resuming: the cancel meant for task 4 came after its 18-second run had ended.
2. **Resume** (`p7/resume`, 16 prompts per run). Dan cancelled array 100689's task 0, the baseline, at image 7 of 16. Nothing of it reached `/scratch`: the output root held only the 4 finished runs. Slurm cancelled evaluation 100690 before it started (`DependencyNeverSatisfied`, `afterok:100689_*(failed)`). The same command again listed the baseline as `todo` and the rest as `done`, and submitted only index 0 (array 100695) and evaluation 100696, which passed all 5. The 4 untouched runs' `perf.json` still name array 100689's tasks (jobs 100692–100694 and 100689); the baseline names job 100695.

Phase 7's jobs used 0.18 GPU-hours of 2.3 requested. Results are locked with `chmod -R a-w` in `/scratch/cmpe261-fa26/group4/results/p7/` (`check`, `resume`; 44 MB).

### Commits

- `eab4bb1 feat(runner): add --list, --index and --skip-existing`: `run_experiments.py` takes several experiment files as one list of runs. A run is `done` (its `perf.json` exists and its `config.yaml` equals the run's config), `todo`, or `conflict` (finished with another config: always an error, so results from different configs never mix). `--list` doesn't import torch, so it runs on g17's head node.
- `a14dd9e feat(runner): stage each run on the node-local SSD`: with `--stage-dir`, a run is written to the job's `$STAGE` and copied to its output directory when it ends, `perf.json` last (as `perf.json.tmp`, then `os.replace`). Stale quality files are removed first. `run_gpu.slurm` uses it too.
- `0294c1b feat(hpc): job arrays with a chained evaluation`: `run_array.sbatch` and `submit_experiment.sh`, tested through `--dry-run`; README.md and DEVELOPMENT.md (§3, §6, §7).
- `docs(hpc): Phase 7 results` (this commit): this log, README.md and DEVELOPMENT.md.

### Findings

- **4 of our jobs run at once on one node.** Array 100683's 4 running tasks all landed on cs003 while cs004 sat idle, and the 5th waited only for our own cap. So the QOS's `MaxTRESPerUser=node=2` counts nodes, not jobs, and the sweep runs 4 at a time: 65.5 GPU-hours is about 16.4 h, not 33.
- **Neighbours didn't slow runs down; the node did.** Seconds per image (speedup over the same set's baseline):

  | Run | Phase 6 pilot: cs001 GPU 0, alone, 200 images | `p7/check`: cs003, 4 at once, 4 images | `p7/resume`: cs003, 16 images |
  |---|---|---|---|
  | baseline | 5.775 | 5.616 | 5.677 (alone on the node) |
  | `ddim_20` | 2.340 (2.47×) | 2.262 (2.48×) | 2.273 (2.50×) |
  | `dpmpp_20` | 2.344 (2.46×) | 2.296 (2.45×) | 2.304 (2.46×) |
  | `deepcache_i3` | 2.357 (2.45×) | 2.306 (2.44×) | 2.316 (2.45×) |
  | `fp16` | 1.468 (3.93×) | 1.412 (3.98×) | 1.424 (3.99×) |

  The baseline beside 3 other runs (5.616 s) was no slower than alone (5.677 s). But every run is 1.7–3.0 % faster on cs003 than on cs001's GPU 0 (`p7/resume` against the pilot), so speedups within cs003 match the pilot's within 1.3 %. Few images, but all five runs agree.
- **Each task spends 12–38 s outside generating:** imports, loading the model, the warm-up and the copy to `/scratch` (fp16 with 4 prompts: 18 s in all, 6 s generating). Fine for the sweep's runs of 1–6 h.
- **`perf.json` records each task's own job ID** (`SLURM_JOB_ID`), not the array's. Array 100683's tasks 0–3 are jobs 100685–100688, and the last task gets the array's own ID, 100683. `sacct -j <array> -o JobID,JobIDRaw` shows both.
- **coe-hpc1 can see a new run a few seconds late.** Just after job 100696 ended, opening `p7/resume/baseline/perf.json` on coe-hpc1 said "No such file", and `ls` a moment later showed it: NFS had cached the failed lookup from before the run existed. `submit_experiment.sh` and the tasks read run states on HPC3, not here.
- **Not checked:** whether the prelude's `EXIT` trap removed the cancelled task's stage directory from cs003's `/tmp`; looking would take a job on cs003. A leftover can't collide with a rerun, whose stage directory has its own job ID in its name.

### Decisions

- **Slurm's own job arrays and dependencies, no workflow manager** (plan approved by Dan, 2026-11-02): nothing is added to the frozen venv.
- **A run is finished when its `perf.json` exists**, written last, and its `config.yaml` equals the run's config. A finished run with another config stops the submission.
- **One array per output root.** Phase 8 submits `00`–`04` together: one cap of 4 GPUs, one output root and one evaluation.
- **`--exclude=cs001,cs002` by default,** until the admins fix those GPUs (open issues).

### Phase 8 preview (dry run on coe-hpc1, 2026-11-02)

```bash
scripts/hpc/submit_experiment.sh --dry-run --output-root /scratch/cmpe261-fa26/group4/results/p8/sweep --time 07:00:00 \
    configs/experiments/00_baseline.yaml configs/experiments/01_steps.yaml configs/experiments/02_schedulers.yaml \
    configs/experiments/03_precision.yaml configs/experiments/04_deepcache.yaml
```

It lists 17 runs, all `todo`, and would submit one array, `--array=0-16%4`, and the evaluation with `p8/sweep/baseline`. Two limits are Phase 8's to set: `--time` must cover the longest run, the baseline at about 5.9 h (`report/pilot/sweep_estimate.csv`); and `--eval-time` must cover scoring all 17 runs in one job, about 2.6 h at Phase 6's 0.15 s per image, against the default 30 minutes.

### Noted for Phase 8 (questions for Dan; nothing changed)

- **Where a run lands moves its time by 2–3 %** (findings), about the size of some differences the sweep measures. The plan's second-node baseline measures it, and the summary's `node` column records it. Options: let runs spread over cs003 and cs004, and compare each to a baseline from its own node; or keep the whole sweep on one node.
- **`01_steps`' `steps_20` and `02_schedulers`' `pndm_20` are the same config** (PNDM, 20 steps, FP32). Running both costs about 2.7 GPU-hours; the pair could also measure timing noise.

## Phase 6: Pilot, R5 and D5, gate G2 (2026-10-30 to 2026-11-02)

### Gate G2: pass; the sweep runs on all 3,632 prompts

The five pilot runs on `data/prompts/subsets/pilot.jsonl` (100 COCO + 100 PartiPrompts, the same prompts and seeds in every run) ran on cs001's GPU 0 with no other job on the node: job 100435 from commit 2ce96f7, except `fp16`, rerun by job 100547 from fe478dc after the noise fix below. Job 100548 scored all five, and `check_run.py --expect-quality` passes all five. Every `perf.json` has `git_dirty: false` and both TF32 flags off.

| Run | s/image | Speedup | Peak GB | `nfe` (`nfe_full`) | CLIP ViT-g/14: all (COCO / Parti) | CLIP ViT-L/14 | LPIPS to baseline |
|---|---|---|---|---|---|---|---|
| baseline: PNDM 50 steps, FP32 | 5.775 | 1.00× | 4.76 | 51 (51) | 30.46 (30.62 / 30.30) | 27.40 | 0 |
| `ddim_20` | 2.340 | 2.47× | 4.76 | 20 (20) | 30.47 (30.49 / 30.46) | 27.35 | 0.346 |
| `dpmpp_20`: DPM-Solver++ 2M | 2.344 | 2.46× | 4.76 | 20 (20) | 30.57 (30.52 / 30.62) | 27.42 | 0.255 |
| `deepcache_i3`: interval 3 | 2.357 | 2.45× | 4.97 | 51 (17) | 30.56 (30.41 / 30.71) | 27.40 | 0.259 |
| `fp16` | 1.468 | 3.93× | 2.61 | 51 (51) | 30.45 (30.57 / 30.33) | 27.36 | 0.012 |

Stage coverage is 0.992–0.998. KID isn't reported from the pilot (decision below), so `report/pilot/summary.csv` leaves it out.

**The sweep's cost** (`report/pilot/sweep_estimate.csv`): the 17 runs of `00`–`04` plus the second-node baseline rerun.

| Prompt set | Sweep | Wall time on 4 GPUs | Phase 9's `05_combined`, separately |
|---|---|---|---|
| all of `eval_prompts.jsonl` (3,632), **chosen** | **65.5 GPU-h** (62.8 generation + 2.7 evaluation) | 16.4 h | 4.1 GPU-h |
| `subsets/parti_550.jsonl` (2,550) | **46.0 GPU-h** (44.1 + 1.9) | 11.5 h | 2.9 GPU-h |

- The cost model is fitted on `baseline`, `fp16` and `deepcache_i3`. It predicts the held-out `ddim_20` and `dpmpp_20` within +0.2 % and −0.0 %.
- Per image, FP32: a full UNet call costs 0.111 s, a call DeepCache serves from its cache 0.010 s, and the rest of the pipeline 0.129 s. FP16: 0.028, 0.0025 and 0.062 s.
- Runs the pilot didn't measure come from the model: PNDM at 10, 20 and 30 steps, the 50-step DDIM, DPM-Solver++ and Euler runs, Euler 20, DeepCache intervals 2 and 5, and the batch-4 runs. Batch 4 uses FP32's measured ratio, 0.838 (Phase 5's `debug_a`); for `fp16_bs4` that's an assumption.
- Job overhead (loading, warm-up, saving PNGs) is 0.112 s per image, from job 100435's 2,997 s; evaluation is 0.150 s per image, from job 100436's 150 s.

**PLAN.md's sanity checks:**

| Check | Result |
|---|---|
| The baseline's ViT-g/14 CLIP Score against DeepCache's published 50-step PLMS numbers: 30.30 (COCO), 29.51 (PartiPrompts) | 30.62 and 30.30: within 0.8, on 100 prompts each against the paper's full sets |
| The 20-step runs are the fastest FP32 runs | **yes**, narrowly: 2.340 and 2.344 s against `deepcache_i3`'s 2.357. FP16 is the fastest run overall |
| DeepCache's speedup against the paper's ≈1.7× (an RTX 2080) | 2.45× at interval 3, closer to the paper's headline 2.3× for SD1.5. A cached call costs 9 % of a full one here |
| FP16 is faster and smaller | **yes**: 3.93× faster, 2.61 GB peak against 4.76, and LPIPS 0.012 against the baseline: nearly the same images |

All five runs' ViT-g/14 CLIP Scores are within 0.12 of the baseline's. At 200 prompts that's noise, so the pilot says nothing yet about quality differences. The quality-vs-latency plot's axis spans only those 0.12 points, which makes them look large.

### What went wrong, and the fixes

1. **Job 99031 ran on the CPU.** P6.1 landed on cs001's GPU 1, which CUDA can't use (findings). Under the configs' `device: auto`, the run fell back to the CPU at 108 s/image and timed out. 1e8652a makes `run_gpu.slurm` and `evaluate.slurm` pass `--device cuda`, so a GPU that CUDA can't use fails the job instead. d68934a adds each GPU's MIG mode to `gpu_diagnose.sbatch`.
2. **DeepCache kept each pipeline alive (job 99828).** fp16, which ran right after `deepcache_i3`, peaked at 6.84 GB against 2.61 GB on its own (Phase 5). DeepCache replaces the UNet's `forward` methods with wrappers that hold its helper, and the helper holds the pipeline: a reference cycle that `del pipe` and `empty_cache()` don't free, so the DeepCache run's FP32 pipeline (about 4.2 GB) was still allocated. Only memory figures were affected, never speed or images. 541ce83 follows the documented cleanup: DeepCache's `helper.disable()` after generating, then `gc.collect()` and `torch.cuda.empty_cache()`, the `flush()` diffusers' own benchmarks run between runs. Job 100435 confirms it: fp16 peaked at 2.61 GB after `deepcache_i3`.
3. **CPU generators gave fp16 other noise (job 100435).** On 2026-11-02 Dan chose to follow the libraries' documented methods, so 2ce96f7 draws each prompt's noise from a CPU generator, as [diffusers recommends](https://huggingface.co/docs/diffusers/en/using-diffusers/reusing_seeds). But diffusers draws the noise in the pipeline's dtype, and a CPU generator's float16 draw is unrelated to its float32 draw from the same seed (correlation −0.015). So job 100435's fp16 images started from other noise than the baseline's, and scored LPIPS 0.76 against it. CUDA generators don't do this, which is why Phase 3's fp16 matched. fe478dc draws the noise in float32 and passes it as `latents`, as diffusers' maintainers advise ([diffusers#11056](https://github.com/huggingface/diffusers/issues/11056)). For FP32 runs that's exactly the noise the pipeline drew itself (pixel-identical on the tiny pipeline), so only fp16 was rerun: LPIPS 0.012. Job 100435's fp16 run is kept in `p6/fp16_job100435`.

### Commits

- `13c0548 exp: add pilot experiment`: `configs/experiments/pilot.yaml`; a test checks each pilot run equals the sweep run of the same name except for `prompts_file`.
- `4b053ab feat(viz): per-group quality tables and column selection in summarize.py`: `--by source|category|challenge` and `--omit`.
- `dac29a5 feat(exp): estimate sweep GPU-hours from pilot timings`: `scripts/estimate_sweep.py`.
- `1e8652a fix(hpc): GPU jobs require CUDA instead of falling back to the CPU`. Its message ties cs001's GPU state to a slurmd restart on cs001 at about 22:50 on 2026-10-30. That restart was cluster-wide (cs002 shows the same `SlurmdStartTime`), and when GPUs 1–3 became unusable isn't known. History isn't rewritten.
- `d68934a feat(hpc): gpu_diagnose shows each GPU's MIG mode`
- `541ce83 fix(bench): free each run's pipeline before the next run starts`
- `2ce96f7 feat(bench): draw each prompt's starting noise from a CPU generator`
- `fe478dc fix(bench): draw the starting noise in float32 so fp16 starts where the baseline does`
- `470a3f3 results: pilot summary and sweep estimate`: `report/pilot/`, this log, README.md and DEVELOPMENT.md.
- `77d29df docs(hpc): record the sweep size chosen at gate G2`: this log, README.md and `data/prompts/README.md`.
- `cd1e96e chore(git): merge main into hpc/p6-pilot`: PR #8 merged at 470a3f3, before 77d29df was pushed; no files changed. 77d29df reached `main` through PR #9.

### Findings

- **cs002's GPU 0 is a hardware fault** (job 99002, 2026-10-30): 73 volatile and 5,562 aggregate uncorrectable DRAM ECC errors, 8 rows remapped for uncorrectable errors, and "Remapping Failure Occurred: Yes". All four allocation probes fail. Keep `--exclude=cs002`; Dan reports it.
- **CUDA can't use cs001's GPUs 1–3.** They read utilization `[N/A]` and 1 MiB used, and CUDA found no usable device on GPU 1 (job 99031). Slurm still hands them out, so a job on cs001 works only when it gets GPU 0. Every Phase 6 job after 99031 got GPU 0. A guess, not confirmed: MIG mode on with no instances. `sbatch --nodelist=cs001 --gres=gpu:a100:3 scripts/hpc/gpu_diagnose.sbatch` would show it, while the node is otherwise free.
- **`coe-hpc3` stopped resolving on 2026-11-01**, even from coe-hpc1. `ssh g17` from coe-hpc1 reaches the same head node.
- **Timings repeat.** The baseline ran at 5.779 s/image over its first 50 images and 5.772 over its last 50, the same as Phase 5's 50-image run (5.767, with CUDA generators). fp16 ran at 1.453 s/image in job 100435 and 1.468 in job 100547 (+1.0 %), on the same GPU.
- **CPU generators cost nothing measurable.** With our schedulers the only random draw is each image's 4×64×64 starting noise; PNDM never draws more, DDIM only at eta > 0, Euler only at `s_churn` > 0, DPM-Solver++ only in its SDE variants.
- **DeepCache holds 0.21 GB of cached features** after the warm-up: 4.22 GB allocated before the timed batches against the baseline's 4.01, and a 4.97 GB peak against 4.76.
- **Three different speedups cost the same here.** 20 DDIM or DPM-Solver++ calls, and DeepCache's 17 full plus 34 cached calls, all take 2.34–2.36 s/image.
- **DDIM changes the images most** (LPIPS 0.346 against the baseline, against 0.255 and 0.259 for DPM-Solver++ and DeepCache). In the grid's first row, its tennis player has a different pose.
- **Evaluation is fast:** 1,000 images scored in 137–150 s.

### Decisions

- **Sweep size: all of `eval_prompts.jsonl`, 3,632 prompts** (Dan, 2026-11-02, gate G2): 65.5 GPU-hours, about 16.4 h on 4 GPUs, against 46.0 GPU-hours and 11.5 h for `parti_550`. Per-Challenge tables get all 1,632 PartiPrompts instead of 550. `configs/base.yaml` already sets `prompts_file: data/prompts/eval_prompts.jsonl` and no file in `00`–`05` overrides it, so Phase 8 has nothing to change (correction 12).
- **Phase 5 approved** (Dan merged PR #7, 2026-10-30).
- **KID: option (a)** (Dan, 2026-10-30): the code stays as it is; KID is read only from the sweeps, and the pilot's tables leave it out.
- **No new GitHub tooling** (Dan, 2026-10-30): no CI, PR template or branch-protection docs.
- **The sweep's GPU-hours are modeled from the pilot's timings,** with no extra probe job (Dan, 2026-10-30).
- **Follow the libraries' documented methods over our own** (Dan, 2026-11-02): CPU generators per prompt, with the noise drawn in float32 and passed as `latents`; between runs, DeepCache's `disable()` and diffusers' `flush()` pattern, in one process; CLIP Score stays by hand until a torchmetrics release fixes CLIPScore for transformers 5 (open issues).
- **PNG figures aren't committed.** Dan's rule is never to commit images, so `report/figures/pilot/` is regenerated from the commands below and kept in `/scratch/cmpe261-fa26/group4/results/p6/figures/`.

### Outputs and how to regenerate them

Locked with `chmod -R a-w`: `/scratch/cmpe261-fa26/group4/results/p6/` (`pilot`, `tiny` from job 99828, `fp16_job100435`, `figures`). Phase 6's jobs used 1.53 GPU-hours, failures included, of 3.8 requested.

On coe-hpc1, from the repo root, with `env/site.sh` sourced (a few seconds each):

```bash
R=/scratch/cmpe261-fa26/group4/results
python scripts/summarize.py --runs "$R/p6/pilot/*" --baseline $R/p6/pilot/baseline \
    --csv report/pilot/summary.csv --by source --omit kid_mean kid_std --out report/figures/pilot \
    --grid-runs baseline ddim_20 dpmpp_20 deepcache_i3 fp16 --grid-ids 0 1 2006 2012
python scripts/estimate_sweep.py --pilot "$R/p6/pilot/*" --pilot-elapsed-s 2997 --eval-elapsed-s 150 \
    --batch-pair $R/p5/debug_a/baseline $R/p5/debug_a/baseline_bs4 --csv report/pilot/sweep_estimate.csv
```

The grid shows the pilot's first two COCO prompts (ids 0 and 1) and first two PartiPrompts (2006, 2012), not ones picked by eye, with the runs as columns in the order given.

## Phase 5: Measurement correctness, R4 and D3 (2026-10-29)

### Gate: pass (jobs 98329–98332; approved by Dan, 2026-10-30)

Dan submitted the four jobs, all with `--exclude=cs002`. They ran from commit `aaba3a0` with a clean tree (`git_dirty: false` in every `perf.json`), offline, on cs001, with both TF32 flags off. `check_run.py` decides each check.

| Check | How | Result |
|---|---|---|
| Every `summary.csv` column filled for the debug runs | P5.4 wrote `p5/summary/summary.csv`, one row per run | **pass**: every column filled for all 4 runs. No run has FID, so `summarize.py` leaves the `fid` column out; each `quality.json` gives the reason in `fid_skipped` ("50 COCO images; FID needs at least 5,000") |
| Deterministic | `check_run.py --compare debug_a/baseline debug_b/baseline`: two separate jobs on different GPUs (0 and 1), the same 50 prompts and seeds | **pass**: 50 of 50 images pixel-identical (largest difference 0), the same NFE |
| The stage times add up | `check_run.py`: text + UNet + VAE within 0.90–1.01 of every batch's latency | **pass**: 0.998 on all 4 debug runs, 0.993–0.998 on P5.1's 4 |
| Not a gate, reported: does batch size change the pixels? | `--compare debug_a/baseline debug_a/baseline_bs4` | **Slightly.** 0 of 50 images identical, largest pixel difference 3 of 255, LPIPS at most 1.7e-5 (mean 8.2e-7). Batch 4 is deterministic itself: `--compare debug_a/baseline_bs4 debug_b/baseline_bs4` gives 50 of 50 identical |

Outputs, locked with `chmod -R a-w`: `/scratch/cmpe261-fa26/group4/results/_checks/98329/` and `/scratch/cmpe261-fa26/group4/results/p5/` (`debug_a`, `debug_b`, `summary`). The compare checks ran on coe-hpc1 (1.5 s each).

### P5.1: the G1 variants, measured (job 98329)

2 smoke prompts each, after the base config's 2 warm-up batches. No other job was on cs001.

| Variant | s/image | Speedup | Peak GB | `nfe` (`nfe_full`) | Text / UNet / VAE, s per image | Coverage |
|---|---|---|---|---|---|---|
| baseline: PNDM 50 steps, FP32 | 5.64 | 1.00× | 4.76 | 51 (51) | 0.009 / 5.51 / 0.109 | 0.998 |
| `fp16` | 1.44 | 3.91× | 2.61 | 51 (51) | 0.009 / 1.38 / 0.040 | 0.993 |
| `dpmpp_20`: DPM-Solver++ 2M, 20 steps | 2.32 | 2.44× | 4.76 | 20 (20) | 0.009 / 2.19 / 0.109 | 0.996 |
| `deepcache_i3`: DeepCache, interval 3 | 2.34 | 2.41× | 4.97 | 51 (17) | 0.009 / 2.21 / 0.110 | 0.996 |

With TF32 off, the FP32 runs slowed down: the baseline from G1's 4.13 to 5.64 s/image (+37 %), `dpmpp_20` from 1.67 to 2.32, `deepcache_i3` from 1.74 to 2.34. FP16 doesn't use TF32 and barely moved (1.46 to 1.44), so its speedup grew from 2.84× to 3.91×. These are still 2-image numbers.

### Debug runs: 50 COCO prompts (jobs 98330, 98331, 98332)

**Debug numbers, not results:** 50 prompts, with the two jobs side by side on cs001.

| Run | s/image | Speedup | Peak GB | `nfe` (`nfe_full`) | Text / UNet / VAE, s per image | Coverage |
|---|---|---|---|---|---|---|
| `debug_a/baseline` | 5.767 | 1.00× | 4.76 | 51 (51) | 0.009 / 5.638 / 0.111 | 0.998 |
| `debug_b/baseline` | 5.761 | 1.00× | 4.76 | 51 (51) | 0.009 / 5.631 / 0.111 | 0.998 |
| `debug_a/baseline_bs4` | 4.832 | 1.19× | 8.77 | 51 (51) | 0.003 / 4.705 / 0.114 | 0.998 |
| `debug_b/baseline_bs4` | 4.825 | 1.20× | 8.77 | 51 (51) | 0.003 / 4.699 / 0.114 | 0.998 |

The batch-to-batch std was 0.03 s at batch 1. At batch 4 it was 2.49 s, because 50 images make 12 batches of 4 and a last batch of 2.

| Run (A and B identical) | CLIP ViT-g/14 | CLIP ViT-L/14 | KID mean (std) | LPIPS to `debug_a/baseline` |
|---|---|---|---|---|
| `baseline` | 30.81 (per image 24.7–44.2) | 27.40 (21.3–34.2) | −0.0033 (2e-7) | 0 |
| `baseline_bs4` | 30.82 | 27.40 | −0.0033 (2e-7) | 8.2e-7 |

### Commits

- `841c75c feat(perf): pin TF32 flags, log run environment, count NFE`
- `ea8c1af feat(perf): add per-stage timing with CUDA events`
- `97582c1 feat(eval): OpenCLIP ViT-g/14 CLIP Score and per-image scores`
- `966ae3c feat(eval): add KID, gate FID at 5,000 images, load weights offline`
- `12c11df feat(run): add --prompts-file to run_experiments.py`
- `c57236e fix(run): reject unknown run names in --only`
- `33b56ea fix(pipeline): pass dtype= instead of the deprecated torch_dtype=`
- `aaba3a0 test(hpc): debug-run determinism check`: `check_run.py --compare`, `quality.json` records its LPIPS baseline, and the docs.
- `docs(hpc): record the Phase 5 jobs and gate` (this commit): the job results in this log, README.md and DEVELOPMENT.md.

### Findings

- **DeepCache 0.1.1 replaces `forward` on the UNet, every down and up block and the mid block** with wrappers that return cached outputs on skipped steps. PyTorch's hooks fire around those wrappers, so a hook on `unet.mid_block` fires on cached steps too. That rules out the mid-block hook proposed in the open issue below. `nfe_full` is counted one level inside, at `unet.mid_block.resnets[0]`, which runs only on full passes. A hook on `unet` counts every call (`nfe`).
- **PNDM makes 51 UNet calls for 50 steps** (and 6 for 5): with `skip_prk_steps`, its second timestep (961 at 50 steps) runs twice. DDIM and DPM-Solver++ make one call per step.
- **DeepCache finds the step with `timesteps.index(t)`.** For PNDM's repeated timestep, the second call gets the first one's index. At 50 PNDM steps, DeepCache's full passes are 25 (interval 2), 17 (3) and 11 (5) of 51 calls. Without the repeat, interval 2 would give 26. `nfe_full` measures what actually runs. Its `start_timestep` is never reset between pipeline calls, which is harmless while the first index is 0.
- **One pipeline call runs the text encoder twice** (the prompt and the empty negative prompt), and the UNet once per step on a doubled batch (classifier-free guidance). The VAE decode is `post_quant_conv`, then `decoder`.
- **CLIP-L by hand matches torchmetrics.** On the first two COCO val images with their captions, on CPU: 18.40965 and 25.50584 by hand, 18.40965 and 25.50585 with torchmetrics' CLIPScore and Phase 2's adapter (`pytest -m slow`).
- **KID on 8 real photos** (CPU, real Inception weights): a set against itself scores −0.138 (the unbiased estimator on identical sets), against other photos 0.005. Scoring again gives the same numbers.
- **The tiny CPU pipeline's stage coverage is 0.912:** with tiny models, tokenizing and converting to PIL take about 9% of the time. The 0.90 bound is for real runs: on the A100, every batch's coverage was 0.993–0.998.
- **Commit `33b56ea`'s message says 116 tests passed; its own tree passes 110.** The 6 compare-mode tests were in the working tree, not the commit. History isn't rewritten; from now on each commit's count is taken on exactly the tree being committed.
- **Runs are deterministic across jobs and GPUs on one node,** with cuDNN's `benchmark` and `deterministic` flags both off: two jobs on GPUs 0 and 1 of cs001 made pixel-identical images at batch 1 and at batch 4. Not tested across nodes yet; every Phase 5 job ran on cs001.
- **Batch size changes the pixels, a little** (largest difference 3 of 255, LPIPS ≤ 1.7e-5). Compare configs at the same batch size; the difference is far below the 0.01 tolerance.
- **The baseline's ViT-g/14 CLIP Score, 30.81 on 50 COCO prompts, is close to DeepCache's published 30.30** for 50-step PLMS on COCO, PLAN.md's Phase 6 sanity check. 50 prompts is noisy: single images range from 24.7 to 44.2.
- **KID says nothing at 50 images.** With subsets of min(1000, n), any n ≤ 1000 makes every subset the whole set, so every subset gives the same value: `kid_std` is 2e-7. The mean, −0.0033, is negative because the unbiased estimator dips below 0 when 50 images can't tell the two sets apart. At debug size KID checks the code path, not the quality. See the open issue.
- **Batch 4 gives 1.19× the throughput for 1.8× the memory** (8.77 GB peak against 4.76). With classifier-free guidance, batch 1 already sends 2 latents through the UNet, so there's less to gain.
- **Evaluation is quick.** P5.4 re-hashed ViT-g's 5.47 GB in 9 s, then scored 200 images in 73 s in all. The 50 photos' Inception features were computed once and reused: 15 s for the first run, 6 s for each one after.
- **`check_run.py` on coe-hpc1 waits on storage when the files aren't cached.** Checking the 4 debug runs (200 PNGs, 84 MB) took 160 s, at about 5 % of a CPU and 40 MB, while P5.4 was reading the same files. Once they were cached, each `--compare` took 1.5 s. For large result sets, use the check inside `evaluate.slurm`.
- **The prelude's `gpus=` field is `CUDA_VISIBLE_DEVICES`,** the GPU's index on the node, not a count: job 98056 printed 1, job 98329 printed 0.
- **Some Phase 6 and 7 work is already done:** PLAN.md's Phase 6 commit 2 (`--summary-out`) is `summarize.py --csv`, and Phase 7's `--prompts-file` is in `run_experiments.py`.

### Decisions

- **CLIP Score by hand, for both models** (Dan, 2026-10-28): 100 × max(cos(image, prompt), 0) per image, for OpenCLIP ViT-g/14 (`clip_vitg`, primary) and CLIP ViT-L/14 (`clip_vitl`), from the pinned weights. A run's score is the mean. torchmetrics' CLIPScore and the transformers 5 adapter are no longer in the metric code; `check_assets.py` keeps the adapter for its own check.
- **NFE has two columns** (Dan, 2026-10-28): `nfe` (every UNet call) and `nfe_full` (calls that ran the whole UNet). They differ only under DeepCache.
- **TF32 off** for matmuls and cuDNN (`configs/base.yaml` `tf32`), set on every run, since the flags are process-global. `check_run` fails a run whose recorded flags differ from its config.
- **KID** uses subsets of min(1000, n) images, drawn with a fixed seed. **FID** is computed only from 5,000 COCO images; below that, `fid_skipped` gives the reason. Real images are center-cropped to 512×512, as before.
- **Determinism tolerance:** B reproduces A if every image is pixel-identical, or each differing one has LPIPS < 0.01 against A's.
- **`summary.csv` per result set,** with a `run` column (the path under the result set), since run names repeat across result sets.

## Phase 4: Frozen prompts, D1 and D2 (2026-10-28)

### Gate: pass

No job ran. Dan chose to build and check the set on coe-hpc1 (2026-10-28): it takes 3 s and 82 MB in one process, inside the login-node rule. `python scripts/prepare_prompts.py` built the files from the shared datasets, and a second run gave identical sha256s. `tests/test_prompts.py` checks the files on every `pytest` run, including a byte-for-byte regeneration wherever the datasets exist.

| Check | Result |
|---|---|
| Full set, `data/prompts/eval_prompts.jsonl` | 3,632 prompts: 2,000 COCO captions (ids 0–1999, 2,000 different images) and all 1,632 PartiPrompts (ids 2000–3631). `seed = 1000 + id` |
| The data doc's counts, as PLAN.md quotes them | 2,000 COCO, 1,632 PartiPrompts, 90 Quantity, 91 Writing & Symbols: **all match**. I haven't seen the data doc itself |
| Subsets, `data/prompts/subsets/` | `debug` 50 (COCO ids 0–49); `pilot` 200 (100 COCO + 100 PartiPrompts); `parti_550` 2,550 (2,000 COCO + 550 PartiPrompts). Rows identical to the full set's; `debug` ⊂ `pilot` ⊂ `parti_550`; Challenge counts as planned |
| Reproducible | byte for byte; the sha256s are in [data/prompts/README.md](../../data/prompts/README.md) |

### Commits

- `60eaef4 feat(data): parse PartiPrompts and add the challenge field`
- `33cbe65 feat(data): add Challenge-stratified sampling and subset files`
- `data: freeze evaluation prompts and subsets` (this commit): the files, `data/prompts/README.md`, the frozen-file tests, and the docs.

### Findings

- **Python's default TSV parsing corrupts a prompt.** With csv's default quoting, `"G I G G L E" painted in thick colorful lettering…` (id 3560, Writing & Symbols) loses its quotes, which matter for text rendering. The parser reads the file literally (`QUOTE_NONE`); every line has exactly 4 fields.
- **Whitespace.** 10 PartiPrompts have leading or trailing spaces, and COCO captions often end in a space; one caption (image 359219, not among the 2,000) contains a newline. Collapsing whitespace changed 3,814 of 26,646 texts and none of their CLIP token ids.
- **One prompt exceeds CLIP's 77 tokens:** id 2358, 84 tokens. It's kept; diffusers truncates it (`truncation=True`) the same way in every config. The longest COCO caption has 58.
- **The old COCO sampling couldn't grow.** `rng.sample(ids, n)` plus an interleaved `rng.choice` reorders the images when `n` changes, so Phase 9's 3,000 extra FID prompts would have changed the first 2,000. Now every image has one seeded order and one caption chosen in advance, so growing the set only appends.

### Decisions

- **`parti_550` holds all 2,000 COCO prompts plus the 550 PartiPrompts** (Dan, 2026-10-28). It's the whole fallback sweep set in one file: each run reads one prompts file, and KID needs the COCO prompts.
- **Built on the login node, with no job** (Dan, 2026-10-28), from measured cost: 3.3 s, 95 MB.
- **The subsets nest** (`debug` ⊂ `pilot` ⊂ `parti_550`), so results on a smaller subset are a slice of those on a larger one.
- **The manual prompts left the evaluation set.** They stay in `manual_prompts.yaml` as the smoke-test set for `gpu_check.py`.
- **PartiPrompts' Note isn't stored**; no metric uses it. Category and Challenge are.

## Phase 3: First GPU job, R3 and G1 (2026-10-27 to 2026-10-28)

### Gate G1: pass (approved by Dan, 2026-10-28)

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
8. **coe-hpc1 has its own Slurm 18.08 client** (`/usr/bin/sbatch`, `sinfo`). It doesn't see HPC3. All queries go through `ssh coe-hpc3` (or `ssh g17` while that name doesn't resolve, Phase 6).
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
12. **Phase 8's commit 1, `exp: set sweep prompt set from gate G2`, is skipped** (2026-11-02). Gate G2 chose the full set, which `configs/base.yaml` already uses, so there's no `prompts_file` to change.
13. **Phase 7, as done** (2026-11-02).
    - Commit 1 doesn't add `--prompts-file`: it has existed since Phase 5.
    - Commit 2 stages the whole run (images, `config.yaml`, `timings.jsonl`, `perf.json`), not only the images, so `perf.json` is copied last and marks a finished run.
    - "Evaluate and summarize jobs" are one job: `evaluate.slurm` already summarizes.
    - The check ran `pilot.yaml`'s 5 runs instead of "2 runs × 4 prompts", so 5 tasks with a cap of 4 also tested the cap. With 4 prompts a task lasted 18–61 s, too short to cancel by hand, so the cancel was redone in a second output root with 16 prompts.
    - PROGRESS.md is updated by the last commit, `docs(hpc): Phase 7 results`, not by commit 3.

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
- **Prompt set** (Phase 4): `data/prompts/eval_prompts.jsonl` and `subsets/`, built by `scripts/prepare_prompts.py` with `--sample-seed 0 --base-seed 1000`; see [data/prompts/README.md](../../data/prompts/README.md).
- **Measurement** (Phase 5): TF32 off; NFE as `nfe` and `nfe_full`; stage times with CUDA events; CLIP Score by hand with ViT-g/14 (primary) and ViT-L/14; KID with subsets of min(1000, n); FID only from 5,000 COCO images. See Phase 5's decisions.
- **Sweep prompt set** (gate G2, Dan, 2026-11-02): all of `eval_prompts.jsonl`, 3,632 prompts.
- **Starting noise** (Phase 6): one CPU generator per prompt, seeded with the prompt's seed; the noise is drawn in float32 and passed as `latents`, cast to the run's dtype. Runs before fe478dc (Phases 3–5, and job 99828) used CUDA generators, so their images don't match later ones. See Phase 6.
- **Orchestration** (Phase 7): `submit_experiment.sh` submits experiment files as one Slurm job array of the unfinished runs, at most 4 at once, excluding cs001 and cs002, with the evaluation chained by `afterok` and `--kill-on-invalid-dep=yes`. A run is finished when its `perf.json` exists and its `config.yaml` matches; runs are staged on the node's SSD and copied with `perf.json` last.
- **Versions are unchanged** despite the CLIPScore incompatibility (Dan, 2026-10-19). An adapter is used instead of `transformers<5`.
- **A100 driver:** 590.44.01 (CUDA ≤ 13.1), which supports the CUDA 12.4 wheels.

## Open issues

- **The whole class is in the group.** `cmpe261-fa26` has 19 members, so every classmate can write to and delete from `group4/`. There's no team-only group. Locking finished data with `chmod -R a-w` is the only guard. Dan could ask the admins for a team group.
- **GPU 0 on cs002 is broken (found 2026-10-28, job 98023).** Submit GPU jobs with `--exclude=cs002` until the admins fix it; Dan reports it to them. Details are in Phase 3's findings; job 99002 (Phase 6) shows a hardware fault: uncorrectable ECC errors and a failed row remap.
- **CUDA can't use cs001's GPUs 1–3 (found 2026-10-31, job 99031).** Slurm still allocates them, and a job that gets one fails at model load (`--device cuda`, since 1e8652a). Dan reports it with cs002's GPU; `gpu_diagnose.sbatch --nodelist=cs001 --gres=gpu:a100:3` would show their MIG mode. Until then a job on cs001 works only on GPU 0; resubmit one that fails there.
- **`coe-hpc3` doesn't resolve (since 2026-11-01).** Use `ssh g17` from coe-hpc1; it's the same head node. Dan reports it.
- **CPU threads aren't limited to the allocation. Handled since Phase 3.** In job 92197 (`-c 2`), torch picked 28 threads, all of c19's physical cores. This fits correction 2: cgroups don't confine the job. `job_prelude.sh` sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` from `SLURM_CPUS_PER_TASK`.
- **Resolved in Phase 5 (Dan, 2026-10-28): CLIP Score is computed by hand, so the metric code doesn't use CLIPScore.** The original issue: torchmetrics' CLIPScore doesn't work with transformers 5 (found by job 92223).
  - **The cause:** in transformers 5.20.1, `CLIPModel.get_image_features` and `get_text_features` return a `BaseModelOutputWithPooling`, with the projected embedding in `.pooler_output`. transformers 4.x returned the tensor itself. torchmetrics 1.9.0 calls `.norm()` on the result and fails with `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'`.
  - **Fixed upstream, not released.** [Lightning-AI/torchmetrics#3407](https://github.com/Lightning-AI/torchmetrics/issues/3407) was fixed on torchmetrics' master in July 2026, with the same `.pooler_output` unwrap our code uses; 1.9.0 (2026-03-09) is still the latest release (checked 2026-11-02). Once a release has the fix, CLIP ViT-L/14 moves to CLIPScore, the method diffusers documents (Dan, 2026-11-02: follow the libraries' methods). ViT-g/14 stays on open_clip, since its weights exist only in open_clip's format.
  - **Downgrading is costly.** The last transformers 4.x release (4.57.6) needs `huggingface_hub<1.0`, but diffusers 0.41 needs `>=1.32`. So `transformers<5` would also downgrade diffusers, mean rebuilding the venv, and redo the Phase 1 gate.
  - **Decision (Dan, 2026-10-19): keep the package versions and use an adapter.**
    - The adapter is a `CLIPModel` subclass whose `get_image_features` and `get_text_features` return `.pooler_output`. It's passed to CLIPScore through its callable loader, with the model loaded from the pinned snapshot.
    - The CLIP-L check in `check_assets.py` uses it. On coe-hpc1, offline, it gives 20.45 with an image's own caption versus 7.66 with another caption.
    - A score computed by hand, `100 * max(cos, 0)`, also gives 20.45, so the metric's definition doesn't change.
  - **Proposed for Phase 5:** use the same adapter in `src/eldm/metrics.py`, or compute the per-image score by hand, which `quality_per_image.csv` needs anyway. Dan confirms which in Phase 5.
- **coe-hpc1 has an old NVIDIA driver (CUDA 10.1).** torch prints a harmless warning there when it probes CUDA. We never use a GPU on coe-hpc1.
- **Resolved in Phase 6 (Dan, 2026-10-30): option (a).** **KID below 1,001 COCO images (found 2026-10-29, job 98332).** With subsets of min(1000, n), `kid_std` is about 0 and `kid_mean` is noise at small n (Phase 5's findings). The pilot's 100 COCO prompts have the same problem, and PLAN.md's Phase 6 already says to skip KID at 200 prompts. The sweeps aren't affected: 2,000 COCO images give subsets of 1,000. The options were:
  - (a) leave the code as it is, and read KID only from the sweeps (chosen);
  - (b) have `evaluate_quality.py` skip KID below a minimum, as it does for FID, with a `kid_skipped` reason;
  - (c) use smaller subsets at small n, such as n/2, so `kid_std` means something.
- **Resolved in Phase 5 (Dan, 2026-10-28): NFE has two columns, `nfe` and `nfe_full`.** The original issue: DeepCache's cached steps still call `unet.forward` and skip only the inner blocks, so a UNet forward hook alone counts the same NFE as the baseline. The proposed hook on `unet.mid_block` would have been wrong: DeepCache wraps the mid block too, so its hook fires on cached steps. Full passes are counted at `unet.mid_block.resnets[0]` (Phase 5's findings).
