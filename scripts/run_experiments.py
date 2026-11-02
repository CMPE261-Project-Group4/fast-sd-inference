"""Run every run defined in one or more experiment files.

Example:
    python scripts/run_experiments.py configs/experiments/01_steps.yaml --num-prompts 20
    python scripts/run_experiments.py configs/experiments/00_baseline.yaml \
        --prompts-file data/prompts/subsets/debug.jsonl

--prompts-file runs on a subset of the frozen prompts (data/prompts/subsets/) without editing
the experiment file; --num-prompts takes the first N of whichever file is used.

Several experiment files make one list of runs, in order; run names must be unique across them.
A run is done when <output_root>/<name>/perf.json, written last, exists and config.yaml there
equals the run's config. --list prints each run's index, name and state. A job array runs one
index per task (--index) and skips finished runs (--skip-existing), so resubmitting redoes only
the runs that didn't finish.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from eldm.config import load_experiment

DONE, TODO, CONFLICT = "done", "todo", "conflict"


def run(cfg: dict, stage_dir: str | None = None) -> dict:
    # eldm.benchmark imports torch and diffusers; --list runs on the head node without them.
    from eldm.benchmark import run as benchmark_run

    return benchmark_run(cfg, stage_dir=stage_dir)


def run_state(cfg: dict) -> str:
    """done: perf.json is there and config.yaml matches; todo: no perf.json; conflict: a run
    finished there with another config."""
    out_dir = Path(cfg["output_root"]) / cfg["name"]
    if not (out_dir / "perf.json").exists():
        return TODO
    try:
        saved = yaml.safe_load((out_dir / "config.yaml").read_text())
    except OSError:
        return CONFLICT
    return DONE if saved == cfg else CONFLICT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", metavar="experiment",
                    help="Path to configs/experiments/*.yaml; several make one list of runs")
    ap.add_argument("--only", nargs="+", help="Run only these run names")
    ap.add_argument("--index", nargs="+", type=int,
                    help="Run only the runs at these 0-based positions in --list ($SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--prompts-file", help="Override prompts_file, e.g. data/prompts/subsets/pilot.jsonl")
    ap.add_argument("--num-prompts", type=int, help="Override num_prompts (for quick checks)")
    ap.add_argument("--device", help="Override device (cuda | mps | cpu)")
    ap.add_argument("--output-root", help="Override output_root")
    ap.add_argument("--list", action="store_true",
                    help="Print each run's index, name and state (done, todo, conflict); exit 1 on a conflict")
    ap.add_argument("--skip-existing", action="store_true", help="Skip done runs; stop on a conflict")
    ap.add_argument("--stage-dir", help="Write each run here first (job scripts pass the node-local "
                    "SSD), then copy it to output_root with perf.json last")
    args = ap.parse_args(argv)

    runs = [cfg for path in args.experiments for cfg in load_experiment(path)]
    names = [r["name"] for r in runs]
    if repeated := sorted({n for n in names if names.count(n) > 1}):
        ap.error(f"run names repeat across the experiment files: {', '.join(repeated)}")
    # Overrides first: a run's state depends on its final config.
    for cfg in runs:
        if args.prompts_file:
            cfg["prompts_file"] = args.prompts_file
        if args.num_prompts is not None:
            cfg["num_prompts"] = args.num_prompts
        if args.device:
            cfg["device"] = args.device
        if args.output_root:
            cfg["output_root"] = args.output_root

    # Indices are positions in the whole list, so they stay the same under --only.
    selected = list(enumerate(runs))
    if args.only:
        if unknown := sorted(set(args.only) - set(names)):
            ap.error(f"--only: no run named {', '.join(unknown)} in {', '.join(args.experiments)}")
        selected = [(i, cfg) for i, cfg in selected if cfg["name"] in args.only]
    if args.index:
        if bad := [i for i in args.index if not 0 <= i < len(runs)]:
            ap.error(f"--index {bad}: there are {len(runs)} runs, 0 to {len(runs) - 1}")
        selected = [(i, cfg) for i, cfg in selected if i in args.index]
        if not selected:
            ap.error("--index and --only select no run together")

    if args.list:
        states = [(i, cfg["name"], run_state(cfg)) for i, cfg in selected]
        for i, name, state in states:
            print(f"{i}\t{name}\t{state}")
        conflicts = [name for _, name, state in states if state == CONFLICT]
        if conflicts:
            print(f"error: finished with another config: {', '.join(conflicts)}", file=sys.stderr)
        return 1 if conflicts else 0

    for _, cfg in selected:
        if args.skip_existing:
            state = run_state(cfg)
            if state == CONFLICT:
                print(f"error: {Path(cfg['output_root']) / cfg['name']} holds a finished run with "
                      "another config.yaml; use another --output-root or move it aside", file=sys.stderr)
                return 1
            if state == DONE:
                print(f"{cfg['name']}: done, skipped")
                continue

        perf = run(cfg, stage_dir=args.stage_dir)
        print(json.dumps({k: v for k, v in perf.items() if k != "env"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
