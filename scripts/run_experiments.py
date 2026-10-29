"""Run every run defined in an experiment file.

Example:
    python scripts/run_experiments.py configs/experiments/01_steps.yaml --num-prompts 20
    python scripts/run_experiments.py configs/experiments/00_baseline.yaml \
        --prompts-file data/prompts/subsets/debug.jsonl

--prompts-file runs on a subset of the frozen prompts (data/prompts/subsets/) without editing
the experiment file; --num-prompts takes the first N of whichever file is used.
"""

from __future__ import annotations

import argparse
import json
import sys

from eldm.benchmark import run
from eldm.config import load_experiment


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", help="Path to configs/experiments/*.yaml")
    ap.add_argument("--only", nargs="+", help="Run only these run names")
    ap.add_argument("--prompts-file", help="Override prompts_file, e.g. data/prompts/subsets/pilot.jsonl")
    ap.add_argument("--num-prompts", type=int, help="Override num_prompts (for quick checks)")
    ap.add_argument("--device", help="Override device (cuda | mps | cpu)")
    ap.add_argument("--output-root", help="Override output_root")
    args = ap.parse_args(argv)

    runs = load_experiment(args.experiment)
    if args.only:
        if unknown := sorted(set(args.only) - {r["name"] for r in runs}):
            ap.error(f"--only: no run named {', '.join(unknown)} in {args.experiment}")
        runs = [r for r in runs if r["name"] in args.only]
    for cfg in runs:
        if args.prompts_file:
            cfg["prompts_file"] = args.prompts_file
        if args.num_prompts is not None:
            cfg["num_prompts"] = args.num_prompts
        if args.device:
            cfg["device"] = args.device
        if args.output_root:
            cfg["output_root"] = args.output_root

        perf = run(cfg)
        print(json.dumps({k: v for k, v in perf.items() if k != "env"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
