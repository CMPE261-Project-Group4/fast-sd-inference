"""Run every run defined in an experiment file.

Example:
    python scripts/run_experiments.py configs/experiments/01_steps.yaml --num-prompts 20
"""

from __future__ import annotations

import argparse
import json

from eldm.benchmark import run
from eldm.config import load_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", help="Path to configs/experiments/*.yaml")
    ap.add_argument("--only", nargs="+", help="Run only these run names")
    ap.add_argument("--num-prompts", type=int, help="Override num_prompts (for quick checks)")
    ap.add_argument("--device", help="Override device (cuda | mps | cpu)")
    ap.add_argument("--output-root", help="Override output_root")
    args = ap.parse_args()

    runs = load_experiment(args.experiment)
    if args.only:
        runs = [r for r in runs if r["name"] in args.only]
    for cfg in runs:
        if args.num_prompts is not None:
            cfg["num_prompts"] = args.num_prompts
        if args.device:
            cfg["device"] = args.device
        if args.output_root:
            cfg["output_root"] = args.output_root

        perf = run(cfg)
        print(json.dumps({k: v for k, v in perf.items() if k != "env"}, indent=2))


if __name__ == "__main__":
    main()
