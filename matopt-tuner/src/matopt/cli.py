from __future__ import annotations

import argparse
import json
import os

from .protocol import MeasurementProfile, Workload
from .runner import MatOptRunner
from .search.lfbo import LFBOConfig
from .session import TuningSession


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="matopt-tuner")
    sub = root.add_subparsers(dest="command", required=True)
    tune = sub.add_parser("tune")
    tune.add_argument("--runner", required=True)
    for name in ("m", "n", "k", "threads"):
        tune.add_argument(f"--{name}", required=True, type=int)
    tune.add_argument("--cpus", required=True)
    tune.add_argument(
        "--objective",
        choices=("one_shot", "steady", "throughput"),
        default="one_shot",
    )
    tune.add_argument("--budget", type=int, default=256)
    tune.add_argument("--search", choices=("random", "lfbo"), default="random")
    tune.add_argument("--seed", type=int, default=19260817)
    tune.add_argument("--history", required=True)
    tune.add_argument("--output", required=True)
    tune.add_argument("--timeout", type=float, default=300)
    tune.add_argument("--warmups", type=int, default=3)
    tune.add_argument("--samples", type=int, default=3)
    tune.add_argument("--minimum-sample-ms", type=float, default=100)
    tune.add_argument("--lfbo-initial", type=int, default=24)
    tune.add_argument("--lfbo-copies", type=int, default=3)
    tune.add_argument("--lfbo-generations", type=int, default=8)
    tune.add_argument("--lfbo-neighbors", type=int, default=256)
    tune.add_argument("--lfbo-radius", type=int, default=2)
    tune.add_argument("--lfbo-quantile", type=float, default=0.10)
    tune.add_argument("--lfbo-fraction", type=float, default=0.10)
    tune.add_argument("--lfbo-similarity-penalty", type=float, default=1.0)
    tune.add_argument("--lfbo-patience", type=int, default=2)
    tune.add_argument("--lfbo-trees", type=int, default=100)
    tune.add_argument("--lfbo-model-jobs", type=int, default=1)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    workload = Workload(args.m, args.n, args.k, args.threads, args.cpus)
    runner = MatOptRunner(args.runner, timeout=args.timeout)
    profile = MeasurementProfile(
        warmups=args.warmups,
        samples=args.samples,
        minimum_sample_ms=args.minimum_sample_ms,
        seed=args.seed,
    )
    result = TuningSession(workload, runner, history=args.history).tune(
        budget=args.budget,
        objective=args.objective,
        seed=args.seed,
        measurement=profile,
        algorithm=args.search,
        lfbo_config=LFBOConfig(
            initial_population=args.lfbo_initial,
            copies=args.lfbo_copies,
            generations=args.lfbo_generations,
            neighbors=args.lfbo_neighbors,
            radius=args.lfbo_radius,
            quantile=args.lfbo_quantile,
            fraction_selected=args.lfbo_fraction,
            similarity_penalty=args.lfbo_similarity_penalty,
            patience=args.lfbo_patience,
            trees=args.lfbo_trees,
            model_jobs=args.lfbo_model_jobs,
        ),
        output=args.output,
    )
    print(
        json.dumps(
            {
                "output": os.path.abspath(args.output),
                "fingerprint": result["fingerprint"],
                "pareto_count": len(result["pareto"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
