from __future__ import annotations

import argparse
from experiments.grid import (
    generate_correlation_grid,
    generate_multivariate_grid,
    generate_quick_multivariate_grid,
)
from experiments.runner import run_experiment_batch
from utils.logger import get_logger

logger = get_logger("run_multivariate")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multivariate drift detection experiments")
    p.add_argument(
        "--mode",
        choices=["quick", "full", "correlation", "quick-correlation"],
        default="quick",
        help=(
            "'quick' for rapid pipeline validation on the marginal-shift grid; "
            "'full' for the full marginal-shift constraint grid; "
            "'correlation' for the supplementary marginal-preserving correlation-shift grid; "
            "'quick-correlation' for a small validation slice of the correlation grid"
        ),
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["credit", "telco", "maintenance"],
        default=None,
        help="Restrict to specific datasets (default: all three)",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "CSV output path. Defaults to results/results.csv for 'quick'/'full' "
            "and results/results_correlation.csv for 'correlation'/'quick-correlation' "
            "-- kept separate by default since the two conditions are structurally "
            "different (different drift_type namespace) and shouldn't be mixed in analysis."
        ),
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh even if the output CSV already exists",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()

    if args.output is not None:
        output_path = args.output
    elif args.mode in ("correlation", "quick-correlation"):
        output_path = "results/results_correlation.csv"
    else:
        output_path = "results/results.csv"

    if args.mode == "quick":
        configs = generate_quick_multivariate_grid(datasets=args.datasets)
        logger.info("Quick multivariate grid: %d configs", len(configs))
    elif args.mode == "full":
        configs = generate_multivariate_grid(datasets=args.datasets)
        logger.info("Full multivariate grid: %d configs", len(configs))
    elif args.mode == "correlation":
        configs = generate_correlation_grid(datasets=args.datasets)
        logger.info("Correlation-shift grid: %d configs", len(configs))
    else:  # quick-correlation
        configs = generate_correlation_grid(
            datasets=args.datasets,
            monitoring_frequencies=[100],
            window_sizes=[100],
            reference_sizes=[500, None],
            n_repetitions=2,
        )
        logger.info("Quick correlation-shift grid: %d configs", len(configs))

    results = run_experiment_batch(
        configs,
        output_path=output_path,
        resume=not args.no_resume,
    )

    logger.info("Done. Results shape: %s -> %s", results.shape, output_path)

if __name__ == "__main__":
    main()
