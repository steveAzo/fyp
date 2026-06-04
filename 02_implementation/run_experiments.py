"""
Main entry point for running the full experiment grid.

Usage (from 02_implementation/):

    # Quick validation (~108 configs, finishes in minutes)
    python run_experiments.py --mode quick

    # Full constraint grid (8 100 configs, ~6 hours)
    python run_experiments.py --mode full

    # Model comparison grid (405 configs, ~25 minutes)
    python run_experiments.py --mode model-comparison

    # Full grid, specific datasets only
    python run_experiments.py --mode full --datasets credit telco

Results are saved incrementally to results/results.csv.
Re-running resumes from where it left off unless --no-resume is passed.
"""
from __future__ import annotations

import argparse

from experiments.grid import generate_full_grid, generate_model_comparison_grid, generate_quick_grid
from experiments.runner import run_experiment_batch
from utils.logger import get_logger

logger = get_logger("run_experiments")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run drift detection experiments")
    p.add_argument(
        "--mode",
        choices=["quick", "full", "model-comparison"],
        default="quick",
        help=(
            "'quick' for pipeline validation; "
            "'full' for the complete constraint grid (8 100 configs); "
            "'model-comparison' for cross-model replication (405 configs)"
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
        default="results/results.csv",
        help="CSV output path (default: results/results.csv)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh even if the output CSV already exists",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "quick":
        configs = generate_quick_grid(datasets=args.datasets)
        logger.info("Quick grid: %d configs", len(configs))
    elif args.mode == "model-comparison":
        configs = generate_model_comparison_grid(datasets=args.datasets)
        logger.info("Model comparison grid: %d configs", len(configs))
    else:
        configs = generate_full_grid(datasets=args.datasets)
        logger.info("Full grid: %d configs", len(configs))

    results = run_experiment_batch(
        configs,
        output_path=args.output,
        resume=not args.no_resume,
    )

    logger.info("Done. Results shape: %s  →  %s", results.shape, args.output)


if __name__ == "__main__":
    main()
