"""Command-line entry point for configuration validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .protocol import validate_protocol
from .runner import run_csv


def _repository_root(value: str) -> Path:
    root = Path(value).expanduser().resolve()
    if not (root / "configs" / "method.yaml").is_file():
        raise argparse.ArgumentTypeError(f"not an ImputeGuide project root: {root}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m imputeguide")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser(
        "validate-config", help="check cross-file method configuration"
    )
    validate_parser.add_argument("--root", type=_repository_root, default=Path.cwd())
    run_parser = subparsers.add_parser(
        "run", help="select an imputer and write one completed table"
    )
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--config-root", type=_repository_root, default=Path.cwd())
    run_parser.add_argument("--clusters", type=int, required=True)
    run_parser.add_argument("--anchor", required=True)
    run_parser.add_argument("--history-ranking", nargs="*", default=[])
    run_parser.add_argument("--probe-ranking", nargs="*", default=[])
    run_parser.add_argument("--opportunity-score", type=float, required=True)
    run_parser.add_argument("--opportunity-threshold", type=float, default=0.0)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--no-header", action="store_true")
    args = parser.parse_args()
    if args.command == "validate-config":
        result = validate_protocol(Path(args.root))
        for error in result.errors:
            print(f"ERROR: {error}")
        if result.release_ready:
            print("ImputeGuide configuration is valid.")
            return 0
        return 2
    try:
        run = run_csv(
            args.input,
            args.output,
            no_header=args.no_header,
            n_clusters=args.clusters,
            anchor=args.anchor,
            historical_ranking=args.history_ranking,
            probe_ranking=args.probe_ranking,
            opportunity_score=args.opportunity_score,
            opportunity_threshold=args.opportunity_threshold,
            config_root=args.config_root,
            seed=args.seed,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    if run.selection.selected_method is None:
        print("No imputation method completed successfully.")
        return 2
    print(f"Selected method: {run.selection.selected_method}")
    print(f"Completed table: {args.output / 'completed.csv'}")
    print(f"Selection trace: {args.output / 'selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
