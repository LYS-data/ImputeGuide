"""Command-line entry point for configuration validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .protocol import validate_protocol


def _repository_root(value: str) -> Path:
    root = Path(value).expanduser().resolve()
    if not (root / "configs" / "method.yaml").is_file():
        raise argparse.ArgumentTypeError(f"not an ImputeGuide project root: {root}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m imputeguide")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate-config", help="check cross-file paper protocol invariants"
    )
    validate.add_argument("--root", type=_repository_root, default=Path.cwd())
    args = parser.parse_args()
    result = validate_protocol(Path(args.root))
    for error in result.errors:
        print(f"ERROR: {error}")
    if result.release_ready:
        print("ImputeGuide configuration is valid.")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
