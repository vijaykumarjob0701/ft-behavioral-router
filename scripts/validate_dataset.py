#!/usr/bin/env python3
"""Validate behavioural routing JSONL seed files against schema v1.0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install when repo root is on PYTHONPATH
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fta_router.dataset import summarise, validate_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate FTA routing dataset JSONL files (schema 1.0)."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=_REPO / "data" / "routing" / "train_seed.jsonl",
        help="Path to train JSONL",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=_REPO / "data" / "routing" / "eval_seed.jsonl",
        help="Path to eval JSONL",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any file has zero rows",
    )
    args = parser.parse_args(argv)

    all_errors: list[str] = []
    for path in (args.train, args.eval):
        if not path.exists():
            all_errors.append(f"missing file: {path}")
            continue
        errs = validate_file(path)
        all_errors.extend(f"{path.name}: {e}" if not e.startswith(path.name) else e for e in errs)
        summary = summarise(path)
        print(f"== {path}")
        print(f"   n={summary['n']}")
        print(f"   primary_action={summary['primary_action']}")
        print(f"   domain={summary['domain']}")
        print(f"   reasoning_level={summary['reasoning_level']}")
        if args.strict and summary["n"] == 0:
            all_errors.append(f"{path}: empty file")

    if all_errors:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("\nVALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
