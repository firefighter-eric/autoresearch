#!/usr/bin/env python3
"""Generate CSV copies from canonical TSV result tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "benchmarks" / "results-summary.tsv"


def convert_tsv_to_csv(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.reader(source, delimiter="\t"))

    if not rows:
        raise SystemExit(f"{input_path} is empty")

    expected_columns = len(rows[0])
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != expected_columns:
            raise SystemExit(
                f"{input_path}:{line_number} has {len(row)} columns; "
                f"expected {expected_columns}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as target:
        csv.writer(target, lineterminator="\n").writerows(rows)

    try:
        display_path = output_path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = output_path
    print(f"Wrote {display_path} ({len(rows) - 1} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="TSV file to convert; defaults to benchmarks/results-summary.tsv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="CSV output path; defaults to the input path with a .csv suffix",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_path.with_suffix(".csv")
    convert_tsv_to_csv(input_path, output_path)


if __name__ == "__main__":
    main()
