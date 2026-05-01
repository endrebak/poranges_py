#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import polars as pl

import polaranges

T = TypeVar("T")


def main() -> None:
    args = parse_args()

    print("polaranges Python BED overlap benchmark", flush=True)
    print(f"core version: {polaranges.benchmark_version()}", flush=True)
    print(f"python: {sys.version.split()[0]}", flush=True)
    print(f"polars: {pl.__version__}", flush=True)
    print("timing unit: seconds", flush=True)
    print("measured path: public Python API over eager Polars DataFrames", flush=True)
    print()

    started = time.perf_counter()
    left = read_bed(args.left)
    left_read_time = time.perf_counter() - started

    started = time.perf_counter()
    right = read_bed(args.right)
    right_read_time = time.perf_counter() - started

    print(f"left:  {left.height} rows from {args.left}", flush=True)
    print(f"right: {right.height} rows from {args.right}", flush=True)
    print(
        "read times (excluded): "
        f"left={left_read_time:.6f}s "
        f"right={right_read_time:.6f}s "
        f"total={left_read_time + right_read_time:.6f}s",
        flush=True,
    )
    print(
        "options: "
        f"multiple={args.multiple} "
        f"slack={args.slack} "
        f"contained={args.contained} "
        f"preserve_input_order={args.preserve_input_order} "
        f"parallel={args.parallel}",
        flush=True,
    )
    print(f"match_by: {args.match_by}", flush=True)
    print(f"repetitions: {args.reps}", flush=True)
    print()

    common_kwargs = dict(
        multiple=args.multiple,
        slack=args.slack,
        contained_intervals_only=args.contained,
        match_by=args.match_by,
        preserve_input_order=args.preserve_input_order,
        parallel_enabled=args.parallel != "serial",
        force_parallel=args.parallel == "force",
    )

    if args.include_pairs:
        print(
            "warning: overlap_pairs_report returns Python lists; "
            "this intentionally includes Python list materialization cost.",
            flush=True,
        )
        pairs_report = run_repeated(
            args.reps,
            lambda: polaranges.overlap_pairs_report(left, right, **common_kwargs),
        )
        pairs = pairs_report.last[:2]
        pair_count = len(pairs[0])
        print_operation_report("overlap_pairs", pair_count, pairs_report)
        print()
        print("last overlap_pairs DataFrame:")
        print(
            pl.DataFrame(
                {
                    "left_row": pairs[0],
                    "right_row": pairs[1],
                }
            )
        )
        print()

    overlap_report = run_repeated(
        args.reps,
        lambda: polaranges.overlap_report(left, right, **common_kwargs),
    )
    overlap = overlap_report.last[0]
    print_operation_report("overlap", overlap.height, overlap_report)
    print()
    print("last overlap DataFrame:")
    print(overlap)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark polaranges from Python over eager Polars DataFrames.",
    )
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--slack", type=int, default=0)
    parser.add_argument("--multiple", choices=["all", "first", "last"], default="all")
    parser.add_argument("--contained", action="store_true")
    parser.add_argument("--no-preserve-order", dest="preserve_input_order", action="store_false")
    parser.add_argument("--match-by", default="Chrom")
    parser.add_argument(
        "--parallel",
        choices=["serial", "auto", "force"],
        default="serial",
        help="Parallel mode forwarded to the Rust overlap kernel.",
    )
    parser.add_argument(
        "--include-pairs",
        action="store_true",
        help="Also time overlap_range_pairs_report, including Python list materialization.",
    )
    args = parser.parse_args()

    if args.reps < 1:
        parser.error("--reps must be a positive integer")

    return args


def read_bed(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=["Chrom", "Start", "End"],
        schema_overrides={
            "Chrom": pl.String,
            "Start": pl.Int64,
            "End": pl.Int64,
        },
        comment_prefix="#",
        truncate_ragged_lines=True,
        columns=[0, 1, 2],
    )


class RepeatedResult:
    def __init__(self, last: T, walls: list[float]) -> None:
        self.last = last
        self.walls = walls


def run_repeated(reps: int, fn: Callable[[], T]) -> RepeatedResult:
    last = fn()
    walls: list[float] = []

    for _ in range(reps):
        started = time.perf_counter()
        last = fn()
        walls.append(time.perf_counter() - started)

    return RepeatedResult(last, walls)


def print_operation_report(label: str, rows: int, repeated: RepeatedResult) -> None:
    wall_total = sum(repeated.walls)
    wall_avg = statistics.fmean(repeated.walls)
    timings = repeated.last[-1]
    python_overhead = wall_avg - timings["total"]
    print(
        f"{label}: rows={rows} "
        f"wall_total={wall_total:.6f}s "
        f"wall_avg={wall_avg:.6f}s "
        f"native_total={timings['total']:.6f}s "
        f"python_overhead={python_overhead:.6f}s "
        f"profile={format_timings(timings)}",
        flush=True,
    )


def format_timings(timings: dict[str, float]) -> str:
    return " ".join(f"{name}={value:.6f}s" for name, value in timings.items())


if __name__ == "__main__":
    main()
