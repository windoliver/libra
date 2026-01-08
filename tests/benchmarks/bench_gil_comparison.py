#!/usr/bin/env python3
"""
GIL vs No-GIL Performance Benchmark.

Compares single-threaded and multi-threaded performance to measure
the impact of free-threading on LIBRA's workloads.

Usage:
    # Standard Python (with GIL)
    python tests/benchmarks/bench_gil_comparison.py

    # Free-threaded Python (without GIL)
    python3.13t tests/benchmarks/bench_gil_comparison.py

    # Force GIL disabled (on free-threaded build)
    PYTHON_GIL=0 python3.13t tests/benchmarks/bench_gil_comparison.py

See Issue #26 and ADR-008 for details.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable


def is_gil_enabled() -> bool:
    """Check if GIL is currently enabled."""
    if hasattr(sys, "_is_gil_enabled"):
        return sys._is_gil_enabled()
    return True


def is_free_threaded_build() -> bool:
    """Check if running on free-threaded Python build."""
    return hasattr(sys, "_is_gil_enabled")


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    single_threaded_time: float
    multi_threaded_time: float
    num_threads: int
    speedup: float

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Single-threaded: {self.single_threaded_time:.4f}s\n"
            f"  Multi-threaded ({self.num_threads} threads): {self.multi_threaded_time:.4f}s\n"
            f"  Speedup: {self.speedup:.2f}x"
        )


def benchmark_cpu_bound(iterations: int = 10_000_000) -> BenchmarkResult:
    """
    Benchmark CPU-bound work (arithmetic operations).

    This is where free-threading should show the most benefit.
    """

    def cpu_work(n: int) -> int:
        """CPU-intensive calculation."""
        total = 0
        for i in range(n):
            total += i * i % 1000
        return total

    num_threads = 4
    work_per_thread = iterations // num_threads

    # Single-threaded
    start = time.perf_counter()
    cpu_work(iterations)
    single_time = time.perf_counter() - start

    # Multi-threaded
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(cpu_work, work_per_thread) for _ in range(num_threads)]
        for f in as_completed(futures):
            f.result()
    multi_time = time.perf_counter() - start

    return BenchmarkResult(
        name="CPU-Bound (arithmetic)",
        single_threaded_time=single_time,
        multi_threaded_time=multi_time,
        num_threads=num_threads,
        speedup=single_time / multi_time if multi_time > 0 else 0,
    )


def benchmark_polars_aggregation(rows: int = 1_000_000) -> BenchmarkResult:
    """
    Benchmark Polars DataFrame aggregation.

    Polars uses Rust internally, so it releases the GIL during computation.
    """
    import polars as pl

    # Create test data
    df = pl.DataFrame({
        "group": [f"g{i % 100}" for i in range(rows)],
        "value": list(range(rows)),
    })

    def aggregate() -> pl.DataFrame:
        return df.group_by("group").agg([
            pl.col("value").sum().alias("sum"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").std().alias("std"),
        ])

    num_threads = 4

    # Single aggregation (baseline)
    start = time.perf_counter()
    aggregate()
    single_time = time.perf_counter() - start

    # Multiple concurrent aggregations
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(aggregate) for _ in range(num_threads)]
        for f in as_completed(futures):
            f.result()
    multi_time = time.perf_counter() - start

    # For Polars, we expect similar times due to internal parallelization
    return BenchmarkResult(
        name="Polars Aggregation",
        single_threaded_time=single_time,
        multi_threaded_time=multi_time,
        num_threads=num_threads,
        speedup=single_time / multi_time if multi_time > 0 else 0,
    )


def benchmark_json_encoding(iterations: int = 100_000) -> BenchmarkResult:
    """
    Benchmark JSON encoding with msgspec.

    This simulates encoding market data events.
    """
    import msgspec

    data = {
        "symbol": "BTC/USDT",
        "timestamp": 1704067200000,
        "bid": 42150.50,
        "ask": 42151.00,
        "bid_size": 1.5,
        "ask_size": 2.0,
    }

    def encode_many(count: int) -> int:
        total_bytes = 0
        for _ in range(count):
            total_bytes += len(msgspec.json.encode(data))
        return total_bytes

    num_threads = 4
    work_per_thread = iterations // num_threads

    # Single-threaded
    start = time.perf_counter()
    encode_many(iterations)
    single_time = time.perf_counter() - start

    # Multi-threaded
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(encode_many, work_per_thread) for _ in range(num_threads)]
        for f in as_completed(futures):
            f.result()
    multi_time = time.perf_counter() - start

    return BenchmarkResult(
        name="JSON Encoding (msgspec)",
        single_threaded_time=single_time,
        multi_threaded_time=multi_time,
        num_threads=num_threads,
        speedup=single_time / multi_time if multi_time > 0 else 0,
    )


def benchmark_dict_operations(iterations: int = 1_000_000) -> BenchmarkResult:
    """
    Benchmark dictionary operations (common in event processing).
    """

    def dict_work(count: int) -> int:
        d: dict[str, int] = {}
        for i in range(count):
            key = f"key_{i % 1000}"
            d[key] = d.get(key, 0) + 1
        return len(d)

    num_threads = 4
    work_per_thread = iterations // num_threads

    # Single-threaded
    start = time.perf_counter()
    dict_work(iterations)
    single_time = time.perf_counter() - start

    # Multi-threaded (each thread has its own dict)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(dict_work, work_per_thread) for _ in range(num_threads)]
        for f in as_completed(futures):
            f.result()
    multi_time = time.perf_counter() - start

    return BenchmarkResult(
        name="Dict Operations",
        single_threaded_time=single_time,
        multi_threaded_time=multi_time,
        num_threads=num_threads,
        speedup=single_time / multi_time if multi_time > 0 else 0,
    )


def benchmark_list_comprehension(iterations: int = 100) -> BenchmarkResult:
    """
    Benchmark list comprehension (common Python pattern).
    """

    def list_work(size: int) -> list[int]:
        return [i * i for i in range(size)]

    size = 100_000
    num_threads = 4
    work_per_thread = iterations // num_threads

    def run_iterations(count: int) -> int:
        total = 0
        for _ in range(count):
            total += len(list_work(size))
        return total

    # Single-threaded
    start = time.perf_counter()
    run_iterations(iterations)
    single_time = time.perf_counter() - start

    # Multi-threaded
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_iterations, work_per_thread) for _ in range(num_threads)]
        for f in as_completed(futures):
            f.result()
    multi_time = time.perf_counter() - start

    return BenchmarkResult(
        name="List Comprehension",
        single_threaded_time=single_time,
        multi_threaded_time=multi_time,
        num_threads=num_threads,
        speedup=single_time / multi_time if multi_time > 0 else 0,
    )


def run_all_benchmarks() -> list[BenchmarkResult]:
    """Run all benchmarks and return results."""
    benchmarks: list[Callable[[], BenchmarkResult]] = [
        benchmark_cpu_bound,
        benchmark_dict_operations,
        benchmark_list_comprehension,
        benchmark_json_encoding,
        benchmark_polars_aggregation,
    ]

    results: list[BenchmarkResult] = []
    for bench in benchmarks:
        try:
            print(f"Running {bench.__name__}...", flush=True)
            result = bench()
            results.append(result)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    return results


def print_report(results: list[BenchmarkResult]) -> None:
    """Print benchmark report."""
    print("\n" + "=" * 70)
    print("GIL vs No-GIL Performance Benchmark Report")
    print("=" * 70)
    print(f"\nPython: {sys.version}")
    print(f"Free-threaded build: {is_free_threaded_build()}")
    print(f"GIL enabled: {is_gil_enabled()}")
    print("\n" + "-" * 70)

    for result in results:
        print(f"\n{result}")

    print("\n" + "-" * 70)
    print("\nSummary:")
    print("-" * 70)

    # Calculate average speedup
    speedups = [r.speedup for r in results]
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0

    print(f"Average multi-threaded speedup: {avg_speedup:.2f}x")

    if is_free_threaded_build() and not is_gil_enabled():
        print("\nNote: Running with GIL DISABLED (free-threaded mode)")
        if avg_speedup > 2.0:
            print("Result: SIGNIFICANT speedup from free-threading!")
        elif avg_speedup > 1.5:
            print("Result: Moderate speedup from free-threading")
        else:
            print("Result: Limited speedup - workload may not benefit from free-threading")
    else:
        print("\nNote: Running with GIL ENABLED")
        print("Expected: Multi-threaded speedup limited by GIL for CPU-bound tasks")

    print("\n" + "=" * 70)


def save_results_json(results: list[BenchmarkResult], filename: str = "benchmark_results.json") -> None:
    """Save results to JSON file."""
    import json
    from datetime import datetime

    data = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "free_threaded_build": is_free_threaded_build(),
        "gil_enabled": is_gil_enabled(),
        "results": [
            {
                "name": r.name,
                "single_threaded_time": r.single_threaded_time,
                "multi_threaded_time": r.multi_threaded_time,
                "num_threads": r.num_threads,
                "speedup": r.speedup,
            }
            for r in results
        ],
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    print("Starting GIL comparison benchmarks...")
    print("This may take a few minutes.\n")

    results = run_all_benchmarks()
    print_report(results)

    # Optionally save to JSON
    if "--save" in sys.argv:
        save_results_json(results)
