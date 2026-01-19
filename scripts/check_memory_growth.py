#!/usr/bin/env python3
"""Check memory growth from memray stats JSON output (Issue #91).

This script analyzes memray statistics and fails CI if memory growth
exceeds configured thresholds.

Usage:
    # Generate stats JSON
    python -m memray run -o profile.bin scripts/benchmark_memory.py
    python -m memray stats profile.bin --json > memory-stats.json

    # Check thresholds
    python scripts/check_memory_growth.py memory-stats.json

Exit codes:
    0 - Memory within acceptable limits
    1 - Memory exceeds threshold
    2 - Error reading/parsing stats file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Memory thresholds (configurable via environment or args)
DEFAULT_PEAK_MB_THRESHOLD = 500  # Max peak memory in MB
DEFAULT_TOTAL_ALLOCATIONS_THRESHOLD = 10_000_000  # Max total allocations


def format_bytes(n: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def check_memory_stats(
    stats_file: Path,
    peak_mb_threshold: float = DEFAULT_PEAK_MB_THRESHOLD,
    total_allocs_threshold: int = DEFAULT_TOTAL_ALLOCATIONS_THRESHOLD,
) -> tuple[bool, dict]:
    """Check memory stats against thresholds.

    Args:
        stats_file: Path to memray stats JSON file
        peak_mb_threshold: Maximum allowed peak memory in MB
        total_allocs_threshold: Maximum allowed total allocations

    Returns:
        Tuple of (passed, stats_dict)
    """
    try:
        with open(stats_file) as f:
            stats = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Stats file not found: {stats_file}")
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in stats file: {e}")
        sys.exit(2)

    # Extract key metrics
    # Note: memray stats JSON structure may vary by version
    # Common fields: total_allocations, total_memory, peak_memory
    total_allocations = stats.get("total_allocations", 0)
    peak_memory_bytes = stats.get("peak_memory", stats.get("total_memory", 0))
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    # Get allocation breakdown if available
    allocation_breakdown = stats.get("allocations_by_size", {})
    top_allocators = stats.get("top_allocators", [])

    # Check thresholds
    passed = True
    failures = []

    if peak_memory_mb > peak_mb_threshold:
        passed = False
        failures.append(
            f"Peak memory {peak_memory_mb:.2f} MB exceeds threshold {peak_mb_threshold} MB"
        )

    if total_allocations > total_allocs_threshold:
        passed = False
        failures.append(
            f"Total allocations {total_allocations:,} exceeds threshold {total_allocs_threshold:,}"
        )

    return passed, {
        "peak_memory_mb": peak_memory_mb,
        "peak_memory_bytes": peak_memory_bytes,
        "total_allocations": total_allocations,
        "allocation_breakdown": allocation_breakdown,
        "top_allocators": top_allocators[:10] if top_allocators else [],
        "failures": failures,
        "thresholds": {
            "peak_mb": peak_mb_threshold,
            "total_allocations": total_allocs_threshold,
        },
    }


def print_report(stats: dict, passed: bool) -> None:
    """Print memory analysis report."""
    print("=" * 60)
    print("MEMORY ANALYSIS REPORT")
    print("=" * 60)

    print(f"\nPeak Memory:       {stats['peak_memory_mb']:.2f} MB")
    print(f"Total Allocations: {stats['total_allocations']:,}")

    print(f"\nThresholds:")
    print(f"  Peak Memory:     {stats['thresholds']['peak_mb']} MB")
    print(f"  Allocations:     {stats['thresholds']['total_allocations']:,}")

    if stats["top_allocators"]:
        print("\nTop Allocators:")
        for i, alloc in enumerate(stats["top_allocators"][:5], 1):
            if isinstance(alloc, dict):
                location = alloc.get("location", alloc.get("function", "unknown"))
                size = alloc.get("size", alloc.get("total_memory", 0))
                print(f"  {i}. {location}: {format_bytes(size)}")
            else:
                print(f"  {i}. {alloc}")

    print("\n" + "-" * 60)
    if passed:
        print("RESULT: PASSED - Memory within acceptable limits")
    else:
        print("RESULT: FAILED - Memory exceeds thresholds")
        for failure in stats["failures"]:
            print(f"  - {failure}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Check memray stats against memory thresholds"
    )
    parser.add_argument(
        "stats_file",
        type=Path,
        help="Path to memray stats JSON file",
    )
    parser.add_argument(
        "--peak-mb",
        type=float,
        default=DEFAULT_PEAK_MB_THRESHOLD,
        help=f"Peak memory threshold in MB (default: {DEFAULT_PEAK_MB_THRESHOLD})",
    )
    parser.add_argument(
        "--max-allocations",
        type=int,
        default=DEFAULT_TOTAL_ALLOCATIONS_THRESHOLD,
        help=f"Max total allocations (default: {DEFAULT_TOTAL_ALLOCATIONS_THRESHOLD:,})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    passed, stats = check_memory_stats(
        args.stats_file,
        peak_mb_threshold=args.peak_mb,
        total_allocs_threshold=args.max_allocations,
    )

    if args.json:
        stats["passed"] = passed
        print(json.dumps(stats, indent=2))
    else:
        print_report(stats, passed)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
