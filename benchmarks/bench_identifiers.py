#!/usr/bin/env python
"""
Benchmark for interned identifiers (Issue #111).

Compares performance of:
- Symbol (interned) vs str for hash lookups
- Symbol (interned) vs str for equality checks
- Memory usage

Run:
    python benchmarks/bench_identifiers.py
"""

from __future__ import annotations

import sys
import time
from typing import Any

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

from libra.core.identifiers import Symbol, VenueId, clear_all_identifier_caches


def bench_hash_lookup(iterations: int = 1_000_000) -> dict[str, float]:
    """Benchmark dictionary hash lookups."""
    clear_all_identifier_caches()

    symbols_str = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
    symbols_interned = [Symbol(s) for s in symbols_str]

    # Create dictionaries
    dict_str: dict[str, int] = {s: i for i, s in enumerate(symbols_str)}
    dict_interned: dict[Symbol, int] = {s: i for i, s in enumerate(symbols_interned)}

    # Benchmark str lookups
    lookup_key_str = "BTC/USDT"
    start = time.perf_counter_ns()
    for _ in range(iterations):
        _ = dict_str[lookup_key_str]
    str_time = (time.perf_counter_ns() - start) / iterations

    # Benchmark Symbol lookups
    lookup_key_symbol = Symbol("BTC/USDT")
    start = time.perf_counter_ns()
    for _ in range(iterations):
        _ = dict_interned[lookup_key_symbol]
    symbol_time = (time.perf_counter_ns() - start) / iterations

    return {
        "str_lookup_ns": str_time,
        "symbol_lookup_ns": symbol_time,
        "speedup": str_time / symbol_time if symbol_time > 0 else float("inf"),
    }


def bench_hash_computation(iterations: int = 1_000_000) -> dict[str, float]:
    """Benchmark hash computation."""
    clear_all_identifier_caches()

    test_str = "BTC/USDT"
    test_symbol = Symbol("BTC/USDT")

    # Benchmark str hash
    start = time.perf_counter_ns()
    for _ in range(iterations):
        _ = hash(test_str)
    str_time = (time.perf_counter_ns() - start) / iterations

    # Benchmark Symbol hash (pre-computed)
    start = time.perf_counter_ns()
    for _ in range(iterations):
        _ = hash(test_symbol)
    symbol_time = (time.perf_counter_ns() - start) / iterations

    return {
        "str_hash_ns": str_time,
        "symbol_hash_ns": symbol_time,
        "speedup": str_time / symbol_time if symbol_time > 0 else float("inf"),
    }


def bench_equality(iterations: int = 1_000_000) -> dict[str, float]:
    """Benchmark equality comparison."""
    clear_all_identifier_caches()

    str1 = "BTC/USDT"
    str2 = "BTC/USDT"
    symbol1 = Symbol("BTC/USDT")
    symbol2 = Symbol("BTC/USDT")

    # Benchmark str equality
    start = time.perf_counter_ns()
    for _ in range(iterations):
        _ = str1 == str2
    str_time = (time.perf_counter_ns() - start) / iterations

    # Benchmark Symbol equality (identity check)
    start = time.perf_counter_ns()
    for _ in range(iterations):
        _ = symbol1 == symbol2
    symbol_time = (time.perf_counter_ns() - start) / iterations

    return {
        "str_eq_ns": str_time,
        "symbol_eq_ns": symbol_time,
        "speedup": str_time / symbol_time if symbol_time > 0 else float("inf"),
    }


def bench_creation(iterations: int = 100_000) -> dict[str, float]:
    """Benchmark object creation."""
    # Create unique strings for fair comparison
    unique_strs = [f"SYMBOL_{i}" for i in range(iterations)]

    # Benchmark str creation (baseline - strings already exist)
    start = time.perf_counter_ns()
    for s in unique_strs:
        _ = s
    str_time = (time.perf_counter_ns() - start) / iterations

    # Benchmark Symbol creation (cold cache - first time)
    clear_all_identifier_caches()
    start = time.perf_counter_ns()
    for s in unique_strs:
        _ = Symbol(s)
    symbol_cold_time = (time.perf_counter_ns() - start) / iterations

    # Benchmark Symbol creation (warm cache - repeated)
    start = time.perf_counter_ns()
    for s in unique_strs:
        _ = Symbol(s)
    symbol_warm_time = (time.perf_counter_ns() - start) / iterations

    return {
        "str_create_ns": str_time,
        "symbol_cold_create_ns": symbol_cold_time,
        "symbol_warm_create_ns": symbol_warm_time,
        "warm_speedup": symbol_cold_time / symbol_warm_time if symbol_warm_time > 0 else float("inf"),
    }


def bench_memory() -> dict[str, Any]:
    """Benchmark memory usage."""
    import gc

    clear_all_identifier_caches()
    gc.collect()

    # Create many duplicate strings
    symbols_count = 10000
    duplicates = 100

    # Memory for strings (each duplicate is a new reference)
    strings = []
    for i in range(symbols_count):
        for _ in range(duplicates):
            strings.append(f"SYMBOL_{i}")

    # Memory for symbols (each duplicate returns same instance)
    clear_all_identifier_caches()
    symbols = []
    for i in range(symbols_count):
        for _ in range(duplicates):
            symbols.append(Symbol(f"SYMBOL_{i}"))

    # Count unique objects
    unique_strings = len(set(id(s) for s in strings))
    unique_symbols = len(set(id(s) for s in symbols))

    return {
        "total_references": symbols_count * duplicates,
        "unique_string_objects": unique_strings,
        "unique_symbol_objects": unique_symbols,
        "memory_reduction_factor": unique_strings / unique_symbols if unique_symbols > 0 else float("inf"),
    }


def bench_real_world_scenario(iterations: int = 100_000) -> dict[str, float]:
    """
    Benchmark a real-world scenario: message bus routing.

    Simulates looking up handlers for symbol-based subscriptions.
    """
    clear_all_identifier_caches()

    # Setup: 100 symbols, each with a handler
    symbols_str = [f"SYMBOL_{i}" for i in range(100)]
    symbols_interned = [Symbol(s) for s in symbols_str]

    handlers_str: dict[str, list[Any]] = {s: [lambda x: x] for s in symbols_str}
    handlers_interned: dict[Symbol, list[Any]] = {s: [lambda x: x] for s in symbols_interned}

    # Simulate receiving events for random symbols
    import random
    random.seed(42)
    event_symbols_str = [random.choice(symbols_str) for _ in range(iterations)]
    event_symbols_interned = [Symbol(s) for s in event_symbols_str]

    # Benchmark str-based routing
    start = time.perf_counter_ns()
    for sym in event_symbols_str:
        handlers = handlers_str.get(sym, [])
        for h in handlers:
            pass  # Would call handler
    str_time = (time.perf_counter_ns() - start) / iterations

    # Benchmark Symbol-based routing
    start = time.perf_counter_ns()
    for sym in event_symbols_interned:
        handlers = handlers_interned.get(sym, [])
        for h in handlers:
            pass
    symbol_time = (time.perf_counter_ns() - start) / iterations

    return {
        "str_routing_ns": str_time,
        "symbol_routing_ns": symbol_time,
        "speedup": str_time / symbol_time if symbol_time > 0 else float("inf"),
    }


def main():
    """Run all benchmarks and print results."""
    print("=" * 60)
    print("Identifier Interning Benchmarks (Issue #111)")
    print("=" * 60)
    print()

    # Hash computation
    print("1. Hash Computation (pre-computed vs computed)")
    print("-" * 40)
    results = bench_hash_computation()
    print(f"   str hash:    {results['str_hash_ns']:.1f} ns")
    print(f"   Symbol hash: {results['symbol_hash_ns']:.1f} ns")
    print(f"   Speedup:     {results['speedup']:.1f}x")
    print()

    # Hash lookup
    print("2. Dictionary Lookup")
    print("-" * 40)
    results = bench_hash_lookup()
    print(f"   str lookup:    {results['str_lookup_ns']:.1f} ns")
    print(f"   Symbol lookup: {results['symbol_lookup_ns']:.1f} ns")
    print(f"   Speedup:       {results['speedup']:.1f}x")
    print()

    # Equality
    print("3. Equality Comparison (identity vs value)")
    print("-" * 40)
    results = bench_equality()
    print(f"   str ==:    {results['str_eq_ns']:.1f} ns")
    print(f"   Symbol ==: {results['symbol_eq_ns']:.1f} ns")
    print(f"   Speedup:   {results['speedup']:.1f}x")
    print()

    # Creation
    print("4. Object Creation")
    print("-" * 40)
    results = bench_creation()
    print(f"   str (baseline):     {results['str_create_ns']:.1f} ns")
    print(f"   Symbol (cold):      {results['symbol_cold_create_ns']:.1f} ns")
    print(f"   Symbol (warm):      {results['symbol_warm_create_ns']:.1f} ns")
    print(f"   Warm/Cold speedup:  {results['warm_speedup']:.1f}x")
    print()

    # Memory
    print("5. Memory Usage (10K symbols x 100 references)")
    print("-" * 40)
    results = bench_memory()
    print(f"   Total references:   {results['total_references']:,}")
    print(f"   Unique str objects: {results['unique_string_objects']:,}")
    print(f"   Unique Symbol objs: {results['unique_symbol_objects']:,}")
    print(f"   Memory reduction:   {results['memory_reduction_factor']:.0f}x")
    print()

    # Real-world
    print("6. Real-World: Message Bus Routing (100 symbols)")
    print("-" * 40)
    results = bench_real_world_scenario()
    print(f"   str routing:    {results['str_routing_ns']:.1f} ns")
    print(f"   Symbol routing: {results['symbol_routing_ns']:.1f} ns")
    print(f"   Speedup:        {results['speedup']:.1f}x")
    print()

    print("=" * 60)
    print("Summary: Symbol interning provides significant speedup for")
    print("hash operations and memory reduction for repeated symbols.")
    print("=" * 60)


if __name__ == "__main__":
    main()
