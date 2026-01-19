#!/usr/bin/env python3
"""Memory benchmark script for memray profiling (Issue #91).

This script exercises the core components to detect memory leaks:
- Message bus publish/subscribe
- Cache operations
- Gateway operations (paper trading)

Usage:
    # Direct run
    python scripts/benchmark_memory.py

    # With memray profiling
    python -m memray run -o profile.bin scripts/benchmark_memory.py
    python -m memray stats profile.bin
    python -m memray flamegraph profile.bin -o flamegraph.html
"""

from __future__ import annotations

import asyncio
import gc
import sys
from decimal import Decimal

# Add src to path for imports
sys.path.insert(0, "src")


async def benchmark_message_bus(iterations: int = 10000) -> dict:
    """Benchmark message bus memory usage."""
    from libra.core.bus import MessageBus

    bus = MessageBus()
    messages_processed = 0

    async def handler(msg):
        nonlocal messages_processed
        messages_processed += 1

    bus.subscribe("test.topic", handler)
    await bus.start()

    # Publish many messages
    for i in range(iterations):
        await bus.publish("test.topic", {"id": i, "data": f"message_{i}"})

    await bus.stop()

    return {
        "component": "MessageBus",
        "iterations": iterations,
        "messages_processed": messages_processed,
    }


async def benchmark_cache(iterations: int = 10000) -> dict:
    """Benchmark cache memory usage with churn."""
    from libra.core.cache import Cache
    from libra.gateways.protocol import (
        Balance,
        OrderResult,
        OrderSide,
        OrderStatus,
        OrderType,
        Position,
        PositionSide,
        Tick,
    )
    from libra.strategies.protocol import Bar

    cache = Cache(max_bars=1000, max_orders=1000)

    # Add and update many items (simulating real trading)
    for i in range(iterations):
        # Orders (will be evicted after 1000)
        order = OrderResult(
            order_id=f"order_{i}",
            symbol="BTC/USDT",
            status=OrderStatus.FILLED,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
            filled_amount=Decimal("0.1"),
            remaining_amount=Decimal("0"),
            average_price=Decimal("50000"),
            fee=Decimal("0.0001"),
            fee_currency="BTC",
            timestamp_ns=i * 1000000,
            client_order_id=f"client_{i}",
        )
        await cache.add_order(order)

        # Quotes (overwritten)
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal(f"{50000 + i % 100}"),
            ask=Decimal(f"{50001 + i % 100}"),
            last=Decimal(f"{50000 + i % 100}"),
            timestamp_ns=i * 1000000,
        )
        await cache.update_quote(tick)

        # Positions (updated)
        position = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=Decimal(f"{0.1 + (i % 10) * 0.01}"),
            entry_price=Decimal("50000"),
            current_price=Decimal(f"{50000 + i % 100}"),
            unrealized_pnl=Decimal(f"{i % 100}"),
            realized_pnl=Decimal("0"),
        )
        await cache.update_position(position)

        # Bars (trimmed to 1000)
        bar = Bar(
            symbol="BTC/USDT",
            timestamp_ns=i * 3600 * 1000000000,
            open=Decimal("49500"),
            high=Decimal("50500"),
            low=Decimal("49000"),
            close=Decimal(f"{50000 + i % 100}"),
            volume=Decimal("100"),
            timeframe="1h",
        )
        await cache.add_bar(bar)

    stats = cache.stats()
    return {
        "component": "Cache",
        "iterations": iterations,
        "orders": stats["orders"],
        "quotes": stats["quotes"],
        "positions": stats["positions"],
        "bar_series": stats["bar_series"],
    }


async def benchmark_paper_gateway(iterations: int = 1000) -> dict:
    """Benchmark paper gateway memory usage."""
    from libra.gateways.paper_gateway import PaperGateway
    from libra.gateways.protocol import OrderSide, OrderType

    gateway = PaperGateway(config={"initial_balances": {"USDT": 100000}})
    await gateway.connect()

    orders_submitted = 0

    for i in range(iterations):
        # Submit orders
        result = await gateway.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),
        )
        if result.is_success:
            orders_submitted += 1

        # Query state
        _ = await gateway.get_positions()
        _ = await gateway.get_balances()
        _ = gateway.get_open_orders()

    await gateway.disconnect()

    return {
        "component": "PaperGateway",
        "iterations": iterations,
        "orders_submitted": orders_submitted,
    }


async def main():
    """Run all memory benchmarks."""
    print("=" * 60)
    print("LIBRA Memory Benchmark")
    print("=" * 60)

    # Force garbage collection before starting
    gc.collect()

    results = []

    # Run benchmarks
    print("\n[1/3] Message Bus benchmark...")
    result = await benchmark_message_bus(10000)
    results.append(result)
    print(f"      Processed {result['messages_processed']} messages")

    gc.collect()

    print("\n[2/3] Cache benchmark...")
    result = await benchmark_cache(10000)
    results.append(result)
    print(f"      Orders: {result['orders']}, Quotes: {result['quotes']}")

    gc.collect()

    print("\n[3/3] Paper Gateway benchmark...")
    result = await benchmark_paper_gateway(1000)
    results.append(result)
    print(f"      Orders submitted: {result['orders_submitted']}")

    gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['component']}: {r['iterations']} iterations")

    print("\nBenchmark complete. Use memray to analyze memory profile.")
    print("  memray stats profile.bin")
    print("  memray flamegraph profile.bin -o flamegraph.html")


if __name__ == "__main__":
    asyncio.run(main())
