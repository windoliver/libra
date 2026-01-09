#!/usr/bin/env python3
"""
Test TUI with Gateway integration.

This script launches the TUI connected to a PaperGateway to test
the Gateway + Fetcher integration (Issue #27).

Usage:
    uv run python scripts/test_tui_with_gateway.py
"""

import asyncio
from decimal import Decimal

from libra.gateways.paper_gateway import PaperGateway
from libra.tui.app import LibraApp


async def main() -> None:
    """Run TUI with Paper Gateway."""
    # Create Paper Gateway with initial balances via config
    config = {
        "initial_balance": {
            "USDT": Decimal("100000"),
            "BTC": Decimal("1.5"),
            "ETH": Decimal("10"),
        },
        "default_prices": {
            "BTC/USDT": Decimal("50000"),
            "ETH/USDT": Decimal("3000"),
        },
    }
    gateway = PaperGateway(config=config)

    # Connect gateway
    await gateway.connect()

    # Create and run TUI
    app = LibraApp(gateway=gateway, demo_mode=False)

    try:
        await app.run_async()
    finally:
        await gateway.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
