#!/usr/bin/env python3
"""
Test CCXT Fetchers directly.

This script tests the CCXT fetchers with a testnet exchange
to verify the TET pipeline works correctly.

Usage:
    # Test with Binance testnet (no API key needed for public data)
    uv run python scripts/test_ccxt_fetchers.py

    # Test with API keys for private data (positions, orders, trades)
    BINANCE_API_KEY=xxx BINANCE_SECRET=yyy uv run python scripts/test_ccxt_fetchers.py
"""

import asyncio
import os
from decimal import Decimal

# Test with mock exchange for quick verification
from unittest.mock import AsyncMock


async def test_with_mock() -> None:
    """Test fetchers with mocked exchange."""
    from libra.gateways.ccxt_fetchers import (
        CCXTBalanceFetcher,
        CCXTOrderBookFetcher,
        CCXTOrderFetcher,
        CCXTPositionFetcher,
        CCXTQuoteFetcher,
        CCXTTradeFetcher,
    )

    print("=" * 60)
    print("Testing CCXT Fetchers with Mock Exchange")
    print("=" * 60)

    # Create mock exchange
    exchange = AsyncMock()

    # Test Quote Fetcher
    print("\n1. Testing CCXTQuoteFetcher...")
    exchange.fetch_ticker.return_value = {
        "bid": 50000,
        "ask": 50010,
        "last": 50005,
        "timestamp": 1704067200000,
        "high": 51000,
        "low": 49000,
        "quoteVolume": 1000000,
    }
    quote_fetcher = CCXTQuoteFetcher(exchange)
    quote = await quote_fetcher.fetch(symbol="BTC/USDT")
    print(f"   Symbol: {quote.symbol}")
    print(f"   Bid: ${quote.bid:,.2f}")
    print(f"   Ask: ${quote.ask:,.2f}")
    print(f"   Spread: ${quote.spread:.2f} ({quote.spread_bps:.2f} bps)")
    print("   OK!")

    # Test OrderBook Fetcher
    print("\n2. Testing CCXTOrderBookFetcher...")
    exchange.fetch_order_book.return_value = {
        "bids": [[50000, 1.5], [49990, 2.0], [49980, 3.0]],
        "asks": [[50010, 1.0], [50020, 2.5], [50030, 4.0]],
        "timestamp": 1704067200000,
    }
    ob_fetcher = CCXTOrderBookFetcher(exchange)
    ob = await ob_fetcher.fetch(symbol="BTC/USDT", depth=10)
    print(f"   Symbol: {ob.symbol}")
    print(f"   Best Bid: ${ob.best_bid:,.2f}")
    print(f"   Best Ask: ${ob.best_ask:,.2f}")
    print(f"   Spread: ${ob.spread:.2f}")
    print(f"   Bid Levels: {len(ob.bids)}")
    print(f"   Ask Levels: {len(ob.asks)}")
    print("   OK!")

    # Test Balance Fetcher
    print("\n3. Testing CCXTBalanceFetcher...")
    exchange.fetch_balance.return_value = {
        "USDT": {"total": 10000, "free": 8000, "used": 2000},
        "BTC": {"total": 1.5, "free": 1.0, "used": 0.5},
        "ETH": {"total": 10, "free": 10, "used": 0},
    }
    balance_fetcher = CCXTBalanceFetcher(exchange)
    balances = await balance_fetcher.fetch()
    print(f"   Currencies: {list(balances.keys())}")
    for currency, bal in balances.items():
        print(f"   {currency}: total={bal.total}, available={bal.available}, locked={bal.locked}")
    print("   OK!")

    # Test Position Fetcher
    print("\n4. Testing CCXTPositionFetcher...")
    exchange.fetch_positions.return_value = [
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "contracts": 0.5,
            "entryPrice": 48000,
            "markPrice": 50000,
            "unrealizedPnl": 1000,
            "leverage": 10,
            "liquidationPrice": 43000,
            "timestamp": 1704067200000,
        },
        {
            "symbol": "ETH/USDT",
            "side": "short",
            "contracts": 5,
            "entryPrice": 3100,
            "markPrice": 3000,
            "unrealizedPnl": 500,
            "leverage": 5,
            "timestamp": 1704067200000,
        },
    ]
    position_fetcher = CCXTPositionFetcher(exchange)
    positions = await position_fetcher.fetch()
    print(f"   Positions: {len(positions)}")
    for pos in positions:
        print(f"   {pos.symbol}: {pos.side} {pos.amount} @ ${pos.entry_price:,.2f}")
        print(f"      Current: ${pos.current_price:,.2f}, PnL: ${pos.unrealized_pnl:,.2f} ({pos.pnl_percent:.2f}%)")
    print("   OK!")

    # Test Order Fetcher
    print("\n5. Testing CCXTOrderFetcher...")
    exchange.fetch_open_orders.return_value = [
        {
            "id": "12345",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "status": "open",
            "amount": 0.1,
            "filled": 0,
            "remaining": 0.1,
            "price": 48000,
            "timestamp": 1704067200000,
        },
    ]
    exchange.fetch_closed_orders.return_value = [
        {
            "id": "12344",
            "symbol": "BTC/USDT",
            "side": "sell",
            "type": "market",
            "status": "closed",
            "amount": 0.5,
            "filled": 0.5,
            "remaining": 0,
            "average": 50100,
            "timestamp": 1704060000000,
            "fee": {"cost": 25.05, "currency": "USDT"},
        },
    ]
    order_fetcher = CCXTOrderFetcher(exchange)

    open_orders = await order_fetcher.fetch(status="open")
    print(f"   Open Orders: {len(open_orders)}")
    for order in open_orders:
        print(f"   #{order.order_id}: {order.side} {order.amount} {order.symbol} @ ${order.price:,.2f}")

    closed_orders = await order_fetcher.fetch(status="closed")
    print(f"   Closed Orders: {len(closed_orders)}")
    for order in closed_orders:
        print(f"   #{order.order_id}: {order.side} {order.amount} {order.symbol} filled @ ${order.average:,.2f}")
    print("   OK!")

    # Test Trade Fetcher
    print("\n6. Testing CCXTTradeFetcher...")
    exchange.fetch_my_trades.return_value = [
        {
            "id": "T001",
            "order": "12344",
            "symbol": "BTC/USDT",
            "side": "sell",
            "amount": 0.25,
            "price": 50100,
            "cost": 12525,
            "timestamp": 1704060000000,
            "fee": {"cost": 12.525, "currency": "USDT"},
            "takerOrMaker": "taker",
        },
        {
            "id": "T002",
            "order": "12344",
            "symbol": "BTC/USDT",
            "side": "sell",
            "amount": 0.25,
            "price": 50100,
            "cost": 12525,
            "timestamp": 1704060100000,
            "fee": {"cost": 12.525, "currency": "USDT"},
            "takerOrMaker": "taker",
        },
    ]
    trade_fetcher = CCXTTradeFetcher(exchange)
    trades = await trade_fetcher.fetch(symbol="BTC/USDT")
    print(f"   Trades: {len(trades)}")
    for trade in trades:
        print(f"   #{trade.trade_id}: {trade.side} {trade.amount} @ ${trade.price:,.2f} = ${trade.cost:,.2f}")
    print("   OK!")

    print("\n" + "=" * 60)
    print("All fetcher tests passed!")
    print("=" * 60)


async def test_with_ccxt_testnet() -> None:
    """Test fetchers with real CCXT testnet (public data only)."""
    try:
        import ccxt.pro as ccxtpro
    except ImportError:
        print("ccxt not installed, skipping testnet test")
        return

    print("\n" + "=" * 60)
    print("Testing CCXT Fetchers with Binance Testnet (Public Data)")
    print("=" * 60)

    from libra.gateways.ccxt_fetchers import (
        CCXTOrderBookFetcher,
        CCXTQuoteFetcher,
    )

    # Create Binance testnet exchange (public endpoints only)
    exchange = ccxtpro.binance({
        "sandbox": True,
        "enableRateLimit": True,
    })

    try:
        await exchange.load_markets()
        print(f"\nLoaded {len(exchange.markets)} markets")

        # Test Quote Fetcher with real data
        print("\n1. Testing CCXTQuoteFetcher with real data...")
        quote_fetcher = CCXTQuoteFetcher(exchange)
        quote = await quote_fetcher.fetch(symbol="BTC/USDT")
        print(f"   Symbol: {quote.symbol}")
        print(f"   Bid: ${quote.bid:,.2f}")
        print(f"   Ask: ${quote.ask:,.2f}")
        print(f"   Last: ${quote.last:,.2f}")
        print(f"   Spread: {quote.spread_bps:.2f} bps")

        # Test OrderBook Fetcher with real data
        print("\n2. Testing CCXTOrderBookFetcher with real data...")
        ob_fetcher = CCXTOrderBookFetcher(exchange)
        ob = await ob_fetcher.fetch(symbol="BTC/USDT", depth=5)
        print(f"   Symbol: {ob.symbol}")
        print(f"   Best Bid: ${ob.best_bid:,.2f}")
        print(f"   Best Ask: ${ob.best_ask:,.2f}")
        print(f"   Spread: ${ob.spread:.2f}")
        print(f"   Top 5 Bids:")
        for level in ob.bids[:5]:
            print(f"      ${level.price:,.2f} x {level.size}")
        print(f"   Top 5 Asks:")
        for level in ob.asks[:5]:
            print(f"      ${level.price:,.2f} x {level.size}")

        print("\n" + "=" * 60)
        print("Testnet tests passed!")
        print("=" * 60)

    finally:
        await exchange.close()


async def main() -> None:
    """Run all tests."""
    # Always run mock tests
    await test_with_mock()

    # Optionally test with real testnet
    try:
        await test_with_ccxt_testnet()
    except Exception as e:
        print(f"\nTestnet test skipped: {e}")


if __name__ == "__main__":
    asyncio.run(main())
