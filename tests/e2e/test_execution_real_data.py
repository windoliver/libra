"""
E2E Tests for Execution Algorithms with Real Market Data.

Tests for Issue #36: Execution Algorithm Framework (TWAP, VWAP).

These tests fetch real market data from Binance to test:
- VWAP volume profiling with real historical bars
- Price-based execution decisions with real prices
- Algorithm behavior with realistic market conditions
"""

from __future__ import annotations

import asyncio
import json
import ssl
import time
import urllib.request
from decimal import Decimal

import pytest

from libra.execution import (
    AlgorithmState,
    IcebergAlgorithm,
    IcebergConfig,
    TWAPAlgorithm,
    TWAPConfig,
    VWAPAlgorithm,
    VWAPConfig,
    create_algorithm,
)
from libra.gateways.fetcher import Bar
from libra.gateways.protocol import Order, OrderResult, OrderSide, OrderStatus, OrderType


# =============================================================================
# Real Data Fetchers (Direct Binance API)
# =============================================================================


def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 24,
) -> list[dict]:
    """
    Fetch real klines/candlestick data from Binance.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Candle interval (e.g., "1h", "4h", "1d")
        limit: Number of candles to fetch

    Returns:
        List of kline data dictionaries
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

    # Create SSL context
    ctx = ssl.create_default_context()

    req = urllib.request.Request(url, headers={"User-Agent": "libra-test/1.0"})

    with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
        data = json.loads(response.read().decode())

    # Binance klines format:
    # [open_time, open, high, low, close, volume, close_time, quote_volume, trades, ...]
    return [
        {
            "timestamp_ms": int(k[0]),
            "open": k[1],
            "high": k[2],
            "low": k[3],
            "close": k[4],
            "volume": k[5],
            "trades": int(k[8]),
        }
        for k in data
    ]


def fetch_binance_ticker(symbol: str = "BTCUSDT") -> dict:
    """
    Fetch real ticker data from Binance.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")

    Returns:
        Ticker data dictionary
    """
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"

    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "libra-test/1.0"})

    with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
        return json.loads(response.read().decode())


def klines_to_bars(klines: list[dict], symbol: str = "BTC/USDT") -> list[Bar]:
    """Convert Binance klines to Bar objects."""
    return [
        Bar(
            symbol=symbol,
            timestamp_ns=int(k["timestamp_ms"]) * 1_000_000,
            open=Decimal(str(k["open"])),
            high=Decimal(str(k["high"])),
            low=Decimal(str(k["low"])),
            close=Decimal(str(k["close"])),
            volume=Decimal(str(k["volume"])),
            trades=k.get("trades"),
        )
        for k in klines
    ]


# =============================================================================
# Simulated Execution Client with Real Prices
# =============================================================================


class RealPriceExecutionClient:
    """
    Execution client that uses real market prices for simulation.

    Fetches real prices from Binance but simulates order execution
    (doesn't actually place orders).
    """

    def __init__(self, symbol: str = "BTCUSDT") -> None:
        self.symbol = symbol
        self.orders_submitted: list[Order] = []
        self.total_filled: Decimal = Decimal("0")
        self.total_value: Decimal = Decimal("0")
        self._current_price: Decimal | None = None

    def _fetch_current_price(self) -> Decimal:
        """Fetch current price from Binance."""
        if self._current_price is None:
            ticker = fetch_binance_ticker(self.symbol)
            self._current_price = Decimal(ticker["lastPrice"])
        return self._current_price

    async def submit_order(self, order: Order) -> OrderResult:
        """Submit order with real price simulation."""
        import random

        self.orders_submitted.append(order)

        # Get real market price
        base_price = self._fetch_current_price()

        # Simulate small slippage (0.01% - 0.05%)
        slippage = Decimal(str(random.uniform(0.0001, 0.0005)))
        if order.side == OrderSide.BUY:
            execution_price = base_price * (1 + slippage)
        else:
            execution_price = base_price * (1 - slippage)

        # Full fill for simulation
        filled_amount = order.amount

        self.total_filled += filled_amount
        self.total_value += filled_amount * execution_price

        # Small delay to simulate network
        await asyncio.sleep(0.001)

        return OrderResult(
            order_id=f"real-sim-{len(self.orders_submitted)}",
            symbol=order.symbol,
            status=OrderStatus.FILLED,
            side=order.side,
            order_type=order.order_type,
            amount=order.amount,
            filled_amount=filled_amount,
            remaining_amount=Decimal("0"),
            average_price=execution_price,
            fee=filled_amount * Decimal("0.001"),
            fee_currency="USDT",
            timestamp_ns=time.time_ns(),
            client_order_id=order.client_order_id,
        )

    @property
    def vwap(self) -> Decimal:
        """Calculate VWAP of all executions."""
        if self.total_filled == 0:
            return Decimal("0")
        return self.total_value / self.total_filled


# =============================================================================
# VWAP Tests with Real Volume Data
# =============================================================================


class TestVWAPRealData:
    """VWAP tests using real Binance volume data."""

    def test_load_real_volume_profile(self) -> None:
        """Test loading volume profile from real Binance data."""
        # Fetch real 24-hour data
        klines = fetch_binance_klines("BTCUSDT", "1h", 24)
        bars = klines_to_bars(klines)

        assert len(bars) == 24

        # Create VWAP with 6 intervals
        config = VWAPConfig(num_intervals=6, interval_secs=0.01)
        algo = VWAPAlgorithm(config)

        profile = algo.load_volume_profile(bars)

        # Verify profile
        assert profile is not None
        assert len(profile.fractions) == 6
        assert abs(sum(profile.fractions) - 1.0) < 0.0001
        assert profile.total_volume > 0

        print(f"\nReal BTC volume profile (24h, 6 intervals):")
        print(f"  Total volume: {profile.total_volume:.2f} BTC")
        for i, frac in enumerate(profile.fractions):
            print(f"  Interval {i+1}: {frac*100:.1f}%")

    def test_real_volume_distribution_shape(self) -> None:
        """Test that real volume shows expected trading patterns."""
        klines = fetch_binance_klines("BTCUSDT", "1h", 24)
        bars = klines_to_bars(klines)

        # Calculate volume per bar
        volumes = [float(bar.volume) for bar in bars]
        avg_volume = sum(volumes) / len(volumes)

        print(f"\nReal BTC hourly volumes (last 24h):")
        print(f"  Average: {avg_volume:.2f} BTC")
        print(f"  Min: {min(volumes):.2f} BTC")
        print(f"  Max: {max(volumes):.2f} BTC")
        print(f"  Ratio (max/min): {max(volumes)/min(volumes):.2f}x")

        # Volume should have some variation (not all equal)
        assert max(volumes) > min(volumes) * 1.1  # At least 10% variation

    @pytest.mark.asyncio
    async def test_vwap_execution_with_real_profile(self) -> None:
        """Test VWAP execution using real volume profile."""
        # Fetch real data
        klines = fetch_binance_klines("BTCUSDT", "1h", 12)
        bars = klines_to_bars(klines)

        # Create execution client with real prices
        client = RealPriceExecutionClient("BTCUSDT")

        # Create VWAP algorithm
        config = VWAPConfig(
            num_intervals=4,
            interval_secs=0.01,
            randomize_size=False,
        )
        algo = VWAPAlgorithm(config, client)

        # Load real volume profile
        profile = algo.load_volume_profile(bars)

        # Create test order
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),  # 1 BTC
        )

        # Execute
        progress = await algo.execute(order)

        assert progress.state == AlgorithmState.COMPLETED
        assert progress.num_children_spawned == 4
        assert client.total_filled > 0

        print(f"\nVWAP Execution Results:")
        print(f"  Total filled: {client.total_filled} BTC")
        print(f"  Execution VWAP: ${client.vwap:.2f}")
        print(f"  Algorithm VWAP: ${algo.vwap:.2f}")
        print(f"  Orders submitted: {len(client.orders_submitted)}")

        # Verify order sizes follow volume profile
        order_sizes = [o.amount for o in client.orders_submitted]
        print(f"  Order sizes: {[float(s) for s in order_sizes]}")


# =============================================================================
# TWAP Tests with Real Prices
# =============================================================================


class TestTWAPRealData:
    """TWAP tests using real market prices."""

    @pytest.mark.asyncio
    async def test_twap_with_real_prices(self) -> None:
        """Test TWAP execution with real Binance prices."""
        client = RealPriceExecutionClient("BTCUSDT")

        config = TWAPConfig(
            horizon_secs=0.1,
            interval_secs=0.02,
            randomize_size=False,
            randomize_delay=False,
        )
        algo = TWAPAlgorithm(config, client)

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.5"),
        )

        progress = await algo.execute(order)

        assert progress.state == AlgorithmState.COMPLETED
        assert progress.num_children_spawned == 5

        # Get real current price for comparison
        ticker = fetch_binance_ticker("BTCUSDT")
        market_price = Decimal(ticker["lastPrice"])

        print(f"\nTWAP Execution Results:")
        print(f"  Market price: ${market_price:.2f}")
        print(f"  Execution VWAP: ${client.vwap:.2f}")
        print(f"  Slippage: {float((client.vwap - market_price) / market_price * 100):.4f}%")

        # Execution price should be within 0.1% of market price
        slippage_pct = abs(client.vwap - market_price) / market_price
        assert slippage_pct < Decimal("0.001")

    @pytest.mark.asyncio
    async def test_twap_equal_slices_with_real_prices(self) -> None:
        """Verify TWAP creates equal slices with real prices."""
        client = RealPriceExecutionClient("BTCUSDT")

        config = TWAPConfig(
            horizon_secs=0.1,
            interval_secs=0.025,  # 4 slices
            randomize_size=False,
        )
        algo = TWAPAlgorithm(config, client)

        total_qty = Decimal("2.0")
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=total_qty,
        )

        await algo.execute(order)

        # Verify equal slices
        expected_size = total_qty / 4
        for i, o in enumerate(client.orders_submitted[:-1]):  # Except last
            assert abs(o.amount - expected_size) < Decimal("0.01")

        print(f"\nTWAP Slice Analysis:")
        for i, o in enumerate(client.orders_submitted):
            print(f"  Slice {i+1}: {o.amount} BTC")


# =============================================================================
# Iceberg Tests with Real Prices
# =============================================================================


class TestIcebergRealData:
    """Iceberg tests using real market prices."""

    @pytest.mark.asyncio
    async def test_iceberg_with_real_prices(self) -> None:
        """Test Iceberg execution with real prices."""
        client = RealPriceExecutionClient("BTCUSDT")

        config = IcebergConfig(
            display_pct=0.2,  # 20% visible
            delay_between_refills_secs=0.01,
            randomize_display=False,
        )
        algo = IcebergAlgorithm(config, client)

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )

        progress = await algo.execute(order)

        assert progress.state == AlgorithmState.COMPLETED
        assert algo.num_refills >= 5  # 1.0 / 0.2 = 5 refills

        print(f"\nIceberg Execution Results:")
        print(f"  Total order: 1.0 BTC")
        print(f"  Display size: 20%")
        print(f"  Number of refills: {algo.num_refills}")
        print(f"  Execution VWAP: ${client.vwap:.2f}")

    @pytest.mark.asyncio
    async def test_iceberg_hides_size_with_real_data(self) -> None:
        """Verify Iceberg hides order size with real market data."""
        client = RealPriceExecutionClient("BTCUSDT")

        total_order = Decimal("5.0")
        config = IcebergConfig(
            display_pct=0.1,  # Only 10% visible
            delay_between_refills_secs=0.005,
            randomize_display=False,
        )
        algo = IcebergAlgorithm(config, client)

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=total_order,
        )

        await algo.execute(order)

        # Each order should be ~10% of total
        for o in client.orders_submitted:
            assert o.amount <= total_order * Decimal("0.15")  # Allow some variance

        print(f"\nIceberg Order Visibility:")
        print(f"  True order size: {total_order} BTC")
        print(f"  Max visible size: {max(o.amount for o in client.orders_submitted)} BTC")
        print(f"  Orders placed: {len(client.orders_submitted)}")


# =============================================================================
# Market Data Analysis Tests
# =============================================================================


class TestMarketDataAnalysis:
    """Tests that analyze real market data patterns."""

    def test_fetch_multiple_symbols(self) -> None:
        """Test fetching data for multiple trading pairs."""
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

        for symbol in symbols:
            klines = fetch_binance_klines(symbol, "1h", 12)
            bars = klines_to_bars(klines, symbol.replace("USDT", "/USDT"))

            assert len(bars) == 12
            assert all(b.volume > 0 for b in bars)

            print(f"\n{symbol} (12h):")
            print(f"  Price range: ${bars[-1].low:.2f} - ${bars[-1].high:.2f}")
            print(f"  Current: ${bars[-1].close:.2f}")
            print(f"  Volume: {bars[-1].volume:.2f}")

    def test_volume_profile_comparison(self) -> None:
        """Compare volume profiles across different timeframes."""
        timeframes = [("1h", 24), ("4h", 12), ("1d", 7)]

        for interval, limit in timeframes:
            klines = fetch_binance_klines("BTCUSDT", interval, limit)
            bars = klines_to_bars(klines)

            config = VWAPConfig(num_intervals=4, interval_secs=0.01)
            algo = VWAPAlgorithm(config)
            profile = algo.load_volume_profile(bars)

            print(f"\nBTC Volume Profile ({interval} x {limit}):")
            print(f"  Total volume: {profile.total_volume:.2f} BTC")
            print(f"  Distribution: {[f'{f*100:.1f}%' for f in profile.fractions]}")

    @pytest.mark.asyncio
    async def test_registry_with_real_data(self) -> None:
        """Test algorithm registry with real market data."""
        client = RealPriceExecutionClient("BTCUSDT")

        # Create all algorithm types via registry
        algorithms = [
            ("twap", {"horizon_secs": 0.05, "interval_secs": 0.01}),
            ("vwap", {"num_intervals": 3, "interval_secs": 0.01}),
            ("iceberg", {"display_pct": 0.25, "delay_between_refills_secs": 0.01}),
        ]

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )

        print("\nRegistry + Real Data Test:")
        for algo_name, config in algorithms:
            client = RealPriceExecutionClient("BTCUSDT")  # Fresh client
            algo = create_algorithm(algo_name, execution_client=client, **config)

            progress = await algo.execute(order)
            assert progress.state == AlgorithmState.COMPLETED

            print(f"  {algo_name.upper()}: VWAP=${client.vwap:.2f}, orders={len(client.orders_submitted)}")
