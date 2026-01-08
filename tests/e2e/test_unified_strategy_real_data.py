"""
E2E tests for Unified Strategy using REAL market data from Binance.

Demonstrates Issue #37: Same strategy code works in backtest and live.
Uses actual historical data to validate the unified strategy pattern.
"""

from __future__ import annotations

import json
import urllib.request
from decimal import Decimal
from typing import Any

import pytest

from libra.backtest import BacktestConfig, BacktestEngine, BacktestResult
from libra.clients.backtest_execution_client import SlippageModel
from libra.strategies.examples.unified_sma_cross import (
    UnifiedSMACross,
    UnifiedSMACrossConfig,
)
from libra.strategies.protocol import Bar


# =============================================================================
# Real Data Fetching (same as test_backtest_engine_real_data.py)
# =============================================================================


async def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
) -> list[dict[str, Any]]:
    """
    Fetch real OHLCV data from Binance public API.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
        interval: Timeframe (e.g., "1m", "5m", "1h", "1d")
        limit: Number of candles to fetch (max 1000)

    Returns:
        List of OHLCV dicts
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())

        klines = []
        for k in data:
            klines.append({
                "timestamp_ms": k[0],
                "open": Decimal(k[1]),
                "high": Decimal(k[2]),
                "low": Decimal(k[3]),
                "close": Decimal(k[4]),
                "volume": Decimal(k[5]),
            })
        return klines
    except Exception as e:
        pytest.skip(f"Could not fetch Binance data: {e}")
        return []


def klines_to_bars(klines: list[dict[str, Any]], symbol: str, timeframe: str) -> list[Bar]:
    """Convert Binance klines to Bar objects."""
    bars = []
    for k in klines:
        bars.append(
            Bar(
                symbol=symbol,
                timeframe=timeframe,
                timestamp_ns=k["timestamp_ms"] * 1_000_000,  # ms to ns
                open=k["open"],
                high=k["high"],
                low=k["low"],
                close=k["close"],
                volume=k["volume"],
            )
        )
    return bars


# =============================================================================
# E2E Tests for Unified Strategy
# =============================================================================


class TestUnifiedStrategyRealData:
    """E2E tests for UnifiedSMACross with real Binance data."""

    @pytest.fixture
    async def btc_bars(self) -> list[Bar]:
        """Fetch 500 hours of BTC/USDT data."""
        klines = await fetch_binance_klines("BTCUSDT", "1h", 500)
        return klines_to_bars(klines, "BTC/USDT", "1h")

    @pytest.fixture
    async def eth_bars(self) -> list[Bar]:
        """Fetch 500 hours of ETH/USDT data."""
        klines = await fetch_binance_klines("ETHUSDT", "1h", 500)
        return klines_to_bars(klines, "ETH/USDT", "1h")

    @pytest.mark.asyncio
    async def test_unified_sma_strategy_btc(self, btc_bars: list[Bar]) -> None:
        """Test UnifiedSMACross strategy on real BTC data."""
        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("5"),
            maker_fee_rate=Decimal("0.001"),
            taker_fee_rate=Decimal("0.001"),
            verbose=False,
        )

        engine = BacktestEngine(config)

        # Create unified strategy
        strategy = UnifiedSMACross(
            None,  # Gateway injected by engine
            UnifiedSMACrossConfig(
                symbol="BTC/USDT",
                timeframe="1h",
                fast_period=10,
                slow_period=30,
            ),
        )

        engine.add_strategy(strategy)
        engine.add_bars("BTC/USDT", btc_bars)

        result = await engine.run()

        # Validate result structure
        assert result is not None
        assert result.summary.strategy_name == "unified_sma_10_30"
        assert result.summary.symbol == "BTC/USDT"
        assert result.summary.bars_processed == 500
        assert len(result.equity_curve) == 500
        assert result.summary.initial_capital == Decimal("100000")

        # Print summary for visibility
        print(f"\nUnified SMA Strategy Results (BTC/USDT, 500 bars):")
        print(f"  Final Equity: ${result.summary.final_equity:,.2f}")
        print(f"  Total Return: {result.summary.total_return_pct * 100:.2f}%")
        print(f"  Sharpe Ratio: {result.summary.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.summary.max_drawdown_pct * 100:.2f}%")
        print(f"  Total Trades: {result.summary.total_trades}")
        print(f"  Win Rate: {result.summary.win_rate * 100:.1f}%")

    @pytest.mark.asyncio
    async def test_unified_strategy_eth(self, eth_bars: list[Bar]) -> None:
        """Test UnifiedSMACross strategy on real ETH data."""
        config = BacktestConfig(
            initial_capital=Decimal("50000"),
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("10"),
            verbose=False,
        )

        engine = BacktestEngine(config)
        strategy = UnifiedSMACross(
            None,
            UnifiedSMACrossConfig(
                symbol="ETH/USDT",
                timeframe="1h",
                fast_period=5,
                slow_period=20,
            ),
        )

        engine.add_strategy(strategy)
        engine.add_bars("ETH/USDT", eth_bars)

        result = await engine.run()

        assert result is not None
        assert result.summary.strategy_name == "unified_sma_5_20"
        assert result.summary.symbol == "ETH/USDT"
        assert result.summary.bars_processed == 500

        print(f"\nUnified SMA Strategy Results (ETH/USDT, 500 bars):")
        print(f"  Final Equity: ${result.summary.final_equity:,.2f}")
        print(f"  Total Return: {result.summary.total_return_pct * 100:.2f}%")
        print(f"  Total Trades: {result.summary.total_trades}")

    @pytest.mark.asyncio
    async def test_unified_strategy_different_periods(self, btc_bars: list[Bar]) -> None:
        """Test UnifiedSMACross with different period configurations."""
        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("5"),
            verbose=False,
        )

        period_configs = [
            (5, 15),
            (10, 30),
            (20, 50),
        ]

        results: list[BacktestResult] = []

        for fast, slow in period_configs:
            engine = BacktestEngine(config)
            strategy = UnifiedSMACross(
                None,
                UnifiedSMACrossConfig(
                    symbol="BTC/USDT",
                    fast_period=fast,
                    slow_period=slow,
                ),
            )
            engine.add_strategy(strategy)
            engine.add_bars("BTC/USDT", btc_bars)

            result = await engine.run()
            results.append(result)

        # Print comparison
        print("\n" + "=" * 70)
        print("Unified SMA Strategy Period Comparison (BTC/USDT, 500 bars)")
        print("=" * 70)
        print(f"{'Config':<20} {'Return %':>10} {'Sharpe':>10} {'MaxDD %':>10} {'Trades':>8}")
        print("-" * 70)

        for (fast, slow), result in zip(period_configs, results):
            s = result.summary
            print(
                f"SMA({fast},{slow}){' ':<10} "
                f"{s.total_return_pct * 100:>10.2f} "
                f"{s.sharpe_ratio:>10.2f} "
                f"{s.max_drawdown_pct * 100:>10.2f} "
                f"{s.total_trades:>8}"
            )

        print("=" * 70)

        # Validate all ran successfully
        for result in results:
            assert result.summary.bars_processed == 500

    @pytest.mark.asyncio
    async def test_unified_vs_legacy_real_data(self, btc_bars: list[Bar]) -> None:
        """Compare unified and legacy strategies on real data."""
        from libra.strategies.base import BaseStrategy as LegacyStrategy
        from libra.strategies.protocol import Signal, SignalType, StrategyConfig

        class LegacySMA(LegacyStrategy):
            """Legacy SMA strategy for comparison."""

            def __init__(self, config: StrategyConfig, fast: int = 10, slow: int = 30) -> None:
                super().__init__(config)
                self._closes: list[Decimal] = []
                self._fast = fast
                self._slow = slow
                self._has_pos = False

            @property
            def name(self) -> str:
                return f"legacy_sma_{self._fast}_{self._slow}"

            def on_bar(self, bar: Bar) -> Signal | None:
                self._closes.append(bar.close)
                if len(self._closes) < self._slow:
                    return None

                fast_sma = sum(self._closes[-self._fast:]) / self._fast
                slow_sma = sum(self._closes[-self._slow:]) / self._slow

                if fast_sma > slow_sma and not self._has_pos:
                    self._has_pos = True
                    return self._long()
                elif fast_sma < slow_sma and self._has_pos:
                    self._has_pos = False
                    return self._close_long()
                return None

        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("5"),
            verbose=False,
        )

        # Run legacy strategy
        engine1 = BacktestEngine(config)
        legacy = LegacySMA(StrategyConfig(symbol="BTC/USDT", timeframe="1h"))
        engine1.add_strategy(legacy)
        engine1.add_bars("BTC/USDT", btc_bars)
        result1 = await engine1.run()

        # Run unified strategy
        engine2 = BacktestEngine(config)
        unified = UnifiedSMACross(
            None,
            UnifiedSMACrossConfig(
                symbol="BTC/USDT",
                fast_period=10,
                slow_period=30,
            ),
        )
        engine2.add_strategy(unified)
        engine2.add_bars("BTC/USDT", btc_bars)
        result2 = await engine2.run()

        # Print comparison
        print("\n" + "=" * 70)
        print("Legacy vs Unified Strategy Comparison (Real BTC Data)")
        print("=" * 70)
        print(f"{'Strategy':<25} {'Return %':>10} {'Trades':>8} {'Win Rate':>10}")
        print("-" * 70)
        print(
            f"{'Legacy SMA(10,30)':<25} "
            f"{result1.summary.total_return_pct * 100:>10.2f} "
            f"{result1.summary.total_trades:>8} "
            f"{result1.summary.win_rate * 100:>10.1f}%"
        )
        print(
            f"{'Unified SMA(10,30)':<25} "
            f"{result2.summary.total_return_pct * 100:>10.2f} "
            f"{result2.summary.total_trades:>8} "
            f"{result2.summary.win_rate * 100:>10.1f}%"
        )
        print("=" * 70)

        # Both should process same bars
        assert result1.summary.bars_processed == result2.summary.bars_processed == 500
