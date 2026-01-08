"""
End-to-End tests for BacktestEngine using REAL market data from Binance.

Tests the complete backtest pipeline with actual historical price data.
Validates:
- Data loading and processing
- Strategy signal generation
- Order execution simulation
- Metrics calculation accuracy
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from libra.backtest import BacktestConfig, BacktestEngine, BacktestResult
from libra.clients.backtest_execution_client import SlippageModel
from libra.strategies.base import BaseStrategy
from libra.strategies.protocol import Bar, Signal, SignalType, StrategyConfig


# =============================================================================
# Real Data Fetching
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
# Test Strategies
# =============================================================================


class SimpleSMAStrategy(BaseStrategy):
    """Simple Moving Average crossover strategy."""

    def __init__(self, config: StrategyConfig, fast_period: int = 10, slow_period: int = 30) -> None:
        super().__init__(config)
        self._closes: list[Decimal] = []
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._position = 0  # 0: no position, 1: long

    @property
    def name(self) -> str:
        return f"sma_{self._fast_period}_{self._slow_period}"

    def on_bar(self, bar: Bar) -> Signal | None:
        self._closes.append(bar.close)

        if len(self._closes) < self._slow_period:
            return None

        # Calculate SMAs
        fast_sma = sum(self._closes[-self._fast_period:]) / self._fast_period
        slow_sma = sum(self._closes[-self._slow_period:]) / self._slow_period

        # Generate signals on crossover
        if fast_sma > slow_sma and self._position == 0:
            self._position = 1
            return self._long(price=bar.close)
        elif fast_sma < slow_sma and self._position == 1:
            self._position = 0
            return self._close_long(price=bar.close)

        return None

    def on_reset(self) -> None:
        super().on_reset()
        self._closes.clear()
        self._position = 0


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy and hold strategy for baseline comparison."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._bought = False

    @property
    def name(self) -> str:
        return "buy_and_hold"

    def on_bar(self, bar: Bar) -> Signal | None:
        if not self._bought:
            self._bought = True
            return self._long(price=bar.close)
        return None

    def on_reset(self) -> None:
        super().on_reset()
        self._bought = False


class MomentumStrategy(BaseStrategy):
    """Momentum strategy based on ROC (Rate of Change)."""

    def __init__(self, config: StrategyConfig, lookback: int = 20, threshold: float = 0.02) -> None:
        super().__init__(config)
        self._closes: list[Decimal] = []
        self._lookback = lookback
        self._threshold = Decimal(str(threshold))
        self._position = 0

    @property
    def name(self) -> str:
        return f"momentum_{self._lookback}"

    def on_bar(self, bar: Bar) -> Signal | None:
        self._closes.append(bar.close)

        if len(self._closes) < self._lookback + 1:
            return None

        # Calculate Rate of Change
        old_price = self._closes[-self._lookback - 1]
        roc = (bar.close - old_price) / old_price

        # Generate signals
        if roc > self._threshold and self._position == 0:
            self._position = 1
            return self._long(price=bar.close)
        elif roc < -self._threshold and self._position == 1:
            self._position = 0
            return self._close_long(price=bar.close)

        return None

    def on_reset(self) -> None:
        super().on_reset()
        self._closes.clear()
        self._position = 0


# =============================================================================
# E2E Tests
# =============================================================================


class TestBacktestEngineRealData:
    """E2E tests for BacktestEngine with real Binance data."""

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
    async def test_sma_strategy_btc(self, btc_bars: list[Bar]) -> None:
        """Test SMA crossover strategy on BTC data."""
        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("5"),
            maker_fee_rate=Decimal("0.001"),
            taker_fee_rate=Decimal("0.001"),
            verbose=False,
        )

        engine = BacktestEngine(config)
        strategy = SimpleSMAStrategy(
            StrategyConfig(symbol="BTC/USDT", timeframe="1h"),
            fast_period=10,
            slow_period=30,
        )
        engine.add_strategy(strategy)
        engine.add_bars("BTC/USDT", btc_bars)

        result = await engine.run()

        # Validate result structure
        assert result is not None
        assert result.summary.strategy_name == "sma_10_30"
        assert result.summary.symbol == "BTC/USDT"
        assert result.summary.bars_processed == 500
        assert len(result.equity_curve) == 500
        assert result.summary.initial_capital == Decimal("100000")

        # Print summary for visibility
        print(f"\nSMA Strategy Results (BTC/USDT, 500 bars):")
        print(f"  Final Equity: ${result.summary.final_equity:,.2f}")
        print(f"  Total Return: {result.summary.total_return_pct * 100:.2f}%")
        print(f"  Sharpe Ratio: {result.summary.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.summary.max_drawdown_pct * 100:.2f}%")
        print(f"  Total Trades: {result.summary.total_trades}")
        print(f"  Win Rate: {result.summary.win_rate * 100:.1f}%")

    @pytest.mark.asyncio
    async def test_buy_and_hold_btc(self, btc_bars: list[Bar]) -> None:
        """Test buy and hold strategy on BTC data."""
        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            slippage_model=SlippageModel.NONE,
            verbose=False,
        )

        engine = BacktestEngine(config)
        strategy = BuyAndHoldStrategy(
            StrategyConfig(symbol="BTC/USDT", timeframe="1h")
        )
        engine.add_strategy(strategy)
        engine.add_bars("BTC/USDT", btc_bars)

        result = await engine.run()

        assert result is not None
        assert result.summary.bars_processed == 500
        # Buy and hold should have exactly 1 trade (entry)
        assert result.summary.total_trades <= 1

        print(f"\nBuy & Hold Results (BTC/USDT, 500 bars):")
        print(f"  Final Equity: ${result.summary.final_equity:,.2f}")
        print(f"  Total Return: {result.summary.total_return_pct * 100:.2f}%")

    @pytest.mark.asyncio
    async def test_momentum_strategy_eth(self, eth_bars: list[Bar]) -> None:
        """Test momentum strategy on ETH data."""
        config = BacktestConfig(
            initial_capital=Decimal("50000"),
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("10"),
            verbose=False,
        )

        engine = BacktestEngine(config)
        strategy = MomentumStrategy(
            StrategyConfig(symbol="ETH/USDT", timeframe="1h"),
            lookback=20,
            threshold=0.03,
        )
        engine.add_strategy(strategy)
        engine.add_bars("ETH/USDT", eth_bars)

        result = await engine.run()

        assert result is not None
        assert result.summary.strategy_name == "momentum_20"
        assert result.summary.symbol == "ETH/USDT"
        assert result.summary.bars_processed == 500

        print(f"\nMomentum Strategy Results (ETH/USDT, 500 bars):")
        print(f"  Final Equity: ${result.summary.final_equity:,.2f}")
        print(f"  Total Return: {result.summary.total_return_pct * 100:.2f}%")
        print(f"  Sharpe Ratio: {result.summary.sharpe_ratio:.2f}")
        print(f"  Total Trades: {result.summary.total_trades}")

    @pytest.mark.asyncio
    async def test_metrics_calculation(self, btc_bars: list[Bar]) -> None:
        """Test that metrics are calculated correctly."""
        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            slippage_model=SlippageModel.NONE,
            verbose=False,
        )

        engine = BacktestEngine(config)
        strategy = SimpleSMAStrategy(
            StrategyConfig(symbol="BTC/USDT", timeframe="1h"),
            fast_period=5,
            slow_period=15,
        )
        engine.add_strategy(strategy)
        engine.add_bars("BTC/USDT", btc_bars)

        result = await engine.run()

        # Validate metrics are calculated
        summary = result.summary
        assert summary.initial_capital == Decimal("100000")
        assert summary.final_equity > 0
        assert summary.duration_days > 0

        # Sharpe/Sortino should be finite
        assert not (summary.sharpe_ratio != summary.sharpe_ratio)  # Not NaN
        assert not (summary.sortino_ratio != summary.sortino_ratio)

        # Drawdown should be non-negative
        assert summary.max_drawdown >= 0
        assert 0 <= summary.max_drawdown_pct <= 1

        # Win rate should be between 0 and 1
        if summary.total_trades > 0:
            assert 0 <= summary.win_rate <= 1

        # Equity curve should be monotonically timestamped
        for i in range(1, len(result.equity_curve)):
            assert result.equity_curve[i].timestamp_ns > result.equity_curve[i-1].timestamp_ns

    @pytest.mark.asyncio
    async def test_result_dataframe_export(self, btc_bars: list[Bar]) -> None:
        """Test that results can be exported to DataFrame."""
        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            slippage_model=SlippageModel.NONE,
            verbose=False,
        )

        engine = BacktestEngine(config)
        strategy = SimpleSMAStrategy(
            StrategyConfig(symbol="BTC/USDT", timeframe="1h")
        )
        engine.add_strategy(strategy)
        engine.add_bars("BTC/USDT", btc_bars)

        result = await engine.run()

        # Export to DataFrame
        equity_df = result.to_dataframe()
        assert len(equity_df) == 500
        assert "equity" in equity_df.columns
        assert "drawdown" in equity_df.columns

        trades_df = result.trades_to_dataframe()
        # May have 0 or more trades
        assert trades_df is not None


class TestBacktestEngineMultipleRuns:
    """Test running multiple backtests for comparison."""

    @pytest.fixture
    async def bars(self) -> list[Bar]:
        """Fetch 300 hours of BTC/USDT data."""
        klines = await fetch_binance_klines("BTCUSDT", "1h", 300)
        return klines_to_bars(klines, "BTC/USDT", "1h")

    @pytest.mark.asyncio
    async def test_strategy_comparison(self, bars: list[Bar]) -> None:
        """Compare multiple strategies on the same data."""
        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("5"),
            verbose=False,
        )

        strategies = [
            SimpleSMAStrategy(StrategyConfig(symbol="BTC/USDT", timeframe="1h"), 5, 20),
            SimpleSMAStrategy(StrategyConfig(symbol="BTC/USDT", timeframe="1h"), 10, 30),
            SimpleSMAStrategy(StrategyConfig(symbol="BTC/USDT", timeframe="1h"), 20, 50),
            MomentumStrategy(StrategyConfig(symbol="BTC/USDT", timeframe="1h"), 10, 0.02),
            BuyAndHoldStrategy(StrategyConfig(symbol="BTC/USDT", timeframe="1h")),
        ]

        results: list[BacktestResult] = []

        for strategy in strategies:
            engine = BacktestEngine(config)
            engine.add_strategy(strategy)
            engine.add_bars("BTC/USDT", bars)
            result = await engine.run()
            results.append(result)

        # Print comparison
        print("\n" + "=" * 70)
        print("Strategy Comparison (BTC/USDT, 300 bars)")
        print("=" * 70)
        print(f"{'Strategy':<25} {'Return %':>10} {'Sharpe':>10} {'MaxDD %':>10} {'Trades':>8}")
        print("-" * 70)

        for result in results:
            s = result.summary
            print(
                f"{s.strategy_name:<25} "
                f"{s.total_return_pct * 100:>10.2f} "
                f"{s.sharpe_ratio:>10.2f} "
                f"{s.max_drawdown_pct * 100:>10.2f} "
                f"{s.total_trades:>8}"
            )

        print("=" * 70)

        # Validate all ran successfully
        assert len(results) == len(strategies)
        for result in results:
            assert result.summary.bars_processed == 300
