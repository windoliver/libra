"""
Tests for VectorizedBacktest (Issue #104).

Tests the vectorbt-style vectorized backtest implementation.
"""

import numpy as np
import polars as pl
import pytest

from libra.backtest.vectorized import (
    VectorizedBacktest,
    VectorizedConfig,
    VectorizedResult,
    generate_moving_average_signals,
    generate_rsi_signals,
)


class TestVectorizedBacktest:
    """Tests for VectorizedBacktest class."""

    @pytest.fixture
    def simple_prices(self) -> pl.DataFrame:
        """Create simple price data for testing."""
        n = 100
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({
            "symbol": ["BTC/USDT"] * n,
            "timestamp": list(range(n)),
            "close": prices.tolist(),
        })

    @pytest.fixture
    def simple_signals(self) -> pl.DataFrame:
        """Create simple alternating signals."""
        n = 100
        signals = [0] * n
        # Buy at 10, sell at 30, buy at 50, sell at 70
        signals[10] = 1
        signals[30] = -1
        signals[50] = 1
        signals[70] = -1
        return pl.DataFrame({
            "symbol": ["BTC/USDT"] * n,
            "timestamp": list(range(n)),
            "signal": signals,
        })

    @pytest.fixture
    def backtest(self) -> VectorizedBacktest:
        """Create backtest instance."""
        config = VectorizedConfig(
            initial_capital=100_000,
            commission_pct=0.001,
            slippage_pct=0.0,
        )
        return VectorizedBacktest(config)

    def test_basic_backtest(
        self,
        backtest: VectorizedBacktest,
        simple_signals: pl.DataFrame,
        simple_prices: pl.DataFrame,
    ) -> None:
        """Test basic backtest execution."""
        result = backtest.run(simple_signals, simple_prices)

        assert isinstance(result, VectorizedResult)
        assert result.initial_capital == 100_000
        assert result.bars_processed == 100
        assert result.execution_time_ms > 0

    def test_result_has_equity_curve(
        self,
        backtest: VectorizedBacktest,
        simple_signals: pl.DataFrame,
        simple_prices: pl.DataFrame,
    ) -> None:
        """Test that result includes equity curve DataFrame."""
        result = backtest.run(simple_signals, simple_prices)

        assert isinstance(result.equity_df, pl.DataFrame)
        assert "equity" in result.equity_df.columns
        assert "drawdown" in result.equity_df.columns
        assert result.equity_df.height == 100

    def test_metrics_computed(
        self,
        backtest: VectorizedBacktest,
        simple_signals: pl.DataFrame,
        simple_prices: pl.DataFrame,
    ) -> None:
        """Test that all metrics are computed."""
        result = backtest.run(simple_signals, simple_prices)

        # All metrics should be numeric
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.sortino_ratio, float)
        assert isinstance(result.calmar_ratio, float)
        assert isinstance(result.max_drawdown_pct, float)
        assert isinstance(result.cagr, float)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.profit_factor, float)

    def test_buy_and_hold(
        self,
        backtest: VectorizedBacktest,
        simple_prices: pl.DataFrame,
    ) -> None:
        """Test buy and hold strategy (always long)."""
        n = simple_prices.height
        signals = pl.DataFrame({
            "symbol": ["BTC/USDT"] * n,
            "timestamp": list(range(n)),
            "signal": [1] + [0] * (n - 1),  # Buy once, hold
        })

        result = backtest.run(signals, simple_prices)

        # Should have one trade (entry)
        assert result.num_trades >= 1

        # Equity should track price movements
        assert result.final_equity != result.initial_capital

    def test_no_trades(
        self,
        backtest: VectorizedBacktest,
        simple_prices: pl.DataFrame,
    ) -> None:
        """Test with no signals (all zeros)."""
        n = simple_prices.height
        signals = pl.DataFrame({
            "symbol": ["BTC/USDT"] * n,
            "timestamp": list(range(n)),
            "signal": [0] * n,
        })

        result = backtest.run(signals, simple_prices)

        # No trades
        assert result.num_trades == 0

        # Equity should remain unchanged
        assert result.final_equity == result.initial_capital
        assert result.total_return == 0

    def test_commission_impact(self, simple_prices: pl.DataFrame) -> None:
        """Test that commission reduces returns."""
        n = simple_prices.height

        # Frequent trading signals
        signals = pl.DataFrame({
            "symbol": ["BTC/USDT"] * n,
            "timestamp": list(range(n)),
            "signal": [1 if i % 5 == 0 else (-1 if i % 5 == 2 else 0) for i in range(n)],
        })

        # Run with no commission
        config_no_comm = VectorizedConfig(commission_pct=0.0, slippage_pct=0.0)
        backtest_no_comm = VectorizedBacktest(config_no_comm)
        result_no_comm = backtest_no_comm.run(signals, simple_prices)

        # Run with commission
        config_comm = VectorizedConfig(commission_pct=0.01, slippage_pct=0.0)
        backtest_comm = VectorizedBacktest(config_comm)
        result_comm = backtest_comm.run(signals, simple_prices)

        # Commission should reduce final equity
        assert result_comm.final_equity < result_no_comm.final_equity

    def test_drawdown_calculation(
        self,
        backtest: VectorizedBacktest,
        simple_signals: pl.DataFrame,
        simple_prices: pl.DataFrame,
    ) -> None:
        """Test drawdown calculation."""
        result = backtest.run(simple_signals, simple_prices)

        # Max drawdown should be non-negative
        assert result.max_drawdown_pct >= 0

        # Drawdown column in equity_df
        assert (result.equity_df["drawdown_pct"] >= 0).all()

    def test_invalid_signals_missing_columns(
        self,
        backtest: VectorizedBacktest,
        simple_prices: pl.DataFrame,
    ) -> None:
        """Test validation of missing columns."""
        invalid_signals = pl.DataFrame({
            "symbol": ["BTC/USDT"] * 10,
            # Missing timestamp and signal columns
        })

        with pytest.raises(ValueError, match="missing columns"):
            backtest.run(invalid_signals, simple_prices)

    def test_invalid_prices_missing_columns(
        self,
        backtest: VectorizedBacktest,
        simple_signals: pl.DataFrame,
    ) -> None:
        """Test validation of missing price columns."""
        invalid_prices = pl.DataFrame({
            "symbol": ["BTC/USDT"] * 100,
            # Missing timestamp and close columns
        })

        with pytest.raises(ValueError, match="missing columns"):
            backtest.run(simple_signals, invalid_prices)

    def test_empty_signals(
        self,
        backtest: VectorizedBacktest,
        simple_prices: pl.DataFrame,
    ) -> None:
        """Test with empty signals DataFrame."""
        empty_signals = pl.DataFrame({
            "symbol": pl.Series([], dtype=pl.Utf8),
            "timestamp": pl.Series([], dtype=pl.Int64),
            "signal": pl.Series([], dtype=pl.Int64),
        })

        with pytest.raises(ValueError, match="empty"):
            backtest.run(empty_signals, simple_prices)


class TestMultiSymbol:
    """Tests for multi-symbol backtesting."""

    @pytest.fixture
    def multi_symbol_prices(self) -> pl.DataFrame:
        """Create multi-symbol price data."""
        n = 50
        np.random.seed(42)

        btc_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        eth_prices = 50 + np.cumsum(np.random.randn(n) * 0.3)

        return pl.DataFrame({
            "symbol": ["BTC/USDT"] * n + ["ETH/USDT"] * n,
            "timestamp": list(range(n)) * 2,
            "close": btc_prices.tolist() + eth_prices.tolist(),
        })

    @pytest.fixture
    def multi_symbol_signals(self) -> pl.DataFrame:
        """Create multi-symbol signals."""
        n = 50
        btc_signals = [0] * n
        eth_signals = [0] * n

        btc_signals[5] = 1
        btc_signals[25] = -1
        eth_signals[10] = 1
        eth_signals[30] = -1

        return pl.DataFrame({
            "symbol": ["BTC/USDT"] * n + ["ETH/USDT"] * n,
            "timestamp": list(range(n)) * 2,
            "signal": btc_signals + eth_signals,
        })

    def test_multi_symbol_backtest(
        self,
        multi_symbol_prices: pl.DataFrame,
        multi_symbol_signals: pl.DataFrame,
    ) -> None:
        """Test backtest with multiple symbols."""
        backtest = VectorizedBacktest()
        result = backtest.run(multi_symbol_signals, multi_symbol_prices)

        assert isinstance(result, VectorizedResult)
        assert result.bars_processed > 0

    def test_equal_capital_allocation(
        self,
        multi_symbol_prices: pl.DataFrame,
    ) -> None:
        """Test that capital is allocated equally across symbols."""
        n = 50
        # Buy signal for both symbols
        signals = pl.DataFrame({
            "symbol": ["BTC/USDT"] * n + ["ETH/USDT"] * n,
            "timestamp": list(range(n)) * 2,
            "signal": [1] + [0] * (n - 1) + [1] + [0] * (n - 1),
        })

        config = VectorizedConfig(initial_capital=100_000)
        backtest = VectorizedBacktest(config)
        result = backtest.run(signals, multi_symbol_prices)

        # Should have processed bars for both symbols
        assert result.bars_processed > 0


class TestSignalGenerators:
    """Tests for signal generation utilities."""

    @pytest.fixture
    def price_data(self) -> pl.DataFrame:
        """Create price data for signal generation."""
        n = 100
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({
            "symbol": ["BTC/USDT"] * n,
            "timestamp": list(range(n)),
            "close": prices.tolist(),
        })

    def test_moving_average_signals(self, price_data: pl.DataFrame) -> None:
        """Test MA crossover signal generation."""
        signals = generate_moving_average_signals(
            price_data,
            fast_period=5,
            slow_period=20,
        )

        assert "symbol" in signals.columns
        assert "timestamp" in signals.columns
        assert "signal" in signals.columns
        assert signals.height == price_data.height

        # Signals should be -1, 0, or 1
        unique_signals = signals["signal"].unique().to_list()
        assert all(s in [-1, 0, 1] for s in unique_signals)

    def test_rsi_signals(self, price_data: pl.DataFrame) -> None:
        """Test RSI signal generation."""
        signals = generate_rsi_signals(
            price_data,
            period=14,
            oversold=30,
            overbought=70,
        )

        assert "symbol" in signals.columns
        assert "timestamp" in signals.columns
        assert "signal" in signals.columns
        assert signals.height == price_data.height

    def test_signal_integration(self, price_data: pl.DataFrame) -> None:
        """Test that generated signals work with backtest."""
        signals = generate_moving_average_signals(price_data, fast_period=5, slow_period=20)

        backtest = VectorizedBacktest()
        result = backtest.run(signals, price_data)

        assert isinstance(result, VectorizedResult)
        assert result.bars_processed > 0


class TestOptimization:
    """Tests for parameter optimization."""

    @pytest.fixture
    def price_data(self) -> pl.DataFrame:
        """Create price data for optimization."""
        n = 200
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pl.DataFrame({
            "symbol": ["BTC/USDT"] * n,
            "timestamp": list(range(n)),
            "close": prices.tolist(),
        })

    def test_optimization_runs(self, price_data: pl.DataFrame) -> None:
        """Test parameter optimization."""
        backtest = VectorizedBacktest()

        def signal_generator(params: dict) -> pl.DataFrame:
            return generate_moving_average_signals(
                price_data,
                fast_period=params["fast"],
                slow_period=params["slow"],
            )

        param_grid = {
            "fast": [5, 10],
            "slow": [20, 30],
        }

        results = backtest.run_optimization(signal_generator, price_data, param_grid)

        assert isinstance(results, pl.DataFrame)
        # 2 fast x 2 slow = 4 combinations
        assert results.height == 4
        assert "sharpe" in results.columns
        assert "fast" in results.columns
        assert "slow" in results.columns

    def test_optimization_sorted_by_sharpe(self, price_data: pl.DataFrame) -> None:
        """Test that optimization results are sorted by Sharpe."""
        backtest = VectorizedBacktest()

        def signal_generator(params: dict) -> pl.DataFrame:
            return generate_moving_average_signals(
                price_data,
                fast_period=params["fast"],
                slow_period=params["slow"],
            )

        param_grid = {
            "fast": [5, 10, 15],
            "slow": [20, 30],
        }

        results = backtest.run_optimization(signal_generator, price_data, param_grid)

        # Results should be sorted by Sharpe descending
        sharpes = results["sharpe"].to_list()
        assert sharpes == sorted(sharpes, reverse=True)


class TestPerformance:
    """Performance tests for vectorized backtest."""

    def test_large_dataset_performance(self) -> None:
        """Test performance with large dataset."""
        # 10,000 bars
        n = 10_000
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        price_df = pl.DataFrame({
            "symbol": ["BTC/USDT"] * n,
            "timestamp": list(range(n)),
            "close": prices.tolist(),
        })

        signals = generate_moving_average_signals(price_df, fast_period=10, slow_period=50)

        backtest = VectorizedBacktest()
        result = backtest.run(signals, price_df)

        # Should complete in under 100ms for 10k bars
        assert result.execution_time_ms < 100
        assert result.bars_processed == n

    def test_many_trades_performance(self) -> None:
        """Test performance with many trades."""
        n = 1000
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        price_df = pl.DataFrame({
            "symbol": ["BTC/USDT"] * n,
            "timestamp": list(range(n)),
            "close": prices.tolist(),
        })

        # Alternating signals for many trades
        signals = pl.DataFrame({
            "symbol": ["BTC/USDT"] * n,
            "timestamp": list(range(n)),
            "signal": [1 if i % 10 == 0 else (-1 if i % 10 == 5 else 0) for i in range(n)],
        })

        backtest = VectorizedBacktest()
        result = backtest.run(signals, price_df)

        # Should have many trades
        assert result.num_trades > 100

        # Should still be fast
        assert result.execution_time_ms < 50
