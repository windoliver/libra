"""
Unit tests for BacktestResult.

Tests:
- BacktestSummary creation and methods
- BacktestResult creation and conversion methods
- Serialization and deserialization
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from libra.backtest.result import BacktestResult, BacktestSummary, EquityPoint


class TestBacktestSummary:
    """Tests for BacktestSummary."""

    @pytest.fixture
    def sample_summary(self) -> BacktestSummary:
        """Create a sample summary for testing."""
        return BacktestSummary(
            strategy_name="TestStrategy",
            symbol="BTC/USDT",
            timeframe="1h",
            start_time_ns=1704067200_000_000_000,
            end_time_ns=1706745600_000_000_000,
            duration_days=31.0,
            initial_capital=Decimal("100000"),
            final_equity=Decimal("115000"),
            total_return=Decimal("15000"),
            total_return_pct=0.15,
            cagr=0.60,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=3.0,
            max_drawdown=Decimal("5000"),
            max_drawdown_pct=0.05,
            max_drawdown_duration_days=5.0,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.60,
            profit_factor=1.8,
            avg_win=Decimal("500"),
            avg_loss=Decimal("375"),
            avg_trade=Decimal("150"),
            largest_win=Decimal("2000"),
            largest_loss=Decimal("-1000"),
            avg_win_loss_ratio=1.33,
            expectancy=Decimal("150"),
            total_volume=Decimal("1000000"),
            total_fees=Decimal("1000"),
            bars_processed=744,
        )

    def test_duration_str_days(self, sample_summary: BacktestSummary) -> None:
        """Test duration string for days."""
        # sample_summary has 31 days, which formats as months
        assert "1.0 months" == sample_summary.duration_str

        # Create a summary with 15 days
        summary = BacktestSummary(
            strategy_name="Test",
            symbol="BTC/USDT",
            timeframe="1h",
            start_time_ns=0,
            end_time_ns=1,
            duration_days=15.0,
            initial_capital=Decimal("100000"),
            final_equity=Decimal("110000"),
            total_return=Decimal("10000"),
            total_return_pct=0.10,
            cagr=0.40,
            sharpe_ratio=1.0,
            sortino_ratio=1.2,
            calmar_ratio=2.0,
            max_drawdown=Decimal("5000"),
            max_drawdown_pct=0.05,
            max_drawdown_duration_days=5.0,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.60,
            profit_factor=1.5,
            avg_win=Decimal("400"),
            avg_loss=Decimal("300"),
            avg_trade=Decimal("200"),
            largest_win=Decimal("1000"),
            largest_loss=Decimal("-500"),
            avg_win_loss_ratio=1.33,
            expectancy=Decimal("200"),
            total_volume=Decimal("500000"),
            total_fees=Decimal("500"),
            bars_processed=360,
        )
        assert "15.0 days" == summary.duration_str

    def test_duration_str_months(self) -> None:
        """Test duration string for months."""
        summary = BacktestSummary(
            strategy_name="Test",
            symbol="BTC/USDT",
            timeframe="1h",
            start_time_ns=0,
            end_time_ns=1,
            duration_days=90.0,
            initial_capital=Decimal("100000"),
            final_equity=Decimal("110000"),
            total_return=Decimal("10000"),
            total_return_pct=0.10,
            cagr=0.40,
            sharpe_ratio=1.0,
            sortino_ratio=1.2,
            calmar_ratio=2.0,
            max_drawdown=Decimal("5000"),
            max_drawdown_pct=0.05,
            max_drawdown_duration_days=5.0,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.60,
            profit_factor=1.5,
            avg_win=Decimal("400"),
            avg_loss=Decimal("300"),
            avg_trade=Decimal("200"),
            largest_win=Decimal("1000"),
            largest_loss=Decimal("-500"),
            avg_win_loss_ratio=1.33,
            expectancy=Decimal("200"),
            total_volume=Decimal("500000"),
            total_fees=Decimal("500"),
            bars_processed=2160,
        )
        assert "3.0 months" == summary.duration_str

    def test_duration_str_years(self) -> None:
        """Test duration string for years."""
        summary = BacktestSummary(
            strategy_name="Test",
            symbol="BTC/USDT",
            timeframe="1h",
            start_time_ns=0,
            end_time_ns=1,
            duration_days=730.0,
            initial_capital=Decimal("100000"),
            final_equity=Decimal("200000"),
            total_return=Decimal("100000"),
            total_return_pct=1.0,
            cagr=0.50,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=2.5,
            max_drawdown=Decimal("20000"),
            max_drawdown_pct=0.20,
            max_drawdown_duration_days=30.0,
            total_trades=500,
            winning_trades=300,
            losing_trades=200,
            win_rate=0.60,
            profit_factor=1.8,
            avg_win=Decimal("500"),
            avg_loss=Decimal("350"),
            avg_trade=Decimal("200"),
            largest_win=Decimal("5000"),
            largest_loss=Decimal("-2000"),
            avg_win_loss_ratio=1.43,
            expectancy=Decimal("200"),
            total_volume=Decimal("5000000"),
            total_fees=Decimal("5000"),
            bars_processed=17520,
        )
        assert "2.0 years" == summary.duration_str

    def test_to_dict(self, sample_summary: BacktestSummary) -> None:
        """Test conversion to dictionary."""
        d = sample_summary.to_dict()

        assert d["strategy"] == "TestStrategy"
        assert d["symbol"] == "BTC/USDT"
        assert "15.00%" in d["total_return_pct"]
        assert "1.50" in d["sharpe_ratio"]
        assert "60.0%" in d["win_rate"]


class TestEquityPoint:
    """Tests for EquityPoint."""

    def test_create_equity_point(self) -> None:
        """Test creating an equity point."""
        point = EquityPoint(
            timestamp_ns=1704067200_000_000_000,
            equity=Decimal("105000"),
            cash=Decimal("50000"),
            position_value=Decimal("55000"),
            drawdown=Decimal("5000"),
            drawdown_pct=0.0455,
        )

        assert point.equity == Decimal("105000")
        assert point.cash == Decimal("50000")
        assert point.position_value == Decimal("55000")
        assert point.drawdown == Decimal("5000")
        assert abs(point.drawdown_pct - 0.0455) < 0.001


class TestBacktestResult:
    """Tests for BacktestResult."""

    @pytest.fixture
    def sample_result(self) -> BacktestResult:
        """Create a sample result for testing."""
        summary = BacktestSummary(
            strategy_name="TestStrategy",
            symbol="BTC/USDT",
            timeframe="1h",
            start_time_ns=1704067200_000_000_000,
            end_time_ns=1706745600_000_000_000,
            duration_days=31.0,
            initial_capital=Decimal("100000"),
            final_equity=Decimal("115000"),
            total_return=Decimal("15000"),
            total_return_pct=0.15,
            cagr=0.60,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=3.0,
            max_drawdown=Decimal("5000"),
            max_drawdown_pct=0.05,
            max_drawdown_duration_days=5.0,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.60,
            profit_factor=1.8,
            avg_win=Decimal("500"),
            avg_loss=Decimal("375"),
            avg_trade=Decimal("150"),
            largest_win=Decimal("2000"),
            largest_loss=Decimal("-1000"),
            avg_win_loss_ratio=1.33,
            expectancy=Decimal("150"),
            total_volume=Decimal("100000"),
            total_fees=Decimal("100"),
            bars_processed=744,
        )

        equity_curve = [
            EquityPoint(
                timestamp_ns=1704067200_000_000_000 + i * 3600_000_000_000,
                equity=Decimal("100000") + Decimal(str(i * 100)),
                cash=Decimal("50000"),
                position_value=Decimal("50000") + Decimal(str(i * 100)),
                drawdown=Decimal("0"),
                drawdown_pct=0.0,
            )
            for i in range(10)
        ]

        return BacktestResult(
            summary=summary,
            equity_curve=equity_curve,
            trades=[],
            daily_returns=[0.01, 0.02, -0.01, 0.015, 0.005],
            config={"initial_capital": "100000"},
            run_timestamp_ns=1706745600_000_000_000,
        )

    def test_to_dataframe(self, sample_result: BacktestResult) -> None:
        """Test conversion to Polars DataFrame."""
        df = sample_result.to_dataframe()

        assert len(df) == 10
        assert "equity" in df.columns
        assert "cash" in df.columns
        assert "position_value" in df.columns
        assert "drawdown" in df.columns

    def test_trades_to_dataframe_empty(self, sample_result: BacktestResult) -> None:
        """Test trades DataFrame with no trades."""
        df = sample_result.trades_to_dataframe()
        assert len(df) == 0

    def test_print_summary(self, sample_result: BacktestResult, capsys: pytest.CaptureFixture[str]) -> None:
        """Test printing summary."""
        sample_result.print_summary()
        captured = capsys.readouterr()

        assert "TestStrategy" in captured.out
        assert "BTC/USDT" in captured.out
        assert "15.00%" in captured.out
        assert "1.50" in captured.out  # Sharpe ratio
