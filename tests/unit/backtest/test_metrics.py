"""
Unit tests for MetricsCollector.

Tests:
- Equity tracking
- Drawdown calculation
- Trade P&L calculation
- Sharpe/Sortino ratio calculation
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from libra.backtest.metrics import MetricsCollector, TradeRecord


class TestTradeRecord:
    """Tests for TradeRecord struct."""

    def test_create_open_trade(self) -> None:
        """Test creating an open trade record."""
        trade = TradeRecord(
            trade_id="trade_001",
            symbol="BTC/USDT",
            side="long",
            entry_time_ns=1704067200_000_000_000,
            entry_price=Decimal("42000"),
            quantity=Decimal("1.0"),
        )

        assert trade.trade_id == "trade_001"
        assert trade.symbol == "BTC/USDT"
        assert trade.side == "long"
        assert trade.entry_price == Decimal("42000")
        assert trade.quantity == Decimal("1.0")
        assert not trade.is_closed
        assert trade.exit_time_ns is None
        assert trade.pnl is None

    def test_closed_trade_winner(self) -> None:
        """Test a winning trade."""
        trade = TradeRecord(
            trade_id="trade_001",
            symbol="BTC/USDT",
            side="long",
            entry_time_ns=1704067200_000_000_000,
            entry_price=Decimal("42000"),
            quantity=Decimal("1.0"),
            exit_time_ns=1704153600_000_000_000,
            exit_price=Decimal("43000"),
            pnl=Decimal("1000"),
            pnl_pct=0.0238,
        )

        assert trade.is_closed
        assert trade.is_winner
        assert not trade.is_loser
        assert trade.pnl == Decimal("1000")

    def test_closed_trade_loser(self) -> None:
        """Test a losing trade."""
        trade = TradeRecord(
            trade_id="trade_002",
            symbol="BTC/USDT",
            side="long",
            entry_time_ns=1704067200_000_000_000,
            entry_price=Decimal("42000"),
            quantity=Decimal("1.0"),
            exit_time_ns=1704153600_000_000_000,
            exit_price=Decimal("41000"),
            pnl=Decimal("-1000"),
            pnl_pct=-0.0238,
        )

        assert trade.is_closed
        assert not trade.is_winner
        assert trade.is_loser
        assert trade.pnl == Decimal("-1000")


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_initialization(self) -> None:
        """Test collector initialization."""
        collector = MetricsCollector(initial_capital=Decimal("100000"))

        assert collector.initial_capital == Decimal("100000")
        assert collector.bars_processed == 0
        assert len(collector.trades) == 0
        assert len(collector.equity_snapshots) == 0

    def test_record_equity(self) -> None:
        """Test equity recording."""
        collector = MetricsCollector(initial_capital=Decimal("100000"))

        # Record some equity points
        collector.record_equity(
            timestamp_ns=1704067200_000_000_000,
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            position_value=Decimal("0"),
        )
        collector.record_equity(
            timestamp_ns=1704153600_000_000_000,
            equity=Decimal("101000"),
            cash=Decimal("50000"),
            position_value=Decimal("51000"),
        )
        collector.record_equity(
            timestamp_ns=1704240000_000_000_000,
            equity=Decimal("102000"),
            cash=Decimal("50000"),
            position_value=Decimal("52000"),
        )

        assert len(collector.equity_snapshots) == 3
        assert collector.bars_processed == 3

    def test_drawdown_tracking(self) -> None:
        """Test drawdown calculation."""
        collector = MetricsCollector(initial_capital=Decimal("100000"))

        # Start at 100000
        collector.record_equity(1, Decimal("100000"), Decimal("100000"), Decimal("0"))
        assert collector.drawdown.max_drawdown == Decimal("0")

        # Peak at 110000
        collector.record_equity(2, Decimal("110000"), Decimal("110000"), Decimal("0"))
        assert collector.drawdown.peak_equity == Decimal("110000")

        # Drawdown to 100000 (10% drawdown)
        collector.record_equity(3, Decimal("100000"), Decimal("100000"), Decimal("0"))
        assert collector.drawdown.current_drawdown == Decimal("10000")
        assert abs(collector.drawdown.current_drawdown_pct - 0.0909) < 0.01

        # Deeper drawdown to 95000
        collector.record_equity(4, Decimal("95000"), Decimal("95000"), Decimal("0"))
        assert collector.drawdown.max_drawdown == Decimal("15000")

        # Recovery to 115000
        collector.record_equity(5, Decimal("115000"), Decimal("115000"), Decimal("0"))
        assert collector.drawdown.peak_equity == Decimal("115000")
        assert collector.drawdown.current_drawdown == Decimal("0")

    def test_open_and_close_trade(self) -> None:
        """Test opening and closing trades."""
        collector = MetricsCollector(initial_capital=Decimal("100000"))

        # Open a long trade
        trade = collector.open_trade(
            trade_id="trade_001",
            symbol="BTC/USDT",
            side="long",
            entry_time_ns=1704067200_000_000_000,
            entry_price=Decimal("42000"),
            quantity=Decimal("1.0"),
            fees=Decimal("42"),
        )

        assert trade.trade_id == "trade_001"
        assert "BTC/USDT" in collector.open_trades

        # Close the trade with profit
        closed_trade = collector.close_trade(
            symbol="BTC/USDT",
            exit_time_ns=1704153600_000_000_000,
            exit_price=Decimal("43000"),
            fees=Decimal("43"),
        )

        assert closed_trade is not None
        assert closed_trade.is_closed
        # PnL = (43000 - 42000) * 1.0 - 85 = 915
        assert closed_trade.pnl == Decimal("915")
        assert "BTC/USDT" not in collector.open_trades
        assert len(collector.trades) == 1

    def test_short_trade_pnl(self) -> None:
        """Test P&L calculation for short trades."""
        collector = MetricsCollector(initial_capital=Decimal("100000"))

        # Open a short trade
        collector.open_trade(
            trade_id="trade_002",
            symbol="ETH/USDT",
            side="short",
            entry_time_ns=1704067200_000_000_000,
            entry_price=Decimal("2200"),
            quantity=Decimal("10.0"),
            fees=Decimal("22"),
        )

        # Close with profit (price went down)
        closed_trade = collector.close_trade(
            symbol="ETH/USDT",
            exit_time_ns=1704153600_000_000_000,
            exit_price=Decimal("2100"),
            fees=Decimal("21"),
        )

        assert closed_trade is not None
        # Short PnL = (entry - exit) * qty - fees = (2200 - 2100) * 10 - 43 = 957
        assert closed_trade.pnl == Decimal("957")

    def test_sharpe_ratio_calculation(self) -> None:
        """Test Sharpe ratio calculation."""
        collector = MetricsCollector(initial_capital=Decimal("100000"))

        # Simulate 10 days of returns
        ns_per_day = 86400 * 1_000_000_000
        base_equity = Decimal("100000")
        daily_returns = [0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.02, 0.01, -0.02, 0.015]

        for i, ret in enumerate(daily_returns):
            base_equity *= Decimal(str(1 + ret))
            collector.record_equity(
                timestamp_ns=i * ns_per_day,
                equity=base_equity,
                cash=base_equity,
                position_value=Decimal("0"),
            )

        sharpe = collector.calculate_sharpe_ratio()
        # With positive average returns and some volatility, Sharpe should be positive
        assert sharpe > 0

    def test_sortino_ratio_calculation(self) -> None:
        """Test Sortino ratio calculation."""
        collector = MetricsCollector(initial_capital=Decimal("100000"))

        # Simulate 10 days of mostly positive returns
        ns_per_day = 86400 * 1_000_000_000
        base_equity = Decimal("100000")
        # Few negative returns = low downside deviation = high Sortino
        daily_returns = [0.01, 0.02, -0.005, 0.015, 0.005, -0.002, 0.02, 0.01, -0.01, 0.015]

        for i, ret in enumerate(daily_returns):
            base_equity *= Decimal(str(1 + ret))
            collector.record_equity(
                timestamp_ns=i * ns_per_day,
                equity=base_equity,
                cash=base_equity,
                position_value=Decimal("0"),
            )

        sharpe = collector.calculate_sharpe_ratio()
        sortino = collector.calculate_sortino_ratio()

        # Sortino should be higher than Sharpe when there are few downside moves
        assert sortino > sharpe

    def test_reset(self) -> None:
        """Test collector reset."""
        collector = MetricsCollector(initial_capital=Decimal("100000"))

        # Add some data
        collector.record_equity(1, Decimal("100000"), Decimal("100000"), Decimal("0"))
        collector.open_trade(
            trade_id="trade_001",
            symbol="BTC/USDT",
            side="long",
            entry_time_ns=1,
            entry_price=Decimal("42000"),
            quantity=Decimal("1.0"),
        )

        # Reset
        collector.reset()

        assert len(collector.equity_snapshots) == 0
        assert len(collector.trades) == 0
        assert len(collector.open_trades) == 0
        assert collector.bars_processed == 0
        assert collector.total_fees == Decimal("0")

    def test_calculate_summary(self) -> None:
        """Test summary calculation."""
        collector = MetricsCollector(initial_capital=Decimal("100000"))

        # Simulate a backtest with some trades
        ns_per_day = 86400 * 1_000_000_000

        # Record equity over 30 days
        equity = Decimal("100000")
        for i in range(30):
            # Add some variance
            change = Decimal(str(1 + (i % 5 - 2) * 0.01))
            equity *= change
            collector.record_equity(
                timestamp_ns=i * ns_per_day,
                equity=equity,
                cash=equity,
                position_value=Decimal("0"),
            )

        # Add some trades
        collector.record_trade(
            TradeRecord(
                trade_id="t1",
                symbol="BTC/USDT",
                side="long",
                entry_time_ns=0,
                entry_price=Decimal("42000"),
                quantity=Decimal("1.0"),
                exit_time_ns=ns_per_day,
                exit_price=Decimal("43000"),
                pnl=Decimal("1000"),
                pnl_pct=0.0238,
            )
        )
        collector.record_trade(
            TradeRecord(
                trade_id="t2",
                symbol="BTC/USDT",
                side="long",
                entry_time_ns=ns_per_day * 2,
                entry_price=Decimal("43000"),
                quantity=Decimal("1.0"),
                exit_time_ns=ns_per_day * 3,
                exit_price=Decimal("42500"),
                pnl=Decimal("-500"),
                pnl_pct=-0.0116,
            )
        )

        summary = collector.calculate_summary(
            strategy_name="TestStrategy",
            symbol="BTC/USDT",
            timeframe="1h",
        )

        assert summary.strategy_name == "TestStrategy"
        assert summary.symbol == "BTC/USDT"
        assert summary.total_trades == 2
        assert summary.winning_trades == 1
        assert summary.losing_trades == 1
        assert summary.win_rate == 0.5
        assert summary.bars_processed == 30


class TestEquityCurve:
    """Tests for equity curve generation."""

    def test_get_equity_curve(self) -> None:
        """Test getting equity curve as EquityPoint list."""
        collector = MetricsCollector(initial_capital=Decimal("100000"))

        # Record some equity points
        collector.record_equity(1, Decimal("100000"), Decimal("100000"), Decimal("0"))
        collector.record_equity(2, Decimal("105000"), Decimal("50000"), Decimal("55000"))
        collector.record_equity(3, Decimal("103000"), Decimal("50000"), Decimal("53000"))

        curve = collector.get_equity_curve()

        assert len(curve) == 3
        assert curve[0].equity == Decimal("100000")
        assert curve[1].equity == Decimal("105000")
        assert curve[2].equity == Decimal("103000")
        assert curve[2].drawdown_pct > 0  # Should have some drawdown
