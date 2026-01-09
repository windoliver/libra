"""Tests for Performance Tracking Module (Issue #12)."""

from dataclasses import dataclass
from decimal import Decimal
import time

import pytest

from libra.plugins.hummingbot_adapter.performance import (
    OrderFill,
    PerformanceSnapshot,
    PerformanceStats,
    PerformanceTracker,
    Position,
    PositionSide,
    Trade,
)


@dataclass
class MockTradeEvent:
    """Mock trade event for testing."""

    symbol: str
    side: str
    quantity: float
    price: float
    timestamp_ns: int


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self) -> None:
        """Test basic position creation."""
        pos = Position(
            symbol="ETH-USD",
            side=PositionSide.LONG,
            size=Decimal("10"),
            entry_price=Decimal("2000"),
            entry_time_ns=time.time_ns(),
        )

        assert pos.symbol == "ETH-USD"
        assert pos.side == PositionSide.LONG
        assert pos.is_open

    def test_position_unrealized_pnl_long(self) -> None:
        """Test unrealized P&L calculation for long position."""
        pos = Position(
            symbol="ETH-USD",
            side=PositionSide.LONG,
            size=Decimal("10"),
            entry_price=Decimal("2000"),
            entry_time_ns=time.time_ns(),
        )

        # Price goes up - profit
        pnl = pos.unrealized_pnl(Decimal("2100"))
        assert pnl == Decimal("1000")  # 10 * (2100 - 2000) = 1000

        # Price goes down - loss
        pnl = pos.unrealized_pnl(Decimal("1900"))
        assert pnl == Decimal("-1000")

    def test_position_unrealized_pnl_short(self) -> None:
        """Test unrealized P&L calculation for short position."""
        pos = Position(
            symbol="ETH-USD",
            side=PositionSide.SHORT,
            size=Decimal("10"),
            entry_price=Decimal("2000"),
            entry_time_ns=time.time_ns(),
        )

        # Price goes down - profit for short
        pnl = pos.unrealized_pnl(Decimal("1900"))
        assert pnl == Decimal("1000")

        # Price goes up - loss for short
        pnl = pos.unrealized_pnl(Decimal("2100"))
        assert pnl == Decimal("-1000")

    def test_position_total_pnl(self) -> None:
        """Test total P&L including realized."""
        pos = Position(
            symbol="ETH-USD",
            side=PositionSide.LONG,
            size=Decimal("10"),
            entry_price=Decimal("2000"),
            entry_time_ns=time.time_ns(),
            realized_pnl=Decimal("500"),
        )

        total = pos.total_pnl(Decimal("2100"))
        assert total == Decimal("1500")  # 500 realized + 1000 unrealized

    def test_flat_position_pnl(self) -> None:
        """Test P&L for flat position (no open)."""
        pos = Position(
            symbol="ETH-USD",
            side=PositionSide.FLAT,
            size=Decimal("0"),
            entry_price=Decimal("0"),
            entry_time_ns=time.time_ns(),
        )

        assert pos.unrealized_pnl(Decimal("2000")) == Decimal("0")
        assert not pos.is_open


class TestTrade:
    """Tests for Trade dataclass."""

    def test_trade_creation(self) -> None:
        """Test basic trade creation."""
        now = time.time_ns()
        trade = Trade(
            symbol="ETH-USD",
            side="long",
            size=Decimal("10"),
            entry_price=Decimal("2000"),
            exit_price=Decimal("2100"),
            entry_time_ns=now,
            exit_time_ns=now + 60_000_000_000,  # 1 minute later
            realized_pnl=Decimal("1000"),
            commission=Decimal("10"),
        )

        assert trade.symbol == "ETH-USD"
        assert trade.holding_time_ns == 60_000_000_000
        assert trade.is_winner

    def test_trade_return_percentage(self) -> None:
        """Test return percentage calculation."""
        trade = Trade(
            symbol="ETH-USD",
            side="long",
            size=Decimal("10"),
            entry_price=Decimal("2000"),
            exit_price=Decimal("2100"),
            entry_time_ns=0,
            exit_time_ns=0,
            realized_pnl=Decimal("1000"),
            commission=Decimal("0"),
        )

        assert trade.return_pct == 5.0  # (2100 - 2000) / 2000 * 100 = 5%

    def test_losing_trade(self) -> None:
        """Test losing trade detection."""
        trade = Trade(
            symbol="ETH-USD",
            side="long",
            size=Decimal("10"),
            entry_price=Decimal("2000"),
            exit_price=Decimal("1900"),
            entry_time_ns=0,
            exit_time_ns=0,
            realized_pnl=Decimal("-1000"),
            commission=Decimal("10"),
        )

        assert not trade.is_winner
        assert trade.return_pct == -5.0


class TestOrderFill:
    """Tests for OrderFill dataclass."""

    def test_order_fill_creation(self) -> None:
        """Test order fill creation."""
        fill = OrderFill(
            order_id="order-123",
            symbol="ETH-USD",
            side="buy",
            price=Decimal("2000"),
            size=Decimal("10"),
            timestamp_ns=time.time_ns(),
            latency_ns=5_000_000,  # 5ms
        )

        assert fill.order_id == "order-123"
        assert fill.latency_ns == 5_000_000


class TestPerformanceStats:
    """Tests for PerformanceStats dataclass."""

    def test_stats_defaults(self) -> None:
        """Test default values for stats."""
        stats = PerformanceStats()

        assert stats.total_pnl == Decimal("0")
        assert stats.total_trades == 0
        assert stats.win_rate == 0.0
        assert stats.sharpe_ratio == 0.0
        assert stats.max_drawdown_pct == 0.0


class TestPerformanceTracker:
    """Tests for PerformanceTracker."""

    @pytest.fixture
    def tracker(self) -> PerformanceTracker:
        """Create tracker with default settings."""
        return PerformanceTracker(
            initial_capital=Decimal("100000"),
            snapshot_interval_ns=1_000_000_000,  # 1 second for testing
        )

    def test_tracker_initialization(self, tracker: PerformanceTracker) -> None:
        """Test tracker initialization."""
        assert tracker.initial_capital == Decimal("100000")
        assert tracker.equity == Decimal("100000")
        assert tracker.unrealized_pnl == Decimal("0")
        assert tracker.realized_pnl == Decimal("0")

    def test_update_price(self, tracker: PerformanceTracker) -> None:
        """Test price updates."""
        tracker.update_price("ETH-USD", Decimal("2000"))
        assert tracker._current_prices["ETH-USD"] == Decimal("2000")

    def test_on_trade_buy_opens_long(self, tracker: PerformanceTracker) -> None:
        """Test buy trade opens long position."""
        event = MockTradeEvent(
            symbol="ETH-USD",
            side="buy",
            quantity=10.0,
            price=2000.0,
            timestamp_ns=time.time_ns(),
        )

        tracker.on_trade(event)

        pos = tracker.get_position("ETH-USD")
        assert pos is not None
        assert pos.side == PositionSide.LONG
        assert pos.size == Decimal("10")
        assert pos.entry_price == Decimal("2000")

    def test_on_trade_sell_opens_short(self, tracker: PerformanceTracker) -> None:
        """Test sell trade opens short position."""
        event = MockTradeEvent(
            symbol="ETH-USD",
            side="sell",
            quantity=10.0,
            price=2000.0,
            timestamp_ns=time.time_ns(),
        )

        tracker.on_trade(event)

        pos = tracker.get_position("ETH-USD")
        assert pos is not None
        assert pos.side == PositionSide.SHORT
        assert pos.size == Decimal("10")

    def test_on_trade_close_long_records_trade(
        self, tracker: PerformanceTracker
    ) -> None:
        """Test closing long position records trade."""
        # Open long
        buy_event = MockTradeEvent(
            symbol="ETH-USD",
            side="buy",
            quantity=10.0,
            price=2000.0,
            timestamp_ns=time.time_ns(),
        )
        tracker.on_trade(buy_event)

        # Close long
        sell_event = MockTradeEvent(
            symbol="ETH-USD",
            side="sell",
            quantity=10.0,
            price=2100.0,
            timestamp_ns=time.time_ns(),
        )
        tracker.on_trade(sell_event)

        # Check trade was recorded
        trades = tracker.get_trades("ETH-USD")
        assert len(trades) == 1
        assert trades[0].realized_pnl == Decimal("1000")  # 10 * (2100 - 2000)

        # Check position is flat
        pos = tracker.get_position("ETH-USD")
        assert pos is not None
        assert pos.side == PositionSide.FLAT

    def test_partial_close_position(self, tracker: PerformanceTracker) -> None:
        """Test partial position close."""
        # Open long 10 units
        buy_event = MockTradeEvent(
            symbol="ETH-USD",
            side="buy",
            quantity=10.0,
            price=2000.0,
            timestamp_ns=time.time_ns(),
        )
        tracker.on_trade(buy_event)

        # Close 5 units
        sell_event = MockTradeEvent(
            symbol="ETH-USD",
            side="sell",
            quantity=5.0,
            price=2100.0,
            timestamp_ns=time.time_ns(),
        )
        tracker.on_trade(sell_event)

        # Position should be reduced
        pos = tracker.get_position("ETH-USD")
        assert pos is not None
        assert pos.size == Decimal("5")
        assert pos.side == PositionSide.LONG

    def test_position_flip(self, tracker: PerformanceTracker) -> None:
        """Test position flip from long to short."""
        # Open long 5 units
        buy_event = MockTradeEvent(
            symbol="ETH-USD",
            side="buy",
            quantity=5.0,
            price=2000.0,
            timestamp_ns=time.time_ns(),
        )
        tracker.on_trade(buy_event)

        # Sell 10 units (close long 5 + open short 5)
        sell_event = MockTradeEvent(
            symbol="ETH-USD",
            side="sell",
            quantity=10.0,
            price=2100.0,
            timestamp_ns=time.time_ns(),
        )
        tracker.on_trade(sell_event)

        # Should have recorded closing trade
        trades = tracker.get_trades()
        assert len(trades) == 1

        # Should be short now
        pos = tracker.get_position("ETH-USD")
        assert pos is not None
        assert pos.side == PositionSide.SHORT
        assert pos.size == Decimal("5")

    def test_unrealized_pnl_updates_with_price(
        self, tracker: PerformanceTracker
    ) -> None:
        """Test unrealized P&L updates with price changes."""
        # Open long
        buy_event = MockTradeEvent(
            symbol="ETH-USD",
            side="buy",
            quantity=10.0,
            price=2000.0,
            timestamp_ns=time.time_ns(),
        )
        tracker.on_trade(buy_event)

        # Update price
        tracker.update_price("ETH-USD", Decimal("2100"))

        assert tracker.unrealized_pnl == Decimal("1000")

    def test_get_stats_basic(self, tracker: PerformanceTracker) -> None:
        """Test basic statistics calculation."""
        # Make some trades
        trades = [
            ("buy", 10.0, 2000.0),
            ("sell", 10.0, 2100.0),  # +1000 profit
            ("buy", 10.0, 2100.0),
            ("sell", 10.0, 2050.0),  # -500 loss
        ]

        for side, qty, price in trades:
            event = MockTradeEvent(
                symbol="ETH-USD",
                side=side,
                quantity=qty,
                price=price,
                timestamp_ns=time.time_ns(),
            )
            tracker.on_trade(event)

        stats = tracker.get_stats()

        assert stats.total_trades == 2
        assert stats.winning_trades == 1
        assert stats.losing_trades == 1
        assert stats.win_rate == 0.5

    def test_order_fill_tracking(self, tracker: PerformanceTracker) -> None:
        """Test order fill analytics."""
        # Submit order
        tracker.record_order_submit("order-1", Decimal("2000"))

        # Small delay
        import time as time_module

        time_module.sleep(0.001)

        # Record fill
        tracker.record_order_fill(
            "order-1", "ETH-USD", "buy", Decimal("2001"), Decimal("10")
        )

        stats = tracker.get_stats()
        assert stats.filled_orders == 1

    def test_equity_curve(self, tracker: PerformanceTracker) -> None:
        """Test equity curve generation."""
        # Force some snapshots
        tracker._last_snapshot_time = 0

        for i in range(3):
            tracker.update_price("ETH-USD", Decimal(str(2000 + i * 100)))
            tracker._last_snapshot_time = 0  # Force new snapshot

        curve = tracker.get_equity_curve()
        assert len(curve) >= 1

    def test_tracker_reset(self, tracker: PerformanceTracker) -> None:
        """Test resetting tracker state."""
        # Make some trades
        event = MockTradeEvent(
            symbol="ETH-USD",
            side="buy",
            quantity=10.0,
            price=2000.0,
            timestamp_ns=time.time_ns(),
        )
        tracker.on_trade(event)

        # Reset
        tracker.reset()

        assert tracker.equity == tracker.initial_capital
        assert len(tracker._trades) == 0
        assert len(tracker._positions) == 0

    def test_to_dict_export(self, tracker: PerformanceTracker) -> None:
        """Test dictionary export."""
        data = tracker.to_dict()

        assert "equity" in data
        assert "cash" in data
        assert "positions" in data
        assert "stats" in data
        assert data["equity"] == float(tracker.initial_capital)

    def test_get_all_positions(self, tracker: PerformanceTracker) -> None:
        """Test getting all open positions."""
        # Open positions in two symbols
        for symbol in ["ETH-USD", "BTC-USD"]:
            event = MockTradeEvent(
                symbol=symbol,
                side="buy",
                quantity=10.0,
                price=2000.0,
                timestamp_ns=time.time_ns(),
            )
            tracker.on_trade(event)

        positions = tracker.get_all_positions()
        assert len(positions) == 2

    def test_drawdown_tracking(self, tracker: PerformanceTracker) -> None:
        """Test drawdown calculation."""
        # Start with equity at initial capital
        tracker._last_snapshot_time = 0
        tracker.update_price("ETH-USD", Decimal("2000"))

        # Open position
        buy_event = MockTradeEvent(
            symbol="ETH-USD",
            side="buy",
            quantity=10.0,
            price=2000.0,
            timestamp_ns=time.time_ns(),
        )
        tracker.on_trade(buy_event)
        tracker._last_snapshot_time = 0

        # Price goes up - new peak
        tracker.update_price("ETH-USD", Decimal("3000"))
        tracker._last_snapshot_time = 0
        tracker.update_price("ETH-USD", Decimal("3000"))

        # Price drops - drawdown
        tracker.update_price("ETH-USD", Decimal("2500"))
        tracker._last_snapshot_time = 0
        tracker.update_price("ETH-USD", Decimal("2500"))

        curve = tracker.get_drawdown_curve()
        # Should have some drawdown recorded
        assert len(curve) > 0


class TestPerformanceSnapshot:
    """Tests for PerformanceSnapshot dataclass."""

    def test_snapshot_creation(self) -> None:
        """Test snapshot creation."""
        snapshot = PerformanceSnapshot(
            timestamp_ns=time.time_ns(),
            equity=Decimal("105000"),
            cash=Decimal("50000"),
            position_value=Decimal("55000"),
            unrealized_pnl=Decimal("5000"),
            realized_pnl=Decimal("0"),
            total_pnl=Decimal("5000"),
            drawdown_pct=2.5,
        )

        assert snapshot.equity == Decimal("105000")
        assert snapshot.drawdown_pct == 2.5
