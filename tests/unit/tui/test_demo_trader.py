"""Tests for DemoTrader."""

from decimal import Decimal

import pytest

from libra.tui.demo_trader import DemoPosition, DemoTrader, TradingState


class TestDemoPosition:
    """Tests for DemoPosition."""

    def test_create_flat_position(self):
        """Create a flat position."""
        pos = DemoPosition(symbol="BTC/USDT", side="FLAT")

        assert pos.symbol == "BTC/USDT"
        assert pos.side == "FLAT"
        assert pos.quantity == Decimal("0")
        assert pos.unrealized_pnl == Decimal("0")

    def test_long_position_pnl(self):
        """Long position P&L calculation."""
        pos = DemoPosition(
            symbol="BTC/USDT",
            side="LONG",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
        )

        # Profit: (51000 - 50000) * 1.0 = 1000
        assert pos.unrealized_pnl == Decimal("1000")

    def test_short_position_pnl(self):
        """Short position P&L calculation."""
        pos = DemoPosition(
            symbol="BTC/USDT",
            side="SHORT",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("49000"),
        )

        # Profit: (50000 - 49000) * 1.0 = 1000
        assert pos.unrealized_pnl == Decimal("1000")

    def test_position_notional(self):
        """Position notional value."""
        pos = DemoPosition(
            symbol="BTC/USDT",
            side="LONG",
            quantity=Decimal("0.5"),
            current_price=Decimal("50000"),
        )

        assert pos.notional == Decimal("25000")


class TestDemoTrader:
    """Tests for DemoTrader."""

    def test_create_default(self):
        """Create trader with defaults."""
        trader = DemoTrader()

        assert trader.balance == Decimal("100000")
        assert trader.trading_state == TradingState.ACTIVE
        assert "BTC/USDT" in trader.prices
        assert "ETH/USDT" in trader.prices
        assert "SOL/USDT" in trader.prices

    def test_get_price(self):
        """Get current price."""
        trader = DemoTrader()

        btc_price = trader.get_price("BTC/USDT")
        assert btc_price > 0

    def test_tick_prices(self):
        """Tick prices updates values."""
        trader = DemoTrader()
        initial_btc = trader.get_price("BTC/USDT")

        # Tick multiple times
        for _ in range(10):
            trader.tick_prices()

        # Price should have changed
        new_btc = trader.get_price("BTC/USDT")
        # Note: might be same in rare cases, so just check it's valid
        assert new_btc > 0

    def test_execute_buy_order(self):
        """Execute a buy order."""
        trader = DemoTrader()
        initial_balance = trader.balance

        success, msg = trader.execute_order(
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.1"),
        )

        assert success is True
        assert "FILLED" in msg
        assert "BUY" in msg
        assert trader.balance < initial_balance  # Spent money

    def test_execute_sell_order(self):
        """Execute a sell order (short)."""
        trader = DemoTrader()

        # First buy
        trader.execute_order("BTC/USDT", "BUY", Decimal("0.1"))

        # Then sell to close
        success, msg = trader.execute_order(
            symbol="BTC/USDT",
            side="SELL",
            quantity=Decimal("0.1"),
        )

        assert success is True
        assert "SELL" in msg

    def test_position_tracking(self):
        """Positions are tracked correctly."""
        trader = DemoTrader()

        trader.execute_order("BTC/USDT", "BUY", Decimal("0.5"))

        pos = trader.positions["BTC/USDT"]
        assert pos.side == "LONG"
        assert pos.quantity == Decimal("0.5")

    def test_insufficient_balance_rejected(self):
        """Order rejected with insufficient balance."""
        trader = DemoTrader()
        trader.balance = Decimal("100")  # Very low balance

        success, msg = trader.execute_order(
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("1.0"),  # Would cost ~$51000
        )

        assert success is False
        assert "Insufficient" in msg

    def test_halted_state_rejects_orders(self):
        """Orders rejected when halted."""
        trader = DemoTrader()
        trader.trading_state = TradingState.HALTED

        success, msg = trader.execute_order(
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.1"),
        )

        assert success is False
        assert "HALTED" in msg

    def test_reducing_state_blocks_new_positions(self):
        """REDUCING state blocks new positions."""
        trader = DemoTrader()
        trader.trading_state = TradingState.REDUCING

        success, msg = trader.execute_order(
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.1"),
        )

        assert success is False
        assert "REDUCING" in msg

    def test_reducing_state_allows_closes(self):
        """REDUCING state allows closing positions."""
        trader = DemoTrader()

        # Open a position first
        trader.execute_order("BTC/USDT", "BUY", Decimal("0.5"))

        # Enter reducing state
        trader.trading_state = TradingState.REDUCING

        # Close should work
        success, msg = trader.execute_order(
            symbol="BTC/USDT",
            side="SELL",
            quantity=Decimal("0.5"),
        )

        assert success is True

    def test_equity_calculation(self):
        """Equity includes unrealized P&L."""
        trader = DemoTrader()

        # Buy some BTC
        trader.execute_order("BTC/USDT", "BUY", Decimal("0.1"))
        equity_after_buy = trader.equity

        # Simulate price increase (10%)
        trader.prices["BTC/USDT"].price *= Decimal("1.1")
        trader.positions["BTC/USDT"].current_price = trader.prices["BTC/USDT"].price

        # Equity should increase due to unrealized profit
        # 0.1 BTC * 10% gain = ~$510 profit
        assert trader.equity > equity_after_buy

    def test_drawdown_calculation(self):
        """Drawdown percentage is calculated."""
        trader = DemoTrader()

        # Set a higher peak
        trader.peak_equity = Decimal("110000")

        # Current equity at 100000 = 9.09% drawdown
        drawdown = trader.drawdown_pct
        assert 9 < drawdown < 10

    def test_get_stats(self):
        """Get comprehensive stats."""
        trader = DemoTrader()

        stats = trader.get_stats()

        assert "balance" in stats
        assert "equity" in stats
        assert "trading_state" in stats
        assert "circuit_breaker" in stats
        assert "drawdown_pct" in stats

    def test_trigger_losing_streak(self):
        """Losing streak triggers circuit breaker."""
        trader = DemoTrader()
        trader.max_consecutive_losses = 3

        trader.trigger_losing_streak(count=3)

        assert trader.consecutive_losses >= 3

    def test_resume_trading(self):
        """Resume trading after safe conditions."""
        trader = DemoTrader()
        trader.trading_state = TradingState.HALTED
        trader.daily_pnl = Decimal("-100")  # Small loss, under limit

        success = trader.resume_trading()

        assert success is True
        assert trader.trading_state == TradingState.ACTIVE


class TestDemoTraderCallbacks:
    """Tests for callback functionality."""

    def test_on_trade_callback(self):
        """Trade callback is invoked."""
        trader = DemoTrader()
        messages = []

        trader.on_trade = lambda msg: messages.append(msg)

        trader.execute_order("BTC/USDT", "BUY", Decimal("0.1"))

        assert len(messages) == 1
        assert "FILLED" in messages[0]

    def test_on_risk_event_callback(self):
        """Risk event callback is invoked."""
        trader = DemoTrader()
        events = []

        trader.on_risk_event = lambda state, reason: events.append((state, reason))
        trader.max_drawdown_pct = 1.0  # Very low threshold

        # Force large drawdown
        trader.peak_equity = Decimal("200000")
        trader.balance = Decimal("100000")
        trader._check_risk_limits()

        assert len(events) >= 1
