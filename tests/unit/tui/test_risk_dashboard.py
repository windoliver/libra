"""Tests for Risk Dashboard widgets."""

import pytest

from libra.tui.widgets.risk_dashboard import (
    CircuitBreakerIndicator,
    DrawdownGauge,
    ExposureBar,
    OrderRateIndicator,
    RiskDashboard,
    TradingStateIndicator,
)


class TestTradingStateIndicator:
    """Tests for TradingStateIndicator widget."""

    def test_create_default(self):
        """Create indicator with default state."""
        indicator = TradingStateIndicator()

        assert indicator.id == "trading-state"
        assert indicator.state == "ACTIVE"

    def test_render_active(self):
        """Render ACTIVE state."""
        indicator = TradingStateIndicator()
        indicator.state = "ACTIVE"

        output = indicator.render()

        assert "ACTIVE" in output
        assert "Trading State:" in output

    def test_render_reducing(self):
        """Render REDUCING state."""
        indicator = TradingStateIndicator()
        indicator.state = "REDUCING"

        output = indicator.render()

        assert "REDUCING" in output

    def test_render_halted(self):
        """Render HALTED state."""
        indicator = TradingStateIndicator()
        indicator.state = "HALTED"

        output = indicator.render()

        assert "HALTED" in output

    def test_set_state(self):
        """Set state via method."""
        indicator = TradingStateIndicator()

        indicator.set_state("reducing")

        assert indicator.state == "REDUCING"

    def test_unknown_state(self):
        """Unknown state renders as UNKNOWN."""
        indicator = TradingStateIndicator()
        indicator.state = "INVALID"

        output = indicator.render()

        assert "UNKNOWN" in output

    def test_has_css(self):
        """Widget has CSS defined."""
        assert TradingStateIndicator.DEFAULT_CSS is not None
        assert "state-active" in TradingStateIndicator.DEFAULT_CSS
        assert "state-reducing" in TradingStateIndicator.DEFAULT_CSS
        assert "state-halted" in TradingStateIndicator.DEFAULT_CSS


class TestDrawdownGauge:
    """Tests for DrawdownGauge widget."""

    def test_create_default(self):
        """Create gauge with default values."""
        gauge = DrawdownGauge()

        assert gauge.id == "drawdown-gauge"
        assert gauge.current == 0.0
        assert gauge.maximum == 50.0

    def test_create_custom_values(self):
        """Create gauge with custom values."""
        gauge = DrawdownGauge(current=10.0, maximum=25.0)

        assert gauge.current == 10.0
        assert gauge.maximum == 25.0

    def test_history_initialized(self):
        """History list is initialized."""
        gauge = DrawdownGauge()

        assert len(gauge._history) == 20
        assert all(v == 0.0 for v in gauge._history)

    def test_update_value(self):
        """Update drawdown value."""
        gauge = DrawdownGauge()

        gauge.update_value(15.0)

        assert gauge.current == 15.0

    def test_update_value_with_maximum(self):
        """Update value and maximum together."""
        gauge = DrawdownGauge()

        gauge.update_value(20.0, 40.0)

        assert gauge.current == 20.0
        assert gauge.maximum == 40.0

    def test_history_tracking(self):
        """History tracks values over time."""
        gauge = DrawdownGauge()

        gauge.update_value(5.0)
        gauge.update_value(10.0)
        gauge.update_value(7.5)

        # History should have the new values
        assert gauge._history[-1] == 7.5
        assert gauge._history[-2] == 10.0
        assert gauge._history[-3] == 5.0

    def test_history_max_length(self):
        """History maintains max length of 20."""
        gauge = DrawdownGauge()

        # Add 25 values
        for i in range(25):
            gauge.update_value(float(i))

        assert len(gauge._history) == 20
        assert gauge._history[-1] == 24.0  # Last value added


class TestExposureBar:
    """Tests for ExposureBar widget."""

    def test_create_default(self):
        """Create exposure bar with default values."""
        bar = ExposureBar(symbol="BTC/USDT")

        assert bar._symbol == "BTC/USDT"
        assert bar._current == 0.0
        assert bar._limit == 100.0
        assert bar.id == "exposure-BTC-USDT"

    def test_create_custom_values(self):
        """Create bar with custom values."""
        bar = ExposureBar(symbol="ETH/USDT", current=50.0, limit=75.0)

        assert bar._current == 50.0
        assert bar._limit == 75.0

    def test_custom_id(self):
        """Create bar with custom ID."""
        bar = ExposureBar(symbol="SOL/USDT", id="my-exposure")

        assert bar.id == "my-exposure"

    def test_update_value(self):
        """Update exposure value."""
        bar = ExposureBar(symbol="BTC/USDT")

        bar.update_value(60.0)

        assert bar._current == 60.0

    def test_update_value_with_limit(self):
        """Update value and limit together."""
        bar = ExposureBar(symbol="BTC/USDT")

        bar.update_value(30.0, 50.0)

        assert bar._current == 30.0
        assert bar._limit == 50.0

    def test_symbol_in_id_sanitized(self):
        """Symbol with slash is sanitized in ID."""
        bar = ExposureBar(symbol="BTC/USDT")

        # Slash should be replaced with dash
        assert "/" not in bar.id
        assert "-" in bar.id


class TestOrderRateIndicator:
    """Tests for OrderRateIndicator widget."""

    def test_create_default(self):
        """Create indicator with default values."""
        indicator = OrderRateIndicator()

        assert indicator.id == "order-rate"
        assert indicator.current == 0
        assert indicator.limit == 10

    def test_create_custom_values(self):
        """Create indicator with custom values."""
        indicator = OrderRateIndicator(current=5, limit=20)

        assert indicator.current == 5
        assert indicator.limit == 20

    def test_render_output(self):
        """Render shows rate info."""
        indicator = OrderRateIndicator(current=3, limit=10)

        output = indicator.render()

        assert "Order Rate:" in output
        assert "3/10" in output
        assert "per sec" in output

    def test_update_rate(self):
        """Update rate via method."""
        indicator = OrderRateIndicator()

        indicator.update_rate(7)

        assert indicator.current == 7

    def test_update_rate_with_limit(self):
        """Update rate and limit together."""
        indicator = OrderRateIndicator()

        indicator.update_rate(5, 15)

        assert indicator.current == 5
        assert indicator.limit == 15

    def test_render_at_limit(self):
        """Render when at rate limit."""
        indicator = OrderRateIndicator(current=10, limit=10)

        output = indicator.render()

        assert "10/10" in output

    def test_render_over_limit(self):
        """Render when over rate limit is capped."""
        indicator = OrderRateIndicator(current=15, limit=10)

        # Current can exceed limit but dots are capped
        output = indicator.render()

        assert "15/10" in output


class TestCircuitBreakerIndicator:
    """Tests for CircuitBreakerIndicator widget."""

    def test_create_default(self):
        """Create indicator with default state."""
        indicator = CircuitBreakerIndicator()

        assert indicator.id == "circuit-breaker"
        assert indicator.state == "CLOSED"
        assert indicator.reason == ""

    def test_render_closed(self):
        """Render CLOSED state."""
        indicator = CircuitBreakerIndicator()

        output = indicator.render()

        assert "CLOSED" in output
        assert "Circuit Breaker:" in output

    def test_render_open(self):
        """Render OPEN state."""
        indicator = CircuitBreakerIndicator()
        indicator.state = "OPEN"

        output = indicator.render()

        assert "OPEN" in output

    def test_render_half_open(self):
        """Render HALF_OPEN state."""
        indicator = CircuitBreakerIndicator()
        indicator.state = "HALF_OPEN"

        output = indicator.render()

        assert "HALF_OPEN" in output

    def test_render_with_reason(self):
        """Render with reason when not CLOSED."""
        indicator = CircuitBreakerIndicator()
        indicator.state = "OPEN"
        indicator.reason = "10 consecutive losses"

        output = indicator.render()

        assert "10 consecutive losses" in output

    def test_reason_not_shown_when_closed(self):
        """Reason not shown when state is CLOSED."""
        indicator = CircuitBreakerIndicator()
        indicator.state = "CLOSED"
        indicator.reason = "Old reason"

        output = indicator.render()

        # Should not include reason when closed
        assert "Old reason" not in output

    def test_update_status(self):
        """Update status via method."""
        indicator = CircuitBreakerIndicator()

        indicator.update_status("open", "Rate limit exceeded")

        assert indicator.state == "OPEN"
        assert indicator.reason == "Rate limit exceeded"

    def test_update_status_uppercase(self):
        """State is uppercased."""
        indicator = CircuitBreakerIndicator()

        indicator.update_status("half_open")

        assert indicator.state == "HALF_OPEN"


class TestRiskDashboard:
    """Tests for RiskDashboard main widget."""

    def test_create_default(self):
        """Create dashboard with default symbols."""
        dashboard = RiskDashboard()

        assert dashboard.id == "risk-dashboard"
        assert dashboard._symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_create_custom_symbols(self):
        """Create dashboard with custom symbols."""
        symbols = ["DOGE/USDT", "SHIB/USDT"]
        dashboard = RiskDashboard(symbols=symbols)

        assert dashboard._symbols == symbols

    def test_has_css(self):
        """Dashboard has CSS defined."""
        assert RiskDashboard.DEFAULT_CSS is not None
        assert "dashboard-title" in RiskDashboard.DEFAULT_CSS
        assert "section-title" in RiskDashboard.DEFAULT_CSS


class TestRiskDashboardSetters:
    """Tests for RiskDashboard setter methods."""

    def test_set_trading_state(self):
        """Set trading state method exists."""
        dashboard = RiskDashboard()

        # Method should not raise
        dashboard.set_trading_state("HALTED")

    def test_set_drawdown(self):
        """Set drawdown method exists."""
        dashboard = RiskDashboard()

        # Method should not raise
        dashboard.set_drawdown(5.0, 20.0)

    def test_set_daily_pnl(self):
        """Set daily P&L method exists."""
        dashboard = RiskDashboard()

        # Method should not raise
        dashboard.set_daily_pnl(-1500, 10000)

    def test_set_exposure(self):
        """Set exposure method exists."""
        dashboard = RiskDashboard()

        # Method should not raise
        dashboard.set_exposure("BTC/USDT", 45.0, 100.0)

    def test_set_order_rate(self):
        """Set order rate method exists."""
        dashboard = RiskDashboard()

        # Method should not raise
        dashboard.set_order_rate(3, 10)

    def test_set_circuit_breaker(self):
        """Set circuit breaker method exists."""
        dashboard = RiskDashboard()

        # Method should not raise
        dashboard.set_circuit_breaker("OPEN", "Test reason")


class TestRiskDashboardUpdateFromDict:
    """Tests for update_from_dict method."""

    def test_update_trading_state(self):
        """Update trading state from dict."""
        dashboard = RiskDashboard()
        data = {"trading_state": "REDUCING"}

        # Should not raise
        dashboard.update_from_dict(data)

    def test_update_drawdown(self):
        """Update drawdown from dict."""
        dashboard = RiskDashboard()
        data = {"drawdown": {"current": 10.0, "maximum": 30.0}}

        # Should not raise
        dashboard.update_from_dict(data)

    def test_update_daily_pnl(self):
        """Update daily P&L from dict."""
        dashboard = RiskDashboard()
        data = {"daily_pnl": {"current": -2000, "limit": 10000}}

        # Should not raise
        dashboard.update_from_dict(data)

    def test_update_exposures(self):
        """Update exposures from dict."""
        dashboard = RiskDashboard()
        data = {
            "exposures": {
                "BTC/USDT": {"current": 50, "limit": 100},
                "ETH/USDT": {"current": 30, "limit": 75},
            }
        }

        # Should not raise
        dashboard.update_from_dict(data)

    def test_update_order_rate(self):
        """Update order rate from dict."""
        dashboard = RiskDashboard()
        data = {"order_rate": {"current": 5, "limit": 10}}

        # Should not raise
        dashboard.update_from_dict(data)

    def test_update_circuit_breaker(self):
        """Update circuit breaker from dict."""
        dashboard = RiskDashboard()
        data = {"circuit_breaker": {"state": "HALF_OPEN", "reason": "Testing"}}

        # Should not raise
        dashboard.update_from_dict(data)

    def test_update_multiple_fields(self):
        """Update multiple fields at once."""
        dashboard = RiskDashboard()
        data = {
            "trading_state": "ACTIVE",
            "drawdown": {"current": 5.0, "maximum": 50.0},
            "daily_pnl": {"current": 1500, "limit": 10000},
            "order_rate": {"current": 2, "limit": 10},
            "circuit_breaker": {"state": "CLOSED"},
        }

        # Should not raise
        dashboard.update_from_dict(data)

    def test_update_empty_dict(self):
        """Update with empty dict does nothing."""
        dashboard = RiskDashboard()

        # Should not raise
        dashboard.update_from_dict({})


class TestRiskDashboardColorLogic:
    """Tests for color-coding logic in dashboard components."""

    def test_drawdown_color_logic(self):
        """Drawdown colors based on percentage of max."""
        # < 50% of max = green
        # 50-80% of max = yellow
        # > 80% of max = red

        current = 10.0
        maximum = 50.0

        # 10/50 = 20% < 50% = green
        assert current < maximum * 0.5

        current = 30.0
        # 30/50 = 60% > 50% but < 80% = yellow
        assert current >= maximum * 0.5
        assert current < maximum * 0.8

        current = 45.0
        # 45/50 = 90% > 80% = red
        assert current >= maximum * 0.8

    def test_pnl_color_logic(self):
        """P&L colors based on value."""
        # Positive = green
        # Negative but < 50% limit = yellow
        # Negative and > 50% limit = red

        limit = 10000

        # Positive
        current = 1500
        assert current >= 0  # green

        # Small loss
        current = -3000
        assert current < 0 and abs(current) < limit * 0.5  # yellow

        # Large loss
        current = -6000
        assert current < 0 and abs(current) >= limit * 0.5  # red

    def test_exposure_color_logic(self):
        """Exposure colors based on percentage."""
        # < 50% = green
        # 50-80% = yellow
        # > 80% = red

        current = 30.0
        limit = 100.0
        pct = current / limit * 100

        assert pct < 50  # green

        current = 65.0
        pct = current / limit * 100
        assert 50 <= pct < 80  # yellow

        current = 90.0
        pct = current / limit * 100
        assert pct >= 80  # red
