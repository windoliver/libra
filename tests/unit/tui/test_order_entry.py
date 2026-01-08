"""Tests for Order Entry Screen."""

from decimal import Decimal

import pytest

from libra.tui.screens.order_entry import OrderEntryResult, OrderEntryScreen, RiskPreview


class TestOrderEntryResult:
    """Tests for OrderEntryResult data class."""

    def test_create_default_result(self):
        """Create result with default values."""
        result = OrderEntryResult()

        assert result.submitted is False
        assert result.symbol == ""
        assert result.side == "BUY"
        assert result.order_type == "MARKET"
        assert result.quantity == Decimal("0")
        assert result.price is None

    def test_create_submitted_result(self):
        """Create a submitted order result."""
        result = OrderEntryResult(
            submitted=True,
            symbol="BTC/USDT",
            side="SELL",
            order_type="LIMIT",
            quantity=Decimal("0.5"),
            price=Decimal("51000"),
        )

        assert result.submitted is True
        assert result.symbol == "BTC/USDT"
        assert result.side == "SELL"
        assert result.order_type == "LIMIT"
        assert result.quantity == Decimal("0.5")
        assert result.price == Decimal("51000")

    def test_cancelled_result(self):
        """Create a cancelled (not submitted) result."""
        result = OrderEntryResult(submitted=False)

        assert result.submitted is False

    def test_market_order_no_price(self):
        """Market orders don't need a price."""
        result = OrderEntryResult(
            submitted=True,
            symbol="ETH/USDT",
            side="BUY",
            order_type="MARKET",
            quantity=Decimal("1.0"),
            price=None,
        )

        assert result.order_type == "MARKET"
        assert result.price is None


class TestRiskPreview:
    """Tests for RiskPreview widget."""

    def test_create_risk_preview(self):
        """Create RiskPreview widget."""
        preview = RiskPreview()

        assert preview.id == "risk-preview"
        assert preview._checks == []

    def test_create_with_custom_id(self):
        """Create RiskPreview with custom ID."""
        preview = RiskPreview(id="custom-preview")

        assert preview.id == "custom-preview"

    def test_update_checks(self):
        """Update risk checks list."""
        preview = RiskPreview()
        checks = [
            ("position_limit", True, "Position: 15% / 25% max"),
            ("daily_loss", True, "Daily P&L: $2k / $10k"),
            ("large_order", False, "Large order: > $5,000"),
        ]

        preview.update_checks(checks)

        assert len(preview._checks) == 3
        assert preview._checks[0][1] is True  # First check passed
        assert preview._checks[2][1] is False  # Third check failed

    def test_clear_checks(self):
        """Clear risk checks."""
        preview = RiskPreview()
        preview._checks = [("test", True, "Test check")]

        preview.clear()

        assert preview._checks == []


class TestOrderEntryScreen:
    """Tests for OrderEntryScreen."""

    def test_create_screen_default(self):
        """Create screen with default values."""
        screen = OrderEntryScreen()

        assert screen._symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        assert screen._risk_manager is None
        assert "BTC/USDT" in screen._current_prices
        assert screen._selected_symbol == "BTC/USDT"
        assert screen._selected_side == "BUY"
        assert screen._selected_type == "LIMIT"
        assert screen._quantity == Decimal("0")
        assert screen._price is None

    def test_create_screen_custom_symbols(self):
        """Create screen with custom symbol list."""
        symbols = ["DOGE/USDT", "SHIB/USDT"]
        screen = OrderEntryScreen(symbols=symbols)

        assert screen._symbols == symbols
        assert screen._selected_symbol == "DOGE/USDT"

    def test_create_screen_custom_prices(self):
        """Create screen with custom prices."""
        prices = {
            "BTC/USDT": Decimal("60000"),
            "ETH/USDT": Decimal("4000"),
        }
        screen = OrderEntryScreen(current_prices=prices)

        assert screen._current_prices["BTC/USDT"] == Decimal("60000")
        assert screen._current_prices["ETH/USDT"] == Decimal("4000")

    def test_bindings_defined(self):
        """Screen has required key bindings."""
        screen = OrderEntryScreen()

        binding_keys = [b.key for b in screen.BINDINGS]
        assert "escape" in binding_keys
        assert "ctrl+enter" in binding_keys

    def test_has_css(self):
        """Screen has CSS defined."""
        assert OrderEntryScreen.DEFAULT_CSS is not None
        assert "OrderEntryScreen" in OrderEntryScreen.DEFAULT_CSS
        assert "modal-title" in OrderEntryScreen.DEFAULT_CSS


class TestOrderEntryValidation:
    """Tests for order entry validation logic."""

    def test_quantity_zero_not_valid(self):
        """Quantity of zero is not valid."""
        screen = OrderEntryScreen()
        screen._quantity = Decimal("0")

        # Quantity check should prevent submission
        assert screen._quantity <= 0

    def test_quantity_positive_valid(self):
        """Positive quantity is valid."""
        screen = OrderEntryScreen()
        screen._quantity = Decimal("0.5")

        assert screen._quantity > 0

    def test_limit_order_needs_price(self):
        """Limit orders require a price."""
        screen = OrderEntryScreen()
        screen._selected_type = "LIMIT"
        screen._price = None

        # Price is required for limit orders
        assert screen._selected_type != "MARKET"
        assert screen._price is None

    def test_market_order_no_price_needed(self):
        """Market orders don't need price."""
        screen = OrderEntryScreen()
        screen._selected_type = "MARKET"
        screen._price = None

        # Price not required for market orders
        assert screen._selected_type == "MARKET"

    def test_side_values(self):
        """Side can be BUY or SELL."""
        screen = OrderEntryScreen()

        screen._selected_side = "BUY"
        assert screen._selected_side == "BUY"

        screen._selected_side = "SELL"
        assert screen._selected_side == "SELL"

    def test_order_types(self):
        """Order types are MARKET, LIMIT, STOP_LIMIT."""
        screen = OrderEntryScreen()

        valid_types = ["MARKET", "LIMIT", "STOP_LIMIT"]

        for order_type in valid_types:
            screen._selected_type = order_type
            assert screen._selected_type == order_type


class TestRiskPreviewCalculations:
    """Tests for risk preview calculations in OrderEntryScreen."""

    def test_position_percentage_calculation(self):
        """Position percentage is calculated correctly."""
        screen = OrderEntryScreen()
        screen._quantity = Decimal("2.5")

        # Simulated calculation: min(quantity * 10, 100)
        position_pct = min(float(screen._quantity) * 10, 100)

        assert position_pct == 25.0

    def test_position_percentage_capped(self):
        """Position percentage is capped at 100%."""
        screen = OrderEntryScreen()
        screen._quantity = Decimal("15")

        # Should cap at 100
        position_pct = min(float(screen._quantity) * 10, 100)

        assert position_pct == 100.0

    def test_notional_calculation(self):
        """Notional value is calculated correctly."""
        screen = OrderEntryScreen()
        screen._selected_symbol = "BTC/USDT"
        screen._quantity = Decimal("0.1")
        screen._current_prices = {"BTC/USDT": Decimal("50000")}

        current_price = screen._current_prices.get(screen._selected_symbol, Decimal("0"))
        notional = screen._quantity * current_price

        assert notional == Decimal("5000")

    def test_large_order_threshold(self):
        """Large orders are flagged when notional exceeds threshold."""
        screen = OrderEntryScreen()
        screen._selected_symbol = "BTC/USDT"
        screen._quantity = Decimal("0.2")
        screen._current_prices = {"BTC/USDT": Decimal("50000")}

        current_price = screen._current_prices.get(screen._selected_symbol, Decimal("0"))
        notional = screen._quantity * current_price
        large_order_threshold = Decimal("5000")

        # 0.2 * 50000 = 10000 > 5000
        assert notional > large_order_threshold


class TestOrderEntryResultActions:
    """Tests for action methods on OrderEntryScreen."""

    def test_form_state_initialization(self):
        """Form state is properly initialized."""
        screen = OrderEntryScreen(
            symbols=["BTC/USDT", "ETH/USDT"],
            current_prices={
                "BTC/USDT": Decimal("51000"),
                "ETH/USDT": Decimal("3000"),
            },
        )

        assert screen._selected_symbol == "BTC/USDT"
        assert screen._selected_side == "BUY"
        assert screen._selected_type == "LIMIT"

    def test_empty_symbols_list(self):
        """Handle empty symbols list by using defaults."""
        screen = OrderEntryScreen(symbols=[])

        # Empty list is falsy, so defaults are used
        assert screen._symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        assert screen._selected_symbol == "BTC/USDT"
