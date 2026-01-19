"""Tests for TUI reactive price properties (Issue #88)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest


class TestReactivePriceProperties:
    """Test reactive price properties on LibraApp."""

    def test_reactive_properties_exist(self):
        """Verify reactive properties are defined on the class."""
        from libra.tui.app import LibraApp

        # Check that reactive properties are class attributes
        assert hasattr(LibraApp, "btc_price")
        assert hasattr(LibraApp, "eth_price")
        assert hasattr(LibraApp, "sol_price")
        assert hasattr(LibraApp, "total_pnl")

    def test_reactive_default_values(self):
        """Verify reactive properties have correct default values."""
        from libra.tui.app import LibraApp

        # Get default values from the reactive descriptors (_default attr)
        assert LibraApp.btc_price._default == Decimal("51250.00")
        assert LibraApp.eth_price._default == Decimal("3045.00")
        assert LibraApp.sol_price._default == Decimal("142.50")
        assert LibraApp.total_pnl._default == Decimal("347.50")

    def test_watch_total_pnl_method_exists(self):
        """Verify watch_total_pnl method exists for automatic updates."""
        from libra.tui.app import LibraApp

        assert hasattr(LibraApp, "watch_total_pnl")
        assert callable(getattr(LibraApp, "watch_total_pnl"))

    def test_watch_total_pnl_updates_display(self):
        """Test that watch_total_pnl updates the PnL display widget."""
        from libra.tui.app import LibraApp

        # Create a mock app instance
        app = MagicMock(spec=LibraApp)
        app._cached_pnl_display = MagicMock()

        # Call the watch method directly
        LibraApp.watch_total_pnl(app, Decimal("1000.50"))

        # Verify the display was updated
        app._cached_pnl_display.update_pnl.assert_called_once_with(Decimal("1000.50"))

    def test_watch_total_pnl_handles_no_display(self):
        """Test that watch_total_pnl handles missing display gracefully."""
        from libra.tui.app import LibraApp

        # Create a mock app instance with no display
        app = MagicMock(spec=LibraApp)
        app._cached_pnl_display = None

        # Should not raise an exception
        LibraApp.watch_total_pnl(app, Decimal("1000.50"))


class TestPnLDisplay:
    """Test PnLDisplay widget updates."""

    def test_update_pnl_positive(self):
        """Test PnL display with positive value shows up icon."""
        from libra.tui.app import PnLDisplay, ICON_UP

        display = PnLDisplay()
        display._cached_digits = MagicMock()

        display.update_pnl(Decimal("100.50"))

        display._cached_digits.update.assert_called_once()
        call_arg = display._cached_digits.update.call_args[0][0]
        assert ICON_UP in call_arg
        assert "+$100.50" in call_arg

    def test_update_pnl_negative(self):
        """Test PnL display with negative value shows down icon."""
        from libra.tui.app import PnLDisplay, ICON_DOWN

        display = PnLDisplay()
        display._cached_digits = MagicMock()

        display.update_pnl(Decimal("-50.25"))

        display._cached_digits.update.assert_called_once()
        call_arg = display._cached_digits.update.call_args[0][0]
        assert ICON_DOWN in call_arg
        assert "$-50.25" in call_arg
        display._cached_digits.add_class.assert_called_with("negative")

    def test_update_pnl_zero(self):
        """Test PnL display with zero value shows neutral icon."""
        from libra.tui.app import PnLDisplay, ICON_NEUTRAL

        display = PnLDisplay()
        display._cached_digits = MagicMock()

        display.update_pnl(Decimal("0"))

        display._cached_digits.update.assert_called_once()
        call_arg = display._cached_digits.update.call_args[0][0]
        assert ICON_NEUTRAL in call_arg


class TestReactiveIntegration:
    """Integration tests for reactive properties."""

    def test_price_properties_used_in_tick(self):
        """Verify price properties are used correctly in tick methods."""
        from libra.tui.app import LibraApp

        # Verify the source code uses reactive properties (not _underscore versions)
        import inspect
        source = inspect.getsource(LibraApp._lite_tick)

        # Check that reactive properties are used (without underscore prefix)
        assert "self.btc_price" in source
        assert "self.eth_price" in source
        assert "self.sol_price" in source
        assert "self.total_pnl" in source

        # Check that old underscore versions are NOT used
        assert "self._btc_price" not in source
        assert "self._eth_price" not in source
        assert "self._sol_price" not in source
        assert "self._total_pnl" not in source

    def test_reactive_import_present(self):
        """Verify reactive is imported from textual."""
        from libra.tui import app

        assert hasattr(app, "reactive")
