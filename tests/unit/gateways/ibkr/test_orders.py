"""
Unit tests for IBKR Order Type Mappings.

Tests order conversion and status mapping.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from libra.gateways.protocol import Order, OrderSide, OrderStatus, OrderType, TimeInForce

from libra.gateways.ibkr.orders import map_ib_status


class TestMapIBStatus:
    """Tests for map_ib_status function."""

    def test_pending_states(self) -> None:
        """Pending IB states map correctly."""
        assert map_ib_status("PendingSubmit") == OrderStatus.PENDING
        assert map_ib_status("PendingCancel") == OrderStatus.PENDING
        assert map_ib_status("PreSubmitted") == OrderStatus.PENDING
        assert map_ib_status("ApiPending") == OrderStatus.PENDING

    def test_open_state(self) -> None:
        """Submitted maps to OPEN."""
        assert map_ib_status("Submitted") == OrderStatus.OPEN

    def test_filled_states(self) -> None:
        """Filled IB states map correctly."""
        assert map_ib_status("Filled") == OrderStatus.FILLED
        assert map_ib_status("PartiallyFilled") == OrderStatus.PARTIALLY_FILLED

    def test_cancelled_states(self) -> None:
        """Cancelled IB states map correctly."""
        assert map_ib_status("Cancelled") == OrderStatus.CANCELLED
        assert map_ib_status("ApiCancelled") == OrderStatus.CANCELLED

    def test_rejected_state(self) -> None:
        """Inactive maps to REJECTED."""
        assert map_ib_status("Inactive") == OrderStatus.REJECTED

    def test_unknown_state(self) -> None:
        """Unknown status defaults to PENDING."""
        assert map_ib_status("SomeUnknownStatus") == OrderStatus.PENDING


class TestBuildIBOrder:
    """Tests for build_ib_order function.

    These tests require ib_async, so they skip if not installed.
    """

    @pytest.fixture
    def market_order(self) -> Order:
        """Create a market order."""
        return Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
        )

    @pytest.fixture
    def limit_order(self) -> Order:
        """Create a limit order."""
        return Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("50"),
            price=Decimal("150.00"),
        )

    @pytest.fixture
    def stop_order(self) -> Order:
        """Create a stop order."""
        return Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("100"),
            stop_price=Decimal("145.00"),
        )

    def test_build_market_order_import_error(self, market_order: Order) -> None:
        """build_ib_order raises ImportError when ib_async not installed."""
        try:
            import ib_async
            pytest.skip("ib_async is installed")
        except ImportError:
            pass

        from libra.gateways.ibkr.orders import build_ib_order

        with pytest.raises(ImportError, match="ib_async is not installed"):
            build_ib_order(market_order)

    def test_limit_order_requires_price(self) -> None:
        """Limit order without price raises ValueError."""
        try:
            import ib_async
        except ImportError:
            pytest.skip("ib_async not installed")

        from libra.gateways.ibkr.orders import build_ib_order

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("100"),
            # No price!
        )

        with pytest.raises(ValueError, match="Limit order requires price"):
            build_ib_order(order)

    def test_stop_order_requires_stop_price(self) -> None:
        """Stop order without stop_price raises ValueError."""
        try:
            import ib_async
        except ImportError:
            pytest.skip("ib_async not installed")

        from libra.gateways.ibkr.orders import build_ib_order

        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("100"),
            # No stop_price!
        )

        with pytest.raises(ValueError, match="Stop order requires stop_price"):
            build_ib_order(order)
