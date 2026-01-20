"""
Tests for order reconciliation (Issue #109).
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from libra.execution.reconciliation import (
    Discrepancy,
    OrderReconciler,
    ReconciliationAction,
    ReconciliationResult,
)
from libra.gateways.protocol import (
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_execution_client():
    """Create a mock execution client."""
    client = MagicMock()
    client.name = "test-client"
    client.get_open_orders = AsyncMock(return_value=[])
    client.get_positions = AsyncMock(return_value=[])
    client.get_order = AsyncMock()
    client.get_position = AsyncMock()
    return client


@pytest.fixture
def mock_cache():
    """Create a mock cache."""
    cache = MagicMock()
    cache.open_orders = MagicMock(return_value=[])
    cache.positions = MagicMock(return_value=[])
    cache.position = MagicMock(return_value=None)
    cache.order_by_exchange_id = MagicMock(return_value=None)
    cache.add_order = AsyncMock()
    cache.update_position = AsyncMock()
    return cache


@pytest.fixture
def mock_risk_engine():
    """Create a mock risk engine."""
    engine = MagicMock()
    engine.add_open_order = MagicMock()
    engine.remove_open_order = MagicMock()
    return engine


def make_order_result(
    order_id: str,
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    amount: Decimal = Decimal("1.0"),
    price: Decimal | None = Decimal("50000"),
    status: OrderStatus = OrderStatus.OPEN,
    filled_amount: Decimal = Decimal("0"),
    client_order_id: str | None = None,
) -> OrderResult:
    """Create a test OrderResult."""
    return OrderResult(
        order_id=order_id,
        symbol=symbol,
        status=status,
        side=side,
        order_type=OrderType.LIMIT,
        amount=amount,
        filled_amount=filled_amount,
        remaining_amount=amount - filled_amount,
        average_price=None,
        fee=Decimal("0"),
        fee_currency="USDT",
        timestamp_ns=1000000000,
        client_order_id=client_order_id or order_id,
        price=price,
    )


def make_position(
    symbol: str = "BTC/USDT",
    side: PositionSide = PositionSide.LONG,
    amount: Decimal = Decimal("1.0"),
    entry_price: Decimal = Decimal("50000"),
    current_price: Decimal = Decimal("51000"),
) -> Position:
    """Create a test Position."""
    return Position(
        symbol=symbol,
        side=side,
        amount=amount,
        entry_price=entry_price,
        current_price=current_price,
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
    )


# =============================================================================
# ReconciliationResult Tests
# =============================================================================


class TestReconciliationResult:
    """Tests for ReconciliationResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = ReconciliationResult()

        assert result.orders_added == 0
        assert result.orders_removed == 0
        assert result.orders_updated == 0
        assert result.positions_added == 0
        assert result.positions_removed == 0
        assert result.positions_adjusted == 0
        assert result.discrepancies == []
        assert result.warnings == []
        assert result.errors == []
        assert result.success is True

    def test_total_discrepancies(self):
        """Test total_discrepancies property."""
        result = ReconciliationResult()
        assert result.total_discrepancies == 0

        result.discrepancies.append(
            Discrepancy(
                action=ReconciliationAction.ADD_LOCAL,
                entity_type="order",
                identifier="123",
                details="test",
            )
        )
        assert result.total_discrepancies == 1

    def test_had_changes_false(self):
        """Test had_changes when no changes."""
        result = ReconciliationResult()
        assert result.had_changes is False

    def test_had_changes_true_orders_added(self):
        """Test had_changes with orders added."""
        result = ReconciliationResult(orders_added=1)
        assert result.had_changes is True

    def test_had_changes_true_orders_removed(self):
        """Test had_changes with orders removed."""
        result = ReconciliationResult(orders_removed=1)
        assert result.had_changes is True

    def test_had_changes_true_positions_adjusted(self):
        """Test had_changes with positions adjusted."""
        result = ReconciliationResult(positions_adjusted=1)
        assert result.had_changes is True

    def test_summary_no_changes(self):
        """Test summary with no changes."""
        result = ReconciliationResult()
        assert result.summary() == "No discrepancies found"

    def test_summary_with_changes(self):
        """Test summary with changes."""
        result = ReconciliationResult(
            orders_added=2,
            orders_removed=1,
            positions_adjusted=3,
        )
        summary = result.summary()
        assert "orders_added=2" in summary
        assert "orders_removed=1" in summary
        assert "positions_adjusted=3" in summary


# =============================================================================
# OrderReconciler Tests - Order Reconciliation
# =============================================================================


class TestOrderReconcilerOrders:
    """Tests for OrderReconciler order reconciliation."""

    @pytest.mark.asyncio
    async def test_no_discrepancies(self, mock_execution_client, mock_cache):
        """Test when venue and local are in sync."""
        order = make_order_result("order-1")

        mock_execution_client.get_open_orders.return_value = [order]
        mock_cache.open_orders.return_value = [order]

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        result = await reconciler.reconcile()

        assert result.success is True
        assert result.had_changes is False
        assert result.orders_added == 0
        assert result.orders_removed == 0
        assert result.orders_updated == 0

    @pytest.mark.asyncio
    async def test_order_on_venue_not_local(
        self, mock_execution_client, mock_cache, mock_risk_engine
    ):
        """Test order exists on venue but not locally."""
        venue_order = make_order_result("order-1")

        mock_execution_client.get_open_orders.return_value = [venue_order]
        mock_cache.open_orders.return_value = []

        reconciler = OrderReconciler(
            mock_execution_client, mock_cache, mock_risk_engine
        )
        result = await reconciler.reconcile()

        assert result.success is True
        assert result.orders_added == 1
        assert len(result.discrepancies) == 1
        assert result.discrepancies[0].action == ReconciliationAction.ADD_LOCAL
        assert result.discrepancies[0].entity_type == "order"

        # Verify cache was updated
        mock_cache.add_order.assert_called_once_with(venue_order)

        # Verify risk engine was updated
        mock_risk_engine.add_open_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_order_local_not_on_venue(
        self, mock_execution_client, mock_cache, mock_risk_engine
    ):
        """Test order exists locally but not on venue (filled/cancelled)."""
        local_order = make_order_result("order-1")

        mock_execution_client.get_open_orders.return_value = []
        mock_cache.open_orders.return_value = [local_order]

        reconciler = OrderReconciler(
            mock_execution_client, mock_cache, mock_risk_engine
        )
        result = await reconciler.reconcile()

        assert result.success is True
        assert result.orders_removed == 1
        assert len(result.discrepancies) >= 1
        assert result.discrepancies[0].action == ReconciliationAction.REMOVE_LOCAL

        # Verify risk engine was updated
        mock_risk_engine.remove_open_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_order_status_mismatch(self, mock_execution_client, mock_cache):
        """Test order exists both places with different status."""
        venue_order = make_order_result(
            "order-1", status=OrderStatus.PARTIALLY_FILLED, filled_amount=Decimal("0.5")
        )
        local_order = make_order_result("order-1", status=OrderStatus.OPEN)

        mock_execution_client.get_open_orders.return_value = [venue_order]
        mock_cache.open_orders.return_value = [local_order]

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        result = await reconciler.reconcile()

        assert result.success is True
        assert result.orders_updated == 1
        assert result.discrepancies[0].action == ReconciliationAction.UPDATE_LOCAL

        # Verify cache was updated with venue order
        mock_cache.add_order.assert_called_once_with(venue_order)

    @pytest.mark.asyncio
    async def test_order_filled_amount_mismatch(
        self, mock_execution_client, mock_cache
    ):
        """Test order exists both places with different filled amounts."""
        venue_order = make_order_result(
            "order-1", filled_amount=Decimal("0.7")
        )
        local_order = make_order_result(
            "order-1", filled_amount=Decimal("0.3")
        )

        mock_execution_client.get_open_orders.return_value = [venue_order]
        mock_cache.open_orders.return_value = [local_order]

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        result = await reconciler.reconcile()

        assert result.success is True
        assert result.orders_updated == 1
        assert "filled" in result.discrepancies[0].details.lower()

    @pytest.mark.asyncio
    async def test_multiple_orders(self, mock_execution_client, mock_cache):
        """Test reconciling multiple orders."""
        venue_orders = [
            make_order_result("order-1"),  # exists both
            make_order_result("order-2"),  # only on venue
        ]
        local_orders = [
            make_order_result("order-1"),  # exists both
            make_order_result("order-3"),  # only local
        ]

        mock_execution_client.get_open_orders.return_value = venue_orders
        mock_cache.open_orders.return_value = local_orders

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        result = await reconciler.reconcile()

        assert result.success is True
        assert result.orders_added == 1  # order-2
        assert result.orders_removed == 1  # order-3

    @pytest.mark.asyncio
    async def test_skip_order_reconciliation(self, mock_execution_client, mock_cache):
        """Test skipping order reconciliation."""
        mock_execution_client.get_open_orders.return_value = [make_order_result("1")]
        mock_cache.open_orders.return_value = []

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        result = await reconciler.reconcile(reconcile_orders=False)

        # Should not have any order discrepancies
        assert result.orders_added == 0
        mock_execution_client.get_open_orders.assert_not_called()


# =============================================================================
# OrderReconciler Tests - Position Reconciliation
# =============================================================================


class TestOrderReconcilerPositions:
    """Tests for OrderReconciler position reconciliation."""

    @pytest.mark.asyncio
    async def test_no_position_discrepancies(self, mock_execution_client, mock_cache):
        """Test when positions are in sync."""
        position = make_position("BTC/USDT")

        mock_execution_client.get_positions.return_value = [position]
        mock_cache.positions.return_value = [position]

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        result = await reconciler.reconcile()

        assert result.success is True
        assert result.positions_added == 0
        assert result.positions_removed == 0
        assert result.positions_adjusted == 0

    @pytest.mark.asyncio
    async def test_position_on_venue_not_local(
        self, mock_execution_client, mock_cache
    ):
        """Test position exists on venue but not locally."""
        venue_position = make_position("BTC/USDT", amount=Decimal("2.0"))

        mock_execution_client.get_positions.return_value = [venue_position]
        mock_cache.positions.return_value = []

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        result = await reconciler.reconcile()

        assert result.success is True
        assert result.positions_added == 1
        assert result.discrepancies[-1].action == ReconciliationAction.ADD_LOCAL
        assert result.discrepancies[-1].entity_type == "position"

        mock_cache.update_position.assert_called_once_with(venue_position)

    @pytest.mark.asyncio
    async def test_position_local_not_on_venue(
        self, mock_execution_client, mock_cache
    ):
        """Test position exists locally but closed on venue."""
        local_position = make_position("BTC/USDT")

        mock_execution_client.get_positions.return_value = []
        mock_cache.positions.return_value = [local_position]

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        result = await reconciler.reconcile()

        assert result.success is True
        assert result.positions_removed == 1

        # Should update with zero position
        mock_cache.update_position.assert_called_once()
        updated_position = mock_cache.update_position.call_args[0][0]
        assert updated_position.amount == Decimal("0")

    @pytest.mark.asyncio
    async def test_position_amount_mismatch(self, mock_execution_client, mock_cache):
        """Test position exists both places with different amounts."""
        venue_position = make_position("BTC/USDT", amount=Decimal("2.0"))
        local_position = make_position("BTC/USDT", amount=Decimal("1.0"))

        mock_execution_client.get_positions.return_value = [venue_position]
        mock_cache.positions.return_value = [local_position]

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        result = await reconciler.reconcile()

        assert result.success is True
        assert result.positions_adjusted == 1

        mock_cache.update_position.assert_called_once_with(venue_position)

    @pytest.mark.asyncio
    async def test_position_side_mismatch(self, mock_execution_client, mock_cache):
        """Test position exists both places with different sides."""
        venue_position = make_position(
            "BTC/USDT", side=PositionSide.LONG, amount=Decimal("1.0")
        )
        local_position = make_position(
            "BTC/USDT", side=PositionSide.SHORT, amount=Decimal("1.0")
        )

        mock_execution_client.get_positions.return_value = [venue_position]
        mock_cache.positions.return_value = [local_position]

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        result = await reconciler.reconcile()

        assert result.success is True
        assert result.positions_adjusted == 1
        assert "side" in result.discrepancies[-1].details.lower()

    @pytest.mark.asyncio
    async def test_skip_position_reconciliation(
        self, mock_execution_client, mock_cache
    ):
        """Test skipping position reconciliation."""
        mock_execution_client.get_positions.return_value = [make_position()]
        mock_cache.positions.return_value = []

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        result = await reconciler.reconcile(reconcile_positions=False)

        assert result.positions_added == 0
        mock_execution_client.get_positions.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_amount_positions_ignored(
        self, mock_execution_client, mock_cache
    ):
        """Test that zero-amount positions on venue are ignored."""
        zero_position = make_position("BTC/USDT", amount=Decimal("0"))

        mock_execution_client.get_positions.return_value = [zero_position]
        mock_cache.positions.return_value = []

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        result = await reconciler.reconcile()

        assert result.success is True
        assert result.positions_added == 0


# =============================================================================
# OrderReconciler Tests - Error Handling
# =============================================================================


class TestOrderReconcilerErrors:
    """Tests for OrderReconciler error handling."""

    @pytest.mark.asyncio
    async def test_get_open_orders_error(self, mock_execution_client, mock_cache):
        """Test handling error from get_open_orders."""
        mock_execution_client.get_open_orders.side_effect = Exception("API error")

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        result = await reconciler.reconcile()

        assert result.success is False
        assert len(result.errors) > 0
        assert "API error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_get_positions_error(self, mock_execution_client, mock_cache):
        """Test handling error from get_positions."""
        mock_execution_client.get_open_orders.return_value = []
        mock_execution_client.get_positions.side_effect = Exception("Position API error")

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        result = await reconciler.reconcile()

        assert result.success is False
        assert len(result.errors) > 0


# =============================================================================
# OrderReconciler Tests - Single Item Reconciliation
# =============================================================================


class TestOrderReconcilerSingle:
    """Tests for single item reconciliation methods."""

    @pytest.mark.asyncio
    async def test_reconcile_single_order_missing_locally(
        self, mock_execution_client, mock_cache, mock_risk_engine
    ):
        """Test reconciling a single order that's missing locally."""
        venue_order = make_order_result("order-1")
        mock_execution_client.get_order.return_value = venue_order
        mock_cache.order_by_exchange_id.return_value = None

        reconciler = OrderReconciler(
            mock_execution_client, mock_cache, mock_risk_engine
        )
        discrepancy = await reconciler.reconcile_single_order("order-1", "BTC/USDT")

        assert discrepancy is not None
        assert discrepancy.action == ReconciliationAction.ADD_LOCAL
        mock_cache.add_order.assert_called_once_with(venue_order)

    @pytest.mark.asyncio
    async def test_reconcile_single_order_status_changed(
        self, mock_execution_client, mock_cache
    ):
        """Test reconciling a single order with changed status."""
        venue_order = make_order_result("order-1", status=OrderStatus.FILLED)
        local_order = make_order_result("order-1", status=OrderStatus.OPEN)

        mock_execution_client.get_order.return_value = venue_order
        mock_cache.order_by_exchange_id.return_value = local_order

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        discrepancy = await reconciler.reconcile_single_order("order-1", "BTC/USDT")

        assert discrepancy is not None
        assert discrepancy.action == ReconciliationAction.UPDATE_LOCAL

    @pytest.mark.asyncio
    async def test_reconcile_single_order_no_change(
        self, mock_execution_client, mock_cache
    ):
        """Test reconciling a single order with no changes needed."""
        order = make_order_result("order-1")

        mock_execution_client.get_order.return_value = order
        mock_cache.order_by_exchange_id.return_value = order

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        discrepancy = await reconciler.reconcile_single_order("order-1", "BTC/USDT")

        assert discrepancy is None

    @pytest.mark.asyncio
    async def test_reconcile_single_order_fetch_error(
        self, mock_execution_client, mock_cache
    ):
        """Test handling fetch error for single order reconciliation."""
        mock_execution_client.get_order.side_effect = Exception("Not found")

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        discrepancy = await reconciler.reconcile_single_order("order-1", "BTC/USDT")

        assert discrepancy is None

    @pytest.mark.asyncio
    async def test_reconcile_single_position_missing_locally(
        self, mock_execution_client, mock_cache
    ):
        """Test reconciling a single position that's missing locally."""
        venue_position = make_position("BTC/USDT")
        mock_execution_client.get_position.return_value = venue_position
        mock_cache.position.return_value = None

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        discrepancy = await reconciler.reconcile_single_position("BTC/USDT")

        assert discrepancy is not None
        assert discrepancy.action == ReconciliationAction.ADD_LOCAL

    @pytest.mark.asyncio
    async def test_reconcile_single_position_closed_on_venue(
        self, mock_execution_client, mock_cache
    ):
        """Test reconciling when position closed on venue."""
        local_position = make_position("BTC/USDT")
        mock_execution_client.get_position.return_value = None
        mock_cache.position.return_value = local_position

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        discrepancy = await reconciler.reconcile_single_position("BTC/USDT")

        assert discrepancy is not None
        assert discrepancy.action == ReconciliationAction.REMOVE_LOCAL

    @pytest.mark.asyncio
    async def test_reconcile_single_position_amount_changed(
        self, mock_execution_client, mock_cache
    ):
        """Test reconciling when position amount changed."""
        venue_position = make_position("BTC/USDT", amount=Decimal("2.0"))
        local_position = make_position("BTC/USDT", amount=Decimal("1.0"))

        mock_execution_client.get_position.return_value = venue_position
        mock_cache.position.return_value = local_position

        reconciler = OrderReconciler(mock_execution_client, mock_cache)
        discrepancy = await reconciler.reconcile_single_position("BTC/USDT")

        assert discrepancy is not None
        assert discrepancy.action == ReconciliationAction.UPDATE_LOCAL


# =============================================================================
# Discrepancy Tests
# =============================================================================


class TestDiscrepancy:
    """Tests for Discrepancy dataclass."""

    def test_create_discrepancy(self):
        """Test creating a discrepancy."""
        d = Discrepancy(
            action=ReconciliationAction.ADD_LOCAL,
            entity_type="order",
            identifier="order-123",
            details="Added missing order",
            venue_state="OPEN",
            local_state=None,
        )

        assert d.action == ReconciliationAction.ADD_LOCAL
        assert d.entity_type == "order"
        assert d.identifier == "order-123"
        assert d.details == "Added missing order"
        assert d.venue_state == "OPEN"
        assert d.local_state is None

    def test_discrepancy_defaults(self):
        """Test discrepancy default values."""
        d = Discrepancy(
            action=ReconciliationAction.NONE,
            entity_type="position",
            identifier="BTC/USDT",
            details="No change",
        )

        assert d.venue_state is None
        assert d.local_state is None


# =============================================================================
# ReconciliationAction Tests
# =============================================================================


class TestReconciliationAction:
    """Tests for ReconciliationAction enum."""

    def test_action_values(self):
        """Test action enum values."""
        assert ReconciliationAction.NONE.value == "none"
        assert ReconciliationAction.ADD_LOCAL.value == "add_local"
        assert ReconciliationAction.REMOVE_LOCAL.value == "remove_local"
        assert ReconciliationAction.UPDATE_LOCAL.value == "update_local"
