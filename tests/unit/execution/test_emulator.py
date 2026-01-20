"""
Tests for OrderEmulator (Issue #106).

Tests synthetic order types: stop-loss, trailing stops, bracket orders, OCO, OTO.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from libra.execution.emulator import (
    BracketOrder,
    EmulatedOrder,
    EmulatedOrderState,
    EmulatorConfig,
    OrderEmulator,
)
from libra.gateways.protocol import (
    ContingencyType,
    GatewayCapabilities,
    Order,
    OrderSide,
    OrderType,
    TriggerType,
)


@pytest.fixture
def mock_message_bus() -> AsyncMock:
    """Create mock message bus."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def mock_clock() -> MagicMock:
    """Create mock clock."""
    clock = MagicMock()
    clock.timestamp_ns.return_value = 1704067200_000_000_000  # 2024-01-01
    return clock


@pytest.fixture
def emulator(
    mock_message_bus: AsyncMock, mock_clock: MagicMock, limited_capabilities: GatewayCapabilities
) -> OrderEmulator:
    """Create order emulator for testing with limited capabilities."""
    return OrderEmulator(
        message_bus=mock_message_bus,
        clock=mock_clock,
        config=EmulatorConfig(),
        capabilities=limited_capabilities,
    )


@pytest.fixture
def limited_capabilities() -> GatewayCapabilities:
    """Gateway with limited order type support."""
    return GatewayCapabilities(
        market_orders=True,
        limit_orders=True,
        stop_orders=False,  # No native stop support
        stop_limit_orders=False,
        trailing_stop_orders=False,
    )


class TestOrderEmulatorBasic:
    """Basic tests for OrderEmulator."""

    async def test_emulator_creation(
        self, mock_message_bus: AsyncMock, mock_clock: MagicMock
    ) -> None:
        """Test emulator can be created."""
        emulator = OrderEmulator(mock_message_bus, mock_clock)
        assert emulator is not None
        assert emulator.stats.orders_emulated == 0

    async def test_pass_through_market_order(self, emulator: OrderEmulator) -> None:
        """Test market orders pass through without emulation."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )

        result_order, emulated = await emulator.submit_order(order)

        assert result_order == order
        assert emulated is None
        assert emulator.stats.orders_emulated == 0

    async def test_pass_through_limit_order(self, emulator: OrderEmulator) -> None:
        """Test limit orders pass through without emulation."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
        )

        result_order, emulated = await emulator.submit_order(order)

        assert result_order == order
        assert emulated is None


class TestStopOrderEmulation:
    """Tests for stop order emulation."""

    async def test_stop_order_emulated_without_capability(
        self, mock_message_bus: AsyncMock, mock_clock: MagicMock, limited_capabilities: GatewayCapabilities
    ) -> None:
        """Test stop orders are emulated when venue doesn't support them."""
        emulator = OrderEmulator(
            mock_message_bus, mock_clock, capabilities=limited_capabilities
        )

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("48000"),
        )

        result_order, emulated = await emulator.submit_order(order)

        assert result_order is None  # Not passed through
        assert emulated is not None
        assert emulated.state == EmulatedOrderState.PENDING
        assert emulated.trigger_price == Decimal("48000")
        assert emulator.stats.orders_emulated == 1

    async def test_stop_order_triggers_on_price_drop(
        self, emulator: OrderEmulator
    ) -> None:
        """Test sell stop triggers when price drops to stop level."""
        # Submit stop order
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("48000"),
            trailing_offset=Decimal("100"),  # Force emulation
        )

        await emulator.submit_order(order)

        # Price above stop - no trigger
        triggered = await emulator.on_tick("BTC/USDT", last_price=Decimal("50000"))
        assert len(triggered) == 0

        # Price at stop level - trigger
        triggered = await emulator.on_tick("BTC/USDT", last_price=Decimal("48000"))
        assert len(triggered) == 1
        assert triggered[0].order_type == OrderType.MARKET  # Converted to market
        assert emulator.stats.orders_triggered == 1

    async def test_buy_stop_triggers_on_price_rise(
        self, emulator: OrderEmulator
    ) -> None:
        """Test buy stop triggers when price rises to stop level."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("52000"),
            trailing_offset=Decimal("100"),
        )

        await emulator.submit_order(order)

        # Price below stop - no trigger
        triggered = await emulator.on_tick("BTC/USDT", last_price=Decimal("50000"))
        assert len(triggered) == 0

        # Price at stop level - trigger
        triggered = await emulator.on_tick("BTC/USDT", last_price=Decimal("52000"))
        assert len(triggered) == 1


class TestTrailingStopEmulation:
    """Tests for trailing stop emulation."""

    async def test_trailing_stop_always_emulated(self, emulator: OrderEmulator) -> None:
        """Test trailing stops are always emulated locally."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("48000"),
            trailing_offset=Decimal("1000"),  # $1000 trail
            trailing_offset_type="price",
        )

        result_order, emulated = await emulator.submit_order(order)

        assert result_order is None
        assert emulated is not None
        assert emulated.trailing_offset == Decimal("1000")
        assert emulator.stats.orders_emulated == 1

    async def test_trailing_stop_follows_price_up(self, emulator: OrderEmulator) -> None:
        """Test trailing sell stop follows price upward."""
        # Initial price
        await emulator.on_tick("BTC/USDT", last_price=Decimal("50000"))

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            trailing_offset=Decimal("1000"),
            trailing_offset_type="price",
        )

        _, emulated = await emulator.submit_order(order)
        initial_stop = emulated.current_stop_price

        # Price rises
        await emulator.on_tick("BTC/USDT", last_price=Decimal("52000"))

        # Stop should have moved up
        updated_emulated = emulator.get_order(emulated.order_id)
        assert updated_emulated is not None
        assert updated_emulated.current_stop_price > initial_stop
        assert updated_emulated.best_price == Decimal("52000")

    async def test_trailing_stop_does_not_follow_down(
        self, emulator: OrderEmulator
    ) -> None:
        """Test trailing sell stop doesn't follow price downward."""
        await emulator.on_tick("BTC/USDT", last_price=Decimal("50000"))

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            trailing_offset=Decimal("1000"),
            trailing_offset_type="price",
        )

        _, emulated = await emulator.submit_order(order)

        # Price rises first
        await emulator.on_tick("BTC/USDT", last_price=Decimal("52000"))
        high_stop = emulator.get_order(emulated.order_id).current_stop_price

        # Price drops - stop should stay
        await emulator.on_tick("BTC/USDT", last_price=Decimal("51000"))
        current_stop = emulator.get_order(emulated.order_id).current_stop_price

        assert current_stop == high_stop

    async def test_trailing_stop_percent_offset(self, emulator: OrderEmulator) -> None:
        """Test trailing stop with percentage offset."""
        await emulator.on_tick("BTC/USDT", last_price=Decimal("50000"))

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            trailing_offset=Decimal("5"),  # 5%
            trailing_offset_type="percent",
        )

        _, emulated = await emulator.submit_order(order)

        # Stop should be 5% below price
        expected_stop = Decimal("50000") * (1 - Decimal("0.05"))
        assert emulated.current_stop_price == expected_stop


class TestBracketOrders:
    """Tests for bracket order emulation."""

    async def test_bracket_order_creation(self, emulator: OrderEmulator) -> None:
        """Test creating a bracket order (entry + SL + TP)."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            exec_algorithm_params={
                "stop_loss": "48000",
                "take_profit": "55000",
            },
        )

        entry_order, emulated = await emulator.submit_order(order)

        # Entry order should be returned for submission
        assert entry_order is not None
        assert entry_order.order_type == OrderType.LIMIT
        assert entry_order.contingency_type == ContingencyType.OTO

        # Stop loss should be emulated
        assert emulated is not None
        assert emulator.stats.bracket_orders == 1

        # Check bracket was created
        pending = emulator.get_pending_orders("BTC/USDT")
        assert len(pending) >= 2  # SL and TP

    async def test_bracket_sl_tp_opposite_side(self, emulator: OrderEmulator) -> None:
        """Test bracket SL and TP are opposite side to entry."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,  # Long entry
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            exec_algorithm_params={
                "stop_loss": "48000",
                "take_profit": "55000",
            },
        )

        await emulator.submit_order(order)

        pending = emulator.get_pending_orders("BTC/USDT")

        # Both SL and TP should be SELL (opposite to BUY entry)
        for emulated in pending:
            assert emulated.original_order.side == OrderSide.SELL


class TestOCOOrders:
    """Tests for OCO (One-Cancels-Other) orders."""

    async def test_oco_order_creation(self, emulator: OrderEmulator) -> None:
        """Test creating OCO order group."""
        order1 = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("48000"),
            trailing_offset=Decimal("100"),
        )

        order2 = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("55000"),
            trailing_offset=Decimal("100"),  # Force emulation
        )

        emulated_orders = await emulator.submit_oco_orders([order1, order2])

        assert len(emulated_orders) == 2
        assert emulator.stats.oco_orders == 2

        # Check they're linked
        assert emulated_orders[0].linked_order_ids == [emulated_orders[1].order_id]
        assert emulated_orders[1].linked_order_ids == [emulated_orders[0].order_id]

    async def test_oco_cancel_on_fill(self, emulator: OrderEmulator) -> None:
        """Test OCO sibling is cancelled when one fills."""
        order1 = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("48000"),
            trailing_offset=Decimal("100"),
        )

        order2 = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("55000"),
            trailing_offset=Decimal("100"),
        )

        emulated_orders = await emulator.submit_oco_orders([order1, order2])

        # Fill first order
        await emulator.on_order_filled(emulated_orders[0].order_id, Decimal("48000"))

        # Second should be cancelled
        order2_state = emulator.get_order(emulated_orders[1].order_id)
        assert order2_state.state == EmulatedOrderState.CANCELLED

    async def test_oco_requires_two_orders(self, emulator: OrderEmulator) -> None:
        """Test OCO requires at least 2 orders."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("48000"),
        )

        with pytest.raises(ValueError, match="at least 2 orders"):
            await emulator.submit_oco_orders([order])


class TestOTOOrders:
    """Tests for OTO (One-Triggers-Other) orders."""

    async def test_oto_chain_creation(self, emulator: OrderEmulator) -> None:
        """Test creating OTO order chain."""
        parent = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
        )

        child1 = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("48000"),
            trailing_offset=Decimal("100"),
        )

        child2 = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("55000"),
            trailing_offset=Decimal("100"),
        )

        parent_order, child_emulated = await emulator.submit_oto_chain(
            parent, [child1, child2]
        )

        assert parent_order.contingency_type == ContingencyType.OTO
        assert len(child_emulated) == 2
        assert emulator.stats.oto_orders == 2

        # Children should be marked as such
        for child in child_emulated:
            assert child.is_child
            assert child.parent_order_id is not None

    async def test_oto_children_wait_for_parent(self, emulator: OrderEmulator) -> None:
        """Test OTO children don't trigger until parent fills."""
        parent = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
        )

        child = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("48000"),
        )

        parent_order, children = await emulator.submit_oto_chain(parent, [child])
        parent_id = parent_order.client_order_id

        # Price hits child trigger but parent not filled
        triggered = await emulator.on_tick("BTC/USDT", last_price=Decimal("47000"))
        assert len(triggered) == 0

        # Now fill parent
        await emulator.on_order_filled(parent_id, Decimal("50000"))

        # Now child should trigger
        triggered = await emulator.on_tick("BTC/USDT", last_price=Decimal("47000"))
        assert len(triggered) == 1


class TestTriggerTypes:
    """Tests for different trigger types."""

    async def test_trigger_on_bid_ask(self, emulator: OrderEmulator) -> None:
        """Test triggering on bid/ask prices."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("48000"),
            trigger_type=TriggerType.BID_ASK,
        )

        await emulator.submit_order(order)

        # Last price hits but bid doesn't - no trigger for sell
        triggered = await emulator.on_tick(
            "BTC/USDT",
            last_price=Decimal("47000"),
            bid_price=Decimal("49000"),  # Bid still above trigger
        )
        assert len(triggered) == 0

        # Bid hits trigger
        triggered = await emulator.on_tick(
            "BTC/USDT",
            last_price=Decimal("49000"),
            bid_price=Decimal("47000"),
        )
        assert len(triggered) == 1

    async def test_trigger_on_mid_point(self, emulator: OrderEmulator) -> None:
        """Test triggering on mid-point price."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("48000"),
            trigger_type=TriggerType.MID_POINT,
        )

        await emulator.submit_order(order)

        # Mid = (49000 + 47000) / 2 = 48000
        triggered = await emulator.on_tick(
            "BTC/USDT",
            bid_price=Decimal("47000"),
            ask_price=Decimal("49000"),
        )
        assert len(triggered) == 1


class TestOrderManagement:
    """Tests for order management operations."""

    async def test_cancel_emulated_order(self, emulator: OrderEmulator) -> None:
        """Test cancelling an emulated order."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("48000"),
        )

        _, emulated = await emulator.submit_order(order)

        result = await emulator.cancel_order(emulated.order_id)
        assert result is True

        cancelled = emulator.get_order(emulated.order_id)
        assert cancelled.state == EmulatedOrderState.CANCELLED
        assert emulator.stats.orders_cancelled == 1

    async def test_cancel_nonexistent_order(self, emulator: OrderEmulator) -> None:
        """Test cancelling order that doesn't exist."""
        result = await emulator.cancel_order("nonexistent")
        assert result is False

    async def test_cancel_all_for_symbol(self, emulator: OrderEmulator) -> None:
        """Test cancelling all orders for a symbol."""
        order1 = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("48000"),
            trailing_offset=Decimal("100"),
        )

        order2 = Order(
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("3000"),
            trailing_offset=Decimal("100"),
        )

        await emulator.submit_order(order1)
        await emulator.submit_order(order2)

        cancelled = await emulator.cancel_all(symbol="BTC/USDT")
        assert cancelled == 1

        # ETH order should remain
        pending = emulator.get_pending_orders()
        assert len(pending) == 1
        assert pending[0].original_order.symbol == "ETH/USDT"

    async def test_get_pending_orders(self, emulator: OrderEmulator) -> None:
        """Test getting pending orders."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("48000"),
            trailing_offset=Decimal("100"),
        )

        await emulator.submit_order(order)

        pending = emulator.get_pending_orders()
        assert len(pending) == 1

        pending_btc = emulator.get_pending_orders("BTC/USDT")
        assert len(pending_btc) == 1

        pending_eth = emulator.get_pending_orders("ETH/USDT")
        assert len(pending_eth) == 0


class TestEmulatorStats:
    """Tests for emulator statistics."""

    async def test_stats_tracking(self, emulator: OrderEmulator) -> None:
        """Test statistics are tracked correctly."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            stop_price=Decimal("48000"),
        )

        await emulator.submit_order(order)
        assert emulator.stats.orders_emulated == 1

        # Trigger (price falls below stop_price)
        await emulator.on_tick("BTC/USDT", last_price=Decimal("47000"))
        assert emulator.stats.orders_triggered == 1
