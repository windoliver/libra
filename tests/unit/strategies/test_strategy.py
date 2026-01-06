"""Unit tests for Strategy with order management."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libra.core.message_bus import MessageBus
from libra.gateways.protocol import (
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    TimeInForce,
)
from libra.strategies.actor import ComponentState
from libra.strategies.strategy import (
    BaseStrategy,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    OrderSubmittedEvent,
    PositionChangedEvent,
    PositionClosedEvent,
    PositionOpenedEvent,
)


# =============================================================================
# Test Strategy Implementation
# =============================================================================


class SampleStrategy(BaseStrategy):
    """Concrete Strategy implementation for testing."""

    def __init__(self, gateway: Any) -> None:
        super().__init__(gateway)
        self._name = "test_strategy"
        self.order_submitted_events: list[OrderSubmittedEvent] = []
        self.order_accepted_events: list[OrderAcceptedEvent] = []
        self.order_rejected_events: list[OrderRejectedEvent] = []
        self.order_filled_events: list[OrderFilledEvent] = []
        self.order_canceled_events: list[OrderCanceledEvent] = []
        self.position_opened_events: list[PositionOpenedEvent] = []
        self.position_changed_events: list[PositionChangedEvent] = []
        self.position_closed_events: list[PositionClosedEvent] = []

    @property
    def name(self) -> str:
        return self._name

    async def on_order_submitted(self, event: OrderSubmittedEvent) -> None:
        self.order_submitted_events.append(event)

    async def on_order_accepted(self, event: OrderAcceptedEvent) -> None:
        self.order_accepted_events.append(event)

    async def on_order_rejected(self, event: OrderRejectedEvent) -> None:
        self.order_rejected_events.append(event)

    async def on_order_filled(self, event: OrderFilledEvent) -> None:
        self.order_filled_events.append(event)

    async def on_order_canceled(self, event: OrderCanceledEvent) -> None:
        self.order_canceled_events.append(event)

    async def on_position_opened(self, event: PositionOpenedEvent) -> None:
        self.position_opened_events.append(event)

    async def on_position_changed(self, event: PositionChangedEvent) -> None:
        self.position_changed_events.append(event)

    async def on_position_closed(self, event: PositionClosedEvent) -> None:
        self.position_closed_events.append(event)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_gateway() -> MagicMock:
    """Create a mock gateway."""
    gateway = MagicMock()
    gateway.name = "mock_gateway"
    gateway.submit_order = AsyncMock()
    gateway.cancel_order = AsyncMock()
    gateway.cancel_all_orders = AsyncMock()
    gateway.get_position = AsyncMock()
    gateway.get_positions = AsyncMock()
    return gateway


@pytest.fixture
def message_bus() -> MessageBus:
    """Create a MessageBus instance."""
    return MessageBus()


@pytest.fixture
def strategy(mock_gateway: MagicMock) -> SampleStrategy:
    """Create a test strategy."""
    return SampleStrategy(mock_gateway)


@pytest.fixture
async def running_strategy(
    strategy: SampleStrategy, message_bus: MessageBus
) -> SampleStrategy:
    """Create a running strategy."""
    await strategy.initialize(message_bus)
    await strategy.start()
    return strategy


# =============================================================================
# Strategy Initialization Tests
# =============================================================================


class SampleStrategyInitialization:
    """Tests for Strategy initialization."""

    def test_initial_state(self, strategy: SampleStrategy) -> None:
        """Test strategy initial state."""
        assert strategy.state == ComponentState.PRE_INITIALIZED
        assert strategy.signal_count == 0
        assert strategy.order_count == 0

    @pytest.mark.asyncio
    async def test_start(
        self, strategy: SampleStrategy, message_bus: MessageBus
    ) -> None:
        """Test strategy start."""
        await strategy.initialize(message_bus)
        await strategy.start()
        assert strategy.state == ComponentState.RUNNING


# =============================================================================
# Order Submission Tests
# =============================================================================


class TestOrderSubmission:
    """Tests for order submission."""

    @pytest.mark.asyncio
    async def test_submit_market_buy(
        self, running_strategy: SampleStrategy, mock_gateway: MagicMock
    ) -> None:
        """Test market buy order submission."""
        mock_gateway.submit_order.return_value = OrderResult(
            order_id="12345",
            symbol="BTC/USDT",
            status=OrderStatus.FILLED,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
            filled_amount=Decimal("0.1"),
            remaining_amount=Decimal("0"),
            average_price=Decimal("50000"),
            fee=Decimal("0.0001"),
            fee_currency="BTC",
            timestamp_ns=0,
        )
        mock_gateway.get_position.return_value = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
        )

        result = await running_strategy.buy_market("BTC/USDT", Decimal("0.1"))

        assert result.status == OrderStatus.FILLED
        assert result.filled_amount == Decimal("0.1")
        assert running_strategy.order_count == 1

        # Check events were triggered
        assert len(running_strategy.order_submitted_events) == 1
        assert len(running_strategy.order_accepted_events) == 1
        assert len(running_strategy.order_filled_events) == 1

    @pytest.mark.asyncio
    async def test_submit_limit_sell(
        self, running_strategy: SampleStrategy, mock_gateway: MagicMock
    ) -> None:
        """Test limit sell order submission."""
        mock_gateway.submit_order.return_value = OrderResult(
            order_id="12346",
            symbol="ETH/USDT",
            status=OrderStatus.OPEN,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            filled_amount=Decimal("0"),
            remaining_amount=Decimal("1.0"),
            average_price=None,
            fee=Decimal("0"),
            fee_currency="USDT",
            timestamp_ns=0,
            price=Decimal("2000"),
        )
        mock_gateway.get_position.return_value = None

        result = await running_strategy.sell_limit(
            "ETH/USDT", Decimal("1.0"), Decimal("2000")
        )

        assert result.status == OrderStatus.OPEN
        assert result.price == Decimal("2000")

    @pytest.mark.asyncio
    async def test_submit_order_not_running_fails(
        self, strategy: SampleStrategy, message_bus: MessageBus
    ) -> None:
        """Test that submitting order when not running fails."""
        await strategy.initialize(message_bus)
        # Don't start

        with pytest.raises(RuntimeError, match="must be RUNNING"):
            await strategy.buy_market("BTC/USDT", Decimal("0.1"))

    @pytest.mark.asyncio
    async def test_order_rejected(
        self, running_strategy: SampleStrategy, mock_gateway: MagicMock
    ) -> None:
        """Test order rejection handling."""
        mock_gateway.submit_order.side_effect = Exception("Insufficient funds")

        with pytest.raises(Exception, match="Insufficient funds"):
            await running_strategy.buy_market("BTC/USDT", Decimal("1000"))

        # Check rejection event was triggered
        assert len(running_strategy.order_rejected_events) == 1
        assert "Insufficient funds" in running_strategy.order_rejected_events[0].reason


# =============================================================================
# Order Cancellation Tests
# =============================================================================


class TestOrderCancellation:
    """Tests for order cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_order(
        self, running_strategy: SampleStrategy, mock_gateway: MagicMock
    ) -> None:
        """Test order cancellation."""
        mock_gateway.cancel_order.return_value = True

        result = await running_strategy.cancel_order("12345", "BTC/USDT")

        assert result is True
        assert len(running_strategy.order_canceled_events) == 1
        mock_gateway.cancel_order.assert_called_once_with("12345", "BTC/USDT")

    @pytest.mark.asyncio
    async def test_cancel_all_orders(
        self, running_strategy: SampleStrategy, mock_gateway: MagicMock
    ) -> None:
        """Test canceling all orders."""
        mock_gateway.cancel_all_orders.return_value = 3

        result = await running_strategy.cancel_all_orders("BTC/USDT")

        assert result == 3
        mock_gateway.cancel_all_orders.assert_called_once_with("BTC/USDT")


# =============================================================================
# Position Management Tests
# =============================================================================


class TestPositionManagement:
    """Tests for position management."""

    @pytest.mark.asyncio
    async def test_is_long(
        self, running_strategy: SampleStrategy, mock_gateway: MagicMock
    ) -> None:
        """Test is_long check."""
        # Set up position cache
        running_strategy._positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("0"),
        )

        assert running_strategy.is_long("BTC/USDT") is True
        assert running_strategy.is_short("BTC/USDT") is False
        assert running_strategy.is_flat("BTC/USDT") is False

    @pytest.mark.asyncio
    async def test_is_short(
        self, running_strategy: SampleStrategy, mock_gateway: MagicMock
    ) -> None:
        """Test is_short check."""
        running_strategy._positions["ETH/USDT"] = Position(
            symbol="ETH/USDT",
            side=PositionSide.SHORT,
            amount=Decimal("1.0"),
            entry_price=Decimal("2000"),
            current_price=Decimal("1900"),
            unrealized_pnl=Decimal("100"),
            realized_pnl=Decimal("0"),
        )

        assert running_strategy.is_short("ETH/USDT") is True
        assert running_strategy.is_long("ETH/USDT") is False
        assert running_strategy.is_flat("ETH/USDT") is False

    @pytest.mark.asyncio
    async def test_is_flat(self, running_strategy: SampleStrategy) -> None:
        """Test is_flat check."""
        assert running_strategy.is_flat("UNKNOWN/USDT") is True

    @pytest.mark.asyncio
    async def test_close_position(
        self, running_strategy: SampleStrategy, mock_gateway: MagicMock
    ) -> None:
        """Test closing a position."""
        # Set up long position
        running_strategy._positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("0"),
        )

        # Mock the sell to close
        mock_gateway.submit_order.return_value = OrderResult(
            order_id="12347",
            symbol="BTC/USDT",
            status=OrderStatus.FILLED,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=Decimal("0.5"),
            filled_amount=Decimal("0.5"),
            remaining_amount=Decimal("0"),
            average_price=Decimal("51000"),
            fee=Decimal("0.0005"),
            fee_currency="BTC",
            timestamp_ns=0,
        )
        mock_gateway.get_position.return_value = None

        result = await running_strategy.close_position("BTC/USDT")

        assert result is not None
        assert result.side == OrderSide.SELL
        assert result.filled_amount == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_close_position_no_position(
        self, running_strategy: SampleStrategy, mock_gateway: MagicMock
    ) -> None:
        """Test closing when no position exists."""
        mock_gateway.get_position.return_value = None

        result = await running_strategy.close_position("BTC/USDT")

        assert result is None


# =============================================================================
# Lifecycle Tests
# =============================================================================


class SampleStrategyLifecycle:
    """Tests for strategy lifecycle."""

    @pytest.mark.asyncio
    async def test_stop_cancels_orders(
        self, running_strategy: SampleStrategy, mock_gateway: MagicMock
    ) -> None:
        """Test that stopping cancels all orders."""
        mock_gateway.cancel_all_orders.return_value = 2

        await running_strategy.stop()

        mock_gateway.cancel_all_orders.assert_called_once()
        assert running_strategy.state == ComponentState.STOPPED

    @pytest.mark.asyncio
    async def test_reset_clears_state(
        self, running_strategy: SampleStrategy, mock_gateway: MagicMock
    ) -> None:
        """Test that reset clears strategy state."""
        # Add some state
        running_strategy._positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("0"),
        )
        running_strategy._signal_count = 10
        running_strategy._order_count = 5

        mock_gateway.cancel_all_orders.return_value = 0
        await running_strategy.stop()
        await running_strategy.reset()

        assert len(running_strategy._positions) == 0
        assert running_strategy.signal_count == 0
        assert running_strategy.order_count == 0


# =============================================================================
# Signal Creation Tests
# =============================================================================


class TestSignalCreation:
    """Tests for signal creation."""

    @pytest.mark.asyncio
    async def test_create_signal(self, running_strategy: SampleStrategy) -> None:
        """Test creating a signal."""
        from libra.strategies.protocol import SignalType

        signal = running_strategy.create_signal(
            SignalType.LONG,
            "BTC/USDT",
            strength=0.8,
            price=Decimal("50000"),
            metadata={"reason": "momentum"},
        )

        assert signal.signal_type == SignalType.LONG
        assert signal.symbol == "BTC/USDT"
        assert signal.strength == 0.8
        assert signal.price == Decimal("50000")
        assert signal.metadata["reason"] == "momentum"
        assert running_strategy.signal_count == 1
