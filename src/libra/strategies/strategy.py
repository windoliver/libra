"""
Strategy: Trading strategy with order management capabilities.

Extends Actor with:
- Order submission and cancellation
- Position management
- Order and position event handlers

Design references:
- NautilusTrader: https://nautilustrader.io/docs/latest/concepts/strategies
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from libra.core.events import Event, EventType
from libra.gateways.protocol import (
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)
from libra.strategies.actor import BaseActor, ComponentState
from libra.strategies.protocol import Signal, SignalType


if TYPE_CHECKING:
    from libra.core.message_bus import MessageBus
    from libra.gateways.protocol import Gateway, Tick
    from libra.strategies.protocol import Bar


logger = logging.getLogger(__name__)


# =============================================================================
# Order Event Types (for internal use)
# =============================================================================


class OrderEvent:
    """Base class for order events."""

    def __init__(self, order_id: str, symbol: str, timestamp_ns: int) -> None:
        self.order_id = order_id
        self.symbol = symbol
        self.timestamp_ns = timestamp_ns


class OrderSubmittedEvent(OrderEvent):
    """Order has been submitted to the gateway."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        timestamp_ns: int,
        order: Order,
    ) -> None:
        super().__init__(order_id, symbol, timestamp_ns)
        self.order = order


class OrderAcceptedEvent(OrderEvent):
    """Order has been accepted by the exchange."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        timestamp_ns: int,
        result: OrderResult,
    ) -> None:
        super().__init__(order_id, symbol, timestamp_ns)
        self.result = result


class OrderRejectedEvent(OrderEvent):
    """Order has been rejected by the exchange."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        timestamp_ns: int,
        reason: str,
    ) -> None:
        super().__init__(order_id, symbol, timestamp_ns)
        self.reason = reason


class OrderFilledEvent(OrderEvent):
    """Order has been filled (partially or completely)."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        timestamp_ns: int,
        result: OrderResult,
        fill_amount: Decimal,
        fill_price: Decimal,
    ) -> None:
        super().__init__(order_id, symbol, timestamp_ns)
        self.result = result
        self.fill_amount = fill_amount
        self.fill_price = fill_price


class OrderCanceledEvent(OrderEvent):
    """Order has been canceled."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        timestamp_ns: int,
    ) -> None:
        super().__init__(order_id, symbol, timestamp_ns)


# =============================================================================
# Position Event Types
# =============================================================================


class PositionEvent:
    """Base class for position events."""

    def __init__(self, symbol: str, timestamp_ns: int) -> None:
        self.symbol = symbol
        self.timestamp_ns = timestamp_ns


class PositionOpenedEvent(PositionEvent):
    """New position has been opened."""

    def __init__(
        self,
        symbol: str,
        timestamp_ns: int,
        position: Position,
    ) -> None:
        super().__init__(symbol, timestamp_ns)
        self.position = position


class PositionChangedEvent(PositionEvent):
    """Position has been modified."""

    def __init__(
        self,
        symbol: str,
        timestamp_ns: int,
        position: Position,
        previous_amount: Decimal,
    ) -> None:
        super().__init__(symbol, timestamp_ns)
        self.position = position
        self.previous_amount = previous_amount


class PositionClosedEvent(PositionEvent):
    """Position has been closed."""

    def __init__(
        self,
        symbol: str,
        timestamp_ns: int,
        realized_pnl: Decimal,
    ) -> None:
        super().__init__(symbol, timestamp_ns)
        self.realized_pnl = realized_pnl


# =============================================================================
# Strategy Protocol
# =============================================================================


@runtime_checkable
class Strategy(Protocol):
    """
    Trading strategy protocol with order management.

    Extends Actor with order submission, position management,
    and order/position event handlers.

    Lifecycle (inherited from Actor):
        1. initialize(): Wire to system (→ READY)
        2. start(): Begin operation (→ RUNNING)
        3. [running: generate signals, manage orders]
        4. stop(): Graceful shutdown (→ STOPPED)
        5. dispose(): Release resources (→ DISPOSED)

    Examples:
        class MomentumStrategy(BaseStrategy):
            async def on_bar(self, bar: Bar) -> None:
                if self.should_buy(bar):
                    await self.buy_market(bar.symbol, Decimal("0.1"))

            async def on_order_filled(self, event: OrderFilledEvent) -> None:
                self.log.info(f"Order filled: {event.fill_amount} @ {event.fill_price}")

            async def on_position_opened(self, event: PositionOpenedEvent) -> None:
                self.log.info(f"Position opened: {event.position}")
    """

    # -------------------------------------------------------------------------
    # Order Management
    # -------------------------------------------------------------------------

    async def submit_order(self, order: Order) -> OrderResult:
        """
        Submit an order through the gateway.

        Args:
            order: Order to submit

        Returns:
            OrderResult with status and fill info
        """
        ...

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair

        Returns:
            True if canceled, False if already closed
        """
        ...

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            Number of orders canceled
        """
        ...

    async def close_position(self, symbol: str) -> OrderResult | None:
        """
        Close position for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            OrderResult if position existed, None otherwise
        """
        ...

    async def close_all_positions(self) -> list[OrderResult]:
        """
        Close all open positions.

        Returns:
            List of OrderResults for closed positions
        """
        ...

    # -------------------------------------------------------------------------
    # Convenience Order Methods
    # -------------------------------------------------------------------------

    async def buy_market(
        self,
        symbol: str,
        amount: Decimal,
    ) -> OrderResult:
        """Submit a market buy order."""
        ...

    async def sell_market(
        self,
        symbol: str,
        amount: Decimal,
    ) -> OrderResult:
        """Submit a market sell order."""
        ...

    async def buy_limit(
        self,
        symbol: str,
        amount: Decimal,
        price: Decimal,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> OrderResult:
        """Submit a limit buy order."""
        ...

    async def sell_limit(
        self,
        symbol: str,
        amount: Decimal,
        price: Decimal,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> OrderResult:
        """Submit a limit sell order."""
        ...

    # -------------------------------------------------------------------------
    # Position Queries
    # -------------------------------------------------------------------------

    async def get_position(self, symbol: str) -> Position | None:
        """Get current position for a symbol."""
        ...

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        ...

    def is_long(self, symbol: str) -> bool:
        """Check if currently long on symbol."""
        ...

    def is_short(self, symbol: str) -> bool:
        """Check if currently short on symbol."""
        ...

    def is_flat(self, symbol: str) -> bool:
        """Check if no position on symbol."""
        ...

    # -------------------------------------------------------------------------
    # Order Event Handlers
    # -------------------------------------------------------------------------

    async def on_order_submitted(self, event: OrderSubmittedEvent) -> None:
        """Called when order is submitted."""
        ...

    async def on_order_accepted(self, event: OrderAcceptedEvent) -> None:
        """Called when order is accepted by exchange."""
        ...

    async def on_order_rejected(self, event: OrderRejectedEvent) -> None:
        """Called when order is rejected."""
        ...

    async def on_order_filled(self, event: OrderFilledEvent) -> None:
        """Called when order is filled (partial or complete)."""
        ...

    async def on_order_canceled(self, event: OrderCanceledEvent) -> None:
        """Called when order is canceled."""
        ...

    # -------------------------------------------------------------------------
    # Position Event Handlers
    # -------------------------------------------------------------------------

    async def on_position_opened(self, event: PositionOpenedEvent) -> None:
        """Called when a new position is opened."""
        ...

    async def on_position_changed(self, event: PositionChangedEvent) -> None:
        """Called when position size changes."""
        ...

    async def on_position_closed(self, event: PositionClosedEvent) -> None:
        """Called when position is closed."""
        ...


# =============================================================================
# Base Strategy Implementation
# =============================================================================


class BaseStrategy(BaseActor):
    """
    Abstract base class for trading strategies.

    Extends BaseActor with:
    - Gateway integration for order execution
    - Position tracking
    - Order and position event handling
    - Convenience methods for common operations

    Subclasses must implement:
    - name property
    - on_bar() or on_tick() for signal generation

    Example:
        class RSIStrategy(BaseStrategy):
            def __init__(self, gateway: Gateway, config: dict):
                super().__init__(gateway)
                self._rsi_period = config.get("rsi_period", 14)
                self._prices: deque[Decimal] = deque(maxlen=self._rsi_period + 1)

            @property
            def name(self) -> str:
                return "rsi_strategy"

            async def on_bar(self, bar: Bar) -> None:
                self._prices.append(bar.close)
                if len(self._prices) < self._rsi_period + 1:
                    return

                rsi = self._calculate_rsi()
                if rsi < 30 and self.is_flat(bar.symbol):
                    await self.buy_market(bar.symbol, Decimal("0.1"))
                elif rsi > 70 and self.is_long(bar.symbol):
                    await self.close_position(bar.symbol)
    """

    def __init__(self, gateway: Gateway) -> None:
        """
        Initialize strategy with gateway.

        Args:
            gateway: Gateway for order execution
        """
        super().__init__()
        self._gateway = gateway
        self._positions: dict[str, Position] = {}
        self._open_orders: dict[str, OrderResult] = {}
        self._signal_count = 0
        self._order_count = 0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def gateway(self) -> Gateway:
        """Gateway for order execution."""
        return self._gateway

    @property
    def signal_count(self) -> int:
        """Number of signals generated."""
        return self._signal_count

    @property
    def order_count(self) -> int:
        """Number of orders submitted."""
        return self._order_count

    # -------------------------------------------------------------------------
    # Order Submission
    # -------------------------------------------------------------------------

    async def submit_order(self, order: Order) -> OrderResult:
        """Submit an order through the gateway."""
        # Allow orders in RUNNING, DEGRADED, and STOPPING (for graceful cleanup)
        if self.state not in (ComponentState.RUNNING, ComponentState.DEGRADED, ComponentState.STOPPING):
            raise RuntimeError(
                f"Cannot submit order in state {self.state.name}. "
                "Strategy must be RUNNING, DEGRADED, or STOPPING."
            )

        self._order_count += 1
        self.log.info(
            "Submitting order: %s %s %s @ %s",
            order.side.value,
            order.amount,
            order.symbol,
            order.price or "MARKET",
        )

        # Publish submitted event
        import time

        submitted_event = OrderSubmittedEvent(
            order_id=order.client_order_id or str(self._order_count),
            symbol=order.symbol,
            timestamp_ns=time.time_ns(),
            order=order,
        )
        await self.on_order_submitted(submitted_event)

        # Submit through gateway
        try:
            result = await self._gateway.submit_order(order)

            # Track open orders
            if result.is_open:
                self._open_orders[result.order_id] = result

            # Publish accepted event
            accepted_event = OrderAcceptedEvent(
                order_id=result.order_id,
                symbol=result.symbol,
                timestamp_ns=time.time_ns(),
                result=result,
            )
            await self.on_order_accepted(accepted_event)

            # Handle fills
            if result.filled_amount > 0:
                filled_event = OrderFilledEvent(
                    order_id=result.order_id,
                    symbol=result.symbol,
                    timestamp_ns=time.time_ns(),
                    result=result,
                    fill_amount=result.filled_amount,
                    fill_price=result.average_price or Decimal("0"),
                )
                await self.on_order_filled(filled_event)
                await self._update_position(result)

            # Publish bus event
            self.publish_event(
                EventType.ORDER_NEW,
                {"order_id": result.order_id, "symbol": result.symbol, "status": result.status.value},
            )

            return result

        except Exception as e:
            self.log.error("Order rejected: %s", e)
            rejected_event = OrderRejectedEvent(
                order_id=order.client_order_id or str(self._order_count),
                symbol=order.symbol,
                timestamp_ns=time.time_ns(),
                reason=str(e),
            )
            await self.on_order_rejected(rejected_event)
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        self.log.info("Canceling order: %s", order_id)

        result = await self._gateway.cancel_order(order_id, symbol)

        if result:
            self._open_orders.pop(order_id, None)
            import time

            canceled_event = OrderCanceledEvent(
                order_id=order_id,
                symbol=symbol,
                timestamp_ns=time.time_ns(),
            )
            await self.on_order_canceled(canceled_event)

            self.publish_event(
                EventType.ORDER_CANCELLED,
                {"order_id": order_id, "symbol": symbol},
            )

        return result

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all open orders."""
        self.log.info("Canceling all orders for %s", symbol or "all symbols")
        return await self._gateway.cancel_all_orders(symbol)

    # -------------------------------------------------------------------------
    # Position Management
    # -------------------------------------------------------------------------

    async def close_position(self, symbol: str) -> OrderResult | None:
        """Close position for a symbol."""
        position = await self.get_position(symbol)
        if position is None or position.amount == Decimal("0"):
            return None

        self.log.info(
            "Closing position: %s %s %s",
            position.side.value,
            position.amount,
            symbol,
        )

        # Determine order side (opposite of position)
        from libra.gateways.protocol import PositionSide

        if position.side == PositionSide.LONG:
            result = await self.sell_market(symbol, position.amount)
        else:
            result = await self.buy_market(symbol, position.amount)

        return result

    async def close_all_positions(self) -> list[OrderResult]:
        """Close all open positions."""
        self.log.info("Closing all positions")
        positions = await self.get_positions()
        results: list[OrderResult] = []

        for position in positions:
            if position.amount > 0:
                result = await self.close_position(position.symbol)
                if result:
                    results.append(result)

        return results

    async def get_position(self, symbol: str) -> Position | None:
        """Get current position for a symbol."""
        # Check cached positions first
        if symbol in self._positions:
            return self._positions[symbol]
        # Fallback to gateway
        return await self._gateway.get_position(symbol)

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        return await self._gateway.get_positions()

    def is_long(self, symbol: str) -> bool:
        """Check if currently long on symbol."""
        from libra.gateways.protocol import PositionSide

        pos = self._positions.get(symbol)
        return pos is not None and pos.side == PositionSide.LONG and pos.amount > 0

    def is_short(self, symbol: str) -> bool:
        """Check if currently short on symbol."""
        from libra.gateways.protocol import PositionSide

        pos = self._positions.get(symbol)
        return pos is not None and pos.side == PositionSide.SHORT and pos.amount > 0

    def is_flat(self, symbol: str) -> bool:
        """Check if no position on symbol."""
        pos = self._positions.get(symbol)
        return pos is None or pos.amount == Decimal("0")

    async def _update_position(self, result: OrderResult) -> None:
        """Update cached position after order fill."""
        import time

        from libra.gateways.protocol import PositionSide

        symbol = result.symbol
        old_position = self._positions.get(symbol)

        # Get fresh position from gateway
        new_position = await self._gateway.get_position(symbol)

        if new_position is None:
            # Position closed
            if old_position is not None:
                self._positions.pop(symbol, None)
                closed_event = PositionClosedEvent(
                    symbol=symbol,
                    timestamp_ns=time.time_ns(),
                    realized_pnl=old_position.realized_pnl,
                )
                await self.on_position_closed(closed_event)
                self.publish_event(
                    EventType.POSITION_CLOSED,
                    {"symbol": symbol, "pnl": str(old_position.realized_pnl)},
                )
        elif old_position is None or old_position.amount == Decimal("0"):
            # New position opened
            self._positions[symbol] = new_position
            opened_event = PositionOpenedEvent(
                symbol=symbol,
                timestamp_ns=time.time_ns(),
                position=new_position,
            )
            await self.on_position_opened(opened_event)
            self.publish_event(
                EventType.POSITION_OPENED,
                {"symbol": symbol, "side": new_position.side.value, "amount": str(new_position.amount)},
            )
        else:
            # Position changed
            self._positions[symbol] = new_position
            changed_event = PositionChangedEvent(
                symbol=symbol,
                timestamp_ns=time.time_ns(),
                position=new_position,
                previous_amount=old_position.amount,
            )
            await self.on_position_changed(changed_event)
            self.publish_event(
                EventType.POSITION_UPDATED,
                {"symbol": symbol, "amount": str(new_position.amount)},
            )

    # -------------------------------------------------------------------------
    # Convenience Order Methods
    # -------------------------------------------------------------------------

    async def buy_market(self, symbol: str, amount: Decimal) -> OrderResult:
        """Submit a market buy order."""
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=amount,
        )
        return await self.submit_order(order)

    async def sell_market(self, symbol: str, amount: Decimal) -> OrderResult:
        """Submit a market sell order."""
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=amount,
        )
        return await self.submit_order(order)

    async def buy_limit(
        self,
        symbol: str,
        amount: Decimal,
        price: Decimal,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> OrderResult:
        """Submit a limit buy order."""
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=amount,
            price=price,
            time_in_force=time_in_force,
        )
        return await self.submit_order(order)

    async def sell_limit(
        self,
        symbol: str,
        amount: Decimal,
        price: Decimal,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> OrderResult:
        """Submit a limit sell order."""
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=amount,
            price=price,
            time_in_force=time_in_force,
        )
        return await self.submit_order(order)

    # -------------------------------------------------------------------------
    # Signal Helpers
    # -------------------------------------------------------------------------

    def create_signal(
        self,
        signal_type: SignalType,
        symbol: str,
        strength: float = 1.0,
        price: Decimal | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Signal:
        """Create a trading signal."""
        self._signal_count += 1
        return Signal.create(
            signal_type=signal_type,
            symbol=symbol,
            strength=strength,
            price=price,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    # Order Event Handlers (override in subclass)
    # -------------------------------------------------------------------------

    async def on_order_submitted(self, event: OrderSubmittedEvent) -> None:  # noqa: ARG002
        """Called when order is submitted. Override in subclass."""
        pass

    async def on_order_accepted(self, event: OrderAcceptedEvent) -> None:  # noqa: ARG002
        """Called when order is accepted. Override in subclass."""
        pass

    async def on_order_rejected(self, event: OrderRejectedEvent) -> None:  # noqa: ARG002
        """Called when order is rejected. Override in subclass."""
        pass

    async def on_order_filled(self, event: OrderFilledEvent) -> None:  # noqa: ARG002
        """Called when order is filled. Override in subclass."""
        pass

    async def on_order_canceled(self, event: OrderCanceledEvent) -> None:  # noqa: ARG002
        """Called when order is canceled. Override in subclass."""
        pass

    # -------------------------------------------------------------------------
    # Position Event Handlers (override in subclass)
    # -------------------------------------------------------------------------

    async def on_position_opened(self, event: PositionOpenedEvent) -> None:  # noqa: ARG002
        """Called when position is opened. Override in subclass."""
        pass

    async def on_position_changed(self, event: PositionChangedEvent) -> None:  # noqa: ARG002
        """Called when position changes. Override in subclass."""
        pass

    async def on_position_closed(self, event: PositionClosedEvent) -> None:  # noqa: ARG002
        """Called when position is closed. Override in subclass."""
        pass

    # -------------------------------------------------------------------------
    # Lifecycle Hooks
    # -------------------------------------------------------------------------

    async def on_stop(self) -> None:
        """Cancel orders and optionally close positions on stop."""
        await self.cancel_all_orders()
        await super().on_stop()

    async def on_reset(self) -> None:
        """Reset strategy state."""
        self._positions.clear()
        self._open_orders.clear()
        self._signal_count = 0
        self._order_count = 0
        await super().on_reset()
