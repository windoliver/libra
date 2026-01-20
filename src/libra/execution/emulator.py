"""
Order Emulator: Synthetic order types for venues without native support.

Implements Issue #106: OrderEmulator for synthetic order types.

Following NautilusTrader pattern, the OrderEmulator enables:
- Emulating order types not supported by venues (e.g., stop-loss)
- Synthetic order types (bracket orders, trailing stops)
- Local order management before sending to venue
- OCO/OTO contingency orders handled locally

Execution Flow:
    Strategy → OrderEmulator → ExecAlgorithm → RiskEngine → ExecutionEngine

Emulated Order Types:
1. Stop-Loss (for venues without native support)
2. Stop-Limit (for venues without native support)
3. Trailing Stop (always local - track price, adjust stop)
4. Bracket Order (entry + stop-loss + take-profit)
5. OCO (One-Cancels-Other)
6. OTO (One-Triggers-Other)

References:
- NautilusTrader OrderEmulator
- Interactive Brokers TWS API
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any

from libra.core.events import Event, EventType
from libra.gateways.protocol import (
    ContingencyType,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    TriggerType,
)


if TYPE_CHECKING:
    from libra.core.clock import Clock
    from libra.core.message_bus import MessageBus
    from libra.gateways.protocol import GatewayCapabilities


logger = logging.getLogger(__name__)


# =============================================================================
# Emulated Order State
# =============================================================================


class EmulatedOrderState(str, Enum):
    """State of an emulated order."""

    PENDING = "pending"  # Waiting for trigger condition
    TRIGGERED = "triggered"  # Trigger hit, order being submitted
    SUBMITTED = "submitted"  # Sent to venue
    FILLED = "filled"  # Completely filled
    CANCELLED = "cancelled"  # Cancelled by user or system
    REJECTED = "rejected"  # Rejected by venue
    EXPIRED = "expired"  # Time-in-force expired


@dataclass
class EmulatedOrder:
    """
    Tracks state of an emulated order.

    Contains original order and trigger conditions.
    """

    order_id: str
    original_order: Order
    state: EmulatedOrderState = EmulatedOrderState.PENDING
    trigger_price: Decimal | None = None
    trigger_type: TriggerType = TriggerType.LAST_PRICE
    created_at_ns: int = 0
    triggered_at_ns: int | None = None
    submitted_at_ns: int | None = None
    filled_at_ns: int | None = None

    # For trailing stops
    trailing_offset: Decimal | None = None
    trailing_offset_type: str | None = None  # "price" or "percent"
    best_price: Decimal | None = None  # Best price seen (for trailing)
    current_stop_price: Decimal | None = None  # Current adjusted stop

    # For linked orders (OCO, OTO, bracket)
    linked_order_ids: list[str] = field(default_factory=list)
    parent_order_id: str | None = None
    is_child: bool = False

    # Result tracking
    submitted_order: Order | None = None
    result: OrderResult | None = None


@dataclass
class BracketOrder:
    """
    Bracket order: Entry + Stop-Loss + Take-Profit.

    All three legs tracked together.
    """

    bracket_id: str
    entry_order_id: str
    stop_loss_order_id: str
    take_profit_order_id: str
    symbol: str
    side: OrderSide
    state: str = "pending"  # pending, active, closed
    entry_filled: bool = False


# =============================================================================
# Order Emulator
# =============================================================================


@dataclass
class EmulatorConfig:
    """Configuration for OrderEmulator."""

    # Tick processing
    tick_buffer_size: int = 1000  # Max ticks to buffer

    # Trailing stop
    trailing_update_threshold: float = 0.001  # 0.1% min price move to update

    # Timeout
    trigger_timeout_secs: float = 86400.0  # 24 hours default

    # Risk
    max_emulated_orders: int = 1000


@dataclass
class EmulatorStats:
    """Statistics for order emulator."""

    orders_emulated: int = 0
    orders_triggered: int = 0
    orders_submitted: int = 0
    orders_cancelled: int = 0
    trailing_updates: int = 0
    bracket_orders: int = 0
    oco_orders: int = 0
    oto_orders: int = 0


class OrderEmulator:
    """
    Emulates order types not natively supported by venues.

    Monitors price ticks and triggers emulated orders when
    conditions are met. Handles:
    - Stop orders (stop-loss, stop-limit)
    - Trailing stops
    - Bracket orders
    - OCO (One-Cancels-Other)
    - OTO (One-Triggers-Other)

    Example:
        emulator = OrderEmulator(message_bus, clock)

        # Submit a trailing stop
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            amount=Decimal("1.0"),
            trailing_offset=Decimal("100"),  # $100 trail
            trailing_offset_type="price",
        )
        await emulator.submit_order(order)

        # Process ticks to update trailing stops
        await emulator.on_tick(tick)
    """

    def __init__(
        self,
        message_bus: MessageBus,
        clock: Clock,
        config: EmulatorConfig | None = None,
        capabilities: GatewayCapabilities | None = None,
    ) -> None:
        """
        Initialize order emulator.

        Args:
            message_bus: For publishing events
            clock: For timestamps
            config: Emulator configuration
            capabilities: Gateway capabilities (to check what needs emulation)
        """
        self._bus = message_bus
        self._clock = clock
        self._config = config or EmulatorConfig()
        self._capabilities = capabilities

        # Order tracking
        self._emulated_orders: dict[str, EmulatedOrder] = {}
        self._bracket_orders: dict[str, BracketOrder] = {}
        self._oco_groups: dict[str, list[str]] = {}  # group_id -> order_ids
        self._oto_chains: dict[str, list[str]] = {}  # parent_id -> child_ids
        self._oto_filled_parents: set[str] = set()  # Track filled OTO parents

        # Price tracking (per symbol)
        self._last_prices: dict[str, Decimal] = {}
        self._last_bids: dict[str, Decimal] = {}
        self._last_asks: dict[str, Decimal] = {}

        # Stats
        self._stats = EmulatorStats()

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Callback for order submission (set by ExecutionEngine)
        self._submit_callback: Any = None

        logger.info("OrderEmulator initialized")

    @property
    def stats(self) -> EmulatorStats:
        """Get emulator statistics."""
        return self._stats

    def set_submit_callback(self, callback: Any) -> None:
        """Set callback for submitting triggered orders."""
        self._submit_callback = callback

    # =========================================================================
    # Order Submission
    # =========================================================================

    async def submit_order(
        self,
        order: Order,
    ) -> tuple[Order | None, EmulatedOrder | None]:
        """
        Process order - emulate if needed, pass through otherwise.

        Args:
            order: Order to process

        Returns:
            Tuple of (order_to_submit, emulated_order)
            - If fully emulated: (None, EmulatedOrder)
            - If pass-through: (Order, None)
            - If modified: (modified_Order, EmulatedOrder)
        """
        async with self._lock:
            # Check if order needs emulation
            if self._should_emulate(order):
                emulated = await self._create_emulated_order(order)
                self._stats.orders_emulated += 1
                logger.info(
                    "Order emulated: %s %s (trigger: %s)",
                    order.symbol,
                    order.order_type.value,
                    emulated.trigger_price,
                )
                return None, emulated

            # Check for bracket order
            if self._is_bracket_order(order):
                entry, emulated = await self._create_bracket_order(order)
                return entry, emulated

            # Pass through
            return order, None

    def _should_emulate(self, order: Order) -> bool:
        """Check if order type needs local emulation."""
        # Trailing stops are always emulated locally
        if order.trailing_offset is not None:
            return True

        # Check capabilities
        if self._capabilities is not None:
            if order.order_type == OrderType.STOP and not self._capabilities.stop_orders:
                return True
            if (
                order.order_type == OrderType.STOP_LIMIT
                and not self._capabilities.stop_limit_orders
            ):
                return True

        # Stop orders with trigger_type need emulation
        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if order.trigger_type is not None and order.trigger_type != TriggerType.LAST_PRICE:
                return True

        return False

    def _is_bracket_order(self, order: Order) -> bool:
        """Check if this is a bracket order request."""
        # Bracket orders have exec_algorithm_params with stop_loss and take_profit
        if order.exec_algorithm_params:
            return (
                "stop_loss" in order.exec_algorithm_params
                and "take_profit" in order.exec_algorithm_params
            )
        return False

    async def _create_emulated_order(self, order: Order) -> EmulatedOrder:
        """Create emulated order from original."""
        order_id = order.client_order_id or f"emu_{uuid.uuid4().hex[:12]}"
        now_ns = self._clock.timestamp_ns()

        # Determine trigger price
        if order.trailing_offset is not None:
            # Trailing stop - initial trigger calculated from current price
            current_price = self._last_prices.get(order.symbol)
            if current_price:
                trigger_price = self._calculate_trailing_trigger(
                    order.side,
                    current_price,
                    order.trailing_offset,
                    order.trailing_offset_type or "price",
                )
            else:
                trigger_price = order.stop_price
        else:
            trigger_price = order.stop_price

        emulated = EmulatedOrder(
            order_id=order_id,
            original_order=order,
            state=EmulatedOrderState.PENDING,
            trigger_price=trigger_price,
            trigger_type=order.trigger_type or TriggerType.LAST_PRICE,
            created_at_ns=now_ns,
            trailing_offset=order.trailing_offset,
            trailing_offset_type=order.trailing_offset_type,
            current_stop_price=trigger_price,
            best_price=self._last_prices.get(order.symbol),
        )

        self._emulated_orders[order_id] = emulated

        # Publish event
        await self._publish_emulated_event(emulated, "created")

        return emulated

    async def _create_bracket_order(
        self, order: Order
    ) -> tuple[Order, EmulatedOrder | None]:
        """
        Create bracket order (entry + SL + TP).

        Returns the entry order to submit immediately and
        creates emulated orders for SL and TP.
        """
        params = order.exec_algorithm_params or {}
        stop_loss_price = Decimal(str(params["stop_loss"]))
        take_profit_price = Decimal(str(params["take_profit"]))

        bracket_id = f"bracket_{uuid.uuid4().hex[:8]}"
        now_ns = self._clock.timestamp_ns()

        # Entry order (modify to remove bracket params)
        entry_order = Order(
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            amount=order.amount,
            price=order.price,
            client_order_id=f"{bracket_id}_entry",
            timestamp_ns=now_ns,
            contingency_type=ContingencyType.OTO,
        )

        # Stop-loss order (opposite side)
        sl_side = OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY
        sl_order = Order(
            symbol=order.symbol,
            side=sl_side,
            order_type=OrderType.STOP,
            amount=order.amount,
            stop_price=stop_loss_price,
            client_order_id=f"{bracket_id}_sl",
            timestamp_ns=now_ns,
            parent_order_id=entry_order.client_order_id,
            contingency_type=ContingencyType.OCO,
        )

        # Take-profit order (opposite side)
        tp_order = Order(
            symbol=order.symbol,
            side=sl_side,
            order_type=OrderType.LIMIT,
            amount=order.amount,
            price=take_profit_price,
            client_order_id=f"{bracket_id}_tp",
            timestamp_ns=now_ns,
            parent_order_id=entry_order.client_order_id,
            contingency_type=ContingencyType.OCO,
        )

        # Create emulated stop-loss
        sl_emulated = EmulatedOrder(
            order_id=sl_order.client_order_id or "",
            original_order=sl_order,
            state=EmulatedOrderState.PENDING,
            trigger_price=stop_loss_price,
            created_at_ns=now_ns,
            is_child=True,
            parent_order_id=entry_order.client_order_id,
        )

        # Create emulated take-profit (limit order, triggers on entry fill)
        tp_emulated = EmulatedOrder(
            order_id=tp_order.client_order_id or "",
            original_order=tp_order,
            state=EmulatedOrderState.PENDING,
            created_at_ns=now_ns,
            is_child=True,
            parent_order_id=entry_order.client_order_id,
        )

        # Link SL and TP as OCO
        sl_emulated.linked_order_ids = [tp_emulated.order_id]
        tp_emulated.linked_order_ids = [sl_emulated.order_id]

        # Store
        self._emulated_orders[sl_emulated.order_id] = sl_emulated
        self._emulated_orders[tp_emulated.order_id] = tp_emulated

        # Create bracket tracking
        bracket = BracketOrder(
            bracket_id=bracket_id,
            entry_order_id=entry_order.client_order_id or "",
            stop_loss_order_id=sl_emulated.order_id,
            take_profit_order_id=tp_emulated.order_id,
            symbol=order.symbol,
            side=order.side,
        )
        self._bracket_orders[bracket_id] = bracket

        # Track OTO: entry triggers SL and TP
        self._oto_chains[entry_order.client_order_id or ""] = [
            sl_emulated.order_id,
            tp_emulated.order_id,
        ]

        # Track OCO: SL and TP cancel each other
        oco_group_id = f"oco_{bracket_id}"
        self._oco_groups[oco_group_id] = [sl_emulated.order_id, tp_emulated.order_id]

        self._stats.bracket_orders += 1
        logger.info(
            "Bracket order created: %s (SL: %s, TP: %s)",
            bracket_id,
            stop_loss_price,
            take_profit_price,
        )

        return entry_order, sl_emulated

    # =========================================================================
    # OCO / OTO Management
    # =========================================================================

    async def submit_oco_orders(self, orders: list[Order]) -> list[EmulatedOrder]:
        """
        Submit OCO (One-Cancels-Other) order group.

        When any order in the group fills, others are cancelled.

        Args:
            orders: List of orders (2+) to link as OCO

        Returns:
            List of emulated orders
        """
        if len(orders) < 2:
            raise ValueError("OCO requires at least 2 orders")

        async with self._lock:
            oco_group_id = f"oco_{uuid.uuid4().hex[:8]}"
            emulated_orders = []
            order_ids = []

            for order in orders:
                emulated = await self._create_emulated_order(order)
                emulated_orders.append(emulated)
                order_ids.append(emulated.order_id)

            # Link all orders
            for emulated in emulated_orders:
                emulated.linked_order_ids = [
                    oid for oid in order_ids if oid != emulated.order_id
                ]

            self._oco_groups[oco_group_id] = order_ids
            self._stats.oco_orders += len(orders)

            logger.info("OCO group created: %s with %d orders", oco_group_id, len(orders))
            return emulated_orders

    async def submit_oto_chain(
        self, parent_order: Order, child_orders: list[Order]
    ) -> tuple[Order, list[EmulatedOrder]]:
        """
        Submit OTO (One-Triggers-Other) order chain.

        Parent order fill triggers submission of child orders.

        Args:
            parent_order: Order that triggers children
            child_orders: Orders to submit when parent fills

        Returns:
            Tuple of (parent_order, list of child emulated orders)
        """
        async with self._lock:
            parent_id = parent_order.client_order_id or f"oto_{uuid.uuid4().hex[:8]}"

            # Modify parent to have OTO contingency
            parent = Order(
                symbol=parent_order.symbol,
                side=parent_order.side,
                order_type=parent_order.order_type,
                amount=parent_order.amount,
                price=parent_order.price,
                stop_price=parent_order.stop_price,
                client_order_id=parent_id,
                contingency_type=ContingencyType.OTO,
                timestamp_ns=self._clock.timestamp_ns(),
            )

            # Create emulated child orders (pending until parent fills)
            child_emulated = []
            child_ids = []

            for child in child_orders:
                emulated = await self._create_emulated_order(child)
                emulated.parent_order_id = parent_id
                emulated.is_child = True
                child_emulated.append(emulated)
                child_ids.append(emulated.order_id)

            self._oto_chains[parent_id] = child_ids
            self._stats.oto_orders += len(child_orders)

            logger.info(
                "OTO chain created: parent=%s, children=%d", parent_id, len(child_orders)
            )
            return parent, child_emulated

    # =========================================================================
    # Price Tick Processing
    # =========================================================================

    async def on_tick(
        self,
        symbol: str,
        last_price: Decimal | None = None,
        bid_price: Decimal | None = None,
        ask_price: Decimal | None = None,
    ) -> list[Order]:
        """
        Process price tick - check emulated orders for triggers.

        Args:
            symbol: Trading symbol
            last_price: Last trade price
            bid_price: Best bid
            ask_price: Best ask

        Returns:
            List of orders triggered and ready to submit
        """
        async with self._lock:
            # Update price cache
            if last_price is not None:
                self._last_prices[symbol] = last_price
            if bid_price is not None:
                self._last_bids[symbol] = bid_price
            if ask_price is not None:
                self._last_asks[symbol] = ask_price

            triggered_orders: list[Order] = []

            # Check all pending emulated orders for this symbol
            for order_id, emulated in list(self._emulated_orders.items()):
                if emulated.original_order.symbol != symbol:
                    continue
                if emulated.state != EmulatedOrderState.PENDING:
                    continue
                if emulated.is_child and not self._is_parent_filled(emulated):
                    continue

                # Get trigger price based on trigger type
                check_price = self._get_check_price(emulated, last_price, bid_price, ask_price)
                if check_price is None:
                    continue

                # Update trailing stops
                if emulated.trailing_offset is not None:
                    self._update_trailing_stop(emulated, check_price)

                # Check trigger condition
                if self._is_triggered(emulated, check_price):
                    order = await self._trigger_order(emulated)
                    if order:
                        triggered_orders.append(order)

            return triggered_orders

    def _get_check_price(
        self,
        emulated: EmulatedOrder,
        last_price: Decimal | None,
        bid_price: Decimal | None,
        ask_price: Decimal | None,
    ) -> Decimal | None:
        """Get price to check against trigger based on trigger type."""
        trigger_type = emulated.trigger_type

        if trigger_type == TriggerType.LAST_PRICE:
            return last_price
        elif trigger_type == TriggerType.BID_ASK:
            # For sells, use bid; for buys, use ask
            if emulated.original_order.side == OrderSide.SELL:
                return bid_price
            else:
                return ask_price
        elif trigger_type == TriggerType.MID_POINT:
            if bid_price and ask_price:
                return (bid_price + ask_price) / 2
            return None
        elif trigger_type == TriggerType.MARK_PRICE:
            # Mark price would come from derivatives exchange
            return last_price
        else:
            return last_price

    def _update_trailing_stop(self, emulated: EmulatedOrder, current_price: Decimal) -> None:
        """Update trailing stop price based on current price."""
        if emulated.trailing_offset is None:
            return

        order = emulated.original_order
        offset = emulated.trailing_offset
        offset_type = emulated.trailing_offset_type or "price"

        # Track best price
        if emulated.best_price is None:
            emulated.best_price = current_price
        elif order.side == OrderSide.SELL:
            # Trailing sell stop follows price up
            if current_price > emulated.best_price:
                emulated.best_price = current_price
        else:
            # Trailing buy stop follows price down
            if current_price < emulated.best_price:
                emulated.best_price = current_price

        # Calculate new stop price
        new_stop = self._calculate_trailing_trigger(
            order.side, emulated.best_price, offset, offset_type
        )

        # Only update if materially different
        if emulated.current_stop_price is not None:
            threshold = self._config.trailing_update_threshold
            pct_change = abs(new_stop - emulated.current_stop_price) / emulated.current_stop_price

            if pct_change < Decimal(str(threshold)):
                return

        emulated.current_stop_price = new_stop
        emulated.trigger_price = new_stop
        self._stats.trailing_updates += 1

        logger.debug(
            "Trailing stop updated: %s -> %s (best: %s)",
            emulated.order_id,
            new_stop,
            emulated.best_price,
        )

    def _calculate_trailing_trigger(
        self,
        side: OrderSide,
        best_price: Decimal,
        offset: Decimal,
        offset_type: str,
    ) -> Decimal:
        """Calculate trailing stop trigger price."""
        if offset_type == "percent":
            offset_amount = best_price * offset / Decimal("100")
        else:
            offset_amount = offset

        if side == OrderSide.SELL:
            # Sell stop triggers below best price
            return best_price - offset_amount
        else:
            # Buy stop triggers above best price
            return best_price + offset_amount

    def _is_triggered(self, emulated: EmulatedOrder, current_price: Decimal) -> bool:
        """Check if emulated order trigger condition is met."""
        trigger_price = emulated.trigger_price or emulated.current_stop_price
        if trigger_price is None:
            return False

        order = emulated.original_order

        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if order.side == OrderSide.SELL:
                # Sell stop triggers when price falls to/below trigger
                return current_price <= trigger_price
            else:
                # Buy stop triggers when price rises to/above trigger
                return current_price >= trigger_price

        # For limit orders (take-profit in bracket), check price reach
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.SELL:
                # Take-profit sell triggers when price rises to limit
                return current_price >= (order.price or Decimal("0"))
            else:
                # Take-profit buy triggers when price falls to limit
                return current_price <= (order.price or Decimal("inf"))

        return False

    async def _trigger_order(self, emulated: EmulatedOrder) -> Order | None:
        """Trigger emulated order - prepare for submission."""
        emulated.state = EmulatedOrderState.TRIGGERED
        emulated.triggered_at_ns = self._clock.timestamp_ns()
        self._stats.orders_triggered += 1

        order = emulated.original_order

        # Convert stop to market order for execution
        if order.order_type == OrderType.STOP:
            submit_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                amount=order.amount,
                client_order_id=emulated.order_id,
                timestamp_ns=self._clock.timestamp_ns(),
                parent_order_id=emulated.parent_order_id,
            )
        elif order.order_type == OrderType.STOP_LIMIT:
            submit_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                amount=order.amount,
                price=order.price,
                client_order_id=emulated.order_id,
                timestamp_ns=self._clock.timestamp_ns(),
                parent_order_id=emulated.parent_order_id,
            )
        else:
            submit_order = order

        emulated.submitted_order = submit_order

        logger.info(
            "Order triggered: %s %s %s @ %s",
            emulated.order_id,
            order.side.value,
            order.symbol,
            emulated.trigger_price,
        )

        await self._publish_emulated_event(emulated, "triggered")

        return submit_order

    def _is_parent_filled(self, emulated: EmulatedOrder) -> bool:
        """Check if parent order is filled (for OTO children)."""
        if emulated.parent_order_id is None:
            return True

        # Check if parent is in filled OTO parents
        if emulated.parent_order_id in self._oto_filled_parents:
            return True

        # Check if parent is in filled brackets
        for bracket in self._bracket_orders.values():
            if bracket.entry_order_id == emulated.parent_order_id:
                return bracket.entry_filled

        # Parent exists but not filled yet
        return False

    # =========================================================================
    # Order Status Updates
    # =========================================================================

    async def on_order_filled(self, order_id: str, fill_price: Decimal) -> None:
        """
        Handle order fill - trigger OTO children, cancel OCO siblings.

        Args:
            order_id: Filled order ID
            fill_price: Fill price
        """
        async with self._lock:
            # Update emulated order state
            if order_id in self._emulated_orders:
                emulated = self._emulated_orders[order_id]
                emulated.state = EmulatedOrderState.FILLED
                emulated.filled_at_ns = self._clock.timestamp_ns()

                # Cancel OCO siblings
                await self._cancel_oco_siblings(emulated)

            # Trigger OTO children
            if order_id in self._oto_chains:
                self._oto_filled_parents.add(order_id)  # Mark parent as filled
                child_ids = self._oto_chains[order_id]
                for child_id in child_ids:
                    if child_id in self._emulated_orders:
                        child = self._emulated_orders[child_id]
                        child.is_child = False  # Enable triggering

                logger.info("OTO children enabled: %s -> %s", order_id, child_ids)

            # Update bracket state
            for bracket in self._bracket_orders.values():
                if bracket.entry_order_id == order_id:
                    bracket.entry_filled = True
                    bracket.state = "active"
                    logger.info("Bracket entry filled: %s", bracket.bracket_id)
                elif order_id in (bracket.stop_loss_order_id, bracket.take_profit_order_id):
                    bracket.state = "closed"
                    logger.info("Bracket closed: %s", bracket.bracket_id)

    async def _cancel_oco_siblings(self, filled_order: EmulatedOrder) -> None:
        """Cancel other orders in OCO group when one fills."""
        for sibling_id in filled_order.linked_order_ids:
            if sibling_id in self._emulated_orders:
                sibling = self._emulated_orders[sibling_id]
                if sibling.state == EmulatedOrderState.PENDING:
                    sibling.state = EmulatedOrderState.CANCELLED
                    self._stats.orders_cancelled += 1
                    await self._publish_emulated_event(sibling, "cancelled")
                    logger.info(
                        "OCO cancelled: %s (sibling of %s filled)",
                        sibling_id,
                        filled_order.order_id,
                    )

    async def on_order_cancelled(self, order_id: str) -> None:
        """Handle order cancellation."""
        async with self._lock:
            if order_id in self._emulated_orders:
                emulated = self._emulated_orders[order_id]
                emulated.state = EmulatedOrderState.CANCELLED
                await self._publish_emulated_event(emulated, "cancelled")

    async def on_order_rejected(self, order_id: str, reason: str) -> None:
        """Handle order rejection."""
        async with self._lock:
            if order_id in self._emulated_orders:
                emulated = self._emulated_orders[order_id]
                emulated.state = EmulatedOrderState.REJECTED
                await self._publish_emulated_event(emulated, "rejected", reason=reason)

    # =========================================================================
    # Order Management
    # =========================================================================

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel emulated order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled, False if not found
        """
        async with self._lock:
            if order_id not in self._emulated_orders:
                return False

            emulated = self._emulated_orders[order_id]
            if emulated.state not in (
                EmulatedOrderState.PENDING,
                EmulatedOrderState.TRIGGERED,
            ):
                return False

            emulated.state = EmulatedOrderState.CANCELLED
            self._stats.orders_cancelled += 1
            await self._publish_emulated_event(emulated, "cancelled")

            logger.info("Emulated order cancelled: %s", order_id)
            return True

    async def cancel_all(self, symbol: str | None = None) -> int:
        """
        Cancel all pending emulated orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            Number of orders cancelled
        """
        count = 0
        async with self._lock:
            for order_id, emulated in self._emulated_orders.items():
                if emulated.state != EmulatedOrderState.PENDING:
                    continue
                if symbol and emulated.original_order.symbol != symbol:
                    continue

                emulated.state = EmulatedOrderState.CANCELLED
                count += 1
                await self._publish_emulated_event(emulated, "cancelled")

        self._stats.orders_cancelled += count
        logger.info("Cancelled %d emulated orders", count)
        return count

    def get_order(self, order_id: str) -> EmulatedOrder | None:
        """Get emulated order by ID."""
        return self._emulated_orders.get(order_id)

    def get_pending_orders(self, symbol: str | None = None) -> list[EmulatedOrder]:
        """Get all pending emulated orders."""
        orders = [
            o for o in self._emulated_orders.values()
            if o.state == EmulatedOrderState.PENDING
        ]
        if symbol:
            orders = [o for o in orders if o.original_order.symbol == symbol]
        return orders

    def get_bracket_order(self, bracket_id: str) -> BracketOrder | None:
        """Get bracket order by ID."""
        return self._bracket_orders.get(bracket_id)

    # =========================================================================
    # Events
    # =========================================================================

    async def _publish_emulated_event(
        self,
        emulated: EmulatedOrder,
        action: str,
        reason: str | None = None,
    ) -> None:
        """Publish emulated order event."""
        # Map action to event type
        event_type_map = {
            "created": EventType.ORDER_NEW,
            "triggered": EventType.ORDER_SUBMITTED,
            "cancelled": EventType.ORDER_CANCELLED,
            "rejected": EventType.ORDER_REJECTED,
            "filled": EventType.ORDER_FILLED,
        }
        event_type = event_type_map.get(action, EventType.ORDER_NEW)

        event = Event.create(
            event_type=event_type,
            source="order_emulator",
            payload={
                "action": action,
                "emulated": True,
                "order_id": emulated.order_id,
                "symbol": emulated.original_order.symbol,
                "side": emulated.original_order.side.value,
                "order_type": emulated.original_order.order_type.value,
                "state": emulated.state.value,
                "trigger_price": str(emulated.trigger_price) if emulated.trigger_price else None,
                "trailing_offset": str(emulated.trailing_offset) if emulated.trailing_offset else None,
                "reason": reason,
            },
        )
        await self._bus.publish("order_emulator", event)
