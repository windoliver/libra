"""
BacktestExecutionClient: Simulated order execution for backtesting.

Provides realistic order fill simulation with:
    - Multiple fill models (market, limit with queue position)
    - Configurable slippage models
    - Partial fill support
    - Position and balance tracking

Design inspired by:
    - NautilusTrader BacktestExecutionClient
    - hftbacktest fill models
    - Backtrader broker simulation

See: https://github.com/windoliver/libra/issues/33
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any

from libra.clients.execution_client import (
    BaseExecutionClient,
    InsufficientFundsError,
    OrderError,
    OrderNotFoundError,
)
from libra.gateways.protocol import (
    Balance,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Tick,
)


if TYPE_CHECKING:
    from libra.core.clock import Clock


# =============================================================================
# Slippage Models
# =============================================================================


class SlippageModel(str, Enum):
    """Available slippage models."""

    NONE = "none"  # No slippage (optimistic)
    FIXED = "fixed"  # Fixed basis points
    VOLUME = "volume"  # Based on order size vs market volume
    STOCHASTIC = "stochastic"  # Random slippage (normal distribution)


class BaseSlippageModel(ABC):
    """Base class for slippage calculation."""

    @abstractmethod
    def calculate(
        self,
        order: Order,
        tick: Tick,
        fill_price: Decimal,
    ) -> Decimal:
        """
        Calculate slippage for an order.

        Args:
            order: Order being filled
            tick: Current market tick
            fill_price: Base fill price before slippage

        Returns:
            Slippage amount (positive = worse for trader)
        """
        ...


class NoSlippage(BaseSlippageModel):
    """No slippage model (optimistic)."""

    def calculate(
        self,
        order: Order,
        tick: Tick,
        fill_price: Decimal,
    ) -> Decimal:
        return Decimal("0")


class FixedSlippage(BaseSlippageModel):
    """Fixed basis points slippage."""

    def __init__(self, bps: Decimal = Decimal("5")) -> None:
        """
        Initialize fixed slippage.

        Args:
            bps: Basis points of slippage (default 5 bps = 0.05%)
        """
        self.bps = bps

    def calculate(
        self,
        order: Order,
        tick: Tick,
        fill_price: Decimal,
    ) -> Decimal:
        # Slippage is always unfavorable
        slippage = fill_price * (self.bps / Decimal("10000"))
        if order.side == OrderSide.BUY:
            return slippage  # Pay more
        else:
            return -slippage  # Receive less


class VolumeSlippage(BaseSlippageModel):
    """Volume-based slippage (larger orders have more slippage)."""

    def __init__(
        self,
        base_bps: Decimal = Decimal("2"),
        volume_impact: Decimal = Decimal("0.1"),
    ) -> None:
        """
        Initialize volume-based slippage.

        Args:
            base_bps: Base slippage in basis points
            volume_impact: Additional bps per 1% of daily volume
        """
        self.base_bps = base_bps
        self.volume_impact = volume_impact

    def calculate(
        self,
        order: Order,
        tick: Tick,
        fill_price: Decimal,
    ) -> Decimal:
        # Estimate volume impact (simplified)
        daily_volume = tick.volume_24h or Decimal("1000000")
        order_pct = (order.amount * fill_price) / daily_volume * 100

        total_bps = self.base_bps + (self.volume_impact * order_pct)
        slippage = fill_price * (total_bps / Decimal("10000"))

        if order.side == OrderSide.BUY:
            return slippage
        else:
            return -slippage


class StochasticSlippage(BaseSlippageModel):
    """Random slippage with normal distribution."""

    def __init__(
        self,
        mean_bps: Decimal = Decimal("3"),
        std_bps: Decimal = Decimal("2"),
    ) -> None:
        """
        Initialize stochastic slippage.

        Args:
            mean_bps: Mean slippage in basis points
            std_bps: Standard deviation in basis points
        """
        self.mean_bps = mean_bps
        self.std_bps = std_bps

    def calculate(
        self,
        order: Order,
        tick: Tick,
        fill_price: Decimal,
    ) -> Decimal:
        import random

        # Generate random slippage (always non-negative for realism)
        bps = max(Decimal("0"), Decimal(str(random.gauss(float(self.mean_bps), float(self.std_bps)))))
        slippage = fill_price * (bps / Decimal("10000"))

        if order.side == OrderSide.BUY:
            return slippage
        else:
            return -slippage


# =============================================================================
# Fill Models
# =============================================================================


class FillModel(ABC):
    """Base class for order fill simulation."""

    @abstractmethod
    def check_fill(
        self,
        order: Order,
        tick: Tick,
        position_in_queue: int = 0,
    ) -> tuple[Decimal | None, Decimal]:
        """
        Check if order should fill and at what price/quantity.

        Args:
            order: Order to check
            tick: Current market tick
            position_in_queue: Queue position for limit orders

        Returns:
            (fill_price, fill_quantity) or (None, 0) if no fill
        """
        ...


class ImmediateFillModel(FillModel):
    """
    Immediate fill at market price.

    Market orders fill at ask (buy) or bid (sell).
    Limit orders fill if price is crossed.
    """

    def check_fill(
        self,
        order: Order,
        tick: Tick,
        position_in_queue: int = 0,
    ) -> tuple[Decimal | None, Decimal]:
        if order.order_type == OrderType.MARKET:
            # Market order fills immediately at current price
            if order.side == OrderSide.BUY:
                return tick.ask, order.amount
            else:
                return tick.bid, order.amount

        elif order.order_type == OrderType.LIMIT:
            # Limit order fills if price crosses
            if order.side == OrderSide.BUY:
                if tick.ask <= order.price:
                    return order.price, order.amount
            else:
                if tick.bid >= order.price:
                    return order.price, order.amount

        return None, Decimal("0")


class QueuePositionFillModel(FillModel):
    """
    Fill model that considers queue position for limit orders.

    Based on hftbacktest probability queue models.
    More realistic for limit order fills.
    """

    def __init__(
        self,
        fill_probability_at_touch: Decimal = Decimal("0.5"),
        fill_probability_decay: Decimal = Decimal("0.1"),
    ) -> None:
        """
        Initialize queue position fill model.

        Args:
            fill_probability_at_touch: Probability of fill when at best price
            fill_probability_decay: Decrease in probability per queue position
        """
        self.fill_prob_touch = fill_probability_at_touch
        self.fill_prob_decay = fill_probability_decay

    def check_fill(
        self,
        order: Order,
        tick: Tick,
        position_in_queue: int = 0,
    ) -> tuple[Decimal | None, Decimal]:
        import random

        if order.order_type == OrderType.MARKET:
            # Market orders always fill
            if order.side == OrderSide.BUY:
                return tick.ask, order.amount
            else:
                return tick.bid, order.amount

        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                # Check if price crosses
                if tick.ask < order.price:
                    # Aggressive cross - definitely fills
                    return order.price, order.amount
                elif tick.ask == order.price:
                    # At touch - probabilistic fill based on queue
                    fill_prob = float(self.fill_prob_touch - self.fill_prob_decay * position_in_queue)
                    if random.random() < fill_prob:
                        return order.price, order.amount
            else:
                if tick.bid > order.price:
                    return order.price, order.amount
                elif tick.bid == order.price:
                    fill_prob = float(self.fill_prob_touch - self.fill_prob_decay * position_in_queue)
                    if random.random() < fill_prob:
                        return order.price, order.amount

        return None, Decimal("0")


# =============================================================================
# BacktestExecutionClient
# =============================================================================


class BacktestExecutionClient(BaseExecutionClient):
    """
    Execution client for backtesting with simulated order fills.

    Simulates realistic order execution with:
        - Configurable fill models
        - Configurable slippage models
        - Position and balance tracking
        - Order lifecycle events

    Examples:
        client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("10000")},
            slippage_model=SlippageModel.FIXED,
            maker_fee=Decimal("0.001"),
            taker_fee=Decimal("0.001"),
        )

        await client.connect()
        result = await client.submit_order(order)

        # Process market data to trigger fills
        await client.process_tick(tick)

    Thread Safety:
        Not thread-safe. Use from a single async context.
    """

    def __init__(
        self,
        clock: Clock,
        initial_balance: dict[str, Decimal] | None = None,
        slippage_model: SlippageModel = SlippageModel.FIXED,
        slippage_bps: Decimal = Decimal("5"),
        fill_model: FillModel | None = None,
        maker_fee: Decimal = Decimal("0.001"),  # 0.1%
        taker_fee: Decimal = Decimal("0.001"),  # 0.1%
        name: str = "backtest-exec",
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize backtest execution client.

        Args:
            clock: Clock instance (must be in BACKTEST mode)
            initial_balance: Starting balances (currency -> amount)
            slippage_model: Type of slippage model
            slippage_bps: Slippage in basis points (for FIXED model)
            fill_model: Custom fill model (default: ImmediateFillModel)
            maker_fee: Maker fee rate (default 0.1%)
            taker_fee: Taker fee rate (default 0.1%)
            name: Client identifier
            config: Optional configuration
        """
        super().__init__(name, config)
        self.clock = clock

        # Fee structure
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        # Slippage model
        self._slippage: BaseSlippageModel
        if slippage_model == SlippageModel.NONE:
            self._slippage = NoSlippage()
        elif slippage_model == SlippageModel.FIXED:
            self._slippage = FixedSlippage(slippage_bps)
        elif slippage_model == SlippageModel.VOLUME:
            self._slippage = VolumeSlippage()
        elif slippage_model == SlippageModel.STOCHASTIC:
            self._slippage = StochasticSlippage()
        else:
            self._slippage = FixedSlippage(slippage_bps)

        # Fill model
        self._fill_model = fill_model or ImmediateFillModel()

        # Account state
        self._balances: dict[str, Balance] = {}
        self._positions: dict[str, Position] = {}

        # Initialize balances
        if initial_balance:
            for currency, amount in initial_balance.items():
                self._balances[currency] = Balance(
                    currency=currency,
                    total=amount,
                    available=amount,
                    locked=Decimal("0"),
                )

        # Order tracking
        self._pending_orders: dict[str, Order] = {}  # order_id -> Order
        self._order_queue_position: dict[str, int] = {}  # order_id -> queue position
        self._order_history: list[OrderResult] = []
        self._next_order_id = 1

        # Event stream
        self._order_updates: asyncio.Queue[OrderResult] = asyncio.Queue()

        # Latest ticks per symbol (for fill checking)
        self._latest_ticks: dict[str, Tick] = {}

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect (initialize for backtesting)."""
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect and clean up."""
        self._connected = False

    # -------------------------------------------------------------------------
    # Order Management
    # -------------------------------------------------------------------------

    async def submit_order(self, order: Order) -> OrderResult:
        """
        Submit an order for simulated execution.

        Market orders are filled immediately against latest tick.
        Limit orders are queued and filled when price crosses.
        """
        # Generate order ID
        order_id = f"backtest-{self._next_order_id}"
        self._next_order_id += 1

        # Create order with ID
        order_with_id = order.with_id(order_id)

        # Check balance
        if not self._check_balance(order_with_id):
            result = OrderResult(
                order_id=order_id,
                symbol=order.symbol,
                status=OrderStatus.REJECTED,
                side=order.side,
                order_type=order.order_type,
                amount=order.amount,
                filled_amount=Decimal("0"),
                remaining_amount=order.amount,
                average_price=None,
                fee=Decimal("0"),
                fee_currency=order.symbol.split("/")[1],
                timestamp_ns=self.clock.timestamp_ns(),
                client_order_id=order.client_order_id,
            )
            await self._order_updates.put(result)
            return result

        # Lock balance
        self._lock_balance(order_with_id)

        # For market orders, try immediate fill
        if order.order_type == OrderType.MARKET:
            tick = self._latest_ticks.get(order.symbol)
            if tick:
                fill_price, fill_qty = self._fill_model.check_fill(order_with_id, tick)
                if fill_price is not None:
                    return await self._execute_fill(order_with_id, fill_price, fill_qty, tick)

        # Queue order for later fill
        self._pending_orders[order_id] = order_with_id
        self._order_queue_position[order_id] = 0  # Start at front of queue
        self._open_orders[order_id] = OrderResult(
            order_id=order_id,
            symbol=order.symbol,
            status=OrderStatus.OPEN,
            side=order.side,
            order_type=order.order_type,
            amount=order.amount,
            filled_amount=Decimal("0"),
            remaining_amount=order.amount,
            average_price=None,
            fee=Decimal("0"),
            fee_currency=order.symbol.split("/")[1],
            timestamp_ns=self.clock.timestamp_ns(),
            client_order_id=order.client_order_id,
            price=order.price,
        )

        result = self._open_orders[order_id]
        await self._order_updates.put(result)
        return result

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a pending order."""
        if order_id not in self._pending_orders:
            if order_id in self._open_orders:
                # Already filled or cancelled
                return False
            raise OrderNotFoundError(f"Order {order_id} not found")

        order = self._pending_orders.pop(order_id)
        self._order_queue_position.pop(order_id, None)

        # Unlock balance
        self._unlock_balance(order)

        # Create cancelled result
        result = OrderResult(
            order_id=order_id,
            symbol=symbol,
            status=OrderStatus.CANCELLED,
            side=order.side,
            order_type=order.order_type,
            amount=order.amount,
            filled_amount=Decimal("0"),
            remaining_amount=order.amount,
            average_price=None,
            fee=Decimal("0"),
            fee_currency=symbol.split("/")[1],
            timestamp_ns=self.clock.timestamp_ns(),
            client_order_id=order.client_order_id,
        )

        self._open_orders[order_id] = result
        self._order_history.append(result)
        await self._order_updates.put(result)

        return True

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all pending orders."""
        cancelled = 0
        orders_to_cancel = list(self._pending_orders.items())

        for order_id, order in orders_to_cancel:
            if symbol is None or order.symbol == symbol:
                await self.cancel_order(order_id, order.symbol)
                cancelled += 1

        return cancelled

    async def modify_order(
        self,
        order_id: str,
        symbol: str,
        price: Any | None = None,
        amount: Any | None = None,
    ) -> OrderResult:
        """Modify a pending order (cancel and replace)."""
        if order_id not in self._pending_orders:
            raise OrderNotFoundError(f"Order {order_id} not found")

        old_order = self._pending_orders[order_id]

        # Cancel old order
        await self.cancel_order(order_id, symbol)

        # Create new order with modifications
        new_order = Order(
            symbol=old_order.symbol,
            side=old_order.side,
            order_type=old_order.order_type,
            amount=Decimal(str(amount)) if amount else old_order.amount,
            price=Decimal(str(price)) if price else old_order.price,
            client_order_id=old_order.client_order_id,
            time_in_force=old_order.time_in_force,
        )

        # Submit new order
        return await self.submit_order(new_order)

    # -------------------------------------------------------------------------
    # Order Queries
    # -------------------------------------------------------------------------

    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        """Get order status."""
        if order_id in self._open_orders:
            return self._open_orders[order_id]

        for result in self._order_history:
            if result.order_id == order_id:
                return result

        raise OrderNotFoundError(f"Order {order_id} not found")

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Get all open orders."""
        results = []
        for order_id in self._pending_orders:
            if order_id in self._open_orders:
                result = self._open_orders[order_id]
                if symbol is None or result.symbol == symbol:
                    results.append(result)
        return results

    async def get_order_history(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[OrderResult]:
        """Get order history."""
        history = self._order_history
        if symbol:
            history = [r for r in history if r.symbol == symbol]
        return history[-limit:]

    def stream_order_updates(self) -> AsyncIterator[OrderResult]:
        """Stream order updates."""
        return self._stream_updates()

    async def _stream_updates(self) -> AsyncIterator[OrderResult]:
        """Internal async generator for order updates."""
        while self._connected:
            try:
                update = await asyncio.wait_for(
                    self._order_updates.get(),
                    timeout=0.1,
                )
                yield update
            except asyncio.TimeoutError:
                continue

    # -------------------------------------------------------------------------
    # Account State
    # -------------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Get all positions."""
        return list(self._positions.values())

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for symbol."""
        return self._positions.get(symbol)

    async def get_balances(self) -> dict[str, Balance]:
        """Get all balances."""
        return self._balances.copy()

    async def get_balance(self, currency: str) -> Balance | None:
        """Get balance for currency."""
        return self._balances.get(currency)

    # -------------------------------------------------------------------------
    # Reconciliation
    # -------------------------------------------------------------------------

    async def reconcile_orders(self) -> int:
        """Reconcile orders (no-op in backtest)."""
        return 0

    async def reconcile_positions(self) -> int:
        """Reconcile positions (no-op in backtest)."""
        return 0

    # -------------------------------------------------------------------------
    # Market Data Processing
    # -------------------------------------------------------------------------

    async def process_tick(self, tick: Tick) -> list[OrderResult]:
        """
        Process a market tick and check for order fills.

        Call this method for each tick received from BacktestDataClient.

        Args:
            tick: Market tick data

        Returns:
            List of OrderResults for any fills that occurred
        """
        self._latest_ticks[tick.symbol] = tick
        fills: list[OrderResult] = []

        # Check pending orders for fills
        orders_to_check = [
            (oid, o) for oid, o in self._pending_orders.items()
            if o.symbol == tick.symbol
        ]

        for order_id, order in orders_to_check:
            queue_pos = self._order_queue_position.get(order_id, 0)
            fill_price, fill_qty = self._fill_model.check_fill(order, tick, queue_pos)

            if fill_price is not None and fill_qty > 0:
                result = await self._execute_fill(order, fill_price, fill_qty, tick)
                fills.append(result)

                # Remove from pending if fully filled
                if result.status == OrderStatus.FILLED:
                    self._pending_orders.pop(order_id, None)
                    self._order_queue_position.pop(order_id, None)

        # Update queue positions for remaining orders
        for order_id in self._order_queue_position:
            self._order_queue_position[order_id] += 1

        return fills

    async def process_bar(self, bar: Any) -> list[OrderResult]:
        """
        Process a bar and check for order fills.

        Converts bar to synthetic ticks (O, H, L, C) and processes each.

        Args:
            bar: Bar data (from strategies.protocol.Bar)

        Returns:
            List of OrderResults for any fills that occurred
        """
        fills: list[OrderResult] = []

        # Create synthetic ticks from bar (O→H→L→C or O→L→H→C)
        # Determine likely sequence based on close vs open
        if bar.close >= bar.open:
            # Bullish bar: likely went down first, then up
            prices = [bar.open, bar.low, bar.high, bar.close]
        else:
            # Bearish bar: likely went up first, then down
            prices = [bar.open, bar.high, bar.low, bar.close]

        for price in prices:
            tick = Tick(
                symbol=bar.symbol,
                bid=price,
                ask=price,
                last=price,
                timestamp_ns=bar.timestamp_ns,
            )
            bar_fills = await self.process_tick(tick)
            fills.extend(bar_fills)

        return fills

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _check_balance(self, order: Order) -> bool:
        """Check if sufficient balance for order."""
        quote = order.symbol.split("/")[1]
        base = order.symbol.split("/")[0]

        if order.side == OrderSide.BUY:
            # Need quote currency to buy
            required = order.amount * (order.price or Decimal("0"))
            balance = self._balances.get(quote)
            if not balance or balance.available < required:
                return False
        else:
            # Need base currency to sell
            balance = self._balances.get(base)
            if not balance or balance.available < order.amount:
                return False

        return True

    def _lock_balance(self, order: Order) -> None:
        """Lock balance for pending order."""
        quote = order.symbol.split("/")[1]
        base = order.symbol.split("/")[0]

        if order.side == OrderSide.BUY:
            required = order.amount * (order.price or Decimal("100000"))
            balance = self._balances[quote]
            self._balances[quote] = Balance(
                currency=quote,
                total=balance.total,
                available=balance.available - required,
                locked=balance.locked + required,
            )
        else:
            balance = self._balances[base]
            self._balances[base] = Balance(
                currency=base,
                total=balance.total,
                available=balance.available - order.amount,
                locked=balance.locked + order.amount,
            )

    def _unlock_balance(self, order: Order) -> None:
        """Unlock balance for cancelled order."""
        quote = order.symbol.split("/")[1]
        base = order.symbol.split("/")[0]

        if order.side == OrderSide.BUY:
            required = order.amount * (order.price or Decimal("100000"))
            balance = self._balances[quote]
            self._balances[quote] = Balance(
                currency=quote,
                total=balance.total,
                available=balance.available + required,
                locked=balance.locked - required,
            )
        else:
            balance = self._balances[base]
            self._balances[base] = Balance(
                currency=base,
                total=balance.total,
                available=balance.available + order.amount,
                locked=balance.locked - order.amount,
            )

    async def _execute_fill(
        self,
        order: Order,
        fill_price: Decimal,
        fill_qty: Decimal,
        tick: Tick,
    ) -> OrderResult:
        """Execute an order fill."""
        # Apply slippage
        slippage = self._slippage.calculate(order, tick, fill_price)
        final_price = fill_price + slippage

        # Calculate fee
        fee_rate = self.taker_fee if order.order_type == OrderType.MARKET else self.maker_fee
        fee = fill_qty * final_price * fee_rate

        quote = order.symbol.split("/")[1]
        base = order.symbol.split("/")[0]

        # Update balances
        if order.side == OrderSide.BUY:
            # Deduct quote, add base
            cost = fill_qty * final_price + fee
            quote_bal = self._balances[quote]
            self._balances[quote] = Balance(
                currency=quote,
                total=quote_bal.total - cost,
                available=quote_bal.available,  # Was already locked
                locked=quote_bal.locked - (fill_qty * (order.price or final_price)),
            )

            # Add base currency
            base_bal = self._balances.get(base, Balance(base, Decimal("0"), Decimal("0"), Decimal("0")))
            self._balances[base] = Balance(
                currency=base,
                total=base_bal.total + fill_qty,
                available=base_bal.available + fill_qty,
                locked=base_bal.locked,
            )
        else:
            # Deduct base, add quote
            proceeds = fill_qty * final_price - fee
            base_bal = self._balances[base]
            self._balances[base] = Balance(
                currency=base,
                total=base_bal.total - fill_qty,
                available=base_bal.available,  # Was already locked
                locked=base_bal.locked - fill_qty,
            )

            quote_bal = self._balances.get(quote, Balance(quote, Decimal("0"), Decimal("0"), Decimal("0")))
            self._balances[quote] = Balance(
                currency=quote,
                total=quote_bal.total + proceeds,
                available=quote_bal.available + proceeds,
                locked=quote_bal.locked,
            )

        # Update position
        self._update_position(order, fill_qty, final_price)

        # Create result
        result = OrderResult(
            order_id=order.id or f"backtest-{self._next_order_id}",
            symbol=order.symbol,
            status=OrderStatus.FILLED,
            side=order.side,
            order_type=order.order_type,
            amount=order.amount,
            filled_amount=fill_qty,
            remaining_amount=order.amount - fill_qty,
            average_price=final_price,
            fee=fee,
            fee_currency=quote,
            timestamp_ns=self.clock.timestamp_ns(),
            client_order_id=order.client_order_id,
            price=order.price,
        )

        self._open_orders[result.order_id] = result
        self._order_history.append(result)
        await self._order_updates.put(result)

        return result

    def _update_position(
        self,
        order: Order,
        fill_qty: Decimal,
        fill_price: Decimal,
    ) -> None:
        """Update position after fill."""
        symbol = order.symbol
        current = self._positions.get(symbol)

        if order.side == OrderSide.BUY:
            if current is None or current.side == PositionSide.FLAT:
                # Open new long
                self._positions[symbol] = Position(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    amount=fill_qty,
                    entry_price=fill_price,
                    current_price=fill_price,
                    unrealized_pnl=Decimal("0"),
                    realized_pnl=Decimal("0"),
                )
            elif current.side == PositionSide.LONG:
                # Add to long
                new_amount = current.amount + fill_qty
                new_entry = (current.entry_price * current.amount + fill_price * fill_qty) / new_amount
                self._positions[symbol] = Position(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    amount=new_amount,
                    entry_price=new_entry,
                    current_price=fill_price,
                    unrealized_pnl=Decimal("0"),
                    realized_pnl=current.realized_pnl,
                )
            else:
                # Close/reduce short
                if fill_qty >= current.amount:
                    # Close short
                    pnl = (current.entry_price - fill_price) * current.amount
                    remaining = fill_qty - current.amount
                    if remaining > 0:
                        # Flip to long
                        self._positions[symbol] = Position(
                            symbol=symbol,
                            side=PositionSide.LONG,
                            amount=remaining,
                            entry_price=fill_price,
                            current_price=fill_price,
                            unrealized_pnl=Decimal("0"),
                            realized_pnl=current.realized_pnl + pnl,
                        )
                    else:
                        self._positions[symbol] = Position(
                            symbol=symbol,
                            side=PositionSide.FLAT,
                            amount=Decimal("0"),
                            entry_price=Decimal("0"),
                            current_price=fill_price,
                            unrealized_pnl=Decimal("0"),
                            realized_pnl=current.realized_pnl + pnl,
                        )
                else:
                    # Reduce short
                    pnl = (current.entry_price - fill_price) * fill_qty
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        side=PositionSide.SHORT,
                        amount=current.amount - fill_qty,
                        entry_price=current.entry_price,
                        current_price=fill_price,
                        unrealized_pnl=Decimal("0"),
                        realized_pnl=current.realized_pnl + pnl,
                    )
        else:
            # SELL logic (mirror of BUY)
            if current is None or current.side == PositionSide.FLAT:
                # Open new short
                self._positions[symbol] = Position(
                    symbol=symbol,
                    side=PositionSide.SHORT,
                    amount=fill_qty,
                    entry_price=fill_price,
                    current_price=fill_price,
                    unrealized_pnl=Decimal("0"),
                    realized_pnl=Decimal("0"),
                )
            elif current.side == PositionSide.SHORT:
                # Add to short
                new_amount = current.amount + fill_qty
                new_entry = (current.entry_price * current.amount + fill_price * fill_qty) / new_amount
                self._positions[symbol] = Position(
                    symbol=symbol,
                    side=PositionSide.SHORT,
                    amount=new_amount,
                    entry_price=new_entry,
                    current_price=fill_price,
                    unrealized_pnl=Decimal("0"),
                    realized_pnl=current.realized_pnl,
                )
            else:
                # Close/reduce long
                if fill_qty >= current.amount:
                    pnl = (fill_price - current.entry_price) * current.amount
                    remaining = fill_qty - current.amount
                    if remaining > 0:
                        self._positions[symbol] = Position(
                            symbol=symbol,
                            side=PositionSide.SHORT,
                            amount=remaining,
                            entry_price=fill_price,
                            current_price=fill_price,
                            unrealized_pnl=Decimal("0"),
                            realized_pnl=current.realized_pnl + pnl,
                        )
                    else:
                        self._positions[symbol] = Position(
                            symbol=symbol,
                            side=PositionSide.FLAT,
                            amount=Decimal("0"),
                            entry_price=Decimal("0"),
                            current_price=fill_price,
                            unrealized_pnl=Decimal("0"),
                            realized_pnl=current.realized_pnl + pnl,
                        )
                else:
                    pnl = (fill_price - current.entry_price) * fill_qty
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        side=PositionSide.LONG,
                        amount=current.amount - fill_qty,
                        entry_price=current.entry_price,
                        current_price=fill_price,
                        unrealized_pnl=Decimal("0"),
                        realized_pnl=current.realized_pnl + pnl,
                    )

    # -------------------------------------------------------------------------
    # Backtest Control
    # -------------------------------------------------------------------------

    def reset(self, initial_balance: dict[str, Decimal] | None = None) -> None:
        """Reset client state for a new backtest run."""
        self._pending_orders.clear()
        self._order_queue_position.clear()
        self._open_orders.clear()
        self._order_history.clear()
        self._positions.clear()
        self._latest_ticks.clear()
        self._next_order_id = 1

        # Reset balances
        if initial_balance:
            self._balances.clear()
            for currency, amount in initial_balance.items():
                self._balances[currency] = Balance(
                    currency=currency,
                    total=amount,
                    available=amount,
                    locked=Decimal("0"),
                )

    def get_equity(self) -> Decimal:
        """Calculate total equity (balances + unrealized P&L)."""
        equity = Decimal("0")

        # Sum all balance totals (simplified - should convert to base currency)
        for balance in self._balances.values():
            equity += balance.total

        # Add unrealized P&L from positions
        for position in self._positions.values():
            equity += position.unrealized_pnl

        return equity

    def get_realized_pnl(self) -> Decimal:
        """Get total realized P&L from closed positions."""
        total = Decimal("0")
        for position in self._positions.values():
            total += position.realized_pnl
        return total
