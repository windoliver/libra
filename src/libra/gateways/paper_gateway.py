"""
Paper Trading Gateway: Simulated trading for testing and development.

Features:
- Realistic order simulation with fill modeling
- Configurable slippage (fixed, volume-based, or stochastic)
- Position tracking with real-time P&L
- Market and limit order support
- Queue position simulation for limit orders
- Partial fill support

Slippage Models:
- NONE: No slippage (ideal fills)
- FIXED: Fixed basis points per trade
- VOLUME: Volume-weighted quadratic model (Zipline-style)
- STOCHASTIC: Random slippage within bounds

Usage:
    config = {
        "initial_balance": {"USDT": 10000, "BTC": 0.1},
        "slippage_model": "volume",
        "slippage_bps": 5,  # 5 basis points
    }

    async with PaperGateway(config) as gateway:
        # Connect to price feed (optional)
        gateway.set_price_feed(ccxt_gateway)

        # Place orders
        order = Order(...)
        result = await gateway.submit_order(order)

        # Check positions
        positions = await gateway.get_positions()
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from libra.gateways.protocol import (
    Balance,
    BaseGateway,
    GatewayCapabilities,
    GatewayError,
    InsufficientFundsError,
    Order,
    OrderBook,
    OrderError,
    OrderNotFoundError,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PAPER_GATEWAY_CAPABILITIES,
    Position,
    PositionSide,
    Tick,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


logger = logging.getLogger(__name__)


# =============================================================================
# Slippage Models
# =============================================================================


class SlippageModel(str, Enum):
    """Available slippage models."""

    NONE = "none"  # No slippage (ideal fills)
    FIXED = "fixed"  # Fixed basis points
    VOLUME = "volume"  # Volume-weighted quadratic
    STOCHASTIC = "stochastic"  # Random within bounds


@dataclass
class SlippageConfig:
    """Configuration for slippage simulation."""

    model: SlippageModel = SlippageModel.FIXED
    fixed_bps: Decimal = Decimal("5")  # 5 basis points for FIXED model
    volume_impact: Decimal = Decimal("0.1")  # Price impact factor for VOLUME
    volume_limit: Decimal = Decimal("0.025")  # Max 2.5% of volume per bar
    stochastic_min_bps: Decimal = Decimal("0")  # Min slippage for STOCHASTIC
    stochastic_max_bps: Decimal = Decimal("10")  # Max slippage for STOCHASTIC


# =============================================================================
# Internal State
# =============================================================================


@dataclass
class OpenOrder:
    """Internal representation of an open order."""

    order: Order
    order_id: str
    status: OrderStatus
    filled_amount: Decimal
    average_price: Decimal | None
    created_ns: int
    trades: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PositionState:
    """Internal position state for a symbol."""

    symbol: str
    amount: Decimal = Decimal("0")  # Positive = long, negative = short
    entry_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    @property
    def side(self) -> PositionSide:
        if self.amount > 0:
            return PositionSide.LONG
        if self.amount < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT


# =============================================================================
# Paper Trading Gateway
# =============================================================================


class PaperGateway(BaseGateway):
    """
    Paper trading gateway for simulated order execution.

    Simulates realistic order fills with configurable slippage,
    tracks positions and P&L, and supports market/limit orders.

    Configuration:
        config = {
            "initial_balance": {"USDT": 10000},
            "slippage_model": "volume",  # none, fixed, volume, stochastic
            "slippage_bps": 5,  # For fixed model
            "maker_fee_bps": 1,  # 0.01% maker fee
            "taker_fee_bps": 5,  # 0.05% taker fee
        }

    Example:
        gateway = PaperGateway(config)
        await gateway.connect()

        # Provide price feed
        gateway.update_price("BTC/USDT", Decimal("50000"), Decimal("50001"))

        # Place order
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )
        result = await gateway.submit_order(order)
        print(f"Filled at {result.average_price}")

        # Check P&L
        positions = await gateway.get_positions()
        for pos in positions:
            print(f"{pos.symbol}: {pos.unrealized_pnl}")
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize paper trading gateway.

        Args:
            config: Configuration dict with:
                - initial_balance: Dict of currency -> amount
                - slippage_model: "none", "fixed", "volume", "stochastic"
                - slippage_bps: Basis points for fixed model
                - maker_fee_bps: Maker fee in basis points
                - taker_fee_bps: Taker fee in basis points
            **kwargs: Additional arguments passed to BaseGateway
        """
        super().__init__(name="paper", config=config, **kwargs)

        # Parse config
        cfg = config or {}
        initial_balance = cfg.get("initial_balance", {"USDT": Decimal("10000")})

        # Initialize balances
        self._balances: dict[str, Decimal] = {}
        self._locked: dict[str, Decimal] = defaultdict(Decimal)

        for currency, amount in initial_balance.items():
            self._balances[currency] = Decimal(str(amount))

        # Slippage configuration
        slippage_model = SlippageModel(cfg.get("slippage_model", "fixed"))
        self._slippage = SlippageConfig(
            model=slippage_model,
            fixed_bps=Decimal(str(cfg.get("slippage_bps", 5))),
        )

        # Fee configuration (in basis points)
        self._maker_fee_bps = Decimal(str(cfg.get("maker_fee_bps", 1)))  # 0.01%
        self._taker_fee_bps = Decimal(str(cfg.get("taker_fee_bps", 5)))  # 0.05%

        # State
        self._orders: dict[str, OpenOrder] = {}  # order_id -> OpenOrder
        self._positions: dict[str, PositionState] = {}  # symbol -> PositionState
        self._prices: dict[str, tuple[Decimal, Decimal]] = {}  # symbol -> (bid, ask)
        self._tick_queue: asyncio.Queue[Tick] = asyncio.Queue(maxsize=1000)
        self._order_counter = 0
        self._dropped_ticks = 0  # Counter for backpressure monitoring (Issue #79)

        # Price feed gateway (optional)
        self._price_feed: BaseGateway | None = None

    # -------------------------------------------------------------------------
    # Capabilities (Issue #24)
    # -------------------------------------------------------------------------

    @property
    def capabilities(self) -> GatewayCapabilities:
        """Get gateway capabilities."""
        return PAPER_GATEWAY_CAPABILITIES

    @property
    def dropped_ticks(self) -> int:
        """
        Get count of dropped ticks due to backpressure (Issue #79).

        Use this metric to monitor queue health and detect backpressure events.
        """
        return self._dropped_ticks

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect (no-op for paper trading)."""
        self._connected = True
        logger.info(f"{self.name}: Paper trading gateway connected")

    async def disconnect(self) -> None:
        """Disconnect (no-op for paper trading)."""
        self._connected = False
        logger.info(f"{self.name}: Paper trading gateway disconnected")

    def set_price_feed(self, gateway: BaseGateway) -> None:
        """
        Set external gateway for price feed.

        Args:
            gateway: Gateway to get prices from (e.g., CCXTGateway)
        """
        self._price_feed = gateway

    def update_price(self, symbol: str, bid: Decimal, ask: Decimal) -> None:
        """
        Update price for a symbol (for testing).

        Args:
            symbol: Trading pair
            bid: Best bid price
            ask: Best ask price
        """
        self._prices[symbol] = (bid, ask)

        # Create and queue tick
        tick = Tick(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=(bid + ask) / 2,
            timestamp_ns=time.time_ns(),
        )
        try:
            self._tick_queue.put_nowait(tick)
        except asyncio.QueueFull:
            # Queue is full - drop oldest tick and log (Issue #79)
            self._dropped_ticks += 1
            logger.warning(
                "Tick dropped due to backpressure: %s @ %s (total dropped: %d)",
                tick.symbol,
                tick.timestamp_ns,
                self._dropped_ticks,
            )
            try:
                self._tick_queue.get_nowait()  # Remove oldest
                self._tick_queue.put_nowait(tick)  # Add newest
            except asyncio.QueueEmpty:
                pass

        # Process limit orders
        asyncio.create_task(self._process_limit_orders(symbol))

    # -------------------------------------------------------------------------
    # Market Data
    # -------------------------------------------------------------------------

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols."""
        self._subscribed_symbols.update(symbols)
        logger.info(f"{self.name}: Subscribed to {symbols}")

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols."""
        self._subscribed_symbols -= set(symbols)

    async def stream_ticks(self) -> AsyncIterator[Tick]:
        """Stream tick data."""
        while self._connected:
            try:
                tick = await asyncio.wait_for(
                    self._tick_queue.get(),
                    timeout=1.0,
                )
                yield tick
            except TimeoutError:
                continue

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get simulated order book."""
        bid, ask = self._get_price(symbol)

        # Generate synthetic order book levels
        bids = []
        asks = []
        spread = ask - bid
        level_size = spread / 10

        for i in range(depth):
            bid_price = bid - (level_size * i)
            ask_price = ask + (level_size * i)
            size = Decimal("1.0") * (1 + i * Decimal("0.5"))  # Increasing size
            bids.append((bid_price, size))
            asks.append((ask_price, size))

        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp_ns=time.time_ns(),
        )

    async def get_ticker(self, symbol: str) -> Tick:
        """Get current ticker."""
        bid, ask = self._get_price(symbol)
        return Tick(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=(bid + ask) / 2,
            timestamp_ns=time.time_ns(),
        )

    def _get_price(self, symbol: str) -> tuple[Decimal, Decimal]:
        """Get bid/ask price for symbol."""
        if symbol in self._prices:
            return self._prices[symbol]
        raise GatewayError(f"No price data for {symbol}")

    # -------------------------------------------------------------------------
    # Order Execution
    # -------------------------------------------------------------------------

    async def submit_order(self, order: Order) -> OrderResult:
        """
        Submit an order for simulated execution.

        Market orders are filled immediately.
        Limit orders are queued and filled when price crosses.
        """
        if not self._connected:
            raise GatewayError("Not connected")

        # Generate order ID
        self._order_counter += 1
        order_id = f"paper-{self._order_counter}-{uuid4().hex[:8]}"

        # Create internal order
        open_order = OpenOrder(
            order=order.with_id(order_id).with_timestamp(),
            order_id=order_id,
            status=OrderStatus.PENDING,
            filled_amount=Decimal("0"),
            average_price=None,
            created_ns=time.time_ns(),
        )

        # Execute based on order type
        if order.order_type == OrderType.MARKET:
            return await self._execute_market_order(open_order)
        if order.order_type == OrderType.LIMIT:
            return await self._submit_limit_order(open_order)
        raise OrderError(f"Unsupported order type: {order.order_type}")

    async def _execute_market_order(self, open_order: OpenOrder) -> OrderResult:
        """Execute market order immediately."""
        order = open_order.order

        # Get current price
        try:
            bid, ask = self._get_price(order.symbol)
        except GatewayError:
            # No price data - reject order
            open_order.status = OrderStatus.REJECTED
            self._orders[open_order.order_id] = open_order
            return self._create_order_result(open_order)

        # Determine fill price (buy at ask, sell at bid)
        base_price = ask if order.side == OrderSide.BUY else bid

        # Apply slippage
        fill_price = self._apply_slippage(base_price, order.side, order.amount)

        # Check balance
        if order.side == OrderSide.BUY:
            # Need quote currency
            quote_currency = order.symbol.split("/")[1]
            required = order.amount * fill_price
            if self._get_available(quote_currency) < required:
                raise InsufficientFundsError(
                    f"Insufficient {quote_currency}: need {required}, have {self._get_available(quote_currency)}"
                )
        else:
            # Need base currency
            base_currency = order.symbol.split("/")[0]
            if self._get_available(base_currency) < order.amount:
                raise InsufficientFundsError(
                    f"Insufficient {base_currency}: need {order.amount}, have {self._get_available(base_currency)}"
                )

        # Execute fill
        self._execute_fill(open_order, order.amount, fill_price, is_maker=False)

        # Update status
        open_order.status = OrderStatus.FILLED
        open_order.filled_amount = order.amount
        open_order.average_price = fill_price

        self._orders[open_order.order_id] = open_order
        return self._create_order_result(open_order)

    async def _submit_limit_order(self, open_order: OpenOrder) -> OrderResult:
        """Submit limit order (queued for later execution)."""
        order = open_order.order

        if order.price is None:
            raise OrderError("Limit order requires price")

        # Lock funds
        if order.side == OrderSide.BUY:
            quote_currency = order.symbol.split("/")[1]
            required = order.amount * order.price
            if self._get_available(quote_currency) < required:
                raise InsufficientFundsError(f"Insufficient {quote_currency}")
            self._locked[quote_currency] += required
        else:
            base_currency = order.symbol.split("/")[0]
            if self._get_available(base_currency) < order.amount:
                raise InsufficientFundsError(f"Insufficient {base_currency}")
            self._locked[base_currency] += order.amount

        open_order.status = OrderStatus.OPEN
        self._orders[open_order.order_id] = open_order

        logger.debug(f"{self.name}: Limit order queued: {open_order.order_id}")

        return self._create_order_result(open_order)

    async def _process_limit_orders(self, symbol: str) -> None:
        """Process limit orders when price updates."""
        try:
            bid, ask = self._get_price(symbol)
        except GatewayError:
            return

        for order_id, open_order in list(self._orders.items()):
            if open_order.status not in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
                continue

            order = open_order.order
            if order.symbol != symbol:
                continue

            if order.order_type != OrderType.LIMIT or order.price is None:
                continue

            # Check if order can be filled
            should_fill = False
            if (order.side == OrderSide.BUY and ask <= order.price) or (
                order.side == OrderSide.SELL and bid >= order.price
            ):
                should_fill = True

            if should_fill:
                # Fill at limit price (maker order)
                remaining = order.amount - open_order.filled_amount
                self._execute_fill(open_order, remaining, order.price, is_maker=True)

                # Update status
                open_order.status = OrderStatus.FILLED
                open_order.filled_amount = order.amount
                open_order.average_price = order.price

                # Unlock remaining funds
                self._unlock_order_funds(order)

                logger.debug(f"{self.name}: Limit order filled: {order_id}")

    def _execute_fill(
        self,
        open_order: OpenOrder,
        amount: Decimal,
        price: Decimal,
        is_maker: bool,
    ) -> None:
        """Execute a fill and update balances/positions."""
        order = open_order.order
        base_currency, quote_currency = order.symbol.split("/")

        # Calculate fee
        fee_bps = self._maker_fee_bps if is_maker else self._taker_fee_bps
        fee = (amount * price * fee_bps) / Decimal("10000")

        # Update balances
        if order.side == OrderSide.BUY:
            # Pay quote, receive base
            cost = amount * price + fee
            self._balances[quote_currency] = self._balances.get(quote_currency, Decimal("0")) - cost
            self._balances[base_currency] = self._balances.get(base_currency, Decimal("0")) + amount
        else:
            # Pay base, receive quote
            proceeds = amount * price - fee
            self._balances[base_currency] = self._balances.get(base_currency, Decimal("0")) - amount
            self._balances[quote_currency] = (
                self._balances.get(quote_currency, Decimal("0")) + proceeds
            )

        # Update position
        self._update_position(order.symbol, order.side, amount, price)

        # Record trade
        open_order.trades.append(
            {
                "price": str(price),
                "amount": str(amount),
                "fee": str(fee),
                "timestamp": time.time_ns(),
                "is_maker": is_maker,
            }
        )

    def _update_position(
        self,
        symbol: str,
        side: OrderSide,
        amount: Decimal,
        price: Decimal,
    ) -> None:
        """Update position state after a fill."""
        if symbol not in self._positions:
            self._positions[symbol] = PositionState(symbol=symbol)

        pos = self._positions[symbol]

        # Calculate position change
        new_amount = pos.amount + amount if side == OrderSide.BUY else pos.amount - amount

        # Calculate new entry price (weighted average)
        if pos.amount == 0:
            # Opening new position
            pos.entry_price = price
        elif (pos.amount > 0 and side == OrderSide.BUY) or (
            pos.amount < 0 and side == OrderSide.SELL
        ):
            # Adding to position
            total_cost = (abs(pos.amount) * pos.entry_price) + (amount * price)
            pos.entry_price = total_cost / (abs(pos.amount) + amount)
        else:
            # Reducing position - realize P&L
            close_amount = min(abs(pos.amount), amount)
            if pos.amount > 0:
                pnl = close_amount * (price - pos.entry_price)
            else:
                pnl = close_amount * (pos.entry_price - price)
            pos.realized_pnl += pnl

        pos.amount = new_amount

    def _apply_slippage(
        self,
        price: Decimal,
        side: OrderSide,
        amount: Decimal,
    ) -> Decimal:
        """Apply slippage model to get fill price."""
        if self._slippage.model == SlippageModel.NONE:
            return price

        if self._slippage.model == SlippageModel.FIXED:
            slippage_pct = self._slippage.fixed_bps / Decimal("10000")
            if side == OrderSide.BUY:
                return price * (1 + slippage_pct)
            return price * (1 - slippage_pct)

        if self._slippage.model == SlippageModel.VOLUME:
            # Zipline-style volume impact: price * (1 + impact * volume_share^2)
            # Simplified without actual volume data
            volume_share = min(amount / Decimal("100"), self._slippage.volume_limit)
            impact = self._slippage.volume_impact * (volume_share**2)
            if side == OrderSide.BUY:
                return price * (1 + impact)
            return price * (1 - impact)

        # SlippageModel.STOCHASTIC
        # Random slippage within bounds
        min_bps = float(self._slippage.stochastic_min_bps)
        max_bps = float(self._slippage.stochastic_max_bps)
        slippage_bps = Decimal(str(random.uniform(min_bps, max_bps)))
        slippage_pct = slippage_bps / Decimal("10000")
        if side == OrderSide.BUY:
            return price * (1 + slippage_pct)
        return price * (1 - slippage_pct)

    def _unlock_order_funds(self, order: Order) -> None:
        """Unlock funds locked by a limit order."""
        if order.side == OrderSide.BUY and order.price:
            quote_currency = order.symbol.split("/")[1]
            locked = order.amount * order.price
            self._locked[quote_currency] = max(
                Decimal("0"),
                self._locked.get(quote_currency, Decimal("0")) - locked,
            )
        else:
            base_currency = order.symbol.split("/")[0]
            self._locked[base_currency] = max(
                Decimal("0"),
                self._locked.get(base_currency, Decimal("0")) - order.amount,
            )

    def _get_available(self, currency: str) -> Decimal:
        """Get available balance (total - locked)."""
        total = self._balances.get(currency, Decimal("0"))
        locked = self._locked.get(currency, Decimal("0"))
        return total - locked

    # -------------------------------------------------------------------------
    # Order Management
    # -------------------------------------------------------------------------

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        if order_id not in self._orders:
            raise OrderNotFoundError(f"Order {order_id} not found")

        open_order = self._orders[order_id]
        if open_order.status not in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
            return False

        # Unlock funds
        self._unlock_order_funds(open_order.order)

        open_order.status = OrderStatus.CANCELLED
        return True

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all open orders."""
        count = 0
        for order_id, open_order in list(self._orders.items()):
            if open_order.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
                if symbol is None or open_order.order.symbol == symbol:
                    await self.cancel_order(order_id, open_order.order.symbol)
                    count += 1
        return count

    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        """Get order status."""
        if order_id not in self._orders:
            raise OrderNotFoundError(f"Order {order_id} not found")
        return self._create_order_result(self._orders[order_id])

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Get all open orders."""
        results = []
        for open_order in self._orders.values():
            if open_order.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
                if symbol is None or open_order.order.symbol == symbol:
                    results.append(self._create_order_result(open_order))
        return results

    async def get_order_history(
        self, symbol: str | None = None, limit: int | None = None
    ) -> list[OrderResult]:
        """Get order history (all orders, open and closed, Issue #27)."""
        results = []
        for open_order in self._orders.values():
            if symbol is None or open_order.order.symbol == symbol:
                results.append(self._create_order_result(open_order))

        # Sort by timestamp descending
        results.sort(key=lambda r: r.timestamp_ns, reverse=True)

        if limit is not None:
            results = results[:limit]

        return results

    async def get_trades(
        self, symbol: str | None = None, limit: int | None = None
    ) -> list:
        """Get trade/fill history (Issue #27)."""
        from libra.gateways.fetcher import TradeRecord

        trades = []
        for open_order in self._orders.values():
            if symbol is not None and open_order.order.symbol != symbol:
                continue

            for trade in open_order.trades:
                trades.append(
                    TradeRecord(
                        trade_id=f"{open_order.order_id}-{trade['timestamp']}",
                        order_id=open_order.order_id,
                        symbol=open_order.order.symbol,
                        side=open_order.order.side.value,
                        amount=Decimal(trade["amount"]),
                        price=Decimal(trade["price"]),
                        cost=Decimal(trade["amount"]) * Decimal(trade["price"]),
                        fee=Decimal(trade["fee"]) if trade.get("fee") else None,
                        fee_currency=open_order.order.symbol.split("/")[1],
                        timestamp_ns=trade["timestamp"],
                        taker_or_maker="maker" if trade.get("is_maker", False) else "taker",
                    )
                )

        # Sort by timestamp descending
        trades.sort(key=lambda t: t.timestamp_ns, reverse=True)

        if limit is not None:
            trades = trades[:limit]

        return trades

    def _create_order_result(self, open_order: OpenOrder) -> OrderResult:
        """Create OrderResult from internal state."""
        order = open_order.order
        fee = sum((Decimal(t["fee"]) for t in open_order.trades), Decimal("0"))
        return OrderResult(
            order_id=open_order.order_id,
            symbol=order.symbol,
            status=open_order.status,
            side=order.side,
            order_type=order.order_type,
            amount=order.amount,
            filled_amount=open_order.filled_amount,
            remaining_amount=order.amount - open_order.filled_amount,
            average_price=open_order.average_price,
            fee=fee,
            fee_currency=order.symbol.split("/")[1],  # Quote currency
            timestamp_ns=time.time_ns(),
            created_ns=open_order.created_ns,
            client_order_id=order.client_order_id,
            price=order.price,
            trades=open_order.trades if open_order.trades else None,
        )

    # -------------------------------------------------------------------------
    # Account
    # -------------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Get all open positions with P&L."""
        positions = []
        for pos in self._positions.values():
            if pos.amount == 0:
                continue

            # Get current price for P&L calculation
            try:
                bid, ask = self._get_price(pos.symbol)
                current_price = (bid + ask) / 2
            except GatewayError:
                current_price = pos.entry_price

            # Calculate unrealized P&L
            if pos.amount > 0:
                unrealized_pnl = pos.amount * (current_price - pos.entry_price)
            else:
                unrealized_pnl = abs(pos.amount) * (pos.entry_price - current_price)

            positions.append(
                Position(
                    symbol=pos.symbol,
                    side=pos.side,
                    amount=abs(pos.amount),
                    entry_price=pos.entry_price,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=pos.realized_pnl,
                    timestamp_ns=time.time_ns(),
                )
            )

        return positions

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def get_balances(self) -> dict[str, Balance]:
        """Get all balances."""
        balances = {}
        for currency, total in self._balances.items():
            if total != 0:
                locked = self._locked.get(currency, Decimal("0"))
                balances[currency] = Balance(
                    currency=currency,
                    total=total,
                    available=total - locked,
                    locked=locked,
                )
        return balances

    async def get_balance(self, currency: str) -> Balance | None:
        """Get balance for a currency."""
        balances = await self.get_balances()
        return balances.get(currency)

    # -------------------------------------------------------------------------
    # Paper Trading Specific
    # -------------------------------------------------------------------------

    def reset(self, initial_balance: dict[str, Decimal] | None = None) -> None:
        """
        Reset paper trading state.

        Args:
            initial_balance: Optional new initial balance
        """
        if initial_balance:
            self._balances = {k: Decimal(str(v)) for k, v in initial_balance.items()}
        else:
            # Reset to config initial balance
            initial = self._config.get("initial_balance", {"USDT": Decimal("10000")})
            self._balances = {k: Decimal(str(v)) for k, v in initial.items()}

        self._locked.clear()
        self._orders.clear()
        self._positions.clear()
        self._order_counter = 0

        logger.info(f"{self.name}: State reset")

    def get_total_equity(self) -> Decimal:
        """
        Calculate total account equity.

        Returns:
            Total equity in quote currency (assumes USDT-denominated)
        """
        # Start with USDT balance
        equity = self._balances.get("USDT", Decimal("0"))

        # Add value of other currencies (simplified - assumes /USDT pairs)
        for currency, amount in self._balances.items():
            if currency == "USDT" or amount == 0:
                continue

            symbol = f"{currency}/USDT"
            try:
                bid, ask = self._get_price(symbol)
                equity += amount * (bid + ask) / 2
            except GatewayError:
                pass  # Skip if no price

        # Add unrealized P&L from positions
        for pos in self._positions.values():
            if pos.amount == 0:
                continue

            try:
                bid, ask = self._get_price(pos.symbol)
                current_price = (bid + ask) / 2
                if pos.amount > 0:
                    equity += pos.amount * (current_price - pos.entry_price)
                else:
                    equity += abs(pos.amount) * (pos.entry_price - current_price)
            except GatewayError:
                pass

        return equity

    def get_trade_history(self) -> list[dict[str, Any]]:
        """
        Get all executed trades.

        Returns:
            List of trade dicts with price, amount, fee, etc.
        """
        trades = []
        for open_order in self._orders.values():
            for trade in open_order.trades:
                trades.append(
                    {
                        "order_id": open_order.order_id,
                        "symbol": open_order.order.symbol,
                        "side": open_order.order.side.value,
                        **trade,
                    }
                )
        return sorted(trades, key=lambda t: t["timestamp"])
