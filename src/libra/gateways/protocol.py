"""
Gateway Protocol: Unified interface for all brokers/exchanges.

Defines:
- Order, OrderResult, Position, Tick data structures
- Gateway Protocol with runtime_checkable decorator
- Abstract base class for implementations

Design inspired by:
- NautilusTrader adapters
- vnpy BaseGateway
- Hummingbot connectors

Performance:
- Uses msgspec.Struct for 4x faster serialization than dataclass
- Decimal for precise monetary calculations
- Nanosecond timestamps for consistency with events.py
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import msgspec


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from libra.core.sessions import MarketSessionManager


# =============================================================================
# Enums
# =============================================================================


class OrderSide(str, Enum):
    """Order side (buy or sell)."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status lifecycle."""

    PENDING = "pending"  # Submitted, not yet acknowledged
    OPEN = "open"  # Acknowledged, waiting to fill
    FILLED = "filled"  # Completely filled
    PARTIALLY_FILLED = "partially_filled"  # Partially filled, still open
    CANCELLED = "cancelled"  # Cancelled by user or system
    REJECTED = "rejected"  # Rejected by exchange
    EXPIRED = "expired"  # Time-in-force expired


class TimeInForce(str, Enum):
    """Time-in-force for orders."""

    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date
    DAY = "DAY"  # Day order


class PositionSide(str, Enum):
    """Position side for derivatives."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"  # No position


class ContingencyType(str, Enum):
    """
    Order contingency/relationship types.

    Inspired by NautilusTrader and Interactive Brokers patterns.

    Types:
    - NONE: Standalone order
    - OTO: One-Triggers-Other (parent fill → submit children)
    - OCO: One-Cancels-Other (one fills → cancel others)
    - OUO: One-Updates-Other (partial fill → reduce other)
    """

    NONE = "none"  # Standalone order
    OTO = "oto"  # One-Triggers-Other
    OCO = "oco"  # One-Cancels-Other
    OUO = "ouo"  # One-Updates-Other


class TriggerType(str, Enum):
    """
    Trigger price type for conditional orders.

    Used with stop orders and emulated order types.
    """

    LAST_PRICE = "last_price"  # Trigger on last trade price
    BID_ASK = "bid_ask"  # Trigger on bid (sell) or ask (buy)
    MID_POINT = "mid_point"  # Trigger on mid-point
    MARK_PRICE = "mark_price"  # Trigger on mark price (derivatives)


# =============================================================================
# Gateway Capabilities (Issue #24 - Extensible Protocol Design)
# =============================================================================


class GatewayCapabilities(msgspec.Struct, frozen=True, gc=False):
    """
    Declares capabilities supported by a gateway.

    Enables feature negotiation and graceful degradation.
    Inspired by CCXT's exchange.has pattern.

    Use Cases:
    - Order validation before submission
    - Automatic order emulation for unsupported types
    - Strategy compatibility checks
    - Documentation generation

    Examples:
        # Check before submitting stop order
        if gateway.capabilities.stop_orders:
            await gateway.submit_order(stop_order)
        else:
            # Emulate with price monitoring
            await emulator.monitor(stop_order)

        # Check margin trading support
        if not gateway.capabilities.margin_trading:
            raise UnsupportedFeatureError("Margin not supported")
    """

    # Order Types
    market_orders: bool = True
    limit_orders: bool = True
    stop_orders: bool = False
    stop_limit_orders: bool = False
    trailing_stop_orders: bool = False
    bracket_orders: bool = False
    oco_orders: bool = False
    oto_orders: bool = False

    # Time-in-Force
    tif_gtc: bool = True  # Good Till Cancelled
    tif_ioc: bool = False  # Immediate Or Cancel
    tif_fok: bool = False  # Fill Or Kill
    tif_gtd: bool = False  # Good Till Date
    tif_day: bool = False  # Day order

    # Market Data
    streaming_ticks: bool = True
    streaming_orderbook: bool = False
    streaming_trades: bool = False
    historical_bars: bool = True
    historical_trades: bool = False

    # Trading Features
    margin_trading: bool = False
    futures_trading: bool = False
    options_trading: bool = False
    short_selling: bool = False
    reduce_only: bool = False
    post_only: bool = False

    # Position Management
    position_tracking: bool = True
    hedge_mode: bool = False  # Separate long/short positions

    # Rate Limits
    max_orders_per_second: int = 10
    max_requests_per_minute: int = 1200

    # Order Emulation (gateway can emulate missing types)
    emulated_stop_orders: bool = False
    emulated_trailing_stops: bool = False
    emulated_bracket_orders: bool = False

    @property
    def supported_order_types(self) -> frozenset[OrderType]:
        """Get set of supported order types for fast lookup."""
        types: set[OrderType] = set()
        if self.market_orders:
            types.add(OrderType.MARKET)
        if self.limit_orders:
            types.add(OrderType.LIMIT)
        if self.stop_orders:
            types.add(OrderType.STOP)
        if self.stop_limit_orders:
            types.add(OrderType.STOP_LIMIT)
        return frozenset(types)

    @property
    def supported_time_in_force(self) -> frozenset[TimeInForce]:
        """Get set of supported time-in-force options."""
        tifs: set[TimeInForce] = set()
        if self.tif_gtc:
            tifs.add(TimeInForce.GTC)
        if self.tif_ioc:
            tifs.add(TimeInForce.IOC)
        if self.tif_fok:
            tifs.add(TimeInForce.FOK)
        if self.tif_gtd:
            tifs.add(TimeInForce.GTD)
        if self.tif_day:
            tifs.add(TimeInForce.DAY)
        return frozenset(tifs)

    def supports_order_type(self, order_type: OrderType) -> bool:
        """Check if order type is supported."""
        return order_type in self.supported_order_types

    def supports_time_in_force(self, tif: TimeInForce) -> bool:
        """Check if time-in-force is supported."""
        return tif in self.supported_time_in_force


# Default capabilities for common gateway types
PAPER_GATEWAY_CAPABILITIES = GatewayCapabilities(
    market_orders=True,
    limit_orders=True,
    stop_orders=True,
    stop_limit_orders=True,
    trailing_stop_orders=False,
    bracket_orders=False,
    streaming_ticks=True,
    streaming_orderbook=True,
    position_tracking=True,
    margin_trading=True,
    short_selling=True,
)

CCXT_DEFAULT_CAPABILITIES = GatewayCapabilities(
    market_orders=True,
    limit_orders=True,
    stop_orders=False,  # Varies by exchange
    stop_limit_orders=False,  # Varies by exchange
    streaming_ticks=True,
    streaming_orderbook=True,
    historical_bars=True,
    # Emulation available for missing features
    emulated_stop_orders=True,
    emulated_trailing_stops=True,
)


# =============================================================================
# Data Structures (msgspec.Struct for performance)
# =============================================================================


class Order(msgspec.Struct, frozen=True, gc=False):
    """
    Universal order representation.

    Immutable, hashable, and optimized for fast serialization.
    Compatible with any exchange via Gateway adapters.

    Examples:
        # Market buy order
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )

        # Limit sell order with stop loss
        order = Order(
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("2000.00"),
            stop_price=Decimal("1950.00"),
            time_in_force=TimeInForce.GTC,
        )

        # Order with TWAP execution algorithm
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100.0"),
            exec_algorithm="twap",
            exec_algorithm_params={"horizon_secs": 300, "interval_secs": 30},
        )
    """

    # Required fields
    symbol: str  # Trading pair (e.g., "BTC/USDT")
    side: OrderSide
    order_type: OrderType
    amount: Decimal  # Order quantity

    # Optional fields with defaults
    id: str | None = None  # Exchange-assigned order ID
    client_order_id: str | None = None  # Client-assigned ID for tracking
    price: Decimal | None = None  # Limit price (required for LIMIT orders)
    stop_price: Decimal | None = None  # Trigger price for STOP orders
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False  # Only reduce position (derivatives)
    post_only: bool = False  # Only maker orders (no taker)
    leverage: int | None = None  # Leverage for derivatives
    timestamp_ns: int | None = None  # Creation time (nanoseconds)
    extended_hours: bool = False  # Allow pre-market/after-hours (Issue #62)

    # Execution algorithm fields (Issue #36)
    exec_algorithm: str | None = None  # Algorithm name: "twap", "vwap", "iceberg", "pov"
    exec_algorithm_params: dict[str, Any] | None = None  # Algorithm-specific config
    parent_order_id: str | None = None  # For child orders spawned by algorithms

    # Contingency order fields (Issue #24)
    contingency_type: ContingencyType = ContingencyType.NONE
    linked_order_ids: tuple[str, ...] | None = None  # IDs of related orders (OCO, OTO)

    # Trailing stop fields (Issue #24)
    trailing_offset: Decimal | None = None  # Trailing distance
    trailing_offset_type: str | None = None  # "price" or "percent"
    trigger_type: TriggerType | None = None  # What price triggers the order

    def with_id(self, order_id: str) -> Order:
        """Return new Order with exchange-assigned ID."""
        return Order(
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            amount=self.amount,
            id=order_id,
            client_order_id=self.client_order_id,
            price=self.price,
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
            reduce_only=self.reduce_only,
            post_only=self.post_only,
            leverage=self.leverage,
            timestamp_ns=self.timestamp_ns,
            extended_hours=self.extended_hours,
            exec_algorithm=self.exec_algorithm,
            exec_algorithm_params=self.exec_algorithm_params,
            parent_order_id=self.parent_order_id,
            contingency_type=self.contingency_type,
            linked_order_ids=self.linked_order_ids,
            trailing_offset=self.trailing_offset,
            trailing_offset_type=self.trailing_offset_type,
            trigger_type=self.trigger_type,
        )

    def with_timestamp(self) -> Order:
        """Return new Order with current timestamp."""
        return Order(
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            amount=self.amount,
            id=self.id,
            client_order_id=self.client_order_id,
            price=self.price,
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
            reduce_only=self.reduce_only,
            post_only=self.post_only,
            leverage=self.leverage,
            timestamp_ns=time.time_ns(),
            extended_hours=self.extended_hours,
            exec_algorithm=self.exec_algorithm,
            exec_algorithm_params=self.exec_algorithm_params,
            parent_order_id=self.parent_order_id,
            contingency_type=self.contingency_type,
            linked_order_ids=self.linked_order_ids,
            trailing_offset=self.trailing_offset,
            trailing_offset_type=self.trailing_offset_type,
            trigger_type=self.trigger_type,
        )

    def with_exec_algorithm(
        self,
        algorithm: str,
        params: dict[str, Any] | None = None,
    ) -> Order:
        """Return new Order with execution algorithm specified."""
        return Order(
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            amount=self.amount,
            id=self.id,
            client_order_id=self.client_order_id,
            price=self.price,
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
            reduce_only=self.reduce_only,
            post_only=self.post_only,
            leverage=self.leverage,
            timestamp_ns=self.timestamp_ns,
            extended_hours=self.extended_hours,
            exec_algorithm=algorithm,
            exec_algorithm_params=params,
            parent_order_id=self.parent_order_id,
            contingency_type=self.contingency_type,
            linked_order_ids=self.linked_order_ids,
            trailing_offset=self.trailing_offset,
            trailing_offset_type=self.trailing_offset_type,
            trigger_type=self.trigger_type,
        )

    def with_contingency(
        self,
        contingency_type: ContingencyType,
        linked_order_ids: tuple[str, ...] | None = None,
    ) -> Order:
        """Return new Order with contingency relationship specified."""
        return Order(
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            amount=self.amount,
            id=self.id,
            client_order_id=self.client_order_id,
            price=self.price,
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
            reduce_only=self.reduce_only,
            post_only=self.post_only,
            leverage=self.leverage,
            timestamp_ns=self.timestamp_ns,
            extended_hours=self.extended_hours,
            exec_algorithm=self.exec_algorithm,
            exec_algorithm_params=self.exec_algorithm_params,
            parent_order_id=self.parent_order_id,
            contingency_type=contingency_type,
            linked_order_ids=linked_order_ids,
            trailing_offset=self.trailing_offset,
            trailing_offset_type=self.trailing_offset_type,
            trigger_type=self.trigger_type,
        )

    def with_trailing_stop(
        self,
        offset: Decimal,
        offset_type: str = "price",
        trigger: TriggerType = TriggerType.LAST_PRICE,
    ) -> Order:
        """Return new Order configured as trailing stop."""
        return Order(
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            amount=self.amount,
            id=self.id,
            client_order_id=self.client_order_id,
            price=self.price,
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
            reduce_only=self.reduce_only,
            post_only=self.post_only,
            leverage=self.leverage,
            timestamp_ns=self.timestamp_ns,
            extended_hours=self.extended_hours,
            exec_algorithm=self.exec_algorithm,
            exec_algorithm_params=self.exec_algorithm_params,
            parent_order_id=self.parent_order_id,
            contingency_type=self.contingency_type,
            linked_order_ids=self.linked_order_ids,
            trailing_offset=offset,
            trailing_offset_type=offset_type,
            trigger_type=trigger,
        )

    @property
    def is_algo_order(self) -> bool:
        """Check if this order should use an execution algorithm."""
        return self.exec_algorithm is not None

    @property
    def is_child_order(self) -> bool:
        """Check if this is a child order spawned by an algorithm."""
        return self.parent_order_id is not None

    @property
    def is_contingent(self) -> bool:
        """Check if this order has contingency relationships."""
        return self.contingency_type != ContingencyType.NONE

    @property
    def is_trailing_stop(self) -> bool:
        """Check if this is a trailing stop order."""
        return self.trailing_offset is not None


class OrderResult(msgspec.Struct, frozen=True, gc=False):
    """
    Result of order submission or query.

    Returned by Gateway.order() and Gateway.get_order().

    Examples:
        # Fully filled market order
        result = OrderResult(
            order_id="12345",
            client_order_id="my-order-1",
            symbol="BTC/USDT",
            status=OrderStatus.FILLED,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
            filled_amount=Decimal("0.1"),
            remaining_amount=Decimal("0"),
            average_price=Decimal("50000.00"),
            fee=Decimal("0.00005"),
            fee_currency="BTC",
            timestamp_ns=time.time_ns(),
        )
    """

    # Identification
    order_id: str
    symbol: str
    status: OrderStatus

    # Order details
    side: OrderSide
    order_type: OrderType
    amount: Decimal  # Original order amount

    # Fill information
    filled_amount: Decimal  # Amount filled so far
    remaining_amount: Decimal  # Amount still open
    average_price: Decimal | None  # Volume-weighted average fill price

    # Fees
    fee: Decimal  # Total fees paid
    fee_currency: str  # Currency of fees

    # Timestamps (nanoseconds)
    timestamp_ns: int  # Last update time
    created_ns: int | None = None  # Creation time

    # Optional
    client_order_id: str | None = None
    price: Decimal | None = None  # Limit price if applicable
    stop_price: Decimal | None = None
    trades: list[dict[str, Any]] | None = None  # Individual fills

    @property
    def is_open(self) -> bool:
        """Check if order is still open."""
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING)

    @property
    def is_closed(self) -> bool:
        """Check if order is closed (filled, cancelled, etc.)."""
        return not self.is_open

    @property
    def fill_percent(self) -> Decimal:
        """Percentage of order filled (0-100)."""
        if self.amount == 0:
            return Decimal("0")
        return (self.filled_amount / self.amount) * 100


class Position(msgspec.Struct, frozen=True, gc=False):
    """
    Current position in an instrument.

    Represents an open position with P&L tracking.

    Examples:
        # Long BTC position
        position = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=Decimal("0.5"),
            entry_price=Decimal("48000.00"),
            current_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("1000.00"),
            realized_pnl=Decimal("0"),
            leverage=1,
        )
    """

    symbol: str
    side: PositionSide
    amount: Decimal  # Position size (always positive)
    entry_price: Decimal  # Average entry price
    current_price: Decimal  # Current mark price
    unrealized_pnl: Decimal  # Unrealized profit/loss
    realized_pnl: Decimal  # Realized profit/loss

    # Optional fields
    leverage: int = 1  # Leverage multiplier
    liquidation_price: Decimal | None = None  # Liquidation price (derivatives)
    margin: Decimal | None = None  # Margin used
    margin_type: str | None = None  # "cross" or "isolated"
    timestamp_ns: int | None = None  # Last update time

    @property
    def notional_value(self) -> Decimal:
        """Position notional value (amount * current_price)."""
        return self.amount * self.current_price

    @property
    def total_pnl(self) -> Decimal:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def pnl_percent(self) -> Decimal:
        """P&L as percentage of entry value."""
        entry_value = self.amount * self.entry_price
        if entry_value == 0:
            return Decimal("0")
        return (self.unrealized_pnl / entry_value) * 100


class Tick(msgspec.Struct, frozen=True, gc=False):
    """
    Market tick data (quote/trade).

    Real-time price update from exchange.

    Examples:
        # BTC/USDT tick
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("49999.00"),
            ask=Decimal("50001.00"),
            last=Decimal("50000.00"),
            bid_size=Decimal("1.5"),
            ask_size=Decimal("2.0"),
            volume_24h=Decimal("15000.0"),
            timestamp_ns=time.time_ns(),
        )
    """

    symbol: str
    bid: Decimal  # Best bid price
    ask: Decimal  # Best ask price
    last: Decimal  # Last trade price
    timestamp_ns: int  # Nanoseconds since epoch

    # Optional fields
    bid_size: Decimal | None = None  # Size at best bid
    ask_size: Decimal | None = None  # Size at best ask
    last_size: Decimal | None = None  # Size of last trade
    volume_24h: Decimal | None = None  # 24h volume
    high_24h: Decimal | None = None  # 24h high
    low_24h: Decimal | None = None  # 24h low
    open_24h: Decimal | None = None  # 24h open
    change_24h: Decimal | None = None  # 24h change percent

    @property
    def mid(self) -> Decimal:
        """Mid price (average of bid and ask)."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> Decimal:
        """Spread in basis points."""
        if self.mid == 0:
            return Decimal("0")
        return (self.spread / self.mid) * 10000

    @property
    def timestamp_sec(self) -> float:
        """Timestamp in seconds (float) for compatibility."""
        return self.timestamp_ns / 1_000_000_000


class OrderBook(msgspec.Struct, frozen=True, gc=False):
    """
    Order book snapshot.

    Contains bids and asks with prices and sizes.

    Examples:
        orderbook = OrderBook(
            symbol="BTC/USDT",
            bids=[(Decimal("49999"), Decimal("1.5")), (Decimal("49998"), Decimal("2.0"))],
            asks=[(Decimal("50001"), Decimal("1.0")), (Decimal("50002"), Decimal("3.0"))],
            timestamp_ns=time.time_ns(),
        )
    """

    symbol: str
    bids: list[tuple[Decimal, Decimal]]  # [(price, size), ...] sorted desc
    asks: list[tuple[Decimal, Decimal]]  # [(price, size), ...] sorted asc
    timestamp_ns: int

    @property
    def best_bid(self) -> Decimal | None:
        """Best bid price."""
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        """Best ask price."""
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        """Mid price."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Decimal | None:
        """Bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


class Balance(msgspec.Struct, frozen=True, gc=False):
    """
    Account balance for a single currency.

    Examples:
        balance = Balance(
            currency="USDT",
            total=Decimal("10000.00"),
            available=Decimal("8000.00"),
            locked=Decimal("2000.00"),
        )
    """

    currency: str
    total: Decimal  # Total balance
    available: Decimal  # Available for trading
    locked: Decimal  # Locked in orders/positions

    @property
    def used_percent(self) -> Decimal:
        """Percentage of balance locked."""
        if self.total == 0:
            return Decimal("0")
        return (self.locked / self.total) * 100


# =============================================================================
# Gateway Protocol
# =============================================================================


@runtime_checkable
class Gateway(Protocol):
    """
    Unified gateway interface for all brokers/exchanges.

    This protocol defines the contract that all gateway implementations
    must follow. Use `isinstance(obj, Gateway)` for runtime checking.

    Note: @runtime_checkable only checks method/attribute existence,
    not signatures. Use static type checking for full validation.

    Implementations:
    - CCXTGateway: 100+ exchanges via CCXT
    - PaperGateway: Simulated paper trading

    Examples:
        async with CCXTGateway("binance", config) as gateway:
            # Subscribe to market data
            await gateway.subscribe(["BTC/USDT", "ETH/USDT"])

            # Stream ticks
            async for tick in gateway.stream_ticks():
                print(f"{tick.symbol}: {tick.last}")

            # Place order
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("0.01"),
            )
            result = await gateway.submit_order(order)
            print(f"Order {result.order_id}: {result.status}")
    """

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Gateway identifier (e.g., "binance", "paper").

        Used for logging, event sources, and configuration.
        """
        ...

    @property
    def is_connected(self) -> bool:
        """
        Check if gateway is connected and ready.

        Returns:
            True if connected and authenticated, False otherwise.
        """
        ...

    @property
    def capabilities(self) -> GatewayCapabilities:
        """
        Gateway capabilities for feature negotiation.

        Returns capabilities this gateway supports, enabling:
        - Order type validation before submission
        - Automatic order emulation for unsupported types
        - Strategy compatibility checks

        Returns:
            GatewayCapabilities describing supported features.

        Example:
            if gateway.capabilities.stop_orders:
                await gateway.submit_order(stop_order)
            elif gateway.capabilities.emulated_stop_orders:
                # Use emulation
                ...
        """
        ...

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect to the exchange/broker.

        Establishes WebSocket connections, authenticates,
        and prepares for trading.

        Raises:
            ConnectionError: If connection fails.
            AuthenticationError: If authentication fails.
        """
        ...

    async def disconnect(self) -> None:
        """
        Disconnect from the exchange/broker.

        Closes all connections gracefully.
        Should be safe to call multiple times.
        """
        ...

    # -------------------------------------------------------------------------
    # Market Data
    # -------------------------------------------------------------------------

    async def subscribe(self, symbols: list[str]) -> None:
        """
        Subscribe to market data for symbols.

        Args:
            symbols: List of trading pairs (e.g., ["BTC/USDT", "ETH/USDT"])

        Raises:
            ValueError: If symbol format is invalid.
            ConnectionError: If not connected.
        """
        ...

    async def unsubscribe(self, symbols: list[str]) -> None:
        """
        Unsubscribe from market data.

        Args:
            symbols: List of trading pairs to unsubscribe from.
        """
        ...

    def stream_ticks(self) -> AsyncIterator[Tick]:
        """
        Stream real-time tick data.

        Yields ticks for all subscribed symbols.
        Use `subscribe()` first to specify symbols.

        Yields:
            Tick objects as they arrive.

        Example:
            await gateway.subscribe(["BTC/USDT"])
            async for tick in gateway.stream_ticks():
                print(f"{tick.symbol}: {tick.last}")
        """
        ...

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """
        Get current order book snapshot.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            depth: Number of levels (default 20)

        Returns:
            OrderBook with bids and asks.

        Raises:
            ValueError: If symbol is invalid.
        """
        ...

    async def get_ticker(self, symbol: str) -> Tick:
        """
        Get current ticker (quote) for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Current Tick data.
        """
        ...

    # -------------------------------------------------------------------------
    # Trading
    # -------------------------------------------------------------------------

    async def submit_order(self, order: Order) -> OrderResult:
        """
        Submit an order to the exchange.

        Args:
            order: Order to submit

        Returns:
            OrderResult with status and fill info.

        Raises:
            OrderError: If order is rejected.
            InsufficientFundsError: If not enough balance.

        Example:
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                amount=Decimal("0.1"),
                price=Decimal("50000"),
            )
            result = await gateway.submit_order(order)
        """
        ...

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair (required by some exchanges)

        Returns:
            True if cancelled, False if already closed.

        Raises:
            OrderNotFoundError: If order doesn't exist.
        """
        ...

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Optional symbol to filter. None = all symbols.

        Returns:
            Number of orders cancelled.
        """
        ...

    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        """
        Get current status of an order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair

        Returns:
            Current OrderResult.

        Raises:
            OrderNotFoundError: If order doesn't exist.
        """
        ...

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol to filter.

        Returns:
            List of open OrderResults.
        """
        ...

    async def get_order_history(
        self, symbol: str | None = None, limit: int | None = None
    ) -> list[OrderResult]:
        """
        Get order history (open and closed orders).

        Uses TET pipeline with CCXTOrderFetcher (Issue #27).

        Args:
            symbol: Optional symbol to filter.
            limit: Maximum number of orders to return.

        Returns:
            List of OrderResults sorted by timestamp descending.
        """
        ...

    async def get_trades(
        self, symbol: str | None = None, limit: int | None = None
    ) -> list:
        """
        Get trade/fill history.

        Uses TET pipeline with CCXTTradeFetcher (Issue #27).

        Args:
            symbol: Optional symbol to filter.
            limit: Maximum number of trades to return.

        Returns:
            List of TradeRecord objects sorted by timestamp descending.
        """
        ...

    # -------------------------------------------------------------------------
    # Account
    # -------------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """
        Get all open positions.

        Returns:
            List of current positions with P&L.
        """
        ...

    async def get_position(self, symbol: str) -> Position | None:
        """
        Get position for a specific symbol.

        Args:
            symbol: Trading pair

        Returns:
            Position if exists, None otherwise.
        """
        ...

    async def get_balances(self) -> dict[str, Balance]:
        """
        Get account balances.

        Returns:
            Dict mapping currency to Balance.
            Example: {"USDT": Balance(...), "BTC": Balance(...)}
        """
        ...

    async def get_balance(self, currency: str) -> Balance | None:
        """
        Get balance for a specific currency.

        Args:
            currency: Currency code (e.g., "USDT", "BTC")

        Returns:
            Balance if exists, None otherwise.
        """
        ...


# =============================================================================
# Abstract Base Class
# =============================================================================


class BaseGateway(ABC):
    """
    Abstract base class for Gateway implementations.

    Provides common functionality:
    - Configuration management
    - Connection state tracking
    - Event publishing to MessageBus
    - Logging

    Subclasses must implement all abstract methods from Gateway protocol.
    """

    def __init__(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        session_manager: MarketSessionManager | None = None,
        max_concurrent_requests: int = 10,
    ) -> None:
        """
        Initialize gateway.

        Args:
            name: Gateway identifier
            config: Configuration dict (API keys, etc.)
            session_manager: Market session manager for stock/options gateways
            max_concurrent_requests: Maximum concurrent API requests (Issue #80)
        """
        self._name = name
        self._config = config or {}
        self._connected = False
        self._subscribed_symbols: set[str] = set()
        self._session_manager = session_manager

        # API concurrency control (Issue #80)
        self._max_concurrent_requests = max_concurrent_requests
        self._api_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._semaphore_wait_time_total: float = 0.0  # Total time spent waiting
        self._semaphore_acquisitions: int = 0  # Number of acquisitions

    @property
    def session_manager(self) -> MarketSessionManager | None:
        """Market session manager for trading hours validation."""
        return self._session_manager

    def validate_session(self, order: Order) -> tuple[bool, str | None]:
        """
        Validate order against market session rules.

        Args:
            order: Order to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self._session_manager is None:
            # No session manager = no validation (e.g., crypto gateways)
            return True, None

        return self._session_manager.validate_order_timing(
            extended_hours=order.extended_hours,
        )

    @property
    def name(self) -> str:
        """Gateway identifier."""
        return self._name

    @property
    def is_connected(self) -> bool:
        """Connection status."""
        return self._connected

    @property
    def subscribed_symbols(self) -> set[str]:
        """Currently subscribed symbols."""
        return self._subscribed_symbols.copy()

    # -------------------------------------------------------------------------
    # API Concurrency Control (Issue #80)
    # -------------------------------------------------------------------------

    @property
    def api_semaphore_stats(self) -> dict[str, float | int]:
        """
        Get API semaphore statistics for monitoring.

        Returns:
            Dict with:
            - max_concurrent: Maximum allowed concurrent requests
            - total_wait_time: Total seconds spent waiting for semaphore
            - acquisitions: Number of semaphore acquisitions
            - avg_wait_time: Average wait time per acquisition
        """
        avg_wait = (
            self._semaphore_wait_time_total / self._semaphore_acquisitions
            if self._semaphore_acquisitions > 0
            else 0.0
        )
        return {
            "max_concurrent": self._max_concurrent_requests,
            "total_wait_time": self._semaphore_wait_time_total,
            "acquisitions": self._semaphore_acquisitions,
            "avg_wait_time": avg_wait,
        }

    @asynccontextmanager
    async def _api_call(self) -> AsyncGenerator[None, None]:
        """
        Context manager for API calls with semaphore and metrics.

        Use this to wrap all API calls to enforce concurrency limits
        and track wait times.

        Example:
            async with self._api_call():
                return await self._exchange.create_order(...)
        """
        start = time.monotonic()
        async with self._api_semaphore:
            wait_time = time.monotonic() - start
            self._semaphore_wait_time_total += wait_time
            self._semaphore_acquisitions += 1
            yield

    # -------------------------------------------------------------------------
    # Context Manager Support
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> BaseGateway:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    # -------------------------------------------------------------------------
    # Abstract Methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Connect to exchange."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        ...

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to market data."""
        ...

    @abstractmethod
    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from market data."""
        ...

    @abstractmethod
    def stream_ticks(self) -> AsyncIterator[Tick]:
        """Stream tick data."""
        ...

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get order book."""
        ...

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Tick:
        """Get ticker."""
        ...

    @abstractmethod
    async def submit_order(self, order: Order) -> OrderResult:
        """Submit order."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order."""
        ...

    @abstractmethod
    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all orders."""
        ...

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        """Get order status."""
        ...

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Get open orders."""
        ...

    @abstractmethod
    async def get_order_history(
        self, symbol: str | None = None, limit: int | None = None
    ) -> list[OrderResult]:
        """Get order history (Issue #27)."""
        ...

    @abstractmethod
    async def get_trades(
        self, symbol: str | None = None, limit: int | None = None
    ) -> list:
        """Get trade/fill history (Issue #27)."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get positions."""
        ...

    @abstractmethod
    async def get_position(self, symbol: str) -> Position | None:
        """Get position for symbol."""
        ...

    @abstractmethod
    async def get_balances(self) -> dict[str, Balance]:
        """Get balances."""
        ...

    @abstractmethod
    async def get_balance(self, currency: str) -> Balance | None:
        """Get balance for currency."""
        ...


# =============================================================================
# Exceptions
# =============================================================================


class GatewayError(Exception):
    """Base exception for gateway errors."""


class ConnectionError(GatewayError):
    """Failed to connect to exchange."""


class AuthenticationError(GatewayError):
    """Failed to authenticate with exchange."""


class OrderError(GatewayError):
    """Order-related error."""


class OrderNotFoundError(OrderError):
    """Order not found."""


class InsufficientFundsError(OrderError):
    """Insufficient funds for order."""


class RateLimitError(GatewayError):
    """Rate limit exceeded."""


class MarketClosedError(OrderError):
    """Market is closed, order cannot be placed."""


# =============================================================================
# Encoders/Decoders
# =============================================================================

# Fast JSON serialization for gateway types
_encoder = msgspec.json.Encoder()
_order_decoder = msgspec.json.Decoder(Order)
_order_result_decoder = msgspec.json.Decoder(OrderResult)
_position_decoder = msgspec.json.Decoder(Position)
_tick_decoder = msgspec.json.Decoder(Tick)


def encode_order(order: Order) -> bytes:
    """Encode Order to JSON bytes."""
    return _encoder.encode(order)


def decode_order(data: bytes) -> Order:
    """Decode Order from JSON bytes."""
    return _order_decoder.decode(data)


def encode_order_result(result: OrderResult) -> bytes:
    """Encode OrderResult to JSON bytes."""
    return _encoder.encode(result)


def decode_order_result(data: bytes) -> OrderResult:
    """Decode OrderResult from JSON bytes."""
    return _order_result_decoder.decode(data)
