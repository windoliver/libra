"""
GatewayFetcher: Provider/Fetcher Pattern for Gateway Layer.

Implements Issue #27: Provider/Fetcher Pattern for Gateway Layer.

This module adopts OpenBB's proven 3-stage TET pipeline (Transform-Extract-Transform):
1. transform_query() - Validate and convert params to gateway-specific query
2. extract_data() - Fetch raw data from provider
3. transform_data() - Normalize raw data to standard format

Benefits:
- Consistent interface across all gateways
- Clear separation of concerns
- Easy testing of each stage independently
- Proven at scale (OpenBB has 56K+ stars)

Usage:
    class MyBarFetcher(GatewayFetcher[BarQuery, list[Bar]]):
        def transform_query(self, params: dict) -> BarQuery:
            return BarQuery(**params)

        async def extract_data(self, query: BarQuery, **kwargs) -> Any:
            return await self._client.fetch_ohlcv(query.symbol)

        def transform_data(self, query: BarQuery, raw: Any) -> list[Bar]:
            return [Bar(...) for row in raw]

    # Use the fetcher
    fetcher = MyBarFetcher()
    bars = await fetcher.fetch(symbol="BTC/USDT", interval="1h")

References:
- OpenBB Fetcher: openbb_platform/core/openbb_core/provider/abstract/fetcher.py
- Issue #27: Provider/Fetcher Pattern for Gateway Layer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import msgspec


if TYPE_CHECKING:
    pass


# =============================================================================
# Type Variables
# =============================================================================

Q = TypeVar("Q", bound="BaseQuery")  # Query type
R = TypeVar("R")  # Response type


# =============================================================================
# Query Types (Input)
# =============================================================================


@dataclass(frozen=True)
class BaseQuery:
    """Base class for all query types."""

    pass


@dataclass(frozen=True)
class BarQuery(BaseQuery):
    """
    Query parameters for fetching bar/OHLCV data.

    Examples:
        # Fetch 100 hourly BTC bars
        query = BarQuery(
            symbol="BTC/USDT",
            interval="1h",
            limit=100,
        )

        # Fetch bars with date range
        query = BarQuery(
            symbol="ETH/USDT",
            interval="1d",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
        )
    """

    symbol: str
    interval: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d, etc.
    limit: int | None = None  # Number of bars to fetch
    start: datetime | None = None  # Start time
    end: datetime | None = None  # End time
    provider: str | None = None  # Optional: specific provider


@dataclass(frozen=True)
class TickQuery(BaseQuery):
    """
    Query parameters for fetching ticker/quote data.

    Examples:
        query = TickQuery(symbol="BTC/USDT")
    """

    symbol: str
    provider: str | None = None


@dataclass(frozen=True)
class OrderBookQuery(BaseQuery):
    """
    Query parameters for fetching order book data.

    Examples:
        query = OrderBookQuery(symbol="BTC/USDT", depth=20)
    """

    symbol: str
    depth: int = 20  # Number of levels
    provider: str | None = None


@dataclass(frozen=True)
class BalanceQuery(BaseQuery):
    """
    Query parameters for fetching account balance.

    Examples:
        # Get all balances
        query = BalanceQuery()

        # Get specific currency
        query = BalanceQuery(currency="USDT")
    """

    currency: str | None = None  # None = all currencies


@dataclass(frozen=True)
class PositionQuery(BaseQuery):
    """
    Query parameters for fetching positions.

    Examples:
        # Get all positions
        query = PositionQuery()

        # Get specific symbol
        query = PositionQuery(symbol="BTC/USDT")
    """

    symbol: str | None = None  # None = all positions


@dataclass(frozen=True)
class OrderQuery(BaseQuery):
    """
    Query parameters for fetching orders.

    Examples:
        # Get all open orders
        query = OrderQuery(status="open")

        # Get orders for specific symbol
        query = OrderQuery(symbol="BTC/USDT", status="open")
    """

    symbol: str | None = None
    order_id: str | None = None
    status: str | None = None  # "open", "closed", "all"
    limit: int | None = None
    since: datetime | None = None


@dataclass(frozen=True)
class TradeQuery(BaseQuery):
    """
    Query parameters for fetching trade history.

    Examples:
        query = TradeQuery(symbol="BTC/USDT", limit=100)
    """

    symbol: str | None = None
    limit: int | None = None
    since: datetime | None = None


# =============================================================================
# Response Types (Output) - Using msgspec.Struct for performance
# =============================================================================


class Bar(msgspec.Struct, frozen=True, gc=False):
    """
    OHLCV bar data (candle).

    Immutable and optimized for fast serialization.

    Examples:
        bar = Bar(
            symbol="BTC/USDT",
            timestamp_ns=1704067200_000_000_000,
            open=Decimal("42000"),
            high=Decimal("42500"),
            low=Decimal("41800"),
            close=Decimal("42300"),
            volume=Decimal("1500.5"),
        )
    """

    symbol: str
    timestamp_ns: int  # Nanoseconds since epoch
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    interval: str = "1h"
    trades: int | None = None  # Number of trades (if available)

    @property
    def timestamp_ms(self) -> int:
        """Timestamp in milliseconds."""
        return self.timestamp_ns // 1_000_000

    @property
    def timestamp_sec(self) -> float:
        """Timestamp in seconds."""
        return self.timestamp_ns / 1_000_000_000

    @property
    def datetime(self) -> datetime:
        """Timestamp as datetime."""
        return datetime.fromtimestamp(self.timestamp_sec)


class Quote(msgspec.Struct, frozen=True, gc=False):
    """
    Market quote/ticker data.

    Examples:
        quote = Quote(
            symbol="BTC/USDT",
            bid=Decimal("42000"),
            ask=Decimal("42001"),
            last=Decimal("42000.50"),
            timestamp_ns=time.time_ns(),
        )
    """

    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    timestamp_ns: int
    bid_size: Decimal | None = None
    ask_size: Decimal | None = None
    volume_24h: Decimal | None = None
    high_24h: Decimal | None = None
    low_24h: Decimal | None = None
    change_24h_pct: Decimal | None = None

    @property
    def mid(self) -> Decimal:
        """Mid price."""
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


class OrderBookLevel(msgspec.Struct, frozen=True, gc=False):
    """Single order book level."""

    price: Decimal
    size: Decimal


class OrderBookSnapshot(msgspec.Struct, frozen=True, gc=False):
    """
    Order book snapshot.

    Examples:
        orderbook = OrderBookSnapshot(
            symbol="BTC/USDT",
            bids=[OrderBookLevel(price=Decimal("42000"), size=Decimal("1.5"))],
            asks=[OrderBookLevel(price=Decimal("42001"), size=Decimal("2.0"))],
            timestamp_ns=time.time_ns(),
        )
    """

    symbol: str
    bids: list[OrderBookLevel]  # Sorted descending by price
    asks: list[OrderBookLevel]  # Sorted ascending by price
    timestamp_ns: int

    @property
    def best_bid(self) -> Decimal | None:
        """Best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        """Best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        """Mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Decimal | None:
        """Bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


class AccountBalance(msgspec.Struct, frozen=True, gc=False):
    """
    Account balance for a single currency.

    Examples:
        balance = AccountBalance(
            currency="USDT",
            total=Decimal("10000"),
            available=Decimal("8000"),
            locked=Decimal("2000"),
        )
    """

    currency: str
    total: Decimal
    available: Decimal
    locked: Decimal


class AccountPosition(msgspec.Struct, frozen=True, gc=False):
    """
    Account position for a single symbol.

    Represents an open position with P&L tracking.

    Examples:
        position = AccountPosition(
            symbol="BTC/USDT",
            side="long",
            amount=Decimal("0.5"),
            entry_price=Decimal("48000.00"),
            current_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("1000.00"),
            leverage=1,
        )
    """

    symbol: str
    side: str  # "long", "short", "flat"
    amount: Decimal  # Position size (always positive)
    entry_price: Decimal  # Average entry price
    current_price: Decimal  # Current mark price
    unrealized_pnl: Decimal  # Unrealized profit/loss
    timestamp_ns: int | None = None  # Last update time
    leverage: int = 1  # Leverage multiplier
    liquidation_price: Decimal | None = None  # Liquidation price (derivatives)
    margin: Decimal | None = None  # Margin used
    margin_type: str | None = None  # "cross" or "isolated"
    realized_pnl: Decimal | None = None  # Realized profit/loss

    @property
    def notional_value(self) -> Decimal:
        """Position notional value (amount * current_price)."""
        return self.amount * self.current_price

    @property
    def pnl_percent(self) -> Decimal:
        """P&L as percentage of entry value."""
        entry_value = self.amount * self.entry_price
        if entry_value == 0:
            return Decimal("0")
        return (self.unrealized_pnl / entry_value) * 100


class AccountOrder(msgspec.Struct, frozen=True, gc=False):
    """
    Order record from exchange.

    Represents an order (open or historical) with fill information.

    Examples:
        order = AccountOrder(
            order_id="12345",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            status="filled",
            amount=Decimal("0.1"),
            filled=Decimal("0.1"),
            price=Decimal("50000"),
            average=Decimal("49980"),
            timestamp_ns=time.time_ns(),
        )
    """

    order_id: str
    symbol: str
    side: str  # "buy", "sell"
    order_type: str  # "market", "limit", "stop", etc.
    status: str  # "open", "closed", "canceled"
    amount: Decimal  # Original order amount
    filled: Decimal  # Amount filled
    timestamp_ns: int  # Order creation time
    price: Decimal | None = None  # Limit price
    average: Decimal | None = None  # Average fill price
    remaining: Decimal | None = None  # Remaining amount
    cost: Decimal | None = None  # Total cost (filled * average)
    fee: Decimal | None = None  # Fee amount
    fee_currency: str | None = None  # Fee currency
    client_order_id: str | None = None  # Client order ID
    stop_price: Decimal | None = None  # Stop trigger price
    time_in_force: str | None = None  # "GTC", "IOC", "FOK", etc.
    reduce_only: bool = False  # Only reduce position
    post_only: bool = False  # Only maker orders

    @property
    def is_open(self) -> bool:
        """Check if order is still open."""
        return self.status in ("open", "partially_filled")

    @property
    def fill_percent(self) -> Decimal:
        """Percentage of order filled (0-100)."""
        if self.amount == 0:
            return Decimal("0")
        return (self.filled / self.amount) * 100


class TradeRecord(msgspec.Struct, frozen=True, gc=False):
    """
    Trade/fill record from exchange.

    Represents an individual trade execution.

    Examples:
        trade = TradeRecord(
            trade_id="T12345",
            order_id="O12345",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.1"),
            price=Decimal("50000"),
            cost=Decimal("5000"),
            timestamp_ns=time.time_ns(),
        )
    """

    trade_id: str
    order_id: str
    symbol: str
    side: str  # "buy", "sell"
    amount: Decimal  # Trade amount
    price: Decimal  # Trade price
    cost: Decimal  # Total cost (amount * price)
    timestamp_ns: int  # Trade execution time
    fee: Decimal | None = None  # Fee amount
    fee_currency: str | None = None  # Fee currency
    taker_or_maker: str | None = None  # "taker" or "maker"


# =============================================================================
# Fetcher Protocol & Abstract Base Class
# =============================================================================


class GatewayFetcher(ABC, Generic[Q, R]):
    """
    Abstract base class for gateway data fetchers.

    Implements the 3-stage TET pipeline:
    1. transform_query() - Convert params dict to typed query
    2. extract_data() - Fetch raw data from provider
    3. transform_data() - Normalize to standard response

    Subclasses must implement all three abstract methods.

    Example:
        class CCXTBarFetcher(GatewayFetcher[BarQuery, list[Bar]]):
            def __init__(self, exchange):
                self._exchange = exchange

            def transform_query(self, params: dict) -> BarQuery:
                return BarQuery(
                    symbol=params["symbol"],
                    interval=params.get("interval", "1h"),
                    limit=params.get("limit", 100),
                )

            async def extract_data(self, query: BarQuery, **kwargs) -> Any:
                return await self._exchange.fetch_ohlcv(
                    query.symbol,
                    query.interval,
                    limit=query.limit,
                )

            def transform_data(self, query: BarQuery, raw: Any) -> list[Bar]:
                return [
                    Bar(
                        symbol=query.symbol,
                        interval=query.interval,
                        timestamp_ns=int(candle[0] * 1_000_000),
                        open=Decimal(str(candle[1])),
                        high=Decimal(str(candle[2])),
                        low=Decimal(str(candle[3])),
                        close=Decimal(str(candle[4])),
                        volume=Decimal(str(candle[5])),
                    )
                    for candle in raw
                ]
    """

    @abstractmethod
    def transform_query(self, params: dict[str, Any]) -> Q:
        """
        Transform input parameters to typed query.

        Stage 1 of the TET pipeline.
        Validates parameters and converts to gateway-specific query format.

        Args:
            params: Raw parameter dictionary

        Returns:
            Typed query object

        Raises:
            ValueError: If required parameters are missing
            TypeError: If parameters have wrong types
        """
        ...

    @abstractmethod
    async def extract_data(self, query: Q, **kwargs: Any) -> Any:
        """
        Fetch raw data from the provider.

        Stage 2 of the TET pipeline.
        Makes API call to external service and returns raw response.

        Args:
            query: Typed query from transform_query()
            **kwargs: Additional provider-specific options (credentials, etc.)

        Returns:
            Raw data from provider (format varies by provider)

        Raises:
            ConnectionError: If provider is unreachable
            AuthenticationError: If credentials are invalid
            RateLimitError: If rate limit exceeded
        """
        ...

    @abstractmethod
    def transform_data(self, query: Q, raw: Any) -> R:
        """
        Transform raw data to standard response format.

        Stage 3 of the TET pipeline.
        Normalizes provider-specific data to consistent schema.

        Args:
            query: Original query (for context)
            raw: Raw data from extract_data()

        Returns:
            Normalized response in standard format

        Raises:
            ValueError: If raw data is malformed
        """
        ...

    async def fetch(self, **params: Any) -> R:
        """
        Execute the full TET pipeline.

        This is the main entry point for using a fetcher.
        Calls transform_query -> extract_data -> transform_data.

        Args:
            **params: Parameters to pass to transform_query()

        Returns:
            Normalized response data

        Example:
            fetcher = CCXTBarFetcher(exchange)
            bars = await fetcher.fetch(
                symbol="BTC/USDT",
                interval="1h",
                limit=100,
            )
        """
        query = self.transform_query(params)
        raw = await self.extract_data(query)
        return self.transform_data(query, raw)

    async def fetch_with_query(self, query: Q, **kwargs: Any) -> R:
        """
        Execute pipeline with pre-built query.

        Useful when you already have a typed query object.

        Args:
            query: Pre-built query object
            **kwargs: Additional options for extract_data()

        Returns:
            Normalized response data
        """
        raw = await self.extract_data(query, **kwargs)
        return self.transform_data(query, raw)


# =============================================================================
# Fetcher Registry
# =============================================================================


@dataclass
class FetcherRegistry:
    """
    Registry for gateway fetchers.

    Allows looking up fetchers by name and type.

    Example:
        registry = FetcherRegistry()
        registry.register("ccxt", "bar", CCXTBarFetcher)
        registry.register("ccxt", "quote", CCXTQuoteFetcher)

        # Get fetcher
        bar_fetcher_cls = registry.get("ccxt", "bar")
        fetcher = bar_fetcher_cls(exchange)
    """

    _fetchers: dict[str, dict[str, type[GatewayFetcher[Any, Any]]]] = field(
        default_factory=dict
    )

    def register(
        self,
        gateway_name: str,
        data_type: str,
        fetcher_class: type[GatewayFetcher[Any, Any]],
    ) -> None:
        """
        Register a fetcher class.

        Args:
            gateway_name: Name of the gateway (e.g., "ccxt", "openbb")
            data_type: Type of data (e.g., "bar", "quote", "orderbook")
            fetcher_class: The fetcher class to register
        """
        if gateway_name not in self._fetchers:
            self._fetchers[gateway_name] = {}
        self._fetchers[gateway_name][data_type] = fetcher_class

    def get(
        self,
        gateway_name: str,
        data_type: str,
    ) -> type[GatewayFetcher[Any, Any]] | None:
        """
        Get a registered fetcher class.

        Args:
            gateway_name: Name of the gateway
            data_type: Type of data

        Returns:
            Fetcher class or None if not found
        """
        return self._fetchers.get(gateway_name, {}).get(data_type)

    def list_gateways(self) -> list[str]:
        """List all registered gateway names."""
        return list(self._fetchers.keys())

    def list_data_types(self, gateway_name: str) -> list[str]:
        """List data types available for a gateway."""
        return list(self._fetchers.get(gateway_name, {}).keys())


# Global registry instance
fetcher_registry = FetcherRegistry()


# =============================================================================
# Utility Functions
# =============================================================================


def timestamp_to_ns(ts: int | float | datetime) -> int:
    """
    Convert various timestamp formats to nanoseconds.

    The function detects the timestamp unit based on magnitude:
    - Nanoseconds: > 1e18 (e.g., 1704067200_000_000_000)
    - Microseconds: > 1e15 (e.g., 1704067200_000_000)
    - Milliseconds: > 1e12 (e.g., 1704067200_000)
    - Seconds: <= 1e12 (e.g., 1704067200)

    Args:
        ts: Timestamp as int (ms/us/ns), float (seconds), or datetime

    Returns:
        Nanoseconds since epoch

    Examples:
        >>> timestamp_to_ns(1704067200)           # seconds
        1704067200000000000
        >>> timestamp_to_ns(1704067200_000)       # milliseconds
        1704067200000000000
        >>> timestamp_to_ns(datetime(2024, 1, 1)) # datetime
        1704067200000000000
    """
    if isinstance(ts, datetime):
        return int(ts.timestamp() * 1_000_000_000)
    if isinstance(ts, float):
        # Assume seconds (float timestamps are typically in seconds)
        return int(ts * 1_000_000_000)

    # Integer timestamps - detect unit by magnitude
    # Epoch times for reference (2024-01-01):
    # - Nanoseconds: ~1.7e18
    # - Microseconds: ~1.7e15
    # - Milliseconds: ~1.7e12
    # - Seconds: ~1.7e9
    if ts > 1_000_000_000_000_000_000:
        # Already nanoseconds (> 1e18)
        return ts
    if ts > 1_000_000_000_000_000:
        # Microseconds (> 1e15)
        return ts * 1_000
    if ts > 1_000_000_000_000:
        # Milliseconds (> 1e12)
        return ts * 1_000_000
    # Seconds (<= 1e12)
    return ts * 1_000_000_000
