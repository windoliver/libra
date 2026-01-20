"""
DataClient Protocol: Market data operations interface.

Defines the contract for all market data client implementations.
Handles subscriptions, streaming, and historical data requests.

Design inspired by NautilusTrader's DataClient architecture.
See: https://nautilustrader.io/docs/latest/concepts/adapters/

Implementations:
    - CCXTDataClient: 100+ exchanges via CCXT
    - PaperDataClient: Simulated market data
    - BacktestDataClient: Historical data replay

See: https://github.com/windoliver/libra/issues/33
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import msgspec


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from decimal import Decimal

    from libra.gateways.protocol import InstrumentStatusEvent, OrderBook, Tick
    from libra.strategies.protocol import Bar


# =============================================================================
# Instrument Data Structure
# =============================================================================


class Instrument(msgspec.Struct, frozen=True, gc=False):
    """
    Trading instrument definition.

    Contains exchange-specific information about a tradable pair.
    Immutable for thread safety and hashable for use in sets/dicts.

    Attributes:
        symbol: Unified symbol format (e.g., "BTC/USDT")
        base: Base currency (e.g., "BTC")
        quote: Quote currency (e.g., "USDT")
        exchange: Exchange identifier (e.g., "binance")
        lot_size: Minimum order size increment
        tick_size: Minimum price increment
        min_quantity: Minimum order quantity (Issue #113)
        max_quantity: Maximum order quantity (Issue #113)
        min_notional: Minimum order value (e.g., $10)
        contract_type: "spot", "perpetual", "future", "option"
        is_active: Whether trading is currently enabled

    Examples:
        instrument = Instrument(
            symbol="BTC/USDT",
            base="BTC",
            quote="USDT",
            exchange="binance",
            lot_size=Decimal("0.00001"),
            tick_size=Decimal("0.01"),
            min_quantity=Decimal("0.00001"),
            max_quantity=Decimal("1000"),
            min_notional=Decimal("10"),
            contract_type="spot",
            is_active=True,
        )
    """

    symbol: str
    base: str
    quote: str
    exchange: str

    # Trading constraints (use Any to avoid Decimal import complexity)
    lot_size: Any  # Decimal - minimum order size increment
    tick_size: Any  # Decimal - minimum price increment
    min_quantity: Any | None = None  # Decimal - minimum order quantity (Issue #113)
    max_quantity: Any | None = None  # Decimal - maximum order quantity (Issue #113)
    min_notional: Any | None = None  # Decimal - minimum order value

    # Instrument type
    contract_type: str = "spot"  # "spot", "perpetual", "future", "option"
    is_active: bool = True

    # Optional fields
    maker_fee: Any | None = None  # Decimal - maker fee rate
    taker_fee: Any | None = None  # Decimal - taker fee rate
    leverage_max: int | None = None  # Maximum leverage for derivatives
    expiry: datetime | None = None  # Expiry for futures/options


# =============================================================================
# DataClient Protocol
# =============================================================================


@runtime_checkable
class DataClient(Protocol):
    """
    Market data client protocol.

    Defines the interface for subscribing to and receiving market data.
    All market data operations (subscriptions, historical requests) go through here.

    This protocol uses structural subtyping - any class implementing these
    methods is considered a DataClient, no inheritance required.

    Thread Safety:
        Implementations should be thread-safe. Subscription state should be
        protected by locks if accessed from multiple coroutines.

    Connection Management:
        - connect() must be called before any data operations
        - disconnect() should be called for cleanup
        - Implementations should handle reconnection internally

    Examples:
        async with CCXTDataClient("binance") as client:
            await client.subscribe_ticks("BTC/USDT")

            async for tick in client.stream_ticks():
                print(f"{tick.symbol}: {tick.last}")

        # Or for backtesting:
        client = BacktestDataClient(historical_data, clock)
        await client.connect()
        bars = await client.request_bars("BTC/USDT", "1h", start, end)
    """

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Client identifier (e.g., "binance-data", "backtest-data").

        Used for logging, metrics, and routing.
        """
        ...

    @property
    def is_connected(self) -> bool:
        """
        Check if client is connected and ready.

        Returns:
            True if connected and can receive data, False otherwise.
        """
        ...

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect to the data source.

        Establishes WebSocket connections, authenticates if required,
        and prepares for data reception.

        Raises:
            ConnectionError: If connection fails.
            AuthenticationError: If authentication fails.

        Note:
            Must be called before any subscription or request operations.
            Implementations should handle reconnection internally.
        """
        ...

    async def disconnect(self) -> None:
        """
        Disconnect from the data source.

        Closes all connections and cleans up resources.
        Should be safe to call multiple times (idempotent).

        Note:
            After disconnect, connect() must be called again before use.
        """
        ...

    # -------------------------------------------------------------------------
    # Real-time Subscriptions
    # -------------------------------------------------------------------------

    async def subscribe_ticks(self, symbol: str) -> None:
        """
        Subscribe to real-time tick/quote data for a symbol.

        Ticks are received via stream_ticks() after subscription.

        Args:
            symbol: Trading pair in unified format (e.g., "BTC/USDT")

        Raises:
            ValueError: If symbol format is invalid.
            ConnectionError: If not connected.

        Note:
            Duplicate subscriptions are ignored (idempotent).
        """
        ...

    async def subscribe_bars(self, symbol: str, timeframe: str) -> None:
        """
        Subscribe to real-time bar/candlestick data.

        Bars are received via stream_bars() after subscription.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Bar interval (e.g., "1m", "5m", "1h", "1d")

        Raises:
            ValueError: If symbol or timeframe is invalid.
            ConnectionError: If not connected.
        """
        ...

    async def subscribe_orderbook(self, symbol: str, depth: int = 10) -> None:
        """
        Subscribe to real-time order book updates.

        Order book snapshots are received via stream_orderbooks() after subscription.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            depth: Number of price levels (default 10, max varies by exchange)

        Raises:
            ValueError: If symbol is invalid or depth exceeds limits.
            ConnectionError: If not connected.
        """
        ...

    async def unsubscribe_ticks(self, symbol: str) -> None:
        """
        Unsubscribe from tick data for a symbol.

        Args:
            symbol: Trading pair to unsubscribe from.

        Note:
            Unsubscribing from non-subscribed symbol is a no-op (idempotent).
        """
        ...

    async def unsubscribe_bars(self, symbol: str, timeframe: str) -> None:
        """
        Unsubscribe from bar data.

        Args:
            symbol: Trading pair
            timeframe: Bar interval
        """
        ...

    async def unsubscribe_orderbook(self, symbol: str) -> None:
        """
        Unsubscribe from order book updates.

        Args:
            symbol: Trading pair
        """
        ...

    # -------------------------------------------------------------------------
    # Data Streams
    # -------------------------------------------------------------------------

    def stream_ticks(self) -> AsyncIterator[Tick]:
        """
        Stream real-time tick data for all subscribed symbols.

        Yields ticks as they arrive from the data source.
        Subscribe to symbols first using subscribe_ticks().

        Yields:
            Tick objects in arrival order.

        Note:
            The iterator continues until disconnect() or unsubscribe all.
            Backpressure: if consumer is slow, ticks may be buffered or dropped
            depending on implementation.

        Example:
            async for tick in client.stream_ticks():
                print(f"{tick.symbol}: bid={tick.bid} ask={tick.ask}")
        """
        ...

    def stream_bars(self) -> AsyncIterator[Bar]:
        """
        Stream real-time bar data for all subscribed symbols/timeframes.

        Yields bars as they complete (not partial bars).

        Yields:
            Bar objects as they complete.

        Example:
            async for bar in client.stream_bars():
                print(f"{bar.symbol} {bar.timeframe}: close={bar.close}")
        """
        ...

    def stream_orderbooks(self) -> AsyncIterator[OrderBook]:
        """
        Stream order book updates for all subscribed symbols.

        Yields order book snapshots on each update.

        Yields:
            OrderBook objects with current bids/asks.
        """
        ...

    # -------------------------------------------------------------------------
    # Instrument Status Subscriptions (Issue #110)
    # -------------------------------------------------------------------------

    async def subscribe_instrument_status(self, symbol: str) -> None:
        """
        Subscribe to instrument status updates (Issue #110).

        Receives updates for trading halts, session changes, and other
        instrument state changes via stream_instrument_status().

        Args:
            symbol: Trading pair (e.g., "AAPL", "BTC/USDT")

        Note:
            Not all data sources support instrument status updates.
            Raises NotImplementedError if not supported.
        """
        ...

    async def unsubscribe_instrument_status(self, symbol: str) -> None:
        """
        Unsubscribe from instrument status updates.

        Args:
            symbol: Trading pair to unsubscribe from.
        """
        ...

    def stream_instrument_status(self) -> AsyncIterator[InstrumentStatusEvent]:
        """
        Stream instrument status updates for subscribed symbols (Issue #110).

        Yields status events for trading halts, session changes,
        and other instrument state changes.

        Yields:
            InstrumentStatusEvent objects on status changes.

        Example:
            async for event in client.stream_instrument_status():
                if event.status == InstrumentStatus.HALT:
                    print(f"{event.symbol} halted: {event.halt_reason_text}")
        """
        ...

    # -------------------------------------------------------------------------
    # Historical Data Requests
    # -------------------------------------------------------------------------

    async def request_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Bar]:
        """
        Request historical bar data.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Bar interval (e.g., "1m", "1h", "1d")
            start: Start time (inclusive)
            end: End time (inclusive)
            limit: Maximum number of bars (None = no limit)

        Returns:
            List of Bar objects sorted by timestamp ascending.

        Raises:
            ValueError: If parameters are invalid.
            DataNotAvailableError: If historical data is not available.

        Note:
            Large requests may be paginated internally.
            Exchange rate limits apply.
        """
        ...

    async def request_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Tick]:
        """
        Request historical tick data.

        Args:
            symbol: Trading pair
            start: Start time (inclusive)
            end: End time (inclusive)
            limit: Maximum number of ticks

        Returns:
            List of Tick objects sorted by timestamp ascending.

        Note:
            Not all data sources support tick-level historical data.
            This may be expensive in terms of data volume.
        """
        ...

    # -------------------------------------------------------------------------
    # Instrument Information
    # -------------------------------------------------------------------------

    async def get_instruments(self) -> list[Instrument]:
        """
        Get all available trading instruments.

        Returns:
            List of Instrument objects with trading constraints.

        Note:
            May be cached - call periodically to refresh.
        """
        ...

    async def get_instrument(self, symbol: str) -> Instrument | None:
        """
        Get instrument information for a specific symbol.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")

        Returns:
            Instrument if found, None otherwise.
        """
        ...

    # -------------------------------------------------------------------------
    # Snapshots
    # -------------------------------------------------------------------------

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """
        Get current order book snapshot.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            depth: Number of price levels (default 20)

        Returns:
            OrderBook with current bids and asks.

        Note:
            This is a REST request, not streaming. Use subscribe_orderbook()
            for real-time updates.
        """
        ...

    async def get_ticker(self, symbol: str) -> Tick:
        """
        Get current ticker (quote) for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Current Tick data with bid/ask/last prices.

        Note:
            This is a REST request. Use subscribe_ticks() for streaming.
        """
        ...


# =============================================================================
# Abstract Base Class (Optional)
# =============================================================================


class BaseDataClient(ABC):
    """
    Abstract base class for DataClient implementations.

    Provides common functionality:
    - Connection state tracking
    - Subscription management
    - Configuration handling

    Subclasses must implement all abstract methods.
    Use this base class for code reuse, or implement DataClient protocol directly.

    Examples:
        class CCXTDataClient(BaseDataClient):
            async def connect(self) -> None:
                self._exchange = ccxt.pro.binance()
                await self._exchange.load_markets()
                self._connected = True
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        """
        Initialize data client.

        Args:
            name: Client identifier
            config: Configuration dict (API keys, endpoints, etc.)
        """
        self._name = name
        self._config = config or {}
        self._connected = False
        self._subscribed_ticks: set[str] = set()
        self._subscribed_bars: dict[str, set[str]] = {}  # symbol -> set of timeframes
        self._subscribed_orderbooks: set[str] = set()

    @property
    def name(self) -> str:
        """Client identifier."""
        return self._name

    @property
    def is_connected(self) -> bool:
        """Connection status."""
        return self._connected

    @property
    def subscribed_ticks(self) -> set[str]:
        """Currently subscribed tick symbols."""
        return self._subscribed_ticks.copy()

    @property
    def subscribed_bars(self) -> dict[str, set[str]]:
        """Currently subscribed bar symbols and timeframes."""
        return {k: v.copy() for k, v in self._subscribed_bars.items()}

    @property
    def subscribed_orderbooks(self) -> set[str]:
        """Currently subscribed orderbook symbols."""
        return self._subscribed_orderbooks.copy()

    # -------------------------------------------------------------------------
    # Context Manager Support
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> BaseDataClient:
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
        """Connect to data source."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source."""
        ...

    @abstractmethod
    async def subscribe_ticks(self, symbol: str) -> None:
        """Subscribe to tick data."""
        ...

    @abstractmethod
    async def subscribe_bars(self, symbol: str, timeframe: str) -> None:
        """Subscribe to bar data."""
        ...

    @abstractmethod
    async def subscribe_orderbook(self, symbol: str, depth: int = 10) -> None:
        """Subscribe to order book."""
        ...

    @abstractmethod
    async def unsubscribe_ticks(self, symbol: str) -> None:
        """Unsubscribe from tick data."""
        ...

    @abstractmethod
    async def unsubscribe_bars(self, symbol: str, timeframe: str) -> None:
        """Unsubscribe from bar data."""
        ...

    @abstractmethod
    async def unsubscribe_orderbook(self, symbol: str) -> None:
        """Unsubscribe from order book."""
        ...

    @abstractmethod
    def stream_ticks(self) -> AsyncIterator[Tick]:
        """Stream tick data."""
        ...

    @abstractmethod
    def stream_bars(self) -> AsyncIterator[Bar]:
        """Stream bar data."""
        ...

    @abstractmethod
    def stream_orderbooks(self) -> AsyncIterator[OrderBook]:
        """Stream order book data."""
        ...

    @abstractmethod
    async def request_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Bar]:
        """Request historical bars."""
        ...

    @abstractmethod
    async def request_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Tick]:
        """Request historical ticks."""
        ...

    @abstractmethod
    async def get_instruments(self) -> list[Instrument]:
        """Get available instruments."""
        ...

    @abstractmethod
    async def get_instrument(self, symbol: str) -> Instrument | None:
        """Get instrument info."""
        ...

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get order book snapshot."""
        ...

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Tick:
        """Get current ticker."""
        ...


# =============================================================================
# Exceptions
# =============================================================================


class DataClientError(Exception):
    """Base exception for data client errors."""


class DataNotAvailableError(DataClientError):
    """Historical data not available for requested range."""


class SubscriptionError(DataClientError):
    """Subscription failed."""
