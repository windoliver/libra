"""
Time-Series Database Protocol.

Defines the interface for all time-series database implementations.
Allows swapping QuestDB for TimescaleDB, ClickHouse, etc. without
changing application code.

Design:
- Async-first for non-blocking I/O
- Supports both streaming ingestion and batch queries
- ASOF JOIN support for backtest accuracy
- Polars DataFrame integration for analytics
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import msgspec


if TYPE_CHECKING:
    import polars as pl

    from libra.gateways.protocol import OrderResult, Position, Tick
    from libra.strategies.protocol import Bar


# =============================================================================
# Trade Record (for persistence)
# =============================================================================


class TradeRecord(msgspec.Struct, frozen=True, gc=False):
    """
    Trade record for database persistence.

    Captures all information about an executed trade for audit,
    analytics, and compliance purposes.

    Note: Uses Any for Decimal fields to avoid import complexity
    while maintaining the struct's performance benefits.
    """

    # Identification
    trade_id: str
    order_id: str
    symbol: str
    exchange: str

    # Trade details
    side: str  # "buy" or "sell"
    amount: Any  # Decimal
    price: Any  # Decimal
    fee: Any  # Decimal
    fee_currency: str

    # Timestamps (nanoseconds)
    timestamp_ns: int
    order_timestamp_ns: int | None = None

    # Strategy attribution
    strategy: str | None = None
    signal_id: str | None = None

    # P&L (calculated post-trade)
    realized_pnl: Any | None = None  # Decimal
    position_after: Any | None = None  # Decimal

    # Metadata
    metadata: dict[str, Any] | None = None


# =============================================================================
# Time-Series Database Protocol
# =============================================================================


@runtime_checkable
class TimeSeriesDB(Protocol):
    """
    Protocol for time-series database implementations.

    Defines the contract for tick data storage, OHLCV queries,
    and trade history persistence.

    Implementations:
    - AsyncQuestDBClient: QuestDB with ILP ingestion + asyncpg queries
    - (Future) TimescaleClient: TimescaleDB implementation
    - (Future) MockTimeSeriesDB: In-memory for testing

    Thread Safety:
        Implementations should be thread-safe. Connection pools
        handle concurrent access internally.

    Examples:
        async with AsyncQuestDBClient(config) as db:
            # Ingest ticks
            await db.ingest_tick(tick)

            # Query bars
            bars = await db.get_bars("BTC/USDT", "1h", start, end)

            # ASOF JOIN for backtest
            df = await db.asof_join_signals("prices", "signals", "BTC/USDT")
    """

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Initialize database connections.

        Creates connection pools and prepares for operations.
        Must be called before any other methods.

        Raises:
            ConnectionError: If connection fails.
        """
        ...

    async def close(self) -> None:
        """
        Close all database connections.

        Releases connection pools and cleans up resources.
        Safe to call multiple times (idempotent).
        """
        ...

    @property
    def is_connected(self) -> bool:
        """Check if database is connected and ready."""
        ...

    # -------------------------------------------------------------------------
    # Tick Data
    # -------------------------------------------------------------------------

    async def ingest_tick(self, tick: Tick) -> None:
        """
        Ingest a single tick.

        Uses ILP for high-throughput ingestion.
        Automatically batches for efficiency.

        Args:
            tick: Tick data to store.
        """
        ...

    async def ingest_ticks(self, ticks: list[Tick]) -> None:
        """
        Batch ingest multiple ticks.

        More efficient than individual ingestion for bulk data.

        Args:
            ticks: List of ticks to store.
        """
        ...

    async def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Tick]:
        """
        Query historical tick data.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            start: Start time (inclusive)
            end: End time (inclusive)
            limit: Maximum number of ticks (None = no limit)

        Returns:
            List of Tick objects sorted by timestamp.
        """
        ...

    async def get_latest_tick(self, symbol: str) -> Tick | None:
        """
        Get the most recent tick for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Latest Tick or None if no data.
        """
        ...

    # -------------------------------------------------------------------------
    # Bar Data (OHLCV)
    # -------------------------------------------------------------------------

    async def ingest_bar(self, bar: Bar) -> None:
        """
        Ingest a single OHLCV bar.

        Args:
            bar: Bar data to store.
        """
        ...

    async def ingest_bars(self, bars: list[Bar]) -> None:
        """
        Batch ingest multiple bars.

        Args:
            bars: List of bars to store.
        """
        ...

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Bar]:
        """
        Query historical bar data.

        Args:
            symbol: Trading pair
            timeframe: Bar interval (e.g., "1m", "1h", "1d")
            start: Start time (inclusive)
            end: End time (inclusive)
            limit: Maximum number of bars

        Returns:
            List of Bar objects sorted by timestamp.
        """
        ...

    async def get_bars_df(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Query bars as Polars DataFrame.

        More efficient for analytics - avoids object creation overhead.

        Args:
            symbol: Trading pair
            timeframe: Bar interval
            start: Start time
            end: End time

        Returns:
            Polars DataFrame with columns: timestamp, open, high, low, close, volume
        """
        ...

    # -------------------------------------------------------------------------
    # Trade History
    # -------------------------------------------------------------------------

    async def ingest_trade(self, trade: TradeRecord) -> None:
        """
        Store a trade record.

        Args:
            trade: Trade record to store.
        """
        ...

    async def get_trades(
        self,
        symbol: str | None = None,
        strategy: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[TradeRecord]:
        """
        Query trade history.

        Args:
            symbol: Filter by symbol (None = all)
            strategy: Filter by strategy (None = all)
            start: Start time (None = no limit)
            end: End time (None = no limit)
            limit: Maximum number of trades

        Returns:
            List of TradeRecord objects sorted by timestamp.
        """
        ...

    # -------------------------------------------------------------------------
    # Backtest Support (ASOF JOIN)
    # -------------------------------------------------------------------------

    async def asof_join_signals(
        self,
        prices_table: str,
        signals_table: str,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """
        ASOF JOIN prices with signals for backtesting.

        Joins each price row with the most recent signal at or before
        that timestamp. Prevents look-ahead bias in backtests.

        Args:
            prices_table: Table containing price data
            signals_table: Table containing signals
            symbol: Trading pair to filter
            start: Start time (optional)
            end: End time (optional)

        Returns:
            Polars DataFrame with price and signal columns.

        Example:
            # Get prices with signals for backtest
            df = await db.asof_join_signals("ohlcv", "signals", "BTC/USDT")
            # df has: timestamp, open, high, low, close, volume, signal, strength
        """
        ...

    # -------------------------------------------------------------------------
    # Schema Management
    # -------------------------------------------------------------------------

    async def create_tables(self) -> None:
        """
        Create required database tables if they don't exist.

        Tables:
        - ticks: Real-time tick data
        - ohlcv: OHLCV bar data
        - trades: Trade history
        - signals: Trading signals (for backtest)
        """
        ...

    async def health_check(self) -> bool:
        """
        Check database health.

        Returns:
            True if database is healthy and responding.
        """
        ...


# =============================================================================
# Abstract Base Class
# =============================================================================


class BaseTimeSeriesDB(ABC):
    """
    Abstract base class for TimeSeriesDB implementations.

    Provides common functionality:
    - Connection state tracking
    - Context manager support
    - Default implementations where possible

    Subclasses must implement all abstract methods.
    """

    def __init__(self) -> None:
        """Initialize database client."""
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Connection status."""
        return self._connected

    async def __aenter__(self) -> BaseTimeSeriesDB:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    # Abstract methods that must be implemented
    @abstractmethod
    async def connect(self) -> None:
        """Connect to database."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close database connection."""
        ...

    @abstractmethod
    async def ingest_tick(self, tick: Tick) -> None:
        """Ingest a tick."""
        ...

    @abstractmethod
    async def ingest_ticks(self, ticks: list[Tick]) -> None:
        """Ingest multiple ticks."""
        ...

    @abstractmethod
    async def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Tick]:
        """Get historical ticks."""
        ...

    @abstractmethod
    async def get_latest_tick(self, symbol: str) -> Tick | None:
        """Get latest tick."""
        ...

    @abstractmethod
    async def ingest_bar(self, bar: Bar) -> None:
        """Ingest a bar."""
        ...

    @abstractmethod
    async def ingest_bars(self, bars: list[Bar]) -> None:
        """Ingest multiple bars."""
        ...

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Bar]:
        """Get historical bars."""
        ...

    @abstractmethod
    async def get_bars_df(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Get bars as DataFrame."""
        ...

    @abstractmethod
    async def ingest_trade(self, trade: TradeRecord) -> None:
        """Ingest a trade record."""
        ...

    @abstractmethod
    async def get_trades(
        self,
        symbol: str | None = None,
        strategy: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[TradeRecord]:
        """Get trade history."""
        ...

    @abstractmethod
    async def asof_join_signals(
        self,
        prices_table: str,
        signals_table: str,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """ASOF JOIN for backtest."""
        ...

    @abstractmethod
    async def create_tables(self) -> None:
        """Create database tables."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check database health."""
        ...
