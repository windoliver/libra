"""
Async QuestDB Client for LIBRA.

High-performance time-series database client with:
- ILP ingestion (5M+ rows/sec, GIL-released during flush)
- asyncpg queries (native async, binary protocol)
- Polars DataFrame integration
- ASOF JOIN support for backtesting

Architecture:
- Ingestion: ILP via questdb.Sender (sync, but run in executor)
- Queries: asyncpg with connection pooling (native async)

See:
- https://questdb.com/docs/clients/ingest-python/
- https://questdb.com/docs/query/pgwire/python/
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import polars as pl

from libra.data.config import QuestDBConfig
from libra.data.protocol import BaseTimeSeriesDB, TradeRecord
from libra.gateways.protocol import Tick
from libra.strategies.protocol import Bar


if TYPE_CHECKING:
    import asyncpg
    from questdb.ingress import Sender


logger = logging.getLogger(__name__)


# =============================================================================
# SQL Schema Definitions
# =============================================================================

CREATE_TICKS_TABLE = """
CREATE TABLE IF NOT EXISTS ticks (
    timestamp TIMESTAMP,
    symbol SYMBOL,
    exchange SYMBOL,
    bid DOUBLE,
    ask DOUBLE,
    last DOUBLE,
    bid_size DOUBLE,
    ask_size DOUBLE,
    volume_24h DOUBLE
) TIMESTAMP(timestamp) PARTITION BY DAY WAL
DEDUP UPSERT KEYS(timestamp, symbol);
"""

CREATE_OHLCV_TABLE = """
CREATE TABLE IF NOT EXISTS ohlcv (
    timestamp TIMESTAMP,
    symbol SYMBOL,
    timeframe SYMBOL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    trades INT
) TIMESTAMP(timestamp) PARTITION BY MONTH WAL
DEDUP UPSERT KEYS(timestamp, symbol, timeframe);
"""

CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    timestamp TIMESTAMP,
    trade_id SYMBOL,
    order_id SYMBOL,
    symbol SYMBOL,
    exchange SYMBOL,
    side SYMBOL,
    amount DOUBLE,
    price DOUBLE,
    fee DOUBLE,
    fee_currency SYMBOL,
    strategy SYMBOL,
    realized_pnl DOUBLE
) TIMESTAMP(timestamp) PARTITION BY MONTH WAL;
"""

CREATE_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    timestamp TIMESTAMP,
    symbol SYMBOL,
    signal_type SYMBOL,
    strength DOUBLE,
    price DOUBLE,
    strategy SYMBOL
) TIMESTAMP(timestamp) PARTITION BY DAY WAL;
"""


# =============================================================================
# AsyncQuestDBClient Implementation
# =============================================================================


class AsyncQuestDBClient(BaseTimeSeriesDB):
    """
    Async QuestDB client with ILP ingestion and asyncpg queries.

    Performance characteristics:
    - Ingestion: 1-5M rows/sec (ILP, network-bound)
    - Queries: <10ms for typical OHLCV queries
    - Connection pool: 2-10 connections (configurable)

    Thread safety:
    - asyncpg pool handles concurrent queries
    - ILP sender accessed via executor (thread-safe)

    Examples:
        config = QuestDBConfig(host="localhost")

        async with AsyncQuestDBClient(config) as db:
            # Ingest ticks
            await db.ingest_tick(tick)

            # Query bars
            bars = await db.get_bars("BTC/USDT", "1h", start, end)

            # Get DataFrame for analysis
            df = await db.get_bars_df("BTC/USDT", "1h", start, end)
    """

    def __init__(self, config: QuestDBConfig) -> None:
        """
        Initialize QuestDB client.

        Args:
            config: QuestDB connection configuration.
        """
        super().__init__()
        self._config = config
        self._pool: asyncpg.Pool | None = None
        self._sender: Sender | None = None
        self._sender_lock = asyncio.Lock()

    @property
    def config(self) -> QuestDBConfig:
        """Get configuration."""
        return self._config

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Initialize database connections.

        Creates:
        - asyncpg connection pool for queries
        - ILP sender for ingestion
        """
        if self._connected:
            return

        try:
            # Import here to make questdb optional dependency
            import asyncpg
            from questdb.ingress import Sender

            # Create asyncpg pool for queries
            self._pool = await asyncpg.create_pool(
                self._config.pg_dsn,
                min_size=self._config.pool_min_size,
                max_size=self._config.pool_max_size,
                command_timeout=self._config.command_timeout,
            )

            # Create ILP sender for ingestion
            self._sender = Sender.from_conf(self._config.ilp_conf)

            self._connected = True
            logger.info(
                "Connected to QuestDB at %s (PG: %d, ILP: %d)",
                self._config.host,
                self._config.pg_port,
                self._config.ilp_port,
            )

        except Exception as e:
            logger.exception("Failed to connect to QuestDB")
            raise ConnectionError(f"QuestDB connection failed: {e}") from e

    async def close(self) -> None:
        """Close all database connections."""
        if not self._connected:
            return

        try:
            if self._sender:
                # Flush and close sender
                await self._run_sender_sync(lambda s: s.flush())
                self._sender.close()
                self._sender = None

            if self._pool:
                await self._pool.close()
                self._pool = None

            self._connected = False
            logger.info("Disconnected from QuestDB")

        except Exception:
            logger.exception("Error closing QuestDB connections")

    async def _run_sender_sync(self, func: Any) -> Any:
        """
        Run a sender operation in executor.

        ILP sender is synchronous but releases GIL during flush,
        so we run it in a thread pool to not block the event loop.
        """
        loop = asyncio.get_event_loop()
        async with self._sender_lock:
            return await loop.run_in_executor(None, func, self._sender)

    # -------------------------------------------------------------------------
    # Tick Data
    # -------------------------------------------------------------------------

    async def ingest_tick(self, tick: Tick) -> None:
        """Ingest a single tick via ILP."""
        if not self._sender:
            raise ConnectionError("Not connected to QuestDB")

        def _ingest(sender: Sender) -> None:
            from questdb.ingress import TimestampNanos

            sender.row(
                "ticks",
                symbols={
                    "symbol": tick.symbol,
                    "exchange": "unknown",  # TODO: add exchange to Tick
                },
                columns={
                    "bid": float(tick.bid),
                    "ask": float(tick.ask),
                    "last": float(tick.last),
                    "bid_size": float(tick.bid_size) if tick.bid_size else None,
                    "ask_size": float(tick.ask_size) if tick.ask_size else None,
                    "volume_24h": float(tick.volume_24h) if tick.volume_24h else None,
                },
                at=TimestampNanos(tick.timestamp_ns),
            )

        await self._run_sender_sync(_ingest)

    async def ingest_ticks(self, ticks: list[Tick]) -> None:
        """Batch ingest multiple ticks."""
        if not self._sender or not ticks:
            return

        def _ingest(sender: Sender) -> None:
            from questdb.ingress import TimestampNanos

            for tick in ticks:
                sender.row(
                    "ticks",
                    symbols={
                        "symbol": tick.symbol,
                        "exchange": "unknown",
                    },
                    columns={
                        "bid": float(tick.bid),
                        "ask": float(tick.ask),
                        "last": float(tick.last),
                        "bid_size": float(tick.bid_size) if tick.bid_size else None,
                        "ask_size": float(tick.ask_size) if tick.ask_size else None,
                        "volume_24h": float(tick.volume_24h) if tick.volume_24h else None,
                    },
                    at=TimestampNanos(tick.timestamp_ns),
                )
            sender.flush()

        await self._run_sender_sync(_ingest)

    async def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Tick]:
        """Query historical tick data."""
        if not self._pool:
            raise ConnectionError("Not connected to QuestDB")

        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
            SELECT timestamp, symbol, bid, ask, last, bid_size, ask_size, volume_24h
            FROM ticks
            WHERE symbol = $1
              AND timestamp >= $2
              AND timestamp <= $3
            ORDER BY timestamp
            {limit_clause}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, start, end)

        return [
            Tick(
                symbol=row["symbol"],
                bid=Decimal(str(row["bid"])),
                ask=Decimal(str(row["ask"])),
                last=Decimal(str(row["last"])),
                timestamp_ns=int(row["timestamp"].timestamp() * 1_000_000_000),
                bid_size=Decimal(str(row["bid_size"])) if row["bid_size"] else None,
                ask_size=Decimal(str(row["ask_size"])) if row["ask_size"] else None,
                volume_24h=Decimal(str(row["volume_24h"])) if row["volume_24h"] else None,
            )
            for row in rows
        ]

    async def get_latest_tick(self, symbol: str) -> Tick | None:
        """Get the most recent tick for a symbol."""
        if not self._pool:
            raise ConnectionError("Not connected to QuestDB")

        query = """
            SELECT timestamp, symbol, bid, ask, last, bid_size, ask_size, volume_24h
            FROM ticks
            WHERE symbol = $1
            ORDER BY timestamp DESC
            LIMIT 1
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, symbol)

        if not row:
            return None

        return Tick(
            symbol=row["symbol"],
            bid=Decimal(str(row["bid"])),
            ask=Decimal(str(row["ask"])),
            last=Decimal(str(row["last"])),
            timestamp_ns=int(row["timestamp"].timestamp() * 1_000_000_000),
            bid_size=Decimal(str(row["bid_size"])) if row["bid_size"] else None,
            ask_size=Decimal(str(row["ask_size"])) if row["ask_size"] else None,
            volume_24h=Decimal(str(row["volume_24h"])) if row["volume_24h"] else None,
        )

    # -------------------------------------------------------------------------
    # Bar Data (OHLCV)
    # -------------------------------------------------------------------------

    async def ingest_bar(self, bar: Bar) -> None:
        """Ingest a single OHLCV bar."""
        if not self._sender:
            raise ConnectionError("Not connected to QuestDB")

        def _ingest(sender: Sender) -> None:
            from questdb.ingress import TimestampNanos

            sender.row(
                "ohlcv",
                symbols={
                    "symbol": bar.symbol,
                    "timeframe": bar.timeframe,
                },
                columns={
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "trades": bar.trades,
                },
                at=TimestampNanos(bar.timestamp_ns),
            )

        await self._run_sender_sync(_ingest)

    async def ingest_bars(self, bars: list[Bar]) -> None:
        """Batch ingest multiple bars."""
        if not self._sender or not bars:
            return

        def _ingest(sender: Sender) -> None:
            from questdb.ingress import TimestampNanos

            for bar in bars:
                sender.row(
                    "ohlcv",
                    symbols={
                        "symbol": bar.symbol,
                        "timeframe": bar.timeframe,
                    },
                    columns={
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                        "trades": bar.trades,
                    },
                    at=TimestampNanos(bar.timestamp_ns),
                )
            sender.flush()

        await self._run_sender_sync(_ingest)

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Bar]:
        """Query historical bar data."""
        if not self._pool:
            raise ConnectionError("Not connected to QuestDB")

        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
            SELECT timestamp, symbol, timeframe, open, high, low, close, volume, trades
            FROM ohlcv
            WHERE symbol = $1
              AND timeframe = $2
              AND timestamp >= $3
              AND timestamp <= $4
            ORDER BY timestamp
            {limit_clause}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, timeframe, start, end)

        return [
            Bar(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                timestamp_ns=int(row["timestamp"].timestamp() * 1_000_000_000),
                open=Decimal(str(row["open"])),
                high=Decimal(str(row["high"])),
                low=Decimal(str(row["low"])),
                close=Decimal(str(row["close"])),
                volume=Decimal(str(row["volume"])),
                trades=row["trades"],
            )
            for row in rows
        ]

    async def get_bars_df(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Query bars as Polars DataFrame for efficient analytics."""
        if not self._pool:
            raise ConnectionError("Not connected to QuestDB")

        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = $1
              AND timeframe = $2
              AND timestamp >= $3
              AND timestamp <= $4
            ORDER BY timestamp
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, timeframe, start, end)

        # Convert to Polars DataFrame
        if not rows:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                }
            )

        return pl.DataFrame(
            {
                "timestamp": [row["timestamp"] for row in rows],
                "open": [float(row["open"]) for row in rows],
                "high": [float(row["high"]) for row in rows],
                "low": [float(row["low"]) for row in rows],
                "close": [float(row["close"]) for row in rows],
                "volume": [float(row["volume"]) for row in rows],
            }
        )

    # -------------------------------------------------------------------------
    # Trade History
    # -------------------------------------------------------------------------

    async def ingest_trade(self, trade: TradeRecord) -> None:
        """Store a trade record."""
        if not self._sender:
            raise ConnectionError("Not connected to QuestDB")

        def _ingest(sender: Sender) -> None:
            from questdb.ingress import TimestampNanos

            sender.row(
                "trades",
                symbols={
                    "trade_id": trade.trade_id,
                    "order_id": trade.order_id,
                    "symbol": trade.symbol,
                    "exchange": trade.exchange,
                    "side": trade.side,
                    "fee_currency": trade.fee_currency,
                    "strategy": trade.strategy or "unknown",
                },
                columns={
                    "amount": float(trade.amount),
                    "price": float(trade.price),
                    "fee": float(trade.fee),
                    "realized_pnl": float(trade.realized_pnl) if trade.realized_pnl else None,
                },
                at=TimestampNanos(trade.timestamp_ns),
            )

        await self._run_sender_sync(_ingest)

    async def get_trades(
        self,
        symbol: str | None = None,
        strategy: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[TradeRecord]:
        """Query trade history."""
        if not self._pool:
            raise ConnectionError("Not connected to QuestDB")

        # Build dynamic query
        conditions = []
        params: list[Any] = []
        param_idx = 1

        if symbol:
            conditions.append(f"symbol = ${param_idx}")
            params.append(symbol)
            param_idx += 1

        if strategy:
            conditions.append(f"strategy = ${param_idx}")
            params.append(strategy)
            param_idx += 1

        if start:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(start)
            param_idx += 1

        if end:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(end)
            param_idx += 1

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
            SELECT timestamp, trade_id, order_id, symbol, exchange,
                   side, amount, price, fee, fee_currency, strategy, realized_pnl
            FROM trades
            {where_clause}
            ORDER BY timestamp DESC
            {limit_clause}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [
            TradeRecord(
                trade_id=row["trade_id"],
                order_id=row["order_id"],
                symbol=row["symbol"],
                exchange=row["exchange"],
                side=row["side"],
                amount=Decimal(str(row["amount"])),
                price=Decimal(str(row["price"])),
                fee=Decimal(str(row["fee"])),
                fee_currency=row["fee_currency"],
                timestamp_ns=int(row["timestamp"].timestamp() * 1_000_000_000),
                strategy=row["strategy"],
                realized_pnl=Decimal(str(row["realized_pnl"])) if row["realized_pnl"] else None,
            )
            for row in rows
        ]

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

        Uses QuestDB's native ASOF JOIN to match each price row with
        the most recent signal at or before that timestamp.
        """
        if not self._pool:
            raise ConnectionError("Not connected to QuestDB")

        # Build time filter
        time_filters = []
        params: list[Any] = [symbol]
        param_idx = 2

        if start:
            time_filters.append(f"p.timestamp >= ${param_idx}")
            params.append(start)
            param_idx += 1

        if end:
            time_filters.append(f"p.timestamp <= ${param_idx}")
            params.append(end)
            param_idx += 1

        time_clause = f"AND {' AND '.join(time_filters)}" if time_filters else ""

        query = f"""
            SELECT
                p.timestamp,
                p.open,
                p.high,
                p.low,
                p.close,
                p.volume,
                s.signal_type,
                s.strength
            FROM {prices_table} p
            ASOF JOIN {signals_table} s ON (p.symbol = s.symbol)
            WHERE p.symbol = $1
            {time_clause}
            ORDER BY p.timestamp
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        if not rows:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                    "signal_type": pl.Utf8,
                    "strength": pl.Float64,
                }
            )

        return pl.DataFrame(
            {
                "timestamp": [row["timestamp"] for row in rows],
                "open": [float(row["open"]) for row in rows],
                "high": [float(row["high"]) for row in rows],
                "low": [float(row["low"]) for row in rows],
                "close": [float(row["close"]) for row in rows],
                "volume": [float(row["volume"]) for row in rows],
                "signal_type": [row["signal_type"] for row in rows],
                "strength": [float(row["strength"]) if row["strength"] else None for row in rows],
            }
        )

    # -------------------------------------------------------------------------
    # Schema Management
    # -------------------------------------------------------------------------

    async def create_tables(self) -> None:
        """Create required database tables."""
        if not self._pool:
            raise ConnectionError("Not connected to QuestDB")

        async with self._pool.acquire() as conn:
            await conn.execute(CREATE_TICKS_TABLE)
            await conn.execute(CREATE_OHLCV_TABLE)
            await conn.execute(CREATE_TRADES_TABLE)
            await conn.execute(CREATE_SIGNALS_TABLE)

        logger.info("Created QuestDB tables: ticks, ohlcv, trades, signals")

    async def health_check(self) -> bool:
        """Check database health with a simple query."""
        if not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception:
            logger.exception("Health check failed")
            return False

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    async def flush(self) -> None:
        """Flush pending ingestion data."""
        if self._sender:
            await self._run_sender_sync(lambda s: s.flush())

    async def count_rows(self, table: str) -> int:
        """Get row count for a table."""
        if not self._pool:
            raise ConnectionError("Not connected to QuestDB")

        async with self._pool.acquire() as conn:
            result = await conn.fetchval(f"SELECT count() FROM {table}")  # noqa: S608
            return int(result) if result else 0
