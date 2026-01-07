"""
BacktestDataClient: Historical data replay for backtesting.

Replays historical market data synchronized with the backtest clock.
Enables running the same strategy code in backtest and live trading.

Features:
    - Event-driven data replay (ticks, bars, orderbooks)
    - Synchronization with Clock.BACKTEST mode
    - Support for multiple data sources (CSV, Parquet, database)
    - Look-ahead bias prevention
    - Multi-symbol support

Design inspired by:
    - NautilusTrader BacktestDataClient
    - hftbacktest data replay

See: https://github.com/windoliver/libra/issues/33
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from libra.clients.data_client import BaseDataClient, DataNotAvailableError, Instrument
from libra.gateways.protocol import OrderBook, Tick
from libra.strategies.protocol import Bar


if TYPE_CHECKING:
    from libra.core.clock import Clock


# =============================================================================
# Data Source Protocols
# =============================================================================


class DataSource:
    """
    Base class for backtest data sources.

    Implement this interface to support different data formats:
    - CSVDataSource: Load from CSV files
    - ParquetDataSource: Load from Parquet files
    - DatabaseDataSource: Load from databases
    """

    async def load_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        """Load historical bars."""
        raise NotImplementedError

    async def load_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Tick]:
        """Load historical ticks."""
        raise NotImplementedError


class CSVDataSource(DataSource):
    """
    Load data from CSV files.

    Expected file format:
        bars: timestamp,open,high,low,close,volume
        ticks: timestamp,bid,ask,last,volume

    File naming convention:
        {data_dir}/{symbol.replace('/', '_')}_{timeframe}.csv
        e.g., data/BTC_USDT_1h.csv

    Examples:
        source = CSVDataSource("./data")
        bars = await source.load_bars("BTC/USDT", "1h", start, end)
    """

    def __init__(self, data_dir: str | Path) -> None:
        """
        Initialize CSV data source.

        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)

    async def load_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        """Load bars from CSV file."""
        # Convert symbol format: BTC/USDT -> BTC_USDT
        symbol_safe = symbol.replace("/", "_")
        filepath = self.data_dir / f"{symbol_safe}_{timeframe}.csv"

        if not filepath.exists():
            raise DataNotAvailableError(f"File not found: {filepath}")

        bars: list[Bar] = []

        # Use polars for fast CSV reading if available
        try:
            import polars as pl

            df = pl.read_csv(filepath)

            # Ensure required columns
            required = ["timestamp", "open", "high", "low", "close", "volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise DataNotAvailableError(f"Missing columns: {missing}")

            # Convert timestamp to datetime and filter
            if df["timestamp"].dtype == pl.Utf8:
                df = df.with_columns(pl.col("timestamp").str.to_datetime())
            elif df["timestamp"].dtype == pl.Int64:
                # Assume milliseconds
                df = df.with_columns(
                    pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("timestamp")
                )

            # Filter by date range
            df = df.filter(
                (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
            ).sort("timestamp")

            # Convert to Bar objects
            for row in df.iter_rows(named=True):
                bar = Bar(
                    symbol=symbol,
                    timestamp_ns=int(row["timestamp"].timestamp() * 1_000_000_000),
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume=Decimal(str(row["volume"])),
                    timeframe=timeframe,
                )
                bars.append(bar)

        except ImportError:
            # Fallback to standard CSV
            import csv

            with open(filepath) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse timestamp
                    ts = row["timestamp"]
                    if ts.isdigit():
                        # Milliseconds
                        dt = datetime.fromtimestamp(int(ts) / 1000)
                    else:
                        # ISO format
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))

                    # Filter by range
                    if dt < start or dt > end:
                        continue

                    bar = Bar(
                        symbol=symbol,
                        timestamp_ns=int(dt.timestamp() * 1_000_000_000),
                        open=Decimal(row["open"]),
                        high=Decimal(row["high"]),
                        low=Decimal(row["low"]),
                        close=Decimal(row["close"]),
                        volume=Decimal(row["volume"]),
                        timeframe=timeframe,
                    )
                    bars.append(bar)

            # Sort by timestamp
            bars.sort(key=lambda b: b.timestamp_ns)

        return bars

    async def load_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Tick]:
        """Load ticks from CSV file."""
        symbol_safe = symbol.replace("/", "_")
        filepath = self.data_dir / f"{symbol_safe}_ticks.csv"

        if not filepath.exists():
            raise DataNotAvailableError(f"File not found: {filepath}")

        ticks: list[Tick] = []

        try:
            import polars as pl

            df = pl.read_csv(filepath)

            # Convert and filter
            if df["timestamp"].dtype == pl.Utf8:
                df = df.with_columns(pl.col("timestamp").str.to_datetime())
            elif df["timestamp"].dtype == pl.Int64:
                df = df.with_columns(
                    pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("timestamp")
                )

            df = df.filter(
                (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
            ).sort("timestamp")

            for row in df.iter_rows(named=True):
                tick = Tick(
                    symbol=symbol,
                    timestamp_ns=int(row["timestamp"].timestamp() * 1_000_000_000),
                    bid=Decimal(str(row.get("bid", row.get("last", 0)))),
                    ask=Decimal(str(row.get("ask", row.get("last", 0)))),
                    last=Decimal(str(row.get("last", row.get("price", 0)))),
                    last_size=Decimal(str(row.get("volume", row.get("size", 0))))
                    if "volume" in row or "size" in row
                    else None,
                )
                ticks.append(tick)

        except ImportError:
            import csv

            with open(filepath) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row["timestamp"]
                    if ts.isdigit():
                        dt = datetime.fromtimestamp(int(ts) / 1000)
                    else:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))

                    if dt < start or dt > end:
                        continue

                    tick = Tick(
                        symbol=symbol,
                        timestamp_ns=int(dt.timestamp() * 1_000_000_000),
                        bid=Decimal(row.get("bid", row.get("last", "0"))),
                        ask=Decimal(row.get("ask", row.get("last", "0"))),
                        last=Decimal(row.get("last", row.get("price", "0"))),
                    )
                    ticks.append(tick)

            ticks.sort(key=lambda t: t.timestamp_ns)

        return ticks


class InMemoryDataSource(DataSource):
    """
    In-memory data source for testing.

    Useful for unit tests or when data is already loaded in memory.

    Examples:
        source = InMemoryDataSource()
        source.add_bars("BTC/USDT", "1h", bars)
        client = BacktestDataClient(source, clock)
    """

    def __init__(self) -> None:
        self._bars: dict[tuple[str, str], list[Bar]] = {}  # (symbol, timeframe) -> bars
        self._ticks: dict[str, list[Tick]] = {}  # symbol -> ticks

    def add_bars(self, symbol: str, timeframe: str, bars: list[Bar]) -> None:
        """Add bars to the data source."""
        key = (symbol, timeframe)
        if key not in self._bars:
            self._bars[key] = []
        self._bars[key].extend(bars)
        self._bars[key].sort(key=lambda b: b.timestamp_ns)

    def add_ticks(self, symbol: str, ticks: list[Tick]) -> None:
        """Add ticks to the data source."""
        if symbol not in self._ticks:
            self._ticks[symbol] = []
        self._ticks[symbol].extend(ticks)
        self._ticks[symbol].sort(key=lambda t: t.timestamp_ns)

    async def load_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        """Load bars from memory."""
        key = (symbol, timeframe)
        if key not in self._bars:
            raise DataNotAvailableError(f"No data for {symbol} {timeframe}")

        start_ns = int(start.timestamp() * 1_000_000_000)
        end_ns = int(end.timestamp() * 1_000_000_000)

        return [
            b for b in self._bars[key] if start_ns <= b.timestamp_ns <= end_ns
        ]

    async def load_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Tick]:
        """Load ticks from memory."""
        if symbol not in self._ticks:
            raise DataNotAvailableError(f"No tick data for {symbol}")

        start_ns = int(start.timestamp() * 1_000_000_000)
        end_ns = int(end.timestamp() * 1_000_000_000)

        return [
            t for t in self._ticks[symbol] if start_ns <= t.timestamp_ns <= end_ns
        ]


# =============================================================================
# BacktestDataClient
# =============================================================================


class BacktestDataClient(BaseDataClient):
    """
    Data client for backtesting with historical data replay.

    Replays historical market data synchronized with a backtest clock.
    The same strategy code works identically in backtest and live trading.

    Features:
        - Chronological event replay
        - Multi-symbol support
        - Bar and tick level data
        - Synthetic order book generation (from ticks)
        - Look-ahead bias prevention

    Examples:
        # Create client with CSV data
        source = CSVDataSource("./data")
        clock = Clock(ClockMode.BACKTEST)
        client = BacktestDataClient(source, clock)

        # Load data and run backtest
        await client.connect()
        await client.subscribe_bars("BTC/USDT", "1h")

        async for bar in client.stream_bars():
            signal = strategy.on_bar(bar)
            if signal:
                await exec_client.submit_order(...)

    Thread Safety:
        Not thread-safe. Use from a single async context.

    Performance:
        - Pre-loads data on subscribe (faster iteration)
        - Uses sorted event queue for multi-symbol replay
        - Polars for fast CSV/Parquet loading
    """

    def __init__(
        self,
        data_source: DataSource,
        clock: Clock,
        name: str = "backtest-data",
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize backtest data client.

        Args:
            data_source: Source for historical data (CSV, Parquet, etc.)
            clock: Clock instance (must be in BACKTEST mode)
            name: Client identifier
            config: Optional configuration
        """
        super().__init__(name, config)
        self.data_source = data_source
        self.clock = clock

        # Event queues for replay
        self._bar_events: list[Bar] = []
        self._tick_events: list[Tick] = []
        self._orderbook_events: list[OrderBook] = []

        # Replay state
        self._bar_index = 0
        self._tick_index = 0
        self._orderbook_index = 0

        # Streaming control
        self._streaming = False
        self._stop_event = asyncio.Event()

        # Pre-loaded data ranges
        self._data_start: datetime | None = None
        self._data_end: datetime | None = None

    def configure_range(self, start: datetime, end: datetime) -> None:
        """
        Configure the backtest date range.

        Data will be loaded for this range when subscribing.

        Args:
            start: Backtest start time
            end: Backtest end time
        """
        self._data_start = start
        self._data_end = end

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect (prepare for data loading)."""
        self._connected = True
        self._streaming = False
        self._stop_event.clear()

    async def disconnect(self) -> None:
        """Disconnect and clean up."""
        self._streaming = False
        self._stop_event.set()
        self._connected = False

        # Clear data
        self._bar_events.clear()
        self._tick_events.clear()
        self._orderbook_events.clear()

    # -------------------------------------------------------------------------
    # Subscriptions (load data)
    # -------------------------------------------------------------------------

    async def subscribe_ticks(self, symbol: str) -> None:
        """Subscribe to tick data (loads historical ticks)."""
        if not self._data_start or not self._data_end:
            raise ValueError("Configure date range first via configure_range()")

        self._subscribed_ticks.add(symbol)

        # Load tick data
        ticks = await self.data_source.load_ticks(
            symbol, self._data_start, self._data_end
        )
        self._tick_events.extend(ticks)
        self._tick_events.sort(key=lambda t: t.timestamp_ns)

    async def subscribe_bars(self, symbol: str, timeframe: str) -> None:
        """Subscribe to bar data (loads historical bars)."""
        if not self._data_start or not self._data_end:
            raise ValueError("Configure date range first via configure_range()")

        if symbol not in self._subscribed_bars:
            self._subscribed_bars[symbol] = set()
        self._subscribed_bars[symbol].add(timeframe)

        # Load bar data
        bars = await self.data_source.load_bars(
            symbol, timeframe, self._data_start, self._data_end
        )
        self._bar_events.extend(bars)
        self._bar_events.sort(key=lambda b: b.timestamp_ns)

    async def subscribe_orderbook(self, symbol: str, depth: int = 10) -> None:
        """Subscribe to order book (synthetic from ticks in backtest)."""
        self._subscribed_orderbooks.add(symbol)
        # Note: Order books are synthetically generated from ticks during replay

    async def unsubscribe_ticks(self, symbol: str) -> None:
        """Unsubscribe from tick data."""
        self._subscribed_ticks.discard(symbol)
        # Remove events for this symbol
        self._tick_events = [t for t in self._tick_events if t.symbol != symbol]

    async def unsubscribe_bars(self, symbol: str, timeframe: str) -> None:
        """Unsubscribe from bar data."""
        if symbol in self._subscribed_bars:
            self._subscribed_bars[symbol].discard(timeframe)
            if not self._subscribed_bars[symbol]:
                del self._subscribed_bars[symbol]
        # Remove events for this symbol/timeframe
        self._bar_events = [
            b for b in self._bar_events
            if not (b.symbol == symbol and b.timeframe == timeframe)
        ]

    async def unsubscribe_orderbook(self, symbol: str) -> None:
        """Unsubscribe from order book."""
        self._subscribed_orderbooks.discard(symbol)

    # -------------------------------------------------------------------------
    # Data Streams (replay)
    # -------------------------------------------------------------------------

    async def stream_ticks(self) -> AsyncIterator[Tick]:
        """
        Stream historical tick data synchronized with backtest clock.

        Yields ticks in chronological order, advancing the clock as needed.
        """
        self._streaming = True
        self._tick_index = 0

        while self._tick_index < len(self._tick_events) and self._streaming:
            tick = self._tick_events[self._tick_index]

            # Wait for clock to reach tick time
            while self.clock.timestamp_ns() < tick.timestamp_ns:
                if not self._streaming:
                    return
                # Advance clock (in backtest mode, this is controlled externally)
                await asyncio.sleep(0.0001)  # Yield to event loop

            self._tick_index += 1
            yield tick

    async def stream_bars(self) -> AsyncIterator[Bar]:
        """
        Stream historical bar data synchronized with backtest clock.

        Yields bars in chronological order, advancing the clock as needed.
        This is the primary method for bar-based backtesting.
        """
        self._streaming = True
        self._bar_index = 0

        while self._bar_index < len(self._bar_events) and self._streaming:
            bar = self._bar_events[self._bar_index]

            # Wait for clock to reach bar time
            while self.clock.timestamp_ns() < bar.timestamp_ns:
                if not self._streaming:
                    return
                await asyncio.sleep(0.0001)

            self._bar_index += 1
            yield bar

    async def stream_orderbooks(self) -> AsyncIterator[OrderBook]:
        """
        Stream synthetic order books (generated from ticks).

        In backtesting, order books are synthetically created from tick data.
        """
        self._streaming = True

        for tick in self._tick_events:
            if tick.symbol not in self._subscribed_orderbooks:
                continue

            while self.clock.timestamp_ns() < tick.timestamp_ns:
                if not self._streaming:
                    return
                await asyncio.sleep(0.0001)

            # Create synthetic order book from tick
            orderbook = self._create_synthetic_orderbook(tick)
            yield orderbook

    def _create_synthetic_orderbook(self, tick: Tick, depth: int = 5) -> OrderBook:
        """Create synthetic order book from tick data."""
        spread = tick.spread if tick.spread > 0 else tick.last * Decimal("0.0001")
        tick_size = spread / 10

        # Generate synthetic levels
        bids = []
        asks = []
        for i in range(depth):
            bid_price = tick.bid - (tick_size * i)
            ask_price = tick.ask + (tick_size * i)
            size = Decimal("1.0") / (i + 1)  # Decreasing size at worse prices
            bids.append((bid_price, size))
            asks.append((ask_price, size))

        return OrderBook(
            symbol=tick.symbol,
            bids=bids,
            asks=asks,
            timestamp_ns=tick.timestamp_ns,
        )

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
        """Request historical bars from data source."""
        bars = await self.data_source.load_bars(symbol, timeframe, start, end)
        if limit:
            bars = bars[:limit]
        return bars

    async def request_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Tick]:
        """Request historical ticks from data source."""
        ticks = await self.data_source.load_ticks(symbol, start, end)
        if limit:
            ticks = ticks[:limit]
        return ticks

    # -------------------------------------------------------------------------
    # Instrument Information
    # -------------------------------------------------------------------------

    async def get_instruments(self) -> list[Instrument]:
        """Get available instruments (from subscribed symbols)."""
        instruments = []
        for symbol in self._subscribed_ticks | set(self._subscribed_bars.keys()):
            parts = symbol.split("/")
            if len(parts) == 2:
                instruments.append(
                    Instrument(
                        symbol=symbol,
                        base=parts[0],
                        quote=parts[1],
                        exchange="backtest",
                        lot_size=Decimal("0.00001"),
                        tick_size=Decimal("0.01"),
                        contract_type="spot",
                        is_active=True,
                    )
                )
        return instruments

    async def get_instrument(self, symbol: str) -> Instrument | None:
        """Get instrument info."""
        instruments = await self.get_instruments()
        for inst in instruments:
            if inst.symbol == symbol:
                return inst
        return None

    # -------------------------------------------------------------------------
    # Snapshots
    # -------------------------------------------------------------------------

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get current order book (from latest tick)."""
        # Find latest tick for symbol
        for tick in reversed(self._tick_events[:self._tick_index]):
            if tick.symbol == symbol:
                return self._create_synthetic_orderbook(tick, depth)

        raise DataNotAvailableError(f"No tick data for {symbol}")

    async def get_ticker(self, symbol: str) -> Tick:
        """Get current ticker (from latest tick)."""
        for tick in reversed(self._tick_events[:self._tick_index]):
            if tick.symbol == symbol:
                return tick

        # If no ticks, try to create from latest bar
        for bar in reversed(self._bar_events[:self._bar_index]):
            if bar.symbol == symbol:
                return Tick(
                    symbol=symbol,
                    bid=bar.close,
                    ask=bar.close,
                    last=bar.close,
                    timestamp_ns=bar.timestamp_ns,
                )

        raise DataNotAvailableError(f"No data for {symbol}")

    # -------------------------------------------------------------------------
    # Backtest Control
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset replay state for a new backtest run."""
        self._bar_index = 0
        self._tick_index = 0
        self._orderbook_index = 0
        self._streaming = False
        self._stop_event.clear()

    def stop(self) -> None:
        """Stop streaming."""
        self._streaming = False
        self._stop_event.set()

    @property
    def progress(self) -> float:
        """Get replay progress (0.0 to 1.0)."""
        total_bars = len(self._bar_events)
        if total_bars == 0:
            return 1.0
        return self._bar_index / total_bars

    @property
    def remaining_bars(self) -> int:
        """Number of bars remaining to replay."""
        return len(self._bar_events) - self._bar_index

    @property
    def remaining_ticks(self) -> int:
        """Number of ticks remaining to replay."""
        return len(self._tick_events) - self._tick_index
