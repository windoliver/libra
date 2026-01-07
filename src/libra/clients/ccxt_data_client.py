"""
CCXTDataClient: Market data client using CCXT.

Provides real-time market data from 100+ exchanges via ccxt.pro.

Features:
    - Async WebSocket streaming (ticks, orderbooks, bars)
    - Historical data requests (OHLCV)
    - Instrument information
    - Automatic reconnection
    - orjson for fast JSON parsing

See: https://github.com/windoliver/libra/issues/33
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from libra.clients.data_client import (
    BaseDataClient,
    DataClientError,
    DataNotAvailableError,
    Instrument,
    SubscriptionError,
)
from libra.gateways.protocol import OrderBook, Tick
from libra.strategies.protocol import Bar


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


logger = logging.getLogger(__name__)


# =============================================================================
# CCXTDataClient
# =============================================================================


class CCXTDataClient(BaseDataClient):
    """
    Market data client using CCXT for 100+ exchanges.

    Uses ccxt.pro for native async WebSocket support.

    Supported exchanges (partial list):
        - binance, binanceusdm (futures)
        - bybit
        - okx
        - kraken
        - coinbase
        - kucoin
        - gate
        - huobi
        - ... and 100+ more

    Configuration:
        config = {
            "testnet": True,  # Use testnet/sandbox
            "options": {
                "defaultType": "future",  # spot, future, swap
            },
        }

    Example:
        async with CCXTDataClient("binance") as client:
            await client.subscribe_ticks("BTC/USDT")

            async for tick in client.stream_ticks():
                print(f"{tick.symbol}: {tick.last}")

    Note:
        No API keys required for market data (public endpoints).
    """

    def __init__(
        self,
        exchange_id: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize CCXT data client.

        Args:
            exchange_id: CCXT exchange ID (e.g., "binance", "bybit")
            config: Configuration options (testnet, defaultType, etc.)
        """
        super().__init__(name=f"{exchange_id}-data", config=config)
        self._exchange_id = exchange_id
        self._exchange: Any = None  # ccxt.pro exchange instance

        # Streaming
        self._tick_queue: asyncio.Queue[Tick] = asyncio.Queue(maxsize=10000)
        self._bar_queue: asyncio.Queue[Bar] = asyncio.Queue(maxsize=10000)
        self._orderbook_queue: asyncio.Queue[OrderBook] = asyncio.Queue(maxsize=1000)
        self._stream_tasks: list[asyncio.Task[Any]] = []
        self._stop_streaming = asyncio.Event()

        # Instrument cache
        self._instruments: dict[str, Instrument] = {}

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect to the exchange.

        Initializes CCXT exchange and loads markets.
        No authentication needed for market data.

        Raises:
            DataClientError: If connection fails.
        """
        if self._connected:
            logger.warning(f"{self.name}: Already connected")
            return

        try:
            # Import ccxt.pro (async version)
            import ccxt.pro as ccxtpro

            # Get exchange class
            exchange_class = getattr(ccxtpro, self._exchange_id, None)
            if exchange_class is None:
                raise DataClientError(f"Unknown exchange: {self._exchange_id}")

            # Build exchange options
            options: dict[str, Any] = {
                "enableRateLimit": True,
            }

            # Enable orjson for faster JSON parsing
            try:
                import orjson  # noqa: F401

                options["enableOrjson"] = True
                logger.debug(f"{self.name}: orjson enabled")
            except ImportError:
                pass

            # Configure testnet if requested
            if self._config.get("testnet", False):
                options["sandbox"] = True

            # Add user-provided options
            if "options" in self._config:
                options.update(self._config["options"])

            # Create exchange instance (no API keys for data)
            self._exchange = exchange_class({"options": options})

            # Load markets
            logger.info(f"{self.name}: Loading markets...")
            await self._exchange.load_markets()

            # Cache instrument info
            await self._load_instruments()

            self._connected = True
            logger.info(f"{self.name}: Connected, {len(self._instruments)} instruments")

        except Exception as e:
            raise DataClientError(f"Failed to connect: {e}") from e

    async def disconnect(self) -> None:
        """
        Disconnect from the exchange.

        Stops all streaming tasks and closes connections.
        """
        if not self._connected:
            return

        logger.info(f"{self.name}: Disconnecting...")

        # Signal streams to stop
        self._stop_streaming.set()

        # Cancel streaming tasks
        for task in self._stream_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._stream_tasks.clear()

        # Close exchange connection
        if self._exchange:
            try:
                await self._exchange.close()
            except Exception as e:
                logger.warning(f"{self.name}: Error closing: {e}")

        self._connected = False
        self._stop_streaming.clear()
        logger.info(f"{self.name}: Disconnected")

    # -------------------------------------------------------------------------
    # Subscriptions
    # -------------------------------------------------------------------------

    async def subscribe_ticks(self, symbol: str) -> None:
        """Subscribe to tick/quote data."""
        if not self._connected:
            raise DataClientError("Not connected")

        if symbol not in self._exchange.markets:
            raise SubscriptionError(f"Invalid symbol: {symbol}")

        if symbol in self._subscribed_ticks:
            return  # Already subscribed

        self._subscribed_ticks.add(symbol)
        logger.info(f"{self.name}: Subscribed to ticks: {symbol}")

        # Start ticker stream if not running
        self._ensure_ticker_stream()

    async def subscribe_bars(self, symbol: str, timeframe: str) -> None:
        """Subscribe to bar/OHLCV data."""
        if not self._connected:
            raise DataClientError("Not connected")

        if symbol not in self._exchange.markets:
            raise SubscriptionError(f"Invalid symbol: {symbol}")

        if timeframe not in self._exchange.timeframes:
            raise SubscriptionError(f"Invalid timeframe: {timeframe}")

        if symbol not in self._subscribed_bars:
            self._subscribed_bars[symbol] = set()
        self._subscribed_bars[symbol].add(timeframe)

        logger.info(f"{self.name}: Subscribed to bars: {symbol} {timeframe}")

        # Start OHLCV stream if not running
        self._ensure_ohlcv_stream()

    async def subscribe_orderbook(self, symbol: str, depth: int = 10) -> None:
        """Subscribe to order book updates."""
        if not self._connected:
            raise DataClientError("Not connected")

        if symbol not in self._exchange.markets:
            raise SubscriptionError(f"Invalid symbol: {symbol}")

        self._subscribed_orderbooks.add(symbol)
        logger.info(f"{self.name}: Subscribed to orderbook: {symbol}")

        # Start orderbook stream if not running
        self._ensure_orderbook_stream()

    async def unsubscribe_ticks(self, symbol: str) -> None:
        """Unsubscribe from tick data."""
        self._subscribed_ticks.discard(symbol)
        logger.info(f"{self.name}: Unsubscribed from ticks: {symbol}")

    async def unsubscribe_bars(self, symbol: str, timeframe: str) -> None:
        """Unsubscribe from bar data."""
        if symbol in self._subscribed_bars:
            self._subscribed_bars[symbol].discard(timeframe)
            if not self._subscribed_bars[symbol]:
                del self._subscribed_bars[symbol]

    async def unsubscribe_orderbook(self, symbol: str) -> None:
        """Unsubscribe from order book."""
        self._subscribed_orderbooks.discard(symbol)

    # -------------------------------------------------------------------------
    # Streaming
    # -------------------------------------------------------------------------

    def _ensure_ticker_stream(self) -> None:
        """Ensure ticker stream task is running."""
        if not any(t.get_name() == "ticker_stream" for t in self._stream_tasks):
            task = asyncio.create_task(self._stream_tickers_loop(), name="ticker_stream")
            self._stream_tasks.append(task)

    def _ensure_ohlcv_stream(self) -> None:
        """Ensure OHLCV stream task is running."""
        if not any(t.get_name() == "ohlcv_stream" for t in self._stream_tasks):
            task = asyncio.create_task(self._stream_ohlcv_loop(), name="ohlcv_stream")
            self._stream_tasks.append(task)

    def _ensure_orderbook_stream(self) -> None:
        """Ensure orderbook stream task is running."""
        if not any(t.get_name() == "orderbook_stream" for t in self._stream_tasks):
            task = asyncio.create_task(self._stream_orderbook_loop(), name="orderbook_stream")
            self._stream_tasks.append(task)

    async def _stream_tickers_loop(self) -> None:
        """Internal ticker streaming loop."""
        logger.info(f"{self.name}: Starting ticker stream")

        while not self._stop_streaming.is_set():
            try:
                if not self._subscribed_ticks:
                    await asyncio.sleep(0.1)
                    continue

                symbols = list(self._subscribed_ticks)
                tickers = await self._exchange.watch_tickers(symbols)

                for symbol, data in tickers.items():
                    tick = self._convert_ticker(symbol, data)
                    self._queue_put(self._tick_queue, tick)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"{self.name}: Ticker stream error: {e}")
                await asyncio.sleep(1)

        logger.info(f"{self.name}: Ticker stream stopped")

    async def _stream_ohlcv_loop(self) -> None:
        """Internal OHLCV streaming loop."""
        logger.info(f"{self.name}: Starting OHLCV stream")

        while not self._stop_streaming.is_set():
            try:
                if not self._subscribed_bars:
                    await asyncio.sleep(0.1)
                    continue

                # Watch OHLCV for each symbol/timeframe
                for symbol, timeframes in list(self._subscribed_bars.items()):
                    for timeframe in timeframes:
                        try:
                            ohlcv = await self._exchange.watch_ohlcv(symbol, timeframe)
                            for candle in ohlcv:
                                bar = self._convert_ohlcv(symbol, timeframe, candle)
                                self._queue_put(self._bar_queue, bar)
                        except Exception as e:
                            logger.warning(f"{self.name}: OHLCV error {symbol}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"{self.name}: OHLCV stream error: {e}")
                await asyncio.sleep(1)

        logger.info(f"{self.name}: OHLCV stream stopped")

    async def _stream_orderbook_loop(self) -> None:
        """Internal orderbook streaming loop."""
        logger.info(f"{self.name}: Starting orderbook stream")

        while not self._stop_streaming.is_set():
            try:
                if not self._subscribed_orderbooks:
                    await asyncio.sleep(0.1)
                    continue

                for symbol in list(self._subscribed_orderbooks):
                    try:
                        data = await self._exchange.watch_order_book(symbol)
                        orderbook = self._convert_orderbook(symbol, data)
                        self._queue_put(self._orderbook_queue, orderbook)
                    except Exception as e:
                        logger.warning(f"{self.name}: Orderbook error {symbol}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"{self.name}: Orderbook stream error: {e}")
                await asyncio.sleep(1)

        logger.info(f"{self.name}: Orderbook stream stopped")

    def _queue_put(self, queue: asyncio.Queue[Any], item: Any) -> None:
        """Put item in queue, dropping oldest if full."""
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
                queue.put_nowait(item)
            except asyncio.QueueEmpty:
                pass

    async def stream_ticks(self) -> AsyncIterator[Tick]:
        """Stream tick data for subscribed symbols."""
        while not self._stop_streaming.is_set():
            try:
                tick = await asyncio.wait_for(self._tick_queue.get(), timeout=1.0)
                yield tick
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def stream_bars(self) -> AsyncIterator[Bar]:
        """Stream bar data for subscribed symbols."""
        while not self._stop_streaming.is_set():
            try:
                bar = await asyncio.wait_for(self._bar_queue.get(), timeout=1.0)
                yield bar
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def stream_orderbooks(self) -> AsyncIterator[OrderBook]:
        """Stream orderbook data for subscribed symbols."""
        while not self._stop_streaming.is_set():
            try:
                ob = await asyncio.wait_for(self._orderbook_queue.get(), timeout=1.0)
                yield ob
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    # -------------------------------------------------------------------------
    # Historical Data
    # -------------------------------------------------------------------------

    async def request_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Bar]:
        """Request historical OHLCV bars."""
        if not self._connected:
            raise DataClientError("Not connected")

        try:
            since = int(start.timestamp() * 1000)
            until = int(end.timestamp() * 1000)

            bars: list[Bar] = []
            current_since = since

            while current_since < until:
                # Fetch batch
                ohlcv = await self._exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current_since,
                    limit=limit or 1000,
                )

                if not ohlcv:
                    break

                for candle in ohlcv:
                    bar = self._convert_ohlcv(symbol, timeframe, candle)
                    if bar.timestamp_ns <= until * 1_000_000:
                        bars.append(bar)

                # Move to next batch
                last_ts = ohlcv[-1][0]
                if last_ts <= current_since:
                    break
                current_since = last_ts + 1

                if limit and len(bars) >= limit:
                    break

            return bars[:limit] if limit else bars

        except Exception as e:
            raise DataNotAvailableError(f"Failed to fetch bars: {e}") from e

    async def request_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Tick]:
        """Request historical trades (as ticks)."""
        if not self._connected:
            raise DataClientError("Not connected")

        try:
            since = int(start.timestamp() * 1000)

            trades = await self._exchange.fetch_trades(
                symbol,
                since=since,
                limit=limit or 1000,
            )

            ticks: list[Tick] = []
            for trade in trades:
                ts = trade.get("timestamp", 0) or 0
                if ts > int(end.timestamp() * 1000):
                    break

                tick = Tick(
                    symbol=symbol,
                    bid=Decimal(str(trade.get("price", 0))),
                    ask=Decimal(str(trade.get("price", 0))),
                    last=Decimal(str(trade.get("price", 0))),
                    timestamp_ns=int(ts * 1_000_000),
                    last_size=Decimal(str(trade.get("amount", 0))),
                )
                ticks.append(tick)

            return ticks[:limit] if limit else ticks

        except Exception as e:
            raise DataNotAvailableError(f"Failed to fetch trades: {e}") from e

    # -------------------------------------------------------------------------
    # Instrument Information
    # -------------------------------------------------------------------------

    async def _load_instruments(self) -> None:
        """Load and cache instrument information."""
        for symbol, market in self._exchange.markets.items():
            instrument = Instrument(
                symbol=symbol,
                base=market.get("base", ""),
                quote=market.get("quote", ""),
                exchange=self._exchange_id,
                lot_size=Decimal(str(market.get("precision", {}).get("amount", "0.00001"))),
                tick_size=Decimal(str(market.get("precision", {}).get("price", "0.01"))),
                min_notional=Decimal(str(market.get("limits", {}).get("cost", {}).get("min", 0) or 0)),
                contract_type=market.get("type", "spot"),
                is_active=market.get("active", True),
                maker_fee=Decimal(str(market.get("maker", 0) or 0)),
                taker_fee=Decimal(str(market.get("taker", 0) or 0)),
            )
            self._instruments[symbol] = instrument

    async def get_instruments(self) -> list[Instrument]:
        """Get all available instruments."""
        return list(self._instruments.values())

    async def get_instrument(self, symbol: str) -> Instrument | None:
        """Get instrument info for a symbol."""
        return self._instruments.get(symbol)

    # -------------------------------------------------------------------------
    # Snapshots
    # -------------------------------------------------------------------------

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get current order book snapshot."""
        if not self._connected:
            raise DataClientError("Not connected")

        try:
            data = await self._exchange.fetch_order_book(symbol, limit=depth)
            return self._convert_orderbook(symbol, data)
        except Exception as e:
            raise DataClientError(f"Failed to fetch orderbook: {e}") from e

    async def get_ticker(self, symbol: str) -> Tick:
        """Get current ticker for a symbol."""
        if not self._connected:
            raise DataClientError("Not connected")

        try:
            data = await self._exchange.fetch_ticker(symbol)
            return self._convert_ticker(symbol, data)
        except Exception as e:
            raise DataClientError(f"Failed to fetch ticker: {e}") from e

    # -------------------------------------------------------------------------
    # Converters
    # -------------------------------------------------------------------------

    def _convert_ticker(self, symbol: str, data: dict[str, Any]) -> Tick:
        """Convert CCXT ticker to Tick."""
        return Tick(
            symbol=symbol,
            bid=Decimal(str(data.get("bid", 0) or 0)),
            ask=Decimal(str(data.get("ask", 0) or 0)),
            last=Decimal(str(data.get("last", 0) or 0)),
            timestamp_ns=int((data.get("timestamp", 0) or 0) * 1_000_000),
            bid_size=Decimal(str(data.get("bidVolume", 0) or 0)) if data.get("bidVolume") else None,
            ask_size=Decimal(str(data.get("askVolume", 0) or 0)) if data.get("askVolume") else None,
            volume_24h=Decimal(str(data.get("quoteVolume", 0) or 0)) if data.get("quoteVolume") else None,
            high_24h=Decimal(str(data.get("high", 0) or 0)) if data.get("high") else None,
            low_24h=Decimal(str(data.get("low", 0) or 0)) if data.get("low") else None,
            open_24h=Decimal(str(data.get("open", 0) or 0)) if data.get("open") else None,
            change_24h=Decimal(str(data.get("percentage", 0) or 0)) if data.get("percentage") else None,
        )

    def _convert_ohlcv(self, symbol: str, timeframe: str, candle: list[Any]) -> Bar:
        """Convert CCXT OHLCV to Bar."""
        return Bar(
            symbol=symbol,
            timestamp_ns=int(candle[0] * 1_000_000),  # ms to ns
            open=Decimal(str(candle[1])),
            high=Decimal(str(candle[2])),
            low=Decimal(str(candle[3])),
            close=Decimal(str(candle[4])),
            volume=Decimal(str(candle[5])),
            timeframe=timeframe,
        )

    def _convert_orderbook(self, symbol: str, data: dict[str, Any]) -> OrderBook:
        """Convert CCXT orderbook to OrderBook."""
        bids = [(Decimal(str(p)), Decimal(str(s))) for p, s in data.get("bids", [])]
        asks = [(Decimal(str(p)), Decimal(str(s))) for p, s in data.get("asks", [])]

        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp_ns=int((data.get("timestamp", 0) or time.time() * 1000) * 1_000_000),
        )
