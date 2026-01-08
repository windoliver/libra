"""
CCXT Fetcher Implementations: Provider/Fetcher pattern for CCXT gateway.

Implements Issue #27: Provider/Fetcher Pattern for Gateway Layer.

This module provides concrete fetcher implementations for CCXT exchanges,
following the 3-stage TET pipeline (Transform-Extract-Transform).

Available Fetchers:
- CCXTBarFetcher: Fetch OHLCV bar data
- CCXTQuoteFetcher: Fetch ticker/quote data
- CCXTOrderBookFetcher: Fetch order book snapshots
- CCXTBalanceFetcher: Fetch account balances

Usage:
    from ccxt.pro import binance

    exchange = binance({"apiKey": "...", "secret": "..."})
    await exchange.load_markets()

    # Use bar fetcher
    bar_fetcher = CCXTBarFetcher(exchange)
    bars = await bar_fetcher.fetch(symbol="BTC/USDT", interval="1h", limit=100)

    # Use quote fetcher
    quote_fetcher = CCXTQuoteFetcher(exchange)
    quote = await quote_fetcher.fetch(symbol="BTC/USDT")

References:
- OpenBB Fetcher pattern: openbb_platform/core/openbb_core/provider/abstract/fetcher.py
- Issue #27: Provider/Fetcher Pattern for Gateway Layer
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from libra.gateways.fetcher import (
    AccountBalance,
    BalanceQuery,
    Bar,
    BarQuery,
    GatewayFetcher,
    OrderBookLevel,
    OrderBookQuery,
    OrderBookSnapshot,
    Quote,
    TickQuery,
    fetcher_registry,
    timestamp_to_ns,
)


if TYPE_CHECKING:
    pass


# =============================================================================
# CCXT Bar Fetcher
# =============================================================================


class CCXTBarFetcher(GatewayFetcher[BarQuery, list[Bar]]):
    """
    Fetcher for OHLCV bar data via CCXT.

    Transforms CCXT candle data to standard Bar format.

    Example:
        exchange = ccxt.pro.binance()
        await exchange.load_markets()

        fetcher = CCXTBarFetcher(exchange)
        bars = await fetcher.fetch(
            symbol="BTC/USDT",
            interval="1h",
            limit=100,
        )
        print(f"Fetched {len(bars)} bars")
        print(f"Latest close: {bars[-1].close}")
    """

    def __init__(self, exchange: Any) -> None:
        """
        Initialize fetcher with CCXT exchange.

        Args:
            exchange: CCXT exchange instance (ccxt.pro or ccxt)
        """
        self._exchange = exchange

    def transform_query(self, params: dict[str, Any]) -> BarQuery:
        """
        Transform params to BarQuery.

        Required params:
            symbol: Trading pair (e.g., "BTC/USDT")

        Optional params:
            interval: Timeframe (default "1h")
            limit: Number of bars (default 100)
            start: Start datetime
            end: End datetime
        """
        symbol = params.get("symbol")
        if not symbol:
            raise ValueError("symbol is required")

        return BarQuery(
            symbol=symbol,
            interval=params.get("interval", "1h"),
            limit=params.get("limit", 100),
            start=params.get("start"),
            end=params.get("end"),
            provider=params.get("provider"),
        )

    async def extract_data(self, query: BarQuery, **_kwargs: Any) -> Any:
        """
        Fetch raw OHLCV data from CCXT.

        CCXT returns data as:
        [[timestamp_ms, open, high, low, close, volume], ...]
        """
        since = None
        if query.start:
            since = int(query.start.timestamp() * 1000)

        return await self._exchange.fetch_ohlcv(
            query.symbol,
            query.interval,
            since=since,
            limit=query.limit,
        )

    def transform_data(self, query: BarQuery, raw: Any) -> list[Bar]:
        """
        Transform CCXT OHLCV to standard Bar format.

        CCXT format: [timestamp_ms, open, high, low, close, volume]
        """
        bars = []
        for candle in raw:
            bar = Bar(
                symbol=query.symbol,
                interval=query.interval,
                timestamp_ns=timestamp_to_ns(candle[0]),  # ms to ns
                open=Decimal(str(candle[1])),
                high=Decimal(str(candle[2])),
                low=Decimal(str(candle[3])),
                close=Decimal(str(candle[4])),
                volume=Decimal(str(candle[5])),
            )
            bars.append(bar)

        # Filter by end date if specified
        if query.end:
            end_ns = timestamp_to_ns(query.end)
            bars = [b for b in bars if b.timestamp_ns <= end_ns]

        return bars


# =============================================================================
# CCXT Quote Fetcher
# =============================================================================


class CCXTQuoteFetcher(GatewayFetcher[TickQuery, Quote]):
    """
    Fetcher for ticker/quote data via CCXT.

    Transforms CCXT ticker to standard Quote format.

    Example:
        fetcher = CCXTQuoteFetcher(exchange)
        quote = await fetcher.fetch(symbol="BTC/USDT")
        print(f"BTC bid: {quote.bid}, ask: {quote.ask}")
        print(f"Spread: {quote.spread_bps:.2f} bps")
    """

    def __init__(self, exchange: Any) -> None:
        """Initialize with CCXT exchange."""
        self._exchange = exchange

    def transform_query(self, params: dict[str, Any]) -> TickQuery:
        """Transform params to TickQuery."""
        symbol = params.get("symbol")
        if not symbol:
            raise ValueError("symbol is required")

        return TickQuery(
            symbol=symbol,
            provider=params.get("provider"),
        )

    async def extract_data(self, query: TickQuery, **_kwargs: Any) -> Any:
        """Fetch raw ticker data from CCXT."""
        return await self._exchange.fetch_ticker(query.symbol)

    def transform_data(self, query: TickQuery, raw: Any) -> Quote:
        """
        Transform CCXT ticker to standard Quote format.

        CCXT ticker fields:
        - bid, ask, last
        - bidVolume, askVolume
        - high, low, open, close
        - quoteVolume, percentage
        - timestamp
        """
        return Quote(
            symbol=query.symbol,
            bid=Decimal(str(raw.get("bid", 0) or 0)),
            ask=Decimal(str(raw.get("ask", 0) or 0)),
            last=Decimal(str(raw.get("last", 0) or 0)),
            timestamp_ns=timestamp_to_ns(raw.get("timestamp", 0) or 0),
            bid_size=Decimal(str(raw.get("bidVolume", 0) or 0))
            if raw.get("bidVolume")
            else None,
            ask_size=Decimal(str(raw.get("askVolume", 0) or 0))
            if raw.get("askVolume")
            else None,
            volume_24h=Decimal(str(raw.get("quoteVolume", 0) or 0))
            if raw.get("quoteVolume")
            else None,
            high_24h=Decimal(str(raw.get("high", 0) or 0)) if raw.get("high") else None,
            low_24h=Decimal(str(raw.get("low", 0) or 0)) if raw.get("low") else None,
            change_24h_pct=Decimal(str(raw.get("percentage", 0) or 0))
            if raw.get("percentage")
            else None,
        )


# =============================================================================
# CCXT Order Book Fetcher
# =============================================================================


class CCXTOrderBookFetcher(GatewayFetcher[OrderBookQuery, OrderBookSnapshot]):
    """
    Fetcher for order book data via CCXT.

    Transforms CCXT orderbook to standard OrderBookSnapshot format.

    Example:
        fetcher = CCXTOrderBookFetcher(exchange)
        orderbook = await fetcher.fetch(symbol="BTC/USDT", depth=20)
        print(f"Best bid: {orderbook.best_bid}")
        print(f"Best ask: {orderbook.best_ask}")
        print(f"Spread: {orderbook.spread}")
    """

    def __init__(self, exchange: Any) -> None:
        """Initialize with CCXT exchange."""
        self._exchange = exchange

    def transform_query(self, params: dict[str, Any]) -> OrderBookQuery:
        """Transform params to OrderBookQuery."""
        symbol = params.get("symbol")
        if not symbol:
            raise ValueError("symbol is required")

        return OrderBookQuery(
            symbol=symbol,
            depth=params.get("depth", 20),
            provider=params.get("provider"),
        )

    async def extract_data(self, query: OrderBookQuery, **_kwargs: Any) -> Any:
        """Fetch raw order book from CCXT."""
        return await self._exchange.fetch_order_book(
            query.symbol,
            limit=query.depth,
        )

    def transform_data(self, query: OrderBookQuery, raw: Any) -> OrderBookSnapshot:
        """
        Transform CCXT orderbook to standard format.

        CCXT format:
        {
            "bids": [[price, size], ...],
            "asks": [[price, size], ...],
            "timestamp": ms,
        }
        """
        bids = [
            OrderBookLevel(
                price=Decimal(str(level[0])),
                size=Decimal(str(level[1])),
            )
            for level in raw.get("bids", [])
        ]

        asks = [
            OrderBookLevel(
                price=Decimal(str(level[0])),
                size=Decimal(str(level[1])),
            )
            for level in raw.get("asks", [])
        ]

        return OrderBookSnapshot(
            symbol=query.symbol,
            bids=bids,
            asks=asks,
            timestamp_ns=timestamp_to_ns(raw.get("timestamp", 0) or 0),
        )


# =============================================================================
# CCXT Balance Fetcher
# =============================================================================


class CCXTBalanceFetcher(GatewayFetcher[BalanceQuery, dict[str, AccountBalance]]):
    """
    Fetcher for account balance via CCXT.

    Transforms CCXT balance to standard AccountBalance format.

    Example:
        fetcher = CCXTBalanceFetcher(exchange)
        balances = await fetcher.fetch()
        print(f"USDT: {balances['USDT'].available}")

        # Or get specific currency
        balances = await fetcher.fetch(currency="BTC")
    """

    def __init__(self, exchange: Any) -> None:
        """Initialize with CCXT exchange."""
        self._exchange = exchange

    def transform_query(self, params: dict[str, Any]) -> BalanceQuery:
        """Transform params to BalanceQuery."""
        return BalanceQuery(
            currency=params.get("currency"),
        )

    async def extract_data(self, _query: BalanceQuery, **_kwargs: Any) -> Any:
        """Fetch raw balance data from CCXT."""
        return await self._exchange.fetch_balance()

    def transform_data(
        self, query: BalanceQuery, raw: Any
    ) -> dict[str, AccountBalance]:
        """
        Transform CCXT balance to standard format.

        CCXT format:
        {
            "BTC": {"total": 1.0, "free": 0.8, "used": 0.2},
            "USDT": {"total": 10000, "free": 8000, "used": 2000},
            ...
        }
        """
        balances: dict[str, AccountBalance] = {}

        for currency, info in raw.items():
            # Skip non-balance entries (like 'info', 'timestamp', etc.)
            if not isinstance(info, dict) or "total" not in info:
                continue

            total = Decimal(str(info.get("total", 0) or 0))
            # Skip zero balances unless specifically requested
            if total == 0 and query.currency != currency:
                continue

            # Filter by currency if specified
            if query.currency and currency != query.currency:
                continue

            balances[currency] = AccountBalance(
                currency=currency,
                total=total,
                available=Decimal(str(info.get("free", 0) or 0)),
                locked=Decimal(str(info.get("used", 0) or 0)),
            )

        return balances


# =============================================================================
# Register Fetchers
# =============================================================================


def register_ccxt_fetchers() -> None:
    """Register all CCXT fetchers with the global registry."""
    fetcher_registry.register("ccxt", "bar", CCXTBarFetcher)
    fetcher_registry.register("ccxt", "quote", CCXTQuoteFetcher)
    fetcher_registry.register("ccxt", "orderbook", CCXTOrderBookFetcher)
    fetcher_registry.register("ccxt", "balance", CCXTBalanceFetcher)


# Auto-register on import
register_ccxt_fetchers()
