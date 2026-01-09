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
    AccountOrder,
    AccountPosition,
    BalanceQuery,
    Bar,
    BarQuery,
    GatewayFetcher,
    OrderBookLevel,
    OrderBookQuery,
    OrderBookSnapshot,
    OrderQuery,
    PositionQuery,
    Quote,
    TickQuery,
    TradeQuery,
    TradeRecord,
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
# CCXT Position Fetcher
# =============================================================================


class CCXTPositionFetcher(GatewayFetcher[PositionQuery, list[AccountPosition]]):
    """
    Fetcher for account positions via CCXT.

    Transforms CCXT positions to standard AccountPosition format.
    Works with exchanges that support futures/derivatives trading.

    Example:
        fetcher = CCXTPositionFetcher(exchange)
        positions = await fetcher.fetch()
        for pos in positions:
            print(f"{pos.symbol}: {pos.side} {pos.amount} @ {pos.entry_price}")

        # Get specific symbol
        positions = await fetcher.fetch(symbol="BTC/USDT")
    """

    def __init__(self, exchange: Any) -> None:
        """Initialize with CCXT exchange."""
        self._exchange = exchange

    def transform_query(self, params: dict[str, Any]) -> PositionQuery:
        """Transform params to PositionQuery."""
        return PositionQuery(
            symbol=params.get("symbol"),
        )

    async def extract_data(self, query: PositionQuery, **_kwargs: Any) -> Any:
        """Fetch raw position data from CCXT."""
        symbols = [query.symbol] if query.symbol else None
        return await self._exchange.fetch_positions(symbols)

    def transform_data(
        self, query: PositionQuery, raw: Any
    ) -> list[AccountPosition]:
        """
        Transform CCXT positions to standard format.

        CCXT position format:
        {
            "symbol": "BTC/USDT",
            "side": "long" or "short",
            "contracts": 1.5,
            "entryPrice": 50000.0,
            "markPrice": 51000.0,
            "unrealizedPnl": 1500.0,
            "leverage": 10,
            "liquidationPrice": 45000.0,
            ...
        }
        """
        positions = []

        for pos in raw:
            # Skip empty positions
            contracts = pos.get("contracts", 0) or 0
            if contracts == 0:
                continue

            # Filter by symbol if specified
            symbol = pos.get("symbol", "")
            if query.symbol and symbol != query.symbol:
                continue

            # Determine side
            side = pos.get("side", "flat")
            if not side:
                # Infer from contracts sign or notional
                notional = pos.get("notional", 0) or 0
                side = "long" if notional >= 0 else "short"

            # Calculate unrealized PnL if not provided
            unrealized_pnl = pos.get("unrealizedPnl", 0) or 0

            positions.append(
                AccountPosition(
                    symbol=symbol,
                    side=side.lower(),
                    amount=Decimal(str(abs(contracts))),
                    entry_price=Decimal(str(pos.get("entryPrice", 0) or 0)),
                    current_price=Decimal(str(pos.get("markPrice", 0) or 0)),
                    unrealized_pnl=Decimal(str(unrealized_pnl)),
                    timestamp_ns=timestamp_to_ns(pos.get("timestamp", 0) or 0),
                    leverage=int(pos.get("leverage", 1) or 1),
                    liquidation_price=Decimal(str(pos.get("liquidationPrice", 0)))
                    if pos.get("liquidationPrice")
                    else None,
                    margin=Decimal(str(pos.get("initialMargin", 0) or 0))
                    if pos.get("initialMargin")
                    else None,
                    margin_type=pos.get("marginMode") or pos.get("marginType"),
                    realized_pnl=Decimal(str(pos.get("realizedPnl", 0)))
                    if pos.get("realizedPnl") is not None
                    else None,
                )
            )

        return positions


# =============================================================================
# CCXT Order Fetcher
# =============================================================================


class CCXTOrderFetcher(GatewayFetcher[OrderQuery, list[AccountOrder]]):
    """
    Fetcher for orders via CCXT.

    Transforms CCXT orders to standard AccountOrder format.
    Supports fetching open orders, closed orders, or all orders.

    Example:
        fetcher = CCXTOrderFetcher(exchange)

        # Get open orders
        open_orders = await fetcher.fetch(status="open")

        # Get order history for a symbol
        orders = await fetcher.fetch(symbol="BTC/USDT", status="closed", limit=50)

        # Get specific order
        orders = await fetcher.fetch(order_id="12345", symbol="BTC/USDT")
    """

    def __init__(self, exchange: Any) -> None:
        """Initialize with CCXT exchange."""
        self._exchange = exchange

    def transform_query(self, params: dict[str, Any]) -> OrderQuery:
        """Transform params to OrderQuery."""
        return OrderQuery(
            symbol=params.get("symbol"),
            order_id=params.get("order_id"),
            status=params.get("status", "all"),
            limit=params.get("limit"),
            since=params.get("since"),
        )

    async def extract_data(self, query: OrderQuery, **_kwargs: Any) -> Any:
        """Fetch raw order data from CCXT."""
        since = None
        if query.since:
            since = int(query.since.timestamp() * 1000)

        # If order_id is specified, fetch single order
        if query.order_id and query.symbol:
            order = await self._exchange.fetch_order(query.order_id, query.symbol)
            return [order]

        # Fetch based on status
        if query.status == "open":
            return await self._exchange.fetch_open_orders(
                symbol=query.symbol,
                since=since,
                limit=query.limit,
            )
        elif query.status == "closed":
            return await self._exchange.fetch_closed_orders(
                symbol=query.symbol,
                since=since,
                limit=query.limit,
            )
        else:
            # Fetch all orders (combine open + closed)
            orders = []
            try:
                open_orders = await self._exchange.fetch_open_orders(
                    symbol=query.symbol,
                    since=since,
                    limit=query.limit,
                )
                orders.extend(open_orders)
            except Exception:
                pass

            try:
                closed_orders = await self._exchange.fetch_closed_orders(
                    symbol=query.symbol,
                    since=since,
                    limit=query.limit,
                )
                orders.extend(closed_orders)
            except Exception:
                pass

            return orders

    def transform_data(
        self, query: OrderQuery, raw: Any
    ) -> list[AccountOrder]:
        """
        Transform CCXT orders to standard format.

        CCXT order format:
        {
            "id": "12345",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "status": "open",
            "amount": 0.1,
            "filled": 0.0,
            "remaining": 0.1,
            "price": 50000.0,
            "average": None,
            "timestamp": 1704067200000,
            "fee": {"cost": 0.0001, "currency": "BTC"},
            ...
        }
        """
        orders = []

        for order in raw:
            # Extract fee info
            fee_info = order.get("fee") or {}
            fee = fee_info.get("cost")
            fee_currency = fee_info.get("currency")

            orders.append(
                AccountOrder(
                    order_id=str(order.get("id", "")),
                    symbol=order.get("symbol", ""),
                    side=order.get("side", "").lower(),
                    order_type=order.get("type", "").lower(),
                    status=order.get("status", "").lower(),
                    amount=Decimal(str(order.get("amount", 0) or 0)),
                    filled=Decimal(str(order.get("filled", 0) or 0)),
                    timestamp_ns=timestamp_to_ns(order.get("timestamp", 0) or 0),
                    price=Decimal(str(order.get("price", 0)))
                    if order.get("price")
                    else None,
                    average=Decimal(str(order.get("average", 0)))
                    if order.get("average")
                    else None,
                    remaining=Decimal(str(order.get("remaining", 0)))
                    if order.get("remaining") is not None
                    else None,
                    cost=Decimal(str(order.get("cost", 0)))
                    if order.get("cost")
                    else None,
                    fee=Decimal(str(fee)) if fee is not None else None,
                    fee_currency=fee_currency,
                    client_order_id=order.get("clientOrderId"),
                    stop_price=Decimal(str(order.get("stopPrice", 0)))
                    if order.get("stopPrice")
                    else None,
                    time_in_force=order.get("timeInForce"),
                    reduce_only=order.get("reduceOnly", False) or False,
                    post_only=order.get("postOnly", False) or False,
                )
            )

        return orders


# =============================================================================
# CCXT Trade Fetcher
# =============================================================================


class CCXTTradeFetcher(GatewayFetcher[TradeQuery, list[TradeRecord]]):
    """
    Fetcher for trade history via CCXT.

    Transforms CCXT trades to standard TradeRecord format.
    Returns individual trade/fill records.

    Example:
        fetcher = CCXTTradeFetcher(exchange)

        # Get recent trades for a symbol
        trades = await fetcher.fetch(symbol="BTC/USDT", limit=100)
        for trade in trades:
            print(f"{trade.side} {trade.amount} @ {trade.price}")

        # Get trades since a specific time
        trades = await fetcher.fetch(
            symbol="BTC/USDT",
            since=datetime(2024, 1, 1),
        )
    """

    def __init__(self, exchange: Any) -> None:
        """Initialize with CCXT exchange."""
        self._exchange = exchange

    def transform_query(self, params: dict[str, Any]) -> TradeQuery:
        """Transform params to TradeQuery."""
        return TradeQuery(
            symbol=params.get("symbol"),
            limit=params.get("limit"),
            since=params.get("since"),
        )

    async def extract_data(self, query: TradeQuery, **_kwargs: Any) -> Any:
        """Fetch raw trade data from CCXT."""
        since = None
        if query.since:
            since = int(query.since.timestamp() * 1000)

        return await self._exchange.fetch_my_trades(
            symbol=query.symbol,
            since=since,
            limit=query.limit,
        )

    def transform_data(
        self, query: TradeQuery, raw: Any
    ) -> list[TradeRecord]:
        """
        Transform CCXT trades to standard format.

        CCXT trade format:
        {
            "id": "T12345",
            "order": "O12345",
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.1,
            "price": 50000.0,
            "cost": 5000.0,
            "timestamp": 1704067200000,
            "fee": {"cost": 0.0001, "currency": "BTC"},
            "takerOrMaker": "taker",
            ...
        }
        """
        trades = []

        for trade in raw:
            # Extract fee info
            fee_info = trade.get("fee") or {}
            fee = fee_info.get("cost")
            fee_currency = fee_info.get("currency")

            trades.append(
                TradeRecord(
                    trade_id=str(trade.get("id", "")),
                    order_id=str(trade.get("order", "")),
                    symbol=trade.get("symbol", ""),
                    side=trade.get("side", "").lower(),
                    amount=Decimal(str(trade.get("amount", 0) or 0)),
                    price=Decimal(str(trade.get("price", 0) or 0)),
                    cost=Decimal(str(trade.get("cost", 0) or 0)),
                    timestamp_ns=timestamp_to_ns(trade.get("timestamp", 0) or 0),
                    fee=Decimal(str(fee)) if fee is not None else None,
                    fee_currency=fee_currency,
                    taker_or_maker=trade.get("takerOrMaker"),
                )
            )

        return trades


# =============================================================================
# Register Fetchers
# =============================================================================


def register_ccxt_fetchers() -> None:
    """Register all CCXT fetchers with the global registry."""
    fetcher_registry.register("ccxt", "bar", CCXTBarFetcher)
    fetcher_registry.register("ccxt", "quote", CCXTQuoteFetcher)
    fetcher_registry.register("ccxt", "orderbook", CCXTOrderBookFetcher)
    fetcher_registry.register("ccxt", "balance", CCXTBalanceFetcher)
    fetcher_registry.register("ccxt", "position", CCXTPositionFetcher)
    fetcher_registry.register("ccxt", "order", CCXTOrderFetcher)
    fetcher_registry.register("ccxt", "trade", CCXTTradeFetcher)


# Auto-register on import
register_ccxt_fetchers()
