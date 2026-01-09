"""
Unit tests for GatewayFetcher pattern (Issue #27).

Tests the 3-stage TET pipeline:
1. transform_query() - Convert params to typed query
2. extract_data() - Fetch raw data
3. transform_data() - Normalize to standard format
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import pytest

from libra.gateways.fetcher import (
    AccountBalance,
    AccountOrder,
    AccountPosition,
    BalanceQuery,
    Bar,
    BarQuery,
    BaseQuery,
    FetcherRegistry,
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


# =============================================================================
# Query Type Tests
# =============================================================================


class TestQueryTypes:
    """Tests for query type dataclasses."""

    def test_bar_query_defaults(self) -> None:
        """Test BarQuery with default values."""
        query = BarQuery(symbol="BTC/USDT")
        assert query.symbol == "BTC/USDT"
        assert query.interval == "1h"
        assert query.limit is None
        assert query.start is None
        assert query.end is None

    def test_bar_query_full(self) -> None:
        """Test BarQuery with all parameters."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        query = BarQuery(
            symbol="ETH/USDT",
            interval="1d",
            limit=500,
            start=start,
            end=end,
            provider="binance",
        )
        assert query.symbol == "ETH/USDT"
        assert query.interval == "1d"
        assert query.limit == 500
        assert query.start == start
        assert query.end == end
        assert query.provider == "binance"

    def test_tick_query(self) -> None:
        """Test TickQuery."""
        query = TickQuery(symbol="BTC/USDT")
        assert query.symbol == "BTC/USDT"
        assert query.provider is None

    def test_orderbook_query_defaults(self) -> None:
        """Test OrderBookQuery defaults."""
        query = OrderBookQuery(symbol="BTC/USDT")
        assert query.symbol == "BTC/USDT"
        assert query.depth == 20

    def test_balance_query_all(self) -> None:
        """Test BalanceQuery for all currencies."""
        query = BalanceQuery()
        assert query.currency is None

    def test_balance_query_specific(self) -> None:
        """Test BalanceQuery for specific currency."""
        query = BalanceQuery(currency="USDT")
        assert query.currency == "USDT"

    def test_position_query_all(self) -> None:
        """Test PositionQuery for all symbols."""
        query = PositionQuery()
        assert query.symbol is None

    def test_position_query_specific(self) -> None:
        """Test PositionQuery for specific symbol."""
        query = PositionQuery(symbol="BTC/USDT")
        assert query.symbol == "BTC/USDT"

    def test_order_query_defaults(self) -> None:
        """Test OrderQuery with defaults."""
        query = OrderQuery()
        assert query.symbol is None
        assert query.order_id is None
        assert query.status is None
        assert query.limit is None
        assert query.since is None

    def test_order_query_full(self) -> None:
        """Test OrderQuery with all parameters."""
        since = datetime(2024, 1, 1)
        query = OrderQuery(
            symbol="BTC/USDT",
            order_id="12345",
            status="open",
            limit=100,
            since=since,
        )
        assert query.symbol == "BTC/USDT"
        assert query.order_id == "12345"
        assert query.status == "open"
        assert query.limit == 100
        assert query.since == since

    def test_trade_query_defaults(self) -> None:
        """Test TradeQuery with defaults."""
        query = TradeQuery()
        assert query.symbol is None
        assert query.limit is None
        assert query.since is None

    def test_trade_query_full(self) -> None:
        """Test TradeQuery with all parameters."""
        since = datetime(2024, 1, 1)
        query = TradeQuery(
            symbol="BTC/USDT",
            limit=100,
            since=since,
        )
        assert query.symbol == "BTC/USDT"
        assert query.limit == 100
        assert query.since == since

    def test_query_is_frozen(self) -> None:
        """Test that queries are immutable."""
        query = BarQuery(symbol="BTC/USDT")
        with pytest.raises(AttributeError):
            query.symbol = "ETH/USDT"  # type: ignore


# =============================================================================
# Response Type Tests
# =============================================================================


class TestResponseTypes:
    """Tests for response type structs."""

    def test_bar_struct(self) -> None:
        """Test Bar struct creation and properties."""
        bar = Bar(
            symbol="BTC/USDT",
            interval="1h",
            timestamp_ns=1704067200_000_000_000,  # 2024-01-01 00:00:00 UTC
            open=Decimal("42000"),
            high=Decimal("42500"),
            low=Decimal("41800"),
            close=Decimal("42300"),
            volume=Decimal("1500.5"),
        )
        assert bar.symbol == "BTC/USDT"
        assert bar.interval == "1h"
        assert bar.timestamp_ns == 1704067200_000_000_000
        assert bar.timestamp_ms == 1704067200_000
        assert bar.open == Decimal("42000")
        assert bar.close == Decimal("42300")

    def test_bar_timestamp_sec(self) -> None:
        """Test Bar timestamp_sec property."""
        bar = Bar(
            symbol="BTC/USDT",
            timestamp_ns=1704067200_000_000_000,
            open=Decimal("42000"),
            high=Decimal("42500"),
            low=Decimal("41800"),
            close=Decimal("42300"),
            volume=Decimal("1500.5"),
        )
        assert bar.timestamp_sec == 1704067200.0

    def test_quote_struct(self) -> None:
        """Test Quote struct creation and properties."""
        quote = Quote(
            symbol="BTC/USDT",
            bid=Decimal("42000"),
            ask=Decimal("42010"),
            last=Decimal("42005"),
            timestamp_ns=1704067200_000_000_000,
        )
        assert quote.symbol == "BTC/USDT"
        assert quote.mid == Decimal("42005")
        assert quote.spread == Decimal("10")

    def test_quote_spread_bps(self) -> None:
        """Test Quote spread in basis points."""
        quote = Quote(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=1704067200_000_000_000,
        )
        # Spread: 10, Mid: 50005, BPS: (10/50005)*10000 ≈ 1.9998
        assert quote.spread_bps > Decimal("1.99")
        assert quote.spread_bps < Decimal("2.01")

    def test_orderbook_snapshot(self) -> None:
        """Test OrderBookSnapshot struct."""
        orderbook = OrderBookSnapshot(
            symbol="BTC/USDT",
            bids=[
                OrderBookLevel(price=Decimal("42000"), size=Decimal("1.5")),
                OrderBookLevel(price=Decimal("41990"), size=Decimal("2.0")),
            ],
            asks=[
                OrderBookLevel(price=Decimal("42010"), size=Decimal("1.0")),
                OrderBookLevel(price=Decimal("42020"), size=Decimal("3.0")),
            ],
            timestamp_ns=1704067200_000_000_000,
        )
        assert orderbook.best_bid == Decimal("42000")
        assert orderbook.best_ask == Decimal("42010")
        assert orderbook.mid == Decimal("42005")
        assert orderbook.spread == Decimal("10")

    def test_orderbook_empty(self) -> None:
        """Test OrderBookSnapshot with empty levels."""
        orderbook = OrderBookSnapshot(
            symbol="BTC/USDT",
            bids=[],
            asks=[],
            timestamp_ns=1704067200_000_000_000,
        )
        assert orderbook.best_bid is None
        assert orderbook.best_ask is None
        assert orderbook.mid is None
        assert orderbook.spread is None

    def test_account_balance(self) -> None:
        """Test AccountBalance struct."""
        balance = AccountBalance(
            currency="USDT",
            total=Decimal("10000"),
            available=Decimal("8000"),
            locked=Decimal("2000"),
        )
        assert balance.currency == "USDT"
        assert balance.total == Decimal("10000")
        assert balance.available == Decimal("8000")
        assert balance.locked == Decimal("2000")

    def test_account_position(self) -> None:
        """Test AccountPosition struct creation and properties."""
        position = AccountPosition(
            symbol="BTC/USDT",
            side="long",
            amount=Decimal("0.5"),
            entry_price=Decimal("48000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("1000"),
            leverage=10,
        )
        assert position.symbol == "BTC/USDT"
        assert position.side == "long"
        assert position.amount == Decimal("0.5")
        assert position.entry_price == Decimal("48000")
        assert position.current_price == Decimal("50000")
        assert position.unrealized_pnl == Decimal("1000")
        assert position.leverage == 10

    def test_account_position_notional_value(self) -> None:
        """Test AccountPosition notional value property."""
        position = AccountPosition(
            symbol="BTC/USDT",
            side="long",
            amount=Decimal("2"),
            entry_price=Decimal("45000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("10000"),
        )
        assert position.notional_value == Decimal("100000")

    def test_account_position_pnl_percent(self) -> None:
        """Test AccountPosition P&L percentage."""
        position = AccountPosition(
            symbol="BTC/USDT",
            side="long",
            amount=Decimal("1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("55000"),
            unrealized_pnl=Decimal("5000"),
        )
        # 5000 / (1 * 50000) * 100 = 10%
        assert position.pnl_percent == Decimal("10")

    def test_account_order(self) -> None:
        """Test AccountOrder struct creation and properties."""
        order = AccountOrder(
            order_id="12345",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            status="filled",
            amount=Decimal("0.1"),
            filled=Decimal("0.1"),
            timestamp_ns=1704067200_000_000_000,
            price=Decimal("50000"),
            average=Decimal("49980"),
        )
        assert order.order_id == "12345"
        assert order.symbol == "BTC/USDT"
        assert order.side == "buy"
        assert order.order_type == "limit"
        assert order.status == "filled"
        assert order.amount == Decimal("0.1")
        assert order.filled == Decimal("0.1")
        assert order.price == Decimal("50000")
        assert order.average == Decimal("49980")

    def test_account_order_is_open(self) -> None:
        """Test AccountOrder is_open property."""
        open_order = AccountOrder(
            order_id="1",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            status="open",
            amount=Decimal("1"),
            filled=Decimal("0"),
            timestamp_ns=1704067200_000_000_000,
        )
        assert open_order.is_open is True

        closed_order = AccountOrder(
            order_id="2",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            status="closed",
            amount=Decimal("1"),
            filled=Decimal("1"),
            timestamp_ns=1704067200_000_000_000,
        )
        assert closed_order.is_open is False

    def test_account_order_fill_percent(self) -> None:
        """Test AccountOrder fill percentage."""
        order = AccountOrder(
            order_id="1",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            status="partially_filled",
            amount=Decimal("10"),
            filled=Decimal("7.5"),
            timestamp_ns=1704067200_000_000_000,
        )
        assert order.fill_percent == Decimal("75")

    def test_trade_record(self) -> None:
        """Test TradeRecord struct creation."""
        trade = TradeRecord(
            trade_id="T12345",
            order_id="O12345",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.1"),
            price=Decimal("50000"),
            cost=Decimal("5000"),
            timestamp_ns=1704067200_000_000_000,
            fee=Decimal("5"),
            fee_currency="USDT",
            taker_or_maker="taker",
        )
        assert trade.trade_id == "T12345"
        assert trade.order_id == "O12345"
        assert trade.symbol == "BTC/USDT"
        assert trade.side == "buy"
        assert trade.amount == Decimal("0.1")
        assert trade.price == Decimal("50000")
        assert trade.cost == Decimal("5000")
        assert trade.fee == Decimal("5")
        assert trade.fee_currency == "USDT"
        assert trade.taker_or_maker == "taker"


# =============================================================================
# Fetcher Protocol Tests
# =============================================================================


class MockBarFetcher(GatewayFetcher[BarQuery, list[Bar]]):
    """Mock bar fetcher for testing."""

    def __init__(self) -> None:
        self.extract_called = False
        self.raw_data: list[list[Any]] = []

    def transform_query(self, params: dict[str, Any]) -> BarQuery:
        return BarQuery(
            symbol=params["symbol"],
            interval=params.get("interval", "1h"),
            limit=params.get("limit", 100),
        )

    async def extract_data(self, query: BarQuery, **kwargs: Any) -> Any:
        self.extract_called = True
        return self.raw_data

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


class TestGatewayFetcher:
    """Tests for GatewayFetcher abstract base class."""

    @pytest.mark.asyncio
    async def test_fetch_pipeline(self) -> None:
        """Test full fetch pipeline execution."""
        fetcher = MockBarFetcher()
        fetcher.raw_data = [
            [1704067200000, 42000, 42500, 41800, 42300, 1500.5],
            [1704070800000, 42300, 42800, 42100, 42600, 1800.2],
        ]

        bars = await fetcher.fetch(symbol="BTC/USDT", interval="1h")

        assert len(bars) == 2
        assert bars[0].symbol == "BTC/USDT"
        assert bars[0].open == Decimal("42000")
        assert bars[0].close == Decimal("42300")
        assert bars[1].close == Decimal("42600")
        assert fetcher.extract_called

    @pytest.mark.asyncio
    async def test_fetch_with_query(self) -> None:
        """Test fetch with pre-built query."""
        fetcher = MockBarFetcher()
        fetcher.raw_data = [
            [1704067200000, 42000, 42500, 41800, 42300, 1500.5],
        ]

        query = BarQuery(symbol="ETH/USDT", interval="15m", limit=50)
        bars = await fetcher.fetch_with_query(query)

        assert len(bars) == 1
        assert bars[0].symbol == "ETH/USDT"
        assert bars[0].interval == "15m"

    def test_transform_query_validation(self) -> None:
        """Test transform_query validates required params."""
        fetcher = MockBarFetcher()

        with pytest.raises(KeyError):
            fetcher.transform_query({})  # Missing symbol


# =============================================================================
# Fetcher Registry Tests
# =============================================================================


class TestFetcherRegistry:
    """Tests for FetcherRegistry."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving fetchers."""
        registry = FetcherRegistry()
        registry.register("test", "bar", MockBarFetcher)

        result = registry.get("test", "bar")
        assert result == MockBarFetcher

    def test_get_unregistered(self) -> None:
        """Test getting unregistered fetcher returns None."""
        registry = FetcherRegistry()
        assert registry.get("unknown", "bar") is None

    def test_list_gateways(self) -> None:
        """Test listing registered gateways."""
        registry = FetcherRegistry()
        registry.register("gateway1", "bar", MockBarFetcher)
        registry.register("gateway2", "quote", MockBarFetcher)

        gateways = registry.list_gateways()
        assert "gateway1" in gateways
        assert "gateway2" in gateways

    def test_list_data_types(self) -> None:
        """Test listing data types for a gateway."""
        registry = FetcherRegistry()
        registry.register("test", "bar", MockBarFetcher)
        registry.register("test", "quote", MockBarFetcher)

        data_types = registry.list_data_types("test")
        assert "bar" in data_types
        assert "quote" in data_types

    def test_global_registry_has_ccxt(self) -> None:
        """Test global registry has CCXT fetchers registered."""
        # Import to trigger registration
        from libra.gateways import ccxt_fetchers  # noqa: F401

        gateways = fetcher_registry.list_gateways()
        assert "ccxt" in gateways

        data_types = fetcher_registry.list_data_types("ccxt")
        assert "bar" in data_types
        assert "quote" in data_types
        assert "orderbook" in data_types
        assert "balance" in data_types


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_timestamp_to_ns_datetime(self) -> None:
        """Test converting datetime to nanoseconds."""
        dt = datetime(2024, 1, 1, 0, 0, 0)
        ns = timestamp_to_ns(dt)
        assert ns == int(dt.timestamp() * 1_000_000_000)

    def test_timestamp_to_ns_milliseconds(self) -> None:
        """Test converting milliseconds to nanoseconds."""
        ms = 1704067200000
        ns = timestamp_to_ns(ms)
        assert ns == ms * 1_000_000

    def test_timestamp_to_ns_seconds(self) -> None:
        """Test converting seconds to nanoseconds."""
        sec = 1704067200
        ns = timestamp_to_ns(sec)
        assert ns == sec * 1_000_000_000

    def test_timestamp_to_ns_float(self) -> None:
        """Test converting float seconds to nanoseconds."""
        sec = 1704067200.5
        ns = timestamp_to_ns(sec)
        assert ns == int(sec * 1_000_000_000)

    def test_timestamp_to_ns_already_ns(self) -> None:
        """Test that nanoseconds are passed through."""
        ns_input = 1704067200_000_000_000
        ns = timestamp_to_ns(ns_input)
        assert ns == ns_input


# =============================================================================
# CCXT Fetcher Tests (with mocked exchange)
# =============================================================================


class TestCCXTFetchers:
    """Tests for CCXT-specific fetchers with mocked exchange."""

    @pytest.fixture
    def mock_exchange(self) -> AsyncMock:
        """Create a mock CCXT exchange."""
        exchange = AsyncMock()
        exchange.fetch_ohlcv = AsyncMock()
        exchange.fetch_ticker = AsyncMock()
        exchange.fetch_order_book = AsyncMock()
        exchange.fetch_balance = AsyncMock()
        exchange.fetch_positions = AsyncMock()
        exchange.fetch_open_orders = AsyncMock()
        exchange.fetch_closed_orders = AsyncMock()
        exchange.fetch_order = AsyncMock()
        exchange.fetch_my_trades = AsyncMock()
        return exchange

    @pytest.mark.asyncio
    async def test_ccxt_bar_fetcher(self, mock_exchange: AsyncMock) -> None:
        """Test CCXTBarFetcher with mocked exchange."""
        from libra.gateways.ccxt_fetchers import CCXTBarFetcher

        # Setup mock response
        mock_exchange.fetch_ohlcv.return_value = [
            [1704067200000, 42000, 42500, 41800, 42300, 1500.5],
            [1704070800000, 42300, 42800, 42100, 42600, 1800.2],
        ]

        fetcher = CCXTBarFetcher(mock_exchange)
        bars = await fetcher.fetch(symbol="BTC/USDT", interval="1h", limit=100)

        assert len(bars) == 2
        assert bars[0].symbol == "BTC/USDT"
        assert bars[0].interval == "1h"
        assert bars[0].open == Decimal("42000")
        assert bars[1].close == Decimal("42600")
        mock_exchange.fetch_ohlcv.assert_called_once()

    @pytest.mark.asyncio
    async def test_ccxt_quote_fetcher(self, mock_exchange: AsyncMock) -> None:
        """Test CCXTQuoteFetcher with mocked exchange."""
        from libra.gateways.ccxt_fetchers import CCXTQuoteFetcher

        # Setup mock response
        mock_exchange.fetch_ticker.return_value = {
            "bid": 42000,
            "ask": 42010,
            "last": 42005,
            "timestamp": 1704067200000,
            "high": 43000,
            "low": 41000,
            "quoteVolume": 1000000,
        }

        fetcher = CCXTQuoteFetcher(mock_exchange)
        quote = await fetcher.fetch(symbol="BTC/USDT")

        assert quote.symbol == "BTC/USDT"
        assert quote.bid == Decimal("42000")
        assert quote.ask == Decimal("42010")
        assert quote.last == Decimal("42005")
        mock_exchange.fetch_ticker.assert_called_once()

    @pytest.mark.asyncio
    async def test_ccxt_orderbook_fetcher(self, mock_exchange: AsyncMock) -> None:
        """Test CCXTOrderBookFetcher with mocked exchange."""
        from libra.gateways.ccxt_fetchers import CCXTOrderBookFetcher

        # Setup mock response
        mock_exchange.fetch_order_book.return_value = {
            "bids": [[42000, 1.5], [41990, 2.0]],
            "asks": [[42010, 1.0], [42020, 3.0]],
            "timestamp": 1704067200000,
        }

        fetcher = CCXTOrderBookFetcher(mock_exchange)
        orderbook = await fetcher.fetch(symbol="BTC/USDT", depth=20)

        assert orderbook.symbol == "BTC/USDT"
        assert len(orderbook.bids) == 2
        assert len(orderbook.asks) == 2
        assert orderbook.best_bid == Decimal("42000")
        assert orderbook.best_ask == Decimal("42010")
        mock_exchange.fetch_order_book.assert_called_once()

    @pytest.mark.asyncio
    async def test_ccxt_balance_fetcher(self, mock_exchange: AsyncMock) -> None:
        """Test CCXTBalanceFetcher with mocked exchange."""
        from libra.gateways.ccxt_fetchers import CCXTBalanceFetcher

        # Setup mock response
        mock_exchange.fetch_balance.return_value = {
            "USDT": {"total": 10000, "free": 8000, "used": 2000},
            "BTC": {"total": 1.5, "free": 1.0, "used": 0.5},
            "info": {},  # CCXT metadata - should be filtered
        }

        fetcher = CCXTBalanceFetcher(mock_exchange)
        balances = await fetcher.fetch()

        assert "USDT" in balances
        assert "BTC" in balances
        assert "info" not in balances
        assert balances["USDT"].total == Decimal("10000")
        assert balances["USDT"].available == Decimal("8000")
        mock_exchange.fetch_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_ccxt_balance_fetcher_specific_currency(
        self, mock_exchange: AsyncMock
    ) -> None:
        """Test CCXTBalanceFetcher with specific currency filter."""
        from libra.gateways.ccxt_fetchers import CCXTBalanceFetcher

        mock_exchange.fetch_balance.return_value = {
            "USDT": {"total": 10000, "free": 8000, "used": 2000},
            "BTC": {"total": 1.5, "free": 1.0, "used": 0.5},
        }

        fetcher = CCXTBalanceFetcher(mock_exchange)
        balances = await fetcher.fetch(currency="USDT")

        # Should only have USDT
        assert "USDT" in balances
        assert "BTC" not in balances

    def test_ccxt_bar_fetcher_transform_query_validation(
        self, mock_exchange: AsyncMock
    ) -> None:
        """Test CCXTBarFetcher validates required params."""
        from libra.gateways.ccxt_fetchers import CCXTBarFetcher

        fetcher = CCXTBarFetcher(mock_exchange)

        with pytest.raises(ValueError, match="symbol is required"):
            fetcher.transform_query({})

    def test_ccxt_quote_fetcher_transform_query_validation(
        self, mock_exchange: AsyncMock
    ) -> None:
        """Test CCXTQuoteFetcher validates required params."""
        from libra.gateways.ccxt_fetchers import CCXTQuoteFetcher

        fetcher = CCXTQuoteFetcher(mock_exchange)

        with pytest.raises(ValueError, match="symbol is required"):
            fetcher.transform_query({})

    # =========================================================================
    # Position Fetcher Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_ccxt_position_fetcher(self, mock_exchange: AsyncMock) -> None:
        """Test CCXTPositionFetcher with mocked exchange."""
        from libra.gateways.ccxt_fetchers import CCXTPositionFetcher

        # Setup mock response (CCXT position format)
        mock_exchange.fetch_positions.return_value = [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "contracts": 1.5,
                "entryPrice": 50000.0,
                "markPrice": 51000.0,
                "unrealizedPnl": 1500.0,
                "leverage": 10,
                "liquidationPrice": 45000.0,
                "initialMargin": 7500.0,
                "marginMode": "cross",
                "timestamp": 1704067200000,
            },
            {
                "symbol": "ETH/USDT",
                "side": "short",
                "contracts": 5.0,
                "entryPrice": 2500.0,
                "markPrice": 2400.0,
                "unrealizedPnl": 500.0,
                "leverage": 5,
                "timestamp": 1704067200000,
            },
        ]

        fetcher = CCXTPositionFetcher(mock_exchange)
        positions = await fetcher.fetch()

        assert len(positions) == 2
        assert positions[0].symbol == "BTC/USDT"
        assert positions[0].side == "long"
        assert positions[0].amount == Decimal("1.5")
        assert positions[0].entry_price == Decimal("50000")
        assert positions[0].current_price == Decimal("51000")
        assert positions[0].unrealized_pnl == Decimal("1500")
        assert positions[0].leverage == 10
        assert positions[0].liquidation_price == Decimal("45000")

        assert positions[1].symbol == "ETH/USDT"
        assert positions[1].side == "short"
        assert positions[1].amount == Decimal("5")
        mock_exchange.fetch_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_ccxt_position_fetcher_specific_symbol(
        self, mock_exchange: AsyncMock
    ) -> None:
        """Test CCXTPositionFetcher with specific symbol filter."""
        from libra.gateways.ccxt_fetchers import CCXTPositionFetcher

        mock_exchange.fetch_positions.return_value = [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "contracts": 1.5,
                "entryPrice": 50000.0,
                "markPrice": 51000.0,
                "unrealizedPnl": 1500.0,
            },
        ]

        fetcher = CCXTPositionFetcher(mock_exchange)
        positions = await fetcher.fetch(symbol="BTC/USDT")

        assert len(positions) == 1
        assert positions[0].symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_ccxt_position_fetcher_skips_empty(
        self, mock_exchange: AsyncMock
    ) -> None:
        """Test CCXTPositionFetcher skips empty positions."""
        from libra.gateways.ccxt_fetchers import CCXTPositionFetcher

        mock_exchange.fetch_positions.return_value = [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "contracts": 0,  # Empty position
                "entryPrice": 50000.0,
                "markPrice": 51000.0,
                "unrealizedPnl": 0,
            },
            {
                "symbol": "ETH/USDT",
                "side": "long",
                "contracts": 1.0,  # Has position
                "entryPrice": 2500.0,
                "markPrice": 2600.0,
                "unrealizedPnl": 100.0,
            },
        ]

        fetcher = CCXTPositionFetcher(mock_exchange)
        positions = await fetcher.fetch()

        # Should only have ETH position, BTC is filtered out
        assert len(positions) == 1
        assert positions[0].symbol == "ETH/USDT"

    # =========================================================================
    # Order Fetcher Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_ccxt_order_fetcher_open_orders(
        self, mock_exchange: AsyncMock
    ) -> None:
        """Test CCXTOrderFetcher fetching open orders."""
        from libra.gateways.ccxt_fetchers import CCXTOrderFetcher

        mock_exchange.fetch_open_orders.return_value = [
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
                "timestamp": 1704067200000,
                "fee": {"cost": 0, "currency": "USDT"},
            },
        ]

        fetcher = CCXTOrderFetcher(mock_exchange)
        orders = await fetcher.fetch(status="open")

        assert len(orders) == 1
        assert orders[0].order_id == "12345"
        assert orders[0].symbol == "BTC/USDT"
        assert orders[0].side == "buy"
        assert orders[0].order_type == "limit"
        assert orders[0].status == "open"
        assert orders[0].amount == Decimal("0.1")
        assert orders[0].filled == Decimal("0")
        assert orders[0].price == Decimal("50000")
        mock_exchange.fetch_open_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_ccxt_order_fetcher_closed_orders(
        self, mock_exchange: AsyncMock
    ) -> None:
        """Test CCXTOrderFetcher fetching closed orders."""
        from libra.gateways.ccxt_fetchers import CCXTOrderFetcher

        mock_exchange.fetch_closed_orders.return_value = [
            {
                "id": "12346",
                "symbol": "BTC/USDT",
                "side": "sell",
                "type": "market",
                "status": "closed",
                "amount": 0.5,
                "filled": 0.5,
                "remaining": 0,
                "average": 51000.0,
                "timestamp": 1704067200000,
                "fee": {"cost": 2.55, "currency": "USDT"},
            },
        ]

        fetcher = CCXTOrderFetcher(mock_exchange)
        orders = await fetcher.fetch(status="closed")

        assert len(orders) == 1
        assert orders[0].order_id == "12346"
        assert orders[0].status == "closed"
        assert orders[0].filled == Decimal("0.5")
        assert orders[0].average == Decimal("51000")
        assert orders[0].fee == Decimal("2.55")
        assert orders[0].fee_currency == "USDT"
        mock_exchange.fetch_closed_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_ccxt_order_fetcher_specific_order(
        self, mock_exchange: AsyncMock
    ) -> None:
        """Test CCXTOrderFetcher fetching specific order by ID."""
        from libra.gateways.ccxt_fetchers import CCXTOrderFetcher

        mock_exchange.fetch_order.return_value = {
            "id": "12345",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "status": "filled",
            "amount": 1.0,
            "filled": 1.0,
            "remaining": 0,
            "price": 50000.0,
            "average": 49990.0,
            "timestamp": 1704067200000,
        }

        fetcher = CCXTOrderFetcher(mock_exchange)
        orders = await fetcher.fetch(order_id="12345", symbol="BTC/USDT")

        assert len(orders) == 1
        assert orders[0].order_id == "12345"
        mock_exchange.fetch_order.assert_called_once_with("12345", "BTC/USDT")

    @pytest.mark.asyncio
    async def test_ccxt_order_fetcher_with_since(
        self, mock_exchange: AsyncMock
    ) -> None:
        """Test CCXTOrderFetcher with since filter."""
        from libra.gateways.ccxt_fetchers import CCXTOrderFetcher

        mock_exchange.fetch_open_orders.return_value = []

        fetcher = CCXTOrderFetcher(mock_exchange)
        since = datetime(2024, 1, 1)
        await fetcher.fetch(status="open", since=since)

        # Verify since was converted to milliseconds
        call_args = mock_exchange.fetch_open_orders.call_args
        assert call_args.kwargs["since"] == int(since.timestamp() * 1000)

    # =========================================================================
    # Trade Fetcher Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_ccxt_trade_fetcher(self, mock_exchange: AsyncMock) -> None:
        """Test CCXTTradeFetcher with mocked exchange."""
        from libra.gateways.ccxt_fetchers import CCXTTradeFetcher

        mock_exchange.fetch_my_trades.return_value = [
            {
                "id": "T12345",
                "order": "O12345",
                "symbol": "BTC/USDT",
                "side": "buy",
                "amount": 0.1,
                "price": 50000.0,
                "cost": 5000.0,
                "timestamp": 1704067200000,
                "fee": {"cost": 5.0, "currency": "USDT"},
                "takerOrMaker": "taker",
            },
            {
                "id": "T12346",
                "order": "O12346",
                "symbol": "BTC/USDT",
                "side": "sell",
                "amount": 0.05,
                "price": 51000.0,
                "cost": 2550.0,
                "timestamp": 1704070800000,
                "fee": {"cost": 2.55, "currency": "USDT"},
                "takerOrMaker": "maker",
            },
        ]

        fetcher = CCXTTradeFetcher(mock_exchange)
        trades = await fetcher.fetch(symbol="BTC/USDT", limit=100)

        assert len(trades) == 2

        assert trades[0].trade_id == "T12345"
        assert trades[0].order_id == "O12345"
        assert trades[0].symbol == "BTC/USDT"
        assert trades[0].side == "buy"
        assert trades[0].amount == Decimal("0.1")
        assert trades[0].price == Decimal("50000")
        assert trades[0].cost == Decimal("5000")
        assert trades[0].fee == Decimal("5")
        assert trades[0].fee_currency == "USDT"
        assert trades[0].taker_or_maker == "taker"

        assert trades[1].trade_id == "T12346"
        assert trades[1].side == "sell"
        assert trades[1].taker_or_maker == "maker"
        mock_exchange.fetch_my_trades.assert_called_once()

    @pytest.mark.asyncio
    async def test_ccxt_trade_fetcher_with_since(
        self, mock_exchange: AsyncMock
    ) -> None:
        """Test CCXTTradeFetcher with since filter."""
        from libra.gateways.ccxt_fetchers import CCXTTradeFetcher

        mock_exchange.fetch_my_trades.return_value = []

        fetcher = CCXTTradeFetcher(mock_exchange)
        since = datetime(2024, 1, 1)
        await fetcher.fetch(symbol="BTC/USDT", since=since, limit=50)

        # Verify since was converted to milliseconds
        call_args = mock_exchange.fetch_my_trades.call_args
        assert call_args.kwargs["since"] == int(since.timestamp() * 1000)
        assert call_args.kwargs["limit"] == 50

    def test_global_registry_has_new_fetchers(self) -> None:
        """Test global registry has new CCXT fetchers registered."""
        # Import to trigger registration
        from libra.gateways import ccxt_fetchers  # noqa: F401

        data_types = fetcher_registry.list_data_types("ccxt")
        assert "position" in data_types
        assert "order" in data_types
        assert "trade" in data_types
