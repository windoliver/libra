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
    BalanceQuery,
    Bar,
    BarQuery,
    BaseQuery,
    FetcherRegistry,
    GatewayFetcher,
    OrderBookLevel,
    OrderBookQuery,
    OrderBookSnapshot,
    Quote,
    TickQuery,
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
