"""
E2E tests for Provider/Fetcher Pattern using REAL market data from Binance.

Demonstrates Issue #27: Provider/Fetcher Pattern for Gateway Layer.
Uses actual live data to validate the 3-stage TET pipeline.
"""

from __future__ import annotations

import json
import time
import urllib.request
from decimal import Decimal
from typing import Any

import pytest

from libra.gateways.fetcher import (
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


# =============================================================================
# Direct Binance API Fetchers (No CCXT dependency for E2E tests)
# =============================================================================


class BinanceBarFetcher(GatewayFetcher[BarQuery, list[Bar]]):
    """Bar fetcher using Binance public API directly."""

    BASE_URL = "https://api.binance.com/api/v3"

    def transform_query(self, params: dict[str, Any]) -> BarQuery:
        symbol = params.get("symbol")
        if not symbol:
            raise ValueError("symbol is required")
        return BarQuery(
            symbol=symbol,
            interval=params.get("interval", "1h"),
            limit=params.get("limit", 100),
        )

    async def extract_data(self, query: BarQuery, **_kwargs: Any) -> Any:
        # Convert symbol format: "BTC/USDT" -> "BTCUSDT"
        binance_symbol = query.symbol.replace("/", "")
        url = f"{self.BASE_URL}/klines?symbol={binance_symbol}&interval={query.interval}&limit={query.limit}"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            pytest.skip(f"Could not fetch Binance data: {e}")
            return []

    def transform_data(self, query: BarQuery, raw: Any) -> list[Bar]:
        return [
            Bar(
                symbol=query.symbol,
                interval=query.interval,
                timestamp_ns=timestamp_to_ns(candle[0]),  # ms
                open=Decimal(candle[1]),
                high=Decimal(candle[2]),
                low=Decimal(candle[3]),
                close=Decimal(candle[4]),
                volume=Decimal(candle[5]),
            )
            for candle in raw
        ]


class BinanceQuoteFetcher(GatewayFetcher[TickQuery, Quote]):
    """Quote fetcher using Binance public API directly."""

    BASE_URL = "https://api.binance.com/api/v3"

    def transform_query(self, params: dict[str, Any]) -> TickQuery:
        symbol = params.get("symbol")
        if not symbol:
            raise ValueError("symbol is required")
        return TickQuery(symbol=symbol)

    async def extract_data(self, query: TickQuery, **_kwargs: Any) -> Any:
        binance_symbol = query.symbol.replace("/", "")
        url = f"{self.BASE_URL}/ticker/bookTicker?symbol={binance_symbol}"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                book_ticker = json.loads(response.read().decode())

            # Also get 24h ticker for last price
            url_24h = f"{self.BASE_URL}/ticker/24hr?symbol={binance_symbol}"
            with urllib.request.urlopen(url_24h, timeout=10) as response:
                ticker_24h = json.loads(response.read().decode())

            return {**book_ticker, **ticker_24h}
        except Exception as e:
            pytest.skip(f"Could not fetch Binance data: {e}")
            return {}

    def transform_data(self, query: TickQuery, raw: Any) -> Quote:
        return Quote(
            symbol=query.symbol,
            bid=Decimal(raw.get("bidPrice", "0")),
            ask=Decimal(raw.get("askPrice", "0")),
            last=Decimal(raw.get("lastPrice", "0")),
            timestamp_ns=timestamp_to_ns(int(raw.get("closeTime", 0))),
            bid_size=Decimal(raw.get("bidQty", "0")),
            ask_size=Decimal(raw.get("askQty", "0")),
            volume_24h=Decimal(raw.get("quoteVolume", "0")),
            high_24h=Decimal(raw.get("highPrice", "0")),
            low_24h=Decimal(raw.get("lowPrice", "0")),
            change_24h_pct=Decimal(raw.get("priceChangePercent", "0")),
        )


class BinanceOrderBookFetcher(GatewayFetcher[OrderBookQuery, OrderBookSnapshot]):
    """Order book fetcher using Binance public API directly."""

    BASE_URL = "https://api.binance.com/api/v3"

    def transform_query(self, params: dict[str, Any]) -> OrderBookQuery:
        symbol = params.get("symbol")
        if not symbol:
            raise ValueError("symbol is required")
        return OrderBookQuery(
            symbol=symbol,
            depth=params.get("depth", 20),
        )

    async def extract_data(self, query: OrderBookQuery, **_kwargs: Any) -> Any:
        binance_symbol = query.symbol.replace("/", "")
        url = f"{self.BASE_URL}/depth?symbol={binance_symbol}&limit={query.depth}"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            pytest.skip(f"Could not fetch Binance data: {e}")
            return {}

    def transform_data(self, query: OrderBookQuery, raw: Any) -> OrderBookSnapshot:
        bids = [
            OrderBookLevel(price=Decimal(level[0]), size=Decimal(level[1]))
            for level in raw.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=Decimal(level[0]), size=Decimal(level[1]))
            for level in raw.get("asks", [])
        ]

        return OrderBookSnapshot(
            symbol=query.symbol,
            bids=bids,
            asks=asks,
            timestamp_ns=timestamp_to_ns(int(time.time() * 1000)),
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def bar_fetcher() -> BinanceBarFetcher:
    """Create a Binance bar fetcher."""
    return BinanceBarFetcher()


@pytest.fixture
def quote_fetcher() -> BinanceQuoteFetcher:
    """Create a Binance quote fetcher."""
    return BinanceQuoteFetcher()


@pytest.fixture
def orderbook_fetcher() -> BinanceOrderBookFetcher:
    """Create a Binance order book fetcher."""
    return BinanceOrderBookFetcher()


# =============================================================================
# E2E Tests for Bar Fetcher
# =============================================================================


class TestBarFetcherRealData:
    """E2E tests for BarFetcher with real Binance data."""

    @pytest.mark.asyncio
    async def test_fetch_btc_hourly_bars(self, bar_fetcher: BinanceBarFetcher) -> None:
        """Test fetching real BTC/USDT hourly bars."""
        bars = await bar_fetcher.fetch(
            symbol="BTC/USDT",
            interval="1h",
            limit=100,
        )

        # Validate results
        assert len(bars) == 100
        assert all(isinstance(b, Bar) for b in bars)

        # Check bar structure
        for bar in bars:
            assert bar.symbol == "BTC/USDT"
            assert bar.interval == "1h"
            assert bar.timestamp_ns > 0
            assert bar.open > 0
            assert bar.high >= bar.low
            assert bar.close > 0
            assert bar.volume >= 0

        # Print sample data
        print(f"\nFetched {len(bars)} BTC/USDT 1h bars")
        print(f"  First bar: {bars[0].datetime} - O:{bars[0].open} H:{bars[0].high} L:{bars[0].low} C:{bars[0].close}")
        print(f"  Last bar:  {bars[-1].datetime} - O:{bars[-1].open} H:{bars[-1].high} L:{bars[-1].low} C:{bars[-1].close}")

    @pytest.mark.asyncio
    async def test_fetch_eth_daily_bars(self, bar_fetcher: BinanceBarFetcher) -> None:
        """Test fetching real ETH/USDT daily bars."""
        bars = await bar_fetcher.fetch(
            symbol="ETH/USDT",
            interval="1d",
            limit=30,
        )

        assert len(bars) == 30
        assert bars[0].symbol == "ETH/USDT"
        assert bars[0].interval == "1d"

        # Calculate price range over period
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        print(f"\nETH/USDT 30-day range: ${min(lows):,.2f} - ${max(highs):,.2f}")

    @pytest.mark.asyncio
    async def test_fetch_multiple_symbols(self, bar_fetcher: BinanceBarFetcher) -> None:
        """Test fetching bars for multiple symbols."""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

        results = {}
        for symbol in symbols:
            bars = await bar_fetcher.fetch(symbol=symbol, interval="1h", limit=24)
            results[symbol] = bars

        # Print comparison
        print("\n" + "=" * 60)
        print("24-Hour Price Comparison (Fetcher Pattern)")
        print("=" * 60)
        print(f"{'Symbol':<12} {'Open':>12} {'Close':>12} {'Change %':>10}")
        print("-" * 60)

        for symbol, bars in results.items():
            if bars:
                open_price = bars[0].open
                close_price = bars[-1].close
                change_pct = ((close_price - open_price) / open_price) * 100
                print(f"{symbol:<12} ${float(open_price):>10,.2f} ${float(close_price):>10,.2f} {float(change_pct):>9.2f}%")

        print("=" * 60)

    @pytest.mark.asyncio
    async def test_transform_query_stage(self, bar_fetcher: BinanceBarFetcher) -> None:
        """Test the transform_query stage independently."""
        # Stage 1: transform_query
        query = bar_fetcher.transform_query({
            "symbol": "BTC/USDT",
            "interval": "15m",
            "limit": 50,
        })

        assert isinstance(query, BarQuery)
        assert query.symbol == "BTC/USDT"
        assert query.interval == "15m"
        assert query.limit == 50

    @pytest.mark.asyncio
    async def test_fetch_with_pre_built_query(self, bar_fetcher: BinanceBarFetcher) -> None:
        """Test fetch_with_query using pre-built query object."""
        # Create query directly
        query = BarQuery(
            symbol="BTC/USDT",
            interval="5m",
            limit=10,
        )

        bars = await bar_fetcher.fetch_with_query(query)

        assert len(bars) == 10
        assert all(b.interval == "5m" for b in bars)


# =============================================================================
# E2E Tests for Quote Fetcher
# =============================================================================


class TestQuoteFetcherRealData:
    """E2E tests for QuoteFetcher with real Binance data."""

    @pytest.mark.asyncio
    async def test_fetch_btc_quote(self, quote_fetcher: BinanceQuoteFetcher) -> None:
        """Test fetching real BTC/USDT quote."""
        quote = await quote_fetcher.fetch(symbol="BTC/USDT")

        # Validate quote structure
        assert isinstance(quote, Quote)
        assert quote.symbol == "BTC/USDT"
        assert quote.bid > 0
        assert quote.ask > 0
        assert quote.last > 0
        assert quote.ask >= quote.bid  # Ask should be >= bid

        # Print quote details
        print(f"\nBTC/USDT Quote:")
        print(f"  Bid: ${float(quote.bid):,.2f}")
        print(f"  Ask: ${float(quote.ask):,.2f}")
        print(f"  Last: ${float(quote.last):,.2f}")
        print(f"  Mid: ${float(quote.mid):,.2f}")
        print(f"  Spread: ${float(quote.spread):,.2f} ({float(quote.spread_bps):.2f} bps)")

    @pytest.mark.asyncio
    async def test_fetch_multiple_quotes(self, quote_fetcher: BinanceQuoteFetcher) -> None:
        """Test fetching quotes for multiple symbols."""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"]

        # Print header
        print("\n" + "=" * 70)
        print("Real-Time Quotes (Fetcher Pattern)")
        print("=" * 70)
        print(f"{'Symbol':<12} {'Bid':>12} {'Ask':>12} {'Spread':>10} {'Spread BPS':>10}")
        print("-" * 70)

        for symbol in symbols:
            quote = await quote_fetcher.fetch(symbol=symbol)
            print(
                f"{symbol:<12} "
                f"${float(quote.bid):>10,.2f} "
                f"${float(quote.ask):>10,.2f} "
                f"${float(quote.spread):>8,.4f} "
                f"{float(quote.spread_bps):>9.2f}"
            )

        print("=" * 70)


# =============================================================================
# E2E Tests for OrderBook Fetcher
# =============================================================================


class TestOrderBookFetcherRealData:
    """E2E tests for OrderBookFetcher with real Binance data."""

    @pytest.mark.asyncio
    async def test_fetch_btc_orderbook(self, orderbook_fetcher: BinanceOrderBookFetcher) -> None:
        """Test fetching real BTC/USDT order book."""
        orderbook = await orderbook_fetcher.fetch(symbol="BTC/USDT", depth=20)

        # Validate structure
        assert isinstance(orderbook, OrderBookSnapshot)
        assert orderbook.symbol == "BTC/USDT"
        assert len(orderbook.bids) <= 20
        assert len(orderbook.asks) <= 20
        assert orderbook.best_bid is not None
        assert orderbook.best_ask is not None
        assert orderbook.best_ask >= orderbook.best_bid

        # Print order book summary
        print(f"\nBTC/USDT Order Book (depth=20):")
        print(f"  Best Bid: ${float(orderbook.best_bid):,.2f}")
        print(f"  Best Ask: ${float(orderbook.best_ask):,.2f}")
        print(f"  Mid Price: ${float(orderbook.mid):,.2f}")
        print(f"  Spread: ${float(orderbook.spread):,.2f}")

        # Calculate total liquidity at top 5 levels
        bid_liquidity = sum(level.size for level in orderbook.bids[:5])
        ask_liquidity = sum(level.size for level in orderbook.asks[:5])
        print(f"  Top 5 Bid Liquidity: {float(bid_liquidity):.4f} BTC")
        print(f"  Top 5 Ask Liquidity: {float(ask_liquidity):.4f} BTC")

    @pytest.mark.asyncio
    async def test_orderbook_structure(self, orderbook_fetcher: BinanceOrderBookFetcher) -> None:
        """Test order book level structure."""
        orderbook = await orderbook_fetcher.fetch(symbol="ETH/USDT", depth=10)

        # Bids should be sorted descending
        bid_prices = [level.price for level in orderbook.bids]
        assert bid_prices == sorted(bid_prices, reverse=True), "Bids should be sorted descending"

        # Asks should be sorted ascending
        ask_prices = [level.price for level in orderbook.asks]
        assert ask_prices == sorted(ask_prices), "Asks should be sorted ascending"

        print(f"\nETH/USDT Order Book verified:")
        print(f"  {len(orderbook.bids)} bid levels (descending)")
        print(f"  {len(orderbook.asks)} ask levels (ascending)")


# =============================================================================
# E2E Tests for Pipeline Stages
# =============================================================================


class TestFetcherPipelineRealData:
    """E2E tests demonstrating the 3-stage TET pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_stages(self, bar_fetcher: BinanceBarFetcher) -> None:
        """Test each stage of the TET pipeline independently."""
        # Stage 1: Transform Query
        print("\n" + "=" * 60)
        print("TET Pipeline Demonstration")
        print("=" * 60)

        params = {"symbol": "BTC/USDT", "interval": "1h", "limit": 5}
        print(f"\nInput params: {params}")

        query = bar_fetcher.transform_query(params)
        print(f"\nStage 1 - transform_query:")
        print(f"  Query type: {type(query).__name__}")
        print(f"  symbol: {query.symbol}")
        print(f"  interval: {query.interval}")
        print(f"  limit: {query.limit}")

        # Stage 2: Extract Data
        raw = await bar_fetcher.extract_data(query)
        print(f"\nStage 2 - extract_data:")
        print(f"  Raw data type: {type(raw).__name__}")
        print(f"  Number of candles: {len(raw)}")
        print(f"  First candle (raw): {raw[0]}")

        # Stage 3: Transform Data
        bars = bar_fetcher.transform_data(query, raw)
        print(f"\nStage 3 - transform_data:")
        print(f"  Output type: list[{type(bars[0]).__name__}]")
        print(f"  Number of bars: {len(bars)}")
        print(f"  First bar: {bars[0]}")

        print("=" * 60)

    @pytest.mark.asyncio
    async def test_registry_has_ccxt_fetchers(self) -> None:
        """Test that the fetcher registry has CCXT fetchers registered."""
        # Verify registry has ccxt gateway
        gateways = fetcher_registry.list_gateways()
        assert "ccxt" in gateways

        # Verify data types
        data_types = fetcher_registry.list_data_types("ccxt")
        assert "bar" in data_types
        assert "quote" in data_types
        assert "orderbook" in data_types
        assert "balance" in data_types

        print(f"\nFetcher Registry:")
        print(f"  Gateways: {gateways}")
        print(f"  CCXT data types: {data_types}")


# =============================================================================
# E2E Tests for Performance
# =============================================================================


class TestFetcherPerformance:
    """E2E tests for fetcher performance."""

    @pytest.mark.asyncio
    async def test_bar_fetch_performance(self, bar_fetcher: BinanceBarFetcher) -> None:
        """Test performance of bar fetching."""
        # Warm up
        await bar_fetcher.fetch(symbol="BTC/USDT", interval="1h", limit=10)

        # Time multiple fetches
        iterations = 5
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            bars = await bar_fetcher.fetch(symbol="BTC/USDT", interval="1h", limit=100)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            assert len(bars) == 100

        avg_time = sum(times) / len(times)
        print(f"\nBar Fetch Performance ({iterations} iterations):")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Min time: {min(times)*1000:.2f}ms")
        print(f"  Max time: {max(times)*1000:.2f}ms")

    @pytest.mark.asyncio
    async def test_quote_fetch_performance(self, quote_fetcher: BinanceQuoteFetcher) -> None:
        """Test performance of quote fetching."""
        # Warm up
        await quote_fetcher.fetch(symbol="BTC/USDT")

        # Time multiple fetches
        iterations = 5
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            quote = await quote_fetcher.fetch(symbol="BTC/USDT")
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            assert quote.bid > 0

        avg_time = sum(times) / len(times)
        print(f"\nQuote Fetch Performance ({iterations} iterations):")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Min time: {min(times)*1000:.2f}ms")
        print(f"  Max time: {max(times)*1000:.2f}ms")


# =============================================================================
# E2E Test for Complete Workflow
# =============================================================================


class TestCompleteWorkflow:
    """E2E test demonstrating complete fetcher workflow."""

    @pytest.mark.asyncio
    async def test_market_analysis_workflow(
        self,
        bar_fetcher: BinanceBarFetcher,
        quote_fetcher: BinanceQuoteFetcher,
        orderbook_fetcher: BinanceOrderBookFetcher,
    ) -> None:
        """
        Demonstrate a complete market analysis workflow using fetchers.

        This simulates what a trading strategy might do:
        1. Fetch current quote
        2. Fetch historical bars
        3. Check order book liquidity
        4. Calculate metrics
        """
        symbol = "BTC/USDT"

        # Step 1: Get current quote
        quote = await quote_fetcher.fetch(symbol=symbol)

        # Step 2: Get historical bars (last 24 hours)
        bars = await bar_fetcher.fetch(symbol=symbol, interval="1h", limit=24)

        # Step 3: Get order book
        orderbook = await orderbook_fetcher.fetch(symbol=symbol, depth=10)

        # Step 4: Calculate metrics
        # Price metrics
        current_price = quote.last
        price_24h_ago = bars[0].open
        price_change = ((current_price - price_24h_ago) / price_24h_ago) * 100

        # Volatility (simple: high-low range)
        avg_range = sum((b.high - b.low) / b.low * 100 for b in bars) / len(bars)

        # Liquidity (top 5 levels)
        bid_liquidity = sum(level.size * level.price for level in orderbook.bids[:5])
        ask_liquidity = sum(level.size * level.price for level in orderbook.asks[:5])

        # Print analysis
        print("\n" + "=" * 60)
        print(f"Market Analysis: {symbol}")
        print("=" * 60)
        print(f"\nPrice Metrics:")
        print(f"  Current Price: ${float(current_price):,.2f}")
        print(f"  24h Change: {float(price_change):+.2f}%")
        print(f"  Bid-Ask Spread: {float(quote.spread_bps):.2f} bps")

        print(f"\nVolatility:")
        print(f"  Avg Hourly Range: {float(avg_range):.3f}%")

        print(f"\nLiquidity (Top 5 Levels):")
        print(f"  Bid Side: ${float(bid_liquidity):,.2f}")
        print(f"  Ask Side: ${float(ask_liquidity):,.2f}")

        print("\n" + "=" * 60)
        print("Market analysis workflow completed successfully!")
        print("=" * 60)

        # Assertions
        assert current_price > 0
        assert len(bars) == 24
        assert orderbook.best_bid is not None
