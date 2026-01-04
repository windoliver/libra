"""
Integration tests for CCXT Gateway.

These tests connect to real exchanges to verify functionality.

Test Categories:
1. Public data tests - No API key required (uses mainnet for public data)
2. Authenticated tests - Require BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_SECRET

Environment Variables:
    BINANCE_TESTNET_API_KEY: API key for Binance testnet
    BINANCE_TESTNET_SECRET: Secret for Binance testnet

To get testnet credentials:
1. Go to https://testnet.binance.vision/
2. Login with GitHub
3. Generate API key

Running tests:
    # Public data only (no API key needed)
    pytest tests/integration/test_ccxt_gateway.py -v -m "not authenticated"

    # All tests (requires API keys)
    export BINANCE_TESTNET_API_KEY="your-key"
    export BINANCE_TESTNET_SECRET="your-secret"
    pytest tests/integration/test_ccxt_gateway.py -v

Note: Some tests may be skipped if the exchange is geo-restricted.
"""

import os
from decimal import Decimal

import pytest

from libra.gateways import (
    CCXTGateway,
    InsufficientFundsError,
    Order,
    OrderError,
    OrderSide,
    OrderType,
)
from libra.gateways.protocol import ConnectionError as GatewayConnectionError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def binance_config() -> dict:
    """Configuration for Binance (public data, mainnet)."""
    return {}  # No testnet for public data - mainnet works without API keys


@pytest.fixture
def binance_testnet_auth_config() -> dict | None:
    """Configuration for Binance testnet with authentication."""
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY")
    secret = os.environ.get("BINANCE_TESTNET_SECRET")

    if not api_key or not secret:
        return None

    return {
        "api_key": api_key,
        "secret": secret,
        "testnet": True,
    }


def has_testnet_credentials() -> bool:
    """Check if testnet credentials are available."""
    return bool(
        os.environ.get("BINANCE_TESTNET_API_KEY") and os.environ.get("BINANCE_TESTNET_SECRET")
    )


# Skip marker for authenticated tests
requires_auth = pytest.mark.skipif(
    not has_testnet_credentials(),
    reason="Requires BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_SECRET env vars",
)


def skip_if_geo_restricted(func):
    """Decorator to skip tests if exchange is geo-restricted."""
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except GatewayConnectionError as e:
            if "451" in str(e) or "restricted" in str(e).lower():
                pytest.skip("Exchange geo-restricted from this location")
            raise
        except Exception as e:
            if "451" in str(e) or "restricted" in str(e).lower():
                pytest.skip("Exchange geo-restricted from this location")
            raise

    return wrapper


# =============================================================================
# Public Data Tests (No API Key Required)
# =============================================================================


class TestCCXTGatewayConnection:
    """Tests for gateway connection (public data)."""

    @pytest.mark.asyncio
    @skip_if_geo_restricted
    async def test_connect_to_binance(self, binance_config: dict) -> None:
        """Test connecting to Binance."""
        gateway = CCXTGateway("binance", binance_config)

        try:
            await gateway.connect()

            assert gateway.is_connected
            assert gateway.name == "binance"

            # Should have loaded markets
            market_info = gateway.get_market_info("BTC/USDT")
            assert market_info is not None
            assert market_info["symbol"] == "BTC/USDT"

        finally:
            await gateway.disconnect()

        assert not gateway.is_connected

    @pytest.mark.asyncio
    @skip_if_geo_restricted
    async def test_context_manager(self, binance_config: dict) -> None:
        """Test async context manager."""
        async with CCXTGateway("binance", binance_config) as gateway:
            assert gateway.is_connected

        assert not gateway.is_connected


class TestCCXTGatewayPublicData:
    """Tests for public market data (no authentication required)."""

    @pytest.mark.asyncio
    @skip_if_geo_restricted
    async def test_get_ticker(self, binance_config: dict) -> None:
        """Test fetching ticker data."""
        async with CCXTGateway("binance", binance_config) as gateway:
            ticker = await gateway.get_ticker("BTC/USDT")

            assert ticker.symbol == "BTC/USDT"
            assert ticker.bid > Decimal("0")
            assert ticker.ask > Decimal("0")
            assert ticker.last > Decimal("0")
            assert ticker.ask >= ticker.bid  # Ask should be >= bid
            assert ticker.spread >= Decimal("0")

    @pytest.mark.asyncio
    @skip_if_geo_restricted
    async def test_get_orderbook(self, binance_config: dict) -> None:
        """Test fetching order book."""
        async with CCXTGateway("binance", binance_config) as gateway:
            orderbook = await gateway.get_orderbook("BTC/USDT", depth=10)

            assert orderbook.symbol == "BTC/USDT"
            assert len(orderbook.bids) > 0
            assert len(orderbook.asks) > 0
            assert orderbook.best_bid is not None
            assert orderbook.best_ask is not None
            assert orderbook.best_ask >= orderbook.best_bid

            # Verify bid/ask structure (price, size)
            for price, size in orderbook.bids[:5]:
                assert price > Decimal("0")
                assert size > Decimal("0")

            for price, size in orderbook.asks[:5]:
                assert price > Decimal("0")
                assert size > Decimal("0")

    @pytest.mark.asyncio
    @skip_if_geo_restricted
    async def test_get_multiple_tickers(self, binance_config: dict) -> None:
        """Test fetching multiple tickers."""
        async with CCXTGateway("binance", binance_config) as gateway:
            symbols = ["BTC/USDT", "ETH/USDT"]

            for symbol in symbols:
                ticker = await gateway.get_ticker(symbol)
                assert ticker.symbol == symbol
                assert ticker.last > Decimal("0")


class TestCCXTGatewayStreaming:
    """Tests for WebSocket streaming."""

    @pytest.mark.asyncio
    @skip_if_geo_restricted
    async def test_subscribe_and_stream_ticks(self, binance_config: dict) -> None:
        """Test subscribing and streaming tick data."""
        async with CCXTGateway("binance", binance_config) as gateway:
            # Subscribe to BTC/USDT
            await gateway.subscribe(["BTC/USDT"])

            # Collect a few ticks
            ticks_received = []
            tick_count = 0
            max_ticks = 3

            async for tick in gateway.stream_ticks():
                ticks_received.append(tick)
                tick_count += 1

                assert tick.symbol == "BTC/USDT"
                assert tick.bid > Decimal("0")
                assert tick.ask > Decimal("0")

                if tick_count >= max_ticks:
                    break

            assert len(ticks_received) >= max_ticks

    @pytest.mark.asyncio
    @skip_if_geo_restricted
    async def test_subscribe_multiple_symbols(self, binance_config: dict) -> None:
        """Test subscribing to multiple symbols."""
        async with CCXTGateway("binance", binance_config) as gateway:
            symbols = ["BTC/USDT", "ETH/USDT"]
            await gateway.subscribe(symbols)

            # Collect ticks from both symbols
            symbols_seen = set()
            tick_count = 0
            max_ticks = 10

            async for tick in gateway.stream_ticks():
                symbols_seen.add(tick.symbol)
                tick_count += 1

                if tick_count >= max_ticks:
                    break

            # Should have seen at least one symbol
            assert len(symbols_seen) >= 1
            # All seen symbols should be in our subscription
            assert symbols_seen.issubset(set(symbols))


# =============================================================================
# Authenticated Tests (Requires API Keys)
# =============================================================================


@pytest.mark.authenticated
class TestCCXTGatewayAuthenticated:
    """Tests requiring authentication (testnet API keys)."""

    @requires_auth
    @pytest.mark.asyncio
    async def test_get_balances(self, binance_testnet_auth_config: dict) -> None:
        """Test fetching account balances."""
        if binance_testnet_auth_config is None:
            pytest.skip("No testnet credentials")

        async with CCXTGateway("binance", binance_testnet_auth_config) as gateway:
            balances = await gateway.get_balances()

            # Should return a dict (may be empty on fresh testnet account)
            assert isinstance(balances, dict)

            # If we have balances, verify structure
            for currency, balance in balances.items():
                assert balance.currency == currency
                assert balance.total >= Decimal("0")
                assert balance.available >= Decimal("0")
                assert balance.locked >= Decimal("0")

    @requires_auth
    @pytest.mark.asyncio
    async def test_get_open_orders(self, binance_testnet_auth_config: dict) -> None:
        """Test fetching open orders."""
        if binance_testnet_auth_config is None:
            pytest.skip("No testnet credentials")

        async with CCXTGateway("binance", binance_testnet_auth_config) as gateway:
            orders = await gateway.get_open_orders("BTC/USDT")

            # Should return a list (may be empty)
            assert isinstance(orders, list)

    @requires_auth
    @pytest.mark.asyncio
    async def test_submit_and_cancel_limit_order(self, binance_testnet_auth_config: dict) -> None:
        """Test submitting and cancelling a limit order."""
        if binance_testnet_auth_config is None:
            pytest.skip("No testnet credentials")

        async with CCXTGateway("binance", binance_testnet_auth_config) as gateway:
            # Get current price
            ticker = await gateway.get_ticker("BTC/USDT")

            # Place limit buy order far below market (won't fill)
            limit_price = ticker.bid * Decimal("0.5")  # 50% below market

            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                amount=Decimal("0.001"),  # Minimum size
                price=limit_price,
            )

            try:
                result = await gateway.submit_order(order)

                assert result.order_id is not None
                assert result.symbol == "BTC/USDT"
                assert result.side == OrderSide.BUY

                # Cancel the order
                cancelled = await gateway.cancel_order(result.order_id, "BTC/USDT")
                assert cancelled

            except InsufficientFundsError:
                pytest.skip("Insufficient funds on testnet account")
            except OrderError as e:
                if "MIN_NOTIONAL" in str(e):
                    pytest.skip("Order too small for exchange minimum")
                raise


# =============================================================================
# Performance Tests
# =============================================================================


class TestCCXTGatewayPerformance:
    """Performance-related tests."""

    @pytest.mark.asyncio
    @skip_if_geo_restricted
    async def test_ticker_latency(self, binance_config: dict) -> None:
        """Test that ticker fetch is reasonably fast."""
        import time

        async with CCXTGateway("binance", binance_config) as gateway:
            # Warm up
            await gateway.get_ticker("BTC/USDT")

            # Measure latency
            latencies = []
            for _ in range(5):
                start = time.perf_counter()
                await gateway.get_ticker("BTC/USDT")
                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)

            avg_latency = sum(latencies) / len(latencies)

            # Should be under 1 second on average (network dependent)
            assert avg_latency < 1000, f"Average latency too high: {avg_latency}ms"

    @pytest.mark.asyncio
    @skip_if_geo_restricted
    async def test_orderbook_latency(self, binance_config: dict) -> None:
        """Test that orderbook fetch is reasonably fast."""
        import time

        async with CCXTGateway("binance", binance_config) as gateway:
            # Warm up
            await gateway.get_orderbook("BTC/USDT")

            # Measure latency
            latencies = []
            for _ in range(5):
                start = time.perf_counter()
                await gateway.get_orderbook("BTC/USDT", depth=20)
                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)

            avg_latency = sum(latencies) / len(latencies)

            # Should be under 1 second on average
            assert avg_latency < 1000, f"Average latency too high: {avg_latency}ms"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestCCXTGatewayErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    @skip_if_geo_restricted
    async def test_invalid_symbol(self, binance_config: dict) -> None:
        """Test handling of invalid symbol."""
        async with CCXTGateway("binance", binance_config) as gateway:
            with pytest.raises(Exception):  # CCXT raises various exceptions
                await gateway.get_ticker("INVALID/SYMBOL")

    @pytest.mark.asyncio
    @skip_if_geo_restricted
    async def test_subscribe_invalid_symbol(self, binance_config: dict) -> None:
        """Test subscribing to invalid symbol."""
        async with CCXTGateway("binance", binance_config) as gateway:
            with pytest.raises(ValueError):
                await gateway.subscribe(["INVALID/SYMBOL"])

    @pytest.mark.asyncio
    async def test_operations_when_disconnected(self) -> None:
        """Test that operations fail gracefully when disconnected."""
        from libra.gateways.protocol import ConnectionError

        gateway = CCXTGateway("binance", {})
        # Don't connect

        with pytest.raises(ConnectionError):
            await gateway.get_ticker("BTC/USDT")

        with pytest.raises(ConnectionError):
            await gateway.get_orderbook("BTC/USDT")
