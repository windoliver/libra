"""Tests for the RedisStateStore."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libra.core.state_store import RedisConfig, RedisStateStore
from libra.gateways.protocol import (
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> RedisConfig:
    """Create test Redis config."""
    return RedisConfig(
        url="redis://localhost:6379",
        max_connections=5,
        order_ttl=3600,
        position_ttl=7200,
    )


@pytest.fixture
def sample_order() -> OrderResult:
    """Create sample order for testing."""
    return OrderResult(
        order_id="order-123",
        symbol="BTC/USDT",
        status=OrderStatus.OPEN,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        amount=Decimal("0.1"),
        filled_amount=Decimal("0"),
        remaining_amount=Decimal("0.1"),
        average_price=None,
        fee=Decimal("0"),
        fee_currency="USDT",
        timestamp_ns=1000000000,
        client_order_id="client-123",
        price=Decimal("50000"),
    )


@pytest.fixture
def filled_order() -> OrderResult:
    """Create filled order for testing."""
    return OrderResult(
        order_id="order-456",
        symbol="ETH/USDT",
        status=OrderStatus.FILLED,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        amount=Decimal("1.0"),
        filled_amount=Decimal("1.0"),
        remaining_amount=Decimal("0"),
        average_price=Decimal("2000"),
        fee=Decimal("2"),
        fee_currency="USDT",
        timestamp_ns=2000000000,
    )


@pytest.fixture
def sample_position() -> Position:
    """Create sample position for testing."""
    return Position(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        amount=Decimal("0.5"),
        entry_price=Decimal("48000"),
        current_price=Decimal("50000"),
        unrealized_pnl=Decimal("1000"),
        realized_pnl=Decimal("0"),
        leverage=1,
    )


@pytest.fixture
def flat_position() -> Position:
    """Create flat (closed) position for testing."""
    return Position(
        symbol="ETH/USDT",
        side=PositionSide.FLAT,
        amount=Decimal("0"),
        entry_price=Decimal("0"),
        current_price=Decimal("2000"),
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("500"),
        leverage=1,
    )


# =============================================================================
# Configuration Tests
# =============================================================================


class TestRedisConfig:
    """Tests for RedisConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config should have reasonable values."""
        config = RedisConfig()
        assert config.url == "redis://localhost:6379"
        assert config.max_connections == 10
        assert config.order_ttl == 86400
        assert config.position_ttl == 86400 * 7

    def test_custom_config(self, config: RedisConfig) -> None:
        """Custom config should override defaults."""
        assert config.max_connections == 5
        assert config.order_ttl == 3600


# =============================================================================
# Connection Tests
# =============================================================================


class TestConnection:
    """Tests for Redis connection management."""

    @pytest.mark.asyncio
    async def test_connect_success(self, config: RedisConfig) -> None:
        """Should connect successfully to Redis."""
        store = RedisStateStore(config)

        with patch("redis.asyncio") as mock_redis:
            mock_pool = MagicMock()
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = mock_pool
            mock_redis.Redis.return_value = mock_client

            await store.connect()

            assert store.is_connected is True
            mock_redis.ConnectionPool.from_url.assert_called_once()
            mock_client.ping.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_connection(self, config: RedisConfig) -> None:
        """Should close connection and pool."""
        store = RedisStateStore(config)

        with patch("redis.asyncio") as mock_redis:
            mock_pool = AsyncMock()
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = mock_pool
            mock_redis.Redis.return_value = mock_client

            await store.connect()
            await store.close()

            assert store.is_connected is False
            mock_client.aclose.assert_awaited_once()
            mock_pool.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, config: RedisConfig) -> None:
        """Should work as async context manager."""
        with patch("redis.asyncio") as mock_redis:
            mock_pool = AsyncMock()
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = mock_pool
            mock_redis.Redis.return_value = mock_client

            async with RedisStateStore(config) as store:
                assert store.is_connected is True

            mock_client.aclose.assert_awaited_once()

    def test_ensure_connected_raises_when_disconnected(self, config: RedisConfig) -> None:
        """Should raise RuntimeError when not connected."""
        store = RedisStateStore(config)

        with pytest.raises(RuntimeError, match="Not connected to Redis"):
            store._ensure_connected()


# =============================================================================
# Order Persistence Tests
# =============================================================================


class TestOrderPersistence:
    """Tests for order persistence methods."""

    @pytest.mark.asyncio
    async def test_save_order(
        self, config: RedisConfig, sample_order: OrderResult
    ) -> None:
        """Should save order to Redis."""
        store = RedisStateStore(config)

        with patch("redis.asyncio") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis.Redis.return_value = mock_client

            await store.connect()
            await store.save_order(sample_order)

            mock_client.set.assert_awaited_once()
            call_args = mock_client.set.call_args
            assert "libra:orders:order-123" in call_args[0]
            assert call_args[1]["ex"] == config.order_ttl

    @pytest.mark.asyncio
    async def test_get_order_found(
        self, config: RedisConfig, sample_order: OrderResult
    ) -> None:
        """Should retrieve order from Redis."""
        store = RedisStateStore(config)

        with patch("redis.asyncio") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis.Redis.return_value = mock_client

            # Serialize order for mock response
            import msgspec
            encoded = msgspec.json.encode(sample_order)
            mock_client.get.return_value = encoded

            await store.connect()
            result = await store.get_order("order-123")

            assert result is not None
            assert result.order_id == sample_order.order_id
            assert result.symbol == sample_order.symbol

    @pytest.mark.asyncio
    async def test_get_order_not_found(self, config: RedisConfig) -> None:
        """Should return None for missing order."""
        store = RedisStateStore(config)

        with patch("redis.asyncio") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis.Redis.return_value = mock_client
            mock_client.get.return_value = None

            await store.connect()
            result = await store.get_order("nonexistent")

            assert result is None

    @pytest.mark.asyncio
    async def test_delete_order(self, config: RedisConfig) -> None:
        """Should delete order from Redis."""
        store = RedisStateStore(config)

        with patch("redis.asyncio") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis.Redis.return_value = mock_client
            mock_client.delete.return_value = 1

            await store.connect()
            result = await store.delete_order("order-123")

            assert result is True
            mock_client.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_open_orders(
        self, config: RedisConfig, sample_order: OrderResult, filled_order: OrderResult
    ) -> None:
        """Should return only open orders."""
        store = RedisStateStore(config)

        with patch("redis.asyncio") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis.Redis.return_value = mock_client

            # Mock scan_iter to return keys
            import msgspec
            open_encoded = msgspec.json.encode(sample_order)
            filled_encoded = msgspec.json.encode(filled_order)

            # Create async iterator class for proper async for support
            class MockScanIter:
                def __init__(self, keys: list[bytes]) -> None:
                    self.keys = keys
                    self.index = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self.index >= len(self.keys):
                        raise StopAsyncIteration
                    key = self.keys[self.index]
                    self.index += 1
                    return key

            # Use side_effect with lambda to return iterator when called
            mock_client.scan_iter = lambda **_: MockScanIter(
                [b"libra:orders:order-123", b"libra:orders:order-456"]
            )
            mock_client.get.side_effect = [open_encoded, filled_encoded]

            await store.connect()
            result = await store.get_open_orders()

            # Only open order should be returned
            assert len(result) == 1
            assert result[0].order_id == "order-123"


# =============================================================================
# Position Persistence Tests
# =============================================================================


class TestPositionPersistence:
    """Tests for position persistence methods."""

    @pytest.mark.asyncio
    async def test_save_position(
        self, config: RedisConfig, sample_position: Position
    ) -> None:
        """Should save position to Redis."""
        store = RedisStateStore(config)

        with patch("redis.asyncio") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis.Redis.return_value = mock_client

            await store.connect()
            await store.save_position(sample_position)

            mock_client.set.assert_awaited_once()
            call_args = mock_client.set.call_args
            assert "libra:positions:BTC/USDT" in call_args[0]
            assert call_args[1]["ex"] == config.position_ttl

    @pytest.mark.asyncio
    async def test_get_position_found(
        self, config: RedisConfig, sample_position: Position
    ) -> None:
        """Should retrieve position from Redis."""
        store = RedisStateStore(config)

        with patch("redis.asyncio") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis.Redis.return_value = mock_client

            import msgspec
            encoded = msgspec.json.encode(sample_position)
            mock_client.get.return_value = encoded

            await store.connect()
            result = await store.get_position("BTC/USDT")

            assert result is not None
            assert result.symbol == sample_position.symbol
            assert result.amount == sample_position.amount

    @pytest.mark.asyncio
    async def test_get_all_positions_excludes_flat(
        self, config: RedisConfig, sample_position: Position, flat_position: Position
    ) -> None:
        """Should exclude flat positions from results."""
        store = RedisStateStore(config)

        with patch("redis.asyncio") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis.Redis.return_value = mock_client

            import msgspec
            open_encoded = msgspec.json.encode(sample_position)
            flat_encoded = msgspec.json.encode(flat_position)

            # Create async iterator class for proper async for support
            class MockScanIter:
                def __init__(self, keys: list[bytes]) -> None:
                    self.keys = keys
                    self.index = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self.index >= len(self.keys):
                        raise StopAsyncIteration
                    key = self.keys[self.index]
                    self.index += 1
                    return key

            # Use lambda to return iterator when called
            mock_client.scan_iter = lambda **_: MockScanIter(
                [b"libra:positions:BTC/USDT", b"libra:positions:ETH/USDT"]
            )
            mock_client.get.side_effect = [open_encoded, flat_encoded]

            await store.connect()
            result = await store.get_all_positions()

            # Only non-flat position should be returned
            assert len(result) == 1
            assert result[0].symbol == "BTC/USDT"


# =============================================================================
# Crash Recovery Tests
# =============================================================================


class TestCrashRecovery:
    """Tests for crash recovery functionality."""

    @pytest.mark.asyncio
    async def test_restore_state(
        self, config: RedisConfig, sample_order: OrderResult, sample_position: Position
    ) -> None:
        """Should restore both orders and positions."""
        store = RedisStateStore(config)

        with patch("redis.asyncio") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis.Redis.return_value = mock_client

            import msgspec
            order_encoded = msgspec.json.encode(sample_order)
            position_encoded = msgspec.json.encode(sample_position)

            # Create async iterator class for proper async for support
            class MockScanIter:
                def __init__(self, keys: list[bytes]) -> None:
                    self.keys = keys
                    self.index = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self.index >= len(self.keys):
                        raise StopAsyncIteration
                    key = self.keys[self.index]
                    self.index += 1
                    return key

            # First call returns orders, second call returns positions
            call_count = [0]

            def mock_scan_iter(**_):
                call_count[0] += 1
                if call_count[0] == 1:
                    return MockScanIter([b"libra:orders:order-123"])
                return MockScanIter([b"libra:positions:BTC/USDT"])

            mock_client.scan_iter = mock_scan_iter
            mock_client.get.side_effect = [order_encoded, position_encoded]

            await store.connect()
            orders, positions = await store.restore_state()

            assert len(orders) == 1
            assert len(positions) == 1
            assert orders[0].order_id == "order-123"
            assert positions[0].symbol == "BTC/USDT"


# =============================================================================
# Metadata Tests
# =============================================================================


class TestMetadata:
    """Tests for metadata storage."""

    @pytest.mark.asyncio
    async def test_save_and_get_metadata(self, config: RedisConfig) -> None:
        """Should save and retrieve metadata."""
        store = RedisStateStore(config)

        with patch("redis.asyncio") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis.Redis.return_value = mock_client
            mock_client.get.return_value = b"test-value"

            await store.connect()

            await store.save_metadata("test-key", "test-value", ttl=60)
            mock_client.set.assert_awaited_once()

            result = await store.get_metadata("test-key")
            assert result == "test-value"


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_ping_success(self, config: RedisConfig) -> None:
        """Should return True when Redis is healthy."""
        store = RedisStateStore(config)

        with patch("redis.asyncio") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis.Redis.return_value = mock_client

            await store.connect()
            result = await store.ping()

            assert result is True
            # ping called during connect + during ping() method
            assert mock_client.ping.await_count == 2

    @pytest.mark.asyncio
    async def test_ping_failure(self, config: RedisConfig) -> None:
        """Should return False when Redis ping fails."""
        store = RedisStateStore(config)

        with patch("redis.asyncio") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis.Redis.return_value = mock_client

            await store.connect()

            # Make ping fail after connect
            mock_client.ping.side_effect = ConnectionError("Redis down")
            result = await store.ping()

            assert result is False

    def test_stats(self, config: RedisConfig) -> None:
        """Should return correct stats."""
        store = RedisStateStore(config)

        stats = store.stats
        assert stats["connected"] is False
        assert stats["max_connections"] == 5
