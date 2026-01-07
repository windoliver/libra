"""Tests for AsyncQuestDBClient.

Note: These tests use mocking to avoid requiring a real QuestDB instance.
For integration tests with a real database, see tests/integration/test_questdb_integration.py
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libra.data.config import QuestDBConfig
from libra.data.questdb import AsyncQuestDBClient
from libra.gateways.protocol import Tick
from libra.strategies.protocol import Bar


@pytest.fixture
def config() -> QuestDBConfig:
    """Create test configuration."""
    return QuestDBConfig(
        host="localhost",
        pg_port=8812,
        http_port=9000,
    )


@pytest.fixture
def client(config: QuestDBConfig) -> AsyncQuestDBClient:
    """Create test client (not connected)."""
    return AsyncQuestDBClient(config)


class TestAsyncQuestDBClientInit:
    """Tests for client initialization."""

    def test_init_creates_client(self, config: QuestDBConfig) -> None:
        """Test that client is created with config."""
        client = AsyncQuestDBClient(config)

        assert client.config == config
        assert client.is_connected is False
        assert client._pool is None
        assert client._sender is None

    def test_config_property(self, client: AsyncQuestDBClient, config: QuestDBConfig) -> None:
        """Test config property returns the configuration."""
        assert client.config == config
        assert client.config.host == "localhost"


class TestAsyncQuestDBClientConnection:
    """Tests for connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_not_connected_initially(self, client: AsyncQuestDBClient) -> None:
        """Test that client is not connected initially."""
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_sets_connected(self, client: AsyncQuestDBClient) -> None:
        """Test that connect sets connected state."""
        # Patch at the point of import (inside connect method)
        mock_asyncpg = MagicMock()
        mock_pool = AsyncMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        mock_sender_module = MagicMock()
        mock_sender = MagicMock()
        mock_sender_module.Sender.from_conf.return_value = mock_sender

        with patch.dict(
            "sys.modules",
            {"asyncpg": mock_asyncpg, "questdb.ingress": mock_sender_module},
        ):
            await client.connect()

            assert client.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_idempotent(self, client: AsyncQuestDBClient) -> None:
        """Test that calling connect twice doesn't reconnect."""
        mock_asyncpg = MagicMock()
        mock_pool = AsyncMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        mock_sender_module = MagicMock()
        mock_sender = MagicMock()
        mock_sender_module.Sender.from_conf.return_value = mock_sender

        with patch.dict(
            "sys.modules",
            {"asyncpg": mock_asyncpg, "questdb.ingress": mock_sender_module},
        ):
            await client.connect()
            await client.connect()  # Second call should be no-op

            # Should only be called once
            assert mock_asyncpg.create_pool.call_count == 1

    @pytest.mark.asyncio
    async def test_close_disconnects(self, client: AsyncQuestDBClient) -> None:
        """Test that close disconnects the client."""
        mock_asyncpg = MagicMock()
        mock_pool = AsyncMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        mock_sender_module = MagicMock()
        mock_sender = MagicMock()
        mock_sender_module.Sender.from_conf.return_value = mock_sender

        with patch.dict(
            "sys.modules",
            {"asyncpg": mock_asyncpg, "questdb.ingress": mock_sender_module},
        ):
            await client.connect()
            assert client.is_connected is True

            await client.close()
            assert client.is_connected is False
            mock_pool.close.assert_called_once()
            mock_sender.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_idempotent(self, client: AsyncQuestDBClient) -> None:
        """Test that calling close twice is safe."""
        await client.close()  # First call on unconnected client
        await client.close()  # Second call should be no-op

        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_context_manager(self, client: AsyncQuestDBClient) -> None:
        """Test async context manager support."""
        mock_asyncpg = MagicMock()
        mock_pool = AsyncMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        mock_sender_module = MagicMock()
        mock_sender = MagicMock()
        mock_sender_module.Sender.from_conf.return_value = mock_sender

        with patch.dict(
            "sys.modules",
            {"asyncpg": mock_asyncpg, "questdb.ingress": mock_sender_module},
        ):
            async with client:
                assert client.is_connected is True

            assert client.is_connected is False


class TestAsyncQuestDBClientQueries:
    """Tests for query operations."""

    @pytest.mark.asyncio
    async def test_get_ticks_not_connected_raises(self, client: AsyncQuestDBClient) -> None:
        """Test that querying without connection raises."""
        with pytest.raises(ConnectionError, match="Not connected"):
            await client.get_ticks(
                "BTC/USDT",
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
            )

    @pytest.mark.asyncio
    async def test_get_bars_not_connected_raises(self, client: AsyncQuestDBClient) -> None:
        """Test that querying bars without connection raises."""
        with pytest.raises(ConnectionError, match="Not connected"):
            await client.get_bars(
                "BTC/USDT",
                "1h",
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
            )

    @pytest.mark.asyncio
    async def test_ingest_tick_not_connected_raises(self, client: AsyncQuestDBClient) -> None:
        """Test that ingesting without connection raises."""
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50001"),
            last=Decimal("50000.50"),
            timestamp_ns=1704067200000000000,
        )

        with pytest.raises(ConnectionError, match="Not connected"):
            await client.ingest_tick(tick)


class TestAsyncQuestDBClientHealth:
    """Tests for health check."""

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, client: AsyncQuestDBClient) -> None:
        """Test health check returns False when not connected."""
        result = await client.health_check()
        assert result is False


class TestTickDataStructures:
    """Tests for Tick data handling."""

    def test_tick_creation(self) -> None:
        """Test creating a Tick for ingestion."""
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50001"),
            last=Decimal("50000.50"),
            timestamp_ns=1704067200000000000,
            bid_size=Decimal("1.5"),
            ask_size=Decimal("2.0"),
            volume_24h=Decimal("15000"),
        )

        assert tick.symbol == "BTC/USDT"
        assert tick.bid == Decimal("50000")
        assert tick.ask == Decimal("50001")
        assert tick.last == Decimal("50000.50")
        assert tick.mid == Decimal("50000.5")
        assert tick.spread == Decimal("1")


class TestBarDataStructures:
    """Tests for Bar data handling."""

    def test_bar_creation(self) -> None:
        """Test creating a Bar for ingestion."""
        bar = Bar(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ns=1704067200000000000,
            open=Decimal("50000"),
            high=Decimal("50500"),
            low=Decimal("49800"),
            close=Decimal("50200"),
            volume=Decimal("100.5"),
        )

        assert bar.symbol == "BTC/USDT"
        assert bar.timeframe == "1h"
        assert bar.open == Decimal("50000")
        assert bar.high == Decimal("50500")
        assert bar.low == Decimal("49800")
        assert bar.close == Decimal("50200")
        assert bar.volume == Decimal("100.5")
        assert bar.is_bullish is True  # close > open
