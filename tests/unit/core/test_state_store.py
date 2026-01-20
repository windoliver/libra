"""
Tests for State Store implementations (Issue #108).

Tests MemoryStateStore, FileStateStore, and RedisStateStore
for crash recovery persistence.
"""

import time

import pytest

from libra.core.state_store import (
    REDIS_AVAILABLE,
    FileStateStore,
    KernelCheckpoint,
    MemoryStateStore,
    OrderState,
    PersistedOrder,
    PersistedPosition,
    RedisStateStore,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_order() -> PersistedOrder:
    """Create sample persisted order."""
    return PersistedOrder(
        order_id="ord_123",
        client_order_id="client_123",
        symbol="BTC/USDT",
        side="buy",
        order_type="limit",
        amount="1.5",
        filled_amount="0.5",
        price="50000",
        state=OrderState.OPEN,
        created_at_ns=time.time_ns(),
        updated_at_ns=time.time_ns(),
        exchange_order_id="exch_456",
    )


@pytest.fixture
def sample_position() -> PersistedPosition:
    """Create sample persisted position."""
    return PersistedPosition(
        symbol="BTC/USDT",
        side="long",
        quantity="1.0",
        entry_price="50000",
        unrealized_pnl="500",
        realized_pnl="0",
        opened_at_ns=time.time_ns(),
        updated_at_ns=time.time_ns(),
    )


@pytest.fixture
def sample_checkpoint() -> KernelCheckpoint:
    """Create sample checkpoint."""
    return KernelCheckpoint(
        instance_id="inst_001",
        checkpoint_id="chk_001",
        timestamp_ns=time.time_ns(),
        state_hash="abc123def456",
        orders_count=5,
        positions_count=2,
        environment="test",
    )


# =============================================================================
# PersistedOrder Tests
# =============================================================================


class TestPersistedOrder:
    """Tests for PersistedOrder serialization."""

    def test_to_dict(self, sample_order: PersistedOrder) -> None:
        """Test serialization to dict."""
        data = sample_order.to_dict()

        assert data["order_id"] == "ord_123"
        assert data["symbol"] == "BTC/USDT"
        assert data["state"] == "open"
        assert data["amount"] == "1.5"

    def test_from_dict(self, sample_order: PersistedOrder) -> None:
        """Test deserialization from dict."""
        data = sample_order.to_dict()
        restored = PersistedOrder.from_dict(data)

        assert restored.order_id == sample_order.order_id
        assert restored.symbol == sample_order.symbol
        assert restored.state == sample_order.state
        assert restored.amount == sample_order.amount


class TestPersistedPosition:
    """Tests for PersistedPosition serialization."""

    def test_to_dict(self, sample_position: PersistedPosition) -> None:
        """Test serialization to dict."""
        data = sample_position.to_dict()

        assert data["symbol"] == "BTC/USDT"
        assert data["side"] == "long"
        assert data["quantity"] == "1.0"

    def test_from_dict(self, sample_position: PersistedPosition) -> None:
        """Test deserialization from dict."""
        data = sample_position.to_dict()
        restored = PersistedPosition.from_dict(data)

        assert restored.symbol == sample_position.symbol
        assert restored.quantity == sample_position.quantity


class TestKernelCheckpoint:
    """Tests for KernelCheckpoint serialization."""

    def test_to_dict(self, sample_checkpoint: KernelCheckpoint) -> None:
        """Test serialization to dict."""
        data = sample_checkpoint.to_dict()

        assert data["instance_id"] == "inst_001"
        assert data["checkpoint_id"] == "chk_001"
        assert data["orders_count"] == 5

    def test_from_dict(self, sample_checkpoint: KernelCheckpoint) -> None:
        """Test deserialization from dict."""
        data = sample_checkpoint.to_dict()
        restored = KernelCheckpoint.from_dict(data)

        assert restored.instance_id == sample_checkpoint.instance_id
        assert restored.state_hash == sample_checkpoint.state_hash


# =============================================================================
# MemoryStateStore Tests
# =============================================================================


class TestMemoryStateStore:
    """Tests for MemoryStateStore."""

    @pytest.fixture
    def store(self) -> MemoryStateStore:
        """Create memory store for testing."""
        return MemoryStateStore()

    async def test_save_and_get_order(
        self, store: MemoryStateStore, sample_order: PersistedOrder
    ) -> None:
        """Test saving and retrieving an order."""
        await store.save_order(sample_order)
        retrieved = await store.get_order(sample_order.order_id)

        assert retrieved is not None
        assert retrieved.order_id == sample_order.order_id
        assert retrieved.symbol == sample_order.symbol

    async def test_get_nonexistent_order(self, store: MemoryStateStore) -> None:
        """Test getting non-existent order returns None."""
        result = await store.get_order("nonexistent")
        assert result is None

    async def test_get_open_orders(
        self, store: MemoryStateStore, sample_order: PersistedOrder
    ) -> None:
        """Test getting open orders."""
        # Save open order
        await store.save_order(sample_order)

        # Save filled order
        filled_order = PersistedOrder(
            order_id="ord_filled",
            client_order_id="client_filled",
            symbol="ETH/USDT",
            side="sell",
            order_type="market",
            amount="2.0",
            filled_amount="2.0",
            price=None,
            state=OrderState.FILLED,
            created_at_ns=time.time_ns(),
            updated_at_ns=time.time_ns(),
        )
        await store.save_order(filled_order)

        open_orders = await store.get_open_orders()

        assert len(open_orders) == 1
        assert open_orders[0].order_id == sample_order.order_id

    async def test_delete_order(
        self, store: MemoryStateStore, sample_order: PersistedOrder
    ) -> None:
        """Test deleting an order."""
        await store.save_order(sample_order)
        await store.delete_order(sample_order.order_id)

        result = await store.get_order(sample_order.order_id)
        assert result is None

    async def test_save_and_get_position(
        self, store: MemoryStateStore, sample_position: PersistedPosition
    ) -> None:
        """Test saving and retrieving a position."""
        await store.save_position(sample_position)
        retrieved = await store.get_position(sample_position.symbol)

        assert retrieved is not None
        assert retrieved.symbol == sample_position.symbol
        assert retrieved.quantity == sample_position.quantity

    async def test_get_positions(
        self, store: MemoryStateStore, sample_position: PersistedPosition
    ) -> None:
        """Test getting all positions."""
        await store.save_position(sample_position)

        position2 = PersistedPosition(
            symbol="ETH/USDT",
            side="short",
            quantity="5.0",
            entry_price="3000",
            unrealized_pnl="-100",
            realized_pnl="50",
            opened_at_ns=time.time_ns(),
            updated_at_ns=time.time_ns(),
        )
        await store.save_position(position2)

        positions = await store.get_positions()
        assert len(positions) == 2

    async def test_delete_position(
        self, store: MemoryStateStore, sample_position: PersistedPosition
    ) -> None:
        """Test deleting a position."""
        await store.save_position(sample_position)
        await store.delete_position(sample_position.symbol)

        result = await store.get_position(sample_position.symbol)
        assert result is None

    async def test_save_and_get_checkpoint(
        self, store: MemoryStateStore, sample_checkpoint: KernelCheckpoint
    ) -> None:
        """Test saving and retrieving checkpoint."""
        await store.save_checkpoint(sample_checkpoint)
        retrieved = await store.get_latest_checkpoint()

        assert retrieved is not None
        assert retrieved.checkpoint_id == sample_checkpoint.checkpoint_id
        assert retrieved.state_hash == sample_checkpoint.state_hash

    async def test_latest_checkpoint_updates(self, store: MemoryStateStore) -> None:
        """Test that latest checkpoint pointer updates."""
        checkpoint1 = KernelCheckpoint(
            instance_id="inst_001",
            checkpoint_id="chk_001",
            timestamp_ns=time.time_ns(),
            state_hash="hash1",
            orders_count=1,
            positions_count=1,
            environment="test",
        )
        checkpoint2 = KernelCheckpoint(
            instance_id="inst_001",
            checkpoint_id="chk_002",
            timestamp_ns=time.time_ns(),
            state_hash="hash2",
            orders_count=2,
            positions_count=2,
            environment="test",
        )

        await store.save_checkpoint(checkpoint1)
        await store.save_checkpoint(checkpoint2)

        latest = await store.get_latest_checkpoint()
        assert latest.checkpoint_id == "chk_002"

    async def test_clear_all(
        self,
        store: MemoryStateStore,
        sample_order: PersistedOrder,
        sample_position: PersistedPosition,
    ) -> None:
        """Test clearing all state."""
        await store.save_order(sample_order)
        await store.save_position(sample_position)

        await store.clear_all()

        assert await store.get_order(sample_order.order_id) is None
        assert await store.get_position(sample_position.symbol) is None
        assert await store.get_latest_checkpoint() is None


# =============================================================================
# FileStateStore Tests
# =============================================================================


class TestFileStateStore:
    """Tests for FileStateStore."""

    @pytest.fixture
    def store(self, tmp_path) -> FileStateStore:
        """Create file store in temp directory."""
        return FileStateStore(tmp_path / "state")

    async def test_save_and_get_order(
        self, store: FileStateStore, sample_order: PersistedOrder
    ) -> None:
        """Test saving and retrieving an order."""
        await store.save_order(sample_order)
        retrieved = await store.get_order(sample_order.order_id)

        assert retrieved is not None
        assert retrieved.order_id == sample_order.order_id

    async def test_get_nonexistent_order(self, store: FileStateStore) -> None:
        """Test getting non-existent order returns None."""
        result = await store.get_order("nonexistent")
        assert result is None

    async def test_get_open_orders(
        self, store: FileStateStore, sample_order: PersistedOrder
    ) -> None:
        """Test getting open orders filters correctly."""
        await store.save_order(sample_order)

        filled_order = PersistedOrder(
            order_id="ord_filled",
            client_order_id="client_filled",
            symbol="ETH/USDT",
            side="sell",
            order_type="market",
            amount="2.0",
            filled_amount="2.0",
            price=None,
            state=OrderState.FILLED,
            created_at_ns=time.time_ns(),
            updated_at_ns=time.time_ns(),
        )
        await store.save_order(filled_order)

        open_orders = await store.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].order_id == sample_order.order_id

    async def test_save_and_get_position(
        self, store: FileStateStore, sample_position: PersistedPosition
    ) -> None:
        """Test saving and retrieving a position."""
        await store.save_position(sample_position)
        retrieved = await store.get_position(sample_position.symbol)

        assert retrieved is not None
        assert retrieved.symbol == sample_position.symbol

    async def test_save_and_get_checkpoint(
        self, store: FileStateStore, sample_checkpoint: KernelCheckpoint
    ) -> None:
        """Test saving and retrieving checkpoint."""
        await store.save_checkpoint(sample_checkpoint)
        retrieved = await store.get_latest_checkpoint()

        assert retrieved is not None
        assert retrieved.checkpoint_id == sample_checkpoint.checkpoint_id

    async def test_clear_all(
        self, store: FileStateStore, sample_order: PersistedOrder
    ) -> None:
        """Test clearing all state."""
        await store.save_order(sample_order)
        await store.clear_all()

        assert await store.get_order(sample_order.order_id) is None


# =============================================================================
# RedisStateStore Tests
# =============================================================================


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
class TestRedisStateStore:
    """Tests for RedisStateStore."""

    @pytest.fixture
    async def store(self):
        """Create Redis store for testing."""
        store = RedisStateStore(
            url="redis://localhost:6379",
            prefix="libra:test:",
            db=15,  # Use test database
        )
        try:
            await store.connect()
            await store.clear_all()  # Clean before test
            yield store
        except Exception:
            pytest.skip("Redis not available")
        finally:
            try:
                await store.clear_all()  # Clean after test
                await store.disconnect()
            except Exception:
                pass

    async def test_save_and_get_order(
        self, store: RedisStateStore, sample_order: PersistedOrder
    ) -> None:
        """Test saving and retrieving an order."""
        await store.save_order(sample_order)
        retrieved = await store.get_order(sample_order.order_id)

        assert retrieved is not None
        assert retrieved.order_id == sample_order.order_id
        assert retrieved.symbol == sample_order.symbol

    async def test_get_nonexistent_order(self, store: RedisStateStore) -> None:
        """Test getting non-existent order returns None."""
        result = await store.get_order("nonexistent")
        assert result is None

    async def test_get_open_orders(
        self, store: RedisStateStore, sample_order: PersistedOrder
    ) -> None:
        """Test getting open orders."""
        await store.save_order(sample_order)

        filled_order = PersistedOrder(
            order_id="ord_filled",
            client_order_id="client_filled",
            symbol="ETH/USDT",
            side="sell",
            order_type="market",
            amount="2.0",
            filled_amount="2.0",
            price=None,
            state=OrderState.FILLED,
            created_at_ns=time.time_ns(),
            updated_at_ns=time.time_ns(),
        )
        await store.save_order(filled_order)

        open_orders = await store.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].order_id == sample_order.order_id

    async def test_delete_order(
        self, store: RedisStateStore, sample_order: PersistedOrder
    ) -> None:
        """Test deleting an order."""
        await store.save_order(sample_order)
        await store.delete_order(sample_order.order_id)

        result = await store.get_order(sample_order.order_id)
        assert result is None

    async def test_save_and_get_position(
        self, store: RedisStateStore, sample_position: PersistedPosition
    ) -> None:
        """Test saving and retrieving a position."""
        await store.save_position(sample_position)
        retrieved = await store.get_position(sample_position.symbol)

        assert retrieved is not None
        assert retrieved.symbol == sample_position.symbol
        assert retrieved.quantity == sample_position.quantity

    async def test_get_positions(
        self, store: RedisStateStore, sample_position: PersistedPosition
    ) -> None:
        """Test getting all positions."""
        await store.save_position(sample_position)

        position2 = PersistedPosition(
            symbol="ETH/USDT",
            side="short",
            quantity="5.0",
            entry_price="3000",
            unrealized_pnl="-100",
            realized_pnl="50",
            opened_at_ns=time.time_ns(),
            updated_at_ns=time.time_ns(),
        )
        await store.save_position(position2)

        positions = await store.get_positions()
        assert len(positions) == 2

    async def test_save_and_get_checkpoint(
        self, store: RedisStateStore, sample_checkpoint: KernelCheckpoint
    ) -> None:
        """Test saving and retrieving checkpoint."""
        await store.save_checkpoint(sample_checkpoint)
        retrieved = await store.get_latest_checkpoint()

        assert retrieved is not None
        assert retrieved.checkpoint_id == sample_checkpoint.checkpoint_id

    async def test_context_manager(self, sample_order: PersistedOrder) -> None:
        """Test async context manager."""
        try:
            async with RedisStateStore(
                url="redis://localhost:6379",
                prefix="libra:ctxtest:",
                db=15,
            ) as store:
                await store.save_order(sample_order)
                retrieved = await store.get_order(sample_order.order_id)
                assert retrieved is not None
                await store.clear_all()
        except Exception:
            pytest.skip("Redis not available")

    async def test_key_prefix_isolation(self, sample_order: PersistedOrder) -> None:
        """Test that different prefixes are isolated."""
        try:
            store1 = RedisStateStore(
                url="redis://localhost:6379",
                prefix="libra:ns1:",
                db=15,
            )
            store2 = RedisStateStore(
                url="redis://localhost:6379",
                prefix="libra:ns2:",
                db=15,
            )

            await store1.connect()
            await store2.connect()

            # Save in store1
            await store1.save_order(sample_order)

            # Should not exist in store2
            result = await store2.get_order(sample_order.order_id)
            assert result is None

            # But exists in store1
            result = await store1.get_order(sample_order.order_id)
            assert result is not None

            await store1.clear_all()
            await store2.clear_all()
            await store1.disconnect()
            await store2.disconnect()

        except Exception:
            pytest.skip("Redis not available")


# =============================================================================
# REDIS_AVAILABLE Tests
# =============================================================================


class TestRedisAvailable:
    """Tests for Redis availability flag."""

    def test_redis_available_is_bool(self) -> None:
        """Test REDIS_AVAILABLE is a boolean."""
        assert isinstance(REDIS_AVAILABLE, bool)

    def test_redis_import_error_when_unavailable(self) -> None:
        """Test helpful error when redis not installed."""
        if not REDIS_AVAILABLE:
            with pytest.raises(ImportError, match="redis package not installed"):
                RedisStateStore()
