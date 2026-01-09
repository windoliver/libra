"""Tests for Tiered Storage (Issue #23).

Tests:
- InMemoryBackend operations
- ParquetBackend operations
- TieredStorage tier selection and migration
- Storage statistics
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from libra.data.storage import (
    InMemoryBackend,
    ParquetBackend,
    StorageStats,
    StorageTier,
    TierConfig,
    TieredStorage,
    create_default_storage,
)


class TestStorageTier:
    """Tests for StorageTier enum."""

    def test_tier_values(self) -> None:
        """Test all tier values."""
        assert StorageTier.HOT.value == "hot"
        assert StorageTier.WARM.value == "warm"
        assert StorageTier.COLD.value == "cold"


class TestTierConfig:
    """Tests for TierConfig."""

    def test_default_hot(self) -> None:
        """Test default hot tier config."""
        config = TierConfig.default_hot()

        assert config.tier == StorageTier.HOT
        assert config.max_age == timedelta(hours=1)
        assert config.max_size_mb == 1024
        assert config.compression is False

    def test_default_warm(self) -> None:
        """Test default warm tier config."""
        config = TierConfig.default_warm()

        assert config.tier == StorageTier.WARM
        assert config.max_age == timedelta(days=30)
        assert config.compression is True
        assert config.path == Path("./data/warm")

    def test_default_cold(self) -> None:
        """Test default cold tier config."""
        config = TierConfig.default_cold()

        assert config.tier == StorageTier.COLD
        assert config.max_age == timedelta(days=365)
        assert config.compression is True

    def test_custom_path(self) -> None:
        """Test config with custom path."""
        path = Path("/custom/path")
        config = TierConfig.default_warm(path=path)

        assert config.path == path


class TestInMemoryBackend:
    """Tests for InMemoryBackend."""

    @pytest.fixture
    def backend(self) -> InMemoryBackend:
        """Create test backend."""
        return InMemoryBackend(max_size_mb=100)

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame."""
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ],
                "value": [1.0, 2.0, 3.0],
            }
        )

    def test_tier_property(self, backend: InMemoryBackend) -> None:
        """Test tier is HOT."""
        assert backend.tier == StorageTier.HOT

    @pytest.mark.asyncio
    async def test_write_and_read(
        self, backend: InMemoryBackend, sample_df: pl.DataFrame
    ) -> None:
        """Test write and read."""
        await backend.write("test_key", sample_df)
        result = await backend.read("test_key")

        assert result is not None
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, backend: InMemoryBackend) -> None:
        """Test reading nonexistent key."""
        result = await backend.read("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_read_with_time_filter(
        self, backend: InMemoryBackend, sample_df: pl.DataFrame
    ) -> None:
        """Test reading with time filter."""
        await backend.write("test_key", sample_df)

        result = await backend.read(
            "test_key",
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 2),
        )

        assert result is not None
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_delete(
        self, backend: InMemoryBackend, sample_df: pl.DataFrame
    ) -> None:
        """Test delete operation."""
        await backend.write("test_key", sample_df)

        result = await backend.delete("test_key")
        assert result is True

        result = await backend.read("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, backend: InMemoryBackend) -> None:
        """Test deleting nonexistent key."""
        result = await backend.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists(
        self, backend: InMemoryBackend, sample_df: pl.DataFrame
    ) -> None:
        """Test exists check."""
        await backend.write("test_key", sample_df)

        assert await backend.exists("test_key") is True
        assert await backend.exists("other_key") is False

    @pytest.mark.asyncio
    async def test_list_keys(
        self, backend: InMemoryBackend, sample_df: pl.DataFrame
    ) -> None:
        """Test listing keys."""
        await backend.write("btc_data", sample_df)
        await backend.write("eth_data", sample_df)
        await backend.write("other", sample_df)

        # All keys
        keys = await backend.list_keys()
        assert len(keys) == 3

        # With prefix
        keys = await backend.list_keys(prefix="btc")
        assert len(keys) == 1
        assert "btc_data" in keys

    @pytest.mark.asyncio
    async def test_get_metadata(
        self, backend: InMemoryBackend, sample_df: pl.DataFrame
    ) -> None:
        """Test getting metadata."""
        await backend.write("test_key", sample_df)

        meta = await backend.get_metadata("test_key")

        assert meta is not None
        assert "updated_at" in meta
        assert meta["row_count"] == 3
        assert "value" in meta["columns"]

    def test_estimate_size(
        self, backend: InMemoryBackend, sample_df: pl.DataFrame
    ) -> None:
        """Test size estimation."""
        # Empty backend
        assert backend.estimate_size_mb() == 0

        # Can't easily test with async write, but method exists


class TestParquetBackend:
    """Tests for ParquetBackend."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def backend(self, temp_dir: Path) -> ParquetBackend:
        """Create test backend."""
        return ParquetBackend(temp_dir, StorageTier.WARM)

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame."""
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ],
                "value": [1.0, 2.0, 3.0],
            }
        )

    def test_tier_property(self, backend: ParquetBackend) -> None:
        """Test tier property."""
        assert backend.tier == StorageTier.WARM

    @pytest.mark.asyncio
    async def test_write_and_read(
        self, backend: ParquetBackend, sample_df: pl.DataFrame
    ) -> None:
        """Test write and read."""
        await backend.write("test_key", sample_df)
        result = await backend.read("test_key")

        assert result is not None
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, backend: ParquetBackend) -> None:
        """Test reading nonexistent key."""
        result = await backend.read("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(
        self, backend: ParquetBackend, sample_df: pl.DataFrame
    ) -> None:
        """Test delete operation."""
        await backend.write("test_key", sample_df)

        result = await backend.delete("test_key")
        assert result is True

        result = await backend.read("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_exists(
        self, backend: ParquetBackend, sample_df: pl.DataFrame
    ) -> None:
        """Test exists check."""
        await backend.write("test_key", sample_df)

        assert await backend.exists("test_key") is True
        assert await backend.exists("other") is False

    @pytest.mark.asyncio
    async def test_get_metadata(
        self, backend: ParquetBackend, sample_df: pl.DataFrame
    ) -> None:
        """Test getting metadata."""
        await backend.write("test_key", sample_df)

        meta = await backend.get_metadata("test_key")

        assert meta is not None
        assert "size_bytes" in meta
        assert "modified_at" in meta
        assert "path" in meta


class TestTieredStorage:
    """Tests for TieredStorage."""

    @pytest.fixture
    def storage(self) -> TieredStorage:
        """Create tiered storage with in-memory backends."""
        return TieredStorage(
            hot=InMemoryBackend(),
            warm=InMemoryBackend(),
            cold=InMemoryBackend(),
        )

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame."""
        return pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "value": [1.0],
            }
        )

    @pytest.mark.asyncio
    async def test_write_and_read_hot(
        self, storage: TieredStorage, sample_df: pl.DataFrame
    ) -> None:
        """Test write to hot tier and read."""
        await storage.write("test_key", sample_df, tier=StorageTier.HOT)

        result = await storage.read("test_key")
        assert result is not None

    @pytest.mark.asyncio
    async def test_read_auto_tier_selection(
        self, storage: TieredStorage, sample_df: pl.DataFrame
    ) -> None:
        """Test read checks all tiers."""
        # Write to warm tier
        await storage.write("warm_key", sample_df, tier=StorageTier.WARM)

        # Read without specifying tier
        result = await storage.read("warm_key")
        assert result is not None

    @pytest.mark.asyncio
    async def test_read_specific_tier(
        self, storage: TieredStorage, sample_df: pl.DataFrame
    ) -> None:
        """Test reading from specific tier."""
        await storage.write("test_key", sample_df, tier=StorageTier.WARM)

        # Won't find in hot tier
        result = await storage.read("test_key", tier=StorageTier.HOT)
        assert result is None

        # Will find in warm tier
        result = await storage.read("test_key", tier=StorageTier.WARM)
        assert result is not None

    @pytest.mark.asyncio
    async def test_promote(
        self, storage: TieredStorage, sample_df: pl.DataFrame
    ) -> None:
        """Test promoting data to faster tier."""
        await storage.write("test_key", sample_df, tier=StorageTier.WARM)

        result = await storage.promote("test_key", StorageTier.WARM, StorageTier.HOT)
        assert result is True

        # Should now be in hot tier
        hot_result = await storage.read("test_key", tier=StorageTier.HOT)
        assert hot_result is not None

    @pytest.mark.asyncio
    async def test_demote(
        self, storage: TieredStorage, sample_df: pl.DataFrame
    ) -> None:
        """Test demoting data to slower tier."""
        await storage.write("test_key", sample_df, tier=StorageTier.HOT)

        result = await storage.demote(
            "test_key", StorageTier.HOT, StorageTier.WARM, delete_source=True
        )
        assert result is True

        # Should no longer be in hot tier
        hot_result = await storage.read("test_key", tier=StorageTier.HOT)
        assert hot_result is None

        # Should be in warm tier
        warm_result = await storage.read("test_key", tier=StorageTier.WARM)
        assert warm_result is not None

    @pytest.mark.asyncio
    async def test_delete_specific_tier(
        self, storage: TieredStorage, sample_df: pl.DataFrame
    ) -> None:
        """Test deleting from specific tier."""
        await storage.write("test_key", sample_df, tier=StorageTier.HOT)

        result = await storage.delete("test_key", tier=StorageTier.HOT)
        assert result is True

        result = await storage.read("test_key", tier=StorageTier.HOT)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_all_tiers(
        self, storage: TieredStorage, sample_df: pl.DataFrame
    ) -> None:
        """Test deleting from all tiers."""
        await storage.write("test_key", sample_df, tier=StorageTier.HOT)
        await storage.write("test_key", sample_df, tier=StorageTier.WARM)

        result = await storage.delete("test_key")
        assert result is True

        assert await storage.read("test_key", tier=StorageTier.HOT) is None
        assert await storage.read("test_key", tier=StorageTier.WARM) is None

    @pytest.mark.asyncio
    async def test_list_keys(
        self, storage: TieredStorage, sample_df: pl.DataFrame
    ) -> None:
        """Test listing keys across tiers."""
        await storage.write("hot_key", sample_df, tier=StorageTier.HOT)
        await storage.write("warm_key", sample_df, tier=StorageTier.WARM)

        keys = await storage.list_keys()

        assert StorageTier.HOT in keys
        assert StorageTier.WARM in keys
        assert "hot_key" in keys[StorageTier.HOT]
        assert "warm_key" in keys[StorageTier.WARM]

    def test_stats(self, storage: TieredStorage) -> None:
        """Test getting stats."""
        stats = storage.stats

        assert isinstance(stats, StorageStats)
        assert stats.hot_reads == 0
        assert stats.writes == 0

    @pytest.mark.asyncio
    async def test_stats_tracking(
        self, storage: TieredStorage, sample_df: pl.DataFrame
    ) -> None:
        """Test stats are tracked."""
        await storage.write("key", sample_df, tier=StorageTier.HOT)
        await storage.read("key")

        stats = storage.stats
        assert stats.writes >= 1
        assert stats.hot_reads >= 1


class TestCreateDefaultStorage:
    """Tests for create_default_storage factory."""

    def test_creates_tiered_storage(self) -> None:
        """Test factory creates TieredStorage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = create_default_storage(base_path=tmpdir, hot_size_mb=256)

            assert isinstance(storage, TieredStorage)

    def test_creates_all_tiers(self) -> None:
        """Test factory creates all tiers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = create_default_storage(base_path=tmpdir)

            # All tiers should be configured
            assert storage._hot is not None
            assert storage._warm is not None
            assert storage._cold is not None
