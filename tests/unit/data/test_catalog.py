"""Tests for Data Catalog (Issue #23).

Tests:
- DataType and DataResolution enums
- DataRequirement declarations
- DataCatalog source registration and selection
- Provider protocol compliance
"""

from datetime import timedelta
from unittest.mock import AsyncMock

import polars as pl
import pytest

from libra.data.catalog import (
    BaseDataProvider,
    DataAvailability,
    DataCatalog,
    DataProvider,
    DataRequirement,
    DataResolution,
    DataType,
    get_catalog,
    set_catalog,
)


class TestDataType:
    """Tests for DataType enum."""

    def test_data_type_values(self) -> None:
        """Test all DataType values exist."""
        assert DataType.OHLCV.value == "ohlcv"
        assert DataType.TICK.value == "tick"
        assert DataType.ORDERBOOK.value == "orderbook"
        assert DataType.TRADES.value == "trades"
        assert DataType.SENTIMENT.value == "sentiment"
        assert DataType.ONCHAIN.value == "onchain"
        assert DataType.NEWS.value == "news"
        assert DataType.FUNDING.value == "funding"
        assert DataType.INDICATORS.value == "indicators"
        assert DataType.SIGNALS.value == "signals"

    def test_data_type_string_enum(self) -> None:
        """Test DataType is string enum."""
        assert str(DataType.OHLCV) == "DataType.OHLCV"
        assert DataType.OHLCV == "ohlcv"


class TestDataResolution:
    """Tests for DataResolution enum."""

    def test_resolution_values(self) -> None:
        """Test all resolution values."""
        assert DataResolution.TICK.value == "tick"
        assert DataResolution.SECOND_1.value == "1s"
        assert DataResolution.MINUTE_1.value == "1m"
        assert DataResolution.MINUTE_5.value == "5m"
        assert DataResolution.HOUR_1.value == "1h"
        assert DataResolution.DAY_1.value == "1d"


class TestDataRequirement:
    """Tests for DataRequirement."""

    def test_basic_requirement(self) -> None:
        """Test creating basic requirement."""
        req = DataRequirement("BTC/USDT", DataType.OHLCV)

        assert req.symbol == "BTC/USDT"
        assert req.data_type == DataType.OHLCV
        assert req.resolution == "1m"
        assert req.source == "auto"
        assert req.lookback is None
        assert req.required is True

    def test_requirement_with_options(self) -> None:
        """Test requirement with all options."""
        req = DataRequirement(
            symbol="ETH/USDT",
            data_type=DataType.ORDERBOOK,
            resolution="tick",
            source="binance",
            lookback=timedelta(days=7),
            required=False,
        )

        assert req.symbol == "ETH/USDT"
        assert req.data_type == DataType.ORDERBOOK
        assert req.resolution == "tick"
        assert req.source == "binance"
        assert req.lookback == timedelta(days=7)
        assert req.required is False

    def test_requirement_key(self) -> None:
        """Test requirement key generation."""
        req = DataRequirement("BTC/USDT", DataType.OHLCV, "5m")
        assert req.key == "BTC/USDT:ohlcv:5m"

    def test_requirement_hashable(self) -> None:
        """Test requirements can be used in sets."""
        req1 = DataRequirement("BTC/USDT", DataType.OHLCV)
        req2 = DataRequirement("BTC/USDT", DataType.OHLCV)
        req3 = DataRequirement("ETH/USDT", DataType.OHLCV)

        # Same requirements should have same hash
        assert hash(req1) == hash(req2)
        # Can use in set
        reqs = {req1, req2, req3}
        assert len(reqs) == 2

    def test_requirement_frozen(self) -> None:
        """Test requirement is immutable."""
        req = DataRequirement("BTC/USDT", DataType.OHLCV)
        with pytest.raises(AttributeError):
            req.symbol = "ETH/USDT"  # type: ignore


class MockDataProvider(BaseDataProvider):
    """Mock data provider for testing."""

    def __init__(
        self,
        name: str = "mock",
        supported_types: frozenset[DataType] | None = None,
    ) -> None:
        super().__init__(
            name,
            supported_types or frozenset({DataType.OHLCV, DataType.TICK}),
        )
        self._fetch_called = False
        self._last_args: dict | None = None

    async def fetch(
        self,
        symbol: str,
        data_type: DataType,
        resolution: str = "1m",
        start=None,
        end=None,
        limit=None,
    ) -> pl.LazyFrame:
        self._fetch_called = True
        self._last_args = {
            "symbol": symbol,
            "data_type": data_type,
            "resolution": resolution,
            "start": start,
            "end": end,
            "limit": limit,
        }
        # Return empty LazyFrame
        return pl.DataFrame(
            {"timestamp": [], "open": [], "close": []}
        ).lazy()


class TestBaseDataProvider:
    """Tests for BaseDataProvider."""

    def test_provider_name(self) -> None:
        """Test provider name property."""
        provider = MockDataProvider("test_provider")
        assert provider.name == "test_provider"

    def test_provider_supported_types(self) -> None:
        """Test provider supported types."""
        provider = MockDataProvider(
            supported_types=frozenset({DataType.OHLCV, DataType.TRADES})
        )
        assert DataType.OHLCV in provider.supported_types
        assert DataType.TRADES in provider.supported_types
        assert DataType.ORDERBOOK not in provider.supported_types

    def test_provider_supports(self) -> None:
        """Test provider supports check."""
        provider = MockDataProvider()
        assert provider.supports("BTC/USDT", DataType.OHLCV) is True
        assert provider.supports("ETH/USDT", DataType.TICK) is True
        assert provider.supports("BTC/USDT", DataType.ORDERBOOK) is False

    @pytest.mark.asyncio
    async def test_provider_fetch(self) -> None:
        """Test provider fetch."""
        provider = MockDataProvider()
        result = await provider.fetch("BTC/USDT", DataType.OHLCV, "5m")

        assert provider._fetch_called
        assert provider._last_args["symbol"] == "BTC/USDT"
        assert provider._last_args["data_type"] == DataType.OHLCV
        assert provider._last_args["resolution"] == "5m"


class TestDataCatalog:
    """Tests for DataCatalog."""

    def test_empty_catalog(self) -> None:
        """Test empty catalog."""
        catalog = DataCatalog()
        assert catalog.sources == []

    def test_register_source(self) -> None:
        """Test registering a data source."""
        catalog = DataCatalog()
        provider = MockDataProvider("binance")

        catalog.register_source(provider, priority=5)

        assert "binance" in catalog.sources
        assert len(catalog.sources) == 1

    def test_register_multiple_sources(self) -> None:
        """Test registering multiple sources."""
        catalog = DataCatalog()

        catalog.register_source(MockDataProvider("binance"), priority=1)
        catalog.register_source(MockDataProvider("kraken"), priority=2)

        assert "binance" in catalog.sources
        assert "kraken" in catalog.sources
        assert len(catalog.sources) == 2

    def test_unregister_source(self) -> None:
        """Test unregistering a source."""
        catalog = DataCatalog()
        catalog.register_source(MockDataProvider("binance"))

        result = catalog.unregister_source("binance")
        assert result is True
        assert "binance" not in catalog.sources

        # Unregister non-existent
        result = catalog.unregister_source("nonexistent")
        assert result is False

    def test_list_available(self) -> None:
        """Test listing available data."""
        catalog = DataCatalog()
        catalog.register_source(MockDataProvider("binance"))

        available = catalog.list_available("BTC/USDT")

        # Should have OHLCV and TICK (mock provider supports these)
        data_types = [a.data_type for a in available]
        assert DataType.OHLCV in data_types
        assert DataType.TICK in data_types

    def test_list_available_with_filter(self) -> None:
        """Test listing available with data type filter."""
        catalog = DataCatalog()
        catalog.register_source(MockDataProvider("binance"))

        available = catalog.list_available("BTC/USDT", data_type=DataType.OHLCV)

        assert len(available) == 1
        assert available[0].data_type == DataType.OHLCV

    @pytest.mark.asyncio
    async def test_fetch(self) -> None:
        """Test fetching data."""
        catalog = DataCatalog()
        provider = MockDataProvider("binance")
        catalog.register_source(provider)

        req = DataRequirement("BTC/USDT", DataType.OHLCV, "5m")
        result = await catalog.fetch(req)

        assert isinstance(result, pl.LazyFrame)
        assert provider._fetch_called

    @pytest.mark.asyncio
    async def test_fetch_with_preferred_source(self) -> None:
        """Test fetching from preferred source."""
        catalog = DataCatalog()
        binance = MockDataProvider("binance")
        kraken = MockDataProvider("kraken")

        catalog.register_source(binance, priority=10)
        catalog.register_source(kraken, priority=5)

        # Without preferred source, should use kraken (lower priority)
        req = DataRequirement("BTC/USDT", DataType.OHLCV)
        await catalog.fetch(req)
        assert kraken._fetch_called

        # Reset
        kraken._fetch_called = False

        # With preferred source
        req = DataRequirement("BTC/USDT", DataType.OHLCV, source="binance")
        await catalog.fetch(req)
        assert binance._fetch_called
        assert not kraken._fetch_called

    @pytest.mark.asyncio
    async def test_fetch_no_source_raises(self) -> None:
        """Test fetch raises when no source available."""
        catalog = DataCatalog()

        req = DataRequirement("BTC/USDT", DataType.OHLCV)

        with pytest.raises(ValueError, match="No data source available"):
            await catalog.fetch(req)

    @pytest.mark.asyncio
    async def test_fetch_many(self) -> None:
        """Test fetching multiple requirements."""
        catalog = DataCatalog()
        catalog.register_source(MockDataProvider("binance"))

        reqs = [
            DataRequirement("BTC/USDT", DataType.OHLCV),
            DataRequirement("ETH/USDT", DataType.TICK),
        ]

        results = await catalog.fetch_many(reqs)

        assert len(results) == 2
        assert "BTC/USDT:ohlcv:1m" in results
        assert "ETH/USDT:tick:1m" in results

    def test_validate_requirements(self) -> None:
        """Test validating requirements."""
        catalog = DataCatalog()
        catalog.register_source(MockDataProvider("binance"))

        reqs = [
            DataRequirement("BTC/USDT", DataType.OHLCV),
            DataRequirement("BTC/USDT", DataType.ORDERBOOK),  # Not supported
        ]

        missing = catalog.validate_requirements(reqs)

        assert len(missing) == 1
        assert missing[0].data_type == DataType.ORDERBOOK

    def test_get_stats(self) -> None:
        """Test getting catalog stats."""
        catalog = DataCatalog()
        catalog.register_source(MockDataProvider("binance"), priority=5)

        stats = catalog.get_stats()

        assert stats["source_count"] == 1
        assert len(stats["sources"]) == 1
        assert stats["sources"][0]["name"] == "binance"
        assert stats["sources"][0]["priority"] == 5


class TestGlobalCatalog:
    """Tests for global catalog functions."""

    def test_get_set_catalog(self) -> None:
        """Test getting and setting global catalog."""
        original = get_catalog()
        assert isinstance(original, DataCatalog)

        new_catalog = DataCatalog()
        set_catalog(new_catalog)

        assert get_catalog() is new_catalog

        # Restore
        set_catalog(original)


class TestDataAvailability:
    """Tests for DataAvailability."""

    def test_availability_creation(self) -> None:
        """Test creating availability info."""
        avail = DataAvailability(
            symbol="BTC/USDT",
            data_type=DataType.OHLCV,
            source="binance",
            resolution="1m",
        )

        assert avail.symbol == "BTC/USDT"
        assert avail.data_type == DataType.OHLCV
        assert avail.source == "binance"

    def test_availability_key(self) -> None:
        """Test availability key."""
        avail = DataAvailability(
            symbol="BTC/USDT",
            data_type=DataType.OHLCV,
            source="binance",
            resolution="5m",
        )

        assert avail.key == "BTC/USDT:ohlcv:5m:binance"
