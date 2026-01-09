"""
Data Catalog: Registry and discovery for market data sources.

Provides:
- DataType enum for different data categories
- DataRequirement for strategy data declarations
- DataCatalog for source registration and data fetching
- Automatic data source selection

Inspired by:
- QuantConnect's Universe Selection
- Zipline's data bundles
- OpenBB's data providers

Usage:
    # Strategy declares requirements
    class MyStrategy(Strategy):
        data_requirements = [
            DataRequirement("BTC/USDT", DataType.OHLCV, "1m"),
            DataRequirement("BTC/USDT", DataType.ORDERBOOK),
        ]

    # Catalog fetches data
    catalog = DataCatalog()
    catalog.register_source("binance", binance_provider)
    data = await catalog.fetch(requirements[0])

See: https://github.com/windoliver/libra/issues/23
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import polars as pl


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


# =============================================================================
# Data Type Definitions
# =============================================================================


class DataType(str, Enum):
    """
    Types of market data available in the catalog.

    Categories:
    - Price data: OHLCV, TICK, ORDERBOOK
    - Alternative data: SENTIMENT, ONCHAIN, NEWS
    - Derived data: INDICATORS, SIGNALS
    """

    # Price data
    OHLCV = "ohlcv"  # Open, High, Low, Close, Volume bars
    TICK = "tick"  # Real-time bid/ask/last
    ORDERBOOK = "orderbook"  # Order book snapshots/deltas
    TRADES = "trades"  # Individual trade records

    # Alternative data
    SENTIMENT = "sentiment"  # Social sentiment scores
    ONCHAIN = "onchain"  # Blockchain metrics
    NEWS = "news"  # News articles/events
    FUNDING = "funding"  # Funding rates (futures)

    # Derived data
    INDICATORS = "indicators"  # Technical indicators
    SIGNALS = "signals"  # Trading signals


class DataResolution(str, Enum):
    """
    Time resolution for data.

    From tick-level to monthly aggregations.
    """

    TICK = "tick"
    SECOND_1 = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


# =============================================================================
# Data Requirement
# =============================================================================


@dataclass(frozen=True)
class DataRequirement:
    """
    Declares a strategy's data requirement.

    Strategies declare their data needs, and the catalog
    automatically fetches from the best available source.

    Attributes:
        symbol: Trading pair (e.g., "BTC/USDT")
        data_type: Type of data needed
        resolution: Time resolution (default: "1m")
        source: Preferred source or "auto" for automatic selection
        lookback: Historical data needed (e.g., timedelta(days=30))
        required: If True, strategy fails without this data

    Examples:
        # Basic OHLCV requirement
        DataRequirement("BTC/USDT", DataType.OHLCV, "1m")

        # Order book from specific source
        DataRequirement("ETH/USDT", DataType.ORDERBOOK, source="binance")

        # Sentiment data with lookback
        DataRequirement("BTC", DataType.SENTIMENT, lookback=timedelta(days=7))
    """

    symbol: str
    data_type: DataType
    resolution: str = "1m"
    source: str = "auto"  # "auto" selects best available
    lookback: timedelta | None = None
    required: bool = True

    @property
    def key(self) -> str:
        """Unique key for this requirement."""
        return f"{self.symbol}:{self.data_type.value}:{self.resolution}"

    def __hash__(self) -> int:
        return hash(self.key)


# =============================================================================
# Data Provider Protocol
# =============================================================================


@runtime_checkable
class DataProvider(Protocol):
    """
    Protocol for data providers (exchanges, APIs, databases).

    Providers must implement:
    - supports(): Check if data type/symbol is available
    - fetch(): Retrieve data as Polars LazyFrame
    - stream(): Real-time data streaming (optional)
    """

    @property
    def name(self) -> str:
        """Provider identifier."""
        ...

    @property
    def supported_types(self) -> frozenset[DataType]:
        """Data types this provider supports."""
        ...

    def supports(self, symbol: str, data_type: DataType) -> bool:
        """Check if this provider supports the given symbol and data type."""
        ...

    async def fetch(
        self,
        symbol: str,
        data_type: DataType,
        resolution: str = "1m",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pl.LazyFrame:
        """
        Fetch historical data.

        Returns:
            Polars LazyFrame for query optimization
        """
        ...


# =============================================================================
# Base Data Provider
# =============================================================================


class BaseDataProvider(ABC):
    """
    Base class for data providers.

    Subclass this to implement custom data sources.
    """

    def __init__(self, name: str, supported_types: frozenset[DataType]) -> None:
        self._name = name
        self._supported_types = supported_types

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_types(self) -> frozenset[DataType]:
        return self._supported_types

    def supports(self, symbol: str, data_type: DataType) -> bool:
        """Default implementation checks supported types."""
        return data_type in self._supported_types

    @abstractmethod
    async def fetch(
        self,
        symbol: str,
        data_type: DataType,
        resolution: str = "1m",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pl.LazyFrame:
        """Implement in subclass."""
        ...


# =============================================================================
# Data Availability
# =============================================================================


@dataclass
class DataAvailability:
    """
    Information about available data for a symbol.

    Attributes:
        symbol: Trading pair
        data_type: Type of data
        source: Provider name
        resolution: Available resolution
        start: Earliest available data
        end: Latest available data
        row_count: Approximate row count
    """

    symbol: str
    data_type: DataType
    source: str
    resolution: str
    start: datetime | None = None
    end: datetime | None = None
    row_count: int | None = None

    @property
    def key(self) -> str:
        return f"{self.symbol}:{self.data_type.value}:{self.resolution}:{self.source}"


# =============================================================================
# Data Catalog
# =============================================================================


class DataCatalog:
    """
    Central registry for data sources and discovery.

    Features:
    - Register multiple data providers
    - Automatic source selection based on priority
    - Data availability querying
    - Requirement validation

    Example:
        catalog = DataCatalog()

        # Register providers
        catalog.register_source(binance_provider, priority=1)
        catalog.register_source(questdb_provider, priority=0)  # Higher priority

        # Query availability
        available = catalog.list_available("BTC/USDT")
        print(available)  # [OHLCV, TICK, ORDERBOOK, ...]

        # Fetch data
        req = DataRequirement("BTC/USDT", DataType.OHLCV, "1m")
        df = await catalog.fetch(req)

        # Validate strategy requirements
        missing = catalog.validate_requirements(strategy.data_requirements)
    """

    def __init__(self) -> None:
        # source_name -> (provider, priority)
        self._sources: dict[str, tuple[DataProvider, int]] = {}
        # Cache of availability info
        self._availability_cache: dict[str, DataAvailability] = {}

    def register_source(
        self,
        provider: DataProvider,
        priority: int = 10,
    ) -> None:
        """
        Register a data provider.

        Args:
            provider: Data provider instance
            priority: Lower = higher priority (default 10)
        """
        self._sources[provider.name] = (provider, priority)
        logger.info(f"Registered data source: {provider.name} (priority={priority})")

    def unregister_source(self, name: str) -> bool:
        """
        Unregister a data provider.

        Args:
            name: Provider name

        Returns:
            True if removed, False if not found
        """
        if name in self._sources:
            del self._sources[name]
            # Clear related availability cache
            self._availability_cache = {
                k: v for k, v in self._availability_cache.items() if v.source != name
            }
            logger.info(f"Unregistered data source: {name}")
            return True
        return False

    @property
    def sources(self) -> list[str]:
        """List registered source names."""
        return list(self._sources.keys())

    def list_available(
        self,
        symbol: str,
        data_type: DataType | None = None,
    ) -> list[DataAvailability]:
        """
        List available data for a symbol.

        Args:
            symbol: Trading pair
            data_type: Optional filter by data type

        Returns:
            List of DataAvailability info
        """
        available = []

        for name, (provider, _) in self._sources.items():
            for dtype in provider.supported_types:
                if data_type is not None and dtype != data_type:
                    continue

                if provider.supports(symbol, dtype):
                    available.append(
                        DataAvailability(
                            symbol=symbol,
                            data_type=dtype,
                            source=name,
                            resolution="1m",  # Default
                        )
                    )

        return available

    def _select_source(
        self,
        symbol: str,
        data_type: DataType,
        preferred_source: str = "auto",
    ) -> DataProvider | None:
        """
        Select best data source for a requirement.

        Args:
            symbol: Trading pair
            data_type: Type of data
            preferred_source: Specific source or "auto"

        Returns:
            Best matching provider or None
        """
        if preferred_source != "auto":
            if preferred_source in self._sources:
                provider, _ = self._sources[preferred_source]
                if provider.supports(symbol, data_type):
                    return provider
            return None

        # Auto-select by priority (lower = higher priority)
        candidates = []
        for name, (provider, priority) in self._sources.items():
            if provider.supports(symbol, data_type):
                candidates.append((priority, name, provider))

        if not candidates:
            return None

        # Sort by priority (ascending)
        candidates.sort(key=lambda x: x[0])
        return candidates[0][2]

    async def fetch(
        self,
        requirement: DataRequirement,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pl.LazyFrame:
        """
        Fetch data for a requirement.

        Args:
            requirement: Data requirement
            start: Start time (or use lookback)
            end: End time (default: now)
            limit: Max rows to return

        Returns:
            Polars LazyFrame with data

        Raises:
            ValueError: If no source available
        """
        # Calculate start from lookback if not provided
        if start is None and requirement.lookback is not None:
            end = end or datetime.utcnow()
            start = end - requirement.lookback

        # Select source
        provider = self._select_source(
            requirement.symbol,
            requirement.data_type,
            requirement.source,
        )

        if provider is None:
            raise ValueError(
                f"No data source available for {requirement.key} "
                f"(source={requirement.source})"
            )

        logger.debug(
            f"Fetching {requirement.key} from {provider.name} "
            f"(start={start}, end={end}, limit={limit})"
        )

        return await provider.fetch(
            symbol=requirement.symbol,
            data_type=requirement.data_type,
            resolution=requirement.resolution,
            start=start,
            end=end,
            limit=limit,
        )

    async def fetch_many(
        self,
        requirements: Sequence[DataRequirement],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pl.LazyFrame]:
        """
        Fetch multiple requirements in parallel.

        Args:
            requirements: List of requirements
            start: Start time
            end: End time

        Returns:
            Dict mapping requirement.key to LazyFrame
        """
        import asyncio

        async def fetch_one(req: DataRequirement) -> tuple[str, pl.LazyFrame | None]:
            try:
                df = await self.fetch(req, start=start, end=end)
                return req.key, df
            except Exception as e:
                if req.required:
                    raise
                logger.warning(f"Failed to fetch optional data {req.key}: {e}")
                return req.key, None

        results = await asyncio.gather(*[fetch_one(req) for req in requirements])
        return {k: v for k, v in results if v is not None}

    def validate_requirements(
        self,
        requirements: Sequence[DataRequirement],
    ) -> list[DataRequirement]:
        """
        Validate that all required data is available.

        Args:
            requirements: List of requirements to validate

        Returns:
            List of requirements that cannot be satisfied
        """
        missing = []

        for req in requirements:
            provider = self._select_source(req.symbol, req.data_type, req.source)
            if provider is None and req.required:
                missing.append(req)

        return missing

    def get_stats(self) -> dict[str, Any]:
        """Get catalog statistics."""
        return {
            "source_count": len(self._sources),
            "sources": [
                {"name": name, "priority": priority, "types": list(prov.supported_types)}
                for name, (prov, priority) in self._sources.items()
            ],
            "cache_size": len(self._availability_cache),
        }


# =============================================================================
# Global Catalog Instance
# =============================================================================

# Default global catalog (can be replaced)
_default_catalog: DataCatalog | None = None


def get_catalog() -> DataCatalog:
    """Get the default data catalog."""
    global _default_catalog
    if _default_catalog is None:
        _default_catalog = DataCatalog()
    return _default_catalog


def set_catalog(catalog: DataCatalog) -> None:
    """Set the default data catalog."""
    global _default_catalog
    _default_catalog = catalog
