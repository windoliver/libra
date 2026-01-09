"""
Data layer for LIBRA trading platform.

Provides:
- Time-series database integration (QuestDB)
- Data Catalog for source registration and discovery
- Polars schemas for market data types
- High-performance LRU cache with TTL
- Tiered storage (Hot/Warm/Cold)
- OHLCV downsampling utilities

Primary implementation: QuestDB (ADR-002)
- 5M+ rows/sec ingestion via ILP
- Sub-25ms OHLCV aggregation queries
- ASOF JOIN for backtest accuracy (prevents look-ahead bias)

See: https://github.com/windoliver/libra/issues/21
See: https://github.com/windoliver/libra/issues/23
"""

# Database
from libra.data.config import QuestDBConfig
from libra.data.protocol import TimeSeriesDB
from libra.data.questdb import AsyncQuestDBClient

# Data Catalog
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

# Schemas
from libra.data.schemas import (
    OHLCV_REQUIRED_COLUMNS,
    OHLCV_SCHEMA,
    ORDERBOOK_SCHEMA,
    SIGNAL_SCHEMA,
    TICK_REQUIRED_COLUMNS,
    TICK_SCHEMA,
    TRADE_REQUIRED_COLUMNS,
    TRADE_SCHEMA,
    add_bollinger_bands,
    add_ema,
    add_rsi,
    add_sma,
    bars_to_df,
    create_ohlcv_df,
    create_orderbook_df,
    create_tick_df,
    create_trade_df,
    ticks_to_df,
    validate_ohlcv,
    validate_tick,
)

# Cache
from libra.data.cache import (
    AggregationCache,
    CacheEntry,
    CacheStats,
    DataFrameCache,
    LRUCache,
)

# Storage
from libra.data.storage import (
    InMemoryBackend,
    ParquetBackend,
    StorageBackend,
    StorageStats,
    StorageTier,
    TierConfig,
    TieredStorage,
    create_default_storage,
)

# Downsampling
from libra.data.downsample import (
    RESOLUTION_SECONDS,
    VALID_DOWNSAMPLE,
    apply_retention_policy,
    can_downsample,
    downsample_multi_step,
    downsample_ohlcv,
    downsample_ohlcv_lazy,
    downsample_to_multiple,
    find_downsample_path,
    get_downsample_factor,
    get_resolution_seconds,
    ticks_to_bars,
    validate_ohlcv_integrity,
)


__all__ = [
    # Database
    "AsyncQuestDBClient",
    "QuestDBConfig",
    "TimeSeriesDB",
    # Catalog
    "BaseDataProvider",
    "DataAvailability",
    "DataCatalog",
    "DataProvider",
    "DataRequirement",
    "DataResolution",
    "DataType",
    "get_catalog",
    "set_catalog",
    # Schemas
    "OHLCV_REQUIRED_COLUMNS",
    "OHLCV_SCHEMA",
    "ORDERBOOK_SCHEMA",
    "SIGNAL_SCHEMA",
    "TICK_REQUIRED_COLUMNS",
    "TICK_SCHEMA",
    "TRADE_REQUIRED_COLUMNS",
    "TRADE_SCHEMA",
    "add_bollinger_bands",
    "add_ema",
    "add_rsi",
    "add_sma",
    "bars_to_df",
    "create_ohlcv_df",
    "create_orderbook_df",
    "create_tick_df",
    "create_trade_df",
    "ticks_to_df",
    "validate_ohlcv",
    "validate_tick",
    # Cache
    "AggregationCache",
    "CacheEntry",
    "CacheStats",
    "DataFrameCache",
    "LRUCache",
    # Storage
    "InMemoryBackend",
    "ParquetBackend",
    "StorageBackend",
    "StorageStats",
    "StorageTier",
    "TierConfig",
    "TieredStorage",
    "create_default_storage",
    # Downsampling
    "RESOLUTION_SECONDS",
    "VALID_DOWNSAMPLE",
    "apply_retention_policy",
    "can_downsample",
    "downsample_multi_step",
    "downsample_ohlcv",
    "downsample_ohlcv_lazy",
    "downsample_to_multiple",
    "find_downsample_path",
    "get_downsample_factor",
    "get_resolution_seconds",
    "ticks_to_bars",
    "validate_ohlcv_integrity",
]
