"""
Tiered Storage: Hot/Warm/Cold data management.

Provides:
- Hot tier: In-memory for active symbols (sub-millisecond access)
- Warm tier: Local SSD/database for recent history (millisecond access)
- Cold tier: Object storage for archives (second+ access)
- Automatic migration between tiers

Storage Strategy:
- Hot: Last N minutes of tick data, current session bars
- Warm: Recent weeks of daily data, historical bars
- Cold: Archived data (months/years), compressed

See: https://github.com/windoliver/libra/issues/23
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import polars as pl


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# =============================================================================
# Storage Tier Definitions
# =============================================================================


class StorageTier(str, Enum):
    """Storage tiers by access speed and cost."""

    HOT = "hot"  # In-memory, sub-ms access
    WARM = "warm"  # Local SSD/database, ms access
    COLD = "cold"  # Object storage, second+ access


@dataclass
class TierConfig:
    """Configuration for a storage tier."""

    tier: StorageTier
    max_age: timedelta  # Max age before migration to next tier
    max_size_mb: int  # Max storage size in MB
    compression: bool = False  # Enable compression
    path: Path | None = None  # Storage path (for warm/cold)

    @classmethod
    def default_hot(cls) -> TierConfig:
        """Default hot tier config."""
        return cls(
            tier=StorageTier.HOT,
            max_age=timedelta(hours=1),
            max_size_mb=1024,  # 1GB
        )

    @classmethod
    def default_warm(cls, path: Path | None = None) -> TierConfig:
        """Default warm tier config."""
        return cls(
            tier=StorageTier.WARM,
            max_age=timedelta(days=30),
            max_size_mb=10240,  # 10GB
            compression=True,
            path=path or Path("./data/warm"),
        )

    @classmethod
    def default_cold(cls, path: Path | None = None) -> TierConfig:
        """Default cold tier config."""
        return cls(
            tier=StorageTier.COLD,
            max_age=timedelta(days=365),  # 1 year
            max_size_mb=102400,  # 100GB
            compression=True,
            path=path or Path("./data/cold"),
        )


# =============================================================================
# Storage Backend Protocol
# =============================================================================


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for storage backends."""

    @property
    def tier(self) -> StorageTier:
        """Storage tier."""
        ...

    async def read(
        self,
        key: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame | None:
        """Read data from storage."""
        ...

    async def write(self, key: str, data: pl.DataFrame) -> None:
        """Write data to storage."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete data from storage."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if data exists."""
        ...

    async def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with optional prefix."""
        ...

    async def get_metadata(self, key: str) -> dict[str, Any] | None:
        """Get metadata for a key."""
        ...


# =============================================================================
# In-Memory Backend (Hot Tier)
# =============================================================================


class InMemoryBackend:
    """
    In-memory storage for hot tier.

    Fast access, limited capacity.
    """

    def __init__(self, max_size_mb: int = 1024) -> None:
        self._data: dict[str, pl.DataFrame] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._max_size_mb = max_size_mb
        self._lock = asyncio.Lock()

    @property
    def tier(self) -> StorageTier:
        return StorageTier.HOT

    async def read(
        self,
        key: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame | None:
        async with self._lock:
            df = self._data.get(key)
            if df is None:
                return None

            # Apply time filter
            if "timestamp" in df.columns:
                if start is not None:
                    df = df.filter(pl.col("timestamp") >= start)
                if end is not None:
                    df = df.filter(pl.col("timestamp") <= end)

            return df

    async def write(self, key: str, data: pl.DataFrame) -> None:
        async with self._lock:
            self._data[key] = data
            self._metadata[key] = {
                "updated_at": time.time(),
                "row_count": len(data),
                "columns": data.columns,
            }

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._data:
                del self._data[key]
                self._metadata.pop(key, None)
                return True
            return False

    async def exists(self, key: str) -> bool:
        return key in self._data

    async def list_keys(self, prefix: str = "") -> list[str]:
        return [k for k in self._data.keys() if k.startswith(prefix)]

    async def get_metadata(self, key: str) -> dict[str, Any] | None:
        return self._metadata.get(key)

    def estimate_size_mb(self) -> float:
        """Estimate current memory usage."""
        total_bytes = 0
        for df in self._data.values():
            total_bytes += df.estimated_size()
        return total_bytes / (1024 * 1024)


# =============================================================================
# Parquet Backend (Warm/Cold Tier)
# =============================================================================


class ParquetBackend:
    """
    Parquet file storage for warm/cold tiers.

    Features:
    - Columnar storage for efficient queries
    - Optional compression (snappy, zstd)
    - Time-partitioned files
    """

    def __init__(
        self,
        base_path: Path,
        tier: StorageTier = StorageTier.WARM,
        compression: Literal["snappy", "zstd", "lz4", "gzip", "uncompressed"] = "snappy",
    ) -> None:
        self._base_path = Path(base_path)
        self._tier = tier
        self._compression = compression
        self._base_path.mkdir(parents=True, exist_ok=True)

    @property
    def tier(self) -> StorageTier:
        return self._tier

    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path."""
        # Replace invalid path chars
        safe_key = key.replace("/", "_").replace(":", "_")
        return self._base_path / f"{safe_key}.parquet"

    async def read(
        self,
        key: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame | None:
        path = self._key_to_path(key)
        if not path.exists():
            return None

        try:
            # Use lazy scan for efficient filtering
            lf = pl.scan_parquet(path)

            # Use collect_schema().names() to avoid performance warning
            columns = lf.collect_schema().names()
            if "timestamp" in columns:
                if start is not None:
                    lf = lf.filter(pl.col("timestamp") >= start)
                if end is not None:
                    lf = lf.filter(pl.col("timestamp") <= end)

            return lf.collect()
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None

    async def write(self, key: str, data: pl.DataFrame) -> None:
        path = self._key_to_path(key)
        try:
            data.write_parquet(path, compression=self._compression)
            logger.debug(f"Wrote {len(data)} rows to {path}")
        except Exception as e:
            logger.error(f"Error writing {path}: {e}")
            raise

    async def delete(self, key: str) -> bool:
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    async def exists(self, key: str) -> bool:
        return self._key_to_path(key).exists()

    async def list_keys(self, prefix: str = "") -> list[str]:
        keys = []
        for path in self._base_path.glob("*.parquet"):
            key = path.stem.replace("_", "/", 1)  # Restore first separator
            if key.startswith(prefix):
                keys.append(key)
        return keys

    async def get_metadata(self, key: str) -> dict[str, Any] | None:
        path = self._key_to_path(key)
        if not path.exists():
            return None

        stat = path.stat()
        return {
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
            "path": str(path),
        }


# =============================================================================
# Bcolz Backend (Zipline-style columnar storage) - Issue #105
# =============================================================================

# Optional bcolz import
try:
    import bcolz

    BCOLZ_AVAILABLE = True
except ImportError:
    BCOLZ_AVAILABLE = False
    bcolz = None  # type: ignore


class BcolzBackend:
    """
    Bcolz columnar storage for Zipline-style data bundles.

    Features:
    - Compressed columnar format (blosc compression)
    - Memory-mapped access for large datasets
    - Efficient time range queries
    - Used by Zipline for years of historical data

    Provides ~2-5x faster reads than Parquet for time-series data
    with better compression ratios.

    Example:
        backend = BcolzBackend(Path("./data/bcolz"))
        await backend.write("BTC/USDT:1d", df)
        df = await backend.read("BTC/USDT:1d", start=datetime(2024, 1, 1))

    Note: Requires bcolz-zipline package: pip install libra[bcolz]
    """

    def __init__(
        self,
        base_path: Path,
        tier: StorageTier = StorageTier.WARM,
        cparams: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize bcolz backend.

        Args:
            base_path: Base directory for bcolz ctables
            tier: Storage tier classification
            cparams: Compression parameters (default: blosc with lz4)
        """
        if not BCOLZ_AVAILABLE:
            raise ImportError(
                "bcolz not available. Install with: pip install libra[bcolz]"
            )

        self._base_path = Path(base_path)
        self._tier = tier
        self._base_path.mkdir(parents=True, exist_ok=True)

        # Default compression: blosc with lz4 (fast compression/decompression)
        self._cparams = cparams or {
            "cname": "lz4",
            "clevel": 5,
            "shuffle": bcolz.SHUFFLE,
        }

    @property
    def tier(self) -> StorageTier:
        return self._tier

    def _key_to_path(self, key: str) -> Path:
        """Convert key to directory path."""
        safe_key = key.replace("/", "_").replace(":", "_")
        return self._base_path / safe_key

    async def read(
        self,
        key: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame | None:
        """
        Read data from bcolz ctable.

        Supports efficient time range queries using boolean indexing.

        Args:
            key: Data key (e.g., "BTC/USDT:1d")
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)

        Returns:
            Polars DataFrame or None if not found
        """
        path = self._key_to_path(key)
        if not path.exists():
            return None

        try:
            ctable = bcolz.open(str(path), mode="r")

            # Build mask for time range query
            if "timestamp" in ctable.names and (start is not None or end is not None):
                timestamps = ctable["timestamp"][:]

                # Convert datetime to appropriate format
                if start is not None:
                    if hasattr(start, "timestamp"):
                        start_val = int(start.timestamp() * 1_000_000_000)  # ns
                    else:
                        start_val = start
                else:
                    start_val = None

                if end is not None:
                    if hasattr(end, "timestamp"):
                        end_val = int(end.timestamp() * 1_000_000_000)  # ns
                    else:
                        end_val = end
                else:
                    end_val = None

                # Apply mask
                import numpy as np

                mask = np.ones(len(ctable), dtype=bool)
                if start_val is not None:
                    mask &= timestamps >= start_val
                if end_val is not None:
                    mask &= timestamps <= end_val

                # Extract filtered data
                data = {col: ctable[col][mask] for col in ctable.names}
            else:
                # No filtering, read all data
                data = {col: ctable[col][:] for col in ctable.names}

            return pl.DataFrame(data)

        except Exception as e:
            logger.error(f"Error reading bcolz {path}: {e}")
            return None

    async def write(self, key: str, data: pl.DataFrame) -> None:
        """
        Write data to bcolz ctable.

        Creates a new ctable or overwrites existing one.

        Args:
            key: Data key
            data: Polars DataFrame to write
        """
        path = self._key_to_path(key)

        try:
            # Remove existing ctable if present
            if path.exists():
                import shutil

                shutil.rmtree(path)

            # Convert Polars columns to numpy arrays
            columns = {}
            for col in data.columns:
                arr = data[col].to_numpy()
                columns[col] = arr

            # Create ctable with compression
            bcolz.ctable(
                columns=columns,
                rootdir=str(path),
                mode="w",
                cparams=bcolz.cparams(**self._cparams),
            )

            logger.debug(f"Wrote {len(data)} rows to bcolz {path}")

        except Exception as e:
            logger.error(f"Error writing bcolz {path}: {e}")
            raise

    async def delete(self, key: str) -> bool:
        """Delete bcolz ctable."""
        path = self._key_to_path(key)
        if path.exists():
            import shutil

            shutil.rmtree(path)
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if ctable exists."""
        return self._key_to_path(key).exists()

    async def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys (ctable directories)."""
        keys = []
        for path in self._base_path.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                key = path.name.replace("_", "/", 1)
                if key.startswith(prefix):
                    keys.append(key)
        return keys

    async def get_metadata(self, key: str) -> dict[str, Any] | None:
        """Get metadata for ctable."""
        path = self._key_to_path(key)
        if not path.exists():
            return None

        try:
            ctable = bcolz.open(str(path), mode="r")
            return {
                "nrows": ctable.nrows,
                "columns": ctable.names,
                "dtype": {col: str(ctable[col].dtype) for col in ctable.names},
                "cparams": str(ctable.cparams),
                "path": str(path),
                "size_bytes": sum(f.stat().st_size for f in path.rglob("*") if f.is_file()),
            }
        except Exception as e:
            logger.error(f"Error getting bcolz metadata {path}: {e}")
            return None

    async def append(self, key: str, data: pl.DataFrame) -> None:
        """
        Append data to existing ctable.

        More efficient than rewriting for incremental updates.

        Args:
            key: Data key
            data: Data to append
        """
        path = self._key_to_path(key)

        if not path.exists():
            # No existing data, create new
            await self.write(key, data)
            return

        try:
            ctable = bcolz.open(str(path), mode="a")

            # Append row by row (bcolz doesn't support bulk append easily)
            for i in range(len(data)):
                row = tuple(data[col][i] for col in ctable.names)
                ctable.append([row])

            ctable.flush()
            logger.debug(f"Appended {len(data)} rows to bcolz {path}")

        except Exception as e:
            logger.error(f"Error appending to bcolz {path}: {e}")
            raise


class BcolzDataStore:
    """
    High-level bcolz data store for OHLCV bar data.

    Provides a simple interface for storing and retrieving
    historical bar data in Zipline-style bundles.

    Example:
        store = BcolzDataStore("./data/bundles")

        # Save bars
        store.save_bars("BTC/USDT", df)

        # Load with time range
        df = store.load_bars("BTC/USDT", start=datetime(2024, 1, 1))

        # List available symbols
        symbols = store.list_symbols()
    """

    def __init__(self, path: str | Path) -> None:
        """
        Initialize bcolz data store.

        Args:
            path: Base directory for data bundles
        """
        if not BCOLZ_AVAILABLE:
            raise ImportError(
                "bcolz not available. Install with: pip install libra[bcolz]"
            )

        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

    def save_bars(
        self,
        symbol: str,
        df: pl.DataFrame,
        timeframe: str = "1d",
    ) -> None:
        """
        Save OHLCV bars to bcolz ctable.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            df: DataFrame with columns: timestamp, open, high, low, close, volume
            timeframe: Bar timeframe (e.g., "1d", "1h", "1m")
        """
        # Validate required columns
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Create directory structure: path/symbol/timeframe/
        safe_symbol = symbol.replace("/", "_")
        ctable_path = self._path / safe_symbol / timeframe

        if ctable_path.exists():
            import shutil

            shutil.rmtree(ctable_path)

        ctable_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy and create ctable
        columns = {
            "timestamp": df["timestamp"].to_numpy(),
            "open": df["open"].to_numpy().astype("float64"),
            "high": df["high"].to_numpy().astype("float64"),
            "low": df["low"].to_numpy().astype("float64"),
            "close": df["close"].to_numpy().astype("float64"),
            "volume": df["volume"].to_numpy().astype("float64"),
        }

        bcolz.ctable(
            columns=columns,
            rootdir=str(ctable_path),
            mode="w",
            cparams=bcolz.cparams(cname="lz4", clevel=5, shuffle=bcolz.SHUFFLE),
        )

        logger.info(f"Saved {len(df)} bars for {symbol}/{timeframe}")

    def load_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: datetime | int | None = None,
        end: datetime | int | None = None,
    ) -> pl.DataFrame | None:
        """
        Load OHLCV bars from bcolz ctable.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            start: Start timestamp (datetime or nanoseconds)
            end: End timestamp (datetime or nanoseconds)

        Returns:
            Polars DataFrame or None if not found
        """
        safe_symbol = symbol.replace("/", "_")
        ctable_path = self._path / safe_symbol / timeframe

        if not ctable_path.exists():
            return None

        ctable = bcolz.open(str(ctable_path), mode="r")
        timestamps = ctable["timestamp"][:]

        # Convert datetime to ns if needed
        import numpy as np

        if start is not None:
            if isinstance(start, datetime):
                start = int(start.timestamp() * 1_000_000_000)
        if end is not None:
            if isinstance(end, datetime):
                end = int(end.timestamp() * 1_000_000_000)

        # Build mask
        mask = np.ones(len(ctable), dtype=bool)
        if start is not None:
            mask &= timestamps >= start
        if end is not None:
            mask &= timestamps <= end

        # Extract data
        data = {col: ctable[col][mask] for col in ctable.names}
        return pl.DataFrame(data)

    def list_symbols(self) -> list[str]:
        """List all available symbols."""
        symbols = []
        for path in self._path.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                symbol = path.name.replace("_", "/")
                symbols.append(symbol)
        return sorted(symbols)

    def list_timeframes(self, symbol: str) -> list[str]:
        """List available timeframes for a symbol."""
        safe_symbol = symbol.replace("/", "_")
        symbol_path = self._path / safe_symbol

        if not symbol_path.exists():
            return []

        return sorted(
            p.name for p in symbol_path.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    def get_info(self, symbol: str, timeframe: str = "1d") -> dict[str, Any] | None:
        """Get info about stored data."""
        safe_symbol = symbol.replace("/", "_")
        ctable_path = self._path / safe_symbol / timeframe

        if not ctable_path.exists():
            return None

        ctable = bcolz.open(str(ctable_path), mode="r")
        timestamps = ctable["timestamp"][:]

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "nrows": ctable.nrows,
            "start_timestamp": int(timestamps.min()) if len(timestamps) > 0 else None,
            "end_timestamp": int(timestamps.max()) if len(timestamps) > 0 else None,
            "columns": ctable.names,
            "compression": str(ctable.cparams),
        }

    def delete(self, symbol: str, timeframe: str | None = None) -> bool:
        """
        Delete stored data.

        Args:
            symbol: Symbol to delete
            timeframe: Specific timeframe (None = all timeframes)

        Returns:
            True if deleted
        """
        import shutil

        safe_symbol = symbol.replace("/", "_")

        if timeframe:
            path = self._path / safe_symbol / timeframe
        else:
            path = self._path / safe_symbol

        if path.exists():
            shutil.rmtree(path)
            return True
        return False


# =============================================================================
# Tiered Storage Manager
# =============================================================================


@dataclass
class StorageStats:
    """Statistics for tiered storage."""

    hot_reads: int = 0
    warm_reads: int = 0
    cold_reads: int = 0
    writes: int = 0
    migrations: int = 0
    hot_size_mb: float = 0.0
    warm_size_mb: float = 0.0


class TieredStorage:
    """
    Manages data across hot/warm/cold tiers.

    Features:
    - Automatic tier selection for reads
    - Write-through to appropriate tier
    - Background migration of old data
    - Unified query interface

    Example:
        storage = TieredStorage(
            hot=InMemoryBackend(max_size_mb=512),
            warm=ParquetBackend(Path("./data/warm")),
            cold=ParquetBackend(Path("./data/cold"), compression="zstd"),
        )

        # Write to hot tier
        await storage.write("BTC/USDT:1m", df, tier=StorageTier.HOT)

        # Read (automatically checks all tiers)
        df = await storage.read("BTC/USDT:1m")

        # Run migration (move old data down)
        await storage.migrate()
    """

    def __init__(
        self,
        hot: StorageBackend | None = None,
        warm: StorageBackend | None = None,
        cold: StorageBackend | None = None,
    ) -> None:
        self._hot = hot or InMemoryBackend()
        self._warm = warm
        self._cold = cold
        self._stats = StorageStats()
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> StorageStats:
        """Get storage statistics."""
        if isinstance(self._hot, InMemoryBackend):
            self._stats.hot_size_mb = self._hot.estimate_size_mb()
        return self._stats

    async def read(
        self,
        key: str,
        start: datetime | None = None,
        end: datetime | None = None,
        tier: StorageTier | None = None,
    ) -> pl.DataFrame | None:
        """
        Read data, checking tiers in order.

        Args:
            key: Data key
            start: Start timestamp
            end: End timestamp
            tier: Specific tier to read from (None = check all)

        Returns:
            DataFrame or None if not found
        """
        # Check specific tier
        if tier is not None:
            backend = self._get_backend(tier)
            if backend:
                return await backend.read(key, start, end)
            return None

        # Check tiers in order: hot -> warm -> cold
        for backend, stat_attr in [
            (self._hot, "hot_reads"),
            (self._warm, "warm_reads"),
            (self._cold, "cold_reads"),
        ]:
            if backend is None:
                continue

            data = await backend.read(key, start, end)
            if data is not None:
                setattr(self._stats, stat_attr, getattr(self._stats, stat_attr) + 1)
                return data

        return None

    async def write(
        self,
        key: str,
        data: pl.DataFrame,
        tier: StorageTier = StorageTier.HOT,
    ) -> None:
        """
        Write data to specified tier.

        Args:
            key: Data key
            data: DataFrame to write
            tier: Target tier (default: HOT)
        """
        backend = self._get_backend(tier)
        if backend is None:
            raise ValueError(f"No backend configured for tier {tier}")

        await backend.write(key, data)
        self._stats.writes += 1

    async def promote(self, key: str, from_tier: StorageTier, to_tier: StorageTier) -> bool:
        """
        Promote data to a faster tier.

        Args:
            key: Data key
            from_tier: Source tier
            to_tier: Target tier

        Returns:
            True if promoted, False if failed
        """
        source = self._get_backend(from_tier)
        target = self._get_backend(to_tier)

        if source is None or target is None:
            return False

        data = await source.read(key)
        if data is None:
            return False

        await target.write(key, data)
        self._stats.migrations += 1
        return True

    async def demote(
        self,
        key: str,
        from_tier: StorageTier,
        to_tier: StorageTier,
        delete_source: bool = True,
    ) -> bool:
        """
        Demote data to a slower tier.

        Args:
            key: Data key
            from_tier: Source tier
            to_tier: Target tier
            delete_source: Delete from source after copy

        Returns:
            True if demoted, False if failed
        """
        source = self._get_backend(from_tier)
        target = self._get_backend(to_tier)

        if source is None or target is None:
            return False

        data = await source.read(key)
        if data is None:
            return False

        await target.write(key, data)

        if delete_source:
            await source.delete(key)

        self._stats.migrations += 1
        return True

    async def migrate(
        self,
        hot_max_age: timedelta = timedelta(hours=1),
        warm_max_age: timedelta = timedelta(days=30),
    ) -> int:
        """
        Migrate old data down tiers.

        Args:
            hot_max_age: Max age for hot tier
            warm_max_age: Max age for warm tier

        Returns:
            Number of migrations performed
        """
        migrations = 0
        now = time.time()

        # Migrate hot -> warm
        if self._hot and self._warm:
            hot_keys = await self._hot.list_keys()
            for key in hot_keys:
                meta = await self._hot.get_metadata(key)
                if meta and now - meta.get("updated_at", now) > hot_max_age.total_seconds():
                    if await self.demote(key, StorageTier.HOT, StorageTier.WARM):
                        migrations += 1

        # Migrate warm -> cold
        if self._warm and self._cold:
            warm_keys = await self._warm.list_keys()
            for key in warm_keys:
                meta = await self._warm.get_metadata(key)
                if meta and now - meta.get("modified_at", now) > warm_max_age.total_seconds():
                    if await self.demote(key, StorageTier.WARM, StorageTier.COLD):
                        migrations += 1

        return migrations

    async def delete(self, key: str, tier: StorageTier | None = None) -> bool:
        """Delete data from specified tier or all tiers."""
        deleted = False

        if tier is not None:
            backend = self._get_backend(tier)
            if backend:
                deleted = await backend.delete(key)
        else:
            # Delete from all tiers
            for backend in [self._hot, self._warm, self._cold]:
                if backend and await backend.delete(key):
                    deleted = True

        return deleted

    async def list_keys(
        self,
        prefix: str = "",
        tier: StorageTier | None = None,
    ) -> dict[StorageTier, list[str]]:
        """
        List all keys across tiers.

        Returns:
            Dict mapping tier to list of keys
        """
        result: dict[StorageTier, list[str]] = {}

        backends = (
            [(tier, self._get_backend(tier))]
            if tier
            else [
                (StorageTier.HOT, self._hot),
                (StorageTier.WARM, self._warm),
                (StorageTier.COLD, self._cold),
            ]
        )

        for t, backend in backends:
            if backend:
                keys = await backend.list_keys(prefix)
                if keys:
                    result[t] = keys

        return result

    def _get_backend(self, tier: StorageTier) -> StorageBackend | None:
        """Get backend for tier."""
        return {
            StorageTier.HOT: self._hot,
            StorageTier.WARM: self._warm,
            StorageTier.COLD: self._cold,
        }.get(tier)


# =============================================================================
# Factory Functions
# =============================================================================


def create_default_storage(
    base_path: Path | str = "./data",
    hot_size_mb: int = 512,
) -> TieredStorage:
    """
    Create tiered storage with default configuration.

    Args:
        base_path: Base path for warm/cold storage
        hot_size_mb: Max hot tier size in MB

    Returns:
        Configured TieredStorage instance
    """
    base = Path(base_path)

    return TieredStorage(
        hot=InMemoryBackend(max_size_mb=hot_size_mb),
        warm=ParquetBackend(base / "warm", StorageTier.WARM, "snappy"),
        cold=ParquetBackend(base / "cold", StorageTier.COLD, "zstd"),
    )
