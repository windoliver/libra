"""
Shared memory price cache for zero-copy multi-process data sharing.

Provides:
- Zero-copy price sharing between processes
- Sub-microsecond read/write operations
- Automatic cleanup on exit
- Support for multiple price fields (bid, ask, last, volume)

See: https://github.com/windoliver/libra/issues/86
"""

from __future__ import annotations

import atexit
import logging
import struct
import time
from multiprocessing import shared_memory
from typing import NamedTuple

import numpy as np


logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


class PriceData(NamedTuple):
    """Price data for a single symbol."""

    bid: float
    ask: float
    last: float
    volume: float
    timestamp_ns: int


# Layout: bid(f64) + ask(f64) + last(f64) + volume(f64) + timestamp_ns(i64) = 40 bytes
PRICE_RECORD_SIZE = 40  # 5 fields * 8 bytes
PRICE_DTYPE = np.dtype([
    ("bid", np.float64),
    ("ask", np.float64),
    ("last", np.float64),
    ("volume", np.float64),
    ("timestamp_ns", np.int64),
])


# =============================================================================
# Shared Price Cache
# =============================================================================


class SharedPriceCache:
    """
    Shared memory price cache for zero-copy multi-process data sharing.

    Uses Python's multiprocessing.shared_memory for zero-copy access.
    Multiple processes can read/write prices without serialization overhead.

    Example (Primary process - creates shared memory):
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        cache = SharedPriceCache(symbols, create=True)

        # Update prices
        cache.update_price("BTC/USDT", bid=49999.0, ask=50001.0, last=50000.0)

        # ... worker processes can now read ...

        cache.close()  # Or use context manager

    Example (Worker process - attaches to existing):
        cache = SharedPriceCache(symbols, create=False)

        # Read prices (zero-copy!)
        price = cache.get_price("BTC/USDT")
        print(f"BTC last: {price.last}")

        cache.close()

    Usage with context manager:
        with SharedPriceCache(symbols, create=True) as cache:
            cache.update_price("BTC/USDT", last=50000.0)
    """

    # Default shared memory name
    DEFAULT_NAME = "libra_prices"

    def __init__(
        self,
        symbols: list[str],
        create: bool = True,
        name: str | None = None,
    ) -> None:
        """
        Initialize shared price cache.

        Args:
            symbols: List of trading symbols to track
            create: If True, create new shared memory; if False, attach to existing
            name: Shared memory name (default: "libra_prices")
        """
        self._symbols = list(symbols)
        self._symbol_index: dict[str, int] = {s: i for i, s in enumerate(symbols)}
        self._name = name or self.DEFAULT_NAME
        self._create = create
        self._closed = False

        # Calculate size needed
        size = len(symbols) * PRICE_RECORD_SIZE

        # Header: 8 bytes for symbol count + 8 bytes for version
        header_size = 16
        total_size = header_size + size

        if create:
            # Try to unlink existing (cleanup from previous crash)
            try:
                existing = shared_memory.SharedMemory(name=self._name)
                existing.close()
                existing.unlink()
                logger.debug("Cleaned up existing shared memory: %s", self._name)
            except FileNotFoundError:
                pass

            self._shm = shared_memory.SharedMemory(
                name=self._name,
                create=True,
                size=total_size,
            )
            # Write header
            struct.pack_into("QQ", self._shm.buf, 0, len(symbols), 1)  # count, version
            logger.info(
                "Created shared memory: %s (size=%d, symbols=%d)",
                self._name,
                total_size,
                len(symbols),
            )
        else:
            self._shm = shared_memory.SharedMemory(name=self._name)
            # Verify header
            count, version = struct.unpack_from("QQ", self._shm.buf, 0)
            if count != len(symbols):
                raise ValueError(
                    f"Symbol count mismatch: expected {len(symbols)}, got {count}"
                )
            logger.debug("Attached to shared memory: %s (version=%d)", self._name, version)

        # Create numpy view into shared memory (after header)
        self._prices: np.ndarray = np.ndarray(
            (len(symbols),),
            dtype=PRICE_DTYPE,
            buffer=self._shm.buf,
            offset=header_size,
        )

        # Initialize prices to zero if creating
        if create:
            self._prices.fill(0)

        # Register cleanup
        if create:
            atexit.register(self._cleanup)

        # Stats
        self._reads = 0
        self._writes = 0

    def update_price(
        self,
        symbol: str,
        bid: float | None = None,
        ask: float | None = None,
        last: float | None = None,
        volume: float | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        """
        Update price for a symbol.

        Args:
            symbol: Trading symbol
            bid: Bid price
            ask: Ask price
            last: Last trade price
            volume: Trading volume
            timestamp_ns: Timestamp in nanoseconds (default: current time)
        """
        if self._closed:
            raise RuntimeError("SharedPriceCache is closed")

        idx = self._symbol_index.get(symbol)
        if idx is None:
            raise KeyError(f"Unknown symbol: {symbol}")

        record = self._prices[idx]

        if bid is not None:
            record["bid"] = bid
        if ask is not None:
            record["ask"] = ask
        if last is not None:
            record["last"] = last
        if volume is not None:
            record["volume"] = volume

        record["timestamp_ns"] = timestamp_ns if timestamp_ns is not None else time.time_ns()

        self._writes += 1

    def update_all(
        self,
        symbol: str,
        bid: float,
        ask: float,
        last: float,
        volume: float = 0.0,
        timestamp_ns: int | None = None,
    ) -> None:
        """
        Update all price fields at once (more efficient).

        Args:
            symbol: Trading symbol
            bid: Bid price
            ask: Ask price
            last: Last trade price
            volume: Trading volume
            timestamp_ns: Timestamp in nanoseconds
        """
        if self._closed:
            raise RuntimeError("SharedPriceCache is closed")

        idx = self._symbol_index.get(symbol)
        if idx is None:
            raise KeyError(f"Unknown symbol: {symbol}")

        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
        self._prices[idx] = (bid, ask, last, volume, ts)
        self._writes += 1

    def get_price(self, symbol: str) -> PriceData:
        """
        Get price data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            PriceData named tuple
        """
        if self._closed:
            raise RuntimeError("SharedPriceCache is closed")

        idx = self._symbol_index.get(symbol)
        if idx is None:
            raise KeyError(f"Unknown symbol: {symbol}")

        record = self._prices[idx]
        self._reads += 1

        return PriceData(
            bid=float(record["bid"]),
            ask=float(record["ask"]),
            last=float(record["last"]),
            volume=float(record["volume"]),
            timestamp_ns=int(record["timestamp_ns"]),
        )

    def get_last(self, symbol: str) -> float:
        """Get just the last price (fastest access)."""
        if self._closed:
            raise RuntimeError("SharedPriceCache is closed")

        idx = self._symbol_index.get(symbol)
        if idx is None:
            raise KeyError(f"Unknown symbol: {symbol}")

        self._reads += 1
        return float(self._prices[idx]["last"])

    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        """Get bid and ask prices."""
        if self._closed:
            raise RuntimeError("SharedPriceCache is closed")

        idx = self._symbol_index.get(symbol)
        if idx is None:
            raise KeyError(f"Unknown symbol: {symbol}")

        record = self._prices[idx]
        self._reads += 1
        return float(record["bid"]), float(record["ask"])

    def get_all_prices(self) -> dict[str, PriceData]:
        """Get all prices as a dictionary."""
        if self._closed:
            raise RuntimeError("SharedPriceCache is closed")

        result = {}
        for symbol, idx in self._symbol_index.items():
            record = self._prices[idx]
            result[symbol] = PriceData(
                bid=float(record["bid"]),
                ask=float(record["ask"]),
                last=float(record["last"]),
                volume=float(record["volume"]),
                timestamp_ns=int(record["timestamp_ns"]),
            )
        self._reads += len(self._symbols)
        return result

    def __contains__(self, symbol: str) -> bool:
        """Check if symbol is in cache."""
        return symbol in self._symbol_index

    def __len__(self) -> int:
        """Number of symbols in cache."""
        return len(self._symbols)

    @property
    def symbols(self) -> list[str]:
        """List of tracked symbols."""
        return self._symbols.copy()

    @property
    def stats(self) -> dict[str, int]:
        """Cache statistics."""
        return {
            "reads": self._reads,
            "writes": self._writes,
            "symbols": len(self._symbols),
            "size_bytes": len(self._symbols) * PRICE_RECORD_SIZE + 16,
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Close shared memory (but don't unlink)."""
        if self._closed:
            return

        self._closed = True
        self._shm.close()
        logger.debug("Closed shared memory: %s", self._name)

    def unlink(self) -> None:
        """Unlink (delete) the shared memory. Only call from the creator process."""
        if not self._create:
            logger.warning("Only the creator process should unlink shared memory")
            return

        try:
            self._shm.unlink()
            logger.info("Unlinked shared memory: %s", self._name)
        except FileNotFoundError:
            pass

    def _cleanup(self) -> None:
        """Cleanup handler for atexit."""
        if not self._closed:
            self.close()
        if self._create:
            self.unlink()

    def __enter__(self) -> SharedPriceCache:
        """Context manager entry."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Context manager exit."""
        self.close()
        if self._create:
            self.unlink()

    # =========================================================================
    # Class Methods
    # =========================================================================

    @classmethod
    def exists(cls, name: str | None = None) -> bool:
        """Check if shared memory exists."""
        shm_name = name or cls.DEFAULT_NAME
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            return True
        except FileNotFoundError:
            return False

    @classmethod
    def cleanup(cls, name: str | None = None) -> bool:
        """
        Force cleanup of shared memory (use if process crashed).

        Returns:
            True if cleaned up, False if not found
        """
        shm_name = name or cls.DEFAULT_NAME
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
            logger.info("Force cleaned up shared memory: %s", shm_name)
            return True
        except FileNotFoundError:
            return False
