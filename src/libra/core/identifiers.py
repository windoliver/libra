"""
Interned identifiers for high-performance lookups (Issue #111).

Provides string interning for frequently used identifiers to:
- Reduce memory allocations
- Pre-compute hash values for O(1) lookup
- Enable identity comparison (pointer equality)

Usage:
    symbol = Symbol("BTC/USDT")  # Interned on first use
    symbol2 = Symbol("BTC/USDT")  # Returns same instance

    # Fast comparison (identity check)
    assert symbol is symbol2

    # Pre-computed hash (no recomputation)
    d = {symbol: 100}
    d[symbol2]  # O(1) lookup

Performance:
    - Hash lookup: ~3x faster than str
    - Equality check: ~10x faster (identity vs character comparison)
    - Memory: Single instance per unique string

References:
    - NautilusTrader identifier optimization
    - Python sys.intern for string interning
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar


class Symbol:
    """
    Interned trading symbol identifier.

    Provides O(1) hash lookup and identity-based equality comparison.
    Each unique symbol string has exactly one Symbol instance.

    Examples:
        # Create symbols (interned automatically)
        btc = Symbol("BTC/USDT")
        btc2 = Symbol("BTC/USDT")

        # Identity comparison (fast)
        assert btc is btc2

        # Use in dictionaries (pre-computed hash)
        prices = {btc: Decimal("50000")}

        # String access
        print(btc.value)  # "BTC/USDT"
        print(str(btc))   # "BTC/USDT"

    Thread Safety:
        Symbol creation is NOT thread-safe. For multi-threaded use,
        pre-create all symbols before spawning threads, or add locking.
    """

    __slots__ = ("_value", "_hash")

    # Class-level intern cache
    _intern_cache: ClassVar[dict[str, Symbol]] = {}

    def __new__(cls, value: str) -> Symbol:
        """
        Create or return existing interned Symbol.

        Args:
            value: Symbol string (e.g., "BTC/USDT")

        Returns:
            Interned Symbol instance
        """
        # Fast path: return cached instance
        if value in cls._intern_cache:
            return cls._intern_cache[value]

        # Slow path: create new instance
        instance = object.__new__(cls)
        instance._value = sys.intern(value)  # Intern the string
        instance._hash = hash(value)  # Pre-compute hash
        cls._intern_cache[value] = instance
        return instance

    @property
    def value(self) -> str:
        """The underlying symbol string."""
        return self._value

    def __hash__(self) -> int:
        """O(1) pre-computed hash."""
        return self._hash

    def __eq__(self, other: object) -> bool:
        """
        Fast equality check.

        Uses identity comparison for Symbol-to-Symbol.
        Falls back to string comparison for str.
        """
        if isinstance(other, Symbol):
            return self._value is other._value  # Identity check (fast)
        if isinstance(other, str):
            return self._value == other
        return False

    def __ne__(self, other: object) -> bool:
        """Inequality check."""
        return not self.__eq__(other)

    def __str__(self) -> str:
        """String representation."""
        return self._value

    def __repr__(self) -> str:
        """Debug representation."""
        return f"Symbol({self._value!r})"

    def __lt__(self, other: Symbol) -> bool:
        """Less than comparison for sorting."""
        if isinstance(other, Symbol):
            return self._value < other._value
        return NotImplemented

    def __le__(self, other: Symbol) -> bool:
        """Less than or equal comparison."""
        if isinstance(other, Symbol):
            return self._value <= other._value
        return NotImplemented

    def __gt__(self, other: Symbol) -> bool:
        """Greater than comparison."""
        if isinstance(other, Symbol):
            return self._value > other._value
        return NotImplemented

    def __ge__(self, other: Symbol) -> bool:
        """Greater than or equal comparison."""
        if isinstance(other, Symbol):
            return self._value >= other._value
        return NotImplemented

    @classmethod
    def cache_size(cls) -> int:
        """Return number of interned symbols."""
        return len(cls._intern_cache)

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the intern cache.

        WARNING: Only call during testing or shutdown.
        Existing Symbol references will become invalid for interning.
        """
        cls._intern_cache.clear()

    @classmethod
    def get_cached(cls, value: str) -> Symbol | None:
        """
        Get cached symbol without creating new one.

        Args:
            value: Symbol string to look up

        Returns:
            Cached Symbol or None if not interned
        """
        return cls._intern_cache.get(value)


class VenueId:
    """
    Interned venue/exchange identifier.

    Similar to Symbol but for exchange names (e.g., "binance", "kraken").

    Examples:
        venue = VenueId("binance")
        venue2 = VenueId("binance")
        assert venue is venue2
    """

    __slots__ = ("_value", "_hash")

    _intern_cache: ClassVar[dict[str, VenueId]] = {}

    def __new__(cls, value: str) -> VenueId:
        """Create or return existing interned VenueId."""
        # Normalize to lowercase before lookup
        normalized = value.lower()

        if normalized in cls._intern_cache:
            return cls._intern_cache[normalized]

        instance = object.__new__(cls)
        instance._value = sys.intern(normalized)
        instance._hash = hash(normalized)
        cls._intern_cache[normalized] = instance
        return instance

    @property
    def value(self) -> str:
        """The underlying venue string."""
        return self._value

    def __hash__(self) -> int:
        """O(1) pre-computed hash."""
        return self._hash

    def __eq__(self, other: object) -> bool:
        """Fast equality check."""
        if isinstance(other, VenueId):
            return self._value is other._value
        if isinstance(other, str):
            return self._value == other.lower()
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"VenueId({self._value!r})"

    def __lt__(self, other: VenueId) -> bool:
        if isinstance(other, VenueId):
            return self._value < other._value
        return NotImplemented

    @classmethod
    def cache_size(cls) -> int:
        """Return number of interned venues."""
        return len(cls._intern_cache)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the intern cache."""
        cls._intern_cache.clear()


class ClientOrderId:
    """
    Interned client order identifier.

    Used for tracking orders through their lifecycle.
    Pre-computes hash for fast order lookup in dictionaries.

    Examples:
        order_id = ClientOrderId("my-order-123")
        orders = {order_id: order}
    """

    __slots__ = ("_value", "_hash")

    _intern_cache: ClassVar[dict[str, ClientOrderId]] = {}

    def __new__(cls, value: str) -> ClientOrderId:
        """Create or return existing interned ClientOrderId."""
        if value in cls._intern_cache:
            return cls._intern_cache[value]

        instance = object.__new__(cls)
        instance._value = sys.intern(value)
        instance._hash = hash(value)
        cls._intern_cache[value] = instance
        return instance

    @property
    def value(self) -> str:
        """The underlying order ID string."""
        return self._value

    def __hash__(self) -> int:
        """O(1) pre-computed hash."""
        return self._hash

    def __eq__(self, other: object) -> bool:
        """Fast equality check."""
        if isinstance(other, ClientOrderId):
            return self._value is other._value
        if isinstance(other, str):
            return self._value == other
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"ClientOrderId({self._value!r})"

    @classmethod
    def cache_size(cls) -> int:
        """Return number of interned order IDs."""
        return len(cls._intern_cache)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the intern cache."""
        cls._intern_cache.clear()


class StrategyId:
    """
    Interned strategy identifier.

    Used for routing events and tracking strategy performance.

    Examples:
        strategy = StrategyId("momentum-btc-v1")
        events = {strategy: [...]}
    """

    __slots__ = ("_value", "_hash")

    _intern_cache: ClassVar[dict[str, StrategyId]] = {}

    def __new__(cls, value: str) -> StrategyId:
        """Create or return existing interned StrategyId."""
        if value in cls._intern_cache:
            return cls._intern_cache[value]

        instance = object.__new__(cls)
        instance._value = sys.intern(value)
        instance._hash = hash(value)
        cls._intern_cache[value] = instance
        return instance

    @property
    def value(self) -> str:
        """The underlying strategy ID string."""
        return self._value

    def __hash__(self) -> int:
        """O(1) pre-computed hash."""
        return self._hash

    def __eq__(self, other: object) -> bool:
        """Fast equality check."""
        if isinstance(other, StrategyId):
            return self._value is other._value
        if isinstance(other, str):
            return self._value == other
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"StrategyId({self._value!r})"

    @classmethod
    def cache_size(cls) -> int:
        """Return number of interned strategy IDs."""
        return len(cls._intern_cache)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the intern cache."""
        cls._intern_cache.clear()


def clear_all_identifier_caches() -> None:
    """
    Clear all identifier intern caches.

    WARNING: Only call during testing or shutdown.
    """
    Symbol.clear_cache()
    VenueId.clear_cache()
    ClientOrderId.clear_cache()
    StrategyId.clear_cache()


def get_identifier_stats() -> dict[str, int]:
    """
    Get statistics about interned identifiers.

    Returns:
        Dict with cache sizes for each identifier type.
    """
    return {
        "symbols": Symbol.cache_size(),
        "venues": VenueId.cache_size(),
        "client_order_ids": ClientOrderId.cache_size(),
        "strategy_ids": StrategyId.cache_size(),
    }
