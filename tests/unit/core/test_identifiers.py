"""
Tests for interned identifiers (Issue #111).
"""

from __future__ import annotations

import pytest

from libra.core.identifiers import (
    ClientOrderId,
    StrategyId,
    Symbol,
    VenueId,
    clear_all_identifier_caches,
    get_identifier_stats,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear all caches before each test."""
    clear_all_identifier_caches()
    yield
    clear_all_identifier_caches()


# =============================================================================
# Symbol Tests
# =============================================================================


class TestSymbol:
    """Tests for Symbol class."""

    def test_create_symbol(self):
        """Test creating a symbol."""
        symbol = Symbol("BTC/USDT")
        assert symbol.value == "BTC/USDT"
        assert str(symbol) == "BTC/USDT"

    def test_interning_returns_same_instance(self):
        """Test that interning returns the same instance."""
        symbol1 = Symbol("BTC/USDT")
        symbol2 = Symbol("BTC/USDT")

        assert symbol1 is symbol2
        assert id(symbol1) == id(symbol2)

    def test_different_symbols_different_instances(self):
        """Test that different symbols have different instances."""
        btc = Symbol("BTC/USDT")
        eth = Symbol("ETH/USDT")

        assert btc is not eth
        assert btc != eth

    def test_hash_precomputed(self):
        """Test that hash is pre-computed and consistent."""
        symbol = Symbol("BTC/USDT")

        # Hash should be consistent
        assert hash(symbol) == hash(symbol)
        assert hash(symbol) == hash("BTC/USDT")

    def test_equality_with_symbol(self):
        """Test equality between symbols."""
        symbol1 = Symbol("BTC/USDT")
        symbol2 = Symbol("BTC/USDT")
        symbol3 = Symbol("ETH/USDT")

        assert symbol1 == symbol2
        assert symbol1 != symbol3

    def test_equality_with_str(self):
        """Test equality with string."""
        symbol = Symbol("BTC/USDT")

        assert symbol == "BTC/USDT"
        assert symbol != "ETH/USDT"

    def test_inequality(self):
        """Test inequality operator."""
        symbol1 = Symbol("BTC/USDT")
        symbol2 = Symbol("ETH/USDT")

        assert symbol1 != symbol2
        assert not (symbol1 != Symbol("BTC/USDT"))

    def test_repr(self):
        """Test repr output."""
        symbol = Symbol("BTC/USDT")
        assert repr(symbol) == "Symbol('BTC/USDT')"

    def test_comparison_operators(self):
        """Test comparison operators for sorting."""
        btc = Symbol("BTC/USDT")
        eth = Symbol("ETH/USDT")
        ada = Symbol("ADA/USDT")

        assert ada < btc < eth
        assert ada <= btc <= eth
        assert eth > btc > ada
        assert eth >= btc >= ada

    def test_use_in_dict(self):
        """Test using Symbol as dictionary key."""
        btc = Symbol("BTC/USDT")
        eth = Symbol("ETH/USDT")

        prices = {btc: 50000, eth: 3000}

        # Lookup with same instance
        assert prices[btc] == 50000

        # Lookup with new Symbol (returns same instance)
        assert prices[Symbol("BTC/USDT")] == 50000

    def test_use_in_set(self):
        """Test using Symbol in set."""
        symbols = {Symbol("BTC/USDT"), Symbol("ETH/USDT"), Symbol("BTC/USDT")}

        assert len(symbols) == 2
        assert Symbol("BTC/USDT") in symbols

    def test_cache_size(self):
        """Test cache_size method."""
        assert Symbol.cache_size() == 0

        Symbol("BTC/USDT")
        assert Symbol.cache_size() == 1

        Symbol("ETH/USDT")
        assert Symbol.cache_size() == 2

        # Same symbol doesn't increase count
        Symbol("BTC/USDT")
        assert Symbol.cache_size() == 2

    def test_clear_cache(self):
        """Test clearing the cache."""
        Symbol("BTC/USDT")
        Symbol("ETH/USDT")
        assert Symbol.cache_size() == 2

        Symbol.clear_cache()
        assert Symbol.cache_size() == 0

    def test_get_cached(self):
        """Test get_cached method."""
        assert Symbol.get_cached("BTC/USDT") is None

        btc = Symbol("BTC/USDT")
        assert Symbol.get_cached("BTC/USDT") is btc
        assert Symbol.get_cached("ETH/USDT") is None


# =============================================================================
# VenueId Tests
# =============================================================================


class TestVenueId:
    """Tests for VenueId class."""

    def test_create_venue(self):
        """Test creating a venue ID."""
        venue = VenueId("binance")
        assert venue.value == "binance"

    def test_normalizes_to_lowercase(self):
        """Test that venue IDs are normalized to lowercase."""
        venue1 = VenueId("Binance")
        venue2 = VenueId("BINANCE")
        venue3 = VenueId("binance")

        assert venue1 is venue2
        assert venue2 is venue3
        assert venue1.value == "binance"

    def test_interning(self):
        """Test that interning works."""
        venue1 = VenueId("kraken")
        venue2 = VenueId("kraken")

        assert venue1 is venue2

    def test_equality_with_str(self):
        """Test equality with string (case-insensitive)."""
        venue = VenueId("binance")

        assert venue == "binance"
        assert venue == "Binance"
        assert venue == "BINANCE"

    def test_repr(self):
        """Test repr output."""
        venue = VenueId("binance")
        assert repr(venue) == "VenueId('binance')"


# =============================================================================
# ClientOrderId Tests
# =============================================================================


class TestClientOrderId:
    """Tests for ClientOrderId class."""

    def test_create_order_id(self):
        """Test creating an order ID."""
        order_id = ClientOrderId("my-order-123")
        assert order_id.value == "my-order-123"

    def test_interning(self):
        """Test that interning works."""
        id1 = ClientOrderId("order-1")
        id2 = ClientOrderId("order-1")

        assert id1 is id2

    def test_use_in_dict(self):
        """Test using as dictionary key."""
        id1 = ClientOrderId("order-1")
        orders = {id1: {"status": "open"}}

        assert orders[ClientOrderId("order-1")]["status"] == "open"

    def test_repr(self):
        """Test repr output."""
        order_id = ClientOrderId("order-123")
        assert repr(order_id) == "ClientOrderId('order-123')"


# =============================================================================
# StrategyId Tests
# =============================================================================


class TestStrategyId:
    """Tests for StrategyId class."""

    def test_create_strategy_id(self):
        """Test creating a strategy ID."""
        strategy = StrategyId("momentum-v1")
        assert strategy.value == "momentum-v1"

    def test_interning(self):
        """Test that interning works."""
        s1 = StrategyId("mean-reversion")
        s2 = StrategyId("mean-reversion")

        assert s1 is s2

    def test_repr(self):
        """Test repr output."""
        strategy = StrategyId("my-strategy")
        assert repr(strategy) == "StrategyId('my-strategy')"


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_clear_all_caches(self):
        """Test clearing all caches."""
        Symbol("BTC/USDT")
        VenueId("binance")
        ClientOrderId("order-1")
        StrategyId("strategy-1")

        assert Symbol.cache_size() == 1
        assert VenueId.cache_size() == 1
        assert ClientOrderId.cache_size() == 1
        assert StrategyId.cache_size() == 1

        clear_all_identifier_caches()

        assert Symbol.cache_size() == 0
        assert VenueId.cache_size() == 0
        assert ClientOrderId.cache_size() == 0
        assert StrategyId.cache_size() == 0

    def test_get_identifier_stats(self):
        """Test getting identifier statistics."""
        Symbol("BTC/USDT")
        Symbol("ETH/USDT")
        VenueId("binance")
        ClientOrderId("order-1")
        ClientOrderId("order-2")
        ClientOrderId("order-3")
        StrategyId("strategy-1")

        stats = get_identifier_stats()

        assert stats["symbols"] == 2
        assert stats["venues"] == 1
        assert stats["client_order_ids"] == 3
        assert stats["strategy_ids"] == 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_string(self):
        """Test with empty string."""
        symbol = Symbol("")
        assert symbol.value == ""
        assert Symbol("") is symbol

    def test_unicode_symbols(self):
        """Test with unicode characters."""
        symbol = Symbol("BTC/USD₮")
        assert symbol.value == "BTC/USD₮"
        assert Symbol("BTC/USD₮") is symbol

    def test_long_string(self):
        """Test with long string."""
        long_name = "A" * 1000
        symbol = Symbol(long_name)
        assert symbol.value == long_name
        assert Symbol(long_name) is symbol

    def test_special_characters(self):
        """Test with special characters."""
        symbol = Symbol("BTC-PERP@binance:futures")
        assert symbol.value == "BTC-PERP@binance:futures"

    def test_equality_with_non_string(self):
        """Test equality with non-string types."""
        symbol = Symbol("BTC/USDT")

        assert symbol != 123
        assert symbol != None
        assert symbol != ["BTC/USDT"]
        assert symbol != {"symbol": "BTC/USDT"}

    def test_sorting_list_of_symbols(self):
        """Test sorting a list of symbols."""
        symbols = [Symbol("ETH"), Symbol("BTC"), Symbol("ADA"), Symbol("SOL")]
        sorted_symbols = sorted(symbols)

        assert [s.value for s in sorted_symbols] == ["ADA", "BTC", "ETH", "SOL"]

    def test_many_symbols_memory_efficiency(self):
        """Test that many references use same object."""
        # Create 1000 references to same symbol
        symbols = [Symbol("BTC/USDT") for _ in range(1000)]

        # All should be the same object
        assert all(s is symbols[0] for s in symbols)

        # Cache should only have 1 entry
        assert Symbol.cache_size() == 1


# =============================================================================
# Hash Consistency Tests
# =============================================================================


class TestHashConsistency:
    """Test hash consistency and correctness."""

    def test_hash_equals_string_hash(self):
        """Test that Symbol hash equals string hash."""
        symbol = Symbol("BTC/USDT")
        assert hash(symbol) == hash("BTC/USDT")

    def test_hash_consistent_across_instances(self):
        """Test hash is consistent across lookups."""
        symbol1 = Symbol("ETH/USDT")
        symbol2 = Symbol("ETH/USDT")

        assert hash(symbol1) == hash(symbol2)

    def test_hash_in_frozen_set(self):
        """Test using in frozenset."""
        symbols = frozenset([Symbol("BTC"), Symbol("ETH"), Symbol("BTC")])
        assert len(symbols) == 2
