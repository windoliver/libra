"""Unit tests for Cache component."""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal

import pytest

from libra.core.cache import Cache, RWLock
from libra.gateways.protocol import (
    Balance,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Tick,
)
from libra.strategies.protocol import Bar


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cache() -> Cache:
    """Create a Cache instance."""
    return Cache(max_bars=100, max_orders=100)


@pytest.fixture
def sample_order_result() -> OrderResult:
    """Create a sample order result."""
    return OrderResult(
        order_id="12345",
        symbol="BTC/USDT",
        status=OrderStatus.FILLED,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.1"),
        filled_amount=Decimal("0.1"),
        remaining_amount=Decimal("0"),
        average_price=Decimal("50000"),
        fee=Decimal("0.0001"),
        fee_currency="BTC",
        timestamp_ns=1000000000,
        client_order_id="client_123",
    )


@pytest.fixture
def sample_position() -> Position:
    """Create a sample position."""
    return Position(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        amount=Decimal("0.5"),
        entry_price=Decimal("50000"),
        current_price=Decimal("51000"),
        unrealized_pnl=Decimal("500"),
        realized_pnl=Decimal("100"),
    )


@pytest.fixture
def sample_tick() -> Tick:
    """Create a sample tick."""
    return Tick(
        symbol="BTC/USDT",
        bid=Decimal("49999"),
        ask=Decimal("50001"),
        last=Decimal("50000"),
        timestamp_ns=1000000000,
        volume_24h=Decimal("15000"),
    )


@pytest.fixture
def sample_bar() -> Bar:
    """Create a sample bar."""
    return Bar(
        symbol="BTC/USDT",
        timestamp_ns=1000000000,
        open=Decimal("49500"),
        high=Decimal("50500"),
        low=Decimal("49000"),
        close=Decimal("50000"),
        volume=Decimal("100"),
        timeframe="1h",
    )


@pytest.fixture
def sample_balance() -> Balance:
    """Create a sample balance."""
    return Balance(
        currency="USDT",
        total=Decimal("10000"),
        available=Decimal("8000"),
        locked=Decimal("2000"),
    )


# =============================================================================
# Order Cache Tests
# =============================================================================


class TestOrderCache:
    """Tests for order caching."""

    @pytest.mark.asyncio
    async def test_add_order(self, cache: Cache, sample_order_result: OrderResult) -> None:
        """Test adding an order."""
        await cache.add_order(sample_order_result)

        order = cache.order("client_123")
        assert order is not None
        assert order.order_id == "12345"

    @pytest.mark.asyncio
    async def test_order_by_exchange_id(
        self, cache: Cache, sample_order_result: OrderResult
    ) -> None:
        """Test finding order by exchange ID."""
        await cache.add_order(sample_order_result)

        order = cache.order_by_exchange_id("12345")
        assert order is not None
        assert order.client_order_id == "client_123"

    @pytest.mark.asyncio
    async def test_orders_filter_by_symbol(
        self, cache: Cache, sample_order_result: OrderResult
    ) -> None:
        """Test filtering orders by symbol."""
        await cache.add_order(sample_order_result)

        orders = cache.orders(symbol="BTC/USDT")
        assert len(orders) == 1

        orders = cache.orders(symbol="ETH/USDT")
        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_orders_filter_by_status(
        self, cache: Cache, sample_order_result: OrderResult
    ) -> None:
        """Test filtering orders by status."""
        await cache.add_order(sample_order_result)

        orders = cache.orders(status=OrderStatus.FILLED)
        assert len(orders) == 1

        orders = cache.orders(status=OrderStatus.OPEN)
        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_open_orders(self, cache: Cache) -> None:
        """Test getting open orders."""
        # Add a filled order
        filled = OrderResult(
            order_id="1",
            symbol="BTC/USDT",
            status=OrderStatus.FILLED,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
            filled_amount=Decimal("0.1"),
            remaining_amount=Decimal("0"),
            average_price=Decimal("50000"),
            fee=Decimal("0"),
            fee_currency="BTC",
            timestamp_ns=0,
            client_order_id="filled_1",
        )
        await cache.add_order(filled)

        # Add an open order
        open_order = OrderResult(
            order_id="2",
            symbol="BTC/USDT",
            status=OrderStatus.OPEN,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.2"),
            filled_amount=Decimal("0"),
            remaining_amount=Decimal("0.2"),
            average_price=None,
            fee=Decimal("0"),
            fee_currency="BTC",
            timestamp_ns=0,
            client_order_id="open_1",
        )
        await cache.add_order(open_order)

        open_orders = cache.open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].order_id == "2"

    @pytest.mark.asyncio
    async def test_order_eviction(self, cache: Cache) -> None:
        """Test FIFO eviction when max orders exceeded."""
        cache = Cache(max_orders=5)

        for i in range(10):
            order = OrderResult(
                order_id=str(i),
                symbol="BTC/USDT",
                status=OrderStatus.FILLED,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("0.1"),
                filled_amount=Decimal("0.1"),
                remaining_amount=Decimal("0"),
                average_price=Decimal("50000"),
                fee=Decimal("0"),
                fee_currency="BTC",
                timestamp_ns=0,
                client_order_id=f"client_{i}",
            )
            await cache.add_order(order)

        # Should only have 5 orders (the last 5)
        all_orders = cache.orders()
        assert len(all_orders) == 5

        # First 5 should be evicted
        assert cache.order("client_0") is None
        assert cache.order("client_4") is None
        # Last 5 should exist
        assert cache.order("client_5") is not None
        assert cache.order("client_9") is not None


# =============================================================================
# Position Cache Tests
# =============================================================================


class TestPositionCache:
    """Tests for position caching."""

    @pytest.mark.asyncio
    async def test_update_position(
        self, cache: Cache, sample_position: Position
    ) -> None:
        """Test updating a position."""
        await cache.update_position(sample_position)

        pos = cache.position("BTC/USDT")
        assert pos is not None
        assert pos.amount == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_remove_position_on_zero_amount(self, cache: Cache) -> None:
        """Test that position is removed when amount is zero."""
        pos = Position(
            symbol="BTC/USDT",
            side=PositionSide.FLAT,
            amount=Decimal("0"),
            entry_price=Decimal("0"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("500"),
        )
        await cache.update_position(pos)

        assert cache.position("BTC/USDT") is None

    @pytest.mark.asyncio
    async def test_positions(self, cache: Cache) -> None:
        """Test getting all positions."""
        pos1 = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            amount=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("0"),
        )
        pos2 = Position(
            symbol="ETH/USDT",
            side=PositionSide.SHORT,
            amount=Decimal("2.0"),
            entry_price=Decimal("2000"),
            current_price=Decimal("1900"),
            unrealized_pnl=Decimal("200"),
            realized_pnl=Decimal("0"),
        )

        await cache.update_position(pos1)
        await cache.update_position(pos2)

        positions = cache.positions()
        assert len(positions) == 2

    @pytest.mark.asyncio
    async def test_has_position(self, cache: Cache, sample_position: Position) -> None:
        """Test checking if position exists."""
        assert cache.has_position("BTC/USDT") is False

        await cache.update_position(sample_position)
        assert cache.has_position("BTC/USDT") is True


# =============================================================================
# Quote Cache Tests
# =============================================================================


class TestQuoteCache:
    """Tests for quote caching."""

    @pytest.mark.asyncio
    async def test_update_quote(self, cache: Cache, sample_tick: Tick) -> None:
        """Test updating a quote."""
        await cache.update_quote(sample_tick)

        quote = cache.quote("BTC/USDT")
        assert quote is not None
        assert quote.last == Decimal("50000")

    @pytest.mark.asyncio
    async def test_quotes(self, cache: Cache) -> None:
        """Test getting all quotes."""
        tick1 = Tick(
            symbol="BTC/USDT",
            bid=Decimal("49999"),
            ask=Decimal("50001"),
            last=Decimal("50000"),
            timestamp_ns=1000000000,
        )
        tick2 = Tick(
            symbol="ETH/USDT",
            bid=Decimal("1999"),
            ask=Decimal("2001"),
            last=Decimal("2000"),
            timestamp_ns=1000000000,
        )

        await cache.update_quote(tick1)
        await cache.update_quote(tick2)

        quotes = cache.quotes()
        assert len(quotes) == 2

    @pytest.mark.asyncio
    async def test_price(self, cache: Cache, sample_tick: Tick) -> None:
        """Test getting last price."""
        assert cache.price("BTC/USDT") is None

        await cache.update_quote(sample_tick)
        assert cache.price("BTC/USDT") == Decimal("50000")

    @pytest.mark.asyncio
    async def test_mid_price(self, cache: Cache, sample_tick: Tick) -> None:
        """Test getting mid price."""
        await cache.update_quote(sample_tick)
        assert cache.mid_price("BTC/USDT") == Decimal("50000")


# =============================================================================
# Bar Cache Tests
# =============================================================================


class TestBarCache:
    """Tests for bar caching."""

    @pytest.mark.asyncio
    async def test_add_bar(self, cache: Cache, sample_bar: Bar) -> None:
        """Test adding a bar."""
        await cache.add_bar(sample_bar)

        bar = cache.bar("BTC/USDT", "1h")
        assert bar is not None
        assert bar.close == Decimal("50000")

    @pytest.mark.asyncio
    async def test_bars(self, cache: Cache) -> None:
        """Test getting multiple bars."""
        for i in range(5):
            bar = Bar(
                symbol="BTC/USDT",
                timestamp_ns=i * 3600 * 1_000_000_000,
                open=Decimal("49500"),
                high=Decimal("50500"),
                low=Decimal("49000"),
                close=Decimal(str(50000 + i * 100)),
                volume=Decimal("100"),
                timeframe="1h",
            )
            await cache.add_bar(bar)

        bars = cache.bars("BTC/USDT", "1h")
        assert len(bars) == 5

        bars = cache.bars("BTC/USDT", "1h", count=3)
        assert len(bars) == 3
        # Should be most recent 3
        assert bars[-1].close == Decimal("50400")

    @pytest.mark.asyncio
    async def test_bar_eviction(self, cache: Cache) -> None:
        """Test bar eviction when max exceeded."""
        cache = Cache(max_bars=5)

        for i in range(10):
            bar = Bar(
                symbol="BTC/USDT",
                timestamp_ns=i * 3600 * 1_000_000_000,
                open=Decimal("49500"),
                high=Decimal("50500"),
                low=Decimal("49000"),
                close=Decimal(str(50000 + i)),
                volume=Decimal("100"),
                timeframe="1h",
            )
            await cache.add_bar(bar)

        bars = cache.bars("BTC/USDT", "1h")
        assert len(bars) == 5
        # Should have most recent 5
        assert bars[0].close == Decimal("50005")
        assert bars[-1].close == Decimal("50009")


# =============================================================================
# Balance Cache Tests
# =============================================================================


class TestBalanceCache:
    """Tests for balance caching."""

    @pytest.mark.asyncio
    async def test_update_balance(self, cache: Cache, sample_balance: Balance) -> None:
        """Test updating a balance."""
        await cache.update_balance(sample_balance)

        balance = cache.balance("USDT")
        assert balance is not None
        assert balance.total == Decimal("10000")

    @pytest.mark.asyncio
    async def test_update_balances(self, cache: Cache) -> None:
        """Test updating multiple balances."""
        balances = {
            "USDT": Balance(
                currency="USDT",
                total=Decimal("10000"),
                available=Decimal("8000"),
                locked=Decimal("2000"),
            ),
            "BTC": Balance(
                currency="BTC",
                total=Decimal("1.5"),
                available=Decimal("1.0"),
                locked=Decimal("0.5"),
            ),
        }

        await cache.update_balances(balances)

        assert cache.balance("USDT") is not None
        assert cache.balance("BTC") is not None

    @pytest.mark.asyncio
    async def test_balances(self, cache: Cache) -> None:
        """Test getting all balances."""
        await cache.update_balance(
            Balance(currency="USDT", total=Decimal("10000"), available=Decimal("8000"), locked=Decimal("2000"))
        )
        await cache.update_balance(
            Balance(currency="BTC", total=Decimal("1.5"), available=Decimal("1.0"), locked=Decimal("0.5"))
        )

        balances = cache.balances()
        assert len(balances) == 2

    @pytest.mark.asyncio
    async def test_available_balance(self, cache: Cache, sample_balance: Balance) -> None:
        """Test getting available balance."""
        assert cache.available_balance("USDT") == Decimal("0")

        await cache.update_balance(sample_balance)
        assert cache.available_balance("USDT") == Decimal("8000")


# =============================================================================
# Utility Tests
# =============================================================================


class TestCacheUtility:
    """Tests for cache utility methods."""

    @pytest.mark.asyncio
    async def test_clear(self, cache: Cache, sample_order_result: OrderResult, sample_position: Position) -> None:
        """Test clearing all cache data."""
        await cache.add_order(sample_order_result)
        await cache.update_position(sample_position)

        await cache.clear()

        assert len(cache.orders()) == 0
        assert len(cache.positions()) == 0

    @pytest.mark.asyncio
    async def test_stats(self, cache: Cache) -> None:
        """Test cache statistics."""
        stats = cache.stats()

        assert "orders" in stats
        assert "positions" in stats
        assert "quotes" in stats
        assert "bar_series" in stats
        assert "balances" in stats


# =============================================================================
# RWLock Tests (Issue #90)
# =============================================================================


class TestRWLock:
    """Tests for RWLock implementation."""

    @pytest.mark.asyncio
    async def test_writer_exclusive(self) -> None:
        """Test that writers have exclusive access."""
        lock = RWLock()
        results = []

        async def writer(n: int) -> None:
            async with lock.writer:
                results.append(f"start_{n}")
                await asyncio.sleep(0.01)
                results.append(f"end_{n}")

        # Run two writers concurrently
        await asyncio.gather(writer(1), writer(2))

        # Writers should be serialized (start_1, end_1, start_2, end_2) or vice versa
        # Not interleaved (start_1, start_2, ...)
        assert results[0].startswith("start_")
        assert results[1].startswith("end_")
        assert results[0][6] == results[1][4]  # Same writer number

    @pytest.mark.asyncio
    async def test_readers_concurrent(self) -> None:
        """Test that multiple readers can run concurrently."""
        lock = RWLock()
        active_readers = []
        max_concurrent = 0

        async def reader(n: int) -> None:
            nonlocal max_concurrent
            async with lock.reader:
                active_readers.append(n)
                max_concurrent = max(max_concurrent, len(active_readers))
                await asyncio.sleep(0.02)
                active_readers.remove(n)

        # Run 5 readers concurrently
        await asyncio.gather(*[reader(i) for i in range(5)])

        # At some point, multiple readers should have been active
        assert max_concurrent > 1

    @pytest.mark.asyncio
    async def test_writer_waits_for_readers(self) -> None:
        """Test that writer waits for active readers to finish."""
        lock = RWLock()
        events = []

        async def reader() -> None:
            async with lock.reader:
                events.append("reader_start")
                await asyncio.sleep(0.05)
                events.append("reader_end")

        async def writer() -> None:
            await asyncio.sleep(0.01)  # Let reader start first
            async with lock.writer:
                events.append("writer")

        await asyncio.gather(reader(), writer())

        # Writer should wait for reader to finish
        assert events == ["reader_start", "reader_end", "writer"]

    @pytest.mark.asyncio
    async def test_reader_count(self) -> None:
        """Test reader_count property."""
        lock = RWLock()
        assert lock.reader_count == 0

        async with lock.reader:
            assert lock.reader_count == 1

        assert lock.reader_count == 0

    @pytest.mark.asyncio
    async def test_is_write_locked(self) -> None:
        """Test is_write_locked property."""
        lock = RWLock()
        assert lock.is_write_locked is False

        async with lock.writer:
            assert lock.is_write_locked is True

        assert lock.is_write_locked is False


class TestCacheRWLockIntegration:
    """Integration tests for Cache with RWLock."""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, cache: Cache) -> None:
        """Test concurrent read operations."""
        # Add some data
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50001"),
            last=Decimal("50000"),
            timestamp_ns=1000,
        )
        await cache.update_quote(tick)

        read_count = 0

        async def read_task() -> None:
            nonlocal read_count
            for _ in range(100):
                _ = cache.quote("BTC/USDT")
                read_count += 1

        # Run 10 concurrent readers
        await asyncio.gather(*[read_task() for _ in range(10)])

        assert read_count == 1000

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, cache: Cache) -> None:
        """Test concurrent write operations."""
        async def write_task(n: int) -> None:
            tick = Tick(
                symbol=f"SYM_{n}/USDT",
                bid=Decimal("100"),
                ask=Decimal("101"),
                last=Decimal("100"),
                timestamp_ns=n,
            )
            await cache.update_quote(tick)

        # Run 50 concurrent writers
        await asyncio.gather(*[write_task(i) for i in range(50)])

        # All writes should succeed
        quotes = cache.quotes()
        assert len(quotes) == 50

    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, cache: Cache) -> None:
        """Test concurrent read and write operations."""
        # Add initial data
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50001"),
            last=Decimal("50000"),
            timestamp_ns=1000,
        )
        await cache.update_quote(tick)

        errors = []

        async def reader() -> None:
            try:
                for _ in range(100):
                    quote = cache.quote("BTC/USDT")
                    if quote is not None:
                        _ = quote.last  # Access a property
                    await asyncio.sleep(0)
            except Exception as e:
                errors.append(str(e))

        async def writer() -> None:
            try:
                for i in range(100):
                    tick = Tick(
                        symbol="BTC/USDT",
                        bid=Decimal(f"{50000 + i}"),
                        ask=Decimal(f"{50001 + i}"),
                        last=Decimal(f"{50000 + i}"),
                        timestamp_ns=1000 + i,
                    )
                    await cache.update_quote(tick)
                    await asyncio.sleep(0)
            except Exception as e:
                errors.append(str(e))

        # Run readers and writers concurrently
        await asyncio.gather(
            *[reader() for _ in range(5)],
            *[writer() for _ in range(3)],
        )

        assert len(errors) == 0, f"Errors: {errors}"

    @pytest.mark.asyncio
    async def test_cache_uses_single_lock(self, cache: Cache) -> None:
        """Test that cache uses single RWLock instead of 5 separate locks."""
        # Verify the cache has _rwlock attribute
        assert hasattr(cache, "_rwlock")
        assert isinstance(cache._rwlock, RWLock)

        # Verify old lock attributes don't exist
        assert not hasattr(cache, "_order_lock")
        assert not hasattr(cache, "_position_lock")
        assert not hasattr(cache, "_quote_lock")
        assert not hasattr(cache, "_bar_lock")
        assert not hasattr(cache, "_balance_lock")


class TestRWLockPerformance:
    """Performance tests for RWLock (Issue #90)."""

    @pytest.mark.asyncio
    async def test_lock_acquisition_overhead(self) -> None:
        """Benchmark lock acquisition overhead."""
        lock = RWLock()

        # Measure reader lock acquisition
        start = time.perf_counter()
        for _ in range(1000):
            async with lock.reader:
                pass
        reader_time = time.perf_counter() - start

        # Measure writer lock acquisition
        start = time.perf_counter()
        for _ in range(1000):
            async with lock.writer:
                pass
        writer_time = time.perf_counter() - start

        # Should complete reasonably fast (< 1 second for 1000 iterations)
        assert reader_time < 1.0, f"Reader time: {reader_time:.3f}s"
        assert writer_time < 1.0, f"Writer time: {writer_time:.3f}s"

        print(f"Reader lock avg: {reader_time * 1000:.3f}ms per 1000 ops")
        print(f"Writer lock avg: {writer_time * 1000:.3f}ms per 1000 ops")
