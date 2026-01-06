"""Unit tests for Cache component."""

from __future__ import annotations

from decimal import Decimal

import pytest

from libra.core.cache import Cache
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
