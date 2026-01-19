"""
Tests for Paper Trading Gateway.

Tests:
- Order simulation (market and limit)
- Fill modeling with slippage
- Position tracking and P&L calculation
- Balance management
- Error handling
"""

from decimal import Decimal

import pytest

from libra.gateways import (
    InsufficientFundsError,
    Order,
    OrderNotFoundError,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperGateway,
    PositionSide,
)


@pytest.fixture
def paper_gateway() -> PaperGateway:
    """Create a paper gateway with test config."""
    config = {
        "initial_balance": {
            "USDT": Decimal("10000"),
            "BTC": Decimal("0.5"),
        },
        "slippage_model": "fixed",
        "slippage_bps": 5,
        "maker_fee_bps": 1,
        "taker_fee_bps": 5,
    }
    return PaperGateway(config)


@pytest.fixture
async def connected_gateway(paper_gateway: PaperGateway) -> PaperGateway:
    """Create a connected paper gateway."""
    await paper_gateway.connect()
    # Set initial prices
    paper_gateway.update_price("BTC/USDT", Decimal("50000"), Decimal("50010"))
    paper_gateway.update_price("ETH/USDT", Decimal("2000"), Decimal("2002"))
    return paper_gateway


class TestPaperGatewayConnection:
    """Tests for connection management."""

    @pytest.mark.asyncio
    async def test_connect(self, paper_gateway: PaperGateway) -> None:
        """Test connecting to paper gateway."""
        assert not paper_gateway.is_connected

        await paper_gateway.connect()

        assert paper_gateway.is_connected

    @pytest.mark.asyncio
    async def test_disconnect(self, paper_gateway: PaperGateway) -> None:
        """Test disconnecting from paper gateway."""
        await paper_gateway.connect()
        assert paper_gateway.is_connected

        await paper_gateway.disconnect()

        assert not paper_gateway.is_connected

    @pytest.mark.asyncio
    async def test_context_manager(self, paper_gateway: PaperGateway) -> None:
        """Test async context manager."""
        async with paper_gateway:
            assert paper_gateway.is_connected

        assert not paper_gateway.is_connected


class TestMarketOrders:
    """Tests for market order execution."""

    @pytest.mark.asyncio
    async def test_market_buy_order(self, connected_gateway: PaperGateway) -> None:
        """Test executing a market buy order."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )

        result = await connected_gateway.submit_order(order)

        assert result.status == OrderStatus.FILLED
        assert result.filled_amount == Decimal("0.1")
        assert result.remaining_amount == Decimal("0")
        assert result.average_price is not None
        # Price should include slippage (buying at ask + slippage)
        assert result.average_price > Decimal("50010")

    @pytest.mark.asyncio
    async def test_market_sell_order(self, connected_gateway: PaperGateway) -> None:
        """Test executing a market sell order."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )

        result = await connected_gateway.submit_order(order)

        assert result.status == OrderStatus.FILLED
        assert result.filled_amount == Decimal("0.1")
        # Price should include slippage (selling at bid - slippage)
        assert result.average_price is not None
        assert result.average_price < Decimal("50000")

    @pytest.mark.asyncio
    async def test_market_order_updates_balance(self, connected_gateway: PaperGateway) -> None:
        """Test that market order updates balances correctly."""
        initial_usdt = await connected_gateway.get_balance("USDT")
        assert initial_usdt is not None
        initial_usdt_amount = initial_usdt.total

        initial_btc = await connected_gateway.get_balance("BTC")
        assert initial_btc is not None
        initial_btc_amount = initial_btc.total

        # Buy BTC
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )
        await connected_gateway.submit_order(order)

        # Check balances changed
        new_usdt = await connected_gateway.get_balance("USDT")
        new_btc = await connected_gateway.get_balance("BTC")

        assert new_usdt is not None
        assert new_btc is not None
        assert new_usdt.total < initial_usdt_amount  # USDT decreased
        assert new_btc.total > initial_btc_amount  # BTC increased
        assert new_btc.total == initial_btc_amount + Decimal("0.1")

    @pytest.mark.asyncio
    async def test_insufficient_funds_buy(self, connected_gateway: PaperGateway) -> None:
        """Test that buy order fails with insufficient funds."""
        # Try to buy more BTC than we can afford
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1000"),  # Would cost 50M USDT
        )

        with pytest.raises(InsufficientFundsError):
            await connected_gateway.submit_order(order)

    @pytest.mark.asyncio
    async def test_insufficient_funds_sell(self, connected_gateway: PaperGateway) -> None:
        """Test that sell order fails with insufficient funds."""
        # Try to sell more BTC than we have
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),  # Only have 0.5 BTC
        )

        with pytest.raises(InsufficientFundsError):
            await connected_gateway.submit_order(order)


class TestLimitOrders:
    """Tests for limit order execution."""

    @pytest.mark.asyncio
    async def test_limit_buy_order_queued(self, connected_gateway: PaperGateway) -> None:
        """Test that limit buy order is queued (not immediately filled)."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("49000"),  # Below current ask
        )

        result = await connected_gateway.submit_order(order)

        assert result.status == OrderStatus.OPEN
        assert result.filled_amount == Decimal("0")

    @pytest.mark.asyncio
    async def test_limit_order_fills_when_price_crosses(
        self, connected_gateway: PaperGateway
    ) -> None:
        """Test that limit order fills when price crosses."""
        # Place limit buy below market
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("49500"),
        )

        result = await connected_gateway.submit_order(order)
        assert result.status == OrderStatus.OPEN

        # Update price to cross the limit
        connected_gateway.update_price("BTC/USDT", Decimal("49400"), Decimal("49500"))

        # Wait for async task to process
        import asyncio

        await asyncio.sleep(0.1)

        # Check order is now filled
        updated_result = await connected_gateway.get_order(result.order_id, "BTC/USDT")
        assert updated_result.status == OrderStatus.FILLED
        assert updated_result.average_price == Decimal("49500")

    @pytest.mark.asyncio
    async def test_limit_order_locks_funds(self, connected_gateway: PaperGateway) -> None:
        """Test that limit order locks funds."""
        initial_balance = await connected_gateway.get_balance("USDT")
        assert initial_balance is not None

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("49000"),  # Locks 4900 USDT
        )

        await connected_gateway.submit_order(order)

        new_balance = await connected_gateway.get_balance("USDT")
        assert new_balance is not None
        assert new_balance.locked == Decimal("4900")
        assert new_balance.available == initial_balance.total - Decimal("4900")

    @pytest.mark.asyncio
    async def test_cancel_limit_order_unlocks_funds(self, connected_gateway: PaperGateway) -> None:
        """Test that cancelling limit order unlocks funds."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("49000"),
        )

        result = await connected_gateway.submit_order(order)

        # Cancel the order
        cancelled = await connected_gateway.cancel_order(result.order_id, "BTC/USDT")
        assert cancelled

        # Check funds unlocked
        balance = await connected_gateway.get_balance("USDT")
        assert balance is not None
        assert balance.locked == Decimal("0")


class TestPositionTracking:
    """Tests for position tracking."""

    @pytest.mark.asyncio
    async def test_position_created_on_buy(self, connected_gateway: PaperGateway) -> None:
        """Test that position is created after buy."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )

        await connected_gateway.submit_order(order)

        position = await connected_gateway.get_position("BTC/USDT")
        assert position is not None
        assert position.side == PositionSide.LONG
        assert position.amount == Decimal("0.1")

    @pytest.mark.asyncio
    async def test_position_increases_on_additional_buy(
        self, connected_gateway: PaperGateway
    ) -> None:
        """Test that position increases with additional buys."""
        # First buy (small amount to fit within 10000 USDT balance)
        order1 = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.05"),
        )
        await connected_gateway.submit_order(order1)

        # Second buy
        order2 = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.05"),
        )
        await connected_gateway.submit_order(order2)

        position = await connected_gateway.get_position("BTC/USDT")
        assert position is not None
        assert position.amount == Decimal("0.1")

    @pytest.mark.asyncio
    async def test_position_closed_on_sell(self, connected_gateway: PaperGateway) -> None:
        """Test that position is closed when fully sold."""
        # Buy
        buy_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )
        await connected_gateway.submit_order(buy_order)

        # Sell same amount
        sell_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )
        await connected_gateway.submit_order(sell_order)

        position = await connected_gateway.get_position("BTC/USDT")
        assert position is None  # Position closed

    @pytest.mark.asyncio
    async def test_unrealized_pnl_calculation(self, connected_gateway: PaperGateway) -> None:
        """Test unrealized P&L calculation."""
        # Buy at current price (~50005) - use small amount to fit in balance
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )
        await connected_gateway.submit_order(order)

        # Update price higher
        connected_gateway.update_price("BTC/USDT", Decimal("51000"), Decimal("51010"))

        position = await connected_gateway.get_position("BTC/USDT")
        assert position is not None
        assert position.unrealized_pnl > Decimal("0")  # Profit


class TestSlippageModels:
    """Tests for slippage simulation."""

    @pytest.mark.asyncio
    async def test_no_slippage_model(self) -> None:
        """Test no slippage model."""
        config = {
            "initial_balance": {"USDT": Decimal("100000")},
            "slippage_model": "none",
        }
        gateway = PaperGateway(config)
        await gateway.connect()
        gateway.update_price("BTC/USDT", Decimal("50000"), Decimal("50010"))

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )
        result = await gateway.submit_order(order)

        # Should fill exactly at ask price
        assert result.average_price == Decimal("50010")

    @pytest.mark.asyncio
    async def test_fixed_slippage_model(self) -> None:
        """Test fixed slippage model."""
        config = {
            "initial_balance": {"USDT": Decimal("100000")},
            "slippage_model": "fixed",
            "slippage_bps": 10,  # 10 bps = 0.1%
        }
        gateway = PaperGateway(config)
        await gateway.connect()
        gateway.update_price("BTC/USDT", Decimal("50000"), Decimal("50000"))

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )
        result = await gateway.submit_order(order)

        # Should be ask price * (1 + 0.001) = 50050
        assert result.average_price == Decimal("50050")

    @pytest.mark.asyncio
    async def test_volume_slippage_model(self) -> None:
        """Test volume-based slippage model."""
        config = {
            "initial_balance": {"USDT": Decimal("10000000")},
            "slippage_model": "volume",
        }
        gateway = PaperGateway(config)
        await gateway.connect()
        gateway.update_price("BTC/USDT", Decimal("50000"), Decimal("50000"))

        # Small order - minimal slippage
        small_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.01"),
        )
        small_result = await gateway.submit_order(small_order)

        # Large order - more slippage
        large_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("10.0"),
        )
        large_result = await gateway.submit_order(large_order)

        # Larger order should have more slippage
        assert large_result.average_price > small_result.average_price


class TestOrderManagement:
    """Tests for order management."""

    @pytest.mark.asyncio
    async def test_get_open_orders(self, connected_gateway: PaperGateway) -> None:
        """Test getting open orders."""
        # Place multiple limit orders
        for i in range(3):
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                amount=Decimal("0.01"),
                price=Decimal("45000") - Decimal(str(i * 100)),
            )
            await connected_gateway.submit_order(order)

        open_orders = await connected_gateway.get_open_orders()
        assert len(open_orders) == 3

    @pytest.mark.asyncio
    async def test_get_open_orders_by_symbol(self, connected_gateway: PaperGateway) -> None:
        """Test getting open orders filtered by symbol."""
        # BTC order
        btc_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.01"),
            price=Decimal("45000"),
        )
        await connected_gateway.submit_order(btc_order)

        # ETH order
        eth_order = Order(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("1800"),
        )
        await connected_gateway.submit_order(eth_order)

        btc_orders = await connected_gateway.get_open_orders("BTC/USDT")
        eth_orders = await connected_gateway.get_open_orders("ETH/USDT")

        assert len(btc_orders) == 1
        assert len(eth_orders) == 1

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, connected_gateway: PaperGateway) -> None:
        """Test cancelling all orders."""
        # Place multiple orders
        for i in range(3):
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                amount=Decimal("0.01"),
                price=Decimal("45000") - Decimal(str(i * 100)),
            )
            await connected_gateway.submit_order(order)

        cancelled = await connected_gateway.cancel_all_orders()
        assert cancelled == 3

        open_orders = await connected_gateway.get_open_orders()
        assert len(open_orders) == 0

    @pytest.mark.asyncio
    async def test_get_nonexistent_order(self, connected_gateway: PaperGateway) -> None:
        """Test getting non-existent order raises error."""
        with pytest.raises(OrderNotFoundError):
            await connected_gateway.get_order("nonexistent-id", "BTC/USDT")


class TestBalanceManagement:
    """Tests for balance management."""

    @pytest.mark.asyncio
    async def test_get_all_balances(self, connected_gateway: PaperGateway) -> None:
        """Test getting all balances."""
        balances = await connected_gateway.get_balances()

        assert "USDT" in balances
        assert "BTC" in balances
        assert balances["USDT"].total == Decimal("10000")
        assert balances["BTC"].total == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_get_single_balance(self, connected_gateway: PaperGateway) -> None:
        """Test getting single balance."""
        balance = await connected_gateway.get_balance("USDT")

        assert balance is not None
        assert balance.currency == "USDT"
        assert balance.total == Decimal("10000")

    @pytest.mark.asyncio
    async def test_get_nonexistent_balance(self, connected_gateway: PaperGateway) -> None:
        """Test getting non-existent balance returns None."""
        balance = await connected_gateway.get_balance("DOGE")
        assert balance is None


class TestPaperGatewayUtilities:
    """Tests for paper gateway utility methods."""

    @pytest.mark.asyncio
    async def test_reset(self, connected_gateway: PaperGateway) -> None:
        """Test resetting gateway state."""
        # Make some trades
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )
        await connected_gateway.submit_order(order)

        # Reset
        connected_gateway.reset()

        # Check state is cleared
        positions = await connected_gateway.get_positions()
        assert len(positions) == 0

        balance = await connected_gateway.get_balance("USDT")
        assert balance is not None
        assert balance.total == Decimal("10000")

    @pytest.mark.asyncio
    async def test_get_total_equity(self, connected_gateway: PaperGateway) -> None:
        """Test calculating total equity."""
        # Initial equity should be ~10000 + 0.5 * 50000 = 35000
        equity = connected_gateway.get_total_equity()
        assert equity > Decimal("34000")
        assert equity < Decimal("36000")

    @pytest.mark.asyncio
    async def test_get_trade_history(self, connected_gateway: PaperGateway) -> None:
        """Test getting trade history."""
        # Execute some trades
        for _ in range(3):
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("0.01"),
            )
            await connected_gateway.submit_order(order)

        history = connected_gateway.get_trade_history()
        assert len(history) == 3
        assert all("price" in trade for trade in history)
        assert all("amount" in trade for trade in history)


class TestBackpressureHandling:
    """Tests for tick queue backpressure handling (Issue #79)."""

    @pytest.mark.asyncio
    async def test_dropped_ticks_counter_initial(self) -> None:
        """Test dropped ticks counter starts at zero."""
        gateway = PaperGateway()
        await gateway.connect()

        assert gateway.dropped_ticks == 0

    @pytest.mark.asyncio
    async def test_dropped_ticks_on_queue_full(self) -> None:
        """Test dropped ticks counter increments when queue is full."""
        # Create gateway with very small queue for testing
        gateway = PaperGateway()
        await gateway.connect()
        gateway._tick_queue._maxsize = 2  # Override queue size for test

        # Fill the queue
        gateway.update_price("BTC/USDT", Decimal("50000"), Decimal("50010"))
        gateway.update_price("BTC/USDT", Decimal("50001"), Decimal("50011"))

        # This should trigger backpressure (queue full)
        gateway.update_price("BTC/USDT", Decimal("50002"), Decimal("50012"))

        # Should have dropped one tick
        assert gateway.dropped_ticks == 1

    @pytest.mark.asyncio
    async def test_dropped_ticks_accumulates(self) -> None:
        """Test dropped ticks counter accumulates over time."""
        gateway = PaperGateway()
        await gateway.connect()
        gateway._tick_queue._maxsize = 1  # Very small queue

        # Multiple updates that will cause drops
        for i in range(5):
            gateway.update_price("BTC/USDT", Decimal(50000 + i), Decimal(50010 + i))

        # Should have dropped 4 ticks (first goes in, rest trigger backpressure)
        assert gateway.dropped_ticks == 4
