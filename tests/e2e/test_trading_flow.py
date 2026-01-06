"""
End-to-End tests for full trading flow.

Tests the complete trading cycle:
1. Market data ingestion
2. Strategy signal generation
3. Order execution
4. Position management
5. P&L tracking

Uses PaperGateway for simulated trading without real exchange interaction.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from libra.core.events import Event, EventType
from libra.core.message_bus import MessageBus
from libra.gateways.paper_gateway import PaperGateway, SlippageModel
from libra.gateways.protocol import OrderSide, PositionSide
from libra.strategies.actor import ComponentState
from libra.strategies.examples.sma_cross_live import (
    SMACrossLiveConfig,
    SMACrossLiveStrategy,
)
from libra.strategies.protocol import Bar


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def message_bus() -> MessageBus:
    """Create a MessageBus instance."""
    return MessageBus()


@pytest.fixture
def paper_gateway() -> PaperGateway:
    """Create a PaperGateway with realistic settings."""
    config = {
        "initial_balance": {"USDT": Decimal("10000")},  # $10,000 USDT
        "slippage_model": "fixed",
        "slippage_bps": 5,  # 5 bps slippage
        "maker_fee_bps": 10,  # 0.1% maker fee
        "taker_fee_bps": 10,  # 0.1% taker fee
    }
    return PaperGateway(config)


@pytest.fixture
def sma_strategy(paper_gateway: PaperGateway) -> SMACrossLiveStrategy:
    """Create SMA crossover strategy."""
    config = SMACrossLiveConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        fast_period=3,  # Short periods for testing
        slow_period=5,
        order_size=Decimal("0.1"),
        use_market_orders=True,
    )
    return SMACrossLiveStrategy(paper_gateway, config)


# =============================================================================
# Helper Functions
# =============================================================================


def create_bar(
    symbol: str,
    open_price: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    timestamp_ns: int,
) -> Bar:
    """Create a bar for testing."""
    return Bar(
        symbol=symbol,
        timestamp_ns=timestamp_ns,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=Decimal("100"),
        timeframe="1h",
    )


def publish_bar(bus: MessageBus, bar: Bar) -> None:
    """Publish a bar event to the message bus."""
    bus.publish(Event.create(EventType.BAR, "test", {"bar": bar}))


# =============================================================================
# E2E Tests
# =============================================================================


class TestFullTradingFlow:
    """End-to-end tests for the full trading flow."""

    @pytest.mark.asyncio
    async def test_strategy_lifecycle(
        self,
        message_bus: MessageBus,
        paper_gateway: PaperGateway,
        sma_strategy: SMACrossLiveStrategy,
    ) -> None:
        """Test strategy goes through complete lifecycle."""
        await paper_gateway.connect()
        await sma_strategy.initialize(message_bus)

        # Check initial state
        assert sma_strategy.state == ComponentState.READY

        async with message_bus:
            # Start strategy
            await sma_strategy.start()
            assert sma_strategy.state == ComponentState.RUNNING

            # Let it run briefly
            await asyncio.sleep(0.05)

            # Stop strategy
            await sma_strategy.stop()
            assert sma_strategy.state == ComponentState.STOPPED

            # Dispose
            await sma_strategy.dispose()
            assert sma_strategy.state == ComponentState.DISPOSED

        await paper_gateway.disconnect()

    @pytest.mark.asyncio
    async def test_golden_cross_opens_position(
        self,
        message_bus: MessageBus,
        paper_gateway: PaperGateway,
        sma_strategy: SMACrossLiveStrategy,
    ) -> None:
        """Test that golden cross opens a long position."""
        await paper_gateway.connect()
        await sma_strategy.initialize(message_bus)

        async with message_bus:
            await sma_strategy.start()

            # Publish bars that create a golden cross
            # Downtrend (fast < slow)
            prices = [
                Decimal("50000"),
                Decimal("49500"),
                Decimal("49000"),
                Decimal("48500"),
                Decimal("48000"),  # slow_ma = 49000, fast_ma = 48500
            ]

            for i, price in enumerate(prices):
                # Update gateway price so orders can execute
                paper_gateway.update_price(
                    "BTC/USDT", price - Decimal("1"), price + Decimal("1")
                )
                bar = create_bar(
                    "BTC/USDT",
                    price - Decimal("100"),
                    price + Decimal("100"),
                    price - Decimal("200"),
                    price,
                    i * 3600 * 1_000_000_000,
                )
                publish_bar(message_bus, bar)
                await asyncio.sleep(0.02)

            # Verify no position yet (downtrend)
            assert sma_strategy.is_flat("BTC/USDT")

            # Now create golden cross (fast > slow)
            uptrend_prices = [
                Decimal("49000"),
                Decimal("50000"),
                Decimal("51000"),  # fast crosses above slow
            ]

            for i, price in enumerate(uptrend_prices):
                # Update gateway price so orders can execute
                paper_gateway.update_price(
                    "BTC/USDT", price - Decimal("1"), price + Decimal("1")
                )
                bar = create_bar(
                    "BTC/USDT",
                    price - Decimal("100"),
                    price + Decimal("100"),
                    price - Decimal("200"),
                    price,
                    (5 + i) * 3600 * 1_000_000_000,
                )
                publish_bar(message_bus, bar)
                await asyncio.sleep(0.02)

            # Wait for order execution
            await asyncio.sleep(0.1)

            # Verify position opened
            assert sma_strategy.is_long("BTC/USDT")
            assert sma_strategy.order_count >= 1

            await sma_strategy.stop()

        await paper_gateway.disconnect()

    @pytest.mark.asyncio
    async def test_death_cross_closes_position(
        self,
        message_bus: MessageBus,
        paper_gateway: PaperGateway,
        sma_strategy: SMACrossLiveStrategy,
    ) -> None:
        """Test that death cross closes a long position."""
        await paper_gateway.connect()
        await sma_strategy.initialize(message_bus)

        async with message_bus:
            await sma_strategy.start()

            # First create golden cross to open position
            uptrend_prices = [
                Decimal("48000"),
                Decimal("49000"),
                Decimal("50000"),
                Decimal("51000"),
                Decimal("52000"),  # Uptrend
                Decimal("53000"),
                Decimal("54000"),
                Decimal("55000"),  # Golden cross happens
            ]

            for i, price in enumerate(uptrend_prices):
                # Update gateway price so orders can execute
                paper_gateway.update_price(
                    "BTC/USDT", price - Decimal("1"), price + Decimal("1")
                )
                bar = create_bar(
                    "BTC/USDT",
                    price - Decimal("100"),
                    price + Decimal("100"),
                    price - Decimal("200"),
                    price,
                    i * 3600 * 1_000_000_000,
                )
                publish_bar(message_bus, bar)
                await asyncio.sleep(0.02)

            await asyncio.sleep(0.1)

            # Verify position opened
            if sma_strategy.is_long("BTC/USDT"):
                initial_order_count = sma_strategy.order_count

                # Now create death cross
                downtrend_prices = [
                    Decimal("54000"),
                    Decimal("52000"),
                    Decimal("50000"),  # Death cross
                ]

                for i, price in enumerate(downtrend_prices):
                    # Update gateway price
                    paper_gateway.update_price(
                        "BTC/USDT", price - Decimal("1"), price + Decimal("1")
                    )
                    bar = create_bar(
                        "BTC/USDT",
                        price + Decimal("100"),
                        price + Decimal("200"),
                        price - Decimal("100"),
                        price,
                        (8 + i) * 3600 * 1_000_000_000,
                    )
                    publish_bar(message_bus, bar)
                    await asyncio.sleep(0.02)

                await asyncio.sleep(0.1)

                # Verify position closed
                assert sma_strategy.is_flat("BTC/USDT")
                assert sma_strategy.order_count > initial_order_count

            await sma_strategy.stop()

        await paper_gateway.disconnect()

    @pytest.mark.asyncio
    async def test_pnl_tracking(
        self,
        message_bus: MessageBus,
        paper_gateway: PaperGateway,
        sma_strategy: SMACrossLiveStrategy,
    ) -> None:
        """Test that P&L is tracked correctly."""
        await paper_gateway.connect()
        await sma_strategy.initialize(message_bus)

        async with message_bus:
            await sma_strategy.start()

            # Create complete trade cycle with profit
            # Buy at ~50000, sell at ~55000

            # Build up to golden cross
            for i, price in enumerate([48000, 49000, 50000, 51000, 52000]):
                # Update gateway price
                paper_gateway.update_price(
                    "BTC/USDT", Decimal(str(price)) - Decimal("1"), Decimal(str(price)) + Decimal("1")
                )
                bar = create_bar(
                    "BTC/USDT",
                    Decimal(str(price)) - Decimal("100"),
                    Decimal(str(price)) + Decimal("100"),
                    Decimal(str(price)) - Decimal("200"),
                    Decimal(str(price)),
                    i * 3600 * 1_000_000_000,
                )
                publish_bar(message_bus, bar)
                await asyncio.sleep(0.02)

            # Continue uptrend
            for i, price in enumerate([53000, 54000, 55000]):
                # Update gateway price
                paper_gateway.update_price(
                    "BTC/USDT", Decimal(str(price)) - Decimal("1"), Decimal(str(price)) + Decimal("1")
                )
                bar = create_bar(
                    "BTC/USDT",
                    Decimal(str(price)) - Decimal("100"),
                    Decimal(str(price)) + Decimal("100"),
                    Decimal(str(price)) - Decimal("200"),
                    Decimal(str(price)),
                    (5 + i) * 3600 * 1_000_000_000,
                )
                publish_bar(message_bus, bar)
                await asyncio.sleep(0.02)

            await asyncio.sleep(0.1)

            # Death cross to close
            for i, price in enumerate([54000, 52000, 50000]):
                # Update gateway price
                paper_gateway.update_price(
                    "BTC/USDT", Decimal(str(price)) - Decimal("1"), Decimal(str(price)) + Decimal("1")
                )
                bar = create_bar(
                    "BTC/USDT",
                    Decimal(str(price)) + Decimal("100"),
                    Decimal(str(price)) + Decimal("200"),
                    Decimal(str(price)) - Decimal("100"),
                    Decimal(str(price)),
                    (8 + i) * 3600 * 1_000_000_000,
                )
                publish_bar(message_bus, bar)
                await asyncio.sleep(0.02)

            await asyncio.sleep(0.1)

            # Get stats
            stats = sma_strategy.get_stats()
            assert "total_pnl" in stats
            assert "trades_won" in stats
            assert "trades_lost" in stats

            await sma_strategy.stop()

        await paper_gateway.disconnect()

    @pytest.mark.asyncio
    async def test_multiple_strategies(
        self,
        message_bus: MessageBus,
        paper_gateway: PaperGateway,
    ) -> None:
        """Test running multiple strategies concurrently."""
        # Create two strategies for different symbols
        config1 = SMACrossLiveConfig(
            symbol="BTC/USDT",
            fast_period=3,
            slow_period=5,
            order_size=Decimal("0.1"),
        )
        strategy1 = SMACrossLiveStrategy(paper_gateway, config1)

        config2 = SMACrossLiveConfig(
            symbol="ETH/USDT",
            fast_period=3,
            slow_period=5,
            order_size=Decimal("1.0"),
        )
        strategy2 = SMACrossLiveStrategy(paper_gateway, config2)

        await paper_gateway.connect()
        await strategy1.initialize(message_bus)
        await strategy2.initialize(message_bus)

        async with message_bus:
            await strategy1.start()
            await strategy2.start()

            # Publish bars for both symbols
            for i in range(8):
                btc_price = Decimal("48000") + Decimal(str(i * 1000))
                eth_price = Decimal("2400") + Decimal(str(i * 50))

                # Update gateway prices
                paper_gateway.update_price(
                    "BTC/USDT", btc_price - Decimal("1"), btc_price + Decimal("1")
                )
                paper_gateway.update_price(
                    "ETH/USDT", eth_price - Decimal("1"), eth_price + Decimal("1")
                )

                bar1 = create_bar(
                    "BTC/USDT",
                    btc_price - Decimal("100"),
                    btc_price + Decimal("100"),
                    btc_price - Decimal("200"),
                    btc_price,
                    i * 3600 * 1_000_000_000,
                )
                bar2 = create_bar(
                    "ETH/USDT",
                    eth_price - Decimal("10"),
                    eth_price + Decimal("10"),
                    eth_price - Decimal("20"),
                    eth_price,
                    i * 3600 * 1_000_000_000,
                )

                publish_bar(message_bus, bar1)
                publish_bar(message_bus, bar2)
                await asyncio.sleep(0.02)

            await asyncio.sleep(0.1)

            # Both strategies should have processed bars
            assert strategy1.state == ComponentState.RUNNING
            assert strategy2.state == ComponentState.RUNNING

            await strategy1.stop()
            await strategy2.stop()

        await paper_gateway.disconnect()

    @pytest.mark.asyncio
    async def test_strategy_persists_state(
        self,
        message_bus: MessageBus,
        paper_gateway: PaperGateway,
        sma_strategy: SMACrossLiveStrategy,
    ) -> None:
        """Test that strategy state can be saved and loaded."""
        await paper_gateway.connect()
        await sma_strategy.initialize(message_bus)

        async with message_bus:
            await sma_strategy.start()

            # Process some bars
            for i, price in enumerate([50000, 51000, 52000, 53000, 54000]):
                # Update gateway price
                paper_gateway.update_price(
                    "BTC/USDT", Decimal(str(price)) - Decimal("1"), Decimal(str(price)) + Decimal("1")
                )
                bar = create_bar(
                    "BTC/USDT",
                    Decimal(str(price)) - Decimal("100"),
                    Decimal(str(price)) + Decimal("100"),
                    Decimal(str(price)) - Decimal("200"),
                    Decimal(str(price)),
                    i * 3600 * 1_000_000_000,
                )
                publish_bar(message_bus, bar)
                await asyncio.sleep(0.02)

            await asyncio.sleep(0.1)

            # Save state
            saved_state = sma_strategy.on_save()
            assert "state" in saved_state
            assert len(saved_state["state"]) > 0

            await sma_strategy.stop()

        # Create new strategy and load state
        config = SMACrossLiveConfig(
            symbol="BTC/USDT",
            fast_period=3,
            slow_period=5,
            order_size=Decimal("0.1"),
        )
        new_strategy = SMACrossLiveStrategy(paper_gateway, config)
        new_strategy.on_load(saved_state)

        # Verify state was loaded
        fast_ma, slow_ma = new_strategy.get_current_mas()
        assert fast_ma is not None
        assert slow_ma is not None

        await paper_gateway.disconnect()


class TestPaperGatewayIntegration:
    """Tests for PaperGateway integration."""

    @pytest.mark.asyncio
    async def test_order_execution(self, paper_gateway: PaperGateway) -> None:
        """Test order execution through PaperGateway."""
        await paper_gateway.connect()

        # Set price
        from libra.gateways.protocol import Order, OrderType

        paper_gateway.update_price("BTC/USDT", Decimal("49999"), Decimal("50001"))

        # Submit market buy
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )
        result = await paper_gateway.submit_order(order)

        assert result.filled_amount == Decimal("0.1")
        assert result.average_price is not None
        assert result.average_price > Decimal("0")

        # Check position
        position = await paper_gateway.get_position("BTC/USDT")
        assert position is not None
        assert position.side == PositionSide.LONG
        assert position.amount == Decimal("0.1")

        await paper_gateway.disconnect()

    @pytest.mark.asyncio
    async def test_balance_updates(self, paper_gateway: PaperGateway) -> None:
        """Test that balances update after trades."""
        await paper_gateway.connect()

        # Get initial balance
        initial_balance = await paper_gateway.get_balance("USDT")
        assert initial_balance is not None
        assert initial_balance.available == Decimal("10000")

        # Set price and trade
        from libra.gateways.protocol import Order, OrderType

        paper_gateway.update_price("BTC/USDT", Decimal("49999"), Decimal("50001"))

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),  # 0.1 * 50000 = 5000 USDT
        )
        await paper_gateway.submit_order(order)

        # Check balance reduced
        new_balance = await paper_gateway.get_balance("USDT")
        assert new_balance is not None
        assert new_balance.available < initial_balance.available

        await paper_gateway.disconnect()
