"""
End-to-End tests for DataClient/ExecutionClient architecture.

Tests the complete client architecture flow:
1. BacktestDataClient with realistic price data
2. BacktestExecutionClient with fill/slippage models
3. TradingKernel integration with clients
4. Complete backtest simulation with P&L tracking
5. Position management and balance tracking

Uses realistic BTC/USDT price patterns to simulate trading scenarios.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from libra.clients import (
    BacktestDataClient,
    BacktestExecutionClient,
    InMemoryDataSource,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    Tick,
)
from libra.clients.backtest_execution_client import SlippageModel
from libra.core.clock import Clock, ClockType
from libra.core.kernel import KernelConfig, TradingKernel
from libra.strategies.protocol import Bar


if TYPE_CHECKING:
    pass


# =============================================================================
# Test Data Generation
# =============================================================================


def generate_btc_bars(
    start_price: Decimal = Decimal("50000"),
    num_bars: int = 100,
    trend: str = "up",  # "up", "down", "range"
    volatility: Decimal = Decimal("0.02"),  # 2% per bar
    start_time: datetime | None = None,
) -> list[Bar]:
    """
    Generate realistic BTC/USDT bars for testing.

    Args:
        start_price: Starting price
        num_bars: Number of bars to generate
        trend: Price trend direction
        volatility: Price volatility per bar
        start_time: Starting timestamp

    Returns:
        List of Bar objects
    """
    bars: list[Bar] = []
    current_price = start_price

    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0, 0)

    # Trend drift per bar
    if trend == "up":
        drift = Decimal("0.005")  # 0.5% up per bar
    elif trend == "down":
        drift = Decimal("-0.005")  # 0.5% down per bar
    else:
        drift = Decimal("0")

    for i in range(num_bars):
        # Calculate OHLC with some randomness
        # Using a deterministic pattern based on index for reproducibility
        phase = (i % 10) / 10  # 0 to 0.9

        # Open is previous close (or start_price for first bar)
        open_price = current_price

        # Calculate intrabar movement
        intrabar_change = current_price * volatility * Decimal(str(phase - 0.5))

        # High and low
        high = open_price + abs(intrabar_change) + current_price * volatility * Decimal("0.3")
        low = open_price - abs(intrabar_change) - current_price * volatility * Decimal("0.3")

        # Close with trend
        close = open_price * (Decimal("1") + drift + Decimal(str((phase - 0.5) * 0.01)))

        # Ensure high >= close and open, low <= close and open
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume varies
        volume = Decimal("1000") * (Decimal("1") + Decimal(str(phase)))

        # Timestamp
        bar_time = start_time + timedelta(hours=i)
        timestamp_ns = int(bar_time.timestamp() * 1_000_000_000)

        bar = Bar(
            symbol="BTC/USDT",
            timestamp_ns=timestamp_ns,
            open=open_price.quantize(Decimal("0.01")),
            high=high.quantize(Decimal("0.01")),
            low=low.quantize(Decimal("0.01")),
            close=close.quantize(Decimal("0.01")),
            volume=volume.quantize(Decimal("0.001")),
            timeframe="1h",
        )
        bars.append(bar)

        # Update current price for next bar
        current_price = close

    return bars


def generate_ticks_from_bars(bars: list[Bar]) -> list[Tick]:
    """Generate ticks from bars for testing."""
    ticks: list[Tick] = []

    for bar in bars:
        # Create a tick at bar close
        spread = bar.close * Decimal("0.0001")  # 1 bps spread
        tick = Tick(
            symbol=bar.symbol,
            bid=bar.close - spread,
            ask=bar.close + spread,
            last=bar.close,
            timestamp_ns=bar.timestamp_ns,
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            volume_24h=bar.volume * Decimal("24"),
        )
        ticks.append(tick)

    return ticks


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clock() -> Clock:
    """Create a backtest clock."""
    return Clock(ClockType.BACKTEST)


@pytest.fixture
def start_time() -> datetime:
    """Test start time."""
    return datetime(2024, 1, 1, 0, 0, 0)


@pytest.fixture
def uptrend_bars(start_time: datetime) -> list[Bar]:
    """Generate uptrending BTC bars."""
    return generate_btc_bars(
        start_price=Decimal("45000"),
        num_bars=50,
        trend="up",
        volatility=Decimal("0.015"),
        start_time=start_time,
    )


@pytest.fixture
def downtrend_bars(start_time: datetime) -> list[Bar]:
    """Generate downtrending BTC bars."""
    return generate_btc_bars(
        start_price=Decimal("55000"),
        num_bars=50,
        trend="down",
        volatility=Decimal("0.015"),
        start_time=start_time,
    )


@pytest.fixture
def range_bars(start_time: datetime) -> list[Bar]:
    """Generate range-bound BTC bars."""
    return generate_btc_bars(
        start_price=Decimal("50000"),
        num_bars=50,
        trend="range",
        volatility=Decimal("0.02"),
        start_time=start_time,
    )


# =============================================================================
# BacktestDataClient Tests
# =============================================================================


class TestBacktestDataClientE2E:
    """End-to-end tests for BacktestDataClient."""

    @pytest.mark.asyncio
    async def test_connect_and_subscribe(
        self, uptrend_bars: list[Bar], clock: Clock, start_time: datetime
    ) -> None:
        """Test connecting and subscribing to data."""
        data_source = InMemoryDataSource()
        data_source.add_bars("BTC/USDT", "1h", uptrend_bars)
        client = BacktestDataClient(data_source, clock)

        await client.connect()
        assert client.is_connected

        # Configure range before subscribing
        end_time = start_time + timedelta(hours=len(uptrend_bars))
        client.configure_range(start_time, end_time)

        # Subscribe to bars
        await client.subscribe_bars("BTC/USDT", "1h")
        assert "BTC/USDT" in client.subscribed_bars

        await client.disconnect()
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_request_historical_bars(
        self, uptrend_bars: list[Bar], clock: Clock, start_time: datetime
    ) -> None:
        """Test requesting historical bars."""
        data_source = InMemoryDataSource()
        data_source.add_bars("BTC/USDT", "1h", uptrend_bars)
        client = BacktestDataClient(data_source, clock)

        await client.connect()

        # Configure range
        end_time = start_time + timedelta(hours=len(uptrend_bars))
        client.configure_range(start_time, end_time)

        # Request bars
        bars = await client.request_bars(
            symbol="BTC/USDT",
            timeframe="1h",
            start=start_time,
            end=end_time,
            limit=20,
        )

        assert len(bars) == 20
        assert all(b.symbol == "BTC/USDT" for b in bars)
        assert all(b.timeframe == "1h" for b in bars)

        # Verify bars are in chronological order
        for i in range(1, len(bars)):
            assert bars[i].timestamp_ns > bars[i - 1].timestamp_ns

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_stream_bars(
        self, uptrend_bars: list[Bar], clock: Clock, start_time: datetime
    ) -> None:
        """Test streaming bars."""
        data_source = InMemoryDataSource()
        data_source.add_bars("BTC/USDT", "1h", uptrend_bars[:10])  # First 10 bars
        client = BacktestDataClient(data_source, clock)

        await client.connect()

        # Configure range
        end_time = start_time + timedelta(hours=10)
        client.configure_range(start_time, end_time)

        await client.subscribe_bars("BTC/USDT", "1h")

        # Stream bars
        streamed_bars: list[Bar] = []
        async for bar in client.stream_bars():
            streamed_bars.append(bar)
            if len(streamed_bars) >= 10:
                break

        assert len(streamed_bars) == 10

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_get_orderbook_synthetic(
        self, uptrend_bars: list[Bar], clock: Clock, start_time: datetime
    ) -> None:
        """Test synthetic orderbook generation."""
        ticks = generate_ticks_from_bars(uptrend_bars[:10])
        data_source = InMemoryDataSource()
        data_source.add_bars("BTC/USDT", "1h", uptrend_bars)
        data_source.add_ticks("BTC/USDT", ticks)
        client = BacktestDataClient(data_source, clock)

        await client.connect()

        # Configure range and subscribe to ticks first
        end_time = start_time + timedelta(hours=10)
        client.configure_range(start_time, end_time)
        await client.subscribe_ticks("BTC/USDT")

        # Stream at least one tick to advance the tick index
        tick_count = 0
        async for tick in client.stream_ticks():
            tick_count += 1
            if tick_count >= 1:
                break

        orderbook = await client.get_orderbook("BTC/USDT", depth=5)

        assert orderbook.symbol == "BTC/USDT"
        assert len(orderbook.bids) == 5
        assert len(orderbook.asks) == 5

        # Verify bid < ask
        best_bid = orderbook.bids[0][0]
        best_ask = orderbook.asks[0][0]
        assert best_bid < best_ask

        await client.disconnect()


# =============================================================================
# BacktestExecutionClient Tests
# =============================================================================


class TestBacktestExecutionClientE2E:
    """End-to-end tests for BacktestExecutionClient."""

    @pytest.mark.asyncio
    async def test_connect_and_get_balances(self, clock: Clock) -> None:
        """Test connecting and getting balances."""
        client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("50000"), "BTC": Decimal("1.5")},
        )

        await client.connect()
        assert client.is_connected

        balances = await client.get_balances()
        assert "USDT" in balances
        assert "BTC" in balances
        assert balances["USDT"].total == Decimal("50000")
        assert balances["BTC"].total == Decimal("1.5")

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_market_order_execution(
        self, uptrend_bars: list[Bar], clock: Clock
    ) -> None:
        """Test market order execution with realistic slippage."""
        client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("10"),  # 10 bps
        )

        await client.connect()

        # Process a bar to set current price
        bar = uptrend_bars[0]
        tick = Tick(
            symbol="BTC/USDT",
            bid=bar.close - Decimal("5"),
            ask=bar.close + Decimal("5"),
            last=bar.close,
            timestamp_ns=bar.timestamp_ns,
        )
        await client.process_tick(tick)

        # Submit market buy
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )
        result = await client.submit_order(order)

        # Verify fill
        assert result.status == OrderStatus.FILLED
        assert result.filled_amount == Decimal("1.0")
        assert result.average_price is not None

        # Slippage should increase price for buy
        expected_base = bar.close + Decimal("5")  # Ask price
        assert result.average_price >= expected_base

        # Verify position
        position = await client.get_position("BTC/USDT")
        assert position is not None
        assert position.side == PositionSide.LONG
        assert position.amount == Decimal("1.0")

        # Verify balance decreased
        balance = await client.get_balance("USDT")
        assert balance is not None
        assert balance.available < Decimal("100000")

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_limit_order_queue(
        self, uptrend_bars: list[Bar], clock: Clock
    ) -> None:
        """Test limit order execution."""
        client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},
            slippage_model=SlippageModel.NONE,
        )

        await client.connect()

        # Process initial tick
        bar = uptrend_bars[0]
        tick = Tick(
            symbol="BTC/USDT",
            bid=bar.close - Decimal("5"),
            ask=bar.close + Decimal("5"),
            last=bar.close,
            timestamp_ns=bar.timestamp_ns,
        )
        await client.process_tick(tick)

        # Submit limit buy below current price
        limit_price = bar.close - Decimal("100")  # $100 below
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=limit_price,
        )
        result = await client.submit_order(order)

        # Should be open (not immediately filled)
        assert result.status == OrderStatus.OPEN

        # Process bars until price touches limit
        for bar in uptrend_bars[1:20]:
            tick = Tick(
                symbol="BTC/USDT",
                bid=bar.low,
                ask=bar.low + Decimal("10"),
                last=bar.low,
                timestamp_ns=bar.timestamp_ns,
            )
            await client.process_tick(tick)
            await client.process_bar(bar)

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_position_close_with_pnl(
        self, uptrend_bars: list[Bar], clock: Clock
    ) -> None:
        """Test opening and closing position with P&L tracking."""
        client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},
            slippage_model=SlippageModel.NONE,
        )

        await client.connect()

        initial_balance = await client.get_balance("USDT")
        assert initial_balance is not None
        initial_usdt = initial_balance.total

        # Buy at first bar
        bar1 = uptrend_bars[0]
        tick1 = Tick(
            symbol="BTC/USDT",
            bid=bar1.close - Decimal("5"),
            ask=bar1.close + Decimal("5"),
            last=bar1.close,
            timestamp_ns=bar1.timestamp_ns,
        )
        await client.process_tick(tick1)

        buy_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )
        buy_result = await client.submit_order(buy_order)
        buy_price = buy_result.average_price
        assert buy_price is not None

        # Sell at later bar (should be higher price in uptrend)
        bar2 = uptrend_bars[20]
        tick2 = Tick(
            symbol="BTC/USDT",
            bid=bar2.close - Decimal("5"),
            ask=bar2.close + Decimal("5"),
            last=bar2.close,
            timestamp_ns=bar2.timestamp_ns,
        )
        await client.process_tick(tick2)

        sell_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )
        sell_result = await client.submit_order(sell_order)
        sell_price = sell_result.average_price
        assert sell_price is not None

        # Position should be flat
        position = await client.get_position("BTC/USDT")
        assert position is None or position.side == PositionSide.FLAT

        # Calculate expected P&L (sell - buy)
        expected_pnl = sell_price - buy_price

        # Verify balance increased (uptrend = profit)
        final_balance = await client.get_balance("USDT")
        assert final_balance is not None
        actual_pnl = final_balance.total - initial_usdt

        # In uptrend, we should have profit
        assert actual_pnl > 0, f"Expected profit in uptrend, got {actual_pnl}"

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_cancel_order(
        self, uptrend_bars: list[Bar], clock: Clock
    ) -> None:
        """Test cancelling an open order."""
        client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},
        )

        await client.connect()

        # Process tick
        bar = uptrend_bars[0]
        tick = Tick(
            symbol="BTC/USDT",
            bid=bar.close - Decimal("5"),
            ask=bar.close + Decimal("5"),
            last=bar.close,
            timestamp_ns=bar.timestamp_ns,
        )
        await client.process_tick(tick)

        # Submit limit order far from market (won't fill)
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=bar.close - Decimal("5000"),  # Way below market
        )
        result = await client.submit_order(order)
        assert result.status == OrderStatus.OPEN

        # Cancel it
        cancelled = await client.cancel_order(result.order_id, "BTC/USDT")
        assert cancelled

        # Verify no open orders
        open_orders = await client.get_open_orders()
        assert len(open_orders) == 0

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_insufficient_funds(
        self, uptrend_bars: list[Bar], clock: Clock
    ) -> None:
        """Test order rejection due to insufficient funds."""
        client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("1000")},  # Only $1000
        )

        await client.connect()

        # Process tick
        bar = uptrend_bars[0]
        tick = Tick(
            symbol="BTC/USDT",
            bid=bar.close - Decimal("5"),
            ask=bar.close + Decimal("5"),
            last=bar.close,
            timestamp_ns=bar.timestamp_ns,
        )
        await client.process_tick(tick)

        # Try to buy more than we can afford
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("10.0"),  # 10 BTC at ~$45000 = $450k
            price=bar.close,
        )
        result = await client.submit_order(order)

        # Should be rejected
        assert result.status == OrderStatus.REJECTED

        await client.disconnect()


# =============================================================================
# Slippage Model Tests
# =============================================================================


class TestSlippageModelsE2E:
    """End-to-end tests for different slippage models."""

    @pytest.mark.asyncio
    async def test_no_slippage(self, uptrend_bars: list[Bar], clock: Clock) -> None:
        """Test execution with no slippage."""
        client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},
            slippage_model=SlippageModel.NONE,
        )

        await client.connect()

        bar = uptrend_bars[0]
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("45000"),
            ask=Decimal("45010"),
            last=bar.close,
            timestamp_ns=bar.timestamp_ns,
        )
        await client.process_tick(tick)

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )
        result = await client.submit_order(order)

        # Should fill at exactly the ask price
        assert result.average_price == Decimal("45010")

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_fixed_slippage(self, uptrend_bars: list[Bar], clock: Clock) -> None:
        """Test execution with fixed basis point slippage."""
        client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("20"),  # 20 bps = 0.2%
        )

        await client.connect()

        bar = uptrend_bars[0]
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=bar.close,
            timestamp_ns=bar.timestamp_ns,
        )
        await client.process_tick(tick)

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )
        result = await client.submit_order(order)

        # Slippage: 50010 * 0.002 = 100.02
        # Expected: 50010 + 100.02 = 50110.02
        assert result.average_price is not None
        expected = Decimal("50010") * (Decimal("1") + Decimal("20") / Decimal("10000"))
        assert abs(result.average_price - expected) < Decimal("0.01")

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_volume_slippage(self, uptrend_bars: list[Bar], clock: Clock) -> None:
        """Test execution with volume-based slippage."""
        client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("10000000")},  # $10M to buy large
            slippage_model=SlippageModel.VOLUME,
        )

        await client.connect()

        bar = uptrend_bars[0]
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=bar.close,
            timestamp_ns=bar.timestamp_ns,
            volume_24h=Decimal("50000000"),  # $50M volume
        )
        await client.process_tick(tick)

        # Small order - minimal slippage
        small_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),  # Small
        )
        small_result = await client.submit_order(small_order)

        # Large order - more slippage
        large_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("50"),  # Large
        )
        large_result = await client.submit_order(large_order)

        # Large order should have more slippage
        assert large_result.average_price is not None
        assert small_result.average_price is not None
        # The difference in slippage should be significant
        small_slippage = small_result.average_price - Decimal("50010")
        large_slippage = large_result.average_price - Decimal("50010")
        assert large_slippage > small_slippage

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_stochastic_slippage(self, uptrend_bars: list[Bar], clock: Clock) -> None:
        """Test execution with stochastic slippage."""
        client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("1000000")},
            slippage_model=SlippageModel.STOCHASTIC,
        )

        await client.connect()

        bar = uptrend_bars[0]
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=bar.close,
            timestamp_ns=bar.timestamp_ns,
        )
        await client.process_tick(tick)

        # Execute multiple orders and verify variance
        prices: list[Decimal] = []
        for _ in range(10):
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("0.1"),
            )
            result = await client.submit_order(order)
            assert result.average_price is not None
            prices.append(result.average_price)

        # Prices should vary due to stochastic slippage
        unique_prices = set(prices)
        assert len(unique_prices) > 1, "Stochastic slippage should produce varying prices"

        await client.disconnect()


# =============================================================================
# Integration Tests
# =============================================================================


class TestClientIntegrationE2E:
    """End-to-end integration tests for DataClient + ExecutionClient."""

    @pytest.mark.asyncio
    async def test_complete_backtest_simulation(
        self, uptrend_bars: list[Bar], clock: Clock, start_time: datetime
    ) -> None:
        """Test a complete backtest simulation with both clients."""
        # Setup data client
        ticks = generate_ticks_from_bars(uptrend_bars)
        data_source = InMemoryDataSource()
        data_source.add_bars("BTC/USDT", "1h", uptrend_bars)
        data_source.add_ticks("BTC/USDT", ticks)
        data_client = BacktestDataClient(data_source, clock)

        # Setup execution client
        exec_client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},
            slippage_model=SlippageModel.FIXED,
            slippage_bps=Decimal("5"),
        )

        await data_client.connect()
        await exec_client.connect()

        # Configure data range
        end_time = start_time + timedelta(hours=len(uptrend_bars))
        data_client.configure_range(start_time, end_time)

        # Subscribe to data
        await data_client.subscribe_bars("BTC/USDT", "1h")

        # Get initial balance
        initial_balance = await exec_client.get_balance("USDT")
        assert initial_balance is not None
        initial_usdt = initial_balance.total

        bars_processed = 0
        position_opened = False

        async for bar in data_client.stream_bars():
            bars_processed += 1

            # Update execution client with current price
            tick = Tick(
                symbol=bar.symbol,
                bid=bar.close - Decimal("5"),
                ask=bar.close + Decimal("5"),
                last=bar.close,
                timestamp_ns=bar.timestamp_ns,
            )
            await exec_client.process_tick(tick)

            # Buy on first bar
            if bars_processed == 1 and not position_opened:
                order = Order(
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    amount=Decimal("1.0"),
                )
                result = await exec_client.submit_order(order)
                assert result.status == OrderStatus.FILLED
                position_opened = True

            # Sell on last bar
            if bars_processed == len(uptrend_bars):
                position = await exec_client.get_position("BTC/USDT")
                if position and position.amount > 0:
                    order = Order(
                        symbol="BTC/USDT",
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        amount=position.amount,
                    )
                    result = await exec_client.submit_order(order)
                    assert result.status == OrderStatus.FILLED
                break

        # Verify results
        assert bars_processed == len(uptrend_bars)

        final_balance = await exec_client.get_balance("USDT")
        assert final_balance is not None

        # In uptrend, we should have profit (minus slippage and spread)
        pnl = final_balance.total - initial_usdt
        assert pnl > 0, f"Expected profit in uptrend, got PnL: {pnl}"

        await data_client.disconnect()
        await exec_client.disconnect()

    @pytest.mark.asyncio
    async def test_multiple_symbols_trading(self, clock: Clock, start_time: datetime) -> None:
        """Test trading multiple symbols simultaneously."""
        # Generate data for multiple symbols
        btc_bars = generate_btc_bars(
            start_price=Decimal("50000"),
            num_bars=30,
            trend="up",
            start_time=start_time,
        )
        eth_bars = [
            Bar(
                symbol="ETH/USDT",
                timestamp_ns=b.timestamp_ns,
                open=b.open * Decimal("0.06"),  # Scale to ETH prices
                high=b.high * Decimal("0.06"),
                low=b.low * Decimal("0.06"),
                close=b.close * Decimal("0.06"),
                volume=b.volume * Decimal("10"),
                timeframe=b.timeframe,
            )
            for b in btc_bars
        ]

        data_source = InMemoryDataSource()
        data_source.add_bars("BTC/USDT", "1h", btc_bars)
        data_source.add_bars("ETH/USDT", "1h", eth_bars)

        data_client = BacktestDataClient(data_source, clock)
        exec_client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},
        )

        await data_client.connect()
        await exec_client.connect()

        # Trade both symbols
        for symbol, bars in [("BTC/USDT", btc_bars), ("ETH/USDT", eth_bars)]:
            bar = bars[0]
            tick = Tick(
                symbol=symbol,
                bid=bar.close - Decimal("1"),
                ask=bar.close + Decimal("1"),
                last=bar.close,
                timestamp_ns=bar.timestamp_ns,
            )
            await exec_client.process_tick(tick)

            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("0.1"),
            )
            result = await exec_client.submit_order(order)
            assert result.status == OrderStatus.FILLED

        # Verify we have positions in both
        positions = await exec_client.get_positions()
        symbols_with_position = {p.symbol for p in positions}
        assert "BTC/USDT" in symbols_with_position
        assert "ETH/USDT" in symbols_with_position

        await data_client.disconnect()
        await exec_client.disconnect()


# =============================================================================
# TradingKernel Integration Tests
# =============================================================================


class TestTradingKernelClientIntegration:
    """Test TradingKernel integration with new client architecture."""

    @pytest.mark.asyncio
    async def test_kernel_with_clients(
        self, uptrend_bars: list[Bar], start_time: datetime
    ) -> None:
        """Test TradingKernel lifecycle with DataClient and ExecutionClient."""
        # Create a dedicated clock for backtest
        clock = Clock(ClockType.BACKTEST)

        # Create clients
        data_source = InMemoryDataSource()
        data_source.add_bars("BTC/USDT", "1h", uptrend_bars)
        data_client = BacktestDataClient(data_source, clock)
        exec_client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},
        )

        # Create kernel
        config = KernelConfig(environment="backtest")
        kernel = TradingKernel(config)

        # Set clients
        kernel.set_clients(data_client, exec_client)

        # Verify clients are set
        assert kernel.data_client is data_client
        assert kernel.execution_client is exec_client

        # Start kernel
        await kernel.start_async()
        assert kernel.is_running
        assert data_client.is_connected
        assert exec_client.is_connected

        # Stop kernel
        await kernel.stop_async()
        assert kernel.is_stopped
        assert not data_client.is_connected
        assert not exec_client.is_connected

        await kernel.dispose()

    @pytest.mark.asyncio
    async def test_kernel_context_manager_with_clients(
        self, uptrend_bars: list[Bar], start_time: datetime
    ) -> None:
        """Test TradingKernel context manager with clients."""
        clock = Clock(ClockType.BACKTEST)

        data_source = InMemoryDataSource()
        data_source.add_bars("BTC/USDT", "1h", uptrend_bars)
        data_client = BacktestDataClient(data_source, clock)
        exec_client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},
        )

        config = KernelConfig(environment="backtest")
        kernel = TradingKernel(config)
        kernel.set_data_client(data_client)
        kernel.set_execution_client(exec_client)

        async with kernel:
            assert kernel.is_running
            assert data_client.is_connected
            assert exec_client.is_connected

            # Can access clients through kernel
            assert kernel.data_client is not None
            assert kernel.execution_client is not None

        # After context, should be stopped
        assert kernel.is_stopped

    @pytest.mark.asyncio
    async def test_kernel_health_check(
        self, uptrend_bars: list[Bar], start_time: datetime
    ) -> None:
        """Test kernel health check with clients."""
        clock = Clock(ClockType.BACKTEST)

        data_source = InMemoryDataSource()
        data_source.add_bars("BTC/USDT", "1h", uptrend_bars)
        data_client = BacktestDataClient(data_source, clock)
        exec_client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("100000")},
        )

        config = KernelConfig(environment="backtest")
        kernel = TradingKernel(config)
        kernel.set_clients(data_client, exec_client)

        async with kernel:
            health = kernel.health_check()

            assert health["kernel"]["state"] == "RUNNING"
            assert health["kernel"]["environment"] == "backtest"
            assert kernel.is_healthy()

        # After context, no longer healthy
        assert not kernel.is_healthy()


# =============================================================================
# Performance Tests
# =============================================================================


class TestClientPerformanceE2E:
    """Performance tests for client architecture."""

    @pytest.mark.asyncio
    async def test_high_frequency_order_execution(self) -> None:
        """Test rapid order execution performance."""
        clock = Clock(ClockType.BACKTEST)
        client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("10000000")},  # $10M
            slippage_model=SlippageModel.NONE,
        )

        await client.connect()

        # Set initial tick
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=1,
        )
        await client.process_tick(tick)

        # Execute many orders rapidly
        import time
        num_orders = 1000

        start = time.perf_counter()
        for i in range(num_orders):
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                amount=Decimal("0.01"),
            )
            result = await client.submit_order(order)
            assert result.status == OrderStatus.FILLED

        elapsed = time.perf_counter() - start
        orders_per_second = num_orders / elapsed

        # Should handle at least 1000 orders/second
        assert orders_per_second > 1000, f"Only {orders_per_second:.0f} orders/sec"

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_large_bar_dataset_streaming(self, start_time: datetime) -> None:
        """Test streaming large datasets efficiently."""
        clock = Clock(ClockType.BACKTEST)

        # Generate 1000 bars
        large_bars = generate_btc_bars(
            start_price=Decimal("50000"),
            num_bars=1000,
            trend="range",
            start_time=start_time,
        )

        data_source = InMemoryDataSource()
        data_source.add_bars("BTC/USDT", "1h", large_bars)
        client = BacktestDataClient(data_source, clock)

        await client.connect()

        end_time = start_time + timedelta(hours=1000)
        client.configure_range(start_time, end_time)
        await client.subscribe_bars("BTC/USDT", "1h")

        import time
        start = time.perf_counter()

        count = 0
        async for bar in client.stream_bars():
            count += 1

        elapsed = time.perf_counter() - start
        bars_per_second = count / elapsed

        assert count == 1000
        # Should process at least 10k bars/second
        assert bars_per_second > 10000, f"Only {bars_per_second:.0f} bars/sec"

        await client.disconnect()
