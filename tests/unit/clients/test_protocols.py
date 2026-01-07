"""
Tests for DataClient and ExecutionClient protocols.

Tests cover:
- Protocol runtime checkability
- BacktestDataClient functionality
- BacktestExecutionClient functionality
- Fill model behavior
- Slippage model behavior
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from libra.clients import (
    BacktestDataClient,
    BacktestExecutionClient,
    DataClient,
    ExecutionClient,
    FixedSlippage,
    ImmediateFillModel,
    InMemoryDataSource,
    NoSlippage,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    QueuePositionFillModel,
    SlippageModel,
    Tick,
)
from libra.strategies.protocol import Bar


# =============================================================================
# Mock Clock for Testing
# =============================================================================


class MockClock:
    """Simple mock clock for testing."""

    def __init__(self, start_time: datetime | None = None) -> None:
        self._time_ns = int((start_time or datetime.now()).timestamp() * 1_000_000_000)

    def timestamp_ns(self) -> int:
        """Return current time in nanoseconds."""
        return self._time_ns

    def advance(self, seconds: float) -> None:
        self._time_ns += int(seconds * 1_000_000_000)

    def set_time(self, time_ns: int) -> None:
        self._time_ns = time_ns


# =============================================================================
# Protocol Tests
# =============================================================================


class TestProtocols:
    """Test that protocols are runtime checkable."""

    def test_data_client_protocol_is_checkable(self) -> None:
        """DataClient should be runtime checkable."""
        assert isinstance(BacktestDataClient, type)

    def test_execution_client_protocol_is_checkable(self) -> None:
        """ExecutionClient should be runtime checkable."""
        assert isinstance(BacktestExecutionClient, type)


# =============================================================================
# Slippage Model Tests
# =============================================================================


class TestSlippageModels:
    """Test slippage model implementations."""

    def test_no_slippage(self) -> None:
        """NoSlippage should return zero slippage."""
        model = NoSlippage()
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=0,
        )

        slippage = model.calculate(order, tick, Decimal("50010"))
        assert slippage == Decimal("0")

    def test_fixed_slippage_buy(self) -> None:
        """FixedSlippage should add slippage for buys."""
        model = FixedSlippage(bps=Decimal("10"))  # 10 bps = 0.1%
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=0,
        )

        slippage = model.calculate(order, tick, Decimal("50000"))
        # 10 bps of 50000 = 50000 * 10/10000 = 50
        assert slippage == Decimal("50")

    def test_fixed_slippage_sell(self) -> None:
        """FixedSlippage should subtract slippage for sells."""
        model = FixedSlippage(bps=Decimal("10"))
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=0,
        )

        slippage = model.calculate(order, tick, Decimal("50000"))
        # 10 bps of 50000 = -50 (negative for sells = receive less)
        assert slippage == Decimal("-50")


# =============================================================================
# Fill Model Tests
# =============================================================================


class TestFillModels:
    """Test fill model implementations."""

    def test_immediate_fill_market_buy(self) -> None:
        """ImmediateFillModel should fill market buy at ask."""
        model = ImmediateFillModel()
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=0,
        )

        price, qty = model.check_fill(order, tick)
        assert price == Decimal("50010")  # Fill at ask
        assert qty == Decimal("1.0")

    def test_immediate_fill_market_sell(self) -> None:
        """ImmediateFillModel should fill market sell at bid."""
        model = ImmediateFillModel()
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
        )
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=0,
        )

        price, qty = model.check_fill(order, tick)
        assert price == Decimal("50000")  # Fill at bid
        assert qty == Decimal("1.0")

    def test_immediate_fill_limit_buy_crosses(self) -> None:
        """ImmediateFillModel should fill limit buy when ask <= limit."""
        model = ImmediateFillModel()
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50020"),  # Limit above ask
        )
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=0,
        )

        price, qty = model.check_fill(order, tick)
        assert price == Decimal("50020")  # Fill at limit price
        assert qty == Decimal("1.0")

    def test_immediate_fill_limit_buy_no_fill(self) -> None:
        """ImmediateFillModel should not fill limit buy when ask > limit."""
        model = ImmediateFillModel()
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("49990"),  # Limit below ask
        )
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=0,
        )

        price, qty = model.check_fill(order, tick)
        assert price is None
        assert qty == Decimal("0")


# =============================================================================
# BacktestDataClient Tests
# =============================================================================


class TestBacktestDataClient:
    """Test BacktestDataClient functionality."""

    @pytest.fixture
    def clock(self) -> MockClock:
        """Create mock clock."""
        return MockClock(datetime(2024, 1, 1))

    @pytest.fixture
    def data_source(self) -> InMemoryDataSource:
        """Create in-memory data source with test data."""
        source = InMemoryDataSource()

        # Add test bars
        base_time = datetime(2024, 1, 1)
        bars = []
        for i in range(10):
            bar = Bar(
                symbol="BTC/USDT",
                timestamp_ns=int((base_time + timedelta(hours=i)).timestamp() * 1e9),
                open=Decimal("50000") + i * 100,
                high=Decimal("50100") + i * 100,
                low=Decimal("49900") + i * 100,
                close=Decimal("50050") + i * 100,
                volume=Decimal("100"),
                timeframe="1h",
            )
            bars.append(bar)

        source.add_bars("BTC/USDT", "1h", bars)
        return source

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, clock: MockClock, data_source: InMemoryDataSource) -> None:
        """Test connect and disconnect lifecycle."""
        client = BacktestDataClient(data_source, clock)

        assert not client.is_connected
        await client.connect()
        assert client.is_connected
        await client.disconnect()
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_subscribe_bars(self, clock: MockClock, data_source: InMemoryDataSource) -> None:
        """Test subscribing to bar data loads data."""
        client = BacktestDataClient(data_source, clock)
        client.configure_range(
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
        )

        await client.connect()
        await client.subscribe_bars("BTC/USDT", "1h")

        assert "BTC/USDT" in client.subscribed_bars
        assert "1h" in client.subscribed_bars["BTC/USDT"]

    @pytest.mark.asyncio
    async def test_request_bars(self, clock: MockClock, data_source: InMemoryDataSource) -> None:
        """Test requesting historical bars."""
        client = BacktestDataClient(data_source, clock)
        await client.connect()

        bars = await client.request_bars(
            "BTC/USDT",
            "1h",
            datetime(2024, 1, 1),
            datetime(2024, 1, 1, 5),
        )

        assert len(bars) == 6  # Hours 0-5
        assert bars[0].symbol == "BTC/USDT"
        assert bars[0].timeframe == "1h"


# =============================================================================
# BacktestExecutionClient Tests
# =============================================================================


class TestBacktestExecutionClient:
    """Test BacktestExecutionClient functionality."""

    @pytest.fixture
    def clock(self) -> MockClock:
        """Create mock clock."""
        return MockClock(datetime(2024, 1, 1))

    @pytest.fixture
    def client(self, clock: MockClock) -> BacktestExecutionClient:
        """Create execution client with initial balance."""
        return BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("10000"), "BTC": Decimal("0")},
            slippage_model=SlippageModel.NONE,
        )

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, client: BacktestExecutionClient) -> None:
        """Test connect and disconnect lifecycle."""
        assert not client.is_connected
        await client.connect()
        assert client.is_connected
        await client.disconnect()
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_get_balances(self, client: BacktestExecutionClient) -> None:
        """Test getting initial balances."""
        await client.connect()
        balances = await client.get_balances()

        assert "USDT" in balances
        assert balances["USDT"].total == Decimal("10000")
        assert balances["USDT"].available == Decimal("10000")

    @pytest.mark.asyncio
    async def test_submit_market_order_buy(self, client: BacktestExecutionClient) -> None:
        """Test submitting and filling a market buy order."""
        await client.connect()

        # Set up latest tick
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=0,
        )
        await client.process_tick(tick)

        # Submit market buy
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )
        result = await client.submit_order(order)

        assert result.status == OrderStatus.FILLED
        assert result.filled_amount == Decimal("0.1")
        assert result.average_price == Decimal("50010")  # Filled at ask

    @pytest.mark.asyncio
    async def test_submit_market_order_sell(self, client: BacktestExecutionClient) -> None:
        """Test submitting and filling a market sell order."""
        await client.connect()

        # First buy some BTC
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=0,
        )
        await client.process_tick(tick)

        buy_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )
        await client.submit_order(buy_order)

        # Now sell
        sell_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )
        result = await client.submit_order(sell_order)

        assert result.status == OrderStatus.FILLED
        assert result.filled_amount == Decimal("0.1")
        assert result.average_price == Decimal("50000")  # Filled at bid

    @pytest.mark.asyncio
    async def test_submit_limit_order_queued(self, client: BacktestExecutionClient) -> None:
        """Test submitting a limit order that doesn't immediately fill."""
        await client.connect()

        # Set up latest tick
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=0,
        )
        await client.process_tick(tick)

        # Submit limit buy below market
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("49000"),  # Below current ask
        )
        result = await client.submit_order(order)

        assert result.status == OrderStatus.OPEN
        assert result.filled_amount == Decimal("0")

    @pytest.mark.asyncio
    async def test_cancel_order(self, client: BacktestExecutionClient) -> None:
        """Test cancelling a pending order."""
        await client.connect()

        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=0,
        )
        await client.process_tick(tick)

        # Submit limit order
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("49000"),
        )
        result = await client.submit_order(order)

        # Cancel it
        cancelled = await client.cancel_order(result.order_id, "BTC/USDT")
        assert cancelled

        # Check it's cancelled
        order_status = await client.get_order(result.order_id, "BTC/USDT")
        assert order_status.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_position_tracking(self, client: BacktestExecutionClient) -> None:
        """Test that positions are tracked correctly."""
        await client.connect()

        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=0,
        )
        await client.process_tick(tick)

        # Buy to open position
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
        )
        await client.submit_order(order)

        # Check position
        position = await client.get_position("BTC/USDT")
        assert position is not None
        assert position.amount == Decimal("0.1")

    @pytest.mark.asyncio
    async def test_insufficient_funds_rejected(self, client: BacktestExecutionClient) -> None:
        """Test that orders are rejected when insufficient funds."""
        await client.connect()

        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=0,
        )
        await client.process_tick(tick)

        # Try to buy more than we can afford with a limit order
        # 10000 USDT balance, trying to buy 1 BTC at 50000 = 50000 USDT needed
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),  # Would cost 50000 USDT, we only have 10000
        )
        result = await client.submit_order(order)

        assert result.status == OrderStatus.REJECTED


# =============================================================================
# Integration Tests
# =============================================================================


class TestBacktestIntegration:
    """Integration tests for backtest data + execution clients."""

    @pytest.mark.asyncio
    async def test_bar_processing_triggers_fills(self) -> None:
        """Test that processing bars triggers limit order fills."""
        clock = MockClock(datetime(2024, 1, 1))

        exec_client = BacktestExecutionClient(
            clock=clock,
            initial_balance={"USDT": Decimal("10000")},
            slippage_model=SlippageModel.NONE,
        )
        await exec_client.connect()

        # Submit limit buy below current market
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("49500"),  # Below market
        )

        # Set initial tick above limit price
        tick = Tick(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            timestamp_ns=0,
        )
        await exec_client.process_tick(tick)

        result = await exec_client.submit_order(order)
        assert result.status == OrderStatus.OPEN

        # Process bar that dips below limit price
        bar = Bar(
            symbol="BTC/USDT",
            timestamp_ns=clock.timestamp_ns(),
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49400"),  # Low below limit price
            close=Decimal("49800"),
            volume=Decimal("100"),
            timeframe="1h",
        )

        fills = await exec_client.process_bar(bar)

        # Order should have filled
        assert len(fills) == 1
        assert fills[0].status == OrderStatus.FILLED
        assert fills[0].average_price == Decimal("49500")
