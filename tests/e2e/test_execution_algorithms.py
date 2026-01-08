"""
E2E Tests for Execution Algorithms.

Tests for Issue #36: Execution Algorithm Framework (TWAP, VWAP).

These tests simulate real execution scenarios with mock execution clients
that behave like real exchanges (partial fills, slippage, etc.).
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal

import pytest

from libra.execution import (
    AlgorithmState,
    IcebergAlgorithm,
    IcebergConfig,
    TWAPAlgorithm,
    TWAPConfig,
    VWAPAlgorithm,
    VWAPConfig,
    create_algorithm,
    list_algorithms,
)
from libra.gateways.fetcher import Bar
from libra.gateways.protocol import Order, OrderResult, OrderSide, OrderStatus, OrderType


# =============================================================================
# Test Fixtures
# =============================================================================


class SimulatedExecutionClient:
    """
    Simulated execution client for E2E testing.

    Simulates realistic market behavior including:
    - Partial fills
    - Random slippage
    - Variable fill times
    """

    def __init__(
        self,
        fill_rate: float = 1.0,  # 1.0 = always fill, 0.5 = 50% fill rate
        slippage_bps: float = 5.0,  # Average slippage in basis points
        base_price: Decimal = Decimal("42000"),
    ) -> None:
        self.fill_rate = fill_rate
        self.slippage_bps = slippage_bps
        self.base_price = base_price
        self.orders_submitted: list[Order] = []
        self.total_filled: Decimal = Decimal("0")
        self.total_value: Decimal = Decimal("0")

    async def submit_order(self, order: Order) -> OrderResult:
        """Submit order with simulated execution."""
        import random

        self.orders_submitted.append(order)

        # Simulate partial fill based on fill_rate
        fill_pct = random.uniform(self.fill_rate * 0.8, min(1.0, self.fill_rate * 1.2))
        filled_amount = order.amount * Decimal(str(fill_pct))

        # Simulate price slippage
        slippage_factor = random.uniform(-self.slippage_bps, self.slippage_bps) / 10000
        if order.side == OrderSide.BUY:
            execution_price = self.base_price * Decimal(str(1 + slippage_factor))
        else:
            execution_price = self.base_price * Decimal(str(1 - slippage_factor))

        self.total_filled += filled_amount
        self.total_value += filled_amount * execution_price

        # Small delay to simulate network latency
        await asyncio.sleep(0.001)

        return OrderResult(
            order_id=f"sim-{len(self.orders_submitted)}",
            symbol=order.symbol,
            status=OrderStatus.FILLED if fill_pct >= 0.99 else OrderStatus.PARTIALLY_FILLED,
            side=order.side,
            order_type=order.order_type,
            amount=order.amount,
            filled_amount=filled_amount,
            remaining_amount=order.amount - filled_amount,
            average_price=execution_price,
            fee=filled_amount * Decimal("0.001"),  # 0.1% fee
            fee_currency="USDT",
            timestamp_ns=time.time_ns(),
            client_order_id=order.client_order_id,
        )

    @property
    def vwap(self) -> Decimal:
        """Calculate VWAP of all executions."""
        if self.total_filled == 0:
            return Decimal("0")
        return self.total_value / self.total_filled


@pytest.fixture
def simulated_client() -> SimulatedExecutionClient:
    """Create a simulated execution client."""
    return SimulatedExecutionClient()


@pytest.fixture
def sample_order() -> Order:
    """Create a sample order for testing."""
    return Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("100"),
        client_order_id="e2e-test-order",
    )


@pytest.fixture
def historical_bars() -> list[Bar]:
    """Create historical bar data for VWAP volume profiling."""
    base_ts = int(time.time() * 1_000_000_000) - 12 * 3600_000_000_000

    # Simulate typical trading day volume pattern (U-shaped)
    volumes = [
        300, 250, 200, 150, 120, 100,  # Morning decline
        100, 110, 130, 160, 220, 350,  # Afternoon increase
    ]

    bars = []
    for i, vol in enumerate(volumes):
        bars.append(
            Bar(
                symbol="BTC/USDT",
                timestamp_ns=base_ts + i * 3600_000_000_000,
                open=Decimal("42000") + Decimal(str(i * 10)),
                high=Decimal("42100") + Decimal(str(i * 10)),
                low=Decimal("41900") + Decimal(str(i * 10)),
                close=Decimal("42050") + Decimal(str(i * 10)),
                volume=Decimal(str(vol)),
            )
        )
    return bars


# =============================================================================
# TWAP E2E Tests
# =============================================================================


class TestTWAPE2E:
    """E2E tests for TWAP algorithm."""

    @pytest.mark.asyncio
    async def test_twap_full_execution(
        self,
        simulated_client: SimulatedExecutionClient,
        sample_order: Order,
    ) -> None:
        """Test full TWAP execution with simulated fills."""
        config = TWAPConfig(
            horizon_secs=0.1,  # 100ms total
            interval_secs=0.02,  # 20ms intervals = 5 slices
            randomize_size=False,
            randomize_delay=False,
        )
        algo = TWAPAlgorithm(config, simulated_client)

        progress = await algo.execute(sample_order)

        assert progress.state == AlgorithmState.COMPLETED
        assert progress.num_children_spawned == 5
        assert progress.executed_quantity > 0
        assert simulated_client.total_filled > 0

    @pytest.mark.asyncio
    async def test_twap_with_randomization(
        self,
        simulated_client: SimulatedExecutionClient,
        sample_order: Order,
    ) -> None:
        """Test TWAP with size and delay randomization."""
        config = TWAPConfig(
            horizon_secs=0.1,
            interval_secs=0.02,
            randomize_size=True,
            randomize_delay=True,
            randomization_pct=0.2,
        )
        algo = TWAPAlgorithm(config, simulated_client)

        progress = await algo.execute(sample_order)

        assert progress.state == AlgorithmState.COMPLETED
        # Should still complete despite randomization
        assert progress.num_children_spawned >= 4  # May vary due to randomization

    @pytest.mark.asyncio
    async def test_twap_cancellation(
        self,
        simulated_client: SimulatedExecutionClient,
        sample_order: Order,
    ) -> None:
        """Test TWAP cancellation mid-execution."""
        config = TWAPConfig(
            horizon_secs=1.0,  # Long enough to cancel
            interval_secs=0.1,
        )
        algo = TWAPAlgorithm(config, simulated_client)

        # Start execution
        task = asyncio.create_task(algo.execute(sample_order))

        # Wait a bit then cancel
        await asyncio.sleep(0.15)
        await algo.cancel()

        progress = await task

        assert progress.state == AlgorithmState.CANCELLED
        assert progress.num_children_spawned > 0
        assert progress.num_children_spawned < config.num_slices  # Didn't finish all

    @pytest.mark.asyncio
    async def test_twap_progress_tracking(
        self,
        simulated_client: SimulatedExecutionClient,
        sample_order: Order,
    ) -> None:
        """Test TWAP progress tracking during execution."""
        config = TWAPConfig(
            horizon_secs=0.1,
            interval_secs=0.05,
            randomize_size=False,
            randomize_delay=False,
        )
        algo = TWAPAlgorithm(config, simulated_client)

        progress = await algo.execute(sample_order)

        assert progress.total_quantity == Decimal("100")
        assert progress.executed_quantity > 0
        assert progress.completion_pct > 0


# =============================================================================
# VWAP E2E Tests
# =============================================================================


class TestVWAPE2E:
    """E2E tests for VWAP algorithm."""

    @pytest.mark.asyncio
    async def test_vwap_with_volume_profile(
        self,
        simulated_client: SimulatedExecutionClient,
        sample_order: Order,
        historical_bars: list[Bar],
    ) -> None:
        """Test VWAP execution with historical volume profile."""
        config = VWAPConfig(
            num_intervals=6,
            interval_secs=0.02,
        )
        algo = VWAPAlgorithm(config, simulated_client)

        # Load volume profile
        profile = algo.load_volume_profile(historical_bars)
        assert len(profile.fractions) == 6
        assert abs(sum(profile.fractions) - 1.0) < 0.0001

        # Execute
        progress = await algo.execute(sample_order)

        assert progress.state == AlgorithmState.COMPLETED
        assert progress.num_children_spawned == 6

    @pytest.mark.asyncio
    async def test_vwap_tracks_running_vwap(
        self,
        simulated_client: SimulatedExecutionClient,
        sample_order: Order,
    ) -> None:
        """Test that VWAP algorithm tracks running VWAP."""
        config = VWAPConfig(
            num_intervals=4,
            interval_secs=0.01,
        )
        algo = VWAPAlgorithm(config, simulated_client)

        await algo.execute(sample_order)

        # Should have calculated running VWAP
        assert algo.vwap > 0
        # VWAP should be close to simulated client's VWAP
        client_vwap = simulated_client.vwap
        assert abs(algo.vwap - client_vwap) < Decimal("100")  # Within $100

    @pytest.mark.asyncio
    async def test_vwap_equal_distribution_fallback(
        self,
        simulated_client: SimulatedExecutionClient,
        sample_order: Order,
    ) -> None:
        """Test VWAP falls back to equal distribution without profile."""
        config = VWAPConfig(
            num_intervals=4,
            interval_secs=0.01,
        )
        algo = VWAPAlgorithm(config, simulated_client)

        # Don't load profile - should use equal distribution
        progress = await algo.execute(sample_order)

        assert progress.state == AlgorithmState.COMPLETED
        assert progress.num_children_spawned == 4


# =============================================================================
# Iceberg E2E Tests
# =============================================================================


class TestIcebergE2E:
    """E2E tests for Iceberg algorithm."""

    @pytest.mark.asyncio
    async def test_iceberg_full_execution(
        self,
        simulated_client: SimulatedExecutionClient,
        sample_order: Order,
    ) -> None:
        """Test full Iceberg execution."""
        config = IcebergConfig(
            display_pct=0.2,  # 20% visible
            delay_between_refills_secs=0.01,
            randomize_display=False,
        )
        algo = IcebergAlgorithm(config, simulated_client)

        progress = await algo.execute(sample_order)

        assert progress.state == AlgorithmState.COMPLETED
        # Should have multiple refills (100 / 20% = ~5)
        assert algo.num_refills >= 5
        assert simulated_client.total_filled > 0

    @pytest.mark.asyncio
    async def test_iceberg_fixed_display_qty(
        self,
        simulated_client: SimulatedExecutionClient,
    ) -> None:
        """Test Iceberg with fixed display quantity."""
        config = IcebergConfig(
            display_qty=Decimal("10"),
            delay_between_refills_secs=0.01,
            randomize_display=False,
        )
        algo = IcebergAlgorithm(config, simulated_client)

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
        )

        progress = await algo.execute(order)

        assert progress.state == AlgorithmState.COMPLETED
        # ~10 refills expected
        assert algo.num_refills >= 10

    @pytest.mark.asyncio
    async def test_iceberg_with_randomization(
        self,
        simulated_client: SimulatedExecutionClient,
        sample_order: Order,
    ) -> None:
        """Test Iceberg with display quantity randomization."""
        config = IcebergConfig(
            display_pct=0.25,
            randomize_display=True,
            randomization_pct=0.2,
            delay_between_refills_secs=0.01,
        )
        algo = IcebergAlgorithm(config, simulated_client)

        progress = await algo.execute(sample_order)

        assert progress.state == AlgorithmState.COMPLETED
        assert algo.num_refills >= 3  # At least some refills


# =============================================================================
# Registry E2E Tests
# =============================================================================


class TestRegistryE2E:
    """E2E tests for algorithm registry."""

    def test_list_algorithms_returns_all(self) -> None:
        """Test that all algorithms are listed."""
        algos = list_algorithms()
        assert "twap" in algos
        assert "vwap" in algos
        assert "iceberg" in algos

    @pytest.mark.asyncio
    async def test_create_and_execute_twap(
        self,
        simulated_client: SimulatedExecutionClient,
        sample_order: Order,
    ) -> None:
        """Test creating and executing TWAP via registry."""
        algo = create_algorithm(
            "twap",
            execution_client=simulated_client,
            horizon_secs=0.1,
            interval_secs=0.05,
        )

        progress = await algo.execute(sample_order)
        assert progress.state == AlgorithmState.COMPLETED

    @pytest.mark.asyncio
    async def test_create_and_execute_vwap(
        self,
        simulated_client: SimulatedExecutionClient,
        sample_order: Order,
    ) -> None:
        """Test creating and executing VWAP via registry."""
        algo = create_algorithm(
            "vwap",
            execution_client=simulated_client,
            num_intervals=4,
            interval_secs=0.01,
        )

        progress = await algo.execute(sample_order)
        assert progress.state == AlgorithmState.COMPLETED

    @pytest.mark.asyncio
    async def test_create_and_execute_iceberg(
        self,
        simulated_client: SimulatedExecutionClient,
        sample_order: Order,
    ) -> None:
        """Test creating and executing Iceberg via registry."""
        algo = create_algorithm(
            "iceberg",
            execution_client=simulated_client,
            display_pct=0.25,
            delay_between_refills_secs=0.01,
        )

        progress = await algo.execute(sample_order)
        assert progress.state == AlgorithmState.COMPLETED


# =============================================================================
# Execution Quality Tests
# =============================================================================


class TestExecutionQuality:
    """Tests for execution quality metrics."""

    @pytest.mark.asyncio
    async def test_twap_execution_spread(
        self,
        sample_order: Order,
    ) -> None:
        """Test that TWAP spreads execution across time."""
        client = SimulatedExecutionClient()

        config = TWAPConfig(
            horizon_secs=0.1,
            interval_secs=0.02,
            randomize_size=False,
            randomize_delay=False,
        )
        algo = TWAPAlgorithm(config, client)

        await algo.execute(sample_order)

        # Should have 5 orders
        assert len(client.orders_submitted) == 5

        # Each slice should be approximately equal (except last which absorbs remainder)
        slice_sizes = [o.amount for o in client.orders_submitted]
        expected_size = sample_order.amount / 5

        # Check first 4 slices are close to expected
        for size in slice_sizes[:-1]:
            # Allow small variance due to rounding
            assert abs(size - expected_size) < Decimal("1")

        # Last slice handles remainder - just check it's positive
        assert slice_sizes[-1] > 0

    @pytest.mark.asyncio
    async def test_vwap_follows_volume_profile(
        self,
        sample_order: Order,
        historical_bars: list[Bar],
    ) -> None:
        """Test that VWAP follows the volume profile distribution."""
        client = SimulatedExecutionClient()

        config = VWAPConfig(
            num_intervals=6,
            interval_secs=0.01,
            randomize_size=False,
        )
        algo = VWAPAlgorithm(config, client)
        profile = algo.load_volume_profile(historical_bars)

        await algo.execute(sample_order)

        # Check that order sizes follow volume profile
        assert len(client.orders_submitted) == 6

        # First interval should have larger order (higher volume at start)
        first_order_size = client.orders_submitted[0].amount
        # Middle intervals should have smaller orders
        middle_order_size = client.orders_submitted[2].amount

        # Volume profile is U-shaped, so ends should be larger than middle
        assert first_order_size > middle_order_size or \
               client.orders_submitted[-1].amount > middle_order_size

    @pytest.mark.asyncio
    async def test_iceberg_hides_order_size(
        self,
        sample_order: Order,
    ) -> None:
        """Test that Iceberg hides true order size."""
        client = SimulatedExecutionClient()

        config = IcebergConfig(
            display_pct=0.1,  # Only show 10%
            delay_between_refills_secs=0.001,
            randomize_display=False,
        )
        algo = IcebergAlgorithm(config, client)

        await algo.execute(sample_order)

        # Each individual order should be much smaller than total
        for order in client.orders_submitted:
            # Each order should be around 10% of total (±some variance)
            assert order.amount <= sample_order.amount * Decimal("0.15")

        # But total should still fill the order
        assert client.total_filled > sample_order.amount * Decimal("0.9")


# =============================================================================
# Stress Tests
# =============================================================================


class TestStress:
    """Stress tests for execution algorithms."""

    @pytest.mark.asyncio
    async def test_large_order_twap(self) -> None:
        """Test TWAP with large order."""
        client = SimulatedExecutionClient()

        large_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("10000"),  # Large order
        )

        config = TWAPConfig(
            horizon_secs=0.5,
            interval_secs=0.01,
            randomize_size=True,
        )
        algo = TWAPAlgorithm(config, client)

        progress = await algo.execute(large_order)

        assert progress.state == AlgorithmState.COMPLETED
        assert progress.num_children_spawned == 50  # 500ms / 10ms

    @pytest.mark.asyncio
    async def test_many_intervals_vwap(self) -> None:
        """Test VWAP with many intervals."""
        client = SimulatedExecutionClient()

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1000"),
        )

        config = VWAPConfig(
            num_intervals=20,
            interval_secs=0.01,
        )
        algo = VWAPAlgorithm(config, client)

        progress = await algo.execute(order)

        assert progress.state == AlgorithmState.COMPLETED
        assert progress.num_children_spawned == 20

    @pytest.mark.asyncio
    async def test_small_display_iceberg(self) -> None:
        """Test Iceberg with very small display quantity."""
        client = SimulatedExecutionClient()

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
        )

        config = IcebergConfig(
            display_pct=0.01,  # Only 1% visible
            delay_between_refills_secs=0.001,
            randomize_display=False,
        )
        algo = IcebergAlgorithm(config, client)

        progress = await algo.execute(order)

        assert progress.state == AlgorithmState.COMPLETED
        # ~100 refills expected (100 / 1%)
        assert algo.num_refills >= 90
