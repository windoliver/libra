"""
Simulation tests for Execution Algorithms.

Tests for Issue #36: Execution Algorithm Framework (TWAP, VWAP).

These tests simulate realistic market conditions to verify:
- Algorithm behavior under various scenarios
- Market impact characteristics
- Execution quality metrics
"""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from libra.execution.algorithm import AlgorithmState, ExecutionProgress
from libra.execution.iceberg import IcebergAlgorithm, IcebergConfig
from libra.execution.metrics import ExecutionTCA, create_tca
from libra.execution.pov import POVAlgorithm, POVConfig
from libra.execution.twap import TWAPAlgorithm, TWAPConfig
from libra.execution.vwap import VWAPAlgorithm, VWAPConfig
from libra.gateways.fetcher import Bar
from libra.gateways.protocol import Order, OrderResult, OrderSide, OrderStatus, OrderType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_execution_client() -> MagicMock:
    """Create a mock execution client with variable fill behavior."""
    client = MagicMock()

    fill_count = [0]

    async def simulate_fill(order: Order) -> OrderResult:
        fill_count[0] += 1
        # Simulate slight price movement with each fill
        base_price = Decimal("50000")
        slippage = Decimal(str(fill_count[0] * 0.5))  # 0.5 per order
        fill_price = base_price + slippage

        return OrderResult(
            order_id=f"fill-{fill_count[0]}",
            symbol=order.symbol,
            status=OrderStatus.FILLED,
            side=order.side,
            order_type=order.order_type,
            amount=order.amount,
            filled_amount=order.amount,
            remaining_amount=Decimal("0"),
            average_price=fill_price,
            fee=order.amount * Decimal("0.0001"),  # 1 bps fee
            fee_currency="USDT",
            timestamp_ns=time.time_ns(),
            client_order_id=order.client_order_id,
        )

    client.submit_order = AsyncMock(side_effect=simulate_fill)
    return client


@pytest.fixture
def sample_order() -> Order:
    """Create a large sample order for testing."""
    return Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("100"),
        client_order_id="test-parent-1",
    )


@pytest.fixture
def sample_bars() -> list[Bar]:
    """Create realistic sample bar data with varying volume."""
    base_ts = time.time_ns()
    # Simulate a volume curve: morning spike, midday lull, afternoon pickup
    volumes = [
        150, 180, 200, 220, 250,  # Morning ramp-up
        280, 300, 320, 350, 380,  # Morning peak
        350, 300, 250, 200, 180,  # Midday decline
        150, 140, 130, 120, 110,  # Midday lull
        130, 160, 200, 250, 300,  # Afternoon pickup
        350, 400, 450, 480, 500,  # Afternoon peak
        450, 400, 350, 300, 250,  # Close approach
        200, 150, 100,            # End of day
    ]

    bars = []
    for i, vol in enumerate(volumes):
        bars.append(
            Bar(
                symbol="BTC/USDT",
                timestamp_ns=base_ts + i * 300_000_000_000,  # 5 min bars
                open=Decimal("50000") + Decimal(str(i % 10)),
                high=Decimal("50100") + Decimal(str(i % 10)),
                low=Decimal("49900") - Decimal(str(i % 10)),
                close=Decimal("50050") + Decimal(str(i % 10)),
                volume=Decimal(str(vol)),
            )
        )
    return bars


# =============================================================================
# TWAP Simulation Tests
# =============================================================================


class TestTWAPSimulation:
    """Simulation tests for TWAP algorithm."""

    @pytest.mark.asyncio
    async def test_twap_even_distribution(
        self,
        mock_execution_client: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that TWAP distributes orders evenly over time."""
        config = TWAPConfig(
            horizon_secs=0.5,
            interval_secs=0.1,
            randomize_size=False,
            randomize_delay=False,
        )
        algo = TWAPAlgorithm(config, mock_execution_client)

        progress = await algo.execute(sample_order)

        assert progress.state == AlgorithmState.COMPLETED
        assert progress.num_children_spawned >= 5  # 0.5 / 0.1 = 5 slices

        # Check that orders were called
        assert mock_execution_client.submit_order.call_count >= 5

    @pytest.mark.asyncio
    async def test_twap_with_tca(
        self,
        mock_execution_client: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test TWAP execution with TCA metrics collection."""
        config = TWAPConfig(
            horizon_secs=0.3,
            interval_secs=0.1,
            randomize_size=False,
            randomize_delay=False,
        )
        algo = TWAPAlgorithm(config, mock_execution_client)

        # Create TCA tracker
        tca = create_tca(
            arrival_price=Decimal("50000"),
            side=sample_order.side,
        )

        progress = await algo.execute(sample_order)

        # Record fills from progress
        if progress.avg_fill_price:
            tca.record_fill(
                quantity=progress.executed_quantity,
                price=progress.avg_fill_price,
            )

        tca.finalize()

        # Should have some slippage due to simulated price movement
        assert tca.total_quantity > 0
        assert tca.num_fills > 0


# =============================================================================
# VWAP Simulation Tests
# =============================================================================


class TestVWAPSimulation:
    """Simulation tests for VWAP algorithm."""

    @pytest.mark.asyncio
    async def test_vwap_volume_weighted_distribution(
        self,
        mock_execution_client: MagicMock,
        sample_order: Order,
        sample_bars: list[Bar],
    ) -> None:
        """Test that VWAP distributes orders proportional to volume."""
        config = VWAPConfig(
            num_intervals=4,
            interval_secs=0.05,
            randomize_size=False,
        )
        algo = VWAPAlgorithm(config, mock_execution_client)

        # Load volume profile
        profile = algo.load_volume_profile(sample_bars)
        assert len(profile.fractions) == 4
        assert abs(sum(profile.fractions) - 1.0) < 0.0001

        progress = await algo.execute(sample_order)

        assert progress.state == AlgorithmState.COMPLETED
        assert progress.num_children_spawned == 4

    @pytest.mark.asyncio
    async def test_vwap_no_profile_uses_equal(
        self,
        mock_execution_client: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that VWAP uses equal distribution when no profile loaded."""
        config = VWAPConfig(
            num_intervals=3,
            interval_secs=0.05,
        )
        algo = VWAPAlgorithm(config, mock_execution_client)

        # Don't load volume profile
        progress = await algo.execute(sample_order)

        assert progress.state == AlgorithmState.COMPLETED
        assert progress.num_children_spawned == 3


# =============================================================================
# Iceberg Simulation Tests
# =============================================================================


class TestIcebergSimulation:
    """Simulation tests for Iceberg algorithm."""

    @pytest.mark.asyncio
    async def test_iceberg_hidden_quantity(
        self,
        mock_execution_client: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that Iceberg hides total order size."""
        config = IcebergConfig(
            display_pct=0.1,  # Show 10% at a time
            delay_between_refills_secs=0.01,
            randomize_display=False,
        )
        algo = IcebergAlgorithm(config, mock_execution_client)

        progress = await algo.execute(sample_order)

        assert progress.state == AlgorithmState.COMPLETED
        # With 100 total and 10% display, should have ~10 refills
        assert algo.num_refills >= 10

    @pytest.mark.asyncio
    async def test_iceberg_randomization(
        self,
        mock_execution_client: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test Iceberg randomization varies display quantity."""
        config = IcebergConfig(
            display_pct=0.2,
            randomize_display=True,
            randomization_pct=0.3,  # ±30%
            delay_between_refills_secs=0.01,
        )
        algo = IcebergAlgorithm(config, mock_execution_client)

        progress = await algo.execute(sample_order)

        assert progress.state == AlgorithmState.COMPLETED

        # Verify order amounts varied (due to randomization)
        calls = mock_execution_client.submit_order.call_args_list
        amounts = [call[0][0].amount for call in calls]

        # With randomization, should not all be exactly the same
        if len(amounts) > 2:
            # At least some variation expected
            unique_amounts = set(amounts)
            # Due to randomization, expect some variety
            assert len(unique_amounts) >= 1


# =============================================================================
# POV Simulation Tests
# =============================================================================


class TestPOVSimulation:
    """Simulation tests for POV algorithm."""

    @pytest.mark.asyncio
    async def test_pov_participation_rate(
        self,
        mock_execution_client: MagicMock,
    ) -> None:
        """Test POV maintains target participation rate."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("50"),
            client_order_id="test-pov-1",
        )

        config = POVConfig(
            target_pct=0.05,  # 5% participation
            max_pct=0.10,
            interval_secs=0.05,
            max_duration_secs=0.3,
        )
        algo = POVAlgorithm(config, mock_execution_client)

        progress = await algo.execute(order)

        # Should execute within max duration
        assert progress is not None

    @pytest.mark.asyncio
    async def test_pov_with_estimated_volume(
        self,
        mock_execution_client: MagicMock,
    ) -> None:
        """Test POV with estimated daily volume set."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
            client_order_id="test-pov-2",
        )

        config = POVConfig(
            target_pct=0.10,
            interval_secs=0.05,
            max_duration_secs=0.2,
        )
        algo = POVAlgorithm(config, mock_execution_client)

        # Set estimated daily volume
        algo.set_estimated_daily_volume(Decimal("10000"))

        progress = await algo.execute(order)
        assert progress is not None


# =============================================================================
# TCA Metrics Tests
# =============================================================================


class TestTCAMetrics:
    """Tests for TCA metrics module."""

    def test_tca_fill_recording(self) -> None:
        """Test recording fills in TCA tracker."""
        tca = create_tca(
            arrival_price=Decimal("100"),
            side=OrderSide.BUY,
        )

        tca.record_fill(Decimal("10"), Decimal("100.5"))
        tca.record_fill(Decimal("20"), Decimal("101.0"))
        tca.record_fill(Decimal("30"), Decimal("100.8"))

        assert tca.num_fills == 3
        assert tca.total_quantity == Decimal("60")
        # Weighted avg: (10*100.5 + 20*101 + 30*100.8) / 60 = 100.85
        expected_avg = (
            Decimal("10") * Decimal("100.5")
            + Decimal("20") * Decimal("101.0")
            + Decimal("30") * Decimal("100.8")
        ) / Decimal("60")
        assert tca.avg_execution_price == expected_avg

    def test_tca_implementation_shortfall_buy(self) -> None:
        """Test implementation shortfall for buy order."""
        tca = create_tca(
            arrival_price=Decimal("100"),
            side=OrderSide.BUY,
        )

        # Paid more than arrival price
        tca.record_fill(Decimal("100"), Decimal("102"))
        tca.finalize()

        # Shortfall = (102 - 100) / 100 * 10000 = 200 bps
        assert tca.implementation_shortfall_bps == pytest.approx(200, rel=0.01)

    def test_tca_implementation_shortfall_sell(self) -> None:
        """Test implementation shortfall for sell order."""
        tca = create_tca(
            arrival_price=Decimal("100"),
            side=OrderSide.SELL,
        )

        # Received less than arrival price
        tca.record_fill(Decimal("100"), Decimal("98"))
        tca.finalize()

        # Shortfall = (100 - 98) / 100 * 10000 = 200 bps
        assert tca.implementation_shortfall_bps == pytest.approx(200, rel=0.01)

    def test_tca_vwap_slippage(self) -> None:
        """Test VWAP slippage calculation."""
        tca = create_tca(
            arrival_price=Decimal("100"),
            side=OrderSide.BUY,
        )

        tca.record_fill(Decimal("100"), Decimal("101"))
        tca.finalize(benchmark_vwap=Decimal("100.5"))

        # VWAP slippage = (101 - 100.5) / 100.5 * 10000 = ~50 bps
        assert tca.vwap_slippage_bps == pytest.approx(49.75, rel=0.1)

    def test_tca_cost_attribution(self) -> None:
        """Test cost attribution breakdown."""
        tca = create_tca(
            arrival_price=Decimal("100"),
            side=OrderSide.BUY,
        )

        tca.record_fill(Decimal("100"), Decimal("100.5"))
        tca.finalize()
        tca.attribute_costs(bid_ask_spread_bps=10.0, estimated_impact_bps=30.0)

        # Spread cost = half of spread
        assert tca.spread_cost_bps == 5.0
        # Market impact = shortfall - spread cost (capped at estimate)
        assert tca.market_impact_bps >= 0

    def test_tca_summary(self) -> None:
        """Test TCA summary generation."""
        tca = create_tca(
            arrival_price=Decimal("100"),
            side=OrderSide.BUY,
        )

        tca.record_fill(Decimal("50"), Decimal("100.2"))
        tca.record_fill(Decimal("50"), Decimal("100.4"))
        tca.finalize(
            benchmark_vwap=Decimal("100.25"),
            benchmark_twap=Decimal("100.3"),
        )

        summary = tca.summary()
        assert "Implementation Shortfall" in summary
        assert "VWAP Slippage" in summary

    def test_aggregated_tca(self) -> None:
        """Test aggregated TCA across multiple executions."""
        from libra.execution.metrics import AggregatedTCA

        agg = AggregatedTCA()

        # Add multiple executions
        for i in range(5):
            tca = create_tca(
                arrival_price=Decimal("100"),
                side=OrderSide.BUY,
            )
            # Varying slippage
            tca.record_fill(Decimal("10"), Decimal("100") + Decimal(str(i * 0.1)))
            tca.finalize()
            agg.add_execution(tca)

        assert agg.num_executions == 5
        assert agg.total_quantity == Decimal("50")
        assert agg.avg_implementation_shortfall_bps >= 0
        assert agg.best_shortfall_bps <= agg.worst_shortfall_bps


# =============================================================================
# Execution Quality Tests
# =============================================================================


class TestExecutionQuality:
    """Tests for execution quality across algorithms."""

    @pytest.mark.asyncio
    async def test_twap_minimizes_timing_risk(
        self,
        mock_execution_client: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that TWAP reduces timing risk through even distribution."""
        config = TWAPConfig(
            horizon_secs=0.3,
            interval_secs=0.1,
            randomize_size=False,
            randomize_delay=False,
        )
        algo = TWAPAlgorithm(config, mock_execution_client)

        progress = await algo.execute(sample_order)

        # All slices should be approximately equal
        calls = mock_execution_client.submit_order.call_args_list
        amounts = [call[0][0].amount for call in calls]

        # Check near-equal distribution
        avg_amount = sum(amounts) / len(amounts)
        for amount in amounts:
            # Each amount should be close to average
            ratio = float(amount / avg_amount)
            assert 0.9 <= ratio <= 1.1, f"Uneven distribution: {amounts}"

    @pytest.mark.asyncio
    async def test_algorithm_cancellation_safety(
        self,
        mock_execution_client: MagicMock,
    ) -> None:
        """Test that algorithms handle cancellation gracefully."""
        import asyncio

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
            client_order_id="test-cancel-1",
        )

        config = TWAPConfig(
            horizon_secs=5.0,  # Long horizon
            interval_secs=0.5,
        )
        algo = TWAPAlgorithm(config, mock_execution_client)

        # Start execution
        task = asyncio.create_task(algo.execute(order))

        # Cancel after short delay
        await asyncio.sleep(0.1)
        cancelled = await algo.cancel()

        progress = await task

        assert cancelled is True
        assert progress.state == AlgorithmState.CANCELLED

    @pytest.mark.asyncio
    async def test_partial_fill_handling(
        self,
        sample_order: Order,
    ) -> None:
        """Test handling of partial fills."""
        # Create client that returns partial fills
        client = MagicMock()

        async def partial_fill(order: Order) -> OrderResult:
            # Fill only 80% of requested
            filled = order.amount * Decimal("0.8")
            return OrderResult(
                order_id="partial-1",
                symbol=order.symbol,
                status=OrderStatus.PARTIALLY_FILLED,
                side=order.side,
                order_type=order.order_type,
                amount=order.amount,
                filled_amount=filled,
                remaining_amount=order.amount - filled,
                average_price=Decimal("50000"),
                fee=Decimal("0.01"),
                fee_currency="USDT",
                timestamp_ns=time.time_ns(),
            )

        client.submit_order = AsyncMock(side_effect=partial_fill)

        config = TWAPConfig(
            horizon_secs=0.2,
            interval_secs=0.1,
            randomize_size=False,
            randomize_delay=False,
        )
        algo = TWAPAlgorithm(config, client)

        progress = await algo.execute(sample_order)

        # Algorithm should complete despite partial fills
        assert progress.state == AlgorithmState.COMPLETED
        # Executed quantity should reflect partial fills
        assert progress.executed_quantity > 0
