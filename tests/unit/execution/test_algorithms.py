"""
Unit tests for Execution Algorithms.

Tests for Issue #36: Execution Algorithm Framework (TWAP, VWAP).
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from libra.execution.algorithm import (
    AlgorithmState,
    BaseExecAlgorithm,
    ExecutionMetrics,
    ExecutionProgress,
)
from libra.execution.iceberg import IcebergAlgorithm, IcebergConfig, create_iceberg
from libra.execution.registry import (
    AlgorithmRegistry,
    create_algorithm,
    get_algorithm_registry,
    list_algorithms,
)
from libra.execution.twap import TWAPAlgorithm, TWAPConfig, create_twap
from libra.execution.vwap import VWAPAlgorithm, VWAPConfig, VolumeProfile, create_vwap
from libra.gateways.fetcher import Bar
from libra.gateways.protocol import Order, OrderResult, OrderSide, OrderStatus, OrderType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_execution_client() -> MagicMock:
    """Create a mock execution client."""
    import time

    client = MagicMock()
    client.submit_order = AsyncMock(
        return_value=OrderResult(
            order_id="test-order-1",
            symbol="BTC/USDT",
            status=OrderStatus.FILLED,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("10"),
            filled_amount=Decimal("10"),
            remaining_amount=Decimal("0"),
            average_price=Decimal("42000"),
            fee=Decimal("0.001"),
            fee_currency="BTC",
            timestamp_ns=time.time_ns(),
            client_order_id="test-client-1",
        )
    )
    return client


@pytest.fixture
def sample_order() -> Order:
    """Create a sample order for testing."""
    return Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("100"),
        client_order_id="test-parent-1",
    )


@pytest.fixture
def sample_bars() -> list[Bar]:
    """Create sample bar data for VWAP volume profiling."""
    import time

    base_ts = int(time.time() * 1_000_000_000)
    bars = []
    volumes = [100, 200, 150, 300, 250, 180, 220, 280, 320, 200, 150, 100]

    for i, vol in enumerate(volumes):
        bars.append(
            Bar(
                symbol="BTC/USDT",
                timestamp_ns=base_ts + i * 3600_000_000_000,  # 1 hour apart
                open=Decimal("42000"),
                high=Decimal("42500"),
                low=Decimal("41800"),
                close=Decimal("42300"),
                volume=Decimal(str(vol)),
            )
        )
    return bars


# =============================================================================
# ExecutionProgress Tests
# =============================================================================


class TestExecutionProgress:
    """Tests for ExecutionProgress dataclass."""

    def test_init_sets_remaining(self) -> None:
        """Test that remaining is set to total on init."""
        progress = ExecutionProgress(
            parent_order_id="test-1",
            total_quantity=Decimal("100"),
        )
        assert progress.remaining_quantity == Decimal("100")

    def test_completion_pct_zero(self) -> None:
        """Test completion percentage at start."""
        progress = ExecutionProgress(
            parent_order_id="test-1",
            total_quantity=Decimal("100"),
        )
        assert progress.completion_pct == 0.0

    def test_completion_pct_partial(self) -> None:
        """Test completion percentage when partially filled."""
        progress = ExecutionProgress(
            parent_order_id="test-1",
            total_quantity=Decimal("100"),
            executed_quantity=Decimal("50"),
        )
        assert progress.completion_pct == 50.0

    def test_completion_pct_full(self) -> None:
        """Test completion percentage when fully filled."""
        progress = ExecutionProgress(
            parent_order_id="test-1",
            total_quantity=Decimal("100"),
            executed_quantity=Decimal("100"),
        )
        assert progress.completion_pct == 100.0

    def test_completion_pct_zero_total(self) -> None:
        """Test completion percentage with zero total."""
        progress = ExecutionProgress(
            parent_order_id="test-1",
            total_quantity=Decimal("0"),
        )
        assert progress.completion_pct == 100.0

    def test_is_complete_remaining(self) -> None:
        """Test is_complete when there's remaining."""
        progress = ExecutionProgress(
            parent_order_id="test-1",
            total_quantity=Decimal("100"),
            remaining_quantity=Decimal("50"),
        )
        assert not progress.is_complete

    def test_is_complete_no_remaining(self) -> None:
        """Test is_complete when fully filled."""
        progress = ExecutionProgress(
            parent_order_id="test-1",
            total_quantity=Decimal("100"),
            executed_quantity=Decimal("100"),
            remaining_quantity=Decimal("0"),
            state=AlgorithmState.COMPLETED,
        )
        assert progress.is_complete

    def test_is_complete_cancelled(self) -> None:
        """Test is_complete when cancelled."""
        progress = ExecutionProgress(
            parent_order_id="test-1",
            total_quantity=Decimal("100"),
            remaining_quantity=Decimal("50"),
            state=AlgorithmState.CANCELLED,
        )
        assert progress.is_complete


# =============================================================================
# ExecutionMetrics Tests
# =============================================================================


class TestExecutionMetrics:
    """Tests for ExecutionMetrics dataclass."""

    def test_calculate_shortfall_buy(self) -> None:
        """Test implementation shortfall calculation for buy."""
        metrics = ExecutionMetrics(
            arrival_price=Decimal("100"),
            avg_execution_price=Decimal("102"),
        )
        metrics.calculate_shortfall(OrderSide.BUY)
        assert metrics.implementation_shortfall == Decimal("2")

    def test_calculate_shortfall_sell(self) -> None:
        """Test implementation shortfall calculation for sell."""
        metrics = ExecutionMetrics(
            arrival_price=Decimal("100"),
            avg_execution_price=Decimal("98"),
        )
        metrics.calculate_shortfall(OrderSide.SELL)
        assert metrics.implementation_shortfall == Decimal("2")

    def test_calculate_shortfall_no_avg_price(self) -> None:
        """Test shortfall calculation without average price."""
        metrics = ExecutionMetrics(arrival_price=Decimal("100"))
        metrics.calculate_shortfall(OrderSide.BUY)
        assert metrics.implementation_shortfall is None


# =============================================================================
# TWAPConfig Tests
# =============================================================================


class TestTWAPConfig:
    """Tests for TWAPConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TWAPConfig()
        assert config.horizon_secs == 60.0
        assert config.interval_secs == 5.0
        assert config.randomize_size is True
        assert config.randomize_delay is True
        assert config.randomization_pct == 0.1

    def test_num_slices_calculation(self) -> None:
        """Test number of slices calculation."""
        config = TWAPConfig(horizon_secs=120, interval_secs=10)
        assert config.num_slices == 12

    def test_num_slices_minimum(self) -> None:
        """Test minimum of 1 slice when interval equals horizon."""
        config = TWAPConfig(horizon_secs=10, interval_secs=10)
        assert config.num_slices == 1

    def test_invalid_horizon(self) -> None:
        """Test validation of horizon_secs."""
        with pytest.raises(ValueError, match="horizon_secs must be positive"):
            TWAPConfig(horizon_secs=0)

    def test_invalid_interval(self) -> None:
        """Test validation of interval_secs."""
        with pytest.raises(ValueError, match="interval_secs must be positive"):
            TWAPConfig(interval_secs=0)

    def test_interval_exceeds_horizon(self) -> None:
        """Test validation when interval > horizon."""
        with pytest.raises(ValueError, match="interval_secs cannot exceed horizon_secs"):
            TWAPConfig(horizon_secs=10, interval_secs=20)

    def test_invalid_randomization(self) -> None:
        """Test validation of randomization_pct."""
        with pytest.raises(ValueError, match="randomization_pct must be between"):
            TWAPConfig(randomization_pct=1.5)


# =============================================================================
# TWAPAlgorithm Tests
# =============================================================================


class TestTWAPAlgorithm:
    """Tests for TWAPAlgorithm."""

    def test_algorithm_id(self) -> None:
        """Test algorithm ID property."""
        algo = TWAPAlgorithm()
        assert algo.algorithm_id == "twap"

    def test_config_property(self) -> None:
        """Test config property."""
        config = TWAPConfig(horizon_secs=120)
        algo = TWAPAlgorithm(config)
        assert algo.config.horizon_secs == 120

    @pytest.mark.asyncio
    async def test_execute_no_client(self, sample_order: Order) -> None:
        """Test execute raises without execution client."""
        algo = TWAPAlgorithm()
        with pytest.raises(RuntimeError, match="Execution client not set"):
            await algo.execute(sample_order)

    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        mock_execution_client: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test successful TWAP execution."""
        config = TWAPConfig(
            horizon_secs=0.1,  # Very short for testing
            interval_secs=0.05,
            randomize_size=False,
            randomize_delay=False,
        )
        algo = TWAPAlgorithm(config, mock_execution_client)

        progress = await algo.execute(sample_order)

        assert progress is not None
        assert progress.state == AlgorithmState.COMPLETED
        assert progress.num_children_spawned > 0
        assert mock_execution_client.submit_order.called

    @pytest.mark.asyncio
    async def test_cancel(
        self,
        mock_execution_client: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test cancellation."""
        config = TWAPConfig(
            horizon_secs=10,
            interval_secs=1,
        )
        algo = TWAPAlgorithm(config, mock_execution_client)

        # Start execution in background
        task = asyncio.create_task(algo.execute(sample_order))

        # Cancel after brief delay
        await asyncio.sleep(0.05)
        cancelled = await algo.cancel()

        assert cancelled is True

        # Wait for completion
        progress = await task
        assert progress.state == AlgorithmState.CANCELLED


class TestCreateTwap:
    """Tests for create_twap factory function."""

    def test_creates_algorithm(self) -> None:
        """Test factory creates algorithm."""
        algo = create_twap(horizon_secs=120, interval_secs=10)
        assert isinstance(algo, TWAPAlgorithm)
        assert algo.config.horizon_secs == 120
        assert algo.config.interval_secs == 10

    def test_with_execution_client(self, mock_execution_client: MagicMock) -> None:
        """Test factory with execution client."""
        algo = create_twap(execution_client=mock_execution_client)
        assert algo._execution_client is mock_execution_client


# =============================================================================
# VWAPConfig Tests
# =============================================================================


class TestVWAPConfig:
    """Tests for VWAPConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = VWAPConfig()
        assert config.num_intervals == 12
        assert config.interval_secs == 300.0
        assert config.max_participation_pct == 0.01
        assert config.use_adaptive is True

    def test_horizon_secs_property(self) -> None:
        """Test horizon calculation."""
        config = VWAPConfig(num_intervals=12, interval_secs=300)
        assert config.horizon_secs == 3600  # 12 * 300

    def test_invalid_num_intervals(self) -> None:
        """Test validation of num_intervals."""
        with pytest.raises(ValueError, match="num_intervals must be positive"):
            VWAPConfig(num_intervals=0)

    def test_invalid_interval_secs(self) -> None:
        """Test validation of interval_secs."""
        with pytest.raises(ValueError, match="interval_secs must be positive"):
            VWAPConfig(interval_secs=0)

    def test_invalid_max_participation(self) -> None:
        """Test validation of max_participation_pct."""
        with pytest.raises(ValueError, match="max_participation_pct must be between"):
            VWAPConfig(max_participation_pct=1.5)


# =============================================================================
# VolumeProfile Tests
# =============================================================================


class TestVolumeProfile:
    """Tests for VolumeProfile dataclass."""

    def test_normalizes_fractions(self) -> None:
        """Test that fractions are normalized to sum to 1.0."""
        profile = VolumeProfile(fractions=[1, 2, 3, 4])
        assert abs(sum(profile.fractions) - 1.0) < 0.0001

    def test_empty_fractions(self) -> None:
        """Test with empty fractions."""
        profile = VolumeProfile(fractions=[])
        assert profile.fractions == []

    def test_zero_fractions(self) -> None:
        """Test with zero fractions."""
        profile = VolumeProfile(fractions=[0, 0, 0])
        # Should stay as zeros (can't normalize)
        assert profile.fractions == [0, 0, 0]


# =============================================================================
# VWAPAlgorithm Tests
# =============================================================================


class TestVWAPAlgorithm:
    """Tests for VWAPAlgorithm."""

    def test_algorithm_id(self) -> None:
        """Test algorithm ID property."""
        algo = VWAPAlgorithm()
        assert algo.algorithm_id == "vwap"

    def test_vwap_property_initial(self) -> None:
        """Test initial VWAP value."""
        algo = VWAPAlgorithm()
        assert algo.vwap == Decimal("0")

    def test_update_vwap(self) -> None:
        """Test VWAP calculation."""
        algo = VWAPAlgorithm()

        # First trade: price=100, volume=10
        vwap = algo.update_vwap(Decimal("100"), Decimal("10"))
        assert vwap == Decimal("100")

        # Second trade: price=110, volume=10
        # VWAP = (100*10 + 110*10) / 20 = 2100 / 20 = 105
        vwap = algo.update_vwap(Decimal("110"), Decimal("10"))
        assert vwap == Decimal("105")

    def test_load_volume_profile(self, sample_bars: list[Bar]) -> None:
        """Test loading volume profile from bars."""
        config = VWAPConfig(num_intervals=4)
        algo = VWAPAlgorithm(config)

        profile = algo.load_volume_profile(sample_bars)

        assert profile is not None
        assert len(profile.fractions) == 4
        assert abs(sum(profile.fractions) - 1.0) < 0.0001
        assert profile.num_bars == len(sample_bars)

    def test_load_volume_profile_empty(self) -> None:
        """Test loading volume profile with empty bars."""
        algo = VWAPAlgorithm()
        profile = algo.load_volume_profile([])

        # Should create equal distribution
        assert profile is not None
        assert len(profile.fractions) == 12  # Default num_intervals
        assert abs(sum(profile.fractions) - 1.0) < 0.0001

    def test_is_price_favorable_buy(self) -> None:
        """Test price favorability for buy orders."""
        algo = VWAPAlgorithm()
        algo.update_vwap(Decimal("100"), Decimal("10"))

        # Price below VWAP is favorable for buy
        assert algo.is_price_favorable(Decimal("95"), OrderSide.BUY) is True
        assert algo.is_price_favorable(Decimal("100"), OrderSide.BUY) is True
        assert algo.is_price_favorable(Decimal("105"), OrderSide.BUY) is False

    def test_is_price_favorable_sell(self) -> None:
        """Test price favorability for sell orders."""
        algo = VWAPAlgorithm()
        algo.update_vwap(Decimal("100"), Decimal("10"))

        # Price above VWAP is favorable for sell
        assert algo.is_price_favorable(Decimal("105"), OrderSide.SELL) is True
        assert algo.is_price_favorable(Decimal("100"), OrderSide.SELL) is True
        assert algo.is_price_favorable(Decimal("95"), OrderSide.SELL) is False

    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        mock_execution_client: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test successful VWAP execution."""
        config = VWAPConfig(
            num_intervals=2,
            interval_secs=0.05,
        )
        algo = VWAPAlgorithm(config, mock_execution_client)

        progress = await algo.execute(sample_order)

        assert progress is not None
        assert progress.state == AlgorithmState.COMPLETED
        assert progress.num_children_spawned > 0


class TestCreateVwap:
    """Tests for create_vwap factory function."""

    def test_creates_algorithm(self) -> None:
        """Test factory creates algorithm."""
        algo = create_vwap(num_intervals=6, interval_secs=600)
        assert isinstance(algo, VWAPAlgorithm)
        assert algo.config.num_intervals == 6
        assert algo.config.interval_secs == 600


# =============================================================================
# IcebergConfig Tests
# =============================================================================


class TestIcebergConfig:
    """Tests for IcebergConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = IcebergConfig()
        assert config.display_qty is None
        assert config.display_pct == 0.05
        assert config.randomize_display is True
        assert config.randomization_pct == 0.2

    def test_invalid_display_qty(self) -> None:
        """Test validation of display_qty."""
        with pytest.raises(ValueError, match="display_qty must be positive"):
            IcebergConfig(display_qty=Decimal("0"))

    def test_invalid_display_pct(self) -> None:
        """Test validation of display_pct."""
        with pytest.raises(ValueError, match="display_pct must be between"):
            IcebergConfig(display_pct=1.5)

    def test_invalid_refill_threshold(self) -> None:
        """Test validation of refill_threshold_pct."""
        with pytest.raises(ValueError, match="refill_threshold_pct must be between"):
            IcebergConfig(refill_threshold_pct=1.5)


# =============================================================================
# IcebergAlgorithm Tests
# =============================================================================


class TestIcebergAlgorithm:
    """Tests for IcebergAlgorithm."""

    def test_algorithm_id(self) -> None:
        """Test algorithm ID property."""
        algo = IcebergAlgorithm()
        assert algo.algorithm_id == "iceberg"

    def test_num_refills_initial(self) -> None:
        """Test initial refill count."""
        algo = IcebergAlgorithm()
        assert algo.num_refills == 0

    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        mock_execution_client: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test successful iceberg execution."""
        config = IcebergConfig(
            display_pct=0.2,  # 20% visible
            delay_between_refills_secs=0.01,
        )
        algo = IcebergAlgorithm(config, mock_execution_client)

        progress = await algo.execute(sample_order)

        assert progress is not None
        assert progress.state == AlgorithmState.COMPLETED
        assert algo.num_refills > 0
        assert mock_execution_client.submit_order.called

    @pytest.mark.asyncio
    async def test_execute_with_fixed_display_qty(
        self,
        mock_execution_client: MagicMock,
    ) -> None:
        """Test iceberg with fixed display quantity."""
        config = IcebergConfig(
            display_qty=Decimal("10"),
            delay_between_refills_secs=0.01,
            randomize_display=False,
        )
        algo = IcebergAlgorithm(config, mock_execution_client)

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
        )

        progress = await algo.execute(order)

        assert progress is not None
        # Should have ~10 refills (100 / 10)
        assert algo.num_refills >= 10


class TestCreateIceberg:
    """Tests for create_iceberg factory function."""

    def test_creates_algorithm(self) -> None:
        """Test factory creates algorithm."""
        algo = create_iceberg(display_pct=0.1, randomize=True)
        assert isinstance(algo, IcebergAlgorithm)
        assert algo.config.display_pct == 0.1
        assert algo.config.randomize_display is True

    def test_with_fixed_display_qty(self) -> None:
        """Test factory with fixed display quantity."""
        algo = create_iceberg(display_qty=Decimal("5"))
        assert algo.config.display_qty == Decimal("5")


# =============================================================================
# AlgorithmRegistry Tests
# =============================================================================


class TestAlgorithmRegistry:
    """Tests for AlgorithmRegistry."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving algorithms."""
        registry = AlgorithmRegistry()
        registry.register("test", TWAPAlgorithm)

        result = registry.get("test")
        assert result is TWAPAlgorithm

    def test_register_duplicate(self) -> None:
        """Test registering duplicate name raises error."""
        registry = AlgorithmRegistry()
        registry.register("test", TWAPAlgorithm)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", VWAPAlgorithm)

    def test_unregister(self) -> None:
        """Test unregistering algorithm."""
        registry = AlgorithmRegistry()
        registry.register("test", TWAPAlgorithm)

        assert registry.unregister("test") is True
        assert registry.get("test") is None

    def test_unregister_not_found(self) -> None:
        """Test unregistering non-existent algorithm."""
        registry = AlgorithmRegistry()
        assert registry.unregister("nonexistent") is False

    def test_list_algorithms(self) -> None:
        """Test listing registered algorithms."""
        registry = AlgorithmRegistry()
        registry.register("twap", TWAPAlgorithm)
        registry.register("vwap", VWAPAlgorithm)

        algos = registry.list_algorithms()
        assert "twap" in algos
        assert "vwap" in algos

    def test_contains(self) -> None:
        """Test __contains__ method."""
        registry = AlgorithmRegistry()
        registry.register("test", TWAPAlgorithm)

        assert "test" in registry
        assert "other" not in registry

    def test_len(self) -> None:
        """Test __len__ method."""
        registry = AlgorithmRegistry()
        assert len(registry) == 0

        registry.register("test", TWAPAlgorithm)
        assert len(registry) == 1

    def test_create(self, mock_execution_client: MagicMock) -> None:
        """Test creating algorithm instance."""
        registry = AlgorithmRegistry()
        registry.register("twap", TWAPAlgorithm)

        algo = registry.create(
            "twap",
            execution_client=mock_execution_client,
            horizon_secs=120,
        )

        assert isinstance(algo, TWAPAlgorithm)
        assert algo.config.horizon_secs == 120

    def test_create_not_found(self) -> None:
        """Test creating non-existent algorithm raises error."""
        registry = AlgorithmRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.create("nonexistent")


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_algorithm_registry(self) -> None:
        """Test getting global registry."""
        registry = get_algorithm_registry()
        assert isinstance(registry, AlgorithmRegistry)

        # Should have default algorithms
        assert "twap" in registry
        assert "vwap" in registry
        assert "iceberg" in registry

    def test_list_algorithms_function(self) -> None:
        """Test list_algorithms convenience function."""
        algos = list_algorithms()
        assert "twap" in algos
        assert "vwap" in algos
        assert "iceberg" in algos

    def test_create_algorithm_function(self) -> None:
        """Test create_algorithm convenience function."""
        algo = create_algorithm("twap", horizon_secs=60)
        assert isinstance(algo, TWAPAlgorithm)
        assert algo.config.horizon_secs == 60


# =============================================================================
# Base Algorithm Tests
# =============================================================================


class TestBaseExecAlgorithm:
    """Tests for BaseExecAlgorithm utility methods."""

    def test_randomize_quantity(self) -> None:
        """Test quantity randomization."""
        base_qty = Decimal("100")

        # Run multiple times to test randomization
        results = []
        for _ in range(100):
            qty = BaseExecAlgorithm._randomize_quantity(base_qty, 0.1)
            results.append(float(qty))

        # Should have variation
        assert min(results) < 100
        assert max(results) > 100
        # Should be within ±10%
        assert all(90 <= r <= 110 for r in results)

    def test_randomize_delay(self) -> None:
        """Test delay randomization."""
        base_delay = 1.0

        # Run multiple times
        results = []
        for _ in range(100):
            delay = BaseExecAlgorithm._randomize_delay(base_delay, 0.1)
            results.append(delay)

        # Should have variation
        assert min(results) < 1.0
        assert max(results) > 1.0
        # Should be within ±10%
        assert all(0.9 <= r <= 1.1 for r in results)

    def test_get_progress_initial(self) -> None:
        """Test get_progress before execution."""
        algo = TWAPAlgorithm()
        assert algo.get_progress() is None

    def test_get_metrics_initial(self) -> None:
        """Test get_metrics before execution."""
        algo = TWAPAlgorithm()
        assert algo.get_metrics() is None

    def test_set_execution_client(self, mock_execution_client: MagicMock) -> None:
        """Test setting execution client."""
        algo = TWAPAlgorithm()
        algo.set_execution_client(mock_execution_client)
        assert algo._execution_client is mock_execution_client
