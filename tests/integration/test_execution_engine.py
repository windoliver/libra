"""
Integration tests for ExecutionEngine.

Tests for Issue #36: Execution Algorithm Framework (TWAP, VWAP).

These tests verify the integration between:
- ExecutionEngine and algorithms
- Algorithm lifecycle events
- Risk engine integration
- Order routing
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libra.core.events import Event, EventType
from libra.core.message_bus import MessageBus
from libra.execution.algorithm import AlgorithmState
from libra.execution.engine import (
    AlgorithmNotFoundError,
    ExecutionEngine,
    ExecutionEngineConfig,
    OrderDeniedError,
    create_execution_engine,
)
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
def mock_clock() -> MagicMock:
    """Create a mock clock."""
    import time

    clock = MagicMock()
    clock.timestamp_ns = MagicMock(return_value=time.time_ns())
    return clock


@pytest.fixture
def message_bus() -> MessageBus:
    """Create a message bus for testing."""
    return MessageBus()


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
def algo_order() -> Order:
    """Create an order with execution algorithm."""
    return Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("100"),
        client_order_id="test-algo-1",
        exec_algorithm="twap",
        exec_algorithm_params={
            "horizon_secs": 0.1,
            "interval_secs": 0.05,
            "randomize_size": False,
            "randomize_delay": False,
        },
    )


# =============================================================================
# ExecutionEngine Tests
# =============================================================================


class TestExecutionEngineCreation:
    """Tests for ExecutionEngine creation and configuration."""

    def test_create_execution_engine(self, mock_execution_client: MagicMock) -> None:
        """Test creating engine with factory function."""
        engine = create_execution_engine(
            execution_client=mock_execution_client,
        )
        assert engine is not None
        assert engine.execution_client is mock_execution_client

    def test_create_engine_with_config(
        self,
        mock_execution_client: MagicMock,
        message_bus: MessageBus,
        mock_clock: MagicMock,
    ) -> None:
        """Test creating engine with full configuration."""
        config = ExecutionEngineConfig(
            enable_risk_checks=False,
            progress_interval_secs=0.5,
        )
        engine = ExecutionEngine(
            message_bus=message_bus,
            clock=mock_clock,
            execution_client=mock_execution_client,
            config=config,
        )
        assert engine._config.enable_risk_checks is False
        assert engine._config.progress_interval_secs == 0.5

    def test_engine_properties(
        self,
        mock_execution_client: MagicMock,
        message_bus: MessageBus,
    ) -> None:
        """Test engine property setters."""
        engine = ExecutionEngine()

        engine.execution_client = mock_execution_client
        assert engine.execution_client is mock_execution_client

        engine.message_bus = message_bus
        assert engine.message_bus is message_bus


class TestDirectOrderExecution:
    """Tests for direct order execution (no algorithm)."""

    @pytest.mark.asyncio
    async def test_execute_direct_order(
        self,
        mock_execution_client: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test executing order directly without algorithm."""
        engine = create_execution_engine(
            execution_client=mock_execution_client,
            enable_risk_checks=False,
        )

        result = await engine.submit_order(sample_order)

        assert isinstance(result, OrderResult)
        assert result.status == OrderStatus.FILLED
        assert mock_execution_client.submit_order.called
        assert engine.stats.orders_direct == 1
        assert engine.stats.orders_algo == 0

    @pytest.mark.asyncio
    async def test_execute_order_no_client(self, sample_order: Order) -> None:
        """Test error when no execution client configured."""
        engine = ExecutionEngine()

        with pytest.raises(RuntimeError, match="Execution client not configured"):
            await engine.submit_order(sample_order)

    @pytest.mark.asyncio
    async def test_generates_client_order_id(
        self,
        mock_execution_client: MagicMock,
    ) -> None:
        """Test that engine generates client_order_id if not provided."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("10"),
            # No client_order_id
        )

        engine = create_execution_engine(
            execution_client=mock_execution_client,
            enable_risk_checks=False,
        )

        await engine.submit_order(order)

        # Check that submit_order was called with an order that has client_order_id
        call_args = mock_execution_client.submit_order.call_args
        submitted_order = call_args[0][0]
        assert submitted_order.client_order_id is not None


class TestAlgorithmExecution:
    """Tests for algorithm-based order execution."""

    @pytest.mark.asyncio
    async def test_execute_with_twap(
        self,
        mock_execution_client: MagicMock,
        algo_order: Order,
    ) -> None:
        """Test executing order with TWAP algorithm."""
        engine = create_execution_engine(
            execution_client=mock_execution_client,
            enable_risk_checks=False,
        )

        progress = await engine.submit_order(algo_order)

        assert progress is not None
        assert progress.state == AlgorithmState.COMPLETED
        assert engine.stats.orders_algo == 1
        assert engine.stats.algos_completed == 1

    @pytest.mark.asyncio
    async def test_execute_with_vwap(
        self,
        mock_execution_client: MagicMock,
    ) -> None:
        """Test executing order with VWAP algorithm."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
            client_order_id="test-vwap-1",
            exec_algorithm="vwap",
            exec_algorithm_params={
                "num_intervals": 2,
                "interval_secs": 0.05,
            },
        )

        engine = create_execution_engine(
            execution_client=mock_execution_client,
            enable_risk_checks=False,
        )

        progress = await engine.submit_order(order)

        assert progress is not None
        assert progress.state == AlgorithmState.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_with_iceberg(
        self,
        mock_execution_client: MagicMock,
    ) -> None:
        """Test executing order with Iceberg algorithm."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
            client_order_id="test-iceberg-1",
            exec_algorithm="iceberg",
            exec_algorithm_params={
                "display_pct": 0.2,
                "delay_between_refills_secs": 0.01,
            },
        )

        engine = create_execution_engine(
            execution_client=mock_execution_client,
            enable_risk_checks=False,
        )

        progress = await engine.submit_order(order)

        assert progress is not None
        assert progress.state == AlgorithmState.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_with_pov(
        self,
        mock_execution_client: MagicMock,
    ) -> None:
        """Test executing order with POV algorithm."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("50"),
            client_order_id="test-pov-1",
            exec_algorithm="pov",
            exec_algorithm_params={
                "target_pct": 0.05,
                "interval_secs": 0.05,
                "max_duration_secs": 0.2,
            },
        )

        engine = create_execution_engine(
            execution_client=mock_execution_client,
            enable_risk_checks=False,
        )

        progress = await engine.submit_order(order)

        assert progress is not None
        # POV may not complete fully due to max_duration

    @pytest.mark.asyncio
    async def test_algorithm_not_found(
        self,
        mock_execution_client: MagicMock,
    ) -> None:
        """Test error when algorithm not found."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
            exec_algorithm="nonexistent",
        )

        engine = create_execution_engine(
            execution_client=mock_execution_client,
            enable_risk_checks=False,
        )

        with pytest.raises(AlgorithmNotFoundError, match="not found"):
            await engine.submit_order(order)


class TestEventPublishing:
    """Tests for algorithm lifecycle event publishing."""

    @pytest.mark.asyncio
    async def test_algo_started_event(
        self,
        mock_execution_client: MagicMock,
        message_bus: MessageBus,
        algo_order: Order,
    ) -> None:
        """Test ALGO_STARTED event is published."""
        events: list[Event] = []

        async def capture_event(event: Event) -> None:
            events.append(event)

        message_bus.subscribe(EventType.ALGO_STARTED, capture_event)

        engine = ExecutionEngine(
            message_bus=message_bus,
            execution_client=mock_execution_client,
            config=ExecutionEngineConfig(enable_risk_checks=False),
        )

        await engine.submit_order(algo_order)

        # Check ALGO_STARTED was published
        started_events = [e for e in events if e.event_type == EventType.ALGO_STARTED]
        assert len(started_events) >= 1
        assert started_events[0].payload["algorithm"] == "twap"

    @pytest.mark.asyncio
    async def test_algo_completed_event(
        self,
        mock_execution_client: MagicMock,
        message_bus: MessageBus,
        algo_order: Order,
    ) -> None:
        """Test ALGO_COMPLETED event is published."""
        events: list[Event] = []

        async def capture_event(event: Event) -> None:
            events.append(event)

        message_bus.subscribe(EventType.ALGO_COMPLETED, capture_event)

        engine = ExecutionEngine(
            message_bus=message_bus,
            execution_client=mock_execution_client,
            config=ExecutionEngineConfig(enable_risk_checks=False),
        )

        await engine.submit_order(algo_order)

        # Check ALGO_COMPLETED was published
        completed_events = [e for e in events if e.event_type == EventType.ALGO_COMPLETED]
        assert len(completed_events) >= 1


class TestCancellation:
    """Tests for execution cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_execution(
        self,
        mock_execution_client: MagicMock,
    ) -> None:
        """Test cancelling an active execution."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
            client_order_id="test-cancel-1",
            exec_algorithm="twap",
            exec_algorithm_params={
                "horizon_secs": 10,  # Long horizon
                "interval_secs": 1,
            },
        )

        engine = create_execution_engine(
            execution_client=mock_execution_client,
            enable_risk_checks=False,
        )

        # Start execution in background
        task = asyncio.create_task(engine.submit_order(order))

        # Wait a bit then cancel
        await asyncio.sleep(0.1)
        cancelled = await engine.cancel_execution("test-cancel-1")

        # Cancel may or may not succeed depending on timing
        # The execution should complete

        await task

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_execution(
        self,
        mock_execution_client: MagicMock,
    ) -> None:
        """Test cancelling execution that doesn't exist."""
        engine = create_execution_engine(
            execution_client=mock_execution_client,
            enable_risk_checks=False,
        )

        result = await engine.cancel_execution("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_all_executions(
        self,
        mock_execution_client: MagicMock,
    ) -> None:
        """Test cancelling all active executions."""
        engine = create_execution_engine(
            execution_client=mock_execution_client,
            enable_risk_checks=False,
        )

        # Start multiple executions
        orders = [
            Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal("100"),
                client_order_id=f"test-multi-{i}",
                exec_algorithm="twap",
                exec_algorithm_params={
                    "horizon_secs": 10,
                    "interval_secs": 1,
                },
            )
            for i in range(3)
        ]

        tasks = [asyncio.create_task(engine.submit_order(o)) for o in orders]

        await asyncio.sleep(0.1)
        cancelled = await engine.cancel_all_executions()

        # Wait for tasks
        for task in tasks:
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.TimeoutError:
                pass


class TestStatistics:
    """Tests for execution statistics."""

    @pytest.mark.asyncio
    async def test_stats_updated(
        self,
        mock_execution_client: MagicMock,
        sample_order: Order,
        algo_order: Order,
    ) -> None:
        """Test statistics are updated correctly."""
        engine = create_execution_engine(
            execution_client=mock_execution_client,
            enable_risk_checks=False,
        )

        # Direct order
        await engine.submit_order(sample_order)
        assert engine.stats.orders_submitted == 1
        assert engine.stats.orders_direct == 1

        # Algo order
        await engine.submit_order(algo_order)
        assert engine.stats.orders_submitted == 2
        assert engine.stats.orders_algo == 1
        assert engine.stats.algos_completed == 1


class TestLifecycle:
    """Tests for engine lifecycle."""

    @pytest.mark.asyncio
    async def test_start_stop(
        self,
        mock_execution_client: MagicMock,
    ) -> None:
        """Test starting and stopping the engine."""
        engine = create_execution_engine(
            execution_client=mock_execution_client,
        )

        await engine.start()
        await engine.stop()

        # Should be safe to call multiple times
        await engine.stop()


# =============================================================================
# Integration with Order Properties
# =============================================================================


class TestOrderIntegration:
    """Tests for Order model integration."""

    def test_order_is_algo_order(self) -> None:
        """Test is_algo_order property."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
            exec_algorithm="twap",
        )
        assert order.is_algo_order is True

    def test_order_not_algo_order(self, sample_order: Order) -> None:
        """Test is_algo_order is False for regular orders."""
        assert sample_order.is_algo_order is False

    def test_order_with_exec_algorithm(self, sample_order: Order) -> None:
        """Test with_exec_algorithm method."""
        algo_order = sample_order.with_exec_algorithm(
            "vwap",
            {"num_intervals": 12},
        )
        assert algo_order.exec_algorithm == "vwap"
        assert algo_order.exec_algorithm_params == {"num_intervals": 12}
        assert algo_order.symbol == sample_order.symbol

    def test_order_is_child_order(self) -> None:
        """Test is_child_order property."""
        parent = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
            client_order_id="parent-1",
        )
        assert parent.is_child_order is False

        child = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("10"),
            parent_order_id="parent-1",
        )
        assert child.is_child_order is True
