"""Tests for ExecAlgorithm spawn helper methods (Issue #114)."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from libra.execution.algorithm import BaseExecAlgorithm, ExecutionProgress
from libra.gateways.protocol import (
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


# =============================================================================
# Test Algorithm Implementation
# =============================================================================


class MockExecAlgorithm(BaseExecAlgorithm):
    """Concrete implementation for testing spawn helpers."""

    @property
    def algorithm_id(self) -> str:
        return "test_algo"

    async def _execute_strategy(self, order: Order) -> None:
        """Empty strategy - tests call spawn methods directly."""
        pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_execution_client() -> MagicMock:
    """Mock execution client for testing."""
    client = MagicMock()
    client.submit_order = AsyncMock()
    return client


@pytest.fixture
def mock_order_result() -> OrderResult:
    """Mock order result for successful submission."""
    return OrderResult(
        order_id="ex-12345",
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
        timestamp_ns=0,
    )


@pytest.fixture
def parent_order() -> Order:
    """Sample parent order."""
    return Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("1.0"),
        client_order_id="parent-123",
    )


@pytest.fixture
async def algorithm(
    mock_execution_client: MagicMock,
    mock_order_result: OrderResult,
    parent_order: Order,
) -> MockExecAlgorithm:
    """Algorithm with initialized parent order."""
    mock_execution_client.submit_order.return_value = mock_order_result

    algo = MockExecAlgorithm(execution_client=mock_execution_client)
    # Initialize internal state by starting execute
    algo._execution_client = mock_execution_client
    algo._parent_order = parent_order
    algo._progress = ExecutionProgress(
        parent_order_id=parent_order.client_order_id or "unknown",
        total_quantity=parent_order.amount,
    )
    algo._spawn_sequence = 0
    algo._child_orders = []
    algo._cancelled = False

    return algo


# =============================================================================
# Tests for spawn_market
# =============================================================================


class TestSpawnMarket:
    """Tests for spawn_market method."""

    @pytest.mark.asyncio
    async def test_spawn_market_basic(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """Basic spawn_market creates and submits market order."""
        result = await algorithm.spawn_market(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        assert result is not None
        mock_execution_client.submit_order.assert_called_once()

        # Check the order submitted
        submitted_order = mock_execution_client.submit_order.call_args[0][0]
        assert submitted_order.symbol == "BTC/USDT"
        assert submitted_order.side == OrderSide.BUY
        assert submitted_order.order_type == OrderType.MARKET
        assert submitted_order.amount == Decimal("0.1")
        assert submitted_order.parent_order_id == "parent-123"
        assert submitted_order.client_order_id == "parent-123-1"

    @pytest.mark.asyncio
    async def test_spawn_market_different_side(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """spawn_market can use different side from parent."""
        # Parent is BUY, child is SELL
        result = await algorithm.spawn_market(
            side=OrderSide.SELL,
            quantity=Decimal("0.1"),
        )

        submitted_order = mock_execution_client.submit_order.call_args[0][0]
        assert submitted_order.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_spawn_market_reduce_only(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """spawn_market supports reduce_only parameter."""
        result = await algorithm.spawn_market(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            reduce_only=True,
        )

        submitted_order = mock_execution_client.submit_order.call_args[0][0]
        assert submitted_order.reduce_only is True

    @pytest.mark.asyncio
    async def test_spawn_market_increments_sequence(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """spawn_market increments spawn sequence."""
        await algorithm.spawn_market(side=OrderSide.BUY, quantity=Decimal("0.1"))
        await algorithm.spawn_market(side=OrderSide.BUY, quantity=Decimal("0.1"))
        await algorithm.spawn_market(side=OrderSide.BUY, quantity=Decimal("0.1"))

        assert algorithm.child_count == 3

        # Check client_order_ids
        calls = mock_execution_client.submit_order.call_args_list
        assert calls[0][0][0].client_order_id == "parent-123-1"
        assert calls[1][0][0].client_order_id == "parent-123-2"
        assert calls[2][0][0].client_order_id == "parent-123-3"

    @pytest.mark.asyncio
    async def test_spawn_market_updates_progress(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """spawn_market updates execution progress."""
        await algorithm.spawn_market(side=OrderSide.BUY, quantity=Decimal("0.1"))

        assert algorithm._progress.num_children_spawned == 1
        assert algorithm._progress.num_children_filled == 1  # Mock result is filled
        assert algorithm._progress.executed_quantity == Decimal("0.1")

    @pytest.mark.asyncio
    async def test_spawn_market_cancelled(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """spawn_market returns None when cancelled."""
        algorithm._cancelled = True

        result = await algorithm.spawn_market(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        assert result is None
        mock_execution_client.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_spawn_market_no_parent_order(
        self,
        mock_execution_client: MagicMock,
    ) -> None:
        """spawn_market raises error if no parent order."""
        algo = MockExecAlgorithm(execution_client=mock_execution_client)

        with pytest.raises(RuntimeError, match="No parent order set"):
            await algo.spawn_market(side=OrderSide.BUY, quantity=Decimal("0.1"))

    @pytest.mark.asyncio
    async def test_spawn_market_no_execution_client(
        self,
        parent_order: Order,
    ) -> None:
        """spawn_market raises error if no execution client."""
        algo = MockExecAlgorithm()
        algo._parent_order = parent_order

        with pytest.raises(RuntimeError, match="Execution client not set"):
            await algo.spawn_market(side=OrderSide.BUY, quantity=Decimal("0.1"))


# =============================================================================
# Tests for spawn_limit
# =============================================================================


class TestSpawnLimit:
    """Tests for spawn_limit method."""

    @pytest.mark.asyncio
    async def test_spawn_limit_basic(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """Basic spawn_limit creates and submits limit order."""
        result = await algorithm.spawn_limit(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            price=Decimal("49000"),
        )

        assert result is not None
        mock_execution_client.submit_order.assert_called_once()

        submitted_order = mock_execution_client.submit_order.call_args[0][0]
        assert submitted_order.order_type == OrderType.LIMIT
        assert submitted_order.price == Decimal("49000")
        assert submitted_order.parent_order_id == "parent-123"

    @pytest.mark.asyncio
    async def test_spawn_limit_time_in_force(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """spawn_limit supports time_in_force parameter."""
        await algorithm.spawn_limit(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            price=Decimal("49000"),
            time_in_force=TimeInForce.IOC,
        )

        submitted_order = mock_execution_client.submit_order.call_args[0][0]
        assert submitted_order.time_in_force == TimeInForce.IOC

    @pytest.mark.asyncio
    async def test_spawn_limit_post_only(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """spawn_limit supports post_only parameter."""
        await algorithm.spawn_limit(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            price=Decimal("49000"),
            post_only=True,
        )

        submitted_order = mock_execution_client.submit_order.call_args[0][0]
        assert submitted_order.post_only is True

    @pytest.mark.asyncio
    async def test_spawn_limit_reduce_only(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """spawn_limit supports reduce_only parameter."""
        await algorithm.spawn_limit(
            side=OrderSide.SELL,
            quantity=Decimal("0.1"),
            price=Decimal("51000"),
            reduce_only=True,
        )

        submitted_order = mock_execution_client.submit_order.call_args[0][0]
        assert submitted_order.reduce_only is True

    @pytest.mark.asyncio
    async def test_spawn_limit_all_parameters(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """spawn_limit with all parameters set."""
        await algorithm.spawn_limit(
            side=OrderSide.SELL,
            quantity=Decimal("0.5"),
            price=Decimal("51000"),
            time_in_force=TimeInForce.FOK,
            post_only=False,
            reduce_only=True,
        )

        submitted_order = mock_execution_client.submit_order.call_args[0][0]
        assert submitted_order.side == OrderSide.SELL
        assert submitted_order.amount == Decimal("0.5")
        assert submitted_order.price == Decimal("51000")
        assert submitted_order.time_in_force == TimeInForce.FOK
        assert submitted_order.post_only is False
        assert submitted_order.reduce_only is True


# =============================================================================
# Tests for spawn_market_to_limit
# =============================================================================


class TestSpawnMarketToLimit:
    """Tests for spawn_market_to_limit method."""

    @pytest.mark.asyncio
    async def test_spawn_market_to_limit_basic(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """Basic spawn_market_to_limit creates order with IOC."""
        result = await algorithm.spawn_market_to_limit(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        assert result is not None
        mock_execution_client.submit_order.assert_called_once()

        submitted_order = mock_execution_client.submit_order.call_args[0][0]
        assert submitted_order.order_type == OrderType.MARKET
        assert submitted_order.time_in_force == TimeInForce.IOC
        assert submitted_order.parent_order_id == "parent-123"

    @pytest.mark.asyncio
    async def test_spawn_market_to_limit_reduce_only(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """spawn_market_to_limit supports reduce_only."""
        await algorithm.spawn_market_to_limit(
            side=OrderSide.SELL,
            quantity=Decimal("0.1"),
            reduce_only=True,
        )

        submitted_order = mock_execution_client.submit_order.call_args[0][0]
        assert submitted_order.reduce_only is True


# =============================================================================
# Tests for child_count and child_orders properties
# =============================================================================


class TestChildTracking:
    """Tests for child order tracking."""

    @pytest.mark.asyncio
    async def test_child_count_property(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """child_count property returns spawn sequence."""
        assert algorithm.child_count == 0

        await algorithm.spawn_market(side=OrderSide.BUY, quantity=Decimal("0.1"))
        assert algorithm.child_count == 1

        await algorithm.spawn_limit(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            price=Decimal("49000"),
        )
        assert algorithm.child_count == 2

    @pytest.mark.asyncio
    async def test_child_orders_property(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """child_orders property returns copy of child orders list."""
        await algorithm.spawn_market(side=OrderSide.BUY, quantity=Decimal("0.1"))
        await algorithm.spawn_limit(
            side=OrderSide.SELL,
            quantity=Decimal("0.2"),
            price=Decimal("51000"),
        )

        children = algorithm.child_orders
        assert len(children) == 2

        # Verify it's a copy
        children.pop()
        assert len(algorithm.child_orders) == 2

    @pytest.mark.asyncio
    async def test_child_orders_have_correct_spawn_id(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """Child orders have correct spawn_id set."""
        await algorithm.spawn_market(side=OrderSide.BUY, quantity=Decimal("0.1"))

        children = algorithm.child_orders
        assert children[0].spawn_id == "parent-123"
        assert children[0].spawn_sequence == 1

    @pytest.mark.asyncio
    async def test_child_orders_track_parent_order_id(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """Child orders have parent_order_id set correctly."""
        await algorithm.spawn_market(side=OrderSide.BUY, quantity=Decimal("0.1"))

        children = algorithm.child_orders
        assert children[0].order.parent_order_id == "parent-123"


# =============================================================================
# Tests for error handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in spawn methods."""

    @pytest.mark.asyncio
    async def test_spawn_market_handles_submit_error(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """spawn_market handles submission errors gracefully."""
        mock_execution_client.submit_order.side_effect = Exception("Connection error")

        result = await algorithm.spawn_market(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        assert result is None
        # Child was still tracked before error
        assert algorithm.child_count == 1

    @pytest.mark.asyncio
    async def test_spawn_limit_handles_submit_error(
        self,
        algorithm: MockExecAlgorithm,
        mock_execution_client: MagicMock,
    ) -> None:
        """spawn_limit handles submission errors gracefully."""
        mock_execution_client.submit_order.side_effect = Exception("Rate limited")

        result = await algorithm.spawn_limit(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            price=Decimal("49000"),
        )

        assert result is None
