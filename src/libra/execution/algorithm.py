"""
Execution Algorithm Framework: Protocol and base class for execution algorithms.

Implements Issue #36: Execution Algorithm Framework (TWAP, VWAP).

Execution algorithms enable sophisticated order management by splitting large orders
into smaller child orders executed strategically over time to minimize market impact.

Architecture (inspired by NautilusTrader):
    Strategy → ExecAlgorithm → RiskEngine → ExecutionClient

Key Concepts:
- Parent Order: The original large order to be executed
- Child Order: Smaller orders spawned from the parent
- exec_spawn_id: Links child orders to their parent

References:
- NautilusTrader: https://nautilustrader.io/docs/nightly/concepts/execution/
- QuantConnect Lean: VolumeWeightedAveragePriceExecutionModel
- MQL5 Execution Algorithms pattern
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from libra.gateways.protocol import Order, OrderResult, OrderSide, OrderType


if TYPE_CHECKING:
    from libra.clients.execution_client import ExecutionClient


logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================


class AlgorithmState(Enum):
    """State of an execution algorithm instance."""

    PENDING = "pending"  # Waiting to start
    RUNNING = "running"  # Actively executing
    PAUSED = "paused"  # Temporarily paused
    COMPLETED = "completed"  # Successfully finished
    CANCELLED = "cancelled"  # Cancelled by user
    FAILED = "failed"  # Failed with error


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ChildOrder:
    """
    Represents a child order spawned from a parent order.

    Attributes:
        order: The actual order object
        spawn_id: ID of the parent order
        spawn_sequence: Sequence number (1, 2, 3, ...)
        scheduled_time_ns: When this order should be executed (optional)
    """

    order: Order
    spawn_id: str
    spawn_sequence: int
    scheduled_time_ns: int | None = None


@dataclass
class ExecutionProgress:
    """
    Tracks progress of an execution algorithm.

    Attributes:
        parent_order_id: ID of the parent order
        total_quantity: Total quantity to execute
        executed_quantity: Quantity already executed
        remaining_quantity: Quantity still to execute
        num_children_spawned: Number of child orders created
        num_children_filled: Number of child orders filled
        avg_fill_price: Volume-weighted average fill price
        state: Current algorithm state
    """

    parent_order_id: str
    total_quantity: Decimal
    executed_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal = Decimal("0")
    num_children_spawned: int = 0
    num_children_filled: int = 0
    avg_fill_price: Decimal | None = None
    state: AlgorithmState = AlgorithmState.PENDING

    def __post_init__(self) -> None:
        if self.remaining_quantity == Decimal("0"):
            self.remaining_quantity = self.total_quantity

    @property
    def completion_pct(self) -> float:
        """Percentage of order completed."""
        if self.total_quantity == 0:
            return 100.0
        return float(self.executed_quantity / self.total_quantity * 100)

    @property
    def is_complete(self) -> bool:
        """Whether execution is complete."""
        return self.remaining_quantity <= 0 or self.state in (
            AlgorithmState.COMPLETED,
            AlgorithmState.CANCELLED,
            AlgorithmState.FAILED,
        )


@dataclass
class ExecutionMetrics:
    """
    Metrics for evaluating execution quality.

    Attributes:
        arrival_price: Price when algorithm started
        avg_execution_price: VWAP of all fills
        benchmark_price: TWAP/VWAP benchmark for comparison
        implementation_shortfall: Cost vs arrival price
        benchmark_deviation: Deviation from benchmark
        total_slippage: Total slippage incurred
        num_orders: Total orders submitted
        fill_rate: Percentage of orders filled
    """

    arrival_price: Decimal
    avg_execution_price: Decimal | None = None
    benchmark_price: Decimal | None = None
    implementation_shortfall: Decimal | None = None
    benchmark_deviation: Decimal | None = None
    total_slippage: Decimal = Decimal("0")
    num_orders: int = 0
    fill_rate: float = 0.0

    def calculate_shortfall(self, side: OrderSide) -> None:
        """Calculate implementation shortfall."""
        if self.avg_execution_price is None:
            return

        if side == OrderSide.BUY:
            # For buys, shortfall is positive if we paid more than arrival
            self.implementation_shortfall = self.avg_execution_price - self.arrival_price
        else:
            # For sells, shortfall is positive if we received less than arrival
            self.implementation_shortfall = self.arrival_price - self.avg_execution_price


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class ExecAlgorithm(Protocol):
    """
    Protocol for execution algorithms.

    Execution algorithms receive parent orders and spawn child orders
    to execute them strategically over time.

    Example:
        class MyTWAP(ExecAlgorithm):
            @property
            def algorithm_id(self) -> str:
                return "my_twap"

            async def execute(self, order: Order) -> ExecutionProgress:
                # Split order into slices and execute over time
                ...
    """

    @property
    def algorithm_id(self) -> str:
        """Unique identifier for this algorithm type."""
        ...

    async def execute(self, order: Order) -> ExecutionProgress:
        """
        Execute a parent order using this algorithm.

        Args:
            order: The parent order to execute

        Returns:
            ExecutionProgress tracking the execution
        """
        ...

    async def cancel(self) -> bool:
        """
        Cancel the current execution.

        Returns:
            True if cancelled successfully
        """
        ...

    def get_progress(self) -> ExecutionProgress | None:
        """Get current execution progress."""
        ...


# =============================================================================
# Base Implementation
# =============================================================================


class BaseExecAlgorithm(ABC):
    """
    Abstract base class for execution algorithms.

    Provides common functionality for child order spawning,
    progress tracking, and execution client integration.

    Subclasses must implement:
    - algorithm_id property
    - _execute_strategy() method with the specific algorithm logic

    Example:
        class TWAPAlgorithm(BaseExecAlgorithm):
            @property
            def algorithm_id(self) -> str:
                return "twap"

            async def _execute_strategy(self, order: Order) -> None:
                num_slices = 10
                slice_qty = order.amount / num_slices
                for i in range(num_slices):
                    await asyncio.sleep(self._interval)
                    await self._submit_child(slice_qty)
    """

    def __init__(
        self,
        execution_client: ExecutionClient | None = None,
    ) -> None:
        """
        Initialize the execution algorithm.

        Args:
            execution_client: Client for submitting orders (can be set later)
        """
        self._execution_client = execution_client
        self._progress: ExecutionProgress | None = None
        self._metrics: ExecutionMetrics | None = None
        self._parent_order: Order | None = None
        self._child_orders: list[ChildOrder] = []
        self._spawn_sequence: int = 0
        self._cancelled: bool = False
        self._lock = asyncio.Lock()

        self.log = logging.getLogger(f"{__name__}.{self.algorithm_id}")

    @property
    @abstractmethod
    def algorithm_id(self) -> str:
        """Unique identifier for this algorithm type."""
        ...

    @abstractmethod
    async def _execute_strategy(self, order: Order) -> None:
        """
        Implement the specific execution strategy.

        This method should:
        1. Calculate slice sizes and timing
        2. Call _spawn_and_submit() for each child order
        3. Handle any algorithm-specific logic

        Args:
            order: The parent order to execute
        """
        ...

    def set_execution_client(self, client: ExecutionClient) -> None:
        """Set the execution client for order submission."""
        self._execution_client = client

    async def execute(self, order: Order) -> ExecutionProgress:
        """
        Execute a parent order using this algorithm.

        Args:
            order: The parent order to execute

        Returns:
            ExecutionProgress tracking the execution
        """
        if self._execution_client is None:
            raise RuntimeError("Execution client not set")

        async with self._lock:
            # Initialize tracking
            self._parent_order = order
            self._progress = ExecutionProgress(
                parent_order_id=order.client_order_id or str(uuid.uuid4()),
                total_quantity=order.amount,
            )
            self._progress.state = AlgorithmState.RUNNING
            self._spawn_sequence = 0
            self._child_orders = []
            self._cancelled = False

            # Initialize metrics with arrival price
            # In real implementation, would get current market price
            self._metrics = ExecutionMetrics(
                arrival_price=order.price or Decimal("0"),
            )

            self.log.info(
                "Starting %s execution: %s %s %s @ %s",
                self.algorithm_id.upper(),
                order.side.value,
                order.amount,
                order.symbol,
                order.price or "MARKET",
            )

        try:
            # Execute the strategy
            await self._execute_strategy(order)

            # Mark as completed if not cancelled
            if not self._cancelled and self._progress:
                self._progress.state = AlgorithmState.COMPLETED

        except Exception as e:
            self.log.exception("Execution failed: %s", e)
            if self._progress:
                self._progress.state = AlgorithmState.FAILED
            raise

        return self._progress

    async def cancel(self) -> bool:
        """Cancel the current execution."""
        self._cancelled = True
        if self._progress:
            self._progress.state = AlgorithmState.CANCELLED
        self.log.info("Execution cancelled")
        return True

    def get_progress(self) -> ExecutionProgress | None:
        """Get current execution progress."""
        return self._progress

    def get_metrics(self) -> ExecutionMetrics | None:
        """Get execution quality metrics."""
        return self._metrics

    # -------------------------------------------------------------------------
    # Child Order Management
    # -------------------------------------------------------------------------

    def _spawn_market(
        self,
        quantity: Decimal,
        scheduled_time_ns: int | None = None,
    ) -> ChildOrder:
        """
        Spawn a market child order.

        Args:
            quantity: Quantity to execute
            scheduled_time_ns: Optional scheduled execution time

        Returns:
            ChildOrder ready for submission
        """
        if self._parent_order is None:
            raise RuntimeError("No parent order set")

        self._spawn_sequence += 1
        parent_id = self._parent_order.client_order_id or "unknown"
        child_id = f"{parent_id}-E{self._spawn_sequence}"

        child_order = Order(
            symbol=self._parent_order.symbol,
            side=self._parent_order.side,
            order_type=OrderType.MARKET,
            amount=quantity,
            client_order_id=child_id,
        )

        child = ChildOrder(
            order=child_order,
            spawn_id=parent_id,
            spawn_sequence=self._spawn_sequence,
            scheduled_time_ns=scheduled_time_ns,
        )

        self._child_orders.append(child)
        if self._progress:
            self._progress.num_children_spawned += 1

        return child

    def _spawn_limit(
        self,
        quantity: Decimal,
        price: Decimal,
        scheduled_time_ns: int | None = None,
    ) -> ChildOrder:
        """
        Spawn a limit child order.

        Args:
            quantity: Quantity to execute
            price: Limit price
            scheduled_time_ns: Optional scheduled execution time

        Returns:
            ChildOrder ready for submission
        """
        if self._parent_order is None:
            raise RuntimeError("No parent order set")

        self._spawn_sequence += 1
        parent_id = self._parent_order.client_order_id or "unknown"
        child_id = f"{parent_id}-E{self._spawn_sequence}"

        child_order = Order(
            symbol=self._parent_order.symbol,
            side=self._parent_order.side,
            order_type=OrderType.LIMIT,
            amount=quantity,
            price=price,
            client_order_id=child_id,
        )

        child = ChildOrder(
            order=child_order,
            spawn_id=parent_id,
            spawn_sequence=self._spawn_sequence,
            scheduled_time_ns=scheduled_time_ns,
        )

        self._child_orders.append(child)
        if self._progress:
            self._progress.num_children_spawned += 1

        return child

    async def _submit_child(self, child: ChildOrder) -> OrderResult | None:
        """
        Submit a child order for execution.

        Args:
            child: The child order to submit

        Returns:
            OrderResult if successful, None otherwise
        """
        if self._execution_client is None:
            raise RuntimeError("Execution client not set")

        if self._cancelled:
            self.log.debug("Skipping child order - execution cancelled")
            return None

        try:
            result = await self._execution_client.submit_order(child.order)
            self._update_progress_from_fill(result)
            return result

        except Exception as e:
            self.log.error("Failed to submit child order: %s", e)
            return None

    async def _spawn_and_submit(
        self,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> OrderResult | None:
        """
        Spawn and immediately submit a child order.

        Args:
            quantity: Quantity to execute
            price: Limit price (None for market order)

        Returns:
            OrderResult if successful
        """
        if price is None:
            child = self._spawn_market(quantity)
        else:
            child = self._spawn_limit(quantity, price)

        return await self._submit_child(child)

    def _update_progress_from_fill(self, result: OrderResult) -> None:
        """Update progress tracking from an order result."""
        if self._progress is None:
            return

        if result.filled_amount > 0:
            self._progress.executed_quantity += result.filled_amount
            self._progress.remaining_quantity -= result.filled_amount
            self._progress.num_children_filled += 1

            # Update average fill price
            if result.average_price and self._metrics:
                self._update_avg_price(result.filled_amount, result.average_price)

            self.log.debug(
                "Child filled: %s @ %s (%.1f%% complete)",
                result.filled_amount,
                result.average_price,
                self._progress.completion_pct,
            )

    def _update_avg_price(self, quantity: Decimal, price: Decimal) -> None:
        """Update volume-weighted average execution price."""
        if self._metrics is None or self._progress is None:
            return

        prev_qty = self._progress.executed_quantity - quantity
        prev_avg = self._metrics.avg_execution_price or Decimal("0")

        if prev_qty > 0:
            total_value = prev_avg * prev_qty + price * quantity
            self._metrics.avg_execution_price = total_value / self._progress.executed_quantity
        else:
            self._metrics.avg_execution_price = price

        # Also update progress avg_fill_price for consistency
        self._progress.avg_fill_price = self._metrics.avg_execution_price

        self._metrics.num_orders += 1

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _randomize_quantity(
        quantity: Decimal,
        randomization_pct: float = 0.1,
    ) -> Decimal:
        """
        Add randomization to order quantity.

        Helps avoid detection by other algorithms.

        Args:
            quantity: Base quantity
            randomization_pct: Max percentage deviation (e.g., 0.1 = ±10%)

        Returns:
            Randomized quantity
        """
        factor = 1.0 + random.uniform(-randomization_pct, randomization_pct)
        return Decimal(str(float(quantity) * factor))

    @staticmethod
    def _randomize_delay(
        delay_seconds: float,
        randomization_pct: float = 0.1,
    ) -> float:
        """
        Add randomization to delay between orders.

        Args:
            delay_seconds: Base delay
            randomization_pct: Max percentage deviation

        Returns:
            Randomized delay in seconds
        """
        factor = 1.0 + random.uniform(-randomization_pct, randomization_pct)
        return delay_seconds * factor
