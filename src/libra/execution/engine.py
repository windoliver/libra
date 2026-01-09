"""
Execution Engine: Orchestrates order execution through algorithms.

Implements Issue #36: Execution Algorithm Framework (TWAP, VWAP).

The ExecutionEngine is the central coordinator for order execution:
- Routes orders with exec_algorithm to appropriate algorithms
- Directly executes orders without algorithms
- Publishes algorithm lifecycle events to MessageBus
- Integrates with RiskEngine for pre-trade validation
- Tracks active algorithm executions

Architecture:
    Strategy → RiskEngine → ExecutionEngine → ExecAlgorithm → ExecutionClient
                                           ↘ (direct) → ExecutionClient

Example:
    engine = ExecutionEngine(
        message_bus=bus,
        clock=clock,
        execution_client=client,
        risk_engine=risk_engine,
    )

    # Order with algorithm
    order = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("100"),
        exec_algorithm="twap",
        exec_algorithm_params={"horizon_secs": 300},
    )
    progress = await engine.submit_order(order)

    # Direct order (no algorithm)
    order = Order(
        symbol="ETH/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        amount=Decimal("10"),
        price=Decimal("2000"),
    )
    result = await engine.submit_order(order)

References:
- NautilusTrader ExecutionEngine
- QuantConnect Algorithm Framework
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from libra.core.events import Event, EventType
from libra.execution.algorithm import (
    AlgorithmState,
    BaseExecAlgorithm,
    ExecutionProgress,
)
from libra.execution.registry import get_algorithm_registry
from libra.gateways.protocol import Order, OrderResult


if TYPE_CHECKING:
    from libra.clients.execution_client import ExecutionClient
    from libra.core.clock import Clock
    from libra.core.message_bus import MessageBus
    from libra.risk.engine import RiskEngine


logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ExecutionEngineError(Exception):
    """Base exception for execution engine errors."""


class OrderDeniedError(ExecutionEngineError):
    """Order denied by risk engine."""

    def __init__(self, reason: str, order: Order | None = None):
        self.reason = reason
        self.order = order
        super().__init__(f"Order denied: {reason}")


class AlgorithmNotFoundError(ExecutionEngineError):
    """Specified algorithm not found in registry."""


class AlgorithmExecutionError(ExecutionEngineError):
    """Error during algorithm execution."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ActiveExecution:
    """Tracks an active algorithm execution."""

    order_id: str
    algorithm: BaseExecAlgorithm
    order: Order
    progress: ExecutionProgress | None = None
    task: asyncio.Task[ExecutionProgress] | None = None
    started_at_ns: int = 0
    completed_at_ns: int | None = None


@dataclass
class ExecutionEngineConfig:
    """Configuration for ExecutionEngine."""

    # Risk integration
    enable_risk_checks: bool = True
    block_on_risk_denial: bool = True

    # Algorithm defaults
    default_algorithm: str | None = None  # Use this algorithm if none specified
    default_algorithm_params: dict[str, Any] = field(default_factory=dict)

    # Progress reporting
    progress_interval_secs: float = 1.0  # Interval for ALGO_PROGRESS events

    # Cancellation
    cancel_on_risk_breach: bool = True  # Cancel active algos on risk breach


@dataclass
class ExecutionEngineStats:
    """Statistics for the execution engine."""

    orders_submitted: int = 0
    orders_direct: int = 0  # Orders executed without algorithm
    orders_algo: int = 0  # Orders executed with algorithm
    orders_denied: int = 0
    algos_completed: int = 0
    algos_cancelled: int = 0
    algos_failed: int = 0
    child_orders_total: int = 0
    total_volume_executed: Decimal = Decimal("0")


# =============================================================================
# Execution Engine
# =============================================================================


class ExecutionEngine:
    """
    Central orchestrator for order execution.

    Routes orders through execution algorithms or directly to the execution
    client based on the order's exec_algorithm field.

    Features:
    - Algorithm-based order execution (TWAP, VWAP, Iceberg, POV)
    - Direct order execution (bypass algorithms)
    - Risk engine integration for pre-trade validation
    - Event publishing for algorithm lifecycle
    - Active execution tracking and cancellation

    Example:
        engine = ExecutionEngine(bus, clock, client)

        # Submit order with TWAP algorithm
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
            exec_algorithm="twap",
            exec_algorithm_params={"horizon_secs": 300},
        )

        progress = await engine.submit_order(order)
        print(f"Execution completed: {progress.completion_pct}%")
    """

    def __init__(
        self,
        message_bus: MessageBus | None = None,
        clock: Clock | None = None,
        execution_client: ExecutionClient | None = None,
        risk_engine: RiskEngine | None = None,
        config: ExecutionEngineConfig | None = None,
    ) -> None:
        """
        Initialize the execution engine.

        Args:
            message_bus: Message bus for event publishing
            clock: Clock for timestamps
            execution_client: Client for submitting orders
            risk_engine: Risk engine for pre-trade validation
            config: Engine configuration
        """
        self._bus = message_bus
        self._clock = clock
        self._execution_client = execution_client
        self._risk_engine = risk_engine
        self._config = config or ExecutionEngineConfig()

        # Algorithm registry
        self._registry = get_algorithm_registry()

        # Active executions tracking
        self._active_executions: dict[str, ActiveExecution] = {}
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = ExecutionEngineStats()

        self._log = logging.getLogger(f"{__name__}.ExecutionEngine")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def execution_client(self) -> ExecutionClient | None:
        """Get the execution client."""
        return self._execution_client

    @execution_client.setter
    def execution_client(self, client: ExecutionClient) -> None:
        """Set the execution client."""
        self._execution_client = client

    @property
    def message_bus(self) -> MessageBus | None:
        """Get the message bus."""
        return self._bus

    @message_bus.setter
    def message_bus(self, bus: MessageBus) -> None:
        """Set the message bus."""
        self._bus = bus

    @property
    def risk_engine(self) -> RiskEngine | None:
        """Get the risk engine."""
        return self._risk_engine

    @risk_engine.setter
    def risk_engine(self, engine: RiskEngine) -> None:
        """Set the risk engine."""
        self._risk_engine = engine

    @property
    def active_executions(self) -> dict[str, ActiveExecution]:
        """Get active executions (read-only copy)."""
        return self._active_executions.copy()

    @property
    def stats(self) -> ExecutionEngineStats:
        """Get execution statistics."""
        return self._stats

    # -------------------------------------------------------------------------
    # Order Submission
    # -------------------------------------------------------------------------

    async def submit_order(
        self,
        order: Order,
    ) -> ExecutionProgress | OrderResult:
        """
        Submit an order for execution.

        If the order has exec_algorithm specified, routes to the appropriate
        algorithm. Otherwise, executes directly via the execution client.

        Args:
            order: Order to execute

        Returns:
            ExecutionProgress if algorithm used, OrderResult if direct

        Raises:
            OrderDeniedError: If risk checks fail
            AlgorithmNotFoundError: If specified algorithm doesn't exist
            RuntimeError: If execution client not configured
        """
        if self._execution_client is None:
            raise RuntimeError("Execution client not configured")

        # Ensure order has client_order_id
        if order.client_order_id is None:
            order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                amount=order.amount,
                id=order.id,
                client_order_id=str(uuid.uuid4()),
                price=order.price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                reduce_only=order.reduce_only,
                post_only=order.post_only,
                leverage=order.leverage,
                timestamp_ns=order.timestamp_ns,
                exec_algorithm=order.exec_algorithm,
                exec_algorithm_params=order.exec_algorithm_params,
                parent_order_id=order.parent_order_id,
            )

        self._stats.orders_submitted += 1

        # Risk validation
        if self._config.enable_risk_checks and self._risk_engine:
            await self._validate_with_risk_engine(order)

        # Determine execution path
        algo_name = order.exec_algorithm or self._config.default_algorithm

        if algo_name:
            return await self._execute_with_algorithm(order, algo_name)
        else:
            return await self._execute_direct(order)

    async def _validate_with_risk_engine(self, order: Order) -> None:
        """Validate order with risk engine."""
        if self._risk_engine is None:
            return

        try:
            # RiskEngine.validate_order is synchronous and returns RiskCheckResult
            # We need a current price for validation - use order price or 0 as fallback
            current_price = order.price or Decimal("0")
            result = self._risk_engine.validate_order(order, current_price)

            if not result.passed:
                self._stats.orders_denied += 1

                # Publish denial event
                await self._publish_event(
                    EventType.ORDER_DENIED,
                    {
                        "order_id": order.client_order_id,
                        "symbol": order.symbol,
                        "reason": result.reason,
                        "check_name": result.check_name,
                    },
                )

                if self._config.block_on_risk_denial:
                    raise OrderDeniedError(result.reason or "Risk check failed", order)

        except OrderDeniedError:
            raise
        except Exception as e:
            self._log.warning("Risk check error (allowing order): %s", e)

    async def _execute_direct(self, order: Order) -> OrderResult:
        """Execute order directly without algorithm."""
        self._stats.orders_direct += 1

        self._log.info(
            "Direct execution: %s %s %s @ %s",
            order.side.value,
            order.amount,
            order.symbol,
            order.price or "MARKET",
        )

        # execution_client is guaranteed to be not None here (checked in submit_order)
        client = self._execution_client
        if client is None:
            raise RuntimeError("Execution client not configured")
        result = await client.submit_order(order)

        # Update stats
        if result.filled_amount > 0:
            self._stats.total_volume_executed += result.filled_amount

        return result

    async def _execute_with_algorithm(
        self,
        order: Order,
        algo_name: str,
    ) -> ExecutionProgress:
        """Execute order using specified algorithm."""
        self._stats.orders_algo += 1

        # Get algorithm parameters
        params = order.exec_algorithm_params or {}
        if self._config.default_algorithm_params:
            # Merge defaults with order-specific params
            merged_params = {**self._config.default_algorithm_params, **params}
        else:
            merged_params = params

        # Create algorithm instance
        try:
            algorithm = self._registry.create(
                algo_name,
                execution_client=self._execution_client,
                **merged_params,
            )
        except KeyError as e:
            raise AlgorithmNotFoundError(f"Algorithm '{algo_name}' not found") from e

        order_id = order.client_order_id or str(uuid.uuid4())

        # Track active execution
        execution = ActiveExecution(
            order_id=order_id,
            algorithm=algorithm,
            order=order,
            started_at_ns=self._get_timestamp_ns(),
        )

        async with self._lock:
            self._active_executions[order_id] = execution

        self._log.info(
            "Algorithm execution: %s for %s %s %s",
            algo_name.upper(),
            order.side.value,
            order.amount,
            order.symbol,
        )

        # Publish started event
        await self._publish_event(
            EventType.ALGO_STARTED,
            {
                "order_id": order_id,
                "symbol": order.symbol,
                "algorithm": algo_name,
                "side": order.side.value,
                "amount": str(order.amount),
                "params": merged_params,
            },
        )

        try:
            # Execute algorithm
            progress = await algorithm.execute(order)
            execution.progress = progress
            execution.completed_at_ns = self._get_timestamp_ns()

            # Update stats based on outcome
            if progress.state == AlgorithmState.COMPLETED:
                self._stats.algos_completed += 1
                await self._publish_event(
                    EventType.ALGO_COMPLETED,
                    {
                        "order_id": order_id,
                        "symbol": order.symbol,
                        "algorithm": algo_name,
                        "completion_pct": progress.completion_pct,
                        "executed_quantity": str(progress.executed_quantity),
                        "avg_fill_price": str(progress.avg_fill_price)
                        if progress.avg_fill_price
                        else None,
                        "num_children": progress.num_children_spawned,
                    },
                )
            elif progress.state == AlgorithmState.CANCELLED:
                self._stats.algos_cancelled += 1
                await self._publish_event(
                    EventType.ALGO_CANCELLED,
                    {
                        "order_id": order_id,
                        "symbol": order.symbol,
                        "algorithm": algo_name,
                        "completion_pct": progress.completion_pct,
                    },
                )

            # Update volume stats
            if progress.executed_quantity > 0:
                self._stats.total_volume_executed += progress.executed_quantity
                self._stats.child_orders_total += progress.num_children_spawned

            return progress

        except Exception as e:
            self._stats.algos_failed += 1
            self._log.exception("Algorithm execution failed: %s", e)

            await self._publish_event(
                EventType.ALGO_FAILED,
                {
                    "order_id": order_id,
                    "symbol": order.symbol,
                    "algorithm": algo_name,
                    "error": str(e),
                },
            )

            raise AlgorithmExecutionError(f"Algorithm '{algo_name}' failed: {e}") from e

        finally:
            async with self._lock:
                self._active_executions.pop(order_id, None)

    # -------------------------------------------------------------------------
    # Execution Management
    # -------------------------------------------------------------------------

    async def cancel_execution(self, order_id: str) -> bool:
        """
        Cancel an active algorithm execution.

        Args:
            order_id: Client order ID of the execution to cancel

        Returns:
            True if cancelled, False if not found or already completed
        """
        async with self._lock:
            execution = self._active_executions.get(order_id)

        if execution is None:
            self._log.warning("Cannot cancel: execution %s not found", order_id)
            return False

        try:
            cancelled = await execution.algorithm.cancel()

            if cancelled:
                self._log.info("Cancelled execution: %s", order_id)
                await self._publish_event(
                    EventType.ALGO_CANCELLED,
                    {
                        "order_id": order_id,
                        "symbol": execution.order.symbol,
                        "algorithm": execution.algorithm.algorithm_id,
                        "reason": "user_requested",
                    },
                )

            return cancelled

        except Exception as e:
            self._log.error("Failed to cancel execution %s: %s", order_id, e)
            return False

    async def cancel_all_executions(self, symbol: str | None = None) -> int:
        """
        Cancel all active executions.

        Args:
            symbol: Optional symbol filter

        Returns:
            Number of executions cancelled
        """
        cancelled_count = 0

        async with self._lock:
            order_ids = list(self._active_executions.keys())

        for order_id in order_ids:
            execution = self._active_executions.get(order_id)
            if execution is None:
                continue

            if symbol and execution.order.symbol != symbol:
                continue

            if await self.cancel_execution(order_id):
                cancelled_count += 1

        self._log.info("Cancelled %d executions", cancelled_count)
        return cancelled_count

    def get_execution(self, order_id: str) -> ActiveExecution | None:
        """Get an active execution by order ID."""
        return self._active_executions.get(order_id)

    def get_progress(self, order_id: str) -> ExecutionProgress | None:
        """Get progress for an active execution."""
        execution = self._active_executions.get(order_id)
        if execution and execution.algorithm:
            return execution.algorithm.get_progress()
        return None

    # -------------------------------------------------------------------------
    # Event Publishing
    # -------------------------------------------------------------------------

    async def _publish_event(
        self,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> None:
        """Publish an event to the message bus."""
        if self._bus is None:
            return

        event = Event.create(
            event_type=event_type,
            source="execution_engine",
            payload=payload,
        )

        try:
            self._bus.publish(event)
        except Exception as e:
            self._log.error("Failed to publish event: %s", e)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _get_timestamp_ns(self) -> int:
        """Get current timestamp in nanoseconds."""
        if self._clock:
            return self._clock.timestamp_ns()
        import time

        return time.time_ns()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the execution engine."""
        self._log.info("ExecutionEngine started")

    async def stop(self) -> None:
        """Stop the execution engine and cancel active executions."""
        self._log.info("Stopping ExecutionEngine...")

        # Cancel all active executions
        await self.cancel_all_executions()

        self._log.info(
            "ExecutionEngine stopped. Stats: submitted=%d, direct=%d, algo=%d, "
            "completed=%d, cancelled=%d, failed=%d",
            self._stats.orders_submitted,
            self._stats.orders_direct,
            self._stats.orders_algo,
            self._stats.algos_completed,
            self._stats.algos_cancelled,
            self._stats.algos_failed,
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_execution_engine(
    message_bus: MessageBus | None = None,
    clock: Clock | None = None,
    execution_client: ExecutionClient | None = None,
    risk_engine: RiskEngine | None = None,
    enable_risk_checks: bool = True,
) -> ExecutionEngine:
    """
    Create an ExecutionEngine with common settings.

    Args:
        message_bus: Message bus for events
        clock: Clock for timestamps
        execution_client: Client for order submission
        risk_engine: Risk engine for validation
        enable_risk_checks: Whether to enable risk checks

    Returns:
        Configured ExecutionEngine instance
    """
    config = ExecutionEngineConfig(
        enable_risk_checks=enable_risk_checks,
    )

    return ExecutionEngine(
        message_bus=message_bus,
        clock=clock,
        execution_client=execution_client,
        risk_engine=risk_engine,
        config=config,
    )
