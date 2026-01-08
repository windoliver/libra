"""
TWAP (Time-Weighted Average Price) Execution Algorithm.

Implements Issue #36: Execution Algorithm Framework (TWAP, VWAP).

TWAP executes orders by evenly spreading them over a specified time horizon.
This helps reduce market impact by minimizing the concentration of trade
size at any given time.

Key Features:
- Configurable time horizon and intervals
- Optional randomization to avoid detection
- Support for both market and limit orders
- Progress tracking and metrics

Example:
    config = TWAPConfig(
        horizon_secs=60.0,    # Execute over 1 minute
        interval_secs=5.0,    # 5 seconds between slices
        randomize_size=True,  # Add size randomization
        randomize_delay=True, # Add timing randomization
    )

    twap = TWAPAlgorithm(config)
    twap.set_execution_client(execution_client)

    progress = await twap.execute(large_order)
    print(f"Completed: {progress.completion_pct}%")

References:
- NautilusTrader TWAP: nautilus_trader.examples.algorithms.twap
- Empirica TWAP Strategy: https://empirica.io/blog/twap-strategy/
- QuantInsti TWAP: https://blog.quantinsti.com/twap/
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from libra.execution.algorithm import BaseExecAlgorithm, ExecutionProgress
from libra.gateways.protocol import Order


if TYPE_CHECKING:
    from libra.clients.execution_client import ExecutionClient


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TWAPConfig:
    """
    Configuration for TWAP execution algorithm.

    Attributes:
        horizon_secs: Total time window for execution (seconds)
        interval_secs: Time between individual order executions (seconds)
        randomize_size: Whether to randomize slice sizes (±randomization_pct)
        randomize_delay: Whether to randomize delays between orders
        randomization_pct: Max percentage for randomization (0.1 = ±10%)
        use_limit_orders: Use limit orders instead of market orders
        limit_offset_bps: Offset from mid price for limit orders (basis points)
        min_slice_qty: Minimum quantity per slice (prevents tiny orders)

    Example:
        # Execute over 2 minutes with 10 slices
        config = TWAPConfig(
            horizon_secs=120.0,
            interval_secs=12.0,  # 120/10 = 12 seconds per slice
            randomize_size=True,
            randomize_delay=True,
        )
    """

    horizon_secs: float = 60.0
    interval_secs: float = 5.0
    randomize_size: bool = True
    randomize_delay: bool = True
    randomization_pct: float = 0.1  # ±10%
    use_limit_orders: bool = False
    limit_offset_bps: float = 5.0  # 5 basis points from mid
    min_slice_qty: Decimal | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.horizon_secs <= 0:
            raise ValueError("horizon_secs must be positive")
        if self.interval_secs <= 0:
            raise ValueError("interval_secs must be positive")
        if self.interval_secs > self.horizon_secs:
            raise ValueError("interval_secs cannot exceed horizon_secs")
        if self.randomization_pct < 0 or self.randomization_pct > 1:
            raise ValueError("randomization_pct must be between 0 and 1")

    @property
    def num_slices(self) -> int:
        """Calculate number of slices based on horizon and interval."""
        return max(1, int(self.horizon_secs / self.interval_secs))


# =============================================================================
# TWAP Algorithm Implementation
# =============================================================================


class TWAPAlgorithm(BaseExecAlgorithm):
    """
    Time-Weighted Average Price (TWAP) execution algorithm.

    Splits a large order into equal-sized slices executed at regular
    time intervals. Optional randomization helps avoid detection by
    other market participants.

    Algorithm Flow:
    1. Calculate slice quantity: total_qty / num_slices
    2. For each slice:
       a. Wait for interval (with optional randomization)
       b. Submit slice order (with optional size randomization)
       c. Track progress and update metrics
    3. Handle any remaining quantity in final slice

    Example:
        # Buy 100 BTC over 10 minutes with 1-minute intervals
        config = TWAPConfig(horizon_secs=600, interval_secs=60)
        twap = TWAPAlgorithm(config)
        twap.set_execution_client(client)

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
        )
        progress = await twap.execute(order)
    """

    def __init__(
        self,
        config: TWAPConfig | None = None,
        execution_client: ExecutionClient | None = None,
    ) -> None:
        """
        Initialize TWAP algorithm.

        Args:
            config: TWAP configuration (uses defaults if None)
            execution_client: Client for order submission
        """
        super().__init__(execution_client)
        self._config = config or TWAPConfig()

    @property
    def algorithm_id(self) -> str:
        """Unique identifier for TWAP algorithm."""
        return "twap"

    @property
    def config(self) -> TWAPConfig:
        """Get current configuration."""
        return self._config

    async def _execute_strategy(self, order: Order) -> None:
        """
        Execute TWAP strategy.

        Splits the order into equal slices and executes them
        at regular intervals over the time horizon.
        """
        num_slices = self._config.num_slices
        base_slice_qty = order.amount / num_slices

        self.log.info(
            "TWAP: %d slices of ~%s over %s seconds (interval=%s)",
            num_slices,
            base_slice_qty,
            self._config.horizon_secs,
            self._config.interval_secs,
        )

        # Track executed quantity to handle rounding
        executed = Decimal("0")

        for i in range(num_slices):
            # Check for cancellation
            if self._cancelled:
                self.log.info("TWAP cancelled at slice %d/%d", i + 1, num_slices)
                break

            # Calculate slice quantity
            if i == num_slices - 1:
                # Final slice: use remaining quantity to handle rounding
                slice_qty = order.amount - executed
            else:
                slice_qty = base_slice_qty
                if self._config.randomize_size:
                    slice_qty = self._randomize_quantity(
                        slice_qty,
                        self._config.randomization_pct,
                    )

            # Enforce minimum slice quantity
            if self._config.min_slice_qty and slice_qty < self._config.min_slice_qty:
                slice_qty = self._config.min_slice_qty

            # Don't exceed remaining quantity
            remaining = order.amount - executed
            slice_qty = min(slice_qty, remaining)

            if slice_qty <= 0:
                break

            # Submit slice
            self.log.debug(
                "TWAP slice %d/%d: %s (%.1f%% complete)",
                i + 1,
                num_slices,
                slice_qty,
                float(executed / order.amount * 100) if order.amount > 0 else 0,
            )

            result = await self._spawn_and_submit(
                quantity=slice_qty,
                price=order.price if self._config.use_limit_orders else None,
            )

            if result and result.filled_amount > 0:
                executed += result.filled_amount

            # Wait for next interval (except after last slice)
            if i < num_slices - 1 and not self._cancelled:
                delay = self._config.interval_secs
                if self._config.randomize_delay:
                    delay = self._randomize_delay(
                        delay,
                        self._config.randomization_pct,
                    )
                await asyncio.sleep(delay)

        self.log.info(
            "TWAP complete: executed %s of %s (%.1f%%)",
            executed,
            order.amount,
            float(executed / order.amount * 100) if order.amount > 0 else 0,
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_twap(
    horizon_secs: float = 60.0,
    interval_secs: float = 5.0,
    randomize: bool = True,
    execution_client: ExecutionClient | None = None,
) -> TWAPAlgorithm:
    """
    Create a TWAP algorithm with common settings.

    Args:
        horizon_secs: Total execution time window
        interval_secs: Time between slices
        randomize: Enable both size and delay randomization
        execution_client: Optional execution client

    Returns:
        Configured TWAPAlgorithm instance

    Example:
        twap = create_twap(
            horizon_secs=300,  # 5 minutes
            interval_secs=30,  # 30 second intervals
            randomize=True,
        )
    """
    config = TWAPConfig(
        horizon_secs=horizon_secs,
        interval_secs=interval_secs,
        randomize_size=randomize,
        randomize_delay=randomize,
    )
    return TWAPAlgorithm(config, execution_client)
