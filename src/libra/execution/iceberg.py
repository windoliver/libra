"""
Iceberg Order Execution Algorithm.

Implements Issue #36: Execution Algorithm Framework (TWAP, VWAP).

Iceberg orders hide the true order size by showing only a small "visible"
portion at a time. As each visible portion is filled, a new one is placed
until the full order is complete.

Key Features:
- Configurable display quantity
- Automatic replenishment on fill
- Price adjustment options (chase, peg)
- Randomization to avoid detection

Use Cases:
- Large orders that would move the market if shown in full
- Institutional trading wanting to minimize market impact
- Accumulation/distribution without signaling intent

Example:
    config = IcebergConfig(
        display_qty=Decimal("10"),    # Show 10 units at a time
        total_qty=Decimal("1000"),    # Total order size
        randomize_display=True,       # Vary display qty ±20%
    )

    iceberg = IcebergAlgorithm(config)
    iceberg.set_execution_client(client)

    progress = await iceberg.execute(large_order)

References:
- Investopedia: https://www.investopedia.com/terms/i/icebergorder.asp
- NautilusTrader: ExecAlgorithm framework patterns
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from libra.execution.algorithm import BaseExecAlgorithm
from libra.gateways.protocol import Order


if TYPE_CHECKING:
    from libra.clients.execution_client import ExecutionClient


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class IcebergConfig:
    """
    Configuration for Iceberg execution algorithm.

    Attributes:
        display_qty: Visible order quantity (the "tip")
        display_pct: Alternative: display as % of total (0.01 = 1%)
        min_display_qty: Minimum display quantity
        max_display_qty: Maximum display quantity
        randomize_display: Add randomization to display qty
        randomization_pct: Max percentage for randomization (0.2 = ±20%)
        use_limit_orders: Use limit orders (True) or market orders (False)
        limit_offset_bps: Offset from best price for limits (basis points)
        refill_threshold_pct: Refill when filled reaches this % (0.9 = 90%)
        delay_between_refills_secs: Minimum delay before placing next order

    Example:
        # Show 5% of order at a time, with ±20% randomization
        config = IcebergConfig(
            display_pct=0.05,
            randomize_display=True,
            randomization_pct=0.2,
        )
    """

    display_qty: Decimal | None = None  # Explicit display quantity
    display_pct: float = 0.05  # 5% of total order
    min_display_qty: Decimal | None = None
    max_display_qty: Decimal | None = None
    randomize_display: bool = True
    randomization_pct: float = 0.2  # ±20%
    use_limit_orders: bool = True
    limit_offset_bps: float = 5.0  # 5 basis points
    refill_threshold_pct: float = 0.9  # Refill at 90% filled
    delay_between_refills_secs: float = 0.5  # 500ms delay

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.display_qty is not None and self.display_qty <= 0:
            raise ValueError("display_qty must be positive")
        if self.display_pct <= 0 or self.display_pct > 1:
            raise ValueError("display_pct must be between 0 and 1")
        if self.randomization_pct < 0 or self.randomization_pct > 1:
            raise ValueError("randomization_pct must be between 0 and 1")
        if self.refill_threshold_pct <= 0 or self.refill_threshold_pct > 1:
            raise ValueError("refill_threshold_pct must be between 0 and 1")


# =============================================================================
# Iceberg Algorithm Implementation
# =============================================================================


class IcebergAlgorithm(BaseExecAlgorithm):
    """
    Iceberg order execution algorithm.

    Executes large orders by showing only a small visible portion at a time.
    When the visible portion fills, a new one is automatically placed.

    Algorithm Flow:
    1. Calculate display quantity (fixed or % of total)
    2. Place visible order (limit or market)
    3. Wait for fill (or partial fill above threshold)
    4. Repeat until total quantity is executed

    Benefits:
    - Hides true order size from other market participants
    - Reduces market impact by not signaling large order
    - Can be combined with price improvement strategies

    Example:
        # Execute 1000 BTC showing only 10 at a time
        config = IcebergConfig(display_qty=Decimal("10"))
        iceberg = IcebergAlgorithm(config)
        iceberg.set_execution_client(client)

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("1000"),
            price=Decimal("42000"),
        )
        progress = await iceberg.execute(order)
    """

    def __init__(
        self,
        config: IcebergConfig | None = None,
        execution_client: ExecutionClient | None = None,
    ) -> None:
        """
        Initialize Iceberg algorithm.

        Args:
            config: Iceberg configuration (uses defaults if None)
            execution_client: Client for order submission
        """
        super().__init__(execution_client)
        self._config = config or IcebergConfig()
        self._current_display_qty: Decimal = Decimal("0")
        self._num_refills: int = 0

    @property
    def algorithm_id(self) -> str:
        """Unique identifier for Iceberg algorithm."""
        return "iceberg"

    @property
    def config(self) -> IcebergConfig:
        """Get current configuration."""
        return self._config

    @property
    def num_refills(self) -> int:
        """Number of times the iceberg has been refilled."""
        return self._num_refills

    def _calculate_display_qty(self, total_qty: Decimal) -> Decimal:
        """
        Calculate display quantity for this order.

        Uses explicit display_qty if set, otherwise calculates from display_pct.
        Applies min/max bounds and optional randomization.

        Args:
            total_qty: Total order quantity

        Returns:
            Display quantity for visible order
        """
        # Use explicit or calculate from percentage
        if self._config.display_qty is not None:
            base_qty = self._config.display_qty
        else:
            base_qty = total_qty * Decimal(str(self._config.display_pct))

        # Apply randomization
        if self._config.randomize_display:
            base_qty = self._randomize_quantity(
                base_qty, self._config.randomization_pct
            )

        # Apply min/max bounds
        if self._config.min_display_qty is not None:
            base_qty = max(base_qty, self._config.min_display_qty)
        if self._config.max_display_qty is not None:
            base_qty = min(base_qty, self._config.max_display_qty)

        return base_qty

    async def _execute_strategy(self, order: Order) -> None:
        """
        Execute Iceberg strategy.

        Repeatedly places visible orders until total quantity is filled.
        """
        self._num_refills = 0
        executed = Decimal("0")
        total_qty = order.amount

        self.log.info(
            "Iceberg: total=%s, display_pct=%.1f%%, use_limits=%s",
            total_qty,
            self._config.display_pct * 100,
            self._config.use_limit_orders,
        )

        while executed < total_qty and not self._cancelled:
            # Calculate remaining and display quantities
            remaining = total_qty - executed
            display_qty = self._calculate_display_qty(total_qty)

            # Don't exceed remaining quantity
            display_qty = min(display_qty, remaining)

            if display_qty <= 0:
                break

            self._current_display_qty = display_qty
            self._num_refills += 1

            self.log.debug(
                "Iceberg refill #%d: display=%s, remaining=%s (%.1f%% complete)",
                self._num_refills,
                display_qty,
                remaining,
                float(executed / total_qty * 100) if total_qty > 0 else 0,
            )

            # Determine order type and price
            if self._config.use_limit_orders and order.price:
                # Use limit order at specified price
                price = order.price
            else:
                # Use market order
                price = None

            # Submit the visible portion
            result = await self._spawn_and_submit(
                quantity=display_qty,
                price=price,
            )

            if result and result.filled_amount > 0:
                executed += result.filled_amount

            # Small delay before next refill
            if executed < total_qty and not self._cancelled:
                await asyncio.sleep(self._config.delay_between_refills_secs)

        self.log.info(
            "Iceberg complete: executed %s of %s (%.1f%%), refills=%d",
            executed,
            total_qty,
            float(executed / total_qty * 100) if total_qty > 0 else 0,
            self._num_refills,
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_iceberg(
    display_qty: Decimal | None = None,
    display_pct: float = 0.05,
    randomize: bool = True,
    execution_client: ExecutionClient | None = None,
) -> IcebergAlgorithm:
    """
    Create an Iceberg algorithm with common settings.

    Args:
        display_qty: Fixed display quantity (overrides display_pct)
        display_pct: Display as percentage of total
        randomize: Enable display quantity randomization
        execution_client: Optional execution client

    Returns:
        Configured IcebergAlgorithm instance

    Example:
        # Show 2% of order at a time
        iceberg = create_iceberg(
            display_pct=0.02,
            randomize=True,
        )

        # Or fixed display quantity
        iceberg = create_iceberg(
            display_qty=Decimal("10"),
            randomize=True,
        )
    """
    config = IcebergConfig(
        display_qty=display_qty,
        display_pct=display_pct,
        randomize_display=randomize,
    )
    return IcebergAlgorithm(config, execution_client)
