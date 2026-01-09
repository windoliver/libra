"""
POV (Percentage of Volume) Execution Algorithm.

Implements Issue #36: Execution Algorithm Framework (TWAP, VWAP).

POV (also known as Participation Rate or Target Volume) executes orders
as a percentage of the market's trading volume. This adaptive approach
trades more when volume is high and less when volume is low.

Key Features:
- Target participation rate (e.g., 5% of market volume)
- Maximum participation cap to limit market impact
- Real-time volume adaptation
- Price limit support
- Time-bound execution windows

Use Cases:
- Large orders that need to track market activity
- Minimize footprint relative to market volume
- When market timing is less critical than participation

Example:
    config = POVConfig(
        target_pct=0.05,          # Target 5% of volume
        max_pct=0.20,             # Never exceed 20%
        min_order_qty=Decimal("0.01"),
        interval_secs=60.0,       # Check volume every minute
    )

    pov = POVAlgorithm(config)
    pov.set_execution_client(client)

    progress = await pov.execute(large_order)

References:
- IBKR Accumulate/Distribute: https://www.ibkrguides.com/traderworkstation/algos.htm
- Empirica POV: https://empirica.io/blog/pov-algorithm/
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from libra.execution.algorithm import BaseExecAlgorithm
from libra.gateways.protocol import Order


if TYPE_CHECKING:
    from libra.clients.data_client import DataClient
    from libra.clients.execution_client import ExecutionClient


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class POVConfig:
    """
    Configuration for POV (Percentage of Volume) execution algorithm.

    Attributes:
        target_pct: Target participation rate (0.05 = 5% of volume)
        max_pct: Maximum participation rate cap (0.20 = 20%)
        min_pct: Minimum participation rate (0.01 = 1%)
        interval_secs: Volume sampling interval (seconds)
        min_order_qty: Minimum order quantity per slice
        max_order_qty: Maximum order quantity per slice
        use_limit_orders: Use limit orders instead of market
        limit_offset_bps: Offset from best price for limits (basis points)
        randomize_size: Add randomization to slice sizes
        randomization_pct: Max percentage for randomization
        max_duration_secs: Maximum execution duration (optional)
        price_limit: Stop execution if price crosses limit (optional)

    Example:
        # Execute at 5% of market volume
        config = POVConfig(
            target_pct=0.05,
            max_pct=0.15,
            interval_secs=30.0,
        )
    """

    target_pct: float = 0.05  # 5% participation
    max_pct: float = 0.20  # 20% max participation
    min_pct: float = 0.01  # 1% minimum participation
    interval_secs: float = 60.0  # 1 minute sampling
    min_order_qty: Decimal | None = None
    max_order_qty: Decimal | None = None
    use_limit_orders: bool = False
    limit_offset_bps: float = 5.0
    randomize_size: bool = True
    randomization_pct: float = 0.15  # ±15%
    max_duration_secs: float | None = None
    price_limit: Decimal | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.target_pct <= 0 or self.target_pct > 1:
            raise ValueError("target_pct must be between 0 and 1")
        if self.max_pct <= 0 or self.max_pct > 1:
            raise ValueError("max_pct must be between 0 and 1")
        if self.min_pct < 0 or self.min_pct >= self.target_pct:
            raise ValueError("min_pct must be between 0 and target_pct")
        if self.target_pct > self.max_pct:
            raise ValueError("target_pct cannot exceed max_pct")
        if self.interval_secs <= 0:
            raise ValueError("interval_secs must be positive")
        if self.randomization_pct < 0 or self.randomization_pct > 1:
            raise ValueError("randomization_pct must be between 0 and 1")


# =============================================================================
# Volume Tracker
# =============================================================================


@dataclass
class VolumeSnapshot:
    """Snapshot of market volume for an interval."""

    timestamp_ns: int
    volume: Decimal
    interval_secs: float


class VolumeTracker:
    """
    Tracks market volume for POV calculations.

    In production, this would integrate with real-time market data.
    For simulation, it can use historical volume patterns.
    """

    def __init__(self, symbol: str, data_client: DataClient | None = None):
        self.symbol = symbol
        self._data_client = data_client
        self._snapshots: list[VolumeSnapshot] = []
        self._estimated_daily_volume: Decimal | None = None

    async def get_interval_volume(self, interval_secs: float) -> Decimal:
        """
        Get estimated volume for the interval.

        In production, this would query real-time volume data.
        Returns estimated volume based on historical patterns or
        a conservative estimate if data unavailable.
        """
        if self._data_client is not None:
            try:
                # Try to get recent trades or volume data
                # This is a simplified implementation
                return await self._fetch_recent_volume(interval_secs)
            except Exception:
                pass

        # Fallback: estimate based on daily volume
        if self._estimated_daily_volume:
            # Assume volume distributed across 24 hours
            hourly_volume = self._estimated_daily_volume / Decimal("24")
            return hourly_volume * Decimal(str(interval_secs / 3600))

        # Conservative fallback: return a small default
        return Decimal("100")

    async def _fetch_recent_volume(self, interval_secs: float) -> Decimal:
        """Fetch recent volume from data client."""
        # This would integrate with real market data
        # For now, return a placeholder
        return Decimal("100")

    def set_estimated_daily_volume(self, volume: Decimal) -> None:
        """Set estimated daily volume for fallback calculations."""
        self._estimated_daily_volume = volume


# =============================================================================
# POV Algorithm Implementation
# =============================================================================


class POVAlgorithm(BaseExecAlgorithm):
    """
    Percentage of Volume (POV) execution algorithm.

    Executes orders as a target percentage of market volume,
    adapting execution rate to match market activity.

    Algorithm Flow:
    1. At each interval:
       a. Query/estimate market volume for interval
       b. Calculate target quantity: volume × target_pct
       c. Apply min/max constraints
       d. Submit slice order
    2. Repeat until total quantity executed or cancelled
    3. Respect max_duration if configured

    Benefits:
    - Adapts to market conditions automatically
    - Lower market impact during low-volume periods
    - Captures liquidity during high-volume periods

    Example:
        config = POVConfig(target_pct=0.05, interval_secs=60)
        pov = POVAlgorithm(config)
        pov.set_execution_client(client)

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("1000"),
        )
        progress = await pov.execute(order)
    """

    def __init__(
        self,
        config: POVConfig | None = None,
        execution_client: ExecutionClient | None = None,
        data_client: DataClient | None = None,
    ) -> None:
        """
        Initialize POV algorithm.

        Args:
            config: POV configuration (uses defaults if None)
            execution_client: Client for order submission
            data_client: Client for volume data (optional)
        """
        super().__init__(execution_client)
        self._config = config or POVConfig()
        self._data_client = data_client
        self._volume_tracker: VolumeTracker | None = None
        self._start_time_ns: int = 0

    @property
    def algorithm_id(self) -> str:
        """Unique identifier for POV algorithm."""
        return "pov"

    @property
    def config(self) -> POVConfig:
        """Get current configuration."""
        return self._config

    def set_data_client(self, client: DataClient) -> None:
        """Set data client for volume tracking."""
        self._data_client = client

    def set_estimated_daily_volume(self, volume: Decimal) -> None:
        """Set estimated daily volume for the symbol."""
        if self._volume_tracker:
            self._volume_tracker.set_estimated_daily_volume(volume)

    async def _execute_strategy(self, order: Order) -> None:
        """
        Execute POV strategy.

        Participates at target percentage of market volume.
        """
        import time

        self._start_time_ns = time.time_ns()

        # Initialize volume tracker
        self._volume_tracker = VolumeTracker(order.symbol, self._data_client)

        executed = Decimal("0")
        total_qty = order.amount
        intervals_executed = 0

        self.log.info(
            "POV: target=%.1f%%, max=%.1f%%, interval=%ss",
            self._config.target_pct * 100,
            self._config.max_pct * 100,
            self._config.interval_secs,
        )

        while executed < total_qty and not self._cancelled:
            # Check max duration
            if self._config.max_duration_secs:
                elapsed_secs = (time.time_ns() - self._start_time_ns) / 1e9
                if elapsed_secs >= self._config.max_duration_secs:
                    self.log.info("POV: max duration reached")
                    break

            # Get interval volume
            interval_volume = await self._volume_tracker.get_interval_volume(
                self._config.interval_secs
            )

            # Calculate target quantity for this interval
            target_qty = interval_volume * Decimal(str(self._config.target_pct))

            # Apply min/max participation
            min_qty = interval_volume * Decimal(str(self._config.min_pct))
            max_qty = interval_volume * Decimal(str(self._config.max_pct))

            target_qty = max(target_qty, min_qty)
            target_qty = min(target_qty, max_qty)

            # Apply order size constraints
            if self._config.min_order_qty and target_qty < self._config.min_order_qty:
                target_qty = self._config.min_order_qty

            if self._config.max_order_qty and target_qty > self._config.max_order_qty:
                target_qty = self._config.max_order_qty

            # Apply randomization
            if self._config.randomize_size:
                target_qty = self._randomize_quantity(
                    target_qty, self._config.randomization_pct
                )

            # Don't exceed remaining quantity
            remaining = total_qty - executed
            slice_qty = min(target_qty, remaining)

            if slice_qty <= 0:
                break

            intervals_executed += 1

            self.log.debug(
                "POV interval %d: target=%s (volume=%s, %.1f%% complete)",
                intervals_executed,
                slice_qty,
                interval_volume,
                float(executed / total_qty * 100) if total_qty > 0 else 0,
            )

            # Submit slice
            result = await self._spawn_and_submit(
                quantity=slice_qty,
                price=order.price if self._config.use_limit_orders else None,
            )

            if result and result.filled_amount > 0:
                executed += result.filled_amount

            # Wait for next interval (unless done or cancelled)
            if executed < total_qty and not self._cancelled:
                delay = self._config.interval_secs
                if self._config.randomize_size:
                    delay = self._randomize_delay(delay, self._config.randomization_pct)
                await asyncio.sleep(delay)

        self.log.info(
            "POV complete: executed %s of %s (%.1f%%), intervals=%d",
            executed,
            total_qty,
            float(executed / total_qty * 100) if total_qty > 0 else 0,
            intervals_executed,
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_pov(
    target_pct: float = 0.05,
    max_pct: float = 0.20,
    interval_secs: float = 60.0,
    execution_client: ExecutionClient | None = None,
) -> POVAlgorithm:
    """
    Create a POV algorithm with common settings.

    Args:
        target_pct: Target participation rate (0.05 = 5%)
        max_pct: Maximum participation rate (0.20 = 20%)
        interval_secs: Volume sampling interval
        execution_client: Optional execution client

    Returns:
        Configured POVAlgorithm instance

    Example:
        # Participate at 10% of volume
        pov = create_pov(
            target_pct=0.10,
            max_pct=0.25,
            interval_secs=30,
        )
    """
    config = POVConfig(
        target_pct=target_pct,
        max_pct=max_pct,
        interval_secs=interval_secs,
    )
    return POVAlgorithm(config, execution_client)
