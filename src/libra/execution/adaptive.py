"""
Adaptive Execution Features.

Implements Issue #36: Execution Algorithm Framework (TWAP, VWAP).

Provides adaptive execution capabilities that can be mixed into
execution algorithms to enable real-time adjustment based on
market conditions.

Key Features:
- Dynamic urgency adjustment based on volume deviations
- Price improvement logic for limit orders
- Market impact estimation
- Fill rate monitoring and adjustment
- Real-time spread monitoring

Use Cases:
- Algorithms that need to adapt to changing market conditions
- High-volume periods requiring increased participation
- Low-volume periods requiring reduced aggression

Example:
    class AdaptiveTWAP(AdaptiveMixin, TWAPAlgorithm):
        async def _execute_strategy(self, order: Order) -> None:
            # Use adaptive features
            urgency = self.calculate_urgency(actual_volume, expected_volume)
            adjusted_delay = self.adjust_delay_for_urgency(base_delay, urgency)
            ...

References:
- BestEx Research Adaptive Optimal: https://www.bestexresearch.com/insights/adaptive
- QuestDB Adaptive Algorithms: https://questdb.com/glossary/adaptive-trading-algorithms/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from libra.gateways.protocol import OrderSide, Tick


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class UrgencyLevel(str, Enum):
    """Predefined urgency levels."""

    PASSIVE = "passive"  # Minimize market impact
    LOW = "low"  # Below average urgency
    NORMAL = "normal"  # Standard execution
    HIGH = "high"  # Faster execution
    AGGRESSIVE = "aggressive"  # Maximum speed, higher impact


@dataclass
class AdaptiveConfig:
    """
    Configuration for adaptive execution features.

    Attributes:
        enable_urgency_adjustment: Dynamically adjust execution speed
        enable_price_improvement: Try to get better prices
        enable_spread_monitoring: Monitor and react to spread changes
        enable_fill_rate_adjustment: Adjust based on fill success rate

        base_urgency: Starting urgency multiplier (1.0 = normal)
        min_urgency: Minimum urgency multiplier
        max_urgency: Maximum urgency multiplier
        urgency_adjustment_rate: How quickly urgency changes

        volume_deviation_threshold: Volume change % to trigger adjustment
        spread_threshold_bps: Spread in bps to trigger adjustment
        fill_rate_threshold: Fill rate below this triggers adjustment

        price_improvement_bps: Target price improvement in basis points
        max_price_chase_bps: Maximum price to chase for fills
    """

    # Feature flags
    enable_urgency_adjustment: bool = True
    enable_price_improvement: bool = True
    enable_spread_monitoring: bool = True
    enable_fill_rate_adjustment: bool = True

    # Urgency parameters
    base_urgency: float = 1.0
    min_urgency: float = 0.3
    max_urgency: float = 3.0
    urgency_adjustment_rate: float = 0.2  # 20% adjustment per trigger

    # Thresholds
    volume_deviation_threshold: float = 0.30  # 30% deviation triggers
    spread_threshold_bps: float = 20.0  # Wide spread threshold
    fill_rate_threshold: float = 0.80  # 80% fill rate target

    # Price improvement
    price_improvement_bps: float = 2.0  # Target 2 bps improvement
    max_price_chase_bps: float = 10.0  # Max 10 bps chase

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.min_urgency <= 0:
            raise ValueError("min_urgency must be positive")
        if self.max_urgency <= self.min_urgency:
            raise ValueError("max_urgency must exceed min_urgency")
        if self.base_urgency < self.min_urgency or self.base_urgency > self.max_urgency:
            raise ValueError("base_urgency must be between min and max")


@dataclass
class AdaptiveState:
    """
    Tracks adaptive execution state.

    Updated during execution to track market conditions
    and execution performance.
    """

    # Current urgency
    current_urgency: float = 1.0

    # Volume tracking
    expected_volume: Decimal = Decimal("0")
    actual_volume: Decimal = Decimal("0")
    volume_deviation: float = 0.0

    # Fill tracking
    orders_submitted: int = 0
    orders_filled: int = 0
    fill_rate: float = 1.0

    # Price tracking
    arrival_price: Decimal | None = None
    current_price: Decimal | None = None
    avg_execution_price: Decimal | None = None

    # Spread tracking
    current_spread_bps: float = 0.0
    avg_spread_bps: float = 0.0

    # Adjustment counts
    urgency_increases: int = 0
    urgency_decreases: int = 0
    price_improvements: int = 0


@dataclass
class MarketConditions:
    """
    Current market conditions snapshot.

    Used for adaptive decision making.
    """

    timestamp_ns: int
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: Decimal | None = None
    volatility: float | None = None

    @property
    def mid(self) -> Decimal:
        """Mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        if self.mid == 0:
            return 0.0
        return float(self.spread / self.mid * 10000)


# =============================================================================
# Adaptive Mixin
# =============================================================================


class AdaptiveMixin:
    """
    Mixin class providing adaptive execution capabilities.

    Mix this into execution algorithm classes to add adaptive
    features that adjust execution based on market conditions.

    Features:
    - Urgency adjustment based on volume/fill rate
    - Price improvement calculations
    - Spread monitoring
    - Fill rate tracking

    Example:
        class AdaptiveTWAP(AdaptiveMixin, TWAPAlgorithm):
            def __init__(self, config, adaptive_config=None):
                TWAPAlgorithm.__init__(self, config)
                self.init_adaptive(adaptive_config)

            async def _execute_strategy(self, order):
                # Use adaptive features during execution
                ...
    """

    _adaptive_config: AdaptiveConfig
    _adaptive_state: AdaptiveState
    _log: logging.Logger

    def init_adaptive(self, config: AdaptiveConfig | None = None) -> None:
        """
        Initialize adaptive features.

        Call this in the subclass __init__ to set up adaptive state.

        Args:
            config: Adaptive configuration (uses defaults if None)
        """
        self._adaptive_config = config or AdaptiveConfig()
        self._adaptive_state = AdaptiveState(
            current_urgency=self._adaptive_config.base_urgency
        )
        self._log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def adaptive_config(self) -> AdaptiveConfig:
        """Get adaptive configuration."""
        return self._adaptive_config

    @property
    def adaptive_state(self) -> AdaptiveState:
        """Get current adaptive state."""
        return self._adaptive_state

    @property
    def current_urgency(self) -> float:
        """Get current urgency level."""
        return self._adaptive_state.current_urgency

    # -------------------------------------------------------------------------
    # Urgency Adjustment
    # -------------------------------------------------------------------------

    def adjust_urgency_for_volume(
        self,
        actual_volume: Decimal,
        expected_volume: Decimal,
    ) -> float:
        """
        Adjust urgency based on volume deviation.

        If actual volume exceeds expected, increase urgency to
        capture the liquidity. If volume is lower, decrease to
        avoid market impact.

        Args:
            actual_volume: Actual observed volume
            expected_volume: Expected volume from profile

        Returns:
            Updated urgency multiplier
        """
        if not self._adaptive_config.enable_urgency_adjustment:
            return self._adaptive_state.current_urgency

        if expected_volume == 0:
            return self._adaptive_state.current_urgency

        # Calculate deviation
        deviation = float((actual_volume - expected_volume) / expected_volume)
        self._adaptive_state.volume_deviation = deviation
        self._adaptive_state.actual_volume = actual_volume
        self._adaptive_state.expected_volume = expected_volume

        threshold = self._adaptive_config.volume_deviation_threshold
        adjustment_rate = self._adaptive_config.urgency_adjustment_rate

        if deviation > threshold:
            # Volume higher than expected - increase urgency
            self._adaptive_state.current_urgency *= 1 + adjustment_rate
            self._adaptive_state.urgency_increases += 1
            self._log.debug(
                "Urgency increased: volume %.1f%% above expected",
                deviation * 100,
            )

        elif deviation < -threshold:
            # Volume lower than expected - decrease urgency
            self._adaptive_state.current_urgency *= 1 - adjustment_rate
            self._adaptive_state.urgency_decreases += 1
            self._log.debug(
                "Urgency decreased: volume %.1f%% below expected",
                abs(deviation) * 100,
            )

        # Clamp to bounds
        self._adaptive_state.current_urgency = max(
            self._adaptive_config.min_urgency,
            min(self._adaptive_config.max_urgency, self._adaptive_state.current_urgency),
        )

        return self._adaptive_state.current_urgency

    def adjust_urgency_for_fill_rate(self) -> float:
        """
        Adjust urgency based on fill rate.

        If fills are below target, increase aggression.

        Returns:
            Updated urgency multiplier
        """
        if not self._adaptive_config.enable_fill_rate_adjustment:
            return self._adaptive_state.current_urgency

        if self._adaptive_state.orders_submitted == 0:
            return self._adaptive_state.current_urgency

        fill_rate = (
            self._adaptive_state.orders_filled / self._adaptive_state.orders_submitted
        )
        self._adaptive_state.fill_rate = fill_rate

        if fill_rate < self._adaptive_config.fill_rate_threshold:
            # Low fill rate - increase aggression
            adjustment = self._adaptive_config.urgency_adjustment_rate
            self._adaptive_state.current_urgency *= 1 + adjustment
            self._adaptive_state.current_urgency = min(
                self._adaptive_config.max_urgency,
                self._adaptive_state.current_urgency,
            )
            self._log.debug(
                "Urgency increased: fill rate %.1f%% below threshold",
                fill_rate * 100,
            )

        return self._adaptive_state.current_urgency

    # -------------------------------------------------------------------------
    # Delay Adjustment
    # -------------------------------------------------------------------------

    def adjust_delay_for_urgency(self, base_delay: float) -> float:
        """
        Adjust execution delay based on current urgency.

        Higher urgency = shorter delays.

        Args:
            base_delay: Base delay in seconds

        Returns:
            Adjusted delay in seconds
        """
        # Inverse relationship: higher urgency = shorter delay
        # urgency 0.5 -> delay * 2
        # urgency 1.0 -> delay * 1
        # urgency 2.0 -> delay * 0.5
        adjusted = base_delay / self._adaptive_state.current_urgency
        return max(0.1, adjusted)  # Minimum 100ms delay

    def adjust_quantity_for_urgency(self, base_quantity: Decimal) -> Decimal:
        """
        Adjust order quantity based on current urgency.

        Higher urgency = larger orders.

        Args:
            base_quantity: Base quantity

        Returns:
            Adjusted quantity
        """
        # Direct relationship: higher urgency = larger quantity
        factor = Decimal(str(self._adaptive_state.current_urgency))
        return base_quantity * factor

    # -------------------------------------------------------------------------
    # Price Improvement
    # -------------------------------------------------------------------------

    def calculate_improved_price(
        self,
        market_price: Decimal,
        side: OrderSide,
    ) -> Decimal:
        """
        Calculate price with improvement for limit orders.

        Tries to get better execution by placing limit orders
        slightly better than the current price.

        Args:
            market_price: Current market price
            side: Order side (BUY/SELL)

        Returns:
            Price with improvement applied
        """
        if not self._adaptive_config.enable_price_improvement:
            return market_price

        improvement_bps = Decimal(str(self._adaptive_config.price_improvement_bps))
        improvement = market_price * improvement_bps / Decimal("10000")

        from libra.gateways.protocol import OrderSide

        if side == OrderSide.BUY:
            # For buys, improve by offering less (lower price)
            improved_price = market_price - improvement
        else:
            # For sells, improve by asking more (higher price)
            improved_price = market_price + improvement

        self._adaptive_state.price_improvements += 1
        return improved_price

    def should_chase_price(
        self,
        target_price: Decimal,
        current_price: Decimal,
        side: OrderSide,
    ) -> bool:
        """
        Determine if we should chase the price for a fill.

        Args:
            target_price: Our target/limit price
            current_price: Current market price
            side: Order side

        Returns:
            True if we should adjust price to chase
        """
        from libra.gateways.protocol import OrderSide

        max_chase_bps = Decimal(str(self._adaptive_config.max_price_chase_bps))
        max_chase = target_price * max_chase_bps / Decimal("10000")

        if side == OrderSide.BUY:
            # For buys, market moved up - chase if within limit
            deviation = current_price - target_price
            return deviation <= max_chase
        else:
            # For sells, market moved down - chase if within limit
            deviation = target_price - current_price
            return deviation <= max_chase

    # -------------------------------------------------------------------------
    # Spread Monitoring
    # -------------------------------------------------------------------------

    def update_spread_tracking(self, tick: Tick) -> None:
        """
        Update spread tracking from tick data.

        Args:
            tick: Current market tick
        """
        if not self._adaptive_config.enable_spread_monitoring:
            return

        current_spread_bps = float(tick.spread_bps)
        self._adaptive_state.current_spread_bps = current_spread_bps

        # Update average (simple moving average)
        if self._adaptive_state.avg_spread_bps == 0:
            self._adaptive_state.avg_spread_bps = current_spread_bps
        else:
            alpha = 0.1  # Smoothing factor
            self._adaptive_state.avg_spread_bps = (
                alpha * current_spread_bps
                + (1 - alpha) * self._adaptive_state.avg_spread_bps
            )

    def is_spread_acceptable(self) -> bool:
        """
        Check if current spread is acceptable for execution.

        Returns:
            True if spread is below threshold
        """
        return (
            self._adaptive_state.current_spread_bps
            <= self._adaptive_config.spread_threshold_bps
        )

    # -------------------------------------------------------------------------
    # Fill Tracking
    # -------------------------------------------------------------------------

    def record_order_submission(self) -> None:
        """Record that an order was submitted."""
        self._adaptive_state.orders_submitted += 1

    def record_order_fill(self, filled: bool = True) -> None:
        """Record order fill result."""
        if filled:
            self._adaptive_state.orders_filled += 1

        # Update fill rate
        if self._adaptive_state.orders_submitted > 0:
            self._adaptive_state.fill_rate = (
                self._adaptive_state.orders_filled
                / self._adaptive_state.orders_submitted
            )

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def reset_adaptive_state(self) -> None:
        """Reset adaptive state for new execution."""
        self._adaptive_state = AdaptiveState(
            current_urgency=self._adaptive_config.base_urgency
        )

    def get_adaptive_summary(self) -> dict[str, float | int]:
        """Get summary of adaptive behavior during execution."""
        return {
            "final_urgency": self._adaptive_state.current_urgency,
            "urgency_increases": self._adaptive_state.urgency_increases,
            "urgency_decreases": self._adaptive_state.urgency_decreases,
            "fill_rate": self._adaptive_state.fill_rate,
            "price_improvements": self._adaptive_state.price_improvements,
            "avg_spread_bps": self._adaptive_state.avg_spread_bps,
            "volume_deviation": self._adaptive_state.volume_deviation,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def urgency_from_level(level: UrgencyLevel) -> float:
    """
    Get urgency multiplier from predefined level.

    Args:
        level: Urgency level enum

    Returns:
        Urgency multiplier
    """
    mapping = {
        UrgencyLevel.PASSIVE: 0.5,
        UrgencyLevel.LOW: 0.75,
        UrgencyLevel.NORMAL: 1.0,
        UrgencyLevel.HIGH: 1.5,
        UrgencyLevel.AGGRESSIVE: 2.5,
    }
    return mapping[level]


def create_adaptive_config(
    urgency_level: UrgencyLevel = UrgencyLevel.NORMAL,
    enable_all: bool = True,
) -> AdaptiveConfig:
    """
    Create adaptive config from urgency level.

    Args:
        urgency_level: Predefined urgency level
        enable_all: Enable all adaptive features

    Returns:
        AdaptiveConfig instance
    """
    base_urgency = urgency_from_level(urgency_level)

    return AdaptiveConfig(
        enable_urgency_adjustment=enable_all,
        enable_price_improvement=enable_all,
        enable_spread_monitoring=enable_all,
        enable_fill_rate_adjustment=enable_all,
        base_urgency=base_urgency,
    )
