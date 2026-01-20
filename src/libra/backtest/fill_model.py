"""
FillModel for realistic order fill simulation.

Implements Issue #107: Realistic order fills with slippage, partial fills,
and queue position modeling for backtesting.

References:
- NautilusTrader: https://nautilustrader.io/docs/latest/concepts/backtesting/
- Zipline slippage models: Volume-weighted quadratic impact

Fill Model Types:
- IMMEDIATE: Fill at order price (simple, current behavior)
- PROBABILISTIC: Probability-based fills with queue modeling
- ORDER_BOOK: Fill against actual book levels (requires L2 data)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libra.gateways.protocol import Order, OrderBook, OrderSide


logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class FillModelType(Enum):
    """Fill simulation model types."""

    IMMEDIATE = "immediate"  # Fill at order price (no slippage)
    PROBABILISTIC = "probabilistic"  # Probability-based with queue modeling
    ORDER_BOOK = "order_book"  # Fill against order book levels


class SlippageType(Enum):
    """Slippage calculation methods."""

    NONE = "none"  # No slippage
    FIXED = "fixed"  # Fixed ticks/bps
    VOLUME = "volume"  # Volume-weighted impact
    STOCHASTIC = "stochastic"  # Random within bounds


# =============================================================================
# Fill Result
# =============================================================================


@dataclass
class FillResult:
    """Result of a fill simulation."""

    filled: bool  # Whether any fill occurred
    fill_price: Decimal | None  # Actual fill price
    fill_quantity: Decimal  # Quantity filled (may be partial)
    slippage_ticks: int  # Ticks of slippage applied
    is_partial: bool  # True if partial fill
    queue_position: float  # Estimated queue position when filled
    reason: str  # Reason for fill/no-fill

    @classmethod
    def no_fill(cls, reason: str) -> FillResult:
        """Create no-fill result."""
        return cls(
            filled=False,
            fill_price=None,
            fill_quantity=Decimal("0"),
            slippage_ticks=0,
            is_partial=False,
            queue_position=1.0,
            reason=reason,
        )

    @classmethod
    def full_fill(
        cls,
        price: Decimal,
        quantity: Decimal,
        slippage_ticks: int = 0,
        queue_position: float = 0.0,
    ) -> FillResult:
        """Create full fill result."""
        return cls(
            filled=True,
            fill_price=price,
            fill_quantity=quantity,
            slippage_ticks=slippage_ticks,
            is_partial=False,
            queue_position=queue_position,
            reason="filled",
        )


# =============================================================================
# Fill Model Configuration
# =============================================================================


@dataclass
class FillModel:
    """
    Configure realistic fill simulation for backtesting.

    This model determines how orders are filled during simulation,
    including slippage, partial fills, and queue position effects.

    Example:
        # Conservative model for HFT strategies
        model = FillModel(
            model_type=FillModelType.PROBABILISTIC,
            prob_fill_on_limit=0.5,  # 50% fill probability at touch
            queue_position_pct=0.8,  # Back of queue
            slippage_type=SlippageType.VOLUME,
        )

        # Optimistic model for position trading
        model = FillModel(
            model_type=FillModelType.IMMEDIATE,
            prob_fill_on_limit=1.0,
            enable_partial_fills=False,
        )
    """

    # Model type
    model_type: FillModelType = FillModelType.PROBABILISTIC

    # Slippage configuration
    slippage_type: SlippageType = SlippageType.FIXED
    prob_slippage: float = 0.3  # Probability of slippage occurring
    slippage_ticks: int = 1  # Ticks of slippage when it occurs
    max_slippage_ticks: int = 5  # Maximum slippage cap
    slippage_bps: Decimal = Decimal("5")  # Basis points for fixed slippage
    volume_impact: Decimal = Decimal("0.1")  # Price impact for volume model

    # Limit order fill probability
    prob_fill_on_limit: float = 0.7  # Probability limit fills at touch
    prob_fill_on_stop: float = 0.95  # Probability stop triggers
    prob_fill_on_stop_limit: float = 0.8  # Probability stop-limit fills

    # Queue position simulation
    queue_position_pct: float = 0.5  # Where in queue (0=front, 1=back)
    queue_position_std: float = 0.1  # Standard deviation for randomization

    # Partial fills
    enable_partial_fills: bool = True
    min_fill_pct: float = 0.1  # Minimum fill percentage
    max_partial_fills: int = 5  # Maximum number of partial fills

    # Volume constraints
    max_volume_pct: float = 0.1  # Max percentage of bar volume
    volume_participation_rate: float = 0.05  # Default participation rate

    # Tick size (for slippage calculation)
    tick_size: Decimal = Decimal("0.01")

    # Random seed for reproducibility
    random_seed: int | None = None

    # Internal state
    _rng: random.Random = field(default_factory=random.Random, repr=False)

    def __post_init__(self) -> None:
        """Initialize random state."""
        if self.random_seed is not None:
            self._rng = random.Random(self.random_seed)
        self._validate()

    def _validate(self) -> None:
        """Validate configuration."""
        if not 0 <= self.prob_slippage <= 1:
            raise ValueError("prob_slippage must be between 0 and 1")
        if not 0 <= self.prob_fill_on_limit <= 1:
            raise ValueError("prob_fill_on_limit must be between 0 and 1")
        if not 0 <= self.prob_fill_on_stop <= 1:
            raise ValueError("prob_fill_on_stop must be between 0 and 1")
        if not 0 <= self.queue_position_pct <= 1:
            raise ValueError("queue_position_pct must be between 0 and 1")
        if not 0 < self.min_fill_pct <= 1:
            raise ValueError("min_fill_pct must be between 0 and 1")
        if self.slippage_ticks < 0:
            raise ValueError("slippage_ticks must be non-negative")
        if self.max_slippage_ticks < self.slippage_ticks:
            raise ValueError("max_slippage_ticks must be >= slippage_ticks")

    # =========================================================================
    # Market Order Fills
    # =========================================================================

    def simulate_market_fill(
        self,
        side: OrderSide,
        quantity: Decimal,
        current_price: Decimal,
        bar_volume: Decimal | None = None,
    ) -> FillResult:
        """
        Simulate market order fill with potential slippage.

        Market orders always fill but may experience slippage based on:
        - Probability of slippage occurring
        - Slippage model (fixed/volume/stochastic)
        - Order size relative to volume

        Args:
            side: Order side (BUY/SELL)
            quantity: Order quantity
            current_price: Current market price
            bar_volume: Volume of current bar (optional, for volume model)

        Returns:
            FillResult with fill details
        """
        from libra.gateways.protocol import OrderSide

        if self.model_type == FillModelType.IMMEDIATE:
            return FillResult.full_fill(current_price, quantity)

        # Calculate slippage
        slippage_ticks = self._calculate_slippage(quantity, bar_volume)

        # Apply slippage to price
        tick_adjustment = Decimal(str(slippage_ticks)) * self.tick_size
        if side == OrderSide.BUY:
            fill_price = current_price + tick_adjustment
        else:
            fill_price = current_price - tick_adjustment

        # Ensure non-negative price
        fill_price = max(fill_price, self.tick_size)

        # Check for partial fill based on volume
        fill_qty = quantity
        is_partial = False

        if bar_volume is not None and bar_volume > 0 and self.enable_partial_fills:
            max_fill = bar_volume * Decimal(str(self.max_volume_pct))
            if quantity > max_fill:
                fill_qty = max(max_fill, quantity * Decimal(str(self.min_fill_pct)))
                is_partial = True

        return FillResult(
            filled=True,
            fill_price=fill_price,
            fill_quantity=fill_qty,
            slippage_ticks=slippage_ticks,
            is_partial=is_partial,
            queue_position=0.0,  # Market orders don't queue
            reason="market_fill",
        )

    def _calculate_slippage(
        self,
        quantity: Decimal,
        bar_volume: Decimal | None,
    ) -> int:
        """Calculate slippage in ticks."""
        if self.slippage_type == SlippageType.NONE:
            return 0

        # Check if slippage occurs
        if self._rng.random() > self.prob_slippage:
            return 0

        if self.slippage_type == SlippageType.FIXED:
            return self.slippage_ticks

        if self.slippage_type == SlippageType.VOLUME:
            if bar_volume is None or bar_volume == 0:
                return self.slippage_ticks

            # Volume impact: slippage increases with order size
            volume_share = float(quantity / bar_volume)
            impact = float(self.volume_impact) * (volume_share ** 2)
            ticks = int(impact / float(self.tick_size)) + 1
            return min(ticks, self.max_slippage_ticks)

        # STOCHASTIC
        return self._rng.randint(0, self.max_slippage_ticks)

    # =========================================================================
    # Limit Order Fills
    # =========================================================================

    def simulate_limit_fill(
        self,
        side: OrderSide,
        quantity: Decimal,
        limit_price: Decimal,
        current_price: Decimal,
        bar_volume: Decimal | None = None,
        time_at_price_pct: float = 0.0,
    ) -> FillResult:
        """
        Simulate limit order fill based on queue position and volume.

        Limit orders may not fill even when price touches the level due to:
        - Queue position (orders ahead need to fill first)
        - Available volume at the price level
        - Random fill probability

        Args:
            side: Order side (BUY/SELL)
            quantity: Order quantity
            limit_price: Limit price
            current_price: Current market price
            bar_volume: Volume of current bar (optional)
            time_at_price_pct: Percentage of bar where price was at limit

        Returns:
            FillResult with fill details or no-fill
        """
        from libra.gateways.protocol import OrderSide

        # Check if price is favorable
        is_favorable = (
            (side == OrderSide.BUY and current_price <= limit_price)
            or (side == OrderSide.SELL and current_price >= limit_price)
        )

        if not is_favorable:
            return FillResult.no_fill("price_not_reached")

        if self.model_type == FillModelType.IMMEDIATE:
            return FillResult.full_fill(limit_price, quantity)

        # Probabilistic fill model
        # Adjust probability based on queue position and time at price
        queue_pos = self._get_queue_position()
        fill_prob = self.prob_fill_on_limit * (1 - queue_pos) * max(0.1, time_at_price_pct + 0.5)

        if self._rng.random() > fill_prob:
            return FillResult.no_fill(f"queue_position_{queue_pos:.2f}")

        # Calculate fill quantity based on volume
        fill_qty = quantity
        is_partial = False

        if bar_volume is not None and bar_volume > 0 and self.enable_partial_fills:
            # Volume available at this price level
            volume_at_price = bar_volume * Decimal(str(self.volume_participation_rate))
            volume_ahead = volume_at_price * Decimal(str(queue_pos))
            available = volume_at_price - volume_ahead

            if available <= 0:
                return FillResult.no_fill("no_volume_available")

            if available < quantity:
                fill_qty = max(available, quantity * Decimal(str(self.min_fill_pct)))
                is_partial = True

        return FillResult(
            filled=True,
            fill_price=limit_price,
            fill_quantity=fill_qty,
            slippage_ticks=0,  # Limit orders fill at limit price
            is_partial=is_partial,
            queue_position=queue_pos,
            reason="limit_fill",
        )

    def _get_queue_position(self) -> float:
        """Get randomized queue position."""
        pos = self._rng.gauss(self.queue_position_pct, self.queue_position_std)
        return max(0.0, min(1.0, pos))

    # =========================================================================
    # Stop Order Fills
    # =========================================================================

    def simulate_stop_fill(
        self,
        side: OrderSide,
        quantity: Decimal,
        stop_price: Decimal,
        current_price: Decimal,
        bar_high: Decimal | None = None,
        bar_low: Decimal | None = None,
    ) -> FillResult:
        """
        Simulate stop order trigger and fill.

        Stop orders trigger when price reaches stop level, then fill
        as market orders with potential slippage.

        Args:
            side: Order side (BUY/SELL)
            quantity: Order quantity
            stop_price: Stop trigger price
            current_price: Current market price
            bar_high: Bar high (to check if stop was triggered)
            bar_low: Bar low (to check if stop was triggered)

        Returns:
            FillResult with fill details
        """
        from libra.gateways.protocol import OrderSide

        # Check if stop is triggered
        is_triggered = False

        if side == OrderSide.BUY:
            # Buy stop triggers when price rises to stop
            if bar_high is not None:
                is_triggered = bar_high >= stop_price
            else:
                is_triggered = current_price >= stop_price
        else:
            # Sell stop triggers when price falls to stop
            if bar_low is not None:
                is_triggered = bar_low <= stop_price
            else:
                is_triggered = current_price <= stop_price

        if not is_triggered:
            return FillResult.no_fill("stop_not_triggered")

        # Probabilistic trigger (gap risk, etc.)
        if self.model_type == FillModelType.PROBABILISTIC:
            if self._rng.random() > self.prob_fill_on_stop:
                return FillResult.no_fill("stop_trigger_failed")

        # Stop triggered - fill as market order at stop price (with slippage)
        return self.simulate_market_fill(
            side=side,
            quantity=quantity,
            current_price=stop_price,
            bar_volume=None,
        )

    # =========================================================================
    # Order Book Fills
    # =========================================================================

    def simulate_order_book_fill(
        self,
        side: OrderSide,
        quantity: Decimal,
        order_book: OrderBook,
        is_market: bool = True,
    ) -> FillResult:
        """
        Simulate fill against actual order book levels.

        Uses L2/L3 data to fill sequentially through price levels,
        providing realistic fill simulation for larger orders.

        Args:
            side: Order side (BUY/SELL)
            quantity: Order quantity
            order_book: Current order book snapshot
            is_market: True for market order, False for limit at best

        Returns:
            FillResult with weighted average fill price
        """
        from libra.gateways.protocol import OrderSide

        if self.model_type != FillModelType.ORDER_BOOK:
            # Fall back to simple fill
            if side == OrderSide.BUY and order_book.asks:
                return FillResult.full_fill(order_book.asks[0][0], quantity)
            if side == OrderSide.SELL and order_book.bids:
                return FillResult.full_fill(order_book.bids[0][0], quantity)
            return FillResult.no_fill("empty_order_book")

        # Get relevant side of book
        levels = order_book.asks if side == OrderSide.BUY else order_book.bids

        if not levels:
            return FillResult.no_fill("empty_order_book")

        # Walk through levels
        remaining = quantity
        total_cost = Decimal("0")
        levels_consumed = 0

        for price, size in levels:
            if remaining <= 0:
                break

            fill_at_level = min(remaining, size)
            total_cost += fill_at_level * price
            remaining -= fill_at_level
            levels_consumed += 1

        filled_qty = quantity - remaining

        if filled_qty == 0:
            return FillResult.no_fill("insufficient_liquidity")

        # Calculate VWAP
        vwap = total_cost / filled_qty

        # Calculate slippage in ticks from best price
        best_price = levels[0][0]
        if side == OrderSide.BUY:
            slippage = (vwap - best_price) / self.tick_size
        else:
            slippage = (best_price - vwap) / self.tick_size

        return FillResult(
            filled=True,
            fill_price=vwap,
            fill_quantity=filled_qty,
            slippage_ticks=int(slippage),
            is_partial=remaining > 0,
            queue_position=0.0,
            reason=f"book_fill_{levels_consumed}_levels",
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def with_seed(self, seed: int) -> FillModel:
        """Create copy with specific random seed."""
        return FillModel(
            model_type=self.model_type,
            slippage_type=self.slippage_type,
            prob_slippage=self.prob_slippage,
            slippage_ticks=self.slippage_ticks,
            max_slippage_ticks=self.max_slippage_ticks,
            slippage_bps=self.slippage_bps,
            volume_impact=self.volume_impact,
            prob_fill_on_limit=self.prob_fill_on_limit,
            prob_fill_on_stop=self.prob_fill_on_stop,
            prob_fill_on_stop_limit=self.prob_fill_on_stop_limit,
            queue_position_pct=self.queue_position_pct,
            queue_position_std=self.queue_position_std,
            enable_partial_fills=self.enable_partial_fills,
            min_fill_pct=self.min_fill_pct,
            max_partial_fills=self.max_partial_fills,
            max_volume_pct=self.max_volume_pct,
            volume_participation_rate=self.volume_participation_rate,
            tick_size=self.tick_size,
            random_seed=seed,
        )


# =============================================================================
# Preset Fill Models
# =============================================================================


def create_immediate_model() -> FillModel:
    """Create simple immediate fill model (no slippage)."""
    return FillModel(
        model_type=FillModelType.IMMEDIATE,
        slippage_type=SlippageType.NONE,
        enable_partial_fills=False,
    )


def create_realistic_model(
    slippage_bps: Decimal = Decimal("5"),
    fill_probability: float = 0.7,
    queue_position: float = 0.5,
) -> FillModel:
    """
    Create realistic probabilistic fill model.

    Good default for most backtesting scenarios.
    """
    return FillModel(
        model_type=FillModelType.PROBABILISTIC,
        slippage_type=SlippageType.FIXED,
        slippage_bps=slippage_bps,
        prob_slippage=0.3,
        slippage_ticks=1,
        prob_fill_on_limit=fill_probability,
        queue_position_pct=queue_position,
        enable_partial_fills=True,
    )


def create_conservative_model() -> FillModel:
    """
    Create conservative fill model for realistic HFT simulation.

    Uses pessimistic assumptions about queue position and fills.
    """
    return FillModel(
        model_type=FillModelType.PROBABILISTIC,
        slippage_type=SlippageType.VOLUME,
        prob_slippage=0.5,
        slippage_ticks=2,
        max_slippage_ticks=10,
        volume_impact=Decimal("0.2"),
        prob_fill_on_limit=0.3,  # Only 30% fill at touch
        prob_fill_on_stop=0.85,  # Some stop slippage
        queue_position_pct=0.8,  # Back of queue
        queue_position_std=0.15,
        enable_partial_fills=True,
        min_fill_pct=0.05,
        max_volume_pct=0.02,  # Only 2% of volume
    )


def create_order_book_model(tick_size: Decimal = Decimal("0.01")) -> FillModel:
    """Create order book fill model for L2 data."""
    return FillModel(
        model_type=FillModelType.ORDER_BOOK,
        slippage_type=SlippageType.NONE,  # Slippage from book walk
        tick_size=tick_size,
        enable_partial_fills=True,
    )
