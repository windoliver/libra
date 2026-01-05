"""
Position Sizing Algorithms for LIBRA.

Implements industry-standard position sizing methods:
- Fixed Percentage: Risk fixed % of equity per trade
- Volatility Adjusted (ATR): Scale position by market volatility
- Kelly Criterion: Mathematically optimal sizing for growth

All methods return consistent PositionSizeResult for easy integration.

References:
    - Kelly: https://www.quantifiedstrategies.com/kelly-criterion-position-sizing/
    - ATR: https://raposa.trade/blog/atr-and-how-top-traders-size-their-positions/
    - Volatility: https://www.luxalgo.com/blog/5-position-sizing-methods-for-high-volatility-trades/
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass
class PositionSizeResult:
    """
    Result of position size calculation.

    Provides the calculated size along with metadata about
    the calculation method and risk parameters used.

    Attributes:
        size: Position size in base currency units
        method: Name of sizing method used
        risk_amount: Dollar amount at risk for this position
        metadata: Additional calculation details
    """

    size: Decimal
    method: str
    risk_amount: Decimal
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if result represents a valid position."""
        return self.size > Decimal("0")


class PositionSizer(ABC):
    """
    Abstract base class for position sizing algorithms.

    All position sizers implement a common interface for
    calculating position sizes based on risk parameters.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this sizing method."""
        ...

    @abstractmethod
    def calculate(
        self,
        equity: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal | None = None,
        **kwargs: Any,
    ) -> PositionSizeResult:
        """
        Calculate position size.

        Args:
            equity: Current account equity
            entry_price: Planned entry price
            stop_loss_price: Stop loss price (optional for some methods)
            **kwargs: Method-specific parameters

        Returns:
            PositionSizeResult with calculated size
        """
        ...


@dataclass
class FixedPercentageSizer(PositionSizer):
    """
    Fixed percentage of equity risk per trade.

    The most common and recommended approach for most traders.
    Typically risk 1-3% of equity per trade.

    Formula:
        Position Size = (Equity × Risk%) / |Entry - StopLoss|

    Examples:
        sizer = FixedPercentageSizer(risk_percent=Decimal("0.02"))  # 2% risk

        result = sizer.calculate(
            equity=Decimal("100000"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),  # $1000 risk per unit
        )
        # Result: size=2.0 (risking $2000 total)

    Attributes:
        risk_percent: Fraction of equity to risk (0.02 = 2%)
        max_position_pct: Maximum position as % of equity (cap)
    """

    risk_percent: Decimal = Decimal("0.02")  # 2% default
    max_position_pct: Decimal = Decimal("0.25")  # 25% max position

    @property
    def name(self) -> str:
        return "fixed_percentage"

    def calculate(
        self,
        equity: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal | None = None,
        **kwargs: Any,
    ) -> PositionSizeResult:
        """
        Calculate position size based on fixed percentage risk.

        Args:
            equity: Current account equity
            entry_price: Planned entry price
            stop_loss_price: Stop loss price (required)

        Returns:
            PositionSizeResult with calculated size

        Raises:
            ValueError: If stop_loss_price not provided or equals entry
        """
        if stop_loss_price is None:
            raise ValueError("stop_loss_price required for fixed percentage sizing")

        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == Decimal("0"):
            raise ValueError("Entry and stop loss prices cannot be equal")

        risk_amount = equity * self.risk_percent
        size = risk_amount / risk_per_unit

        # Apply maximum position cap
        max_notional = equity * self.max_position_pct
        max_size = max_notional / entry_price
        size = min(size, max_size)

        return PositionSizeResult(
            size=size.quantize(Decimal("0.00000001")),
            method=self.name,
            risk_amount=risk_amount,
            metadata={
                "risk_percent": float(self.risk_percent),
                "risk_per_unit": float(risk_per_unit),
                "max_position_pct": float(self.max_position_pct),
                "capped": size == max_size,
            },
        )


@dataclass
class VolatilityAdjustedSizer(PositionSizer):
    """
    ATR-based volatility-adjusted position sizing.

    Scales position size inversely to current volatility.
    Higher volatility = smaller position to maintain consistent risk.

    Formula:
        Position Size = (Equity × Risk%) / (ATR × Multiplier)

    This approach:
    - Reduces position size during volatile markets
    - Increases position size during calm markets
    - Maintains consistent dollar risk regardless of volatility

    Examples:
        sizer = VolatilityAdjustedSizer(
            risk_percent=Decimal("0.02"),
            atr_multiplier=Decimal("2.0"),
        )

        result = sizer.calculate(
            equity=Decimal("100000"),
            entry_price=Decimal("50000"),
            atr=Decimal("500"),  # Current 14-period ATR
        )

    Attributes:
        risk_percent: Fraction of equity to risk
        atr_multiplier: Multiplier for ATR (typically 1.5-3.0)
        max_position_pct: Maximum position as % of equity
    """

    risk_percent: Decimal = Decimal("0.02")
    atr_multiplier: Decimal = Decimal("2.0")
    max_position_pct: Decimal = Decimal("0.25")

    @property
    def name(self) -> str:
        return "volatility_adjusted"

    def calculate(
        self,
        equity: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal | None = None,
        atr: Decimal | None = None,
        **kwargs: Any,
    ) -> PositionSizeResult:
        """
        Calculate position size based on ATR volatility.

        Args:
            equity: Current account equity
            entry_price: Planned entry price
            stop_loss_price: Not used (ATR defines risk)
            atr: Current Average True Range value (required)

        Returns:
            PositionSizeResult with calculated size

        Raises:
            ValueError: If ATR not provided or is zero
        """
        if atr is None or atr <= Decimal("0"):
            raise ValueError("Positive ATR required for volatility-adjusted sizing")

        risk_amount = equity * self.risk_percent
        risk_per_unit = atr * self.atr_multiplier
        size = risk_amount / risk_per_unit

        # Apply maximum position cap
        max_notional = equity * self.max_position_pct
        max_size = max_notional / entry_price
        size = min(size, max_size)

        return PositionSizeResult(
            size=size.quantize(Decimal("0.00000001")),
            method=self.name,
            risk_amount=risk_amount,
            metadata={
                "risk_percent": float(self.risk_percent),
                "atr": float(atr),
                "atr_multiplier": float(self.atr_multiplier),
                "implied_stop_distance": float(risk_per_unit),
                "capped": size == max_size,
            },
        )


@dataclass
class KellyCriterionSizer(PositionSizer):
    """
    Kelly Criterion position sizing.

    Mathematically optimal sizing for maximizing long-term growth rate.
    Based on historical win rate and reward/risk ratio.

    Formula:
        Kelly% = W - (1 - W) / R

        where:
            W = Win probability (historical win rate)
            R = Reward/Risk ratio (avg win / avg loss)

    Best Practices:
        - Use fractional Kelly (0.25-0.5) to reduce volatility
        - Cap maximum position at 20-25% of equity
        - Requires sufficient trade history for accurate estimates
        - Negative Kelly = don't trade (no edge)

    Examples:
        sizer = KellyCriterionSizer(
            kelly_fraction=Decimal("0.5"),  # Half-Kelly
            max_position_pct=Decimal("0.20"),
        )

        result = sizer.calculate(
            equity=Decimal("100000"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),
            win_rate=Decimal("0.55"),      # 55% win rate
            avg_win=Decimal("1000"),       # Average win
            avg_loss=Decimal("500"),       # Average loss
        )

    Attributes:
        kelly_fraction: Fraction of full Kelly to use (0.5 = half-Kelly)
        max_position_pct: Maximum position as % of equity
        min_trades_required: Minimum trade history for valid calculation
    """

    kelly_fraction: Decimal = Decimal("0.5")  # Half-Kelly (conservative)
    max_position_pct: Decimal = Decimal("0.20")  # Cap at 20%
    min_trades_required: int = 30  # Minimum trades for statistical validity

    @property
    def name(self) -> str:
        return "kelly_criterion"

    def calculate(
        self,
        equity: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal | None = None,
        win_rate: Decimal | None = None,
        avg_win: Decimal | None = None,
        avg_loss: Decimal | None = None,
        **kwargs: Any,
    ) -> PositionSizeResult:
        """
        Calculate position size using Kelly Criterion.

        Args:
            equity: Current account equity
            entry_price: Planned entry price
            stop_loss_price: Stop loss price (required for position sizing)
            win_rate: Historical win rate (0.0 to 1.0)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)

        Returns:
            PositionSizeResult with calculated size

        Raises:
            ValueError: If required parameters missing or invalid
        """
        # Validate inputs
        if None in (win_rate, avg_win, avg_loss):
            raise ValueError("Kelly requires win_rate, avg_win, and avg_loss")

        if stop_loss_price is None:
            raise ValueError("stop_loss_price required for Kelly sizing")

        if avg_loss <= Decimal("0"):
            raise ValueError("avg_loss must be positive")

        if not (Decimal("0") <= win_rate <= Decimal("1")):
            raise ValueError("win_rate must be between 0 and 1")

        # Calculate Kelly percentage
        reward_risk = abs(avg_win / avg_loss)
        raw_kelly = win_rate - ((Decimal("1") - win_rate) / reward_risk)

        # Apply fractional Kelly and bounds
        if raw_kelly <= Decimal("0"):
            # Negative Kelly = no edge, don't trade
            return PositionSizeResult(
                size=Decimal("0"),
                method=self.name,
                risk_amount=Decimal("0"),
                metadata={
                    "raw_kelly": float(raw_kelly),
                    "no_edge": True,
                    "win_rate": float(win_rate),
                    "reward_risk": float(reward_risk),
                },
            )

        kelly_pct = raw_kelly * self.kelly_fraction
        kelly_pct = min(kelly_pct, self.max_position_pct)

        # Calculate position size
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == Decimal("0"):
            raise ValueError("Entry and stop loss prices cannot be equal")

        risk_amount = equity * kelly_pct
        size = risk_amount / risk_per_unit

        # Apply notional cap
        max_notional = equity * self.max_position_pct
        max_size = max_notional / entry_price
        size = min(size, max_size)

        return PositionSizeResult(
            size=size.quantize(Decimal("0.00000001")),
            method=self.name,
            risk_amount=risk_amount,
            metadata={
                "raw_kelly": float(raw_kelly),
                "fractional_kelly": float(kelly_pct),
                "kelly_fraction": float(self.kelly_fraction),
                "win_rate": float(win_rate),
                "reward_risk": float(reward_risk),
                "capped": kelly_pct == self.max_position_pct or size == max_size,
            },
        )


@dataclass
class FixedQuantitySizer(PositionSizer):
    """
    Fixed quantity position sizing.

    Always uses the same position size regardless of price or volatility.
    Simple but doesn't adapt to changing conditions.

    Useful for:
    - Initial testing
    - Very small accounts
    - Specific lot size requirements

    Attributes:
        quantity: Fixed position size to use
        max_notional: Maximum notional value cap
    """

    quantity: Decimal = Decimal("1.0")
    max_notional: Decimal = Decimal("100000")

    @property
    def name(self) -> str:
        return "fixed_quantity"

    def calculate(
        self,
        equity: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal | None = None,
        **kwargs: Any,
    ) -> PositionSizeResult:
        """Calculate fixed quantity position size."""
        # Check notional value limit
        notional = self.quantity * entry_price
        if notional > self.max_notional:
            size = self.max_notional / entry_price
        else:
            size = self.quantity

        # Calculate implied risk if stop loss provided
        if stop_loss_price is not None:
            risk_amount = size * abs(entry_price - stop_loss_price)
        else:
            risk_amount = Decimal("0")

        return PositionSizeResult(
            size=size.quantize(Decimal("0.00000001")),
            method=self.name,
            risk_amount=risk_amount,
            metadata={
                "fixed_quantity": float(self.quantity),
                "notional_capped": notional > self.max_notional,
            },
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_sizer(
    method: str,
    **kwargs: Any,
) -> PositionSizer:
    """
    Factory function to create position sizers.

    Args:
        method: Sizing method name
        **kwargs: Method-specific configuration

    Returns:
        Configured PositionSizer instance

    Raises:
        ValueError: If method not recognized
    """
    sizers = {
        "fixed_percentage": FixedPercentageSizer,
        "volatility_adjusted": VolatilityAdjustedSizer,
        "kelly_criterion": KellyCriterionSizer,
        "fixed_quantity": FixedQuantitySizer,
    }

    if method not in sizers:
        valid = ", ".join(sizers.keys())
        raise ValueError(f"Unknown sizing method '{method}'. Valid: {valid}")

    # Convert float kwargs to Decimal where needed
    decimal_fields = {
        "risk_percent",
        "max_position_pct",
        "atr_multiplier",
        "kelly_fraction",
        "quantity",
        "max_notional",
    }

    converted_kwargs = {}
    for key, value in kwargs.items():
        if key in decimal_fields and not isinstance(value, Decimal):
            converted_kwargs[key] = Decimal(str(value))
        else:
            converted_kwargs[key] = value

    return sizers[method](**converted_kwargs)
