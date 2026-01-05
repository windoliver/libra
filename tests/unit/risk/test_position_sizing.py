"""Tests for Position Sizing algorithms."""

from decimal import Decimal

import pytest

from libra.risk.position_sizing import (
    FixedPercentageSizer,
    FixedQuantitySizer,
    KellyCriterionSizer,
    PositionSizeResult,
    VolatilityAdjustedSizer,
    create_sizer,
)


class TestPositionSizeResult:
    """Tests for PositionSizeResult."""

    def test_valid_result(self):
        """Valid result with positive size."""
        result = PositionSizeResult(
            size=Decimal("0.5"),
            method="test",
            risk_amount=Decimal("1000"),
        )

        assert result.is_valid is True

    def test_invalid_result_zero_size(self):
        """Invalid result with zero size."""
        result = PositionSizeResult(
            size=Decimal("0"),
            method="test",
            risk_amount=Decimal("0"),
        )

        assert result.is_valid is False


class TestFixedPercentageSizer:
    """Tests for FixedPercentageSizer."""

    def test_basic_calculation(self):
        """Basic fixed percentage calculation."""
        sizer = FixedPercentageSizer(
            risk_percent=Decimal("0.02"),
            max_position_pct=Decimal("1.0"),  # No cap for this test
        )

        result = sizer.calculate(
            equity=Decimal("100000"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),  # $1000 risk per unit
        )

        # 2% of 100k = $2000 risk
        # $2000 / $1000 per unit = 2 units
        assert result.size == Decimal("2.00000000")
        assert result.method == "fixed_percentage"
        assert result.risk_amount == Decimal("2000")

    def test_requires_stop_loss(self):
        """Stop loss price is required."""
        sizer = FixedPercentageSizer()

        with pytest.raises(ValueError, match="stop_loss_price required"):
            sizer.calculate(
                equity=Decimal("100000"),
                entry_price=Decimal("50000"),
                stop_loss_price=None,
            )

    def test_stop_loss_equals_entry_error(self):
        """Entry and stop loss cannot be equal."""
        sizer = FixedPercentageSizer()

        with pytest.raises(ValueError, match="cannot be equal"):
            sizer.calculate(
                equity=Decimal("100000"),
                entry_price=Decimal("50000"),
                stop_loss_price=Decimal("50000"),
            )

    def test_max_position_cap(self):
        """Position size capped at max percentage."""
        sizer = FixedPercentageSizer(
            risk_percent=Decimal("0.10"),  # 10%
            max_position_pct=Decimal("0.05"),  # 5% cap
        )

        result = sizer.calculate(
            equity=Decimal("100000"),
            entry_price=Decimal("1000"),
            stop_loss_price=Decimal("999"),  # Tiny $1 risk per unit
        )

        # Without cap: 10k / 1 = 10,000 units
        # With 5% cap: 5k notional / 1000 = 5 units max
        assert result.size == Decimal("5.00000000")
        assert result.metadata["capped"] is True

    def test_name_property(self):
        """Sizer has correct name."""
        sizer = FixedPercentageSizer()
        assert sizer.name == "fixed_percentage"


class TestVolatilityAdjustedSizer:
    """Tests for VolatilityAdjustedSizer."""

    def test_basic_calculation(self):
        """Basic ATR-based calculation."""
        sizer = VolatilityAdjustedSizer(
            risk_percent=Decimal("0.02"),
            atr_multiplier=Decimal("2.0"),
            max_position_pct=Decimal("1.0"),  # No cap for this test
        )

        result = sizer.calculate(
            equity=Decimal("100000"),
            entry_price=Decimal("50000"),
            atr=Decimal("500"),  # $500 ATR
        )

        # 2% of 100k = $2000 risk
        # ATR * 2 = $1000 risk per unit
        # $2000 / $1000 = 2 units
        assert result.size == Decimal("2.00000000")
        assert result.method == "volatility_adjusted"
        assert result.metadata["atr"] == 500.0
        assert result.metadata["implied_stop_distance"] == 1000.0

    def test_requires_atr(self):
        """ATR value is required."""
        sizer = VolatilityAdjustedSizer()

        with pytest.raises(ValueError, match="ATR required"):
            sizer.calculate(
                equity=Decimal("100000"),
                entry_price=Decimal("50000"),
                atr=None,
            )

    def test_zero_atr_error(self):
        """ATR cannot be zero."""
        sizer = VolatilityAdjustedSizer()

        with pytest.raises(ValueError, match="Positive ATR required"):
            sizer.calculate(
                equity=Decimal("100000"),
                entry_price=Decimal("50000"),
                atr=Decimal("0"),
            )

    def test_higher_volatility_smaller_position(self):
        """Higher volatility results in smaller position."""
        sizer = VolatilityAdjustedSizer(
            risk_percent=Decimal("0.02"),
            atr_multiplier=Decimal("2.0"),
            max_position_pct=Decimal("1.0"),  # No cap for this test
        )

        low_vol = sizer.calculate(
            equity=Decimal("100000"),
            entry_price=Decimal("50000"),
            atr=Decimal("250"),  # Low vol
        )

        high_vol = sizer.calculate(
            equity=Decimal("100000"),
            entry_price=Decimal("50000"),
            atr=Decimal("1000"),  # High vol
        )

        assert low_vol.size > high_vol.size

    def test_name_property(self):
        """Sizer has correct name."""
        sizer = VolatilityAdjustedSizer()
        assert sizer.name == "volatility_adjusted"


class TestKellyCriterionSizer:
    """Tests for KellyCriterionSizer."""

    def test_basic_calculation(self):
        """Basic Kelly criterion calculation."""
        sizer = KellyCriterionSizer(
            kelly_fraction=Decimal("1.0"),  # Full Kelly for testing
            max_position_pct=Decimal("1.0"),  # No cap for testing
        )

        result = sizer.calculate(
            equity=Decimal("100000"),
            entry_price=Decimal("1"),  # Very low price to avoid notional cap
            stop_loss_price=Decimal("0.99"),  # $0.01 risk per unit
            win_rate=Decimal("0.60"),  # 60% win rate
            avg_win=Decimal("1000"),
            avg_loss=Decimal("500"),  # 2:1 reward/risk
        )

        # Kelly = 0.60 - (0.40 / 2.0) = 0.60 - 0.20 = 0.40 (40%)
        # Verify raw Kelly is correct
        assert result.metadata["raw_kelly"] == pytest.approx(0.40, rel=0.01)
        # Position size depends on risk per unit
        # 40% of 100k = $40000 risk amount
        assert float(result.risk_amount) == pytest.approx(40000.0, rel=0.01)

    def test_half_kelly(self):
        """Half-Kelly reduces position size."""
        sizer = KellyCriterionSizer(
            kelly_fraction=Decimal("0.5"),  # Half Kelly
            max_position_pct=Decimal("1.0"),
        )

        result = sizer.calculate(
            equity=Decimal("100000"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),
            win_rate=Decimal("0.60"),
            avg_win=Decimal("1000"),
            avg_loss=Decimal("500"),
        )

        # Full Kelly = 40%, Half Kelly = 20%
        assert result.metadata["fractional_kelly"] == pytest.approx(0.20, rel=0.01)

    def test_negative_kelly_no_trade(self):
        """Negative Kelly returns zero size (no edge)."""
        sizer = KellyCriterionSizer()

        result = sizer.calculate(
            equity=Decimal("100000"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),
            win_rate=Decimal("0.40"),  # 40% win rate
            avg_win=Decimal("500"),
            avg_loss=Decimal("1000"),  # Bad reward/risk
        )

        # Kelly = 0.40 - (0.60 / 0.5) = 0.40 - 1.20 = -0.80 (negative)
        assert result.size == Decimal("0")
        assert result.metadata["no_edge"] is True

    def test_max_position_cap(self):
        """Position capped at max percentage."""
        sizer = KellyCriterionSizer(
            kelly_fraction=Decimal("1.0"),
            max_position_pct=Decimal("0.10"),  # 10% cap
        )

        result = sizer.calculate(
            equity=Decimal("100000"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),
            win_rate=Decimal("0.60"),
            avg_win=Decimal("1000"),
            avg_loss=Decimal("500"),  # Would give 40% Kelly
        )

        # Capped at 10%
        assert result.metadata["capped"] is True

    def test_requires_all_parameters(self):
        """All Kelly parameters required."""
        sizer = KellyCriterionSizer()

        with pytest.raises(ValueError, match="Kelly requires"):
            sizer.calculate(
                equity=Decimal("100000"),
                entry_price=Decimal("50000"),
                stop_loss_price=Decimal("49000"),
                win_rate=Decimal("0.55"),
                avg_win=None,  # Missing
                avg_loss=Decimal("500"),
            )

    def test_requires_stop_loss(self):
        """Stop loss required for position sizing."""
        sizer = KellyCriterionSizer()

        with pytest.raises(ValueError, match="stop_loss_price required"):
            sizer.calculate(
                equity=Decimal("100000"),
                entry_price=Decimal("50000"),
                stop_loss_price=None,
                win_rate=Decimal("0.55"),
                avg_win=Decimal("1000"),
                avg_loss=Decimal("500"),
            )

    def test_win_rate_bounds(self):
        """Win rate must be between 0 and 1."""
        sizer = KellyCriterionSizer()

        with pytest.raises(ValueError, match="win_rate must be between"):
            sizer.calculate(
                equity=Decimal("100000"),
                entry_price=Decimal("50000"),
                stop_loss_price=Decimal("49000"),
                win_rate=Decimal("1.5"),  # Invalid
                avg_win=Decimal("1000"),
                avg_loss=Decimal("500"),
            )

    def test_name_property(self):
        """Sizer has correct name."""
        sizer = KellyCriterionSizer()
        assert sizer.name == "kelly_criterion"


class TestFixedQuantitySizer:
    """Tests for FixedQuantitySizer."""

    def test_basic_calculation(self):
        """Basic fixed quantity calculation."""
        sizer = FixedQuantitySizer(quantity=Decimal("0.5"))

        result = sizer.calculate(
            equity=Decimal("100000"),
            entry_price=Decimal("50000"),
        )

        assert result.size == Decimal("0.50000000")
        assert result.method == "fixed_quantity"

    def test_notional_cap(self):
        """Position capped by max notional value."""
        sizer = FixedQuantitySizer(
            quantity=Decimal("10.0"),
            max_notional=Decimal("100000"),
        )

        result = sizer.calculate(
            equity=Decimal("1000000"),
            entry_price=Decimal("50000"),  # 10 units = $500k > $100k max
        )

        # Capped at $100k / $50k = 2 units
        assert result.size == Decimal("2.00000000")

    def test_risk_amount_with_stop_loss(self):
        """Calculate risk amount when stop loss provided."""
        sizer = FixedQuantitySizer(quantity=Decimal("1.0"))

        result = sizer.calculate(
            equity=Decimal("100000"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),  # $1000 risk per unit
        )

        assert result.risk_amount == Decimal("1000")

    def test_name_property(self):
        """Sizer has correct name."""
        sizer = FixedQuantitySizer()
        assert sizer.name == "fixed_quantity"


class TestCreateSizer:
    """Tests for create_sizer factory function."""

    def test_create_fixed_percentage(self):
        """Create FixedPercentageSizer via factory."""
        sizer = create_sizer("fixed_percentage", risk_percent=0.03)

        assert isinstance(sizer, FixedPercentageSizer)
        assert sizer.risk_percent == Decimal("0.03")

    def test_create_volatility_adjusted(self):
        """Create VolatilityAdjustedSizer via factory."""
        sizer = create_sizer("volatility_adjusted", atr_multiplier=2.5)

        assert isinstance(sizer, VolatilityAdjustedSizer)
        assert sizer.atr_multiplier == Decimal("2.5")

    def test_create_kelly_criterion(self):
        """Create KellyCriterionSizer via factory."""
        sizer = create_sizer("kelly_criterion", kelly_fraction=0.25)

        assert isinstance(sizer, KellyCriterionSizer)
        assert sizer.kelly_fraction == Decimal("0.25")

    def test_create_fixed_quantity(self):
        """Create FixedQuantitySizer via factory."""
        sizer = create_sizer("fixed_quantity", quantity=2.5)

        assert isinstance(sizer, FixedQuantitySizer)
        assert sizer.quantity == Decimal("2.5")

    def test_unknown_method_error(self):
        """Unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown sizing method"):
            create_sizer("unknown_method")
