"""
Unit tests for Greeks Models.

Tests Greeks and GreeksSnapshot.

Issue #63: Options Data Models
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from libra.core.options import (
    Greeks,
    GreeksSnapshot,
    greeks_from_dict,
    zero_greeks,
)


# =============================================================================
# Greeks Tests
# =============================================================================


class TestGreeks:
    """Tests for Greeks struct."""

    @pytest.fixture
    def sample_greeks(self) -> Greeks:
        """Create sample Greeks."""
        return Greeks(
            delta=Decimal("0.65"),
            gamma=Decimal("0.02"),
            theta=Decimal("-0.05"),
            vega=Decimal("0.15"),
            rho=Decimal("0.03"),
            iv=Decimal("0.35"),
        )

    def test_creation(self, sample_greeks: Greeks) -> None:
        """Greeks fields are set correctly."""
        assert sample_greeks.delta == Decimal("0.65")
        assert sample_greeks.gamma == Decimal("0.02")
        assert sample_greeks.theta == Decimal("-0.05")
        assert sample_greeks.vega == Decimal("0.15")
        assert sample_greeks.rho == Decimal("0.03")
        assert sample_greeks.iv == Decimal("0.35")
        assert sample_greeks.timestamp_ns is None

    def test_delta_dollars(self, sample_greeks: Greeks) -> None:
        """delta_dollars scales by 100."""
        # 0.65 * 100 = 65
        assert sample_greeks.delta_dollars == Decimal("65")

    def test_iv_percent(self, sample_greeks: Greeks) -> None:
        """iv_percent converts to percentage."""
        # 0.35 * 100 = 35
        assert sample_greeks.iv_percent == Decimal("35")

    def test_scale_long_position(self, sample_greeks: Greeks) -> None:
        """scale() multiplies Greeks by position size."""
        # 10 contracts * 100 multiplier = 1000
        scaled = sample_greeks.scale(10)
        assert scaled.delta == Decimal("650.00")  # 0.65 * 10 * 100
        assert scaled.gamma == Decimal("20.00")  # 0.02 * 10 * 100
        assert scaled.theta == Decimal("-50.00")  # -0.05 * 10 * 100
        assert scaled.vega == Decimal("150.00")  # 0.15 * 10 * 100
        assert scaled.rho == Decimal("30.00")  # 0.03 * 10 * 100
        # IV unchanged
        assert scaled.iv == Decimal("0.35")

    def test_scale_short_position(self, sample_greeks: Greeks) -> None:
        """scale() handles negative quantity (short position)."""
        scaled = sample_greeks.scale(-10)
        assert scaled.delta == Decimal("-650.00")
        assert scaled.theta == Decimal("50.00")  # Positive theta for short

    def test_scale_custom_multiplier(self, sample_greeks: Greeks) -> None:
        """scale() accepts custom multiplier."""
        scaled = sample_greeks.scale(1, multiplier=50)  # Mini options
        assert scaled.delta == Decimal("32.5")  # 0.65 * 1 * 50

    def test_addition(self, sample_greeks: Greeks) -> None:
        """Greeks can be added together."""
        other = Greeks(
            delta=Decimal("0.35"),
            gamma=Decimal("0.01"),
            theta=Decimal("-0.03"),
            vega=Decimal("0.10"),
            rho=Decimal("0.02"),
            iv=Decimal("0.30"),
        )
        combined = sample_greeks + other
        assert combined.delta == Decimal("1.00")  # 0.65 + 0.35
        assert combined.gamma == Decimal("0.03")  # 0.02 + 0.01
        assert combined.theta == Decimal("-0.08")  # -0.05 + -0.03
        assert combined.vega == Decimal("0.25")  # 0.15 + 0.10
        # IV is averaged
        assert combined.iv == Decimal("0.325")  # (0.35 + 0.30) / 2


class TestZeroGreeks:
    """Tests for zero_greeks factory."""

    def test_all_zeros(self) -> None:
        """zero_greeks returns all zeros."""
        greeks = zero_greeks()
        assert greeks.delta == Decimal("0")
        assert greeks.gamma == Decimal("0")
        assert greeks.theta == Decimal("0")
        assert greeks.vega == Decimal("0")
        assert greeks.rho == Decimal("0")
        assert greeks.iv == Decimal("0")


class TestGreeksFromDict:
    """Tests for greeks_from_dict factory."""

    def test_basic_dict(self) -> None:
        """greeks_from_dict parses dictionary."""
        data = {
            "delta": 0.65,
            "gamma": 0.02,
            "theta": -0.05,
            "vega": 0.15,
            "rho": 0.03,
            "iv": 0.35,
        }
        greeks = greeks_from_dict(data)
        assert greeks.delta == Decimal("0.65")
        assert greeks.iv == Decimal("0.35")

    def test_implied_volatility_key(self) -> None:
        """greeks_from_dict accepts implied_volatility key."""
        data = {
            "delta": 0.5,
            "gamma": 0.01,
            "theta": -0.02,
            "vega": 0.1,
            "rho": 0.01,
            "implied_volatility": 0.40,
        }
        greeks = greeks_from_dict(data)
        assert greeks.iv == Decimal("0.40")

    def test_missing_values_default_zero(self) -> None:
        """greeks_from_dict defaults missing values to zero."""
        data = {"delta": 0.5}
        greeks = greeks_from_dict(data)
        assert greeks.delta == Decimal("0.5")
        assert greeks.gamma == Decimal("0")
        assert greeks.iv == Decimal("0")


# =============================================================================
# GreeksSnapshot Tests
# =============================================================================


class TestGreeksSnapshot:
    """Tests for GreeksSnapshot struct."""

    @pytest.fixture
    def sample_greeks(self) -> Greeks:
        """Create sample Greeks."""
        return Greeks(
            delta=Decimal("0.65"),
            gamma=Decimal("0.02"),
            theta=Decimal("-0.05"),
            vega=Decimal("0.15"),
            rho=Decimal("0.03"),
            iv=Decimal("0.35"),
        )

    @pytest.fixture
    def sample_snapshot(self, sample_greeks: Greeks) -> GreeksSnapshot:
        """Create sample snapshot."""
        return GreeksSnapshot(
            greeks=sample_greeks,
            underlying_price=Decimal("155.00"),
            option_price=Decimal("7.50"),
            bid=Decimal("7.40"),
            ask=Decimal("7.60"),
            volume=1500,
            open_interest=25000,
        )

    def test_creation(self, sample_snapshot: GreeksSnapshot) -> None:
        """Snapshot fields are set correctly."""
        assert sample_snapshot.underlying_price == Decimal("155.00")
        assert sample_snapshot.option_price == Decimal("7.50")
        assert sample_snapshot.bid == Decimal("7.40")
        assert sample_snapshot.ask == Decimal("7.60")
        assert sample_snapshot.volume == 1500
        assert sample_snapshot.open_interest == 25000

    def test_bid_ask_spread(self, sample_snapshot: GreeksSnapshot) -> None:
        """bid_ask_spread calculates correctly."""
        # 7.60 - 7.40 = 0.20
        assert sample_snapshot.bid_ask_spread == Decimal("0.20")

    def test_mid_price(self, sample_snapshot: GreeksSnapshot) -> None:
        """mid_price calculates correctly."""
        # (7.40 + 7.60) / 2 = 7.50
        assert sample_snapshot.mid_price == Decimal("7.50")

    def test_spread_pct(self, sample_snapshot: GreeksSnapshot) -> None:
        """spread_pct calculates correctly."""
        # 0.20 / 7.50 * 100 = 2.666...%
        spread_pct = sample_snapshot.spread_pct
        assert spread_pct > Decimal("2.6")
        assert spread_pct < Decimal("2.7")

    def test_spread_pct_zero_mid(self, sample_greeks: Greeks) -> None:
        """spread_pct handles zero mid price."""
        snapshot = GreeksSnapshot(
            greeks=sample_greeks,
            underlying_price=Decimal("0"),
            option_price=Decimal("0"),
            bid=Decimal("0"),
            ask=Decimal("0"),
            volume=0,
            open_interest=0,
        )
        assert snapshot.spread_pct == Decimal("0")

    def test_is_liquid_true(self, sample_snapshot: GreeksSnapshot) -> None:
        """is_liquid returns True for liquid options."""
        # Spread ~2.7%, OI 25000 -> liquid
        assert sample_snapshot.is_liquid is True

    def test_is_liquid_wide_spread(self, sample_greeks: Greeks) -> None:
        """is_liquid returns False for wide spreads."""
        snapshot = GreeksSnapshot(
            greeks=sample_greeks,
            underlying_price=Decimal("100"),
            option_price=Decimal("1.00"),
            bid=Decimal("0.80"),
            ask=Decimal("1.20"),  # 40% spread
            volume=100,
            open_interest=1000,
        )
        assert snapshot.is_liquid is False

    def test_is_liquid_low_oi(self, sample_greeks: Greeks) -> None:
        """is_liquid returns False for low open interest."""
        snapshot = GreeksSnapshot(
            greeks=sample_greeks,
            underlying_price=Decimal("100"),
            option_price=Decimal("5.00"),
            bid=Decimal("4.99"),
            ask=Decimal("5.01"),  # Tight spread
            volume=10,
            open_interest=50,  # Low OI
        )
        assert snapshot.is_liquid is False
