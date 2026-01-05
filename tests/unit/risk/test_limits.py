"""Tests for RiskLimits configuration."""

from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from libra.risk.limits import RiskLimits, SymbolLimits


class TestSymbolLimits:
    """Tests for SymbolLimits."""

    def test_create_symbol_limits(self):
        """Create symbol limits with all fields."""
        limits = SymbolLimits(
            max_position_size=Decimal("1.0"),
            max_notional_per_order=Decimal("50000"),
            max_order_rate=5,
        )

        assert limits.max_position_size == Decimal("1.0")
        assert limits.max_notional_per_order == Decimal("50000")
        assert limits.max_order_rate == 5

    def test_symbol_limits_frozen(self):
        """Symbol limits should be immutable."""
        limits = SymbolLimits(
            max_position_size=Decimal("1.0"),
            max_notional_per_order=Decimal("50000"),
        )

        with pytest.raises(AttributeError):
            limits.max_position_size = Decimal("2.0")


class TestRiskLimits:
    """Tests for RiskLimits."""

    def test_create_with_defaults(self):
        """Create RiskLimits with default values."""
        limits = RiskLimits()

        assert limits.max_total_exposure == Decimal("100000")
        assert limits.max_single_position_pct == Decimal("0.20")
        assert limits.max_daily_loss_pct == Decimal("-0.03")
        assert limits.max_orders_per_second == 10

    def test_create_with_custom_values(self):
        """Create RiskLimits with custom values."""
        limits = RiskLimits(
            max_total_exposure=Decimal("500000"),
            max_single_position_pct=Decimal("0.10"),
            max_daily_loss_pct=Decimal("-0.05"),
            max_weekly_loss_pct=Decimal("-0.10"),
            max_total_drawdown_pct=Decimal("-0.20"),
        )

        assert limits.max_total_exposure == Decimal("500000")
        assert limits.max_single_position_pct == Decimal("0.10")
        assert limits.max_daily_loss_pct == Decimal("-0.05")

    def test_validation_daily_loss_must_be_negative(self):
        """Daily loss limit must be negative."""
        with pytest.raises(ValueError, match="max_daily_loss_pct must be negative"):
            RiskLimits(max_daily_loss_pct=Decimal("0.03"))

    def test_validation_weekly_loss_must_be_negative(self):
        """Weekly loss limit must be negative."""
        with pytest.raises(ValueError, match="max_weekly_loss_pct must be negative"):
            RiskLimits(max_weekly_loss_pct=Decimal("0.05"))

    def test_validation_drawdown_must_be_negative(self):
        """Max drawdown must be negative."""
        with pytest.raises(ValueError, match="max_total_drawdown_pct must be negative"):
            RiskLimits(max_total_drawdown_pct=Decimal("0.10"))

    def test_validation_position_pct_bounds(self):
        """Single position percentage must be between 0 and 1."""
        with pytest.raises(ValueError, match="max_single_position_pct must be between 0 and 1"):
            RiskLimits(max_single_position_pct=Decimal("1.5"))

        with pytest.raises(ValueError, match="max_single_position_pct must be between 0 and 1"):
            RiskLimits(max_single_position_pct=Decimal("-0.1"))

    def test_validation_rate_limits_positive(self):
        """Rate limits must be positive."""
        with pytest.raises(ValueError, match="max_orders_per_second must be positive"):
            RiskLimits(max_orders_per_second=0)

    def test_validation_loss_hierarchy(self):
        """Daily loss should be >= weekly loss >= total drawdown."""
        # Daily tighter than weekly - invalid
        with pytest.raises(ValueError, match="max_daily_loss_pct should be >= max_weekly_loss_pct"):
            RiskLimits(
                max_daily_loss_pct=Decimal("-0.10"),  # More than weekly
                max_weekly_loss_pct=Decimal("-0.05"),
            )

    def test_get_symbol_limits_configured(self):
        """Get limits for a configured symbol."""
        btc_limits = SymbolLimits(
            max_position_size=Decimal("2.0"),
            max_notional_per_order=Decimal("100000"),
            max_order_rate=10,
        )

        limits = RiskLimits(symbol_limits={"BTC/USDT": btc_limits})

        result = limits.get_symbol_limits("BTC/USDT")
        assert result.max_position_size == Decimal("2.0")
        assert result.max_notional_per_order == Decimal("100000")

    def test_get_symbol_limits_default(self):
        """Get default limits for unconfigured symbol."""
        limits = RiskLimits(
            default_max_position_size=Decimal("5.0"),
            default_max_notional_per_order=Decimal("25000"),
        )

        result = limits.get_symbol_limits("UNKNOWN/USDT")
        assert result.max_position_size == Decimal("5.0")
        assert result.max_notional_per_order == Decimal("25000")

    def test_from_dict(self):
        """Create RiskLimits from dictionary."""
        data = {
            "max_total_exposure": 200000,
            "max_single_position_pct": 0.15,
            "max_daily_loss_pct": -0.04,
            "max_weekly_loss_pct": -0.08,
            "max_total_drawdown_pct": -0.20,
            "max_orders_per_second": 20,
            "symbol_limits": {
                "BTC/USDT": {
                    "max_position_size": 1.5,
                    "max_notional_per_order": 75000,
                    "max_order_rate": 8,
                }
            },
        }

        limits = RiskLimits.from_dict(data)

        assert limits.max_total_exposure == Decimal("200000")
        assert limits.max_single_position_pct == Decimal("0.15")
        assert limits.max_orders_per_second == 20

        btc = limits.get_symbol_limits("BTC/USDT")
        assert btc.max_position_size == Decimal("1.5")

    def test_from_yaml(self, tmp_path):
        """Load RiskLimits from YAML file."""
        yaml_content = """
max_total_exposure: 150000
max_single_position_pct: 0.25
max_daily_loss_pct: -0.03
max_weekly_loss_pct: -0.07
max_total_drawdown_pct: -0.15
max_orders_per_second: 15
symbol_limits:
  ETH/USDT:
    max_position_size: 10.0
    max_notional_per_order: 30000
"""
        yaml_file = tmp_path / "risk_limits.yaml"
        yaml_file.write_text(yaml_content)

        limits = RiskLimits.from_yaml(yaml_file)

        assert limits.max_total_exposure == Decimal("150000")
        assert limits.max_orders_per_second == 15

        eth = limits.get_symbol_limits("ETH/USDT")
        assert eth.max_position_size == Decimal("10.0")

    def test_to_dict(self):
        """Convert RiskLimits to dictionary."""
        limits = RiskLimits(
            max_total_exposure=Decimal("100000"),
            symbol_limits={
                "BTC/USDT": SymbolLimits(
                    max_position_size=Decimal("1.0"),
                    max_notional_per_order=Decimal("50000"),
                )
            },
        )

        data = limits.to_dict()

        assert data["max_total_exposure"] == "100000"
        assert "BTC/USDT" in data["symbol_limits"]
        assert data["symbol_limits"]["BTC/USDT"]["max_position_size"] == "1.0"
