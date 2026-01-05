"""
Risk Limits Configuration for LIBRA.

Defines all configurable risk limits including:
- Position limits (per-symbol and global)
- Loss limits (daily, weekly, total drawdown)
- Order rate limits
- Circuit breaker thresholds

Supports YAML configuration for easy adjustment without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import msgspec


# =============================================================================
# Per-Symbol Limits
# =============================================================================


class SymbolLimits(msgspec.Struct, frozen=True):
    """
    Risk limits for a specific trading symbol.

    Attributes:
        max_position_size: Maximum position size in base currency
        max_notional_per_order: Maximum order value in quote currency
        max_order_rate: Maximum orders per second for this symbol
    """

    max_position_size: Decimal
    max_notional_per_order: Decimal
    max_order_rate: int = 10


# =============================================================================
# Global Risk Limits
# =============================================================================


@dataclass
class RiskLimits:
    """
    Complete risk limits configuration.

    Loaded from YAML for easy adjustment without code changes.
    All percentage values are expressed as decimals (0.02 = 2%).

    Examples:
        # Load from YAML
        limits = RiskLimits.from_yaml(Path("config/risk_limits.yaml"))

        # Create programmatically
        limits = RiskLimits(
            max_total_exposure=Decimal("100000"),
            max_single_position_pct=Decimal("0.20"),
            max_daily_loss_pct=Decimal("-0.03"),
            max_weekly_loss_pct=Decimal("-0.07"),
            max_total_drawdown_pct=Decimal("-0.15"),
        )
    """

    # === Global Position Limits ===
    max_total_exposure: Decimal = Decimal("100000")  # Max total portfolio value
    max_single_position_pct: Decimal = Decimal("0.20")  # Max 20% in single position

    # === Loss Limits (negative values) ===
    max_daily_loss_pct: Decimal = Decimal("-0.03")  # -3% daily limit
    max_weekly_loss_pct: Decimal = Decimal("-0.07")  # -7% weekly limit
    max_total_drawdown_pct: Decimal = Decimal("-0.15")  # -15% max drawdown

    # === Circuit Breaker Thresholds ===
    circuit_breaker_drawdown_pct: Decimal = Decimal("-0.05")  # -5% triggers
    circuit_breaker_cooldown_seconds: int = 300  # 5 minute cooldown
    circuit_breaker_max_consecutive_losses: int = 10  # 10 losses triggers

    # === Rate Limits (Token Bucket) ===
    max_orders_per_second: int = 10
    max_orders_per_minute: int = 100

    # === Per-Symbol Limits ===
    symbol_limits: dict[str, SymbolLimits] = field(default_factory=dict)

    # === Default Symbol Limits (used if symbol not in symbol_limits) ===
    default_max_position_size: Decimal = Decimal("1.0")
    default_max_notional_per_order: Decimal = Decimal("10000")
    default_max_order_rate: int = 10

    def __post_init__(self) -> None:
        """Validate configuration on creation."""
        self.validate()

    @classmethod
    def from_yaml(cls, path: Path) -> RiskLimits:
        """
        Load and validate limits from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Validated RiskLimits instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If configuration is invalid
        """
        # Import yaml here to keep it optional
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RiskLimits:
        """
        Create RiskLimits from dictionary.

        Handles type conversion for Decimal values and nested SymbolLimits.
        """
        # Convert top-level Decimal fields
        decimal_fields = {
            "max_total_exposure",
            "max_single_position_pct",
            "max_daily_loss_pct",
            "max_weekly_loss_pct",
            "max_total_drawdown_pct",
            "circuit_breaker_drawdown_pct",
            "default_max_position_size",
            "default_max_notional_per_order",
        }

        converted = {}
        for key, value in data.items():
            if key in decimal_fields and value is not None:
                converted[key] = Decimal(str(value))
            elif key == "symbol_limits" and value is not None:
                # Convert nested symbol limits
                converted[key] = {
                    symbol: SymbolLimits(
                        max_position_size=Decimal(str(limits.get("max_position_size", 1.0))),
                        max_notional_per_order=Decimal(
                            str(limits.get("max_notional_per_order", 10000))
                        ),
                        max_order_rate=limits.get("max_order_rate", 10),
                    )
                    for symbol, limits in value.items()
                }
            else:
                converted[key] = value

        return cls(**converted)

    def validate(self) -> None:
        """
        Validate all configuration values.

        Raises:
            ValueError: If any value is invalid
        """
        # Loss limits must be negative
        if self.max_daily_loss_pct >= Decimal("0"):
            raise ValueError("max_daily_loss_pct must be negative (e.g., -0.03 for -3%)")

        if self.max_weekly_loss_pct >= Decimal("0"):
            raise ValueError("max_weekly_loss_pct must be negative")

        if self.max_total_drawdown_pct >= Decimal("0"):
            raise ValueError("max_total_drawdown_pct must be negative")

        if self.circuit_breaker_drawdown_pct >= Decimal("0"):
            raise ValueError("circuit_breaker_drawdown_pct must be negative")

        # Position percentage must be between 0 and 1
        if not (Decimal("0") < self.max_single_position_pct <= Decimal("1")):
            raise ValueError("max_single_position_pct must be between 0 and 1")

        # Rate limits must be positive
        if self.max_orders_per_second <= 0:
            raise ValueError("max_orders_per_second must be positive")

        if self.max_orders_per_minute <= 0:
            raise ValueError("max_orders_per_minute must be positive")

        # Cooldown must be positive
        if self.circuit_breaker_cooldown_seconds <= 0:
            raise ValueError("circuit_breaker_cooldown_seconds must be positive")

        # Exposure must be positive
        if self.max_total_exposure <= Decimal("0"):
            raise ValueError("max_total_exposure must be positive")

        # Validate hierarchy: daily < weekly < total
        if self.max_daily_loss_pct < self.max_weekly_loss_pct:
            raise ValueError("max_daily_loss_pct should be >= max_weekly_loss_pct")

        if self.max_weekly_loss_pct < self.max_total_drawdown_pct:
            raise ValueError("max_weekly_loss_pct should be >= max_total_drawdown_pct")

    def get_symbol_limits(self, symbol: str) -> SymbolLimits:
        """
        Get limits for a specific symbol.

        Returns symbol-specific limits if configured, otherwise returns
        default limits.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")

        Returns:
            SymbolLimits for the symbol
        """
        if symbol in self.symbol_limits:
            return self.symbol_limits[symbol]

        # Return default limits
        return SymbolLimits(
            max_position_size=self.default_max_position_size,
            max_notional_per_order=self.default_max_notional_per_order,
            max_order_rate=self.default_max_order_rate,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_total_exposure": str(self.max_total_exposure),
            "max_single_position_pct": str(self.max_single_position_pct),
            "max_daily_loss_pct": str(self.max_daily_loss_pct),
            "max_weekly_loss_pct": str(self.max_weekly_loss_pct),
            "max_total_drawdown_pct": str(self.max_total_drawdown_pct),
            "circuit_breaker_drawdown_pct": str(self.circuit_breaker_drawdown_pct),
            "circuit_breaker_cooldown_seconds": self.circuit_breaker_cooldown_seconds,
            "circuit_breaker_max_consecutive_losses": self.circuit_breaker_max_consecutive_losses,
            "max_orders_per_second": self.max_orders_per_second,
            "max_orders_per_minute": self.max_orders_per_minute,
            "symbol_limits": {
                symbol: {
                    "max_position_size": str(limits.max_position_size),
                    "max_notional_per_order": str(limits.max_notional_per_order),
                    "max_order_rate": limits.max_order_rate,
                }
                for symbol, limits in self.symbol_limits.items()
            },
            "default_max_position_size": str(self.default_max_position_size),
            "default_max_notional_per_order": str(self.default_max_notional_per_order),
            "default_max_order_rate": self.default_max_order_rate,
        }
