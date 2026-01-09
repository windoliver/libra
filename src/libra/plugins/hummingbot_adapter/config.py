"""
Configuration schema for Hummingbot Adapter Plugin (Issue #12).

Defines configuration for market making strategies:
- Avellaneda-Stoikov parameters
- Pure market making settings
- Cross-exchange market making (XEMM) settings
- Inventory management parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


class StrategyType(str, Enum):
    """Market making strategy types."""

    AVELLANEDA_STOIKOV = "avellaneda_stoikov"
    PURE_MARKET_MAKING = "pure_market_making"
    CROSS_EXCHANGE_MM = "cross_exchange_mm"


class InventorySkewMode(str, Enum):
    """How to handle inventory imbalance."""

    NONE = "none"  # No inventory adjustment
    LINEAR = "linear"  # Linear price skew
    EXPONENTIAL = "exponential"  # Exponential skew for aggressive rebalancing


class TimeframeMode(str, Enum):
    """Trading session timeframe mode."""

    INFINITE = "infinite"  # 24/7 trading
    DAILY = "daily"  # Daily sessions with reset
    CUSTOM = "custom"  # Custom start/end times


@dataclass
class AvellanedaStoikovConfig:
    """
    Avellaneda-Stoikov strategy parameters.

    Based on "High-frequency Trading in a Limit Order Book" (2008).
    Adapted for crypto markets with 24/7 trading support.
    """

    # Risk aversion parameter (gamma)
    # Higher = more risk-averse = wider spreads
    # Range: 0.0 to 1.0
    risk_aversion: float = 0.5

    # Order book liquidity parameter (kappa)
    # Higher = more liquid market = tighter spreads
    # Typical range: 0.1 to 10.0
    order_book_depth: float = 1.5

    # Order amount shape factor (eta)
    # Affects order size based on inventory distance
    # Range: 0.0 to 2.0
    order_amount_shape: float = 1.0

    # Volatility calculation window (seconds)
    volatility_window: int = 300  # 5 minutes

    # Minimum time between quote updates (seconds)
    min_quote_refresh: float = 1.0

    # Timeframe mode for the strategy
    timeframe_mode: TimeframeMode = TimeframeMode.INFINITE

    # For DAILY/CUSTOM modes: session duration in hours
    session_duration_hours: float = 24.0

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.risk_aversion <= 1.0:
            raise ValueError("risk_aversion must be between 0.0 and 1.0")
        if self.order_book_depth <= 0:
            raise ValueError("order_book_depth must be positive")
        if self.order_amount_shape < 0:
            raise ValueError("order_amount_shape must be non-negative")


@dataclass
class InventoryConfig:
    """Inventory management configuration."""

    # Target inventory ratio (0.5 = balanced between base and quote)
    target_ratio: float = 0.5

    # Acceptable deviation from target before aggressive rebalancing
    # E.g., 0.2 means rebalance if ratio is outside [0.3, 0.7]
    tolerance: float = 0.2

    # Maximum inventory as percentage of total portfolio
    max_inventory_pct: float = 0.8

    # Skew mode for price adjustment
    skew_mode: InventorySkewMode = InventorySkewMode.LINEAR

    # Skew intensity (higher = more aggressive price adjustment)
    skew_intensity: float = 2.0

    # Enable hanging orders (orders that persist until filled)
    hanging_orders: bool = False

    # Maximum age for hanging orders (seconds)
    hanging_order_max_age: int = 3600

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.target_ratio <= 1.0:
            raise ValueError("target_ratio must be between 0.0 and 1.0")
        if not 0.0 <= self.tolerance <= 0.5:
            raise ValueError("tolerance must be between 0.0 and 0.5")
        if not 0.0 < self.max_inventory_pct <= 1.0:
            raise ValueError("max_inventory_pct must be between 0.0 and 1.0")


@dataclass
class XEMMConfig:
    """Cross-Exchange Market Making configuration."""

    # Maker exchange (where we place limit orders)
    maker_exchange: str = ""

    # Taker exchange (where we hedge with market orders)
    taker_exchange: str = ""

    # Minimum profitability threshold (as decimal, e.g., 0.001 = 0.1%)
    min_profitability: Decimal = Decimal("0.001")

    # Whether to actively hedge on taker exchange
    active_hedging: bool = True

    # Maximum position to hold before forced hedging
    max_unhedged_position: Decimal = Decimal("0")

    # Taker fee (for profitability calculation)
    taker_fee: Decimal = Decimal("0.001")

    # Maker fee (rebate if negative)
    maker_fee: Decimal = Decimal("0.0005")

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.maker_exchange:
            raise ValueError("maker_exchange is required for XEMM")
        if not self.taker_exchange:
            raise ValueError("taker_exchange is required for XEMM")
        if self.min_profitability < 0:
            raise ValueError("min_profitability must be non-negative")


@dataclass
class HummingbotAdapterConfig:
    """
    Main configuration for Hummingbot Adapter Plugin.

    Supports multiple strategy types with shared and strategy-specific settings.
    """

    # Strategy selection
    strategy_type: StrategyType = StrategyType.AVELLANEDA_STOIKOV

    # Trading pair symbol (e.g., "BTC/USDT")
    symbol: str = ""

    # Order amount in base currency
    order_amount: Decimal = Decimal("0.01")

    # Minimum spread (as decimal, e.g., 0.001 = 0.1%)
    min_spread: Decimal = Decimal("0.001")

    # Maximum spread (as decimal)
    max_spread: Decimal = Decimal("0.05")

    # Number of order levels on each side
    order_levels: int = 1

    # Spread between order levels (as decimal)
    level_spread: Decimal = Decimal("0.001")

    # Order refresh time (seconds)
    order_refresh_time: float = 15.0

    # Cancel orders on stop
    cancel_on_stop: bool = True

    # Strategy-specific configurations
    avellaneda: AvellanedaStoikovConfig = field(default_factory=AvellanedaStoikovConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)
    xemm: XEMMConfig = field(default_factory=XEMMConfig)

    # Extra configuration for extensibility
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate all configuration parameters."""
        if not self.symbol:
            raise ValueError("symbol is required")
        if self.order_amount <= 0:
            raise ValueError("order_amount must be positive")
        if self.min_spread < 0:
            raise ValueError("min_spread must be non-negative")
        if self.max_spread <= self.min_spread:
            raise ValueError("max_spread must be greater than min_spread")
        if self.order_levels < 1:
            raise ValueError("order_levels must be at least 1")

        # Validate nested configs
        self.avellaneda.validate()
        self.inventory.validate()

        if self.strategy_type == StrategyType.CROSS_EXCHANGE_MM:
            self.xemm.validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HummingbotAdapterConfig:
        """Create configuration from dictionary."""
        # Extract nested configs
        avellaneda_data = data.pop("avellaneda", {})
        inventory_data = data.pop("inventory", {})
        xemm_data = data.pop("xemm", {})

        # Handle enum conversion
        if "strategy_type" in data and isinstance(data["strategy_type"], str):
            data["strategy_type"] = StrategyType(data["strategy_type"])

        # Handle Decimal conversion
        for field_name in ["order_amount", "min_spread", "max_spread", "level_spread"]:
            if field_name in data and not isinstance(data[field_name], Decimal):
                data[field_name] = Decimal(str(data[field_name]))

        # Create nested configs
        if "timeframe_mode" in avellaneda_data and isinstance(avellaneda_data["timeframe_mode"], str):
            avellaneda_data["timeframe_mode"] = TimeframeMode(avellaneda_data["timeframe_mode"])

        if "skew_mode" in inventory_data and isinstance(inventory_data["skew_mode"], str):
            inventory_data["skew_mode"] = InventorySkewMode(inventory_data["skew_mode"])

        for field_name in ["min_profitability", "max_unhedged_position", "taker_fee", "maker_fee"]:
            if field_name in xemm_data and not isinstance(xemm_data[field_name], Decimal):
                xemm_data[field_name] = Decimal(str(xemm_data[field_name]))

        config = cls(
            avellaneda=AvellanedaStoikovConfig(**avellaneda_data) if avellaneda_data else AvellanedaStoikovConfig(),
            inventory=InventoryConfig(**inventory_data) if inventory_data else InventoryConfig(),
            xemm=XEMMConfig(**xemm_data) if xemm_data else XEMMConfig(),
            **data,
        )

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "strategy_type": self.strategy_type.value,
            "symbol": self.symbol,
            "order_amount": str(self.order_amount),
            "min_spread": str(self.min_spread),
            "max_spread": str(self.max_spread),
            "order_levels": self.order_levels,
            "level_spread": str(self.level_spread),
            "order_refresh_time": self.order_refresh_time,
            "cancel_on_stop": self.cancel_on_stop,
            "avellaneda": {
                "risk_aversion": self.avellaneda.risk_aversion,
                "order_book_depth": self.avellaneda.order_book_depth,
                "order_amount_shape": self.avellaneda.order_amount_shape,
                "volatility_window": self.avellaneda.volatility_window,
                "min_quote_refresh": self.avellaneda.min_quote_refresh,
                "timeframe_mode": self.avellaneda.timeframe_mode.value,
                "session_duration_hours": self.avellaneda.session_duration_hours,
            },
            "inventory": {
                "target_ratio": self.inventory.target_ratio,
                "tolerance": self.inventory.tolerance,
                "max_inventory_pct": self.inventory.max_inventory_pct,
                "skew_mode": self.inventory.skew_mode.value,
                "skew_intensity": self.inventory.skew_intensity,
                "hanging_orders": self.inventory.hanging_orders,
                "hanging_order_max_age": self.inventory.hanging_order_max_age,
            },
            "xemm": {
                "maker_exchange": self.xemm.maker_exchange,
                "taker_exchange": self.xemm.taker_exchange,
                "min_profitability": str(self.xemm.min_profitability),
                "active_hedging": self.xemm.active_hedging,
                "max_unhedged_position": str(self.xemm.max_unhedged_position),
                "taker_fee": str(self.xemm.taker_fee),
                "maker_fee": str(self.xemm.maker_fee),
            },
            "extra": self.extra,
        }
