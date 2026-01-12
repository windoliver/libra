"""
Margin Monitor.

Monitors margin utilization and liquidation risk for leveraged trading:
- Per-position margin tracking
- Portfolio-level margin aggregation
- Liquidation price calculation
- Margin call early warning system
- Cross-margin vs isolated margin support

References:
- Exchange margin specifications (Binance, FTX, etc.)
- CFTC margin requirements
- Portfolio margining best practices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MarginMode(Enum):
    """Margin mode for position."""

    ISOLATED = "isolated"  # Margin dedicated to single position
    CROSS = "cross"  # Margin shared across positions


class PositionSide(Enum):
    """Position side."""

    LONG = "long"
    SHORT = "short"


class MarginAlertLevel(Enum):
    """Margin alert severity levels."""

    HEALTHY = "healthy"  # > 50% buffer
    CAUTION = "caution"  # 25-50% buffer
    WARNING = "warning"  # 10-25% buffer
    CRITICAL = "critical"  # < 10% buffer
    LIQUIDATION = "liquidation"  # At liquidation


@dataclass
class MarginPosition:
    """Margin data for a single position."""

    symbol: str
    side: PositionSide
    size: Decimal  # Position size in base currency
    entry_price: Decimal
    mark_price: Decimal
    leverage: Decimal
    margin_mode: MarginMode

    # Margin requirements
    initial_margin: Decimal  # Required to open position
    maintenance_margin: Decimal  # Required to keep position
    margin_balance: Decimal  # Current margin allocated

    # Calculated fields
    notional_value: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    margin_ratio: Decimal = Decimal("0")  # maintenance_margin / margin_balance
    liquidation_price: Decimal = Decimal("0")
    distance_to_liquidation_pct: float = 0.0

    def __post_init__(self) -> None:
        """Calculate derived fields."""
        self.notional_value = abs(self.size * self.mark_price)

        # Calculate unrealized PnL
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = self.size * (self.mark_price - self.entry_price)
        else:
            self.unrealized_pnl = self.size * (self.entry_price - self.mark_price)

        # Calculate margin ratio
        if self.margin_balance > 0:
            self.margin_ratio = self.maintenance_margin / self.margin_balance

        # Calculate liquidation price
        self._calculate_liquidation_price()

    def _calculate_liquidation_price(self) -> None:
        """
        Calculate liquidation price.

        For longs: price drops until margin_balance + unrealized_pnl = maintenance_margin
        For shorts: price rises until margin_balance + unrealized_pnl = maintenance_margin
        """
        if self.size == 0:
            return

        margin_available = self.margin_balance - self.maintenance_margin

        if self.side == PositionSide.LONG:
            # Liquidation when: margin_balance + size * (liq_price - entry_price) = maintenance_margin
            # liq_price = entry_price - margin_available / size
            price_buffer = margin_available / abs(self.size)
            self.liquidation_price = self.entry_price - price_buffer
            self.liquidation_price = max(Decimal("0"), self.liquidation_price)
        else:
            # For shorts, liquidation when price rises
            price_buffer = margin_available / abs(self.size)
            self.liquidation_price = self.entry_price + price_buffer

        # Distance to liquidation
        if self.liquidation_price > 0 and self.mark_price > 0:
            if self.side == PositionSide.LONG:
                distance = (self.mark_price - self.liquidation_price) / self.mark_price
            else:
                distance = (self.liquidation_price - self.mark_price) / self.mark_price
            self.distance_to_liquidation_pct = float(distance) * 100


@dataclass
class PortfolioMargin:
    """Portfolio-level margin aggregation."""

    total_initial_margin: Decimal
    total_maintenance_margin: Decimal
    total_margin_balance: Decimal
    total_notional: Decimal
    total_unrealized_pnl: Decimal

    # Utilization metrics
    margin_utilization: float  # maintenance / balance
    available_margin: Decimal
    margin_level: float  # balance / maintenance (inverse of ratio)

    # Risk metrics
    effective_leverage: float  # notional / equity
    largest_position_pct: float
    position_count: int

    # Alert
    alert_level: MarginAlertLevel

    def __post_init__(self) -> None:
        """Calculate alert level."""
        if self.margin_level <= 1.0:
            self.alert_level = MarginAlertLevel.LIQUIDATION
        elif self.margin_level < 1.1:
            self.alert_level = MarginAlertLevel.CRITICAL
        elif self.margin_level < 1.25:
            self.alert_level = MarginAlertLevel.WARNING
        elif self.margin_level < 1.5:
            self.alert_level = MarginAlertLevel.CAUTION
        else:
            self.alert_level = MarginAlertLevel.HEALTHY


@dataclass
class MarginAlert:
    """Alert for margin-related events."""

    alert_type: str  # "low_margin", "approaching_liquidation", "liquidation"
    symbol: str | None  # None for portfolio-level
    description: str
    current_value: float
    threshold: float
    severity: MarginAlertLevel
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MarginConfig:
    """Configuration for margin monitoring."""

    # Alert thresholds (margin_level = balance / maintenance)
    caution_threshold: float = 1.5  # 50% buffer
    warning_threshold: float = 1.25  # 25% buffer
    critical_threshold: float = 1.1  # 10% buffer

    # Position limits
    max_leverage: Decimal = Decimal("10")
    max_position_pct: float = 25.0  # Max % of portfolio in single position
    max_notional_per_symbol: Decimal | None = None

    # Alert cooldown (seconds)
    alert_cooldown: int = 300  # 5 minutes


class MarginMonitor:
    """
    Margin Monitor for leveraged trading.

    Tracks margin utilization across positions and provides
    early warning for liquidation risk.

    Example:
        monitor = MarginMonitor(config=MarginConfig(warning_threshold=1.3))

        # Add positions
        position = MarginPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=Decimal("1.5"),
            entry_price=Decimal("45000"),
            mark_price=Decimal("44000"),
            leverage=Decimal("10"),
            margin_mode=MarginMode.CROSS,
            initial_margin=Decimal("6750"),
            maintenance_margin=Decimal("450"),
            margin_balance=Decimal("6000"),
        )
        monitor.update_position(position)

        # Get portfolio margin status
        portfolio = monitor.get_portfolio_margin()
        print(f"Margin utilization: {portfolio.margin_utilization:.1%}")
        print(f"Alert level: {portfolio.alert_level.value}")

        # Check alerts
        alerts = monitor.get_alerts()
        for alert in alerts:
            print(f"[{alert.severity.value}] {alert.description}")
    """

    def __init__(
        self,
        config: MarginConfig | None = None,
        account_equity: Decimal | None = None,
    ) -> None:
        """
        Initialize margin monitor.

        Args:
            config: Margin monitoring configuration
            account_equity: Total account equity for calculations
        """
        self.config = config or MarginConfig()
        self._account_equity = account_equity or Decimal("0")
        self._positions: dict[str, MarginPosition] = {}
        self._alerts: list[MarginAlert] = []
        self._last_alert_time: dict[str, datetime] = {}

    def update_account_equity(self, equity: Decimal) -> None:
        """Update account equity."""
        self._account_equity = equity

    def update_position(self, position: MarginPosition) -> list[MarginAlert]:
        """
        Update or add a position.

        Args:
            position: Position margin data

        Returns:
            List of new alerts triggered
        """
        self._positions[position.symbol] = position
        return self._check_position_alerts(position)

    def remove_position(self, symbol: str) -> None:
        """Remove a closed position."""
        self._positions.pop(symbol, None)

    def update_mark_price(
        self,
        symbol: str,
        mark_price: Decimal,
    ) -> list[MarginAlert]:
        """
        Update mark price for a position.

        Args:
            symbol: Trading symbol
            mark_price: Current mark price

        Returns:
            List of new alerts triggered
        """
        if symbol not in self._positions:
            return []

        pos = self._positions[symbol]

        # Create new position with updated price
        updated = MarginPosition(
            symbol=pos.symbol,
            side=pos.side,
            size=pos.size,
            entry_price=pos.entry_price,
            mark_price=mark_price,
            leverage=pos.leverage,
            margin_mode=pos.margin_mode,
            initial_margin=pos.initial_margin,
            maintenance_margin=pos.maintenance_margin,
            margin_balance=pos.margin_balance + (
                (mark_price - pos.mark_price) * pos.size
                if pos.side == PositionSide.LONG
                else (pos.mark_price - mark_price) * pos.size
            ),
        )

        self._positions[symbol] = updated
        return self._check_position_alerts(updated)

    def get_position(self, symbol: str) -> MarginPosition | None:
        """Get position margin data."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> list[MarginPosition]:
        """Get all position margin data."""
        return list(self._positions.values())

    def get_portfolio_margin(self) -> PortfolioMargin:
        """
        Calculate portfolio-level margin aggregation.

        Returns:
            PortfolioMargin with aggregated metrics
        """
        if not self._positions:
            return PortfolioMargin(
                total_initial_margin=Decimal("0"),
                total_maintenance_margin=Decimal("0"),
                total_margin_balance=self._account_equity,
                total_notional=Decimal("0"),
                total_unrealized_pnl=Decimal("0"),
                margin_utilization=0.0,
                available_margin=self._account_equity,
                margin_level=float("inf"),
                effective_leverage=0.0,
                largest_position_pct=0.0,
                position_count=0,
                alert_level=MarginAlertLevel.HEALTHY,
            )

        total_initial = sum(
            (p.initial_margin for p in self._positions.values()), Decimal("0")
        )
        total_maintenance = sum(
            (p.maintenance_margin for p in self._positions.values()), Decimal("0")
        )
        total_notional = sum(
            (p.notional_value for p in self._positions.values()), Decimal("0")
        )
        total_pnl = sum(
            (p.unrealized_pnl for p in self._positions.values()), Decimal("0")
        )

        # For cross margin, we use account equity
        # For isolated, we sum position balances
        has_cross = any(
            p.margin_mode == MarginMode.CROSS for p in self._positions.values()
        )

        if has_cross:
            total_balance = self._account_equity
        else:
            total_balance = sum(
                (p.margin_balance for p in self._positions.values()), Decimal("0")
            )

        # Utilization and level
        if total_balance > 0:
            margin_utilization = float(total_maintenance / total_balance)
            margin_level = float(total_balance / total_maintenance) if total_maintenance > 0 else float("inf")
        else:
            margin_utilization = 1.0 if total_maintenance > 0 else 0.0
            margin_level = 0.0 if total_maintenance > 0 else float("inf")

        available = total_balance - total_initial

        # Effective leverage
        equity = total_balance + total_pnl
        effective_leverage = float(total_notional / equity) if equity > 0 else 0.0

        # Largest position
        largest_notional = max(p.notional_value for p in self._positions.values())
        largest_pct = float(largest_notional / total_notional) * 100 if total_notional > 0 else 0.0

        return PortfolioMargin(
            total_initial_margin=total_initial,
            total_maintenance_margin=total_maintenance,
            total_margin_balance=total_balance,
            total_notional=total_notional,
            total_unrealized_pnl=total_pnl,
            margin_utilization=margin_utilization,
            available_margin=available,
            margin_level=margin_level,
            effective_leverage=effective_leverage,
            largest_position_pct=largest_pct,
            position_count=len(self._positions),
            alert_level=MarginAlertLevel.HEALTHY,  # Calculated in __post_init__
        )

    def calculate_liquidation_price(
        self,
        symbol: str,  # noqa: ARG002
        entry_price: Decimal,
        leverage: Decimal,
        side: PositionSide,
        maintenance_margin_rate: Decimal = Decimal("0.005"),  # 0.5%
    ) -> Decimal:
        """
        Calculate theoretical liquidation price.

        Args:
            symbol: Trading symbol (for future extension)
            entry_price: Position entry price
            leverage: Position leverage
            side: Long or short
            maintenance_margin_rate: Maintenance margin rate

        Returns:
            Estimated liquidation price
        """
        # For isolated margin:
        # Long: liq_price = entry_price * (1 - 1/leverage + maintenance_rate)
        # Short: liq_price = entry_price * (1 + 1/leverage - maintenance_rate)

        leverage_pct = Decimal("1") / leverage
        maint_rate = maintenance_margin_rate

        if side == PositionSide.LONG:
            liq_price = entry_price * (Decimal("1") - leverage_pct + maint_rate)
        else:
            liq_price = entry_price * (Decimal("1") + leverage_pct - maint_rate)

        return max(Decimal("0"), liq_price)

    def calculate_max_position_size(
        self,
        symbol: str,  # noqa: ARG002
        price: Decimal,
        leverage: Decimal,
        available_margin: Decimal | None = None,
    ) -> Decimal:
        """
        Calculate maximum position size given margin constraints.

        Args:
            symbol: Trading symbol (for future extension)
            price: Current price
            leverage: Desired leverage
            available_margin: Override available margin

        Returns:
            Maximum position size in base currency
        """
        if available_margin is None:
            portfolio = self.get_portfolio_margin()
            available_margin = portfolio.available_margin

        if available_margin <= 0 or price <= 0:
            return Decimal("0")

        # Max notional = available_margin * leverage
        max_notional = available_margin * leverage

        # Apply config limits
        if self.config.max_notional_per_symbol:
            max_notional = min(max_notional, self.config.max_notional_per_symbol)

        # Convert to position size
        max_size = max_notional / price

        return max_size

    def calculate_required_margin(
        self,
        size: Decimal,
        price: Decimal,
        leverage: Decimal,
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate required margin for a position.

        Args:
            size: Position size
            price: Entry price
            leverage: Leverage

        Returns:
            Tuple of (initial_margin, maintenance_margin)
        """
        notional = abs(size * price)
        initial_margin = notional / leverage
        # Assume 0.5% maintenance margin rate
        maintenance_margin = notional * Decimal("0.005")

        return initial_margin, maintenance_margin

    def _check_position_alerts(self, position: MarginPosition) -> list[MarginAlert]:
        """Check for position-level alerts."""
        new_alerts: list[MarginAlert] = []
        now = datetime.utcnow()

        # Check cooldown
        alert_key = f"position_{position.symbol}"
        last_alert = self._last_alert_time.get(alert_key)
        if last_alert and (now - last_alert).total_seconds() < self.config.alert_cooldown:
            return []

        # Check distance to liquidation
        if position.distance_to_liquidation_pct < 5:
            alert = MarginAlert(
                alert_type="approaching_liquidation",
                symbol=position.symbol,
                description=f"{position.symbol} is {position.distance_to_liquidation_pct:.1f}% from liquidation",
                current_value=position.distance_to_liquidation_pct,
                threshold=5.0,
                severity=MarginAlertLevel.CRITICAL,
            )
            new_alerts.append(alert)
            self._last_alert_time[alert_key] = now
        elif position.distance_to_liquidation_pct < 10:
            alert = MarginAlert(
                alert_type="low_margin",
                symbol=position.symbol,
                description=f"{position.symbol} margin buffer is low ({position.distance_to_liquidation_pct:.1f}%)",
                current_value=position.distance_to_liquidation_pct,
                threshold=10.0,
                severity=MarginAlertLevel.WARNING,
            )
            new_alerts.append(alert)
            self._last_alert_time[alert_key] = now

        # Check leverage
        if position.leverage > self.config.max_leverage:
            alert = MarginAlert(
                alert_type="excess_leverage",
                symbol=position.symbol,
                description=f"{position.symbol} leverage ({position.leverage}x) exceeds limit ({self.config.max_leverage}x)",
                current_value=float(position.leverage),
                threshold=float(self.config.max_leverage),
                severity=MarginAlertLevel.WARNING,
            )
            new_alerts.append(alert)

        self._alerts.extend(new_alerts)
        return new_alerts

    def check_portfolio_alerts(self) -> list[MarginAlert]:
        """Check for portfolio-level alerts."""
        portfolio = self.get_portfolio_margin()
        new_alerts: list[MarginAlert] = []
        now = datetime.utcnow()

        # Check cooldown
        last_alert = self._last_alert_time.get("portfolio")
        if last_alert and (now - last_alert).total_seconds() < self.config.alert_cooldown:
            return []

        # Check margin level thresholds
        if portfolio.margin_level <= 1.0:
            alert = MarginAlert(
                alert_type="liquidation",
                symbol=None,
                description="Portfolio is at liquidation level!",
                current_value=portfolio.margin_level,
                threshold=1.0,
                severity=MarginAlertLevel.LIQUIDATION,
            )
            new_alerts.append(alert)
        elif portfolio.margin_level < self.config.critical_threshold:
            alert = MarginAlert(
                alert_type="critical_margin",
                symbol=None,
                description=f"Portfolio margin level critical ({portfolio.margin_level:.2f})",
                current_value=portfolio.margin_level,
                threshold=self.config.critical_threshold,
                severity=MarginAlertLevel.CRITICAL,
            )
            new_alerts.append(alert)
        elif portfolio.margin_level < self.config.warning_threshold:
            alert = MarginAlert(
                alert_type="low_margin",
                symbol=None,
                description=f"Portfolio margin level warning ({portfolio.margin_level:.2f})",
                current_value=portfolio.margin_level,
                threshold=self.config.warning_threshold,
                severity=MarginAlertLevel.WARNING,
            )
            new_alerts.append(alert)

        # Check position concentration
        if portfolio.largest_position_pct > self.config.max_position_pct:
            alert = MarginAlert(
                alert_type="concentration",
                symbol=None,
                description=f"Position concentration ({portfolio.largest_position_pct:.1f}%) exceeds limit",
                current_value=portfolio.largest_position_pct,
                threshold=self.config.max_position_pct,
                severity=MarginAlertLevel.WARNING,
            )
            new_alerts.append(alert)

        if new_alerts:
            self._last_alert_time["portfolio"] = now
            self._alerts.extend(new_alerts)

        return new_alerts

    def get_alerts(self, limit: int = 20) -> list[MarginAlert]:
        """Get recent alerts."""
        return self._alerts[-limit:]

    def get_critical_alerts(self) -> list[MarginAlert]:
        """Get only critical and liquidation alerts."""
        return [
            a for a in self._alerts
            if a.severity in (MarginAlertLevel.CRITICAL, MarginAlertLevel.LIQUIDATION)
        ]

    def clear_alerts(self) -> None:
        """Clear alert history."""
        self._alerts.clear()
        self._last_alert_time.clear()

    def get_margin_summary(self) -> dict[str, Any]:
        """
        Get comprehensive margin summary.

        Returns:
            Dict with margin metrics and alerts
        """
        portfolio = self.get_portfolio_margin()
        positions = self.get_all_positions()

        # Sort positions by risk (closest to liquidation first)
        positions_sorted = sorted(
            positions,
            key=lambda p: p.distance_to_liquidation_pct,
        )

        return {
            "portfolio": {
                "total_margin": float(portfolio.total_margin_balance),
                "maintenance_required": float(portfolio.total_maintenance_margin),
                "available_margin": float(portfolio.available_margin),
                "margin_utilization": portfolio.margin_utilization,
                "margin_level": portfolio.margin_level,
                "effective_leverage": portfolio.effective_leverage,
                "alert_level": portfolio.alert_level.value,
            },
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "size": float(p.size),
                    "leverage": float(p.leverage),
                    "unrealized_pnl": float(p.unrealized_pnl),
                    "liquidation_price": float(p.liquidation_price),
                    "distance_to_liquidation": p.distance_to_liquidation_pct,
                    "margin_ratio": float(p.margin_ratio),
                }
                for p in positions_sorted
            ],
            "at_risk_positions": [
                p.symbol for p in positions_sorted
                if p.distance_to_liquidation_pct < 20
            ],
            "recent_alerts": [
                {
                    "type": a.alert_type,
                    "symbol": a.symbol,
                    "severity": a.severity.value,
                    "description": a.description,
                }
                for a in self._alerts[-5:]
            ],
        }


def create_margin_monitor(
    account_equity: Decimal = Decimal("0"),
    max_leverage: Decimal = Decimal("10"),
    warning_threshold: float = 1.25,
) -> MarginMonitor:
    """
    Factory function to create margin monitor.

    Args:
        account_equity: Initial account equity
        max_leverage: Maximum allowed leverage
        warning_threshold: Margin level warning threshold

    Returns:
        Configured MarginMonitor
    """
    config = MarginConfig(
        max_leverage=max_leverage,
        warning_threshold=warning_threshold,
    )
    return MarginMonitor(config=config, account_equity=account_equity)
