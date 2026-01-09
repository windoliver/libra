"""
Scheduled Rebalance Strategy: Time-based portfolio rebalancing.

Demonstrates Issue #24 extensible protocol features:
- StrategyType.SCHEDULED for time-based execution
- DateRule and TimeRule for scheduling
- ScheduledTask for task registration

This strategy rebalances a portfolio to target weights on a schedule,
similar to QuantConnect's scheduled universe selection.

Example use cases:
- Weekly rebalancing to maintain target allocations
- Monthly momentum rotation strategies
- Daily end-of-day portfolio adjustments
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal

from libra.strategies.base import BaseStrategy


logger = logging.getLogger(__name__)
from libra.strategies.protocol import (
    DateRule,
    ScheduledTask,
    Signal,
    SignalType,
    StrategyConfig,
    StrategyType,
    TimeRule,
)


@dataclass
class RebalanceConfig(StrategyConfig):
    """
    Configuration for Scheduled Rebalance Strategy.

    Attributes:
        symbol: Primary symbol (inherited)
        timeframe: Timeframe (inherited)
        target_weights: Target portfolio allocation per symbol
        rebalance_threshold: Min deviation % before rebalancing
        date_rule: When to check for rebalancing (default: weekly)
        time_rule: Time of day to rebalance (default: market open)
        offset_minutes: Minutes after time_rule to execute

    Examples:
        config = RebalanceConfig(
            symbol="BTC/USDT",
            target_weights={
                "BTC/USDT": Decimal("0.60"),
                "ETH/USDT": Decimal("0.40"),
            },
            rebalance_threshold=Decimal("0.05"),  # 5% deviation
            date_rule=DateRule.WEEK_START,
            time_rule=TimeRule.MARKET_OPEN,
        )
    """

    strategy_type: StrategyType = StrategyType.SCHEDULED
    target_weights: dict[str, Decimal] = field(
        default_factory=lambda: {"BTC/USDT": Decimal("1.0")}
    )
    rebalance_threshold: Decimal = Decimal("0.05")  # 5% deviation triggers rebalance
    date_rule: DateRule = DateRule.WEEK_START
    time_rule: TimeRule = TimeRule.MARKET_OPEN
    offset_minutes: int = 30  # 30 min after market open


class ScheduledRebalanceStrategy(BaseStrategy):
    """
    Scheduled portfolio rebalancing strategy.

    Executes on a time-based schedule (not event-driven) to:
    1. Calculate current portfolio weights
    2. Compare to target weights
    3. Generate rebalance signals if deviation exceeds threshold

    This demonstrates the SCHEDULED strategy type from Issue #24,
    inspired by:
    - Zipline's schedule_function
    - QuantConnect's scheduled events
    - Backtrader's timer-based execution

    Example:
        config = RebalanceConfig(
            symbol="BTC/USDT",
            target_weights={
                "BTC/USDT": Decimal("0.60"),
                "ETH/USDT": Decimal("0.40"),
            },
        )
        strategy = ScheduledRebalanceStrategy(config)
        strategy.on_start()

        # Called by executor on schedule (not on every bar)
        signals = strategy.on_schedule()
    """

    # Strategy type declaration (Issue #24)
    strategy_type = StrategyType.SCHEDULED

    def __init__(self, config: RebalanceConfig) -> None:
        """
        Initialize Scheduled Rebalance Strategy.

        Args:
            config: Strategy configuration with target weights
        """
        super().__init__(config)
        self._config: RebalanceConfig = config

        # Current portfolio state (would be populated by executor)
        self._current_weights: dict[str, Decimal] = {}
        self._current_prices: dict[str, Decimal] = {}
        self._last_rebalance_ns: int = 0
        self._rebalance_count: int = 0

        # Register scheduled tasks
        self._scheduled_tasks: list[ScheduledTask] = [
            ScheduledTask(
                func_name="on_schedule",
                date_rule=config.date_rule,
                time_rule=config.time_rule,
                offset_minutes=config.offset_minutes,
            ),
        ]

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "scheduled_rebalance"

    @property
    def symbols(self) -> list[str]:
        """All symbols in the target portfolio."""
        return list(self._config.target_weights.keys())

    @property
    def scheduled_tasks(self) -> list[ScheduledTask]:
        """Get registered scheduled tasks."""
        return self._scheduled_tasks

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_start(self) -> None:
        """Initialize strategy state."""
        super().on_start()
        self._current_weights = {}
        self._current_prices = {}
        self._last_rebalance_ns = 0
        self._rebalance_count = 0

    def on_stop(self) -> None:
        """Cleanup strategy state."""
        super().on_stop()

    def on_reset(self) -> None:
        """Reset strategy between backtest runs."""
        super().on_reset()
        self._current_weights = {}
        self._current_prices = {}
        self._last_rebalance_ns = 0
        self._rebalance_count = 0

    # -------------------------------------------------------------------------
    # Scheduled Execution
    # -------------------------------------------------------------------------

    def on_schedule(self) -> list[Signal]:
        """
        Called on schedule (not on every bar).

        This is the main entry point for scheduled strategies.
        The executor calls this based on the ScheduledTask configuration.

        Returns:
            List of rebalancing signals
        """
        signals: list[Signal] = []

        # Calculate weight deviations
        deviations = self._calculate_deviations()

        # Generate rebalance signals for symbols exceeding threshold
        for symbol, deviation in deviations.items():
            if abs(deviation) > self._config.rebalance_threshold:
                signal = self._generate_rebalance_signal(symbol, deviation)
                if signal:
                    signals.append(signal)

        if signals:
            self._last_rebalance_ns = time.time_ns()
            self._rebalance_count += 1
            logger.info(
                f"Rebalance #{self._rebalance_count}: {len(signals)} signals"
            )

        return signals

    def _calculate_deviations(self) -> dict[str, Decimal]:
        """
        Calculate deviation of current weights from target.

        Returns:
            Dict of symbol -> deviation (positive = overweight)
        """
        deviations: dict[str, Decimal] = {}

        for symbol, target_weight in self._config.target_weights.items():
            current_weight = self._current_weights.get(symbol, Decimal("0"))
            deviations[symbol] = current_weight - target_weight

        return deviations

    def _generate_rebalance_signal(
        self, symbol: str, deviation: Decimal
    ) -> Signal | None:
        """
        Generate a rebalance signal for a symbol.

        Args:
            symbol: Symbol to rebalance
            deviation: Current weight deviation (positive = overweight)

        Returns:
            Signal to adjust position
        """
        target_weight = self._config.target_weights.get(symbol, Decimal("0"))

        if deviation > 0:
            # Overweight - reduce position
            if deviation > self._config.rebalance_threshold:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=symbol,
                    timestamp_ns=time.time_ns(),
                    strength=float(abs(deviation)),  # How much to reduce
                    price=self._current_prices.get(symbol),
                    metadata={
                        "reason": "rebalance_reduce",
                        "deviation": str(deviation),
                        "target_weight": str(target_weight),
                    },
                )
        else:
            # Underweight - increase position
            if abs(deviation) > self._config.rebalance_threshold:
                return Signal(
                    signal_type=SignalType.LONG,
                    symbol=symbol,
                    timestamp_ns=time.time_ns(),
                    strength=float(abs(deviation)),  # How much to add
                    price=self._current_prices.get(symbol),
                    metadata={
                        "reason": "rebalance_increase",
                        "deviation": str(deviation),
                        "target_weight": str(target_weight),
                    },
                )

        return None

    # -------------------------------------------------------------------------
    # Portfolio State Updates
    # -------------------------------------------------------------------------

    def update_portfolio_state(
        self,
        weights: dict[str, Decimal],
        prices: dict[str, Decimal],
    ) -> None:
        """
        Update current portfolio state.

        Called by executor before scheduled tasks run.

        Args:
            weights: Current portfolio weights per symbol
            prices: Current prices per symbol
        """
        self._current_weights = weights.copy()
        self._current_prices = prices.copy()

    # -------------------------------------------------------------------------
    # State Persistence
    # -------------------------------------------------------------------------

    def on_save(self) -> dict[str, bytes]:
        """Save strategy state."""
        state = {
            "current_weights": {k: str(v) for k, v in self._current_weights.items()},
            "current_prices": {k: str(v) for k, v in self._current_prices.items()},
            "last_rebalance_ns": self._last_rebalance_ns,
            "rebalance_count": self._rebalance_count,
        }
        return {"state": json.dumps(state).encode()}

    def on_load(self, state: dict[str, bytes]) -> None:
        """Load strategy state."""
        if "state" not in state:
            return

        data = json.loads(state["state"].decode())
        self._current_weights = {
            k: Decimal(v) for k, v in data.get("current_weights", {}).items()
        }
        self._current_prices = {
            k: Decimal(v) for k, v in data.get("current_prices", {}).items()
        }
        self._last_rebalance_ns = data.get("last_rebalance_ns", 0)
        self._rebalance_count = data.get("rebalance_count", 0)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get current strategy status."""
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "target_weights": {
                k: str(v) for k, v in self._config.target_weights.items()
            },
            "current_weights": {
                k: str(v) for k, v in self._current_weights.items()
            },
            "rebalance_count": self._rebalance_count,
            "threshold": str(self._config.rebalance_threshold),
            "schedule": f"{self._config.date_rule.value}@{self._config.time_rule.value}+{self._config.offset_minutes}min",
        }
