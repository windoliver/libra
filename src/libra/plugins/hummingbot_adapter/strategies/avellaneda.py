"""
Avellaneda-Stoikov Market Making Strategy (Issue #12).

Implementation of the optimal market making strategy from:
"High-frequency Trading in a Limit Order Book" (Avellaneda & Stoikov, 2008)

Key concepts:
- Reservation price: The price at which the market maker is indifferent to trading
- Optimal spread: The spread that maximizes expected utility
- Inventory risk: Adjusting quotes based on current inventory position

Adapted for crypto markets with:
- 24/7 trading support (infinite timeframe mode)
- Real-time volatility estimation
- Configurable risk parameters
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libra.plugins.hummingbot_adapter.config import AvellanedaStoikovConfig, InventoryConfig


@dataclass
class Quote:
    """A two-sided quote (bid and ask)."""

    bid_price: Decimal
    ask_price: Decimal
    bid_size: Decimal
    ask_size: Decimal
    timestamp_ns: int

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> Decimal:
        """Calculate spread."""
        return self.ask_price - self.bid_price

    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid = self.mid_price
        if mid == 0:
            return 0.0
        return float(self.spread / mid)


@dataclass
class VolatilityEstimator:
    """
    Real-time volatility estimation using exponential weighted moving average.

    Uses returns-based volatility calculation suitable for high-frequency data.
    """

    window_seconds: int = 300  # 5 minutes default
    min_samples: int = 10
    _prices: deque = field(default_factory=lambda: deque(maxlen=1000))
    _timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    _returns: deque = field(default_factory=lambda: deque(maxlen=1000))

    def update(self, price: Decimal, timestamp_ns: int) -> None:
        """Update with new price observation."""
        if self._prices:
            # Calculate log return
            last_price = self._prices[-1]
            if last_price > 0 and price > 0:
                log_return = math.log(float(price) / float(last_price))
                self._returns.append(log_return)

        self._prices.append(price)
        self._timestamps.append(timestamp_ns)

        # Trim old data outside window
        cutoff_ns = timestamp_ns - (self.window_seconds * 1_000_000_000)
        while self._timestamps and self._timestamps[0] < cutoff_ns:
            self._timestamps.popleft()
            self._prices.popleft()
            if self._returns:
                self._returns.popleft()

    def get_volatility(self) -> float:
        """
        Calculate annualized volatility.

        Returns volatility as a decimal (e.g., 0.5 = 50% annualized vol).
        """
        if len(self._returns) < self.min_samples:
            return 0.0

        # Calculate standard deviation of returns
        returns = list(self._returns)
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)

        # Annualize (assuming ~1 second between observations)
        # sqrt(365 * 24 * 3600) for per-second data
        annualization_factor = math.sqrt(365 * 24 * 3600)

        return std_dev * annualization_factor

    def reset(self) -> None:
        """Reset all stored data."""
        self._prices.clear()
        self._timestamps.clear()
        self._returns.clear()


class AvellanedaStoikovStrategy:
    """
    Avellaneda-Stoikov optimal market making strategy.

    This strategy calculates:
    1. Reservation price (r): Where the MM is indifferent to trading
       r = s - q * gamma * sigma^2 * (T - t)

    2. Optimal spread (delta): The spread that maximizes utility
       delta = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)

    Where:
    - s = mid price
    - q = inventory (positive = long, negative = short)
    - gamma = risk aversion parameter
    - sigma = volatility
    - T - t = time remaining (1.0 for infinite mode)
    - kappa = order book depth parameter
    """

    def __init__(
        self,
        config: AvellanedaStoikovConfig,
        inventory_config: InventoryConfig,
        min_spread: Decimal = Decimal("0.001"),
        max_spread: Decimal = Decimal("0.05"),
    ) -> None:
        """
        Initialize Avellaneda-Stoikov strategy.

        Args:
            config: Avellaneda-Stoikov specific configuration
            inventory_config: Inventory management configuration
            min_spread: Minimum allowed spread (as decimal)
            max_spread: Maximum allowed spread (as decimal)
        """
        self.config = config
        self.inventory_config = inventory_config
        self.min_spread = min_spread
        self.max_spread = max_spread

        # State
        self._volatility_estimator = VolatilityEstimator(
            window_seconds=config.volatility_window
        )
        self._last_quote_time_ns: int = 0
        self._current_inventory: Decimal = Decimal("0")
        self._target_inventory: Decimal = Decimal("0")

    @property
    def gamma(self) -> float:
        """Risk aversion parameter."""
        return self.config.risk_aversion

    @property
    def kappa(self) -> float:
        """Order book depth parameter."""
        return self.config.order_book_depth

    def update_price(self, price: Decimal, timestamp_ns: int) -> None:
        """Update volatility estimator with new price."""
        self._volatility_estimator.update(price, timestamp_ns)

    def set_inventory(
        self,
        base_balance: Decimal,
        quote_balance: Decimal,
        mid_price: Decimal,
    ) -> None:
        """
        Set current inventory position.

        Args:
            base_balance: Amount of base currency held
            quote_balance: Amount of quote currency held
            mid_price: Current mid price for valuation
        """
        # Calculate inventory as deviation from target
        if mid_price <= 0:
            self._current_inventory = Decimal("0")
            return

        # Total portfolio value in quote currency
        total_value = base_balance * mid_price + quote_balance

        if total_value <= 0:
            self._current_inventory = Decimal("0")
            return

        # Current ratio of base value to total
        current_ratio = (base_balance * mid_price) / total_value

        # Target inventory in base units
        target_base = (total_value * Decimal(str(self.inventory_config.target_ratio))) / mid_price

        # Inventory deviation (positive = too much base, negative = too little)
        self._current_inventory = base_balance - target_base
        self._target_inventory = target_base

    def calculate_reservation_price(
        self,
        mid_price: Decimal,
        time_remaining: float = 1.0,
    ) -> Decimal:
        """
        Calculate the reservation price.

        The reservation price is where the market maker is indifferent to trading.
        When inventory is positive (long), reservation price is below mid to encourage selling.
        When inventory is negative (short), reservation price is above mid to encourage buying.

        Args:
            mid_price: Current market mid price
            time_remaining: Time remaining as fraction (1.0 for infinite mode)

        Returns:
            Reservation price
        """
        sigma = self._volatility_estimator.get_volatility()

        if sigma == 0:
            # No volatility data yet, use mid price
            return mid_price

        # q * gamma * sigma^2 * (T - t)
        q = float(self._current_inventory)
        adjustment = Decimal(str(q * self.gamma * (sigma ** 2) * time_remaining))

        reservation = mid_price - adjustment

        return reservation

    def calculate_optimal_spread(
        self,
        time_remaining: float = 1.0,
    ) -> Decimal:
        """
        Calculate the optimal spread.

        The optimal spread balances:
        - Earning the spread when orders are filled
        - Risk of adverse price movement

        Args:
            time_remaining: Time remaining as fraction (1.0 for infinite mode)

        Returns:
            Optimal spread as decimal
        """
        sigma = self._volatility_estimator.get_volatility()

        if sigma == 0:
            # No volatility data, use minimum spread
            return self.min_spread

        # gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)
        gamma = self.gamma
        kappa = self.kappa

        if gamma == 0:
            # Zero risk aversion = minimum spread
            return self.min_spread

        term1 = gamma * (sigma ** 2) * time_remaining
        term2 = (2 / gamma) * math.log(1 + gamma / kappa)

        optimal_spread = Decimal(str(term1 + term2))

        # Clamp to min/max
        return max(self.min_spread, min(self.max_spread, optimal_spread))

    def calculate_inventory_skew(self) -> tuple[Decimal, Decimal]:
        """
        Calculate bid/ask skew based on inventory.

        Returns:
            Tuple of (bid_skew, ask_skew) as multipliers
        """
        if self._target_inventory == 0:
            return Decimal("1"), Decimal("1")

        # Inventory ratio: how far from target (-1 to 1 range approximately)
        inventory_ratio = float(self._current_inventory / self._target_inventory)

        # Clamp to reasonable range
        inventory_ratio = max(-2.0, min(2.0, inventory_ratio))

        intensity = self.inventory_config.skew_intensity

        if self.inventory_config.skew_mode.value == "linear":
            # Linear skew
            bid_skew = Decimal(str(1.0 + inventory_ratio * intensity * 0.5))
            ask_skew = Decimal(str(1.0 - inventory_ratio * intensity * 0.5))

        elif self.inventory_config.skew_mode.value == "exponential":
            # Exponential skew for more aggressive rebalancing
            bid_skew = Decimal(str(math.exp(inventory_ratio * intensity * 0.5)))
            ask_skew = Decimal(str(math.exp(-inventory_ratio * intensity * 0.5)))

        else:
            # No skew
            bid_skew = Decimal("1")
            ask_skew = Decimal("1")

        # Ensure skews are positive
        bid_skew = max(Decimal("0.1"), bid_skew)
        ask_skew = max(Decimal("0.1"), ask_skew)

        return bid_skew, ask_skew

    def generate_quote(
        self,
        mid_price: Decimal,
        order_size: Decimal,
        timestamp_ns: int,
        time_remaining: float = 1.0,
    ) -> Quote | None:
        """
        Generate optimal bid/ask quote.

        Args:
            mid_price: Current market mid price
            order_size: Base order size
            timestamp_ns: Current timestamp in nanoseconds
            time_remaining: Time remaining as fraction (1.0 for infinite mode)

        Returns:
            Quote with optimal bid/ask prices and sizes, or None if too soon
        """
        # Check minimum refresh time
        min_refresh_ns = int(self.config.min_quote_refresh * 1_000_000_000)
        if timestamp_ns - self._last_quote_time_ns < min_refresh_ns:
            return None

        # Update price for volatility calculation
        self.update_price(mid_price, timestamp_ns)

        # Calculate reservation price and optimal spread
        reservation_price = self.calculate_reservation_price(mid_price, time_remaining)
        optimal_spread = self.calculate_optimal_spread(time_remaining)

        # Calculate base bid/ask from reservation price
        half_spread = optimal_spread / 2
        base_bid = reservation_price - half_spread
        base_ask = reservation_price + half_spread

        # Apply inventory skew
        bid_skew, ask_skew = self.calculate_inventory_skew()

        # Skew adjusts the spread asymmetrically
        # Higher bid_skew = wider bid spread (less aggressive buying)
        # Higher ask_skew = wider ask spread (less aggressive selling)
        bid_price = reservation_price - (half_spread * bid_skew)
        ask_price = reservation_price + (half_spread * ask_skew)

        # Ensure minimum spread
        actual_spread = ask_price - bid_price
        if actual_spread < self.min_spread * mid_price:
            adjustment = (self.min_spread * mid_price - actual_spread) / 2
            bid_price -= adjustment
            ask_price += adjustment

        # Round to reasonable precision (8 decimal places)
        bid_price = bid_price.quantize(Decimal("0.00000001"))
        ask_price = ask_price.quantize(Decimal("0.00000001"))

        # Calculate order sizes (can be adjusted based on inventory)
        bid_size = order_size
        ask_size = order_size

        # Reduce size on the side that increases inventory imbalance
        if self._current_inventory > 0:
            # Long inventory - reduce bid size
            reduction = min(Decimal("0.5"), abs(self._current_inventory / self._target_inventory) * Decimal("0.3"))
            bid_size = order_size * (1 - reduction)
        elif self._current_inventory < 0:
            # Short inventory - reduce ask size
            reduction = min(Decimal("0.5"), abs(self._current_inventory / self._target_inventory) * Decimal("0.3"))
            ask_size = order_size * (1 - reduction)

        self._last_quote_time_ns = timestamp_ns

        return Quote(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size.quantize(Decimal("0.00000001")),
            ask_size=ask_size.quantize(Decimal("0.00000001")),
            timestamp_ns=timestamp_ns,
        )

    def reset(self) -> None:
        """Reset strategy state."""
        self._volatility_estimator.reset()
        self._last_quote_time_ns = 0
        self._current_inventory = Decimal("0")
        self._target_inventory = Decimal("0")
