"""
Pure Market Making Strategy (Issue #12).

A simpler market making strategy that places orders at fixed spreads
around the mid price, with inventory-based adjustments.

Features:
- Fixed spread market making
- Multiple order levels
- Inventory skew for risk management
- Hanging orders support
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libra.plugins.hummingbot_adapter.config import InventoryConfig


@dataclass
class OrderLevel:
    """An order at a specific price level."""

    price: Decimal
    size: Decimal
    level: int  # 0 = best, 1 = second best, etc.


@dataclass
class TwoSidedOrder:
    """A two-sided market making order set."""

    bids: list[OrderLevel]
    asks: list[OrderLevel]
    timestamp_ns: int
    mid_price: Decimal

    @property
    def best_bid(self) -> Decimal | None:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Decimal | None:
        """Calculate spread between best bid and ask."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


class PureMarketMakingStrategy:
    """
    Pure market making strategy with fixed spreads.

    This is a simpler alternative to Avellaneda-Stoikov that:
    - Places orders at fixed percentage spreads from mid price
    - Supports multiple order levels
    - Adjusts prices based on inventory position
    - Can maintain "hanging" orders that persist until filled
    """

    def __init__(
        self,
        inventory_config: InventoryConfig,
        base_spread: Decimal = Decimal("0.002"),  # 0.2%
        order_levels: int = 1,
        level_spread: Decimal = Decimal("0.001"),  # 0.1% between levels
        min_spread: Decimal = Decimal("0.001"),
        max_spread: Decimal = Decimal("0.05"),
    ) -> None:
        """
        Initialize Pure Market Making strategy.

        Args:
            inventory_config: Inventory management configuration
            base_spread: Base spread from mid price (as decimal)
            order_levels: Number of order levels on each side
            level_spread: Spread between consecutive levels
            min_spread: Minimum allowed spread
            max_spread: Maximum allowed spread
        """
        self.inventory_config = inventory_config
        self.base_spread = base_spread
        self.order_levels = order_levels
        self.level_spread = level_spread
        self.min_spread = min_spread
        self.max_spread = max_spread

        # State
        self._current_inventory: Decimal = Decimal("0")
        self._target_inventory: Decimal = Decimal("0")
        self._hanging_orders: list[OrderLevel] = []

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
        if mid_price <= 0:
            self._current_inventory = Decimal("0")
            return

        # Total portfolio value in quote currency
        total_value = base_balance * mid_price + quote_balance

        if total_value <= 0:
            self._current_inventory = Decimal("0")
            return

        # Target inventory in base units
        target_base = (total_value * Decimal(str(self.inventory_config.target_ratio))) / mid_price

        # Inventory deviation
        self._current_inventory = base_balance - target_base
        self._target_inventory = target_base

    def calculate_inventory_skew(self) -> tuple[Decimal, Decimal]:
        """
        Calculate bid/ask price adjustments based on inventory.

        Returns:
            Tuple of (bid_adjustment, ask_adjustment) as decimals to add to spread
        """
        if self._target_inventory == 0:
            return Decimal("0"), Decimal("0")

        # Inventory ratio
        inventory_ratio = float(self._current_inventory / self._target_inventory)
        inventory_ratio = max(-2.0, min(2.0, inventory_ratio))

        intensity = self.inventory_config.skew_intensity

        # Positive inventory (long) -> widen bid, tighten ask
        # Negative inventory (short) -> tighten bid, widen ask
        skew = Decimal(str(inventory_ratio * intensity * 0.002))  # 0.2% per unit

        # bid_adjustment positive = wider bid spread
        # ask_adjustment positive = wider ask spread
        bid_adjustment = skew
        ask_adjustment = -skew

        return bid_adjustment, ask_adjustment

    def generate_orders(
        self,
        mid_price: Decimal,
        order_size: Decimal,
        timestamp_ns: int,
    ) -> TwoSidedOrder:
        """
        Generate market making orders.

        Args:
            mid_price: Current market mid price
            order_size: Base order size for each level
            timestamp_ns: Current timestamp in nanoseconds

        Returns:
            TwoSidedOrder with bid and ask order levels
        """
        bids: list[OrderLevel] = []
        asks: list[OrderLevel] = []

        # Get inventory-based adjustments
        bid_adj, ask_adj = self.calculate_inventory_skew()

        for level in range(self.order_levels):
            # Calculate spread for this level
            level_offset = self.level_spread * level
            total_bid_spread = self.base_spread + level_offset + bid_adj
            total_ask_spread = self.base_spread + level_offset + ask_adj

            # Clamp spreads
            total_bid_spread = max(self.min_spread, min(self.max_spread, total_bid_spread))
            total_ask_spread = max(self.min_spread, min(self.max_spread, total_ask_spread))

            # Calculate prices
            bid_price = mid_price * (1 - total_bid_spread)
            ask_price = mid_price * (1 + total_ask_spread)

            # Round to reasonable precision
            bid_price = bid_price.quantize(Decimal("0.00000001"))
            ask_price = ask_price.quantize(Decimal("0.00000001"))

            # Adjust size for outer levels (smaller)
            level_size = order_size * Decimal(str(0.8 ** level))
            level_size = level_size.quantize(Decimal("0.00000001"))

            bids.append(OrderLevel(price=bid_price, size=level_size, level=level))
            asks.append(OrderLevel(price=ask_price, size=level_size, level=level))

        return TwoSidedOrder(
            bids=bids,
            asks=asks,
            timestamp_ns=timestamp_ns,
            mid_price=mid_price,
        )

    def should_refresh_orders(
        self,
        current_orders: TwoSidedOrder | None,
        new_mid_price: Decimal,
        price_threshold: Decimal = Decimal("0.001"),  # 0.1%
    ) -> bool:
        """
        Check if orders should be refreshed based on price movement.

        Args:
            current_orders: Currently active orders
            new_mid_price: New market mid price
            price_threshold: Price change threshold to trigger refresh

        Returns:
            True if orders should be refreshed
        """
        if current_orders is None:
            return True

        # Check if price moved significantly
        price_change = abs(new_mid_price - current_orders.mid_price) / current_orders.mid_price

        return price_change >= price_threshold

    def add_hanging_order(self, order: OrderLevel) -> None:
        """Add an order to the hanging orders list."""
        if self.inventory_config.hanging_orders:
            self._hanging_orders.append(order)

    def get_hanging_orders(self) -> list[OrderLevel]:
        """Get current hanging orders."""
        return self._hanging_orders.copy()

    def clear_hanging_orders(self) -> None:
        """Clear all hanging orders."""
        self._hanging_orders.clear()

    def reset(self) -> None:
        """Reset strategy state."""
        self._current_inventory = Decimal("0")
        self._target_inventory = Decimal("0")
        self._hanging_orders.clear()
