"""
Cross-Exchange Market Making (XEMM) Strategy (Issue #12).

XEMM exploits price discrepancies between two exchanges:
- Maker exchange: Places limit orders (earns spread + maker rebate)
- Taker exchange: Hedges with market orders when limit orders fill

This is a hybrid of market making and arbitrage:
- Market making risk: Earning spread on the maker exchange
- Arbitrage benefit: Hedging removes inventory risk
- Main risk: Execution risk (hedge might not fill at expected price)

Key concepts:
- Active hedging: Immediately hedge fills on taker exchange
- Profitability threshold: Minimum profit required to place orders
- Position limits: Maximum unhedged position allowed
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libra.plugins.hummingbot_adapter.config import XEMMConfig


class HedgeDirection(str, Enum):
    """Direction of hedge trade."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class CrossExchangeQuote:
    """Quote prices from both exchanges."""

    maker_bid: Decimal
    maker_ask: Decimal
    taker_bid: Decimal
    taker_ask: Decimal
    timestamp_ns: int

    @property
    def maker_mid(self) -> Decimal:
        """Maker exchange mid price."""
        return (self.maker_bid + self.maker_ask) / 2

    @property
    def taker_mid(self) -> Decimal:
        """Taker exchange mid price."""
        return (self.taker_bid + self.taker_ask) / 2

    @property
    def price_difference(self) -> Decimal:
        """Price difference between exchanges (taker - maker)."""
        return self.taker_mid - self.maker_mid

    @property
    def price_difference_pct(self) -> float:
        """Price difference as percentage."""
        if self.maker_mid == 0:
            return 0.0
        return float(self.price_difference / self.maker_mid)


@dataclass
class XEMMOrder:
    """Order for cross-exchange market making."""

    price: Decimal
    size: Decimal
    side: str  # "buy" or "sell"
    exchange: str  # "maker" or "taker"
    is_hedge: bool = False


@dataclass
class XEMMOpportunity:
    """A profitable XEMM opportunity."""

    # Maker side order
    maker_price: Decimal
    maker_size: Decimal
    maker_side: str  # "buy" or "sell"

    # Expected hedge
    hedge_price: Decimal
    hedge_side: str

    # Profitability
    expected_profit: Decimal
    expected_profit_pct: float

    timestamp_ns: int


@dataclass
class HedgeInstruction:
    """Instruction to execute a hedge trade."""

    side: HedgeDirection
    size: Decimal
    max_price: Decimal | None  # Maximum price for buy hedge
    min_price: Decimal | None  # Minimum price for sell hedge
    urgency: str = "normal"  # "normal", "urgent", "immediate"


class CrossExchangeMarketMakingStrategy:
    """
    Cross-Exchange Market Making strategy.

    Places limit orders on one exchange (maker) and hedges fills
    on another exchange (taker) to capture price discrepancies.
    """

    def __init__(
        self,
        config: XEMMConfig,
        min_spread: Decimal = Decimal("0.001"),
        max_spread: Decimal = Decimal("0.02"),
    ) -> None:
        """
        Initialize XEMM strategy.

        Args:
            config: XEMM configuration
            min_spread: Minimum spread for maker orders
            max_spread: Maximum spread for maker orders
        """
        self.config = config
        self.min_spread = min_spread
        self.max_spread = max_spread

        # Position tracking
        self._unhedged_position: Decimal = Decimal("0")  # Positive = long, negative = short
        self._total_profit: Decimal = Decimal("0")
        self._trade_count: int = 0

    @property
    def maker_exchange(self) -> str:
        """Get maker exchange name."""
        return self.config.maker_exchange

    @property
    def taker_exchange(self) -> str:
        """Get taker exchange name."""
        return self.config.taker_exchange

    def calculate_profitability(
        self,
        maker_price: Decimal,
        taker_price: Decimal,
        side: str,
    ) -> tuple[Decimal, float]:
        """
        Calculate expected profitability of a trade.

        For a BUY on maker (we buy low):
        - We buy at maker_price on maker exchange
        - We sell at taker_price on taker exchange
        - Profit = taker_price - maker_price - fees

        For a SELL on maker (we sell high):
        - We sell at maker_price on maker exchange
        - We buy at taker_price on taker exchange
        - Profit = maker_price - taker_price - fees

        Args:
            maker_price: Price on maker exchange
            taker_price: Price on taker exchange (for hedging)
            side: "buy" or "sell" on maker exchange

        Returns:
            Tuple of (absolute profit, profit percentage)
        """
        maker_fee = self.config.maker_fee
        taker_fee = self.config.taker_fee
        total_fees = maker_fee + taker_fee

        if side == "buy":
            # Buy on maker, sell on taker
            gross_profit = taker_price - maker_price
        else:
            # Sell on maker, buy on taker
            gross_profit = maker_price - taker_price

        # Subtract fees (as percentage of trade value)
        avg_price = (maker_price + taker_price) / 2
        fee_cost = avg_price * total_fees
        net_profit = gross_profit - fee_cost

        # Profit percentage
        profit_pct = float(net_profit / avg_price) if avg_price > 0 else 0.0

        return net_profit, profit_pct

    def find_opportunities(
        self,
        quotes: CrossExchangeQuote,
        order_size: Decimal,
    ) -> list[XEMMOpportunity]:
        """
        Find profitable XEMM opportunities.

        Args:
            quotes: Current quotes from both exchanges
            order_size: Size of orders to place

        Returns:
            List of profitable opportunities
        """
        opportunities: list[XEMMOpportunity] = []
        min_profit = self.config.min_profitability

        # Check BUY opportunity on maker
        # We would buy at maker_ask and sell at taker_bid
        buy_profit, buy_profit_pct = self.calculate_profitability(
            maker_price=quotes.maker_ask,
            taker_price=quotes.taker_bid,
            side="buy",
        )

        if buy_profit_pct >= float(min_profit):
            opportunities.append(
                XEMMOpportunity(
                    maker_price=quotes.maker_ask,
                    maker_size=order_size,
                    maker_side="buy",
                    hedge_price=quotes.taker_bid,
                    hedge_side="sell",
                    expected_profit=buy_profit * order_size,
                    expected_profit_pct=buy_profit_pct,
                    timestamp_ns=quotes.timestamp_ns,
                )
            )

        # Check SELL opportunity on maker
        # We would sell at maker_bid and buy at taker_ask
        sell_profit, sell_profit_pct = self.calculate_profitability(
            maker_price=quotes.maker_bid,
            taker_price=quotes.taker_ask,
            side="sell",
        )

        if sell_profit_pct >= float(min_profit):
            opportunities.append(
                XEMMOpportunity(
                    maker_price=quotes.maker_bid,
                    maker_size=order_size,
                    maker_side="sell",
                    hedge_price=quotes.taker_ask,
                    hedge_side="buy",
                    expected_profit=sell_profit * order_size,
                    expected_profit_pct=sell_profit_pct,
                    timestamp_ns=quotes.timestamp_ns,
                )
            )

        return opportunities

    def generate_maker_orders(
        self,
        quotes: CrossExchangeQuote,
        order_size: Decimal,
    ) -> list[XEMMOrder]:
        """
        Generate maker exchange orders.

        Places orders that would be profitable if filled and hedged.

        Args:
            quotes: Current quotes from both exchanges
            order_size: Size of orders to place

        Returns:
            List of orders to place on maker exchange
        """
        orders: list[XEMMOrder] = []

        # Find opportunities
        opportunities = self.find_opportunities(quotes, order_size)

        for opp in opportunities:
            # Check position limits
            if opp.maker_side == "buy":
                new_position = self._unhedged_position + opp.maker_size
            else:
                new_position = self._unhedged_position - opp.maker_size

            max_position = self.config.max_unhedged_position
            if max_position > 0 and abs(new_position) > max_position:
                continue

            orders.append(
                XEMMOrder(
                    price=opp.maker_price,
                    size=opp.maker_size,
                    side=opp.maker_side,
                    exchange="maker",
                    is_hedge=False,
                )
            )

        return orders

    def on_maker_fill(
        self,
        side: str,
        size: Decimal,
        fill_price: Decimal,
    ) -> HedgeInstruction | None:
        """
        Handle a fill on the maker exchange.

        Updates position and generates hedge instruction if active hedging is enabled.

        Args:
            side: "buy" or "sell"
            size: Filled size
            fill_price: Fill price

        Returns:
            HedgeInstruction if hedging is needed, None otherwise
        """
        # Update unhedged position
        if side == "buy":
            self._unhedged_position += size
        else:
            self._unhedged_position -= size

        self._trade_count += 1

        # Generate hedge instruction if active hedging enabled
        if not self.config.active_hedging:
            return None

        if side == "buy":
            # We bought on maker, need to sell on taker
            return HedgeInstruction(
                side=HedgeDirection.SELL,
                size=size,
                max_price=None,
                min_price=fill_price * (1 - self.config.min_profitability),
                urgency="normal",
            )
        else:
            # We sold on maker, need to buy on taker
            return HedgeInstruction(
                side=HedgeDirection.BUY,
                size=size,
                max_price=fill_price * (1 + self.config.min_profitability),
                min_price=None,
                urgency="normal",
            )

    def on_hedge_fill(
        self,
        side: HedgeDirection,
        size: Decimal,
        fill_price: Decimal,
        original_maker_price: Decimal,
    ) -> None:
        """
        Handle a hedge fill on the taker exchange.

        Updates position and records profit.

        Args:
            side: Hedge direction
            size: Filled size
            fill_price: Fill price
            original_maker_price: The price of the original maker fill
        """
        # Update unhedged position
        if side == HedgeDirection.BUY:
            self._unhedged_position += size
        else:
            self._unhedged_position -= size

        # Calculate realized profit
        if side == HedgeDirection.SELL:
            # We bought on maker, sold on taker
            profit = (fill_price - original_maker_price) * size
        else:
            # We sold on maker, bought on taker
            profit = (original_maker_price - fill_price) * size

        # Subtract fees
        fee_cost = (original_maker_price * self.config.maker_fee + fill_price * self.config.taker_fee) * size
        net_profit = profit - fee_cost

        self._total_profit += net_profit

    def needs_forced_hedge(self) -> bool:
        """Check if we need to force a hedge due to position limits."""
        max_position = self.config.max_unhedged_position
        if max_position <= 0:
            return False
        return abs(self._unhedged_position) > max_position

    def get_forced_hedge_instruction(self) -> HedgeInstruction | None:
        """Get instruction for forced hedge if needed."""
        if not self.needs_forced_hedge():
            return None

        if self._unhedged_position > 0:
            # Long position, need to sell
            return HedgeInstruction(
                side=HedgeDirection.SELL,
                size=abs(self._unhedged_position),
                max_price=None,
                min_price=None,  # Market order
                urgency="urgent",
            )
        else:
            # Short position, need to buy
            return HedgeInstruction(
                side=HedgeDirection.BUY,
                size=abs(self._unhedged_position),
                max_price=None,  # Market order
                min_price=None,
                urgency="urgent",
            )

    @property
    def unhedged_position(self) -> Decimal:
        """Current unhedged position."""
        return self._unhedged_position

    @property
    def total_profit(self) -> Decimal:
        """Total realized profit."""
        return self._total_profit

    @property
    def trade_count(self) -> int:
        """Total number of trades."""
        return self._trade_count

    def reset(self) -> None:
        """Reset strategy state."""
        self._unhedged_position = Decimal("0")
        self._total_profit = Decimal("0")
        self._trade_count = 0
