"""
Buying Power Calculator with Settlement Awareness.

Calculates available buying power based on account type,
settlement status, and margin rules.
"""

from __future__ import annotations

from decimal import Decimal

from libra.core.settlement.models import AccountType, BuyingPower
from libra.core.settlement.tracker import SettlementTracker


class BuyingPowerCalculator:
    """
    Calculates buying power based on account type and settlement status.

    Different account types have different rules:
    - Cash: Only settled funds available
    - Margin: Can use unsettled + margin (typically 2x)
    - PDT: Pattern Day Trader, 4x intraday buying power

    Example:
        >>> tracker = SettlementTracker()
        >>> calculator = BuyingPowerCalculator(tracker, account_type=AccountType.CASH)
        >>> bp = calculator.calculate(cash_balance=Decimal("10000"))
        >>> bp.buying_power
        Decimal('10000')  # Full cash available
        >>> # After a pending buy
        >>> tracker.add_trade("T1", "AAPL", "buy", Decimal("100"), Decimal("50"))
        >>> bp = calculator.calculate(cash_balance=Decimal("10000"))
        >>> bp.buying_power
        Decimal('5000')  # Cash minus pending debit
    """

    # Default margin rates
    DEFAULT_MARGIN_RATE = Decimal("0.5")  # 50% margin = 2x leverage
    PDT_MARGIN_RATE = Decimal("0.25")  # 25% margin = 4x leverage

    # PDT minimum equity requirement
    PDT_MIN_EQUITY = Decimal("25000")

    def __init__(
        self,
        tracker: SettlementTracker,
        account_type: AccountType = AccountType.MARGIN,
        margin_rate: Decimal | None = None,
    ):
        """
        Initialize buying power calculator.

        Args:
            tracker: Settlement tracker for pending settlements
            account_type: Type of account (cash, margin, pdt)
            margin_rate: Custom margin rate (default based on account type)
        """
        self.tracker = tracker
        self.account_type = account_type

        if margin_rate is not None:
            self.margin_rate = margin_rate
        elif account_type == AccountType.PDT:
            self.margin_rate = self.PDT_MARGIN_RATE
        elif account_type == AccountType.MARGIN:
            self.margin_rate = self.DEFAULT_MARGIN_RATE
        else:
            self.margin_rate = Decimal(1)  # Cash account, no margin

    def calculate(
        self,
        cash_balance: Decimal,
        portfolio_value: Decimal = Decimal(0),
        margin_used: Decimal = Decimal(0),
        currency: str = "USD",
    ) -> BuyingPower:
        """
        Calculate available buying power.

        Args:
            cash_balance: Current cash balance
            portfolio_value: Total value of securities
            margin_used: Amount of margin currently in use
            currency: Currency code

        Returns:
            BuyingPower breakdown
        """
        cash = self.tracker.get_cash_balance(cash_balance, currency)
        warnings: list[str] = []

        if self.account_type == AccountType.CASH:
            # Cash account: only settled funds
            buying_power = cash.available_cash
            day_trade_bp = Decimal(0)
            margin_available = Decimal(0)

            if cash.pending_debits > 0:
                warnings.append(
                    f"${cash.pending_debits:,.2f} pending settlement (buys)"
                )

        elif self.account_type == AccountType.MARGIN:
            # Margin account:
            # Cash + (portfolio_value * margin_rate) - margin_used
            margin_available = portfolio_value * self.margin_rate
            buying_power = (
                cash.total_cash + margin_available - margin_used - cash.pending_debits
            )
            day_trade_bp = Decimal(0)

            if margin_used > margin_available * Decimal("0.8"):
                warnings.append("Approaching margin limit")

        elif self.account_type == AccountType.PDT:
            # Pattern Day Trader: 4x intraday buying power
            equity = cash.total_cash + portfolio_value

            if equity < self.PDT_MIN_EQUITY:
                warnings.append(
                    f"Below PDT minimum equity (${self.PDT_MIN_EQUITY:,.0f})"
                )
                # Fall back to margin rules
                margin_available = portfolio_value * self.DEFAULT_MARGIN_RATE
                buying_power = cash.total_cash + margin_available - margin_used
                day_trade_bp = buying_power
            else:
                # 4x intraday buying power
                margin_available = portfolio_value * Decimal("3")  # 4x total
                buying_power = cash.total_cash + margin_available - margin_used
                day_trade_bp = equity * 4 - margin_used

        else:
            # Unknown account type, use cash rules
            buying_power = cash.available_cash
            day_trade_bp = Decimal(0)
            margin_available = Decimal(0)

        return BuyingPower(
            account_type=self.account_type,
            currency=currency,
            total_cash=cash.total_cash,
            settled_cash=cash.settled_cash,
            buying_power=max(Decimal(0), buying_power),
            margin_available=margin_available,
            margin_used=margin_used,
            day_trade_buying_power=max(Decimal(0), day_trade_bp),
            warnings=tuple(warnings),
        )

    def can_buy(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        current_cash: Decimal,
        portfolio_value: Decimal = Decimal(0),
        margin_used: Decimal = Decimal(0),
    ) -> tuple[bool, str]:
        """
        Check if a buy order is allowed given buying power.

        Args:
            symbol: Symbol to buy
            quantity: Number of shares
            price: Price per share
            current_cash: Current cash balance
            portfolio_value: Portfolio value (for margin)
            margin_used: Current margin usage

        Returns:
            (allowed, reason) tuple
        """
        order_cost = quantity * price
        bp = self.calculate(current_cash, portfolio_value, margin_used)

        if order_cost > bp.buying_power:
            return False, (
                f"Insufficient buying power. "
                f"Need ${order_cost:,.2f}, have ${bp.buying_power:,.2f}"
            )

        return True, "Order allowed"

    def can_sell(
        self,
        symbol: str,
        quantity: Decimal,
        current_position: Decimal,
    ) -> tuple[bool, str]:
        """
        Check if a sell order is allowed given settlement constraints.

        For cash accounts, checks for potential Good Faith Violations.

        Args:
            symbol: Symbol to sell
            quantity: Number of shares to sell
            current_position: Current position size

        Returns:
            (allowed, reason) tuple
        """
        if quantity > current_position:
            return False, (
                f"Cannot sell {quantity} shares, only {current_position} held"
            )

        if self.account_type == AccountType.CASH:
            # Check for potential GFV
            position = self.tracker.get_settled_position(symbol, current_position)

            if quantity > position.settled_quantity:
                return False, (
                    f"Selling {quantity} shares but only {position.settled_quantity} "
                    f"settled. Would cause Good Faith Violation."
                )

        return True, "Order allowed"

    def get_max_shares(
        self,
        price: Decimal,
        current_cash: Decimal,
        portfolio_value: Decimal = Decimal(0),
        margin_used: Decimal = Decimal(0),
    ) -> int:
        """
        Calculate maximum shares that can be purchased.

        Args:
            price: Price per share
            current_cash: Current cash balance
            portfolio_value: Portfolio value (for margin)
            margin_used: Current margin usage

        Returns:
            Maximum number of whole shares
        """
        bp = self.calculate(current_cash, portfolio_value, margin_used)

        if price <= 0:
            return 0

        return int(bp.buying_power / price)

    def estimate_margin_impact(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        current_margin_used: Decimal,
        portfolio_value: Decimal,
    ) -> dict[str, Decimal]:
        """
        Estimate the impact of a trade on margin.

        Args:
            symbol: Symbol to trade
            side: "buy" or "sell"
            quantity: Number of shares
            price: Price per share
            current_margin_used: Current margin usage
            portfolio_value: Current portfolio value

        Returns:
            Dict with margin impact details
        """
        trade_value = quantity * price

        if side.lower() == "buy":
            # Buying increases margin usage
            margin_requirement = trade_value * self.margin_rate
            new_margin_used = current_margin_used + margin_requirement
            new_portfolio = portfolio_value + trade_value
        else:
            # Selling decreases margin usage
            margin_freed = trade_value * self.margin_rate
            new_margin_used = max(Decimal(0), current_margin_used - margin_freed)
            new_portfolio = portfolio_value - trade_value

        margin_available = new_portfolio * self.margin_rate

        return {
            "trade_value": trade_value,
            "margin_requirement": trade_value * self.margin_rate,
            "new_margin_used": new_margin_used,
            "margin_available": margin_available,
            "margin_utilization": (
                new_margin_used / margin_available * 100
                if margin_available > 0
                else Decimal(0)
            ),
        }
