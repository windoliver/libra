"""
Settlement Data Models for T+1 Stock and Options Settlement.

US stocks and options moved to T+1 settlement on May 28, 2024.
This module provides models for tracking pending settlements,
calculating buying power, and avoiding Good Faith Violations.

Settlement Rules:
- US Stocks: T+1 (next business day)
- US Options: T+1 (same-day for expiry assignment)
- US Treasuries: T+1
- Crypto: T+0 (instant)
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
import msgspec


class SettlementStatus(str, Enum):
    """Settlement status of a trade."""

    PENDING = "pending"  # Trade executed, not settled
    SETTLING = "settling"  # Settlement in progress (end of day)
    SETTLED = "settled"  # Fully settled
    FAILED = "failed"  # Settlement failed


class SettlementType(str, Enum):
    """Type of settlement."""

    REGULAR = "regular"  # Standard T+1
    SAME_DAY = "same_day"  # Same-day settlement (crypto, some ETFs)
    EXERCISE = "exercise"  # Option exercise settlement
    ASSIGNMENT = "assignment"  # Option assignment settlement
    DIVIDEND = "dividend"  # Dividend payment
    CORPORATE_ACTION = "corporate_action"  # Merger, split, etc.


class SecurityType(str, Enum):
    """Security type for settlement rules."""

    STOCK = "stock"  # T+1
    OPTION = "option"  # T+1
    ETF = "etf"  # T+1
    TREASURY = "treasury"  # T+1
    MUTUAL_FUND = "mutual_fund"  # T+1 to T+2
    CRYPTO = "crypto"  # T+0


class AccountType(str, Enum):
    """Account type for settlement and buying power rules."""

    CASH = "cash"  # Can only use settled funds
    MARGIN = "margin"  # Can trade with unsettled funds
    PDT = "pdt"  # Pattern Day Trader (4x buying power)


# Settlement days by security type
SETTLEMENT_DAYS: dict[SecurityType, int] = {
    SecurityType.STOCK: 1,  # T+1
    SecurityType.OPTION: 1,  # T+1
    SecurityType.ETF: 1,  # T+1
    SecurityType.TREASURY: 1,  # T+1
    SecurityType.MUTUAL_FUND: 2,  # T+2 (varies)
    SecurityType.CRYPTO: 0,  # T+0 (instant)
}


class PendingSettlement(msgspec.Struct, kw_only=True, frozen=False):
    """
    A trade pending settlement.

    Tracks the state of a trade from execution to settlement.
    """

    settlement_id: str
    trade_id: str

    # Trade details
    symbol: str
    side: str  # "buy" or "sell"
    quantity: Decimal
    price: Decimal

    # Settlement info
    settlement_type: SettlementType
    security_type: SecurityType
    trade_date: date
    settlement_date: date
    status: SettlementStatus = SettlementStatus.PENDING

    # Cash impact (positive = receive, negative = pay)
    cash_amount: Decimal

    # For options
    is_option: bool = False
    underlying: str | None = None

    # Timestamps
    executed_at: datetime
    settled_at: datetime | None = None

    @property
    def days_to_settlement(self) -> int:
        """Days until settlement (may be negative if past due)."""
        return (self.settlement_date - date.today()).days

    @property
    def is_settled(self) -> bool:
        """Check if settlement is complete."""
        return self.status == SettlementStatus.SETTLED

    @property
    def is_pending(self) -> bool:
        """Check if settlement is still pending."""
        return self.status == SettlementStatus.PENDING

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy trade."""
        return self.side.lower() == "buy"

    @property
    def is_sell(self) -> bool:
        """Check if this is a sell trade."""
        return self.side.lower() == "sell"


class SettledPosition(msgspec.Struct, frozen=True, gc=False):
    """
    Breakdown of settled vs unsettled shares for a position.

    Used to determine shares available to sell without GFV risk.
    """

    symbol: str
    total_quantity: Decimal
    settled_quantity: Decimal
    unsettled_quantity: Decimal
    pending_buys: Decimal
    pending_sells: Decimal

    @property
    def available_to_sell(self) -> Decimal:
        """Shares available to sell without Good Faith Violation risk."""
        return self.settled_quantity

    @property
    def pct_settled(self) -> Decimal:
        """Percentage of position that is settled."""
        if self.total_quantity == 0:
            return Decimal("100")
        return (self.settled_quantity / self.total_quantity) * 100


class CashBalance(msgspec.Struct, frozen=True, gc=False):
    """
    Breakdown of cash balances by settlement status.

    Used for buying power calculations based on account type.
    """

    currency: str
    total_cash: Decimal
    settled_cash: Decimal
    unsettled_cash: Decimal
    pending_debits: Decimal  # Money going out (buys)
    pending_credits: Decimal  # Money coming in (sells)

    @property
    def available_cash(self) -> Decimal:
        """Cash available for trading in a cash account (conservative)."""
        return max(Decimal(0), self.settled_cash - self.pending_debits)

    @property
    def net_pending(self) -> Decimal:
        """Net pending cash flow (positive = incoming)."""
        return self.pending_credits - self.pending_debits


class GoodFaithViolation(msgspec.Struct, frozen=True, gc=False):
    """
    Record of a Good Faith Violation.

    A GFV occurs in cash accounts when you:
    1. Buy a security with unsettled funds
    2. Sell that security before the original funds settle

    Three GFVs in 12 months = account restricted to settled funds for 90 days.
    """

    violation_id: str
    account_id: str
    symbol: str
    description: str
    violation_date: date
    buy_trade_id: str
    sell_trade_id: str
    amount: Decimal  # Amount of the violation


class GFVStatus(msgspec.Struct, frozen=True, gc=False):
    """
    Good Faith Violation status for an account.

    Tracks violation count and restriction status.
    """

    account_id: str
    violation_count: int  # In last 12 months
    max_violations: int = 3
    is_restricted: bool = False
    restriction_end_date: date | None = None
    violations: tuple[GoodFaithViolation, ...] = ()

    @property
    def violations_remaining(self) -> int:
        """Number of violations before account restriction."""
        return max(0, self.max_violations - self.violation_count)

    @property
    def is_at_risk(self) -> bool:
        """Check if one more violation would trigger restriction."""
        return self.violations_remaining == 1


class BuyingPower(msgspec.Struct, frozen=True, gc=False):
    """
    Buying power breakdown for an account.

    Different account types have different buying power rules:
    - Cash: Only settled funds
    - Margin: Cash + margin (typically 2x)
    - PDT: Cash + 4x intraday leverage
    """

    # Required fields first
    account_type: AccountType
    currency: str
    total_cash: Decimal
    settled_cash: Decimal
    buying_power: Decimal

    # Optional fields with defaults
    margin_available: Decimal = Decimal(0)
    margin_used: Decimal = Decimal(0)
    day_trade_buying_power: Decimal = Decimal(0)  # PDT only
    warnings: tuple[str, ...] = ()

    @property
    def has_warnings(self) -> bool:
        """Check if there are any buying power warnings."""
        return len(self.warnings) > 0


class SettlementEvent(msgspec.Struct, frozen=True, gc=False):
    """
    Event emitted when a settlement status changes.

    Published to MessageBus for other components to react.
    """

    event_type: str  # "settlement_pending", "settlement_complete", etc.
    settlement_id: str
    trade_id: str
    symbol: str
    status: SettlementStatus
    timestamp: datetime

    # Optional details
    cash_amount: Decimal | None = None
    quantity: Decimal | None = None
