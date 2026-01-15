"""
T+1 Settlement Handler for Stock and Options Trading.

This module provides comprehensive settlement tracking for US equities
and options, which moved to T+1 settlement on May 28, 2024.

Key Features:
- Track pending settlements (buy/sell trades)
- Calculate settlement-aware buying power
- Prevent Good Faith Violations (cash accounts)
- Handle options exercise/assignment settlements
- Support cash, margin, and PDT account types

Example:
    >>> from libra.core.settlement import (
    ...     SettlementTracker,
    ...     BuyingPowerCalculator,
    ...     AccountType,
    ... )
    >>> # Create tracker for a margin account
    >>> tracker = SettlementTracker(account_type=AccountType.MARGIN)
    >>> # Add a buy trade
    >>> settlement = tracker.add_trade(
    ...     trade_id="T001",
    ...     symbol="AAPL",
    ...     side="buy",
    ...     quantity=Decimal("100"),
    ...     price=Decimal("150.00"),
    ... )
    >>> print(f"Settles on: {settlement.settlement_date}")
    >>> # Calculate buying power
    >>> calc = BuyingPowerCalculator(tracker)
    >>> bp = calc.calculate(cash_balance=Decimal("50000"))
    >>> print(f"Buying power: ${bp.buying_power:,.2f}")

Settlement Rules:
- US Stocks: T+1 (next business day)
- US Options: T+1 (same-day for expiry assignment)
- Crypto: T+0 (instant settlement)

Account Types:
- Cash: Can only use settled funds (GFV risk)
- Margin: Can trade with unsettled funds + margin
- PDT: Pattern Day Trader, 4x intraday buying power
"""

from libra.core.settlement.models import (
    # Enums
    AccountType,
    SecurityType,
    SettlementStatus,
    SettlementType,
    # Data models
    BuyingPower,
    CashBalance,
    GFVStatus,
    GoodFaithViolation,
    PendingSettlement,
    SettledPosition,
    SettlementEvent,
    # Constants
    SETTLEMENT_DAYS,
)
from libra.core.settlement.calendar import (
    SettlementCalendar,
    get_settlement_calendar,
    get_settlement_date,
)
from libra.core.settlement.tracker import SettlementTracker
from libra.core.settlement.calculator import BuyingPowerCalculator
from libra.core.settlement.violations import GFVTracker, GFVCheckResult


__all__ = [
    # Enums
    "AccountType",
    "SecurityType",
    "SettlementStatus",
    "SettlementType",
    # Data models
    "BuyingPower",
    "CashBalance",
    "GFVStatus",
    "GoodFaithViolation",
    "PendingSettlement",
    "SettledPosition",
    "SettlementEvent",
    # Constants
    "SETTLEMENT_DAYS",
    # Calendar
    "SettlementCalendar",
    "get_settlement_calendar",
    "get_settlement_date",
    # Core classes
    "SettlementTracker",
    "BuyingPowerCalculator",
    "GFVTracker",
    "GFVCheckResult",
]
