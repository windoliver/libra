"""
Option Exercise and Assignment Models.

Provides data structures for handling option exercise and assignment events:
- Exercise types (voluntary, automatic, assigned)
- Expiration actions
- Exercise events with settlement details

Issue #63: Options Data Models
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum

import msgspec

from libra.core.options.models import OptionContract


# =============================================================================
# Enums
# =============================================================================


class ExerciseType(str, Enum):
    """
    Type of exercise event.

    - VOLUNTARY: Option holder chose to exercise early
    - AUTOMATIC: Option auto-exercised at expiry (ITM by >= $0.01)
    - ASSIGNED: Option writer was assigned (counterparty to exercise)
    """

    VOLUNTARY = "voluntary"  # Holder chose to exercise
    AUTOMATIC = "automatic"  # Auto-exercised at expiry (ITM > $0.01)
    ASSIGNED = "assigned"  # Writer was assigned


class ExpirationAction(str, Enum):
    """
    Action to take at option expiration.

    - EXERCISE: Exercise the option (take shares)
    - EXPIRE: Let it expire worthless
    - CLOSE: Close position before expiry
    - DO_NOT_EXERCISE: Explicitly don't exercise ITM option
    """

    EXERCISE = "exercise"  # Exercise the option
    EXPIRE = "expire"  # Let it expire worthless
    CLOSE = "close"  # Close position before expiry
    DO_NOT_EXERCISE = "do_not_exercise"  # Explicitly don't exercise ITM


# =============================================================================
# Exercise Event
# =============================================================================


class ExerciseEvent(msgspec.Struct, frozen=True, gc=False, kw_only=True):
    """
    Option exercise or assignment event.

    Represents the exercise of an option, resulting in
    delivery/receipt of underlying shares.

    Attributes:
        event_id: Unique identifier for this event
        contract: The option contract being exercised
        exercise_type: Type of exercise (voluntary, automatic, assigned)
        quantity: Number of contracts exercised (always positive)
        shares: Number of shares involved (quantity * multiplier)
        stock_price: Exercise price (strike price)
        cash_amount: Cash exchanged (shares * strike)
        settlement_date: Settlement date (T+1 for stocks)
        event_timestamp_ns: When exercise occurred (nanoseconds)
        notification_timestamp_ns: When notification was received

    Examples:
        # Long call exercise - buying shares
        event = ExerciseEvent(
            event_id="EX-001",
            contract=call_contract,
            exercise_type=ExerciseType.VOLUNTARY,
            quantity=10,
            shares=1000,  # 10 contracts * 100 shares
            stock_price=Decimal("150.00"),  # Strike price
            cash_amount=Decimal("150000.00"),  # 1000 * 150
            settlement_date=date(2025, 1, 20),
            event_timestamp_ns=time.time_ns(),
        )

        # Check stock direction
        if event.stock_direction > 0:
            print("Receiving shares")
        else:
            print("Delivering shares")
    """

    event_id: str
    contract: OptionContract
    exercise_type: ExerciseType
    quantity: int  # Contracts exercised (always positive)

    # Resulting stock transaction
    shares: int  # Shares delivered (quantity * multiplier)
    stock_price: Decimal  # Exercise price (strike)
    cash_amount: Decimal  # Cash exchanged (shares * strike)

    # Settlement
    settlement_date: date  # T+1 for stocks

    # Timestamps
    event_timestamp_ns: int
    notification_timestamp_ns: int | None = None

    @property
    def is_exercise(self) -> bool:
        """
        True if option holder exercised (long position).

        Exercise occurs when holder wants to take the underlying position.
        """
        return self.exercise_type in (ExerciseType.VOLUNTARY, ExerciseType.AUTOMATIC)

    @property
    def is_assignment(self) -> bool:
        """
        True if option writer was assigned (short position).

        Assignment is the counterparty event when someone exercises.
        """
        return self.exercise_type == ExerciseType.ASSIGNED

    @property
    def stock_direction(self) -> int:
        """
        Direction of stock movement: +1 for receive shares, -1 for deliver.

        Long call exercise: +1 (buy shares at strike)
        Short call assignment: -1 (sell shares at strike)
        Long put exercise: -1 (sell shares at strike)
        Short put assignment: +1 (buy shares at strike)
        """
        if self.contract.is_call:
            return 1 if self.is_exercise else -1
        else:  # Put
            return -1 if self.is_exercise else 1

    @property
    def cash_direction(self) -> int:
        """
        Direction of cash flow: +1 for receive cash, -1 for pay cash.

        Opposite of stock_direction:
        - Receive shares = pay cash
        - Deliver shares = receive cash
        """
        return -self.stock_direction

    @property
    def is_early_exercise(self) -> bool:
        """
        True if this was an early exercise (before expiration).

        Only possible for American-style options.
        """
        return self.exercise_type == ExerciseType.VOLUNTARY


# =============================================================================
# Helper Functions
# =============================================================================


def calculate_exercise_shares(quantity: int, multiplier: int = 100) -> int:
    """
    Calculate number of shares involved in exercise.

    Args:
        quantity: Number of contracts
        multiplier: Contract multiplier (default 100)

    Returns:
        Total shares (quantity * multiplier)
    """
    return quantity * multiplier


def calculate_exercise_cash(
    shares: int, strike: Decimal
) -> Decimal:
    """
    Calculate cash amount for exercise.

    Args:
        shares: Number of shares
        strike: Strike price

    Returns:
        Total cash amount (shares * strike)
    """
    return Decimal(shares) * strike


def create_exercise_event(
    event_id: str,
    contract: OptionContract,
    exercise_type: ExerciseType,
    quantity: int,
    settlement_date: date,
    event_timestamp_ns: int,
) -> ExerciseEvent:
    """
    Create an exercise event with calculated fields.

    Convenience function that calculates shares and cash amount.

    Args:
        event_id: Unique identifier
        contract: Option contract
        exercise_type: Type of exercise
        quantity: Number of contracts
        settlement_date: Settlement date
        event_timestamp_ns: Event timestamp

    Returns:
        ExerciseEvent with all fields populated
    """
    shares = calculate_exercise_shares(quantity, contract.multiplier)
    cash_amount = calculate_exercise_cash(shares, contract.strike)

    return ExerciseEvent(
        event_id=event_id,
        contract=contract,
        exercise_type=exercise_type,
        quantity=quantity,
        shares=shares,
        stock_price=contract.strike,
        cash_amount=cash_amount,
        settlement_date=settlement_date,
        event_timestamp_ns=event_timestamp_ns,
    )
