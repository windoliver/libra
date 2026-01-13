"""
Alpaca Symbol Utilities.

Handles symbol format conversions:
- OCC option symbol format (AAPL250117C00150000)
- Stock symbols normalization
- Symbol validation

Issue #61: Alpaca Gateway - Stock & Options Execution
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime


# OCC symbol regex: UNDERLYING(1-6 chars) + YYMMDD + C/P + 8-digit strike
OCC_PATTERN = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


@dataclass(frozen=True)
class OptionSymbolComponents:
    """Parsed components of an OCC option symbol."""

    underlying: str
    expiration: date
    option_type: str  # "call" or "put"
    strike: float

    def to_occ(self) -> str:
        """Convert back to OCC symbol format."""
        return to_occ_symbol(
            underlying=self.underlying,
            expiration=self.expiration,
            option_type=self.option_type,
            strike=self.strike,
        )


def to_occ_symbol(
    underlying: str,
    expiration: date,
    option_type: str,
    strike: float,
) -> str:
    """
    Convert option parameters to OCC symbol format.

    The OCC (Options Clearing Corporation) symbol format is:
    - Underlying symbol (1-6 characters, padded with spaces if needed)
    - Expiration date (YYMMDD)
    - Option type (C for call, P for put)
    - Strike price (8 digits, multiplied by 1000)

    Args:
        underlying: Stock ticker (e.g., "AAPL")
        expiration: Option expiration date
        option_type: "call", "put", "C", or "P"
        strike: Strike price (e.g., 150.00)

    Returns:
        OCC symbol string (e.g., "AAPL250117C00150000")

    Example:
        >>> to_occ_symbol("AAPL", date(2025, 1, 17), "call", 150.00)
        'AAPL250117C00150000'
    """
    # Normalize underlying
    underlying = underlying.upper().strip()
    if not underlying or len(underlying) > 6:
        raise ValueError(f"Invalid underlying symbol: {underlying}")

    # Format expiration
    exp_str = expiration.strftime("%y%m%d")

    # Normalize option type
    option_type_upper = option_type.upper()
    if option_type_upper in ("CALL", "C"):
        type_char = "C"
    elif option_type_upper in ("PUT", "P"):
        type_char = "P"
    else:
        raise ValueError(f"Invalid option type: {option_type}")

    # Format strike (multiply by 1000, 8 digits)
    if strike <= 0:
        raise ValueError(f"Strike must be positive: {strike}")
    strike_int = int(round(strike * 1000))
    strike_str = f"{strike_int:08d}"

    return f"{underlying}{exp_str}{type_char}{strike_str}"


def from_occ_symbol(occ_symbol: str) -> OptionSymbolComponents:
    """
    Parse OCC symbol back to components.

    Args:
        occ_symbol: OCC format symbol (e.g., "AAPL250117C00150000")

    Returns:
        OptionSymbolComponents with underlying, expiration, type, strike

    Raises:
        ValueError: If symbol format is invalid

    Example:
        >>> result = from_occ_symbol("AAPL250117C00150000")
        >>> result.underlying
        'AAPL'
        >>> result.strike
        150.0
    """
    occ_symbol = occ_symbol.upper().strip()

    match = OCC_PATTERN.match(occ_symbol)
    if not match:
        raise ValueError(f"Invalid OCC symbol format: {occ_symbol}")

    underlying, exp_str, type_char, strike_str = match.groups()

    # Parse expiration
    try:
        expiration = datetime.strptime(exp_str, "%y%m%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid expiration date in OCC symbol: {exp_str}") from e

    # Parse option type
    option_type = "call" if type_char == "C" else "put"

    # Parse strike (divide by 1000)
    strike = int(strike_str) / 1000.0

    return OptionSymbolComponents(
        underlying=underlying,
        expiration=expiration,
        option_type=option_type,
        strike=strike,
    )


def is_option_symbol(symbol: str) -> bool:
    """
    Check if a symbol is in OCC option format.

    Args:
        symbol: Symbol to check

    Returns:
        True if symbol matches OCC format, False otherwise

    Example:
        >>> is_option_symbol("AAPL250117C00150000")
        True
        >>> is_option_symbol("AAPL")
        False
    """
    return bool(OCC_PATTERN.match(symbol.upper().strip()))


def normalize_symbol(symbol: str) -> str:
    """
    Normalize a stock symbol for Alpaca.

    - Uppercase
    - Strip whitespace
    - Remove common suffixes

    Args:
        symbol: Stock symbol (e.g., "aapl", "AAPL.US")

    Returns:
        Normalized symbol (e.g., "AAPL")

    Example:
        >>> normalize_symbol("aapl")
        'AAPL'
        >>> normalize_symbol("BRK.B")
        'BRK.B'
    """
    symbol = symbol.upper().strip()

    # Remove common exchange suffixes
    for suffix in (".US", ".NYSE", ".NASDAQ", ".AMEX"):
        if symbol.endswith(suffix):
            symbol = symbol[: -len(suffix)]
            break

    return symbol


def get_underlying(symbol: str) -> str:
    """
    Get the underlying symbol from a stock or option symbol.

    Args:
        symbol: Stock symbol or OCC option symbol

    Returns:
        Underlying stock symbol

    Example:
        >>> get_underlying("AAPL")
        'AAPL'
        >>> get_underlying("AAPL250117C00150000")
        'AAPL'
    """
    if is_option_symbol(symbol):
        return from_occ_symbol(symbol).underlying
    return normalize_symbol(symbol)


def format_option_display(occ_symbol: str) -> str:
    """
    Format OCC symbol for human-readable display.

    Args:
        occ_symbol: OCC format symbol

    Returns:
        Human-readable string (e.g., "AAPL Jan 17 '25 $150 Call")

    Example:
        >>> format_option_display("AAPL250117C00150000")
        "AAPL Jan 17 '25 $150.00 Call"
    """
    components = from_occ_symbol(occ_symbol)
    exp_str = components.expiration.strftime("%b %d '%y")
    type_str = components.option_type.title()
    return f"{components.underlying} {exp_str} ${components.strike:.2f} {type_str}"
