"""
Interactive Brokers Contract Builders.

Converts libra instruments to IB contract objects.

Issue #64: Interactive Brokers Gateway - Full Options Lifecycle
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from libra.core.options import OptionContract, StrategyLeg


def _get_ib_async():
    """Lazy import ib_async to avoid import errors when not installed."""
    try:
        import ib_async

        return ib_async
    except ImportError as e:
        raise ImportError(
            "ib_async is not installed. Install with: pip install ib_async"
        ) from e


def build_stock(
    symbol: str,
    exchange: str = "SMART",
    currency: str = "USD",
) -> Any:
    """
    Build IB Stock contract.

    Args:
        symbol: Stock ticker (e.g., "AAPL")
        exchange: Exchange (default: SMART for best routing)
        currency: Currency code (default: USD)

    Returns:
        ib_async.Stock contract

    Example:
        stock = build_stock("AAPL")
        await ib.qualifyContractsAsync(stock)
    """
    ib = _get_ib_async()
    return ib.Stock(symbol, exchange, currency)


def build_option(
    contract: OptionContract,
    exchange: str = "SMART",
    currency: str = "USD",
) -> Any:
    """
    Build IB Option contract from core OptionContract.

    Args:
        contract: Core OptionContract from libra.core.options
        exchange: Exchange (default: SMART)
        currency: Currency code

    Returns:
        ib_async.Option contract

    Example:
        from libra.core.options import OptionContract, OptionType

        core_contract = OptionContract(
            symbol="AAPL250117C00150000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("150"),
            expiration=date(2025, 1, 17),
        )
        ib_option = build_option(core_contract)
        await ib.qualifyContractsAsync(ib_option)
    """
    ib = _get_ib_async()
    return ib.Option(
        symbol=contract.underlying,
        lastTradeDateOrContractMonth=contract.expiration.strftime("%Y%m%d"),
        strike=float(contract.strike),
        right="C" if contract.is_call else "P",
        exchange=exchange,
        currency=currency,
    )


def build_option_from_params(
    underlying: str,
    expiration: str,  # Format: YYYYMMDD
    strike: float,
    right: str,  # "C" or "P"
    exchange: str = "SMART",
    currency: str = "USD",
) -> Any:
    """
    Build IB Option contract from raw parameters.

    Args:
        underlying: Underlying ticker
        expiration: Expiration date as YYYYMMDD string
        strike: Strike price
        right: "C" for call, "P" for put
        exchange: Exchange
        currency: Currency

    Returns:
        ib_async.Option contract
    """
    ib = _get_ib_async()
    return ib.Option(
        symbol=underlying,
        lastTradeDateOrContractMonth=expiration,
        strike=strike,
        right=right,
        exchange=exchange,
        currency=currency,
    )


def build_forex(
    pair: str,
    exchange: str = "IDEALPRO",
) -> Any:
    """
    Build IB Forex contract.

    Args:
        pair: Currency pair (e.g., "EURUSD")
        exchange: Exchange (default: IDEALPRO)

    Returns:
        ib_async.Forex contract
    """
    ib = _get_ib_async()
    return ib.Forex(pair, exchange)


def build_future(
    symbol: str,
    expiration: str,  # Format: YYYYMM
    exchange: str = "CME",
    currency: str = "USD",
) -> Any:
    """
    Build IB Future contract.

    Args:
        symbol: Future ticker (e.g., "ES")
        expiration: Contract month as YYYYMM
        exchange: Exchange (e.g., CME, CBOT)
        currency: Currency

    Returns:
        ib_async.Future contract
    """
    ib = _get_ib_async()
    return ib.Future(
        symbol=symbol,
        lastTradeDateOrContractMonth=expiration,
        exchange=exchange,
        currency=currency,
    )


def build_combo(
    underlying: str,
    legs: list[StrategyLeg],
    exchange: str = "SMART",
    currency: str = "USD",
) -> Any:
    """
    Build IB Combo (BAG) contract for multi-leg orders.

    This requires qualifying each leg first to get contract IDs.

    Args:
        underlying: Underlying symbol
        legs: List of StrategyLeg from libra.core.options.strategies
        exchange: Exchange
        currency: Currency

    Returns:
        ib_async.Contract configured as BAG

    Note:
        Legs must be qualified before use:
        ```python
        for leg in legs:
            option = build_option(leg.contract)
            await ib.qualifyContractsAsync(option)
            leg_conId = option.conId
        ```
    """
    ib = _get_ib_async()

    combo = ib.Contract()
    combo.symbol = underlying
    combo.secType = "BAG"
    combo.currency = currency
    combo.exchange = exchange

    combo_legs = []
    for leg in legs:
        combo_leg = ib.ComboLeg()
        # Note: conId must be set after qualifying the option contract
        combo_leg.ratio = abs(leg.quantity)
        combo_leg.action = "BUY" if leg.quantity > 0 else "SELL"
        combo_leg.exchange = exchange
        combo_legs.append(combo_leg)

    combo.comboLegs = combo_legs
    return combo


def parse_symbol(symbol: str) -> tuple[str, str]:
    """
    Parse libra symbol format to determine asset class.

    Args:
        symbol: Symbol in various formats
            - "AAPL" -> stock
            - "AAPL250117C00150000" -> option (OCC format)
            - "BTC/USD" -> crypto (if ever supported)

    Returns:
        Tuple of (asset_class, underlying)
        asset_class: "stock", "option", "crypto"
    """
    # OCC format: 21 chars, 6 for root, 6 for date, 1 for type, 8 for strike
    if len(symbol) == 21 and symbol[6:12].isdigit():
        # Likely OCC option symbol
        underlying = symbol[:6].rstrip()
        return "option", underlying

    # Simple stock
    if "/" not in symbol:
        return "stock", symbol

    # Pair format (e.g., BTC/USD)
    return "crypto", symbol.split("/")[0]
