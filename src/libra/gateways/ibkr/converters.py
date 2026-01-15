"""
Interactive Brokers to Core Model Converters.

Converts IB API responses to libra core models.

Issue #64: Interactive Brokers Gateway - Full Options Lifecycle
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from libra.gateways.protocol import (
    Balance,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
    PositionSide,
    Tick,
)

from libra.gateways.ibkr.orders import map_ib_status


if TYPE_CHECKING:
    pass


def ticker_to_tick(ticker: Any, symbol: str) -> Tick:
    """
    Convert IB Ticker to libra Tick.

    Args:
        ticker: ib_async.Ticker object
        symbol: Symbol string

    Returns:
        Libra Tick object
    """
    return Tick(
        symbol=symbol,
        bid=Decimal(str(ticker.bid)) if ticker.bid is not None else Decimal("0"),
        ask=Decimal(str(ticker.ask)) if ticker.ask is not None else Decimal("0"),
        last=Decimal(str(ticker.last)) if ticker.last is not None else Decimal("0"),
        timestamp_ns=time.time_ns(),
        bid_size=Decimal(str(ticker.bidSize)) if ticker.bidSize else None,
        ask_size=Decimal(str(ticker.askSize)) if ticker.askSize else None,
        last_size=Decimal(str(ticker.lastSize)) if ticker.lastSize else None,
        volume_24h=Decimal(str(ticker.volume)) if ticker.volume else None,
        high_24h=Decimal(str(ticker.high)) if ticker.high else None,
        low_24h=Decimal(str(ticker.low)) if ticker.low else None,
        open_24h=Decimal(str(ticker.open)) if ticker.open else None,
    )


def ticker_to_greeks(ticker: Any) -> "Greeks":
    """
    Convert IB Ticker.modelGreeks to libra Greeks.

    Args:
        ticker: ib_async.Ticker with modelGreeks populated

    Returns:
        Libra Greeks object

    Raises:
        ValueError: If modelGreeks is not available
    """
    from libra.core.options import Greeks

    if ticker.modelGreeks is None:
        raise ValueError("Ticker does not have modelGreeks populated")

    greeks = ticker.modelGreeks
    return Greeks(
        delta=Decimal(str(greeks.delta)) if greeks.delta is not None else Decimal("0"),
        gamma=Decimal(str(greeks.gamma)) if greeks.gamma is not None else Decimal("0"),
        theta=Decimal(str(greeks.theta)) if greeks.theta is not None else Decimal("0"),
        vega=Decimal(str(greeks.vega)) if greeks.vega is not None else Decimal("0"),
        rho=Decimal(str(greeks.rho)) if greeks.rho is not None else Decimal("0"),
        iv=Decimal(str(greeks.impliedVol)) if greeks.impliedVol is not None else Decimal("0"),
        timestamp_ns=time.time_ns(),
    )


def ib_position_to_position(ib_pos: Any) -> Position:
    """
    Convert IB Position to libra Position.

    Args:
        ib_pos: ib_async position object from ib.positions()

    Returns:
        Libra Position object
    """
    # IB position has: account, contract, position (qty), avgCost
    quantity = Decimal(str(ib_pos.position))
    avg_cost = Decimal(str(ib_pos.avgCost))

    # Determine side
    if quantity > 0:
        side = PositionSide.LONG
    elif quantity < 0:
        side = PositionSide.SHORT
    else:
        side = PositionSide.FLAT

    # Build symbol from contract
    contract = ib_pos.contract
    if hasattr(contract, "localSymbol") and contract.localSymbol:
        symbol = contract.localSymbol
    else:
        symbol = contract.symbol

    return Position(
        symbol=symbol,
        side=side,
        amount=abs(quantity),
        entry_price=avg_cost,
        current_price=avg_cost,  # Will be updated with market data
        unrealized_pnl=Decimal("0"),  # Will be updated
        realized_pnl=Decimal("0"),
        timestamp_ns=time.time_ns(),
    )


def ib_account_value_to_balance(
    account_values: list[Any],
    currency: str = "USD",
) -> Balance:
    """
    Convert IB account values to libra Balance.

    Args:
        account_values: List of AccountValue from ib.accountValues()
        currency: Currency to extract (default: USD)

    Returns:
        Libra Balance object
    """
    total = Decimal("0")
    available = Decimal("0")

    for av in account_values:
        if av.currency != currency:
            continue

        if av.tag == "TotalCashValue":
            total = Decimal(str(av.value))
        elif av.tag == "AvailableFunds":
            available = Decimal(str(av.value))
        elif av.tag == "BuyingPower":
            # Alternative if AvailableFunds not present
            if available == 0:
                available = Decimal(str(av.value))

    locked = total - available if total > available else Decimal("0")

    return Balance(
        currency=currency,
        total=total,
        available=available,
        locked=locked,
    )


def trade_to_order_result(trade: Any) -> OrderResult:
    """
    Convert IB Trade to libra OrderResult.

    Args:
        trade: ib_async.Trade object

    Returns:
        Libra OrderResult object
    """
    order = trade.order
    status = trade.orderStatus
    contract = trade.contract

    # Determine order type
    order_type_map = {
        "MKT": OrderType.MARKET,
        "LMT": OrderType.LIMIT,
        "STP": OrderType.STOP,
        "STP LMT": OrderType.STOP_LIMIT,
    }
    order_type = order_type_map.get(order.orderType, OrderType.MARKET)

    # Determine side
    side = OrderSide.BUY if order.action == "BUY" else OrderSide.SELL

    # Build symbol
    if hasattr(contract, "localSymbol") and contract.localSymbol:
        symbol = contract.localSymbol
    else:
        symbol = contract.symbol

    # Calculate amounts
    total_qty = Decimal(str(order.totalQuantity))
    filled_qty = Decimal(str(status.filled)) if status.filled else Decimal("0")
    remaining = Decimal(str(status.remaining)) if status.remaining else total_qty - filled_qty

    # Average fill price
    avg_price = (
        Decimal(str(status.avgFillPrice))
        if status.avgFillPrice and status.avgFillPrice > 0
        else None
    )

    # Fees from commission reports
    total_fee = Decimal("0")
    for fill in trade.fills:
        if fill.commissionReport:
            total_fee += Decimal(str(fill.commissionReport.commission))

    return OrderResult(
        order_id=str(order.orderId),
        symbol=symbol,
        status=map_ib_status(status.status),
        side=side,
        order_type=order_type,
        amount=total_qty,
        filled_amount=filled_qty,
        remaining_amount=remaining,
        average_price=avg_price,
        fee=total_fee,
        fee_currency="USD",  # IB reports in base currency
        timestamp_ns=time.time_ns(),
        client_order_id=order.clientId if hasattr(order, "clientId") else None,
        price=Decimal(str(order.lmtPrice)) if hasattr(order, "lmtPrice") and order.lmtPrice else None,
        stop_price=Decimal(str(order.auxPrice)) if hasattr(order, "auxPrice") and order.auxPrice else None,
    )


def option_chain_params_to_dict(chain_params: Any) -> dict:
    """
    Convert IB option chain parameters to dict.

    Args:
        chain_params: Result from reqSecDefOptParamsAsync

    Returns:
        Dict with expirations and strikes
    """
    if not chain_params:
        return {"expirations": [], "strikes": []}

    # chain_params is a list of OptionChain objects
    all_expirations = set()
    all_strikes = set()

    for chain in chain_params:
        if hasattr(chain, "expirations"):
            all_expirations.update(chain.expirations)
        if hasattr(chain, "strikes"):
            all_strikes.update(chain.strikes)

    return {
        "expirations": sorted(all_expirations),
        "strikes": sorted(all_strikes),
    }
