"""
Interactive Brokers Order Type Mappings.

Converts libra Order objects to IB order types.

Issue #64: Interactive Brokers Gateway - Full Options Lifecycle
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from libra.gateways.protocol import Order, OrderSide, OrderStatus, OrderType, TimeInForce


if TYPE_CHECKING:
    pass


def _get_ib_async():
    """Lazy import ib_async."""
    try:
        import ib_async

        return ib_async
    except ImportError as e:
        raise ImportError(
            "ib_async is not installed. Install with: pip install ib_async"
        ) from e


def build_ib_order(order: Order) -> Any:
    """
    Convert libra Order to IB order object.

    Args:
        order: Libra Order object

    Returns:
        ib_async order object (MarketOrder, LimitOrder, etc.)

    Raises:
        ValueError: If order type is not supported
    """
    ib = _get_ib_async()

    action = "BUY" if order.side == OrderSide.BUY else "SELL"
    quantity = float(order.amount)

    if order.order_type == OrderType.MARKET:
        return ib.MarketOrder(action, quantity)

    elif order.order_type == OrderType.LIMIT:
        if order.price is None:
            raise ValueError("Limit order requires price")
        return ib.LimitOrder(action, quantity, float(order.price))

    elif order.order_type == OrderType.STOP:
        if order.stop_price is None:
            raise ValueError("Stop order requires stop_price")
        return ib.StopOrder(action, quantity, float(order.stop_price))

    elif order.order_type == OrderType.STOP_LIMIT:
        if order.stop_price is None or order.price is None:
            raise ValueError("Stop-limit order requires both stop_price and price")
        return ib.StopLimitOrder(
            action, quantity, float(order.price), float(order.stop_price)
        )

    else:
        raise ValueError(f"Unsupported order type: {order.order_type}")


def apply_time_in_force(ib_order: Any, tif: TimeInForce) -> None:
    """
    Apply time-in-force to IB order (mutates order).

    Args:
        ib_order: IB order object to modify
        tif: Time-in-force setting
    """
    tif_map = {
        TimeInForce.GTC: "GTC",
        TimeInForce.DAY: "DAY",
        TimeInForce.IOC: "IOC",
        TimeInForce.FOK: "FOK",
        TimeInForce.GTD: "GTD",
    }
    ib_order.tif = tif_map.get(tif, "GTC")


def apply_extended_hours(ib_order: Any, extended: bool) -> None:
    """
    Enable extended hours trading on IB order (mutates order).

    Args:
        ib_order: IB order object
        extended: Whether to allow extended hours
    """
    if extended:
        ib_order.outsideRth = True


def map_ib_status(ib_status: str) -> OrderStatus:
    """
    Map IB order status to libra OrderStatus.

    Args:
        ib_status: IB status string

    Returns:
        OrderStatus enum value
    """
    status_map = {
        # Active states
        "PendingSubmit": OrderStatus.PENDING,
        "PendingCancel": OrderStatus.PENDING,
        "PreSubmitted": OrderStatus.PENDING,
        "Submitted": OrderStatus.OPEN,
        # Filled states
        "Filled": OrderStatus.FILLED,
        "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
        # Terminal states
        "Cancelled": OrderStatus.CANCELLED,
        "ApiCancelled": OrderStatus.CANCELLED,
        "Inactive": OrderStatus.REJECTED,
        "ApiPending": OrderStatus.PENDING,
    }
    return status_map.get(ib_status, OrderStatus.PENDING)


def build_bracket_order(
    parent_order: Order,
    take_profit_price: Decimal,
    stop_loss_price: Decimal,
) -> tuple[Any, Any, Any]:
    """
    Build IB bracket order (entry + take profit + stop loss).

    Args:
        parent_order: Entry order
        take_profit_price: Take profit limit price
        stop_loss_price: Stop loss trigger price

    Returns:
        Tuple of (parent, take_profit, stop_loss) IB orders
    """
    ib = _get_ib_async()

    action = "BUY" if parent_order.side == OrderSide.BUY else "SELL"
    quantity = float(parent_order.amount)
    exit_action = "SELL" if action == "BUY" else "BUY"

    # Parent order
    if parent_order.order_type == OrderType.MARKET:
        parent = ib.MarketOrder(action, quantity)
    elif parent_order.order_type == OrderType.LIMIT:
        parent = ib.LimitOrder(action, quantity, float(parent_order.price))
    else:
        raise ValueError(f"Bracket parent must be market or limit, got {parent_order.order_type}")

    parent.transmit = False  # Don't transmit until all orders ready

    # Take profit (limit order)
    take_profit = ib.LimitOrder(exit_action, quantity, float(take_profit_price))
    take_profit.parentId = parent.orderId
    take_profit.transmit = False

    # Stop loss (stop order)
    stop_loss = ib.StopOrder(exit_action, quantity, float(stop_loss_price))
    stop_loss.parentId = parent.orderId
    stop_loss.transmit = True  # Transmit all orders

    return parent, take_profit, stop_loss


def build_oco_orders(
    order1: Order,
    order2: Order,
    oca_group: str,
) -> tuple[Any, Any]:
    """
    Build OCO (One-Cancels-Other) order pair.

    When one order fills, the other is cancelled.

    Args:
        order1: First order
        order2: Second order
        oca_group: Unique identifier for the OCA group

    Returns:
        Tuple of two IB orders linked via OCA
    """
    ib_order1 = build_ib_order(order1)
    ib_order2 = build_ib_order(order2)

    # Link orders via OCA group
    ib_order1.ocaGroup = oca_group
    ib_order1.ocaType = 1  # Cancel remaining orders

    ib_order2.ocaGroup = oca_group
    ib_order2.ocaType = 1

    return ib_order1, ib_order2


def build_trailing_stop(
    order: Order,
    trailing_amount: Decimal | None = None,
    trailing_percent: Decimal | None = None,
) -> Any:
    """
    Build trailing stop order.

    Args:
        order: Base order (side and quantity)
        trailing_amount: Trail by fixed amount
        trailing_percent: Trail by percentage

    Returns:
        IB trailing stop order
    """
    ib = _get_ib_async()

    action = "BUY" if order.side == OrderSide.BUY else "SELL"
    quantity = float(order.amount)

    if trailing_amount is not None:
        # Trailing by fixed amount
        trail_order = ib.Order()
        trail_order.action = action
        trail_order.orderType = "TRAIL"
        trail_order.totalQuantity = quantity
        trail_order.auxPrice = float(trailing_amount)
        return trail_order
    elif trailing_percent is not None:
        # Trailing by percentage
        trail_order = ib.Order()
        trail_order.action = action
        trail_order.orderType = "TRAIL"
        trail_order.totalQuantity = quantity
        trail_order.trailingPercent = float(trailing_percent)
        return trail_order
    else:
        raise ValueError("Must specify either trailing_amount or trailing_percent")
