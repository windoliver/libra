"""
Orders Router for LIBRA API (Issue #30).

Endpoints:
- GET /orders - List orders
- POST /orders - Create order
- GET /orders/{id} - Get order details
- DELETE /orders/{id} - Cancel order
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status

from libra.api.deps import get_current_active_user, require_scope
from libra.api.schemas import (
    OrderCancelResponse,
    OrderCreate,
    OrderListResponse,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
)


router = APIRouter()

# In-memory order storage for demo
_orders: dict[str, dict[str, Any]] = {}


def _init_demo_orders() -> None:
    """Initialize demo orders if empty."""
    if _orders:
        return

    now = datetime.now(timezone.utc)

    _orders["ord_001"] = {
        "id": "ord_001",
        "client_order_id": "client_001",
        "symbol": "BTC/USDT",
        "side": OrderSide.BUY,
        "order_type": OrderType.LIMIT,
        "status": OrderStatus.FILLED,
        "quantity": Decimal("0.5"),
        "filled_quantity": Decimal("0.5"),
        "remaining_quantity": Decimal("0"),
        "price": Decimal("42500.00"),
        "average_price": Decimal("42485.00"),
        "stop_price": None,
        "time_in_force": "GTC",
        "created_at": now,
        "updated_at": now,
        "filled_at": now,
        "exec_algorithm": None,
    }

    _orders["ord_002"] = {
        "id": "ord_002",
        "client_order_id": "client_002",
        "symbol": "ETH/USDT",
        "side": OrderSide.BUY,
        "order_type": OrderType.LIMIT,
        "status": OrderStatus.ACCEPTED,
        "quantity": Decimal("5.0"),
        "filled_quantity": Decimal("0"),
        "remaining_quantity": Decimal("5.0"),
        "price": Decimal("2200.00"),
        "average_price": None,
        "stop_price": None,
        "time_in_force": "GTC",
        "created_at": now,
        "updated_at": now,
        "filled_at": None,
        "exec_algorithm": None,
    }

    _orders["ord_003"] = {
        "id": "ord_003",
        "client_order_id": None,
        "symbol": "BTC/USDT",
        "side": OrderSide.SELL,
        "order_type": OrderType.STOP_LIMIT,
        "status": OrderStatus.ACCEPTED,
        "quantity": Decimal("0.25"),
        "filled_quantity": Decimal("0"),
        "remaining_quantity": Decimal("0.25"),
        "price": Decimal("41000.00"),
        "average_price": None,
        "stop_price": Decimal("41500.00"),
        "time_in_force": "GTC",
        "created_at": now,
        "updated_at": now,
        "filled_at": None,
        "exec_algorithm": None,
    }


@router.get("", response_model=OrderListResponse)
async def list_orders(
    current_user: Annotated[dict, Depends(get_current_active_user)],
    symbol: str | None = Query(None, description="Filter by symbol"),
    status_filter: OrderStatus | None = Query(None, alias="status"),
    side: OrderSide | None = Query(None, description="Filter by order side"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> OrderListResponse:
    """
    List orders.

    Supports filtering by symbol, status, and side.
    """
    _init_demo_orders()

    orders = list(_orders.values())

    # Apply filters
    if symbol:
        symbol_upper = symbol.upper()
        orders = [o for o in orders if o["symbol"].upper() == symbol_upper]

    if status_filter:
        orders = [o for o in orders if o["status"] == status_filter]

    if side:
        orders = [o for o in orders if o["side"] == side]

    total = len(orders)
    orders = orders[offset : offset + limit]

    return OrderListResponse(
        orders=[OrderResponse(**o) for o in orders],
        total=total,
    )


@router.post("", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(
    order: OrderCreate,
    current_user: Annotated[dict, Depends(require_scope("write"))],
) -> OrderResponse:
    """
    Create a new order.

    Requires write scope.

    **Order types:**
    - market: Execute immediately at market price
    - limit: Execute at specified price or better
    - stop: Trigger market order when stop price is reached
    - stop_limit: Trigger limit order when stop price is reached

    **Execution algorithms** (optional):
    - twap: Time-weighted average price
    - vwap: Volume-weighted average price
    - iceberg: Large order split into smaller visible chunks
    """
    _init_demo_orders()

    # Validate limit orders have price
    if order.order_type == OrderType.LIMIT and order.price is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit orders require a price",
        )

    # Validate stop orders have stop_price
    if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and order.stop_price is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stop orders require a stop_price",
        )

    order_id = f"ord_{uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    order_data = {
        "id": order_id,
        "client_order_id": None,
        "symbol": order.symbol.upper(),
        "side": order.side,
        "order_type": order.order_type,
        "status": OrderStatus.SUBMITTED,
        "quantity": order.quantity,
        "filled_quantity": Decimal("0"),
        "remaining_quantity": order.quantity,
        "price": order.price,
        "average_price": None,
        "stop_price": order.stop_price,
        "time_in_force": order.time_in_force,
        "created_at": now,
        "updated_at": now,
        "filled_at": None,
        "exec_algorithm": order.exec_algorithm,
    }

    _orders[order_id] = order_data

    # Simulate quick acceptance
    order_data["status"] = OrderStatus.ACCEPTED

    return OrderResponse(**order_data)


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str,
    current_user: Annotated[dict, Depends(get_current_active_user)],
) -> OrderResponse:
    """Get order details by ID."""
    _init_demo_orders()

    if order_id not in _orders:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found",
        )

    return OrderResponse(**_orders[order_id])


@router.delete("/{order_id}", response_model=OrderCancelResponse)
async def cancel_order(
    order_id: str,
    current_user: Annotated[dict, Depends(require_scope("write"))],
) -> OrderCancelResponse:
    """
    Cancel an order.

    Requires write scope. Only open orders can be cancelled.
    """
    _init_demo_orders()

    if order_id not in _orders:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found",
        )

    order = _orders[order_id]

    # Check if order can be cancelled
    cancellable_statuses = {
        OrderStatus.PENDING,
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
    }

    if order["status"] not in cancellable_statuses:
        return OrderCancelResponse(
            id=order_id,
            cancelled=False,
            message=f"Order cannot be cancelled. Status: {order['status'].value}",
        )

    # Cancel the order
    order["status"] = OrderStatus.CANCELLED
    order["updated_at"] = datetime.now(timezone.utc)

    return OrderCancelResponse(
        id=order_id,
        cancelled=True,
        message="Order cancelled successfully",
    )
