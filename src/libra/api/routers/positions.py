"""
Positions Router for LIBRA API (Issue #30).

Endpoints:
- GET /positions - List all positions
- GET /positions/{symbol} - Get position for symbol
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from libra.api.deps import get_current_active_user
from libra.api.schemas import PositionListResponse, PositionResponse, PositionSide


router = APIRouter()

# In-memory position storage for demo
_positions: dict[str, dict[str, Any]] = {}


def _init_demo_positions() -> None:
    """Initialize demo positions if empty."""
    if _positions:
        return

    now = datetime.now(timezone.utc)

    _positions["BTC/USDT"] = {
        "symbol": "BTC/USDT",
        "side": PositionSide.LONG,
        "size": Decimal("0.5"),
        "entry_price": Decimal("42500.00"),
        "current_price": Decimal("43250.00"),
        "unrealized_pnl": Decimal("375.00"),
        "unrealized_pnl_pct": 1.76,
        "realized_pnl": Decimal("1250.50"),
        "leverage": 1.0,
        "liquidation_price": None,
        "margin": None,
        "opened_at": now,
        "updated_at": now,
    }

    _positions["ETH/USDT"] = {
        "symbol": "ETH/USDT",
        "side": PositionSide.LONG,
        "size": Decimal("2.5"),
        "entry_price": Decimal("2250.00"),
        "current_price": Decimal("2285.00"),
        "unrealized_pnl": Decimal("87.50"),
        "unrealized_pnl_pct": 1.56,
        "realized_pnl": Decimal("450.25"),
        "leverage": 1.0,
        "liquidation_price": None,
        "margin": None,
        "opened_at": now,
        "updated_at": now,
    }


@router.get("", response_model=PositionListResponse)
async def list_positions(
    current_user: Annotated[dict, Depends(get_current_active_user)],
    side: PositionSide | None = Query(None, description="Filter by position side"),
    min_size: Decimal | None = Query(None, description="Minimum position size", ge=0),
) -> PositionListResponse:
    """
    List all open positions.

    Supports filtering by side and minimum size.
    """
    _init_demo_positions()

    positions = list(_positions.values())

    # Filter by side
    if side:
        positions = [p for p in positions if p["side"] == side]

    # Filter by minimum size
    if min_size is not None:
        positions = [p for p in positions if p["size"] >= min_size]

    # Calculate totals
    total_unrealized = sum(p["unrealized_pnl"] for p in positions)
    total_realized = sum(p["realized_pnl"] for p in positions)

    return PositionListResponse(
        positions=[PositionResponse(**p) for p in positions],
        total_unrealized_pnl=total_unrealized,
        total_realized_pnl=total_realized,
    )


@router.get("/{symbol}", response_model=PositionResponse)
async def get_position(
    symbol: str,
    current_user: Annotated[dict, Depends(get_current_active_user)],
) -> PositionResponse:
    """Get position for a specific symbol."""
    _init_demo_positions()

    # Normalize symbol (support both BTC/USDT and BTCUSDT formats)
    normalized = symbol.upper().replace("-", "/")
    if "/" not in normalized and len(normalized) > 3:
        # Try to split (e.g., BTCUSDT -> BTC/USDT)
        # Common quote currencies
        for quote in ["USDT", "USDC", "USD", "BTC", "ETH"]:
            if normalized.endswith(quote):
                base = normalized[: -len(quote)]
                normalized = f"{base}/{quote}"
                break

    if normalized not in _positions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No position found for {symbol}",
        )

    return PositionResponse(**_positions[normalized])
