"""
Market Data Router for LIBRA API (Issue #30).

Endpoints:
- GET /quotes/{symbol} - Get current quote/ticker
- GET /bars/{symbol} - Get OHLCV bars
- GET /symbols - List available symbols
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from libra.api.deps import get_optional_user
from libra.api.schemas import BarResponse, BarsResponse, QuoteResponse


router = APIRouter()


# Demo market data
_symbols = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "MATIC/USDT",
    "LINK/USDT",
]

_base_prices = {
    "BTC/USDT": Decimal("43250.00"),
    "ETH/USDT": Decimal("2285.00"),
    "SOL/USDT": Decimal("98.50"),
    "DOGE/USDT": Decimal("0.0825"),
    "XRP/USDT": Decimal("0.5250"),
    "ADA/USDT": Decimal("0.4850"),
    "AVAX/USDT": Decimal("35.75"),
    "DOT/USDT": Decimal("7.25"),
    "MATIC/USDT": Decimal("0.8150"),
    "LINK/USDT": Decimal("14.25"),
}


def _normalize_symbol(symbol: str) -> str:
    """Normalize symbol format."""
    normalized = symbol.upper().replace("-", "/")
    if "/" not in normalized and len(normalized) > 3:
        for quote in ["USDT", "USDC", "USD", "BTC", "ETH"]:
            if normalized.endswith(quote):
                base = normalized[: -len(quote)]
                return f"{base}/{quote}"
    return normalized


def _generate_quote(symbol: str) -> QuoteResponse:
    """Generate simulated quote data."""
    base_price = _base_prices.get(symbol, Decimal("100.00"))

    # Add some randomness
    spread_pct = Decimal("0.0005")  # 0.05% spread
    change_pct = random.uniform(-0.02, 0.02)

    mid = base_price * (1 + Decimal(str(change_pct)))
    bid = mid * (1 - spread_pct)
    ask = mid * (1 + spread_pct)

    return QuoteResponse(
        symbol=symbol,
        bid=bid.quantize(Decimal("0.01")),
        bid_size=Decimal(str(random.uniform(0.5, 10.0))).quantize(Decimal("0.001")),
        ask=ask.quantize(Decimal("0.01")),
        ask_size=Decimal(str(random.uniform(0.5, 10.0))).quantize(Decimal("0.001")),
        last=mid.quantize(Decimal("0.01")),
        last_size=Decimal(str(random.uniform(0.1, 2.0))).quantize(Decimal("0.001")),
        volume_24h=Decimal(str(random.uniform(10000, 100000))).quantize(Decimal("0.01")),
        change_24h=change_pct * 100,
        high_24h=(base_price * Decimal("1.025")).quantize(Decimal("0.01")),
        low_24h=(base_price * Decimal("0.975")).quantize(Decimal("0.01")),
        timestamp=datetime.now(timezone.utc),
    )


def _generate_bars(symbol: str, timeframe: str, limit: int) -> list[BarResponse]:
    """Generate simulated OHLCV bars."""
    base_price = _base_prices.get(symbol, Decimal("100.00"))

    # Parse timeframe
    tf_minutes = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }.get(timeframe, 60)

    bars = []
    now = datetime.now(timezone.utc)
    current_price = base_price

    for i in range(limit):
        bar_time = now - timedelta(minutes=tf_minutes * (limit - i - 1))

        # Random OHLCV
        change = Decimal(str(random.uniform(-0.01, 0.01)))
        open_price = current_price
        close_price = open_price * (1 + change)
        high_price = max(open_price, close_price) * Decimal(str(1 + random.uniform(0, 0.005)))
        low_price = min(open_price, close_price) * Decimal(str(1 - random.uniform(0, 0.005)))
        volume = Decimal(str(random.uniform(100, 1000)))

        bars.append(
            BarResponse(
                symbol=symbol,
                timestamp=bar_time,
                open=open_price.quantize(Decimal("0.01")),
                high=high_price.quantize(Decimal("0.01")),
                low=low_price.quantize(Decimal("0.01")),
                close=close_price.quantize(Decimal("0.01")),
                volume=volume.quantize(Decimal("0.01")),
            )
        )

        current_price = close_price

    return bars


@router.get("/symbols")
async def list_symbols(
    _current_user: Annotated[dict | None, Depends(get_optional_user)] = None,
) -> dict:
    """
    List available trading symbols.

    This endpoint is public (no auth required).
    """
    return {
        "symbols": _symbols,
        "total": len(_symbols),
    }


@router.get("/quotes/{symbol}", response_model=QuoteResponse)
async def get_quote(
    symbol: str,
    _current_user: Annotated[dict | None, Depends(get_optional_user)] = None,
) -> QuoteResponse:
    """
    Get current quote/ticker for a symbol.

    This endpoint is public (no auth required).
    """
    normalized = _normalize_symbol(symbol)

    if normalized not in _base_prices:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Symbol {symbol} not found",
        )

    return _generate_quote(normalized)


@router.get("/bars/{symbol}", response_model=BarsResponse)
async def get_bars(
    symbol: str,
    timeframe: str = Query("1h", pattern="^(1m|5m|15m|1h|4h|1d)$"),
    limit: int = Query(100, ge=1, le=1000),
    _current_user: Annotated[dict | None, Depends(get_optional_user)] = None,
) -> BarsResponse:
    """
    Get OHLCV bars for a symbol.

    This endpoint is public (no auth required).

    **Timeframes:**
    - 1m: 1 minute
    - 5m: 5 minutes
    - 15m: 15 minutes
    - 1h: 1 hour
    - 4h: 4 hours
    - 1d: 1 day
    """
    normalized = _normalize_symbol(symbol)

    if normalized not in _base_prices:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Symbol {symbol} not found",
        )

    bars = _generate_bars(normalized, timeframe, limit)

    return BarsResponse(
        symbol=normalized,
        timeframe=timeframe,
        bars=bars,
        total=len(bars),
    )
