"""
Strategy Management Router for LIBRA API (Issue #30).

Endpoints:
- GET /strategies - List all strategies
- POST /strategies - Create strategy
- GET /strategies/{id} - Get strategy details
- PUT /strategies/{id} - Update strategy
- DELETE /strategies/{id} - Delete strategy
- POST /strategies/{id}/start - Start strategy
- POST /strategies/{id}/stop - Stop strategy
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status

from libra.api.deps import get_current_active_user, require_scope
from libra.api.schemas import (
    StrategyCreate,
    StrategyListResponse,
    StrategyResponse,
    StrategyStatus,
    StrategyUpdate,
)


router = APIRouter()

# In-memory strategy storage for demo
_strategies: dict[str, dict[str, Any]] = {}


def _init_demo_strategies() -> None:
    """Initialize demo strategies if empty."""
    if _strategies:
        return

    now = datetime.now(timezone.utc)
    _strategies["strat_sma_cross_btc"] = {
        "id": "strat_sma_cross_btc",
        "name": "SMA Crossover BTC",
        "description": "Simple moving average crossover strategy for BTC/USDT",
        "strategy_type": "sma_cross",
        "symbols": ["BTC/USDT"],
        "parameters": {"fast_period": 10, "slow_period": 20},
        "status": StrategyStatus.STOPPED,
        "created_at": now,
        "updated_at": None,
        "total_pnl": Decimal("1250.50"),
        "total_trades": 45,
        "win_rate": 0.62,
    }
    _strategies["strat_momentum_eth"] = {
        "id": "strat_momentum_eth",
        "name": "Momentum ETH",
        "description": "Momentum-based strategy for ETH/USDT",
        "strategy_type": "momentum",
        "symbols": ["ETH/USDT"],
        "parameters": {"lookback": 14, "threshold": 0.02},
        "status": StrategyStatus.RUNNING,
        "created_at": now,
        "updated_at": now,
        "total_pnl": Decimal("890.25"),
        "total_trades": 28,
        "win_rate": 0.57,
    }


@router.get("", response_model=StrategyListResponse)
async def list_strategies(
    current_user: Annotated[dict, Depends(get_current_active_user)],
    status_filter: StrategyStatus | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> StrategyListResponse:
    """
    List all strategies.

    Supports filtering by status and pagination.
    """
    _init_demo_strategies()

    strategies = list(_strategies.values())

    # Filter by status if provided
    if status_filter:
        strategies = [s for s in strategies if s["status"] == status_filter]

    total = len(strategies)
    strategies = strategies[offset : offset + limit]

    return StrategyListResponse(
        strategies=[StrategyResponse(**s) for s in strategies],
        total=total,
    )


@router.post("", response_model=StrategyResponse, status_code=status.HTTP_201_CREATED)
async def create_strategy(
    strategy: StrategyCreate,
    current_user: Annotated[dict, Depends(require_scope("write"))],
) -> StrategyResponse:
    """
    Create a new strategy.

    Requires write scope.
    """
    _init_demo_strategies()

    strategy_id = f"strat_{uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    strategy_data = {
        "id": strategy_id,
        "name": strategy.name,
        "description": strategy.description,
        "strategy_type": strategy.strategy_type,
        "symbols": strategy.symbols,
        "parameters": strategy.parameters,
        "status": StrategyStatus.STOPPED,
        "created_at": now,
        "updated_at": None,
        "total_pnl": Decimal("0"),
        "total_trades": 0,
        "win_rate": 0.0,
    }

    _strategies[strategy_id] = strategy_data
    return StrategyResponse(**strategy_data)


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: str,
    current_user: Annotated[dict, Depends(get_current_active_user)],
) -> StrategyResponse:
    """Get strategy details by ID."""
    _init_demo_strategies()

    if strategy_id not in _strategies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        )

    return StrategyResponse(**_strategies[strategy_id])


@router.put("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: str,
    update: StrategyUpdate,
    current_user: Annotated[dict, Depends(require_scope("write"))],
) -> StrategyResponse:
    """
    Update a strategy.

    Requires write scope. Only stopped strategies can be updated.
    """
    _init_demo_strategies()

    if strategy_id not in _strategies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        )

    strategy = _strategies[strategy_id]

    if strategy["status"] == StrategyStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot update a running strategy. Stop it first.",
        )

    # Update fields
    if update.name is not None:
        strategy["name"] = update.name
    if update.description is not None:
        strategy["description"] = update.description
    if update.symbols is not None:
        strategy["symbols"] = update.symbols
    if update.parameters is not None:
        strategy["parameters"] = update.parameters

    strategy["updated_at"] = datetime.now(timezone.utc)

    return StrategyResponse(**strategy)


@router.delete("/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_strategy(
    strategy_id: str,
    current_user: Annotated[dict, Depends(require_scope("write"))],
) -> None:
    """
    Delete a strategy.

    Requires write scope. Only stopped strategies can be deleted.
    """
    _init_demo_strategies()

    if strategy_id not in _strategies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        )

    strategy = _strategies[strategy_id]

    if strategy["status"] == StrategyStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete a running strategy. Stop it first.",
        )

    del _strategies[strategy_id]


@router.post("/{strategy_id}/start", response_model=StrategyResponse)
async def start_strategy(
    strategy_id: str,
    current_user: Annotated[dict, Depends(require_scope("write"))],
) -> StrategyResponse:
    """
    Start a strategy.

    Requires write scope.
    """
    _init_demo_strategies()

    if strategy_id not in _strategies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        )

    strategy = _strategies[strategy_id]

    if strategy["status"] == StrategyStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Strategy is already running",
        )

    if strategy["status"] == StrategyStatus.STARTING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Strategy is already starting",
        )

    # Transition to running
    strategy["status"] = StrategyStatus.RUNNING
    strategy["updated_at"] = datetime.now(timezone.utc)

    return StrategyResponse(**strategy)


@router.post("/{strategy_id}/stop", response_model=StrategyResponse)
async def stop_strategy(
    strategy_id: str,
    current_user: Annotated[dict, Depends(require_scope("write"))],
) -> StrategyResponse:
    """
    Stop a running strategy.

    Requires write scope.
    """
    _init_demo_strategies()

    if strategy_id not in _strategies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        )

    strategy = _strategies[strategy_id]

    if strategy["status"] == StrategyStatus.STOPPED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Strategy is already stopped",
        )

    if strategy["status"] == StrategyStatus.STOPPING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Strategy is already stopping",
        )

    # Transition to stopped
    strategy["status"] = StrategyStatus.STOPPED
    strategy["updated_at"] = datetime.now(timezone.utc)

    return StrategyResponse(**strategy)
