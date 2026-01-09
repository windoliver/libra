"""
Pydantic schemas for API request/response models (Issue #30).

All API models use Pydantic v2 for:
- Request validation
- Response serialization
- OpenAPI schema generation
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Enums
# =============================================================================


class StrategyStatus(str, Enum):
    """Strategy execution status."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class OrderSide(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order execution status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(str, Enum):
    """Position side."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


# =============================================================================
# Base Models
# =============================================================================


class APIResponse(BaseModel):
    """Standard API response wrapper."""

    model_config = ConfigDict(from_attributes=True)

    success: bool = True
    message: str = ""
    data: Any = None


class PaginatedResponse(BaseModel):
    """Paginated response for list endpoints."""

    model_config = ConfigDict(from_attributes=True)

    items: list[Any]
    total: int
    page: int = 1
    page_size: int = 50
    has_more: bool = False


# =============================================================================
# Strategy Schemas
# =============================================================================


class StrategyBase(BaseModel):
    """Base strategy model."""

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(..., description="Strategy name", min_length=1, max_length=100)
    description: str = Field("", description="Strategy description")
    symbols: list[str] = Field(default_factory=list, description="Trading symbols")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")


class StrategyCreate(StrategyBase):
    """Schema for creating a strategy."""

    strategy_type: str = Field(..., description="Strategy type (e.g., 'sma_cross')")


class StrategyUpdate(BaseModel):
    """Schema for updating a strategy."""

    model_config = ConfigDict(from_attributes=True)

    name: str | None = None
    description: str | None = None
    symbols: list[str] | None = None
    parameters: dict[str, Any] | None = None


class StrategyResponse(StrategyBase):
    """Strategy response model."""

    id: str = Field(..., description="Unique strategy ID")
    strategy_type: str = Field(..., description="Strategy type")
    status: StrategyStatus = Field(StrategyStatus.STOPPED, description="Current status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")

    # Performance metrics
    total_pnl: Decimal = Field(Decimal("0"), description="Total P&L")
    total_trades: int = Field(0, description="Total trades executed")
    win_rate: float = Field(0.0, description="Win rate percentage")


class StrategyListResponse(BaseModel):
    """Response for listing strategies."""

    model_config = ConfigDict(from_attributes=True)

    strategies: list[StrategyResponse]
    total: int


# =============================================================================
# Position Schemas
# =============================================================================


class PositionResponse(BaseModel):
    """Position response model."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str = Field(..., description="Trading symbol")
    side: PositionSide = Field(..., description="Position side")
    size: Decimal = Field(..., description="Position size")
    entry_price: Decimal = Field(..., description="Average entry price")
    current_price: Decimal = Field(..., description="Current market price")
    unrealized_pnl: Decimal = Field(..., description="Unrealized P&L")
    unrealized_pnl_pct: float = Field(..., description="Unrealized P&L percentage")
    realized_pnl: Decimal = Field(Decimal("0"), description="Realized P&L")
    leverage: float = Field(1.0, description="Position leverage")
    liquidation_price: Decimal | None = Field(None, description="Liquidation price")
    margin: Decimal | None = Field(None, description="Margin used")
    opened_at: datetime = Field(..., description="Position open time")
    updated_at: datetime = Field(..., description="Last update time")


class PositionListResponse(BaseModel):
    """Response for listing positions."""

    model_config = ConfigDict(from_attributes=True)

    positions: list[PositionResponse]
    total_unrealized_pnl: Decimal
    total_realized_pnl: Decimal


# =============================================================================
# Order Schemas
# =============================================================================


class OrderCreate(BaseModel):
    """Schema for creating an order."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Order side")
    order_type: OrderType = Field(OrderType.MARKET, description="Order type")
    quantity: Decimal = Field(..., description="Order quantity", gt=0)
    price: Decimal | None = Field(None, description="Limit price (required for limit orders)")
    stop_price: Decimal | None = Field(None, description="Stop price (for stop orders)")
    time_in_force: str = Field("GTC", description="Time in force (GTC, IOC, FOK)")
    reduce_only: bool = Field(False, description="Reduce-only order")
    post_only: bool = Field(False, description="Post-only order")

    # Execution algorithm (Issue #36)
    exec_algorithm: str | None = Field(None, description="Execution algorithm (twap, vwap, iceberg)")
    exec_algorithm_params: dict[str, Any] | None = Field(None, description="Algorithm parameters")


class OrderResponse(BaseModel):
    """Order response model."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Order ID")
    client_order_id: str | None = Field(None, description="Client order ID")
    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Order side")
    order_type: OrderType = Field(..., description="Order type")
    status: OrderStatus = Field(..., description="Order status")
    quantity: Decimal = Field(..., description="Order quantity")
    filled_quantity: Decimal = Field(Decimal("0"), description="Filled quantity")
    remaining_quantity: Decimal = Field(..., description="Remaining quantity")
    price: Decimal | None = Field(None, description="Limit price")
    average_price: Decimal | None = Field(None, description="Average fill price")
    stop_price: Decimal | None = Field(None, description="Stop price")
    time_in_force: str = Field("GTC", description="Time in force")
    created_at: datetime = Field(..., description="Creation time")
    updated_at: datetime = Field(..., description="Last update time")
    filled_at: datetime | None = Field(None, description="Fill time")

    # Execution algorithm info
    exec_algorithm: str | None = Field(None, description="Execution algorithm used")


class OrderListResponse(BaseModel):
    """Response for listing orders."""

    model_config = ConfigDict(from_attributes=True)

    orders: list[OrderResponse]
    total: int


class OrderCancelResponse(BaseModel):
    """Response for order cancellation."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    cancelled: bool
    message: str = ""


# =============================================================================
# Market Data Schemas
# =============================================================================


class QuoteResponse(BaseModel):
    """Quote/ticker response model."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str = Field(..., description="Trading symbol")
    bid: Decimal = Field(..., description="Best bid price")
    bid_size: Decimal = Field(..., description="Best bid size")
    ask: Decimal = Field(..., description="Best ask price")
    ask_size: Decimal = Field(..., description="Best ask size")
    last: Decimal = Field(..., description="Last trade price")
    last_size: Decimal = Field(..., description="Last trade size")
    volume_24h: Decimal = Field(..., description="24h volume")
    change_24h: float = Field(..., description="24h price change percentage")
    high_24h: Decimal = Field(..., description="24h high")
    low_24h: Decimal = Field(..., description="24h low")
    timestamp: datetime = Field(..., description="Quote timestamp")


class BarResponse(BaseModel):
    """OHLCV bar response model."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Bar timestamp")
    open: Decimal = Field(..., description="Open price")
    high: Decimal = Field(..., description="High price")
    low: Decimal = Field(..., description="Low price")
    close: Decimal = Field(..., description="Close price")
    volume: Decimal = Field(..., description="Volume")


class BarsResponse(BaseModel):
    """Response for historical bars."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    timeframe: str
    bars: list[BarResponse]
    total: int


# =============================================================================
# Authentication Schemas
# =============================================================================


class Token(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token expiry in seconds")


class TokenData(BaseModel):
    """Token payload data."""

    username: str | None = None
    scopes: list[str] = Field(default_factory=list)


class UserCreate(BaseModel):
    """Schema for creating a user."""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    email: str | None = None


class UserResponse(BaseModel):
    """User response model."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    username: str
    email: str | None = None
    is_active: bool = True
    created_at: datetime


# =============================================================================
# WebSocket Schemas
# =============================================================================


class WSMessage(BaseModel):
    """WebSocket message model."""

    type: str = Field(..., description="Message type")
    channel: str = Field(..., description="Channel/topic")
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSSubscribe(BaseModel):
    """WebSocket subscription request."""

    action: str = Field("subscribe", description="subscribe or unsubscribe")
    channels: list[str] = Field(..., description="Channels to subscribe to")
