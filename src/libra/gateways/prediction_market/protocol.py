"""
Prediction Market Protocol: Data structures for prediction markets.

Defines unified data structures for prediction market platforms:
- Polymarket (crypto, USDC-based)
- Kalshi (regulated, USD-based)
- Metaculus (reputation-based forecasting)
- Manifold Markets (play-money)

Design inspired by:
- LIBRA Gateway Protocol (Issue #24)
- OpenBB Fetcher Pattern (Issue #27)
- Polymarket CLOB API
- Kalshi Trading API

Performance:
- Uses msgspec.Struct for 4x faster serialization than dataclass
- Decimal for precise probability/price calculations
- Nanosecond timestamps for consistency
"""

from __future__ import annotations

import time
from decimal import Decimal
from enum import Enum
from typing import Any

import msgspec


# =============================================================================
# Enums
# =============================================================================


class MarketStatus(str, Enum):
    """Prediction market status."""

    OPEN = "open"  # Market is open for trading
    CLOSED = "closed"  # Trading closed, awaiting resolution
    RESOLVED = "resolved"  # Market resolved with outcome
    CANCELLED = "cancelled"  # Market cancelled, positions refunded


class OutcomeType(str, Enum):
    """Type of outcome structure."""

    BINARY = "binary"  # Yes/No outcome
    MULTIPLE = "multiple"  # Multiple choice outcomes
    SCALAR = "scalar"  # Numeric range outcome


class MarketType(str, Enum):
    """Type of market mechanism."""

    CLOB = "clob"  # Central Limit Order Book
    AMM = "amm"  # Automated Market Maker
    LMSR = "lmsr"  # Logarithmic Market Scoring Rule
    CPMM = "cpmm"  # Constant Product Market Maker
    REPUTATION = "reputation"  # Reputation-based (non-monetary)


class PredictionOrderSide(str, Enum):
    """Order side for prediction markets."""

    BUY = "buy"
    SELL = "sell"


class PredictionOrderType(str, Enum):
    """Order type for prediction markets."""

    MARKET = "market"
    LIMIT = "limit"
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date


class PredictionOrderStatus(str, Enum):
    """Order status for prediction markets."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


# =============================================================================
# Core Data Structures
# =============================================================================


class Outcome(msgspec.Struct, frozen=True, gc=False):
    """
    Single outcome in a prediction market.

    Represents one possible result with current trading data.

    Examples:
        outcome = Outcome(
            outcome_id="yes",
            name="Yes",
            probability=Decimal("0.65"),
            price=Decimal("0.65"),
            volume=Decimal("150000"),
        )
    """

    outcome_id: str  # Unique identifier (e.g., "yes", "no", token_id)
    name: str  # Display name (e.g., "Yes", "No", "Trump", "Biden")
    probability: Decimal  # Current probability (0.0 - 1.0)
    price: Decimal  # Current trading price (typically equals probability)
    volume: Decimal = Decimal("0")  # Total volume traded

    # Optional fields
    token_id: str | None = None  # Token ID for on-chain markets
    open_interest: Decimal | None = None  # Open interest
    winner: bool | None = None  # True if this outcome won (after resolution)


class PredictionMarket(msgspec.Struct, frozen=True, gc=False):
    """
    Universal prediction market representation.

    Provides a unified view across different prediction market platforms.

    Examples:
        market = PredictionMarket(
            market_id="0x123abc",
            platform="polymarket",
            title="Will BTC exceed $100k in 2024?",
            description="Resolves YES if Bitcoin...",
            category="crypto",
            outcomes=(
                Outcome(outcome_id="yes", name="Yes", ...),
                Outcome(outcome_id="no", name="No", ...),
            ),
            status=MarketStatus.OPEN,
            volume=Decimal("500000"),
            liquidity=Decimal("100000"),
        )
    """

    # Required fields
    market_id: str  # Platform-specific market ID
    platform: str  # Platform name (polymarket, kalshi, metaculus, manifold)
    title: str  # Market question
    outcomes: tuple[Outcome, ...]  # Available outcomes

    # Market metadata
    status: MarketStatus = MarketStatus.OPEN
    outcome_type: OutcomeType = OutcomeType.BINARY
    market_type: MarketType = MarketType.CLOB

    # Optional metadata
    description: str | None = None  # Detailed description
    category: str | None = None  # Market category (politics, crypto, sports)
    tags: tuple[str, ...] = ()  # Additional tags
    slug: str | None = None  # URL-friendly identifier
    url: str | None = None  # Direct link to market

    # Trading data
    volume: Decimal = Decimal("0")  # Total volume traded
    volume_24h: Decimal = Decimal("0")  # 24-hour volume
    liquidity: Decimal = Decimal("0")  # Available liquidity
    num_traders: int = 0  # Number of unique traders

    # Resolution
    resolution_date: int | None = None  # Expected resolution (timestamp_ns)
    close_date: int | None = None  # Trading close date (timestamp_ns)
    resolved_at: int | None = None  # Actual resolution time (timestamp_ns)
    resolution_source: str | None = None  # Oracle/source for resolution
    winning_outcome: str | None = None  # ID of winning outcome (if resolved)

    # Timestamps
    created_at: int | None = None  # Creation time (timestamp_ns)
    updated_at: int | None = None  # Last update time (timestamp_ns)

    @property
    def is_open(self) -> bool:
        """Check if market is open for trading."""
        return self.status == MarketStatus.OPEN

    @property
    def is_resolved(self) -> bool:
        """Check if market is resolved."""
        return self.status == MarketStatus.RESOLVED

    @property
    def best_yes_price(self) -> Decimal | None:
        """Get best YES price (for binary markets)."""
        for outcome in self.outcomes:
            if outcome.name.lower() in ("yes", "true"):
                return outcome.price
        return self.outcomes[0].price if self.outcomes else None

    @property
    def best_no_price(self) -> Decimal | None:
        """Get best NO price (for binary markets)."""
        for outcome in self.outcomes:
            if outcome.name.lower() in ("no", "false"):
                return outcome.price
        # For binary, NO price = 1 - YES price
        yes_price = self.best_yes_price
        if yes_price is not None:
            return Decimal("1") - yes_price
        return None


class PredictionQuote(msgspec.Struct, frozen=True, gc=False):
    """
    Quote for a prediction market outcome.

    Represents current bid/ask prices for a specific outcome.

    Examples:
        quote = PredictionQuote(
            market_id="0x123abc",
            outcome_id="yes",
            platform="polymarket",
            bid=Decimal("0.64"),
            ask=Decimal("0.66"),
            mid=Decimal("0.65"),
            timestamp_ns=time.time_ns(),
        )
    """

    market_id: str
    outcome_id: str
    platform: str
    bid: Decimal  # Best bid price
    ask: Decimal  # Best ask price
    mid: Decimal  # Mid price
    timestamp_ns: int

    # Optional
    bid_size: Decimal | None = None  # Size at best bid
    ask_size: Decimal | None = None  # Size at best ask
    last_price: Decimal | None = None  # Last trade price
    last_size: Decimal | None = None  # Last trade size

    @property
    def spread(self) -> Decimal:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> Decimal:
        """Spread in basis points."""
        if self.mid == 0:
            return Decimal("0")
        return (self.spread / self.mid) * 10000


class PredictionOrderBookLevel(msgspec.Struct, frozen=True, gc=False):
    """Single order book level for prediction market."""

    price: Decimal  # Price (0.0 - 1.0)
    size: Decimal  # Size in shares/contracts


class PredictionOrderBook(msgspec.Struct, frozen=True, gc=False):
    """
    Order book for a prediction market outcome.

    Examples:
        orderbook = PredictionOrderBook(
            market_id="0x123abc",
            outcome_id="yes",
            platform="polymarket",
            bids=[(Decimal("0.64"), Decimal("1000")), ...],
            asks=[(Decimal("0.66"), Decimal("500")), ...],
            timestamp_ns=time.time_ns(),
        )
    """

    market_id: str
    outcome_id: str
    platform: str
    bids: tuple[PredictionOrderBookLevel, ...]  # Sorted desc by price
    asks: tuple[PredictionOrderBookLevel, ...]  # Sorted asc by price
    timestamp_ns: int

    @property
    def best_bid(self) -> Decimal | None:
        """Best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        """Best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        """Mid price."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Decimal | None:
        """Bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


# =============================================================================
# Trading Data Structures (Phase 3)
# =============================================================================


class PredictionOrder(msgspec.Struct, frozen=True, gc=False):
    """
    Order for prediction market trading.

    Examples:
        order = PredictionOrder(
            market_id="0x123abc",
            outcome_id="yes",
            platform="polymarket",
            side=PredictionOrderSide.BUY,
            order_type=PredictionOrderType.LIMIT,
            size=Decimal("100"),
            price=Decimal("0.65"),
        )
    """

    market_id: str
    outcome_id: str
    platform: str
    side: PredictionOrderSide
    order_type: PredictionOrderType
    size: Decimal  # Number of shares/contracts

    # Optional
    price: Decimal | None = None  # Limit price (required for limit orders)
    order_id: str | None = None  # Assigned after submission
    client_order_id: str | None = None  # Client-assigned ID
    expiration: int | None = None  # Expiration timestamp_ns (for GTD)
    timestamp_ns: int | None = None  # Creation time

    def with_id(self, order_id: str) -> PredictionOrder:
        """Return new order with assigned ID."""
        return PredictionOrder(
            market_id=self.market_id,
            outcome_id=self.outcome_id,
            platform=self.platform,
            side=self.side,
            order_type=self.order_type,
            size=self.size,
            price=self.price,
            order_id=order_id,
            client_order_id=self.client_order_id,
            expiration=self.expiration,
            timestamp_ns=self.timestamp_ns,
        )

    def with_timestamp(self) -> PredictionOrder:
        """Return new order with current timestamp."""
        return PredictionOrder(
            market_id=self.market_id,
            outcome_id=self.outcome_id,
            platform=self.platform,
            side=self.side,
            order_type=self.order_type,
            size=self.size,
            price=self.price,
            order_id=self.order_id,
            client_order_id=self.client_order_id,
            expiration=self.expiration,
            timestamp_ns=time.time_ns(),
        )


class PredictionOrderResult(msgspec.Struct, frozen=True, gc=False):
    """
    Result of prediction order submission.

    Examples:
        result = PredictionOrderResult(
            order_id="ord_123",
            market_id="0x123abc",
            outcome_id="yes",
            platform="polymarket",
            status=PredictionOrderStatus.FILLED,
            side=PredictionOrderSide.BUY,
            size=Decimal("100"),
            filled_size=Decimal("100"),
            average_price=Decimal("0.65"),
            timestamp_ns=time.time_ns(),
        )
    """

    order_id: str
    market_id: str
    outcome_id: str
    platform: str
    status: PredictionOrderStatus
    side: PredictionOrderSide
    size: Decimal  # Original size
    filled_size: Decimal  # Filled amount
    timestamp_ns: int

    # Fill information
    average_price: Decimal | None = None  # Average fill price
    remaining_size: Decimal | None = None  # Remaining unfilled

    # Cost/fees
    cost: Decimal | None = None  # Total cost in settlement currency
    fee: Decimal | None = None  # Trading fee

    # Optional
    client_order_id: str | None = None
    price: Decimal | None = None  # Limit price if applicable
    trades: list[dict[str, Any]] | None = None  # Individual fills

    @property
    def is_open(self) -> bool:
        """Check if order is still open."""
        return self.status in (
            PredictionOrderStatus.OPEN,
            PredictionOrderStatus.PARTIALLY_FILLED,
            PredictionOrderStatus.PENDING,
        )

    @property
    def fill_percent(self) -> Decimal:
        """Percentage filled (0-100)."""
        if self.size == 0:
            return Decimal("0")
        return (self.filled_size / self.size) * 100


class PredictionPosition(msgspec.Struct, frozen=True, gc=False):
    """
    Position in a prediction market.

    Examples:
        position = PredictionPosition(
            market_id="0x123abc",
            outcome_id="yes",
            platform="polymarket",
            size=Decimal("500"),
            avg_price=Decimal("0.60"),
            current_price=Decimal("0.65"),
            unrealized_pnl=Decimal("25"),
        )
    """

    market_id: str
    outcome_id: str
    platform: str
    size: Decimal  # Number of shares held
    avg_price: Decimal  # Average entry price
    current_price: Decimal  # Current market price
    unrealized_pnl: Decimal  # Unrealized profit/loss

    # Optional
    realized_pnl: Decimal = Decimal("0")  # Realized P&L
    cost_basis: Decimal | None = None  # Total cost basis
    market_value: Decimal | None = None  # Current market value
    timestamp_ns: int | None = None

    @property
    def total_pnl(self) -> Decimal:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def pnl_percent(self) -> Decimal:
        """P&L as percentage of cost basis."""
        if self.avg_price == 0 or self.size == 0:
            return Decimal("0")
        cost = self.size * self.avg_price
        return (self.unrealized_pnl / cost) * 100


# =============================================================================
# Platform-Specific Extensions
# =============================================================================


class PolymarketMarketData(msgspec.Struct, frozen=True, gc=False):
    """Polymarket-specific market data."""

    condition_id: str  # CTF condition ID
    question_id: str  # Question ID
    tokens: tuple[str, ...]  # Token IDs for outcomes
    clob_token_ids: tuple[str, ...]  # CLOB token IDs
    maker_base_fee: Decimal = Decimal("0")
    taker_base_fee: Decimal = Decimal("0")
    active: bool = True
    closed: bool = False
    accepting_orders: bool = True
    minimum_order_size: Decimal = Decimal("1")
    minimum_tick_size: Decimal = Decimal("0.01")


class KalshiMarketData(msgspec.Struct, frozen=True, gc=False):
    """Kalshi-specific market data."""

    ticker: str  # Kalshi ticker
    series_ticker: str  # Series ticker
    event_ticker: str  # Event ticker
    can_close_early: bool = False
    expected_expiration_time: int | None = None
    strike_type: str | None = None  # For ranged markets
    floor_strike: Decimal | None = None
    cap_strike: Decimal | None = None


class MetaculusQuestionData(msgspec.Struct, frozen=True, gc=False):
    """Metaculus-specific question data."""

    question_id: int
    page_url: str
    author_id: int | None = None
    community_prediction: Decimal | None = None  # Community median
    metaculus_prediction: Decimal | None = None  # AI prediction
    resolution_criteria: str | None = None
    fine_print: str | None = None
    publish_time: int | None = None
    close_time: int | None = None
    effected_close_time: int | None = None


class ManifoldMarketData(msgspec.Struct, frozen=True, gc=False):
    """Manifold-specific market data."""

    creator_id: str
    creator_username: str
    pool: dict[str, Decimal] | None = None  # AMM pool
    p: Decimal | None = None  # AMM probability parameter
    total_liquidity: Decimal = Decimal("0")
    subsidized_total_liquidity: Decimal = Decimal("0")
    mechanism: str = "cpmm-1"  # dpm-2, cpmm-1, etc.
    visibility: str = "public"


# =============================================================================
# Gateway Capabilities
# =============================================================================


class PredictionMarketCapabilities(msgspec.Struct, frozen=True, gc=False):
    """
    Capabilities of a prediction market provider.

    Used for feature negotiation and graceful degradation.
    """

    # Trading
    supports_trading: bool = False
    supports_limit_orders: bool = False
    supports_market_orders: bool = False

    # Data
    supports_orderbook: bool = False
    supports_trades: bool = False
    supports_positions: bool = False
    supports_historical_prices: bool = False

    # Market Types
    supports_binary: bool = True
    supports_multiple_choice: bool = False
    supports_scalar: bool = False

    # Other
    is_real_money: bool = False
    is_regulated: bool = False
    settlement_currency: str = "USD"

    # Rate Limits
    max_requests_per_minute: int = 60


# Provider capability presets
POLYMARKET_CAPABILITIES = PredictionMarketCapabilities(
    supports_trading=True,
    supports_limit_orders=True,
    supports_market_orders=True,
    supports_orderbook=True,
    supports_trades=True,
    supports_positions=True,
    supports_historical_prices=True,
    supports_binary=True,
    supports_multiple_choice=True,
    is_real_money=True,
    is_regulated=False,
    settlement_currency="USDC",
    max_requests_per_minute=120,
)

KALSHI_CAPABILITIES = PredictionMarketCapabilities(
    supports_trading=True,
    supports_limit_orders=True,
    supports_market_orders=True,
    supports_orderbook=True,
    supports_trades=True,
    supports_positions=True,
    supports_historical_prices=True,
    supports_binary=True,
    supports_scalar=True,
    is_real_money=True,
    is_regulated=True,
    settlement_currency="USD",
    max_requests_per_minute=60,
)

METACULUS_CAPABILITIES = PredictionMarketCapabilities(
    supports_trading=False,
    supports_orderbook=False,
    supports_positions=False,
    supports_historical_prices=True,
    supports_binary=True,
    supports_scalar=True,
    is_real_money=False,
    is_regulated=False,
    settlement_currency="",
    max_requests_per_minute=60,
)

MANIFOLD_CAPABILITIES = PredictionMarketCapabilities(
    supports_trading=True,
    supports_limit_orders=True,
    supports_market_orders=True,
    supports_orderbook=False,  # AMM-based
    supports_trades=True,
    supports_positions=True,
    supports_historical_prices=True,
    supports_binary=True,
    supports_multiple_choice=True,
    is_real_money=False,  # Play money
    is_regulated=False,
    settlement_currency="M$",
    max_requests_per_minute=60,
)


# =============================================================================
# Exceptions
# =============================================================================


class PredictionMarketError(Exception):
    """Base exception for prediction market errors."""


class ProviderNotAvailableError(PredictionMarketError):
    """Provider is not available or configured."""


class MarketNotFoundError(PredictionMarketError):
    """Market not found."""


class InvalidOrderError(PredictionMarketError):
    """Invalid order parameters."""


class InsufficientBalanceError(PredictionMarketError):
    """Insufficient balance for order."""


class RateLimitError(PredictionMarketError):
    """Rate limit exceeded."""


class AuthenticationError(PredictionMarketError):
    """Authentication failed."""
