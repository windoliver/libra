"""
Prediction Market Query Types.

Query types for the TET (Transform-Extract-Transform) pipeline.
Following the pattern from Issue #27: Provider/Fetcher Pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from libra.gateways.fetcher import BaseQuery
from libra.gateways.prediction_market.protocol import (
    MarketStatus,
    OutcomeType,
    PredictionOrderSide,
    PredictionOrderType,
)


# =============================================================================
# Market Queries
# =============================================================================


@dataclass(frozen=True)
class PredictionMarketQuery(BaseQuery):
    """
    Query for fetching prediction markets.

    Examples:
        # Get all open crypto markets
        query = PredictionMarketQuery(
            category="crypto",
            status=MarketStatus.OPEN,
            limit=50,
        )

        # Search for specific market
        query = PredictionMarketQuery(
            search="bitcoin",
            provider="polymarket",
        )
    """

    provider: str | None = None  # polymarket, kalshi, metaculus, manifold
    category: str | None = None  # crypto, politics, sports, science, etc.
    status: MarketStatus | None = None  # Filter by status
    outcome_type: OutcomeType | None = None  # binary, multiple, scalar
    search: str | None = None  # Search term
    limit: int = 100  # Max results
    offset: int = 0  # Pagination offset
    sort_by: str | None = None  # volume, liquidity, close_date, created
    sort_order: str = "desc"  # asc, desc


@dataclass(frozen=True)
class MarketDetailQuery(BaseQuery):
    """
    Query for fetching a specific market's details.

    Examples:
        query = MarketDetailQuery(
            market_id="0x123abc",
            provider="polymarket",
        )
    """

    market_id: str
    provider: str


@dataclass(frozen=True)
class MarketSearchQuery(BaseQuery):
    """
    Query for searching markets across providers.

    Examples:
        query = MarketSearchQuery(
            search="presidential election 2024",
            providers=["polymarket", "kalshi"],
        )
    """

    search: str
    providers: tuple[str, ...] | None = None  # None = all providers
    limit: int = 50


# =============================================================================
# Quote/Price Queries
# =============================================================================


@dataclass(frozen=True)
class PredictionQuoteQuery(BaseQuery):
    """
    Query for fetching quotes for a market outcome.

    Examples:
        query = PredictionQuoteQuery(
            market_id="0x123abc",
            outcome_id="yes",
            provider="polymarket",
        )
    """

    market_id: str
    outcome_id: str
    provider: str


@dataclass(frozen=True)
class MultiQuoteQuery(BaseQuery):
    """
    Query for fetching multiple quotes at once.

    Examples:
        query = MultiQuoteQuery(
            markets=[
                ("0x123abc", "yes"),
                ("0x456def", "no"),
            ],
            provider="polymarket",
        )
    """

    markets: tuple[tuple[str, str], ...]  # List of (market_id, outcome_id)
    provider: str


# =============================================================================
# Order Book Queries
# =============================================================================


@dataclass(frozen=True)
class PredictionOrderBookQuery(BaseQuery):
    """
    Query for fetching order book for a market outcome.

    Examples:
        query = PredictionOrderBookQuery(
            market_id="0x123abc",
            outcome_id="yes",
            provider="polymarket",
            depth=20,
        )
    """

    market_id: str
    outcome_id: str
    provider: str
    depth: int = 20  # Number of levels


# =============================================================================
# Historical Data Queries
# =============================================================================


@dataclass(frozen=True)
class PriceHistoryQuery(BaseQuery):
    """
    Query for fetching historical prices.

    Examples:
        query = PriceHistoryQuery(
            market_id="0x123abc",
            outcome_id="yes",
            provider="polymarket",
            interval="1h",
            start=datetime(2024, 1, 1),
        )
    """

    market_id: str
    outcome_id: str | None = None  # None = all outcomes
    provider: str = ""
    interval: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d
    start: datetime | None = None
    end: datetime | None = None
    limit: int = 1000


@dataclass(frozen=True)
class TradeHistoryQuery(BaseQuery):
    """
    Query for fetching trade history.

    Examples:
        query = TradeHistoryQuery(
            market_id="0x123abc",
            provider="polymarket",
            limit=100,
        )
    """

    market_id: str
    outcome_id: str | None = None
    provider: str = ""
    limit: int = 100
    since: datetime | None = None


# =============================================================================
# Position Queries
# =============================================================================


@dataclass(frozen=True)
class PositionQuery(BaseQuery):
    """
    Query for fetching user positions.

    Examples:
        # Get all positions
        query = PositionQuery(provider="polymarket")

        # Get position for specific market
        query = PositionQuery(
            provider="polymarket",
            market_id="0x123abc",
        )
    """

    provider: str
    market_id: str | None = None  # None = all positions
    outcome_id: str | None = None


@dataclass(frozen=True)
class BalanceQuery(BaseQuery):
    """
    Query for fetching account balance.

    Examples:
        query = BalanceQuery(provider="polymarket")
    """

    provider: str
    currency: str | None = None  # None = all currencies


# =============================================================================
# Order Queries
# =============================================================================


@dataclass(frozen=True)
class OrderQuery(BaseQuery):
    """
    Query for fetching orders.

    Examples:
        query = OrderQuery(
            provider="polymarket",
            status="open",
        )
    """

    provider: str
    market_id: str | None = None
    order_id: str | None = None
    status: str | None = None  # open, filled, cancelled, all
    limit: int = 100
    since: datetime | None = None


@dataclass(frozen=True)
class CreateOrderQuery(BaseQuery):
    """
    Query for creating a new order.

    Examples:
        query = CreateOrderQuery(
            market_id="0x123abc",
            outcome_id="yes",
            provider="polymarket",
            side=PredictionOrderSide.BUY,
            order_type=PredictionOrderType.LIMIT,
            size=100,
            price=0.65,
        )
    """

    market_id: str
    outcome_id: str
    provider: str
    side: PredictionOrderSide
    order_type: PredictionOrderType
    size: float  # Will be converted to Decimal
    price: float | None = None  # Required for limit orders
    client_order_id: str | None = None
    expiration: datetime | None = None


@dataclass(frozen=True)
class CancelOrderQuery(BaseQuery):
    """
    Query for cancelling an order.

    Examples:
        query = CancelOrderQuery(
            order_id="ord_123",
            provider="polymarket",
        )
    """

    order_id: str
    provider: str
    market_id: str | None = None  # Required by some providers


# =============================================================================
# Aggregation Queries
# =============================================================================


@dataclass(frozen=True)
class ArbitrageQuery(BaseQuery):
    """
    Query for finding arbitrage opportunities across platforms.

    Examples:
        query = ArbitrageQuery(
            min_spread=Decimal("0.02"),
            providers=["polymarket", "kalshi"],
        )
    """

    min_spread: float = 0.01  # Minimum spread to report
    providers: tuple[str, ...] | None = None  # None = all available
    category: str | None = None


@dataclass(frozen=True)
class ForecastComparisonQuery(BaseQuery):
    """
    Query for comparing forecasts across platforms.

    Examples:
        query = ForecastComparisonQuery(
            search="2024 election",
            providers=["polymarket", "metaculus", "manifold"],
        )
    """

    search: str
    providers: tuple[str, ...] | None = None
    limit: int = 20
