"""
Query types for OpenBB Data Gateway.

Defines typed query parameters for all OpenBB data endpoints.
All queries are frozen dataclasses for immutability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from libra.gateways.fetcher import BaseQuery


# =============================================================================
# Equity Queries
# =============================================================================


@dataclass(frozen=True)
class EquityHistoricalQuery(BaseQuery):
    """
    Query for equity historical price data.

    Examples:
        # Basic query with defaults
        query = EquityHistoricalQuery(symbol="AAPL")

        # Full query with all parameters
        query = EquityHistoricalQuery(
            symbol="AAPL",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            interval="1d",
            provider="yfinance",
            adjustment="splits_and_dividends",
        )
    """

    symbol: str
    start_date: date | None = None
    end_date: date | None = None
    interval: str = "1d"  # 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1W, 1M
    provider: str = "yfinance"  # yfinance, fmp, polygon, intrinio, alpha_vantage
    adjustment: str = "splits_only"  # splits_only, splits_and_dividends, unadjusted
    extended_hours: bool = False


@dataclass(frozen=True)
class QuoteQuery(BaseQuery):
    """
    Query for current equity/crypto quote.

    Examples:
        query = QuoteQuery(symbol="AAPL", provider="yfinance")
    """

    symbol: str
    provider: str = "yfinance"


# =============================================================================
# Crypto Queries
# =============================================================================


@dataclass(frozen=True)
class CryptoHistoricalQuery(BaseQuery):
    """
    Query for cryptocurrency historical price data.

    Examples:
        query = CryptoHistoricalQuery(
            symbol="BTCUSD",
            start_date=date(2024, 1, 1),
            interval="1h",
            provider="fmp",
        )
    """

    symbol: str  # BTCUSD, ETHUSD, BTC-USD, etc.
    start_date: date | None = None
    end_date: date | None = None
    interval: str = "1d"  # 1m, 5m, 15m, 1h, 4h, 1d
    provider: str = "yfinance"  # yfinance, fmp, polygon, tiingo


# =============================================================================
# Fundamentals Queries
# =============================================================================


@dataclass(frozen=True)
class FundamentalsQuery(BaseQuery):
    """
    Query for company fundamental data.

    Examples:
        # Income statement
        query = FundamentalsQuery(
            symbol="AAPL",
            statement="income",
            period="quarter",
            limit=8,
        )

        # Balance sheet
        query = FundamentalsQuery(
            symbol="MSFT",
            statement="balance",
            period="annual",
        )

        # Key ratios
        query = FundamentalsQuery(
            symbol="GOOGL",
            statement="ratios",
        )
    """

    symbol: str
    statement: Literal[
        "income", "balance", "cash", "ratios", "metrics", "profile"
    ] = "income"
    period: Literal["annual", "quarter", "ttm"] = "annual"
    limit: int = 4
    provider: str = "fmp"  # fmp, polygon, intrinio, yfinance


# =============================================================================
# Options Queries
# =============================================================================


@dataclass(frozen=True)
class OptionsChainQuery(BaseQuery):
    """
    Query for options chain data with Greeks.

    Examples:
        # Get full chain
        query = OptionsChainQuery(symbol="AAPL")

        # Filter by expiration and type
        query = OptionsChainQuery(
            symbol="TSLA",
            expiration=date(2024, 3, 15),
            option_type="call",
        )
    """

    symbol: str
    expiration: date | None = None
    option_type: Literal["call", "put"] | None = None  # None = both
    moneyness: Literal["otm", "itm", "all"] = "all"
    strike_min: float | None = None
    strike_max: float | None = None
    provider: str = "cboe"  # cboe, tradier, intrinio, yfinance


# =============================================================================
# Economic Data Queries
# =============================================================================


@dataclass(frozen=True)
class EconomicSeriesQuery(BaseQuery):
    """
    Query for FRED economic data series.

    Examples:
        # GDP data
        query = EconomicSeriesQuery(series_id="GDP")

        # Inflation (CPI) with transformation
        query = EconomicSeriesQuery(
            series_id="CPIAUCSL",
            start_date=date(2020, 1, 1),
            transform="pc1",  # percent change
        )

        # Multiple series
        query = EconomicSeriesQuery(
            series_id="UNRATE,FEDFUNDS",
            frequency="monthly",
        )
    """

    series_id: str  # FRED series ID(s), comma-separated for multiple
    start_date: date | None = None
    end_date: date | None = None
    frequency: Literal[
        "daily", "weekly", "monthly", "quarterly", "annual"
    ] | None = None
    transform: Literal[
        "chg", "ch1", "pch", "pc1", "pca", "cch", "cca", "log"
    ] | None = None
    aggregation: Literal["avg", "sum", "eop"] | None = None  # end of period
    provider: str = "fred"


# =============================================================================
# News Queries
# =============================================================================


@dataclass(frozen=True)
class NewsQuery(BaseQuery):
    """
    Query for company or market news.

    Examples:
        query = NewsQuery(
            symbol="AAPL",
            limit=20,
            provider="fmp",
        )
    """

    symbol: str | None = None  # None = market news
    limit: int = 20
    start_date: date | None = None
    end_date: date | None = None
    provider: str = "fmp"  # fmp, benzinga, intrinio


# =============================================================================
# Search Queries
# =============================================================================


@dataclass(frozen=True)
class SymbolSearchQuery(BaseQuery):
    """
    Query for symbol search.

    Examples:
        query = SymbolSearchQuery(query="apple", provider="sec")
    """

    query: str
    provider: str = "sec"  # sec, nasdaq, intrinio
    limit: int = 10
