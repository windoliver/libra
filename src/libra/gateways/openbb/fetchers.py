"""
OpenBB Data Fetchers.

Implements the GatewayFetcher TET pipeline pattern for OpenBB data endpoints.
Each fetcher handles a specific data type (equity, crypto, fundamentals, etc.).

Note: OpenBB's main interface is synchronous, so we wrap calls in run_in_executor
for async compatibility with LIBRA's gateway architecture.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import msgspec

from libra.gateways.fetcher import (
    Bar,
    GatewayFetcher,
    Quote,
    fetcher_registry,
    timestamp_to_ns,
)
from libra.gateways.openbb.queries import (
    CryptoHistoricalQuery,
    EconomicSeriesQuery,
    EquityHistoricalQuery,
    FundamentalsQuery,
    OptionsChainQuery,
    QuoteQuery,
)


if TYPE_CHECKING:
    pass


# =============================================================================
# Response Data Types (using msgspec.Struct for performance)
# =============================================================================


class FundamentalRecord(msgspec.Struct, frozen=True, gc=False):
    """Single fundamental data record (one period)."""

    period: str  # "Q1 2024", "FY 2023", etc.
    period_ending: date | None = None
    # Income statement fields
    revenue: Decimal | None = None
    cost_of_revenue: Decimal | None = None
    gross_profit: Decimal | None = None
    operating_expenses: Decimal | None = None
    operating_income: Decimal | None = None
    ebitda: Decimal | None = None
    net_income: Decimal | None = None
    eps: Decimal | None = None
    eps_diluted: Decimal | None = None
    # Balance sheet fields
    total_assets: Decimal | None = None
    total_liabilities: Decimal | None = None
    total_equity: Decimal | None = None
    cash_and_equivalents: Decimal | None = None
    total_debt: Decimal | None = None
    # Cash flow fields
    operating_cash_flow: Decimal | None = None
    capital_expenditure: Decimal | None = None
    free_cash_flow: Decimal | None = None
    # Ratios
    pe_ratio: Decimal | None = None
    pb_ratio: Decimal | None = None
    ps_ratio: Decimal | None = None
    roe: Decimal | None = None
    roa: Decimal | None = None
    debt_to_equity: Decimal | None = None
    current_ratio: Decimal | None = None
    quick_ratio: Decimal | None = None
    gross_margin: Decimal | None = None
    operating_margin: Decimal | None = None
    net_margin: Decimal | None = None


class OptionContract(msgspec.Struct, frozen=True, gc=False):
    """Single option contract with Greeks."""

    contract_symbol: str
    underlying: str
    expiration: date
    strike: Decimal
    option_type: str  # "call" or "put"
    bid: Decimal | None = None
    ask: Decimal | None = None
    last: Decimal | None = None
    volume: int | None = None
    open_interest: int | None = None
    implied_volatility: Decimal | None = None
    # Greeks
    delta: Decimal | None = None
    gamma: Decimal | None = None
    theta: Decimal | None = None
    vega: Decimal | None = None
    rho: Decimal | None = None
    # Additional
    in_the_money: bool | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    last_trade_time: datetime | None = None


class EconomicDataPoint(msgspec.Struct, frozen=True, gc=False):
    """Single economic data observation."""

    date: date
    value: Decimal
    series_id: str | None = None


class CompanyProfile(msgspec.Struct, frozen=True, gc=False):
    """Company profile/overview data."""

    symbol: str
    name: str
    exchange: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: Decimal | None = None
    employees: int | None = None
    description: str | None = None
    website: str | None = None
    ceo: str | None = None
    country: str | None = None
    currency: str | None = None
    # Price data
    price: Decimal | None = None
    change: Decimal | None = None
    change_percent: Decimal | None = None
    volume: int | None = None
    avg_volume: int | None = None
    # Valuation
    pe_ratio: Decimal | None = None
    forward_pe: Decimal | None = None
    peg_ratio: Decimal | None = None
    price_to_book: Decimal | None = None
    price_to_sales: Decimal | None = None
    # Dividends
    dividend_yield: Decimal | None = None
    dividend_per_share: Decimal | None = None
    ex_dividend_date: date | None = None
    # 52-week
    week_52_high: Decimal | None = None
    week_52_low: Decimal | None = None
    # Shares
    shares_outstanding: int | None = None
    shares_float: int | None = None
    beta: Decimal | None = None


# =============================================================================
# Utility Functions
# =============================================================================


def _to_decimal(value: Any) -> Decimal | None:
    """Safely convert value to Decimal."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _to_date(value: Any) -> date | None:
    """Safely convert value to date."""
    if value is None:
        return None
    # Check datetime first since datetime is a subclass of date
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


async def _run_sync(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run synchronous function in executor for async compatibility."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# =============================================================================
# Equity Historical Fetcher
# =============================================================================


class OpenBBEquityBarFetcher(GatewayFetcher[EquityHistoricalQuery, list[Bar]]):
    """
    Fetches historical equity price data via OpenBB.

    Supports providers: yfinance, fmp, polygon, intrinio, alpha_vantage, tiingo.

    Example:
        fetcher = OpenBBEquityBarFetcher()
        bars = await fetcher.fetch(
            symbol="AAPL",
            start_date=date(2024, 1, 1),
            interval="1d",
            provider="yfinance",
        )
    """

    def transform_query(self, params: dict[str, Any]) -> EquityHistoricalQuery:
        """Transform params dict to typed query."""
        return EquityHistoricalQuery(
            symbol=params["symbol"],
            start_date=params.get("start_date"),
            end_date=params.get("end_date"),
            interval=params.get("interval", "1d"),
            provider=params.get("provider", "yfinance"),
            adjustment=params.get("adjustment", "splits_only"),
            extended_hours=params.get("extended_hours", False),
        )

    async def extract_data(self, query: EquityHistoricalQuery, **kwargs: Any) -> Any:
        """Fetch raw data from OpenBB."""
        try:
            from openbb import obb
        except ImportError as e:
            raise ImportError(
                "OpenBB is not installed. Install with: pip install openbb openbb-yfinance"
            ) from e

        result = await _run_sync(
            obb.equity.price.historical,
            symbol=query.symbol,
            start_date=query.start_date,
            end_date=query.end_date,
            interval=query.interval,
            provider=query.provider,
        )
        return result

    def transform_data(
        self, query: EquityHistoricalQuery, raw: Any
    ) -> list[Bar]:
        """Transform OpenBB OBBject to list of Bar."""
        if raw is None or not hasattr(raw, "results"):
            return []

        bars = []
        for record in raw.results:
            # Handle different field names across providers
            record_dict = record.__dict__ if hasattr(record, "__dict__") else {}

            # Get timestamp
            ts = getattr(record, "date", None)
            if ts is None:
                continue

            bars.append(
                Bar(
                    symbol=query.symbol,
                    timestamp_ns=timestamp_to_ns(ts) if isinstance(ts, (int, float, datetime)) else int(datetime.combine(ts, datetime.min.time()).timestamp() * 1_000_000_000),
                    open=Decimal(str(getattr(record, "open", 0))),
                    high=Decimal(str(getattr(record, "high", 0))),
                    low=Decimal(str(getattr(record, "low", 0))),
                    close=Decimal(str(getattr(record, "close", 0))),
                    volume=Decimal(str(getattr(record, "volume", 0) or 0)),
                    interval=query.interval,
                    trades=getattr(record, "transactions", None),
                )
            )
        return bars


# =============================================================================
# Crypto Historical Fetcher
# =============================================================================


class OpenBBCryptoBarFetcher(GatewayFetcher[CryptoHistoricalQuery, list[Bar]]):
    """
    Fetches historical cryptocurrency price data via OpenBB.

    Supports providers: yfinance, fmp, polygon, tiingo.

    Example:
        fetcher = OpenBBCryptoBarFetcher()
        bars = await fetcher.fetch(
            symbol="BTCUSD",
            start_date=date(2024, 1, 1),
            interval="1h",
            provider="fmp",
        )
    """

    def transform_query(self, params: dict[str, Any]) -> CryptoHistoricalQuery:
        """Transform params dict to typed query."""
        return CryptoHistoricalQuery(
            symbol=params["symbol"],
            start_date=params.get("start_date"),
            end_date=params.get("end_date"),
            interval=params.get("interval", "1d"),
            provider=params.get("provider", "yfinance"),
        )

    async def extract_data(self, query: CryptoHistoricalQuery, **kwargs: Any) -> Any:
        """Fetch raw data from OpenBB."""
        try:
            from openbb import obb
        except ImportError as e:
            raise ImportError(
                "OpenBB is not installed. Install with: pip install openbb"
            ) from e

        result = await _run_sync(
            obb.crypto.price.historical,
            symbol=query.symbol,
            start_date=query.start_date,
            end_date=query.end_date,
            interval=query.interval,
            provider=query.provider,
        )
        return result

    def transform_data(
        self, query: CryptoHistoricalQuery, raw: Any
    ) -> list[Bar]:
        """Transform OpenBB OBBject to list of Bar."""
        if raw is None or not hasattr(raw, "results"):
            return []

        bars = []
        for record in raw.results:
            ts = getattr(record, "date", None)
            if ts is None:
                continue

            bars.append(
                Bar(
                    symbol=query.symbol,
                    timestamp_ns=timestamp_to_ns(ts) if isinstance(ts, (int, float, datetime)) else int(datetime.combine(ts, datetime.min.time()).timestamp() * 1_000_000_000),
                    open=Decimal(str(getattr(record, "open", 0))),
                    high=Decimal(str(getattr(record, "high", 0))),
                    low=Decimal(str(getattr(record, "low", 0))),
                    close=Decimal(str(getattr(record, "close", 0))),
                    volume=Decimal(str(getattr(record, "volume", 0) or 0)),
                    interval=query.interval,
                )
            )
        return bars


# =============================================================================
# Quote Fetcher
# =============================================================================


class OpenBBQuoteFetcher(GatewayFetcher[QuoteQuery, Quote | None]):
    """
    Fetches current quote data via OpenBB.

    Example:
        fetcher = OpenBBQuoteFetcher()
        quote = await fetcher.fetch(symbol="AAPL", provider="yfinance")
    """

    def transform_query(self, params: dict[str, Any]) -> QuoteQuery:
        """Transform params dict to typed query."""
        return QuoteQuery(
            symbol=params["symbol"],
            provider=params.get("provider", "yfinance"),
        )

    async def extract_data(self, query: QuoteQuery, **kwargs: Any) -> Any:
        """Fetch raw data from OpenBB."""
        try:
            from openbb import obb
        except ImportError as e:
            raise ImportError(
                "OpenBB is not installed. Install with: pip install openbb"
            ) from e

        result = await _run_sync(
            obb.equity.price.quote,
            symbol=query.symbol,
            provider=query.provider,
        )
        return result

    def transform_data(self, query: QuoteQuery, raw: Any) -> Quote | None:
        """Transform OpenBB OBBject to Quote."""
        if raw is None or not hasattr(raw, "results") or not raw.results:
            return None

        record = raw.results[0] if isinstance(raw.results, list) else raw.results
        import time

        return Quote(
            symbol=query.symbol,
            bid=_to_decimal(getattr(record, "bid", 0)) or Decimal("0"),
            ask=_to_decimal(getattr(record, "ask", 0)) or Decimal("0"),
            last=_to_decimal(getattr(record, "last_price", None) or getattr(record, "price", 0)) or Decimal("0"),
            timestamp_ns=time.time_ns(),
            bid_size=_to_decimal(getattr(record, "bid_size", None)),
            ask_size=_to_decimal(getattr(record, "ask_size", None)),
            volume_24h=_to_decimal(getattr(record, "volume", None)),
            high_24h=_to_decimal(getattr(record, "high", None)),
            low_24h=_to_decimal(getattr(record, "low", None)),
            change_24h_pct=_to_decimal(getattr(record, "change_percent", None)),
        )


# =============================================================================
# Fundamentals Fetcher
# =============================================================================


class OpenBBFundamentalsFetcher(
    GatewayFetcher[FundamentalsQuery, list[FundamentalRecord]]
):
    """
    Fetches company fundamental data via OpenBB.

    Supports statements: income, balance, cash, ratios, metrics, profile.
    Supports providers: fmp, polygon, intrinio, yfinance.

    Example:
        fetcher = OpenBBFundamentalsFetcher()
        income = await fetcher.fetch(
            symbol="AAPL",
            statement="income",
            period="quarter",
            limit=8,
            provider="fmp",
        )
    """

    def transform_query(self, params: dict[str, Any]) -> FundamentalsQuery:
        """Transform params dict to typed query."""
        return FundamentalsQuery(
            symbol=params["symbol"],
            statement=params.get("statement", "income"),
            period=params.get("period", "annual"),
            limit=params.get("limit", 4),
            provider=params.get("provider", "fmp"),
        )

    async def extract_data(self, query: FundamentalsQuery, **kwargs: Any) -> Any:
        """Fetch raw data from OpenBB."""
        try:
            from openbb import obb
        except ImportError as e:
            raise ImportError(
                "OpenBB is not installed. Install with: pip install openbb openbb-fmp"
            ) from e

        # Map statement type to OpenBB endpoint
        if query.statement == "income":
            result = await _run_sync(
                obb.equity.fundamental.income,
                symbol=query.symbol,
                period=query.period,
                limit=query.limit,
                provider=query.provider,
            )
        elif query.statement == "balance":
            result = await _run_sync(
                obb.equity.fundamental.balance,
                symbol=query.symbol,
                period=query.period,
                limit=query.limit,
                provider=query.provider,
            )
        elif query.statement == "cash":
            result = await _run_sync(
                obb.equity.fundamental.cash,
                symbol=query.symbol,
                period=query.period,
                limit=query.limit,
                provider=query.provider,
            )
        elif query.statement == "ratios":
            result = await _run_sync(
                obb.equity.fundamental.ratios,
                symbol=query.symbol,
                period=query.period,
                limit=query.limit,
                provider=query.provider,
            )
        elif query.statement == "metrics":
            result = await _run_sync(
                obb.equity.fundamental.metrics,
                symbol=query.symbol,
                provider=query.provider,
            )
        elif query.statement == "profile":
            result = await _run_sync(
                obb.equity.profile,
                symbol=query.symbol,
                provider=query.provider,
            )
        else:
            raise ValueError(f"Unknown statement type: {query.statement}")

        return result

    def transform_data(
        self, query: FundamentalsQuery, raw: Any
    ) -> list[FundamentalRecord]:
        """Transform OpenBB OBBject to list of FundamentalRecord."""
        if raw is None or not hasattr(raw, "results"):
            return []

        results = raw.results if isinstance(raw.results, list) else [raw.results]
        records = []

        for record in results:
            # Build period string
            period_date = getattr(record, "period_ending", None) or getattr(
                record, "date", None
            )
            fiscal_period = getattr(record, "fiscal_period", None) or getattr(
                record, "period", ""
            )
            period_str = str(fiscal_period)
            if period_date:
                period_str = f"{period_str} {period_date}"

            records.append(
                FundamentalRecord(
                    period=period_str,
                    period_ending=_to_date(period_date),
                    # Income
                    revenue=_to_decimal(getattr(record, "revenue", None)),
                    cost_of_revenue=_to_decimal(
                        getattr(record, "cost_of_revenue", None)
                    ),
                    gross_profit=_to_decimal(getattr(record, "gross_profit", None)),
                    operating_expenses=_to_decimal(
                        getattr(record, "operating_expenses", None)
                    ),
                    operating_income=_to_decimal(
                        getattr(record, "operating_income", None)
                    ),
                    ebitda=_to_decimal(getattr(record, "ebitda", None)),
                    net_income=_to_decimal(getattr(record, "net_income", None)),
                    eps=_to_decimal(getattr(record, "eps", None) or getattr(record, "basic_eps", None)),
                    eps_diluted=_to_decimal(getattr(record, "eps_diluted", None) or getattr(record, "diluted_eps", None)),
                    # Balance
                    total_assets=_to_decimal(getattr(record, "total_assets", None)),
                    total_liabilities=_to_decimal(
                        getattr(record, "total_liabilities", None)
                    ),
                    total_equity=_to_decimal(
                        getattr(record, "total_equity", None)
                        or getattr(record, "total_stockholders_equity", None)
                    ),
                    cash_and_equivalents=_to_decimal(
                        getattr(record, "cash_and_cash_equivalents", None)
                    ),
                    total_debt=_to_decimal(getattr(record, "total_debt", None)),
                    # Cash flow
                    operating_cash_flow=_to_decimal(
                        getattr(record, "operating_cash_flow", None)
                    ),
                    capital_expenditure=_to_decimal(
                        getattr(record, "capital_expenditure", None)
                    ),
                    free_cash_flow=_to_decimal(
                        getattr(record, "free_cash_flow", None)
                    ),
                    # Ratios
                    pe_ratio=_to_decimal(
                        getattr(record, "pe_ratio", None)
                        or getattr(record, "price_earnings_ratio", None)
                    ),
                    pb_ratio=_to_decimal(
                        getattr(record, "pb_ratio", None)
                        or getattr(record, "price_to_book", None)
                    ),
                    ps_ratio=_to_decimal(
                        getattr(record, "ps_ratio", None)
                        or getattr(record, "price_to_sales", None)
                    ),
                    roe=_to_decimal(
                        getattr(record, "roe", None)
                        or getattr(record, "return_on_equity", None)
                    ),
                    roa=_to_decimal(
                        getattr(record, "roa", None)
                        or getattr(record, "return_on_assets", None)
                    ),
                    debt_to_equity=_to_decimal(
                        getattr(record, "debt_to_equity", None)
                    ),
                    current_ratio=_to_decimal(getattr(record, "current_ratio", None)),
                    quick_ratio=_to_decimal(getattr(record, "quick_ratio", None)),
                    gross_margin=_to_decimal(
                        getattr(record, "gross_profit_margin", None)
                        or getattr(record, "gross_margin", None)
                    ),
                    operating_margin=_to_decimal(
                        getattr(record, "operating_income_margin", None)
                        or getattr(record, "operating_margin", None)
                    ),
                    net_margin=_to_decimal(
                        getattr(record, "net_income_margin", None)
                        or getattr(record, "net_margin", None)
                        or getattr(record, "profit_margin", None)
                    ),
                )
            )

        return records


# =============================================================================
# Options Chain Fetcher
# =============================================================================


class OpenBBOptionsFetcher(
    GatewayFetcher[OptionsChainQuery, list[OptionContract]]
):
    """
    Fetches options chain data with Greeks via OpenBB.

    Supports providers: cboe (default), tradier, intrinio, yfinance.

    Example:
        fetcher = OpenBBOptionsFetcher()
        options = await fetcher.fetch(
            symbol="AAPL",
            expiration=date(2024, 3, 15),
            option_type="call",
            provider="cboe",
        )
    """

    def transform_query(self, params: dict[str, Any]) -> OptionsChainQuery:
        """Transform params dict to typed query."""
        return OptionsChainQuery(
            symbol=params["symbol"],
            expiration=params.get("expiration"),
            option_type=params.get("option_type"),
            moneyness=params.get("moneyness", "all"),
            strike_min=params.get("strike_min"),
            strike_max=params.get("strike_max"),
            provider=params.get("provider", "cboe"),
        )

    async def extract_data(self, query: OptionsChainQuery, **kwargs: Any) -> Any:
        """Fetch raw data from OpenBB."""
        try:
            from openbb import obb
        except ImportError as e:
            raise ImportError(
                "OpenBB is not installed. Install with: pip install openbb"
            ) from e

        # Build kwargs for OpenBB call
        call_kwargs: dict[str, Any] = {
            "symbol": query.symbol,
            "provider": query.provider,
        }

        if query.option_type:
            call_kwargs["option_type"] = query.option_type
        if query.moneyness and query.moneyness != "all":
            call_kwargs["moneyness"] = query.moneyness

        result = await _run_sync(obb.derivatives.options.chains, **call_kwargs)
        return result

    def transform_data(
        self, query: OptionsChainQuery, raw: Any
    ) -> list[OptionContract]:
        """Transform OpenBB OBBject to list of OptionContract."""
        if raw is None or not hasattr(raw, "results"):
            return []

        results = raw.results if isinstance(raw.results, list) else [raw.results]
        contracts = []

        for record in results:
            expiration_raw = getattr(record, "expiration", None)
            expiration = _to_date(expiration_raw)
            if expiration is None:
                continue

            # Apply strike filters
            strike = _to_decimal(getattr(record, "strike", 0)) or Decimal("0")
            if query.strike_min and float(strike) < query.strike_min:
                continue
            if query.strike_max and float(strike) > query.strike_max:
                continue

            # Apply expiration filter
            if query.expiration and expiration != query.expiration:
                continue

            contracts.append(
                OptionContract(
                    contract_symbol=getattr(record, "contract_symbol", ""),
                    underlying=query.symbol,
                    expiration=expiration,
                    strike=strike,
                    option_type=getattr(record, "option_type", ""),
                    bid=_to_decimal(getattr(record, "bid", None)),
                    ask=_to_decimal(getattr(record, "ask", None)),
                    last=_to_decimal(getattr(record, "last_price", None) or getattr(record, "last", None)),
                    volume=getattr(record, "volume", None),
                    open_interest=getattr(record, "open_interest", None),
                    implied_volatility=_to_decimal(getattr(record, "implied_volatility", None) or getattr(record, "iv", None)),
                    # Greeks
                    delta=_to_decimal(getattr(record, "delta", None)),
                    gamma=_to_decimal(getattr(record, "gamma", None)),
                    theta=_to_decimal(getattr(record, "theta", None)),
                    vega=_to_decimal(getattr(record, "vega", None)),
                    rho=_to_decimal(getattr(record, "rho", None)),
                    in_the_money=getattr(record, "in_the_money", None),
                    bid_size=getattr(record, "bid_size", None),
                    ask_size=getattr(record, "ask_size", None),
                )
            )

        return contracts


# =============================================================================
# Economic Data Fetcher
# =============================================================================


class OpenBBEconomicFetcher(
    GatewayFetcher[EconomicSeriesQuery, list[EconomicDataPoint]]
):
    """
    Fetches FRED economic data series via OpenBB.

    Example:
        fetcher = OpenBBEconomicFetcher()

        # GDP data
        gdp = await fetcher.fetch(series_id="GDP")

        # Inflation with percent change transformation
        cpi = await fetcher.fetch(
            series_id="CPIAUCSL",
            start_date=date(2020, 1, 1),
            transform="pc1",
        )
    """

    def transform_query(self, params: dict[str, Any]) -> EconomicSeriesQuery:
        """Transform params dict to typed query."""
        return EconomicSeriesQuery(
            series_id=params["series_id"],
            start_date=params.get("start_date"),
            end_date=params.get("end_date"),
            frequency=params.get("frequency"),
            transform=params.get("transform"),
            aggregation=params.get("aggregation"),
            provider=params.get("provider", "fred"),
        )

    async def extract_data(self, query: EconomicSeriesQuery, **kwargs: Any) -> Any:
        """Fetch raw data from OpenBB."""
        try:
            from openbb import obb
        except ImportError as e:
            raise ImportError(
                "OpenBB is not installed. Install with: pip install openbb openbb-fred"
            ) from e

        # Build kwargs
        call_kwargs: dict[str, Any] = {
            "symbol": query.series_id,
            "provider": query.provider,
        }
        if query.start_date:
            call_kwargs["start_date"] = query.start_date
        if query.end_date:
            call_kwargs["end_date"] = query.end_date
        if query.frequency:
            call_kwargs["frequency"] = query.frequency
        if query.transform:
            call_kwargs["transform"] = query.transform
        if query.aggregation:
            call_kwargs["aggregation_method"] = query.aggregation

        result = await _run_sync(obb.economy.fred_series, **call_kwargs)
        return result

    def transform_data(
        self, query: EconomicSeriesQuery, raw: Any
    ) -> list[EconomicDataPoint]:
        """Transform OpenBB OBBject to list of EconomicDataPoint."""
        if raw is None or not hasattr(raw, "results"):
            return []

        results = raw.results if isinstance(raw.results, list) else [raw.results]
        points = []

        for record in results:
            date_val = getattr(record, "date", None)
            value = getattr(record, "value", None)

            if date_val is None or value is None:
                continue

            points.append(
                EconomicDataPoint(
                    date=_to_date(date_val) or date_val,
                    value=_to_decimal(value) or Decimal("0"),
                    series_id=query.series_id.split(",")[0],  # First series if multiple
                )
            )

        return points


# =============================================================================
# Company Profile Fetcher
# =============================================================================


class OpenBBProfileFetcher(GatewayFetcher[QuoteQuery, CompanyProfile | None]):
    """
    Fetches company profile/overview data via OpenBB.

    Example:
        fetcher = OpenBBProfileFetcher()
        profile = await fetcher.fetch(symbol="AAPL", provider="fmp")
    """

    def transform_query(self, params: dict[str, Any]) -> QuoteQuery:
        """Transform params dict to typed query."""
        return QuoteQuery(
            symbol=params["symbol"],
            provider=params.get("provider", "fmp"),
        )

    async def extract_data(self, query: QuoteQuery, **kwargs: Any) -> Any:
        """Fetch raw data from OpenBB."""
        try:
            from openbb import obb
        except ImportError as e:
            raise ImportError(
                "OpenBB is not installed. Install with: pip install openbb openbb-fmp"
            ) from e

        result = await _run_sync(
            obb.equity.profile,
            symbol=query.symbol,
            provider=query.provider,
        )
        return result

    def transform_data(self, query: QuoteQuery, raw: Any) -> CompanyProfile | None:
        """Transform OpenBB OBBject to CompanyProfile."""
        if raw is None or not hasattr(raw, "results") or not raw.results:
            return None

        record = raw.results[0] if isinstance(raw.results, list) else raw.results

        return CompanyProfile(
            symbol=query.symbol,
            name=getattr(record, "name", "") or getattr(record, "company_name", ""),
            exchange=getattr(record, "exchange", None),
            sector=getattr(record, "sector", None),
            industry=getattr(record, "industry", None),
            market_cap=_to_decimal(getattr(record, "market_cap", None)),
            employees=getattr(record, "employees", None) or getattr(record, "full_time_employees", None),
            description=getattr(record, "description", None) or getattr(record, "long_business_summary", None),
            website=getattr(record, "website", None),
            ceo=getattr(record, "ceo", None),
            country=getattr(record, "country", None),
            currency=getattr(record, "currency", None),
            price=_to_decimal(getattr(record, "price", None)),
            change=_to_decimal(getattr(record, "change", None) or getattr(record, "changes", None)),
            change_percent=_to_decimal(getattr(record, "change_percent", None)),
            volume=getattr(record, "volume", None) or getattr(record, "avg_volume", None),
            avg_volume=getattr(record, "avg_volume", None),
            pe_ratio=_to_decimal(getattr(record, "pe", None) or getattr(record, "pe_ratio", None) or getattr(record, "trailing_pe", None)),
            forward_pe=_to_decimal(getattr(record, "forward_pe", None)),
            peg_ratio=_to_decimal(getattr(record, "peg_ratio", None)),
            price_to_book=_to_decimal(getattr(record, "price_to_book", None) or getattr(record, "pb_ratio", None)),
            price_to_sales=_to_decimal(getattr(record, "price_to_sales", None) or getattr(record, "ps_ratio", None)),
            dividend_yield=_to_decimal(getattr(record, "dividend_yield", None)),
            dividend_per_share=_to_decimal(getattr(record, "dividend_per_share", None) or getattr(record, "last_div", None)),
            ex_dividend_date=_to_date(getattr(record, "ex_dividend_date", None)),
            week_52_high=_to_decimal(getattr(record, "year_high", None) or getattr(record, "fifty_two_week_high", None)),
            week_52_low=_to_decimal(getattr(record, "year_low", None) or getattr(record, "fifty_two_week_low", None)),
            shares_outstanding=getattr(record, "shares_outstanding", None),
            shares_float=getattr(record, "shares_float", None) or getattr(record, "float_shares", None),
            beta=_to_decimal(getattr(record, "beta", None)),
        )


# =============================================================================
# Registry Registration
# =============================================================================


def register_openbb_fetchers() -> None:
    """Register all OpenBB fetchers with the global registry."""
    fetcher_registry.register("openbb", "equity_bar", OpenBBEquityBarFetcher)
    fetcher_registry.register("openbb", "crypto_bar", OpenBBCryptoBarFetcher)
    fetcher_registry.register("openbb", "quote", OpenBBQuoteFetcher)
    fetcher_registry.register("openbb", "fundamentals", OpenBBFundamentalsFetcher)
    fetcher_registry.register("openbb", "options", OpenBBOptionsFetcher)
    fetcher_registry.register("openbb", "economic", OpenBBEconomicFetcher)
    fetcher_registry.register("openbb", "profile", OpenBBProfileFetcher)


# Auto-register on import
register_openbb_fetchers()
