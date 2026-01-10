"""Tests for OpenBB fetchers."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from libra.gateways.openbb.fetchers import (
    CompanyProfile,
    EconomicDataPoint,
    FundamentalRecord,
    OpenBBCryptoBarFetcher,
    OpenBBEconomicFetcher,
    OpenBBEquityBarFetcher,
    OpenBBFundamentalsFetcher,
    OpenBBOptionsFetcher,
    OpenBBProfileFetcher,
    OpenBBQuoteFetcher,
    OptionContract,
    _to_date,
    _to_decimal,
)
from libra.gateways.openbb.queries import (
    CryptoHistoricalQuery,
    EconomicSeriesQuery,
    EquityHistoricalQuery,
    FundamentalsQuery,
    OptionsChainQuery,
    QuoteQuery,
)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_to_decimal_with_float(self) -> None:
        """Test converting float to Decimal."""
        result = _to_decimal(42.5)
        assert result == Decimal("42.5")

    def test_to_decimal_with_int(self) -> None:
        """Test converting int to Decimal."""
        result = _to_decimal(100)
        assert result == Decimal("100")

    def test_to_decimal_with_string(self) -> None:
        """Test converting string to Decimal."""
        result = _to_decimal("123.45")
        assert result == Decimal("123.45")

    def test_to_decimal_with_none(self) -> None:
        """Test None returns None."""
        result = _to_decimal(None)
        assert result is None

    def test_to_decimal_with_invalid(self) -> None:
        """Test invalid value returns None."""
        result = _to_decimal("not a number")
        assert result is None

    def test_to_date_with_date(self) -> None:
        """Test converting date to date."""
        d = date(2024, 1, 15)
        result = _to_date(d)
        assert result == d

    def test_to_date_with_datetime(self) -> None:
        """Test converting datetime to date."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = _to_date(dt)
        assert result == date(2024, 1, 15)

    def test_to_date_with_string(self) -> None:
        """Test converting string to date."""
        result = _to_date("2024-01-15")
        assert result == date(2024, 1, 15)

    def test_to_date_with_none(self) -> None:
        """Test None returns None."""
        result = _to_date(None)
        assert result is None


class TestOpenBBEquityBarFetcher:
    """Tests for OpenBBEquityBarFetcher."""

    def test_transform_query_basic(self) -> None:
        """Test basic query transformation."""
        fetcher = OpenBBEquityBarFetcher()

        query = fetcher.transform_query({"symbol": "AAPL"})

        assert isinstance(query, EquityHistoricalQuery)
        assert query.symbol == "AAPL"
        assert query.interval == "1d"
        assert query.provider == "yfinance"

    def test_transform_query_full(self) -> None:
        """Test full query transformation."""
        fetcher = OpenBBEquityBarFetcher()

        query = fetcher.transform_query({
            "symbol": "MSFT",
            "start_date": date(2024, 1, 1),
            "end_date": date(2024, 12, 31),
            "interval": "1h",
            "provider": "polygon",
            "adjustment": "splits_and_dividends",
            "extended_hours": True,
        })

        assert query.symbol == "MSFT"
        assert query.start_date == date(2024, 1, 1)
        assert query.interval == "1h"
        assert query.provider == "polygon"

    def test_transform_data_empty(self) -> None:
        """Test transforming empty data."""
        fetcher = OpenBBEquityBarFetcher()
        query = EquityHistoricalQuery(symbol="AAPL")

        result = fetcher.transform_data(query, None)
        assert result == []

    def test_transform_data_no_results(self) -> None:
        """Test transforming data with no results."""
        fetcher = OpenBBEquityBarFetcher()
        query = EquityHistoricalQuery(symbol="AAPL")

        mock_response = MagicMock()
        del mock_response.results  # No results attribute

        result = fetcher.transform_data(query, mock_response)
        assert result == []


class TestOpenBBCryptoBarFetcher:
    """Tests for OpenBBCryptoBarFetcher."""

    def test_transform_query(self) -> None:
        """Test query transformation."""
        fetcher = OpenBBCryptoBarFetcher()

        query = fetcher.transform_query({
            "symbol": "BTCUSD",
            "interval": "4h",
            "provider": "fmp",
        })

        assert isinstance(query, CryptoHistoricalQuery)
        assert query.symbol == "BTCUSD"
        assert query.interval == "4h"
        assert query.provider == "fmp"


class TestOpenBBQuoteFetcher:
    """Tests for OpenBBQuoteFetcher."""

    def test_transform_query(self) -> None:
        """Test query transformation."""
        fetcher = OpenBBQuoteFetcher()

        query = fetcher.transform_query({"symbol": "AAPL", "provider": "fmp"})

        assert isinstance(query, QuoteQuery)
        assert query.symbol == "AAPL"
        assert query.provider == "fmp"

    def test_transform_data_empty(self) -> None:
        """Test transforming empty quote data."""
        fetcher = OpenBBQuoteFetcher()
        query = QuoteQuery(symbol="AAPL")

        result = fetcher.transform_data(query, None)
        assert result is None


class TestOpenBBFundamentalsFetcher:
    """Tests for OpenBBFundamentalsFetcher."""

    def test_transform_query_income(self) -> None:
        """Test income statement query transformation."""
        fetcher = OpenBBFundamentalsFetcher()

        query = fetcher.transform_query({
            "symbol": "AAPL",
            "statement": "income",
            "period": "quarter",
            "limit": 8,
        })

        assert isinstance(query, FundamentalsQuery)
        assert query.symbol == "AAPL"
        assert query.statement == "income"
        assert query.period == "quarter"
        assert query.limit == 8

    def test_transform_query_balance(self) -> None:
        """Test balance sheet query transformation."""
        fetcher = OpenBBFundamentalsFetcher()

        query = fetcher.transform_query({
            "symbol": "MSFT",
            "statement": "balance",
        })

        assert query.statement == "balance"

    def test_transform_data_empty(self) -> None:
        """Test transforming empty fundamentals data."""
        fetcher = OpenBBFundamentalsFetcher()
        query = FundamentalsQuery(symbol="AAPL")

        result = fetcher.transform_data(query, None)
        assert result == []


class TestOpenBBOptionsFetcher:
    """Tests for OpenBBOptionsFetcher."""

    def test_transform_query_basic(self) -> None:
        """Test basic options query transformation."""
        fetcher = OpenBBOptionsFetcher()

        query = fetcher.transform_query({"symbol": "AAPL"})

        assert isinstance(query, OptionsChainQuery)
        assert query.symbol == "AAPL"
        assert query.provider == "cboe"

    def test_transform_query_filtered(self) -> None:
        """Test filtered options query."""
        fetcher = OpenBBOptionsFetcher()

        query = fetcher.transform_query({
            "symbol": "TSLA",
            "expiration": date(2024, 3, 15),
            "option_type": "call",
            "strike_min": 200.0,
            "strike_max": 300.0,
        })

        assert query.symbol == "TSLA"
        assert query.expiration == date(2024, 3, 15)
        assert query.option_type == "call"
        assert query.strike_min == 200.0
        assert query.strike_max == 300.0


class TestOpenBBEconomicFetcher:
    """Tests for OpenBBEconomicFetcher."""

    def test_transform_query_basic(self) -> None:
        """Test basic economic query transformation."""
        fetcher = OpenBBEconomicFetcher()

        query = fetcher.transform_query({"series_id": "GDP"})

        assert isinstance(query, EconomicSeriesQuery)
        assert query.series_id == "GDP"
        assert query.provider == "fred"

    def test_transform_query_with_transform(self) -> None:
        """Test query with data transformation."""
        fetcher = OpenBBEconomicFetcher()

        query = fetcher.transform_query({
            "series_id": "CPIAUCSL",
            "transform": "pc1",
            "frequency": "monthly",
        })

        assert query.series_id == "CPIAUCSL"
        assert query.transform == "pc1"
        assert query.frequency == "monthly"


class TestDataModels:
    """Tests for data model classes."""

    def test_fundamental_record(self) -> None:
        """Test FundamentalRecord creation."""
        record = FundamentalRecord(
            period="Q1 2024",
            revenue=Decimal("100000000000"),
            net_income=Decimal("25000000000"),
            eps=Decimal("1.53"),
        )

        assert record.period == "Q1 2024"
        assert record.revenue == Decimal("100000000000")
        assert record.net_income == Decimal("25000000000")
        assert record.eps == Decimal("1.53")

    def test_option_contract(self) -> None:
        """Test OptionContract creation."""
        contract = OptionContract(
            contract_symbol="AAPL240315C00190000",
            underlying="AAPL",
            expiration=date(2024, 3, 15),
            strike=Decimal("190.00"),
            option_type="call",
            bid=Decimal("5.50"),
            ask=Decimal("5.60"),
            delta=Decimal("0.65"),
            gamma=Decimal("0.02"),
            theta=Decimal("-0.05"),
            vega=Decimal("0.15"),
        )

        assert contract.contract_symbol == "AAPL240315C00190000"
        assert contract.strike == Decimal("190.00")
        assert contract.option_type == "call"
        assert contract.delta == Decimal("0.65")

    def test_economic_data_point(self) -> None:
        """Test EconomicDataPoint creation."""
        point = EconomicDataPoint(
            date=date(2024, 1, 1),
            value=Decimal("28000.5"),
            series_id="GDP",
        )

        assert point.date == date(2024, 1, 1)
        assert point.value == Decimal("28000.5")
        assert point.series_id == "GDP"

    def test_company_profile(self) -> None:
        """Test CompanyProfile creation."""
        profile = CompanyProfile(
            symbol="AAPL",
            name="Apple Inc.",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=Decimal("3000000000000"),
            pe_ratio=Decimal("28.5"),
            dividend_yield=Decimal("0.005"),
        )

        assert profile.symbol == "AAPL"
        assert profile.name == "Apple Inc."
        assert profile.market_cap == Decimal("3000000000000")
        assert profile.pe_ratio == Decimal("28.5")


class TestFetcherRegistry:
    """Tests for fetcher registration."""

    def test_fetchers_registered(self) -> None:
        """Test that all fetchers are registered."""
        from libra.gateways.fetcher import fetcher_registry

        # Check OpenBB fetchers are registered
        assert fetcher_registry.get("openbb", "equity_bar") is not None
        assert fetcher_registry.get("openbb", "crypto_bar") is not None
        assert fetcher_registry.get("openbb", "quote") is not None
        assert fetcher_registry.get("openbb", "fundamentals") is not None
        assert fetcher_registry.get("openbb", "options") is not None
        assert fetcher_registry.get("openbb", "economic") is not None
        assert fetcher_registry.get("openbb", "profile") is not None

    def test_openbb_in_gateway_list(self) -> None:
        """Test OpenBB is in gateway list."""
        from libra.gateways.fetcher import fetcher_registry

        gateways = fetcher_registry.list_gateways()
        assert "openbb" in gateways

    def test_openbb_data_types(self) -> None:
        """Test OpenBB data types are listed."""
        from libra.gateways.fetcher import fetcher_registry

        data_types = fetcher_registry.list_data_types("openbb")
        assert "equity_bar" in data_types
        assert "crypto_bar" in data_types
        assert "fundamentals" in data_types
