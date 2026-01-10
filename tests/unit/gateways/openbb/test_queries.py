"""Tests for OpenBB query types."""

from __future__ import annotations

from datetime import date

import pytest

from libra.gateways.openbb.queries import (
    CryptoHistoricalQuery,
    EconomicSeriesQuery,
    EquityHistoricalQuery,
    FundamentalsQuery,
    OptionsChainQuery,
    QuoteQuery,
)


class TestEquityHistoricalQuery:
    """Tests for EquityHistoricalQuery."""

    def test_basic_query(self) -> None:
        """Test creating a basic query."""
        query = EquityHistoricalQuery(symbol="AAPL")

        assert query.symbol == "AAPL"
        assert query.interval == "1d"
        assert query.provider == "yfinance"
        assert query.start_date is None
        assert query.end_date is None

    def test_full_query(self) -> None:
        """Test creating a query with all parameters."""
        query = EquityHistoricalQuery(
            symbol="MSFT",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            interval="1h",
            provider="polygon",
            adjustment="splits_and_dividends",
            extended_hours=True,
        )

        assert query.symbol == "MSFT"
        assert query.start_date == date(2024, 1, 1)
        assert query.end_date == date(2024, 12, 31)
        assert query.interval == "1h"
        assert query.provider == "polygon"
        assert query.adjustment == "splits_and_dividends"
        assert query.extended_hours is True

    def test_immutable(self) -> None:
        """Test query is immutable (frozen dataclass)."""
        query = EquityHistoricalQuery(symbol="AAPL")

        with pytest.raises(Exception):  # FrozenInstanceError
            query.symbol = "MSFT"  # type: ignore


class TestCryptoHistoricalQuery:
    """Tests for CryptoHistoricalQuery."""

    def test_basic_query(self) -> None:
        """Test creating a basic query."""
        query = CryptoHistoricalQuery(symbol="BTCUSD")

        assert query.symbol == "BTCUSD"
        assert query.interval == "1d"
        assert query.provider == "yfinance"

    def test_with_dates(self) -> None:
        """Test query with date range."""
        query = CryptoHistoricalQuery(
            symbol="ETHUSD",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            interval="4h",
            provider="fmp",
        )

        assert query.symbol == "ETHUSD"
        assert query.start_date == date(2024, 1, 1)
        assert query.interval == "4h"
        assert query.provider == "fmp"


class TestQuoteQuery:
    """Tests for QuoteQuery."""

    def test_basic_query(self) -> None:
        """Test creating a basic quote query."""
        query = QuoteQuery(symbol="AAPL")

        assert query.symbol == "AAPL"
        assert query.provider == "yfinance"

    def test_with_provider(self) -> None:
        """Test quote query with specific provider."""
        query = QuoteQuery(symbol="GOOGL", provider="fmp")

        assert query.symbol == "GOOGL"
        assert query.provider == "fmp"


class TestFundamentalsQuery:
    """Tests for FundamentalsQuery."""

    def test_default_values(self) -> None:
        """Test default values."""
        query = FundamentalsQuery(symbol="AAPL")

        assert query.symbol == "AAPL"
        assert query.statement == "income"
        assert query.period == "annual"
        assert query.limit == 4
        assert query.provider == "fmp"

    def test_balance_sheet(self) -> None:
        """Test balance sheet query."""
        query = FundamentalsQuery(
            symbol="MSFT",
            statement="balance",
            period="quarter",
            limit=8,
        )

        assert query.statement == "balance"
        assert query.period == "quarter"
        assert query.limit == 8

    def test_ratios(self) -> None:
        """Test ratios query."""
        query = FundamentalsQuery(
            symbol="GOOGL",
            statement="ratios",
        )

        assert query.statement == "ratios"


class TestOptionsChainQuery:
    """Tests for OptionsChainQuery."""

    def test_default_values(self) -> None:
        """Test default values."""
        query = OptionsChainQuery(symbol="AAPL")

        assert query.symbol == "AAPL"
        assert query.expiration is None
        assert query.option_type is None
        assert query.moneyness == "all"
        assert query.provider == "cboe"

    def test_filtered_query(self) -> None:
        """Test filtered options query."""
        query = OptionsChainQuery(
            symbol="TSLA",
            expiration=date(2024, 3, 15),
            option_type="call",
            moneyness="otm",
            strike_min=200.0,
            strike_max=300.0,
            provider="tradier",
        )

        assert query.expiration == date(2024, 3, 15)
        assert query.option_type == "call"
        assert query.moneyness == "otm"
        assert query.strike_min == 200.0
        assert query.strike_max == 300.0
        assert query.provider == "tradier"


class TestEconomicSeriesQuery:
    """Tests for EconomicSeriesQuery."""

    def test_basic_query(self) -> None:
        """Test basic FRED query."""
        query = EconomicSeriesQuery(series_id="GDP")

        assert query.series_id == "GDP"
        assert query.provider == "fred"
        assert query.frequency is None
        assert query.transform is None

    def test_with_transformation(self) -> None:
        """Test query with data transformation."""
        query = EconomicSeriesQuery(
            series_id="CPIAUCSL",
            start_date=date(2020, 1, 1),
            frequency="monthly",
            transform="pc1",
        )

        assert query.series_id == "CPIAUCSL"
        assert query.start_date == date(2020, 1, 1)
        assert query.frequency == "monthly"
        assert query.transform == "pc1"

    def test_multiple_series(self) -> None:
        """Test multiple series query."""
        query = EconomicSeriesQuery(
            series_id="UNRATE,FEDFUNDS",
            aggregation="avg",
        )

        assert query.series_id == "UNRATE,FEDFUNDS"
        assert query.aggregation == "avg"
