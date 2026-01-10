"""
OpenBB Data Gateway.

High-level gateway class that provides a unified interface to all OpenBB data.
This is a DATA-ONLY gateway - use CCXT gateway for order execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from libra.gateways.fetcher import Bar, Quote
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
)


logger = logging.getLogger(__name__)


class OpenBBNotInstalledError(Exception):
    """Raised when OpenBB is not installed."""

    def __init__(self, message: str | None = None) -> None:
        default_message = (
            "OpenBB Platform is not installed. Install with:\n"
            "  pip install openbb openbb-yfinance openbb-fmp\n"
            "Or for all providers:\n"
            "  pip install openbb[all]"
        )
        super().__init__(message or default_message)


@dataclass
class OpenBBCredentials:
    """Credentials for OpenBB data providers."""

    fmp_api_key: str | None = None
    polygon_api_key: str | None = None
    fred_api_key: str | None = None
    intrinio_api_key: str | None = None
    tradier_api_key: str | None = None
    alpha_vantage_api_key: str | None = None
    tiingo_api_key: str | None = None
    benzinga_api_key: str | None = None


@dataclass
class ProviderStatus:
    """Status of a data provider."""

    name: str
    available: bool
    has_credentials: bool
    rate_limit: str | None = None
    last_error: str | None = None


@dataclass
class OpenBBGateway:
    """
    OpenBB Data Gateway for LIBRA.

    Provides unified access to market data, fundamentals, options, and economic data
    through OpenBB Platform's 30+ data providers.

    This is a DATA-ONLY gateway. For trading/order execution, use CCXTGateway.

    Example:
        gateway = OpenBBGateway()
        await gateway.connect()

        # Equity data
        bars = await gateway.get_equity_historical("AAPL", interval="1d")
        profile = await gateway.get_company_profile("AAPL")

        # Fundamentals
        income = await gateway.get_fundamentals("AAPL", statement="income")

        # Options
        chain = await gateway.get_options_chain("AAPL")

        # Economic data
        gdp = await gateway.get_economic_series("GDP")
    """

    credentials: OpenBBCredentials = field(default_factory=OpenBBCredentials)
    default_equity_provider: str = "yfinance"
    default_crypto_provider: str = "yfinance"
    default_fundamentals_provider: str = "fmp"
    default_options_provider: str = "cboe"
    default_economic_provider: str = "fred"

    # Internal state
    _connected: bool = field(default=False, init=False)
    _openbb_available: bool = field(default=False, init=False)

    # Fetchers (lazy-initialized)
    _equity_fetcher: OpenBBEquityBarFetcher | None = field(default=None, init=False)
    _crypto_fetcher: OpenBBCryptoBarFetcher | None = field(default=None, init=False)
    _quote_fetcher: OpenBBQuoteFetcher | None = field(default=None, init=False)
    _fundamentals_fetcher: OpenBBFundamentalsFetcher | None = field(
        default=None, init=False
    )
    _options_fetcher: OpenBBOptionsFetcher | None = field(default=None, init=False)
    _economic_fetcher: OpenBBEconomicFetcher | None = field(default=None, init=False)
    _profile_fetcher: OpenBBProfileFetcher | None = field(default=None, init=False)

    async def connect(self) -> None:
        """
        Connect to OpenBB and verify availability.

        Sets up credentials and checks that OpenBB is installed.

        Raises:
            OpenBBNotInstalledError: If OpenBB is not installed.
        """
        try:
            from openbb import obb

            self._openbb_available = True

            # Set credentials if provided
            if self.credentials.fmp_api_key:
                obb.user.credentials.fmp_api_key = self.credentials.fmp_api_key
            if self.credentials.polygon_api_key:
                obb.user.credentials.polygon_api_key = self.credentials.polygon_api_key
            if self.credentials.fred_api_key:
                obb.user.credentials.fred_api_key = self.credentials.fred_api_key
            if self.credentials.intrinio_api_key:
                obb.user.credentials.intrinio_api_key = self.credentials.intrinio_api_key
            if self.credentials.tradier_api_key:
                obb.user.credentials.tradier_api_key = self.credentials.tradier_api_key
            if self.credentials.alpha_vantage_api_key:
                obb.user.credentials.alpha_vantage_api_key = (
                    self.credentials.alpha_vantage_api_key
                )
            if self.credentials.tiingo_api_key:
                obb.user.credentials.tiingo_api_key = self.credentials.tiingo_api_key
            if self.credentials.benzinga_api_key:
                obb.user.credentials.benzinga_api_key = self.credentials.benzinga_api_key

            # Initialize fetchers
            self._equity_fetcher = OpenBBEquityBarFetcher()
            self._crypto_fetcher = OpenBBCryptoBarFetcher()
            self._quote_fetcher = OpenBBQuoteFetcher()
            self._fundamentals_fetcher = OpenBBFundamentalsFetcher()
            self._options_fetcher = OpenBBOptionsFetcher()
            self._economic_fetcher = OpenBBEconomicFetcher()
            self._profile_fetcher = OpenBBProfileFetcher()

            self._connected = True
            logger.info("OpenBB gateway connected successfully")

        except ImportError as e:
            raise OpenBBNotInstalledError() from e

    async def disconnect(self) -> None:
        """Disconnect from OpenBB (cleanup)."""
        self._connected = False
        self._equity_fetcher = None
        self._crypto_fetcher = None
        self._quote_fetcher = None
        self._fundamentals_fetcher = None
        self._options_fetcher = None
        self._economic_fetcher = None
        self._profile_fetcher = None
        logger.info("OpenBB gateway disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if gateway is connected."""
        return self._connected

    def _ensure_connected(self) -> None:
        """Ensure gateway is connected."""
        if not self._connected:
            raise RuntimeError("OpenBB gateway not connected. Call connect() first.")

    # =========================================================================
    # Equity Data
    # =========================================================================

    async def get_equity_historical(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
        interval: str = "1d",
        provider: str | None = None,
        adjustment: str = "splits_only",
    ) -> list[Bar]:
        """
        Get historical equity price data.

        Args:
            symbol: Stock ticker (e.g., "AAPL", "MSFT")
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Bar interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1W, 1M)
            provider: Data provider (yfinance, fmp, polygon, etc.)
            adjustment: Price adjustment (splits_only, splits_and_dividends, unadjusted)

        Returns:
            List of Bar objects with OHLCV data
        """
        self._ensure_connected()
        assert self._equity_fetcher is not None

        return await self._equity_fetcher.fetch(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            provider=provider or self.default_equity_provider,
            adjustment=adjustment,
        )

    async def get_quote(
        self, symbol: str, provider: str | None = None
    ) -> Quote | None:
        """
        Get current quote for a symbol.

        Args:
            symbol: Stock ticker
            provider: Data provider

        Returns:
            Quote object or None if not available
        """
        self._ensure_connected()
        assert self._quote_fetcher is not None

        return await self._quote_fetcher.fetch(
            symbol=symbol,
            provider=provider or self.default_equity_provider,
        )

    async def get_company_profile(
        self, symbol: str, provider: str | None = None
    ) -> CompanyProfile | None:
        """
        Get company profile/overview.

        Args:
            symbol: Stock ticker
            provider: Data provider (fmp, yfinance)

        Returns:
            CompanyProfile object or None
        """
        self._ensure_connected()
        assert self._profile_fetcher is not None

        return await self._profile_fetcher.fetch(
            symbol=symbol,
            provider=provider or self.default_fundamentals_provider,
        )

    # =========================================================================
    # Crypto Data
    # =========================================================================

    async def get_crypto_historical(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
        interval: str = "1d",
        provider: str | None = None,
    ) -> list[Bar]:
        """
        Get historical cryptocurrency price data.

        Args:
            symbol: Crypto pair (e.g., "BTCUSD", "ETHUSD", "BTC-USD")
            start_date: Start date
            end_date: End date
            interval: Bar interval
            provider: Data provider (yfinance, fmp, polygon, tiingo)

        Returns:
            List of Bar objects
        """
        self._ensure_connected()
        assert self._crypto_fetcher is not None

        return await self._crypto_fetcher.fetch(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            provider=provider or self.default_crypto_provider,
        )

    # =========================================================================
    # Fundamentals
    # =========================================================================

    async def get_fundamentals(
        self,
        symbol: str,
        statement: str = "income",
        period: str = "annual",
        limit: int = 4,
        provider: str | None = None,
    ) -> list[FundamentalRecord]:
        """
        Get company fundamental data.

        Args:
            symbol: Stock ticker
            statement: Type of statement (income, balance, cash, ratios, metrics)
            period: Period type (annual, quarter, ttm)
            limit: Number of periods to return
            provider: Data provider (fmp, polygon, intrinio, yfinance)

        Returns:
            List of FundamentalRecord objects
        """
        self._ensure_connected()
        assert self._fundamentals_fetcher is not None

        return await self._fundamentals_fetcher.fetch(
            symbol=symbol,
            statement=statement,
            period=period,
            limit=limit,
            provider=provider or self.default_fundamentals_provider,
        )

    async def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 4,
        provider: str | None = None,
    ) -> list[FundamentalRecord]:
        """Get income statement data (convenience method)."""
        return await self.get_fundamentals(
            symbol=symbol, statement="income", period=period, limit=limit, provider=provider
        )

    async def get_balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 4,
        provider: str | None = None,
    ) -> list[FundamentalRecord]:
        """Get balance sheet data (convenience method)."""
        return await self.get_fundamentals(
            symbol=symbol, statement="balance", period=period, limit=limit, provider=provider
        )

    async def get_cash_flow(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 4,
        provider: str | None = None,
    ) -> list[FundamentalRecord]:
        """Get cash flow statement (convenience method)."""
        return await self.get_fundamentals(
            symbol=symbol, statement="cash", period=period, limit=limit, provider=provider
        )

    async def get_ratios(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 4,
        provider: str | None = None,
    ) -> list[FundamentalRecord]:
        """Get financial ratios (convenience method)."""
        return await self.get_fundamentals(
            symbol=symbol, statement="ratios", period=period, limit=limit, provider=provider
        )

    # =========================================================================
    # Options
    # =========================================================================

    async def get_options_chain(
        self,
        symbol: str,
        expiration: date | None = None,
        option_type: str | None = None,
        moneyness: str = "all",
        strike_min: float | None = None,
        strike_max: float | None = None,
        provider: str | None = None,
    ) -> list[OptionContract]:
        """
        Get options chain with Greeks.

        Args:
            symbol: Underlying symbol
            expiration: Filter by expiration date
            option_type: Filter by type ("call" or "put")
            moneyness: Filter by moneyness ("otm", "itm", "all")
            strike_min: Minimum strike price
            strike_max: Maximum strike price
            provider: Data provider (cboe, tradier, intrinio, yfinance)

        Returns:
            List of OptionContract objects with Greeks
        """
        self._ensure_connected()
        assert self._options_fetcher is not None

        return await self._options_fetcher.fetch(
            symbol=symbol,
            expiration=expiration,
            option_type=option_type,
            moneyness=moneyness,
            strike_min=strike_min,
            strike_max=strike_max,
            provider=provider or self.default_options_provider,
        )

    async def get_options_expirations(
        self, symbol: str, provider: str | None = None
    ) -> list[date]:
        """
        Get available option expiration dates.

        Args:
            symbol: Underlying symbol
            provider: Data provider

        Returns:
            List of available expiration dates
        """
        chain = await self.get_options_chain(
            symbol=symbol, provider=provider or self.default_options_provider
        )
        expirations = sorted(set(c.expiration for c in chain))
        return expirations

    # =========================================================================
    # Economic Data
    # =========================================================================

    async def get_economic_series(
        self,
        series_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
        frequency: str | None = None,
        transform: str | None = None,
        provider: str | None = None,
    ) -> list[EconomicDataPoint]:
        """
        Get FRED economic data series.

        Args:
            series_id: FRED series ID (e.g., "GDP", "UNRATE", "CPIAUCSL")
            start_date: Start date
            end_date: End date
            frequency: Aggregation frequency (daily, weekly, monthly, quarterly, annual)
            transform: Data transformation (chg, ch1, pch, pc1, pca, cch, cca, log)
            provider: Data provider (fred)

        Returns:
            List of EconomicDataPoint objects
        """
        self._ensure_connected()
        assert self._economic_fetcher is not None

        return await self._economic_fetcher.fetch(
            series_id=series_id,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            transform=transform,
            provider=provider or self.default_economic_provider,
        )

    # =========================================================================
    # Provider Status
    # =========================================================================

    def get_provider_status(self) -> list[ProviderStatus]:
        """
        Get status of available data providers.

        Returns:
            List of ProviderStatus for each provider
        """
        providers = []

        # Check each provider
        provider_configs = [
            ("yfinance", True, False, "2000/hour (unofficial)"),
            ("fmp", True, self.credentials.fmp_api_key is not None, "250/day free"),
            ("polygon", True, self.credentials.polygon_api_key is not None, "5/min free"),
            ("fred", True, self.credentials.fred_api_key is not None, "Unlimited"),
            ("cboe", True, False, "Unknown"),
            ("intrinio", True, self.credentials.intrinio_api_key is not None, "Varies"),
            ("tradier", True, self.credentials.tradier_api_key is not None, "Varies"),
            ("alpha_vantage", True, self.credentials.alpha_vantage_api_key is not None, "5/min free"),
            ("tiingo", True, self.credentials.tiingo_api_key is not None, "50/hr free"),
            ("benzinga", True, self.credentials.benzinga_api_key is not None, "Varies"),
        ]

        for name, available, has_creds, rate_limit in provider_configs:
            providers.append(
                ProviderStatus(
                    name=name,
                    available=available,
                    has_credentials=has_creds,
                    rate_limit=rate_limit,
                )
            )

        return providers

    def get_available_providers(self, data_type: str) -> list[str]:
        """
        Get providers available for a specific data type.

        Args:
            data_type: Type of data (equity, crypto, fundamentals, options, economic)

        Returns:
            List of provider names
        """
        provider_map = {
            "equity": ["yfinance", "fmp", "polygon", "intrinio", "alpha_vantage", "tiingo", "cboe", "tradier"],
            "crypto": ["yfinance", "fmp", "polygon", "tiingo"],
            "fundamentals": ["fmp", "polygon", "intrinio", "yfinance"],
            "options": ["cboe", "tradier", "intrinio", "yfinance"],
            "economic": ["fred"],
        }
        return provider_map.get(data_type, [])
