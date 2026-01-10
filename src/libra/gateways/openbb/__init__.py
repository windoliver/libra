"""
OpenBB Data Gateway for LIBRA.

Implements Issue #28: OpenBB Data Gateway Integration.

Provides unified access to 30+ data providers through OpenBB Platform:
- Equity prices (Yahoo Finance, FMP, Polygon, etc.)
- Crypto prices (Binance, CoinGecko, etc.)
- Fundamentals (SEC, FMP, Intrinio)
- Options chains with Greeks (CBOE, Tradier)
- Economic data (FRED, BLS, World Bank)

Usage:
    from libra.gateways.openbb import OpenBBGateway

    gateway = OpenBBGateway()
    await gateway.connect()

    # Fetch equity historical data
    bars = await gateway.get_equity_historical("AAPL", interval="1d")

    # Fetch fundamentals
    income = await gateway.get_fundamentals("AAPL", statement="income")

    # Fetch options chain
    options = await gateway.get_options_chain("AAPL")

    # Fetch FRED economic data
    gdp = await gateway.get_economic_series("GDP")
"""

from libra.gateways.openbb.gateway import (
    OpenBBCredentials,
    OpenBBGateway,
    OpenBBNotInstalledError,
    ProviderStatus,
)
from libra.gateways.openbb.queries import (
    CryptoHistoricalQuery,
    EconomicSeriesQuery,
    EquityHistoricalQuery,
    FundamentalsQuery,
    OptionsChainQuery,
    QuoteQuery,
)
from libra.gateways.openbb.fetchers import (
    OpenBBCryptoBarFetcher,
    OpenBBEconomicFetcher,
    OpenBBEquityBarFetcher,
    OpenBBFundamentalsFetcher,
    OpenBBOptionsFetcher,
    OpenBBQuoteFetcher,
    register_openbb_fetchers,
)

__all__ = [
    # Gateway
    "OpenBBGateway",
    "OpenBBCredentials",
    "OpenBBNotInstalledError",
    "ProviderStatus",
    # Queries
    "EquityHistoricalQuery",
    "CryptoHistoricalQuery",
    "FundamentalsQuery",
    "OptionsChainQuery",
    "EconomicSeriesQuery",
    "QuoteQuery",
    # Fetchers
    "OpenBBEquityBarFetcher",
    "OpenBBCryptoBarFetcher",
    "OpenBBFundamentalsFetcher",
    "OpenBBOptionsFetcher",
    "OpenBBEconomicFetcher",
    "OpenBBQuoteFetcher",
    "register_openbb_fetchers",
]
