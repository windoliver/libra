"""
Alpaca Gateway for US Stocks and Options.

Provides execution and market data via Alpaca's API-first brokerage:
- Commission-free stock/ETF trading
- Options trading (Level 1-3 including multi-leg)
- Paper trading for testing
- Real-time WebSocket streaming

Issue #61: Alpaca Gateway - Stock & Options Execution
"""

from libra.gateways.alpaca.config import AlpacaConfig, AlpacaCredentials
from libra.gateways.alpaca.gateway import AlpacaGateway, ALPACA_CAPABILITIES
from libra.gateways.alpaca.symbols import (
    to_occ_symbol,
    from_occ_symbol,
    normalize_symbol,
    is_option_symbol,
)

__all__ = [
    # Config
    "AlpacaConfig",
    "AlpacaCredentials",
    # Gateway
    "AlpacaGateway",
    "ALPACA_CAPABILITIES",
    # Symbols
    "to_occ_symbol",
    "from_occ_symbol",
    "normalize_symbol",
    "is_option_symbol",
]
