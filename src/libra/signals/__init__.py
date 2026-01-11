"""
LIBRA Signals Module.

Whale activity detection and trading signals across multiple asset classes.

Crypto (Tier 1 - Free - CCXT Order Book):
- Order book imbalance detection
- Large wall detection (single-price whales)
- Ladder wall detection (distributed orders)
- Volume spike analysis
- Large trade detection

Crypto (Tier 2 - Optional - API Keys Required):
- Whale Alert API integration
- Dune Analytics queries
- Exchange inflow/outflow tracking

Prediction Markets:
- Large bets on Polymarket, Kalshi, etc.
- Market-moving trades
- Smart money wallet tracking
- Position change detection

Stocks/Equities:
- Unusual options activity
- Options sweeps
- Dark pool transactions
- Block trades
- Insider filings (Form 4)
- Institutional holdings (13F)

See: https://github.com/windoliver/libra/issues/38
"""

from libra.signals.protocol import (
    # Enums
    AssetClass,
    SignalDirection,
    SignalSource,
    WhaleSignalType,
    # Config
    WhaleThresholds,
    # Core types
    OrderBookAnalysis,
    WhaleSignal,
    WhaleTransaction,
)
from libra.signals.order_flow import OrderFlowAnalyzer
from libra.signals.aggregator import (
    AggregatorConfig,
    SignalStats,
    WhaleSignalAggregator,
    create_aggregator,
)

# Prediction market analyzer
from libra.signals.prediction_market import (
    PredictionMarketBet,
    PredictionMarketState,
    PredictionMarketThresholds,
    PredictionMarketWhaleAnalyzer,
    PolymarketProvider,
    create_demo_pm_signals,
)

# Stock whale analyzer
from libra.signals.stock_whale import (
    BlockTrade,
    DarkPoolTrade,
    InsiderTrade,
    InstitutionalHolding,
    OptionsFlow,
    OptionType,
    OrderType,
    InsiderTransactionType,
    SECEdgarProvider,
    StockWhaleAnalyzer,
    StockWhaleThresholds,
    UnusualWhalesProvider,
    create_demo_stock_signals,
)

# Optional on-chain providers (may require httpx)
try:
    from libra.signals.onchain import (
        BaseOnChainProvider,
        DuneConfig,
        DuneProvider,
        MockOnChainProvider,
        WhaleAlertConfig,
        WhaleAlertProvider,
    )
    ONCHAIN_AVAILABLE = True
except ImportError:
    ONCHAIN_AVAILABLE = False
    BaseOnChainProvider = None  # type: ignore
    DuneConfig = None  # type: ignore
    DuneProvider = None  # type: ignore
    MockOnChainProvider = None  # type: ignore
    WhaleAlertConfig = None  # type: ignore
    WhaleAlertProvider = None  # type: ignore


__all__ = [
    # Protocol
    "AssetClass",
    "SignalDirection",
    "SignalSource",
    "WhaleSignalType",
    "WhaleThresholds",
    "OrderBookAnalysis",
    "WhaleSignal",
    "WhaleTransaction",
    # Order Flow (Tier 1 - Crypto)
    "OrderFlowAnalyzer",
    # Aggregator
    "AggregatorConfig",
    "SignalStats",
    "WhaleSignalAggregator",
    "create_aggregator",
    # On-Chain (Tier 2 - Crypto)
    "ONCHAIN_AVAILABLE",
    "BaseOnChainProvider",
    "WhaleAlertConfig",
    "WhaleAlertProvider",
    "DuneConfig",
    "DuneProvider",
    "MockOnChainProvider",
    # Prediction Markets
    "PredictionMarketBet",
    "PredictionMarketState",
    "PredictionMarketThresholds",
    "PredictionMarketWhaleAnalyzer",
    "PolymarketProvider",
    "create_demo_pm_signals",
    # Stocks/Equities
    "BlockTrade",
    "DarkPoolTrade",
    "InsiderTrade",
    "InstitutionalHolding",
    "OptionsFlow",
    "OptionType",
    "OrderType",
    "InsiderTransactionType",
    "SECEdgarProvider",
    "StockWhaleAnalyzer",
    "StockWhaleThresholds",
    "UnusualWhalesProvider",
    "create_demo_stock_signals",
]
