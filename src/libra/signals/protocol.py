"""
Whale Activity Detection Protocol.

Defines signal types and data structures for whale activity detection.

Signal Types:
- Order flow signals (from exchange order books)
- On-chain signals (from blockchain data)
- Aggregated signals (combined from multiple sources)

See: https://github.com/windoliver/libra/issues/38
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

import time


# =============================================================================
# Signal Types
# =============================================================================


class WhaleSignalType(str, Enum):
    """
    Types of whale activity signals.

    Crypto Order Flow (Tier 1 - Free, uses CCXT):
    - ORDER_IMBALANCE: Strong bid/ask volume imbalance
    - LARGE_WALL: Big order at single price point
    - LADDER_WALL: Distributed orders at sequential prices
    - VOLUME_SPIKE: Unusual trading volume
    - LARGE_TRADE: Single large trade execution

    Crypto On-Chain (Tier 2 - Optional):
    - EXCHANGE_INFLOW: Tokens moving to exchange (potential sell)
    - EXCHANGE_OUTFLOW: Tokens leaving exchange (accumulation)
    - WHALE_TRANSFER: Large transfer between wallets
    - DORMANT_ACTIVATION: Old wallet suddenly active
    - MINT_BURN: Large token mint or burn event

    Prediction Markets:
    - PM_LARGE_BET: Large bet on prediction market outcome
    - PM_POSITION_CHANGE: Whale changing position significantly
    - PM_MARKET_MOVE: Single trade moving market price significantly
    - PM_SMART_MONEY: Known smart money wallet activity

    Stocks/Equities:
    - OPTIONS_UNUSUAL: Unusual options activity (volume, OI)
    - OPTIONS_SWEEP: Large options sweep order
    - DARK_POOL: Large dark pool transaction
    - BLOCK_TRADE: Large block trade on exchange
    - INSIDER_FILING: Insider buying/selling (Form 4)
    - INST_13F: Institutional position change (13F filing)
    """

    # Tier 1: Crypto Order Flow (Free)
    ORDER_IMBALANCE = "order_imbalance"
    LARGE_WALL = "large_wall"
    LADDER_WALL = "ladder_wall"
    VOLUME_SPIKE = "volume_spike"
    LARGE_TRADE = "large_trade"

    # Tier 2: Crypto On-Chain (Optional)
    EXCHANGE_INFLOW = "exchange_inflow"
    EXCHANGE_OUTFLOW = "exchange_outflow"
    WHALE_TRANSFER = "whale_transfer"
    DORMANT_ACTIVATION = "dormant_activation"
    MINT_BURN = "mint_burn"

    # Prediction Markets
    PM_LARGE_BET = "pm_large_bet"
    PM_POSITION_CHANGE = "pm_position_change"
    PM_MARKET_MOVE = "pm_market_move"
    PM_SMART_MONEY = "pm_smart_money"

    # Stocks/Equities
    OPTIONS_UNUSUAL = "options_unusual"
    OPTIONS_SWEEP = "options_sweep"
    DARK_POOL = "dark_pool"
    BLOCK_TRADE = "block_trade"
    INSIDER_FILING = "insider_filing"
    INST_13F = "inst_13f"


class SignalDirection(str, Enum):
    """Direction implied by the signal."""

    BULLISH = "bullish"    # Likely price increase
    BEARISH = "bearish"    # Likely price decrease
    NEUTRAL = "neutral"    # Unclear direction


class SignalSource(str, Enum):
    """Source of the whale signal."""

    # Crypto
    ORDER_BOOK = "order_book"        # Exchange order book analysis
    TRADES = "trades"                # Trade tape analysis
    WHALE_ALERT = "whale_alert"      # Whale Alert API
    DUNE = "dune"                    # Dune Analytics
    NANSEN = "nansen"                # Nansen API
    GLASSNODE = "glassnode"          # Glassnode API

    # Prediction Markets
    POLYMARKET = "polymarket"        # Polymarket API
    KALSHI = "kalshi"                # Kalshi API
    METACULUS = "metaculus"          # Metaculus API
    MANIFOLD = "manifold"            # Manifold Markets

    # Stocks/Equities
    UNUSUAL_WHALES = "unusual_whales"  # Unusual Whales API
    FINRA = "finra"                  # FINRA dark pool data
    SEC_EDGAR = "sec_edgar"          # SEC EDGAR filings
    OPTIONS_FLOW = "options_flow"    # Options flow data
    OPENBB = "openbb"                # OpenBB data

    CUSTOM = "custom"                # Custom data source


class AssetClass(str, Enum):
    """Asset class for whale signals."""

    CRYPTO = "crypto"
    PREDICTION_MARKET = "prediction_market"
    STOCK = "stock"
    OPTIONS = "options"
    FUTURES = "futures"


# =============================================================================
# Thresholds Configuration
# =============================================================================


@dataclass
class WhaleThresholds:
    """
    Configurable thresholds for whale detection.

    Adjust based on market conditions and asset.
    """

    # Order book imbalance threshold (0.3 = 30% more volume on one side)
    imbalance_threshold: float = 0.3

    # Minimum wall size as percentage of total visible book
    wall_pct_threshold: float = 0.01  # 1%

    # Minimum wall size in quote currency (e.g., USDT)
    wall_min_value: Decimal = Decimal("50000")

    # Volume spike multiplier (vs 20-period average)
    volume_spike_multiplier: float = 3.0

    # Large trade threshold in quote currency
    large_trade_min_value: Decimal = Decimal("100000")

    # On-chain thresholds (USD)
    whale_transfer_min_usd: Decimal = Decimal("1000000")  # $1M
    exchange_flow_min_usd: Decimal = Decimal("500000")    # $500K

    # Dormant wallet activation threshold (days inactive)
    dormant_days_threshold: int = 365

    # Number of order book levels to analyze
    orderbook_depth: int = 20

    @classmethod
    def for_btc(cls) -> WhaleThresholds:
        """Thresholds tuned for BTC markets."""
        return cls(
            wall_min_value=Decimal("100000"),
            large_trade_min_value=Decimal("500000"),
            whale_transfer_min_usd=Decimal("10000000"),
        )

    @classmethod
    def for_eth(cls) -> WhaleThresholds:
        """Thresholds tuned for ETH markets."""
        return cls(
            wall_min_value=Decimal("50000"),
            large_trade_min_value=Decimal("250000"),
            whale_transfer_min_usd=Decimal("5000000"),
        )

    @classmethod
    def for_altcoin(cls) -> WhaleThresholds:
        """Thresholds for smaller altcoins."""
        return cls(
            wall_min_value=Decimal("10000"),
            large_trade_min_value=Decimal("50000"),
            whale_transfer_min_usd=Decimal("500000"),
            imbalance_threshold=0.25,
        )


# =============================================================================
# Whale Signal
# =============================================================================


@dataclass
class WhaleSignal:
    """
    A detected whale activity signal.

    Attributes:
        signal_type: Type of whale activity detected
        symbol: Trading pair (e.g., "BTC/USDT")
        timestamp_ns: Detection time in nanoseconds
        strength: Signal confidence 0.0-1.0
        direction: Implied market direction
        value_usd: Estimated USD value involved
        source: Data source that generated signal
        metadata: Additional context (prices, addresses, etc.)

    Example:
        signal = WhaleSignal(
            signal_type=WhaleSignalType.LARGE_WALL,
            symbol="BTC/USDT",
            timestamp_ns=time.time_ns(),
            strength=0.85,
            direction=SignalDirection.BULLISH,
            value_usd=Decimal("2500000"),
            source=SignalSource.ORDER_BOOK,
            metadata={
                "side": "bid",
                "price": "42500.00",
                "size": "58.82",
                "pct_of_book": "3.2%",
            }
        )
    """

    signal_type: WhaleSignalType
    symbol: str
    timestamp_ns: int
    strength: float  # 0.0 to 1.0
    direction: SignalDirection
    value_usd: Decimal
    source: SignalSource
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate signal data."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be 0.0-1.0, got {self.strength}")

    @classmethod
    def create(
        cls,
        signal_type: WhaleSignalType,
        symbol: str,
        strength: float,
        direction: SignalDirection,
        value_usd: Decimal,
        source: SignalSource,
        metadata: dict[str, Any] | None = None,
    ) -> WhaleSignal:
        """Factory method with auto-timestamp."""
        return cls(
            signal_type=signal_type,
            symbol=symbol,
            timestamp_ns=time.time_ns(),
            strength=strength,
            direction=direction,
            value_usd=value_usd,
            source=source,
            metadata=metadata or {},
        )

    @property
    def timestamp_sec(self) -> float:
        """Timestamp in seconds."""
        return self.timestamp_ns / 1_000_000_000

    @property
    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        return self.direction == SignalDirection.BULLISH

    @property
    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        return self.direction == SignalDirection.BEARISH

    @property
    def is_tier1(self) -> bool:
        """Check if signal is from Tier 1 (free) sources."""
        return self.signal_type in {
            WhaleSignalType.ORDER_IMBALANCE,
            WhaleSignalType.LARGE_WALL,
            WhaleSignalType.LADDER_WALL,
            WhaleSignalType.VOLUME_SPIKE,
            WhaleSignalType.LARGE_TRADE,
        }

    @property
    def is_onchain(self) -> bool:
        """Check if signal is from on-chain data."""
        return self.signal_type in {
            WhaleSignalType.EXCHANGE_INFLOW,
            WhaleSignalType.EXCHANGE_OUTFLOW,
            WhaleSignalType.WHALE_TRANSFER,
            WhaleSignalType.DORMANT_ACTIVATION,
            WhaleSignalType.MINT_BURN,
        }

    @property
    def is_prediction_market(self) -> bool:
        """Check if signal is from prediction markets."""
        return self.signal_type in {
            WhaleSignalType.PM_LARGE_BET,
            WhaleSignalType.PM_POSITION_CHANGE,
            WhaleSignalType.PM_MARKET_MOVE,
            WhaleSignalType.PM_SMART_MONEY,
        }

    @property
    def is_stock(self) -> bool:
        """Check if signal is from stock/equity markets."""
        return self.signal_type in {
            WhaleSignalType.OPTIONS_UNUSUAL,
            WhaleSignalType.OPTIONS_SWEEP,
            WhaleSignalType.DARK_POOL,
            WhaleSignalType.BLOCK_TRADE,
            WhaleSignalType.INSIDER_FILING,
            WhaleSignalType.INST_13F,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "signal_type": self.signal_type.value,
            "symbol": self.symbol,
            "timestamp_ns": self.timestamp_ns,
            "strength": self.strength,
            "direction": self.direction.value,
            "value_usd": str(self.value_usd),
            "source": self.source.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WhaleSignal:
        """Create from dictionary."""
        return cls(
            signal_type=WhaleSignalType(data["signal_type"]),
            symbol=data["symbol"],
            timestamp_ns=data["timestamp_ns"],
            strength=data["strength"],
            direction=SignalDirection(data["direction"]),
            value_usd=Decimal(data["value_usd"]),
            source=SignalSource(data["source"]),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# On-Chain Transaction
# =============================================================================


@dataclass
class WhaleTransaction:
    """
    An on-chain whale transaction.

    Used by on-chain providers (Whale Alert, Dune, etc.).
    """

    tx_hash: str
    blockchain: str  # "ethereum", "bitcoin", "solana", etc.
    timestamp: int   # Unix timestamp
    from_address: str
    to_address: str
    amount: Decimal
    amount_usd: Decimal
    symbol: str      # Token symbol
    tx_type: str     # "transfer", "mint", "burn", etc.

    # Optional enrichment
    from_owner: str | None = None    # Known entity name
    to_owner: str | None = None      # Known entity name
    from_type: str | None = None     # "exchange", "whale", "contract", etc.
    to_type: str | None = None       # "exchange", "whale", "contract", etc.

    @property
    def is_exchange_inflow(self) -> bool:
        """Check if this is an exchange inflow."""
        return self.to_type == "exchange"

    @property
    def is_exchange_outflow(self) -> bool:
        """Check if this is an exchange outflow."""
        return self.from_type == "exchange"

    def to_whale_signal(self) -> WhaleSignal:
        """Convert to WhaleSignal."""
        # Determine signal type and direction
        if self.is_exchange_inflow:
            signal_type = WhaleSignalType.EXCHANGE_INFLOW
            direction = SignalDirection.BEARISH  # Potential sell
        elif self.is_exchange_outflow:
            signal_type = WhaleSignalType.EXCHANGE_OUTFLOW
            direction = SignalDirection.BULLISH  # Accumulation
        elif self.tx_type == "mint":
            signal_type = WhaleSignalType.MINT_BURN
            direction = SignalDirection.BEARISH  # Supply increase
        elif self.tx_type == "burn":
            signal_type = WhaleSignalType.MINT_BURN
            direction = SignalDirection.BULLISH  # Supply decrease
        else:
            signal_type = WhaleSignalType.WHALE_TRANSFER
            direction = SignalDirection.NEUTRAL

        # Calculate strength based on value
        strength = min(1.0, float(self.amount_usd) / 10_000_000)  # $10M = 1.0

        return WhaleSignal.create(
            signal_type=signal_type,
            symbol=self.symbol,
            strength=strength,
            direction=direction,
            value_usd=self.amount_usd,
            source=SignalSource.WHALE_ALERT,
            metadata={
                "tx_hash": self.tx_hash,
                "blockchain": self.blockchain,
                "from_address": self.from_address,
                "to_address": self.to_address,
                "from_owner": self.from_owner,
                "to_owner": self.to_owner,
                "amount": str(self.amount),
                "tx_type": self.tx_type,
            },
        )


# =============================================================================
# Order Book Analysis Result
# =============================================================================


@dataclass
class OrderBookAnalysis:
    """
    Result of order book analysis.

    Contains computed metrics for whale detection.
    """

    symbol: str
    timestamp_ns: int

    # Imbalance metrics
    bid_volume: Decimal        # Total bid volume in top N levels
    ask_volume: Decimal        # Total ask volume in top N levels
    imbalance_ratio: float     # (bid - ask) / (bid + ask), range -1 to 1

    # Wall detection
    largest_bid_wall: Decimal  # Largest single bid order value
    largest_ask_wall: Decimal  # Largest single ask order value
    bid_wall_price: Decimal | None = None
    ask_wall_price: Decimal | None = None

    # Book depth metrics
    total_book_value: Decimal = Decimal("0")
    bid_pct: float = 0.0       # Bid volume as % of total
    ask_pct: float = 0.0       # Ask volume as % of total

    # Spread metrics
    spread_bps: float = 0.0    # Spread in basis points
    mid_price: Decimal = Decimal("0")

    @property
    def has_strong_imbalance(self) -> bool:
        """Check if there's a significant order book imbalance."""
        return abs(self.imbalance_ratio) > 0.3

    @property
    def imbalance_direction(self) -> SignalDirection:
        """Get direction implied by imbalance."""
        if self.imbalance_ratio > 0.3:
            return SignalDirection.BULLISH
        elif self.imbalance_ratio < -0.3:
            return SignalDirection.BEARISH
        return SignalDirection.NEUTRAL


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "WhaleSignalType",
    "SignalDirection",
    "SignalSource",
    "AssetClass",
    # Config
    "WhaleThresholds",
    # Core types
    "WhaleSignal",
    "WhaleTransaction",
    "OrderBookAnalysis",
]
