"""
Stock/Equity Whale Analyzer.

Detects whale activity in stock markets through:
- Unusual options activity
- Options sweeps
- Dark pool transactions
- Block trades
- Insider filings (Form 4)
- Institutional holdings (13F)

See: https://github.com/windoliver/libra/issues/38
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

from libra.signals.protocol import (
    AssetClass,
    SignalDirection,
    SignalSource,
    WhaleSignal,
    WhaleSignalType,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class OptionType(str, Enum):
    """Option contract type."""
    CALL = "call"
    PUT = "put"


class OrderType(str, Enum):
    """Order execution type."""
    SWEEP = "sweep"  # Aggressive sweep across exchanges
    BLOCK = "block"  # Large block trade
    SPLIT = "split"  # Split across multiple fills


class InsiderTransactionType(str, Enum):
    """Type of insider transaction."""
    BUY = "buy"
    SELL = "sell"
    EXERCISE = "exercise"  # Option exercise
    GIFT = "gift"


# =============================================================================
# Thresholds Configuration
# =============================================================================


@dataclass
class StockWhaleThresholds:
    """
    Configurable thresholds for stock market whale detection.
    """

    # Options thresholds
    unusual_options_min_premium: Decimal = Decimal("100000")  # $100k premium
    unusual_options_volume_ratio: float = 3.0  # 3x average volume
    options_sweep_min_premium: Decimal = Decimal("50000")  # $50k sweep

    # Dark pool thresholds
    dark_pool_min_value: Decimal = Decimal("1000000")  # $1M dark pool trade
    dark_pool_pct_of_volume: float = 0.05  # 5% of daily volume

    # Block trade thresholds
    block_trade_min_value: Decimal = Decimal("500000")  # $500k block
    block_trade_min_shares: int = 10000  # 10k shares minimum

    # Insider thresholds
    insider_min_value: Decimal = Decimal("100000")  # $100k insider trade

    # Institutional thresholds
    inst_position_change_pct: float = 0.10  # 10% position change
    inst_new_position_min: Decimal = Decimal("10000000")  # $10M new position

    @classmethod
    def for_large_cap(cls) -> StockWhaleThresholds:
        """Thresholds for large-cap stocks (AAPL, MSFT, etc.)."""
        return cls(
            unusual_options_min_premium=Decimal("250000"),
            dark_pool_min_value=Decimal("5000000"),
            block_trade_min_value=Decimal("2000000"),
            inst_new_position_min=Decimal("50000000"),
        )

    @classmethod
    def for_small_cap(cls) -> StockWhaleThresholds:
        """Thresholds for small-cap stocks."""
        return cls(
            unusual_options_min_premium=Decimal("25000"),
            dark_pool_min_value=Decimal("250000"),
            block_trade_min_value=Decimal("100000"),
            inst_new_position_min=Decimal("1000000"),
            block_trade_min_shares=5000,
        )


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class OptionsFlow:
    """
    Unusual options activity record.
    """

    symbol: str                    # Underlying ticker
    option_type: OptionType        # Call or Put
    strike: Decimal                # Strike price
    expiration: str                # Expiration date (YYYY-MM-DD)
    premium: Decimal               # Total premium paid
    volume: int                    # Contracts traded
    open_interest: int             # Open interest
    order_type: OrderType          # Sweep, block, etc.
    side: str                      # "buy" or "sell"
    timestamp: int                 # Unix timestamp

    # Optional enrichment
    implied_volatility: float | None = None
    delta: float | None = None
    underlying_price: Decimal | None = None
    days_to_expiry: int | None = None

    @property
    def volume_oi_ratio(self) -> float:
        """Volume to open interest ratio."""
        if self.open_interest == 0:
            return float("inf") if self.volume > 0 else 0.0
        return self.volume / self.open_interest


@dataclass
class DarkPoolTrade:
    """
    Dark pool (off-exchange) transaction.
    """

    symbol: str
    price: Decimal
    shares: int
    value_usd: Decimal
    timestamp: int
    venue: str | None = None       # ATS/dark pool name
    pct_of_daily_volume: float | None = None

    @property
    def is_significant(self) -> bool:
        """Check if trade is significant vs daily volume."""
        return (
            self.pct_of_daily_volume is not None
            and self.pct_of_daily_volume >= 0.01  # 1% of daily volume
        )


@dataclass
class BlockTrade:
    """
    Large block trade on exchange.
    """

    symbol: str
    price: Decimal
    shares: int
    value_usd: Decimal
    side: str              # "buy" or "sell" (if known)
    timestamp: int
    exchange: str | None = None


@dataclass
class InsiderTrade:
    """
    Insider transaction from SEC Form 4.
    """

    symbol: str
    insider_name: str
    insider_title: str          # CEO, CFO, Director, etc.
    transaction_type: InsiderTransactionType
    shares: int
    price: Decimal
    value_usd: Decimal
    shares_owned_after: int     # Total shares after transaction
    filing_date: str            # YYYY-MM-DD
    transaction_date: str       # YYYY-MM-DD

    @property
    def is_cluster(self) -> bool:
        """Check if part of insider buying cluster (placeholder)."""
        return False


@dataclass
class InstitutionalHolding:
    """
    Institutional position from 13F filing.
    """

    symbol: str
    institution_name: str
    shares: int
    value_usd: Decimal
    pct_of_portfolio: float
    shares_change: int           # Change from previous filing
    pct_change: float            # Percent change
    filing_date: str             # YYYY-MM-DD
    report_date: str             # Quarter end date

    @property
    def is_new_position(self) -> bool:
        """Check if this is a new position."""
        return self.pct_change == float("inf") or self.pct_change > 10.0


# =============================================================================
# Provider Protocol
# =============================================================================


class StockDataProvider(Protocol):
    """Protocol for stock market data providers."""

    async def get_options_flow(
        self,
        symbol: str | None = None,
        min_premium: Decimal | None = None,
        limit: int = 100,
    ) -> list[OptionsFlow]:
        """Get unusual options activity."""
        ...

    async def get_dark_pool_trades(
        self,
        symbol: str | None = None,
        min_value: Decimal | None = None,
        limit: int = 100,
    ) -> list[DarkPoolTrade]:
        """Get dark pool transactions."""
        ...

    async def get_insider_trades(
        self,
        symbol: str | None = None,
        days_back: int = 7,
    ) -> list[InsiderTrade]:
        """Get recent insider trades."""
        ...

    async def get_institutional_changes(
        self,
        symbol: str | None = None,
    ) -> list[InstitutionalHolding]:
        """Get institutional position changes."""
        ...


# =============================================================================
# Unusual Whales Provider
# =============================================================================


class UnusualWhalesProvider:
    """
    Provider for Unusual Whales data.

    Unusual Whales tracks options flow and dark pool data.
    API: https://docs.unusualwhales.com/
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str = "https://api.unusualwhales.com/api",
    ) -> None:
        """
        Initialize Unusual Whales provider.

        Args:
            api_key: API key for Unusual Whales
            api_url: API base URL
        """
        self.api_key = api_key
        self.api_url = api_url
        self._http_client: Any = None

    async def _get_client(self) -> Any:
        """Get or create HTTP client."""
        if self._http_client is None:
            import httpx
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                headers=headers,
            )
        return self._http_client

    async def get_options_flow(
        self,
        symbol: str | None = None,
        min_premium: Decimal | None = None,
        limit: int = 100,
    ) -> list[OptionsFlow]:
        """
        Get unusual options activity from Unusual Whales.

        Note: This is a placeholder. Full implementation would use
        the Unusual Whales API with proper authentication.
        """
        logger.debug(
            "UnusualWhales get_options_flow: symbol=%s, min_premium=%s",
            symbol,
            min_premium,
        )
        return []

    async def get_dark_pool_trades(
        self,
        symbol: str | None = None,
        min_value: Decimal | None = None,
        limit: int = 100,
    ) -> list[DarkPoolTrade]:
        """Get dark pool transactions."""
        logger.debug(
            "UnusualWhales get_dark_pool_trades: symbol=%s, min_value=%s",
            symbol,
            min_value,
        )
        return []

    async def get_insider_trades(
        self,
        symbol: str | None = None,
        days_back: int = 7,
    ) -> list[InsiderTrade]:
        """Get recent insider trades."""
        logger.debug(
            "UnusualWhales get_insider_trades: symbol=%s, days=%d",
            symbol,
            days_back,
        )
        return []

    async def get_institutional_changes(
        self,
        symbol: str | None = None,
    ) -> list[InstitutionalHolding]:
        """Get institutional position changes."""
        logger.debug(
            "UnusualWhales get_institutional_changes: symbol=%s",
            symbol,
        )
        return []

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


# =============================================================================
# SEC EDGAR Provider
# =============================================================================


class SECEdgarProvider:
    """
    Provider for SEC EDGAR filings.

    Fetches Form 4 (insider trades) and 13F (institutional holdings).
    """

    def __init__(
        self,
        user_agent: str = "LIBRA Trading System contact@example.com",
    ) -> None:
        """
        Initialize SEC EDGAR provider.

        Args:
            user_agent: User agent string (SEC requires identification)
        """
        self.user_agent = user_agent
        self._http_client: Any = None

    async def _get_client(self) -> Any:
        """Get or create HTTP client."""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                headers={"User-Agent": self.user_agent},
            )
        return self._http_client

    async def get_insider_trades(
        self,
        symbol: str | None = None,
        days_back: int = 7,
    ) -> list[InsiderTrade]:
        """
        Get insider trades from SEC Form 4 filings.

        Note: Placeholder implementation. Full version would parse
        EDGAR RSS feeds and XML filings.
        """
        logger.debug(
            "SECEdgar get_insider_trades: symbol=%s, days=%d",
            symbol,
            days_back,
        )
        return []

    async def get_institutional_changes(
        self,
        symbol: str | None = None,
    ) -> list[InstitutionalHolding]:
        """
        Get institutional holdings changes from 13F filings.

        Note: 13F filings are quarterly with 45-day delay.
        """
        logger.debug(
            "SECEdgar get_institutional_changes: symbol=%s",
            symbol,
        )
        return []

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


# =============================================================================
# Stock Whale Analyzer
# =============================================================================


class StockWhaleAnalyzer:
    """
    Analyzes stock market activity for whale signals.

    Detects:
    - Unusual options activity (OPTIONS_UNUSUAL)
    - Options sweeps (OPTIONS_SWEEP)
    - Dark pool transactions (DARK_POOL)
    - Block trades (BLOCK_TRADE)
    - Insider buying/selling (INSIDER_FILING)
    - Institutional position changes (INST_13F)
    """

    def __init__(
        self,
        thresholds: StockWhaleThresholds | None = None,
        options_provider: StockDataProvider | None = None,
        sec_provider: SECEdgarProvider | None = None,
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            thresholds: Detection thresholds
            options_provider: Provider for options/dark pool data
            sec_provider: Provider for SEC filings
        """
        self.thresholds = thresholds or StockWhaleThresholds()
        self.options_provider = options_provider
        self.sec_provider = sec_provider

    def analyze_options_flow(self, flow: OptionsFlow) -> list[WhaleSignal]:
        """
        Analyze options flow for whale signals.

        Args:
            flow: Options activity record

        Returns:
            List of detected whale signals
        """
        signals: list[WhaleSignal] = []

        # Check for unusual options activity
        if flow.premium >= self.thresholds.unusual_options_min_premium:
            if flow.volume_oi_ratio >= self.thresholds.unusual_options_volume_ratio:
                signals.append(self._create_unusual_options_signal(flow))

        # Check for options sweep
        if (
            flow.order_type == OrderType.SWEEP
            and flow.premium >= self.thresholds.options_sweep_min_premium
        ):
            signals.append(self._create_options_sweep_signal(flow))

        return signals

    def analyze_dark_pool(self, trade: DarkPoolTrade) -> WhaleSignal | None:
        """
        Analyze dark pool trade for whale signal.

        Args:
            trade: Dark pool transaction

        Returns:
            Whale signal if significant, None otherwise
        """
        if trade.value_usd < self.thresholds.dark_pool_min_value:
            return None

        # Check if significant vs daily volume
        if trade.pct_of_daily_volume is not None:
            if trade.pct_of_daily_volume < self.thresholds.dark_pool_pct_of_volume:
                return None

        return self._create_dark_pool_signal(trade)

    def analyze_block_trade(self, trade: BlockTrade) -> WhaleSignal | None:
        """
        Analyze block trade for whale signal.

        Args:
            trade: Block trade record

        Returns:
            Whale signal if significant, None otherwise
        """
        if trade.value_usd < self.thresholds.block_trade_min_value:
            return None

        if trade.shares < self.thresholds.block_trade_min_shares:
            return None

        return self._create_block_trade_signal(trade)

    def analyze_insider_trade(self, trade: InsiderTrade) -> WhaleSignal | None:
        """
        Analyze insider trade for whale signal.

        Args:
            trade: Insider transaction

        Returns:
            Whale signal if significant, None otherwise
        """
        if trade.value_usd < self.thresholds.insider_min_value:
            return None

        # Only report buys and sells (not exercises or gifts)
        if trade.transaction_type not in {
            InsiderTransactionType.BUY,
            InsiderTransactionType.SELL,
        }:
            return None

        return self._create_insider_signal(trade)

    def analyze_institutional_holding(
        self,
        holding: InstitutionalHolding,
    ) -> WhaleSignal | None:
        """
        Analyze institutional holding change for whale signal.

        Args:
            holding: Institutional position

        Returns:
            Whale signal if significant, None otherwise
        """
        # Check for new significant position
        if holding.is_new_position:
            if holding.value_usd >= self.thresholds.inst_new_position_min:
                return self._create_institutional_signal(holding, is_new=True)

        # Check for significant position change
        if abs(holding.pct_change) >= self.thresholds.inst_position_change_pct * 100:
            return self._create_institutional_signal(holding, is_new=False)

        return None

    def _create_unusual_options_signal(self, flow: OptionsFlow) -> WhaleSignal:
        """Create unusual options activity signal."""
        # Direction based on option type and side
        if flow.option_type == OptionType.CALL:
            direction = (
                SignalDirection.BULLISH
                if flow.side == "buy"
                else SignalDirection.BEARISH
            )
        else:  # PUT
            direction = (
                SignalDirection.BEARISH
                if flow.side == "buy"
                else SignalDirection.BULLISH
            )

        strength = min(
            1.0,
            float(flow.premium) / float(self.thresholds.unusual_options_min_premium * 5),
        )

        return WhaleSignal.create(
            signal_type=WhaleSignalType.OPTIONS_UNUSUAL,
            symbol=flow.symbol,
            strength=strength,
            direction=direction,
            value_usd=flow.premium,
            source=SignalSource.OPTIONS_FLOW,
            metadata={
                "option_type": flow.option_type.value,
                "strike": str(flow.strike),
                "expiration": flow.expiration,
                "volume": flow.volume,
                "open_interest": flow.open_interest,
                "volume_oi_ratio": f"{flow.volume_oi_ratio:.1f}x",
                "side": flow.side,
                "asset_class": AssetClass.OPTIONS.value,
            },
        )

    def _create_options_sweep_signal(self, flow: OptionsFlow) -> WhaleSignal:
        """Create options sweep signal."""
        if flow.option_type == OptionType.CALL:
            direction = (
                SignalDirection.BULLISH
                if flow.side == "buy"
                else SignalDirection.BEARISH
            )
        else:
            direction = (
                SignalDirection.BEARISH
                if flow.side == "buy"
                else SignalDirection.BULLISH
            )

        # Sweeps get higher strength (urgent execution)
        strength = min(
            1.0,
            0.7 + float(flow.premium) / float(self.thresholds.options_sweep_min_premium * 20),
        )

        return WhaleSignal.create(
            signal_type=WhaleSignalType.OPTIONS_SWEEP,
            symbol=flow.symbol,
            strength=strength,
            direction=direction,
            value_usd=flow.premium,
            source=SignalSource.OPTIONS_FLOW,
            metadata={
                "option_type": flow.option_type.value,
                "strike": str(flow.strike),
                "expiration": flow.expiration,
                "volume": flow.volume,
                "side": flow.side,
                "order_type": "sweep",
                "asset_class": AssetClass.OPTIONS.value,
            },
        )

    def _create_dark_pool_signal(self, trade: DarkPoolTrade) -> WhaleSignal:
        """Create dark pool signal."""
        # Dark pool trades don't reveal direction easily
        # Use price vs VWAP or other heuristics in real implementation
        direction = SignalDirection.NEUTRAL

        strength = min(
            1.0,
            float(trade.value_usd) / float(self.thresholds.dark_pool_min_value * 10),
        )

        return WhaleSignal.create(
            signal_type=WhaleSignalType.DARK_POOL,
            symbol=trade.symbol,
            strength=strength,
            direction=direction,
            value_usd=trade.value_usd,
            source=SignalSource.FINRA,
            metadata={
                "shares": trade.shares,
                "price": str(trade.price),
                "venue": trade.venue,
                "pct_of_daily_volume": (
                    f"{trade.pct_of_daily_volume:.2%}"
                    if trade.pct_of_daily_volume
                    else None
                ),
                "asset_class": AssetClass.STOCK.value,
            },
        )

    def _create_block_trade_signal(self, trade: BlockTrade) -> WhaleSignal:
        """Create block trade signal."""
        if trade.side == "buy":
            direction = SignalDirection.BULLISH
        elif trade.side == "sell":
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        strength = min(
            1.0,
            float(trade.value_usd) / float(self.thresholds.block_trade_min_value * 10),
        )

        return WhaleSignal.create(
            signal_type=WhaleSignalType.BLOCK_TRADE,
            symbol=trade.symbol,
            strength=strength,
            direction=direction,
            value_usd=trade.value_usd,
            source=SignalSource.OPTIONS_FLOW,
            metadata={
                "shares": trade.shares,
                "price": str(trade.price),
                "side": trade.side,
                "exchange": trade.exchange,
                "asset_class": AssetClass.STOCK.value,
            },
        )

    def _create_insider_signal(self, trade: InsiderTrade) -> WhaleSignal:
        """Create insider filing signal."""
        direction = (
            SignalDirection.BULLISH
            if trade.transaction_type == InsiderTransactionType.BUY
            else SignalDirection.BEARISH
        )

        # Weight by insider title importance
        title_weight = 1.0
        if "CEO" in trade.insider_title.upper():
            title_weight = 1.5
        elif "CFO" in trade.insider_title.upper():
            title_weight = 1.3
        elif "DIRECTOR" in trade.insider_title.upper():
            title_weight = 1.1

        strength = min(
            1.0,
            (float(trade.value_usd) / float(self.thresholds.insider_min_value * 10))
            * title_weight,
        )

        return WhaleSignal.create(
            signal_type=WhaleSignalType.INSIDER_FILING,
            symbol=trade.symbol,
            strength=strength,
            direction=direction,
            value_usd=trade.value_usd,
            source=SignalSource.SEC_EDGAR,
            metadata={
                "insider_name": trade.insider_name,
                "insider_title": trade.insider_title,
                "transaction_type": trade.transaction_type.value,
                "shares": trade.shares,
                "price": str(trade.price),
                "shares_owned_after": trade.shares_owned_after,
                "filing_date": trade.filing_date,
                "asset_class": AssetClass.STOCK.value,
            },
        )

    def _create_institutional_signal(
        self,
        holding: InstitutionalHolding,
        is_new: bool,
    ) -> WhaleSignal:
        """Create institutional 13F signal."""
        if is_new or holding.shares_change > 0:
            direction = SignalDirection.BULLISH
        elif holding.shares_change < 0:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        strength = min(
            1.0,
            float(holding.value_usd) / float(self.thresholds.inst_new_position_min * 5),
        )

        return WhaleSignal.create(
            signal_type=WhaleSignalType.INST_13F,
            symbol=holding.symbol,
            strength=strength,
            direction=direction,
            value_usd=holding.value_usd,
            source=SignalSource.SEC_EDGAR,
            metadata={
                "institution": holding.institution_name,
                "shares": holding.shares,
                "shares_change": holding.shares_change,
                "pct_change": f"{holding.pct_change:.1f}%",
                "pct_of_portfolio": f"{holding.pct_of_portfolio:.2f}%",
                "is_new_position": is_new,
                "filing_date": holding.filing_date,
                "asset_class": AssetClass.STOCK.value,
            },
        )

    async def poll_signals(self) -> list[WhaleSignal]:
        """
        Poll all providers for new whale signals.

        Returns:
            List of detected whale signals
        """
        signals: list[WhaleSignal] = []

        # Poll options flow provider
        if self.options_provider:
            try:
                flows = await self.options_provider.get_options_flow(
                    min_premium=self.thresholds.unusual_options_min_premium,
                )
                for flow in flows:
                    signals.extend(self.analyze_options_flow(flow))

                dark_pools = await self.options_provider.get_dark_pool_trades(
                    min_value=self.thresholds.dark_pool_min_value,
                )
                for trade in dark_pools:
                    if signal := self.analyze_dark_pool(trade):
                        signals.append(signal)

            except Exception as e:
                logger.error("Error polling options provider: %s", e)

        # Poll SEC provider
        if self.sec_provider:
            try:
                insider_trades = await self.sec_provider.get_insider_trades()
                for trade in insider_trades:
                    if signal := self.analyze_insider_trade(trade):
                        signals.append(signal)

                inst_holdings = await self.sec_provider.get_institutional_changes()
                for holding in inst_holdings:
                    if signal := self.analyze_institutional_holding(holding):
                        signals.append(signal)

            except Exception as e:
                logger.error("Error polling SEC provider: %s", e)

        return signals


# =============================================================================
# Demo/Mock Data
# =============================================================================


def create_demo_stock_signals() -> list[WhaleSignal]:
    """Create demo stock whale signals for TUI testing."""
    now = time.time_ns()

    return [
        WhaleSignal(
            signal_type=WhaleSignalType.OPTIONS_SWEEP,
            symbol="NVDA",
            timestamp_ns=now - 120_000_000_000,  # 2 min ago
            strength=0.92,
            direction=SignalDirection.BULLISH,
            value_usd=Decimal("2500000"),
            source=SignalSource.UNUSUAL_WHALES,
            metadata={
                "option_type": "call",
                "strike": "950",
                "expiration": "2024-01-19",
                "volume": 5000,
                "side": "buy",
                "order_type": "sweep",
                "asset_class": "options",
            },
        ),
        WhaleSignal(
            signal_type=WhaleSignalType.OPTIONS_UNUSUAL,
            symbol="TSLA",
            timestamp_ns=now - 480_000_000_000,  # 8 min ago
            strength=0.78,
            direction=SignalDirection.BEARISH,
            value_usd=Decimal("850000"),
            source=SignalSource.OPTIONS_FLOW,
            metadata={
                "option_type": "put",
                "strike": "220",
                "expiration": "2024-02-16",
                "volume": 12000,
                "open_interest": 3500,
                "volume_oi_ratio": "3.4x",
                "side": "buy",
                "asset_class": "options",
            },
        ),
        WhaleSignal(
            signal_type=WhaleSignalType.DARK_POOL,
            symbol="AAPL",
            timestamp_ns=now - 900_000_000_000,  # 15 min ago
            strength=0.65,
            direction=SignalDirection.NEUTRAL,
            value_usd=Decimal("15000000"),
            source=SignalSource.FINRA,
            metadata={
                "shares": 75000,
                "price": "200.00",
                "venue": "SIGMA-X",
                "pct_of_daily_volume": "2.5%",
                "asset_class": "stock",
            },
        ),
        WhaleSignal(
            signal_type=WhaleSignalType.INSIDER_FILING,
            symbol="META",
            timestamp_ns=now - 3600_000_000_000,  # 1 hour ago
            strength=0.88,
            direction=SignalDirection.BULLISH,
            value_usd=Decimal("5000000"),
            source=SignalSource.SEC_EDGAR,
            metadata={
                "insider_name": "Mark Zuckerberg",
                "insider_title": "CEO",
                "transaction_type": "buy",
                "shares": 10000,
                "price": "500.00",
                "filing_date": "2024-01-10",
                "asset_class": "stock",
            },
        ),
        WhaleSignal(
            signal_type=WhaleSignalType.INST_13F,
            symbol="MSFT",
            timestamp_ns=now - 7200_000_000_000,  # 2 hours ago
            strength=0.72,
            direction=SignalDirection.BULLISH,
            value_usd=Decimal("500000000"),
            source=SignalSource.SEC_EDGAR,
            metadata={
                "institution": "Berkshire Hathaway",
                "shares": 1200000,
                "shares_change": 400000,
                "pct_change": "50.0%",
                "pct_of_portfolio": "2.5%",
                "is_new_position": False,
                "filing_date": "2024-01-05",
                "asset_class": "stock",
            },
        ),
        WhaleSignal(
            signal_type=WhaleSignalType.BLOCK_TRADE,
            symbol="AMD",
            timestamp_ns=now - 1800_000_000_000,  # 30 min ago
            strength=0.55,
            direction=SignalDirection.BEARISH,
            value_usd=Decimal("8500000"),
            source=SignalSource.OPTIONS_FLOW,
            metadata={
                "shares": 50000,
                "price": "170.00",
                "side": "sell",
                "exchange": "NYSE",
                "asset_class": "stock",
            },
        ),
    ]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "OptionType",
    "OrderType",
    "InsiderTransactionType",
    # Config
    "StockWhaleThresholds",
    # Data structures
    "OptionsFlow",
    "DarkPoolTrade",
    "BlockTrade",
    "InsiderTrade",
    "InstitutionalHolding",
    # Providers
    "StockDataProvider",
    "UnusualWhalesProvider",
    "SECEdgarProvider",
    # Analyzer
    "StockWhaleAnalyzer",
    # Demo
    "create_demo_stock_signals",
]
