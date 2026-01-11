"""
Prediction Market Whale Analyzer.

Detects whale activity on prediction markets like Polymarket, Kalshi, etc.

Signals:
- PM_LARGE_BET: Large bet on a market outcome
- PM_POSITION_CHANGE: Whale significantly changing position
- PM_MARKET_MOVE: Single trade moving market price
- PM_SMART_MONEY: Known smart money wallet activity

See: https://github.com/windoliver/libra/issues/38
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
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
# Thresholds Configuration
# =============================================================================


@dataclass
class PredictionMarketThresholds:
    """
    Configurable thresholds for prediction market whale detection.
    """

    # Minimum bet size in USD to be considered "large"
    large_bet_min_usd: Decimal = Decimal("10000")

    # Position change threshold (percent of market liquidity)
    position_change_pct: float = 0.05  # 5% of market

    # Market move threshold (price impact from single trade)
    market_move_threshold: float = 0.02  # 2% price move

    # Volume spike multiplier vs average
    volume_spike_multiplier: float = 3.0

    # Known smart money addresses (can be populated from config)
    smart_money_addresses: set[str] = field(default_factory=set)

    @classmethod
    def for_polymarket(cls) -> PredictionMarketThresholds:
        """Thresholds tuned for Polymarket."""
        return cls(
            large_bet_min_usd=Decimal("25000"),
            position_change_pct=0.03,
            market_move_threshold=0.015,
        )

    @classmethod
    def for_kalshi(cls) -> PredictionMarketThresholds:
        """Thresholds tuned for Kalshi."""
        return cls(
            large_bet_min_usd=Decimal("5000"),  # Lower limits on Kalshi
            position_change_pct=0.05,
            market_move_threshold=0.02,
        )


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PredictionMarketBet:
    """
    A bet/trade on a prediction market.
    """

    market_id: str          # Market identifier
    market_title: str       # Human-readable market name
    outcome: str            # "YES" or "NO" (or specific outcome name)
    side: str               # "buy" or "sell"
    price: Decimal          # Price paid (0-1 range typically)
    size: Decimal           # Number of shares/contracts
    value_usd: Decimal      # Total USD value
    timestamp: int          # Unix timestamp
    trader_address: str | None = None  # Wallet address if known
    source: SignalSource = SignalSource.POLYMARKET


@dataclass
class PredictionMarketState:
    """
    Current state of a prediction market.
    """

    market_id: str
    market_title: str
    outcomes: list[str]           # e.g., ["YES", "NO"]
    prices: dict[str, Decimal]    # Outcome -> price
    liquidity_usd: Decimal        # Total liquidity
    volume_24h_usd: Decimal       # 24h volume
    open_interest: Decimal        # Open interest value


# =============================================================================
# Provider Protocol
# =============================================================================


class PredictionMarketProvider(Protocol):
    """Protocol for prediction market data providers."""

    async def get_recent_bets(
        self,
        market_id: str | None = None,
        min_value_usd: Decimal | None = None,
        limit: int = 100,
    ) -> list[PredictionMarketBet]:
        """Get recent bets, optionally filtered."""
        ...

    async def get_market_state(
        self,
        market_id: str,
    ) -> PredictionMarketState | None:
        """Get current market state."""
        ...

    async def get_trader_positions(
        self,
        trader_address: str,
    ) -> dict[str, Decimal]:
        """Get trader's positions by market ID."""
        ...


# =============================================================================
# Polymarket Provider
# =============================================================================


class PolymarketProvider:
    """
    Provider for Polymarket data.

    Uses Polymarket's public API and on-chain data.
    """

    def __init__(
        self,
        api_url: str = "https://clob.polymarket.com",
        known_whales: set[str] | None = None,
    ) -> None:
        """
        Initialize Polymarket provider.

        Args:
            api_url: Polymarket CLOB API URL
            known_whales: Set of known whale wallet addresses
        """
        self.api_url = api_url
        self.known_whales = known_whales or set()
        self._http_client: Any = None

    async def _get_client(self) -> Any:
        """Get or create HTTP client."""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def get_recent_bets(
        self,
        market_id: str | None = None,
        min_value_usd: Decimal | None = None,
        limit: int = 100,
    ) -> list[PredictionMarketBet]:
        """
        Get recent bets from Polymarket.

        Note: This is a simplified implementation. Full implementation
        would query the Polymarket API and/or on-chain data.
        """
        # TODO: Implement actual Polymarket API integration
        # For now, return empty list
        logger.debug(
            "Polymarket get_recent_bets called: market=%s, min_value=%s",
            market_id,
            min_value_usd,
        )
        return []

    async def get_market_state(
        self,
        market_id: str,
    ) -> PredictionMarketState | None:
        """Get current state of a Polymarket market."""
        # TODO: Implement actual API call
        logger.debug("Polymarket get_market_state called: market=%s", market_id)
        return None

    async def get_trader_positions(
        self,
        trader_address: str,
    ) -> dict[str, Decimal]:
        """Get trader's positions across markets."""
        # TODO: Implement actual API call
        logger.debug(
            "Polymarket get_trader_positions called: trader=%s",
            trader_address,
        )
        return {}

    def is_known_whale(self, address: str) -> bool:
        """Check if address is a known whale."""
        return address.lower() in {a.lower() for a in self.known_whales}

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


# =============================================================================
# Prediction Market Whale Analyzer
# =============================================================================


class PredictionMarketWhaleAnalyzer:
    """
    Analyzes prediction market activity for whale signals.

    Detects:
    - Large bets (PM_LARGE_BET)
    - Significant position changes (PM_POSITION_CHANGE)
    - Market-moving trades (PM_MARKET_MOVE)
    - Smart money activity (PM_SMART_MONEY)
    """

    def __init__(
        self,
        thresholds: PredictionMarketThresholds | None = None,
        providers: list[PredictionMarketProvider] | None = None,
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            thresholds: Detection thresholds
            providers: Data providers (Polymarket, Kalshi, etc.)
        """
        self.thresholds = thresholds or PredictionMarketThresholds()
        self.providers = providers or []

        # Track position changes
        self._position_history: dict[str, dict[str, Decimal]] = {}  # trader -> market -> size
        self._market_prices: dict[str, dict[str, Decimal]] = {}  # market -> outcome -> price

    def analyze_bet(
        self,
        bet: PredictionMarketBet,
        market_state: PredictionMarketState | None = None,
    ) -> list[WhaleSignal]:
        """
        Analyze a prediction market bet for whale signals.

        Args:
            bet: The bet to analyze
            market_state: Current market state (optional)

        Returns:
            List of detected whale signals
        """
        signals: list[WhaleSignal] = []

        # Check for large bet
        if bet.value_usd >= self.thresholds.large_bet_min_usd:
            signals.append(self._create_large_bet_signal(bet, market_state))

        # Check for market move (if we have market state)
        if market_state:
            price_impact = self._estimate_price_impact(bet, market_state)
            if price_impact >= self.thresholds.market_move_threshold:
                signals.append(
                    self._create_market_move_signal(bet, market_state, price_impact)
                )

        # Check for smart money (if address is known)
        if bet.trader_address:
            if self._is_smart_money(bet.trader_address):
                signals.append(self._create_smart_money_signal(bet, market_state))

        return signals

    def analyze_position_change(
        self,
        trader_address: str,
        market_id: str,
        old_position: Decimal,
        new_position: Decimal,
        market_state: PredictionMarketState,
    ) -> WhaleSignal | None:
        """
        Analyze a position change for whale signals.

        Args:
            trader_address: Trader wallet address
            market_id: Market identifier
            old_position: Previous position size
            new_position: Current position size
            market_state: Current market state

        Returns:
            Whale signal if significant, None otherwise
        """
        position_change = abs(new_position - old_position)
        change_pct = float(position_change) / float(market_state.liquidity_usd)

        if change_pct >= self.thresholds.position_change_pct:
            direction = (
                SignalDirection.BULLISH
                if new_position > old_position
                else SignalDirection.BEARISH
            )

            # Calculate strength based on change magnitude
            strength = min(1.0, change_pct / 0.20)  # 20% = max strength

            return WhaleSignal.create(
                signal_type=WhaleSignalType.PM_POSITION_CHANGE,
                symbol=market_state.market_title,
                strength=strength,
                direction=direction,
                value_usd=position_change,
                source=SignalSource.POLYMARKET,
                metadata={
                    "market_id": market_id,
                    "trader_address": trader_address,
                    "old_position": str(old_position),
                    "new_position": str(new_position),
                    "change_pct": f"{change_pct:.2%}",
                    "asset_class": AssetClass.PREDICTION_MARKET.value,
                },
            )

        return None

    def _create_large_bet_signal(
        self,
        bet: PredictionMarketBet,
        market_state: PredictionMarketState | None,
    ) -> WhaleSignal:
        """Create a large bet signal."""
        # Determine direction based on bet type
        if bet.side == "buy":
            direction = (
                SignalDirection.BULLISH
                if bet.outcome.upper() == "YES"
                else SignalDirection.BEARISH
            )
        else:
            direction = (
                SignalDirection.BEARISH
                if bet.outcome.upper() == "YES"
                else SignalDirection.BULLISH
            )

        # Calculate strength based on bet size
        strength = min(
            1.0,
            float(bet.value_usd) / float(self.thresholds.large_bet_min_usd * 10),
        )

        return WhaleSignal.create(
            signal_type=WhaleSignalType.PM_LARGE_BET,
            symbol=bet.market_title,
            strength=strength,
            direction=direction,
            value_usd=bet.value_usd,
            source=bet.source,
            metadata={
                "market_id": bet.market_id,
                "outcome": bet.outcome,
                "side": bet.side,
                "price": str(bet.price),
                "size": str(bet.size),
                "trader_address": bet.trader_address,
                "liquidity_usd": str(market_state.liquidity_usd) if market_state else None,
                "asset_class": AssetClass.PREDICTION_MARKET.value,
            },
        )

    def _create_market_move_signal(
        self,
        bet: PredictionMarketBet,
        market_state: PredictionMarketState,
        price_impact: float,
    ) -> WhaleSignal:
        """Create a market move signal."""
        # Direction based on price impact
        if bet.side == "buy":
            direction = (
                SignalDirection.BULLISH
                if bet.outcome.upper() == "YES"
                else SignalDirection.BEARISH
            )
        else:
            direction = (
                SignalDirection.BEARISH
                if bet.outcome.upper() == "YES"
                else SignalDirection.BULLISH
            )

        # Strength based on price impact magnitude
        strength = min(1.0, price_impact / 0.10)  # 10% move = max strength

        return WhaleSignal.create(
            signal_type=WhaleSignalType.PM_MARKET_MOVE,
            symbol=bet.market_title,
            strength=strength,
            direction=direction,
            value_usd=bet.value_usd,
            source=bet.source,
            metadata={
                "market_id": bet.market_id,
                "outcome": bet.outcome,
                "price_impact": f"{price_impact:.2%}",
                "pre_trade_price": str(market_state.prices.get(bet.outcome, Decimal("0"))),
                "bet_size": str(bet.size),
                "asset_class": AssetClass.PREDICTION_MARKET.value,
            },
        )

    def _create_smart_money_signal(
        self,
        bet: PredictionMarketBet,
        market_state: PredictionMarketState | None,
    ) -> WhaleSignal:
        """Create a smart money signal."""
        if bet.side == "buy":
            direction = (
                SignalDirection.BULLISH
                if bet.outcome.upper() == "YES"
                else SignalDirection.BEARISH
            )
        else:
            direction = (
                SignalDirection.BEARISH
                if bet.outcome.upper() == "YES"
                else SignalDirection.BULLISH
            )

        # Smart money gets high strength by default
        strength = 0.85

        return WhaleSignal.create(
            signal_type=WhaleSignalType.PM_SMART_MONEY,
            symbol=bet.market_title,
            strength=strength,
            direction=direction,
            value_usd=bet.value_usd,
            source=bet.source,
            metadata={
                "market_id": bet.market_id,
                "outcome": bet.outcome,
                "side": bet.side,
                "trader_address": bet.trader_address,
                "is_known_whale": True,
                "asset_class": AssetClass.PREDICTION_MARKET.value,
            },
        )

    def _estimate_price_impact(
        self,
        bet: PredictionMarketBet,
        market_state: PredictionMarketState,
    ) -> float:
        """
        Estimate price impact of a bet.

        Uses simplified constant product formula approximation.
        """
        if market_state.liquidity_usd <= 0:
            return 0.0

        # Simplified impact estimation
        # Real implementation would use AMM curve
        impact = float(bet.value_usd) / float(market_state.liquidity_usd)
        return min(impact, 1.0)

    def _is_smart_money(self, address: str) -> bool:
        """Check if address is known smart money."""
        return address.lower() in {
            a.lower() for a in self.thresholds.smart_money_addresses
        }

    async def poll_signals(self) -> list[WhaleSignal]:
        """
        Poll all providers for new whale signals.

        Returns:
            List of detected whale signals
        """
        signals: list[WhaleSignal] = []

        for provider in self.providers:
            try:
                # Get recent large bets
                bets = await provider.get_recent_bets(
                    min_value_usd=self.thresholds.large_bet_min_usd,
                )

                for bet in bets:
                    # Get market state for context
                    market_state = await provider.get_market_state(bet.market_id)
                    signals.extend(self.analyze_bet(bet, market_state))

            except Exception as e:
                logger.error("Error polling provider: %s", e)
                continue

        return signals

    def reset(self) -> None:
        """Reset analyzer state."""
        self._position_history.clear()
        self._market_prices.clear()


# =============================================================================
# Demo/Mock Data
# =============================================================================


def create_demo_pm_signals() -> list[WhaleSignal]:
    """Create demo prediction market whale signals for TUI testing."""
    now = time.time_ns()

    return [
        WhaleSignal(
            signal_type=WhaleSignalType.PM_LARGE_BET,
            symbol="Will Bitcoin reach $100k by Dec 2024?",
            timestamp_ns=now - 300_000_000_000,  # 5 min ago
            strength=0.88,
            direction=SignalDirection.BULLISH,
            value_usd=Decimal("150000"),
            source=SignalSource.POLYMARKET,
            metadata={
                "market_id": "btc-100k-2024",
                "outcome": "YES",
                "side": "buy",
                "price": "0.42",
                "size": "357143",
                "asset_class": "prediction_market",
            },
        ),
        WhaleSignal(
            signal_type=WhaleSignalType.PM_MARKET_MOVE,
            symbol="2024 US Presidential Election - Republican",
            timestamp_ns=now - 600_000_000_000,  # 10 min ago
            strength=0.75,
            direction=SignalDirection.BULLISH,
            value_usd=Decimal("500000"),
            source=SignalSource.POLYMARKET,
            metadata={
                "market_id": "us-pres-2024-rep",
                "outcome": "YES",
                "price_impact": "3.2%",
                "pre_trade_price": "0.48",
                "asset_class": "prediction_market",
            },
        ),
        WhaleSignal(
            signal_type=WhaleSignalType.PM_SMART_MONEY,
            symbol="Fed Rate Cut March 2024",
            timestamp_ns=now - 1800_000_000_000,  # 30 min ago
            strength=0.92,
            direction=SignalDirection.BEARISH,
            value_usd=Decimal("75000"),
            source=SignalSource.KALSHI,
            metadata={
                "market_id": "fed-rate-mar-2024",
                "outcome": "NO",
                "side": "buy",
                "trader_address": "0x742d35Cc6634C0532925a3b8...",
                "is_known_whale": True,
                "asset_class": "prediction_market",
            },
        ),
        WhaleSignal(
            signal_type=WhaleSignalType.PM_POSITION_CHANGE,
            symbol="ETH/BTC Ratio > 0.06 by Q2 2024",
            timestamp_ns=now - 3600_000_000_000,  # 1 hour ago
            strength=0.68,
            direction=SignalDirection.BULLISH,
            value_usd=Decimal("200000"),
            source=SignalSource.POLYMARKET,
            metadata={
                "market_id": "eth-btc-ratio-q2",
                "old_position": "50000",
                "new_position": "250000",
                "change_pct": "8.5%",
                "asset_class": "prediction_market",
            },
        ),
    ]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PredictionMarketThresholds",
    "PredictionMarketBet",
    "PredictionMarketState",
    "PredictionMarketProvider",
    "PolymarketProvider",
    "PredictionMarketWhaleAnalyzer",
    "create_demo_pm_signals",
]
