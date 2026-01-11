"""
Whale Signal Aggregator.

Combines signals from multiple sources (order flow + on-chain) into
a unified signal stream with deduplication and ranking.

See: https://github.com/windoliver/libra/issues/38
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from libra.signals.order_flow import OrderFlowAnalyzer
from libra.signals.protocol import (
    SignalDirection,
    WhaleSignal,
    WhaleThresholds,
)

if TYPE_CHECKING:
    from libra.gateways.protocol import Bar, OrderBook
    from libra.signals.onchain import BaseOnChainProvider


logger = logging.getLogger(__name__)


@dataclass
class AggregatorConfig:
    """Configuration for signal aggregator."""

    # Enable/disable signal sources
    enable_order_flow: bool = True
    enable_onchain: bool = False  # Requires API keys

    # Signal filtering
    min_strength: float = 0.3  # Minimum signal strength to emit
    min_value_usd: Decimal = Decimal("100000")  # Minimum USD value

    # Deduplication window (seconds)
    dedup_window_sec: float = 60.0

    # Maximum signals to keep in history
    max_history: int = 1000

    # Aggregation settings
    combine_similar_signals: bool = True
    boost_confirmed_signals: float = 1.2  # Boost when multiple sources agree


@dataclass
class SignalStats:
    """Statistics about signal generation."""

    total_signals: int = 0
    signals_by_type: dict[str, int] = field(default_factory=dict)
    signals_by_direction: dict[str, int] = field(default_factory=dict)
    total_value_usd: Decimal = Decimal("0")
    avg_strength: float = 0.0


class WhaleSignalAggregator:
    """
    Aggregates whale signals from multiple sources.

    Combines:
    - Order flow analysis (Tier 1 - free)
    - On-chain data (Tier 2 - optional)

    Features:
    - Signal deduplication
    - Strength boosting for confirmed signals
    - Configurable filtering
    - Signal history and statistics

    Example:
        aggregator = WhaleSignalAggregator()

        # Add on-chain provider (optional)
        aggregator.add_onchain_provider(whale_alert_provider)

        # Get signals from order book
        signals = await aggregator.get_signals(orderbook=ob, bars=bars)

        # Get all recent signals
        history = aggregator.get_history()

        # Get statistics
        stats = aggregator.get_stats()
    """

    def __init__(
        self,
        config: AggregatorConfig | None = None,
        thresholds: WhaleThresholds | None = None,
    ) -> None:
        """
        Initialize aggregator.

        Args:
            config: Aggregator configuration
            thresholds: Whale detection thresholds
        """
        self.config = config or AggregatorConfig()
        self.thresholds = thresholds or WhaleThresholds()

        # Order flow analyzer (Tier 1)
        self._order_flow = OrderFlowAnalyzer(thresholds=self.thresholds)

        # On-chain providers (Tier 2)
        self._onchain_providers: list[BaseOnChainProvider] = []

        # Signal history
        self._history: deque[WhaleSignal] = deque(maxlen=self.config.max_history)

        # Statistics
        self._stats = SignalStats()

        # Callbacks
        self._callbacks: list[callable] = []

    def add_onchain_provider(self, provider: BaseOnChainProvider) -> None:
        """
        Add an on-chain data provider.

        Args:
            provider: Provider instance (must be connected)
        """
        self._onchain_providers.append(provider)
        logger.info(f"Added on-chain provider: {provider.name}")

    def remove_onchain_provider(self, name: str) -> bool:
        """
        Remove an on-chain provider by name.

        Args:
            name: Provider name

        Returns:
            True if removed
        """
        for i, provider in enumerate(self._onchain_providers):
            if provider.name == name:
                self._onchain_providers.pop(i)
                logger.info(f"Removed on-chain provider: {name}")
                return True
        return False

    def on_signal(self, callback: callable) -> None:
        """
        Register a callback for new signals.

        Args:
            callback: Function(signal: WhaleSignal) to call
        """
        self._callbacks.append(callback)

    async def get_signals(
        self,
        orderbook: OrderBook | None = None,
        bars: list[Bar] | None = None,
        symbol: str | None = None,
    ) -> list[WhaleSignal]:
        """
        Get aggregated whale signals from all sources.

        Args:
            orderbook: Order book snapshot (for order flow)
            bars: Recent OHLCV bars (for volume analysis)
            symbol: Symbol for on-chain queries

        Returns:
            List of aggregated signals
        """
        all_signals: list[WhaleSignal] = []

        # Tier 1: Order flow analysis
        if self.config.enable_order_flow:
            flow_signals = self._order_flow.get_signals(orderbook, bars)
            all_signals.extend(flow_signals)

        # Tier 2: On-chain data
        if self.config.enable_onchain and self._onchain_providers:
            onchain_signals = await self._get_onchain_signals(symbol)
            all_signals.extend(onchain_signals)

        # Filter by minimum criteria
        filtered = self._filter_signals(all_signals)

        # Combine similar signals
        if self.config.combine_similar_signals:
            filtered = self._combine_signals(filtered)

        # Update history and stats
        for signal in filtered:
            self._add_to_history(signal)
            self._notify_callbacks(signal)

        return filtered

    async def _get_onchain_signals(
        self,
        symbol: str | None = None,
    ) -> list[WhaleSignal]:
        """Get signals from all on-chain providers."""
        signals: list[WhaleSignal] = []

        tasks = [
            provider.get_signals(
                min_value_usd=self.thresholds.whale_transfer_min_usd,
            )
            for provider in self._onchain_providers
            if provider.is_connected
        ]

        if not tasks:
            return signals

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    # Filter by symbol if specified
                    if symbol:
                        result = [s for s in result if symbol in s.symbol]
                    signals.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"On-chain provider error: {result}")
        except Exception as e:
            logger.error(f"Failed to get on-chain signals: {e}")

        return signals

    def _filter_signals(self, signals: list[WhaleSignal]) -> list[WhaleSignal]:
        """Filter signals by minimum criteria."""
        return [
            s for s in signals
            if s.strength >= self.config.min_strength
            and s.value_usd >= self.config.min_value_usd
        ]

    def _combine_signals(self, signals: list[WhaleSignal]) -> list[WhaleSignal]:
        """
        Combine similar signals and boost strength.

        Signals are combined if they have the same:
        - Symbol
        - Signal type
        - Direction
        - Within dedup window
        """
        if not signals:
            return signals

        combined: dict[str, WhaleSignal] = {}

        for signal in signals:
            key = f"{signal.symbol}:{signal.signal_type.value}:{signal.direction.value}"

            if key in combined:
                existing = combined[key]
                # Boost strength when multiple sources agree
                boosted_strength = min(
                    1.0,
                    existing.strength * self.config.boost_confirmed_signals
                )
                # Take higher value
                combined_value = max(existing.value_usd, signal.value_usd)
                # Merge metadata
                merged_metadata = {**existing.metadata, **signal.metadata}
                merged_metadata["combined_sources"] = merged_metadata.get(
                    "combined_sources", 1
                ) + 1

                # Create combined signal
                combined[key] = WhaleSignal(
                    signal_type=existing.signal_type,
                    symbol=existing.symbol,
                    timestamp_ns=max(existing.timestamp_ns, signal.timestamp_ns),
                    strength=boosted_strength,
                    direction=existing.direction,
                    value_usd=combined_value,
                    source=existing.source,
                    metadata=merged_metadata,
                )
            else:
                combined[key] = signal

        return list(combined.values())

    def _add_to_history(self, signal: WhaleSignal) -> None:
        """Add signal to history and update stats."""
        self._history.append(signal)

        # Update statistics
        self._stats.total_signals += 1
        self._stats.total_value_usd += signal.value_usd

        type_key = signal.signal_type.value
        self._stats.signals_by_type[type_key] = (
            self._stats.signals_by_type.get(type_key, 0) + 1
        )

        dir_key = signal.direction.value
        self._stats.signals_by_direction[dir_key] = (
            self._stats.signals_by_direction.get(dir_key, 0) + 1
        )

        # Update average strength
        total = self._stats.total_signals
        self._stats.avg_strength = (
            (self._stats.avg_strength * (total - 1) + signal.strength) / total
        )

    def _notify_callbacks(self, signal: WhaleSignal) -> None:
        """Notify registered callbacks of new signal."""
        for callback in self._callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.warning(f"Signal callback error: {e}")

    def get_history(
        self,
        limit: int | None = None,
        symbol: str | None = None,
        direction: SignalDirection | None = None,
    ) -> list[WhaleSignal]:
        """
        Get signal history.

        Args:
            limit: Maximum signals to return
            symbol: Filter by symbol
            direction: Filter by direction

        Returns:
            List of historical signals (newest first)
        """
        signals = list(self._history)

        # Apply filters
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        if direction:
            signals = [s for s in signals if s.direction == direction]

        # Sort by timestamp (newest first)
        signals.sort(key=lambda s: s.timestamp_ns, reverse=True)

        if limit:
            signals = signals[:limit]

        return signals

    def get_stats(self) -> SignalStats:
        """Get signal statistics."""
        return self._stats

    def get_summary(self) -> dict[str, Any]:
        """Get aggregator summary."""
        return {
            "config": {
                "enable_order_flow": self.config.enable_order_flow,
                "enable_onchain": self.config.enable_onchain,
                "min_strength": self.config.min_strength,
                "min_value_usd": str(self.config.min_value_usd),
            },
            "providers": {
                "order_flow": self.config.enable_order_flow,
                "onchain": [p.name for p in self._onchain_providers],
            },
            "stats": {
                "total_signals": self._stats.total_signals,
                "by_type": self._stats.signals_by_type,
                "by_direction": self._stats.signals_by_direction,
                "total_value_usd": str(self._stats.total_value_usd),
                "avg_strength": f"{self._stats.avg_strength:.2f}",
            },
            "history_size": len(self._history),
        }

    def reset(self) -> None:
        """Reset aggregator state."""
        self._order_flow.reset()
        self._history.clear()
        self._stats = SignalStats()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_aggregator(
    enable_onchain: bool = False,
    thresholds: WhaleThresholds | None = None,
) -> WhaleSignalAggregator:
    """
    Create a configured signal aggregator.

    Args:
        enable_onchain: Enable on-chain providers
        thresholds: Detection thresholds

    Returns:
        Configured aggregator
    """
    config = AggregatorConfig(
        enable_order_flow=True,
        enable_onchain=enable_onchain,
    )
    return WhaleSignalAggregator(config=config, thresholds=thresholds)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AggregatorConfig",
    "SignalStats",
    "WhaleSignalAggregator",
    "create_aggregator",
]
