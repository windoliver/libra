"""
Order Flow Analyzer for Whale Detection.

Tier 1 implementation using exchange order book data (free via CCXT).

Detects:
- Order book imbalance (buy/sell pressure)
- Large walls (single-price whale orders)
- Ladder walls (distributed whale orders)
- Volume spikes (unusual trading activity)
- Large trades (significant individual trades)

See: https://github.com/windoliver/libra/issues/38
"""

from __future__ import annotations

import logging
import time
from collections import deque
from decimal import Decimal
from typing import TYPE_CHECKING

from libra.signals.protocol import (
    OrderBookAnalysis,
    SignalDirection,
    SignalSource,
    WhaleSignal,
    WhaleSignalType,
    WhaleThresholds,
)

if TYPE_CHECKING:
    from libra.gateways.protocol import Bar, OrderBook


logger = logging.getLogger(__name__)


class OrderFlowAnalyzer:
    """
    Analyze order flow to detect whale activity.

    Uses exchange order book and trade data to identify:
    - Large orders that may move the market
    - Accumulation/distribution patterns
    - Unusual volume activity

    Example:
        analyzer = OrderFlowAnalyzer()

        # Analyze order book
        signals = analyzer.analyze_orderbook(orderbook)

        # Check for volume spikes
        signal = analyzer.analyze_volume(bars)

        # Get all signals
        all_signals = analyzer.get_signals(orderbook, bars)
    """

    def __init__(
        self,
        thresholds: WhaleThresholds | None = None,
        volume_window: int = 20,
    ) -> None:
        """
        Initialize analyzer.

        Args:
            thresholds: Detection thresholds (uses defaults if None)
            volume_window: Number of bars for volume average calculation
        """
        self.thresholds = thresholds or WhaleThresholds()
        self.volume_window = volume_window

        # Volume history for spike detection
        self._volume_history: dict[str, deque[Decimal]] = {}

        # Recent signals cache (dedup)
        self._recent_signals: deque[tuple[str, int]] = deque(maxlen=100)

    def analyze_orderbook(self, orderbook: OrderBook) -> list[WhaleSignal]:
        """
        Analyze order book for whale activity.

        Detects:
        - Order imbalance
        - Large walls
        - Ladder patterns

        Args:
            orderbook: Order book snapshot

        Returns:
            List of detected whale signals
        """
        signals: list[WhaleSignal] = []

        # Compute order book analysis
        analysis = self._compute_orderbook_analysis(orderbook)

        # Check for order imbalance
        imbalance_signal = self._detect_imbalance(analysis)
        if imbalance_signal:
            signals.append(imbalance_signal)

        # Check for large walls
        wall_signals = self._detect_walls(orderbook, analysis)
        signals.extend(wall_signals)

        # Check for ladder patterns
        ladder_signals = self._detect_ladders(orderbook, analysis)
        signals.extend(ladder_signals)

        return signals

    def analyze_volume(self, bars: list[Bar]) -> WhaleSignal | None:
        """
        Analyze trading volume for spikes.

        Args:
            bars: Recent OHLCV bars

        Returns:
            Volume spike signal if detected
        """
        if not bars:
            return None

        latest = bars[-1]
        symbol = latest.symbol

        # Update volume history
        if symbol not in self._volume_history:
            self._volume_history[symbol] = deque(maxlen=self.volume_window)

        history = self._volume_history[symbol]

        # Need enough history
        if len(history) < self.volume_window // 2:
            history.append(latest.volume)
            return None

        # Calculate average volume
        avg_volume = sum(history) / len(history)

        # Add current to history
        history.append(latest.volume)

        if avg_volume == 0:
            return None

        # Check for spike
        volume_ratio = float(latest.volume / avg_volume)

        if volume_ratio >= self.thresholds.volume_spike_multiplier:
            # Determine direction from price action
            if latest.close > latest.open:
                direction = SignalDirection.BULLISH
            elif latest.close < latest.open:
                direction = SignalDirection.BEARISH
            else:
                direction = SignalDirection.NEUTRAL

            # Estimate value (volume * price)
            value_usd = latest.volume * latest.close

            # Strength based on spike magnitude
            strength = min(1.0, volume_ratio / 10.0)

            return WhaleSignal.create(
                signal_type=WhaleSignalType.VOLUME_SPIKE,
                symbol=symbol,
                strength=strength,
                direction=direction,
                value_usd=value_usd,
                source=SignalSource.TRADES,
                metadata={
                    "volume": str(latest.volume),
                    "avg_volume": str(avg_volume),
                    "volume_ratio": f"{volume_ratio:.2f}x",
                    "close": str(latest.close),
                    "open": str(latest.open),
                },
            )

        return None

    def analyze_trade(
        self,
        symbol: str,
        side: str,
        price: Decimal,
        size: Decimal,
        value_usd: Decimal | None = None,
    ) -> WhaleSignal | None:
        """
        Analyze a single trade for whale activity.

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            price: Trade price
            size: Trade size
            value_usd: Trade value in USD (calculated if None)

        Returns:
            Large trade signal if threshold exceeded
        """
        # Calculate value if not provided
        if value_usd is None:
            value_usd = price * size

        if value_usd < self.thresholds.large_trade_min_value:
            return None

        direction = SignalDirection.BULLISH if side == "buy" else SignalDirection.BEARISH

        # Strength based on value
        strength = min(1.0, float(value_usd / (self.thresholds.large_trade_min_value * 10)))

        return WhaleSignal.create(
            signal_type=WhaleSignalType.LARGE_TRADE,
            symbol=symbol,
            strength=strength,
            direction=direction,
            value_usd=value_usd,
            source=SignalSource.TRADES,
            metadata={
                "side": side,
                "price": str(price),
                "size": str(size),
            },
        )

    def get_signals(
        self,
        orderbook: OrderBook | None = None,
        bars: list[Bar] | None = None,
    ) -> list[WhaleSignal]:
        """
        Get all detected whale signals.

        Args:
            orderbook: Order book snapshot (optional)
            bars: Recent OHLCV bars (optional)

        Returns:
            List of all detected signals
        """
        signals: list[WhaleSignal] = []

        if orderbook:
            signals.extend(self.analyze_orderbook(orderbook))

        if bars:
            volume_signal = self.analyze_volume(bars)
            if volume_signal:
                signals.append(volume_signal)

        # Deduplicate similar signals
        signals = self._deduplicate_signals(signals)

        return signals

    def _compute_orderbook_analysis(self, orderbook: OrderBook) -> OrderBookAnalysis:
        """Compute order book metrics."""
        depth = self.thresholds.orderbook_depth
        timestamp_ns = orderbook.timestamp_ns

        # Get top N levels
        bids = orderbook.bids[:depth] if orderbook.bids else []
        asks = orderbook.asks[:depth] if orderbook.asks else []

        # Calculate volumes (price * size for each level)
        bid_volume = sum(Decimal(str(p)) * Decimal(str(s)) for p, s in bids)
        ask_volume = sum(Decimal(str(p)) * Decimal(str(s)) for p, s in asks)

        total_volume = bid_volume + ask_volume

        # Imbalance ratio
        if total_volume > 0:
            imbalance = float((bid_volume - ask_volume) / total_volume)
        else:
            imbalance = 0.0

        # Find largest walls
        largest_bid = Decimal("0")
        largest_bid_price = None
        for price, size in bids:
            value = Decimal(str(price)) * Decimal(str(size))
            if value > largest_bid:
                largest_bid = value
                largest_bid_price = Decimal(str(price))

        largest_ask = Decimal("0")
        largest_ask_price = None
        for price, size in asks:
            value = Decimal(str(price)) * Decimal(str(size))
            if value > largest_ask:
                largest_ask = value
                largest_ask_price = Decimal(str(price))

        # Spread and mid price
        if bids and asks:
            best_bid = Decimal(str(bids[0][0]))
            best_ask = Decimal(str(asks[0][0]))
            mid_price = (best_bid + best_ask) / 2
            spread_bps = float((best_ask - best_bid) / mid_price * 10000)
        else:
            mid_price = Decimal("0")
            spread_bps = 0.0

        return OrderBookAnalysis(
            symbol=orderbook.symbol,
            timestamp_ns=timestamp_ns,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance_ratio=imbalance,
            largest_bid_wall=largest_bid,
            largest_ask_wall=largest_ask,
            bid_wall_price=largest_bid_price,
            ask_wall_price=largest_ask_price,
            total_book_value=total_volume,
            bid_pct=float(bid_volume / total_volume * 100) if total_volume > 0 else 0,
            ask_pct=float(ask_volume / total_volume * 100) if total_volume > 0 else 0,
            spread_bps=spread_bps,
            mid_price=mid_price,
        )

    def _detect_imbalance(self, analysis: OrderBookAnalysis) -> WhaleSignal | None:
        """Detect order book imbalance."""
        threshold = self.thresholds.imbalance_threshold

        if abs(analysis.imbalance_ratio) < threshold:
            return None

        # Strong imbalance detected
        if analysis.imbalance_ratio > 0:
            direction = SignalDirection.BULLISH
            dominant_side = "bid"
            dominant_volume = analysis.bid_volume
        else:
            direction = SignalDirection.BEARISH
            dominant_side = "ask"
            dominant_volume = analysis.ask_volume

        # Strength based on imbalance magnitude
        strength = min(1.0, abs(analysis.imbalance_ratio) / 0.6)  # 60% imbalance = 1.0

        return WhaleSignal.create(
            signal_type=WhaleSignalType.ORDER_IMBALANCE,
            symbol=analysis.symbol,
            strength=strength,
            direction=direction,
            value_usd=dominant_volume,
            source=SignalSource.ORDER_BOOK,
            metadata={
                "imbalance_ratio": f"{analysis.imbalance_ratio:.3f}",
                "dominant_side": dominant_side,
                "bid_volume": str(analysis.bid_volume),
                "ask_volume": str(analysis.ask_volume),
                "bid_pct": f"{analysis.bid_pct:.1f}%",
                "ask_pct": f"{analysis.ask_pct:.1f}%",
            },
        )

    def _detect_walls(
        self,
        orderbook: OrderBook,
        analysis: OrderBookAnalysis,
    ) -> list[WhaleSignal]:
        """Detect large order walls."""
        signals: list[WhaleSignal] = []
        min_value = self.thresholds.wall_min_value
        min_pct = self.thresholds.wall_pct_threshold

        total_book = analysis.total_book_value
        if total_book == 0:
            return signals

        # Check bid wall
        if analysis.largest_bid_wall >= min_value:
            wall_pct = float(analysis.largest_bid_wall / total_book)
            if wall_pct >= min_pct:
                strength = min(1.0, wall_pct / 0.05)  # 5% of book = 1.0

                signals.append(WhaleSignal.create(
                    signal_type=WhaleSignalType.LARGE_WALL,
                    symbol=analysis.symbol,
                    strength=strength,
                    direction=SignalDirection.BULLISH,  # Support wall
                    value_usd=analysis.largest_bid_wall,
                    source=SignalSource.ORDER_BOOK,
                    metadata={
                        "side": "bid",
                        "price": str(analysis.bid_wall_price),
                        "value": str(analysis.largest_bid_wall),
                        "pct_of_book": f"{wall_pct * 100:.2f}%",
                    },
                ))

        # Check ask wall
        if analysis.largest_ask_wall >= min_value:
            wall_pct = float(analysis.largest_ask_wall / total_book)
            if wall_pct >= min_pct:
                strength = min(1.0, wall_pct / 0.05)

                signals.append(WhaleSignal.create(
                    signal_type=WhaleSignalType.LARGE_WALL,
                    symbol=analysis.symbol,
                    strength=strength,
                    direction=SignalDirection.BEARISH,  # Resistance wall
                    value_usd=analysis.largest_ask_wall,
                    source=SignalSource.ORDER_BOOK,
                    metadata={
                        "side": "ask",
                        "price": str(analysis.ask_wall_price),
                        "value": str(analysis.largest_ask_wall),
                        "pct_of_book": f"{wall_pct * 100:.2f}%",
                    },
                ))

        return signals

    def _detect_ladders(
        self,
        orderbook: OrderBook,
        analysis: OrderBookAnalysis,
    ) -> list[WhaleSignal]:
        """
        Detect ladder patterns (distributed whale orders).

        Ladder pattern: Multiple orders of similar size at sequential prices.
        """
        signals: list[WhaleSignal] = []
        min_orders = 3  # Minimum orders to form a ladder
        size_tolerance = 0.1  # 10% size variance allowed

        for side, orders in [("bid", orderbook.bids), ("ask", orderbook.asks)]:
            if len(orders) < min_orders:
                continue

            # Look for sequences of similar-sized orders
            i = 0
            while i < len(orders) - min_orders + 1:
                base_size = Decimal(str(orders[i][1]))
                ladder_orders = [(orders[i][0], orders[i][1])]

                for j in range(i + 1, len(orders)):
                    size = Decimal(str(orders[j][1]))
                    # Check if size is within tolerance
                    if abs(float((size - base_size) / base_size)) <= size_tolerance:
                        ladder_orders.append((orders[j][0], orders[j][1]))
                    else:
                        break

                if len(ladder_orders) >= min_orders:
                    # Calculate total ladder value
                    total_value = sum(
                        Decimal(str(p)) * Decimal(str(s))
                        for p, s in ladder_orders
                    )

                    if total_value >= self.thresholds.wall_min_value:
                        prices = [Decimal(str(p)) for p, _ in ladder_orders]
                        direction = (
                            SignalDirection.BULLISH if side == "bid"
                            else SignalDirection.BEARISH
                        )

                        # Strength based on ladder size
                        strength = min(
                            1.0,
                            float(total_value / (self.thresholds.wall_min_value * 5))
                        )

                        signals.append(WhaleSignal.create(
                            signal_type=WhaleSignalType.LADDER_WALL,
                            symbol=analysis.symbol,
                            strength=strength,
                            direction=direction,
                            value_usd=total_value,
                            source=SignalSource.ORDER_BOOK,
                            metadata={
                                "side": side,
                                "num_orders": len(ladder_orders),
                                "price_range": f"{min(prices)}-{max(prices)}",
                                "order_size": str(base_size),
                                "total_value": str(total_value),
                            },
                        ))

                    # Skip past this ladder
                    i += len(ladder_orders)
                else:
                    i += 1

        return signals

    def _deduplicate_signals(self, signals: list[WhaleSignal]) -> list[WhaleSignal]:
        """Remove duplicate signals within short time window."""
        unique_signals: list[WhaleSignal] = []
        now = time.time_ns()
        window_ns = 60_000_000_000  # 60 seconds

        for signal in signals:
            # Create dedup key
            key = f"{signal.signal_type.value}:{signal.symbol}:{signal.direction.value}"
            key_hash = hash(key)

            # Check if we've seen this recently
            is_duplicate = False
            for seen_key, seen_time in self._recent_signals:
                if seen_key == key and (now - seen_time) < window_ns:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_signals.append(signal)
                self._recent_signals.append((key, now))

        return unique_signals

    def reset(self) -> None:
        """Reset analyzer state."""
        self._volume_history.clear()
        self._recent_signals.clear()


# =============================================================================
# Exports
# =============================================================================

__all__ = ["OrderFlowAnalyzer"]
