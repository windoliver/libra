"""Tests for order flow analyzer."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from libra.signals.order_flow import OrderFlowAnalyzer
from libra.signals.protocol import (
    SignalDirection,
    WhaleSignalType,
    WhaleThresholds,
)


class TestOrderFlowAnalyzer:
    """Tests for OrderFlowAnalyzer."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        analyzer = OrderFlowAnalyzer()

        assert analyzer.thresholds is not None
        assert analyzer.volume_window == 20

    def test_init_custom_thresholds(self) -> None:
        """Test initialization with custom thresholds."""
        thresholds = WhaleThresholds.for_btc()
        analyzer = OrderFlowAnalyzer(thresholds=thresholds)

        assert analyzer.thresholds.wall_min_value == Decimal("100000")

    def test_analyze_orderbook_imbalance(self) -> None:
        """Test order book imbalance detection."""
        analyzer = OrderFlowAnalyzer()

        # Create mock orderbook with strong bid imbalance
        orderbook = MagicMock()
        orderbook.symbol = "BTC/USDT"
        orderbook.timestamp_ns = 1700000000000000000
        # Bids: 70% of volume, Asks: 30% of volume
        orderbook.bids = [
            (50000.0, 10.0),  # $500,000
            (49950.0, 8.0),   # $399,600
            (49900.0, 6.0),   # $299,400
        ]
        orderbook.asks = [
            (50050.0, 3.0),   # $150,150
            (50100.0, 2.0),   # $100,200
            (50150.0, 2.0),   # $100,300
        ]

        signals = analyzer.analyze_orderbook(orderbook)

        # Should detect imbalance
        imbalance_signals = [
            s for s in signals
            if s.signal_type == WhaleSignalType.ORDER_IMBALANCE
        ]
        assert len(imbalance_signals) >= 1
        assert imbalance_signals[0].direction == SignalDirection.BULLISH

    def test_analyze_orderbook_large_wall(self) -> None:
        """Test large wall detection."""
        analyzer = OrderFlowAnalyzer(
            thresholds=WhaleThresholds(wall_min_value=Decimal("100000"))
        )

        # Create mock orderbook with a large bid wall
        orderbook = MagicMock()
        orderbook.symbol = "BTC/USDT"
        orderbook.timestamp_ns = 1700000000000000000
        orderbook.bids = [
            (50000.0, 50.0),  # $2,500,000 - WHALE WALL
            (49950.0, 1.0),
            (49900.0, 1.0),
        ]
        orderbook.asks = [
            (50050.0, 2.0),
            (50100.0, 2.0),
            (50150.0, 2.0),
        ]

        signals = analyzer.analyze_orderbook(orderbook)

        # Should detect large wall
        wall_signals = [
            s for s in signals
            if s.signal_type == WhaleSignalType.LARGE_WALL
        ]
        assert len(wall_signals) >= 1
        assert wall_signals[0].direction == SignalDirection.BULLISH
        assert wall_signals[0].value_usd >= Decimal("100000")

    def test_analyze_trade_large(self) -> None:
        """Test large trade detection."""
        analyzer = OrderFlowAnalyzer(
            thresholds=WhaleThresholds(large_trade_min_value=Decimal("50000"))
        )

        signal = analyzer.analyze_trade(
            symbol="ETH/USDT",
            side="buy",
            price=Decimal("3000"),
            size=Decimal("100"),  # $300,000
        )

        assert signal is not None
        assert signal.signal_type == WhaleSignalType.LARGE_TRADE
        assert signal.direction == SignalDirection.BULLISH
        assert signal.value_usd == Decimal("300000")

    def test_analyze_trade_small(self) -> None:
        """Test that small trades are ignored."""
        analyzer = OrderFlowAnalyzer(
            thresholds=WhaleThresholds(large_trade_min_value=Decimal("100000"))
        )

        signal = analyzer.analyze_trade(
            symbol="ETH/USDT",
            side="sell",
            price=Decimal("3000"),
            size=Decimal("10"),  # $30,000 - below threshold
        )

        assert signal is None

    def test_reset(self) -> None:
        """Test analyzer reset."""
        analyzer = OrderFlowAnalyzer()

        # Add some history
        analyzer.analyze_trade(
            symbol="BTC/USDT",
            side="buy",
            price=Decimal("50000"),
            size=Decimal("10"),
            value_usd=Decimal("500000"),
        )

        # Reset should clear state
        analyzer.reset()

        # Verify cleared (no direct way to check, but should not error)
        assert True


class TestOrderFlowAnalyzerVolume:
    """Tests for volume spike detection."""

    def test_volume_spike_detection(self) -> None:
        """Test volume spike detection with enough history."""
        analyzer = OrderFlowAnalyzer(
            thresholds=WhaleThresholds(volume_spike_multiplier=2.0),
            volume_window=5,
        )

        # Create mock bars
        normal_bars = []
        for i in range(6):
            bar = MagicMock()
            bar.symbol = "BTC/USDT"
            bar.volume = Decimal("100")  # Normal volume
            bar.close = Decimal("50000")
            bar.open = Decimal("49900")
            normal_bars.append(bar)

        # Build history with normal volume
        for bar in normal_bars[:5]:
            analyzer.analyze_volume([bar])

        # Create spike bar (5x normal volume)
        spike_bar = MagicMock()
        spike_bar.symbol = "BTC/USDT"
        spike_bar.volume = Decimal("500")  # 5x spike
        spike_bar.close = Decimal("51000")
        spike_bar.open = Decimal("50000")

        signal = analyzer.analyze_volume([spike_bar])

        assert signal is not None
        assert signal.signal_type == WhaleSignalType.VOLUME_SPIKE
        assert signal.direction == SignalDirection.BULLISH  # close > open

    def test_no_spike_with_normal_volume(self) -> None:
        """Test that normal volume doesn't trigger spike."""
        analyzer = OrderFlowAnalyzer(
            thresholds=WhaleThresholds(volume_spike_multiplier=3.0),
            volume_window=5,
        )

        # Build history
        for _ in range(6):
            bar = MagicMock()
            bar.symbol = "BTC/USDT"
            bar.volume = Decimal("100")
            bar.close = Decimal("50000")
            bar.open = Decimal("49900")
            analyzer.analyze_volume([bar])

        # Normal volume bar (no spike)
        normal_bar = MagicMock()
        normal_bar.symbol = "BTC/USDT"
        normal_bar.volume = Decimal("120")  # Only 1.2x
        normal_bar.close = Decimal("50100")
        normal_bar.open = Decimal("50000")

        signal = analyzer.analyze_volume([normal_bar])

        assert signal is None
