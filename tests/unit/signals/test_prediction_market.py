"""Tests for prediction market whale analyzer."""

from __future__ import annotations

from decimal import Decimal

import pytest

from libra.signals.prediction_market import (
    PredictionMarketBet,
    PredictionMarketState,
    PredictionMarketThresholds,
    PredictionMarketWhaleAnalyzer,
    create_demo_pm_signals,
)
from libra.signals.protocol import (
    SignalDirection,
    SignalSource,
    WhaleSignalType,
)


class TestPredictionMarketThresholds:
    """Tests for PredictionMarketThresholds configuration."""

    def test_default_thresholds(self) -> None:
        """Test default threshold values."""
        thresholds = PredictionMarketThresholds()

        assert thresholds.large_bet_min_usd == Decimal("10000")
        assert thresholds.position_change_pct == 0.05
        assert thresholds.market_move_threshold == 0.02

    def test_polymarket_thresholds(self) -> None:
        """Test Polymarket-specific thresholds."""
        thresholds = PredictionMarketThresholds.for_polymarket()

        assert thresholds.large_bet_min_usd == Decimal("25000")
        assert thresholds.position_change_pct == 0.03

    def test_kalshi_thresholds(self) -> None:
        """Test Kalshi-specific thresholds."""
        thresholds = PredictionMarketThresholds.for_kalshi()

        assert thresholds.large_bet_min_usd == Decimal("5000")


class TestPredictionMarketBet:
    """Tests for PredictionMarketBet data class."""

    def test_create_bet(self) -> None:
        """Test bet creation."""
        bet = PredictionMarketBet(
            market_id="btc-100k-2024",
            market_title="Will BTC reach $100k by Dec 2024?",
            outcome="YES",
            side="buy",
            price=Decimal("0.42"),
            size=Decimal("10000"),
            value_usd=Decimal("4200"),
            timestamp=1700000000,
            trader_address="0x1234",
            source=SignalSource.POLYMARKET,
        )

        assert bet.market_id == "btc-100k-2024"
        assert bet.outcome == "YES"
        assert bet.value_usd == Decimal("4200")


class TestPredictionMarketWhaleAnalyzer:
    """Tests for PredictionMarketWhaleAnalyzer."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        analyzer = PredictionMarketWhaleAnalyzer()

        assert analyzer.thresholds is not None
        assert analyzer.thresholds.large_bet_min_usd == Decimal("10000")

    def test_init_custom_thresholds(self) -> None:
        """Test initialization with custom thresholds."""
        thresholds = PredictionMarketThresholds.for_polymarket()
        analyzer = PredictionMarketWhaleAnalyzer(thresholds=thresholds)

        assert analyzer.thresholds.large_bet_min_usd == Decimal("25000")

    def test_analyze_large_bet_bullish(self) -> None:
        """Test large bet detection for bullish YES bet."""
        analyzer = PredictionMarketWhaleAnalyzer(
            thresholds=PredictionMarketThresholds(
                large_bet_min_usd=Decimal("5000")
            )
        )

        bet = PredictionMarketBet(
            market_id="test-market",
            market_title="Test Market?",
            outcome="YES",
            side="buy",
            price=Decimal("0.50"),
            size=Decimal("20000"),
            value_usd=Decimal("10000"),
            timestamp=1700000000,
        )

        signals = analyzer.analyze_bet(bet)

        assert len(signals) >= 1
        large_bet_signal = next(
            (s for s in signals if s.signal_type == WhaleSignalType.PM_LARGE_BET),
            None,
        )
        assert large_bet_signal is not None
        assert large_bet_signal.direction == SignalDirection.BULLISH
        assert large_bet_signal.is_prediction_market is True

    def test_analyze_large_bet_bearish(self) -> None:
        """Test large bet detection for bearish NO bet."""
        analyzer = PredictionMarketWhaleAnalyzer(
            thresholds=PredictionMarketThresholds(
                large_bet_min_usd=Decimal("5000")
            )
        )

        bet = PredictionMarketBet(
            market_id="test-market",
            market_title="Test Market?",
            outcome="NO",
            side="buy",
            price=Decimal("0.30"),
            size=Decimal("30000"),
            value_usd=Decimal("9000"),
            timestamp=1700000000,
        )

        signals = analyzer.analyze_bet(bet)

        large_bet_signal = next(
            (s for s in signals if s.signal_type == WhaleSignalType.PM_LARGE_BET),
            None,
        )
        assert large_bet_signal is not None
        assert large_bet_signal.direction == SignalDirection.BEARISH

    def test_small_bet_no_signal(self) -> None:
        """Test that small bets don't generate signals."""
        analyzer = PredictionMarketWhaleAnalyzer(
            thresholds=PredictionMarketThresholds(
                large_bet_min_usd=Decimal("10000")
            )
        )

        bet = PredictionMarketBet(
            market_id="test-market",
            market_title="Test Market?",
            outcome="YES",
            side="buy",
            price=Decimal("0.50"),
            size=Decimal("1000"),
            value_usd=Decimal("500"),  # Below threshold
            timestamp=1700000000,
        )

        signals = analyzer.analyze_bet(bet)

        assert len(signals) == 0

    def test_smart_money_detection(self) -> None:
        """Test smart money wallet detection."""
        smart_wallet = "0xknownwhale123"
        analyzer = PredictionMarketWhaleAnalyzer(
            thresholds=PredictionMarketThresholds(
                large_bet_min_usd=Decimal("100000"),  # High threshold
                smart_money_addresses={smart_wallet},
            )
        )

        bet = PredictionMarketBet(
            market_id="test-market",
            market_title="Test Market?",
            outcome="YES",
            side="buy",
            price=Decimal("0.50"),
            size=Decimal("1000"),
            value_usd=Decimal("500"),  # Small bet
            timestamp=1700000000,
            trader_address=smart_wallet,
        )

        signals = analyzer.analyze_bet(bet)

        smart_money_signal = next(
            (s for s in signals if s.signal_type == WhaleSignalType.PM_SMART_MONEY),
            None,
        )
        assert smart_money_signal is not None
        assert smart_money_signal.strength == 0.85

    def test_market_move_signal(self) -> None:
        """Test market move detection with price impact."""
        analyzer = PredictionMarketWhaleAnalyzer(
            thresholds=PredictionMarketThresholds(
                large_bet_min_usd=Decimal("100000"),  # Won't trigger large bet
                market_move_threshold=0.01,  # 1% threshold
            )
        )

        market_state = PredictionMarketState(
            market_id="test-market",
            market_title="Test Market?",
            outcomes=["YES", "NO"],
            prices={"YES": Decimal("0.50"), "NO": Decimal("0.50")},
            liquidity_usd=Decimal("100000"),
            volume_24h_usd=Decimal("50000"),
            open_interest=Decimal("200000"),
        )

        bet = PredictionMarketBet(
            market_id="test-market",
            market_title="Test Market?",
            outcome="YES",
            side="buy",
            price=Decimal("0.50"),
            size=Decimal("10000"),
            value_usd=Decimal("5000"),  # 5% of liquidity
            timestamp=1700000000,
        )

        signals = analyzer.analyze_bet(bet, market_state)

        market_move_signal = next(
            (s for s in signals if s.signal_type == WhaleSignalType.PM_MARKET_MOVE),
            None,
        )
        assert market_move_signal is not None
        assert market_move_signal.direction == SignalDirection.BULLISH

    def test_position_change_signal(self) -> None:
        """Test position change detection."""
        analyzer = PredictionMarketWhaleAnalyzer(
            thresholds=PredictionMarketThresholds(
                position_change_pct=0.05,  # 5% threshold
            )
        )

        market_state = PredictionMarketState(
            market_id="test-market",
            market_title="Test Market?",
            outcomes=["YES", "NO"],
            prices={"YES": Decimal("0.60")},
            liquidity_usd=Decimal("1000000"),
            volume_24h_usd=Decimal("100000"),
            open_interest=Decimal("500000"),
        )

        signal = analyzer.analyze_position_change(
            trader_address="0x1234",
            market_id="test-market",
            old_position=Decimal("10000"),
            new_position=Decimal("110000"),  # 10% of liquidity change
            market_state=market_state,
        )

        assert signal is not None
        assert signal.signal_type == WhaleSignalType.PM_POSITION_CHANGE
        assert signal.direction == SignalDirection.BULLISH

    def test_reset(self) -> None:
        """Test analyzer reset."""
        analyzer = PredictionMarketWhaleAnalyzer()
        analyzer.reset()
        # Should not error
        assert True


class TestDemoPMSignals:
    """Tests for demo prediction market signals."""

    def test_create_demo_signals(self) -> None:
        """Test demo signal creation."""
        signals = create_demo_pm_signals()

        assert len(signals) == 4
        assert all(s.is_prediction_market for s in signals)

    def test_demo_signal_types(self) -> None:
        """Test that demo includes all PM signal types."""
        signals = create_demo_pm_signals()
        signal_types = {s.signal_type for s in signals}

        expected = {
            WhaleSignalType.PM_LARGE_BET,
            WhaleSignalType.PM_MARKET_MOVE,
            WhaleSignalType.PM_SMART_MONEY,
            WhaleSignalType.PM_POSITION_CHANGE,
        }
        assert signal_types == expected

    def test_demo_signal_sources(self) -> None:
        """Test demo signal sources."""
        signals = create_demo_pm_signals()
        sources = {s.source for s in signals}

        assert SignalSource.POLYMARKET in sources
        assert SignalSource.KALSHI in sources
