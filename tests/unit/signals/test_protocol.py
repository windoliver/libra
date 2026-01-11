"""Tests for whale signal protocol."""

from __future__ import annotations

from decimal import Decimal

import pytest

from libra.signals.protocol import (
    OrderBookAnalysis,
    SignalDirection,
    SignalSource,
    WhaleSignal,
    WhaleSignalType,
    WhaleThresholds,
    WhaleTransaction,
)


class TestWhaleSignalType:
    """Tests for WhaleSignalType enum."""

    def test_tier1_types(self) -> None:
        """Test Tier 1 (order flow) signal types."""
        tier1_types = [
            WhaleSignalType.ORDER_IMBALANCE,
            WhaleSignalType.LARGE_WALL,
            WhaleSignalType.LADDER_WALL,
            WhaleSignalType.VOLUME_SPIKE,
            WhaleSignalType.LARGE_TRADE,
        ]
        for t in tier1_types:
            assert t.value is not None

    def test_tier2_types(self) -> None:
        """Test Tier 2 (on-chain) signal types."""
        tier2_types = [
            WhaleSignalType.EXCHANGE_INFLOW,
            WhaleSignalType.EXCHANGE_OUTFLOW,
            WhaleSignalType.WHALE_TRANSFER,
            WhaleSignalType.DORMANT_ACTIVATION,
            WhaleSignalType.MINT_BURN,
        ]
        for t in tier2_types:
            assert t.value is not None


class TestWhaleThresholds:
    """Tests for WhaleThresholds configuration."""

    def test_default_thresholds(self) -> None:
        """Test default threshold values."""
        thresholds = WhaleThresholds()

        assert thresholds.imbalance_threshold == 0.3
        assert thresholds.wall_pct_threshold == 0.01
        assert thresholds.volume_spike_multiplier == 3.0

    def test_btc_thresholds(self) -> None:
        """Test BTC-specific thresholds."""
        thresholds = WhaleThresholds.for_btc()

        assert thresholds.wall_min_value == Decimal("100000")
        assert thresholds.whale_transfer_min_usd == Decimal("10000000")

    def test_eth_thresholds(self) -> None:
        """Test ETH-specific thresholds."""
        thresholds = WhaleThresholds.for_eth()

        assert thresholds.wall_min_value == Decimal("50000")
        assert thresholds.whale_transfer_min_usd == Decimal("5000000")

    def test_altcoin_thresholds(self) -> None:
        """Test altcoin thresholds."""
        thresholds = WhaleThresholds.for_altcoin()

        assert thresholds.wall_min_value == Decimal("10000")
        assert thresholds.imbalance_threshold == 0.25


class TestWhaleSignal:
    """Tests for WhaleSignal data class."""

    def test_create_signal(self) -> None:
        """Test signal creation with factory method."""
        signal = WhaleSignal.create(
            signal_type=WhaleSignalType.ORDER_IMBALANCE,
            symbol="BTC/USDT",
            strength=0.75,
            direction=SignalDirection.BULLISH,
            value_usd=Decimal("2500000"),
            source=SignalSource.ORDER_BOOK,
            metadata={"imbalance_ratio": "0.42"},
        )

        assert signal.signal_type == WhaleSignalType.ORDER_IMBALANCE
        assert signal.symbol == "BTC/USDT"
        assert signal.strength == 0.75
        assert signal.direction == SignalDirection.BULLISH
        assert signal.value_usd == Decimal("2500000")
        assert signal.source == SignalSource.ORDER_BOOK
        assert signal.timestamp_ns > 0

    def test_signal_properties(self) -> None:
        """Test signal computed properties."""
        signal = WhaleSignal.create(
            signal_type=WhaleSignalType.LARGE_WALL,
            symbol="ETH/USDT",
            strength=0.85,
            direction=SignalDirection.BEARISH,
            value_usd=Decimal("1000000"),
            source=SignalSource.ORDER_BOOK,
        )

        assert signal.is_bearish is True
        assert signal.is_bullish is False
        assert signal.is_tier1 is True
        assert signal.is_onchain is False
        assert signal.timestamp_sec > 0

    def test_onchain_signal(self) -> None:
        """Test on-chain signal properties."""
        signal = WhaleSignal.create(
            signal_type=WhaleSignalType.EXCHANGE_INFLOW,
            symbol="BTC",
            strength=0.90,
            direction=SignalDirection.BEARISH,
            value_usd=Decimal("15000000"),
            source=SignalSource.WHALE_ALERT,
        )

        assert signal.is_tier1 is False
        assert signal.is_onchain is True

    def test_signal_serialization(self) -> None:
        """Test signal to/from dict conversion."""
        signal = WhaleSignal.create(
            signal_type=WhaleSignalType.VOLUME_SPIKE,
            symbol="SOL/USDT",
            strength=0.65,
            direction=SignalDirection.BULLISH,
            value_usd=Decimal("500000"),
            source=SignalSource.TRADES,
            metadata={"volume_ratio": "3.5x"},
        )

        data = signal.to_dict()
        restored = WhaleSignal.from_dict(data)

        assert restored.signal_type == signal.signal_type
        assert restored.symbol == signal.symbol
        assert restored.strength == signal.strength
        assert restored.direction == signal.direction
        assert restored.value_usd == signal.value_usd

    def test_invalid_strength(self) -> None:
        """Test validation of signal strength."""
        with pytest.raises(ValueError):
            WhaleSignal.create(
                signal_type=WhaleSignalType.ORDER_IMBALANCE,
                symbol="BTC/USDT",
                strength=1.5,  # Invalid - must be 0-1
                direction=SignalDirection.BULLISH,
                value_usd=Decimal("1000000"),
                source=SignalSource.ORDER_BOOK,
            )


class TestWhaleTransaction:
    """Tests for WhaleTransaction data class."""

    def test_create_transaction(self) -> None:
        """Test transaction creation."""
        tx = WhaleTransaction(
            tx_hash="0xabc123",
            blockchain="ethereum",
            timestamp=1700000000,
            from_address="0x1234",
            to_address="0x5678",
            amount=Decimal("5000"),
            amount_usd=Decimal("15000000"),
            symbol="ETH",
            tx_type="transfer",
            from_type="whale",
            to_type="exchange",
        )

        assert tx.tx_hash == "0xabc123"
        assert tx.blockchain == "ethereum"
        assert tx.is_exchange_inflow is True
        assert tx.is_exchange_outflow is False

    def test_exchange_outflow(self) -> None:
        """Test exchange outflow detection."""
        tx = WhaleTransaction(
            tx_hash="0xdef456",
            blockchain="bitcoin",
            timestamp=1700000000,
            from_address="bc1qexchange",
            to_address="bc1qwhale",
            amount=Decimal("500"),
            amount_usd=Decimal("25000000"),
            symbol="BTC",
            tx_type="transfer",
            from_type="exchange",
            to_type="whale",
        )

        assert tx.is_exchange_inflow is False
        assert tx.is_exchange_outflow is True

    def test_to_whale_signal(self) -> None:
        """Test conversion to WhaleSignal."""
        tx = WhaleTransaction(
            tx_hash="0xghi789",
            blockchain="ethereum",
            timestamp=1700000000,
            from_address="0xwhale",
            to_address="0xexchange",
            amount=Decimal("10000"),
            amount_usd=Decimal("30000000"),
            symbol="ETH",
            tx_type="transfer",
            from_type="whale",
            to_type="exchange",
        )

        signal = tx.to_whale_signal()

        assert signal.signal_type == WhaleSignalType.EXCHANGE_INFLOW
        assert signal.direction == SignalDirection.BEARISH
        assert signal.symbol == "ETH"
        assert signal.value_usd == Decimal("30000000")
        assert signal.source == SignalSource.WHALE_ALERT


class TestOrderBookAnalysis:
    """Tests for OrderBookAnalysis data class."""

    def test_create_analysis(self) -> None:
        """Test analysis creation."""
        analysis = OrderBookAnalysis(
            symbol="BTC/USDT",
            timestamp_ns=1700000000000000000,
            bid_volume=Decimal("5000000"),
            ask_volume=Decimal("3000000"),
            imbalance_ratio=0.25,
            largest_bid_wall=Decimal("500000"),
            largest_ask_wall=Decimal("300000"),
        )

        assert analysis.symbol == "BTC/USDT"
        assert analysis.bid_volume == Decimal("5000000")
        assert analysis.has_strong_imbalance is False  # 0.25 < 0.3

    def test_strong_imbalance(self) -> None:
        """Test strong imbalance detection."""
        analysis = OrderBookAnalysis(
            symbol="ETH/USDT",
            timestamp_ns=1700000000000000000,
            bid_volume=Decimal("7000000"),
            ask_volume=Decimal("3000000"),
            imbalance_ratio=0.4,
            largest_bid_wall=Decimal("500000"),
            largest_ask_wall=Decimal("200000"),
        )

        assert analysis.has_strong_imbalance is True
        assert analysis.imbalance_direction == SignalDirection.BULLISH

    def test_bearish_imbalance(self) -> None:
        """Test bearish imbalance direction."""
        analysis = OrderBookAnalysis(
            symbol="SOL/USDT",
            timestamp_ns=1700000000000000000,
            bid_volume=Decimal("2000000"),
            ask_volume=Decimal("6000000"),
            imbalance_ratio=-0.5,
            largest_bid_wall=Decimal("100000"),
            largest_ask_wall=Decimal("400000"),
        )

        assert analysis.has_strong_imbalance is True
        assert analysis.imbalance_direction == SignalDirection.BEARISH
