"""Tests for stock whale analyzer."""

from __future__ import annotations

from decimal import Decimal

import pytest

from libra.signals.stock_whale import (
    BlockTrade,
    DarkPoolTrade,
    InsiderTrade,
    InsiderTransactionType,
    InstitutionalHolding,
    OptionsFlow,
    OptionType,
    OrderType,
    StockWhaleAnalyzer,
    StockWhaleThresholds,
    create_demo_stock_signals,
)
from libra.signals.protocol import (
    SignalDirection,
    SignalSource,
    WhaleSignalType,
)


class TestStockWhaleThresholds:
    """Tests for StockWhaleThresholds configuration."""

    def test_default_thresholds(self) -> None:
        """Test default threshold values."""
        thresholds = StockWhaleThresholds()

        assert thresholds.unusual_options_min_premium == Decimal("100000")
        assert thresholds.dark_pool_min_value == Decimal("1000000")
        assert thresholds.insider_min_value == Decimal("100000")

    def test_large_cap_thresholds(self) -> None:
        """Test large-cap stock thresholds."""
        thresholds = StockWhaleThresholds.for_large_cap()

        assert thresholds.unusual_options_min_premium == Decimal("250000")
        assert thresholds.dark_pool_min_value == Decimal("5000000")

    def test_small_cap_thresholds(self) -> None:
        """Test small-cap stock thresholds."""
        thresholds = StockWhaleThresholds.for_small_cap()

        assert thresholds.unusual_options_min_premium == Decimal("25000")
        assert thresholds.block_trade_min_shares == 5000


class TestOptionsFlow:
    """Tests for OptionsFlow data class."""

    def test_create_options_flow(self) -> None:
        """Test options flow creation."""
        flow = OptionsFlow(
            symbol="NVDA",
            option_type=OptionType.CALL,
            strike=Decimal("950"),
            expiration="2024-01-19",
            premium=Decimal("500000"),
            volume=5000,
            open_interest=1000,
            order_type=OrderType.SWEEP,
            side="buy",
            timestamp=1700000000,
        )

        assert flow.symbol == "NVDA"
        assert flow.option_type == OptionType.CALL
        assert flow.volume_oi_ratio == 5.0

    def test_volume_oi_ratio_zero_oi(self) -> None:
        """Test volume/OI ratio with zero OI."""
        flow = OptionsFlow(
            symbol="AAPL",
            option_type=OptionType.PUT,
            strike=Decimal("200"),
            expiration="2024-02-16",
            premium=Decimal("100000"),
            volume=1000,
            open_interest=0,
            order_type=OrderType.BLOCK,
            side="buy",
            timestamp=1700000000,
        )

        assert flow.volume_oi_ratio == float("inf")


class TestStockWhaleAnalyzer:
    """Tests for StockWhaleAnalyzer."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        analyzer = StockWhaleAnalyzer()

        assert analyzer.thresholds is not None
        assert analyzer.thresholds.unusual_options_min_premium == Decimal("100000")

    def test_analyze_unusual_options(self) -> None:
        """Test unusual options activity detection."""
        analyzer = StockWhaleAnalyzer(
            thresholds=StockWhaleThresholds(
                unusual_options_min_premium=Decimal("50000"),
                unusual_options_volume_ratio=2.0,
            )
        )

        flow = OptionsFlow(
            symbol="TSLA",
            option_type=OptionType.CALL,
            strike=Decimal("250"),
            expiration="2024-01-19",
            premium=Decimal("100000"),
            volume=3000,
            open_interest=1000,  # 3x ratio
            order_type=OrderType.BLOCK,
            side="buy",
            timestamp=1700000000,
        )

        signals = analyzer.analyze_options_flow(flow)

        unusual_signal = next(
            (s for s in signals if s.signal_type == WhaleSignalType.OPTIONS_UNUSUAL),
            None,
        )
        assert unusual_signal is not None
        assert unusual_signal.direction == SignalDirection.BULLISH
        assert unusual_signal.is_stock is True

    def test_analyze_put_options_bearish(self) -> None:
        """Test that put buys are bearish."""
        analyzer = StockWhaleAnalyzer(
            thresholds=StockWhaleThresholds(
                unusual_options_min_premium=Decimal("50000"),
                unusual_options_volume_ratio=2.0,
            )
        )

        flow = OptionsFlow(
            symbol="SPY",
            option_type=OptionType.PUT,
            strike=Decimal("450"),
            expiration="2024-01-19",
            premium=Decimal("150000"),
            volume=5000,
            open_interest=1000,
            order_type=OrderType.BLOCK,
            side="buy",
            timestamp=1700000000,
        )

        signals = analyzer.analyze_options_flow(flow)

        unusual_signal = next(
            (s for s in signals if s.signal_type == WhaleSignalType.OPTIONS_UNUSUAL),
            None,
        )
        assert unusual_signal is not None
        assert unusual_signal.direction == SignalDirection.BEARISH

    def test_analyze_options_sweep(self) -> None:
        """Test options sweep detection."""
        analyzer = StockWhaleAnalyzer(
            thresholds=StockWhaleThresholds(
                options_sweep_min_premium=Decimal("25000"),
            )
        )

        flow = OptionsFlow(
            symbol="AMD",
            option_type=OptionType.CALL,
            strike=Decimal("180"),
            expiration="2024-02-16",
            premium=Decimal("75000"),
            volume=2000,
            open_interest=500,
            order_type=OrderType.SWEEP,
            side="buy",
            timestamp=1700000000,
        )

        signals = analyzer.analyze_options_flow(flow)

        sweep_signal = next(
            (s for s in signals if s.signal_type == WhaleSignalType.OPTIONS_SWEEP),
            None,
        )
        assert sweep_signal is not None
        assert sweep_signal.direction == SignalDirection.BULLISH
        assert "sweep" in sweep_signal.metadata.get("order_type", "").lower()

    def test_analyze_dark_pool(self) -> None:
        """Test dark pool trade detection."""
        analyzer = StockWhaleAnalyzer(
            thresholds=StockWhaleThresholds(
                dark_pool_min_value=Decimal("500000"),
                dark_pool_pct_of_volume=0.01,  # 1% threshold
            )
        )

        trade = DarkPoolTrade(
            symbol="AAPL",
            price=Decimal("200.00"),
            shares=10000,
            value_usd=Decimal("2000000"),
            timestamp=1700000000,
            venue="SIGMA-X",
            pct_of_daily_volume=0.02,  # 2% > 1% threshold
        )

        signal = analyzer.analyze_dark_pool(trade)

        assert signal is not None
        assert signal.signal_type == WhaleSignalType.DARK_POOL
        assert signal.direction == SignalDirection.NEUTRAL  # Direction unknown

    def test_analyze_small_dark_pool(self) -> None:
        """Test that small dark pool trades are ignored."""
        analyzer = StockWhaleAnalyzer(
            thresholds=StockWhaleThresholds(
                dark_pool_min_value=Decimal("1000000"),
            )
        )

        trade = DarkPoolTrade(
            symbol="AAPL",
            price=Decimal("200.00"),
            shares=1000,
            value_usd=Decimal("200000"),  # Below threshold
            timestamp=1700000000,
        )

        signal = analyzer.analyze_dark_pool(trade)

        assert signal is None

    def test_analyze_block_trade(self) -> None:
        """Test block trade detection."""
        analyzer = StockWhaleAnalyzer(
            thresholds=StockWhaleThresholds(
                block_trade_min_value=Decimal("250000"),
                block_trade_min_shares=5000,
            )
        )

        trade = BlockTrade(
            symbol="MSFT",
            price=Decimal("400.00"),
            shares=10000,
            value_usd=Decimal("4000000"),
            side="buy",
            timestamp=1700000000,
        )

        signal = analyzer.analyze_block_trade(trade)

        assert signal is not None
        assert signal.signal_type == WhaleSignalType.BLOCK_TRADE
        assert signal.direction == SignalDirection.BULLISH

    def test_analyze_insider_buy(self) -> None:
        """Test insider buying detection."""
        analyzer = StockWhaleAnalyzer(
            thresholds=StockWhaleThresholds(
                insider_min_value=Decimal("50000"),
            )
        )

        trade = InsiderTrade(
            symbol="META",
            insider_name="Mark Zuckerberg",
            insider_title="CEO",
            transaction_type=InsiderTransactionType.BUY,
            shares=1000,
            price=Decimal("500.00"),
            value_usd=Decimal("500000"),
            shares_owned_after=10000000,
            filing_date="2024-01-10",
            transaction_date="2024-01-08",
        )

        signal = analyzer.analyze_insider_trade(trade)

        assert signal is not None
        assert signal.signal_type == WhaleSignalType.INSIDER_FILING
        assert signal.direction == SignalDirection.BULLISH
        assert signal.metadata["insider_title"] == "CEO"

    def test_analyze_insider_sell(self) -> None:
        """Test insider selling detection."""
        analyzer = StockWhaleAnalyzer(
            thresholds=StockWhaleThresholds(
                insider_min_value=Decimal("50000"),
            )
        )

        trade = InsiderTrade(
            symbol="AMZN",
            insider_name="Executive",
            insider_title="CFO",
            transaction_type=InsiderTransactionType.SELL,
            shares=5000,
            price=Decimal("180.00"),
            value_usd=Decimal("900000"),
            shares_owned_after=50000,
            filing_date="2024-01-10",
            transaction_date="2024-01-08",
        )

        signal = analyzer.analyze_insider_trade(trade)

        assert signal is not None
        assert signal.direction == SignalDirection.BEARISH

    def test_analyze_insider_exercise_ignored(self) -> None:
        """Test that option exercises are ignored."""
        analyzer = StockWhaleAnalyzer()

        trade = InsiderTrade(
            symbol="GOOG",
            insider_name="Employee",
            insider_title="VP",
            transaction_type=InsiderTransactionType.EXERCISE,
            shares=10000,
            price=Decimal("150.00"),
            value_usd=Decimal("1500000"),
            shares_owned_after=20000,
            filing_date="2024-01-10",
            transaction_date="2024-01-08",
        )

        signal = analyzer.analyze_insider_trade(trade)

        assert signal is None

    def test_analyze_institutional_new_position(self) -> None:
        """Test new institutional position detection."""
        analyzer = StockWhaleAnalyzer(
            thresholds=StockWhaleThresholds(
                inst_new_position_min=Decimal("5000000"),
            )
        )

        holding = InstitutionalHolding(
            symbol="NVDA",
            institution_name="Berkshire Hathaway",
            shares=100000,
            value_usd=Decimal("50000000"),
            pct_of_portfolio=2.5,
            shares_change=100000,
            pct_change=float("inf"),  # New position
            filing_date="2024-01-05",
            report_date="2023-12-31",
        )

        signal = analyzer.analyze_institutional_holding(holding)

        assert signal is not None
        assert signal.signal_type == WhaleSignalType.INST_13F
        assert signal.direction == SignalDirection.BULLISH
        assert signal.metadata["is_new_position"] is True

    def test_analyze_institutional_position_increase(self) -> None:
        """Test institutional position increase detection."""
        analyzer = StockWhaleAnalyzer(
            thresholds=StockWhaleThresholds(
                inst_position_change_pct=0.10,  # 10%
            )
        )

        holding = InstitutionalHolding(
            symbol="MSFT",
            institution_name="Vanguard",
            shares=2000000,
            value_usd=Decimal("800000000"),
            pct_of_portfolio=5.0,
            shares_change=500000,
            pct_change=25.0,  # 25% increase
            filing_date="2024-01-05",
            report_date="2023-12-31",
        )

        signal = analyzer.analyze_institutional_holding(holding)

        assert signal is not None
        assert signal.direction == SignalDirection.BULLISH


class TestDemoStockSignals:
    """Tests for demo stock signals."""

    def test_create_demo_signals(self) -> None:
        """Test demo signal creation."""
        signals = create_demo_stock_signals()

        assert len(signals) == 6
        assert all(s.is_stock for s in signals)

    def test_demo_signal_types(self) -> None:
        """Test that demo includes various stock signal types."""
        signals = create_demo_stock_signals()
        signal_types = {s.signal_type for s in signals}

        expected = {
            WhaleSignalType.OPTIONS_SWEEP,
            WhaleSignalType.OPTIONS_UNUSUAL,
            WhaleSignalType.DARK_POOL,
            WhaleSignalType.INSIDER_FILING,
            WhaleSignalType.INST_13F,
            WhaleSignalType.BLOCK_TRADE,
        }
        assert signal_types == expected

    def test_demo_signal_sources(self) -> None:
        """Test demo signal sources."""
        signals = create_demo_stock_signals()
        sources = {s.source for s in signals}

        assert SignalSource.UNUSUAL_WHALES in sources
        assert SignalSource.SEC_EDGAR in sources
        assert SignalSource.FINRA in sources
