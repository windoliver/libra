"""Tests for funding rate arbitrage strategy."""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from libra.strategies.arbitrage.funding_rate import (
    ArbitrageDirection,
    FundingArbitrageConfig,
    FundingArbitragePosition,
    FundingArbitrageState,
    FundingRateArbitrageStrategy,
    FundingRateData,
    FundingRateMonitor,
    create_demo_funding_rates,
    FUNDING_PERIODS_PER_YEAR,
)
from libra.strategies.protocol import SignalType


# =============================================================================
# FundingRateData Tests
# =============================================================================


class TestFundingRateData:
    """Tests for FundingRateData dataclass."""

    def test_create_funding_rate_data(self) -> None:
        """Test basic creation of FundingRateData."""
        now = time.time_ns()
        next_funding = now + 4 * 3600 * 1_000_000_000  # 4 hours

        data = FundingRateData(
            symbol="BTC/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("0.0001"),
            next_funding_time_ns=next_funding,
            mark_price=Decimal("95000.00"),
            index_price=Decimal("94950.00"),
        )

        assert data.symbol == "BTC/USDT:USDT"
        assert data.exchange == "binance"
        assert data.funding_rate == Decimal("0.0001")
        assert data.mark_price == Decimal("95000.00")
        assert data.index_price == Decimal("94950.00")

    def test_annualized_rate(self) -> None:
        """Test annualized rate calculation."""
        data = FundingRateData(
            symbol="BTC/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("0.0001"),  # 0.01% per 8 hours
            next_funding_time_ns=time.time_ns(),
            mark_price=Decimal("95000.00"),
            index_price=Decimal("95000.00"),
        )

        # 0.01% * 1095 periods = 10.95% APR
        expected_apr = Decimal("0.0001") * FUNDING_PERIODS_PER_YEAR
        assert data.annualized_rate == expected_apr
        assert float(data.annualized_rate) == pytest.approx(0.1095, rel=0.001)

    def test_basis_calculation(self) -> None:
        """Test basis calculation."""
        data = FundingRateData(
            symbol="BTC/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("0.0001"),
            next_funding_time_ns=time.time_ns(),
            mark_price=Decimal("95100.00"),  # 0.1% above index
            index_price=Decimal("95000.00"),
        )

        # Basis = (95100 - 95000) / 95000 = 0.001052...
        assert float(data.basis) == pytest.approx(0.001052, rel=0.01)
        assert float(data.basis_bps) == pytest.approx(10.52, rel=0.01)

    def test_basis_zero_index_price(self) -> None:
        """Test basis calculation with zero index price."""
        data = FundingRateData(
            symbol="BTC/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("0.0001"),
            next_funding_time_ns=time.time_ns(),
            mark_price=Decimal("95000.00"),
            index_price=Decimal("0"),
        )

        assert data.basis == Decimal("0")

    def test_is_positive_negative(self) -> None:
        """Test positive/negative funding rate checks."""
        positive = FundingRateData(
            symbol="BTC/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("0.0001"),
            next_funding_time_ns=time.time_ns(),
            mark_price=Decimal("95000.00"),
            index_price=Decimal("95000.00"),
        )

        negative = FundingRateData(
            symbol="BTC/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("-0.0001"),
            next_funding_time_ns=time.time_ns(),
            mark_price=Decimal("95000.00"),
            index_price=Decimal("95000.00"),
        )

        assert positive.is_positive is True
        assert positive.is_negative is False
        assert negative.is_positive is False
        assert negative.is_negative is True


# =============================================================================
# FundingArbitragePosition Tests
# =============================================================================


class TestFundingArbitragePosition:
    """Tests for FundingArbitragePosition dataclass."""

    def test_create_position(self) -> None:
        """Test basic position creation."""
        position = FundingArbitragePosition(
            symbol="BTC/USDT:USDT",
            direction=ArbitrageDirection.LONG_SPOT_SHORT_PERP,
            spot_size=Decimal("1.0"),
            perp_size=Decimal("1.0"),
            entry_funding_rate=Decimal("0.0003"),
            entry_spot_price=Decimal("95000.00"),
            entry_perp_price=Decimal("95050.00"),
            entry_time_ns=time.time_ns(),
        )

        assert position.symbol == "BTC/USDT:USDT"
        assert position.direction == ArbitrageDirection.LONG_SPOT_SHORT_PERP
        assert position.cumulative_funding == Decimal("0")
        assert position.funding_payments_count == 0

    def test_net_delta_long_spot_short_perp(self) -> None:
        """Test net delta for long spot / short perp position."""
        position = FundingArbitragePosition(
            symbol="BTC/USDT:USDT",
            direction=ArbitrageDirection.LONG_SPOT_SHORT_PERP,
            spot_size=Decimal("1.0"),
            perp_size=Decimal("0.98"),  # Slight imbalance
            entry_funding_rate=Decimal("0.0003"),
            entry_spot_price=Decimal("95000.00"),
            entry_perp_price=Decimal("95050.00"),
            entry_time_ns=time.time_ns(),
        )

        # Long spot (1.0) - Short perp (0.98) = 0.02 net long
        assert position.net_delta == Decimal("0.02")

    def test_net_delta_short_spot_long_perp(self) -> None:
        """Test net delta for short spot / long perp position."""
        position = FundingArbitragePosition(
            symbol="BTC/USDT:USDT",
            direction=ArbitrageDirection.SHORT_SPOT_LONG_PERP,
            spot_size=Decimal("1.0"),
            perp_size=Decimal("1.02"),  # Slight imbalance
            entry_funding_rate=Decimal("-0.0003"),
            entry_spot_price=Decimal("95000.00"),
            entry_perp_price=Decimal("94950.00"),
            entry_time_ns=time.time_ns(),
        )

        # Long perp (1.02) - Short spot (1.0) = 0.02 net long
        assert position.net_delta == Decimal("0.02")

    def test_add_funding_payment(self) -> None:
        """Test adding funding payments."""
        position = FundingArbitragePosition(
            symbol="BTC/USDT:USDT",
            direction=ArbitrageDirection.LONG_SPOT_SHORT_PERP,
            spot_size=Decimal("1.0"),
            perp_size=Decimal("1.0"),
            entry_funding_rate=Decimal("0.0003"),
            entry_spot_price=Decimal("95000.00"),
            entry_perp_price=Decimal("95050.00"),
            entry_time_ns=time.time_ns(),
        )

        # Add first funding payment
        position.add_funding_payment(Decimal("28.50"))  # 0.03% of 95000
        assert position.cumulative_funding == Decimal("28.50")
        assert position.funding_payments_count == 1

        # Add second funding payment
        position.add_funding_payment(Decimal("27.00"))
        assert position.cumulative_funding == Decimal("55.50")
        assert position.funding_payments_count == 2

    def test_avg_funding_per_period(self) -> None:
        """Test average funding per period calculation."""
        position = FundingArbitragePosition(
            symbol="BTC/USDT:USDT",
            direction=ArbitrageDirection.LONG_SPOT_SHORT_PERP,
            spot_size=Decimal("1.0"),
            perp_size=Decimal("1.0"),
            entry_funding_rate=Decimal("0.0003"),
            entry_spot_price=Decimal("95000.00"),
            entry_perp_price=Decimal("95050.00"),
            entry_time_ns=time.time_ns(),
        )

        # No payments yet
        assert position.avg_funding_per_period == Decimal("0")

        # Add payments
        position.add_funding_payment(Decimal("30.00"))
        position.add_funding_payment(Decimal("20.00"))
        position.add_funding_payment(Decimal("25.00"))

        # Average = 75 / 3 = 25
        assert position.avg_funding_per_period == Decimal("25")

    def test_update_pnl_long_spot_short_perp(self) -> None:
        """Test P&L update for long spot / short perp."""
        position = FundingArbitragePosition(
            symbol="BTC/USDT:USDT",
            direction=ArbitrageDirection.LONG_SPOT_SHORT_PERP,
            spot_size=Decimal("1.0"),
            perp_size=Decimal("1.0"),
            entry_funding_rate=Decimal("0.0003"),
            entry_spot_price=Decimal("95000.00"),
            entry_perp_price=Decimal("95000.00"),
            entry_time_ns=time.time_ns(),
        )

        # Price goes up - spot profits, perp loses (but they should offset)
        position.update_pnl(Decimal("96000.00"), Decimal("96000.00"))

        # Spot P&L: (96000 - 95000) * 1 = +1000
        # Perp P&L: (95000 - 96000) * 1 = -1000
        assert position.spot_pnl == Decimal("1000.00")
        assert position.perp_pnl == Decimal("-1000.00")

        # Total should be ~0 (delta neutral)
        assert position.total_pnl == Decimal("0")

    def test_update_pnl_short_spot_long_perp(self) -> None:
        """Test P&L update for short spot / long perp."""
        position = FundingArbitragePosition(
            symbol="BTC/USDT:USDT",
            direction=ArbitrageDirection.SHORT_SPOT_LONG_PERP,
            spot_size=Decimal("1.0"),
            perp_size=Decimal("1.0"),
            entry_funding_rate=Decimal("-0.0003"),
            entry_spot_price=Decimal("95000.00"),
            entry_perp_price=Decimal("95000.00"),
            entry_time_ns=time.time_ns(),
        )

        # Price goes down - spot profits (short), perp loses (long)
        position.update_pnl(Decimal("94000.00"), Decimal("94000.00"))

        # Spot P&L: (95000 - 94000) * 1 = +1000 (short profits on down)
        # Perp P&L: (94000 - 95000) * 1 = -1000 (long loses on down)
        assert position.spot_pnl == Decimal("1000.00")
        assert position.perp_pnl == Decimal("-1000.00")

        # Total should be ~0 (delta neutral)
        assert position.total_pnl == Decimal("0")

    def test_total_pnl_with_funding(self) -> None:
        """Test total P&L including funding."""
        position = FundingArbitragePosition(
            symbol="BTC/USDT:USDT",
            direction=ArbitrageDirection.LONG_SPOT_SHORT_PERP,
            spot_size=Decimal("1.0"),
            perp_size=Decimal("1.0"),
            entry_funding_rate=Decimal("0.0003"),
            entry_spot_price=Decimal("95000.00"),
            entry_perp_price=Decimal("95000.00"),
            entry_time_ns=time.time_ns(),
        )

        # Add funding payments
        position.add_funding_payment(Decimal("50.00"))

        # Update P&L (price unchanged)
        position.update_pnl(Decimal("95000.00"), Decimal("95000.00"))

        # Total = funding + spot_pnl + perp_pnl - fees
        # = 50 + 0 + 0 - 0 = 50
        assert position.total_pnl == Decimal("50.00")


# =============================================================================
# FundingArbitrageConfig Tests
# =============================================================================


class TestFundingArbitrageConfig:
    """Tests for FundingArbitrageConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FundingArbitrageConfig(
            symbol="BTC/USDT",
            timeframe="1h",
        )

        assert config.min_funding_rate == Decimal("0.0001")
        assert config.min_annualized_return == Decimal("0.10")
        assert config.max_leverage == 3
        assert config.max_basis_deviation == Decimal("0.02")

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = FundingArbitrageConfig(
            symbol="ETH/USDT",
            timeframe="1h",
            min_funding_rate=Decimal("0.0002"),
            max_position_size_usd=Decimal("50000"),
            max_leverage=5,
        )

        assert config.min_funding_rate == Decimal("0.0002")
        assert config.max_position_size_usd == Decimal("50000")
        assert config.max_leverage == 5

    def test_invalid_leverage(self) -> None:
        """Test that invalid leverage raises error."""
        with pytest.raises(ValueError, match="max_leverage must be >= 1"):
            FundingArbitrageConfig(
                symbol="BTC/USDT",
                timeframe="1h",
                max_leverage=0,
            )

    def test_invalid_min_funding_rate(self) -> None:
        """Test that negative min_funding_rate raises error."""
        with pytest.raises(ValueError, match="min_funding_rate must be >= 0"):
            FundingArbitrageConfig(
                symbol="BTC/USDT",
                timeframe="1h",
                min_funding_rate=Decimal("-0.0001"),
            )

    def test_invalid_position_size(self) -> None:
        """Test that invalid position size raises error."""
        with pytest.raises(ValueError, match="max_position_size_usd must be > 0"):
            FundingArbitrageConfig(
                symbol="BTC/USDT",
                timeframe="1h",
                max_position_size_usd=Decimal("0"),
            )


# =============================================================================
# FundingRateMonitor Tests
# =============================================================================


class TestFundingRateMonitor:
    """Tests for FundingRateMonitor."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        monitor = FundingRateMonitor()

        assert monitor.providers == {}
        assert monitor.symbols == []
        assert monitor.poll_interval == 60
        assert not monitor.is_running

    def test_init_with_config(self) -> None:
        """Test initialization with configuration."""
        mock_provider = MagicMock()

        monitor = FundingRateMonitor(
            providers={"binance": mock_provider},
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            poll_interval_seconds=30,
        )

        assert "binance" in monitor.providers
        assert len(monitor.symbols) == 2
        assert monitor.poll_interval == 30

    def test_get_rate_not_found(self) -> None:
        """Test get_rate when no data available."""
        monitor = FundingRateMonitor()

        result = monitor.get_rate("BTC/USDT:USDT")
        assert result is None

    def test_get_best_opportunity_empty(self) -> None:
        """Test get_best_opportunity when no data available."""
        monitor = FundingRateMonitor()

        result = monitor.get_best_opportunity()
        assert result is None

    def test_get_all_opportunities_empty(self) -> None:
        """Test get_all_opportunities when no data available."""
        monitor = FundingRateMonitor()

        result = monitor.get_all_opportunities()
        assert result == []

    def test_poll_rates(self) -> None:
        """Test polling funding rates from providers."""
        # Create mock provider
        mock_provider = MagicMock()
        mock_rate_data = FundingRateData(
            symbol="BTC/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("0.0003"),
            next_funding_time_ns=time.time_ns() + 3600 * 1_000_000_000,
            mark_price=Decimal("95000.00"),
            index_price=Decimal("94950.00"),
        )
        mock_provider.fetch_funding_rate = AsyncMock(return_value=mock_rate_data)

        monitor = FundingRateMonitor(
            providers={"binance": mock_provider},
            symbols=["BTC/USDT:USDT"],
        )

        # Poll rates
        rates = asyncio.run(monitor.poll_rates())

        # Verify
        assert "BTC/USDT:USDT" in rates
        assert "binance" in rates["BTC/USDT:USDT"]
        assert rates["BTC/USDT:USDT"]["binance"].funding_rate == Decimal("0.0003")

    def test_get_best_opportunity_with_data(self) -> None:
        """Test get_best_opportunity with actual data."""
        # Create mock providers
        mock_provider = MagicMock()

        async def mock_fetch(symbol: str) -> FundingRateData:
            rates = {
                "BTC/USDT:USDT": Decimal("0.0002"),
                "ETH/USDT:USDT": Decimal("0.0005"),  # Highest
                "SOL/USDT:USDT": Decimal("0.0001"),
            }
            return FundingRateData(
                symbol=symbol,
                exchange="binance",
                funding_rate=rates.get(symbol, Decimal("0")),
                next_funding_time_ns=time.time_ns() + 3600 * 1_000_000_000,
                mark_price=Decimal("100.00"),
                index_price=Decimal("100.00"),
            )

        mock_provider.fetch_funding_rate = AsyncMock(side_effect=mock_fetch)

        monitor = FundingRateMonitor(
            providers={"binance": mock_provider},
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
        )

        # Poll rates
        asyncio.run(monitor.poll_rates())

        # Get best opportunity
        best = monitor.get_best_opportunity()

        assert best is not None
        assert best.symbol == "ETH/USDT:USDT"
        assert best.funding_rate == Decimal("0.0005")


# =============================================================================
# FundingRateArbitrageStrategy Tests
# =============================================================================


class TestFundingRateArbitrageStrategy:
    """Tests for FundingRateArbitrageStrategy."""

    @pytest.fixture
    def config(self) -> FundingArbitrageConfig:
        """Create test configuration."""
        return FundingArbitrageConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            min_funding_rate=Decimal("0.0001"),
            min_annualized_return=Decimal("0.10"),
            max_position_size_usd=Decimal("100000"),
            max_positions=3,
        )

    @pytest.fixture
    def strategy(self, config: FundingArbitrageConfig) -> FundingRateArbitrageStrategy:
        """Create test strategy."""
        return FundingRateArbitrageStrategy(config)

    def test_init(self, strategy: FundingRateArbitrageStrategy) -> None:
        """Test strategy initialization."""
        assert strategy.name == "funding_rate_arbitrage"
        assert strategy.state == FundingArbitrageState.IDLE
        assert len(strategy.positions) == 0
        assert strategy.total_funding_collected == Decimal("0")

    def test_on_start(self, strategy: FundingRateArbitrageStrategy) -> None:
        """Test strategy start."""
        strategy.on_start()

        assert strategy.state == FundingArbitrageState.IDLE

    def test_on_stop(self, strategy: FundingRateArbitrageStrategy) -> None:
        """Test strategy stop."""
        strategy.on_start()
        strategy.on_stop()

        assert strategy.state == FundingArbitrageState.IDLE

    def test_on_reset(self, strategy: FundingRateArbitrageStrategy) -> None:
        """Test strategy reset clears state."""
        strategy.on_start()

        # Register a position
        strategy.register_position(
            symbol="BTC/USDT:USDT",
            direction=ArbitrageDirection.LONG_SPOT_SHORT_PERP,
            spot_size=Decimal("1.0"),
            perp_size=Decimal("1.0"),
            spot_price=Decimal("95000.00"),
            perp_price=Decimal("95050.00"),
            funding_rate=Decimal("0.0003"),
        )

        assert len(strategy.positions) == 1

        # Reset
        strategy.on_reset()

        assert len(strategy.positions) == 0
        assert strategy.state == FundingArbitrageState.IDLE

    def test_evaluate_entry_positive_funding(self, strategy: FundingRateArbitrageStrategy) -> None:
        """Test entry signal generation for positive funding."""
        strategy.on_start()

        rate_data = FundingRateData(
            symbol="BTC/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("0.0003"),  # 0.03% - above threshold
            next_funding_time_ns=time.time_ns() + 3600 * 1_000_000_000,
            mark_price=Decimal("95000.00"),
            index_price=Decimal("94950.00"),  # Small basis
        )

        signal = strategy.on_funding_rate(rate_data)

        assert signal is not None
        assert signal.signal_type == SignalType.SHORT  # Short perp
        assert signal.symbol == "BTC/USDT:USDT"
        assert signal.metadata["direction"] == ArbitrageDirection.LONG_SPOT_SHORT_PERP.value

    def test_evaluate_entry_negative_funding(self, strategy: FundingRateArbitrageStrategy) -> None:
        """Test entry signal generation for negative funding."""
        strategy.on_start()

        rate_data = FundingRateData(
            symbol="BTC/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("-0.0003"),  # -0.03% - negative funding
            next_funding_time_ns=time.time_ns() + 3600 * 1_000_000_000,
            mark_price=Decimal("94950.00"),
            index_price=Decimal("95000.00"),  # Small basis
        )

        signal = strategy.on_funding_rate(rate_data)

        assert signal is not None
        assert signal.signal_type == SignalType.LONG  # Long perp
        assert signal.metadata["direction"] == ArbitrageDirection.SHORT_SPOT_LONG_PERP.value

    def test_no_entry_below_threshold(self, strategy: FundingRateArbitrageStrategy) -> None:
        """Test no entry when funding rate is below threshold."""
        strategy.on_start()

        rate_data = FundingRateData(
            symbol="BTC/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("0.00005"),  # 0.005% - below 0.01% threshold
            next_funding_time_ns=time.time_ns() + 3600 * 1_000_000_000,
            mark_price=Decimal("95000.00"),
            index_price=Decimal("95000.00"),
        )

        signal = strategy.on_funding_rate(rate_data)

        assert signal is None

    def test_no_entry_excessive_basis(self, strategy: FundingRateArbitrageStrategy) -> None:
        """Test no entry when basis exceeds threshold."""
        strategy.on_start()

        rate_data = FundingRateData(
            symbol="BTC/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("0.0005"),  # Good funding rate
            next_funding_time_ns=time.time_ns() + 3600 * 1_000_000_000,
            mark_price=Decimal("98000.00"),  # 3.15% above index - excessive basis
            index_price=Decimal("95000.00"),
        )

        signal = strategy.on_funding_rate(rate_data)

        assert signal is None  # Basis too high

    def test_no_entry_max_positions(self, strategy: FundingRateArbitrageStrategy) -> None:
        """Test no entry when max positions reached."""
        strategy.on_start()

        # Fill up max positions
        for i in range(strategy.arb_config.max_positions):
            strategy.register_position(
                symbol=f"SYM{i}/USDT:USDT",
                direction=ArbitrageDirection.LONG_SPOT_SHORT_PERP,
                spot_size=Decimal("1.0"),
                perp_size=Decimal("1.0"),
                spot_price=Decimal("100.00"),
                perp_price=Decimal("100.00"),
                funding_rate=Decimal("0.0003"),
            )

        # Try to enter new position
        rate_data = FundingRateData(
            symbol="NEW/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("0.0005"),
            next_funding_time_ns=time.time_ns() + 3600 * 1_000_000_000,
            mark_price=Decimal("100.00"),
            index_price=Decimal("100.00"),
        )

        signal = strategy.on_funding_rate(rate_data)

        assert signal is None  # Max positions reached

    def test_register_position(self, strategy: FundingRateArbitrageStrategy) -> None:
        """Test position registration."""
        strategy.on_start()

        position = strategy.register_position(
            symbol="BTC/USDT:USDT",
            direction=ArbitrageDirection.LONG_SPOT_SHORT_PERP,
            spot_size=Decimal("1.0"),
            perp_size=Decimal("1.0"),
            spot_price=Decimal("95000.00"),
            perp_price=Decimal("95050.00"),
            funding_rate=Decimal("0.0003"),
        )

        assert position.symbol == "BTC/USDT:USDT"
        assert "BTC/USDT:USDT" in strategy.positions
        assert strategy.state == FundingArbitrageState.ACTIVE

    def test_close_position(self, strategy: FundingRateArbitrageStrategy) -> None:
        """Test position closing."""
        strategy.on_start()

        # Register position
        strategy.register_position(
            symbol="BTC/USDT:USDT",
            direction=ArbitrageDirection.LONG_SPOT_SHORT_PERP,
            spot_size=Decimal("1.0"),
            perp_size=Decimal("1.0"),
            spot_price=Decimal("95000.00"),
            perp_price=Decimal("95050.00"),
            funding_rate=Decimal("0.0003"),
        )

        # Close position
        closed = strategy.close_position("BTC/USDT:USDT")

        assert closed is not None
        assert "BTC/USDT:USDT" not in strategy.positions
        assert strategy.state == FundingArbitrageState.IDLE

    def test_on_funding_payment(self, strategy: FundingRateArbitrageStrategy) -> None:
        """Test funding payment recording."""
        strategy.on_start()

        # Register position
        strategy.register_position(
            symbol="BTC/USDT:USDT",
            direction=ArbitrageDirection.LONG_SPOT_SHORT_PERP,
            spot_size=Decimal("1.0"),
            perp_size=Decimal("1.0"),
            spot_price=Decimal("95000.00"),
            perp_price=Decimal("95050.00"),
            funding_rate=Decimal("0.0003"),
        )

        # Record funding payment
        strategy.on_funding_payment("BTC/USDT:USDT", Decimal("28.50"))

        assert strategy.total_funding_collected == Decimal("28.50")
        assert strategy.positions["BTC/USDT:USDT"].cumulative_funding == Decimal("28.50")

    def test_exit_on_rate_reversal(self, strategy: FundingRateArbitrageStrategy) -> None:
        """Test exit signal on rate reversal."""
        strategy.on_start()

        # Register position with positive funding
        position = strategy.register_position(
            symbol="BTC/USDT:USDT",
            direction=ArbitrageDirection.LONG_SPOT_SHORT_PERP,
            spot_size=Decimal("1.0"),
            perp_size=Decimal("1.0"),
            spot_price=Decimal("95000.00"),
            perp_price=Decimal("95050.00"),
            funding_rate=Decimal("0.0003"),
        )

        # Add enough funding payments to allow exit
        for _ in range(strategy.arb_config.min_hold_periods):
            position.add_funding_payment(Decimal("28.50"))

        # Rate reverses to negative
        rate_data = FundingRateData(
            symbol="BTC/USDT:USDT",
            exchange="binance",
            funding_rate=Decimal("-0.0002"),  # Reversed to negative
            next_funding_time_ns=time.time_ns() + 3600 * 1_000_000_000,
            mark_price=Decimal("95000.00"),
            index_price=Decimal("95000.00"),
        )

        signal = strategy.on_funding_rate(rate_data)

        assert signal is not None
        assert signal.signal_type == SignalType.CLOSE_SHORT  # Close the short perp
        assert signal.metadata["exit_reason"] == "rate_reversal_to_negative"

    def test_save_and_load_state(self, strategy: FundingRateArbitrageStrategy) -> None:
        """Test state serialization and deserialization."""
        strategy.on_start()

        # Register position and add funding
        strategy.register_position(
            symbol="BTC/USDT:USDT",
            direction=ArbitrageDirection.LONG_SPOT_SHORT_PERP,
            spot_size=Decimal("1.5"),
            perp_size=Decimal("1.5"),
            spot_price=Decimal("95000.00"),
            perp_price=Decimal("95050.00"),
            funding_rate=Decimal("0.0003"),
        )
        strategy.on_funding_payment("BTC/USDT:USDT", Decimal("42.75"))

        # Save state
        state = strategy.on_save()

        # Create new strategy and load state
        new_strategy = FundingRateArbitrageStrategy(strategy.arb_config)
        new_strategy.on_load(state)

        # Verify state restored
        assert "BTC/USDT:USDT" in new_strategy.positions
        assert new_strategy.positions["BTC/USDT:USDT"].spot_size == Decimal("1.5")
        assert new_strategy.positions["BTC/USDT:USDT"].cumulative_funding == Decimal("42.75")
        assert new_strategy.total_funding_collected == Decimal("42.75")


# =============================================================================
# Demo Data Tests
# =============================================================================


class TestDemoFundingRates:
    """Tests for demo funding rate data."""

    def test_create_demo_funding_rates(self) -> None:
        """Test demo data creation."""
        rates = create_demo_funding_rates()

        assert len(rates) == 4

        # Check we have different symbols
        symbols = {r.symbol for r in rates}
        assert "BTC/USDT:USDT" in symbols
        assert "ETH/USDT:USDT" in symbols
        assert "SOL/USDT:USDT" in symbols

    def test_demo_rates_have_valid_values(self) -> None:
        """Test that demo rates have valid values."""
        rates = create_demo_funding_rates()

        for rate in rates:
            assert rate.symbol
            assert rate.exchange
            assert rate.mark_price > 0
            assert rate.index_price > 0
            assert rate.next_funding_time_ns > 0

    def test_demo_includes_positive_and_negative(self) -> None:
        """Test that demo includes both positive and negative rates."""
        rates = create_demo_funding_rates()

        has_positive = any(r.is_positive for r in rates)
        has_negative = any(r.is_negative for r in rates)

        assert has_positive
        assert has_negative
