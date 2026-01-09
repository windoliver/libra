"""Tests for TUI Dashboard Widgets (Issue #8)."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from libra.tui.widgets.enhanced_positions import (
    CompactPositionsTable,
    EnhancedPositionsTable,
    PositionData,
    PositionRow,
    create_demo_positions,
)
from libra.tui.widgets.portfolio_dashboard import (
    AllocationBar,
    AssetAllocationTable,
    AssetHolding,
    DailyPnLCard,
    PeriodReturns,
    PortfolioDashboard,
    PortfolioData,
    TotalValueCard,
)
from libra.tui.screens.position_detail import (
    ClosePositionModal,
    MetricCard,
    PositionActionResult,
    PositionDetailModal,
)
from libra.tui.screens.backtest_modal import (
    BacktestConfig,
    BacktestResult,
    BacktestConfigModal,
    BacktestResultsModal,
    run_demo_backtest,
)
from libra.tui.widgets.enhanced_positions import SortablePositionsTable


# =============================================================================
# PositionData Tests
# =============================================================================


class TestPositionData:
    """Tests for PositionData dataclass."""

    def test_create_position(self):
        """Create a basic position."""
        pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("42500"),
            current_price=Decimal("43200"),
            unrealized_pnl=Decimal("350"),
        )

        assert pos.position_id == "pos-1"
        assert pos.symbol == "BTC/USDT"
        assert pos.side == "LONG"
        assert pos.size == Decimal("0.5")

    def test_pnl_pct_long_profit(self):
        """Calculate P&L percentage for profitable long."""
        pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("1"),
            entry_price=Decimal("100"),
            current_price=Decimal("110"),
            unrealized_pnl=Decimal("10"),
        )

        assert pos.pnl_pct == pytest.approx(10.0, rel=0.01)

    def test_pnl_pct_zero_entry(self):
        """Handle zero entry price."""
        pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("1"),
            entry_price=Decimal("0"),
            current_price=Decimal("100"),
            unrealized_pnl=Decimal("0"),
        )

        assert pos.pnl_pct == 0.0

    def test_notional_value(self):
        """Calculate notional value."""
        pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("42500"),
            current_price=Decimal("43200"),
            unrealized_pnl=Decimal("350"),
        )

        assert pos.notional_value == Decimal("21600")  # 0.5 * 43200

    def test_duration(self):
        """Calculate position duration."""
        now = datetime.now()
        pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("42500"),
            current_price=Decimal("43200"),
            unrealized_pnl=Decimal("350"),
            opened_at=now - timedelta(hours=2, minutes=15),
        )

        duration = pos.duration
        assert duration.total_seconds() >= 2 * 3600  # At least 2 hours

    def test_duration_str_hours_minutes(self):
        """Format duration as hours and minutes."""
        now = datetime.now()
        pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("42500"),
            current_price=Decimal("43200"),
            unrealized_pnl=Decimal("350"),
            opened_at=now - timedelta(hours=2, minutes=15),
        )

        # Should be around "2h 15m"
        duration_str = pos.duration_str
        assert "h" in duration_str
        assert "m" in duration_str

    def test_duration_str_days(self):
        """Format duration with days."""
        now = datetime.now()
        pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("42500"),
            current_price=Decimal("43200"),
            unrealized_pnl=Decimal("350"),
            opened_at=now - timedelta(days=2, hours=5),
        )

        duration_str = pos.duration_str
        assert "d" in duration_str


# =============================================================================
# PositionRow Tests
# =============================================================================


class TestPositionRow:
    """Tests for PositionRow widget."""

    def test_create_position_row(self):
        """Create a position row widget."""
        pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("42500"),
            current_price=Decimal("43200"),
            unrealized_pnl=Decimal("350"),
        )
        row = PositionRow(pos)

        assert row.position == pos
        assert row.collapsed is True

    def test_create_title_long_profit(self):
        """Title shows correct colors for long profit."""
        pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("42500"),
            current_price=Decimal("43200"),
            unrealized_pnl=Decimal("350"),
        )
        row = PositionRow(pos)
        title = row._create_title(pos)

        assert "BTC/USDT" in title
        assert "LONG" in title
        assert "+$350" in title
        assert "green" in title  # Profit color

    def test_create_title_short_loss(self):
        """Title shows correct colors for short loss."""
        pos = PositionData(
            position_id="pos-1",
            symbol="ETH/USDT",
            side="SHORT",
            size=Decimal("2.0"),
            entry_price=Decimal("2800"),
            current_price=Decimal("2850"),
            unrealized_pnl=Decimal("-100"),
        )
        row = PositionRow(pos)
        title = row._create_title(pos)

        assert "ETH/USDT" in title
        assert "SHORT" in title
        assert "red" in title  # Loss color

    def test_has_css(self):
        """Widget has CSS defined."""
        assert PositionRow.DEFAULT_CSS is not None
        assert "position-header" in PositionRow.DEFAULT_CSS

    def test_update_position(self):
        """Update position data."""
        pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("42500"),
            current_price=Decimal("43200"),
            unrealized_pnl=Decimal("350"),
        )
        row = PositionRow(pos)

        # Update with new data
        new_pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("42500"),
            current_price=Decimal("44000"),
            unrealized_pnl=Decimal("750"),
        )
        row.update_position(new_pos)

        assert row.position == new_pos


# =============================================================================
# EnhancedPositionsTable Tests
# =============================================================================


class TestEnhancedPositionsTable:
    """Tests for EnhancedPositionsTable widget."""

    def test_create_empty(self):
        """Create empty positions table."""
        table = EnhancedPositionsTable()

        assert table.positions == []

    def test_create_with_positions(self):
        """Create table with positions."""
        positions = create_demo_positions()
        table = EnhancedPositionsTable(positions=positions)

        assert len(table.positions) == len(positions)

    def test_calculate_total_pnl(self):
        """Calculate total P&L across positions."""
        positions = [
            PositionData(
                position_id="pos-1",
                symbol="BTC/USDT",
                side="LONG",
                size=Decimal("1"),
                entry_price=Decimal("100"),
                current_price=Decimal("110"),
                unrealized_pnl=Decimal("100"),
            ),
            PositionData(
                position_id="pos-2",
                symbol="ETH/USDT",
                side="LONG",
                size=Decimal("1"),
                entry_price=Decimal("100"),
                current_price=Decimal("95"),
                unrealized_pnl=Decimal("-50"),
            ),
        ]
        table = EnhancedPositionsTable(positions=positions)

        assert table._calculate_total_pnl() == Decimal("50")

    def test_total_class_positive(self):
        """Total class is positive for profit."""
        positions = [
            PositionData(
                position_id="pos-1",
                symbol="BTC/USDT",
                side="LONG",
                size=Decimal("1"),
                entry_price=Decimal("100"),
                current_price=Decimal("110"),
                unrealized_pnl=Decimal("100"),
            ),
        ]
        table = EnhancedPositionsTable(positions=positions)

        assert table._total_class() == "positive"

    def test_total_class_negative(self):
        """Total class is negative for loss."""
        positions = [
            PositionData(
                position_id="pos-1",
                symbol="BTC/USDT",
                side="LONG",
                size=Decimal("1"),
                entry_price=Decimal("100"),
                current_price=Decimal("90"),
                unrealized_pnl=Decimal("-100"),
            ),
        ]
        table = EnhancedPositionsTable(positions=positions)

        assert table._total_class() == "negative"

    def test_has_css(self):
        """Widget has CSS defined."""
        assert EnhancedPositionsTable.DEFAULT_CSS is not None
        assert "positions-container" in EnhancedPositionsTable.DEFAULT_CSS


# =============================================================================
# CompactPositionsTable Tests
# =============================================================================


class TestCompactPositionsTable:
    """Tests for CompactPositionsTable widget."""

    def test_create_empty(self):
        """Create empty compact table."""
        table = CompactPositionsTable()

        assert table._positions == []

    def test_create_with_positions(self):
        """Create table with positions."""
        positions = create_demo_positions()
        table = CompactPositionsTable(positions=positions)

        assert len(table._positions) == len(positions)

    def test_has_css(self):
        """Widget has CSS defined."""
        assert CompactPositionsTable.DEFAULT_CSS is not None
        assert "table-title" in CompactPositionsTable.DEFAULT_CSS


# =============================================================================
# Demo Positions Tests
# =============================================================================


class TestDemoPositions:
    """Tests for demo positions generator."""

    def test_create_demo_positions(self):
        """Create demo positions returns list."""
        positions = create_demo_positions()

        assert isinstance(positions, list)
        assert len(positions) > 0

    def test_demo_positions_have_data(self):
        """Demo positions have required data."""
        positions = create_demo_positions()

        for pos in positions:
            assert pos.position_id
            assert pos.symbol
            assert pos.side in ("LONG", "SHORT")
            assert pos.size > 0


# =============================================================================
# AssetHolding Tests
# =============================================================================


class TestAssetHolding:
    """Tests for AssetHolding dataclass."""

    def test_create_holding(self):
        """Create basic asset holding."""
        holding = AssetHolding(
            symbol="BTC",
            amount=Decimal("0.5"),
            value_usd=Decimal("21600"),
            pct_of_portfolio=35.5,
            color="yellow",
        )

        assert holding.symbol == "BTC"
        assert holding.value_usd == Decimal("21600")
        assert holding.pct_of_portfolio == 35.5


# =============================================================================
# PortfolioData Tests
# =============================================================================


class TestPortfolioData:
    """Tests for PortfolioData dataclass."""

    def test_create_portfolio(self):
        """Create portfolio data."""
        portfolio = PortfolioData(
            total_value=Decimal("100000"),
            daily_pnl=Decimal("1250"),
            daily_pnl_pct=1.25,
        )

        assert portfolio.total_value == Decimal("100000")
        assert portfolio.daily_pnl == Decimal("1250")

    def test_default_values(self):
        """Test default values."""
        portfolio = PortfolioData()

        assert portfolio.total_value == Decimal("0")
        assert portfolio.holdings == []


# =============================================================================
# TotalValueCard Tests
# =============================================================================


class TestTotalValueCard:
    """Tests for TotalValueCard widget."""

    def test_create_default(self):
        """Create card with default value."""
        card = TotalValueCard()

        assert card.value == Decimal("0")

    def test_create_with_value(self):
        """Create card with custom value."""
        card = TotalValueCard(value=Decimal("100000"))

        assert card.value == Decimal("100000")

    def test_has_css(self):
        """Widget has CSS defined."""
        assert TotalValueCard.DEFAULT_CSS is not None


# =============================================================================
# DailyPnLCard Tests
# =============================================================================


class TestDailyPnLCard:
    """Tests for DailyPnLCard widget."""

    def test_create_default(self):
        """Create card with default values."""
        card = DailyPnLCard()

        assert card.pnl == Decimal("0")
        assert card.pnl_pct == 0.0

    def test_create_with_values(self):
        """Create card with custom values."""
        card = DailyPnLCard(pnl=Decimal("1250"), pnl_pct=1.25)

        assert card.pnl == Decimal("1250")
        assert card.pnl_pct == 1.25

    def test_has_css(self):
        """Widget has CSS defined."""
        assert DailyPnLCard.DEFAULT_CSS is not None


# =============================================================================
# AllocationBar Tests
# =============================================================================


class TestAllocationBar:
    """Tests for AllocationBar widget."""

    def test_create_empty(self):
        """Create with no allocations."""
        bar = AllocationBar()

        assert bar.allocations == []

    def test_create_with_allocations(self):
        """Create with allocations."""
        allocations = [
            ("BTC", 35.5),
            ("ETH", 24.6),
        ]
        bar = AllocationBar(allocations=allocations)

        assert len(bar.allocations) == 2

    def test_has_css(self):
        """Widget has CSS defined."""
        assert AllocationBar.DEFAULT_CSS is not None


# =============================================================================
# PeriodReturns Tests
# =============================================================================


class TestPeriodReturns:
    """Tests for PeriodReturns widget."""

    def test_create_default(self):
        """Create with default values."""
        returns = PeriodReturns()

        assert returns.daily == Decimal("0")
        assert returns.weekly == Decimal("0")
        assert returns.monthly == Decimal("0")
        assert returns.ytd == Decimal("0")

    def test_update_returns(self):
        """Update returns values."""
        returns = PeriodReturns()
        returns.update_returns(
            daily=Decimal("1250"),
            weekly=Decimal("3450"),
            monthly=Decimal("8200"),
            ytd=Decimal("25430"),
        )

        assert returns.daily == Decimal("1250")
        assert returns.ytd == Decimal("25430")

    def test_has_css(self):
        """Widget has CSS defined."""
        assert PeriodReturns.DEFAULT_CSS is not None


# =============================================================================
# PortfolioDashboard Tests
# =============================================================================


class TestPortfolioDashboard:
    """Tests for PortfolioDashboard widget."""

    def test_create_default(self):
        """Create with default data."""
        dashboard = PortfolioDashboard()

        assert dashboard._data is not None

    def test_create_with_data(self):
        """Create with custom data."""
        data = PortfolioData(
            total_value=Decimal("100000"),
            daily_pnl=Decimal("1250"),
            daily_pnl_pct=1.25,
        )
        dashboard = PortfolioDashboard(data=data)

        assert dashboard._data.total_value == Decimal("100000")

    def test_has_css(self):
        """Widget has CSS defined."""
        assert PortfolioDashboard.DEFAULT_CSS is not None


# =============================================================================
# PositionActionResult Tests
# =============================================================================


class TestPositionActionResult:
    """Tests for PositionActionResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = PositionActionResult()

        assert result.action == "none"
        assert result.position_id == ""

    def test_close_action(self):
        """Test close action."""
        result = PositionActionResult(
            action="close",
            position_id="pos-1",
        )

        assert result.action == "close"
        assert result.position_id == "pos-1"


# =============================================================================
# MetricCard Tests
# =============================================================================


class TestMetricCard:
    """Tests for MetricCard widget."""

    def test_create_basic(self):
        """Create basic metric card."""
        card = MetricCard(label="Test", value="$100")

        assert card._label == "Test"
        assert card._value == "$100"

    def test_create_with_class(self):
        """Create metric card with value class."""
        card = MetricCard(label="P&L", value="+$100", value_class="positive")

        assert card._value_class == "positive"

    def test_has_css(self):
        """Widget has CSS defined."""
        assert MetricCard.DEFAULT_CSS is not None


# =============================================================================
# PositionDetailModal Tests
# =============================================================================


class TestPositionDetailModal:
    """Tests for PositionDetailModal screen."""

    def test_create_modal(self):
        """Create position detail modal."""
        pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("42500"),
            current_price=Decimal("43200"),
            unrealized_pnl=Decimal("350"),
        )
        modal = PositionDetailModal(pos)

        assert modal._position == pos

    def test_liquidation_distance_no_liq(self):
        """Calculate liquidation distance with no liquidation price."""
        pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("42500"),
            current_price=Decimal("43200"),
            unrealized_pnl=Decimal("350"),
        )
        modal = PositionDetailModal(pos)

        distance = modal._liquidation_distance(pos)
        assert distance == 100.0

    def test_liquidation_distance_with_liq(self):
        """Calculate liquidation distance with liquidation price."""
        pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("42500"),
            current_price=Decimal("43200"),
            unrealized_pnl=Decimal("350"),
            liquidation_price=Decimal("38250"),
        )
        modal = PositionDetailModal(pos)

        distance = modal._liquidation_distance(pos)
        assert distance > 0
        assert distance < 100

    def test_has_css(self):
        """Widget has CSS defined."""
        assert PositionDetailModal.DEFAULT_CSS is not None


# =============================================================================
# ClosePositionModal Tests
# =============================================================================


class TestClosePositionModal:
    """Tests for ClosePositionModal screen."""

    def test_create_modal(self):
        """Create close position modal."""
        pos = PositionData(
            position_id="pos-1",
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.5"),
            entry_price=Decimal("42500"),
            current_price=Decimal("43200"),
            unrealized_pnl=Decimal("350"),
        )
        modal = ClosePositionModal(pos)

        assert modal._position == pos

    def test_has_css(self):
        """Widget has CSS defined."""
        assert ClosePositionModal.DEFAULT_CSS is not None


# =============================================================================
# SortablePositionsTable Tests
# =============================================================================


class TestSortablePositionsTable:
    """Tests for SortablePositionsTable widget."""

    def test_create_empty(self):
        """Create empty sortable table."""
        table = SortablePositionsTable()

        assert table._positions == []

    def test_create_with_positions(self):
        """Create table with positions."""
        positions = create_demo_positions()
        table = SortablePositionsTable(positions=positions)

        assert len(table._positions) == len(positions)

    def test_default_sort(self):
        """Default sort is by P&L descending."""
        table = SortablePositionsTable()

        assert table._sort_column == "pnl"
        assert table._sort_reverse is True

    def test_sort_by(self):
        """Sort by specific column."""
        positions = create_demo_positions()
        table = SortablePositionsTable(positions=positions)

        table.sort_by("symbol", reverse=False)

        assert table._sort_column == "symbol"
        assert table._sort_reverse is False

    def test_has_css(self):
        """Widget has CSS defined."""
        assert SortablePositionsTable.DEFAULT_CSS is not None


# =============================================================================
# BacktestConfig Tests
# =============================================================================


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_create_config(self):
        """Create basic backtest config."""
        config = BacktestConfig(
            strategy_id="test-1",
            strategy_name="Test Strategy",
        )

        assert config.strategy_id == "test-1"
        assert config.strategy_name == "Test Strategy"
        assert config.initial_capital == Decimal("10000")

    def test_default_dates(self):
        """Default dates are set."""
        config = BacktestConfig(
            strategy_id="test-1",
            strategy_name="Test Strategy",
        )

        assert config.start_date  # Not empty
        assert config.end_date  # Not empty

    def test_custom_values(self):
        """Custom values are preserved."""
        config = BacktestConfig(
            strategy_id="test-1",
            strategy_name="Test Strategy",
            symbol="ETH/USDT",
            initial_capital=Decimal("50000"),
            commission_pct=0.05,
        )

        assert config.symbol == "ETH/USDT"
        assert config.initial_capital == Decimal("50000")
        assert config.commission_pct == 0.05


# =============================================================================
# BacktestResult Tests
# =============================================================================


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = BacktestResult()

        assert result.success is False
        assert result.total_trades == 0
        assert result.equity_curve == []

    def test_with_values(self):
        """Create result with values."""
        result = BacktestResult(
            success=True,
            total_return=Decimal("1500"),
            total_return_pct=15.0,
            sharpe_ratio=1.5,
            max_drawdown_pct=10.0,
            win_rate=60.0,
            total_trades=50,
        )

        assert result.success is True
        assert result.total_return == Decimal("1500")
        assert result.sharpe_ratio == 1.5


# =============================================================================
# Demo Backtest Tests
# =============================================================================


class TestDemoBacktest:
    """Tests for demo backtest runner."""

    def test_run_demo_backtest(self):
        """Run demo backtest returns result."""
        config = BacktestConfig(
            strategy_id="test-1",
            strategy_name="Test Strategy",
        )

        result = run_demo_backtest(config)

        assert result.success is True
        assert result.total_trades > 0
        assert len(result.equity_curve) > 0
        assert result.config == config

    def test_backtest_has_metrics(self):
        """Demo backtest generates all metrics."""
        config = BacktestConfig(
            strategy_id="test-1",
            strategy_name="Test Strategy",
        )

        result = run_demo_backtest(config)

        assert result.sharpe_ratio != 0
        assert result.max_drawdown_pct != 0
        assert result.win_rate != 0


# =============================================================================
# BacktestConfigModal Tests
# =============================================================================


class TestBacktestConfigModal:
    """Tests for BacktestConfigModal screen."""

    def test_create_modal(self):
        """Create backtest config modal."""
        modal = BacktestConfigModal(
            strategy_id="test-1",
            strategy_name="Test Strategy",
        )

        assert modal._strategy_id == "test-1"
        assert modal._strategy_name == "Test Strategy"

    def test_has_css(self):
        """Widget has CSS defined."""
        assert BacktestConfigModal.DEFAULT_CSS is not None


# =============================================================================
# BacktestResultsModal Tests
# =============================================================================


class TestBacktestResultsModal:
    """Tests for BacktestResultsModal screen."""

    def test_create_modal(self):
        """Create backtest results modal."""
        result = BacktestResult(
            success=True,
            total_return=Decimal("1500"),
        )
        modal = BacktestResultsModal(result)

        assert modal._result == result

    def test_has_css(self):
        """Widget has CSS defined."""
        assert BacktestResultsModal.DEFAULT_CSS is not None
