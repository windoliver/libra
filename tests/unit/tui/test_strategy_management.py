"""Tests for Strategy Management TUI components."""

from decimal import Decimal

import pytest

from libra.tui.widgets.strategy_card import StrategyCard, STRATEGY_STATUS_CONFIG
from libra.tui.widgets.strategy_metrics import (
    MetricCard,
    StrategyMetricsPanel,
)
from libra.tui.widgets.strategy_tree import (
    PositionInfo,
    StrategyInfo,
    StrategyTree,
    StrategyListView,
)
from libra.tui.widgets.parameter_editor import (
    ParameterDefinition,
    ParameterEditor,
    ParameterRow,
    ParameterType,
    IntegerValidator,
    FloatValidator,
    create_sma_cross_parameters,
)
from libra.tui.screens.strategy_edit_modal import (
    StrategyEditModal,
    StrategyCreateModal,
    ConfirmationModal,
    StrategyEditResult,
)


# =============================================================================
# StrategyCard Tests
# =============================================================================


class TestStrategyCard:
    """Tests for StrategyCard widget."""

    def test_create_default(self):
        """Create card with required parameters."""
        card = StrategyCard(
            strategy_id="test_strategy",
            name="Test Strategy",
        )

        assert card.strategy_id == "test_strategy"
        assert card.strategy_name == "Test Strategy"
        assert card.status == "STOPPED"
        assert card.pnl == Decimal("0")

    def test_create_with_all_params(self):
        """Create card with all parameters."""
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        card = StrategyCard(
            strategy_id="my_strategy",
            name="My Strategy",
            status="RUNNING",
            pnl=Decimal("1250.50"),
            pnl_history=history,
        )

        assert card.strategy_id == "my_strategy"
        assert card.strategy_name == "My Strategy"
        assert card.status == "RUNNING"
        assert card.pnl == Decimal("1250.50")
        assert card.pnl_history == history

    def test_format_positive_pnl(self):
        """Format positive P&L."""
        card = StrategyCard(
            strategy_id="test",
            name="Test",
            pnl=Decimal("1234.56"),
        )

        formatted = card._format_pnl()

        assert formatted == "+$1,234.56"

    def test_format_negative_pnl(self):
        """Format negative P&L."""
        card = StrategyCard(
            strategy_id="test",
            name="Test",
            pnl=Decimal("-567.89"),
        )

        formatted = card._format_pnl()

        assert formatted == "-$567.89"

    def test_format_zero_pnl(self):
        """Format zero P&L."""
        card = StrategyCard(
            strategy_id="test",
            name="Test",
            pnl=Decimal("0"),
        )

        formatted = card._format_pnl()

        assert formatted == "$0.00"

    def test_pnl_class_positive(self):
        """P&L class for positive value."""
        card = StrategyCard(strategy_id="test", name="Test", pnl=Decimal("100"))

        assert card._pnl_class() == "positive"

    def test_pnl_class_negative(self):
        """P&L class for negative value."""
        card = StrategyCard(strategy_id="test", name="Test", pnl=Decimal("-100"))

        assert card._pnl_class() == "negative"

    def test_pnl_class_neutral(self):
        """P&L class for zero value."""
        card = StrategyCard(strategy_id="test", name="Test", pnl=Decimal("0"))

        assert card._pnl_class() == "neutral"

    def test_status_icon_running(self):
        """Status icon for RUNNING."""
        card = StrategyCard(strategy_id="test", name="Test", status="RUNNING")

        icon = card._get_status_icon()

        assert "green" in icon
        assert "●" in icon

    def test_status_icon_stopped(self):
        """Status icon for STOPPED."""
        card = StrategyCard(strategy_id="test", name="Test", status="STOPPED")

        icon = card._get_status_icon()

        assert "dim" in icon
        assert "○" in icon

    def test_status_icon_paused(self):
        """Status icon for PAUSED."""
        card = StrategyCard(strategy_id="test", name="Test", status="PAUSED")

        icon = card._get_status_icon()

        assert "yellow" in icon

    def test_status_icon_error(self):
        """Status icon for ERROR."""
        card = StrategyCard(strategy_id="test", name="Test", status="ERROR")

        icon = card._get_status_icon()

        assert "red" in icon
        assert "✗" in icon

    def test_update_data(self):
        """Update multiple fields at once."""
        card = StrategyCard(strategy_id="test", name="Test")

        card.update_data(
            status="RUNNING",
            pnl=Decimal("500"),
            pnl_history=[1.0, 2.0, 3.0],
        )

        assert card.status == "RUNNING"
        assert card.pnl == Decimal("500")
        assert card.pnl_history == [1.0, 2.0, 3.0]

    def test_has_css(self):
        """Widget has CSS defined."""
        assert StrategyCard.DEFAULT_CSS is not None
        assert "pnl-value" in StrategyCard.DEFAULT_CSS
        assert "status-icon" in StrategyCard.DEFAULT_CSS


class TestStrategyStatusConfig:
    """Tests for strategy status configuration."""

    def test_all_statuses_defined(self):
        """All expected statuses are defined."""
        expected_statuses = ["RUNNING", "STOPPED", "PAUSED", "STARTING", "STOPPING", "ERROR", "DEGRADED"]

        for status in expected_statuses:
            assert status in STRATEGY_STATUS_CONFIG

    def test_status_has_icon_and_label(self):
        """Each status has icon and label."""
        for status, config in STRATEGY_STATUS_CONFIG.items():
            assert "icon" in config, f"{status} missing icon"
            assert "label" in config, f"{status} missing label"


# =============================================================================
# StrategyMetricsPanel Tests
# =============================================================================


class TestMetricCard:
    """Tests for MetricCard widget."""

    def test_create_default(self):
        """Create card with label and value."""
        card = MetricCard("Sharpe Ratio", "2.15")

        assert card.value == "2.15"

    def test_threshold_good(self):
        """Value above threshold is good."""
        card = MetricCard(
            "Sharpe",
            "2.5",
            numeric_value=2.5,
            threshold_good=2.0,
        )

        assert card._get_value_class() == "good"

    def test_threshold_bad(self):
        """Value below threshold is bad."""
        card = MetricCard(
            "Sharpe",
            "0.3",
            numeric_value=0.3,
            threshold_bad=0.5,
        )

        assert card._get_value_class() == "bad"

    def test_threshold_warning(self):
        """Value between thresholds is warning."""
        card = MetricCard(
            "Sharpe",
            "1.0",
            numeric_value=1.0,
            threshold_good=2.0,
            threshold_bad=0.5,
        )

        assert card._get_value_class() == "warning"

    def test_inverted_threshold(self):
        """Inverted threshold (lower is better)."""
        # For drawdown, lower is better
        card = MetricCard(
            "Drawdown",
            "5%",
            numeric_value=5.0,
            threshold_good=10.0,
            threshold_bad=30.0,
            invert_threshold=True,
        )

        assert card._get_value_class() == "good"

    def test_inverted_threshold_bad(self):
        """Inverted threshold bad case."""
        card = MetricCard(
            "Drawdown",
            "35%",
            numeric_value=35.0,
            threshold_good=10.0,
            threshold_bad=30.0,
            invert_threshold=True,
        )

        assert card._get_value_class() == "bad"

    def test_update_metric(self):
        """Update both display and numeric value."""
        card = MetricCard("Test", "0", numeric_value=0.0)

        card.update_metric("2.5", 2.5)

        assert card.value == "2.5"
        assert card.numeric_value == 2.5


class TestStrategyMetricsPanel:
    """Tests for StrategyMetricsPanel widget."""

    def test_create_default(self):
        """Create panel with default values."""
        panel = StrategyMetricsPanel()

        assert panel.total_pnl == Decimal("0")
        assert panel.sharpe_ratio == 0.0
        assert panel.win_rate == 0.0

    def test_create_with_values(self):
        """Create panel with custom values."""
        panel = StrategyMetricsPanel(
            total_pnl=Decimal("1250"),
            sharpe_ratio=2.15,
            max_drawdown=12.5,
            win_rate=65.0,
            profit_factor=1.85,
            total_trades=42,
        )

        assert panel.total_pnl == Decimal("1250")
        assert panel.sharpe_ratio == 2.15
        assert panel.max_drawdown == 12.5
        assert panel.win_rate == 65.0
        assert panel.profit_factor == 1.85
        assert panel.total_trades == 42

    def test_format_large_pnl_positive(self):
        """Format positive P&L."""
        panel = StrategyMetricsPanel(total_pnl=Decimal("1234.56"))

        formatted = panel._format_large_pnl()

        assert formatted == "+$1,234.56"

    def test_format_large_pnl_negative(self):
        """Format negative P&L."""
        panel = StrategyMetricsPanel(total_pnl=Decimal("-567.89"))

        formatted = panel._format_large_pnl()

        assert formatted == "-$567.89"

    def test_pnl_class(self):
        """P&L class based on value."""
        panel_positive = StrategyMetricsPanel(total_pnl=Decimal("100"))
        panel_negative = StrategyMetricsPanel(total_pnl=Decimal("-100"))
        panel_zero = StrategyMetricsPanel(total_pnl=Decimal("0"))

        assert panel_positive._pnl_class() == "positive"
        assert panel_negative._pnl_class() == "negative"
        assert panel_zero._pnl_class() == ""

    def test_update_all_metrics(self):
        """Batch update all metrics."""
        panel = StrategyMetricsPanel()

        panel.update_all_metrics(
            total_pnl=Decimal("500"),
            sharpe_ratio=1.5,
            max_drawdown=15.0,
            win_rate=55.0,
            profit_factor=1.2,
            total_trades=20,
        )

        assert panel.total_pnl == Decimal("500")
        assert panel.sharpe_ratio == 1.5
        assert panel.max_drawdown == 15.0
        assert panel.win_rate == 55.0
        assert panel.profit_factor == 1.2
        assert panel.total_trades == 20


# =============================================================================
# StrategyTree Tests
# =============================================================================


class TestPositionInfo:
    """Tests for PositionInfo dataclass."""

    def test_create_long_position(self):
        """Create LONG position."""
        pos = PositionInfo(
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.1"),
            entry_price=Decimal("42000"),
            current_price=Decimal("43000"),
            unrealized_pnl=Decimal("100"),
        )

        assert pos.symbol == "BTC/USDT"
        assert pos.side == "LONG"
        assert pos.size == Decimal("0.1")

    def test_display_text_long(self):
        """Display text for LONG position."""
        pos = PositionInfo(
            symbol="BTC/USDT",
            side="LONG",
            size=Decimal("0.1"),
            entry_price=Decimal("42000"),
            current_price=Decimal("43000"),
            unrealized_pnl=Decimal("100"),
        )

        text = pos.display_text

        assert "LONG" in text
        assert "0.1" in text
        assert "42,000" in text
        assert "100" in text

    def test_display_text_flat(self):
        """Display text for FLAT position."""
        pos = PositionInfo(
            symbol="BTC/USDT",
            side="FLAT",
            size=Decimal("0"),
        )

        text = pos.display_text

        assert "no position" in text


class TestStrategyInfo:
    """Tests for StrategyInfo dataclass."""

    def test_create_strategy(self):
        """Create strategy info."""
        strategy = StrategyInfo(
            strategy_id="test_strat",
            name="Test Strategy",
            status="RUNNING",
            symbols=["BTC/USDT"],
            total_pnl=Decimal("500"),
        )

        assert strategy.strategy_id == "test_strat"
        assert strategy.name == "Test Strategy"
        assert strategy.status == "RUNNING"

    def test_status_icon_running(self):
        """Status icon for running strategy."""
        strategy = StrategyInfo(
            strategy_id="test",
            name="Test",
            status="RUNNING",
        )

        icon = strategy.status_icon

        assert "green" in icon

    def test_status_icon_stopped(self):
        """Status icon for stopped strategy."""
        strategy = StrategyInfo(
            strategy_id="test",
            name="Test",
            status="STOPPED",
        )

        icon = strategy.status_icon

        assert "dim" in icon

    def test_pnl_display_positive(self):
        """P&L display for positive value."""
        strategy = StrategyInfo(
            strategy_id="test",
            name="Test",
            total_pnl=Decimal("1234.56"),
        )

        display = strategy.pnl_display

        assert "green" in display
        assert "+$1,234.56" in display

    def test_pnl_display_negative(self):
        """P&L display for negative value."""
        strategy = StrategyInfo(
            strategy_id="test",
            name="Test",
            total_pnl=Decimal("-567.89"),
        )

        display = strategy.pnl_display

        assert "red" in display


class TestStrategyTree:
    """Tests for StrategyTree widget."""

    def test_create_empty(self):
        """Create empty tree."""
        tree = StrategyTree()

        assert len(tree._strategies) == 0

    def test_create_with_strategies(self):
        """Create tree with strategies."""
        strategies = [
            StrategyInfo(strategy_id="s1", name="Strategy 1"),
            StrategyInfo(strategy_id="s2", name="Strategy 2"),
        ]

        tree = StrategyTree(strategies=strategies)

        assert len(tree._strategies) == 2
        assert "s1" in tree._strategies
        assert "s2" in tree._strategies

    def test_add_strategy(self):
        """Add strategy to tree."""
        tree = StrategyTree()
        strategy = StrategyInfo(strategy_id="new", name="New Strategy")

        tree._strategies[strategy.strategy_id] = strategy

        assert "new" in tree._strategies


# =============================================================================
# ParameterEditor Tests
# =============================================================================


class TestParameterDefinition:
    """Tests for ParameterDefinition dataclass."""

    def test_create_integer_param(self):
        """Create integer parameter."""
        param = ParameterDefinition(
            name="fast_period",
            display_name="Fast Period",
            param_type=ParameterType.INTEGER,
            default=10,
            min_value=1,
            max_value=100,
        )

        assert param.name == "fast_period"
        assert param.param_type == ParameterType.INTEGER
        assert param.default == 10
        assert param.current == 10  # Should be set from default

    def test_create_float_param(self):
        """Create float parameter."""
        param = ParameterDefinition(
            name="threshold",
            display_name="Threshold",
            param_type=ParameterType.FLOAT,
            default=0.02,
            min_value=0.001,
            max_value=0.1,
        )

        assert param.param_type == ParameterType.FLOAT
        assert param.default == 0.02

    def test_create_select_param(self):
        """Create select parameter."""
        param = ParameterDefinition(
            name="timeframe",
            display_name="Timeframe",
            param_type=ParameterType.SELECT,
            default="1h",
            options=[("1 Hour", "1h"), ("4 Hours", "4h"), ("1 Day", "1d")],
        )

        assert param.param_type == ParameterType.SELECT
        assert len(param.options) == 3

    def test_current_defaults_to_default(self):
        """Current value defaults to default."""
        param = ParameterDefinition(
            name="test",
            display_name="Test",
            param_type=ParameterType.INTEGER,
            default=42,
        )

        assert param.current == 42


class TestIntegerValidator:
    """Tests for IntegerValidator."""

    def test_valid_integer(self):
        """Valid integer passes."""
        validator = IntegerValidator()

        result = validator.validate("42")

        assert result.is_valid

    def test_invalid_not_integer(self):
        """Non-integer fails."""
        validator = IntegerValidator()

        result = validator.validate("abc")

        assert not result.is_valid

    def test_invalid_float(self):
        """Float fails integer validation."""
        validator = IntegerValidator()

        result = validator.validate("3.14")

        assert not result.is_valid

    def test_min_value(self):
        """Value below minimum fails."""
        validator = IntegerValidator(min_value=10)

        result = validator.validate("5")

        assert not result.is_valid

    def test_max_value(self):
        """Value above maximum fails."""
        validator = IntegerValidator(max_value=100)

        result = validator.validate("150")

        assert not result.is_valid

    def test_within_range(self):
        """Value within range passes."""
        validator = IntegerValidator(min_value=1, max_value=100)

        result = validator.validate("50")

        assert result.is_valid


class TestFloatValidator:
    """Tests for FloatValidator."""

    def test_valid_float(self):
        """Valid float passes."""
        validator = FloatValidator()

        result = validator.validate("3.14")

        assert result.is_valid

    def test_valid_integer_as_float(self):
        """Integer passes float validation."""
        validator = FloatValidator()

        result = validator.validate("42")

        assert result.is_valid

    def test_invalid_not_number(self):
        """Non-number fails."""
        validator = FloatValidator()

        result = validator.validate("abc")

        assert not result.is_valid

    def test_min_value(self):
        """Value below minimum fails."""
        validator = FloatValidator(min_value=0.01)

        result = validator.validate("0.001")

        assert not result.is_valid

    def test_max_value(self):
        """Value above maximum fails."""
        validator = FloatValidator(max_value=1.0)

        result = validator.validate("1.5")

        assert not result.is_valid


class TestCreateSMAParameters:
    """Tests for create_sma_cross_parameters helper."""

    def test_default_parameters(self):
        """Create default SMA parameters."""
        params = create_sma_cross_parameters()

        assert len(params) == 5

        # Check parameter names
        names = [p.name for p in params]
        assert "symbol" in names
        assert "timeframe" in names
        assert "fast_period" in names
        assert "slow_period" in names
        assert "threshold" in names

    def test_custom_parameters(self):
        """Create SMA parameters with custom values."""
        params = create_sma_cross_parameters(
            fast_period=20,
            slow_period=50,
            threshold=0.05,
            timeframe="4h",
            symbol="ETH/USDT",
        )

        # Find each parameter
        fast = next(p for p in params if p.name == "fast_period")
        slow = next(p for p in params if p.name == "slow_period")
        threshold = next(p for p in params if p.name == "threshold")
        timeframe = next(p for p in params if p.name == "timeframe")
        symbol = next(p for p in params if p.name == "symbol")

        assert fast.default == 20
        assert slow.default == 50
        assert threshold.default == 0.05
        assert timeframe.default == "4h"
        assert symbol.default == "ETH/USDT"


# =============================================================================
# StrategyEditModal Tests
# =============================================================================


class TestStrategyEditResult:
    """Tests for StrategyEditResult dataclass."""

    def test_create_saved_result(self):
        """Create saved result."""
        result = StrategyEditResult(
            saved=True,
            strategy_id="test",
            strategy_name="Test Strategy",
            parameters={"fast_period": 10},
        )

        assert result.saved is True
        assert result.strategy_id == "test"
        assert result.parameters == {"fast_period": 10}

    def test_create_cancelled_result(self):
        """Create cancelled result."""
        result = StrategyEditResult(saved=False)

        assert result.saved is False
        assert result.parameters is None


class TestStrategyEditModal:
    """Tests for StrategyEditModal screen."""

    def test_create_modal(self):
        """Create edit modal."""
        modal = StrategyEditModal(
            strategy_id="test_strategy",
            strategy_name="Test Strategy",
        )

        assert modal._strategy_id == "test_strategy"
        assert modal._strategy_name == "Test Strategy"

    def test_create_with_custom_parameters(self):
        """Create modal with custom parameters."""
        params = [
            ParameterDefinition(
                name="custom",
                display_name="Custom",
                param_type=ParameterType.STRING,
                default="value",
            )
        ]

        modal = StrategyEditModal(
            strategy_id="test",
            strategy_name="Test",
            parameters=params,
        )

        assert len(modal._parameters) == 1


class TestConfirmationModal:
    """Tests for ConfirmationModal screen."""

    def test_create_modal(self):
        """Create confirmation modal."""
        modal = ConfirmationModal(
            title="Delete",
            message="Are you sure?",
            confirm_label="Yes",
            cancel_label="No",
        )

        assert modal._title == "Delete"
        assert modal._message == "Are you sure?"
        assert modal._confirm_label == "Yes"
        assert modal._cancel_label == "No"

    def test_default_labels(self):
        """Default labels are set."""
        modal = ConfirmationModal()

        assert modal._confirm_label == "Yes"
        assert modal._cancel_label == "No"
