"""
Parameter Editor Widget.

Dynamic form for editing strategy parameters with validation.

Features:
- Type-aware input fields (number, string, select)
- Inline validation with error messages
- Autocomplete suggestions for known values
- Real-time parameter preview
- Undo/reset support

Design inspired by:
- NinjaTrader parameter dialogs
- Jupyter notebook widget forms
- VS Code settings editor

Layout:
    +-- PARAMETERS ------------------------------------+
    | fast_period: [10____] (Integer, 1-100)          |
    | slow_period: [30____] (Integer, 1-500)          |
    | threshold:   [0.02__] (Float, 0.001-0.1)        |
    | timeframe:   [1h    ▾] (Select)                 |
    | symbol:      [BTC/USDT] (String, autocomplete)  |
    +-------------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.suggester import SuggestFromList
from textual.validation import Number, ValidationResult, Validator
from textual.widgets import Input, Label, Select, Static, Switch


if TYPE_CHECKING:
    pass


# =============================================================================
# Parameter Type Definitions
# =============================================================================


class ParameterType(Enum):
    """Supported parameter types."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    SELECT = "select"
    SYMBOL = "symbol"  # Special string with symbol suggestions


@dataclass
class ParameterDefinition:
    """
    Definition of a strategy parameter.

    Attributes:
        name: Parameter identifier
        display_name: Human-readable label
        param_type: Type of the parameter
        default: Default value
        current: Current value (may differ from default)
        min_value: Minimum value (for numeric types)
        max_value: Maximum value (for numeric types)
        options: Available options (for SELECT type)
        suggestions: Autocomplete suggestions (for STRING/SYMBOL types)
        description: Help text
        required: Whether the parameter is required
        readonly: Whether the parameter can be edited
    """

    name: str
    display_name: str
    param_type: ParameterType
    default: Any
    current: Any = None
    min_value: float | None = None
    max_value: float | None = None
    options: list[tuple[str, Any]] | None = None  # (label, value) pairs
    suggestions: list[str] | None = None
    description: str = ""
    required: bool = True
    readonly: bool = False

    def __post_init__(self) -> None:
        if self.current is None:
            self.current = self.default


# =============================================================================
# Custom Validators
# =============================================================================


class IntegerValidator(Validator):
    """Validates integer input with optional range."""

    def __init__(
        self,
        min_value: int | None = None,
        max_value: int | None = None,
        failure_description: str | None = None,
    ) -> None:
        super().__init__(failure_description)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: str) -> ValidationResult:
        """Validate the integer value."""
        if not value:
            return self.failure("Value required")

        try:
            int_val = int(value)
        except ValueError:
            return self.failure("Must be an integer")

        if self.min_value is not None and int_val < self.min_value:
            return self.failure(f"Must be >= {self.min_value}")

        if self.max_value is not None and int_val > self.max_value:
            return self.failure(f"Must be <= {self.max_value}")

        return self.success()


class FloatValidator(Validator):
    """Validates float input with optional range."""

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        failure_description: str | None = None,
    ) -> None:
        super().__init__(failure_description)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: str) -> ValidationResult:
        """Validate the float value."""
        if not value:
            return self.failure("Value required")

        try:
            float_val = float(value)
        except ValueError:
            return self.failure("Must be a number")

        if self.min_value is not None and float_val < self.min_value:
            return self.failure(f"Must be >= {self.min_value}")

        if self.max_value is not None and float_val > self.max_value:
            return self.failure(f"Must be <= {self.max_value}")

        return self.success()


# =============================================================================
# Parameter Row Widget
# =============================================================================


class ParameterRow(Horizontal):
    """
    Single parameter input row with label, input, and validation message.
    """

    DEFAULT_CSS = """
    ParameterRow {
        height: 3;
        padding: 0 1;
        margin-bottom: 1;
    }

    ParameterRow .param-label {
        width: 20;
        height: 1;
        content-align: left middle;
    }

    ParameterRow .param-input-container {
        width: 1fr;
        height: 3;
    }

    ParameterRow Input {
        width: 100%;
    }

    ParameterRow Select {
        width: 100%;
    }

    ParameterRow Switch {
        width: auto;
    }

    ParameterRow .param-error {
        height: 1;
        color: $error;
    }

    ParameterRow .param-hint {
        height: 1;
        color: $text-muted;
    }

    ParameterRow.-invalid .param-label {
        color: $error;
    }

    ParameterRow.-readonly .param-label {
        color: $text-muted;
    }
    """

    class Changed(Message):
        """Message sent when parameter value changes."""

        def __init__(self, name: str, value: Any, valid: bool) -> None:
            self.name = name
            self.value = value
            self.valid = valid
            super().__init__()

    def __init__(self, param: ParameterDefinition, id: str | None = None) -> None:
        super().__init__(id=id or f"param-row-{param.name}")
        self.param = param
        self._is_valid = True

    def compose(self) -> ComposeResult:
        # Label
        label_text = self.param.display_name
        if self.param.required:
            label_text += " *"
        yield Label(label_text, classes="param-label")

        # Input container
        with Vertical(classes="param-input-container"):
            yield self._create_input_widget()
            yield Static("", classes="param-hint", id="hint")

        if self.param.readonly:
            self.add_class("-readonly")

    def _create_input_widget(self) -> Input | Select | Switch:
        """Create the appropriate input widget based on parameter type."""
        param = self.param

        if param.param_type == ParameterType.BOOLEAN:
            return Switch(value=bool(param.current), id="input")

        elif param.param_type == ParameterType.SELECT:
            options = param.options or []
            return Select(
                options=options,
                value=param.current,
                id="input",
                allow_blank=not param.required,
            )

        elif param.param_type == ParameterType.INTEGER:
            validator = IntegerValidator(
                min_value=int(param.min_value) if param.min_value is not None else None,
                max_value=int(param.max_value) if param.max_value is not None else None,
            )
            hint = self._get_range_hint()
            return Input(
                value=str(param.current),
                validators=[validator],
                placeholder=hint,
                id="input",
                disabled=param.readonly,
            )

        elif param.param_type == ParameterType.FLOAT:
            validator = FloatValidator(
                min_value=param.min_value,
                max_value=param.max_value,
            )
            hint = self._get_range_hint()
            return Input(
                value=str(param.current),
                validators=[validator],
                placeholder=hint,
                id="input",
                disabled=param.readonly,
            )

        else:  # STRING or SYMBOL
            suggester = None
            if param.suggestions:
                suggester = SuggestFromList(param.suggestions, case_sensitive=False)
            return Input(
                value=str(param.current) if param.current else "",
                suggester=suggester,
                placeholder=param.description or "",
                id="input",
                disabled=param.readonly,
            )

    def _get_range_hint(self) -> str:
        """Generate hint text for numeric ranges."""
        parts = []
        if self.param.min_value is not None:
            parts.append(f"min: {self.param.min_value}")
        if self.param.max_value is not None:
            parts.append(f"max: {self.param.max_value}")
        return ", ".join(parts) if parts else ""

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        value = event.value
        valid = True

        # Check validation result
        if event.validation_result:
            valid = event.validation_result.is_valid
            self._update_validation_display(event.validation_result)

        # Convert value to appropriate type
        typed_value = self._convert_value(value)

        self._is_valid = valid
        if valid:
            self.remove_class("-invalid")
        else:
            self.add_class("-invalid")

        self.post_message(self.Changed(self.param.name, typed_value, valid))

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        self.post_message(self.Changed(self.param.name, event.value, True))

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes."""
        self.post_message(self.Changed(self.param.name, event.value, True))

    def _update_validation_display(self, result: ValidationResult) -> None:
        """Update the hint/error display."""
        try:
            hint = self.query_one("#hint", Static)
            if result.is_valid:
                hint.update(self._get_range_hint())
                hint.remove_class("param-error")
                hint.add_class("param-hint")
            else:
                failures = result.failure_descriptions
                hint.update(failures[0] if failures else "Invalid")
                hint.remove_class("param-hint")
                hint.add_class("param-error")
        except Exception:
            pass

    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        param = self.param

        if param.param_type == ParameterType.INTEGER:
            try:
                return int(value)
            except ValueError:
                return param.current

        elif param.param_type == ParameterType.FLOAT:
            try:
                return float(value)
            except ValueError:
                return param.current

        return value

    def get_value(self) -> Any:
        """Get the current value."""
        try:
            input_widget = self.query_one("#input")
            if isinstance(input_widget, Switch):
                return input_widget.value
            elif isinstance(input_widget, Select):
                return input_widget.value
            else:
                return self._convert_value(input_widget.value)
        except Exception:
            return self.param.current

    def is_valid(self) -> bool:
        """Check if current value is valid."""
        return self._is_valid

    def reset(self) -> None:
        """Reset to default value."""
        try:
            input_widget = self.query_one("#input")
            if isinstance(input_widget, Switch):
                input_widget.value = bool(self.param.default)
            elif isinstance(input_widget, Select):
                input_widget.value = self.param.default
            else:
                input_widget.value = str(self.param.default)
        except Exception:
            pass


# =============================================================================
# Parameter Editor Widget
# =============================================================================


class ParameterEditor(Container):
    """
    Complete parameter editing form.

    Features:
    - Dynamic form generation from parameter definitions
    - Real-time validation
    - Batch value retrieval
    - Reset to defaults
    """

    DEFAULT_CSS = """
    ParameterEditor {
        height: auto;
        padding: 1;
        border: round $primary-darken-2;
        background: $surface;
    }

    ParameterEditor .editor-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    ParameterEditor .editor-actions {
        height: 3;
        layout: horizontal;
        margin-top: 1;
    }

    ParameterEditor .editor-actions Button {
        margin-right: 1;
    }
    """

    class ValuesChanged(Message):
        """Message sent when any parameter value changes."""

        def __init__(self, values: dict[str, Any], all_valid: bool) -> None:
            self.values = values
            self.all_valid = all_valid
            super().__init__()

    def __init__(
        self,
        parameters: list[ParameterDefinition],
        title: str = "PARAMETERS",
        show_actions: bool = False,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._parameters = parameters
        self._title = title
        self._show_actions = show_actions
        self._values: dict[str, Any] = {}
        self._validity: dict[str, bool] = {}

        # Initialize values
        for param in parameters:
            self._values[param.name] = param.current
            self._validity[param.name] = True

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="editor-title")

        for param in self._parameters:
            yield ParameterRow(param, id=f"param-{param.name}")

    def on_parameter_row_changed(self, event: ParameterRow.Changed) -> None:
        """Handle parameter value changes."""
        self._values[event.name] = event.value
        self._validity[event.name] = event.valid

        self.post_message(self.ValuesChanged(
            values=self._values.copy(),
            all_valid=all(self._validity.values()),
        ))

    def get_values(self) -> dict[str, Any]:
        """Get all current parameter values."""
        return self._values.copy()

    def is_all_valid(self) -> bool:
        """Check if all parameters are valid."""
        return all(self._validity.values())

    def reset_all(self) -> None:
        """Reset all parameters to default values."""
        for param in self._parameters:
            try:
                row = self.query_one(f"#param-{param.name}", ParameterRow)
                row.reset()
                self._values[param.name] = param.default
                self._validity[param.name] = True
            except Exception:
                pass

        self.post_message(self.ValuesChanged(
            values=self._values.copy(),
            all_valid=True,
        ))

    def set_values(self, values: dict[str, Any]) -> None:
        """Set parameter values programmatically."""
        for name, value in values.items():
            if name in self._values:
                try:
                    row = self.query_one(f"#param-{name}", ParameterRow)
                    input_widget = row.query_one("#input")
                    if isinstance(input_widget, Switch):
                        input_widget.value = bool(value)
                    elif isinstance(input_widget, Select):
                        input_widget.value = value
                    else:
                        input_widget.value = str(value)
                    self._values[name] = value
                except Exception:
                    pass


# =============================================================================
# Common Parameter Definitions
# =============================================================================

# Common symbol suggestions for autocomplete
COMMON_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "MATIC/USDT",
]

# Common timeframes
COMMON_TIMEFRAMES = [
    ("1 minute", "1m"),
    ("5 minutes", "5m"),
    ("15 minutes", "15m"),
    ("30 minutes", "30m"),
    ("1 hour", "1h"),
    ("4 hours", "4h"),
    ("1 day", "1d"),
    ("1 week", "1w"),
]


def create_sma_cross_parameters(
    fast_period: int = 10,
    slow_period: int = 30,
    threshold: float = 0.02,
    timeframe: str = "1h",
    symbol: str = "BTC/USDT",
) -> list[ParameterDefinition]:
    """Create parameter definitions for SMA Cross strategy."""
    return [
        ParameterDefinition(
            name="symbol",
            display_name="Symbol",
            param_type=ParameterType.SYMBOL,
            default=symbol,
            suggestions=COMMON_SYMBOLS,
            description="Trading pair",
        ),
        ParameterDefinition(
            name="timeframe",
            display_name="Timeframe",
            param_type=ParameterType.SELECT,
            default=timeframe,
            options=COMMON_TIMEFRAMES,
            description="Candle timeframe",
        ),
        ParameterDefinition(
            name="fast_period",
            display_name="Fast Period",
            param_type=ParameterType.INTEGER,
            default=fast_period,
            min_value=1,
            max_value=100,
            description="Fast SMA period",
        ),
        ParameterDefinition(
            name="slow_period",
            display_name="Slow Period",
            param_type=ParameterType.INTEGER,
            default=slow_period,
            min_value=1,
            max_value=500,
            description="Slow SMA period",
        ),
        ParameterDefinition(
            name="threshold",
            display_name="Threshold",
            param_type=ParameterType.FLOAT,
            default=threshold,
            min_value=0.001,
            max_value=0.1,
            description="Signal threshold",
        ),
    ]
