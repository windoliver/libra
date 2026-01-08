"""
Strategy Edit Modal Screen.

Modal dialog for editing strategy parameters.

Features:
- Parameter form with validation
- Save/Cancel actions
- Keyboard shortcuts (Ctrl+Enter save, Esc cancel)
- Type-safe return values

Design inspired by:
- lazygit confirmation dialogs
- VS Code settings modal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from libra.tui.widgets.parameter_editor import (
    ParameterDefinition,
    ParameterEditor,
    ParameterType,
    create_sma_cross_parameters,
)


if TYPE_CHECKING:
    pass


# =============================================================================
# Strategy Edit Result
# =============================================================================


@dataclass
class StrategyEditResult:
    """Result returned from strategy edit modal."""

    saved: bool = False
    strategy_id: str = ""
    strategy_name: str = ""
    parameters: dict[str, Any] | None = None


# =============================================================================
# Strategy Edit Modal
# =============================================================================


class StrategyEditModal(ModalScreen[StrategyEditResult]):
    """
    Modal screen for editing strategy parameters.

    Returns:
        StrategyEditResult with saved=True if user saved, False if cancelled.
    """

    DEFAULT_CSS = """
    StrategyEditModal {
        align: center middle;
    }

    StrategyEditModal > Container {
        width: 70;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }

    StrategyEditModal .modal-title {
        height: 1;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
        color: $text;
    }

    StrategyEditModal .modal-subtitle {
        height: 1;
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }

    StrategyEditModal .modal-content {
        height: auto;
        max-height: 50;
        overflow-y: auto;
    }

    StrategyEditModal .modal-actions {
        height: 3;
        layout: horizontal;
        align: center middle;
        margin-top: 1;
    }

    StrategyEditModal .modal-actions Button {
        margin: 0 1;
        min-width: 12;
    }

    StrategyEditModal .validation-status {
        height: 1;
        text-align: center;
        margin-top: 1;
    }

    StrategyEditModal .validation-status.valid {
        color: $success;
    }

    StrategyEditModal .validation-status.invalid {
        color: $error;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+s", "save", "Save", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+enter", "save", "Save", show=False),
    ]

    def __init__(
        self,
        strategy_id: str,
        strategy_name: str,
        parameters: list[ParameterDefinition] | None = None,
        title: str = "Edit Strategy",
    ) -> None:
        super().__init__()
        self._strategy_id = strategy_id
        self._strategy_name = strategy_name
        self._parameters = parameters or create_sma_cross_parameters()
        self._title = title
        self._all_valid = True

    def compose(self) -> ComposeResult:
        with Container():
            yield Static(self._title, classes="modal-title")
            yield Static(f"Strategy: {self._strategy_name}", classes="modal-subtitle")

            with Vertical(classes="modal-content"):
                yield ParameterEditor(
                    parameters=self._parameters,
                    title="PARAMETERS",
                    id="param-editor",
                )

            yield Static("All parameters valid", classes="validation-status valid", id="validation-status")

            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Save", variant="primary", id="btn-save")

    def on_parameter_editor_values_changed(self, event: ParameterEditor.ValuesChanged) -> None:
        """Handle parameter validation changes."""
        self._all_valid = event.all_valid

        try:
            status = self.query_one("#validation-status", Static)
            save_btn = self.query_one("#btn-save", Button)

            if event.all_valid:
                status.update("All parameters valid")
                status.remove_class("invalid")
                status.add_class("valid")
                save_btn.disabled = False
            else:
                status.update("Some parameters are invalid")
                status.remove_class("valid")
                status.add_class("invalid")
                save_btn.disabled = True
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "btn-save":
            self.action_save()
        elif event.button.id == "btn-cancel":
            self.action_cancel()

    def action_save(self) -> None:
        """Save and dismiss."""
        if not self._all_valid:
            self.notify("Please fix validation errors", severity="warning")
            return

        try:
            editor = self.query_one("#param-editor", ParameterEditor)
            values = editor.get_values()

            result = StrategyEditResult(
                saved=True,
                strategy_id=self._strategy_id,
                strategy_name=self._strategy_name,
                parameters=values,
            )
            self.dismiss(result)
        except Exception as e:
            self.notify(f"Error saving: {e}", severity="error")

    def action_cancel(self) -> None:
        """Cancel and dismiss."""
        result = StrategyEditResult(
            saved=False,
            strategy_id=self._strategy_id,
            strategy_name=self._strategy_name,
        )
        self.dismiss(result)


# =============================================================================
# Strategy Create Modal
# =============================================================================


class StrategyCreateModal(ModalScreen[StrategyEditResult]):
    """
    Modal screen for creating a new strategy.

    Similar to StrategyEditModal but for new strategies.
    """

    DEFAULT_CSS = """
    StrategyCreateModal {
        align: center middle;
    }

    StrategyCreateModal > Container {
        width: 70;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }

    StrategyCreateModal .modal-title {
        height: 1;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }

    StrategyCreateModal .strategy-type-section {
        height: auto;
        margin-bottom: 1;
    }

    StrategyCreateModal .section-label {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    StrategyCreateModal .modal-content {
        height: auto;
        max-height: 50;
        overflow-y: auto;
    }

    StrategyCreateModal .modal-actions {
        height: 3;
        layout: horizontal;
        align: center middle;
        margin-top: 1;
    }

    StrategyCreateModal .modal-actions Button {
        margin: 0 1;
        min-width: 12;
    }

    StrategyCreateModal .validation-status {
        height: 1;
        text-align: center;
        margin-top: 1;
    }

    StrategyCreateModal .validation-status.valid {
        color: $success;
    }

    StrategyCreateModal .validation-status.invalid {
        color: $error;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+s", "save", "Create", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
    ]

    # Available strategy types
    STRATEGY_TYPES = [
        ("SMA Crossover", "sma_cross", create_sma_cross_parameters),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._strategy_type = "sma_cross"
        self._strategy_name = "New Strategy"
        self._all_valid = True
        self._name_params = [
            ParameterDefinition(
                name="strategy_name",
                display_name="Strategy Name",
                param_type=ParameterType.STRING,
                default="New Strategy",
                description="Name for your strategy",
            ),
        ]

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("CREATE NEW STRATEGY", classes="modal-title")

            with Vertical(classes="modal-content"):
                # Strategy name
                yield ParameterEditor(
                    parameters=self._name_params,
                    title="STRATEGY NAME",
                    id="name-editor",
                )

                # Strategy parameters
                yield ParameterEditor(
                    parameters=create_sma_cross_parameters(),
                    title="PARAMETERS",
                    id="param-editor",
                )

            yield Static("All fields valid", classes="validation-status valid", id="validation-status")

            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Create", variant="primary", id="btn-create")

    def on_parameter_editor_values_changed(self, event: ParameterEditor.ValuesChanged) -> None:
        """Handle validation changes."""
        # Check both editors
        try:
            name_editor = self.query_one("#name-editor", ParameterEditor)
            param_editor = self.query_one("#param-editor", ParameterEditor)
            self._all_valid = name_editor.is_all_valid() and param_editor.is_all_valid()

            status = self.query_one("#validation-status", Static)
            create_btn = self.query_one("#btn-create", Button)

            if self._all_valid:
                status.update("All fields valid")
                status.remove_class("invalid")
                status.add_class("valid")
                create_btn.disabled = False
            else:
                status.update("Please fix validation errors")
                status.remove_class("valid")
                status.add_class("invalid")
                create_btn.disabled = True
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "btn-create":
            self.action_save()
        elif event.button.id == "btn-cancel":
            self.action_cancel()

    def action_save(self) -> None:
        """Create the strategy."""
        if not self._all_valid:
            self.notify("Please fix validation errors", severity="warning")
            return

        try:
            name_editor = self.query_one("#name-editor", ParameterEditor)
            param_editor = self.query_one("#param-editor", ParameterEditor)

            name_values = name_editor.get_values()
            param_values = param_editor.get_values()

            # Generate a simple ID
            import time
            strategy_id = f"strategy_{int(time.time())}"

            result = StrategyEditResult(
                saved=True,
                strategy_id=strategy_id,
                strategy_name=name_values.get("strategy_name", "New Strategy"),
                parameters=param_values,
            )
            self.dismiss(result)
        except Exception as e:
            self.notify(f"Error creating strategy: {e}", severity="error")

    def action_cancel(self) -> None:
        """Cancel creation."""
        self.dismiss(StrategyEditResult(saved=False))


# =============================================================================
# Confirmation Modal
# =============================================================================


class ConfirmationModal(ModalScreen[bool]):
    """
    Simple confirmation modal for destructive actions.

    Returns True if confirmed, False if cancelled.
    """

    DEFAULT_CSS = """
    ConfirmationModal {
        align: center middle;
    }

    ConfirmationModal > Container {
        width: 50;
        height: auto;
        background: $surface;
        border: round $warning;
        padding: 1 2;
    }

    ConfirmationModal .modal-title {
        height: 1;
        text-style: bold;
        text-align: center;
        color: $warning;
        margin-bottom: 1;
    }

    ConfirmationModal .modal-message {
        height: auto;
        text-align: center;
        margin: 1 0;
    }

    ConfirmationModal .modal-actions {
        height: 3;
        layout: horizontal;
        align: center middle;
        margin-top: 1;
    }

    ConfirmationModal .modal-actions Button {
        margin: 0 1;
        min-width: 10;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("y", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
        Binding("enter", "confirm", "Confirm", show=False),
        Binding("escape", "cancel", "Cancel", priority=True),
    ]

    def __init__(
        self,
        title: str = "Confirm Action",
        message: str = "Are you sure?",
        confirm_label: str = "Yes",
        cancel_label: str = "No",
    ) -> None:
        super().__init__()
        self._title = title
        self._message = message
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label

    def compose(self) -> ComposeResult:
        with Container():
            yield Static(self._title, classes="modal-title")
            yield Static(self._message, classes="modal-message")
            with Horizontal(classes="modal-actions"):
                yield Button(self._cancel_label, variant="default", id="btn-cancel")
                yield Button(self._confirm_label, variant="warning", id="btn-confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "btn-confirm":
            self.action_confirm()
        else:
            self.action_cancel()

    def action_confirm(self) -> None:
        """Confirm the action."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Cancel the action."""
        self.dismiss(False)
