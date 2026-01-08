"""
Order Entry Modal Screen.

Professional order entry form inspired by:
- Interactive Brokers TWS (rapid entry modes)
- Bloomberg Terminal (keyboard-first)
- lazygit/lazydocker (vim-style, instant feedback)

Features:
- Symbol selection with search
- BUY/SELL toggle (RadioSet)
- Order type selection (MARKET/LIMIT/STOP_LIMIT)
- Quantity input with USD equivalent
- Price input with market reference
- Real-time risk preview before submission
- Keyboard shortcuts (Ctrl+Enter submit, Esc cancel)
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Select,
    Static,
)


if TYPE_CHECKING:
    from libra.gateways.protocol import Order
    from libra.risk.manager import RiskManager


# =============================================================================
# Order Entry Result
# =============================================================================


class OrderEntryResult:
    """Result returned from the order entry screen."""

    def __init__(
        self,
        submitted: bool = False,
        symbol: str = "",
        side: str = "BUY",
        order_type: str = "MARKET",
        quantity: Decimal = Decimal("0"),
        price: Decimal | None = None,
    ) -> None:
        self.submitted = submitted
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price


# =============================================================================
# Risk Preview Widget
# =============================================================================


class RiskPreview(Static):
    """
    Real-time risk check preview.

    Shows the result of risk validation BEFORE order submission.
    Updates live as user changes order parameters.
    """

    DEFAULT_CSS = """
    RiskPreview {
        height: auto;
        min-height: 6;
        padding: 1;
        background: $surface-darken-1;
        border: round $primary-darken-2;
        margin-top: 1;
    }

    RiskPreview .risk-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    RiskPreview .risk-check {
        height: 1;
    }

    RiskPreview .risk-pass {
        color: $success;
    }

    RiskPreview .risk-fail {
        color: $error;
    }

    RiskPreview .risk-warn {
        color: $warning;
    }
    """

    def __init__(self, id: str = "risk-preview") -> None:
        super().__init__(id=id)
        self._checks: list[tuple[str, bool, str]] = []

    def compose(self) -> ComposeResult:
        yield Static("RISK CHECK", classes="risk-title")
        yield Static("", id="risk-checks")

    def update_checks(self, checks: list[tuple[str, bool, str]]) -> None:
        """
        Update the risk check display.

        Args:
            checks: List of (check_name, passed, message) tuples
        """
        self._checks = checks
        lines = []

        for name, passed, message in checks:
            icon = "[green]OK[/green]" if passed else "[red]!![/red]"
            color = "risk-pass" if passed else "risk-fail"
            lines.append(f"  {icon} {message}")

        try:
            checks_widget = self.query_one("#risk-checks", Static)
            checks_widget.update("\n".join(lines) if lines else "[dim]Enter order details...[/dim]")
        except Exception:
            pass

    def set_loading(self) -> None:
        """Show loading state."""
        try:
            checks_widget = self.query_one("#risk-checks", Static)
            checks_widget.update("[dim]Validating...[/dim]")
        except Exception:
            pass

    def clear(self) -> None:
        """Clear risk checks."""
        self._checks = []
        try:
            checks_widget = self.query_one("#risk-checks", Static)
            checks_widget.update("[dim]Enter order details...[/dim]")
        except Exception:
            pass


# =============================================================================
# Order Entry Screen
# =============================================================================


class OrderEntryScreen(ModalScreen[OrderEntryResult]):
    """
    Modal screen for manual order entry.

    Returns an OrderEntryResult when dismissed.

    Keyboard Shortcuts:
        Ctrl+Enter  Submit order
        Escape      Cancel
        Tab         Next field
        Shift+Tab   Previous field

    Layout:
        +---------------------------------------+
        |            NEW ORDER                  |
        +---------------------------------------+
        | Symbol:   [BTC/USDT          v]      |
        | Side:     (o) BUY   ( ) SELL         |
        | Type:     [LIMIT             v]      |
        | Quantity: [0.10000000________]       |
        | Price:    [51250_____________]       |
        |                                       |
        | +-- RISK CHECK -----------------+    |
        | | OK Position: 15% / 25% max    |    |
        | | OK Daily loss: $2k / $10k     |    |
        | | !! Large order: > $5,000      |    |
        | +-------------------------------+    |
        |                                       |
        |    [Cancel]      [Submit Order]      |
        +---------------------------------------+
    """

    DEFAULT_CSS = """
    OrderEntryScreen {
        align: center middle;
    }

    OrderEntryScreen > Container {
        width: 65;
        height: auto;
        max-height: 90%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    OrderEntryScreen .modal-title {
        text-align: center;
        text-style: bold;
        color: $text;
        background: $primary-darken-2;
        padding: 0 1;
        margin-bottom: 1;
        width: 100%;
    }

    OrderEntryScreen .form-row {
        height: 3;
        margin-bottom: 1;
    }

    OrderEntryScreen .form-label {
        width: 12;
        height: 3;
        content-align: left middle;
        color: $text-muted;
    }

    OrderEntryScreen .form-input {
        width: 1fr;
    }

    OrderEntryScreen Input {
        width: 100%;
    }

    OrderEntryScreen Select {
        width: 100%;
    }

    OrderEntryScreen RadioSet {
        height: 3;
        layout: horizontal;
    }

    OrderEntryScreen RadioButton {
        width: auto;
        margin-right: 2;
    }

    OrderEntryScreen .price-info {
        color: $text-muted;
        text-style: italic;
        margin-left: 1;
    }

    OrderEntryScreen .usd-equivalent {
        color: $success;
        margin-left: 1;
    }

    OrderEntryScreen .button-row {
        margin-top: 1;
        height: 3;
        align: center middle;
    }

    OrderEntryScreen Button {
        margin: 0 1;
    }

    OrderEntryScreen #submit-btn {
        background: $success;
    }

    OrderEntryScreen #cancel-btn {
        background: $error-darken-1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("ctrl+enter", "submit", "Submit", show=True),
    ]

    def __init__(
        self,
        symbols: list[str] | None = None,
        risk_manager: RiskManager | None = None,
        current_prices: dict[str, Decimal] | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """
        Initialize the order entry screen.

        Args:
            symbols: List of available trading symbols
            risk_manager: RiskManager for pre-trade validation
            current_prices: Dict of symbol -> current price
        """
        super().__init__(name=name, id=id, classes=classes)
        self._symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        self._risk_manager = risk_manager
        self._current_prices = current_prices or {
            "BTC/USDT": Decimal("51250.00"),
            "ETH/USDT": Decimal("3045.00"),
            "SOL/USDT": Decimal("142.50"),
        }

        # Form state
        self._selected_symbol: str = self._symbols[0] if self._symbols else ""
        self._selected_side: str = "BUY"
        self._selected_type: str = "LIMIT"
        self._quantity: Decimal = Decimal("0")
        self._price: Decimal | None = None

    def compose(self) -> ComposeResult:
        """Create the order entry form layout."""
        with Container():
            yield Static("NEW ORDER", classes="modal-title")

            # Symbol selection
            with Horizontal(classes="form-row"):
                yield Label("Symbol:", classes="form-label")
                with Container(classes="form-input"):
                    yield Select(
                        [(s, s) for s in self._symbols],
                        value=self._selected_symbol,
                        id="symbol-select",
                    )

            # Side selection (BUY/SELL)
            with Horizontal(classes="form-row"):
                yield Label("Side:", classes="form-label")
                with Container(classes="form-input"):
                    with RadioSet(id="side-radio"):
                        yield RadioButton("BUY", id="side-buy", value=True)
                        yield RadioButton("SELL", id="side-sell")

            # Order type selection
            with Horizontal(classes="form-row"):
                yield Label("Type:", classes="form-label")
                with Container(classes="form-input"):
                    yield Select(
                        [
                            ("MARKET", "MARKET"),
                            ("LIMIT", "LIMIT"),
                            ("STOP_LIMIT", "STOP_LIMIT"),
                        ],
                        value="LIMIT",
                        id="type-select",
                    )

            # Quantity input
            with Horizontal(classes="form-row"):
                yield Label("Quantity:", classes="form-label")
                with Container(classes="form-input"):
                    yield Input(
                        placeholder="0.00000000",
                        id="quantity-input",
                    )

            # USD equivalent display
            with Horizontal():
                yield Label("", classes="form-label")
                yield Static("", id="usd-equivalent", classes="usd-equivalent")

            # Price input (for LIMIT orders)
            with Horizontal(classes="form-row"):
                yield Label("Price:", classes="form-label")
                with Container(classes="form-input"):
                    yield Input(
                        placeholder="Enter price",
                        id="price-input",
                    )

            # Current market price reference
            with Horizontal():
                yield Label("", classes="form-label")
                yield Static("", id="market-price", classes="price-info")

            # Risk preview panel
            yield RiskPreview()

            # Action buttons
            with Horizontal(classes="button-row"):
                yield Button("Cancel (Esc)", id="cancel-btn", variant="error")
                yield Button("Submit Order (Ctrl+Enter)", id="submit-btn", variant="success")

    def on_mount(self) -> None:
        """Initialize form state on mount."""
        self._update_market_price()
        self._update_price_field_visibility()

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes."""
        if event.select.id == "symbol-select":
            self._selected_symbol = str(event.value) if event.value else ""
            self._update_market_price()
            self._update_usd_equivalent()
            self._update_risk_preview()

        elif event.select.id == "type-select":
            self._selected_type = str(event.value) if event.value else "LIMIT"
            self._update_price_field_visibility()
            self._update_risk_preview()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle side selection change."""
        if event.radio_set.id == "side-radio":
            self._selected_side = "BUY" if event.pressed.id == "side-buy" else "SELL"
            self._update_risk_preview()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes for live validation."""
        if event.input.id == "quantity-input":
            try:
                self._quantity = Decimal(event.value) if event.value else Decimal("0")
            except InvalidOperation:
                self._quantity = Decimal("0")
            self._update_usd_equivalent()
            self._update_risk_preview()

        elif event.input.id == "price-input":
            try:
                self._price = Decimal(event.value) if event.value else None
            except InvalidOperation:
                self._price = None
            self._update_risk_preview()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "submit-btn":
            self.action_submit()

    # =========================================================================
    # UI Updates
    # =========================================================================

    def _update_market_price(self) -> None:
        """Update the market price reference display."""
        try:
            price = self._current_prices.get(self._selected_symbol, Decimal("0"))
            market_price = self.query_one("#market-price", Static)
            market_price.update(f"Market: ${price:,.2f}")

            # Pre-fill price input with market price for new orders
            price_input = self.query_one("#price-input", Input)
            if not price_input.value:
                price_input.value = str(price)
                self._price = price
        except Exception:
            pass

    def _update_usd_equivalent(self) -> None:
        """Update USD equivalent display."""
        try:
            price = self._current_prices.get(self._selected_symbol, Decimal("0"))
            usd_value = self._quantity * price

            usd_widget = self.query_one("#usd-equivalent", Static)
            if usd_value > 0:
                usd_widget.update(f"~ ${usd_value:,.2f} USD")
            else:
                usd_widget.update("")
        except Exception:
            pass

    def _update_price_field_visibility(self) -> None:
        """Show/hide price field based on order type."""
        try:
            price_input = self.query_one("#price-input", Input)
            if self._selected_type == "MARKET":
                price_input.disabled = True
                price_input.placeholder = "N/A (Market Order)"
            else:
                price_input.disabled = False
                price_input.placeholder = "Enter price"
        except Exception:
            pass

    def _update_risk_preview(self) -> None:
        """Update the risk preview panel with current order parameters."""
        if self._quantity <= 0:
            try:
                self.query_one(RiskPreview).clear()
            except Exception:
                pass
            return

        # Build risk checks
        checks: list[tuple[str, bool, str]] = []

        # Get current price for calculations
        current_price = self._current_prices.get(self._selected_symbol, Decimal("0"))
        notional = self._quantity * current_price

        # Check 1: Position limit (simulated - would use RiskManager in real impl)
        position_pct = min(float(self._quantity) * 10, 100)  # Simulated
        position_limit = 25.0  # Simulated limit
        checks.append((
            "position_limit",
            position_pct < position_limit,
            f"Position: {position_pct:.0f}% / {position_limit:.0f}% max",
        ))

        # Check 2: Daily loss limit (simulated)
        daily_used = 2000  # Simulated
        daily_limit = 10000
        checks.append((
            "daily_loss",
            daily_used < daily_limit,
            f"Daily P&L: ${daily_used:,} / ${daily_limit:,} limit",
        ))

        # Check 3: Order rate (simulated)
        order_rate = 2
        rate_limit = 5
        checks.append((
            "rate_limit",
            order_rate < rate_limit,
            f"Order rate: {order_rate}/{rate_limit} per sec",
        ))

        # Check 4: Large order warning
        large_order_threshold = Decimal("5000")
        if notional > large_order_threshold:
            checks.append((
                "large_order",
                False,  # Warning, not blocking
                f"Large order: ${notional:,.2f} > ${large_order_threshold:,}",
            ))

        # If risk manager is available, use real validation
        if self._risk_manager and self._quantity > 0:
            self._perform_real_risk_check(checks)

        try:
            self.query_one(RiskPreview).update_checks(checks)
        except Exception:
            pass

    def _perform_real_risk_check(self, checks: list[tuple[str, bool, str]]) -> None:
        """Perform real risk validation using RiskManager."""
        # This would be implemented when integrating with real RiskManager
        # For now, we show simulated checks above
        pass

    # =========================================================================
    # Actions
    # =========================================================================

    def action_cancel(self) -> None:
        """Cancel and close the order entry screen."""
        self.dismiss(OrderEntryResult(submitted=False))

    def action_submit(self) -> None:
        """Validate and submit the order."""
        # Validate required fields
        if not self._selected_symbol:
            self.notify("Please select a symbol", severity="error")
            return

        if self._quantity <= 0:
            self.notify("Please enter a valid quantity", severity="error")
            return

        if self._selected_type != "MARKET" and not self._price:
            self.notify("Please enter a price for limit orders", severity="error")
            return

        # Create result
        result = OrderEntryResult(
            submitted=True,
            symbol=self._selected_symbol,
            side=self._selected_side,
            order_type=self._selected_type,
            quantity=self._quantity,
            price=self._price,
        )

        self.dismiss(result)
