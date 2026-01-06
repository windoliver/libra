"""
Status bar widget showing gateway connection status.
"""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """
    Displays gateway connection status and name.

    Shows a visual indicator (green/red) with gateway name and status text.
    Updates automatically when reactive attributes change.
    """

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        padding: 0 1;
        background: $surface-darken-1;
    }

    StatusBar.connected {
        color: $success;
    }

    StatusBar.disconnected {
        color: $error;
    }
    """

    connected: reactive[bool] = reactive(False)
    gateway_name: reactive[str] = reactive("No Gateway")

    def render(self) -> str:
        """Render the status bar content."""
        if self.connected:
            indicator = "[green]●[/green]"
            status = "Connected"
        else:
            indicator = "[red]●[/red]"
            status = "Disconnected"

        return f"{indicator} [{self.gateway_name}] {status}"

    def watch_connected(self, connected: bool) -> None:
        """Update CSS class when connection state changes."""
        self.remove_class("connected", "disconnected")
        self.add_class("connected" if connected else "disconnected")

    def set_status(self, connected: bool, gateway_name: str | None = None) -> None:
        """
        Update the status bar.

        Args:
            connected: Whether the gateway is connected.
            gateway_name: Name of the gateway (optional).
        """
        self.connected = connected
        if gateway_name is not None:
            self.gateway_name = gateway_name
