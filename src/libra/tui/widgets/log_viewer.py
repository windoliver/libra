"""
Log viewer widget for real-time event display.

Shows scrollable log with color-coded entries by priority/level.
Non-focusable to prevent stealing focus from command input.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import RichLog, Static


if TYPE_CHECKING:
    from libra.core.events import Event


# Priority to color mapping (for Event objects)
PRIORITY_COLORS: dict[int, str] = {
    0: "red bold",  # RISK - highest priority
    1: "yellow",  # ORDERS
    2: "cyan",  # SIGNALS
    3: "dim",  # MARKET_DATA - lowest priority
}

# Level to color mapping (for plain messages)
LEVEL_COLORS: dict[str, str] = {
    "info": "white",
    "warning": "yellow",
    "error": "red bold",
    "success": "green",
    "debug": "dim",
}


class LogViewer(Vertical, can_focus=False):
    """
    Displays real-time event log with color-coding.

    Features:
    - Non-focusable (won't steal focus from input)
    - Auto-scrolls to bottom
    - Max 500 lines (older entries removed)
    - Color-coded by priority/level

    Color scheme:
    - RISK events: red bold
    - ORDER events: yellow
    - SIGNAL events: cyan
    - MARKET_DATA events: dim

    Copy support:
    - Use Ctrl+C with App.copy_to_clipboard()
    - Terminal selection: hold SHIFT (most terminals) or OPTION (iTerm)
    """

    DEFAULT_CSS = """
    LogViewer {
        height: auto;
        min-height: 8;
        max-height: 12;
        border: round $primary-darken-1;
        background: $surface;
    }

    LogViewer > Static {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }

    LogViewer > RichLog {
        height: 1fr;
        background: $surface;
        padding: 0 1;
        scrollbar-gutter: stable;
    }
    """

    MAX_LINES = 500

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("EVENT LOG")
        yield RichLog(
            highlight=True,
            markup=True,
            max_lines=self.MAX_LINES,
            wrap=False,
            auto_scroll=True,
            id="event-log",
        )

    def on_mount(self) -> None:
        """Configure RichLog after mounting."""
        log = self.query_one(RichLog)
        log.can_focus = False  # Prevent focus stealing
        log.auto_links = False  # Don't capture clicks for links

    def log_event(self, event: Event) -> None:
        """
        Log an event to the viewer.

        Args:
            event: The event to log.
        """
        log = self.query_one(RichLog)
        timestamp = datetime.fromtimestamp(
            event.timestamp_sec, tz=timezone.utc
        ).strftime("%H:%M:%S")
        color = PRIORITY_COLORS.get(event.priority, "white")

        # Format payload for display
        payload_str = ""
        if event.payload:
            if "symbol" in event.payload:
                payload_str += f" {event.payload['symbol']}"
            if "price" in event.payload:
                payload_str += f" @ {event.payload['price']:,.2f}"
            elif "last" in event.payload:
                payload_str += f" @ {event.payload['last']:,.2f}"

        log.write(
            f"[dim]{timestamp}[/dim] [{color}]{event.event_type.name}[/{color}]"
            f"[dim]{payload_str}[/dim]"
        )

    def log_message(
        self,
        message: str,
        level: str = "info",
    ) -> None:
        """
        Log a plain message.

        Args:
            message: The message to log.
            level: Log level (info, warning, error, success, debug).
        """
        log = self.query_one(RichLog)
        timestamp = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        color = LEVEL_COLORS.get(level, "white")

        log.write(f"[dim]{timestamp}[/dim] [{color}]{message}[/{color}]")

    def clear_log(self) -> None:
        """Clear all log entries."""
        log = self.query_one(RichLog)
        log.clear()

    def get_log_text(self, max_lines: int = 50) -> str:
        """
        Get plain text content of recent log entries.

        Args:
            max_lines: Maximum number of recent lines to return.

        Returns:
            Plain text of log entries, one per line.
        """
        log = self.query_one(RichLog)
        lines = []

        for line in log.lines[-max_lines:]:
            # Each line is a Strip, extract text from segments
            try:
                plain = "".join(segment.text for segment in line._segments)
                lines.append(plain)
            except (AttributeError, TypeError):
                # Fallback if structure changes
                lines.append(str(line))

        return "\n".join(lines)

    def set_demo_data(self) -> None:
        """Populate with demo log entries."""
        self.log_message("LIBRA Trading Terminal started", "success")
        self.log_message("Connecting to gateway...", "info")
        self.log_message("Gateway connected: Demo", "success")
        self.log_message("Loaded 3 active positions", "info")
        self.log_message("Risk manager initialized", "info")
        self.log_message("Ready for trading", "success")
