"""
Command input widget with keyboard-first design.

Based on Textual best practices:
- Extends Input directly (not wrapped in container)
- ESC binding to blur/unfocus
- Minimal CSS to avoid focus issues
"""

from __future__ import annotations

from typing import ClassVar

from textual.binding import Binding
from textual.message import Message
from textual.widgets import Input


class CommandInput(Input):
    """
    Keyboard-centric command input widget.

    Key bindings:
    - ESC: Blur (unfocus) the input
    - ENTER: Submit command

    Messages:
    - CommandSubmitted: Fired when user submits a command
    - BuiltinCommand: Fired for built-in commands (/help, /quit, etc.)
    """

    # Minimal CSS - avoid specific styling that breaks focus
    DEFAULT_CSS = """
    CommandInput {
        dock: bottom;
        margin: 0 1 1 1;
    }
    """

    # ESC to unfocus - critical for keyboard-first design
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "blur", "Cancel", show=False),
    ]

    class CommandSubmitted(Message):
        """Fired when a command is submitted."""

        def __init__(self, command: str, args: list[str]) -> None:
            self.command = command
            self.args = args
            super().__init__()

    class BuiltinCommand(Message):
        """Fired for built-in command execution."""

        def __init__(self, command: str) -> None:
            self.command = command
            super().__init__()

    BUILTIN_COMMANDS: ClassVar[dict[str, str]] = {
        "/help": "Show available commands",
        "/quit": "Exit application",
        "/exit": "Exit application",
        "/clear": "Clear log viewer",
        "/theme": "Toggle dark/light theme",
        "/status": "Show connection status",
    }

    def __init__(self, id: str = "command-input") -> None:
        """Initialize command input."""
        super().__init__(
            placeholder="Press / or : to type commands, ESC to cancel",
            id=id,
        )

    def action_blur(self) -> None:
        """Blur (unfocus) this input widget."""
        self.value = ""  # Clear on cancel
        self.blur()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command submission."""
        command = event.value.strip()

        # Clear and blur
        self.value = ""
        self.blur()

        if not command:
            return

        # Parse command
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:]

        # Normalize command (add / prefix if missing for known commands)
        if not cmd.startswith("/"):
            prefixed = f"/{cmd}"
            if prefixed in self.BUILTIN_COMMANDS:
                cmd = prefixed

        # Dispatch
        if cmd in self.BUILTIN_COMMANDS:
            self.post_message(self.BuiltinCommand(cmd))
        else:
            self.post_message(self.CommandSubmitted(cmd, args))

    def get_help_text(self) -> str:
        """Get formatted help text for all commands."""
        lines = ["Available commands:"]
        for cmd, desc in sorted(self.BUILTIN_COMMANDS.items()):
            lines.append(f"  {cmd:<12} {desc}")
        return "\n".join(lines)
