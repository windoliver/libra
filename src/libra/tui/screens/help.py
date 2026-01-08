"""
Help screen modal showing all keyboard shortcuts.

Provides instant discoverability of all available shortcuts.
Press ? from anywhere to show this overlay.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static


class HelpScreen(ModalScreen):
    """
    Modal help overlay showing all keyboard shortcuts.

    Press ? or ESC to dismiss.

    Layout:
    ┌─────────────────────────────────────────┐
    │           KEYBOARD SHORTCUTS            │
    ├─────────────────────────────────────────┤
    │  NAVIGATION                             │
    │  ─────────────────────────────────────  │
    │  j / ↓        Move down                 │
    │  k / ↑        Move up                   │
    │  h / ←        Previous tab              │
    │  l / →        Next tab                  │
    │  1-4          Switch to tab             │
    │                                         │
    │  COMMANDS                               │
    │  ─────────────────────────────────────  │
    │  / or :       Focus command input       │
    │  Ctrl+P       Command palette           │
    │  Ctrl+C       Copy log to clipboard     │
    │  ?            Show this help            │
    │                                         │
    │  ACTIONS                                │
    │  ─────────────────────────────────────  │
    │  q            Quit                      │
    │  F1           Help                      │
    │  F2           Toggle theme              │
    │  ESC          Cancel / Close            │
    │                                         │
    │         Press ? or ESC to close         │
    └─────────────────────────────────────────┘
    """

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }

    HelpScreen > Container {
        width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }

    HelpScreen .help-title {
        text-align: center;
        text-style: bold;
        color: $text;
        background: $primary-darken-2;
        padding: 0 1;
        margin-bottom: 1;
    }

    HelpScreen .help-section {
        color: $success;
        text-style: bold;
        margin-top: 1;
    }

    HelpScreen .help-divider {
        color: $primary-darken-1;
    }

    HelpScreen .help-row {
        height: 1;
    }

    HelpScreen .help-key {
        color: $warning;
        text-style: bold;
        width: 16;
    }

    HelpScreen .help-desc {
        color: $text;
    }

    HelpScreen .help-footer {
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
        Binding("question_mark", "dismiss", "Close", show=False),
        Binding("q", "dismiss", "Close", show=False),
    ]

    SHORTCUTS = [
        # (section, [(key, description), ...])
        (
            "NAVIGATION",
            [
                ("j / ↓", "Move down in list"),
                ("k / ↑", "Move up in list"),
                ("h / ←", "Previous tab"),
                ("l / →", "Next tab"),
                ("g", "Go to top"),
                ("G", "Go to bottom"),
                ("1-5", "Switch to tab directly"),
            ],
        ),
        (
            "TRADING",
            [
                ("o / n", "Open order entry form"),
                ("Ctrl+Enter", "Submit order (in form)"),
                ("ESC", "Cancel order / Close modal"),
            ],
        ),
        (
            "COMMANDS",
            [
                ("/ or :", "Focus command input"),
                ("Ctrl+P", "Command palette (fuzzy search)"),
                ("Ctrl+C", "Copy log to clipboard"),
                ("?", "Show this help"),
            ],
        ),
        (
            "ACTIONS",
            [
                ("q", "Quit application"),
                ("F1", "Show help"),
                ("F2", "Toggle dark/light theme"),
                ("ESC", "Cancel / Close modal"),
            ],
        ),
        (
            "COMMANDS (type in input)",
            [
                ("/help", "Show help"),
                ("/status", "Show system status"),
                ("/clear", "Clear event log"),
                ("/theme", "Toggle theme"),
                ("/quit", "Exit application"),
            ],
        ),
    ]

    def compose(self) -> ComposeResult:
        """Create the help overlay layout."""
        with Container():
            yield Static("KEYBOARD SHORTCUTS", classes="help-title")

            with VerticalScroll():
                for section, shortcuts in self.SHORTCUTS:
                    yield Static(section, classes="help-section")
                    yield Static("─" * 40, classes="help-divider")

                    for key, desc in shortcuts:
                        with Container(classes="help-row"):
                            yield Static(f"  {key:<14} {desc}")

            yield Static("Press ? or ESC to close", classes="help-footer")

    def action_dismiss(self) -> None:
        """Close the help screen."""
        self.dismiss()
