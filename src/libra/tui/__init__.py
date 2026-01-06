"""
LIBRA TUI: Terminal User Interface.

Textual-based terminal UI for the trading platform:
- Dashboard layout with balances, positions, and logs
- Real-time event streaming from MessageBus
- Built-in commands for control
- Demo mode for standalone testing

Usage:
    # Run in demo mode
    python -m libra.tui

    # Or import and use programmatically
    from libra.tui import LibraApp, run_app
    app = LibraApp(bus=my_bus, gateway=my_gateway)
    app.run()

Keyboard shortcuts:
    /  or :   Focus command input
    Ctrl+P    Command palette (fuzzy search)
    Ctrl+C    Copy log to clipboard
    Q         Quit
    F1        Help
    F2        Toggle dark/light theme
    ESC       Cancel/unfocus input
"""

from libra.tui.app import LibraApp, run_app


__all__ = [
    "LibraApp",
    "run_app",
]
