"""
Entry point for running LIBRA TUI as a module.

Usage:
    python -m libra.tui          # Run in demo mode (default)
    python -m libra.tui --live   # Run in live mode (requires gateway)
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Parse arguments and run the TUI."""
    parser = argparse.ArgumentParser(
        description="LIBRA Trading Terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m libra.tui          # Run in demo mode with simulated data
    python -m libra.tui --live   # Run in live mode (requires gateway config)

Keyboard shortcuts:
    /  or :   Focus command input
    Ctrl+P    Command palette (fuzzy search)
    Ctrl+C    Copy log to clipboard
    Q         Quit
    F1        Help
    F2        Toggle dark/light theme
    ESC       Cancel/unfocus input
        """,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Run in live mode (requires gateway configuration)",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    from libra.tui.app import run_app

    run_app(demo_mode=not args.live)


if __name__ == "__main__":
    main()
