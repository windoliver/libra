#!/usr/bin/env python3
"""
IBKR + TUI Integration Script.

Connects to Interactive Brokers TWS/Gateway and launches the LIBRA TUI.

Requirements:
    1. pip install -e ".[ibkr,tui]"
    2. TWS or IB Gateway running with API enabled on port 7497 (paper)

Usage:
    python scripts/run_ibkr_tui.py              # Paper trading (default)
    python scripts/run_ibkr_tui.py --live       # Live trading (port 7496)
    python scripts/run_ibkr_tui.py --gateway    # IB Gateway paper (port 4002)
    python scripts/run_ibkr_tui.py --port 7497  # Custom port
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch LIBRA TUI with Interactive Brokers gateway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Port Reference:
  7497  TWS Paper Trading (default)
  7496  TWS Live Trading
  4002  IB Gateway Paper
  4001  IB Gateway Live

Examples:
  python scripts/run_ibkr_tui.py                    # TWS paper trading
  python scripts/run_ibkr_tui.py --live             # TWS live trading
  python scripts/run_ibkr_tui.py --gateway          # IB Gateway paper
  python scripts/run_ibkr_tui.py --gateway --live   # IB Gateway live
  python scripts/run_ibkr_tui.py --demo             # Demo mode (no IBKR)
        """,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live trading port (7496 for TWS, 4001 for Gateway)",
    )
    parser.add_argument(
        "--gateway",
        action="store_true",
        help="Use IB Gateway ports instead of TWS",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port (default: auto-select based on --live/--gateway)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="TWS/Gateway host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=1,
        help="Client ID for IBKR connection (default: 1)",
    )
    parser.add_argument(
        "--account",
        type=str,
        default=None,
        help="Account ID (for multi-account setups)",
    )
    parser.add_argument(
        "--readonly",
        action="store_true",
        help="Connect in read-only mode (no order submission)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run TUI in demo mode without IBKR connection",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Test connection only, don't launch TUI",
    )
    return parser.parse_args()


async def test_connection(gateway: "IBKRGateway") -> bool:
    """Test IBKR connection and print account info."""
    print("\n" + "=" * 50)
    print("IBKR CONNECTION TEST")
    print("=" * 50)

    try:
        # Get positions
        print("\nFetching positions...")
        positions = await gateway.get_positions()
        if positions:
            print(f"  Found {len(positions)} position(s):")
            for pos in positions[:5]:  # Show first 5
                print(f"    - {pos.symbol}: {pos.quantity} @ {pos.entry_price}")
        else:
            print("  No open positions")

        # Get balances
        print("\nFetching balances...")
        balances = await gateway.get_balances()
        if balances:
            print(f"  Found {len(balances)} balance(s):")
            for bal in balances[:5]:  # Show first 5
                print(f"    - {bal.currency}: {bal.total} (available: {bal.available})")
        else:
            print("  No balance data")

        print("\n" + "=" * 50)
        print("CONNECTION TEST PASSED")
        print("=" * 50 + "\n")
        return True

    except Exception as e:
        print(f"\n  ERROR: {e}")
        print("\n" + "=" * 50)
        print("CONNECTION TEST FAILED")
        print("=" * 50 + "\n")
        return False


async def run_with_ibkr(args: argparse.Namespace) -> None:
    """Connect to IBKR and launch TUI."""
    # Import here to provide better error messages
    try:
        from libra.gateways.ibkr import IBKRGateway, IBKRConfig, IBKRPort
    except ImportError as e:
        print("ERROR: IBKR gateway not installed.")
        print("Run: pip install -e '.[ibkr]'")
        print(f"\nDetails: {e}")
        sys.exit(1)

    try:
        from libra.tui import run_app
    except ImportError as e:
        print("ERROR: TUI not installed.")
        print("Run: pip install -e '.[tui]'")
        print(f"\nDetails: {e}")
        sys.exit(1)

    # Determine port
    if args.port:
        port = args.port
    elif args.gateway:
        port = IBKRPort.GATEWAY_LIVE if args.live else IBKRPort.GATEWAY_PAPER
    else:
        port = IBKRPort.TWS_LIVE if args.live else IBKRPort.TWS_PAPER

    # Create config
    config = IBKRConfig(
        host=args.host,
        port=port,
        client_id=args.client_id,
        account=args.account,
        readonly=args.readonly,
    )

    # Display connection info
    mode = "LIVE" if config.is_paper is False else "PAPER"
    conn_type = "IB Gateway" if config.is_gateway else "TWS"

    print("\n" + "=" * 50)
    print("LIBRA Trading Terminal - IBKR Integration")
    print("=" * 50)
    print(f"  Host:      {config.host}")
    print(f"  Port:      {int(config.port)}")
    print(f"  Client ID: {config.client_id}")
    print(f"  Mode:      {mode}")
    print(f"  Type:      {conn_type}")
    print(f"  Account:   {config.account or 'Default'}")
    print(f"  Read-only: {config.readonly}")
    print("=" * 50)
    print("\nConnecting to IBKR...")

    # Create gateway and connect
    gateway = IBKRGateway("ibkr", config)

    try:
        await gateway.connect()
        print(f"Connected: {gateway.is_connected}")

        if args.test_only:
            # Just test connection
            await test_connection(gateway)
        else:
            # Test connection first
            success = await test_connection(gateway)
            if not success:
                print("WARNING: Connection test had issues, but launching TUI anyway...")

            # Launch TUI
            print("Launching TUI...")
            print("Press 'q' to quit, '?' for help\n")
            run_app(gateway=gateway, demo_mode=False)

    except ConnectionError as e:
        print(f"\nERROR: Could not connect to IBKR")
        print(f"  {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure TWS/IB Gateway is running")
        print("  2. Check API is enabled: Configure > API > Settings")
        print("  3. Verify port matches (paper=7497, live=7496)")
        print("  4. Check 'Enable ActiveX and Socket Clients' is checked")
        print("  5. Try a different client_id if another app is connected")
        sys.exit(1)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        if gateway.is_connected:
            print("\nDisconnecting from IBKR...")
            await gateway.disconnect()
            print("Disconnected.")


def run_demo() -> None:
    """Run TUI in demo mode without IBKR."""
    try:
        from libra.tui import run_app
    except ImportError as e:
        print("ERROR: TUI not installed.")
        print("Run: pip install -e '.[tui]'")
        print(f"\nDetails: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("LIBRA Trading Terminal - Demo Mode")
    print("=" * 50)
    print("Running without IBKR connection (simulated data)")
    print("Press 'q' to quit, '?' for help\n")

    run_app(demo_mode=True)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.demo:
        run_demo()
    else:
        asyncio.run(run_with_ibkr(args))


if __name__ == "__main__":
    main()
