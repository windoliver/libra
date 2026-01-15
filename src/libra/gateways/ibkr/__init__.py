"""
Interactive Brokers Gateway for LIBRA.

Implements Issue #64: Interactive Brokers Gateway - Full Options Lifecycle.

Provides institutional-grade access to stocks, options, and futures via
TWS API using the ib_async library. Features include:
- Stock and options trading
- Real-time Greeks via market data
- Option chains without throttling
- Exercise/assignment handling
- Multi-leg combo orders (future)

Requires TWS or IB Gateway running locally.

Usage:
    from libra.gateways.ibkr import IBKRGateway, IBKRConfig

    # Paper trading with TWS
    config = IBKRConfig(port=7497)
    async with IBKRGateway("ibkr", config) as gw:
        positions = await gw.get_positions()

        # Place order
        from libra.gateways.protocol import Order, OrderSide, OrderType
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("100"),
        )
        result = await gw.submit_order(order)

        # Get option Greeks
        from libra.core.options import OptionContract, OptionType
        contract = OptionContract(
            symbol="AAPL250117C00150000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=Decimal("150"),
            expiration=date(2025, 1, 17),
        )
        greeks = await gw.get_greeks(contract)
"""

from libra.gateways.ibkr.config import (
    IBKRConfig,
    IBKRCredentials,
    IBKRPort,
)
from libra.gateways.ibkr.gateway import (
    IBKR_CAPABILITIES,
    IBKRConnectionError,
    IBKRGateway,
    IBKRNotConnectedError,
    IBKRNotInstalledError,
)

__all__ = [
    # Gateway
    "IBKRGateway",
    "IBKR_CAPABILITIES",
    # Config
    "IBKRConfig",
    "IBKRCredentials",
    "IBKRPort",
    # Exceptions
    "IBKRNotInstalledError",
    "IBKRConnectionError",
    "IBKRNotConnectedError",
]
