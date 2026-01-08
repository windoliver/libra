"""
LIBRA Gateways: Broker/exchange connectors.

Unified interface for all brokers/exchanges:
- Gateway Protocol: Standard interface
- CCXT Gateway: 100+ exchanges via CCXT
- Paper Gateway: Simulated paper trading

Usage:
    from libra.gateways import (
        Gateway,
        CCXTGateway,
        PaperGateway,
        Order,
        OrderResult,
        Position,
        Tick,
    )

    # Real trading via CCXT
    async with CCXTGateway("binance", config) as gateway:
        await gateway.subscribe(["BTC/USDT"])
        result = await gateway.submit_order(order)

    # Paper trading for testing
    async with PaperGateway(config) as gateway:
        gateway.update_price("BTC/USDT", bid, ask)
        result = await gateway.submit_order(order)
"""

from libra.gateways.ccxt_fetchers import (
    CCXTBalanceFetcher,
    CCXTBarFetcher,
    CCXTOrderBookFetcher,
    CCXTQuoteFetcher,
)
from libra.gateways.ccxt_gateway import CCXTGateway
from libra.gateways.fetcher import (
    # Query types
    AccountBalance,
    BalanceQuery,
    Bar,
    BarQuery,
    BaseQuery,
    # Fetcher protocol
    FetcherRegistry,
    GatewayFetcher,
    OrderBookLevel,
    OrderBookQuery,
    OrderBookSnapshot,
    OrderQuery,
    PositionQuery,
    Quote,
    TickQuery,
    TradeQuery,
    # Registry
    fetcher_registry,
)
from libra.gateways.paper_gateway import PaperGateway, SlippageConfig, SlippageModel
from libra.gateways.protocol import (
    # Exceptions
    AuthenticationError,
    # Data structures
    Balance,
    # Protocol and base class
    BaseGateway,
    ConnectionError,
    Gateway,
    GatewayError,
    InsufficientFundsError,
    Order,
    OrderBook,
    OrderError,
    OrderNotFoundError,
    OrderResult,
    # Enums
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    RateLimitError,
    Tick,
    TimeInForce,
    # Serialization
    decode_order,
    decode_order_result,
    encode_order,
    encode_order_result,
)


__all__ = [
    # Fetcher pattern (Issue #27)
    "AccountBalance",
    "BalanceQuery",
    "Bar",
    "BarQuery",
    "BaseQuery",
    "CCXTBalanceFetcher",
    "CCXTBarFetcher",
    "CCXTOrderBookFetcher",
    "CCXTQuoteFetcher",
    "FetcherRegistry",
    "GatewayFetcher",
    "OrderBookLevel",
    "OrderBookQuery",
    "OrderBookSnapshot",
    "OrderQuery",
    "PositionQuery",
    "Quote",
    "TickQuery",
    "TradeQuery",
    "fetcher_registry",
    # Exceptions
    "AuthenticationError",
    # Data structures
    "Balance",
    # Protocol and base
    "BaseGateway",
    # Implementations
    "CCXTGateway",
    "ConnectionError",
    "Gateway",
    "GatewayError",
    "InsufficientFundsError",
    "Order",
    "OrderBook",
    "OrderError",
    "OrderNotFoundError",
    "OrderResult",
    # Enums
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PaperGateway",
    "Position",
    "PositionSide",
    "RateLimitError",
    "SlippageConfig",
    "SlippageModel",
    "Tick",
    "TimeInForce",
    # Serialization
    "decode_order",
    "decode_order_result",
    "encode_order",
    "encode_order_result",
]
