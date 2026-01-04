"""
LIBRA: High-performance AI trading platform.

A unified, AI-powered trading platform with:
- LMAX Disruptor-inspired message bus
- Multi-agent LLM intelligence
- Plug-and-play strategy ecosystem
- Real money execution (100+ exchanges via CCXT)

Quick Start:
    from libra.core import Event, EventType, MessageBus
    from libra.gateways import CCXTGateway, PaperGateway, Order, OrderSide, OrderType

    # Create message bus
    bus = MessageBus()

    # Connect to exchange
    async with CCXTGateway("binance", config) as gateway:
        await gateway.subscribe(["BTC/USDT"])
        result = await gateway.submit_order(order)
"""

__version__ = "0.1.0"
__author__ = "Libra Team"

# Re-export core components for convenience
from libra.core import (
    Event,
    EventType,
    MessageBus,
    Priority,
)

# Re-export gateway components
from libra.gateways import (
    CCXTGateway,
    Gateway,
    Order,
    OrderBook,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperGateway,
    Position,
    Tick,
)


__all__ = [
    # Gateways
    "CCXTGateway",
    # Core
    "Event",
    "EventType",
    "Gateway",
    "MessageBus",
    "Order",
    "OrderBook",
    "OrderResult",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PaperGateway",
    "Position",
    "Priority",
    "Tick",
    "__author__",
    # Version
    "__version__",
]
