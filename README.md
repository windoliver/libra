# LIBRA

High-performance AI trading platform with LMAX Disruptor-inspired message bus.

## Features

- **Event-Driven Architecture**: Priority-based message bus (Risk > Orders > Signals > Market Data)
- **High Performance**: Target <100μs dispatch latency, 100K+ events/sec
- **Multi-Exchange**: 100+ exchanges via CCXT
- **AI-Powered**: Multi-agent LLM intelligence for strategy analysis

## Installation

```bash
# Development install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Quick Start

```python
import asyncio
from libra.core import MessageBus, Event, EventType

async def main():
    bus = MessageBus()

    async def on_tick(event: Event):
        print(f"Received: {event.payload}")

    bus.subscribe(EventType.TICK, on_tick)

    async with bus:
        await bus.publish(Event.create(
            EventType.TICK,
            source="gateway.binance",
            payload={"symbol": "BTC/USDT", "price": 50000.0}
        ))
        await asyncio.sleep(0.1)

asyncio.run(main())
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MESSAGE BUS                              │
│  Priority: RISK(0) > ORDERS(1) > SIGNALS(2) > MARKET_DATA(3)│
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
    ┌────▼────┐   ┌─────▼────┐   ┌────▼────┐   ┌────▼────┐
    │  Risk   │   │ Executor │   │Strategy │   │ Gateway │
    │ Manager │   │          │   │         │   │         │
    └─────────┘   └──────────┘   └─────────┘   └─────────┘
```

## License

MIT
