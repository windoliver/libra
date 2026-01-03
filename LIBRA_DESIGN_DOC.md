# AQUARIUS: Next-Generation AI Trading Platform
## Technical Design Document v1.0

**Author:** Aquarius Team
**Date:** January 2026
**Status:** Draft

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Vision & Goals](#2-vision--goals)
3. [Architecture Overview](#3-architecture-overview)
4. [Core Components](#4-core-components)
5. [Integration Strategy](#5-integration-strategy)
6. [Data Architecture](#6-data-architecture)
7. [Agent System](#7-agent-system)
8. [Risk Management](#8-risk-management)
9. [Execution Plan](#9-execution-plan)
10. [Technical Specifications](#10-technical-specifications)

---

## 1. Executive Summary

### 1.1 What is Aquarius?

Aquarius is a unified, AI-powered trading platform that democratizes institutional-grade trading capabilities for retail users. It combines:

- **Multi-Agent LLM Intelligence** (TradingAgents-style analysis)
- **Plug-and-Play Strategy Ecosystem** (Freqtrade, Hummingbot, Jesse, FinRL)
- **Real Money Execution** (100+ exchanges via CCXT)
- **No Vendor Lock-in** (modular adapter architecture)
- **Terminal-First UX** (Textual TUI)

### 1.2 Key Differentiators

| Feature | Traditional Bots | Aquarius |
|---------|-----------------|----------|
| Intelligence | Rule-based | LLM-powered multi-agent |
| Strategy Source | Single framework | Plug any framework |
| Asset Classes | Usually one | Stocks, Crypto, Options, Prediction Markets |
| User Interface | Web/Mobile | Terminal TUI (power users) |
| Execution | Paper or single broker | Unified real-money multi-broker |

### 1.3 Built on Nexus

Aquarius leverages the existing Nexus infrastructure:
- **Plugin System** (`nexus.plugins`) → Strategy adapters
- **Agent System** (`nexus.core.agents`) → Trading agents
- **LangGraph Integration** (`nexus.tools.langgraph`) → Multi-agent orchestration
- **Storage** (`nexus.core.nexus_fs`) → Strategies, configs, results
- **Permissions** (`nexus.core.rebac`) → Agent access control

---

## 2. Vision & Goals

### 2.1 Vision Statement

> "An AI agent that trades smarter than the average human, 24/7, across any market, with institutional-grade risk management—accessible to everyone."

### 2.2 Design Principles

1. **Agent > Human for Grinding Tasks**
   - Agents watch markets 24/7 (humans sleep)
   - Agents execute in milliseconds (humans are slow)
   - Agents have no emotions (humans FOMO/FUD)
   - Agents can run 100 strategies (humans can manage 2-3)

2. **No Vendor Lock-in**
   - Unified interfaces for all brokers
   - Swap strategies without code changes
   - Mix frameworks freely (Freqtrade + Hummingbot + custom)

3. **Real Money Ready**
   - Not just paper trading
   - Production-grade risk management
   - Audit trails for every decision

4. **Progressive Complexity**
   - Beginner: "Grow my $10k safely"
   - Intermediate: Select from strategy marketplace
   - Advanced: Custom ML models + full control

### 2.3 Success Metrics

| Metric | Target |
|--------|--------|
| Strategy Frameworks Integrated | 5+ (Freqtrade, Hummingbot, Jesse, FinRL, OctoBot) |
| Exchanges Supported | 100+ (via CCXT) |
| Latency (signal to order) | <100ms for non-HFT |
| Backtest/Live Code Parity | 100% (same code runs both) |
| Agent Response Time | <5s for analysis requests |

---

## 3. Architecture Overview

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AQUARIUS PLATFORM                                  │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                         PRESENTATION LAYER                                │  │
│  │                                                                           │  │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │  │
│  │   │  TUI App    │  │  REST API   │  │  Telegram   │  │    Voice    │     │  │
│  │   │  (Textual)  │  │  (FastAPI)  │  │     Bot     │  │  (Whisper)  │     │  │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │  │
│  │                                                                           │  │
│  │   Natural Language: "Run momentum strategy on BTC with 2% risk"          │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│  ┌───────────────────────────────────┼───────────────────────────────────────┐  │
│  │                    AGENT ORCHESTRATION LAYER                              │  │
│  │                   (LangGraph + TradingAgents)                             │  │
│  │                                                                           │  │
│  │   ┌─────────────────────────────────────────────────────────────────┐    │  │
│  │   │                    COORDINATOR AGENT                            │    │  │
│  │   │              (Routes requests, synthesizes results)             │    │  │
│  │   └───────────────────────────┬─────────────────────────────────────┘    │  │
│  │                               │                                          │  │
│  │   ┌───────────┬───────────┬───┴───┬───────────┬───────────┐             │  │
│  │   │           │           │       │           │           │             │  │
│  │   ▼           ▼           ▼       ▼           ▼           ▼             │  │
│  │ ┌─────┐   ┌─────┐   ┌─────────┐ ┌─────┐   ┌─────┐   ┌─────────┐        │  │
│  │ │Deep │   │Fund-│   │Sentiment│ │Tech │   │Risk │   │Executor │        │  │
│  │ │Rsrch│   │ament│   │ Analyst │ │Anlst│   │ Mgr │   │  Agent  │        │  │
│  │ └─────┘   └─────┘   └─────────┘ └─────┘   └─────┘   └─────────┘        │  │
│  │                                                                          │  │
│  │   ┌─────────────────────────────────────────────────────────────────┐    │  │
│  │   │              DEBATE MECHANISM (Bull vs Bear)                    │    │  │
│  │   └─────────────────────────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│  ┌───────────────────────────────────┼───────────────────────────────────────┐  │
│  │                         MESSAGE BUS                                       │  │
│  │              (Event-Driven, Async, All Components)                        │  │
│  │                                                                           │  │
│  │   Events: MarketData | Signal | Order | Fill | Risk | Position           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│  ┌───────────────────────────────────┼───────────────────────────────────────┐  │
│  │                     STRATEGY PLUGIN LAYER                                 │  │
│  │                   (Nexus Plugin Architecture)                             │  │
│  │                                                                           │  │
│  │   ┌─────────────────────────────────────────────────────────────────┐    │  │
│  │   │                  UNIFIED STRATEGY PROTOCOL                      │    │  │
│  │   │                                                                 │    │  │
│  │   │   class Strategy(Protocol):                                     │    │  │
│  │   │       async def on_data(self, data: MarketData) -> Signal       │    │  │
│  │   │       async def backtest(self, data: DataFrame) -> Report       │    │  │
│  │   │       async def optimize(self, params: dict) -> BestParams      │    │  │
│  │   └─────────────────────────────────────────────────────────────────┘    │  │
│  │                                   │                                       │  │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │  │
│  │   │Freqtrade│ │Hummingbot│ │  Jesse  │ │  FinRL  │ │ Custom  │           │  │
│  │   │ Adapter │ │ Adapter │ │ Adapter │ │ Adapter │ │ Adapter │           │  │
│  │   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│  ┌───────────────────────────────────┼───────────────────────────────────────┐  │
│  │                    ALGORITHM FRAMEWORK LAYER                              │  │
│  │                 (Inspired by QuantConnect Lean)                           │  │
│  │                                                                           │  │
│  │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   │  │
│  │   │Universe │ → │  Alpha  │ → │Portfolio│ → │  Risk   │ → │Execution│   │  │
│  │   │Selection│   │  Model  │   │ Constr. │   │  Mgmt   │   │  Model  │   │  │
│  │   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│  ┌───────────────────────────────────┼───────────────────────────────────────┐  │
│  │                       GATEWAY LAYER                                       │  │
│  │              (Unified Broker/Exchange Interface)                          │  │
│  │                                                                           │  │
│  │   class Gateway(Protocol):                                                │  │
│  │       async def connect(config: dict) -> None                             │  │
│  │       async def subscribe(symbols: list[str]) -> None                     │  │
│  │       async def order(order: Order) -> OrderResult                        │  │
│  │       async def cancel(order_id: str) -> bool                             │  │
│  │       async def positions() -> list[Position]                             │  │
│  │       async def balance() -> dict[str, Decimal]                           │  │
│  │                                                                           │  │
│  │   ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐           │  │
│  │   │Binance│ │ Bybit │ │Alpaca │ │Hyper- │ │Deribit│ │Kalshi │           │  │
│  │   │Gateway│ │Gateway│ │Gateway│ │liquid │ │Gateway│ │Gateway│           │  │
│  │   └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘           │  │
│  │                                                                           │  │
│  │   Built on: CCXT (100+ exchanges) + Custom adapters                      │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│  ┌───────────────────────────────────┼───────────────────────────────────────┐  │
│  │                         DATA LAYER                                        │  │
│  │                                                                           │  │
│  │   ┌─────────────────────────┐  ┌─────────────────────────┐               │  │
│  │   │      MARKET DATA        │  │    ALTERNATIVE DATA     │               │  │
│  │   │                         │  │                         │               │  │
│  │   │ • CCXT WebSocket        │  │ • News APIs (NewsAPI)   │               │  │
│  │   │ • OpenBB (stocks)       │  │ • Sentiment (Twitter)   │               │  │
│  │   │ • Polygon.io            │  │ • On-chain (Dune)       │               │  │
│  │   │ • Databento (inst.)     │  │ • Options flow          │               │  │
│  │   └─────────────────────────┘  └─────────────────────────┘               │  │
│  │                                                                           │  │
│  │   ┌─────────────────────────────────────────────────────────────────┐    │  │
│  │   │                    TIME-SERIES DATABASE                         │    │  │
│  │   │                      (QuestDB / TimescaleDB)                    │    │  │
│  │   │                                                                 │    │  │
│  │   │   • OHLCV candles            • Order book snapshots            │    │  │
│  │   │   • Tick data                • Trade history                    │    │  │
│  │   │   • Nanosecond precision     • ASOF joins for backtesting      │    │  │
│  │   └─────────────────────────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│  ┌───────────────────────────────────┼───────────────────────────────────────┐  │
│  │                    STORAGE LAYER (NexusFS)                                │  │
│  │                                                                           │  │
│  │   /aquarius/                                                              │  │
│  │   ├── strategies/           # Strategy code & configs                    │  │
│  │   │   ├── freqtrade/                                                     │  │
│  │   │   ├── hummingbot/                                                    │  │
│  │   │   └── custom/                                                        │  │
│  │   ├── agents/               # Agent definitions & state                  │  │
│  │   ├── results/              # Backtest & live results                    │  │
│  │   ├── positions/            # Current positions                          │  │
│  │   ├── history/              # Trade history                              │  │
│  │   └── configs/              # Gateway & strategy configs                 │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Core Language** | Python 3.12+ | Ecosystem compatibility (CCXT, TA-Lib, ML libs) |
| **Performance-Critical** | Rust (via PyO3) | Hot paths like order matching, indicator calc |
| **Message Bus** | In-process async + Redis Pub/Sub | Low latency for single node, scale with Redis |
| **Database** | QuestDB (time-series) + PostgreSQL (relational) | Financial data needs specialized TSDB |
| **Agent Framework** | LangGraph | Proven multi-agent orchestration, integrates with Nexus |
| **TUI Framework** | Textual | Modern Python TUI, async-native |

### 3.3 Hybrid Rust/Python Architecture (ADR-001)

Based on analysis of NautilusTrader (17k stars) and performance benchmarks, we adopt a **hybrid architecture**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PERFORMANCE ANALYSIS                                   │
│                                                                                 │
│  Metric                    │ Pure Python    │ Rust Core      │ Improvement     │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  Quote processing          │ 250-500 μs     │ 12 μs          │ 20-80x faster   │
│  Trade message processing  │ ~400 μs        │ 6 μs           │ ~65x faster     │
│  Cryptographic signing     │ 45 ms          │ 0.05 ms        │ 900x faster     │
│  P99 latency              │ 3ms (GC spikes)│ Predictable    │ Critical        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Components by Language

**Rust Core (`libra-core` crate):**
```
├── Message Bus dispatch loop (target: <1ms)
├── Order matching engine
├── Risk check engine (pre-trade validation)
├── Technical indicator calculations (EMA, RSI, etc.)
├── WebSocket frame parsing
├── Cryptographic signing (ed25519, secp256k1)
└── Time-series data structures
```

**Python Layer:**
```
├── Strategy definitions & adapters (Freqtrade, Hummingbot)
├── LLM Agent orchestration (LangGraph)
├── TUI application (Textual)
├── Configuration management
├── Gateway wrappers (CCXT)
├── Backtesting orchestration
└── Natural language interface
```

#### Why This Split?

1. **GIL Problem**: Python's Global Interpreter Lock prevents true concurrency
2. **GC Latency**: Python's garbage collector causes unpredictable 3ms+ spikes
3. **Ecosystem**: CCXT, pandas, LangGraph, Textual are Python-native
4. **Development Velocity**: Python is 3-5x faster for prototyping
5. **Proven Pattern**: NautilusTrader uses identical architecture

#### Integration via PyO3

```python
# Python side - seamless integration
from libra_core import MessageBus, RiskEngine, indicators

bus = MessageBus()
risk = RiskEngine(limits)

# Rust executes in microseconds, returns to Python
result = risk.check_order(order, positions)
ema_values = indicators.ema(prices, period=20)
```

### 3.4 Database Selection (ADR-002)

**Decision: QuestDB over TimescaleDB**

| Feature | QuestDB | TimescaleDB |
|---------|---------|-------------|
| Ingestion rate | 4-11.4M rows/sec | 620K-1.2M rows/sec |
| ASOF JOIN (backtesting) | **Yes** | No |
| High cardinality scaling | Excellent | Degrades |
| Financial optimization | Native | Generic |

ASOF JOIN is critical for point-in-time accurate backtesting (prevents look-ahead bias).

---

## 4. Core Components

### 4.1 Message Bus

The message bus is the nervous system of Aquarius—all components communicate via async events.

```python
# aquarius/core/message_bus.py

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Coroutine
from collections import defaultdict
import asyncio

class EventType(Enum):
    # Market Data Events
    TICK = auto()
    BAR = auto()
    ORDER_BOOK = auto()
    TRADE = auto()

    # Signal Events
    SIGNAL = auto()
    ALPHA = auto()

    # Order Events
    ORDER_NEW = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    ORDER_REJECTED = auto()

    # Position Events
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_UPDATED = auto()

    # Risk Events
    RISK_LIMIT_BREACH = auto()
    DRAWDOWN_WARNING = auto()
    CIRCUIT_BREAKER = auto()

    # System Events
    GATEWAY_CONNECTED = auto()
    GATEWAY_DISCONNECTED = auto()
    STRATEGY_STARTED = auto()
    STRATEGY_STOPPED = auto()

@dataclass
class Event:
    """Base event class."""
    type: EventType
    timestamp: datetime
    source: str
    data: dict[str, Any]

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class MessageBus:
    """
    Async event-driven message bus.

    Inspired by Nautilus Trader's design:
    - All components communicate via events
    - Decoupled producers and consumers
    - Supports both sync and async handlers
    """

    def __init__(self):
        self._handlers: dict[EventType, list[Callable]] = defaultdict(list)
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], Coroutine]
    ) -> None:
        """Subscribe to an event type."""
        self._handlers[event_type].append(handler)

    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable
    ) -> None:
        """Unsubscribe from an event type."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        await self._queue.put(event)

    async def start(self) -> None:
        """Start processing events."""
        self._running = True
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=0.1
                )
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        """Stop processing events."""
        self._running = False

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all handlers."""
        handlers = self._handlers.get(event.type, [])
        if handlers:
            await asyncio.gather(
                *[handler(event) for handler in handlers],
                return_exceptions=True
            )
```

### 4.2 Gateway Protocol

The Gateway protocol provides a unified interface for all brokers/exchanges.

```python
# aquarius/gateways/protocol.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import AsyncIterator, Protocol, runtime_checkable

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Universal order representation."""
    id: str | None
    symbol: str
    side: OrderSide
    type: OrderType
    amount: Decimal
    price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: str = "GTC"
    reduce_only: bool = False
    post_only: bool = False
    client_order_id: str | None = None

@dataclass
class OrderResult:
    """Result of order submission."""
    order_id: str
    status: OrderStatus
    filled_amount: Decimal
    average_price: Decimal | None
    fee: Decimal
    fee_currency: str
    timestamp: datetime

@dataclass
class Position:
    """Current position."""
    symbol: str
    side: OrderSide
    amount: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal
    liquidation_price: Decimal | None = None

@dataclass
class Tick:
    """Market tick data."""
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: Decimal
    timestamp: datetime

@runtime_checkable
class Gateway(Protocol):
    """
    Unified gateway interface for all brokers/exchanges.

    Inspired by:
    - vnpy's BaseGateway
    - Nautilus Trader's adapters
    - Hummingbot's connectors

    Any broker/exchange can be integrated by implementing this protocol.
    """

    @property
    def name(self) -> str:
        """Gateway identifier."""
        ...

    @property
    def is_connected(self) -> bool:
        """Connection status."""
        ...

    # Connection
    async def connect(self, config: dict) -> None:
        """Connect to the exchange/broker."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the exchange/broker."""
        ...

    # Market Data
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to market data for symbols."""
        ...

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from market data."""
        ...

    def stream_ticks(self) -> AsyncIterator[Tick]:
        """Stream real-time tick data."""
        ...

    async def get_orderbook(
        self,
        symbol: str,
        depth: int = 20
    ) -> dict:
        """Get current order book."""
        ...

    # Trading
    async def order(self, order: Order) -> OrderResult:
        """Submit an order."""
        ...

    async def cancel(self, order_id: str) -> bool:
        """Cancel an order."""
        ...

    async def cancel_all(self, symbol: str | None = None) -> int:
        """Cancel all orders, optionally for a specific symbol."""
        ...

    # Account
    async def positions(self) -> list[Position]:
        """Get all open positions."""
        ...

    async def balance(self) -> dict[str, Decimal]:
        """Get account balances."""
        ...

    async def orders(self, symbol: str | None = None) -> list[Order]:
        """Get open orders."""
        ...
```

### 4.3 Strategy Protocol

The Strategy protocol defines how trading strategies integrate with Aquarius.

```python
# aquarius/strategies/protocol.py

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Protocol, runtime_checkable
import pandas as pd

class SignalType(Enum):
    LONG = "long"
    SHORT = "short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"

@dataclass
class Signal:
    """Trading signal from a strategy."""
    symbol: str
    type: SignalType
    strength: float  # 0.0 to 1.0
    price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    metadata: dict | None = None
    timestamp: datetime | None = None

@dataclass
class BacktestResult:
    """Result of a backtest run."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    trades: pd.DataFrame
    equity_curve: pd.DataFrame

@runtime_checkable
class Strategy(Protocol):
    """
    Unified strategy interface.

    All strategy adapters (Freqtrade, Hummingbot, etc.) implement this.
    Same code runs for backtest and live trading.
    """

    @property
    def name(self) -> str:
        """Strategy name."""
        ...

    @property
    def symbols(self) -> list[str]:
        """Symbols this strategy trades."""
        ...

    async def initialize(self, config: dict) -> None:
        """Initialize strategy with configuration."""
        ...

    async def on_data(self, data: pd.DataFrame) -> Signal | None:
        """
        Process new market data and generate signal.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Signal if action needed, None otherwise
        """
        ...

    async def on_tick(self, tick: Tick) -> Signal | None:
        """
        Process real-time tick (for HFT strategies).

        Args:
            tick: Real-time tick data

        Returns:
            Signal if action needed, None otherwise
        """
        ...

    async def on_fill(self, order_result: OrderResult) -> None:
        """Called when an order is filled."""
        ...

    async def backtest(
        self,
        data: pd.DataFrame,
        initial_capital: Decimal = Decimal("10000"),
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: Historical OHLCV data
            initial_capital: Starting capital

        Returns:
            Backtest results
        """
        ...

    async def optimize(
        self,
        data: pd.DataFrame,
        param_space: dict,
        metric: str = "sharpe_ratio",
    ) -> dict:
        """
        Optimize strategy parameters.

        Args:
            data: Historical data for optimization
            param_space: Parameter search space
            metric: Metric to optimize for

        Returns:
            Best parameters found
        """
        ...
```

### 4.4 Risk Manager

```python
# aquarius/risk/manager.py

from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Callable

class RiskAction(Enum):
    ALLOW = auto()
    REDUCE_SIZE = auto()
    REJECT = auto()
    EMERGENCY_CLOSE = auto()

@dataclass
class RiskLimits:
    """Risk limits configuration."""
    # Position limits
    max_position_size: Decimal = Decimal("0.1")  # 10% of capital per position
    max_positions: int = 10
    max_sector_exposure: Decimal = Decimal("0.3")  # 30% in one sector

    # Loss limits
    max_daily_loss: Decimal = Decimal("0.02")  # 2% daily
    max_weekly_loss: Decimal = Decimal("0.05")  # 5% weekly
    max_drawdown: Decimal = Decimal("0.10")  # 10% max drawdown

    # Order limits
    max_order_size: Decimal = Decimal("0.05")  # 5% per order
    max_orders_per_minute: int = 10

    # Circuit breakers
    volatility_threshold: float = 3.0  # 3x normal volatility
    correlation_threshold: float = 0.8  # Pause if positions too correlated

@dataclass
class RiskCheck:
    """Result of a risk check."""
    action: RiskAction
    reason: str | None = None
    adjusted_size: Decimal | None = None

class RiskManager:
    """
    Centralized risk management.

    Features:
    - Pre-trade risk checks
    - Position limits
    - Drawdown monitoring
    - Circuit breakers
    - Correlation checks
    """

    def __init__(
        self,
        limits: RiskLimits,
        message_bus: MessageBus,
    ):
        self.limits = limits
        self.bus = message_bus
        self._daily_pnl = Decimal("0")
        self._weekly_pnl = Decimal("0")
        self._peak_equity = Decimal("0")
        self._current_equity = Decimal("0")
        self._order_timestamps: list[datetime] = []
        self._circuit_breaker_active = False

    async def check_order(
        self,
        order: Order,
        current_positions: list[Position],
        balance: dict[str, Decimal],
    ) -> RiskCheck:
        """
        Pre-trade risk check.

        Args:
            order: Proposed order
            current_positions: Current positions
            balance: Account balances

        Returns:
            RiskCheck with action to take
        """
        # Circuit breaker check
        if self._circuit_breaker_active:
            return RiskCheck(
                action=RiskAction.REJECT,
                reason="Circuit breaker active"
            )

        # Rate limit check
        if not self._check_rate_limit():
            return RiskCheck(
                action=RiskAction.REJECT,
                reason="Order rate limit exceeded"
            )

        # Position count check
        if len(current_positions) >= self.limits.max_positions:
            existing = any(p.symbol == order.symbol for p in current_positions)
            if not existing:
                return RiskCheck(
                    action=RiskAction.REJECT,
                    reason=f"Max positions ({self.limits.max_positions}) reached"
                )

        # Position size check
        total_equity = sum(balance.values())
        order_value = order.amount * (order.price or Decimal("0"))

        if order_value / total_equity > self.limits.max_order_size:
            adjusted_size = total_equity * self.limits.max_order_size
            return RiskCheck(
                action=RiskAction.REDUCE_SIZE,
                reason="Order exceeds max size",
                adjusted_size=adjusted_size / (order.price or Decimal("1"))
            )

        # Drawdown check
        current_drawdown = self._calculate_drawdown()
        if current_drawdown > self.limits.max_drawdown:
            return RiskCheck(
                action=RiskAction.REJECT,
                reason=f"Max drawdown ({self.limits.max_drawdown:.1%}) exceeded"
            )

        # Daily loss check
        if abs(self._daily_pnl) / total_equity > self.limits.max_daily_loss:
            return RiskCheck(
                action=RiskAction.REJECT,
                reason=f"Daily loss limit ({self.limits.max_daily_loss:.1%}) reached"
            )

        return RiskCheck(action=RiskAction.ALLOW)

    async def trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger emergency circuit breaker."""
        self._circuit_breaker_active = True
        await self.bus.publish(Event(
            type=EventType.CIRCUIT_BREAKER,
            timestamp=datetime.utcnow(),
            source="risk_manager",
            data={"reason": reason, "action": "activated"}
        ))

    async def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (manual action required)."""
        self._circuit_breaker_active = False
        await self.bus.publish(Event(
            type=EventType.CIRCUIT_BREAKER,
            timestamp=datetime.utcnow(),
            source="risk_manager",
            data={"action": "deactivated"}
        ))

    def _check_rate_limit(self) -> bool:
        """Check order rate limit."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        self._order_timestamps = [
            ts for ts in self._order_timestamps if ts > minute_ago
        ]
        if len(self._order_timestamps) >= self.limits.max_orders_per_minute:
            return False
        self._order_timestamps.append(now)
        return True

    def _calculate_drawdown(self) -> Decimal:
        """Calculate current drawdown from peak."""
        if self._peak_equity == 0:
            return Decimal("0")
        return (self._peak_equity - self._current_equity) / self._peak_equity
```

---

## 5. Integration Strategy

### 5.1 Framework Adapters

Each external framework becomes a Nexus plugin implementing the Strategy protocol.

#### 5.1.1 Freqtrade Adapter

```python
# aquarius/plugins/freqtrade_adapter/plugin.py

from nexus.plugins.base import NexusPlugin, PluginMetadata
from aquarius.strategies.protocol import Strategy, Signal, BacktestResult
import pandas as pd

class FreqtradeAdapter(NexusPlugin, Strategy):
    """
    Adapter for Freqtrade strategies.

    Enables running any Freqtrade strategy in Aquarius:
    - Import existing strategies
    - Use FreqAI for ML
    - Access hyperopt optimization
    """

    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="freqtrade-adapter",
            version="0.1.0",
            description="Run Freqtrade strategies in Aquarius",
            author="Aquarius",
            requires=["freqtrade>=2024.1"],
        )

    def commands(self) -> dict:
        return {
            "list": self.list_strategies,
            "backtest": self.run_backtest,
            "import": self.import_strategy,
        }

    async def initialize(self, config: dict) -> None:
        """Load Freqtrade configuration and strategy."""
        from freqtrade.configuration import Configuration
        from freqtrade.resolvers import StrategyResolver

        self._ft_config = Configuration.from_files([config["config_path"]])
        self._strategy = StrategyResolver.load_strategy(self._ft_config)

    async def on_data(self, data: pd.DataFrame) -> Signal | None:
        """Run Freqtrade strategy on new data."""
        # Add indicators
        df = self._strategy.advise_indicators(data.copy(), {"pair": self.symbols[0]})

        # Get entry signal
        df = self._strategy.advise_entry(df, {"pair": self.symbols[0]})

        # Get exit signal
        df = self._strategy.advise_exit(df, {"pair": self.symbols[0]})

        # Convert to Aquarius Signal
        last_row = df.iloc[-1]

        if last_row.get("enter_long", 0) == 1:
            return Signal(
                symbol=self.symbols[0],
                type=SignalType.LONG,
                strength=1.0,
            )
        elif last_row.get("exit_long", 0) == 1:
            return Signal(
                symbol=self.symbols[0],
                type=SignalType.CLOSE_LONG,
                strength=1.0,
            )

        return None

    async def backtest(
        self,
        data: pd.DataFrame,
        initial_capital: Decimal = Decimal("10000"),
    ) -> BacktestResult:
        """Run Freqtrade backtesting engine."""
        from freqtrade.optimize.backtesting import Backtesting

        bt = Backtesting(self._ft_config)
        results = bt.backtest(
            processed=data,
            start_date=data.index[0],
            end_date=data.index[-1],
        )

        # Convert to Aquarius BacktestResult
        return self._convert_results(results, initial_capital)

    async def list_strategies(self) -> list[str]:
        """List available Freqtrade strategies."""
        from freqtrade.resolvers import StrategyResolver
        return StrategyResolver.search_all_objects({}, enum_failed=False)
```

#### 5.1.2 Hummingbot Adapter

```python
# aquarius/plugins/hummingbot_adapter/plugin.py

class HummingbotAdapter(NexusPlugin, Strategy):
    """
    Adapter for Hummingbot market making strategies.

    Enables:
    - Pure market making
    - Avellaneda-Stoikov
    - Cross-exchange market making
    - Arbitrage strategies
    """

    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="hummingbot-adapter",
            version="0.1.0",
            description="Run Hummingbot strategies in Aquarius",
            author="Aquarius",
            requires=["hummingbot>=2.0"],
        )

    def commands(self) -> dict:
        return {
            "mm": self.run_market_making,
            "arb": self.run_arbitrage,
            "configure": self.configure_strategy,
        }

    async def run_market_making(
        self,
        exchange: str,
        trading_pair: str,
        bid_spread: float = 0.001,
        ask_spread: float = 0.001,
        order_amount: float = 0.1,
        strategy: str = "avellaneda",
    ):
        """Run market making strategy."""
        if strategy == "avellaneda":
            from hummingbot.strategy.avellaneda_market_making import (
                AvellanedaMarketMakingStrategy
            )
            self._hb_strategy = AvellanedaMarketMakingStrategy(
                exchange=exchange,
                trading_pair=trading_pair,
                bid_spread=bid_spread,
                ask_spread=ask_spread,
                order_amount=order_amount,
            )

        # Run strategy loop
        await self._run_strategy_loop()
```

#### 5.1.3 TradingAgents Integration

```python
# aquarius/agents/trading_agents.py

from langgraph.graph import StateGraph, END
from typing import TypedDict

class TradingState(TypedDict):
    """State passed between agents."""
    symbol: str
    date: str
    fundamental_analysis: str | None
    sentiment_analysis: str | None
    technical_analysis: str | None
    news_analysis: str | None
    bull_case: str | None
    bear_case: str | None
    debate_result: str | None
    risk_assessment: str | None
    final_decision: dict | None

class TradingAgentsOrchestrator:
    """
    Multi-agent trading analysis using TradingAgents pattern.

    Agents:
    - Fundamentals Analyst: Financial metrics
    - Sentiment Analyst: Social/market mood
    - Technical Analyst: Chart patterns, indicators
    - News Analyst: Current events
    - Bull Researcher: Bullish case
    - Bear Researcher: Bearish case
    - Risk Manager: Risk assessment
    - Trader: Final decision
    """

    def __init__(self, llm_provider):
        self.llm = llm_provider
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the multi-agent graph."""
        workflow = StateGraph(TradingState)

        # Add analyst nodes (parallel execution)
        workflow.add_node("fundamentals", self._fundamentals_agent)
        workflow.add_node("sentiment", self._sentiment_agent)
        workflow.add_node("technical", self._technical_agent)
        workflow.add_node("news", self._news_agent)

        # Add debate nodes
        workflow.add_node("bull", self._bull_researcher)
        workflow.add_node("bear", self._bear_researcher)
        workflow.add_node("debate", self._debate_moderator)

        # Add decision nodes
        workflow.add_node("risk", self._risk_manager)
        workflow.add_node("trader", self._trader_agent)

        # Define flow
        workflow.set_entry_point("fundamentals")

        # Analysts run in parallel (fan-out)
        workflow.add_edge("fundamentals", "sentiment")
        workflow.add_edge("sentiment", "technical")
        workflow.add_edge("technical", "news")

        # Researchers analyze after analysts
        workflow.add_edge("news", "bull")
        workflow.add_edge("bull", "bear")
        workflow.add_edge("bear", "debate")

        # Final decision chain
        workflow.add_edge("debate", "risk")
        workflow.add_conditional_edges(
            "risk",
            self._should_trade,
            {True: "trader", False: END}
        )
        workflow.add_edge("trader", END)

        return workflow.compile()

    async def analyze(self, symbol: str, date: str) -> dict:
        """Run full multi-agent analysis."""
        initial_state = TradingState(
            symbol=symbol,
            date=date,
            fundamental_analysis=None,
            sentiment_analysis=None,
            technical_analysis=None,
            news_analysis=None,
            bull_case=None,
            bear_case=None,
            debate_result=None,
            risk_assessment=None,
            final_decision=None,
        )

        final_state = await self.graph.ainvoke(initial_state)
        return final_state["final_decision"]

    async def _fundamentals_agent(self, state: TradingState) -> dict:
        """Analyze fundamental metrics."""
        prompt = f"""
        Analyze the fundamental metrics for {state['symbol']} as of {state['date']}.

        Consider:
        - Revenue growth
        - Profit margins
        - P/E ratio vs sector
        - Balance sheet health
        - Cash flow

        Provide a concise fundamental analysis.
        """
        analysis = await self.llm.generate(prompt)
        return {"fundamental_analysis": analysis}

    async def _debate_moderator(self, state: TradingState) -> dict:
        """Moderate bull vs bear debate."""
        prompt = f"""
        You are moderating a debate between bullish and bearish analysts.

        BULL CASE:
        {state['bull_case']}

        BEAR CASE:
        {state['bear_case']}

        Synthesize both arguments and provide a balanced conclusion.
        Who has the stronger case? What is the consensus view?
        """
        result = await self.llm.generate(prompt)
        return {"debate_result": result}

    def _should_trade(self, state: TradingState) -> bool:
        """Determine if we should proceed with trade."""
        # Parse risk assessment
        risk = state.get("risk_assessment", "")
        return "PROCEED" in risk.upper()
```

---

## 6. Data Architecture

### 6.1 Time-Series Database

```python
# aquarius/data/tsdb.py

from datetime import datetime
from decimal import Decimal
import pandas as pd

class TimeSeriesDB:
    """
    Time-series database interface.

    Uses QuestDB for:
    - Nanosecond precision timestamps
    - Fast OHLCV queries
    - ASOF joins for backtesting
    - Efficient tick data storage
    """

    def __init__(self, connection_string: str):
        self.conn = connection_string

    async def store_ticks(
        self,
        symbol: str,
        ticks: list[Tick]
    ) -> None:
        """Store tick data."""
        # INSERT INTO ticks (symbol, timestamp, bid, ask, last, volume)
        # VALUES ($1, $2, $3, $4, $5, $6)
        pass

    async def store_bars(
        self,
        symbol: str,
        timeframe: str,
        bars: pd.DataFrame,
    ) -> None:
        """Store OHLCV bars."""
        pass

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Retrieve OHLCV bars."""
        pass

    async def asof_join(
        self,
        left_table: str,
        right_table: str,
        on_column: str,
    ) -> pd.DataFrame:
        """
        ASOF join for point-in-time accurate backtesting.

        Prevents look-ahead bias by joining on timestamp
        with the most recent available data.
        """
        pass
```

### 6.2 Data Feed Manager

```python
# aquarius/data/feed_manager.py

class DataFeedManager:
    """
    Manages multiple data feeds.

    Supports:
    - Real-time WebSocket streams
    - Historical data downloads
    - Alternative data integration
    """

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self._feeds: dict[str, DataFeed] = {}

    async def add_feed(
        self,
        name: str,
        feed_type: str,
        config: dict,
    ) -> None:
        """Add a data feed."""
        if feed_type == "ccxt":
            feed = CCXTDataFeed(config)
        elif feed_type == "polygon":
            feed = PolygonDataFeed(config)
        elif feed_type == "openbb":
            feed = OpenBBDataFeed(config)
        else:
            raise ValueError(f"Unknown feed type: {feed_type}")

        self._feeds[name] = feed

    async def start_streaming(self, symbols: list[str]) -> None:
        """Start real-time data streaming."""
        for feed in self._feeds.values():
            await feed.connect()
            await feed.subscribe(symbols)

            # Forward data to message bus
            async for tick in feed.stream():
                await self.bus.publish(Event(
                    type=EventType.TICK,
                    timestamp=tick.timestamp,
                    source=feed.name,
                    data=tick.__dict__,
                ))
```

---

## 7. Agent System

### 7.1 Agent Types

```python
# aquarius/agents/definitions.py

from dataclasses import dataclass
from enum import Enum

class AgentRole(Enum):
    # Analysis agents
    DEEP_RESEARCHER = "deep_researcher"
    FUNDAMENTALS_ANALYST = "fundamentals_analyst"
    SENTIMENT_ANALYST = "sentiment_analyst"
    TECHNICAL_ANALYST = "technical_analyst"
    NEWS_ANALYST = "news_analyst"

    # Decision agents
    BULL_RESEARCHER = "bull_researcher"
    BEAR_RESEARCHER = "bear_researcher"
    RISK_MANAGER = "risk_manager"

    # Execution agents
    TRADER = "trader"
    PORTFOLIO_MANAGER = "portfolio_manager"

    # Utility agents
    COORDINATOR = "coordinator"
    MONITOR = "monitor"

@dataclass
class AgentDefinition:
    """Definition for a trading agent."""
    id: str
    role: AgentRole
    display_name: str
    description: str

    # LLM configuration
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3

    # Permissions
    can_execute_trades: bool = False
    max_position_size: Decimal | None = None
    allowed_symbols: list[str] | None = None

    # Skills
    skills: list[str] = None  # e.g., ["backtest", "analyze", "optimize"]

    # Prompts
    system_prompt: str | None = None

# Predefined agent configurations
TRADING_AGENTS = {
    AgentRole.DEEP_RESEARCHER: AgentDefinition(
        id="agent_deep_researcher",
        role=AgentRole.DEEP_RESEARCHER,
        display_name="Deep Researcher",
        description="Conducts comprehensive research across 100+ sources",
        model="claude-sonnet-4-20250514",
        temperature=0.2,
        skills=["web_search", "document_analysis", "report_generation"],
        system_prompt="""You are a deep research agent for financial analysis.
        Your job is to:
        1. Search multiple sources (news, filings, social media, research)
        2. Synthesize information into comprehensive reports
        3. Cite all sources
        4. Highlight key risks and opportunities
        """,
    ),

    AgentRole.TRADER: AgentDefinition(
        id="agent_trader",
        role=AgentRole.TRADER,
        display_name="Trader",
        description="Executes trades based on analysis and signals",
        model="claude-sonnet-4-20250514",
        temperature=0.1,  # Low temperature for consistency
        can_execute_trades=True,
        skills=["order_execution", "position_management"],
        system_prompt="""You are a trading execution agent.
        Your job is to:
        1. Receive trading signals from analysis agents
        2. Validate signals against risk parameters
        3. Execute orders with optimal timing
        4. Manage open positions
        5. Report all actions taken
        """,
    ),

    AgentRole.RISK_MANAGER: AgentDefinition(
        id="agent_risk_manager",
        role=AgentRole.RISK_MANAGER,
        display_name="Risk Manager",
        description="Monitors and enforces risk limits",
        model="claude-sonnet-4-20250514",
        temperature=0.1,
        skills=["risk_analysis", "position_monitoring"],
        system_prompt="""You are a risk management agent.
        Your job is to:
        1. Monitor all open positions
        2. Calculate portfolio risk metrics (VaR, drawdown, correlation)
        3. Approve or reject proposed trades based on risk limits
        4. Trigger alerts and circuit breakers when needed
        5. Never allow trades that exceed defined risk parameters
        """,
    ),
}
```

### 7.2 Agent Registration with Nexus

```python
# aquarius/agents/registration.py

from nexus.core.agents import register_agent
from nexus.core.entity_registry import EntityRegistry

async def register_trading_agents(
    user_id: str,
    entity_registry: EntityRegistry,
) -> dict[str, dict]:
    """Register all trading agents for a user."""

    registered = {}

    for role, definition in TRADING_AGENTS.items():
        agent = register_agent(
            user_id=user_id,
            agent_id=definition.id,
            name=definition.display_name,
            metadata={
                "role": role.value,
                "description": definition.description,
                "model": definition.model,
                "can_execute_trades": definition.can_execute_trades,
                "skills": definition.skills,
            },
            entity_registry=entity_registry,
        )
        registered[role.value] = agent

    return registered
```

---

## 8. Risk Management

### 8.1 Risk Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                     RISK MANAGEMENT FRAMEWORK                   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   PRE-TRADE CHECKS                      │    │
│  │                                                         │    │
│  │  • Position size limits (% of portfolio)               │    │
│  │  • Order rate limits (orders/minute)                   │    │
│  │  • Sector/asset concentration limits                   │    │
│  │  • Correlation checks (avoid concentrated risk)        │    │
│  │  • Margin/leverage limits                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  REAL-TIME MONITORING                   │    │
│  │                                                         │    │
│  │  • P&L tracking (daily, weekly, total)                 │    │
│  │  • Drawdown monitoring                                 │    │
│  │  • Volatility tracking                                 │    │
│  │  • Position delta monitoring                           │    │
│  │  • Liquidity risk assessment                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   CIRCUIT BREAKERS                      │    │
│  │                                                         │    │
│  │  Trigger Conditions:                                   │    │
│  │  • Daily loss > 2%  → Pause new trades                 │    │
│  │  • Drawdown > 10%   → Close all positions              │    │
│  │  • Volatility > 3σ  → Reduce position sizes            │    │
│  │  • API errors > 3   → Switch to paper mode             │    │
│  │                                                         │    │
│  │  Recovery:                                              │    │
│  │  • Manual reset required for severe breaches           │    │
│  │  • Auto-resume after cooling period for minor          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    AUDIT & LOGGING                      │    │
│  │                                                         │    │
│  │  • Every order logged with full context                │    │
│  │  • Agent reasoning captured                            │    │
│  │  • Risk decisions documented                           │    │
│  │  • Compliance-ready audit trail                        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Position Sizing

```python
# aquarius/risk/position_sizing.py

from decimal import Decimal
from enum import Enum

class SizingMethod(Enum):
    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENT = "fixed_percent"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"

class PositionSizer:
    """
    Calculate optimal position sizes.

    Methods:
    - Fixed: Always same dollar amount
    - Percent: Fixed % of portfolio
    - Volatility-adjusted: Smaller when volatile
    - Kelly: Optimal based on win rate and payoff
    """

    def __init__(
        self,
        method: SizingMethod = SizingMethod.FIXED_PERCENT,
        base_risk: Decimal = Decimal("0.01"),  # 1% risk per trade
    ):
        self.method = method
        self.base_risk = base_risk

    def calculate(
        self,
        portfolio_value: Decimal,
        entry_price: Decimal,
        stop_loss: Decimal,
        volatility: float | None = None,
        win_rate: float | None = None,
        avg_win: Decimal | None = None,
        avg_loss: Decimal | None = None,
    ) -> Decimal:
        """Calculate position size."""

        if self.method == SizingMethod.FIXED_PERCENT:
            return self._fixed_percent(portfolio_value)

        elif self.method == SizingMethod.VOLATILITY_ADJUSTED:
            return self._volatility_adjusted(
                portfolio_value, volatility or 0.02
            )

        elif self.method == SizingMethod.KELLY_CRITERION:
            return self._kelly(
                portfolio_value,
                win_rate or 0.5,
                avg_win or Decimal("100"),
                avg_loss or Decimal("100"),
            )

        else:  # FIXED_AMOUNT
            return portfolio_value * self.base_risk

    def _fixed_percent(self, portfolio_value: Decimal) -> Decimal:
        """Fixed percentage of portfolio."""
        return portfolio_value * self.base_risk

    def _volatility_adjusted(
        self,
        portfolio_value: Decimal,
        volatility: float,
    ) -> Decimal:
        """Reduce size when volatility is high."""
        base_size = portfolio_value * self.base_risk
        vol_multiplier = Decimal(str(0.02 / max(volatility, 0.01)))  # Target 2% vol
        return base_size * min(vol_multiplier, Decimal("2"))  # Cap at 2x

    def _kelly(
        self,
        portfolio_value: Decimal,
        win_rate: float,
        avg_win: Decimal,
        avg_loss: Decimal,
    ) -> Decimal:
        """Kelly criterion optimal sizing."""
        if avg_loss == 0:
            return Decimal("0")

        b = float(avg_win / avg_loss)  # Win/loss ratio
        p = win_rate
        q = 1 - p

        kelly_pct = (b * p - q) / b

        # Use fractional Kelly (half) for safety
        kelly_pct = max(0, kelly_pct * 0.5)

        return portfolio_value * Decimal(str(kelly_pct))
```

---

## 9. Execution Plan

### 9.1 Phase Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      EXECUTION ROADMAP                          │
│                                                                 │
│  PHASE 1: Foundation                                            │
│  ══════════════════                                             │
│  • Core infrastructure (message bus, gateway protocol)          │
│  • CCXT gateway implementation                                  │
│  • Basic TUI shell                                             │
│  • Risk manager skeleton                                        │
│                                                                 │
│  PHASE 2: First Integration                                     │
│  ═════════════════════════                                      │
│  • Freqtrade adapter                                           │
│  • Basic backtesting engine                                    │
│  • Paper trading mode                                          │
│  • TUI dashboard widgets                                       │
│                                                                 │
│  PHASE 3: Agent Layer                                           │
│  ════════════════════                                           │
│  • TradingAgents integration                                   │
│  • LangGraph orchestration                                     │
│  • Deep research agent                                         │
│  • Natural language interface                                  │
│                                                                 │
│  PHASE 4: Advanced Strategies                                   │
│  ════════════════════════════                                   │
│  • Hummingbot adapter (market making)                          │
│  • Funding rate arbitrage                                      │
│  • Options/derivatives support                                 │
│  • FinRL integration (RL strategies)                           │
│                                                                 │
│  PHASE 5: Production Hardening                                  │
│  ═════════════════════════════                                  │
│  • Full risk management                                        │
│  • Audit logging                                               │
│  • Multi-strategy orchestration                                │
│  • Performance optimization                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Phase 1: Foundation (Weeks 1-2)

#### Deliverables

| Component | Description | Files |
|-----------|-------------|-------|
| Message Bus | Event-driven core | `aquarius/core/message_bus.py` |
| Gateway Protocol | Unified broker interface | `aquarius/gateways/protocol.py` |
| CCXT Gateway | 100+ exchange support | `aquarius/gateways/ccxt_gateway.py` |
| Strategy Protocol | Strategy interface | `aquarius/strategies/protocol.py` |
| Risk Manager | Basic risk checks | `aquarius/risk/manager.py` |
| TUI Shell | Basic Textual app | `aquarius/tui/app.py` |

#### Task Breakdown

```
Week 1:
├── Day 1-2: Message bus implementation
│   ├── Event types definition
│   ├── Async queue processing
│   ├── Handler registration
│   └── Unit tests
│
├── Day 3-4: Gateway protocol + CCXT
│   ├── Protocol definition
│   ├── CCXT wrapper
│   ├── Connection management
│   └── Order submission
│
└── Day 5: Strategy protocol
    ├── Signal types
    ├── Backtest result format
    └── Protocol methods

Week 2:
├── Day 1-2: Risk manager
│   ├── Limit configuration
│   ├── Pre-trade checks
│   ├── Drawdown monitoring
│   └── Circuit breakers
│
├── Day 3-4: TUI shell
│   ├── App skeleton
│   ├── Basic widgets
│   ├── Keyboard bindings
│   └── Status line
│
└── Day 5: Integration testing
    ├── End-to-end flow
    ├── Paper trading test
    └── Documentation
```

### 9.3 Phase 2: First Integration (Weeks 3-4)

#### Deliverables

| Component | Description | Files |
|-----------|-------------|-------|
| Freqtrade Adapter | Run FT strategies | `aquarius/plugins/freqtrade_adapter/` |
| Backtest Engine | Event-driven backtester | `aquarius/backtest/engine.py` |
| Paper Trading | Simulated execution | `aquarius/gateways/paper_gateway.py` |
| Dashboard | TUI dashboard widgets | `aquarius/tui/widgets/` |

#### Task Breakdown

```
Week 3:
├── Day 1-2: Freqtrade adapter
│   ├── Strategy loading
│   ├── Signal conversion
│   ├── Config mapping
│   └── Plugin registration
│
├── Day 3-4: Backtest engine
│   ├── Event-driven simulation
│   ├── Fill modeling
│   ├── Performance metrics
│   └── Result visualization
│
└── Day 5: Paper trading gateway
    ├── Order simulation
    ├── Position tracking
    └── P&L calculation

Week 4:
├── Day 1-2: TUI dashboard
│   ├── Portfolio widget
│   ├── Positions table
│   ├── P&L chart
│   └── Order book view
│
├── Day 3-4: Strategy browser
│   ├── List strategies
│   ├── Configure strategy
│   ├── Start/stop control
│   └── Backtest from TUI
│
└── Day 5: Testing + docs
    ├── Integration tests
    ├── User guide
    └── Example strategies
```

### 9.4 Phase 3: Agent Layer (Weeks 5-6)

#### Deliverables

| Component | Description | Files |
|-----------|-------------|-------|
| TradingAgents | Multi-agent analysis | `aquarius/agents/trading_agents.py` |
| Agent Definitions | Role configurations | `aquarius/agents/definitions.py` |
| LangGraph Workflows | Agent orchestration | `aquarius/agents/workflows/` |
| NL Interface | Natural language commands | `aquarius/nl/parser.py` |

### 9.5 Phase 4: Advanced Strategies (Weeks 7-8)

#### Deliverables

| Component | Description | Files |
|-----------|-------------|-------|
| Hummingbot Adapter | Market making | `aquarius/plugins/hummingbot_adapter/` |
| Funding Arb | Funding rate strategy | `aquarius/strategies/funding_arb.py` |
| Options Support | Greeks, hedging | `aquarius/derivatives/` |
| FinRL Adapter | RL strategies | `aquarius/plugins/finrl_adapter/` |

### 9.6 Phase 5: Production Hardening (Weeks 9-10)

#### Deliverables

| Component | Description | Files |
|-----------|-------------|-------|
| Full Risk Mgmt | Complete risk system | `aquarius/risk/` |
| Audit Logging | Compliance logging | `aquarius/audit/` |
| Multi-Strategy | Portfolio of strategies | `aquarius/portfolio/` |
| Performance | Optimization | Various |

---

## 10. Technical Specifications

### 10.1 Project Structure

```
aquarius/
├── pyproject.toml
├── README.md
├── AQUARIUS_DESIGN_DOC.md
│
├── src/aquarius/
│   ├── __init__.py
│   │
│   ├── core/                    # Core infrastructure
│   │   ├── __init__.py
│   │   ├── message_bus.py       # Event-driven message bus
│   │   ├── events.py            # Event type definitions
│   │   └── config.py            # Configuration management
│   │
│   ├── gateways/                # Broker/exchange connectors
│   │   ├── __init__.py
│   │   ├── protocol.py          # Gateway protocol definition
│   │   ├── ccxt_gateway.py      # CCXT-based gateway
│   │   ├── paper_gateway.py     # Paper trading gateway
│   │   ├── alpaca_gateway.py    # Alpaca-specific gateway
│   │   └── hyperliquid_gateway.py
│   │
│   ├── strategies/              # Strategy system
│   │   ├── __init__.py
│   │   ├── protocol.py          # Strategy protocol
│   │   ├── base.py              # Base strategy class
│   │   └── examples/            # Example strategies
│   │
│   ├── risk/                    # Risk management
│   │   ├── __init__.py
│   │   ├── manager.py           # Risk manager
│   │   ├── limits.py            # Limit definitions
│   │   ├── position_sizing.py   # Sizing algorithms
│   │   └── circuit_breaker.py   # Emergency controls
│   │
│   ├── agents/                  # AI agents
│   │   ├── __init__.py
│   │   ├── definitions.py       # Agent type definitions
│   │   ├── trading_agents.py    # TradingAgents integration
│   │   ├── registration.py      # Nexus agent registration
│   │   └── workflows/           # LangGraph workflows
│   │
│   ├── data/                    # Data management
│   │   ├── __init__.py
│   │   ├── feed_manager.py      # Data feed management
│   │   ├── tsdb.py              # Time-series database
│   │   └── providers/           # Data providers
│   │
│   ├── backtest/                # Backtesting engine
│   │   ├── __init__.py
│   │   ├── engine.py            # Backtest engine
│   │   ├── simulator.py         # Order simulation
│   │   └── metrics.py           # Performance metrics
│   │
│   ├── tui/                     # Terminal UI
│   │   ├── __init__.py
│   │   ├── app.py               # Main Textual app
│   │   ├── screens/             # TUI screens
│   │   ├── widgets/             # Custom widgets
│   │   └── styles.css           # TUI styling
│   │
│   ├── plugins/                 # Framework adapters
│   │   ├── __init__.py
│   │   ├── freqtrade_adapter/
│   │   ├── hummingbot_adapter/
│   │   ├── jesse_adapter/
│   │   └── finrl_adapter/
│   │
│   └── nl/                      # Natural language
│       ├── __init__.py
│       ├── parser.py            # Command parser
│       └── commands.py          # NL command handlers
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── configs/                     # Example configurations
│   ├── gateways/
│   ├── strategies/
│   └── agents/
│
└── docs/
    ├── getting-started.md
    ├── strategies.md
    ├── agents.md
    └── api-reference.md
```

### 10.2 Dependencies

```toml
# pyproject.toml

[project]
name = "aquarius"
version = "0.1.0"
requires-python = ">=3.12"

dependencies = [
    # Core
    "pydantic>=2.0",
    "asyncio>=3.4",
    "aiohttp>=3.9",

    # Exchange connectivity
    "ccxt>=4.0",

    # Data
    "pandas>=2.0",
    "numpy>=1.26",
    "pyarrow>=14.0",

    # Technical analysis
    "pandas-ta>=0.3",
    "ta-lib>=0.4",

    # Agent orchestration
    "langgraph>=0.1",
    "langchain>=0.2",

    # TUI
    "textual>=0.50",
    "rich>=13.0",

    # Database
    "sqlalchemy>=2.0",
    "asyncpg>=0.29",

    # Nexus integration
    "nexus-core>=0.5",
]

[project.optional-dependencies]
freqtrade = ["freqtrade>=2024.1"]
hummingbot = ["hummingbot>=2.0"]
finrl = ["finrl>=0.3", "stable-baselines3>=2.0"]
```

### 10.3 Configuration

```yaml
# configs/aquarius.yaml

# Platform settings
platform:
  name: "Aquarius Trading Platform"
  mode: "paper"  # paper | live
  log_level: "INFO"

# Default gateway
gateway:
  type: "ccxt"
  exchange: "binance"
  testnet: true

# Risk limits
risk:
  max_position_size: 0.1       # 10% of portfolio
  max_daily_loss: 0.02         # 2% daily loss limit
  max_drawdown: 0.10           # 10% max drawdown
  max_orders_per_minute: 10

# Agent configuration
agents:
  default_model: "claude-sonnet-4-20250514"
  temperature: 0.3
  enable_trading_agents: true

# Data sources
data:
  primary: "ccxt"
  alternatives:
    - type: "newsapi"
      enabled: true
    - type: "openbb"
      enabled: false

# TUI settings
tui:
  theme: "dark"
  refresh_rate: 1.0
```

---

## Appendix A: References

### Open Source Projects Analyzed

| Project | Stars | Key Learnings |
|---------|-------|---------------|
| [Nautilus Trader](https://github.com/nautechsystems/nautilus_trader) | 17k | Ports & adapters, message bus, Rust+Python |
| [QuantConnect Lean](https://github.com/QuantConnect/Lean) | 15k | Algorithm framework, handler pattern |
| [Freqtrade](https://github.com/freqtrade/freqtrade) | 46k | Strategy ecosystem, FreqAI |
| [Hummingbot](https://github.com/hummingbot/hummingbot) | 15k | Market making, DEX support |
| [vnpy](https://github.com/vnpy/vnpy) | 35k | Gateway pattern, event engine |
| [CCXT](https://github.com/ccxt/ccxt) | 40k | Unified exchange API |
| [TradingAgents](https://github.com/TauricResearch/TradingAgents) | - | Multi-agent LLM trading |

### Research Papers

1. Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*.
2. TradingAgents: Multi-Agents LLM Financial Trading Framework. arXiv:2412.20138 (2024).

### Industry Reports

1. Gartner: 80% of finance functions will embed AI-driven autonomy by 2030.
2. SEC: HFT accounts for ~55% of U.S. equity volume.

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Adapter** | Plugin that connects external framework to Aquarius |
| **Agent** | LLM-powered autonomous entity with specific role |
| **Circuit Breaker** | Emergency mechanism to halt trading |
| **Gateway** | Unified interface to broker/exchange |
| **Message Bus** | Event-driven communication system |
| **Signal** | Trading instruction from strategy |
| **TUI** | Terminal User Interface |

---

*Document Version: 1.0*
*Last Updated: January 2026*
