# Architecture Decision Records (ADRs)

This document captures key architectural decisions for the LIBRA trading platform.

---

## ADR-001: Hybrid Rust/Python Architecture

### Status
**Accepted** - January 2026

### Context
We need to build a high-performance trading platform that:
- Achieves <100ms signal-to-order latency
- Integrates with Python-native libraries (CCXT, LangGraph, Textual)
- Maintains development velocity
- Scales to high-frequency workloads

### Decision
Adopt a **hybrid Rust/Python architecture** inspired by NautilusTrader:

**Rust Core (`libra-core`):**
- Message bus dispatch loop
- Risk check engine
- Technical indicator calculations
- WebSocket frame parsing
- Cryptographic signing

**Python Layer:**
- Strategy definitions & adapters
- LLM agent orchestration
- TUI application
- Configuration management
- Gateway wrappers

### Rationale

#### Performance Benchmarks (from research)

| Metric | Python | Rust | Improvement |
|--------|--------|------|-------------|
| Quote processing | 250-500 μs | 12 μs | 20-80x |
| Trade message parsing | ~400 μs | 6 μs | ~65x |
| Cryptographic signing | 45 ms | 0.05 ms | 900x |
| P99 latency consistency | 3ms spikes | Predictable | Critical |

#### Why Not Pure Rust?
1. **Ecosystem**: CCXT, pandas, LangGraph are Python-native
2. **Development velocity**: Python is 3-5x faster for prototyping
3. **LLM integration**: All major AI frameworks are Python-first
4. **Adapter compatibility**: Freqtrade/Hummingbot are Python

#### Why Not Pure Python?
1. **GIL**: Prevents true concurrency
2. **GC pauses**: Unpredictable 3ms+ latency spikes
3. **Performance ceiling**: Can't hit <10ms targets
4. **Signing latency**: 45ms vs 0.05ms is unacceptable

### Consequences

**Positive:**
- Best of both worlds (performance + ecosystem)
- Proven pattern (NautilusTrader has 17k stars)
- Gradual migration path (start Python, add Rust)

**Negative:**
- Two languages to maintain
- Build complexity (PyO3, maturin)
- Debugging across language boundary

### References
- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader)
- [PyO3](https://github.com/PyO3/pyo3)
- [Rust in HFT](https://markrbest.github.io/hft-and-rust/)

---

## ADR-002: QuestDB for Time-Series Data

### Status
**Accepted** - January 2026

### Context
Need a time-series database for:
- Tick data storage (millions of rows/day)
- OHLCV bar storage
- Backtesting with point-in-time accuracy
- Real-time analytics

### Decision
Use **QuestDB** as the primary time-series database.

### Rationale

| Feature | QuestDB | TimescaleDB |
|---------|---------|-------------|
| Ingestion rate | 4-11.4M rows/sec | 620K-1.2M rows/sec |
| ASOF JOIN | **Yes** | No |
| High cardinality | Scales well | Degrades |
| Financial optimization | Native | Generic |

**Critical Feature: ASOF JOIN**

ASOF JOIN is essential for point-in-time accurate backtesting:
```sql
SELECT *
FROM signals
ASOF JOIN prices ON (symbol)
WHERE timestamp BETWEEN '2024-01-01' AND '2024-12-31'
```

This prevents look-ahead bias by joining each signal with the most recent price *at that moment*.

TimescaleDB does not support ASOF JOIN natively.

### Consequences

**Positive:**
- 10x faster ingestion than alternatives
- Native financial data support
- ASOF JOIN for backtesting accuracy

**Negative:**
- Less mature ecosystem than PostgreSQL/TimescaleDB
- Fewer integrations
- Smaller community

### References
- [QuestDB vs TimescaleDB Benchmark](https://questdb.com/blog/timescaledb-vs-questdb-comparison/)
- [Time Series Benchmark Suite](https://github.com/questdb/tsbs)

---

## ADR-003: Event-Driven Message Bus Architecture

### Status
**Accepted** - January 2026

### Context
Need a communication pattern between components that:
- Decouples producers and consumers
- Supports async operations
- Handles high throughput
- Enables easy testing

### Decision
Use an **in-process async message bus** with Redis Pub/Sub for multi-node scaling.

### Rationale

**Event Types:**
```
Market Data: TICK, BAR, ORDER_BOOK
Orders: ORDER_NEW, ORDER_FILLED, ORDER_CANCELLED, ORDER_REJECTED
Positions: POSITION_OPENED, POSITION_CLOSED, POSITION_UPDATED
Risk: RISK_LIMIT_BREACH, DRAWDOWN_WARNING, CIRCUIT_BREAKER
System: GATEWAY_CONNECTED, GATEWAY_DISCONNECTED
```

**Architecture:**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Gateway   │────▶│  Message    │────▶│  Strategy   │
│             │     │    Bus      │     │             │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌─────────┐ ┌─────────┐ ┌─────────┐
        │  Risk   │ │   TUI   │ │  Logger │
        │ Manager │ │         │ │         │
        └─────────┘ └─────────┘ └─────────┘
```

**Benefits:**
- Components are decoupled
- Easy to add new subscribers
- Testable in isolation
- Priority queues for risk events

### Consequences

**Positive:**
- Loose coupling between components
- Easy to test and mock
- Supports real-time streaming

**Negative:**
- Complexity of async coordination
- Event ordering guarantees needed
- Debugging event flows can be tricky

### References
- [Cosmic Python - Events and Message Bus](https://www.cosmicpython.com/book/chapter_08_events_and_message_bus.html)
- [Nautilus Trader Message Bus](https://nautilustrader.io/docs/latest/concepts/overview/)

---

## ADR-004: Python MVP First Approach

### Status
**Accepted** - January 2026

### Context
Starting a new trading platform with:
- Uncertain final architecture
- Performance requirements to be validated
- Risk of over-engineering

### Decision
Build a **complete Python MVP first**, then migrate hot paths to Rust.

### Rationale

**Phase 1A (Python MVP):**
- Build everything in Python
- Validate architecture and interfaces
- Get end-to-end flow working
- Measure actual performance bottlenecks

**Phase 1B (Rust Migration):**
- Profile Python implementation
- Identify actual hot paths (not guessed)
- Migrate only what's needed
- Validate 10x+ improvement

**Why This Works:**
1. **Validate Before Optimize**: Don't write Rust for the wrong abstractions
2. **Measure Real Bottlenecks**: Profile shows where Rust is actually needed
3. **Faster Iteration**: Python prototype in days, not weeks
4. **Risk Reduction**: Working system before optimization

### Consequences

**Positive:**
- Fast initial development
- Correct abstractions before optimization
- Data-driven Rust migration decisions

**Negative:**
- Some code rewritten in Rust
- Temporary suboptimal performance
- Two implementations during transition

---

## ADR-005: Textual for Terminal UI

### Status
**Accepted** - January 2026

### Context
Need a terminal-first user interface that:
- Supports async operations
- Has modern styling capabilities
- Works cross-platform
- Integrates with Python ecosystem

### Decision
Use **Textual** for the terminal user interface.

### Rationale

| Framework | Async | Styling | Widgets | Maturity |
|-----------|-------|---------|---------|----------|
| Textual | Native | CSS-like | Rich | Active |
| urwid | Partial | Limited | Basic | Stable |
| blessed | Manual | Basic | DIY | Stable |
| prompt_toolkit | Yes | Custom | Some | Stable |

**Key Textual Features:**
- CSS-like styling
- Built-in widgets (DataTable, Tree, etc.)
- Async-native (works with our message bus)
- Active development
- Great documentation

### Consequences

**Positive:**
- Modern Python TUI framework
- Native async support
- Rich widget library
- CSS-like theming

**Negative:**
- Relatively new (less stable than urwid)
- API may change
- Steeper learning curve than simpler options

### References
- [Textual Documentation](https://textual.textualize.io/)
- [Textual GitHub](https://github.com/Textualize/textual)

---

## ADR-006: CCXT for Exchange Connectivity

### Status
**Accepted** - January 2026

### Context
Need to connect to multiple cryptocurrency exchanges with:
- Unified API
- WebSocket support
- Active maintenance
- Broad exchange coverage

### Decision
Use **CCXT** as the primary exchange connectivity layer.

### Rationale

**Coverage:**
- 100+ exchanges supported
- Unified REST and WebSocket APIs
- Active community (40k+ stars)
- Regular updates for API changes

**Performance Note:**
Default signing is slow (45ms). Must use Coincurve for production:
```python
# With Coincurve: 0.05ms (900x faster)
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})
```

### Consequences

**Positive:**
- Massive exchange coverage
- Unified interface
- Active maintenance
- Large community

**Negative:**
- Abstraction overhead
- Some exchange-specific features hidden
- Need to optimize signing for HFT

### References
- [CCXT GitHub](https://github.com/ccxt/ccxt)
- [CCXT Performance Guide](https://docs.ccxt.com/#/README?id=performance)

---

## Decision Log

| ADR | Decision | Date | Status |
|-----|----------|------|--------|
| 001 | Hybrid Rust/Python | Jan 2026 | Accepted |
| 002 | QuestDB for TSDB | Jan 2026 | Accepted |
| 003 | Event-Driven Bus | Jan 2026 | Accepted |
| 004 | Python MVP First | Jan 2026 | Accepted |
| 005 | Textual for TUI | Jan 2026 | Accepted |
| 006 | CCXT for Exchanges | Jan 2026 | Accepted |

---

*Last Updated: January 2026*
