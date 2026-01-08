# AQUARIUS: Detailed Execution Plan

## Overview

This document provides a granular execution plan for building the Aquarius trading platform. Each phase is broken down into specific tasks with dependencies, priorities, and acceptance criteria.

---

## Phase 1: Foundation (Weeks 1-4)

### Goal
Establish core infrastructure that all other components will build upon.

### Strategy: Python MVP First, Then Rust Hot Paths

Based on deep research of NautilusTrader and HFT performance benchmarks, we split Phase 1:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1 STRATEGY                             │
│                                                                 │
│  Phase 1A (Weeks 1-2): Python MVP                               │
│  ════════════════════════════════                               │
│  • Build everything in Python first                             │
│  • Validate architecture and interfaces                         │
│  • Get end-to-end flow working                                  │
│  • Target: Functional, not optimal                              │
│                                                                 │
│  Phase 1B (Weeks 3-4): Rust Hot Paths                           │
│  ════════════════════════════════════                           │
│  • Profile Python MVP, find actual bottlenecks                  │
│  • Create libra-core Rust crate                                 │
│  • Migrate: Message Bus, Risk Engine, Indicators                │
│  • Target: <100ms signal-to-order, <1ms dispatch                │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Approach?

1. **Validate Before Optimize**: Don't write Rust for the wrong abstractions
2. **Measure Real Bottlenecks**: Profile shows where Rust is actually needed
3. **Faster Iteration**: Python prototype in days, not weeks
4. **Risk Reduction**: Working system before optimization

### 1.1 Message Bus Implementation

**Priority: P0 (Critical Path)**

```
Task 1.1.1: Event Type Definitions
├── File: src/aquarius/core/events.py
├── Description: Define all event types for the system
├── Subtasks:
│   ├── [ ] Define EventType enum (TICK, BAR, ORDER_*, POSITION_*, RISK_*)
│   ├── [ ] Create Event dataclass with timestamp, source, data
│   ├── [ ] Add event serialization/deserialization
│   └── [ ] Write unit tests
├── Acceptance Criteria:
│   ├── All event types documented
│   ├── Events are immutable
│   └── 100% test coverage
└── Estimated Effort: 4 hours

Task 1.1.2: Message Bus Core
├── File: src/aquarius/core/message_bus.py
├── Dependencies: Task 1.1.1
├── Subtasks:
│   ├── [ ] Implement async event queue
│   ├── [ ] Handler registration (subscribe/unsubscribe)
│   ├── [ ] Event dispatch with error handling
│   ├── [ ] Graceful shutdown
│   └── [ ] Integration tests
├── Acceptance Criteria:
│   ├── Events processed in order within partition
│   ├── Handler errors don't crash bus
│   ├── Clean shutdown drains queue
│   └── <1ms latency for dispatch
└── Estimated Effort: 8 hours

Task 1.1.3: Message Bus Extensions
├── File: src/aquarius/core/message_bus.py
├── Dependencies: Task 1.1.2
├── Subtasks:
│   ├── [ ] Add priority queues for risk events
│   ├── [ ] Implement event filtering
│   ├── [ ] Add metrics/monitoring hooks
│   └── [ ] Redis pub/sub for multi-process
├── Acceptance Criteria:
│   ├── Risk events processed first
│   ├── Can filter by event type/source
│   └── Metrics exported to Prometheus format
└── Estimated Effort: 6 hours
```

### 1.2 Gateway Protocol & CCXT Implementation

**Priority: P0 (Critical Path)**

```
Task 1.2.1: Gateway Protocol Definition
├── File: src/aquarius/gateways/protocol.py
├── Description: Define unified gateway interface
├── Subtasks:
│   ├── [ ] Define Order, OrderResult, Position, Tick dataclasses
│   ├── [ ] Create Gateway Protocol with all methods
│   ├── [ ] Document each method with examples
│   └── [ ] Create abstract base class
├── Acceptance Criteria:
│   ├── Protocol is runtime_checkable
│   ├── All return types defined
│   ├── Async throughout
│   └── Compatible with any exchange
└── Estimated Effort: 4 hours

Task 1.2.2: CCXT Gateway Implementation
├── File: src/aquarius/gateways/ccxt_gateway.py
├── Dependencies: Task 1.2.1, Task 1.1.1
├── Subtasks:
│   ├── [ ] Implement connection management
│   ├── [ ] WebSocket data streaming
│   ├── [ ] Order submission/cancellation
│   ├── [ ] Position and balance queries
│   ├── [ ] Error handling and reconnection
│   └── [ ] Integration tests with testnet
├── Acceptance Criteria:
│   ├── Connects to Binance testnet
│   ├── Streams real-time data
│   ├── Submits and tracks orders
│   ├── Handles disconnections gracefully
│   └── <100ms order submission
└── Estimated Effort: 16 hours

Task 1.2.3: Paper Trading Gateway
├── File: src/aquarius/gateways/paper_gateway.py
├── Dependencies: Task 1.2.1
├── Subtasks:
│   ├── [ ] Implement order simulation
│   ├── [ ] Fill modeling (market/limit)
│   ├── [ ] Position tracking
│   ├── [ ] P&L calculation
│   └── [ ] Slippage simulation
├── Acceptance Criteria:
│   ├── Simulates realistic fills
│   ├── Tracks positions accurately
│   ├── Supports all order types
│   └── Configurable slippage
└── Estimated Effort: 8 hours
```

### 1.3 Strategy Protocol

**Priority: P0 (Critical Path)**

```
Task 1.3.1: Strategy Protocol Definition
├── File: src/aquarius/strategies/protocol.py
├── Subtasks:
│   ├── [ ] Define Signal and SignalType
│   ├── [ ] Define BacktestResult
│   ├── [ ] Create Strategy Protocol
│   ├── [ ] Document lifecycle methods
│   └── [ ] Create base implementation
├── Acceptance Criteria:
│   ├── Same interface for backtest and live
│   ├── Clear signal semantics
│   └── Extensible for custom strategies
└── Estimated Effort: 4 hours

Task 1.3.2: Example Strategy
├── File: src/aquarius/strategies/examples/sma_cross.py
├── Dependencies: Task 1.3.1
├── Subtasks:
│   ├── [ ] Implement SMA crossover strategy
│   ├── [ ] Add configurable parameters
│   ├── [ ] Implement backtest method
│   └── [ ] Write tests
├── Acceptance Criteria:
│   ├── Works in backtest mode
│   ├── Works in live mode
│   └── Demonstrates protocol usage
└── Estimated Effort: 4 hours
```

### 1.4 Risk Manager

**Priority: P1 (High)**

```
Task 1.4.1: Risk Limits Configuration
├── File: src/aquarius/risk/limits.py
├── Subtasks:
│   ├── [ ] Define RiskLimits dataclass
│   ├── [ ] Add position/order limits
│   ├── [ ] Add loss limits (daily/weekly/total)
│   ├── [ ] Add circuit breaker thresholds
│   └── [ ] YAML configuration loading
├── Acceptance Criteria:
│   ├── All limits configurable
│   ├── Sensible defaults
│   └── Validates on load
└── Estimated Effort: 3 hours

Task 1.4.2: Risk Manager Core
├── File: src/aquarius/risk/manager.py
├── Dependencies: Task 1.4.1, Task 1.1.2, Task 1.2.1
├── Subtasks:
│   ├── [ ] Pre-trade risk checks
│   ├── [ ] Position sizing validation
│   ├── [ ] Drawdown monitoring
│   ├── [ ] Rate limiting
│   ├── [ ] Circuit breaker logic
│   └── [ ] Integration with message bus
├── Acceptance Criteria:
│   ├── Rejects orders exceeding limits
│   ├── Tracks drawdown in real-time
│   ├── Triggers circuit breaker correctly
│   └── All checks <1ms
└── Estimated Effort: 10 hours

Task 1.4.3: Position Sizing
├── File: src/aquarius/risk/position_sizing.py
├── Dependencies: Task 1.4.1
├── Subtasks:
│   ├── [ ] Fixed percentage sizing
│   ├── [ ] Volatility-adjusted sizing
│   ├── [ ] Kelly criterion
│   └── [ ] Unit tests
├── Acceptance Criteria:
│   ├── All methods produce valid sizes
│   ├── Never exceeds max position
│   └── Documented formulas
└── Estimated Effort: 4 hours
```

### 1.5 TUI Shell

**Priority: P1 (High)**

```
Task 1.5.1: TUI Application Skeleton
├── File: src/aquarius/tui/app.py
├── Subtasks:
│   ├── [ ] Create Textual App class
│   ├── [ ] Define screen layout
│   ├── [ ] Add keyboard bindings
│   ├── [ ] Implement status bar
│   └── [ ] Add dark/light themes
├── Acceptance Criteria:
│   ├── App launches without errors
│   ├── Responsive keyboard navigation
│   ├── Clean visual design
│   └── Graceful exit
└── Estimated Effort: 6 hours

Task 1.5.2: Basic Widgets
├── Files: src/aquarius/tui/widgets/*.py
├── Dependencies: Task 1.5.1
├── Subtasks:
│   ├── [ ] Connection status widget
│   ├── [ ] Balance display widget
│   ├── [ ] Log viewer widget
│   └── [ ] Command input widget
├── Acceptance Criteria:
│   ├── Widgets update in real-time
│   ├── Proper error states
│   └── Consistent styling
└── Estimated Effort: 8 hours
```

### Phase 1A Milestones (Python MVP)

| Milestone | Tasks | Deliverable |
|-----------|-------|-------------|
| M1A.1 | 1.1.1-1.1.2 | Working message bus (Python) |
| M1A.2 | 1.2.1-1.2.2 | Can connect to Binance testnet |
| M1A.3 | 1.2.3, 1.3.1-1.3.2 | Paper trading works |
| M1A.4 | 1.4.1-1.4.3 | Risk checks enforced |
| M1A.5 | 1.5.1-1.5.2 | TUI launches and shows data |

---

### 1.6 Phase 1B: Rust Core Migration (Weeks 3-4)

**Priority: P0 (Performance Critical)**

```
Task 1.6.1: Rust Crate Setup
├── Files: libra-core/Cargo.toml, libra-core/src/lib.rs
├── Description: Initialize Rust crate with PyO3 bindings
├── Subtasks:
│   ├── [ ] Create Cargo workspace structure
│   ├── [ ] Configure PyO3 and maturin
│   ├── [ ] Set up CI for Rust builds
│   ├── [ ] Create Python stub files for IDE support
│   └── [ ] Benchmark harness setup
├── Acceptance Criteria:
│   ├── `import libra_core` works in Python
│   ├── Rust tests pass
│   └── Maturin builds wheel
└── Estimated Effort: 6 hours

Task 1.6.2: Rust Message Bus Core
├── File: libra-core/src/message_bus.rs
├── Dependencies: Task 1.6.1, Task 1.1.2 (Python reference)
├── Subtasks:
│   ├── [ ] Lock-free MPSC queue (crossbeam)
│   ├── [ ] Event dispatch with zero-copy
│   ├── [ ] Handler registration via PyO3
│   ├── [ ] Async runtime integration (tokio)
│   └── [ ] Benchmark vs Python implementation
├── Acceptance Criteria:
│   ├── <1ms dispatch latency (P99)
│   ├── 10x+ improvement over Python
│   ├── Drop-in replacement for Python bus
│   └── No GC pauses
└── Estimated Effort: 16 hours

Task 1.6.3: Rust Risk Engine
├── File: libra-core/src/risk.rs
├── Dependencies: Task 1.6.1
├── Subtasks:
│   ├── [ ] Pre-trade risk checks (all in one pass)
│   ├── [ ] Position limit validation
│   ├── [ ] Drawdown calculation
│   ├── [ ] Rate limiting (token bucket)
│   └── [ ] PyO3 bindings
├── Acceptance Criteria:
│   ├── <100μs per risk check
│   ├── Same logic as Python (validated)
│   └── Zero allocations in hot path
└── Estimated Effort: 12 hours

Task 1.6.4: Rust Technical Indicators
├── File: libra-core/src/indicators.rs
├── Dependencies: Task 1.6.1
├── Subtasks:
│   ├── [ ] EMA, SMA, RSI, MACD
│   ├── [ ] Bollinger Bands
│   ├── [ ] SIMD optimization where applicable
│   ├── [ ] Streaming (incremental) calculation
│   └── [ ] NumPy array integration
├── Acceptance Criteria:
│   ├── 50x+ faster than pandas-ta
│   ├── Identical results (validated)
│   └── Works with numpy arrays directly
└── Estimated Effort: 12 hours

Task 1.6.5: Rust WebSocket Parser
├── File: libra-core/src/ws_parser.rs
├── Dependencies: Task 1.6.1
├── Subtasks:
│   ├── [ ] JSON parsing (simd-json)
│   ├── [ ] Market data normalization
│   ├── [ ] Zero-copy message handling
│   └── [ ] Binance/Bybit format support
├── Acceptance Criteria:
│   ├── <10μs per message parse
│   ├── Handles malformed data gracefully
│   └── Memory efficient
└── Estimated Effort: 8 hours
```

### Phase 1B Milestones (Rust Core)

| Milestone | Tasks | Deliverable |
|-----------|-------|-------------|
| M1B.1 | 1.6.1 | Rust crate builds, imports in Python |
| M1B.2 | 1.6.2 | Message bus <1ms dispatch |
| M1B.3 | 1.6.3 | Risk checks <100μs |
| M1B.4 | 1.6.4-1.6.5 | Indicators 50x faster, WS parsing <10μs |

### Phase 1 Performance Validation

```
Before proceeding to Phase 2, validate:

┌─────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE GATES                            │
│                                                                 │
│  Component          │ Target         │ Measured │ Status       │
│  ─────────────────────────────────────────────────────────────  │
│  Message dispatch   │ <1ms P99       │ ___      │ [ ] Pass     │
│  Risk check         │ <100μs         │ ___      │ [ ] Pass     │
│  Indicator calc     │ 50x vs pandas  │ ___      │ [ ] Pass     │
│  WS message parse   │ <10μs          │ ___      │ [ ] Pass     │
│  Signal-to-order    │ <100ms         │ ___      │ [ ] Pass     │
└─────────────────────────────────────────────────────────────────┘
```

### Message Bus Best Practices (from Deep Research)

Based on analysis of LMAX Disruptor, NautilusTrader, and industry benchmarks:

```
┌─────────────────────────────────────────────────────────────────┐
│              MESSAGE BUS IMPLEMENTATION STRATEGY                 │
│                                                                 │
│  PHASE 1A (Python MVP):                                         │
│  ══════════════════════                                         │
│  • Use asyncio.PriorityQueue with @dataclass(order=True)       │
│  • Priority levels: Risk(0) > Orders(1) > Signals(2) > Data(3) │
│  • Frozen dataclasses with slots=True for memory efficiency    │
│  • W3C Trace Context compatible trace_id/span_id               │
│  • Graceful shutdown with queue draining (5s timeout)          │
│  • Target: <100μs dispatch, 100K-500K events/sec               │
│                                                                 │
│  PHASE 1B (Rust Core):                                          │
│  ═════════════════════                                          │
│  • ringbuf crate for lock-free SPSC ring buffers               │
│  • Separate ring buffer per priority (no heap allocation)       │
│  • Cache-line padding (120 bytes) for sequence counters        │
│  • PyO3 with allow_threads for GIL release                     │
│  • crossbeam-channel for MPSC scenarios                        │
│  • Target: <1μs dispatch, >1M events/sec                       │
│                                                                 │
│  KEY INSIGHTS:                                                  │
│  ═════════════                                                  │
│  • False sharing kills perf 10-100x → cache-line padding       │
│  • Separate queues > heap priority queue in hot path           │
│  • Python GIL = bottleneck for <1μs → requires Rust core       │
│  • SPSC queues much faster than MPSC/MPMC                      │
│  • Batch ops (push_slice) reduce cache sync overhead           │
└─────────────────────────────────────────────────────────────────┘
```

**Reference Implementations:**
- [NautilusTrader MessageBus](https://nautilustrader.io/docs/nightly/concepts/message_bus/)
- [LMAX Disruptor](https://lmax-exchange.github.io/disruptor/)
- [ringbuf crate](https://crates.io/crates/ringbuf)

---

## Phase 2: First Integration (Weeks 5-6)

### Goal
Integrate Freqtrade strategies and build usable backtesting.

### 2.1 Freqtrade Adapter

**Priority: P0 (Critical Path)**

```
Task 2.1.1: Freqtrade Plugin Structure
├── Files: src/libra/plugins/freqtrade_adapter/
├── Subtasks:
│   ├── [ ] Create plugin package structure
│   ├── [ ] Implement StrategyPlugin base class
│   ├── [ ] Define plugin metadata
│   ├── [ ] Register via entry_points in pyproject.toml
│   └── [ ] Configuration schema
├── Acceptance Criteria:
│   ├── Plugin discoverable via discover_strategies()
│   ├── Commands registered
│   └── Config validated
└── Estimated Effort: 4 hours

Task 2.1.2: Strategy Loading
├── File: src/aquarius/plugins/freqtrade_adapter/loader.py
├── Dependencies: Task 2.1.1
├── Subtasks:
│   ├── [ ] Import Freqtrade strategy classes
│   ├── [ ] Map FT config to Aquarius
│   ├── [ ] Handle FT dependencies
│   └── [ ] List available strategies
├── Acceptance Criteria:
│   ├── Can load any FT strategy
│   ├── Config conversion works
│   └── Handles import errors
└── Estimated Effort: 6 hours

Task 2.1.3: Signal Conversion
├── File: src/aquarius/plugins/freqtrade_adapter/converter.py
├── Dependencies: Task 2.1.2, Task 1.3.1
├── Subtasks:
│   ├── [ ] Convert FT signals to Aquarius Signal
│   ├── [ ] Handle entry/exit signals
│   ├── [ ] Map stoploss/takeprofit
│   └── [ ] Preserve signal metadata
├── Acceptance Criteria:
│   ├── All FT signals converted
│   ├── No signal loss
│   └── Metadata preserved
└── Estimated Effort: 4 hours

Task 2.1.4: Freqtrade Backtesting Bridge
├── File: src/aquarius/plugins/freqtrade_adapter/backtest.py
├── Dependencies: Task 2.1.3
├── Subtasks:
│   ├── [ ] Call FT backtesting engine
│   ├── [ ] Convert results to BacktestResult
│   ├── [ ] Generate performance report
│   └── [ ] Handle errors
├── Acceptance Criteria:
│   ├── Can run FT backtest
│   ├── Results match FT native
│   └── Performance metrics correct
└── Estimated Effort: 6 hours
```

### 2.2 Backtest Engine

**Priority: P0 (Critical Path)**

```
Task 2.2.1: Event-Driven Backtest Engine
├── File: src/aquarius/backtest/engine.py
├── Dependencies: Task 1.1.2, Task 1.3.1
├── Subtasks:
│   ├── [ ] Time-based event simulation
│   ├── [ ] Strategy signal processing
│   ├── [ ] Order fill simulation
│   ├── [ ] Position tracking
│   ├── [ ] Prevent look-ahead bias
│   └── [ ] Multi-timeframe support
├── Acceptance Criteria:
│   ├── No look-ahead bias
│   ├── Accurate fill modeling
│   ├── Same code as live trading
│   └── <1 min for 1 year daily data
└── Estimated Effort: 16 hours

Task 2.2.2: Performance Metrics
├── File: src/aquarius/backtest/metrics.py
├── Dependencies: Task 2.2.1
├── Subtasks:
│   ├── [ ] Calculate returns
│   ├── [ ] Sharpe/Sortino ratios
│   ├── [ ] Max drawdown
│   ├── [ ] Win rate and profit factor
│   ├── [ ] Trade analysis
│   └── [ ] Equity curve
├── Acceptance Criteria:
│   ├── Standard metrics calculated
│   ├── Matches industry definitions
│   └── Handles edge cases
└── Estimated Effort: 6 hours

Task 2.2.3: Result Visualization
├── File: src/aquarius/backtest/visualization.py
├── Dependencies: Task 2.2.2
├── Subtasks:
│   ├── [ ] Equity curve chart
│   ├── [ ] Drawdown chart
│   ├── [ ] Trade markers on price
│   ├── [ ] Monthly returns heatmap
│   └── [ ] Export to HTML/PNG
├── Acceptance Criteria:
│   ├── Charts render correctly
│   ├── Interactive in TUI
│   └── Export works
└── Estimated Effort: 6 hours
```

### 2.3 TUI Dashboard

**Priority: P1 (High)**

```
Task 2.3.1: Portfolio Dashboard Widget
├── File: src/aquarius/tui/widgets/portfolio.py
├── Dependencies: Task 1.5.1, Task 1.2.2
├── Subtasks:
│   ├── [ ] Total value display
│   ├── [ ] Asset allocation pie
│   ├── [ ] P&L summary
│   ├── [ ] Real-time updates
│   └── [ ] Styling
├── Acceptance Criteria:
│   ├── Updates every tick
│   ├── Accurate calculations
│   └── Clean design
└── Estimated Effort: 6 hours

Task 2.3.2: Positions Table Widget
├── File: src/aquarius/tui/widgets/positions.py
├── Dependencies: Task 2.3.1
├── Subtasks:
│   ├── [ ] DataTable with positions
│   ├── [ ] Sortable columns
│   ├── [ ] Color-coded P&L
│   ├── [ ] Close position action
│   └── [ ] Position details popup
├── Acceptance Criteria:
│   ├── All positions displayed
│   ├── Real-time P&L
│   ├── Actions work
│   └── Handles many positions
└── Estimated Effort: 6 hours

Task 2.3.3: Strategy Browser
├── File: src/aquarius/tui/screens/strategies.py
├── Dependencies: Task 2.1.1
├── Subtasks:
│   ├── [ ] List all strategies
│   ├── [ ] Show strategy details
│   ├── [ ] Configure parameters
│   ├── [ ] Start/stop controls
│   ├── [ ] Backtest from UI
│   └── [ ] Performance history
├── Acceptance Criteria:
│   ├── Browse all strategies
│   ├── Configure and run
│   ├── View past results
│   └── Clean UX
└── Estimated Effort: 10 hours
```

### Phase 2 Milestones

| Milestone | Tasks | Deliverable |
|-----------|-------|-------------|
| M2.1 | 2.1.1-2.1.4 | Freqtrade strategies run in Aquarius |
| M2.2 | 2.2.1-2.2.3 | Backtest engine with visualization |
| M2.3 | 2.3.1-2.3.3 | Full TUI dashboard |

---

## Phase 3: Agent Layer (Weeks 5-6)

### Goal
Integrate LLM agents for intelligent analysis and decision support.

### 3.1 TradingAgents Integration

**Priority: P0 (Critical Path)**

```
Task 3.1.1: Agent Definitions
├── File: src/aquarius/agents/definitions.py
├── Subtasks:
│   ├── [ ] Define AgentRole enum
│   ├── [ ] Create AgentDefinition dataclass
│   ├── [ ] Configure analyst agents
│   ├── [ ] Configure decision agents
│   ├── [ ] Configure execution agents
│   └── [ ] System prompts for each
├── Acceptance Criteria:
│   ├── All roles defined
│   ├── Prompts are effective
│   └── Configurable models
└── Estimated Effort: 6 hours

Task 3.1.2: LangGraph Workflow
├── File: src/aquarius/agents/workflows/trading_workflow.py
├── Dependencies: Task 3.1.1
├── Subtasks:
│   ├── [ ] Define TradingState
│   ├── [ ] Build analysis graph
│   ├── [ ] Implement debate mechanism
│   ├── [ ] Add decision routing
│   ├── [ ] Error handling
│   └── [ ] Streaming output
├── Acceptance Criteria:
│   ├── Full workflow executes
│   ├── Agents debate effectively
│   ├── Final decision produced
│   └── <30s for full analysis
└── Estimated Effort: 16 hours

Task 3.1.3: Agent Registry
├── File: src/libra/agents/registry.py
├── Dependencies: Task 3.1.1
├── Subtasks:
│   ├── [ ] Create simple AgentRegistry dataclass
│   ├── [ ] Register agent definitions
│   ├── [ ] Set up role-based permissions
│   └── [ ] Configure skills per agent
├── Acceptance Criteria:
│   ├── Agents registered correctly
│   ├── Permissions enforced
│   └── Skills work
└── Estimated Effort: 4 hours
```

### 3.2 Deep Research Agent

**Priority: P1 (High)**

```
Task 3.2.1: Research Agent Implementation
├── File: src/aquarius/agents/deep_research.py
├── Dependencies: Task 3.1.2
├── Subtasks:
│   ├── [ ] Multi-source search
│   ├── [ ] Document analysis
│   ├── [ ] Synthesis and summarization
│   ├── [ ] Citation management
│   ├── [ ] Report generation
│   └── [ ] Caching
├── Acceptance Criteria:
│   ├── Searches 10+ sources
│   ├── Synthesizes findings
│   ├── Cites all sources
│   └── Reports are actionable
└── Estimated Effort: 12 hours

Task 3.2.2: Data Source Integrations
├── File: src/aquarius/agents/data_sources/
├── Dependencies: Task 3.2.1
├── Subtasks:
│   ├── [ ] News API integration
│   ├── [ ] SEC filings (EDGAR)
│   ├── [ ] Social sentiment (Twitter/Reddit)
│   ├── [ ] On-chain data (for crypto)
│   └── [ ] Rate limiting
├── Acceptance Criteria:
│   ├── All sources work
│   ├── Rate limits respected
│   └── Data normalized
└── Estimated Effort: 10 hours
```

### 3.3 Natural Language Interface

**Priority: P1 (High)**

```
Task 3.3.1: Command Parser
├── File: src/aquarius/nl/parser.py
├── Dependencies: Task 3.1.2
├── Subtasks:
│   ├── [ ] Intent recognition
│   ├── [ ] Entity extraction
│   ├── [ ] Command mapping
│   ├── [ ] Ambiguity handling
│   └── [ ] Confirmation flow
├── Acceptance Criteria:
│   ├── Parses common commands
│   ├── Asks for clarification
│   ├── Confirms risky actions
│   └── <1s parse time
└── Estimated Effort: 8 hours

Task 3.3.2: TUI Chat Interface
├── File: src/aquarius/tui/widgets/chat.py
├── Dependencies: Task 3.3.1
├── Subtasks:
│   ├── [ ] Chat input widget
│   ├── [ ] Message history
│   ├── [ ] Agent response display
│   ├── [ ] Streaming responses
│   └── [ ] Rich formatting
├── Acceptance Criteria:
│   ├── Natural conversation
│   ├── Responses stream
│   ├── History preserved
│   └── Commands executed
└── Estimated Effort: 8 hours
```

### Phase 3 Milestones

| Milestone | Tasks | Deliverable |
|-----------|-------|-------------|
| M3.1 | 3.1.1-3.1.3 | Multi-agent analysis works |
| M3.2 | 3.2.1-3.2.2 | Deep research reports generated |
| M3.3 | 3.3.1-3.3.2 | Natural language commands work |

---

## Phase 4: Advanced Strategies (Weeks 7-8)

### Goal
Add market making, arbitrage, and ML-based strategies.

### 4.1 Hummingbot Adapter

```
Task 4.1.1: Hummingbot Plugin
├── Files: src/aquarius/plugins/hummingbot_adapter/
├── Subtasks:
│   ├── [ ] Plugin structure
│   ├── [ ] Market making strategies
│   ├── [ ] Arbitrage strategies
│   ├── [ ] DEX gateway support
│   └── [ ] Configuration
├── Estimated Effort: 16 hours

Task 4.1.2: Avellaneda-Stoikov Implementation
├── File: src/aquarius/strategies/market_making/avellaneda.py
├── Subtasks:
│   ├── [ ] Optimal spread calculation
│   ├── [ ] Inventory management
│   ├── [ ] Order placement logic
│   └── [ ] Performance tracking
├── Estimated Effort: 12 hours
```

### 4.2 Funding Rate Arbitrage

```
Task 4.2.1: Funding Rate Strategy
├── File: src/aquarius/strategies/arbitrage/funding_rate.py
├── Subtasks:
│   ├── [ ] Funding rate monitoring
│   ├── [ ] Position management (spot + perp)
│   ├── [ ] Entry/exit logic
│   ├── [ ] Risk parameters
│   └── [ ] Backtesting
├── Estimated Effort: 12 hours
```

### 4.3 FinRL Adapter

```
Task 4.3.1: FinRL Integration
├── Files: src/aquarius/plugins/finrl_adapter/
├── Subtasks:
│   ├── [ ] Plugin structure
│   ├── [ ] Training pipeline
│   ├── [ ] Model deployment
│   ├── [ ] Live inference
│   └── [ ] Model versioning
├── Estimated Effort: 16 hours
```

### Phase 4 Milestones

| Milestone | Tasks | Deliverable |
|-----------|-------|-------------|
| M4.1 | 4.1.1-4.1.2 | Market making works |
| M4.2 | 4.2.1 | Funding rate arb running |
| M4.3 | 4.3.1 | RL strategies deployable |

---

## Phase 5: Production Hardening (Weeks 9-10)

### Goal
Make the platform production-ready with full risk management, logging, and performance optimization.

### 5.1 Full Risk Management

```
Task 5.1.1: Advanced Risk Checks
├── Subtasks:
│   ├── [ ] Correlation monitoring
│   ├── [ ] VaR calculation
│   ├── [ ] Stress testing
│   ├── [ ] Margin monitoring
│   └── [ ] Liquidation warnings
├── Estimated Effort: 16 hours

Task 5.1.2: Audit Logging
├── Subtasks:
│   ├── [ ] Order audit trail
│   ├── [ ] Agent decision logging
│   ├── [ ] Risk event logging
│   ├── [ ] Log rotation/retention
│   └── [ ] Compliance export
├── Estimated Effort: 10 hours
```

### 5.2 Multi-Strategy Orchestration

```
Task 5.2.1: Portfolio of Strategies
├── Subtasks:
│   ├── [ ] Strategy allocation
│   ├── [ ] Capital distribution
│   ├── [ ] Correlation management
│   ├── [ ] Rebalancing
│   └── [ ] Performance attribution
├── Estimated Effort: 16 hours
```

### 5.3 Performance Optimization

```
Task 5.3.1: Latency Optimization
├── Subtasks:
│   ├── [ ] Profile hot paths
│   ├── [ ] Optimize message bus
│   ├── [ ] Connection pooling
│   ├── [ ] Cython for indicators
│   └── [ ] Rust components
├── Estimated Effort: 16 hours
```

### Phase 5 Milestones

| Milestone | Tasks | Deliverable |
|-----------|-------|-------------|
| M5.1 | 5.1.1-5.1.2 | Production risk management |
| M5.2 | 5.2.1 | Multi-strategy portfolios |
| M5.3 | 5.3.1 | Optimized performance |

---

## Summary

### Total Estimated Effort

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1 | 2 weeks | Core infrastructure, CCXT gateway, basic TUI |
| Phase 2 | 2 weeks | Freqtrade adapter, backtest engine, dashboard |
| Phase 3 | 2 weeks | LLM agents, deep research, NL interface |
| Phase 4 | 2 weeks | Market making, arbitrage, RL strategies |
| Phase 5 | 2 weeks | Production hardening, optimization |
| **Total** | **10 weeks** | **Full platform** |

### Critical Path

```
Message Bus → Gateway Protocol → CCXT Gateway → Strategy Protocol
    → Freqtrade Adapter → Backtest Engine → TUI Dashboard
    → Agent Definitions → LangGraph Workflow → NL Interface
```

### Risk Factors

1. **CCXT complexity**: Exchange APIs change frequently
2. **LLM latency**: Agent responses may be slow
3. **Backtest accuracy**: Ensuring no look-ahead bias
4. **Real money execution**: Requires extensive testing

### Success Criteria

- [ ] Can run Freqtrade strategy on Binance testnet
- [ ] Backtest results match Freqtrade native
- [ ] LLM agents produce actionable analysis
- [ ] TUI is intuitive and responsive
- [ ] Risk limits enforced correctly
- [ ] <100ms signal-to-order latency

---

*Document Version: 1.0*
*Last Updated: January 2026*
