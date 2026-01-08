# LIBRA Issue Execution Order & Complete Roadmap

> Complete execution roadmap for all 53 GitHub issues with dependencies and priorities.
> Updated: January 2026

---

## Issue Summary

| Category | Count | Closed | Open |
|----------|-------|--------|------|
| Phase 0-1: Foundation | 8 | 7 | 1 |
| Phase 2: Backtest | 6 | 0 | 6 |
| Phase 3: AI Agents | 14 | 0 | 14 |
| Phase 3: UI/UX | 4 | 0 | 4 |
| Phase 4: Advanced | 3 | 0 | 3 |
| Phase 5: Production | 4 | 0 | 4 |
| Enhancements | 14 | 0 | 14 |
| **Total** | **53** | **7** | **46** |

---

## Complete Issue List by Phase

### Phase 0-1: Foundation (7 CLOSED, 1 OPEN)

| # | Title | Status | Effort |
|---|-------|--------|--------|
| #1 | Message Bus Implementation | ✅ CLOSED | - |
| #2 | Gateway Protocol & CCXT Implementation | ✅ CLOSED | - |
| #3 | Strategy Protocol Definition | ✅ CLOSED | - |
| #4 | Risk Manager Implementation | ✅ CLOSED | - |
| #5 | TUI Shell Implementation | ✅ CLOSED | - |
| #19 | Project Scaffolding | ✅ CLOSED | - |
| #32 | Actor/Strategy Base Classes | ✅ CLOSED | - |
| #33 | Split Gateway into DataClient/ExecutionClient | ✅ CLOSED | - |
| #35 | TradingKernel Central Orchestrator | ✅ CLOSED | - |
| #34 | Risk Engine with Pre-Trade Validation Pipeline | OPEN | 6h |

### Phase 2: Backtest & Freqtrade

| # | Title | Status | Effort | Dependencies |
|---|-------|--------|--------|--------------|
| #6 | Freqtrade Adapter Plugin | OPEN | 20h | #3, #32 |
| #7 | Backtest Engine Implementation | OPEN | 16h | #1, #3, #4 |
| #37 | Event-Driven Backtest with Unified Strategy Code | OPEN | 12h | #7, #33 |
| #36 | Execution Algorithm Framework (TWAP, VWAP) | OPEN | 10h | #33, #34 |
| #8 | TUI Dashboard Widgets | OPEN | 12h | #5, #7 |
| #40 | **[NEW]** Backtest Results Dashboard Widget | OPEN | 8h | #7, #5 |

### Phase 3A: Agent Infrastructure

| # | Title | Status | Effort | Dependencies |
|---|-------|--------|--------|--------------|
| #44 | **[NEW]** Task Plan Protocol (Evidence-Based) | OPEN | 8h | #35 |
| #45 | **[NEW]** Task Completion Enforcer | OPEN | 10h | #44 |
| #46 | **[NEW]** Context Window Monitor | OPEN | 6h | None |
| #50 | **[NEW]** Error Recovery & Retry Middleware | OPEN | 8h | #45 |
| #53 | **[NEW]** Agent Session Management | OPEN | 6h | #44 |

### Phase 3B: Trading Agent Integration

| # | Title | Status | Effort | Dependencies |
|---|-------|--------|--------|--------------|
| #9 | TradingAgents Multi-Agent Integration | OPEN | 16h | #3, #35 |
| #49 | **[NEW]** Trading Agent Roles & Permissions | OPEN | 10h | #9, #4 |
| #51 | **[NEW]** Agent Prompts & Termination Controls | OPEN | 6h | #49 |
| #52 | **[NEW]** Coordinator + Workers Pattern | OPEN | 12h | #9, #44, #45 |
| #10 | Deep Research Agent | OPEN | 12h | #9 |

### Phase 3C: Agent UX

| # | Title | Status | Effort | Dependencies |
|---|-------|--------|--------|--------------|
| #47 | **[NEW]** HITL Trade Confirmation | OPEN | 8h | #45, #5 |
| #48 | **[NEW]** Agent Progress Reporter | OPEN | 6h | #44, #5 |
| #41 | **[NEW]** Agent Progress Panel & HITL TUI | OPEN | 12h | #47, #48, #5 |
| #11 | Natural Language Interface | OPEN | 16h | #9, #5 |

### Phase 3D: UI/UX Enhancements

| # | Title | Status | Effort | Dependencies |
|---|-------|--------|--------|--------------|
| #42 | **[NEW]** Order Entry & Risk Dashboard | OPEN | 10h | #4, #5 |
| #43 | **[NEW]** Strategy Management Screen | OPEN | 8h | #3, #32, #35 |

### Phase 4: Advanced Strategies

| # | Title | Status | Effort | Dependencies |
|---|-------|--------|--------|--------------|
| #12 | Hummingbot Adapter Plugin | OPEN | 16h | #3, #33 |
| #13 | Funding Rate Arbitrage Strategy | OPEN | 12h | #3, #4 |
| #14 | FinRL Adapter Plugin | OPEN | 16h | #3, #7 |

### Phase 5: Production Hardening

| # | Title | Status | Effort | Dependencies |
|---|-------|--------|--------|--------------|
| #15 | Advanced Risk Management | OPEN | 16h | #4, #34 |
| #16 | Audit Logging System | OPEN | 10h | #35 |
| #17 | Multi-Strategy Orchestration | OPEN | 16h | #35, #32 |
| #18 | Performance Optimization | OPEN | 16h | Rust Core |

### Enhancements (Can Parallelize)

| # | Title | Status | Effort | Dependencies |
|---|-------|--------|--------|--------------|
| #21 | Decision: Database Selection | OPEN | 2h | None |
| #22 | Decision: Nexus Dependency | OPEN | 2h | None |
| #23 | Data Management Subsystem | OPEN | 12h | #7 |
| #24 | Extensible Protocol Design | OPEN | 6h | #3, #2 |
| #25 | Observability Layer | OPEN | 10h | #35 |
| #26 | Python 3.13 Free-Threaded Testing | OPEN | 4h | Rust Core |
| #27 | Provider/Fetcher Pattern | OPEN | 6h | #2 |
| #28 | OpenBB Data Gateway | OPEN | 8h | #27 |
| #29 | Plugin Architecture | OPEN | 8h | #35 |
| #30 | FastAPI REST API | OPEN | 12h | #35 |
| #31 | MCP Server for AI Agent Integration | OPEN | 10h | #9, #30 |
| #38 | Whale Activity Detection Signals | OPEN | 8h | #27 |
| #39 | Prediction Market Gateway | OPEN | 16h | #27 |

---

## Visual Dependency Graph

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    PHASE 0-1 (MOSTLY DONE)                   │
                    │                                                              │
                    │  #1 MessageBus ✅ ──▶ #2 Gateway ✅ ──▶ #3 Strategy ✅       │
                    │        │                   │                  │              │
                    │        ▼                   ▼                  ▼              │
                    │  #4 Risk ✅ ──────▶ #35 Kernel ✅ ◀─── #32 Actor ✅          │
                    │        │                   │                                 │
                    │        ▼                   ▼                                 │
                    │  #5 TUI ✅ ◀─────── #33 DataClient ✅                        │
                    │                            │                                 │
                    │                     #34 Risk Pipeline (OPEN)                 │
                    └────────────────────────────┼─────────────────────────────────┘
                                                 │
    ┌────────────────────────────────────────────┼─────────────────────────────────┐
    │                              PHASE 2: BACKTEST                                │
    │                                                                               │
    │   #6 Freqtrade ────────┬──▶ #7 Backtest Engine ──▶ #37 Unified Backtest      │
    │                        │            │                      │                  │
    │                        │            ▼                      ▼                  │
    │                        │   #40 Backtest Results UI   #8 TUI Dashboard         │
    │                        │                                                      │
    │                        └──▶ #36 Exec Algorithms                               │
    └───────────────────────────────────────────────────────────────────────────────┘
                                                 │
    ┌────────────────────────────────────────────┼─────────────────────────────────┐
    │                        PHASE 3A: AGENT INFRASTRUCTURE                         │
    │                                                                               │
    │   #44 Task Plan ──────┬──▶ #45 Enforcer ──▶ #50 Error Recovery               │
    │          │            │                                                       │
    │          │            └──▶ #53 Session Management                             │
    │          │                                                                    │
    │          └──────────────── #46 Context Monitor (parallel)                     │
    └───────────────────────────────────────────────────────────────────────────────┘
                                                 │
    ┌────────────────────────────────────────────┼─────────────────────────────────┐
    │                      PHASE 3B: TRADING AGENT INTEGRATION                      │
    │                                                                               │
    │   #9 TradingAgents ───┬──▶ #49 Agent Roles ──▶ #51 Prompts/Termination       │
    │          │            │                                                       │
    │          │            └──▶ #52 Coordinator + Workers ──▶ #10 Deep Research   │
    └───────────────────────────────────────────────────────────────────────────────┘
                                                 │
    ┌────────────────────────────────────────────┼─────────────────────────────────┐
    │                           PHASE 3C: AGENT UX                                  │
    │                                                                               │
    │   #47 HITL ──────────┬──▶ #48 Progress Reporter ──▶ #41 Agent Progress TUI   │
    │                      │                                                        │
    │                      └──▶ #11 Natural Language Interface                      │
    └───────────────────────────────────────────────────────────────────────────────┘
                                                 │
    ┌────────────────────────────────────────────┼─────────────────────────────────┐
    │                          PHASE 3D: UI/UX                                      │
    │                                                                               │
    │   #42 Order Entry & Risk Dashboard                                            │
    │   #43 Strategy Management Screen                                              │
    └───────────────────────────────────────────────────────────────────────────────┘
                                                 │
    ┌────────────────────────────────────────────┼─────────────────────────────────┐
    │                      PHASE 4: ADVANCED STRATEGIES                             │
    │                                                                               │
    │   #12 Hummingbot ──┬──▶ #13 Funding Rate Arbitrage                           │
    │                    │                                                          │
    │                    └──▶ #14 FinRL Adapter                                     │
    └───────────────────────────────────────────────────────────────────────────────┘
                                                 │
    ┌────────────────────────────────────────────┼─────────────────────────────────┐
    │                        PHASE 5: PRODUCTION                                    │
    │                                                                               │
    │   #15 Advanced Risk ──▶ #16 Audit Logging ──▶ #17 Multi-Strategy             │
    │                                                                               │
    │   Rust Core ──────────▶ #18 Performance Optimization                          │
    └───────────────────────────────────────────────────────────────────────────────┘
```

---

## Execution Order (Linear)

### Sprint 1 (Weeks 1-2): Foundation Completion
```
Focus: Complete remaining Phase 1 + Decisions
├── #21 Decision: Database Selection (2h)
├── #22 Decision: Nexus Dependency (2h)
├── #34 Risk Engine Pipeline (6h)
└── Total: ~10h
```

### Sprint 2 (Weeks 3-4): Backtest Core
```
Focus: Backtest Engine + Freqtrade
├── #6  Freqtrade Adapter Plugin (20h)
├── #7  Backtest Engine Implementation (16h)
└── Total: ~36h
```

### Sprint 3 (Weeks 5-6): Backtest Advanced + UI
```
Focus: Event-driven backtest + visualization
├── #37 Event-Driven Backtest (12h)
├── #36 Execution Algorithms (10h)
├── #40 Backtest Results Dashboard Widget (8h)
├── #8  TUI Dashboard Widgets (12h)
└── Total: ~42h
```

### Sprint 4 (Weeks 7-8): Agent Infrastructure
```
Focus: Core agent patterns from Aquarius
├── #44 Task Plan Protocol (8h)
├── #45 Task Completion Enforcer (10h)
├── #46 Context Window Monitor (6h)
├── #50 Error Recovery & Retry (8h)
├── #53 Session Management (6h)
└── Total: ~38h
```

### Sprint 5 (Weeks 9-10): Trading Agent Integration
```
Focus: Multi-agent trading system
├── #9  TradingAgents Integration (16h)
├── #49 Trading Agent Roles (10h)
├── #51 Agent Prompts & Termination (6h)
├── #52 Coordinator + Workers (12h)
└── Total: ~44h
```

### Sprint 6 (Weeks 11-12): Agent UX
```
Focus: User-facing agent features
├── #47 HITL Trade Confirmation (8h)
├── #48 Agent Progress Reporter (6h)
├── #41 Agent Progress Panel TUI (12h)
├── #10 Deep Research Agent (12h)
├── #11 Natural Language Interface (16h)
└── Total: ~54h
```

### Sprint 7 (Weeks 13-14): UI/UX + Advanced
```
Focus: Trading UI + advanced strategies
├── #42 Order Entry & Risk Dashboard (10h)
├── #43 Strategy Management Screen (8h)
├── #12 Hummingbot Adapter (16h)
├── #13 Funding Rate Arbitrage (12h)
├── #14 FinRL Adapter (16h)
└── Total: ~62h
```

### Sprint 8 (Weeks 15-16): Production
```
Focus: Production readiness
├── #15 Advanced Risk Management (16h)
├── #16 Audit Logging System (10h)
├── #17 Multi-Strategy Orchestration (16h)
├── #18 Performance Optimization (16h)
└── Total: ~58h
```

### Ongoing (Parallel Work)
```
Can be done anytime after dependencies:
├── #27 Provider/Fetcher Pattern (6h)
├── #28 OpenBB Data Gateway (8h)
├── #29 Plugin Architecture (8h)
├── #30 FastAPI REST API (12h)
├── #31 MCP Server (10h)
├── #38 Whale Activity Detection (8h)
├── #39 Prediction Market Gateway (16h)
├── #23 Data Management Subsystem (12h)
├── #24 Extensible Protocol Design (6h)
├── #25 Observability Layer (10h)
├── #26 Python 3.13 Testing (4h)
└── Total: ~100h (spread across sprints)
```

---

## Agent-Specific Patterns (from Aquarius)

### Termination Controls (#51)

| Control | Threshold | Action |
|---------|-----------|--------|
| Max Iterations | 15+ tool calls | Ask user how to proceed |
| Max Retries | 3+ same action | Try different approach |
| Stuck Detection | No progress | Acknowledge and ask for help |

### Evidence-Based Completion (#44)

All trading steps require evidence to mark complete:
```python
# Cannot complete without evidence
result = await task_plan.complete_step("s001", evidence="")
# Returns: "ERROR: Evidence is REQUIRED"

# With evidence - success
result = await task_plan.complete_step("s001", evidence="RSI=28, SMA crossover detected")
# Returns: "Step s001 marked as completed"
```

### Trading Validation Prompts (#51)

| Task Type | Validation Requirements |
|-----------|------------------------|
| market_analysis | List indicators with values, S/R levels, patterns |
| signal_generation | Signal type, entry price, reasoning, SL/TP |
| risk_check | Position size, notional value, daily loss remaining |
| data_fetch | Timerange, bar count, gap verification |

### Context Window Thresholds (#46)

| Level | Threshold | Action |
|-------|-----------|--------|
| OK | <70% | Continue normally |
| Warning | 70% | "Still have headroom" (anxiety management) |
| Critical | 85% | Trigger compaction |
| Emergency | 95% | Aggressive pruning |

---

## Total Effort Summary

| Phase | Issues | Hours |
|-------|--------|-------|
| Phase 1 (remaining) | 1 | 6h |
| Phase 2 | 6 | 78h |
| Phase 3A: Agent Infra | 5 | 38h |
| Phase 3B: Agent Integration | 5 | 56h |
| Phase 3C: Agent UX | 4 | 42h |
| Phase 3D: UI/UX | 2 | 18h |
| Phase 4 | 3 | 44h |
| Phase 5 | 4 | 58h |
| Enhancements | 13 | 100h |
| **Total** | **43 open** | **~440h** |

At 40h/week = **~11 weeks** for core implementation
With enhancements = **~14 weeks** total

---

## Quick Reference: New Issues Created

### UI/UX Issues (4 new)
- **#40** - Backtest Results Dashboard Widget
- **#41** - Agent Progress Panel & HITL TUI Integration
- **#42** - Order Entry Form & Risk Dashboard Widget
- **#43** - Strategy Management Screen

### Agent Infrastructure Issues (10 new)
- **#44** - Task Plan Protocol with Evidence-Based Completion
- **#45** - Task Completion Enforcer Middleware
- **#46** - Context Window Monitor
- **#47** - Human-in-the-Loop (HITL) Trade Confirmation
- **#48** - Agent Progress Reporter (MessageBus Integration)
- **#49** - Trading Agent Roles & Permissions
- **#50** - Error Recovery & Retry Middleware
- **#51** - Agent Prompts & Termination Controls
- **#52** - Worker Subagent Pattern (Coordinator + Workers)
- **#53** - Agent Session Management

---

*Document Version: 2.0*
*Updated: January 2026*
*Total Issues: 53 (7 closed, 46 open)*
