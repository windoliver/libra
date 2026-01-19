# LIBRA Issue Execution Order

> Updated: January 2026 (includes new agent issues #54-#60 from deep research)

---

## Completed ✅

| # | Title | Status |
|---|-------|--------|
| #1 | Message Bus | ✅ CLOSED |
| #2 | Gateway Protocol & CCXT | ✅ CLOSED |
| #3 | Strategy Protocol | ✅ CLOSED |
| #4 | Risk Manager | ✅ CLOSED |
| #5 | TUI Shell | ✅ CLOSED |
| #35 | TradingKernel | ✅ CLOSED |
| #34 | Risk Engine | ✅ CLOSED |
| #32 | DataClient Protocol | ✅ CLOSED |
| #33 | Position & Order Models | ✅ CLOSED |
| #7 | Backtest Engine | ✅ CLOSED |

---

## Execution Order (43 Open Issues)

### 🔴 SPRINT 1: Core Trading (8h)

| Order | # | Issue | Effort | Deps |
|-------|---|-------|--------|------|
| 1 | #36 | Execution Algorithms (TWAP, VWAP) | 10h | ✅ #35 |

---

### 🟠 SPRINT 2: TUI Widgets - Parallel Track (26h)

| Order | # | Issue | Effort | Deps |
|-------|---|-------|--------|------|
| 2 | #40 | Backtest Results Dashboard | 8h | ✅ #7 |
| 3 | #42 | Order Entry Form & Risk Dashboard | 10h | ✅ Ready |
| 4 | #43 | Strategy Management Screen | 8h | ✅ Ready |

---

### 🟡 SPRINT 3: Agent Infrastructure - Foundation (38h)

**Critical Path: #44 → #45 → #55 → #54**

| Order | # | Issue | Effort | Deps | Priority |
|-------|---|-------|--------|------|----------|
| 5 | #44 | Task Plan Protocol (Evidence-Based) | 8h | ✅ #35 | P0 |
| 6 | #45 | Task Completion Enforcer | 6h | #44 | P0 |
| 7 | #46 | Context Window Monitor | 4h | #44 | P1 |
| 8 | **#55** | **Idle and Stuck Detection** ⭐NEW | 6h | #44 | P1 |
| 9 | **#54** | **Continuous Execution Loop** ⭐NEW | 10h | #45, #55 | P0 |
| 10 | **#56** | **Preemptive Auto-Compaction** ⭐NEW | 8h | #46 | P1 |

---

### 🟢 SPRINT 4: Agent Infrastructure - Trading (36h)

| Order | # | Issue | Effort | Deps | Priority |
|-------|---|-------|--------|------|----------|
| 11 | #49 | Trading Agent Roles & Permissions | 6h | #44 | P0 |
| 12 | #50 | Error Recovery & Retry Middleware | 8h | #45 | P1 |
| 13 | #51 | Agent Prompts & Termination Controls | 6h | #49 | P1 |
| 14 | #47 | HITL Trade Confirmation | 10h | #45, #49 | P1 |
| 15 | **#57** | **Tool Output Truncation** ⭐NEW | 4h | - | P2 |
| 16 | #48 | Agent Progress Reporter | 6h | #47 | P1 |

---

### 🔵 SPRINT 5: Agent Infrastructure - Subagents (28h)

| Order | # | Issue | Effort | Deps | Priority |
|-------|---|-------|--------|------|----------|
| 17 | #52 | Worker Subagent Pattern | 12h | #44, #51 | P0 |
| 18 | **#58** | **Background Task Manager** ⭐NEW | 6h | #52 | P2 |
| 19 | #53 | Agent Session Management | 8h | #52 | P2 |

---

### 🟣 SPRINT 6: Agent Integration (52h)

| Order | # | Issue | Effort | Deps |
|-------|---|-------|--------|------|
| 20 | #9 | TradingAgents Multi-Agent | 16h | #44-#53 |
| 21 | #10 | Deep Research Agent | 12h | #9 |
| 22 | #31 | MCP Server for AI Agent | 8h | #9 |
| 23 | #41 | Agent Progress Panel & HITL UI | 8h | #47, #48 |
| 24 | #11 | Natural Language Interface | 16h | #9, #41 |

---

### ⚪ SPRINT 7: Optional Agent Features (10h)

| Order | # | Issue | Effort | Deps | Priority |
|-------|---|-------|--------|------|----------|
| 25 | **#59** | **Explore Agent Mode** ⭐NEW | 6h | #52 | P3 |
| 26 | **#60** | **Hierarchical AGENTS.md** ⭐NEW | 4h | - | P3 |

---

### ⬜ SPRINT 8: Infrastructure (30h)

| Order | # | Issue | Effort | Deps |
|-------|---|-------|--------|------|
| 27 | #29 | Plugin Architecture | 10h | ✅ #35 |
| 28 | #24 | Extensible Protocol Design | 8h | #29 |
| 29 | #23 | Data Management Subsystem | 8h | - |
| 30 | #25 | Observability Layer | 6h | #23 |
| 31 | #30 | FastAPI REST API | 8h | - |

---

### 🔘 SPRINT 9: Adapters (38h)

| Order | # | Issue | Effort | Deps |
|-------|---|-------|--------|------|
| 32 | #27 | Provider/Fetcher Pattern | 6h | ✅ #2 |
| 33 | #12 | Hummingbot Adapter | 10h | #29 |
| 34 | #14 | FinRL Adapter | 10h | #29, ✅#7 |
| 35 | #28 | OpenBB Data Gateway | 8h | #27 |
| 36 | #39 | Prediction Market Gateway | 8h | #27 |

---

### ⚫ SPRINT 10: Strategies (18h)

| Order | # | Issue | Effort | Deps |
|-------|---|-------|--------|------|
| 37 | #38 | Whale Activity Detection | 8h | #23 |
| 38 | #13 | Funding Rate Arbitrage | 10h | ✅#7, ✅#32 |

---

### 🔴 SPRINT 11: Advanced Features (36h)

| Order | # | Issue | Effort | Deps |
|-------|---|-------|--------|------|
| 39 | #15 | Advanced Risk Management | 12h | ✅ #34 |
| 40 | #16 | Audit Logging System | 8h | #25 |
| 41 | #17 | Multi-Strategy Orchestration | 10h | #9, ✅#35 |
| 42 | #18 | Performance Optimization | 8h | All |

---

### 📋 META

| # | Issue | Notes |
|---|-------|-------|
| #20 | Aquarius Epic | Tracking issue - no work |

---

## Visual Dependency Graph

```
SPRINT 2: TUI (Parallel)
├── #40 Backtest UI ◄── ✅#7
├── #42 Order Entry ◄── ✅Ready
└── #43 Strategy Mgmt ◄── ✅Ready

SPRINT 3-5: AGENT INFRASTRUCTURE (Critical Path)

#44 Task Plan ─────┬──► #45 Enforcer ─────┬──► #54 Continuous Executor ⭐
       │           │                      │              │
       │           ├──► #46 Context ──────┼──► #56 Auto-Compaction ⭐
       │           │                      │
       │           └──► #55 Idle/Stuck ───┘  ⭐
       │
       ├──► #49 Roles ──┬──► #51 Prompts ──┬──► #52 Subagent Pattern
       │                │                  │              │
       │                └──► #47 HITL ─────┤              ├──► #58 Background ⭐
       │                         │         │              │
       │                         └──► #48 Progress        └──► #53 Session
       │
       └──► #50 Error Recovery

       #57 Tool Truncation ⭐ (parallel - no deps)

SPRINT 6: AGENT INTEGRATION

#52 Subagent ──► #9 TradingAgents ──┬──► #10 Deep Research
                        │           │
                        │           ├──► #31 MCP Server
                        │           │
#47 HITL ───────► #41 Agent UI ─────┼──► #11 NL Interface
#48 Progress ─────────┘             │
                                    │
#59 Explore Mode ⭐ ◄───────────────┘  (optional)
#60 AGENTS.md ⭐ (optional, no deps)

SPRINT 7+: INFRASTRUCTURE & ADAPTERS

#29 Plugin ──┬──► #24 Extensible Protocol
             ├──► #12 Hummingbot
             └──► #14 FinRL

#27 Provider ──┬──► #28 OpenBB
               └──► #39 Prediction Markets

#23 Data ──────► #25 Observability ──► #16 Audit Logging

#38 Whale Detection ◄── #23
#13 Funding Arb ◄── ✅#7, ✅#32

#15 Advanced Risk ◄── ✅#34
#17 Multi-Strategy ◄── #9, ✅#35
#18 Performance ◄── All
```

---

## Summary by Sprint

| Sprint | Focus | Issues | Effort |
|--------|-------|--------|--------|
| 1 | Core Trading | 1 | 10h |
| 2 | TUI Widgets | 3 | 26h |
| 3 | Agent Foundation | 6 | 42h |
| 4 | Agent Trading | 6 | 40h |
| 5 | Agent Subagents | 3 | 26h |
| 6 | Agent Integration | 5 | 60h |
| 7 | Optional Agent | 2 | 10h |
| 8 | Infrastructure | 5 | 40h |
| 9 | Adapters | 5 | 42h |
| 10 | Strategies | 2 | 18h |
| 11 | Advanced | 4 | 38h |
| **Total** | **43 issues** | - | **~352h** |

---

## New Issues from Deep Research ⭐

| # | Title | Priority | Effort | Sprint |
|---|-------|----------|--------|--------|
| #54 | Continuous Execution Loop (Ralph-Loop) | P0 | 10h | 3 |
| #55 | Idle and Stuck Detection | P1 | 6h | 3 |
| #56 | Preemptive Auto-Compaction | P1 | 8h | 3 |
| #57 | Tool Output Truncation | P2 | 4h | 4 |
| #58 | Background Task Manager | P2 | 6h | 5 |
| #59 | Explore Agent Mode | P3 | 6h | 7 |
| #60 | Hierarchical AGENTS.md Injection | P3 | 4h | 7 |
| **Total New** | | | **44h** | |

---

## Quick Start: What to Work On Now

### Parallel Tracks (Can Start Immediately)

**Track A - TUI (No blockers):**
1. #42 Order Entry Form
2. #43 Strategy Management
3. #40 Backtest Results Dashboard

**Track B - Agent Infrastructure:**
1. #44 Task Plan Protocol ← **START HERE**
2. #45 Task Completion Enforcer
3. #55 Idle and Stuck Detection
4. #54 Continuous Execution Loop

**Track C - Infrastructure:**
1. #29 Plugin Architecture
2. #30 FastAPI REST API

---

*Document generated from deep research of Aquarius and LangChain deepagents*
