# Agent-Related GitHub Issues for LIBRA

> Issues derived from Aquarius deep research for Phase 3: AI Agent Layer

---

## New Issues Summary

| Issue # | Title | Priority | Effort | Dependencies |
|---------|-------|----------|--------|--------------|
| #41 | Task Plan Protocol with Evidence-Based Completion | P0 | 8h | #35 |
| #42 | Task Completion Enforcer Middleware | P0 | 10h | #41 |
| #43 | Context Window Monitor | P1 | 6h | None |
| #44 | Human-in-the-Loop (HITL) Trade Confirmation | P1 | 8h | #42, #5 |
| #45 | Agent Progress Reporter (TUI Integration) | P1 | 6h | #5, #41 |
| #46 | Trading Agent Roles & Permissions | P0 | 10h | #9, #4 |
| #47 | Agent Session Management | P2 | 6h | #41 |
| #48 | Error Recovery & Retry Middleware | P1 | 8h | #42 |
| #49 | Agent Prompts & System Design | P1 | 6h | #46 |
| #50 | Worker Subagent Pattern (Coordinator + Workers) | P0 | 12h | #9, #41 |

---

## Issue #41: Task Plan Protocol with Evidence-Based Completion

**Labels**: `enhancement`, `ai-agents`, `phase-3`
**Priority**: P0 (Critical - foundation for all agent work)
**Effort**: 8h

### Description

Implement Anthropic's "feature list" pattern for task planning with evidence-based step completion. This ensures all trading analysis steps are properly verified before proceeding.

### Motivation

From Aquarius research:
- Steps can only be marked complete (never deleted/modified) - preserves audit trail
- Evidence string is REQUIRED to complete any step
- Original steps preserved, new steps marked with `source="worker"`
- Validation prompts per task type (research, analysis, execution)

### Implementation

```python
# src/libra/agents/task_plan.py
from dataclasses import dataclass
from typing import Literal
from datetime import datetime

@dataclass
class TradingStep:
    """A single step in the trading analysis plan."""
    id: str                                    # e.g., "s001", "s001a"
    description: str                           # What needs to be done
    validation: str                            # How to verify completion
    completed: bool = False
    evidence: str | None = None                # REQUIRED to mark complete
    completed_at: datetime | None = None
    dependencies: list[str] = None             # e.g., ["s001", "s002"]
    source: Literal["initializer", "worker"] = "initializer"

@dataclass
class TradingPlan:
    """Complete trading analysis plan."""
    objective: str                             # e.g., "Analyze BTC market conditions"
    goal: str                                  # e.g., "Generate trading signals"
    steps: list[TradingStep]
    task_type: Literal["analysis", "research", "execution"] = "analysis"

# Validation prompts by task type
TRADING_VALIDATION_PROMPTS = {
    "market_analysis": """Before marking complete:
1. List indicators calculated with values
2. Include support/resistance levels identified
3. Note any divergences or patterns found""",

    "signal_generation": """Before marking complete:
1. Specify signal type (LONG/SHORT/HOLD)
2. Include entry price and reasoning
3. Note stop-loss and take-profit levels""",

    "risk_check": """Before marking complete:
1. Verify position size within limits
2. Confirm notional value acceptable
3. Note daily loss remaining""",

    "data_fetch": """Before marking complete:
1. Confirm data timerange fetched
2. Note number of bars/ticks retrieved
3. Verify no gaps in data""",
}
```

### Tasks

- [ ] Create `TradingStep` and `TradingPlan` dataclasses
- [ ] Implement `TaskPlanMiddleware` with evidence validation
- [ ] Add `complete_step(step_id, evidence)` - reject if no evidence
- [ ] Add `get_ready_steps()` - return steps with deps satisfied
- [ ] Add `add_step()` - for discoveries during execution
- [ ] Add `get_progress()` - summary with completion percentage
- [ ] Implement validation prompts per task type
- [ ] Add race condition handling (optimistic locking)
- [ ] Unit tests for all operations
- [ ] Integration test with mock agent

### Acceptance Criteria

```python
# Cannot complete without evidence
result = await task_plan.complete_step("s001", evidence="")
assert "ERROR: Evidence is REQUIRED" in result

# With evidence - success
result = await task_plan.complete_step("s001", evidence="Fetched 30 days of OHLCV data")
assert "marked as completed" in result
```

### References

- `AQUARIUS_AGENT_LEARNINGS.md` Section 2
- Anthropic blog: "Effective harnesses for long-running agents"

---

## Issue #42: Task Completion Enforcer Middleware

**Labels**: `enhancement`, `ai-agents`, `phase-3`
**Priority**: P0 (Critical - ensures agents complete workflow)
**Effort**: 10h

### Description

Implement middleware that tracks pending work and forces the agent to continue until all analysis steps are complete. Prevents agents from "forgetting" to evaluate results or complete delegated tasks.

### Motivation

From Aquarius research - the enforcer tracks 3 states:
1. `pending_delegations`: Steps from get_ready_steps() not yet task()'d
2. `pending_evaluations`: Steps task()'d but not evaluate_and_complete()'d
3. `pending_retries`: Steps that need retry

After each model response, checks for pending work and **forces continuation** via injected message.

### Implementation

```python
# src/libra/agents/enforcer.py
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class TradingEnforcerMiddleware:
    """Ensures trading analysis workflow completes."""

    pending_delegations: set[str] = field(default_factory=set)
    pending_evaluations: set[str] = field(default_factory=set)
    pending_retries: set[str] = field(default_factory=set)
    task_plan_middleware: "TaskPlanMiddleware" = None

    async def awrap_tool_call(self, request, handler):
        """Track state based on tool calls."""
        tool_name = request.tool_call.get("name")

        # Block task() if no plan exists yet
        if tool_name == "task" and self.task_plan_middleware:
            if not await self.task_plan_middleware.plan_exists():
                return ToolMessage(
                    content="ERROR: Create task_plan.json first before delegating tasks.",
                    tool_call_id=request.tool_call.get("id"),
                )

        # Track get_ready_steps() → pending_delegations
        if tool_name == "get_ready_steps":
            result = await handler(request)
            for step in self._extract_steps(result):
                self.pending_delegations.add(step["id"])
            return result

        # Track task() → move to pending_evaluations
        if tool_name == "task":
            step_id = self._extract_step_id(request)
            if step_id:
                self.pending_delegations.discard(step_id)
                self.pending_evaluations.add(step_id)

        # Track evaluate_and_complete → remove or add to retries
        if tool_name == "evaluate_and_complete":
            step_id = request.tool_call.get("args", {}).get("step_id")
            decision = request.tool_call.get("args", {}).get("decision")
            self.pending_evaluations.discard(step_id)
            if decision == "retry":
                self.pending_retries.add(step_id)

        return await handler(request)

    async def aafter_model(self, state, runtime):
        """Check after model responds - force continuation if needed."""
        pending_work = []

        if self.pending_delegations:
            pending_work.append(f"Delegate: {sorted(self.pending_delegations)}")
        if self.pending_evaluations:
            pending_work.append(f"Evaluate: {sorted(self.pending_evaluations)}")
        if self.pending_retries:
            pending_work.append(f"Retry: {sorted(self.pending_retries)}")

        if pending_work:
            return {
                "jump_to": "model",
                "messages": [
                    HumanMessage(content=(
                        "[SYSTEM] You have pending work:\n- "
                        + "\n- ".join(pending_work)
                        + "\n\nHandle ALL of these before proceeding."
                    ))
                ],
            }

        return None  # All clear - allow completion
```

### Tasks

- [ ] Create `TradingEnforcerMiddleware` class
- [ ] Implement state tracking for delegations/evaluations/retries
- [ ] Implement `awrap_tool_call()` for tool tracking
- [ ] Implement `aafter_model()` for continuation forcing
- [ ] Add plan existence enforcement
- [ ] Add browser mode bypass (for web automation)
- [ ] Unit tests for state transitions
- [ ] Integration test with full workflow

### Acceptance Criteria

```python
# Agent cannot stop with pending evaluations
enforcer.pending_evaluations = {"s001"}
result = await enforcer.aafter_model(state, runtime)
assert result["jump_to"] == "model"
assert "Evaluate: ['s001']" in result["messages"][0].content
```

### References

- `AQUARIUS_AGENT_LEARNINGS.md` Section 3
- `aquarius/middleware/task_plan.py:TaskCompletionEnforcerMiddleware`

---

## Issue #43: Context Window Monitor

**Labels**: `enhancement`, `ai-agents`, `performance`
**Priority**: P1 (High - prevents context overflow)
**Effort**: 6h

### Description

Implement context window monitoring with research-backed thresholds to prevent overflow and provide "anxiety management" for the agent.

### Motivation

From Aquarius/oh-my-opencode research:
- 70%: Warning ("still have headroom" - reduces agent anxiety)
- 85%: Critical (trigger compaction)
- 95%: Emergency (aggressive pruning)

Market data can be massive - need careful tracking.

### Implementation

```python
# src/libra/agents/context_monitor.py
from dataclasses import dataclass, field

@dataclass
class ContextMetrics:
    """Metrics for context window usage."""
    section_breakdown: dict[str, int] = field(default_factory=dict)
    peak_utilization: float = 0.0
    compactions_performed: int = 0

class MarketContextMonitor:
    """Context monitor specialized for market data."""

    WARN_THRESHOLD = 0.70       # "Still have headroom"
    CRITICAL_THRESHOLD = 0.85   # Trigger compaction
    EMERGENCY_THRESHOLD = 0.95  # Aggressive pruning

    PRUNE_PROTECT = 40_000      # Protect last 40k tokens
    OUTPUT_RESERVE = 32_000     # Reserve for output

    # Market data budget
    MARKET_DATA_BUDGET = 50_000  # tokens for market data section

    CHARS_PER_TOKEN = 3.5  # Anthropic estimate

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        self.max_context = 200_000
        self.usable_context = self.max_context - self.OUTPUT_RESERVE
        self.metrics = ContextMetrics()

    def count_tokens(self, content: str, section: str) -> int:
        """Count tokens for a section."""
        tokens = int(len(content) / self.CHARS_PER_TOKEN)
        self.metrics.section_breakdown[section] = tokens
        return tokens

    def check_thresholds(self) -> tuple[str, float]:
        """Check usage against thresholds."""
        total = sum(self.metrics.section_breakdown.values())
        utilization = total / self.usable_context

        if utilization >= self.EMERGENCY_THRESHOLD:
            return "emergency", utilization
        elif utilization >= self.CRITICAL_THRESHOLD:
            return "critical", utilization
        elif utilization >= self.WARN_THRESHOLD:
            return "warning", utilization
        return "ok", utilization

    def get_headroom_message(self) -> str:
        """Anxiety management message for agent."""
        _, utilization = self.check_thresholds()
        remaining = 1.0 - utilization
        return (
            f"[Context: {utilization:.0%} used, {remaining:.0%} remaining. "
            f"You have adequate headroom - no need to rush.]"
        )
```

### Tasks

- [ ] Create `ContextMetrics` dataclass
- [ ] Implement `MarketContextMonitor` class
- [ ] Add token estimation (3.5 chars/token)
- [ ] Implement threshold checking
- [ ] Add anxiety management message
- [ ] Add market data budget enforcement
- [ ] Unit tests for threshold detection
- [ ] Integration with agent prompts

### References

- `AQUARIUS_AGENT_LEARNINGS.md` Section 4
- oh-my-opencode context anxiety research

---

## Issue #44: Human-in-the-Loop (HITL) Trade Confirmation

**Labels**: `enhancement`, `ai-agents`, `tui`, `phase-3`
**Priority**: P1 (High - safety for trade execution)
**Effort**: 8h

### Description

Implement human-in-the-loop pattern for trade confirmation, allowing the agent to pause execution and ask user for approval before executing trades.

### Motivation

From Aquarius research:
- Agent can interrupt execution to get user input
- State preserved during interrupt
- Critical for trading safety - user must approve before large trades

### Implementation

```python
# src/libra/agents/hitl.py
from langgraph.errors import GraphInterrupt

class TradeConfirmationMiddleware:
    """Human-in-the-loop for trade confirmation."""

    async def ask_user_question(
        self,
        question: str,
        options: list[str] | None = None,
    ) -> str:
        """Ask user a question and wait for response."""
        raise GraphInterrupt(
            value={
                "type": "ask_user_question",
                "question": question,
                "options": options,
            }
        )

    async def confirm_trade(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        reason: str,
    ) -> str:
        """Ask user to confirm trade before execution."""
        return await self.ask_user_question(
            question=(
                f"🔔 TRADE CONFIRMATION REQUIRED\n\n"
                f"Action: {side} {quantity} {symbol}\n"
                f"Price: {price}\n"
                f"Reason: {reason}\n\n"
                f"Do you approve this trade?"
            ),
            options=["✅ Execute", "✏️ Modify", "❌ Cancel"],
        )

    async def confirm_risk_override(
        self,
        check_failed: str,
        details: str,
    ) -> str:
        """Ask user to approve risk limit override."""
        return await self.ask_user_question(
            question=(
                f"⚠️ RISK CHECK FAILED: {check_failed}\n\n"
                f"Details: {details}\n\n"
                f"Override and proceed anyway?"
            ),
            options=["Override", "Cancel Trade"],
        )
```

### Tasks

- [ ] Create `TradeConfirmationMiddleware`
- [ ] Implement `ask_user_question()` with GraphInterrupt
- [ ] Implement `confirm_trade()` for order confirmation
- [ ] Implement `confirm_risk_override()` for risk breaches
- [ ] Add TUI integration for displaying questions
- [ ] Add agent resume after user response
- [ ] Unit tests for interrupt/resume flow
- [ ] Integration test with TUI

### Acceptance Criteria

```python
# Trade confirmation interrupts execution
with pytest.raises(GraphInterrupt) as exc:
    await hitl.confirm_trade("BTC/USDT", "BUY", 0.1, 50000, "RSI oversold")
assert exc.value.value["type"] == "ask_user_question"
assert "BTC/USDT" in exc.value.value["question"]
```

### Dependencies

- Requires: #42 (Enforcer), #5 (TUI Shell)

### References

- `AQUARIUS_AGENT_LEARNINGS.md` Section 9
- `aquarius/middleware/human_in_loop.py`

---

## Issue #45: Agent Progress Reporter (TUI Integration)

**Labels**: `enhancement`, `ai-agents`, `tui`, `phase-3`
**Priority**: P1 (High - user visibility)
**Effort**: 6h

### Description

Implement progress reporting from agents to the TUI, showing analysis status, step completion, and signal generation in real-time.

### Motivation

From Aquarius research:
- SSE streaming for live updates
- Step-by-step progress
- Output file tracking
- Key findings extraction

### Implementation

```python
# src/libra/agents/progress.py
from libra.core.events import Event, EventType

class TUIProgressReporter:
    """Reports trading agent progress to Textual TUI."""

    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus

    async def report_analysis_started(
        self,
        symbols: list[str],
        task_type: str,
    ) -> None:
        """Report analysis started."""
        await self.message_bus.publish(Event(
            type=EventType.AGENT_ANALYSIS_STARTED,
            data={
                "symbols": symbols,
                "task_type": task_type,
            },
        ))

    async def report_step_progress(
        self,
        step_id: str,
        description: str,
        step: int,
        total_steps: int,
    ) -> None:
        """Report step progress."""
        await self.message_bus.publish(Event(
            type=EventType.AGENT_STEP_PROGRESS,
            data={
                "step_id": step_id,
                "description": description,
                "step": step,
                "total_steps": total_steps,
                "percent": int(step / total_steps * 100),
            },
        ))

    async def report_signal_generated(self, signal: Signal) -> None:
        """Report trading signal generated."""
        await self.message_bus.publish(Event(
            type=EventType.AGENT_SIGNAL,
            data=signal.to_dict(),
        ))

    async def report_completed(
        self,
        summary: str,
        signals: list[Signal],
        key_findings: list[str],
    ) -> None:
        """Report analysis completed."""
        await self.message_bus.publish(Event(
            type=EventType.AGENT_ANALYSIS_COMPLETED,
            data={
                "summary": summary,
                "signals": [s.to_dict() for s in signals],
                "key_findings": key_findings,
            },
        ))
```

### Tasks

- [ ] Add new EventTypes for agent progress
- [ ] Create `TUIProgressReporter` class
- [ ] Implement all report methods
- [ ] Add TUI widget for agent progress display
- [ ] Add signal display widget
- [ ] Unit tests for event publishing
- [ ] Integration test with TUI

### Dependencies

- Requires: #5 (TUI Shell), #41 (Task Plan)

### References

- `AQUARIUS_AGENT_LEARNINGS.md` Section 11
- `aquarius/agents/progress_reporter.py`

---

## Issue #46: Trading Agent Roles & Permissions

**Labels**: `enhancement`, `ai-agents`, `security`, `phase-3`
**Priority**: P0 (Critical - security for trading)
**Effort**: 10h

### Description

Define trading agent roles with specific permissions and capabilities. Analysts cannot execute trades, executors require approval, risk managers can halt trading.

### Motivation

From Aquarius research - agents have skill/permission configurations:
- `can_execute_trades: bool`
- `allowed_symbols: list[str]`
- `max_position_size: Decimal`
- Model selection per role (Opus for coordination, Sonnet for workers)

### Implementation

```python
# src/libra/agents/roles.py
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class AgentPermissions:
    """Permissions for a trading agent role."""
    can_execute_trades: bool = False
    can_cancel_orders: bool = False
    can_halt_trading: bool = False
    max_position_size: Decimal | None = None
    allowed_symbols: list[str] | None = None  # None = all symbols
    requires_approval: bool = True

@dataclass
class AgentRole:
    """Definition of a trading agent role."""
    id: str
    name: str
    description: str
    model: str
    temperature: float
    permissions: AgentPermissions
    tools: list[str]
    system_prompt: str | None = None

# Predefined roles
TRADING_ROLES = {
    "analyst": AgentRole(
        id="analyst",
        name="Market Analyst",
        description="Analyzes market data and generates signals",
        model="claude-sonnet-4-5-20250929",
        temperature=0.3,
        permissions=AgentPermissions(
            can_execute_trades=False,
            can_cancel_orders=False,
        ),
        tools=[
            "fetch_market_data",
            "calculate_indicators",
            "web_search",
            "read_file",
            "write_file",
        ],
    ),
    "risk_manager": AgentRole(
        id="risk_manager",
        name="Risk Manager",
        description="Validates trades against risk limits",
        model="claude-sonnet-4-5-20250929",
        temperature=0.1,
        permissions=AgentPermissions(
            can_execute_trades=False,
            can_halt_trading=True,  # Can halt all trading
        ),
        tools=[
            "check_position_limits",
            "calculate_exposure",
            "get_portfolio",
            "halt_trading",
        ],
    ),
    "executor": AgentRole(
        id="executor",
        name="Trade Executor",
        description="Executes approved trades",
        model="claude-sonnet-4-5-20250929",
        temperature=0.1,
        permissions=AgentPermissions(
            can_execute_trades=True,
            can_cancel_orders=True,
            requires_approval=True,  # HITL required
        ),
        tools=[
            "submit_order",
            "cancel_order",
            "get_positions",
            "get_open_orders",
        ],
    ),
    "coordinator": AgentRole(
        id="coordinator",
        name="Coordinator",
        description="Plans tasks and delegates to workers",
        model="claude-opus-4-5-20251101",  # Opus for planning
        temperature=0.5,
        permissions=AgentPermissions(
            can_execute_trades=False,
        ),
        tools=[
            "task",  # Delegate to workers
            "evaluate_and_complete",
            "get_ready_steps",
            "ask_user_question",
        ],
    ),
}
```

### Tasks

- [ ] Create `AgentPermissions` dataclass
- [ ] Create `AgentRole` dataclass
- [ ] Define predefined roles (analyst, risk_manager, executor, coordinator)
- [ ] Implement permission checking in tool calls
- [ ] Add role-based tool filtering
- [ ] Unit tests for permission validation
- [ ] Integration test with full agent flow

### Acceptance Criteria

```python
# Analyst cannot execute trades
analyst = get_agent("analyst")
assert not analyst.permissions.can_execute_trades

# Executor requires approval
executor = get_agent("executor")
assert executor.permissions.requires_approval
```

### Dependencies

- Requires: #9 (TradingAgents), #4 (Risk Manager)

### References

- `AQUARIUS_AGENT_LEARNINGS.md` Section 13
- `aquarius/ROADMAP.md` - Skills/Permissions System

---

## Issue #47: Agent Session Management

**Labels**: `enhancement`, `ai-agents`, `phase-3`
**Priority**: P2 (Medium - cleanup and isolation)
**Effort**: 6h

### Description

Implement session management for agents - session isolation, state persistence, and cleanup.

### Motivation

From Aquarius research:
- Sessions prevent cross-contamination between tasks
- Auto-generated session IDs
- Session-specific file paths
- Progress tracking per session

### Implementation

```python
# src/libra/agents/session.py
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass
class AgentSession:
    """Manages agent session state."""
    session_id: str
    created_at: datetime
    session_root: Path

    @classmethod
    def create(cls, base_path: Path) -> "AgentSession":
        """Create new session with auto-generated ID."""
        session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        session_root = base_path / session_id
        session_root.mkdir(parents=True, exist_ok=True)
        return cls(
            session_id=session_id,
            created_at=datetime.now(),
            session_root=session_root,
        )

    @property
    def plan_path(self) -> Path:
        return self.session_root / "task_plan.json"

    @property
    def progress_path(self) -> Path:
        return self.session_root / "progress.txt"

    @property
    def data_path(self) -> Path:
        return self.session_root / "data"

    @property
    def output_path(self) -> Path:
        return self.session_root / "output"

    async def cleanup(self) -> None:
        """Cleanup session files."""
        # Archive or delete based on policy
        pass
```

### Tasks

- [ ] Create `AgentSession` dataclass
- [ ] Implement session creation with auto ID
- [ ] Add session file paths
- [ ] Implement mode detection (initializer vs worker)
- [ ] Add session cleanup/archival
- [ ] Add session listing and management
- [ ] Unit tests

### Dependencies

- Requires: #41 (Task Plan)

### References

- `AQUARIUS_AGENT_LEARNINGS.md` Section 7
- `aquarius/session/manager.py`

---

## Issue #48: Error Recovery & Retry Middleware

**Labels**: `enhancement`, `ai-agents`, `reliability`, `phase-3`
**Priority**: P1 (High - robustness)
**Effort**: 8h

### Description

Implement error recovery and retry patterns for agent operations - max retries, exponential backoff, race condition handling.

### Motivation

From Aquarius research:
- Max 3 retries per step with tracking
- Optimistic locking for race conditions
- Transient errors trigger automatic retry
- Clear error messages with next action suggestions

### Implementation

```python
# src/libra/agents/recovery.py
import asyncio
from dataclasses import dataclass, field

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 0.2  # seconds
    max_delay: float = 5.0
    exponential_base: float = 1.5

@dataclass
class ErrorRecoveryMiddleware:
    """Handles retries and error recovery for agent operations."""

    retry_counts: dict[str, int] = field(default_factory=dict)
    config: RetryConfig = field(default_factory=RetryConfig)

    TRANSIENT_ERRORS = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    )

    async def with_retry(
        self,
        operation: str,
        func,
        *args,
        **kwargs,
    ):
        """Execute function with retry on transient errors."""
        delay = self.config.base_delay

        for attempt in range(self.config.max_retries):
            try:
                return await func(*args, **kwargs)
            except self.TRANSIENT_ERRORS as e:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * self.config.exponential_base, self.config.max_delay)

    def track_step_retry(self, step_id: str) -> tuple[int, bool]:
        """Track retry count for a step.

        Returns:
            (attempt_number, exceeded_max)
        """
        if step_id not in self.retry_counts:
            self.retry_counts[step_id] = 0
        self.retry_counts[step_id] += 1
        attempt = self.retry_counts[step_id]
        exceeded = attempt > self.config.max_retries
        return attempt, exceeded

    async def complete_with_optimistic_lock(
        self,
        read_func,
        update_func,
        write_func,
        max_attempts: int = 5,
    ):
        """Complete operation with optimistic locking for race conditions."""
        delay = 0.1

        for attempt in range(max_attempts):
            data = await read_func()
            updated = update_func(data)
            await write_func(updated)

            # Verify write succeeded
            await asyncio.sleep(0.05)
            verified = await read_func()

            if self._verify_update(updated, verified):
                return True

            # Race condition - retry with backoff
            await asyncio.sleep(delay)
            delay *= 1.5

        return False
```

### Tasks

- [ ] Create `RetryConfig` dataclass
- [ ] Create `ErrorRecoveryMiddleware` class
- [ ] Implement `with_retry()` for transient errors
- [ ] Implement step retry tracking
- [ ] Implement optimistic locking pattern
- [ ] Add exchange-specific error handling
- [ ] Unit tests for all retry scenarios
- [ ] Integration test with mock failures

### Dependencies

- Requires: #42 (Enforcer)

### References

- `AQUARIUS_AGENT_LEARNINGS.md` Section 5
- `aquarius/middleware/task_plan.py:complete_step`

---

## Issue #49: Agent Prompts & System Design

**Labels**: `enhancement`, `ai-agents`, `phase-3`
**Priority**: P1 (High - agent behavior)
**Effort**: 6h

### Description

Implement comprehensive prompt templates for trading agents - coordinator prompts, worker prompts, termination controls, and debugging guidelines.

### Motivation

From Aquarius research:
- Structured prompts with clear sections
- Efficiency rules (target tool calls)
- Worker output format
- Termination controls (max iterations, stuck detection)
- Debugging guidelines

### Implementation

```python
# src/libra/agents/prompts.py

COORDINATOR_PROMPT = """You are the Trading Coordinator for LIBRA.

## Your Role
- Plan trading analysis tasks
- Delegate to specialist workers (analyst, risk_manager)
- Evaluate results and make decisions
- Do NOT execute trades directly - delegate to executor

## Workflow
1. PHASE 1: Create task_plan.json with analysis steps
2. PHASE 2: Delegate steps via task() to workers
3. PHASE 3: Evaluate results with evaluate_and_complete()
4. PHASE 4: Generate final trading recommendation

## Tools Available
- task(description, subagent_type): Delegate to worker
- evaluate_and_complete(step_id, result, decision): Mark step done/retry
- get_ready_steps(): Get parallelizable steps
- ask_user_question(question, options): Get user input

## Trading Analysis Steps (Typical)
1. Fetch market data (price, volume, orderbook)
2. Calculate technical indicators
3. Analyze market structure
4. Generate signals
5. Validate against risk limits
6. Present recommendation to user

{context_section}
"""

WORKER_PROMPT = """You are a Trading Worker for LIBRA.

## Your Role
- Execute ONE specific task assigned by coordinator
- Return results quickly (target: 1-3 tool calls)
- Do NOT create your own task plans

## Efficiency Rules
- Target: 1-3 tool calls - be fast and focused
- Use direct tools (web_search, fetch_url) not execute_code
- Don't over-research - get key facts and move on
- Act immediately - don't explain, just do it

## Output Format
When done, output:
```
=== WORKER RESULTS ===
STEP_ID: Brief summary (1-2 sentences)
DATA_RETRIEVED: [list key data points]
FILES_CREATED: /path/to/file (if any)
=== STOP ===
```

After outputting === STOP ===, you are DONE. Do not continue.

{task_section}
"""

TERMINATION_CONTROLS = """## Termination Controls

### Max Iterations
If you have made 15+ tool calls without completing:
1. Summarize what you accomplished
2. List what remains incomplete
3. Ask user how to proceed

### Max Retries
If you have retried the same action 3+ times:
1. Stop retrying
2. Summarize the error pattern
3. Try a different approach OR ask for guidance

### Stuck Detection
If making no progress (same errors, circular attempts):
1. STOP and acknowledge you're stuck
2. Explain what you've tried
3. Ask for user input
"""

DEBUGGING_GUIDELINES = """## Debugging

### Stop-and-Reflect Rule (After 2-3 Failed Attempts)
STOP re-running the same command. Instead:
1. List 5-7 different possible causes
2. Rank by likelihood
3. Check most likely with diagnostic command
4. Address systematically

### When to Report to User
After 3 systematic attempts without success:
- Summarize what you tried
- Share the error message
- Suggest next steps
- Do NOT keep retrying same approach
"""
```

### Tasks

- [ ] Create coordinator prompt template
- [ ] Create worker prompt template
- [ ] Add termination controls section
- [ ] Add debugging guidelines
- [ ] Add context injection for situational awareness
- [ ] Add trading-specific few-shot examples
- [ ] Unit tests for prompt generation
- [ ] A/B test different prompt variations

### Dependencies

- Requires: #46 (Trading Agent Roles)

### References

- `AQUARIUS_AGENT_LEARNINGS.md` Section 8
- `aquarius/session/prompts.py`

---

## Issue #50: Worker Subagent Pattern (Coordinator + Workers)

**Labels**: `enhancement`, `ai-agents`, `architecture`, `phase-3`
**Priority**: P0 (Critical - core agent architecture)
**Effort**: 12h

### Description

Implement the coordinator + worker subagent pattern from Aquarius, where a coordinator (Opus) plans and delegates to specialized workers (Sonnet).

### Motivation

From Aquarius research:
- Coordinator uses Opus for planning
- Workers use Sonnet for execution (faster, cheaper)
- `task()` tool for delegation
- Workers return to coordinator for evaluation

### Implementation

```python
# src/libra/agents/factory.py
from deepagents import SubAgent, create_deep_agent

async def create_trading_agent(
    mode: Literal["normal", "deep"] = "deep",
    coordinator_model: str = "claude-opus-4-5-20251101",
    worker_model: str = "claude-sonnet-4-5-20250929",
) -> CompiledStateGraph:
    """Create a trading agent.

    Args:
        mode: "normal" for single agent, "deep" for coordinator + workers
        coordinator_model: Model for coordinator (planning)
        worker_model: Model for workers (execution)
    """
    if mode == "normal":
        return await _create_normal_agent(coordinator_model)

    # DEEP MODE: Coordinator + Workers
    # 1. Create worker subagents
    analyst_worker = SubAgent(
        name="analyst",
        description="Analyzes market data, calculates indicators, identifies patterns",
        system_prompt=WORKER_PROMPT.format(task_section="Focus on market analysis"),
        tools=[
            fetch_market_data,
            calculate_indicators,
            web_search,
        ],
        model=worker_model,
    )

    risk_worker = SubAgent(
        name="risk_manager",
        description="Validates trades against risk limits, calculates exposure",
        system_prompt=WORKER_PROMPT.format(task_section="Focus on risk validation"),
        tools=[
            check_position_limits,
            calculate_exposure,
            get_portfolio,
        ],
        model=worker_model,
    )

    # 2. Create middleware stack
    task_plan = TaskPlanMiddleware(...)
    enforcer = TradingEnforcerMiddleware(task_plan_middleware=task_plan)
    hitl = TradeConfirmationMiddleware()

    middleware = [task_plan, enforcer, hitl]

    # 3. Create coordinator
    agent = create_deep_agent(
        model=coordinator_model,
        system_prompt=COORDINATOR_PROMPT,
        middleware=middleware,
        subagents=[analyst_worker, risk_worker],
        tools=[browser_task, ask_user_question],
    )

    return agent
```

### Tasks

- [ ] Create `create_trading_agent()` factory function
- [ ] Implement analyst worker subagent
- [ ] Implement risk_manager worker subagent
- [ ] Implement executor worker subagent
- [ ] Wire up middleware stack
- [ ] Add session detection (initializer vs worker mode)
- [ ] Unit tests for factory
- [ ] Integration test with full workflow

### Dependencies

- Requires: #9 (TradingAgents), #41 (Task Plan), #42 (Enforcer)

### References

- `AQUARIUS_AGENT_LEARNINGS.md` Section 1
- `aquarius/agent.py:create_aquarius`

---

## Updated Execution Order

### Phase 3: AI Agent Layer (Expanded)

```
PHASE 3A: Agent Infrastructure (Week 7-8)
  #41 Task Plan Protocol ──┬──▶ #42 Enforcer ──▶ #48 Error Recovery
                           │
                           └──▶ #47 Session Management

  #43 Context Monitor (parallel)

PHASE 3B: Trading Integration (Week 9-10)
  #46 Agent Roles ──┬──▶ #49 Prompts
                    │
  #9 TradingAgents ─┴──▶ #50 Coordinator+Workers ──▶ #10 Deep Research

PHASE 3C: User Interface (Week 11)
  #44 HITL ──┬──▶ #45 Progress Reporter
             │
             └──▶ #11 Natural Language Interface
```

### Updated Sprint Planning

```
Sprint 4 (Week 7-8): Agent Infrastructure
├── #41 Task Plan Protocol (8h)
├── #42 Task Completion Enforcer (10h)
├── #43 Context Monitor (6h)
├── #47 Session Management (6h)
├── #48 Error Recovery (8h)
└── Total: ~38h

Sprint 5 (Week 9-10): Trading Agents
├── #46 Trading Agent Roles (10h)
├── #49 Agent Prompts (6h)
├── #9  TradingAgents Integration (16h)
├── #50 Coordinator+Workers Pattern (12h)
└── Total: ~44h

Sprint 6 (Week 11-12): Agent UX
├── #44 HITL Trade Confirmation (8h)
├── #45 Progress Reporter (6h)
├── #10 Deep Research Agent (12h)
├── #11 Natural Language Interface (16h)
└── Total: ~42h
```

### Complete Phase 3 Dependency Graph

```
                    ┌─────────────────────────────────────────────────────┐
                    │              PHASE 3A: Infrastructure                │
                    │                                                      │
#35 TradingKernel ─▶│   #41 Task Plan Protocol ──┬──▶ #42 Enforcer        │
                    │          │                 │         │               │
#4 Risk Manager ───▶│          │                 │         ▼               │
                    │          │                 │   #48 Error Recovery    │
                    │          ▼                 │                         │
                    │   #47 Session Management   │   #43 Context Monitor   │
                    └──────────┼─────────────────┼─────────────────────────┘
                               │                 │
                    ┌──────────┼─────────────────┼─────────────────────────┐
                    │          │     PHASE 3B: Trading Integration         │
                    │          │                 │                         │
                    │          ▼                 ▼                         │
                    │   #46 Agent Roles ◀───── #9 TradingAgents           │
                    │          │                 │                         │
                    │          ▼                 ▼                         │
                    │   #49 Agent Prompts   #50 Coordinator+Workers       │
                    │                            │                         │
                    │                            ▼                         │
                    │                    #10 Deep Research Agent           │
                    └──────────────────────────────────────────────────────┘
                                                 │
                    ┌────────────────────────────┼─────────────────────────┐
                    │              PHASE 3C: User Interface                │
                    │                            │                         │
#5 TUI Shell ──────▶│   #44 HITL ──────────────┬┴──▶ #45 Progress Reporter│
                    │                          │                          │
                    │                          ▼                          │
                    │               #11 Natural Language Interface        │
                    └─────────────────────────────────────────────────────┘
```

---

## Total Effort for Agent Issues

| Issue | Title | Hours |
|-------|-------|-------|
| #41 | Task Plan Protocol | 8h |
| #42 | Task Completion Enforcer | 10h |
| #43 | Context Monitor | 6h |
| #44 | HITL Trade Confirmation | 8h |
| #45 | Progress Reporter | 6h |
| #46 | Trading Agent Roles | 10h |
| #47 | Session Management | 6h |
| #48 | Error Recovery | 8h |
| #49 | Agent Prompts | 6h |
| #50 | Coordinator+Workers | 12h |
| **Total** | **10 new issues** | **80h** |

Combined with existing Phase 3:
- #9 TradingAgents: 16h
- #10 Deep Research: 12h
- #11 NL Interface: 16h

**Phase 3 Grand Total: ~124h** (vs original ~44h)

---

## Quick Create Commands

```bash
# Create all new agent issues
for i in 41 42 43 44 45 46 47 48 49 50; do
  gh issue create \
    --title "$(sed -n "/## Issue #$i/,/^##/p" AGENT_ISSUES.md | head -1 | sed 's/## Issue #[0-9]*: //')" \
    --label "enhancement,ai-agents,phase-3" \
    --body "See AGENT_ISSUES.md for full description"
done
```

---

*Document Version: 1.0*
*Created: January 2026*
*Based on: AQUARIUS_AGENT_LEARNINGS.md deep research*
