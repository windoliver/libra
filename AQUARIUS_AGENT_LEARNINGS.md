# Comprehensive Agent Learnings from Aquarius → Libra

Deep research findings from `/Users/taofeng/aquarius` to inform Libra's Phase 3 AI Agent Layer.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Task Plan Middleware](#2-task-plan-middleware---evidence-based-completion)
3. [Task Completion Enforcer](#3-task-completion-enforcer---keep-agent-going)
4. [Context Window Monitoring](#4-context-window-monitoring)
5. [Error Recovery & Retry Patterns](#5-error-recovery--retry-patterns)
6. [Tool Implementations](#6-tool-implementations)
7. [Session Management](#7-session-management)
8. [Prompts & System Design](#8-prompts--system-design)
9. [Human-in-the-Loop (HITL)](#9-human-in-the-loop-hitl)
10. [Browser Automation](#10-browser-automation)
11. [Streaming & Progress Reporting](#11-streaming--progress-reporting)
12. [Agent Caching](#12-agent-caching)
13. [Implementation Recommendations for Libra](#13-implementation-recommendations-for-libra)

---

## 1. Architecture Overview

### Coordinator + Worker Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                     COORDINATOR AGENT                            │
│              (Claude Opus - planning/delegation)                 │
│                                                                  │
│  Tools: task(), evaluate_and_complete(), get_ready_steps(),     │
│         browser_task(), ask_user_question()                      │
│                                                                  │
│  Does NOT call: mcp_call, execute_code directly                  │
│                                                                  │
│                    ↓ delegates via task()                        │
├─────────────────────────────────────────────────────────────────┤
│   WORKER_SIMPLE           │        WORKER_COMPLEX                │
│  (Sonnet - 1-3 tools)     │      (Sonnet - 5+ tools)            │
│                           │                                      │
│  For: quick searches,     │  For: comprehensive research,        │
│  single API calls,        │  multi-source validation,            │
│  file reads               │  detailed analysis                   │
│                           │                                      │
│  Tools: mcp_call,         │  Tools: mcp_call, execute_code,      │
│  execute_code, web_search,│  web_search, fetch_url,              │
│  fetch_url, file ops      │  file ops, http_request              │
└───────────────────────────┴──────────────────────────────────────┘
```

### Two Execution Modes

```python
# NORMAL MODE: Single agent, no task planning
agent, mode, _ = await create_aquarius(mode="normal")
# - Simpler, faster, cheaper
# - For straightforward tasks
# - No subagents, no task_plan.json

# DEEP MODE: Coordinator + workers with task planning
agent, mode, enforcer = await create_aquarius(mode="deep")
# - For complex multi-step tasks
# - Creates task_plan.json
# - Delegates to worker subagents
# - Enforcer ensures completion
```

---

## 2. Task Plan Middleware - Evidence-Based Completion

### Core Concept: Anthropic's Feature List Pattern

From `aquarius/middleware/task_plan.py`:

```python
class TaskPlanMiddleware(AgentMiddleware):
    """
    Hybrid mode implementation:
    - Original steps (from Initializer): Can only mark as completed, cannot modify/delete
    - New steps: Can be added during Worker mode, marked with source="worker"
    - Evidence is REQUIRED to complete any step
    """
```

### Task Plan JSON Structure

```json
{
    "task_id": "task-20250106-143000",
    "objective": "Analyze BTC market conditions",
    "goal": "Generate trading signals for BTC/USDT",
    "steps": [
        {
            "id": "s001",
            "description": "Fetch historical price data from exchange",
            "validation": "Verify 30 days of OHLCV data retrieved",
            "completed": false,
            "evidence": null,
            "dependencies": [],
            "source": "initializer"
        },
        {
            "id": "s002",
            "description": "Calculate technical indicators (SMA, RSI, MACD)",
            "validation": "Verify all indicators computed without NaN",
            "completed": false,
            "evidence": null,
            "dependencies": ["s001"],
            "source": "initializer"
        }
    ]
}
```

### Evidence-Based Step Completion

```python
async def complete_step(self, step_id: str, evidence: str) -> str:
    """Mark a step as completed with evidence.

    CRITICAL: Evidence is REQUIRED - cannot mark complete without it.
    """
    # Validate evidence - reject if empty
    if not evidence or not evidence.strip():
        return (
            f"ERROR: Cannot complete step {step_id}. "
            "Evidence is REQUIRED to mark a step as complete. "
            "Please provide specific evidence of what was accomplished "
            "(e.g., files created, data found, results obtained)."
        )

    # Update step with evidence and timestamp
    step["completed"] = True
    step["evidence"] = evidence.strip()
    step["completed_at"] = datetime.now(UTC).isoformat()
```

### Validation Prompts by Task Type

```python
VALIDATION_PROMPTS = {
    "research": """Before marking complete:
1. List sources found with URLs
2. Include key quotes or data points
3. Verify information from multiple sources""",

    "analysis": """Before marking complete:
1. Verify output file was created
2. Check data quality (row counts, no nulls)
3. Include output path and summary stats as evidence""",

    "data_processing": """Before marking complete:
1. Verify pipeline ran without errors
2. Check output file exists and has expected format
3. Include stats (input rows, output rows, duration)""",

    "api_integration": """Before marking complete:
1. Test API call with real data
2. Verify response matches expected format
3. Include sample response as evidence""",
}
```

### Hybrid Mode: Original + Added Steps

```python
async def add_step(
    self,
    description: str,
    validation: str | None = None,
    after_step_id: str | None = None,
) -> str:
    """Add a new step during execution.

    Use when you discover additional work is needed.
    New steps are marked with source="worker" for audit trail.
    """
    new_step = {
        "id": self._generate_step_id(plan, after_step_id),  # e.g., "s001a"
        "description": description.strip(),
        "validation": validation or "Verify completion",
        "completed": False,
        "source": "worker",  # Audit: added during execution
        "added_by_session": self.session_id,
    }
```

### Dependency-Based Parallel Execution

```python
async def get_ready_steps(self, limit: int | None = None) -> list[dict]:
    """Get ALL steps where dependencies are satisfied.

    Returns steps that can be executed in parallel.
    """
    completed_ids = {s["id"] for s in steps if s.get("completed")}

    ready = []
    for step in steps:
        if step.get("completed"):
            continue

        # Check if all dependencies are satisfied
        deps = step.get("dependencies", [])
        unmet_deps = [d for d in deps if d not in completed_ids]

        if not unmet_deps:  # All deps met
            ready.append(step)

    return ready  # Returns ALL ready steps for parallel execution
```

### All Steps Complete Detection

```python
# When all steps are done, return special indicator
if not ready and len(completed_ids) == len(steps):
    return [{
        "id": "__ALL_COMPLETE__",
        "description": (
            f"ALL {len(steps)} STEPS COMPLETED! "
            "Synthesize the final response now: "
            "1. Read the output files from data/ and output/ folders. "
            "2. Combine findings into a comprehensive answer. "
            "3. Present the final result to the user. "
            "4. STOP - do not call any more tools after responding."
        ),
        "all_complete": True,
    }]
```

---

## 3. Task Completion Enforcer - Keep Agent Going

### Core Concept: Force Agent to Complete Workflow

From `aquarius/middleware/task_plan.py`:

```python
class TaskCompletionEnforcerMiddleware(AgentMiddleware):
    """Middleware that enforces the task completion workflow.

    Tracks three states:
    1. pending_delegations: Steps from get_ready_steps() not yet task()'d
    2. pending_evaluations: Steps task()'d but not evaluate_and_complete()'d
    3. pending_retries: Steps that need retry (evaluate_and_complete returned "retry")

    After each model response, checks if there are pending items and forces
    continuation via Command(goto="agent") with a reminder message.

    This ensures the model cannot "forget" to:
    - Delegate all ready steps
    - Evaluate all worker results
    - Retry failed steps
    """

    def __init__(self, task_plan_middleware: TaskPlanMiddleware | None = None):
        self.pending_delegations: set[str] = set()  # Steps to delegate
        self.pending_evaluations: set[str] = set()  # Awaiting evaluation
        self.pending_retries: set[str] = set()      # Need retry
```

### State Tracking on Tool Calls

```python
async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage:
    """Track state based on tool calls."""
    tool_name = request.tool_call.get("name")

    # PLAN ENFORCEMENT: Block task() if no plan exists
    if tool_name == "task" and self._task_plan_middleware:
        plan_exists = await self._task_plan_middleware.plan_exists()
        if not plan_exists:
            return ToolMessage(
                content=(
                    "ERROR: Cannot delegate tasks before creating a task plan.\n\n"
                    "You are in INITIALIZER MODE and must follow PHASE 1 before PHASE 2:\n\n"
                    "PHASE 1 (REQUIRED NOW):\n"
                    "- write_file(task_plan.json) - Create the plan with steps\n"
                    "- get_ready_steps() - Find steps ready for execution\n\n"
                    "PHASE 2 (after Phase 1):\n"
                    "- task() - Delegate steps to workers"
                ),
                tool_call_id=request.tool_call.get("id"),
            )

    # Track task() calls - move from pending_delegations to pending_evaluations
    if tool_name == "task":
        step_id = self._extract_step_id(description)
        if step_id:
            self.pending_delegations.discard(step_id)
            self.pending_evaluations.add(step_id)

    # Track evaluate_and_complete - handle completion/retry
    elif tool_name == "evaluate_and_complete":
        step_id = tool_args.get("step_id")
        decision = tool_args.get("decision")

        self.pending_evaluations.discard(step_id)
        self.pending_delegations.discard(step_id)

        if decision == "retry":
            self.pending_retries.add(step_id)

    return await handler(request)
```

### Force Continuation After Model Response

```python
@hook_config(can_jump_to=["model"])
async def aafter_model(self, state: AgentState, runtime: Runtime):
    """Check after model responds - force continuation if needed."""

    # Build list of pending work
    pending_work = []
    actions = []

    if self.pending_delegations:
        steps = sorted(self.pending_delegations)
        pending_work.append(f"Delegate: {steps}")
        actions.append(f'task("{steps[0]}: [description]", subagent_type="worker")')

    if self.pending_evaluations:
        steps = sorted(self.pending_evaluations)
        pending_work.append(f"Evaluate: {steps}")
        actions.append(
            f'evaluate_and_complete(step_id="{steps[0]}", '
            f'worker_result="...", decision="complete", evidence_summary="...")'
        )

    if self.pending_retries:
        steps = sorted(self.pending_retries)
        pending_work.append(f"Retry: {steps}")
        actions.append(f'task("{steps[0]} RETRY: [guidance]", subagent_type="worker")')

    # FORCE CONTINUATION if there's pending work
    if pending_work:
        logger.warning(f"[ENFORCER] !!! FORCING CONTINUATION !!! {pending_work}")
        return {
            "jump_to": "model",  # Jump back to model node
            "messages": [
                HumanMessage(
                    content=(
                        "[SYSTEM REMINDER] You have pending work:\n- "
                        + "\n- ".join(pending_work)
                        + "\n\n"
                        "Handle ALL of these before calling get_ready_steps() again.\n\n"
                        "Examples:\n" + "\n".join(actions)
                    )
                )
            ],
        }

    # All clear - allow completion
    return None
```

### Plan Enforcement in Deep Mode

```python
# If plan doesn't exist and agent tries to finish without creating it
if plan_exists is False and not has_tool_calls:
    return {
        "jump_to": "model",
        "messages": [
            HumanMessage(
                content=(
                    "STOP. You are in INITIALIZER MODE and must create the task plan first.\n\n"
                    "You MUST call write_file to create task_plan.json with this structure:\n"
                    "```json\n"
                    '{"objective": "...", "steps": [{"id": "s001", "description": "...", '
                    '"validation": "...", "completed": false}]}\n'
                    "```\n\n"
                    "DO NOT respond to the user until the task plan is created."
                )
            )
        ],
    }
```

---

## 4. Context Window Monitoring

### Research-Backed Thresholds

From `aquarius/middleware/context_monitor.py`:

```python
class ContextMonitor:
    """Context window monitoring with threshold alerts.

    Thresholds (research-backed):
    - WARN_THRESHOLD (70%): "Anxiety management" - reminds agent there's headroom
    - CRITICAL_THRESHOLD (85%): Trigger preemptive compaction
    - EMERGENCY_THRESHOLD (95%): Aggressive pruning required

    References:
    - oh-my-opencode: Context window anxiety management
    - Anthropic context engineering best practices
    - Claude Code context compaction patterns
    """

    WARN_THRESHOLD = 0.70       # 70% - anxiety management
    CRITICAL_THRESHOLD = 0.85   # 85% - trigger compaction
    EMERGENCY_THRESHOLD = 0.95  # 95% - aggressive pruning

    # Protection zones (from Claude Code)
    PRUNE_PROTECT = 40_000   # Protect last 40k tokens of tool outputs
    PRUNE_MINIMUM = 20_000   # Only prune if >20k tokens available
    OUTPUT_TOKEN_MAX = 32_000  # Reserved for model output
```

### Token Estimation and Tracking

```python
def estimate_tokens(self, text: str) -> int:
    """Estimate token count using character heuristic.

    Uses Anthropic's recommended ~3.5 characters per token.
    """
    CHARS_PER_TOKEN = 3.5
    return int(len(text) / CHARS_PER_TOKEN)

def count_tokens(self, content: str, section: str) -> int:
    """Count tokens for a context section and update metrics."""
    tokens = self.estimate_tokens(content)
    self.metrics.section_breakdown[section] = tokens
    return tokens
```

### Anxiety Management Message

```python
def get_headroom_message(self) -> str:
    """Get a message for the agent about context headroom.

    Used for "anxiety management" - reminding the agent it has room to work.
    """
    _, utilization = self.check_thresholds()
    remaining = 1.0 - utilization
    remaining_tokens = int(self.usable_context * remaining)

    return (
        f"[Context Status: {utilization:.0%} used, {remaining:.0%} remaining "
        f"(~{remaining_tokens:,} tokens). You have adequate headroom to continue "
        f"thorough work - no need to rush.]"
    )
```

### Threshold-Based Actions

```python
def log_usage(self) -> tuple[str, float]:
    """Log context usage and check thresholds."""
    status, utilization = self.check_thresholds()

    if status == "emergency":
        logger.critical(f"🚨 Context at {utilization:.0%} - EMERGENCY! Aggressive pruning required!")
    elif status == "critical":
        logger.critical(f"🚨 Context at {utilization:.0%} - Trigger compaction!")
    elif status == "warning":
        logger.warning(f"⚠️ Context at {utilization:.0%} - Still have headroom")

    return status, utilization
```

---

## 5. Error Recovery & Retry Patterns

### Retry Logic with Max Attempts

From `TaskPlanMiddleware.evaluate_and_complete()`:

```python
# Track retry counts per step
_retry_counts: dict[str, int] = {}
_max_retries: int = 3

async def evaluate_and_complete(
    self,
    step_id: str,
    worker_result: str,
    decision: Literal["complete", "retry", "add_steps"],
    evidence_summary: str | None = None,
    retry_reason: str | None = None,
    retry_guidance: str | None = None,
    new_steps: list[dict] | None = None,
) -> str:
    """Evaluate worker result and take appropriate action."""

    if decision == "retry":
        # Validate retry has reason and guidance
        if not retry_reason:
            return "ERROR: decision='retry' requires retry_reason."
        if not retry_guidance:
            return "ERROR: decision='retry' requires retry_guidance."

        # Track retry count
        if step_id not in self._retry_counts:
            self._retry_counts[step_id] = 0
        self._retry_counts[step_id] += 1
        attempt = self._retry_counts[step_id]

        # Check max retries
        if attempt > self._max_retries:
            await self._append_progress(
                f"STEP {step_id}: FAILED after {self._max_retries} retries - {retry_reason}"
            )
            return (
                f"ERROR: Step {step_id} has exceeded max retries ({self._max_retries}). "
                f"Last failure reason: {retry_reason}. "
                f"Consider using add_step() to break this into smaller steps, "
                f"or mark this step as blocked and continue with other steps."
            )

        return (
            f"RETRY REQUIRED for {step_id} (attempt #{attempt}/{self._max_retries})\n"
            f"Reason: {retry_reason}\n"
            f"Guidance: {retry_guidance}\n\n"
            f"YOUR NEXT ACTION: Call task() with:\n"
            f'task("{step_id} RETRY #{attempt}: {retry_guidance}", subagent_type="worker")'
        )
```

### Race Condition Handling with Optimistic Locking

```python
async def complete_step(self, step_id: str, evidence: str) -> str:
    """Mark step complete with optimistic locking retry."""

    # RACE CONDITION FIX: Retry loop with optimistic locking
    max_retries = 5
    retry_delay = 0.1  # Start with 100ms

    for attempt in range(max_retries):
        # Read current plan
        plan = await self._read_plan()

        # Update the step
        for step in plan.get("steps", []):
            if step.get("id") == step_id:
                step["completed"] = True
                step["evidence"] = evidence
                break

        # Write updated plan
        await self._write_plan(plan)

        # CRITICAL: Verify the write succeeded (optimistic locking check)
        await asyncio.sleep(0.05)  # Small delay for writes to propagate
        verify_plan = await self._read_plan()

        for s in verify_plan.get("steps", []):
            if s.get("id") == step_id and s.get("completed"):
                return f"Step {step_id} marked as completed"

        # Race condition detected - retry with backoff
        await asyncio.sleep(retry_delay)
        retry_delay *= 1.5

    return f"ERROR: Failed to complete step {step_id} after {max_retries} attempts"
```

### Pending Evaluations Race Condition Fix

```python
async def get_ready_steps(self) -> list[dict]:
    """Get ready steps with race condition protection."""

    # RACE CONDITION FIX: Wait for pending evaluate_and_complete() calls
    wait_start = time.time()
    max_wait_seconds = 30

    while True:
        async with self._eval_lock:
            pending = set(self._pending_evaluations)

        if not pending:
            break

        if time.time() - wait_start > max_wait_seconds:
            logger.error(f"TIMEOUT waiting for pending evaluations: {pending}")
            break

        await asyncio.sleep(0.1)  # Brief wait, then re-check

    # Now safe to read task_plan.json
    plan = await self._read_plan()
    # ... rest of logic
```

### Streaming Service Transient Error Retry

```python
# Retry settings for transient API errors
STREAM_MAX_RETRIES = 3
STREAM_RETRY_BASE_DELAY = 2.0  # seconds

# Transient errors that should trigger retry
TRANSIENT_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    ConnectionError,
    ConnectionResetError,
)
```

---

## 6. Tool Implementations

### Tool Execution Middleware

From `aquarius/middleware/mcp_gateway.py`:

```python
class ToolExecutionMiddleware(AgentMiddleware):
    """Provides tools for MCP, code execution, and web access.

    Tools:
    - mcp_call: Call MCP server tools
    - execute_code: Run Python in sandbox
    - web_search: Search the web
    - fetch_url: Fetch webpage as markdown
    - http_request: Make HTTP API calls
    """

    @property
    def tools(self) -> list:
        return [
            self.mcp_call,
            self.execute_code,
            self.web_search,
            self.fetch_url,
            self.http_request,
        ]
```

### MCP Call Pattern

```python
async def mcp_call(
    self,
    server: str,
    tool: str,
    args: dict[str, Any] | None = None,
) -> str:
    """Call an MCP tool on a mounted server.

    Args:
        server: MCP server name (e.g., "brave_search")
        tool: Tool name on that server
        args: Tool arguments

    Example:
        mcp_call("exchange-connector", "get_orderbook", {"symbol": "BTC/USDT"})
    """
```

### Code Execution with Sandbox

```python
async def execute_code(self, code: str) -> str:
    """Execute Python code in sandbox environment.

    - Variables persist within same call only
    - Has aquarius_web pre-imported for web operations
    - Uses /mnt/nexus/ prefix for file paths
    """
```

### Web Search with Date Context

```python
async def web_search(self, query: str, max_results: int = 5) -> str:
    """Search the web for current information.

    TIP: Always include current date for time-sensitive queries.
    Example: web_search("NVIDIA stock price December 2025")
    """
```

### Tool Selection Guidelines (from prompts)

```python
TOOL_SELECTION = """
### Direct Tools (Preferred - No Sandbox Needed)
- web_search: Finding news, stock prices, current events
- fetch_url: Reading documentation, articles
- http_request: Calling REST APIs
- read_file: Viewing uploaded files
- write_file: Saving new reports (NEW files only)
- edit_file: Modifying existing files
- glob: Finding files by pattern
- grep: Searching file contents

### Sandbox Tools (When You Need to Run Code)
- execute: Running scripts, installing packages
- execute_code: Quick calculations, data exploration

### Common Mistakes to Avoid
| Don't Do This             | Do This Instead      | Why                    |
|---------------------------|---------------------|------------------------|
| execute_code for search   | web_search tool     | Direct tool is faster  |
| open() in sandbox         | write_file tool     | Nexus storage persists |
| execute("cat file.txt")   | read_file tool      | Direct tool is cleaner |
"""
```

---

## 7. Session Management

### Session Isolation Pattern

From `aquarius/session/manager.py`:

```python
class SessionManager:
    """Manages session state and mode detection."""

    def __init__(self, backend: NexusBackend, session_id: str | None = None):
        self.backend = backend
        # Auto-generate session ID if not provided
        self.session_id = session_id or f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Session-specific paths
        self._session_root = f"/sessions/{self.session_id}"
        self._plan_path = f"{self._session_root}/task_plan.json"
        self._progress_path = f"{self._session_root}/progress.txt"

    async def detect_mode(self) -> Literal["initializer", "worker"]:
        """Detect session mode based on existing files.

        - initializer: No task_plan.json exists (first run)
        - worker: task_plan.json exists (continuing execution)
        """
        plan_exists = await self.backend.exists(self._plan_path)
        return "worker" if plan_exists else "initializer"
```

### Session File Structure

```
/sessions/{session_id}/
├── task_plan.json      # Current plan with steps
├── progress.txt        # Execution log
├── data/               # Input data files
├── output/             # Generated outputs
└── scripts/            # Saved scripts for execution
```

### Bootstrap Messages by Mode

```python
def get_bootstrap_message(self, mode: str, task: str | None = None) -> str:
    """Get initial message based on mode."""

    if mode == "initializer":
        return f"""New task: {task}

You are in INITIALIZER MODE. Follow these phases:

PHASE 1: Plan Creation
1. Analyze the task requirements
2. Create task_plan.json with steps
3. Call get_ready_steps() to identify parallelizable work

PHASE 2: Execution
4. Delegate steps to workers via task()
5. Evaluate results with evaluate_and_complete()
6. Continue until all steps complete
"""

    elif mode == "worker":
        return """Continuing task execution.

You are in WORKER MODE. The task plan already exists.
1. Call get_ready_steps() to find next steps
2. Delegate to workers or mark complete
3. Continue until all_complete indicator
"""
```

---

## 8. Prompts & System Design

### Coordinator System Prompt Structure

```python
# From get_initializer_prompt() / get_worker_prompt()

system_prompt = _log_context_sections(
    "coordinator",
    agent_persona=agent_system_prompt,      # User-defined persona (optional)
    base_prompt=coordinator_base_prompt,     # Core instructions
    browser_tools=browser_prompt_injection,  # Browser automation docs
    hitl_guidelines=hitl_prompt,            # Human-in-loop docs
    mcp_discovery=mcp_prompt,               # Available MCP tools
)
```

### Worker Prompt Key Rules

```python
EFFICIENCY_RULES_SIMPLE = """<efficiency_rules>
<rule>**Target: 1-3 tool calls** - Be fast and focused</rule>
<rule>**Use direct tools** - web_search, fetch_url directly (NOT via execute_code)</rule>
<rule>**Don't over-research** - Get key facts and move on</rule>
<rule>**Act immediately** - Don't explain, just do it</rule>
<rule>**Do NOT create sub-tasks or todos** - Just do the work directly</rule>
</efficiency_rules>"""
```

### Worker Output Format

```python
WORKER_OUTPUT_FORMAT = """<output_format>
<critical>You MUST stop after outputting your results!</critical>

When you have completed your research task:
<step number="1">Save your findings to a file (if required)</step>
<step number="2">Output your results in the format below</step>
<step number="3">**STOP - DO NOT make any more tool calls**</step>

<results_template>
=== WORKER RESULTS ===
STEP_ID: Brief summary (1-2 sentences)
FILES_CREATED: /path/to/file.md (if any)
SOURCES: url1, url2 (if web research)
=== STOP ===
</results_template>
</output_format>"""
```

### Debugging & Troubleshooting

```python
DEBUGGING_TROUBLESHOOTING = """<debugging>
### Stop-and-Reflect Rule (After 2-3 Failed Attempts)
**STOP re-running the same command. Instead:**
1. List **5-7 different possible causes** of the failure
2. Rank them by likelihood
3. Check the most likely cause with a **diagnostic command**
4. Address causes systematically, starting with highest probability

### When to Report to User
After 3 systematic attempts without success:
- Summarize what you tried
- Share the error message
- Suggest next steps
- **Do NOT keep retrying the same approach**
</debugging>"""
```

### Termination Controls

```python
AGENT_TERMINATION_CONTROLS = """<termination_controls>
<max_iterations>
If you have made **15+ tool calls** without resolving the user's request:
1. Summarize what you accomplished so far
2. List what remains incomplete or blocked
3. Ask the user how to proceed
</max_iterations>

<max_retries>
If you have **retried the same action 3+ times** with errors:
1. Stop retrying the same approach
2. Summarize the error pattern
3. Try a different approach OR ask the user for guidance
</max_retries>

<stuck_detection>
If you are making no progress (same errors, circular attempts):
1. STOP and acknowledge you're stuck
2. Explain what you've tried
3. Ask for user input or alternative approach
</stuck_detection>
</termination_controls>"""
```

---

## 9. Human-in-the-Loop (HITL)

### Ask User Question Pattern

From `aquarius/middleware/human_in_loop.py`:

```python
class AskUserQuestionMiddleware(AgentMiddleware):
    """Allows agent to ask user questions during execution."""

    async def ask_user_question(
        self,
        question: str,
        options: list[str] | None = None,
    ) -> str:
        """Ask the user a question and wait for response.

        Args:
            question: The question to ask
            options: Optional list of suggested answers

        Example:
            ask_user_question(
                "Should I include derivatives in the analysis?",
                options=["Yes", "No", "Only for hedging symbols"]
            )
        """
        # Interrupt execution to get user input
        raise GraphInterrupt(
            value={
                "type": "ask_user_question",
                "question": question,
                "options": options,
            }
        )
```

### HITL Resume Pattern

```python
# When resuming after HITL, skip enforcer to avoid state conflicts
agent, mode, enforcer = await create_aquarius(
    ...,
    skip_enforcer=True,  # Skip enforcer for HITL resume (state not checkpointed)
)
```

---

## 10. Browser Automation

### Browser Task Pattern

From `aquarius/middleware/browser.py`:

```python
class BrowserMiddleware(AgentMiddleware):
    """Browser automation for web tasks requiring authentication."""

    async def browser_task(
        self,
        url: str,
        task: str,
        require_auth: bool = False,
    ) -> str:
        """Run browser automation.

        Args:
            url: Starting URL
            task: What to do (e.g., "Extract current support/resistance levels")
            require_auth: Whether to show live browser for user login

        Flow:
        1. First try WITHOUT auth (uses saved cookies)
        2. If auth needed, call with require_auth=True
        3. User logs in via live browser view
        4. Cookies saved for next time
        5. Always call browser_stop() when done
        """
```

### Browser Mode in Enforcer

```python
# Browser mode skips plan enforcement
if tool_name in ("browser_task", "browser_resume"):
    self._browser_mode = True
    logger.info("Browser tool detected - entering browser mode")
elif tool_name == "browser_stop":
    self._browser_mode = False
    self._browser_just_finished = True  # Allow one response without plan

# Skip enforcement during browser operations
if self._browser_mode:
    return None  # Allow without plan enforcement
```

---

## 11. Streaming & Progress Reporting

### Progress Reporter Pattern

From `aquarius/agents/progress_reporter.py`:

```python
class ProgressReporter:
    """Reports progress as task executes."""

    async def report_started(self, skills: list[str]) -> None:
        """Report task started with skills being used."""
        await self._emit({
            "type": "task_started",
            "skills": skills,
            "timestamp": datetime.now().isoformat(),
        })

    async def report_step_progress(
        self,
        message: str,
        step: int,
        total_steps: int,
    ) -> None:
        """Report step progress."""
        await self._emit({
            "type": "step_progress",
            "message": message,
            "step": step,
            "total_steps": total_steps,
            "percent": int(step / total_steps * 100),
        })

    async def report_completed(
        self,
        summary: str,
        outputs: list[str],
        key_findings: list[str],
    ) -> None:
        """Report task completed."""
        await self._emit({
            "type": "task_completed",
            "summary": summary,
            "outputs": outputs,
            "key_findings": key_findings,
        })
```

### SSE Event Types

```python
# Event types emitted during execution
EVENT_TYPES = {
    "task_started": "Task has started with given skills",
    "step_progress": "Progress on current step",
    "tool_start": "Tool call starting",
    "tool_end": "Tool call completed",
    "worker_start": "Worker subagent starting",
    "worker_end": "Worker subagent completed",
    "text_delta": "Streaming text from model",
    "task_completed": "Task finished successfully",
    "task_failed": "Task failed with error",
    "hitl_question": "Human-in-loop question asked",
}
```

---

## 12. Agent Caching

### Cache Pattern for HITL Resume

From `aquarius/api/services/streaming.py`:

```python
@dataclass
class CachedAgent:
    """Cached agent with related objects for reuse.

    Caching agents avoids expensive recreation on:
    - HITL resume (same session, user answered question)
    - Multi-turn conversations (same session, new messages)
    """
    agent: CompiledStateGraph
    detected_mode: str
    enforcer_middleware: TaskCompletionEnforcerMiddleware | None
    checkpointer: Any
    checkpointer_ctx: Any
    created_at: float
    last_used: float
    hit_count: int = 0

    def is_expired(self, ttl: float = 30 * 60) -> bool:
        """Check if cache entry has expired (30 min default)."""
        return (time.time() - self.last_used) > ttl
```

### Cache Key Generation

```python
def _get_cache_key(
    self,
    session_id: str,
    mode: str,
    team_id: int,
    user_id: int,
    context_items: list[dict] | None = None,
    mentioned_agent: str | None = None,
) -> str:
    """Generate cache key for agent lookup.

    Includes context_items hash and mentioned_agent so agents with
    different context or @mentions are cached separately.
    """
    base_key = f"{session_id}:{mode}:{team_id}:{user_id}"

    if context_items:
        context_hash = hashlib.md5(
            json.dumps(context_items, sort_keys=True).encode()
        ).hexdigest()[:8]
        base_key = f"{base_key}:ctx_{context_hash}"

    if mentioned_agent:
        base_key = f"{base_key}:agent_{mentioned_agent}"

    return base_key
```

---

## 13. Implementation Recommendations for Libra

### Phase 3A: Core Agent Infrastructure (Week 1-2)

#### 1. Task Plan Protocol

```python
# src/libra/agents/task_plan.py
from dataclasses import dataclass
from typing import Literal

@dataclass
class TradingStep:
    id: str
    description: str
    validation: str
    completed: bool = False
    evidence: str | None = None
    dependencies: list[str] = None
    source: Literal["initializer", "worker"] = "initializer"

@dataclass
class TradingPlan:
    objective: str
    goal: str
    steps: list[TradingStep]
    task_type: Literal["analysis", "research", "execution"] = "analysis"
```

#### 2. Task Completion Enforcer for Trading

```python
# src/libra/agents/enforcer.py
class TradingEnforcerMiddleware:
    """Ensures trading analysis workflow completes.

    Tracks:
    - pending_analyses: Market analyses not yet started
    - pending_validations: Analyses awaiting risk validation
    - pending_signals: Signals awaiting execution decision
    """

    async def aafter_model(self, state, runtime):
        """Force continuation if trading workflow incomplete."""
        if self.pending_analyses:
            return self._force_analysis_continuation()
        if self.pending_validations:
            return self._force_risk_validation()
        return None
```

#### 3. Context Monitor for Market Data

```python
# src/libra/agents/context_monitor.py
class MarketContextMonitor(ContextMonitor):
    """Context monitor specialized for market data.

    Market data can be massive - need careful tracking:
    - Historical prices: Summarize, don't include raw
    - Order books: Only top N levels
    - Tick data: Aggregate to bars
    """

    MARKET_DATA_BUDGET = 50_000  # tokens for market data section
```

### Phase 3B: Trading-Specific Patterns (Week 3-4)

#### 4. Evidence-Based Trade Validation

```python
# Validation prompts for trading
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
}
```

#### 5. Trading Agent Roles

```python
# Agent definitions for trading
TRADING_AGENTS = {
    "analyst": {
        "model": "claude-sonnet-4-5-20250929",
        "can_execute_trades": False,
        "tools": ["fetch_market_data", "calculate_indicators", "web_search"],
    },
    "risk_manager": {
        "model": "claude-sonnet-4-5-20250929",
        "can_execute_trades": False,
        "tools": ["check_position_limits", "calculate_exposure", "get_portfolio"],
    },
    "executor": {
        "model": "claude-sonnet-4-5-20250929",
        "can_execute_trades": True,
        "tools": ["submit_order", "cancel_order", "get_positions"],
        "requires_approval": True,  # HITL for trade confirmation
    },
}
```

### Phase 3C: Integration Patterns (Week 5-6)

#### 6. HITL for Trade Confirmation

```python
async def confirm_trade(
    self,
    symbol: str,
    side: str,
    quantity: Decimal,
    price: Decimal,
) -> str:
    """Ask user to confirm trade before execution."""
    return await self.ask_user_question(
        f"Confirm {side} {quantity} {symbol} @ {price}?",
        options=["Execute", "Modify", "Cancel"]
    )
```

#### 7. Progress Reporting to TUI

```python
class TUIProgressReporter:
    """Reports trading agent progress to Textual TUI."""

    async def report_analysis_started(self, symbols: list[str]):
        await self.message_bus.publish(Event(
            type=EventType.AGENT_ANALYSIS_STARTED,
            data={"symbols": symbols},
        ))

    async def report_signal_generated(self, signal: Signal):
        await self.message_bus.publish(Event(
            type=EventType.AGENT_SIGNAL,
            data=signal.to_dict(),
        ))
```

### Key Implementation Order

1. **TaskPlanMiddleware** - Evidence-based completion (Week 1)
2. **TaskCompletionEnforcer** - Keep agent going (Week 1)
3. **ContextMonitor** - Track market data tokens (Week 2)
4. **TradingValidationPrompts** - Trading-specific validation (Week 2)
5. **TradingAgentRoles** - Analyst, Risk, Executor (Week 3)
6. **HITLTradeConfirmation** - User approval for trades (Week 3)
7. **TUIIntegration** - Progress reporting (Week 4)

---

## Summary: Top Patterns to Port

| Priority | Pattern | Source File | Apply To |
|----------|---------|-------------|----------|
| 1 | Evidence-Based Completion | `task_plan.py` | All trading decisions |
| 2 | Task Completion Enforcer | `task_plan.py` | Force analysis completion |
| 3 | Retry with Max Attempts | `task_plan.py` | Exchange API calls |
| 4 | Race Condition Handling | `task_plan.py` | Order state updates |
| 5 | Context Monitoring | `context_monitor.py` | Market data management |
| 6 | HITL Question Pattern | `human_in_loop.py` | Trade confirmation |
| 7 | Worker Output Format | `prompts.py` | Structured analysis results |
| 8 | Agent Caching | `streaming.py` | Multi-turn conversations |
| 9 | Dependency-Based Parallel | `task_plan.py` | Parallel data fetching |
| 10 | Termination Controls | `prompts.py` | Prevent infinite loops |

---

*Generated from deep research of `/Users/taofeng/aquarius` repository*
