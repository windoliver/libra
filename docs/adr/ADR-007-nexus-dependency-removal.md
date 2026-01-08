# ADR-007: Nexus Dependency Removal

**Status:** Accepted
**Date:** 2026-01-08
**Decision Makers:** Libra Team

## Context

The original LIBRA design document referenced Nexus infrastructure for several capabilities:

- **Plugin System** (`nexus.plugins`) for strategy adapters
- **Agent System** (`nexus.core.agents`) for trading agents
- **LangGraph Integration** (`nexus.tools.langgraph`) for multi-agent orchestration
- **Storage** (`nexus.core.nexus_fs`) for file management
- **Permissions** (`nexus.core.rebac`) for access control

Issue #22 raised the question: Should we build the Nexus infrastructure, remove it, or use existing frameworks?

## Decision

**We will use existing Python frameworks instead of building Nexus infrastructure (Option C).**

## Options Considered

### Option A: Build Nexus Infrastructure
- **Pros:** Full control, custom features
- **Cons:** Significant engineering effort (6+ months), delays core trading functionality
- **Verdict:** Rejected - not justified for a trading platform

### Option B: Remove Nexus Dependency (Simplify)
- **Pros:** Simple, fast to implement
- **Cons:** Loses plugin architecture, less extensible
- **Verdict:** Partially adopted - remove Nexus, but keep extensibility

### Option C: Use Existing Frameworks (Selected)
- **Pros:** Battle-tested, well-documented, community support
- **Cons:** Less custom control
- **Verdict:** Selected - best balance of features and effort

## Implementation

Each Nexus component is replaced with a standard Python alternative:

| Nexus Component | Replacement | Rationale |
|-----------------|-------------|-----------|
| `nexus.plugins` | `importlib` + `entry_points` | Standard Python plugin discovery |
| `nexus.core.agents` | Direct LangGraph | Already using LangGraph for orchestration |
| `nexus.tools.langgraph` | `langgraph` directly | No wrapper needed |
| `nexus.core.nexus_fs` | Local filesystem + QuestDB | Simple storage + time-series DB |
| `nexus.core.rebac` | Simple role-based checks | Trading platform doesn't need complex RBAC |

### Plugin System Design

```python
# Using entry_points for plugin discovery
# pyproject.toml
[project.entry-points."libra.strategies"]
freqtrade = "libra.plugins.freqtrade_adapter:FreqtradeAdapter"
hummingbot = "libra.plugins.hummingbot_adapter:HummingbotAdapter"

# Plugin loading
from importlib.metadata import entry_points

def load_strategy_plugins() -> dict[str, type]:
    """Load all registered strategy plugins."""
    plugins = {}
    eps = entry_points(group="libra.strategies")
    for ep in eps:
        plugins[ep.name] = ep.load()
    return plugins
```

### Agent System Design

```python
# Direct LangGraph usage without Nexus wrapper
from langgraph.graph import StateGraph

class TradingAgentOrchestrator:
    """Multi-agent trading analysis using LangGraph directly."""

    def __init__(self, llm):
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(TradingState)
        # Add nodes and edges directly
        workflow.add_node("fundamentals", self._fundamentals_agent)
        workflow.add_node("technical", self._technical_agent)
        # ... etc
        return workflow.compile()
```

### Storage Design

```python
# Local filesystem + QuestDB for time-series
from pathlib import Path
from libra.data import AsyncQuestDBClient

LIBRA_HOME = Path.home() / ".libra"

class Storage:
    """Simple local storage without NexusFS."""

    strategies_dir = LIBRA_HOME / "strategies"
    configs_dir = LIBRA_HOME / "configs"
    results_dir = LIBRA_HOME / "results"

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.strategies_dir, cls.configs_dir, cls.results_dir]:
            d.mkdir(parents=True, exist_ok=True)
```

## Consequences

### Positive
- Faster time to market (weeks instead of months)
- Standard Python patterns (easier for contributors)
- No maintenance burden for custom infrastructure
- Better documentation (standard libraries are well-documented)

### Negative
- Less flexibility than custom solution
- Multiple small dependencies instead of one unified framework

### Neutral
- Need to document plugin development process
- Agent definitions remain similar, just without Nexus registration

## Related Issues

- Issue #22: [Decision] Nexus Dependency - Build or Remove
- Aquarius codebase exploration for patterns

## References

- [Python Entry Points](https://packaging.python.org/en/latest/specifications/entry-points/)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [importlib.metadata](https://docs.python.org/3/library/importlib.metadata.html)
