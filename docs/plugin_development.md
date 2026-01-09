# Plugin Development Guide

This guide covers how to develop plugins for LIBRA using the entry_points-based plugin system.

## Overview

LIBRA uses Python's `importlib.metadata` entry_points for plugin discovery, following PEP 621. This allows:

- **Strategy Plugins**: Custom trading strategies that integrate with the framework
- **Gateway Plugins**: Exchange/broker connectors for order execution
- **Automatic Discovery**: Plugins are discovered at runtime via entry points

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TradingKernel                        │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ discover_   │  │ discover_    │  │ list_strategy │  │
│  │ strategies()│  │ gateways()   │  │ _plugins()    │  │
│  └──────┬──────┘  └──────┬───────┘  └───────────────┘  │
└─────────┼────────────────┼──────────────────────────────┘
          │                │
          ▼                ▼
┌─────────────────────────────────────────────────────────┐
│              importlib.metadata.entry_points            │
│  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │ libra.strategies    │  │ libra.gateways           │  │
│  │ - freqtrade         │  │ - paper                  │  │
│  │ - my_strategy       │  │ - ccxt                   │  │
│  └─────────────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Create a Strategy Plugin

```python
# my_package/strategies.py
from decimal import Decimal
from libra.plugins.base import PluginMetadata, StrategyPlugin


class MyAwesomeStrategy(StrategyPlugin):
    """A custom trading strategy plugin."""

    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata.create(
            name="my_awesome_strategy",
            version="1.0.0",
            description="My awesome trading strategy",
            author="Your Name",
        )

    @property
    def name(self) -> str:
        return "my_awesome_strategy"

    @property
    def symbols(self) -> list[str]:
        return ["BTC/USDT", "ETH/USDT"]

    async def initialize(self, context: dict) -> None:
        """Called when strategy is loaded."""
        self.log.info("Initializing %s", self.name)

    async def on_data(self, data: dict) -> None:
        """Called on new market data."""
        # Your strategy logic here
        pass

    async def shutdown(self) -> None:
        """Called when strategy is unloaded."""
        self.log.info("Shutting down %s", self.name)
```

### 2. Register in pyproject.toml

```toml
# pyproject.toml
[project.entry-points."libra.strategies"]
my_awesome_strategy = "my_package.strategies:MyAwesomeStrategy"
```

### 3. Use the Plugin

```python
from libra.plugins import load_strategy, discover_strategies

# Discover all strategies
strategies = discover_strategies()
print(strategies)  # {'freqtrade': FreqtradeAdapter, 'my_awesome_strategy': MyAwesomeStrategy}

# Load specific strategy
MyStrategy = load_strategy("my_awesome_strategy")
strategy = MyStrategy()
```

## Plugin Types

### Strategy Plugins

Strategy plugins extend `StrategyPlugin` and implement the trading logic interface.

**Required Methods:**

| Method | Description |
|--------|-------------|
| `metadata()` | Returns `PluginMetadata` with name, version, description |
| `name` | Property returning the strategy name |
| `symbols` | Property returning list of trading symbols |
| `initialize(context)` | Async initialization with context dict |
| `on_data(data)` | Async handler for market data |

**Optional Methods:**

| Method | Description |
|--------|-------------|
| `on_tick(tick)` | Handler for tick data |
| `on_fill(fill)` | Handler for order fills |
| `backtest(config)` | Run strategy backtests |
| `optimize(config)` | Run parameter optimization |
| `shutdown()` | Cleanup on unload |

**Example with All Methods:**

```python
class CompleteStrategy(StrategyPlugin):
    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata.create(
            name="complete_strategy",
            version="2.0.0",
            description="A complete strategy example",
            author="LIBRA Team",
            requires=["numpy>=1.20", "pandas>=2.0"],
        )

    @property
    def name(self) -> str:
        return "complete_strategy"

    @property
    def symbols(self) -> list[str]:
        return self._config.get("symbols", ["BTC/USDT"])

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._positions = {}

    async def initialize(self, context: dict) -> None:
        self.bus = context.get("bus")
        self.cache = context.get("cache")
        self.log.info("Initialized with context keys: %s", list(context.keys()))

    async def on_data(self, data: dict) -> None:
        symbol = data.get("symbol")
        price = data.get("price")
        # Trading logic here

    async def on_tick(self, tick: dict) -> None:
        # Real-time tick handling
        pass

    async def on_fill(self, fill: dict) -> None:
        # Order fill handling
        self._positions[fill["symbol"]] = fill["quantity"]

    async def backtest(self, config: dict) -> dict:
        # Run backtest and return results
        return {"sharpe": 1.5, "max_dd": 0.15}

    async def optimize(self, config: dict) -> dict:
        # Parameter optimization
        return {"best_params": {"fast": 10, "slow": 20}}

    async def shutdown(self) -> None:
        self.log.info("Shutting down, closing %d positions", len(self._positions))
```

### Gateway Plugins

Gateway plugins provide exchange/broker connectivity.

**Entry Point Group:** `libra.gateways`

**Registration:**

```toml
[project.entry-points."libra.gateways"]
my_exchange = "my_package.gateways:MyExchangeGateway"
```

**Usage:**

```python
from libra.plugins import load_gateway, discover_gateways

# Discover all gateways
gateways = discover_gateways()
print(gateways)  # {'paper': PaperGateway, 'ccxt': CCXTGateway, 'my_exchange': MyExchangeGateway}

# Load specific gateway
MyGateway = load_gateway("my_exchange")
gateway = MyGateway(config)
```

## Plugin Metadata

The `PluginMetadata` class provides plugin information:

```python
from libra.plugins.base import PluginMetadata

@dataclass
class PluginMetadata:
    name: str           # Plugin identifier
    version: str        # Semantic version (e.g., "1.0.0")
    description: str    # Human-readable description
    author: str         # Author name/email
    requires: list[str] # Optional dependencies
```

**Creating Metadata:**

```python
# Using the factory method (recommended)
meta = PluginMetadata.create(
    name="my_plugin",
    version="1.0.0",
    description="My awesome plugin",
    author="Your Name <you@example.com>",
    requires=["numpy>=1.20"],
)

# Direct construction
meta = PluginMetadata(
    name="my_plugin",
    version="1.0.0",
    description="My awesome plugin",
    author="Your Name",
    requires=(),  # Tuple, not list (immutable)
)
```

## Kernel Integration

The `TradingKernel` automatically discovers plugins at startup:

```python
from libra.core.kernel import TradingKernel, KernelConfig

# Configure plugin discovery
config = KernelConfig(
    discover_plugins=True,   # Enable plugin discovery (default)
    log_plugins=True,        # Log discovered plugins (default)
)

kernel = TradingKernel(config)

# Manual discovery
plugins = kernel.discover_plugins()
print(plugins["strategies"])  # Strategy plugins
print(plugins["gateways"])    # Gateway plugins

# Get specific plugin types
gateways = kernel.get_available_gateways()
strategies = kernel.get_available_strategies()
```

## Publishing Your Plugin

### 1. Package Structure

```
my_libra_plugin/
├── pyproject.toml
├── src/
│   └── my_plugin/
│       ├── __init__.py
│       └── strategy.py
└── tests/
    └── test_strategy.py
```

### 2. pyproject.toml

```toml
[project]
name = "my-libra-plugin"
version = "1.0.0"
description = "A custom LIBRA strategy plugin"
dependencies = [
    "libra>=0.1.0",
]

[project.entry-points."libra.strategies"]
my_strategy = "my_plugin.strategy:MyStrategy"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 3. Install and Test

```bash
# Install in development mode
pip install -e .

# Verify discovery
python -c "from libra.plugins import discover_strategies; print(discover_strategies())"
```

### 4. Publish to PyPI

```bash
# Build
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

## Best Practices

### 1. Version Your Plugins

Use semantic versioning (SemVer):
- MAJOR: Breaking changes
- MINOR: New features, backwards compatible
- PATCH: Bug fixes

### 2. Handle Dependencies

Declare dependencies in `PluginMetadata.requires`:

```python
@classmethod
def metadata(cls) -> PluginMetadata:
    return PluginMetadata.create(
        name="my_strategy",
        version="1.0.0",
        description="Strategy with numpy",
        requires=["numpy>=1.20", "scipy>=1.7"],
    )
```

### 3. Graceful Degradation

Handle missing optional dependencies:

```python
async def initialize(self, context: dict) -> None:
    try:
        import numpy as np
        self._use_numpy = True
    except ImportError:
        self.log.warning("NumPy not available, using fallback")
        self._use_numpy = False
```

### 4. Logging

Use the provided logger:

```python
class MyStrategy(StrategyPlugin):
    def __init__(self):
        import logging
        self.log = logging.getLogger(f"libra.plugins.{self.name}")

    async def on_data(self, data: dict) -> None:
        self.log.debug("Received data: %s", data)
```

### 5. Testing

Test your plugin independently:

```python
# tests/test_my_strategy.py
import pytest
from my_plugin.strategy import MyStrategy

@pytest.mark.asyncio
async def test_strategy_initialization():
    strategy = MyStrategy()
    await strategy.initialize({})
    assert strategy.name == "my_strategy"

@pytest.mark.asyncio
async def test_strategy_on_data():
    strategy = MyStrategy()
    await strategy.initialize({})
    await strategy.on_data({"symbol": "BTC/USDT", "price": 50000})
    # Assert expected behavior
```

## API Reference

### libra.plugins Module

```python
# Discovery
discover_strategies() -> dict[str, type[StrategyPlugin]]
discover_gateways() -> dict[str, type]

# Loading
load_strategy(name: str) -> type[StrategyPlugin]
load_gateway(name: str) -> type

# Metadata
list_strategy_plugins() -> list[PluginMetadata]
```

### Entry Point Groups

| Group | Description |
|-------|-------------|
| `libra.strategies` | Strategy plugins |
| `libra.gateways` | Gateway plugins |

## Troubleshooting

### Plugin Not Discovered

1. Ensure package is installed (`pip install -e .`)
2. Check entry point syntax in pyproject.toml
3. Verify import path is correct
4. Check for import errors in your module

```python
# Debug discovery
import logging
logging.basicConfig(level=logging.DEBUG)

from libra.plugins import discover_strategies
strategies = discover_strategies()
```

### Import Errors

If a plugin fails to load, check the logs:

```
WARNING:libra.plugins.loader:Failed to load plugin my_strategy: ModuleNotFoundError: No module named 'missing_dep'
```

### Version Conflicts

Ensure your plugin's dependencies don't conflict with LIBRA's:

```bash
pip check
```

## See Also

- [Strategy Development Guide](strategy_development.md)
- [PEP 621 - Project Metadata](https://peps.python.org/pep-621/)
- [importlib.metadata Documentation](https://docs.python.org/3/library/importlib.metadata.html)
