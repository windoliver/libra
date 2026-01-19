# LIBRA Core Rust

High-performance Rust computational kernels for the LIBRA trading platform.

## Overview

This crate provides Rust implementations of performance-critical algorithms with Python bindings via PyO3. It achieves 10-100x speedup for computational hot paths.

## Features

- **EWMA Calculations**: Exponentially weighted moving average for volatility and mean
- **Correlation/Covariance**: Fast matrix computations with optional parallelism
- **Value at Risk**: Parametric and historical VaR calculations
- **Message Bus**: High-performance event distribution system

## Installation

### Development Build

```bash
cd libra-core-rs
maturin develop --release
```

### Production Build

```bash
maturin build --release
pip install target/wheels/libra_core_rs-*.whl
```

## Usage

```python
from libra_core_rs import ewma_volatility, correlation_matrix, parametric_var
import numpy as np

# EWMA volatility
returns = np.random.randn(252) * 0.02
vol = ewma_volatility(returns, span=20, annualize=True)

# Correlation matrix
returns_matrix = np.random.randn(100, 5) * 0.02
corr = correlation_matrix(returns_matrix)

# Parametric VaR
var = parametric_var(returns, confidence=0.95, portfolio_value=100000.0)
```

## Benchmarks

Run benchmarks with:

```bash
cargo bench
```

## Development

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Python 3.11+
- maturin (`pip install maturin`)

### Building

```bash
# Development build
maturin develop

# Release build
maturin build --release

# Run tests
cargo test
```

### Project Structure

```
libra-core-rs/
├── Cargo.toml          # Rust dependencies
├── pyproject.toml      # Python/maturin config
├── src/
│   ├── lib.rs          # Main module + PyO3 bindings
│   ├── ewma.rs         # EWMA calculations
│   ├── correlation.rs  # Correlation/covariance
│   ├── var.rs          # Value at Risk
│   └── message_bus.rs  # Message bus
└── python/
    └── libra_core_rs/
        └── __init__.py # Python package
```

## License

MIT License - see [LICENSE](../LICENSE) for details.
