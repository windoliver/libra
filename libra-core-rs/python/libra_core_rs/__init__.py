"""
LIBRA Core Rust Extensions.

High-performance computational kernels implemented in Rust with PyO3 bindings.
Provides 10-100x speedup for hot paths in the trading platform.

Available Functions:
    ewma_volatility: Calculate EWMA volatility from returns
    ewma_mean: Calculate EWMA mean from values
    correlation_matrix: Calculate correlation matrix
    covariance_matrix: Calculate covariance matrix
    parametric_var: Calculate parametric VaR
    historical_var: Calculate historical VaR

Available Classes:
    MessageBus: High-performance message bus for events

Example:
    >>> from libra_core_rs import ewma_volatility, correlation_matrix
    >>> import numpy as np
    >>>
    >>> # Calculate EWMA volatility
    >>> returns = np.random.randn(252) * 0.02
    >>> vol = ewma_volatility(returns, span=20, annualize=True)
    >>>
    >>> # Calculate correlation matrix
    >>> returns_matrix = np.random.randn(100, 5) * 0.02
    >>> corr = correlation_matrix(returns_matrix)

Build Instructions:
    cd libra-core-rs
    maturin develop --release

See: https://github.com/windoliver/libra/issues/96
"""

from __future__ import annotations

try:
    from libra_core_rs.libra_core_rs import (
        __version__,
        correlation_matrix,
        covariance_matrix,
        ewma_mean,
        ewma_volatility,
        historical_var,
        MessageBus,
        parametric_var,
    )

    __all__ = [
        "__version__",
        # EWMA functions
        "ewma_volatility",
        "ewma_mean",
        # Correlation functions
        "correlation_matrix",
        "covariance_matrix",
        # VaR functions
        "parametric_var",
        "historical_var",
        # Message bus
        "MessageBus",
    ]

except ImportError:
    # Rust extension not built yet - provide stub for documentation
    import warnings

    warnings.warn(
        "libra_core_rs Rust extension not built. "
        "Run 'maturin develop --release' in libra-core-rs/ to build.",
        ImportWarning,
        stacklevel=2,
    )

    __version__ = "0.1.0-stub"
    __all__ = ["__version__"]
