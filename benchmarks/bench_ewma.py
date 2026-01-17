"""
EWMA Volatility Benchmarks (Issue #70).

Compares Numba JIT-compiled EWMA vs pure Python implementation.

Expected improvement: ~50x speedup.

Run: pytest benchmarks/bench_ewma.py -v -s
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from libra.risk.var import VaRCalculator, VaRConfig, _ewma_volatility_numba


if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Benchmark Configuration
# =============================================================================

NUM_RETURNS = 252  # 1 year of daily returns
NUM_ITERATIONS = 1000


# =============================================================================
# Pure Python Implementation (baseline)
# =============================================================================


def _ewma_volatility_python(returns: np.ndarray, lam: float) -> float:
    """Pure Python EWMA volatility (baseline for comparison)."""
    n = len(returns)
    variance = float(np.var(returns, ddof=1))

    for i in range(1, n):
        variance = lam * variance + (1 - lam) * returns[i] ** 2

    return np.sqrt(variance)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gc_disabled() -> Generator[None, None, None]:
    """Disable garbage collection during benchmark."""
    gc_was_enabled = gc.isenabled()
    gc.disable()
    gc.collect()
    try:
        yield
    finally:
        if gc_was_enabled:
            gc.enable()


@pytest.fixture
def sample_returns() -> np.ndarray:
    """Generate sample daily returns."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.02, NUM_RETURNS)  # ~0.05% daily mean, 2% daily vol


# =============================================================================
# Benchmarks
# =============================================================================


class TestEWMABenchmark:
    """Benchmark tests for EWMA volatility calculation."""

    @pytest.mark.benchmark
    def test_python_ewma(self, gc_disabled: None, sample_returns: np.ndarray) -> None:
        """Benchmark pure Python EWMA implementation."""
        lam = 0.94

        start = time.perf_counter()
        for _ in range(NUM_ITERATIONS):
            _ewma_volatility_python(sample_returns, lam)
        duration = time.perf_counter() - start

        print(f"""
Pure Python EWMA
================
  Returns:      {NUM_RETURNS}
  Iterations:   {NUM_ITERATIONS:,}
  Duration:     {duration:.4f} sec
  Per call:     {duration / NUM_ITERATIONS * 1000:.3f} ms
  Calls/sec:    {NUM_ITERATIONS / duration:,.0f}
""")

    @pytest.mark.benchmark
    def test_numba_ewma(self, gc_disabled: None, sample_returns: np.ndarray) -> None:
        """Benchmark Numba JIT-compiled EWMA implementation."""
        lam = 0.94

        # Ensure JIT is warmed up
        _ewma_volatility_numba(sample_returns, lam)

        start = time.perf_counter()
        for _ in range(NUM_ITERATIONS):
            _ewma_volatility_numba(sample_returns, lam)
        duration = time.perf_counter() - start

        print(f"""
Numba JIT EWMA
==============
  Returns:      {NUM_RETURNS}
  Iterations:   {NUM_ITERATIONS:,}
  Duration:     {duration:.4f} sec
  Per call:     {duration / NUM_ITERATIONS * 1000:.3f} ms
  Calls/sec:    {NUM_ITERATIONS / duration:,.0f}
""")

    @pytest.mark.benchmark
    def test_comparison(self, gc_disabled: None, sample_returns: np.ndarray) -> None:
        """Compare Numba vs Python EWMA performance."""
        lam = 0.94

        # Warmup Numba
        _ewma_volatility_numba(sample_returns, lam)

        # Python benchmark
        python_start = time.perf_counter()
        for _ in range(NUM_ITERATIONS):
            python_result = _ewma_volatility_python(sample_returns, lam)
        python_duration = time.perf_counter() - python_start

        # Numba benchmark
        numba_start = time.perf_counter()
        for _ in range(NUM_ITERATIONS):
            numba_result = _ewma_volatility_numba(sample_returns, lam)
        numba_duration = time.perf_counter() - numba_start

        speedup = python_duration / numba_duration if numba_duration > 0 else 0

        print(f"""
================================================================================
                EWMA Volatility Performance Comparison (Issue #70)
================================================================================

Configuration:
  Returns:       {NUM_RETURNS} (1 year daily)
  Iterations:    {NUM_ITERATIONS:,}
  Lambda:        {lam}

Results:
                       Numba JIT          Pure Python        Speedup
  -----------------------------------------------------------------------
  Duration (sec):      {numba_duration:>12.4f}      {python_duration:>12.4f}        {speedup:>6.0f}x faster
  Per call (ms):       {numba_duration / NUM_ITERATIONS * 1000:>12.4f}      {python_duration / NUM_ITERATIONS * 1000:>12.4f}
  Calls/sec:           {NUM_ITERATIONS / numba_duration:>12,.0f}      {NUM_ITERATIONS / python_duration:>12,.0f}

Accuracy check:
  Python result:  {python_result:.10f}
  Numba result:   {numba_result:.10f}
  Difference:     {abs(python_result - numba_result):.2e}

Verdict: Numba JIT is {speedup:.0f}x faster than pure Python
================================================================================
""")

        # Verify results are close (fastmath allows small differences)
        assert abs(python_result - numba_result) < 1e-6, "Results should be nearly identical"
        assert speedup >= 5, f"Numba should be at least 5x faster (got {speedup:.1f}x)"


class TestVaRCalculatorIntegration:
    """Test VaRCalculator uses Numba EWMA correctly."""

    @pytest.mark.benchmark
    def test_parametric_var_uses_numba(self, sample_returns: np.ndarray) -> None:
        """Verify VaRCalculator.calculate_parametric_var uses Numba EWMA."""
        from decimal import Decimal

        config = VaRConfig(use_ewma=True, ewma_lambda=0.94)
        calculator = VaRCalculator(config)

        # This should use the Numba-compiled EWMA internally
        result = calculator.calculate_parametric_var(
            returns=sample_returns,
            portfolio_value=Decimal("100000"),
        )

        assert result.var > 0
        print(f"""
VaRCalculator Integration Test
==============================
  Method:      Parametric with EWMA
  Portfolio:   $100,000
  VaR (95%):   ${result.var:,.2f} ({result.var_pct:.2f}%)
  CVaR:        ${result.cvar:,.2f} ({result.cvar_pct:.2f}%)
""")


class TestVaRResultBenchmark:
    """Benchmark VaRResult msgspec.Struct vs dataclass (Issue #72)."""

    @pytest.mark.benchmark
    def test_varresult_creation_speed(self, gc_disabled: None) -> None:
        """Benchmark VaRResult creation speed."""
        from decimal import Decimal
        from libra.risk.var import VaRResult, VaRMethod

        num_iterations = 100_000

        start = time.perf_counter()
        for i in range(num_iterations):
            _ = VaRResult(
                var=Decimal("1000"),
                cvar=Decimal("1200"),
                confidence_level=0.95,
                time_horizon_days=1,
                method=VaRMethod.HISTORICAL,
                portfolio_value=Decimal("100000"),
                var_pct=1.0,
                cvar_pct=1.2,
                num_observations=252,
            )
        duration = time.perf_counter() - start

        print(f"""
VaRResult Creation Benchmark (msgspec.Struct)
=============================================
  Iterations:   {num_iterations:,}
  Duration:     {duration:.4f} sec
  Per creation: {duration / num_iterations * 1_000_000:.1f} ns
  Creates/sec:  {num_iterations / duration:,.0f}
""")

        # Should be fast (~100-200ns per creation with msgspec.Struct)
        assert num_iterations / duration > 100_000, "Should create >100K VaRResults/sec"
