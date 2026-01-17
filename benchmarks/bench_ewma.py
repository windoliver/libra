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

from libra.risk.var import (
    VaRCalculator,
    VaRConfig,
    VaRMethod,
    _ewma_volatility_numba,
    _var_backtest_loop_historical,
)


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


class TestVaRBacktestBenchmark:
    """Benchmark VaR backtest loop (Issue #74)."""

    @pytest.fixture
    def backtest_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample data for backtest benchmark."""
        rng = np.random.default_rng(42)
        n_obs = 1000  # ~4 years of daily data
        returns = rng.normal(0.0005, 0.02, n_obs)
        portfolio_values = 100000 * np.cumprod(1 + returns)
        return returns.astype(np.float64), portfolio_values.astype(np.float64)

    def _backtest_loop_python(
        self,
        returns: np.ndarray,
        portfolio_values: np.ndarray,
        window: int,
        confidence: float,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Pure Python backtest loop (baseline)."""
        n = len(returns) - window
        var_estimates = np.empty(n, dtype=np.float64)
        actual_losses = np.empty(n, dtype=np.float64)
        percentile_level = (1.0 - confidence) * 100.0
        exceptions = 0

        for i in range(window, len(returns)):
            idx = i - window
            window_returns = returns[i - window : i]
            port_value = portfolio_values[i - 1]

            var_pct = np.percentile(window_returns, percentile_level)
            var_estimate = abs(var_pct) * port_value
            var_estimates[idx] = var_estimate

            actual_loss = -returns[i] * port_value
            actual_losses[idx] = actual_loss

            if actual_loss > var_estimate:
                exceptions += 1

        return var_estimates, actual_losses, exceptions

    @pytest.mark.benchmark
    def test_backtest_comparison(
        self, gc_disabled: None, backtest_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Compare Numba JIT vs Python backtest loop (Issue #74)."""
        returns, portfolio_values = backtest_data
        window = 50
        confidence = 0.95
        num_iterations = 50

        # Warmup Numba
        _var_backtest_loop_historical(returns, portfolio_values, window, confidence)

        # Python benchmark
        python_start = time.perf_counter()
        for _ in range(num_iterations):
            python_result = self._backtest_loop_python(
                returns, portfolio_values, window, confidence
            )
        python_duration = time.perf_counter() - python_start

        # Numba benchmark
        numba_start = time.perf_counter()
        for _ in range(num_iterations):
            numba_result = _var_backtest_loop_historical(
                returns, portfolio_values, window, confidence
            )
        numba_duration = time.perf_counter() - numba_start

        speedup = python_duration / numba_duration if numba_duration > 0 else 0

        print(f"""
================================================================================
           VaR Backtest Loop Performance Comparison (Issue #74)
================================================================================

Configuration:
  Observations:  {len(returns)}
  Window size:   {window}
  Confidence:    {confidence}
  Iterations:    {num_iterations}

Results:
                       Numba JIT          Pure Python        Speedup
  -----------------------------------------------------------------------
  Duration (sec):      {numba_duration:>12.4f}      {python_duration:>12.4f}        {speedup:>6.1f}x faster
  Per call (ms):       {numba_duration / num_iterations * 1000:>12.3f}      {python_duration / num_iterations * 1000:>12.3f}
  Calls/sec:           {num_iterations / numba_duration:>12,.1f}      {num_iterations / python_duration:>12,.1f}

Accuracy check:
  Python exceptions:  {python_result[2]}
  Numba exceptions:   {numba_result[2]}
  VaR diff (max):     {np.max(np.abs(python_result[0] - numba_result[0])):.2e}

Verdict: Numba JIT is {speedup:.1f}x faster than pure Python
================================================================================
""")

        # Verify results match
        assert python_result[2] == numba_result[2], "Exception counts should match"
        assert np.allclose(
            python_result[0], numba_result[0], rtol=1e-10
        ), "VaR estimates should match"
        assert speedup >= 2, f"Numba should be at least 2x faster (got {speedup:.1f}x)"

    @pytest.mark.benchmark
    def test_backtest_integration(
        self, backtest_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test VaRCalculator.backtest_var uses JIT function."""
        returns, portfolio_values = backtest_data

        calculator = VaRCalculator()

        # Run backtest with HISTORICAL method (uses JIT)
        result = calculator.backtest_var(
            returns,
            portfolio_values,
            method=VaRMethod.HISTORICAL,
            window_size=50,
        )

        print(f"""
VaRCalculator.backtest_var Integration Test
==========================================
  Method:          HISTORICAL (JIT-accelerated)
  Observations:    {result["n_observations"]}
  Exceptions:      {result["exceptions"]}
  Exception rate:  {result["exception_rate"]:.4f}
  Expected rate:   {result["expected_rate"]:.4f}
  Kupiec p-value:  {result["kupiec_pvalue"]:.4f}
  Model adequate:  {result["model_adequate"]}
""")
