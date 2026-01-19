"""
Pipeline API for Vectorized Factor Computation.

Provides Zipline-style pipeline for computing multiple factors in a single
vectorized pass over historical data. Achieves significant speedup over
event-driven backtesting.

Usage:
    from libra.backtest.pipeline import Pipeline
    from libra.backtest.factors import MomentumFactor, RSIFactor

    pipeline = Pipeline([
        MomentumFactor(window=20),
        RSIFactor(window=14),
    ])

    # Compute all factors at once
    result = pipeline.run(historical_data)

    # Access factor values
    momentum = result["MomentumFactor"]
    rsi = result["RSIFactor"]

See: https://github.com/windoliver/libra/issues/93
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import polars as pl

from libra.backtest.factors import Factor

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Result
# =============================================================================


@dataclass
class PipelineResult:
    """Result of running a pipeline.

    Attributes:
        data: DataFrame with all computed factors
        factor_names: List of factor column names
        metadata: Pipeline execution metadata
    """

    data: pl.DataFrame
    factor_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> pl.Series:
        """Get factor by name."""
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        """Check if factor exists."""
        return key in self.factor_names

    def to_dict(self) -> dict[str, list[Any]]:
        """Convert to dictionary of lists."""
        return self.data.to_dict(as_series=False)

    @property
    def shape(self) -> tuple[int, int]:
        """Get result shape (rows, columns)."""
        return self.data.shape

    def get_factor_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for each factor."""
        stats = {}
        for name in self.factor_names:
            series = self.data[name]
            stats[name] = {
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "null_count": series.null_count(),
            }
        return stats


# =============================================================================
# Pipeline Configuration
# =============================================================================


@dataclass
class PipelineConfig:
    """Pipeline configuration.

    Attributes:
        compute_parallel: Whether to compute factors in parallel
        drop_nulls: Whether to drop rows with null factor values
        validate_inputs: Whether to validate factor inputs
        warmup_period: Number of initial rows to exclude (for window warmup)
    """

    compute_parallel: bool = False  # Polars handles parallelism internally
    drop_nulls: bool = False
    validate_inputs: bool = True
    warmup_period: int | None = None  # Auto-calculated from max window


# =============================================================================
# Pipeline Class
# =============================================================================


class Pipeline:
    """Orchestrates vectorized factor computation.

    The Pipeline class manages multiple factors and computes them efficiently
    in a single pass over historical data using Polars vectorized operations.

    Example:
        pipeline = Pipeline([
            MomentumFactor(window=20),
            MeanReversionFactor(window=10),
            VolatilityFactor(window=30),
        ])

        # Run on historical data
        result = pipeline.run(ohlcv_data)

        # Access factors
        print(result.data)
        print(result.get_factor_stats())

    Features:
        - Vectorized computation (no Python loops over data)
        - Automatic input validation
        - Factor dependency resolution (coming soon)
        - Built-in performance timing
    """

    def __init__(
        self,
        factors: Sequence[Factor] | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            factors: List of factors to compute
            config: Pipeline configuration
        """
        self._factors: list[Factor] = list(factors) if factors else []
        self._config = config or PipelineConfig()
        self._hooks: list[Callable[[PipelineResult], None]] = []

    @property
    def factors(self) -> list[Factor]:
        """Get list of factors."""
        return self._factors.copy()

    @property
    def factor_names(self) -> list[str]:
        """Get list of factor names."""
        return [f.name for f in self._factors]

    def add_factor(self, factor: Factor) -> "Pipeline":
        """Add a factor to the pipeline.

        Args:
            factor: Factor to add

        Returns:
            Self for method chaining
        """
        self._factors.append(factor)
        return self

    def add_factors(self, factors: Sequence[Factor]) -> "Pipeline":
        """Add multiple factors to the pipeline.

        Args:
            factors: Factors to add

        Returns:
            Self for method chaining
        """
        self._factors.extend(factors)
        return self

    def remove_factor(self, name: str) -> "Pipeline":
        """Remove a factor by name.

        Args:
            name: Factor name to remove

        Returns:
            Self for method chaining
        """
        self._factors = [f for f in self._factors if f.name != name]
        return self

    def clear_factors(self) -> "Pipeline":
        """Remove all factors.

        Returns:
            Self for method chaining
        """
        self._factors.clear()
        return self

    def add_hook(self, hook: Callable[[PipelineResult], None]) -> "Pipeline":
        """Add a post-computation hook.

        Args:
            hook: Callback function receiving PipelineResult

        Returns:
            Self for method chaining
        """
        self._hooks.append(hook)
        return self

    def run(
        self,
        data: pl.DataFrame,
        *,
        include_data: bool = True,
    ) -> PipelineResult:
        """Run pipeline on historical data.

        Args:
            data: Input DataFrame with OHLCV data
            include_data: Whether to include original data columns in result

        Returns:
            PipelineResult with computed factors

        Raises:
            ValueError: If no factors registered or data is empty
        """
        if not self._factors:
            raise ValueError("No factors registered in pipeline")

        if data.is_empty():
            raise ValueError("Input data is empty")

        start_time = time.perf_counter()
        computed_columns: list[pl.Series] = []
        factor_timings: dict[str, float] = {}

        # Validate all inputs first
        if self._config.validate_inputs:
            self._validate_all_inputs(data)

        # Compute each factor
        for factor in self._factors:
            factor_start = time.perf_counter()

            try:
                result = factor.compute(data)

                # Handle both Series and DataFrame returns
                if isinstance(result, pl.DataFrame):
                    for col in result.columns:
                        computed_columns.append(result[col])
                else:
                    computed_columns.append(result)

                factor_timings[factor.name] = time.perf_counter() - factor_start

            except Exception as e:
                logger.error("Factor '%s' failed: %s", factor.name, e)
                raise RuntimeError(
                    f"Factor '{factor.name}' computation failed: {e}"
                ) from e

        # Build result DataFrame
        if include_data:
            result_df = data.clone()
            for col in computed_columns:
                result_df = result_df.with_columns(col)
        else:
            result_df = pl.DataFrame(computed_columns)

        # Apply warmup period
        warmup = self._get_warmup_period()
        if warmup > 0 and warmup < len(result_df):
            result_df = result_df.slice(warmup)

        # Drop nulls if configured
        if self._config.drop_nulls:
            result_df = result_df.drop_nulls(subset=self.factor_names)

        total_time = time.perf_counter() - start_time

        # Build metadata
        metadata = {
            "total_time_seconds": total_time,
            "factor_timings": factor_timings,
            "input_rows": len(data),
            "output_rows": len(result_df),
            "warmup_period": warmup,
            "factor_count": len(self._factors),
        }

        result = PipelineResult(
            data=result_df,
            factor_names=self.factor_names,
            metadata=metadata,
        )

        # Run hooks
        for hook in self._hooks:
            try:
                hook(result)
            except Exception as e:
                logger.warning("Pipeline hook failed: %s", e)

        logger.debug(
            "Pipeline computed %d factors on %d rows in %.3fs",
            len(self._factors),
            len(data),
            total_time,
        )

        return result

    def _validate_all_inputs(self, data: pl.DataFrame) -> None:
        """Validate that all factor inputs exist."""
        all_inputs: set[str] = set()
        for factor in self._factors:
            all_inputs.update(factor.inputs)

        missing = all_inputs - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _get_warmup_period(self) -> int:
        """Calculate warmup period from max factor window."""
        if self._config.warmup_period is not None:
            return self._config.warmup_period

        if not self._factors:
            return 0

        return max(f.window for f in self._factors)

    def describe(self) -> dict[str, Any]:
        """Get pipeline description."""
        return {
            "factor_count": len(self._factors),
            "factors": [
                {
                    "name": f.name,
                    "class": f.__class__.__name__,
                    "window": f.window,
                    "inputs": f.inputs,
                }
                for f in self._factors
            ],
            "config": {
                "compute_parallel": self._config.compute_parallel,
                "drop_nulls": self._config.drop_nulls,
                "validate_inputs": self._config.validate_inputs,
                "warmup_period": self._config.warmup_period,
            },
        }

    def __repr__(self) -> str:
        factor_str = ", ".join(f.name for f in self._factors)
        return f"Pipeline([{factor_str}])"

    def __len__(self) -> int:
        return len(self._factors)

    def __iter__(self):
        return iter(self._factors)


# =============================================================================
# Convenience Functions
# =============================================================================


def run_factors(
    data: pl.DataFrame,
    factors: Sequence[Factor],
    **kwargs: Any,
) -> PipelineResult:
    """Run factors on data without creating a Pipeline object.

    Args:
        data: Input DataFrame
        factors: Factors to compute
        **kwargs: Additional arguments for Pipeline.run()

    Returns:
        PipelineResult with computed factors
    """
    return Pipeline(factors).run(data, **kwargs)


def compute_factor(
    data: pl.DataFrame,
    factor: Factor,
) -> pl.Series:
    """Compute a single factor.

    Args:
        data: Input DataFrame
        factor: Factor to compute

    Returns:
        Series with factor values
    """
    factor.validate_inputs(data)
    return factor.compute(data)
