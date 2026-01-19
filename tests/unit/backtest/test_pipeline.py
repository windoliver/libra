"""Tests for Pipeline API (Issue #93)."""

from __future__ import annotations

import polars as pl
import pytest

from libra.backtest.factors import (
    BollingerBandFactor,
    Factor,
    FactorMeta,
    LogReturnsFactor,
    MeanReversionFactor,
    MomentumFactor,
    RankFactor,
    ReturnsFactor,
    RSIFactor,
    VolatilityFactor,
    VWAPDeviationFactor,
    get_factor,
    list_factors,
    register_factor,
)
from libra.backtest.pipeline import (
    Pipeline,
    PipelineConfig,
    PipelineResult,
    compute_factor,
    run_factors,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_ohlcv() -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    import random

    random.seed(42)

    n = 100
    base_price = 100.0
    prices = [base_price]
    for _ in range(n - 1):
        change = random.gauss(0, 0.02)
        prices.append(prices[-1] * (1 + change))

    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1),
                pl.datetime(2024, 4, 10),
                interval="1d",
                eager=True,
            )[:n],
            "open": [p * 0.999 for p in prices],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [random.randint(1000, 10000) for _ in range(n)],
        }
    )


@pytest.fixture
def minimal_data() -> pl.DataFrame:
    """Create minimal data for simple tests."""
    return pl.DataFrame(
        {
            "close": [100.0, 102.0, 101.0, 105.0, 103.0],
            "volume": [1000, 1100, 900, 1200, 1000],
        }
    )


# =============================================================================
# Factor Base Class Tests
# =============================================================================


class TestFactorBase:
    """Tests for Factor base class."""

    def test_abstract_compute(self):
        """Test that Factor is abstract."""
        with pytest.raises(TypeError):
            Factor()  # Can't instantiate abstract class

    def test_custom_window(self):
        """Test custom window parameter."""
        factor = MomentumFactor(window=30)
        assert factor.window == 30

    def test_custom_name(self):
        """Test custom name parameter."""
        factor = MomentumFactor(name="my_momentum")
        assert factor.name == "my_momentum"

    def test_default_name_is_class_name(self):
        """Test default name is class name."""
        factor = MomentumFactor()
        assert factor.name == "MomentumFactor"

    def test_validate_inputs_success(self, minimal_data: pl.DataFrame):
        """Test input validation passes with valid data."""
        factor = MomentumFactor()
        factor.validate_inputs(minimal_data)  # Should not raise

    def test_validate_inputs_missing_column(self, minimal_data: pl.DataFrame):
        """Test input validation fails with missing column."""
        factor = VWAPDeviationFactor()  # Requires 'close' and 'volume'
        data_no_volume = minimal_data.select("close")

        with pytest.raises(ValueError, match="missing required columns"):
            factor.validate_inputs(data_no_volume)

    def test_get_metadata(self):
        """Test factor metadata."""
        factor = MomentumFactor(window=25)
        meta = factor.get_metadata()

        assert isinstance(meta, FactorMeta)
        assert meta.name == "MomentumFactor"
        assert meta.window == 25
        assert "close" in meta.inputs

    def test_repr(self):
        """Test string representation."""
        factor = MomentumFactor(window=15, name="mom_15")
        repr_str = repr(factor)
        assert "MomentumFactor" in repr_str
        assert "window=15" in repr_str
        assert "mom_15" in repr_str


# =============================================================================
# Common Factor Tests
# =============================================================================


class TestMomentumFactor:
    """Tests for MomentumFactor."""

    def test_compute(self, sample_ohlcv: pl.DataFrame):
        """Test momentum computation."""
        factor = MomentumFactor(window=5)
        result = factor.compute(sample_ohlcv)

        assert isinstance(result, pl.Series)
        assert result.name == "MomentumFactor"
        assert len(result) == len(sample_ohlcv)
        # First 5 values should be null (window warmup)
        assert result[:5].null_count() == 5

    def test_momentum_direction(self):
        """Test momentum reflects price direction."""
        data = pl.DataFrame({"close": [100.0, 110.0, 120.0, 130.0, 140.0]})
        factor = MomentumFactor(window=1)
        result = factor.compute(data)

        # All positive momentum (price increasing)
        assert all(v > 0 for v in result[1:].to_list())


class TestMeanReversionFactor:
    """Tests for MeanReversionFactor."""

    def test_compute(self, sample_ohlcv: pl.DataFrame):
        """Test mean reversion computation."""
        factor = MeanReversionFactor(window=10)
        result = factor.compute(sample_ohlcv)

        assert isinstance(result, pl.Series)
        assert len(result) == len(sample_ohlcv)

    def test_z_score_properties(self):
        """Test z-score has expected properties."""
        # Constant price should give z-score of NaN (0/0)
        data = pl.DataFrame({"close": [100.0] * 20})
        factor = MeanReversionFactor(window=10)
        result = factor.compute(data)

        # Should be NaN (0/0) for constant prices - std is 0
        # NaN values are not null in Polars, so check for is_nan
        non_null_values = result[10:]
        assert non_null_values.is_nan().all()


class TestVolatilityFactor:
    """Tests for VolatilityFactor."""

    def test_compute_annualized(self, sample_ohlcv: pl.DataFrame):
        """Test annualized volatility computation."""
        factor = VolatilityFactor(window=20, annualize=True)
        result = factor.compute(sample_ohlcv)

        assert isinstance(result, pl.Series)
        # Annualized vol should be reasonable (typically 0.1 to 1.0)
        valid_values = result.drop_nulls().to_list()
        assert all(0 < v < 5 for v in valid_values if v is not None)

    def test_compute_not_annualized(self, sample_ohlcv: pl.DataFrame):
        """Test non-annualized volatility."""
        factor = VolatilityFactor(window=20, annualize=False)
        result = factor.compute(sample_ohlcv)

        # Non-annualized should be smaller
        valid_values = result.drop_nulls().to_list()
        assert all(v < 0.5 for v in valid_values if v is not None)


class TestRSIFactor:
    """Tests for RSIFactor."""

    def test_compute(self, sample_ohlcv: pl.DataFrame):
        """Test RSI computation."""
        factor = RSIFactor(window=14)
        result = factor.compute(sample_ohlcv)

        assert isinstance(result, pl.Series)
        # RSI should be between 0 and 100
        valid_values = result.drop_nulls().to_list()
        assert all(0 <= v <= 100 for v in valid_values)

    def test_rsi_extreme_up(self):
        """Test RSI for consistently rising prices."""
        data = pl.DataFrame({"close": list(range(1, 51))})  # 1 to 50
        factor = RSIFactor(window=14)
        result = factor.compute(data)

        # Consistently rising = RSI should be 100
        assert result[-1] == 100.0


class TestBollingerBandFactor:
    """Tests for BollingerBandFactor."""

    def test_compute(self, sample_ohlcv: pl.DataFrame):
        """Test Bollinger Band position computation."""
        factor = BollingerBandFactor(window=20, num_std=2.0)
        result = factor.compute(sample_ohlcv)

        assert isinstance(result, pl.Series)
        # Position typically between 0 and 1, but can exceed
        valid_values = result.drop_nulls().to_list()
        assert len(valid_values) > 0


class TestVWAPDeviationFactor:
    """Tests for VWAPDeviationFactor."""

    def test_compute(self, sample_ohlcv: pl.DataFrame):
        """Test VWAP deviation computation."""
        factor = VWAPDeviationFactor(window=10)
        result = factor.compute(sample_ohlcv)

        assert isinstance(result, pl.Series)
        # Deviation should be small (typically -0.1 to 0.1)
        valid_values = result.drop_nulls().to_list()
        assert all(abs(v) < 1 for v in valid_values)


class TestReturnsFactor:
    """Tests for ReturnsFactor."""

    def test_compute(self, minimal_data: pl.DataFrame):
        """Test returns computation."""
        factor = ReturnsFactor(window=1)
        result = factor.compute(minimal_data)

        # Manual calculation: pct_change
        expected_second = (102.0 - 100.0) / 100.0
        assert abs(result[1] - expected_second) < 1e-10


class TestLogReturnsFactor:
    """Tests for LogReturnsFactor."""

    def test_compute(self, minimal_data: pl.DataFrame):
        """Test log returns computation."""
        factor = LogReturnsFactor(window=1)
        result = factor.compute(minimal_data)

        # Log return: ln(102/100)
        import math

        expected_second = math.log(102.0 / 100.0)
        assert abs(result[1] - expected_second) < 1e-10


class TestRankFactor:
    """Tests for RankFactor."""

    def test_compute_on_close(self, minimal_data: pl.DataFrame):
        """Test rank computation on close prices."""
        factor = RankFactor()
        result = factor.compute(minimal_data)

        assert isinstance(result, pl.Series)
        # Ranks should be between 0 and 1
        assert all(0 <= v <= 1 for v in result.to_list())


# =============================================================================
# Factor Registry Tests
# =============================================================================


class TestFactorRegistry:
    """Tests for factor registry functions."""

    def test_list_factors(self):
        """Test listing registered factors."""
        factors = list_factors()
        assert "momentum" in factors
        assert "rsi" in factors
        assert "volatility" in factors

    def test_get_factor(self):
        """Test getting factor by name."""
        factor = get_factor("momentum", window=15)
        assert isinstance(factor, MomentumFactor)
        assert factor.window == 15

    def test_get_unknown_factor(self):
        """Test getting unknown factor raises error."""
        with pytest.raises(KeyError, match="Unknown factor"):
            get_factor("nonexistent_factor")

    def test_register_factor(self):
        """Test registering custom factor."""

        class CustomFactor(Factor):
            def compute(self, data: pl.DataFrame) -> pl.Series:
                return data["close"].alias(self.name)

        register_factor("custom", CustomFactor)
        factor = get_factor("custom")
        assert isinstance(factor, CustomFactor)


# =============================================================================
# Pipeline Tests
# =============================================================================


class TestPipeline:
    """Tests for Pipeline class."""

    def test_create_empty(self):
        """Test creating empty pipeline."""
        pipeline = Pipeline()
        assert len(pipeline) == 0
        assert pipeline.factor_names == []

    def test_create_with_factors(self):
        """Test creating pipeline with factors."""
        factors = [MomentumFactor(), RSIFactor()]
        pipeline = Pipeline(factors)

        assert len(pipeline) == 2
        assert "MomentumFactor" in pipeline.factor_names
        assert "RSIFactor" in pipeline.factor_names

    def test_add_factor(self):
        """Test adding factor to pipeline."""
        pipeline = Pipeline()
        pipeline.add_factor(MomentumFactor())

        assert len(pipeline) == 1

    def test_add_factor_chaining(self):
        """Test method chaining for add_factor."""
        pipeline = (
            Pipeline()
            .add_factor(MomentumFactor())
            .add_factor(RSIFactor())
            .add_factor(VolatilityFactor())
        )

        assert len(pipeline) == 3

    def test_remove_factor(self):
        """Test removing factor from pipeline."""
        pipeline = Pipeline([MomentumFactor(), RSIFactor()])
        pipeline.remove_factor("MomentumFactor")

        assert len(pipeline) == 1
        assert "RSIFactor" in pipeline.factor_names

    def test_clear_factors(self):
        """Test clearing all factors."""
        pipeline = Pipeline([MomentumFactor(), RSIFactor()])
        pipeline.clear_factors()

        assert len(pipeline) == 0

    def test_run_empty_pipeline(self, sample_ohlcv: pl.DataFrame):
        """Test running empty pipeline raises error."""
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="No factors registered"):
            pipeline.run(sample_ohlcv)

    def test_run_empty_data(self):
        """Test running on empty data raises error."""
        pipeline = Pipeline([MomentumFactor()])
        empty_data = pl.DataFrame({"close": []})

        with pytest.raises(ValueError, match="Input data is empty"):
            pipeline.run(empty_data)

    def test_run_single_factor(self, sample_ohlcv: pl.DataFrame):
        """Test running pipeline with single factor."""
        pipeline = Pipeline([MomentumFactor(window=5)])
        result = pipeline.run(sample_ohlcv)

        assert isinstance(result, PipelineResult)
        assert "MomentumFactor" in result.factor_names
        assert "MomentumFactor" in result.data.columns

    def test_run_multiple_factors(self, sample_ohlcv: pl.DataFrame):
        """Test running pipeline with multiple factors."""
        pipeline = Pipeline(
            [
                MomentumFactor(window=10),
                RSIFactor(window=14),
                VolatilityFactor(window=20),
            ]
        )
        result = pipeline.run(sample_ohlcv)

        assert len(result.factor_names) == 3
        assert all(name in result.data.columns for name in result.factor_names)

    def test_run_include_data_true(self, sample_ohlcv: pl.DataFrame):
        """Test that original data is included by default."""
        pipeline = Pipeline([MomentumFactor()])
        result = pipeline.run(sample_ohlcv, include_data=True)

        # Original columns should be present
        assert "close" in result.data.columns
        assert "volume" in result.data.columns

    def test_run_include_data_false(self, sample_ohlcv: pl.DataFrame):
        """Test excluding original data."""
        pipeline = Pipeline([MomentumFactor()])
        result = pipeline.run(sample_ohlcv, include_data=False)

        # Only factor columns should be present
        assert "close" not in result.data.columns
        assert "MomentumFactor" in result.data.columns

    def test_run_with_warmup(self, sample_ohlcv: pl.DataFrame):
        """Test warmup period is applied."""
        config = PipelineConfig(warmup_period=0)  # Disable warmup
        pipeline = Pipeline([MomentumFactor(window=10)], config=config)
        result_no_warmup = pipeline.run(sample_ohlcv)

        config_warmup = PipelineConfig(warmup_period=20)
        pipeline_warmup = Pipeline([MomentumFactor(window=10)], config=config_warmup)
        result_warmup = pipeline_warmup.run(sample_ohlcv)

        assert len(result_warmup.data) < len(result_no_warmup.data)

    def test_run_drop_nulls(self, sample_ohlcv: pl.DataFrame):
        """Test dropping null values."""
        config = PipelineConfig(drop_nulls=True, warmup_period=0)
        pipeline = Pipeline([MomentumFactor(window=10)], config=config)
        result = pipeline.run(sample_ohlcv)

        # Should have no nulls in factor column
        assert result.data["MomentumFactor"].null_count() == 0

    def test_metadata(self, sample_ohlcv: pl.DataFrame):
        """Test pipeline metadata."""
        pipeline = Pipeline([MomentumFactor()])
        result = pipeline.run(sample_ohlcv)

        assert "total_time_seconds" in result.metadata
        assert "factor_timings" in result.metadata
        assert "input_rows" in result.metadata
        assert result.metadata["factor_count"] == 1

    def test_hook_called(self, sample_ohlcv: pl.DataFrame):
        """Test that hooks are called after computation."""
        hook_results = []

        def my_hook(result: PipelineResult) -> None:
            hook_results.append(result)

        pipeline = Pipeline([MomentumFactor()])
        pipeline.add_hook(my_hook)
        pipeline.run(sample_ohlcv)

        assert len(hook_results) == 1
        assert isinstance(hook_results[0], PipelineResult)

    def test_describe(self):
        """Test pipeline description."""
        pipeline = Pipeline(
            [MomentumFactor(window=20), RSIFactor(window=14)],
            config=PipelineConfig(drop_nulls=True),
        )
        desc = pipeline.describe()

        assert desc["factor_count"] == 2
        assert len(desc["factors"]) == 2
        assert desc["config"]["drop_nulls"] is True

    def test_repr(self):
        """Test string representation."""
        pipeline = Pipeline([MomentumFactor(), RSIFactor()])
        repr_str = repr(pipeline)

        assert "Pipeline" in repr_str
        assert "MomentumFactor" in repr_str
        assert "RSIFactor" in repr_str

    def test_iteration(self):
        """Test iterating over pipeline factors."""
        factors = [MomentumFactor(), RSIFactor()]
        pipeline = Pipeline(factors)

        iterated = list(pipeline)
        assert len(iterated) == 2
        assert all(isinstance(f, Factor) for f in iterated)


# =============================================================================
# PipelineResult Tests
# =============================================================================


class TestPipelineResult:
    """Tests for PipelineResult class."""

    def test_getitem(self, sample_ohlcv: pl.DataFrame):
        """Test getting factor by name."""
        pipeline = Pipeline([MomentumFactor()])
        result = pipeline.run(sample_ohlcv)

        factor_data = result["MomentumFactor"]
        assert isinstance(factor_data, pl.Series)

    def test_contains(self, sample_ohlcv: pl.DataFrame):
        """Test checking if factor exists."""
        pipeline = Pipeline([MomentumFactor()])
        result = pipeline.run(sample_ohlcv)

        assert "MomentumFactor" in result
        assert "NonexistentFactor" not in result

    def test_to_dict(self, sample_ohlcv: pl.DataFrame):
        """Test converting to dictionary."""
        pipeline = Pipeline([MomentumFactor()])
        result = pipeline.run(sample_ohlcv)

        data_dict = result.to_dict()
        assert isinstance(data_dict, dict)
        assert "MomentumFactor" in data_dict

    def test_shape(self, sample_ohlcv: pl.DataFrame):
        """Test getting result shape."""
        pipeline = Pipeline([MomentumFactor(), RSIFactor()])
        result = pipeline.run(sample_ohlcv)

        rows, cols = result.shape
        assert rows > 0
        assert cols >= 2  # At least two factors

    def test_get_factor_stats(self, sample_ohlcv: pl.DataFrame):
        """Test getting factor statistics."""
        pipeline = Pipeline([MomentumFactor()])
        result = pipeline.run(sample_ohlcv)

        stats = result.get_factor_stats()
        assert "MomentumFactor" in stats
        assert "mean" in stats["MomentumFactor"]
        assert "std" in stats["MomentumFactor"]


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_run_factors(self, sample_ohlcv: pl.DataFrame):
        """Test run_factors convenience function."""
        result = run_factors(
            sample_ohlcv,
            [MomentumFactor(), RSIFactor()],
        )

        assert isinstance(result, PipelineResult)
        assert len(result.factor_names) == 2

    def test_compute_factor(self, sample_ohlcv: pl.DataFrame):
        """Test compute_factor convenience function."""
        result = compute_factor(sample_ohlcv, MomentumFactor())

        assert isinstance(result, pl.Series)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPipelineIntegration:
    """Integration tests for pipeline."""

    def test_full_workflow(self, sample_ohlcv: pl.DataFrame):
        """Test complete pipeline workflow."""
        # Create pipeline with multiple factors
        pipeline = (
            Pipeline()
            .add_factor(MomentumFactor(window=10, name="mom_10"))
            .add_factor(MomentumFactor(window=20, name="mom_20"))
            .add_factor(RSIFactor(window=14))
            .add_factor(VolatilityFactor(window=20))
            .add_factor(MeanReversionFactor(window=15))
        )

        # Run pipeline
        result = pipeline.run(sample_ohlcv)

        # Verify results
        assert len(result.factor_names) == 5
        assert result.metadata["factor_count"] == 5

        # Get statistics
        stats = result.get_factor_stats()
        assert len(stats) == 5

    def test_factor_correlation(self, sample_ohlcv: pl.DataFrame):
        """Test that different window factors are computed correctly."""
        pipeline = Pipeline(
            [
                MomentumFactor(window=5, name="mom_5"),
                MomentumFactor(window=20, name="mom_20"),
            ]
        )
        result = pipeline.run(sample_ohlcv)

        # Different windows should give different results
        mom_5 = result["mom_5"].drop_nulls()
        mom_20 = result["mom_20"].drop_nulls()

        # They shouldn't be identical (unless data is constant)
        if len(mom_5) > 0 and len(mom_20) > 0:
            assert mom_5[-1] != mom_20[-1] or sample_ohlcv["close"].std() < 1e-10
