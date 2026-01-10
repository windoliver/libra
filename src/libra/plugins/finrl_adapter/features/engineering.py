"""
Feature Engineering Pipeline for FinRL Adapter.

Provides a complete feature engineering pipeline for RL trading.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from libra.plugins.finrl_adapter.features.technical import TechnicalIndicators

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Try to import numpy
try:
    import numpy as np
    from numpy.typing import NDArray

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]
    NDArray = Any  # type: ignore[misc,assignment]


class FeatureEngineer:
    """
    Feature engineering pipeline for RL trading environments.

    This class provides:
    - Technical indicator calculation
    - Feature normalization and scaling
    - Missing value handling
    - Lagged feature generation
    - Data validation and cleaning

    Usage:
        engineer = FeatureEngineer(config)
        df = engineer.process(raw_df)
    """

    def __init__(
        self,
        tech_indicators: list[str] | None = None,
        normalize: bool = True,
        fill_missing: bool = True,
        add_lags: bool = False,
        lag_periods: list[int] | None = None,
        add_turbulence: bool = True,
    ) -> None:
        """
        Initialize the feature engineer.

        Args:
            tech_indicators: List of technical indicators to add.
            normalize: Whether to normalize features.
            fill_missing: Whether to fill missing values.
            add_lags: Whether to add lagged features.
            lag_periods: Periods for lagged features.
            add_turbulence: Whether to add turbulence index.
        """
        self.tech_indicators = tech_indicators or [
            "macd",
            "rsi",
            "cci",
            "dx",
            "boll",
            "sma",
            "atr",
        ]
        self.normalize = normalize
        self.fill_missing = fill_missing
        self.add_lags = add_lags
        self.lag_periods = lag_periods or [1, 5]
        self.add_turbulence = add_turbulence

        self.indicator_calculator = TechnicalIndicators()

        # Storage for normalization parameters
        self._feature_stats: dict[str, dict[str, float]] = {}

    def process(
        self,
        df: Any,
        fit: bool = True,
    ) -> Any:
        """
        Process DataFrame through the feature engineering pipeline.

        Args:
            df: Raw DataFrame with OHLCV data.
            fit: Whether to fit normalization parameters (True for training).

        Returns:
            Processed DataFrame with all features.
        """
        logger.info("Starting feature engineering pipeline")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Step 1: Validate and clean data
        df = self._validate_data(df)

        # Step 2: Add technical indicators
        df = self._add_indicators(df)

        # Step 3: Add lagged features
        if self.add_lags:
            df = self._add_lagged_features(df)

        # Step 4: Add turbulence
        if self.add_turbulence:
            df = self._add_turbulence(df)

        # Step 5: Handle missing values
        if self.fill_missing:
            df = self._fill_missing(df)

        # Step 6: Normalize features
        if self.normalize:
            df = self._normalize_features(df, fit=fit)

        logger.info(
            "Feature engineering complete: %d rows, %d columns",
            len(df),
            len(df.columns),
        )

        return df

    def _validate_data(self, df: Any) -> Any:
        """Validate and clean input data."""
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure date column exists or create index
        if "date" not in df.columns:
            if hasattr(df.index, "name") and df.index.name == "date":
                df = df.reset_index()
            else:
                df["date"] = range(len(df))

        # Sort by date and ticker
        sort_cols = ["date"]
        if "tic" in df.columns:
            sort_cols.append("tic")
        df = df.sort_values(sort_cols)

        # Remove rows with invalid prices
        df = df[df["close"] > 0]

        return df

    def _add_indicators(self, df: Any) -> Any:
        """Add technical indicators to DataFrame."""
        logger.debug("Adding technical indicators: %s", self.tech_indicators)
        return self.indicator_calculator.add_all_indicators(
            df, indicators=self.tech_indicators
        )

    def _add_lagged_features(self, df: Any) -> Any:
        """Add lagged versions of key features."""
        df = df.copy()

        base_features = ["close", "volume", "return"]
        if "return" not in df.columns:
            if "tic" in df.columns:
                df["return"] = df.groupby("tic")["close"].pct_change()
            else:
                df["return"] = df["close"].pct_change()

        for feature in base_features:
            if feature not in df.columns:
                continue
            for lag in self.lag_periods:
                col_name = f"{feature}_lag_{lag}"
                if "tic" in df.columns:
                    df[col_name] = df.groupby("tic")[feature].shift(lag)
                else:
                    df[col_name] = df[feature].shift(lag)

        return df

    def _add_turbulence(self, df: Any) -> Any:
        """Add turbulence index."""
        logger.debug("Adding turbulence index")
        return TechnicalIndicators.calculate_turbulence(df)

    def _fill_missing(self, df: Any) -> Any:
        """Fill missing values in DataFrame."""
        df = df.copy()

        # Forward fill first (carry previous values)
        if "tic" in df.columns:
            df = df.groupby("tic").apply(lambda x: x.ffill()).reset_index(drop=True)
        else:
            df = df.ffill()

        # Backward fill for initial values
        if "tic" in df.columns:
            df = df.groupby("tic").apply(lambda x: x.bfill()).reset_index(drop=True)
        else:
            df = df.bfill()

        # Fill remaining with zeros
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    def _normalize_features(self, df: Any, fit: bool = True) -> Any:
        """
        Normalize numerical features.

        Uses z-score normalization (mean=0, std=1) for most features.
        Prices are normalized relative to initial price.

        Args:
            df: DataFrame to normalize.
            fit: Whether to calculate and store normalization parameters.

        Returns:
            Normalized DataFrame.
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, skipping normalization")
            return df

        df = df.copy()

        # Columns to normalize
        exclude_cols = {"date", "tic", "open", "high", "low", "close", "volume"}
        numeric_cols = df.select_dtypes(include=["number"]).columns
        normalize_cols = [c for c in numeric_cols if c not in exclude_cols]

        for col in normalize_cols:
            if fit:
                mean_val = float(df[col].mean())
                std_val = float(df[col].std())
                if std_val == 0:
                    std_val = 1.0
                self._feature_stats[col] = {"mean": mean_val, "std": std_val}
            elif col not in self._feature_stats:
                # Use default stats if not fitted
                self._feature_stats[col] = {"mean": 0.0, "std": 1.0}

            stats = self._feature_stats[col]
            df[col] = (df[col] - stats["mean"]) / stats["std"]

        return df

    def get_feature_columns(self, df: Any) -> list[str]:
        """
        Get list of feature columns (excluding meta columns).

        Args:
            df: DataFrame with features.

        Returns:
            List of feature column names.
        """
        exclude = {"date", "tic", "index"}
        return [c for c in df.columns if c not in exclude]

    def get_feature_stats(self) -> dict[str, dict[str, float]]:
        """Get stored feature normalization statistics."""
        return self._feature_stats.copy()

    def set_feature_stats(self, stats: dict[str, dict[str, float]]) -> None:
        """
        Set feature normalization statistics.

        Useful for loading pre-computed stats for inference.

        Args:
            stats: Dictionary of feature statistics.
        """
        self._feature_stats = stats.copy()


def prepare_training_data(
    df: Any,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    tech_indicators: list[str] | None = None,
) -> tuple[Any, Any]:
    """
    Prepare training and testing datasets with feature engineering.

    Args:
        df: Raw DataFrame with OHLCV data.
        train_start: Training period start date.
        train_end: Training period end date.
        test_start: Testing period start date.
        test_end: Testing period end date.
        tech_indicators: List of technical indicators to add.

    Returns:
        Tuple of (train_df, test_df) with features.
    """
    engineer = FeatureEngineer(tech_indicators=tech_indicators)

    # Split data
    train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

    # Process training data (fit normalization)
    train_df = engineer.process(train_df, fit=True)

    # Process test data (use training normalization)
    test_df = engineer.process(test_df, fit=False)

    return train_df, test_df
