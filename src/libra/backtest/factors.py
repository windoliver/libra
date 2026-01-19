"""
Vectorized Factor Framework for LIBRA Backtesting.

Provides Zipline-style factor computation using Polars for vectorized operations:
- Factor base class with compute() interface
- Common factors: momentum, mean reversion, volatility
- Window-based and cross-sectional factors

Achieves significant speedup over event-driven backtesting by computing
all factors across the entire backtest period in a single vectorized pass.

See: https://github.com/windoliver/libra/issues/93
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import polars as pl


# =============================================================================
# Factor Base Class
# =============================================================================


@dataclass
class FactorMeta:
    """Metadata about a factor.

    Attributes:
        name: Factor identifier
        description: Human-readable description
        window: Lookback window size (if applicable)
        inputs: Required input columns
        outputs: Output column names
    """

    name: str
    description: str = ""
    window: int = 0
    inputs: list[str] = field(default_factory=lambda: ["close"])
    outputs: list[str] = field(default_factory=list)


class Factor(ABC):
    """Base class for vectorized factor computation.

    Factors transform market data into numerical signals using vectorized
    Polars operations. Each factor implements compute() which receives
    a DataFrame and returns a Series or DataFrame with factor values.

    Example:
        class MomentumFactor(Factor):
            window = 20

            def compute(self, data: pl.DataFrame) -> pl.Series:
                return data["close"].pct_change(self.window)

        factor = MomentumFactor()
        result = factor.compute(historical_data)

    Attributes:
        window: Default lookback window (can be overridden in subclasses)
        name: Factor name (defaults to class name)
        inputs: Required input columns
    """

    window: int = 1
    inputs: list[str] = ["close"]

    def __init__(
        self,
        window: int | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize factor.

        Args:
            window: Override default window size
            name: Override default factor name
        """
        if window is not None:
            self.window = window
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Get factor name."""
        return self._name

    @abstractmethod
    def compute(self, data: pl.DataFrame) -> pl.Series | pl.DataFrame:
        """Compute factor values from input data.

        Args:
            data: Input DataFrame with OHLCV data

        Returns:
            Series or DataFrame with computed factor values
        """
        ...

    def validate_inputs(self, data: pl.DataFrame) -> None:
        """Validate that required inputs exist in data.

        Args:
            data: Input DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        missing = [col for col in self.inputs if col not in data.columns]
        if missing:
            raise ValueError(
                f"Factor '{self.name}' missing required columns: {missing}"
            )

    def get_metadata(self) -> FactorMeta:
        """Get factor metadata."""
        return FactorMeta(
            name=self.name,
            description=self.__doc__ or "",
            window=self.window,
            inputs=self.inputs.copy(),
            outputs=[self.name],
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(window={self.window}, name='{self.name}')"


# =============================================================================
# Common Factors
# =============================================================================


class MomentumFactor(Factor):
    """Price momentum over lookback window.

    Computes percentage change in price over the specified window.
    Positive values indicate upward momentum, negative indicate downward.

    Formula: (close - close.shift(window)) / close.shift(window)
    """

    window = 20
    inputs = ["close"]

    def compute(self, data: pl.DataFrame) -> pl.Series:
        """Compute momentum as percentage change over window."""
        self.validate_inputs(data)
        return (
            data["close"]
            .pct_change(n=self.window)
            .alias(self.name)
        )


class MeanReversionFactor(Factor):
    """Mean reversion signal: deviation from rolling mean.

    Measures how far current price is from its rolling average.
    Positive values indicate price is above average (overbought),
    negative values indicate price is below average (oversold).

    Formula: (close - close.rolling_mean(window)) / close.rolling_std(window)
    """

    window = 20
    inputs = ["close"]

    def compute(self, data: pl.DataFrame) -> pl.Series:
        """Compute z-score deviation from rolling mean."""
        self.validate_inputs(data)
        close = data["close"]
        rolling_mean = close.rolling_mean(window_size=self.window)
        rolling_std = close.rolling_std(window_size=self.window)

        # Z-score: (price - mean) / std
        z_score = (close - rolling_mean) / rolling_std
        return z_score.alias(self.name)


class VolatilityFactor(Factor):
    """Rolling volatility (standard deviation of returns).

    Measures price volatility as the standard deviation of
    log returns over the lookback window.

    Formula: returns.rolling_std(window) * sqrt(252)  # Annualized
    """

    window = 20
    inputs = ["close"]
    annualize: bool = True

    def __init__(
        self,
        window: int | None = None,
        name: str | None = None,
        annualize: bool = True,
    ) -> None:
        """Initialize volatility factor.

        Args:
            window: Lookback window
            name: Factor name
            annualize: Whether to annualize volatility (default True)
        """
        super().__init__(window=window, name=name)
        self.annualize = annualize

    def compute(self, data: pl.DataFrame) -> pl.Series:
        """Compute rolling volatility."""
        self.validate_inputs(data)
        # Log returns
        returns = data["close"].log().diff()

        # Rolling standard deviation
        vol = returns.rolling_std(window_size=self.window)

        # Annualize if requested (assuming daily data, 252 trading days)
        if self.annualize:
            vol = vol * (252**0.5)

        return vol.alias(self.name)


class RSIFactor(Factor):
    """Relative Strength Index (RSI).

    Momentum oscillator measuring speed and magnitude of price changes.
    Values range from 0 to 100:
    - Above 70: Overbought
    - Below 30: Oversold

    Formula: 100 - (100 / (1 + avg_gain / avg_loss))
    """

    window = 14
    inputs = ["close"]

    def compute(self, data: pl.DataFrame) -> pl.Series:
        """Compute RSI indicator."""
        self.validate_inputs(data)
        close = data["close"]

        # Price changes
        delta = close.diff()

        # Separate gains and losses
        gains = delta.clip(lower_bound=0)
        losses = (-delta).clip(lower_bound=0)

        # Rolling averages
        avg_gain = gains.rolling_mean(window_size=self.window)
        avg_loss = losses.rolling_mean(window_size=self.window)

        # RSI calculation
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.alias(self.name)


class BollingerBandFactor(Factor):
    """Bollinger Band position indicator.

    Measures where current price sits within Bollinger Bands.
    Values:
    - Near 1: Price at upper band (overbought)
    - Near 0: Price at lower band (oversold)
    - Near 0.5: Price at middle band

    Formula: (close - lower_band) / (upper_band - lower_band)
    """

    window = 20
    inputs = ["close"]
    num_std: float = 2.0

    def __init__(
        self,
        window: int | None = None,
        name: str | None = None,
        num_std: float = 2.0,
    ) -> None:
        """Initialize Bollinger Band factor.

        Args:
            window: Lookback window for rolling mean/std
            name: Factor name
            num_std: Number of standard deviations for bands
        """
        super().__init__(window=window, name=name)
        self.num_std = num_std

    def compute(self, data: pl.DataFrame) -> pl.Series:
        """Compute Bollinger Band position (0-1 scale)."""
        self.validate_inputs(data)
        close = data["close"]

        middle = close.rolling_mean(window_size=self.window)
        std = close.rolling_std(window_size=self.window)

        upper = middle + (self.num_std * std)
        lower = middle - (self.num_std * std)

        # Position within bands (0 = lower, 1 = upper)
        position = (close - lower) / (upper - lower)

        return position.alias(self.name)


class VWAPDeviationFactor(Factor):
    """Deviation from Volume Weighted Average Price (VWAP).

    Measures how far current price deviates from VWAP.
    Positive values indicate price above VWAP (bullish),
    negative values indicate price below VWAP (bearish).

    Formula: (close - vwap) / vwap
    """

    window = 20
    inputs = ["close", "volume"]

    def compute(self, data: pl.DataFrame) -> pl.Series:
        """Compute VWAP deviation."""
        self.validate_inputs(data)

        close = data["close"]
        volume = data["volume"]

        # Rolling VWAP
        price_volume = close * volume
        rolling_pv = price_volume.rolling_sum(window_size=self.window)
        rolling_vol = volume.rolling_sum(window_size=self.window)
        vwap = rolling_pv / rolling_vol

        # Deviation as percentage
        deviation = (close - vwap) / vwap

        return deviation.alias(self.name)


class ReturnsFactor(Factor):
    """Simple returns over window.

    Computes arithmetic returns: (P_t - P_{t-n}) / P_{t-n}
    """

    window = 1
    inputs = ["close"]

    def compute(self, data: pl.DataFrame) -> pl.Series:
        """Compute simple returns."""
        self.validate_inputs(data)
        return data["close"].pct_change(n=self.window).alias(self.name)


class LogReturnsFactor(Factor):
    """Log returns over window.

    Computes logarithmic returns: ln(P_t / P_{t-n})
    Log returns are time-additive and often preferred for statistical analysis.
    """

    window = 1
    inputs = ["close"]

    def compute(self, data: pl.DataFrame) -> pl.Series:
        """Compute log returns."""
        self.validate_inputs(data)
        close = data["close"]
        return (close.log() - close.shift(self.window).log()).alias(self.name)


class RankFactor(Factor):
    """Cross-sectional rank transformation.

    Converts factor values to percentile ranks within each time period.
    Useful for creating market-neutral signals.

    Note: This is a wrapper factor that transforms another factor's output.
    """

    inputs = ["close"]

    def __init__(
        self,
        base_factor: Factor | None = None,
        window: int | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize rank factor.

        Args:
            base_factor: Factor to rank (or None for close price)
            window: Window for base factor
            name: Factor name
        """
        super().__init__(window=window, name=name or "RankFactor")
        self._base_factor = base_factor

    def compute(self, data: pl.DataFrame) -> pl.Series:
        """Compute percentile rank of values."""
        if self._base_factor:
            values = self._base_factor.compute(data)
        else:
            self.validate_inputs(data)
            values = data["close"]

        # Rank within rolling window (or all data if no window)
        ranked = values.rank() / values.len()
        return ranked.alias(self.name)


# =============================================================================
# Factor Registry
# =============================================================================


_FACTOR_REGISTRY: dict[str, type[Factor]] = {
    "momentum": MomentumFactor,
    "mean_reversion": MeanReversionFactor,
    "volatility": VolatilityFactor,
    "rsi": RSIFactor,
    "bollinger": BollingerBandFactor,
    "vwap_deviation": VWAPDeviationFactor,
    "returns": ReturnsFactor,
    "log_returns": LogReturnsFactor,
    "rank": RankFactor,
}


def get_factor(name: str, **kwargs: Any) -> Factor:
    """Get a factor instance by name.

    Args:
        name: Factor name (e.g., 'momentum', 'rsi')
        **kwargs: Arguments to pass to factor constructor

    Returns:
        Factor instance

    Raises:
        KeyError: If factor name is not registered
    """
    if name not in _FACTOR_REGISTRY:
        raise KeyError(
            f"Unknown factor: '{name}'. Available: {list(_FACTOR_REGISTRY.keys())}"
        )
    return _FACTOR_REGISTRY[name](**kwargs)


def register_factor(name: str, factor_class: type[Factor]) -> None:
    """Register a custom factor.

    Args:
        name: Name to register factor under
        factor_class: Factor class to register
    """
    _FACTOR_REGISTRY[name] = factor_class


def list_factors() -> list[str]:
    """List all registered factor names."""
    return list(_FACTOR_REGISTRY.keys())
