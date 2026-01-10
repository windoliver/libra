"""
Technical Indicators for FinRL Adapter.

Provides technical analysis indicators for feature engineering.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Try to import numpy
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]


class TechnicalIndicators:
    """
    Technical indicator calculator for trading features.

    Provides common technical indicators used in RL trading:
    - Moving averages (SMA, EMA)
    - Momentum indicators (RSI, MACD, CCI, DX)
    - Volatility indicators (Bollinger Bands, ATR)
    - Volume indicators (OBV)
    - Turbulence index

    All methods work with pandas DataFrames and return modified DataFrames
    with indicator columns added.
    """

    # Default indicator parameters
    DEFAULT_PARAMS = {
        "sma_periods": [30, 60],
        "ema_periods": [12, 26],
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2,
        "atr_period": 14,
        "cci_period": 30,
        "dx_period": 30,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the technical indicator calculator.

        Args:
            params: Custom parameters for indicators.
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

    def add_all_indicators(
        self,
        df: Any,
        indicators: list[str] | None = None,
    ) -> Any:
        """
        Add all specified technical indicators to DataFrame.

        Args:
            df: DataFrame with OHLCV columns.
            indicators: List of indicators to add. If None, adds all.

        Returns:
            DataFrame with indicator columns added.
        """
        if indicators is None:
            indicators = [
                "macd",
                "rsi",
                "cci",
                "dx",
                "boll",
                "sma",
                "ema",
                "atr",
                "obv",
            ]

        df = df.copy()

        for indicator in indicators:
            method_name = f"add_{indicator}"
            if hasattr(self, method_name):
                try:
                    df = getattr(self, method_name)(df)
                except Exception as e:
                    logger.warning("Failed to add %s: %s", indicator, e)
            else:
                logger.warning("Unknown indicator: %s", indicator)

        return df

    def add_sma(self, df: Any, periods: list[int] | None = None) -> Any:
        """
        Add Simple Moving Average indicators.

        Args:
            df: DataFrame with 'close' column.
            periods: List of SMA periods.

        Returns:
            DataFrame with SMA columns added.
        """
        periods = periods or self.params["sma_periods"]
        df = df.copy()

        for period in periods:
            col_name = f"close_{period}_sma"
            if "tic" in df.columns:
                df[col_name] = df.groupby("tic")["close"].transform(
                    lambda x: x.rolling(window=period, min_periods=1).mean()
                )
            else:
                df[col_name] = df["close"].rolling(window=period, min_periods=1).mean()

        return df

    def add_ema(self, df: Any, periods: list[int] | None = None) -> Any:
        """
        Add Exponential Moving Average indicators.

        Args:
            df: DataFrame with 'close' column.
            periods: List of EMA periods.

        Returns:
            DataFrame with EMA columns added.
        """
        periods = periods or self.params["ema_periods"]
        df = df.copy()

        for period in periods:
            col_name = f"close_{period}_ema"
            if "tic" in df.columns:
                df[col_name] = df.groupby("tic")["close"].transform(
                    lambda x: x.ewm(span=period, adjust=False).mean()
                )
            else:
                df[col_name] = df["close"].ewm(span=period, adjust=False).mean()

        return df

    def add_rsi(self, df: Any, period: int | None = None) -> Any:
        """
        Add Relative Strength Index indicator.

        Args:
            df: DataFrame with 'close' column.
            period: RSI period.

        Returns:
            DataFrame with RSI column added.
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, skipping RSI")
            return df

        period = period or self.params["rsi_period"]
        df = df.copy()

        def calc_rsi(prices: Any) -> Any:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)

            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()

            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)

        col_name = f"rsi_{period}"
        if "tic" in df.columns:
            df[col_name] = df.groupby("tic")["close"].transform(calc_rsi)
        else:
            df[col_name] = calc_rsi(df["close"])

        return df

    def add_macd(
        self,
        df: Any,
        fast: int | None = None,
        slow: int | None = None,
        signal: int | None = None,
    ) -> Any:
        """
        Add MACD (Moving Average Convergence Divergence) indicator.

        Args:
            df: DataFrame with 'close' column.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal line period.

        Returns:
            DataFrame with MACD columns added.
        """
        fast = fast or self.params["macd_fast"]
        slow = slow or self.params["macd_slow"]
        signal = signal or self.params["macd_signal"]
        df = df.copy()

        def calc_macd(prices: Any) -> Any:
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            return macd_line

        if "tic" in df.columns:
            df["macd"] = df.groupby("tic")["close"].transform(calc_macd)
        else:
            df["macd"] = calc_macd(df["close"])

        return df

    def add_boll(
        self,
        df: Any,
        period: int | None = None,
        std_mult: float | None = None,
    ) -> Any:
        """
        Add Bollinger Bands indicators.

        Args:
            df: DataFrame with 'close' column.
            period: Rolling window period.
            std_mult: Standard deviation multiplier.

        Returns:
            DataFrame with Bollinger Band columns added.
        """
        period = period or self.params["bb_period"]
        std_mult = std_mult or self.params["bb_std"]
        df = df.copy()

        def calc_bands(prices: Any) -> tuple[Any, Any]:
            sma = prices.rolling(window=period, min_periods=1).mean()
            std = prices.rolling(window=period, min_periods=1).std()
            upper = sma + std_mult * std
            lower = sma - std_mult * std
            return upper, lower

        if "tic" in df.columns:
            grouped = df.groupby("tic")["close"]
            df["boll_ub"] = grouped.transform(lambda x: calc_bands(x)[0])
            df["boll_lb"] = grouped.transform(lambda x: calc_bands(x)[1])
        else:
            df["boll_ub"], df["boll_lb"] = calc_bands(df["close"])

        return df

    def add_cci(self, df: Any, period: int | None = None) -> Any:
        """
        Add Commodity Channel Index indicator.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns.
            period: CCI period.

        Returns:
            DataFrame with CCI column added.
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, skipping CCI")
            return df

        period = period or self.params["cci_period"]
        df = df.copy()

        def calc_cci(group: Any) -> Any:
            tp = (group["high"] + group["low"] + group["close"]) / 3
            sma = tp.rolling(window=period, min_periods=1).mean()
            mad = tp.rolling(window=period, min_periods=1).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True
            )
            cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
            return cci.fillna(0)

        col_name = f"cci_{period}"
        if "tic" in df.columns:
            df[col_name] = df.groupby("tic").apply(
                lambda x: calc_cci(x)
            ).reset_index(level=0, drop=True)
        else:
            df[col_name] = calc_cci(df)

        return df

    def add_dx(self, df: Any, period: int | None = None) -> Any:
        """
        Add Directional Movement Index indicator.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns.
            period: DX period.

        Returns:
            DataFrame with DX column added.
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, skipping DX")
            return df

        period = period or self.params["dx_period"]
        df = df.copy()

        def calc_dx(group: Any) -> Any:
            high = group["high"]
            low = group["low"]
            close = group["close"]

            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = np.maximum(np.maximum(tr1, tr2), tr3)

            # Directional Movement
            dm_plus = np.where(
                (high - high.shift(1)) > (low.shift(1) - low),
                np.maximum(high - high.shift(1), 0),
                0,
            )
            dm_minus = np.where(
                (low.shift(1) - low) > (high - high.shift(1)),
                np.maximum(low.shift(1) - low, 0),
                0,
            )

            # Smoothed averages
            import pandas as pd
            atr = pd.Series(tr).rolling(window=period, min_periods=1).mean()
            di_plus = 100 * pd.Series(dm_plus).rolling(period, min_periods=1).mean() / atr.replace(0, np.nan)
            di_minus = 100 * pd.Series(dm_minus).rolling(period, min_periods=1).mean() / atr.replace(0, np.nan)

            # DX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)
            return dx.fillna(0)

        col_name = f"dx_{period}"
        if "tic" in df.columns:
            df[col_name] = df.groupby("tic").apply(
                lambda x: calc_dx(x)
            ).reset_index(level=0, drop=True)
        else:
            df[col_name] = calc_dx(df)

        return df

    def add_atr(self, df: Any, period: int | None = None) -> Any:
        """
        Add Average True Range indicator.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns.
            period: ATR period.

        Returns:
            DataFrame with ATR column added.
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, skipping ATR")
            return df

        period = period or self.params["atr_period"]
        df = df.copy()

        def calc_atr(group: Any) -> Any:
            high = group["high"]
            low = group["low"]
            close = group["close"]

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = np.maximum(np.maximum(tr1, tr2), tr3)

            return tr.rolling(window=period, min_periods=1).mean()

        col_name = f"atr_{period}"
        if "tic" in df.columns:
            df[col_name] = df.groupby("tic").apply(
                lambda x: calc_atr(x)
            ).reset_index(level=0, drop=True)
        else:
            df[col_name] = calc_atr(df)

        return df

    def add_obv(self, df: Any) -> Any:
        """
        Add On-Balance Volume indicator.

        Args:
            df: DataFrame with 'close' and 'volume' columns.

        Returns:
            DataFrame with OBV column added.
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, skipping OBV")
            return df

        df = df.copy()

        def calc_obv(group: Any) -> Any:
            close = group["close"]
            volume = group["volume"]

            direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
            obv = (direction * volume).cumsum()
            return obv

        if "tic" in df.columns:
            df["obv"] = df.groupby("tic").apply(
                lambda x: calc_obv(x)
            ).reset_index(level=0, drop=True)
        else:
            df["obv"] = calc_obv(df)

        return df

    @staticmethod
    def calculate_turbulence(
        df: Any,
        lookback: int = 252,
    ) -> Any:
        """
        Calculate turbulence index for market regime detection.

        Turbulence measures how different the current market state is
        from historical norms using Mahalanobis distance.

        Args:
            df: DataFrame with returns data.
            lookback: Lookback period for covariance estimation.

        Returns:
            DataFrame with turbulence column added.
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, skipping turbulence")
            return df

        df = df.copy()

        # Calculate returns if not present
        if "return" not in df.columns:
            if "tic" in df.columns:
                df["return"] = df.groupby("tic")["close"].pct_change()
            else:
                df["return"] = df["close"].pct_change()

        # Pivot to get returns matrix if multiple tickers
        if "tic" in df.columns and "date" in df.columns:
            returns_df = df.pivot(index="date", columns="tic", values="return")
        else:
            returns_df = df[["return"]].copy()
            returns_df.columns = ["single_stock"]

        turbulence_values = []

        for i in range(len(returns_df)):
            if i < lookback:
                turbulence_values.append(0)
                continue

            historical = returns_df.iloc[i - lookback : i].dropna()
            current = returns_df.iloc[i : i + 1].values

            if len(historical) < lookback // 2:
                turbulence_values.append(0)
                continue

            try:
                mean = historical.mean().values
                cov = historical.cov().values

                # Check if covariance matrix is invertible
                if np.linalg.det(cov) < 1e-10:
                    turbulence_values.append(0)
                    continue

                diff = current - mean
                cov_inv = np.linalg.inv(cov)
                turb = float(np.sqrt(diff @ cov_inv @ diff.T))
                turbulence_values.append(turb)
            except Exception:
                turbulence_values.append(0)

        # Map turbulence back to original DataFrame
        if "tic" in df.columns and "date" in df.columns:
            turb_df = returns_df.copy()
            turb_df["turbulence"] = turbulence_values
            turb_df = turb_df[["turbulence"]].reset_index()
            df = df.merge(turb_df, on="date", how="left")
        else:
            df["turbulence"] = turbulence_values

        df["turbulence"] = df["turbulence"].fillna(0)
        return df
