"""
Correlation Monitor.

Monitors correlations between portfolio assets for risk management:
- Rolling correlation matrix calculation
- Tail correlation (stress correlation)
- Concentration metrics (HHI, effective positions)
- Correlation regime detection
- Diversification ratio

References:
- Markowitz Portfolio Theory
- Lopez de Prado "Advances in Financial ML"
- Risk.net correlation risk management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class CorrelationMethod(Enum):
    """Correlation calculation method."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class CorrelationRegime(Enum):
    """Correlation regime classification."""

    LOW = "low"  # Average correlation < 0.3
    NORMAL = "normal"  # Average correlation 0.3-0.6
    HIGH = "high"  # Average correlation 0.6-0.8
    CRISIS = "crisis"  # Average correlation > 0.8


@dataclass
class CorrelationMatrix:
    """Correlation matrix with metadata."""

    symbols: list[str]
    matrix: np.ndarray
    method: CorrelationMethod
    window_size: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        try:
            idx1 = self.symbols.index(symbol1)
            idx2 = self.symbols.index(symbol2)
            return float(self.matrix[idx1, idx2])
        except ValueError:
            return 0.0

    def get_average_correlation(self) -> float:
        """Get average off-diagonal correlation."""
        n = len(self.symbols)
        if n < 2:
            return 0.0
        # Extract upper triangle (excluding diagonal)
        upper = self.matrix[np.triu_indices(n, k=1)]
        return float(np.mean(upper))

    def get_max_correlation(self) -> tuple[str, str, float]:
        """Get pair with maximum correlation."""
        n = len(self.symbols)
        if n < 2:
            return ("", "", 0.0)

        # Mask diagonal
        masked = self.matrix.copy()
        np.fill_diagonal(masked, -np.inf)

        idx = np.unravel_index(np.argmax(masked), masked.shape)
        return (
            self.symbols[idx[0]],
            self.symbols[idx[1]],
            float(self.matrix[idx]),
        )

    def get_min_correlation(self) -> tuple[str, str, float]:
        """Get pair with minimum correlation."""
        n = len(self.symbols)
        if n < 2:
            return ("", "", 0.0)

        # Mask diagonal
        masked = self.matrix.copy()
        np.fill_diagonal(masked, np.inf)

        idx = np.unravel_index(np.argmin(masked), masked.shape)
        return (
            self.symbols[idx[0]],
            self.symbols[idx[1]],
            float(self.matrix[idx]),
        )


@dataclass
class ConcentrationMetrics:
    """Portfolio concentration metrics."""

    hhi: float  # Herfindahl-Hirschman Index (0-1)
    effective_positions: float  # 1/HHI
    max_weight: float  # Largest position weight
    max_weight_symbol: str
    top_3_weight: float  # Sum of top 3 weights
    is_concentrated: bool  # HHI > 0.25


@dataclass
class DiversificationMetrics:
    """Portfolio diversification metrics."""

    diversification_ratio: float  # DR = sum(w*sigma) / sigma_portfolio
    correlation_contribution: float  # How much correlation affects risk
    marginal_risk: dict[str, float]  # Marginal risk contribution by asset
    risk_parity_weights: dict[str, float]  # Equal risk contribution weights


@dataclass
class CorrelationAlert:
    """Alert for correlation regime change or anomaly."""

    alert_type: str  # "regime_change", "spike", "breakdown"
    description: str
    previous_value: float
    current_value: float
    symbols: list[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: str = "info"  # "info", "warning", "critical"


@dataclass
class CorrelationMonitorState:
    """Current state of correlation monitoring."""

    current_matrix: CorrelationMatrix | None
    regime: CorrelationRegime
    average_correlation: float
    concentration: ConcentrationMetrics | None
    diversification: DiversificationMetrics | None
    recent_alerts: list[CorrelationAlert]
    last_update: datetime = field(default_factory=datetime.utcnow)


class CorrelationMonitor:
    """
    Correlation Monitor for portfolio risk.

    Tracks correlations between assets and monitors for:
    - Correlation regime changes
    - Concentration risk
    - Diversification effectiveness
    - Tail correlation during stress

    Example:
        monitor = CorrelationMonitor(window_size=60)

        # Update with new returns
        matrix = monitor.update(returns_dict)

        # Check concentration
        conc = monitor.calculate_concentration(weights)
        if conc.is_concentrated:
            print(f"Warning: HHI = {conc.hhi:.2f}")

        # Get diversification metrics
        div = monitor.calculate_diversification(returns_dict, weights)
        print(f"Diversification ratio: {div.diversification_ratio:.2f}")
    """

    def __init__(
        self,
        window_size: int = 60,
        method: CorrelationMethod = CorrelationMethod.PEARSON,
        alert_threshold: float = 0.2,
        regime_thresholds: tuple[float, float, float] = (0.3, 0.6, 0.8),
    ) -> None:
        """
        Initialize correlation monitor.

        Args:
            window_size: Rolling window for correlation calculation
            method: Correlation calculation method
            alert_threshold: Change threshold to trigger alerts
            regime_thresholds: (low/normal, normal/high, high/crisis) boundaries
        """
        self.window_size = window_size
        self.method = method
        self.alert_threshold = alert_threshold
        self.regime_thresholds = regime_thresholds

        self._history: dict[str, list[float]] = {}
        self._current_matrix: CorrelationMatrix | None = None
        self._previous_matrix: CorrelationMatrix | None = None
        self._alerts: list[CorrelationAlert] = []
        self._regime: CorrelationRegime = CorrelationRegime.NORMAL

    def update(
        self,
        returns: dict[str, float] | dict[str, Sequence[float]],
    ) -> CorrelationMatrix | None:
        """
        Update correlation matrix with new returns.

        Args:
            returns: Either single return per symbol or full return series

        Returns:
            Updated CorrelationMatrix or None if not enough data
        """
        # Handle single returns vs full series
        first_value = next(iter(returns.values())) if returns else None
        if first_value is not None and isinstance(first_value, (int, float)):
            # Single return - add to history
            for symbol, ret in returns.items():
                if symbol not in self._history:
                    self._history[symbol] = []
                self._history[symbol].append(float(ret))  # type: ignore[arg-type]
                # Trim to window
                if len(self._history[symbol]) > self.window_size:
                    self._history[symbol] = self._history[symbol][-self.window_size :]
        else:
            # Full series - replace history
            self._history = {
                symbol: [float(x) for x in rets][-self.window_size :]  # type: ignore[union-attr]
                for symbol, rets in returns.items()
            }

        # Check if we have enough data
        symbols = list(self._history.keys())
        if len(symbols) < 2:
            return None

        min_len = min(len(self._history[s]) for s in symbols)
        if min_len < 10:  # Need minimum observations
            return None

        # Build return matrix
        returns_matrix = np.column_stack(
            [self._history[s][-min_len:] for s in symbols]
        )

        # Calculate correlation matrix
        if self.method == CorrelationMethod.PEARSON:
            corr_matrix = np.corrcoef(returns_matrix, rowvar=False)
        elif self.method == CorrelationMethod.SPEARMAN:
            corr_matrix, _ = stats.spearmanr(returns_matrix)
            if corr_matrix.ndim == 0:
                corr_matrix = np.array([[1.0, float(corr_matrix)], [float(corr_matrix), 1.0]])
        else:  # Kendall
            n = len(symbols)
            corr_matrix = np.eye(n)
            for i in range(n):
                for j in range(i + 1, n):
                    tau, _ = stats.kendalltau(
                        returns_matrix[:, i], returns_matrix[:, j]
                    )
                    corr_matrix[i, j] = tau
                    corr_matrix[j, i] = tau

        # Store previous matrix
        self._previous_matrix = self._current_matrix

        # Create new matrix
        self._current_matrix = CorrelationMatrix(
            symbols=symbols,
            matrix=corr_matrix,
            method=self.method,
            window_size=min_len,
        )

        # Check for regime change and generate alerts
        self._check_regime_change()
        self._check_correlation_spikes()

        return self._current_matrix

    def calculate_correlation_matrix(
        self,
        returns: dict[str, np.ndarray],
        method: CorrelationMethod | None = None,
    ) -> CorrelationMatrix:
        """
        Calculate correlation matrix from return arrays.

        Args:
            returns: Dict of symbol -> return array
            method: Override default method

        Returns:
            CorrelationMatrix
        """
        method = method or self.method
        symbols = list(returns.keys())

        if len(symbols) < 2:
            raise ValueError("Need at least 2 symbols")

        # Align lengths
        min_len = min(len(returns[s]) for s in symbols)
        returns_matrix = np.column_stack([returns[s][-min_len:] for s in symbols])

        if method == CorrelationMethod.PEARSON:
            corr_matrix = np.corrcoef(returns_matrix, rowvar=False)
        elif method == CorrelationMethod.SPEARMAN:
            corr_matrix, _ = stats.spearmanr(returns_matrix)
            if corr_matrix.ndim == 0:
                corr_matrix = np.array([[1.0, float(corr_matrix)], [float(corr_matrix), 1.0]])
        else:
            n = len(symbols)
            corr_matrix = np.eye(n)
            for i in range(n):
                for j in range(i + 1, n):
                    tau, _ = stats.kendalltau(
                        returns_matrix[:, i], returns_matrix[:, j]
                    )
                    corr_matrix[i, j] = tau
                    corr_matrix[j, i] = tau

        return CorrelationMatrix(
            symbols=symbols,
            matrix=corr_matrix,
            method=method,
            window_size=min_len,
        )

    def calculate_tail_correlation(
        self,
        returns: dict[str, np.ndarray],
        threshold_percentile: float = 10.0,
    ) -> CorrelationMatrix:
        """
        Calculate tail correlation (correlation during extreme moves).

        Uses only returns in the lower/upper percentiles to measure
        how assets correlate during stress periods.

        Args:
            returns: Dict of symbol -> return array
            threshold_percentile: Percentile threshold for tail

        Returns:
            CorrelationMatrix for tail events
        """
        symbols = list(returns.keys())
        if len(symbols) < 2:
            raise ValueError("Need at least 2 symbols")

        # Align lengths
        min_len = min(len(returns[s]) for s in symbols)
        returns_matrix = np.column_stack([returns[s][-min_len:] for s in symbols])

        # Find tail events (either tail)
        lower_threshold = np.percentile(returns_matrix, threshold_percentile, axis=0)
        upper_threshold = np.percentile(returns_matrix, 100 - threshold_percentile, axis=0)

        # Mask for extreme observations (any asset in tail)
        tail_mask = np.any(
            (returns_matrix < lower_threshold) | (returns_matrix > upper_threshold),
            axis=1,
        )

        tail_returns = returns_matrix[tail_mask]

        if len(tail_returns) < 5:
            logger.warning("Too few tail observations, using all data")
            tail_returns = returns_matrix

        corr_matrix = np.corrcoef(tail_returns, rowvar=False)

        return CorrelationMatrix(
            symbols=symbols,
            matrix=corr_matrix,
            method=CorrelationMethod.PEARSON,
            window_size=len(tail_returns),
        )

    def calculate_concentration(
        self,
        weights: dict[str, float | Decimal],
    ) -> ConcentrationMetrics:
        """
        Calculate portfolio concentration metrics.

        Args:
            weights: Dict of symbol -> weight (can be value or proportion)

        Returns:
            ConcentrationMetrics
        """
        if not weights:
            return ConcentrationMetrics(
                hhi=0.0,
                effective_positions=0.0,
                max_weight=0.0,
                max_weight_symbol="",
                top_3_weight=0.0,
                is_concentrated=False,
            )

        # Normalize weights
        total = sum(float(w) for w in weights.values())
        if total == 0:
            return ConcentrationMetrics(
                hhi=0.0,
                effective_positions=0.0,
                max_weight=0.0,
                max_weight_symbol="",
                top_3_weight=0.0,
                is_concentrated=False,
            )

        normalized = {s: float(w) / total for s, w in weights.items()}

        # HHI = sum of squared weights
        hhi = sum(w**2 for w in normalized.values())

        # Effective number of positions = 1/HHI
        effective_positions = 1 / hhi if hhi > 0 else 0

        # Max weight
        max_symbol = max(normalized.keys(), key=lambda s: normalized[s])
        max_weight = normalized[max_symbol]

        # Top 3 weight
        sorted_weights = sorted(normalized.values(), reverse=True)
        top_3_weight = sum(sorted_weights[:3])

        return ConcentrationMetrics(
            hhi=hhi,
            effective_positions=effective_positions,
            max_weight=max_weight,
            max_weight_symbol=max_symbol,
            top_3_weight=top_3_weight,
            is_concentrated=hhi > 0.25,
        )

    def calculate_diversification(
        self,
        returns: dict[str, np.ndarray],
        weights: dict[str, float | Decimal],
    ) -> DiversificationMetrics:
        """
        Calculate portfolio diversification metrics.

        Includes diversification ratio, which measures how much
        correlation reduces portfolio risk vs uncorrelated assets.

        Args:
            returns: Dict of symbol -> return array
            weights: Dict of symbol -> weight

        Returns:
            DiversificationMetrics
        """
        symbols = list(returns.keys())
        if len(symbols) < 2:
            return DiversificationMetrics(
                diversification_ratio=1.0,
                correlation_contribution=0.0,
                marginal_risk={},
                risk_parity_weights={},
            )

        # Normalize weights
        total = sum(float(weights.get(s, 0)) for s in symbols)
        w = np.array([float(weights.get(s, 0)) / total for s in symbols])

        # Build return matrix and calculate covariance
        min_len = min(len(returns[s]) for s in symbols)
        returns_matrix = np.column_stack([returns[s][-min_len:] for s in symbols])
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        volatilities = np.sqrt(np.diag(cov_matrix))

        # Portfolio volatility
        port_variance = w @ cov_matrix @ w
        port_vol = np.sqrt(port_variance)

        # Weighted average volatility (if uncorrelated)
        weighted_vol = w @ volatilities

        # Diversification ratio
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1.0

        # Correlation contribution (how much correlation adds to risk)
        # = 1 - (port_vol / weighted_vol)^2
        corr_contribution = 1 - (port_vol / weighted_vol) ** 2 if weighted_vol > 0 else 0.0

        # Marginal risk contribution
        marginal_risk = {}
        if port_vol > 0:
            marginal_var = cov_matrix @ w / port_vol
            for i, symbol in enumerate(symbols):
                marginal_risk[symbol] = float(marginal_var[i])

        # Risk parity weights (equal risk contribution)
        risk_parity_weights = self._calculate_risk_parity_weights(
            cov_matrix, symbols
        )

        return DiversificationMetrics(
            diversification_ratio=float(div_ratio),
            correlation_contribution=float(corr_contribution),
            marginal_risk=marginal_risk,
            risk_parity_weights=risk_parity_weights,
        )

    def _calculate_risk_parity_weights(
        self,
        cov_matrix: np.ndarray,
        symbols: list[str],
        max_iterations: int = 100,
    ) -> dict[str, float]:
        """
        Calculate risk parity weights (equal risk contribution).

        Uses iterative algorithm to find weights where each asset
        contributes equally to portfolio risk.
        """
        n = len(symbols)
        volatilities = np.sqrt(np.diag(cov_matrix))

        # Initial weights: inverse volatility
        weights = 1 / volatilities
        weights = weights / weights.sum()

        for _ in range(max_iterations):
            # Portfolio volatility
            port_var = weights @ cov_matrix @ weights
            port_vol = np.sqrt(port_var)

            # Marginal risk contribution
            marginal = cov_matrix @ weights / port_vol

            # Risk contribution
            risk_contrib = weights * marginal

            # Target equal contribution
            target = port_vol / n

            # Update weights
            new_weights = weights * (target / risk_contrib)
            new_weights = new_weights / new_weights.sum()

            # Check convergence
            if np.allclose(weights, new_weights, rtol=1e-6):
                break

            weights = new_weights

        return {symbols[i]: float(weights[i]) for i in range(n)}

    def _check_regime_change(self) -> None:
        """Check if correlation regime has changed."""
        if self._current_matrix is None:
            return

        avg_corr = self._current_matrix.get_average_correlation()

        # Determine regime
        if avg_corr < self.regime_thresholds[0]:
            new_regime = CorrelationRegime.LOW
        elif avg_corr < self.regime_thresholds[1]:
            new_regime = CorrelationRegime.NORMAL
        elif avg_corr < self.regime_thresholds[2]:
            new_regime = CorrelationRegime.HIGH
        else:
            new_regime = CorrelationRegime.CRISIS

        # Alert if changed
        if new_regime != self._regime:
            severity = "info"
            if new_regime == CorrelationRegime.HIGH:
                severity = "warning"
            elif new_regime == CorrelationRegime.CRISIS:
                severity = "critical"

            alert = CorrelationAlert(
                alert_type="regime_change",
                description=f"Correlation regime changed from {self._regime.value} to {new_regime.value}",
                previous_value=self.regime_thresholds[1],  # Placeholder
                current_value=avg_corr,
                symbols=self._current_matrix.symbols,
                severity=severity,
            )
            self._alerts.append(alert)
            self._regime = new_regime

    def _check_correlation_spikes(self) -> None:
        """Check for sudden correlation changes between specific pairs."""
        if self._current_matrix is None or self._previous_matrix is None:
            return

        # Compare matrices
        current_symbols = set(self._current_matrix.symbols)
        previous_symbols = set(self._previous_matrix.symbols)
        common_symbols = list(current_symbols & previous_symbols)

        for i, sym1 in enumerate(common_symbols):
            for sym2 in common_symbols[i + 1 :]:
                curr_corr = self._current_matrix.get_correlation(sym1, sym2)
                prev_corr = self._previous_matrix.get_correlation(sym1, sym2)

                change = abs(curr_corr - prev_corr)
                if change > self.alert_threshold:
                    severity = "warning" if change > 0.3 else "info"
                    direction = "increased" if curr_corr > prev_corr else "decreased"

                    alert = CorrelationAlert(
                        alert_type="spike",
                        description=f"Correlation between {sym1} and {sym2} {direction} by {change:.2f}",
                        previous_value=prev_corr,
                        current_value=curr_corr,
                        symbols=[sym1, sym2],
                        severity=severity,
                    )
                    self._alerts.append(alert)

    def get_state(self) -> CorrelationMonitorState:
        """Get current monitor state."""
        return CorrelationMonitorState(
            current_matrix=self._current_matrix,
            regime=self._regime,
            average_correlation=(
                self._current_matrix.get_average_correlation()
                if self._current_matrix
                else 0.0
            ),
            concentration=None,
            diversification=None,
            recent_alerts=self._alerts[-10:],  # Last 10 alerts
        )

    def get_alerts(self, limit: int = 10) -> list[CorrelationAlert]:
        """Get recent alerts."""
        return self._alerts[-limit:]

    def clear_alerts(self) -> None:
        """Clear alert history."""
        self._alerts.clear()

    @property
    def current_matrix(self) -> CorrelationMatrix | None:
        """Get current correlation matrix."""
        return self._current_matrix

    @property
    def regime(self) -> CorrelationRegime:
        """Get current correlation regime."""
        return self._regime


def create_correlation_monitor(
    window_size: int = 60,
    method: str = "pearson",
) -> CorrelationMonitor:
    """
    Factory function to create correlation monitor.

    Args:
        window_size: Rolling window size
        method: Correlation method ("pearson", "spearman", "kendall")

    Returns:
        Configured CorrelationMonitor
    """
    method_enum = CorrelationMethod(method.lower())
    return CorrelationMonitor(window_size=window_size, method=method_enum)
