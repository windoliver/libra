"""
Value at Risk (VaR) Calculator.

Implements multiple VaR calculation methodologies:
- Historical VaR: Non-parametric, uses actual return distribution
- Parametric VaR: Assumes normal distribution, uses volatility
- Monte Carlo VaR: Simulation-based with correlation structure
- Expected Shortfall (CVaR): Average loss beyond VaR threshold

References:
- Basel III regulatory framework
- RiskMetrics Technical Document
- Jorion, "Value at Risk" (3rd Edition)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

import msgspec
import numpy as np
from numba import njit, prange
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Optional Rust backend for EWMA (Issue #97: ~50x additional speedup)
try:
    from libra_core_rs import ewma_volatility_scalar as _ewma_volatility_rust

    RUST_EWMA_AVAILABLE = True
except ImportError:
    RUST_EWMA_AVAILABLE = False
    _ewma_volatility_rust = None  # type: ignore

# Optional Rust backend for Monte Carlo VaR (Issue #100: parallel simulation)
try:
    from libra_core_rs import monte_carlo_var as _monte_carlo_var_rust

    RUST_MONTE_CARLO_AVAILABLE = True
except ImportError:
    RUST_MONTE_CARLO_AVAILABLE = False
    _monte_carlo_var_rust = None  # type: ignore

# Optional GPU backend for large Monte Carlo (Issue #103: 100x+ speedup)
try:
    import cupy as cp

    GPU_AVAILABLE = True
    GPU_DEVICE_NAME = cp.cuda.Device().name.decode() if cp.cuda.is_available() else None
    logger.info("GPU acceleration available: %s", GPU_DEVICE_NAME)
except ImportError:
    GPU_AVAILABLE = False
    GPU_DEVICE_NAME = None
    cp = None  # type: ignore


# =============================================================================
# GPU Monte Carlo VaR (Issue #103: 100x+ speedup for large simulations)
# =============================================================================


# Threshold for GPU usage - below this, CPU/Rust is faster due to transfer overhead
GPU_MIN_SIMULATIONS = 100_000


def _monte_carlo_var_gpu(
    mean_return: float,
    volatility: float,
    horizon: int,
    n_sims: int,
    confidence: float,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    GPU-accelerated Monte Carlo VaR using CuPy (NVIDIA CUDA).

    Provides 100x+ speedup for large simulations (n_sims > 100,000).
    Automatically falls back to CPU for smaller simulations.

    Args:
        mean_return: Estimated mean return
        volatility: Estimated volatility
        horizon: Time horizon in days
        n_sims: Number of simulations
        confidence: Confidence level (e.g., 0.95)
        seed: Optional random seed for reproducibility

    Returns:
        Tuple of (VaR%, CVaR%) as positive losses
    """
    if not GPU_AVAILABLE or cp is None:
        raise RuntimeError("GPU not available")

    # Set seed if provided
    if seed is not None:
        cp.random.seed(seed)

    # Generate simulated returns on GPU
    horizon_f = float(horizon)
    simulated_returns = cp.random.normal(
        mean_return * horizon_f,
        volatility * cp.sqrt(horizon_f),
        n_sims,
    )

    # Sort on GPU
    sorted_returns = cp.sort(simulated_returns)

    # Calculate VaR percentile
    var_idx = int((1.0 - confidence) * n_sims)
    var_idx = min(var_idx, n_sims - 1)

    # VaR: negative return at percentile (positive loss)
    var_pct = float(-sorted_returns[var_idx].get())

    # CVaR: average of worst losses
    if var_idx > 0:
        cvar_pct = float(-cp.mean(sorted_returns[:var_idx]).get())
    else:
        cvar_pct = var_pct

    return max(0.0, var_pct), max(0.0, cvar_pct)


# =============================================================================
# Numba-compiled EWMA (Issue #70: ~50x speedup)
# =============================================================================


@njit(cache=True, fastmath=True)
def _ewma_volatility_numba(returns: np.ndarray, lam: float) -> float:
    """
    Numba JIT-compiled EWMA volatility calculation.

    Uses RiskMetrics methodology with decay factor lambda.
    Compiled to native code for ~50x speedup over pure Python.

    Args:
        returns: Historical returns array
        lam: Decay factor (typically 0.94 for daily data)

    Returns:
        EWMA volatility estimate
    """
    n = len(returns)
    if n == 0:
        return 0.0

    # Calculate mean (manual for Numba compatibility)
    mean = 0.0
    for i in range(n):
        mean += returns[i]
    mean /= n

    # Calculate initial variance (ddof=1)
    variance = 0.0
    for i in range(n):
        variance += (returns[i] - mean) ** 2
    if n > 1:
        variance /= n - 1
    else:
        variance = 0.0

    # Apply EWMA recursion
    for i in range(1, n):
        variance = lam * variance + (1.0 - lam) * returns[i] ** 2

    return np.sqrt(variance)


# =============================================================================
# Numba-compiled VaR Backtest Loop (Issue #74: ~10-50x speedup)
# =============================================================================


@njit(cache=True, parallel=True)
def _var_backtest_loop_historical(
    returns: np.ndarray,
    portfolio_values: np.ndarray,
    window: int,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Numba JIT-compiled VaR backtest loop for historical method.

    Uses parallel execution with prange for multi-core speedup.
    Pre-allocates output arrays for efficiency.

    Args:
        returns: Full historical returns array
        portfolio_values: Portfolio values over time
        window: Rolling window size
        confidence: VaR confidence level (e.g., 0.95)

    Returns:
        Tuple of (var_estimates, actual_losses, exceptions_count)
    """
    n = len(returns) - window
    var_estimates = np.empty(n, dtype=np.float64)
    actual_losses = np.empty(n, dtype=np.float64)
    percentile_level = (1.0 - confidence) * 100.0

    # Thread-local exception counting (will sum after)
    exceptions = 0

    for i in prange(window, len(returns)):
        idx = i - window
        window_returns = returns[i - window : i]
        port_value = portfolio_values[i - 1]

        # Historical VaR: percentile of window returns * portfolio value
        # Note: VaR is typically negative (loss), so we use abs of percentile
        var_pct = np.percentile(window_returns, percentile_level)
        var_estimate = abs(var_pct) * port_value
        var_estimates[idx] = var_estimate

        # Actual loss for this period
        actual_loss = -returns[i] * port_value
        actual_losses[idx] = actual_loss

        # Count exception (actual loss exceeded VaR)
        if actual_loss > var_estimate:
            exceptions += 1

    return var_estimates, actual_losses, exceptions


# Warmup JIT compilation at module load (compiles once, cached to disk)
_ewma_volatility_numba(np.array([0.01, -0.01, 0.02], dtype=np.float64), 0.94)
_var_backtest_loop_historical(
    np.array([0.01, -0.01, 0.02, 0.01, -0.02], dtype=np.float64),
    np.array([100.0, 101.0, 100.0, 102.0, 101.0], dtype=np.float64),
    2,
    0.95,
)


class VaRMethod(Enum):
    """VaR calculation methodology."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


class ConfidenceLevel(Enum):
    """Standard confidence levels for VaR."""

    CL_90 = 0.90
    CL_95 = 0.95
    CL_99 = 0.99
    CL_995 = 0.995


class VaRResult(msgspec.Struct, frozen=True, gc=False):
    """
    Result of VaR calculation.

    Uses msgspec.Struct for ~4x faster creation and ~30% less memory (Issue #72).
    frozen=True makes instances immutable, gc=False reduces GC overhead.
    """

    var: Decimal  # Value at Risk (positive = potential loss)
    cvar: Decimal  # Conditional VaR / Expected Shortfall
    confidence_level: float
    time_horizon_days: int
    method: VaRMethod
    portfolio_value: Decimal
    var_pct: float  # VaR as percentage of portfolio
    cvar_pct: float  # CVaR as percentage of portfolio
    timestamp: datetime | None = None  # Default: caller should set or leave None
    num_observations: int = 0
    num_simulations: int = 0  # For Monte Carlo


@dataclass
class PositionVaR:
    """VaR breakdown by position."""

    symbol: str
    position_value: Decimal
    var: Decimal
    cvar: Decimal
    var_pct: float
    contribution_pct: float  # Contribution to portfolio VaR


@dataclass
class VaRConfig:
    """Configuration for VaR calculations."""

    # Default confidence level
    confidence_level: float = 0.95

    # Time horizon in days (1-day or 10-day per Basel)
    time_horizon_days: int = 1

    # Historical VaR settings
    lookback_days: int = 252  # 1 year of trading days
    min_observations: int = 30  # Minimum for calculation

    # Monte Carlo settings
    num_simulations: int = 10000
    seed: int | None = None
    use_gpu: bool = True  # Use GPU for large MC if available (Issue #103)
    gpu_min_sims: int = 100_000  # Minimum sims to use GPU (below this, CPU is faster)

    # Parametric settings
    use_ewma: bool = True  # Exponentially weighted volatility
    ewma_lambda: float = 0.94  # RiskMetrics decay factor
    use_rust_ewma: bool = True  # Use Rust backend if available (Issue #97)

    # Scaling
    sqrt_time_scaling: bool = True  # Scale VaR by sqrt(time)


class VaRCalculator:
    """
    Value at Risk Calculator.

    Calculates portfolio VaR using multiple methodologies with
    proper time scaling and confidence level adjustments.

    Example:
        calculator = VaRCalculator(config=VaRConfig(confidence_level=0.99))

        # Calculate VaR from historical returns
        result = calculator.calculate_historical_var(
            returns=daily_returns,
            portfolio_value=Decimal("100000"),
        )

        print(f"99% 1-day VaR: ${result.var:,.2f}")
        print(f"Expected Shortfall: ${result.cvar:,.2f}")
    """

    def __init__(self, config: VaRConfig | None = None) -> None:
        """Initialize VaR calculator."""
        self.config = config or VaRConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def calculate_historical_var(
        self,
        returns: Sequence[float] | np.ndarray,
        portfolio_value: Decimal,
        confidence_level: float | None = None,
        time_horizon_days: int | None = None,
    ) -> VaRResult:
        """
        Calculate Historical VaR.

        Uses the empirical distribution of returns without
        assuming any parametric form.

        Args:
            returns: Historical returns (daily)
            portfolio_value: Current portfolio value
            confidence_level: Override config confidence level
            time_horizon_days: Override config time horizon

        Returns:
            VaRResult with VaR and CVaR
        """
        returns_arr = np.asarray(returns)
        conf = confidence_level or self.config.confidence_level
        horizon = time_horizon_days or self.config.time_horizon_days

        if len(returns_arr) < self.config.min_observations:
            raise ValueError(
                f"Need at least {self.config.min_observations} observations, "
                f"got {len(returns_arr)}"
            )

        # Calculate VaR percentile (left tail for losses)
        var_pct = np.percentile(returns_arr, (1 - conf) * 100)

        # Calculate CVaR (Expected Shortfall) - average of returns below VaR
        cvar_returns = returns_arr[returns_arr <= var_pct]
        cvar_pct = float(np.mean(cvar_returns)) if len(cvar_returns) > 0 else var_pct

        # Scale for time horizon (sqrt-time rule)
        if self.config.sqrt_time_scaling and horizon > 1:
            scaling = np.sqrt(horizon)
            var_pct *= scaling
            cvar_pct *= scaling

        # Convert to dollar amounts (negative return = positive VaR)
        var_amount = Decimal(str(abs(var_pct))) * portfolio_value
        cvar_amount = Decimal(str(abs(cvar_pct))) * portfolio_value

        return VaRResult(
            var=var_amount,
            cvar=cvar_amount,
            confidence_level=conf,
            time_horizon_days=horizon,
            method=VaRMethod.HISTORICAL,
            portfolio_value=portfolio_value,
            var_pct=float(abs(var_pct)) * 100,
            cvar_pct=float(abs(cvar_pct)) * 100,
            num_observations=len(returns_arr),
        )

    def calculate_parametric_var(
        self,
        returns: Sequence[float] | np.ndarray,
        portfolio_value: Decimal,
        confidence_level: float | None = None,
        time_horizon_days: int | None = None,
    ) -> VaRResult:
        """
        Calculate Parametric (Variance-Covariance) VaR.

        Assumes returns are normally distributed and uses
        volatility to estimate VaR.

        Args:
            returns: Historical returns for volatility estimation
            portfolio_value: Current portfolio value
            confidence_level: Override config confidence level
            time_horizon_days: Override config time horizon

        Returns:
            VaRResult with VaR and CVaR
        """
        returns_arr = np.asarray(returns)
        conf = confidence_level or self.config.confidence_level
        horizon = time_horizon_days or self.config.time_horizon_days

        if len(returns_arr) < self.config.min_observations:
            raise ValueError(
                f"Need at least {self.config.min_observations} observations, "
                f"got {len(returns_arr)}"
            )

        # Calculate volatility (optionally EWMA)
        if self.config.use_ewma:
            volatility = self._calculate_ewma_volatility(returns_arr)
        else:
            volatility = float(np.std(returns_arr, ddof=1))

        mean_return = float(np.mean(returns_arr))

        # Calculate z-score for confidence level
        z_score = stats.norm.ppf(1 - conf)

        # VaR = -mean + z * volatility (for losses)
        var_pct = -(mean_return + z_score * volatility)

        # CVaR for normal distribution
        # E[X | X < VaR] = mean - volatility * phi(z) / (1-conf)
        pdf_at_z = stats.norm.pdf(z_score)
        cvar_pct = -(mean_return - volatility * pdf_at_z / (1 - conf))

        # Scale for time horizon
        if self.config.sqrt_time_scaling and horizon > 1:
            scaling = np.sqrt(horizon)
            var_pct *= scaling
            cvar_pct *= scaling

        # Convert to dollar amounts
        var_amount = Decimal(str(abs(var_pct))) * portfolio_value
        cvar_amount = Decimal(str(abs(cvar_pct))) * portfolio_value

        return VaRResult(
            var=var_amount,
            cvar=cvar_amount,
            confidence_level=conf,
            time_horizon_days=horizon,
            method=VaRMethod.PARAMETRIC,
            portfolio_value=portfolio_value,
            var_pct=abs(var_pct) * 100,
            cvar_pct=abs(cvar_pct) * 100,
            num_observations=len(returns_arr),
        )

    def calculate_monte_carlo_var(
        self,
        returns: Sequence[float] | np.ndarray,
        portfolio_value: Decimal,
        confidence_level: float | None = None,
        time_horizon_days: int | None = None,
        num_simulations: int | None = None,
    ) -> VaRResult:
        """
        Calculate Monte Carlo VaR.

        Uses Rust parallel simulation when available (Issue #100: ~5-10x speedup).
        Falls back to NumPy implementation otherwise.

        Simulates future returns based on historical distribution
        parameters and calculates VaR from simulated scenarios.

        Args:
            returns: Historical returns for parameter estimation
            portfolio_value: Current portfolio value
            confidence_level: Override config confidence level
            time_horizon_days: Override config time horizon
            num_simulations: Override config number of simulations

        Returns:
            VaRResult with VaR and CVaR
        """
        returns_arr = np.asarray(returns)
        conf = confidence_level or self.config.confidence_level
        horizon = time_horizon_days or self.config.time_horizon_days
        n_sims = num_simulations or self.config.num_simulations

        if len(returns_arr) < self.config.min_observations:
            raise ValueError(
                f"Need at least {self.config.min_observations} observations, "
                f"got {len(returns_arr)}"
            )

        # Estimate parameters
        mean_return = float(np.mean(returns_arr))
        volatility = float(np.std(returns_arr, ddof=1))

        # Use GPU for large simulations (Issue #103: 100x+ speedup)
        if (
            GPU_AVAILABLE
            and self.config.use_gpu
            and n_sims >= self.config.gpu_min_sims
        ):
            var_pct, cvar_pct = _monte_carlo_var_gpu(
                mean_return,
                volatility,
                horizon,
                n_sims,
                conf,
                self.config.seed,
            )
            logger.debug(
                "GPU Monte Carlo VaR: %d sims on %s", n_sims, GPU_DEVICE_NAME
            )
        # Use Rust implementation if available (parallel, faster RNG)
        elif RUST_MONTE_CARLO_AVAILABLE and _monte_carlo_var_rust is not None:
            var_pct, cvar_pct = _monte_carlo_var_rust(
                mean_return,
                volatility,
                horizon,
                n_sims,
                conf,
                self.config.seed,
            )
        else:
            # Fallback to NumPy implementation
            simulated_returns = self._rng.normal(
                mean_return * horizon,
                volatility * np.sqrt(horizon),
                n_sims,
            )

            # Calculate VaR from simulated distribution
            var_pct_raw = np.percentile(simulated_returns, (1 - conf) * 100)

            # Calculate CVaR
            cvar_returns = simulated_returns[simulated_returns <= var_pct_raw]
            cvar_pct_raw = (
                float(np.mean(cvar_returns)) if len(cvar_returns) > 0 else var_pct_raw
            )

            var_pct = abs(var_pct_raw)
            cvar_pct = abs(cvar_pct_raw)

        # Convert to dollar amounts
        var_amount = Decimal(str(var_pct)) * portfolio_value
        cvar_amount = Decimal(str(cvar_pct)) * portfolio_value

        return VaRResult(
            var=var_amount,
            cvar=cvar_amount,
            confidence_level=conf,
            time_horizon_days=horizon,
            method=VaRMethod.MONTE_CARLO,
            portfolio_value=portfolio_value,
            var_pct=var_pct * 100,
            cvar_pct=cvar_pct * 100,
            num_observations=len(returns_arr),
            num_simulations=n_sims,
        )

    def calculate_portfolio_var(
        self,
        position_returns: dict[str, np.ndarray],
        position_values: dict[str, Decimal],
        method: VaRMethod = VaRMethod.HISTORICAL,
        confidence_level: float | None = None,
        time_horizon_days: int | None = None,
    ) -> tuple[VaRResult, list[PositionVaR]]:
        """
        Calculate portfolio VaR with position-level breakdown.

        Accounts for correlations between positions when calculating
        portfolio VaR, and provides marginal VaR contribution.

        Args:
            position_returns: Dict of symbol -> returns array
            position_values: Dict of symbol -> position value
            method: VaR calculation method
            confidence_level: Override config confidence level
            time_horizon_days: Override config time horizon

        Returns:
            Tuple of (portfolio VaR, list of position VaRs)
        """
        conf = confidence_level or self.config.confidence_level
        horizon = time_horizon_days or self.config.time_horizon_days

        symbols = list(position_returns.keys())
        if not symbols:
            raise ValueError("No positions provided")

        # Validate all return series have same length
        lengths = [len(position_returns[s]) for s in symbols]
        if len(set(lengths)) > 1:
            min_len = min(lengths)
            logger.warning(
                "Return series have different lengths, truncating to %d", min_len
            )
            position_returns = {
                s: position_returns[s][-min_len:] for s in symbols
            }

        # Calculate portfolio value
        portfolio_value = sum(position_values.values(), Decimal("0"))

        # Calculate weights
        weights = np.array(
            [float(position_values[s] / portfolio_value) for s in symbols]
        )

        # Build return matrix
        returns_matrix = np.column_stack([position_returns[s] for s in symbols])

        # Calculate portfolio returns
        portfolio_returns = returns_matrix @ weights

        # Calculate portfolio VaR
        if method == VaRMethod.HISTORICAL:
            portfolio_var = self.calculate_historical_var(
                portfolio_returns, portfolio_value, conf, horizon
            )
        elif method == VaRMethod.PARAMETRIC:
            portfolio_var = self.calculate_parametric_var(
                portfolio_returns, portfolio_value, conf, horizon
            )
        else:
            portfolio_var = self.calculate_monte_carlo_var(
                portfolio_returns, portfolio_value, conf, horizon
            )

        # Calculate individual position VaRs
        position_vars: list[PositionVaR] = []
        total_individual_var = Decimal("0")

        for symbol in symbols:
            pos_value = position_values[symbol]
            pos_returns = position_returns[symbol]

            if method == VaRMethod.HISTORICAL:
                pos_var_result = self.calculate_historical_var(
                    pos_returns, pos_value, conf, horizon
                )
            elif method == VaRMethod.PARAMETRIC:
                pos_var_result = self.calculate_parametric_var(
                    pos_returns, pos_value, conf, horizon
                )
            else:
                pos_var_result = self.calculate_monte_carlo_var(
                    pos_returns, pos_value, conf, horizon
                )

            total_individual_var += pos_var_result.var
            position_vars.append(
                PositionVaR(
                    symbol=symbol,
                    position_value=pos_value,
                    var=pos_var_result.var,
                    cvar=pos_var_result.cvar,
                    var_pct=pos_var_result.var_pct,
                    contribution_pct=0.0,  # Calculate after
                )
            )

        # Calculate contribution percentages (marginal VaR contribution)
        # This is simplified - true marginal VaR requires covariance matrix
        for pos_var in position_vars:
            if portfolio_var.var > 0:
                # Weight contribution by individual VaR
                pos_var.contribution_pct = float(pos_var.var / total_individual_var) * 100

        return portfolio_var, position_vars

    def calculate_component_var(
        self,
        position_returns: dict[str, np.ndarray],
        position_values: dict[str, Decimal],
        confidence_level: float | None = None,
    ) -> dict[str, Decimal]:
        """
        Calculate Component VaR (marginal contribution to portfolio VaR).

        Component VaR measures how much each position contributes
        to total portfolio VaR, accounting for correlations.

        Args:
            position_returns: Dict of symbol -> returns array
            position_values: Dict of symbol -> position value
            confidence_level: Override config confidence level

        Returns:
            Dict of symbol -> component VaR
        """
        conf = confidence_level or self.config.confidence_level

        symbols = list(position_returns.keys())
        if not symbols:
            return {}

        # Build return matrix and calculate covariance
        returns_matrix = np.column_stack([position_returns[s] for s in symbols])
        cov_matrix = np.cov(returns_matrix, rowvar=False)

        # Portfolio value and weights
        portfolio_value = sum(position_values.values(), Decimal("0"))
        weights = np.array(
            [float(position_values[s] / portfolio_value) for s in symbols]
        )

        # Portfolio variance (Issue #95: einsum ~15% faster for small matrices)
        port_variance = np.einsum('i,ij,j->', weights, cov_matrix, weights)
        port_volatility = np.sqrt(port_variance)

        # Z-score for confidence level
        z_score = abs(stats.norm.ppf(1 - conf))

        # Marginal VaR = dVaR/dw = z * Cov(r_i, r_p) / sigma_p
        # einsum for matrix-vector: 'ij,j->i'
        marginal_var = z_score * np.einsum('ij,j->i', cov_matrix, weights) / port_volatility

        # Component VaR = w_i * Marginal_VaR_i * Portfolio_Value
        component_vars = {}
        for i, symbol in enumerate(symbols):
            comp_var = weights[i] * marginal_var[i] * float(portfolio_value)
            component_vars[symbol] = Decimal(str(abs(comp_var)))

        return component_vars

    def calculate_incremental_var(
        self,
        current_returns: dict[str, np.ndarray],
        current_values: dict[str, Decimal],
        new_position_returns: np.ndarray,
        new_position_value: Decimal,
        confidence_level: float | None = None,
    ) -> Decimal:
        """
        Calculate Incremental VaR for adding a new position.

        Measures the change in portfolio VaR from adding
        a new position.

        Args:
            current_returns: Current portfolio position returns
            current_values: Current position values
            new_position_returns: Returns for new position
            new_position_value: Value of new position
            confidence_level: Override config confidence level

        Returns:
            Incremental VaR (positive = increases risk)
        """
        conf = confidence_level or self.config.confidence_level

        # Calculate current portfolio VaR
        if current_returns:
            current_var, _ = self.calculate_portfolio_var(
                current_returns, current_values, VaRMethod.PARAMETRIC, conf
            )
        else:
            current_var = VaRResult(
                var=Decimal("0"),
                cvar=Decimal("0"),
                confidence_level=conf,
                time_horizon_days=self.config.time_horizon_days,
                method=VaRMethod.PARAMETRIC,
                portfolio_value=Decimal("0"),
                var_pct=0.0,
                cvar_pct=0.0,
            )

        # Add new position and recalculate
        new_returns = dict(current_returns)
        new_returns["_new_position"] = new_position_returns
        new_values = dict(current_values)
        new_values["_new_position"] = new_position_value

        new_var, _ = self.calculate_portfolio_var(
            new_returns, new_values, VaRMethod.PARAMETRIC, conf
        )

        return new_var.var - current_var.var

    def _calculate_ewma_volatility(self, returns: np.ndarray) -> float:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) volatility.

        Uses RiskMetrics methodology with decay factor lambda.
        Backends (in order of preference when enabled):
        - Rust (Issue #97): ~100x speedup, no GIL, SIMD auto-vectorization
        - Numba (Issue #70): ~50x speedup, JIT-compiled

        Args:
            returns: Historical returns

        Returns:
            EWMA volatility estimate
        """
        # Use Rust backend if available and enabled (Issue #97)
        if self.config.use_rust_ewma and RUST_EWMA_AVAILABLE:
            return _ewma_volatility_rust(returns, lambda_=self.config.ewma_lambda)

        # Fallback to Numba backend
        return _ewma_volatility_numba(returns, self.config.ewma_lambda)

    def backtest_var(
        self,
        returns: np.ndarray,
        portfolio_values: np.ndarray,
        method: VaRMethod = VaRMethod.HISTORICAL,
        confidence_level: float | None = None,
        window_size: int | None = None,
    ) -> dict:
        """
        Backtest VaR model using historical data.

        Calculates VaR exceptions (breaches) and performs
        statistical tests for model adequacy.

        For HISTORICAL method, uses Numba JIT-compiled loop for
        10-50x speedup (Issue #74).

        Args:
            returns: Full historical returns
            portfolio_values: Portfolio values over time
            method: VaR calculation method to test
            confidence_level: Override config confidence level
            window_size: Rolling window size for VaR calculation

        Returns:
            Backtest results including exception rate and tests
        """
        conf = confidence_level or self.config.confidence_level
        window = window_size or self.config.lookback_days

        if len(returns) < window + 1:
            raise ValueError(f"Need at least {window + 1} observations")

        # Use Numba JIT-compiled loop for historical VaR (Issue #74: ~10-50x faster)
        if method == VaRMethod.HISTORICAL:
            # Ensure arrays are float64 for Numba
            returns_f64 = np.asarray(returns, dtype=np.float64)
            portfolio_f64 = np.asarray(portfolio_values, dtype=np.float64)

            var_estimates, actual_losses, exceptions = _var_backtest_loop_historical(
                returns_f64, portfolio_f64, window, conf
            )
        else:
            # Fallback to Python loop for parametric/monte carlo methods
            # Pre-allocate arrays to reduce GC pressure (Issue #82)
            n_obs = len(returns) - window
            var_estimates = np.empty(n_obs, dtype=np.float64)
            actual_losses = np.empty(n_obs, dtype=np.float64)
            exceptions = 0

            for i in range(window, len(returns)):
                idx = i - window
                window_returns = returns[i - window : i]
                port_value = Decimal(str(portfolio_values[i - 1]))

                if method == VaRMethod.PARAMETRIC:
                    var_result = self.calculate_parametric_var(
                        window_returns, port_value, conf, time_horizon_days=1
                    )
                else:
                    var_result = self.calculate_monte_carlo_var(
                        window_returns, port_value, conf, time_horizon_days=1
                    )

                var_estimate = float(var_result.var)
                var_estimates[idx] = var_estimate
                actual_loss = -returns[i] * float(port_value)
                actual_losses[idx] = actual_loss

                if actual_loss > var_estimate:
                    exceptions += 1

        n_observations = len(returns) - window
        exception_rate = exceptions / n_observations
        expected_rate = 1 - conf

        # Kupiec POF test (Proportion of Failures)
        # Tests if exception rate is consistent with confidence level
        if exceptions > 0 and exceptions < n_observations:
            lr_pof = -2 * (
                exceptions * np.log(expected_rate / exception_rate)
                + (n_observations - exceptions)
                * np.log((1 - expected_rate) / (1 - exception_rate))
            )
            pof_pvalue = 1 - stats.chi2.cdf(lr_pof, 1)
        else:
            lr_pof = 0.0
            pof_pvalue = 1.0

        return {
            "method": method.value,
            "confidence_level": conf,
            "n_observations": n_observations,
            "exceptions": exceptions,
            "exception_rate": exception_rate,
            "expected_rate": expected_rate,
            "kupiec_lr_statistic": lr_pof,
            "kupiec_pvalue": pof_pvalue,
            "model_adequate": pof_pvalue > 0.05,  # 5% significance
            "var_estimates": var_estimates,  # Already np.ndarray
            "actual_losses": actual_losses,  # Already np.ndarray
        }


def create_var_calculator(
    confidence_level: float = 0.95,
    time_horizon_days: int = 1,
    lookback_days: int = 252,
    use_ewma: bool = True,
) -> VaRCalculator:
    """
    Factory function to create a VaR calculator.

    Args:
        confidence_level: VaR confidence level (e.g., 0.95, 0.99)
        time_horizon_days: Time horizon for VaR (1 or 10 days)
        lookback_days: Historical lookback window
        use_ewma: Use EWMA for volatility estimation

    Returns:
        Configured VaRCalculator instance
    """
    config = VaRConfig(
        confidence_level=confidence_level,
        time_horizon_days=time_horizon_days,
        lookback_days=lookback_days,
        use_ewma=use_ewma,
    )
    return VaRCalculator(config)
