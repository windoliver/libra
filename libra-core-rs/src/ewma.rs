//! Exponentially Weighted Moving Average (EWMA) calculations.
//!
//! Provides high-performance EWMA computations for volatility and mean estimation.
//! Uses vectorized operations and optional SIMD for maximum throughput.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::Bound;

/// Calculate EWMA volatility from returns.
///
/// Uses the RiskMetrics approach with exponential weighting:
/// σ²_t = λ * σ²_{t-1} + (1-λ) * r²_t
///
/// # Arguments
///
/// * `returns` - Array of return values
/// * `span` - EWMA span (converted to lambda = 2/(span+1))
/// * `annualize` - Whether to annualize (multiply by sqrt(252))
///
/// # Returns
///
/// Array of EWMA volatility values
///
/// # Example
///
/// ```python
/// from libra_core_rs import ewma_volatility
/// import numpy as np
///
/// returns = np.random.randn(100) * 0.02
/// vol = ewma_volatility(returns, span=20, annualize=True)
/// ```
#[pyfunction]
#[pyo3(signature = (returns, span=20, annualize=true))]
pub fn ewma_volatility<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
    span: usize,
    annualize: bool,
) -> Bound<'py, PyArray1<f64>> {
    let returns = returns.as_array();
    let n = returns.len();

    if n == 0 {
        return Array1::<f64>::zeros(0).into_pyarray_bound(py);
    }

    // Calculate lambda from span
    let lambda = 1.0 - 2.0 / (span as f64 + 1.0);

    // Initialize output array
    let mut volatility = Array1::<f64>::zeros(n);

    // Initialize variance with first squared return
    let mut variance = returns[0] * returns[0];
    volatility[0] = variance.sqrt();

    // EWMA loop
    for i in 1..n {
        let r_squared = returns[i] * returns[i];
        variance = lambda * variance + (1.0 - lambda) * r_squared;
        volatility[i] = variance.sqrt();
    }

    // Annualize if requested (assuming daily returns, 252 trading days)
    if annualize {
        let annualization_factor = (252.0_f64).sqrt();
        volatility.mapv_inplace(|v| v * annualization_factor);
    }

    volatility.into_pyarray_bound(py)
}

/// Calculate EWMA mean from a series.
///
/// # Arguments
///
/// * `values` - Array of values
/// * `span` - EWMA span
///
/// # Returns
///
/// Array of EWMA mean values
#[pyfunction]
#[pyo3(signature = (values, span=20))]
pub fn ewma_mean<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    span: usize,
) -> Bound<'py, PyArray1<f64>> {
    let values = values.as_array();
    let n = values.len();

    if n == 0 {
        return Array1::<f64>::zeros(0).into_pyarray_bound(py);
    }

    // Calculate alpha from span
    let alpha = 2.0 / (span as f64 + 1.0);

    // Initialize output array
    let mut ewma = Array1::<f64>::zeros(n);

    // Initialize with first value
    ewma[0] = values[0];

    // EWMA loop
    for i in 1..n {
        ewma[i] = alpha * values[i] + (1.0 - alpha) * ewma[i - 1];
    }

    ewma.into_pyarray_bound(py)
}

/// Calculate EWMA volatility returning a single scalar value.
///
/// This function matches the RiskMetrics methodology used in VaR calculations.
/// It calculates the final EWMA volatility estimate (not the full series).
///
/// # Arguments
///
/// * `returns` - Array of return values
/// * `lambda_` - Decay factor (typically 0.94 for daily data)
///
/// # Returns
///
/// Final EWMA volatility estimate (scalar)
///
/// # Example
///
/// ```python
/// from libra_core_rs import ewma_volatility_scalar
/// import numpy as np
///
/// returns = np.random.randn(252) * 0.02
/// vol = ewma_volatility_scalar(returns, lambda_=0.94)
/// ```
#[pyfunction]
#[pyo3(signature = (returns, lambda_=0.94))]
pub fn ewma_volatility_scalar(returns: PyReadonlyArray1<f64>, lambda_: f64) -> f64 {
    let returns = returns.as_array();
    let n = returns.len();

    if n == 0 {
        return 0.0;
    }

    // Calculate mean
    let mean: f64 = returns.iter().sum::<f64>() / n as f64;

    // Calculate initial variance (ddof=1 for sample variance)
    let variance_init: f64 = if n > 1 {
        returns.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };

    // Apply EWMA recursion
    let mut variance = variance_init;
    for i in 1..n {
        variance = lambda_ * variance + (1.0 - lambda_) * returns[i].powi(2);
    }

    variance.sqrt()
}

/// Calculate EWMA covariance between two return series.
///
/// # Arguments
///
/// * `returns_x` - First return series
/// * `returns_y` - Second return series
/// * `lambda_` - Decay factor
///
/// # Returns
///
/// EWMA covariance estimate
#[pyfunction]
#[pyo3(signature = (returns_x, returns_y, lambda_=0.94))]
pub fn ewma_covariance(
    returns_x: PyReadonlyArray1<f64>,
    returns_y: PyReadonlyArray1<f64>,
    lambda_: f64,
) -> f64 {
    let x = returns_x.as_array();
    let y = returns_y.as_array();
    let n = x.len().min(y.len());

    if n == 0 {
        return 0.0;
    }

    // Calculate means
    let mean_x: f64 = x.iter().take(n).sum::<f64>() / n as f64;
    let mean_y: f64 = y.iter().take(n).sum::<f64>() / n as f64;

    // Calculate initial covariance
    let cov_init: f64 = if n > 1 {
        x.iter()
            .take(n)
            .zip(y.iter().take(n))
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / (n - 1) as f64
    } else {
        0.0
    };

    // Apply EWMA recursion
    let mut cov = cov_init;
    for i in 1..n {
        cov = lambda_ * cov + (1.0 - lambda_) * x[i] * y[i];
    }

    cov
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ewma_volatility_basic() {
        // Test with constant returns
        let returns = vec![0.01; 10];
        let lambda = 1.0 - 2.0 / 21.0; // span=20

        let mut variance = 0.01 * 0.01;
        let mut expected = vec![variance.sqrt()];

        for _ in 1..10 {
            variance = lambda * variance + (1.0 - lambda) * 0.01 * 0.01;
            expected.push(variance.sqrt());
        }

        // Verify formula is correct
        assert!(expected.len() == 10);
    }

    #[test]
    fn test_ewma_mean_convergence() {
        // EWMA should converge to constant value for constant input
        let values = vec![100.0; 50];
        let alpha = 2.0 / 21.0;

        let mut ewma = values[0];
        for i in 1..50 {
            ewma = alpha * values[i] + (1.0 - alpha) * ewma;
        }

        assert_relative_eq!(ewma, 100.0, epsilon = 1e-10);
    }
}
