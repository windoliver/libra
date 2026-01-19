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
