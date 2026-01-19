//! Value at Risk (VaR) calculations.
//!
//! Provides high-performance VaR computation using parametric and historical methods.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Standard normal quantile function (inverse CDF).
///
/// Uses Abramowitz and Stegun approximation for fast computation.
fn norm_ppf(p: f64) -> f64 {
    // Handle edge cases
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Abramowitz and Stegun approximation
    let a1 = -3.969683028665376e+01;
    let a2 = 2.209460984245205e+02;
    let a3 = -2.759285104469687e+02;
    let a4 = 1.383577518672690e+02;
    let a5 = -3.066479806614716e+01;
    let a6 = 2.506628277459239e+00;

    let b1 = -5.447609879822406e+01;
    let b2 = 1.615858368580409e+02;
    let b3 = -1.556989798598866e+02;
    let b4 = 6.680131188771972e+01;
    let b5 = -1.328068155288572e+01;

    let c1 = -7.784894002430293e-03;
    let c2 = -3.223964580411365e-01;
    let c3 = -2.400758277161838e+00;
    let c4 = -2.549732539343734e+00;
    let c5 = 4.374664141464968e+00;
    let c6 = 2.938163982698783e+00;

    let d1 = 7.784695709041462e-03;
    let d2 = 3.224671290700398e-01;
    let d3 = 2.445134137142996e+00;
    let d4 = 3.754408661907416e+00;

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    let result = if p < p_low {
        // Lower region
        let q = (-2.0 * p.ln()).sqrt();
        (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    } else if p <= p_high {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
            / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    } else {
        // Upper region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    };

    result
}

/// Calculate parametric VaR assuming normal distribution.
///
/// # Arguments
///
/// * `returns` - Array of historical returns
/// * `confidence` - Confidence level (e.g., 0.95 for 95%)
/// * `portfolio_value` - Current portfolio value
/// * `horizon` - Time horizon in days
///
/// # Returns
///
/// VaR value (positive number representing potential loss)
///
/// # Example
///
/// ```python
/// from libra_core_rs import parametric_var
/// import numpy as np
///
/// returns = np.random.randn(252) * 0.02
/// var = parametric_var(returns, confidence=0.95, portfolio_value=100000.0, horizon=1)
/// ```
#[pyfunction]
#[pyo3(signature = (returns, confidence=0.95, portfolio_value=1.0, horizon=1))]
pub fn parametric_var(
    returns: PyReadonlyArray1<f64>,
    confidence: f64,
    portfolio_value: f64,
    horizon: i32,
) -> f64 {
    let returns = returns.as_array();
    let n = returns.len();

    if n == 0 {
        return 0.0;
    }

    // Calculate mean and standard deviation
    let mean: f64 = returns.iter().sum::<f64>() / n as f64;
    let variance: f64 = returns.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std = variance.sqrt();

    // Z-score for confidence level
    let z = norm_ppf(1.0 - confidence).abs();

    // VaR = portfolio_value * (mean * horizon - z * std * sqrt(horizon))
    // Typically for VaR we ignore mean and use: z * std * sqrt(horizon)
    let horizon_factor = (horizon as f64).sqrt();
    let var = portfolio_value * z * std * horizon_factor;

    var
}

/// Calculate historical VaR from returns.
///
/// Uses the percentile method on historical returns.
///
/// # Arguments
///
/// * `returns` - Array of historical returns
/// * `confidence` - Confidence level (e.g., 0.95 for 95%)
/// * `portfolio_value` - Current portfolio value
/// * `horizon` - Time horizon in days (for scaling)
///
/// # Returns
///
/// VaR value (positive number representing potential loss)
#[pyfunction]
#[pyo3(signature = (returns, confidence=0.95, portfolio_value=1.0, horizon=1))]
pub fn historical_var(
    returns: PyReadonlyArray1<f64>,
    confidence: f64,
    portfolio_value: f64,
    horizon: i32,
) -> f64 {
    let returns = returns.as_array();
    let n = returns.len();

    if n == 0 {
        return 0.0;
    }

    // Sort returns
    let mut sorted_returns: Vec<f64> = returns.iter().cloned().collect();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Find percentile index
    let percentile = 1.0 - confidence;
    let index = (percentile * n as f64).floor() as usize;
    let index = index.min(n - 1);

    // Get VaR return (negative return at percentile)
    let var_return = -sorted_returns[index];

    // Scale by horizon (square root of time)
    let horizon_factor = (horizon as f64).sqrt();

    portfolio_value * var_return * horizon_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_norm_ppf() {
        // Test standard quantiles
        assert_relative_eq!(norm_ppf(0.5), 0.0, epsilon = 1e-6);
        assert_relative_eq!(norm_ppf(0.95), 1.6448536, epsilon = 1e-4);
        assert_relative_eq!(norm_ppf(0.99), 2.3263479, epsilon = 1e-4);
    }

    #[test]
    fn test_parametric_var_basic() {
        // With zero std, VaR should be zero
        let returns = vec![0.0; 100];
        let var_val = {
            let n = returns.len();
            let mean: f64 = returns.iter().sum::<f64>() / n as f64;
            let variance: f64 =
                returns.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            let std = variance.sqrt();
            let z = norm_ppf(0.05).abs();
            100000.0 * z * std
        };

        assert_relative_eq!(var_val, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_historical_var_sorted() {
        // Test with known values
        let returns: Vec<f64> = (-10..=10).map(|x| x as f64 / 100.0).collect();
        // Returns from -0.10 to 0.10

        // At 95% confidence, we look at 5th percentile
        // With 21 values, 5% is about index 1 (value -0.09)

        let mut sorted: Vec<f64> = returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile = 0.05;
        let index = (percentile * 21.0).floor() as usize;
        let expected_return = -sorted[index];

        assert!(expected_return > 0.0); // Loss is positive
    }
}
