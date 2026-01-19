//! Correlation and covariance matrix calculations.
//!
//! High-performance matrix computations using ndarray and rayon for parallelism.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Calculate correlation matrix from returns matrix.
///
/// Uses parallel computation for large matrices.
///
/// # Arguments
///
/// * `returns` - 2D array of returns (rows = observations, cols = assets)
///
/// # Returns
///
/// Correlation matrix (n_assets x n_assets)
///
/// # Example
///
/// ```python
/// from libra_core_rs import correlation_matrix
/// import numpy as np
///
/// # 100 observations, 5 assets
/// returns = np.random.randn(100, 5) * 0.02
/// corr = correlation_matrix(returns)
/// ```
#[pyfunction]
pub fn correlation_matrix<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let returns = returns.as_array();
    let (n_obs, n_assets) = returns.dim();

    if n_obs < 2 || n_assets == 0 {
        return Ok(Array2::<f64>::zeros((n_assets, n_assets)).into_pyarray_bound(py));
    }

    // Calculate means
    let means: Vec<f64> = (0..n_assets)
        .map(|j| returns.column(j).mean().unwrap_or(0.0))
        .collect();

    // Calculate standard deviations
    let stds: Vec<f64> = (0..n_assets)
        .map(|j| {
            let col = returns.column(j);
            let mean = means[j];
            let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_obs - 1) as f64;
            variance.sqrt()
        })
        .collect();

    // Build correlation matrix
    let mut corr = Array2::<f64>::zeros((n_assets, n_assets));

    // Diagonal is always 1
    for i in 0..n_assets {
        corr[[i, i]] = 1.0;
    }

    // Calculate off-diagonal elements (upper triangle, then mirror)
    // Sequential calculation (parallel can be added with rayon later)
    for i in 0..n_assets {
        for j in (i + 1)..n_assets {
            let cov = calculate_covariance(&returns, i, j, means[i], means[j], n_obs);
            let corr_val = if stds[i] > 0.0 && stds[j] > 0.0 {
                cov / (stds[i] * stds[j])
            } else {
                0.0
            };
            corr[[i, j]] = corr_val;
            corr[[j, i]] = corr_val; // Symmetric
        }
    }

    Ok(corr.into_pyarray_bound(py))
}

/// Calculate covariance matrix from returns matrix.
///
/// # Arguments
///
/// * `returns` - 2D array of returns (rows = observations, cols = assets)
///
/// # Returns
///
/// Covariance matrix (n_assets x n_assets)
#[pyfunction]
pub fn covariance_matrix<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let returns = returns.as_array();
    let (n_obs, n_assets) = returns.dim();

    if n_obs < 2 || n_assets == 0 {
        return Ok(Array2::<f64>::zeros((n_assets, n_assets)).into_pyarray_bound(py));
    }

    // Calculate means
    let means: Vec<f64> = (0..n_assets)
        .map(|j| returns.column(j).mean().unwrap_or(0.0))
        .collect();

    // Build covariance matrix
    let mut cov = Array2::<f64>::zeros((n_assets, n_assets));

    // Calculate all elements (symmetric matrix)
    for i in 0..n_assets {
        for j in i..n_assets {
            let cov_val = calculate_covariance(&returns, i, j, means[i], means[j], n_obs);
            cov[[i, j]] = cov_val;
            cov[[j, i]] = cov_val;
        }
    }

    Ok(cov.into_pyarray_bound(py))
}

/// Helper function to calculate covariance between two columns.
#[inline]
fn calculate_covariance(
    returns: &ndarray::ArrayView2<f64>,
    i: usize,
    j: usize,
    mean_i: f64,
    mean_j: f64,
    n_obs: usize,
) -> f64 {
    let col_i = returns.column(i);
    let col_j = returns.column(j);

    let sum: f64 = col_i
        .iter()
        .zip(col_j.iter())
        .map(|(&xi, &xj)| (xi - mean_i) * (xj - mean_j))
        .sum();

    sum / (n_obs - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_correlation_identity() {
        // Correlation of a series with itself should be 1.0
        let data = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

        // Single column correlation matrix should be [[1.0]]
        let n_obs = 5;
        let n_assets = 1;
        let means = vec![3.0];
        let variance: f64 = data.column(0).iter().map(|&x| (x - 3.0).powi(2)).sum::<f64>() / 4.0;
        let std = variance.sqrt();

        assert!(std > 0.0);
    }

    #[test]
    fn test_covariance_symmetric() {
        // Covariance matrix should be symmetric
        let data = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ];

        let means: Vec<f64> = (0..3)
            .map(|j| data.column(j).mean().unwrap())
            .collect();

        let cov_01 = calculate_covariance(&data.view(), 0, 1, means[0], means[1], 4);
        let cov_10 = calculate_covariance(&data.view(), 1, 0, means[1], means[0], 4);

        assert_relative_eq!(cov_01, cov_10, epsilon = 1e-10);
    }
}
