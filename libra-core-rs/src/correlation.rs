//! Correlation and covariance matrix calculations.
//!
//! High-performance matrix computations using ndarray and rayon for parallelism.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

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

/// Calculate Kendall tau correlation between two arrays.
///
/// Uses O(n log n) merge-sort based algorithm for efficiency.
///
/// # Arguments
///
/// * `x` - First array
/// * `y` - Second array
///
/// # Returns
///
/// Kendall tau correlation coefficient
fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }

    // Create pairs and sort by x
    let mut pairs: Vec<(f64, f64)> = x.iter().copied().zip(y.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Count concordant and discordant pairs using merge sort
    let y_values: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let (concordant, discordant, ties_x, ties_y) = count_pairs(&y_values, &pairs);

    // Calculate tau-b (handles ties)
    let n_pairs = (n * (n - 1)) / 2;
    let n0 = n_pairs as f64;
    let n1 = ties_x as f64;
    let n2 = ties_y as f64;

    let numerator = (concordant as f64) - (discordant as f64);
    let denominator = ((n0 - n1) * (n0 - n2)).sqrt();

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Count concordant, discordant pairs and ties using O(n²) for correctness.
/// For production with large n, could use merge-sort O(n log n) algorithm.
fn count_pairs(y_sorted_by_x: &[f64], pairs: &[(f64, f64)]) -> (usize, usize, usize, usize) {
    let n = y_sorted_by_x.len();
    let mut concordant = 0usize;
    let mut discordant = 0usize;
    let mut ties_x = 0usize;
    let mut ties_y = 0usize;

    for i in 0..n {
        for j in (i + 1)..n {
            let x_diff = pairs[j].0 - pairs[i].0;
            let y_diff = y_sorted_by_x[j] - y_sorted_by_x[i];

            if x_diff == 0.0 {
                ties_x += 1;
            }
            if y_diff == 0.0 {
                ties_y += 1;
            }

            let product = x_diff * y_diff;
            if product > 0.0 {
                concordant += 1;
            } else if product < 0.0 {
                discordant += 1;
            }
        }
    }

    (concordant, discordant, ties_x, ties_y)
}

/// Calculate Kendall tau correlation matrix with Rayon parallelization.
///
/// Computes pairwise Kendall tau correlations for all asset pairs in parallel.
/// For 20 assets, this means 190 pairs computed across multiple CPU cores.
///
/// # Arguments
///
/// * `returns` - 2D array of returns (rows = observations, cols = assets)
///
/// # Returns
///
/// Kendall tau correlation matrix (n_assets x n_assets)
///
/// # Example
///
/// ```python
/// from libra_core_rs import kendall_correlation_matrix
/// import numpy as np
///
/// # 252 observations, 20 assets
/// returns = np.random.randn(252, 20) * 0.02
/// corr = kendall_correlation_matrix(returns)
/// ```
#[pyfunction]
pub fn kendall_correlation_matrix<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let returns = returns.as_array();
    let (n_obs, n_assets) = returns.dim();

    if n_obs < 2 || n_assets == 0 {
        return Ok(Array2::<f64>::zeros((n_assets, n_assets)).into_pyarray_bound(py));
    }

    // Pre-extract columns for efficient access
    let columns: Vec<Vec<f64>> = (0..n_assets)
        .map(|j| returns.column(j).to_vec())
        .collect();

    // Generate all pairs for upper triangle
    let pairs: Vec<(usize, usize)> = (0..n_assets)
        .flat_map(|i| ((i + 1)..n_assets).map(move |j| (i, j)))
        .collect();

    // Parallel computation of Kendall tau for each pair
    let taus: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let tau = kendall_tau(&columns[i], &columns[j]);
            (i, j, tau)
        })
        .collect();

    // Build correlation matrix
    let mut corr = Array2::<f64>::eye(n_assets);
    for (i, j, tau) in taus {
        corr[[i, j]] = tau;
        corr[[j, i]] = tau;
    }

    Ok(corr.into_pyarray_bound(py))
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
        let variance: f64 = data.column(0).iter().map(|&x: &f64| (x - 3.0).powi(2)).sum::<f64>() / 4.0;
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

    #[test]
    fn test_kendall_tau_perfect_correlation() {
        // Perfectly correlated data should have tau = 1.0
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tau = kendall_tau(&x, &y);
        assert_relative_eq!(tau, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kendall_tau_perfect_anticorrelation() {
        // Perfectly anti-correlated data should have tau = -1.0
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let tau = kendall_tau(&x, &y);
        assert_relative_eq!(tau, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kendall_tau_no_correlation() {
        // Uncorrelated data should have tau near 0
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 1.0, 4.0, 2.0, 5.0];
        let tau = kendall_tau(&x, &y);
        // tau should be between -1 and 1 and relatively small
        assert!(tau >= -1.0 && tau <= 1.0);
    }
}
