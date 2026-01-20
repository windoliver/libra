//! LIBRA Core Rust Library
//!
//! High-performance computational kernels for the LIBRA trading platform.
//! Provides 10-100x speedup for hot paths via Rust + PyO3.
//!
//! # Modules
//!
//! - `ewma`: Exponentially Weighted Moving Average calculations
//! - `correlation`: Correlation matrix computations
//! - `var`: Value at Risk calculations
//! - `message_bus`: High-performance message passing
//!
//! # Example
//!
//! ```python
//! from libra_core_rs import ewma_volatility, correlation_matrix
//!
//! # Calculate EWMA volatility
//! vol = ewma_volatility(returns, span=20)
//!
//! # Calculate correlation matrix
//! corr = correlation_matrix(returns_matrix)
//! ```
//!
//! See: https://github.com/windoliver/libra/issues/96

use pyo3::prelude::*;

pub mod correlation;
pub mod ewma;
pub mod message_bus;
pub mod var;

/// LIBRA Core Rust module for Python.
///
/// Exposes high-performance computational functions to Python via PyO3.
#[pymodule]
fn libra_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // EWMA functions
    m.add_function(wrap_pyfunction!(ewma::ewma_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(ewma::ewma_mean, m)?)?;
    m.add_function(wrap_pyfunction!(ewma::ewma_volatility_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(ewma::ewma_covariance, m)?)?;

    // Correlation functions
    m.add_function(wrap_pyfunction!(correlation::correlation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(correlation::covariance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(correlation::kendall_correlation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(correlation::risk_parity_weights, m)?)?;

    // VaR functions
    m.add_function(wrap_pyfunction!(var::parametric_var, m)?)?;
    m.add_function(wrap_pyfunction!(var::historical_var, m)?)?;

    // Message bus class
    m.add_class::<message_bus::MessageBus>()?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
