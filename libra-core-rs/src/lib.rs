//! LIBRA Core Rust Implementation
//!
//! High-performance components for the LIBRA trading system.
//! Provides PyO3 bindings for Python integration.
//!
//! # Components
//! - `RustMessageBus`: High-performance message bus with priority queues
//!
//! # Performance
//! - Target: >100M events/sec dispatch throughput
//! - O(1) publish and dispatch operations
//! - Lock-free where possible

use pyo3::prelude::*;

mod message_bus;

pub use message_bus::{Priority, RustMessageBus, RustMessageBusConfig};

/// LIBRA Core Rust module for Python.
///
/// Usage from Python:
/// ```python
/// from libra.core._rust import RustMessageBus, RustMessageBusConfig, Priority
///
/// config = RustMessageBusConfig()
/// bus = RustMessageBus(config)
/// bus.publish(event)
/// ```
#[pymodule]
fn libra_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustMessageBusConfig>()?;
    m.add_class::<RustMessageBus>()?;
    m.add_class::<Priority>()?;
    Ok(())
}
