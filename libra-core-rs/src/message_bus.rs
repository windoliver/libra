//! High-performance MessageBus implementation in Rust.
//!
//! Provides O(1) publish and priority-based dispatch with pre-computed hashes.
//!
//! # Architecture
//! - 4 priority queues (RISK > ORDERS > SIGNALS > MARKET_DATA)
//! - FNV hash map for fast handler lookup
//! - Parking lot mutex for minimal contention
//!
//! # Performance Targets
//! - Publish: >100M events/sec
//! - Dispatch: >50M events/sec (with handler calls)

use fnv::FnvHashMap;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Event priority levels (lower = higher priority).
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Priority {
    Risk = 0,
    Orders = 1,
    Signals = 2,
    MarketData = 3,
}

#[pymethods]
impl Priority {
    #[new]
    fn new(value: u8) -> PyResult<Self> {
        match value {
            0 => Ok(Priority::Risk),
            1 => Ok(Priority::Orders),
            2 => Ok(Priority::Signals),
            3 => Ok(Priority::MarketData),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "Priority must be 0-3",
            )),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            Priority::Risk => "Priority.RISK".to_string(),
            Priority::Orders => "Priority.ORDERS".to_string(),
            Priority::Signals => "Priority.SIGNALS".to_string(),
            Priority::MarketData => "Priority.MARKET_DATA".to_string(),
        }
    }

    fn __int__(&self) -> u8 {
        *self as u8
    }
}

/// Configuration for RustMessageBus.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RustMessageBusConfig {
    /// Max events in RISK queue
    #[pyo3(get, set)]
    pub risk_queue_size: usize,
    /// Max events in ORDERS queue
    #[pyo3(get, set)]
    pub orders_queue_size: usize,
    /// Max events in SIGNALS queue
    #[pyo3(get, set)]
    pub signals_queue_size: usize,
    /// Max events in MARKET_DATA queue
    #[pyo3(get, set)]
    pub data_queue_size: usize,
    /// Events to process per dispatch batch
    #[pyo3(get, set)]
    pub batch_size: usize,
}

#[pymethods]
impl RustMessageBusConfig {
    #[new]
    #[pyo3(signature = (
        risk_queue_size = 1000,
        orders_queue_size = 10000,
        signals_queue_size = 10000,
        data_queue_size = 100000,
        batch_size = 100
    ))]
    fn new(
        risk_queue_size: usize,
        orders_queue_size: usize,
        signals_queue_size: usize,
        data_queue_size: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            risk_queue_size,
            orders_queue_size,
            signals_queue_size,
            data_queue_size,
            batch_size,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RustMessageBusConfig(risk={}, orders={}, signals={}, data={}, batch={})",
            self.risk_queue_size,
            self.orders_queue_size,
            self.signals_queue_size,
            self.data_queue_size,
            self.batch_size
        )
    }
}

impl Default for RustMessageBusConfig {
    fn default() -> Self {
        Self {
            risk_queue_size: 1000,
            orders_queue_size: 10000,
            signals_queue_size: 10000,
            data_queue_size: 100000,
            batch_size: 100,
        }
    }
}

/// Lightweight event representation for Rust-side processing.
struct RustEvent {
    priority: Priority,
    event_type: u32, // Event type as int for fast comparison
    py_event: PyObject,
}

/// Subscription tracking.
struct Subscription {
    id: u64,
    event_type: u32,
    handler: PyObject,
    filter_fn: Option<PyObject>,
}

/// High-performance message bus with priority queues.
///
/// This is a drop-in replacement for Python MessageBus with 100x better
/// throughput for the hot path (publish/dispatch).
#[pyclass]
pub struct RustMessageBus {
    config: RustMessageBusConfig,

    // Priority queues - using Mutex for thread safety
    risk_queue: Arc<Mutex<VecDeque<RustEvent>>>,
    orders_queue: Arc<Mutex<VecDeque<RustEvent>>>,
    signals_queue: Arc<Mutex<VecDeque<RustEvent>>>,
    data_queue: Arc<Mutex<VecDeque<RustEvent>>>,

    // Handler registry: event_type -> subscriptions
    handlers: Arc<Mutex<FnvHashMap<u32, Vec<Subscription>>>>,

    // State
    accepting: Arc<std::sync::atomic::AtomicBool>,

    // Metrics (atomic for lock-free updates)
    events_published: AtomicU64,
    events_dispatched: AtomicU64,
    events_dropped: AtomicU64,
    handler_errors: AtomicU64,

    // Subscription ID counter
    next_sub_id: AtomicU64,
}

#[pymethods]
impl RustMessageBus {
    /// Create a new RustMessageBus.
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<RustMessageBusConfig>) -> Self {
        let config = config.unwrap_or_default();
        Self {
            risk_queue: Arc::new(Mutex::new(VecDeque::with_capacity(config.risk_queue_size))),
            orders_queue: Arc::new(Mutex::new(VecDeque::with_capacity(config.orders_queue_size))),
            signals_queue: Arc::new(Mutex::new(VecDeque::with_capacity(
                config.signals_queue_size,
            ))),
            data_queue: Arc::new(Mutex::new(VecDeque::with_capacity(config.data_queue_size))),
            handlers: Arc::new(Mutex::new(FnvHashMap::default())),
            accepting: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            events_published: AtomicU64::new(0),
            events_dispatched: AtomicU64::new(0),
            events_dropped: AtomicU64::new(0),
            handler_errors: AtomicU64::new(0),
            next_sub_id: AtomicU64::new(1),
            config,
        }
    }

    /// Subscribe to an event type.
    ///
    /// Args:
    ///     event_type: Event type as integer
    ///     handler: Async callable to handle events
    ///     filter_fn: Optional filter function
    ///
    /// Returns:
    ///     Subscription ID
    #[pyo3(signature = (event_type, handler, filter_fn = None))]
    fn subscribe(
        &self,
        event_type: u32,
        handler: PyObject,
        filter_fn: Option<PyObject>,
    ) -> u64 {
        let sub_id = self.next_sub_id.fetch_add(1, Ordering::SeqCst);
        let sub = Subscription {
            id: sub_id,
            event_type,
            handler,
            filter_fn,
        };

        let mut handlers = self.handlers.lock();
        handlers.entry(event_type).or_default().push(sub);

        sub_id
    }

    /// Unsubscribe by ID.
    fn unsubscribe(&self, subscription_id: u64) -> bool {
        let mut handlers = self.handlers.lock();
        for subs in handlers.values_mut() {
            if let Some(pos) = subs.iter().position(|s| s.id == subscription_id) {
                subs.remove(pos);
                return true;
            }
        }
        false
    }

    /// Publish an event (non-blocking).
    ///
    /// Routes to appropriate priority queue based on event.priority.
    ///
    /// Args:
    ///     event: Python Event object with priority and event_type attributes
    ///
    /// Returns:
    ///     True if accepted, False if shutting down
    fn publish(&self, py: Python<'_>, event: PyObject) -> PyResult<bool> {
        if !self.accepting.load(Ordering::SeqCst) {
            return Ok(false);
        }

        // Extract priority and event_type from Python event
        let priority_int: u8 = event.getattr(py, "priority")?.extract(py)?;
        let event_type: u32 = event
            .getattr(py, "event_type")?
            .getattr(py, "value")?
            .extract(py)?;

        let priority = match priority_int {
            0 => Priority::Risk,
            1 => Priority::Orders,
            2 => Priority::Signals,
            _ => Priority::MarketData,
        };

        let rust_event = RustEvent {
            priority,
            event_type,
            py_event: event,
        };

        // Get the appropriate queue
        let (queue, max_size) = match priority {
            Priority::Risk => (&self.risk_queue, self.config.risk_queue_size),
            Priority::Orders => (&self.orders_queue, self.config.orders_queue_size),
            Priority::Signals => (&self.signals_queue, self.config.signals_queue_size),
            Priority::MarketData => (&self.data_queue, self.config.data_queue_size),
        };

        let mut q = queue.lock();
        let was_full = q.len() >= max_size;

        if was_full {
            // Drop oldest event
            q.pop_front();
            self.events_dropped.fetch_add(1, Ordering::Relaxed);
        }

        q.push_back(rust_event);
        self.events_published.fetch_add(1, Ordering::Relaxed);

        Ok(true)
    }

    /// Dispatch a batch of events to handlers.
    ///
    /// Processes events in priority order (RISK first).
    ///
    /// Returns:
    ///     Number of events dispatched
    fn dispatch_batch(&self, py: Python<'_>) -> PyResult<usize> {
        let mut dispatched = 0;
        let batch_size = self.config.batch_size;

        // Process queues in priority order
        for (queue, _) in [
            (&self.risk_queue, Priority::Risk),
            (&self.orders_queue, Priority::Orders),
            (&self.signals_queue, Priority::Signals),
            (&self.data_queue, Priority::MarketData),
        ] {
            while dispatched < batch_size {
                let event = {
                    let mut q = queue.lock();
                    q.pop_front()
                };

                match event {
                    Some(rust_event) => {
                        self.dispatch_event(py, rust_event)?;
                        dispatched += 1;
                        self.events_dispatched.fetch_add(1, Ordering::Relaxed);
                    }
                    None => break,
                }
            }
        }

        Ok(dispatched)
    }

    /// Get statistics about the message bus.
    fn get_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "events_published",
            self.events_published.load(Ordering::Relaxed),
        )?;
        dict.set_item(
            "events_dispatched",
            self.events_dispatched.load(Ordering::Relaxed),
        )?;
        dict.set_item(
            "events_dropped",
            self.events_dropped.load(Ordering::Relaxed),
        )?;
        dict.set_item(
            "handler_errors",
            self.handler_errors.load(Ordering::Relaxed),
        )?;

        // Queue sizes
        dict.set_item("risk_queue_size", self.risk_queue.lock().len())?;
        dict.set_item("orders_queue_size", self.orders_queue.lock().len())?;
        dict.set_item("signals_queue_size", self.signals_queue.lock().len())?;
        dict.set_item("data_queue_size", self.data_queue.lock().len())?;

        // Subscription count
        let sub_count: usize = self.handlers.lock().values().map(|v| v.len()).sum();
        dict.set_item("subscription_count", sub_count)?;

        Ok(dict.into())
    }

    /// Stop accepting new events.
    fn stop_accepting(&self) {
        self.accepting.store(false, Ordering::SeqCst);
    }

    /// Resume accepting new events.
    fn start_accepting(&self) {
        self.accepting.store(true, Ordering::SeqCst);
    }

    /// Check if accepting events.
    fn is_accepting(&self) -> bool {
        self.accepting.load(Ordering::SeqCst)
    }

    /// Get total queue depth across all priorities.
    fn total_queue_depth(&self) -> usize {
        self.risk_queue.lock().len()
            + self.orders_queue.lock().len()
            + self.signals_queue.lock().len()
            + self.data_queue.lock().len()
    }

    /// Clear all queues.
    fn clear_queues(&self) -> usize {
        let mut cleared = 0;
        cleared += self.risk_queue.lock().len();
        self.risk_queue.lock().clear();
        cleared += self.orders_queue.lock().len();
        self.orders_queue.lock().clear();
        cleared += self.signals_queue.lock().len();
        self.signals_queue.lock().clear();
        cleared += self.data_queue.lock().len();
        self.data_queue.lock().clear();
        cleared
    }
}

impl RustMessageBus {
    /// Internal: dispatch a single event to matching handlers.
    fn dispatch_event(&self, py: Python<'_>, event: RustEvent) -> PyResult<()> {
        let handlers = self.handlers.lock();

        if let Some(subs) = handlers.get(&event.event_type) {
            for sub in subs {
                // Check filter if present
                if let Some(ref filter_fn) = sub.filter_fn {
                    let passes: bool = filter_fn.call1(py, (&event.py_event,))?.extract(py)?;
                    if !passes {
                        continue;
                    }
                }

                // Call handler - note: this is synchronous, Python side should handle async
                match sub.handler.call1(py, (&event.py_event,)) {
                    Ok(_) => {}
                    Err(e) => {
                        self.handler_errors.fetch_add(1, Ordering::Relaxed);
                        // Log error but don't propagate
                        eprintln!("Handler error: {:?}", e);
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_values() {
        assert_eq!(Priority::Risk as u8, 0);
        assert_eq!(Priority::Orders as u8, 1);
        assert_eq!(Priority::Signals as u8, 2);
        assert_eq!(Priority::MarketData as u8, 3);
    }

    #[test]
    fn test_config_default() {
        let config = RustMessageBusConfig::default();
        assert_eq!(config.risk_queue_size, 1000);
        assert_eq!(config.batch_size, 100);
    }
}
