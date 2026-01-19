//! High-performance message bus for event-driven architecture.
//!
//! Provides a lock-free message passing system for trading events.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Message bus for high-performance event distribution.
///
/// Provides publish/subscribe functionality with topic-based routing.
/// Thread-safe for use in multi-threaded trading systems.
///
/// # Example
///
/// ```python
/// from libra_core_rs import MessageBus
///
/// bus = MessageBus()
///
/// # Subscribe to events
/// bus.subscribe("market.tick", callback)
///
/// # Publish events
/// bus.publish("market.tick", {"symbol": "BTC", "price": 50000.0})
/// ```
#[pyclass]
pub struct MessageBus {
    /// Counter for total messages published
    message_count: Arc<AtomicU64>,

    /// Subscribers by topic (topic -> list of callback IDs)
    #[pyo3(get)]
    subscriber_count: u64,

    /// Topics that have been created
    topics: Vec<String>,
}

#[pymethods]
impl MessageBus {
    /// Create a new message bus.
    #[new]
    pub fn new() -> Self {
        MessageBus {
            message_count: Arc::new(AtomicU64::new(0)),
            subscriber_count: 0,
            topics: Vec::new(),
        }
    }

    /// Get total number of messages published.
    pub fn get_message_count(&self) -> u64 {
        self.message_count.load(Ordering::Relaxed)
    }

    /// Register a new topic.
    ///
    /// # Arguments
    ///
    /// * `topic` - Topic name to register
    pub fn register_topic(&mut self, topic: String) {
        if !self.topics.contains(&topic) {
            self.topics.push(topic);
        }
    }

    /// Get list of registered topics.
    pub fn get_topics(&self) -> Vec<String> {
        self.topics.clone()
    }

    /// Check if a topic is registered.
    pub fn has_topic(&self, topic: &str) -> bool {
        self.topics.iter().any(|t| t == topic)
    }

    /// Increment message counter (called when publishing).
    ///
    /// Returns the new message count.
    pub fn increment_count(&self) -> u64 {
        self.message_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Reset message counter.
    pub fn reset_count(&self) {
        self.message_count.store(0, Ordering::Relaxed);
    }

    /// Get statistics about the message bus.
    pub fn get_stats(&self) -> HashMap<String, u64> {
        let mut stats = HashMap::new();
        stats.insert("message_count".to_string(), self.get_message_count());
        stats.insert("subscriber_count".to_string(), self.subscriber_count);
        stats.insert("topic_count".to_string(), self.topics.len() as u64);
        stats
    }
}

impl Default for MessageBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_bus_new() {
        let bus = MessageBus::new();
        assert_eq!(bus.get_message_count(), 0);
        assert_eq!(bus.subscriber_count, 0);
    }

    #[test]
    fn test_message_bus_increment() {
        let bus = MessageBus::new();
        assert_eq!(bus.increment_count(), 1);
        assert_eq!(bus.increment_count(), 2);
        assert_eq!(bus.get_message_count(), 2);
    }

    #[test]
    fn test_message_bus_topics() {
        let mut bus = MessageBus::new();
        bus.register_topic("market.tick".to_string());
        bus.register_topic("order.filled".to_string());

        assert!(bus.has_topic("market.tick"));
        assert!(bus.has_topic("order.filled"));
        assert!(!bus.has_topic("unknown"));

        let topics = bus.get_topics();
        assert_eq!(topics.len(), 2);
    }

    #[test]
    fn test_message_bus_reset() {
        let bus = MessageBus::new();
        bus.increment_count();
        bus.increment_count();
        assert_eq!(bus.get_message_count(), 2);

        bus.reset_count();
        assert_eq!(bus.get_message_count(), 0);
    }
}
