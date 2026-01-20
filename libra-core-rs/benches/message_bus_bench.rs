//! Benchmarks for RustMessageBus.
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use libra_core_rs::{Priority, RustMessageBusConfig};

/// Benchmark config creation.
fn bench_config_creation(c: &mut Criterion) {
    c.bench_function("config_default", |b| {
        b.iter(|| RustMessageBusConfig::default())
    });
}

/// Benchmark priority enum operations.
fn bench_priority(c: &mut Criterion) {
    c.bench_function("priority_to_int", |b| {
        let p = Priority::MarketData;
        b.iter(|| black_box(p as u8))
    });
}

criterion_group!(benches, bench_config_creation, bench_priority);
criterion_main!(benches);
