//! Benchmarks for EWMA calculations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;

/// EWMA volatility calculation (pure Rust, no PyO3).
fn ewma_volatility_rust(returns: &[f64], span: usize, annualize: bool) -> Vec<f64> {
    let n = returns.len();
    if n == 0 {
        return vec![];
    }

    let lambda = 1.0 - 2.0 / (span as f64 + 1.0);
    let mut volatility = vec![0.0; n];

    let mut variance = returns[0] * returns[0];
    volatility[0] = variance.sqrt();

    for i in 1..n {
        let r_squared = returns[i] * returns[i];
        variance = lambda * variance + (1.0 - lambda) * r_squared;
        volatility[i] = variance.sqrt();
    }

    if annualize {
        let factor = (252.0_f64).sqrt();
        for v in &mut volatility {
            *v *= factor;
        }
    }

    volatility
}

fn benchmark_ewma(c: &mut Criterion) {
    // Generate test data
    let returns: Vec<f64> = (0..1000)
        .map(|i| (i as f64 * 0.01).sin() * 0.02)
        .collect();

    c.bench_function("ewma_volatility_1000", |b| {
        b.iter(|| ewma_volatility_rust(black_box(&returns), 20, true))
    });

    let returns_large: Vec<f64> = (0..10000)
        .map(|i| (i as f64 * 0.01).sin() * 0.02)
        .collect();

    c.bench_function("ewma_volatility_10000", |b| {
        b.iter(|| ewma_volatility_rust(black_box(&returns_large), 20, true))
    });
}

criterion_group!(benches, benchmark_ewma);
criterion_main!(benches);
