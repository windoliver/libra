//! Benchmarks for EWMA calculations.
//!
//! Compares Rust EWMA implementations for performance testing.
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// EWMA volatility calculation returning full series (pure Rust, no PyO3).
fn ewma_volatility_series(returns: &[f64], span: usize, annualize: bool) -> Vec<f64> {
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

/// EWMA volatility scalar (matches VaR methodology).
fn ewma_volatility_scalar(returns: &[f64], lambda: f64) -> f64 {
    let n = returns.len();
    if n == 0 {
        return 0.0;
    }

    // Calculate mean
    let mean: f64 = returns.iter().sum::<f64>() / n as f64;

    // Calculate initial variance (ddof=1)
    let variance_init: f64 = if n > 1 {
        returns.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };

    // Apply EWMA recursion
    let mut variance = variance_init;
    for i in 1..n {
        variance = lambda * variance + (1.0 - lambda) * returns[i].powi(2);
    }

    variance.sqrt()
}

fn benchmark_ewma_series(c: &mut Criterion) {
    let mut group = c.benchmark_group("ewma_series");

    for size in [100, 1000, 10000].iter() {
        let returns: Vec<f64> = (0..*size)
            .map(|i| (i as f64 * 0.01).sin() * 0.02)
            .collect();

        group.bench_with_input(BenchmarkId::new("rust", size), &returns, |b, data| {
            b.iter(|| ewma_volatility_series(black_box(data), 20, true))
        });
    }

    group.finish();
}

fn benchmark_ewma_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("ewma_scalar");

    for size in [100, 252, 1000, 10000].iter() {
        let returns: Vec<f64> = (0..*size)
            .map(|i| (i as f64 * 0.01).sin() * 0.02)
            .collect();

        group.bench_with_input(BenchmarkId::new("rust", size), &returns, |b, data| {
            b.iter(|| ewma_volatility_scalar(black_box(data), 0.94))
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_ewma_series, benchmark_ewma_scalar);
criterion_main!(benches);
