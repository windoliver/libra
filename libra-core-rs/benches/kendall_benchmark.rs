//! Benchmarks for Kendall tau correlation calculations.
//!
//! Compares Rust parallel implementation vs sequential for performance testing.
//! Run with: cargo bench --bench kendall_benchmark

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;

/// Kendall tau correlation between two arrays.
fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }

    let mut pairs: Vec<(f64, f64)> = x.iter().copied().zip(y.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut concordant = 0usize;
    let mut discordant = 0usize;
    let mut ties_x = 0usize;
    let mut ties_y = 0usize;

    for i in 0..n {
        for j in (i + 1)..n {
            let x_diff = pairs[j].0 - pairs[i].0;
            let y_diff = pairs[j].1 - pairs[i].1;

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

/// Sequential Kendall correlation matrix.
fn kendall_matrix_sequential(columns: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = columns.len();
    let mut corr = vec![vec![0.0; n]; n];

    for i in 0..n {
        corr[i][i] = 1.0;
        for j in (i + 1)..n {
            let tau = kendall_tau(&columns[i], &columns[j]);
            corr[i][j] = tau;
            corr[j][i] = tau;
        }
    }

    corr
}

/// Parallel Kendall correlation matrix using Rayon.
fn kendall_matrix_parallel(columns: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = columns.len();
    let mut corr = vec![vec![0.0; n]; n];

    // Generate pairs
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    // Parallel computation
    let taus: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let tau = kendall_tau(&columns[i], &columns[j]);
            (i, j, tau)
        })
        .collect();

    // Fill matrix
    for i in 0..n {
        corr[i][i] = 1.0;
    }
    for (i, j, tau) in taus {
        corr[i][j] = tau;
        corr[j][i] = tau;
    }

    corr
}

fn benchmark_kendall_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("kendall_matrix");

    for n_assets in [5, 10, 20].iter() {
        let n_obs = 252;

        // Generate test data
        let columns: Vec<Vec<f64>> = (0..*n_assets)
            .map(|col| {
                (0..n_obs)
                    .map(|row| ((row * col) as f64 * 0.01).sin() * 0.02)
                    .collect()
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("sequential", n_assets),
            &columns,
            |b, data| b.iter(|| kendall_matrix_sequential(black_box(data))),
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", n_assets),
            &columns,
            |b, data| b.iter(|| kendall_matrix_parallel(black_box(data))),
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_kendall_matrix);
criterion_main!(benches);
