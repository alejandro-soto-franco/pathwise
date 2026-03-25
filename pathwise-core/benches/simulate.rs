//! Criterion benchmarks for pathwise-core simulation performance.
//!
//! Run with: cargo bench -p pathwise-core
//! Results are written to target/criterion/

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::SVector;
use pathwise_core::{
    bm, gbm, heston, ou,
    scheme::{euler, milstein, sri},
    simulate, simulate_nd,
};

// ---------------------------------------------------------------------------
// Configurations: (n_paths, n_steps, label)
// ---------------------------------------------------------------------------
const CONFIGS: &[(usize, usize, &str)] = &[
    (1_000, 100, "1k×100"),
    (10_000, 252, "10k×252"),
    (100_000, 50, "100k×50"),
];

// ---------------------------------------------------------------------------
// Scalar process benchmarks (euler)
// ---------------------------------------------------------------------------

fn bench_bm_euler(c: &mut Criterion) {
    let sde = bm();
    let scheme = euler();
    let mut group = c.benchmark_group("bm/euler");
    for &(n_paths, n_steps, label) in CONFIGS {
        group.bench_with_input(
            BenchmarkId::new("paths×steps", label),
            &(n_paths, n_steps),
            |b, &(np, ns)| {
                b.iter(|| {
                    simulate(
                        black_box(&sde.drift),
                        black_box(&sde.diffusion),
                        black_box(&scheme),
                        black_box(0.0_f64),
                        black_box(0.0),
                        black_box(1.0),
                        black_box(np),
                        black_box(ns),
                        black_box(0_u64),
                    )
                    .unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_gbm_euler(c: &mut Criterion) {
    let sde = gbm(0.05, 0.2);
    let scheme = euler();
    let mut group = c.benchmark_group("gbm/euler");
    for &(n_paths, n_steps, label) in CONFIGS {
        group.bench_with_input(
            BenchmarkId::new("paths×steps", label),
            &(n_paths, n_steps),
            |b, &(np, ns)| {
                b.iter(|| {
                    simulate(
                        black_box(&sde.drift),
                        black_box(&sde.diffusion),
                        black_box(&scheme),
                        black_box(1.0_f64),
                        black_box(0.0),
                        black_box(1.0),
                        black_box(np),
                        black_box(ns),
                        black_box(0_u64),
                    )
                    .unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_ou_euler(c: &mut Criterion) {
    let sde = ou(2.0, 1.0, 0.3);
    let scheme = euler();
    let mut group = c.benchmark_group("ou/euler");
    for &(n_paths, n_steps, label) in CONFIGS {
        group.bench_with_input(
            BenchmarkId::new("paths×steps", label),
            &(n_paths, n_steps),
            |b, &(np, ns)| {
                b.iter(|| {
                    simulate(
                        black_box(&sde.drift),
                        black_box(&sde.diffusion),
                        black_box(&scheme),
                        black_box(1.0_f64),
                        black_box(0.0),
                        black_box(1.0),
                        black_box(np),
                        black_box(ns),
                        black_box(0_u64),
                    )
                    .unwrap()
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Scheme comparison: euler vs milstein vs SRI on GBM (10k x 252)
// ---------------------------------------------------------------------------

fn bench_scheme_comparison(c: &mut Criterion) {
    let sde = gbm(0.05, 0.3);
    let n_paths = 10_000;
    let n_steps = 252;
    let mut group = c.benchmark_group("gbm/scheme-comparison/10k×252");

    group.bench_function("euler", |b| {
        b.iter(|| {
            simulate(
                black_box(&sde.drift),
                black_box(&sde.diffusion),
                black_box(&euler()),
                black_box(1.0_f64),
                black_box(0.0),
                black_box(1.0),
                black_box(n_paths),
                black_box(n_steps),
                black_box(0_u64),
            )
            .unwrap()
        })
    });

    group.bench_function("milstein", |b| {
        b.iter(|| {
            simulate(
                black_box(&sde.drift),
                black_box(&sde.diffusion),
                black_box(&milstein()),
                black_box(1.0_f64),
                black_box(0.0),
                black_box(1.0),
                black_box(n_paths),
                black_box(n_steps),
                black_box(0_u64),
            )
            .unwrap()
        })
    });

    group.bench_function("sri", |b| {
        b.iter(|| {
            simulate(
                black_box(&sde.drift),
                black_box(&sde.diffusion),
                black_box(&sri()),
                black_box(1.0_f64),
                black_box(0.0),
                black_box(1.0),
                black_box(n_paths),
                black_box(n_steps),
                black_box(0_u64),
            )
            .unwrap()
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Heston (nD, simulate_nd)
// ---------------------------------------------------------------------------

fn bench_heston_euler(c: &mut Criterion) {
    let sde = heston(0.05, 2.0, 0.04, 0.3, -0.7);
    let scheme = euler();
    let x0 = SVector::<f64, 2>::from([0.0, 0.04]);
    let mut group = c.benchmark_group("heston/euler");
    // Skip 100k config: too slow for nD
    for &(n_paths, n_steps, label) in &CONFIGS[..2] {
        group.bench_with_input(
            BenchmarkId::new("paths×steps", label),
            &(n_paths, n_steps),
            |b, &(np, ns)| {
                b.iter(|| {
                    simulate_nd::<2, _, _, _>(
                        black_box(&sde.drift),
                        black_box(&sde.diffusion),
                        black_box(&scheme),
                        black_box(x0),
                        black_box(0.0),
                        black_box(1.0),
                        black_box(np),
                        black_box(ns),
                        black_box(0_u64),
                    )
                    .unwrap()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bm_euler,
    bench_gbm_euler,
    bench_ou_euler,
    bench_scheme_comparison,
    bench_heston_euler,
);
criterion_main!(benches);
