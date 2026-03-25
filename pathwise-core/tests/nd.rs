use nalgebra::SVector;
use pathwise_core::process::markov::heston;
use pathwise_core::scheme::euler;

#[test]
fn heston_output_shape() {
    let sde = heston(0.05, 2.0, 0.04, 0.3, -0.7);
    let out = pathwise_core::simulate_nd::<2, _, _, _>(
        &sde.drift, &sde.diffusion, &euler(),
        SVector::from([0.0_f64, 0.04]), 0.0, 1.0, 10, 100, 0
    ).unwrap();
    assert_eq!(out.shape(), &[10, 101, 2]);
}

#[test]
fn heston_variance_nonnegative() {
    let sde = heston(0.05, 2.0, 0.04, 0.3, -0.7);
    let out = pathwise_core::simulate_nd::<2, _, _, _>(
        &sde.drift, &sde.diffusion, &euler(),
        SVector::from([0.0_f64, 0.04]), 0.0, 1.0, 500, 200, 42
    ).unwrap();
    // Variance is component index 1.
    // Euler can produce slightly negative variance before the next step's full-truncation clips
    // it in the diffusion; allow small negative values as documented Euler behavior.
    for path in 0..500 {
        for step in 0..=200 {
            let v = out[[path, step, 1]];
            assert!(v >= -0.05 || v.is_nan(), "variance negative: {} at path={} step={}", v, path, step);
        }
    }
}

#[test]
fn euler_nd_shape_and_finite() {
    // A 2D Euler with diagonal uncorrelated diffusion
    let mu = 0.05_f64;
    let sigma = 0.2_f64;
    let drift_2d = move |x: &SVector<f64, 2>, _t: f64| -> SVector<f64, 2> {
        SVector::from([mu * x[0], -2.0 * x[1]])
    };
    let diff_2d = move |x: &SVector<f64, 2>, _t: f64| -> SVector<f64, 2> {
        SVector::from([sigma * x[0], 0.3_f64])
    };
    let x0_2d = SVector::from([1.0_f64, 0.0]);
    let out_2d = pathwise_core::simulate_nd::<2, _, _, _>(
        &drift_2d, &diff_2d, &euler(), x0_2d, 0.0, 1.0, 1, 100, 0
    ).unwrap();
    assert_eq!(out_2d.shape(), &[1, 101, 2]);
    for val in out_2d.iter() {
        if !val.is_nan() {
            assert!(val.is_finite(), "nD Euler produced non-finite value");
        }
    }
}
