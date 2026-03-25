use cartan_manifolds::{sphere::Sphere, so::SpecialOrthogonal};
use nalgebra::{SMatrix, SVector};
use pathwise_geo::{GeodesicEuler, GeodesicMilstein, ManifoldSDE, manifold_simulate, manifold_simulate_with_scheme};

fn sphere_sde() -> ManifoldSDE<
    Sphere<3>,
    impl Fn(&SVector<f64, 3>, f64) -> SVector<f64, 3> + Send + Sync,
    impl Fn(&SVector<f64, 3>, f64) -> SVector<f64, 3> + Send + Sync,
> {
    ManifoldSDE::new(
        Sphere::<3>,
        |_x: &SVector<f64, 3>, _t: f64| SVector::zeros(),
        |x: &SVector<f64, 3>, _t: f64| {
            // Tangent projection of e1 onto sphere at x: e1 - (e1 . x)*x
            let e1 = SVector::from([1.0_f64, 0.0, 0.0]);
            e1 - x * x.dot(&e1)
        },
    )
}

#[test]
fn geodesic_euler_stays_on_sphere() {
    let sde = sphere_sde();
    let x0 = SVector::from([0.0_f64, 0.0, 1.0]);
    let paths = manifold_simulate(&sde, &GeodesicEuler, x0, 0.0, 1.0, 500, 200, 42);
    for path in &paths {
        for point in path {
            let norm: f64 = point.norm();
            assert!((norm - 1.0).abs() < 1e-6, "point off sphere: norm={}", norm);
        }
    }
}

#[test]
fn geodesic_euler_stays_on_so3() {
    let so3 = SpecialOrthogonal::<3>;
    // Lie algebra element (skew-symmetric basis vector).
    let omega_alg = SMatrix::<f64, 3, 3>::from_row_slice(&[
        0.0, -1.0, 0.0,
        1.0,  0.0, 0.0,
        0.0,  0.0, 0.0,
    ]);
    // Tangent at R is R * omega_alg (left-translated). This keeps the
    // tangent vector in T_R(SO3) for all R, so exp stays on the manifold.
    let sde = ManifoldSDE::new(
        so3,
        |_x: &SMatrix<f64, 3, 3>, _t: f64| SMatrix::zeros(),
        move |x: &SMatrix<f64, 3, 3>, _t: f64| x * omega_alg,
    );
    let x0 = SMatrix::<f64, 3, 3>::identity();
    let paths = manifold_simulate(&sde, &GeodesicEuler, x0, 0.0, 1.0, 100, 100, 0);
    for path in &paths {
        for r in path {
            let rtr = r.transpose() * r;
            let id = SMatrix::<f64, 3, 3>::identity();
            assert!(
                (rtr - id).norm() < 1e-6,
                "SO3: R^T R != I, diff={}",
                (rtr - id).norm()
            );
        }
    }
}

#[test]
fn geodesic_milstein_stays_on_sphere() {
    let s2 = Sphere::<3>;
    let sde = ManifoldSDE::new(
        s2,
        |_x: &SVector<f64, 3>, _t: f64| SVector::zeros(),
        |x: &SVector<f64, 3>, _t: f64| {
            let e1 = SVector::from([1.0_f64, 0.0, 0.0]);
            e1 - x * x.dot(&e1)
        },
    );
    let x0 = SVector::from([0.0_f64, 0.0, 1.0]);
    let paths = manifold_simulate_with_scheme(&sde, &GeodesicMilstein::new(), x0, 0.0, 1.0, 100, 100, 0);
    for path in &paths {
        for point in path {
            let norm = point.norm();
            assert!((norm - 1.0).abs() < 1e-6, "Milstein point off sphere: {}", norm);
        }
    }
}

#[test]
fn paths_to_array_shape() {
    use pathwise_geo::paths_to_array;
    let sde = sphere_sde();
    let x0 = SVector::from([0.0_f64, 0.0, 1.0]);
    let paths = manifold_simulate(&sde, &GeodesicEuler, x0, 0.0, 1.0, 10, 50, 0);
    let arr = paths_to_array(&paths, &Sphere::<3>, &x0);
    assert_eq!(arr.shape(), &[10, 51, 3]);
}
