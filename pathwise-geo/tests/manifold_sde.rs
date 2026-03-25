use cartan_manifolds::{sphere::Sphere, so::SpecialOrthogonal, spd::Spd};
use nalgebra::{SMatrix, SVector};
use pathwise_geo::{GeodesicEuler, GeodesicMilstein, GeodesicSRI, ManifoldSDE, manifold_simulate, manifold_simulate_with_scheme};
use pathwise_geo::ou_on_with_diffusion;

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
fn ou_on_sphere_mean_reverts() {
    let s2 = Sphere::<3>;
    let mu = SVector::from([0.0_f64, 0.0, 1.0]);
    let x0 = SVector::from([0.0_f64, 0.0, -1.0]);
    let sde = ou_on_with_diffusion(s2, 2.0, mu,
        |x: &SVector<f64, 3>, _t: f64| {
            let e1 = SVector::from([1.0_f64, 0.0, 0.0]);
            e1 - x * x.dot(&e1)
        });
    let paths = manifold_simulate_with_scheme(&sde, &GeodesicEuler, x0, 0.0, 2.0, 500, 400, 42);
    let dist_late: f64 = paths.iter().map(|p| {
        let x = &p[400];
        x.dot(&mu).clamp(-1.0, 1.0).acos()
    }).sum::<f64>() / paths.len() as f64;
    let dist_early: f64 = paths.iter().map(|p| {
        let x = &p[20];
        x.dot(&mu).clamp(-1.0, 1.0).acos()
    }).sum::<f64>() / paths.len() as f64;
    println!("OU S2: dist_early={:.4} dist_late={:.4}", dist_early, dist_late);
    assert!(dist_late < dist_early * 0.95,
        "OU should mean-revert: dist_late={:.4} dist_early={:.4}", dist_late, dist_early);
}

#[test]
fn geodesic_sri_stays_on_sphere() {
    let s2 = Sphere::<3>;
    let sde = ManifoldSDE::new(
        s2,
        |_x: &SVector<f64, 3>, _t: f64| SVector::<f64, 3>::zeros(),
        |x: &SVector<f64, 3>, _t: f64| {
            let e1 = SVector::from([1.0_f64, 0.0, 0.0]);
            e1 - x * x.dot(&e1)
        },
    );
    let x0 = SVector::from([0.0_f64, 0.0, 1.0]);
    let paths = manifold_simulate_with_scheme(&sde, &GeodesicSRI::new(), x0, 0.0, 1.0, 100, 100, 0);
    for path in &paths {
        for point in path {
            let norm = point.norm();
            assert!((norm - 1.0).abs() < 1e-6, "SRI point off sphere: {}", norm);
        }
    }
}

#[test]
fn ou_on_spd_stays_positive_definite() {
    let spd = Spd::<2>;
    let mu = SMatrix::<f64, 2, 2>::identity();
    let x0 = SMatrix::<f64, 2, 2>::identity() * 2.0;
    let sde = ou_on_with_diffusion(
        spd, 1.0, mu,
        |_x: &SMatrix<f64, 2, 2>, _t: f64| {
            SMatrix::from_row_slice(&[0.1, 0.05, 0.05, 0.1])
        },
    );
    let paths = manifold_simulate_with_scheme(&sde, &GeodesicEuler, x0, 0.0, 1.0, 100, 100, 0);
    for path in &paths {
        for mat in path {
            let eig = mat.symmetric_eigen();
            for &ev in eig.eigenvalues.iter() {
                assert!(ev > -1e-8, "SPD path produced non-positive eigenvalue: {}", ev);
            }
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
