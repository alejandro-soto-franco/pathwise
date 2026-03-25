use crate::sde::ManifoldSDE;
use cartan_core::Manifold;

/// Brownian motion on a manifold with an explicit diffusion direction.
///
/// Zero drift; `diffusion_gen(x, t)` provides the tangent vector g(x, t).
/// Directional (1D noise) -- isotropic BM requires a frame field (future work).
pub fn brownian_motion_on_with_diffusion<M, G>(
    manifold: M,
    diffusion_gen: G,
) -> ManifoldSDE<
    M,
    impl Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    G,
>
where
    M: Manifold + Clone + Send + Sync,
    M::Point: Clone + Send + Sync,
    M::Tangent: Clone + Default + Send + Sync,
    G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
{
    ManifoldSDE::new(
        manifold,
        |_x: &M::Point, _t: f64| M::Tangent::default(),
        diffusion_gen,
    )
}

/// Ornstein-Uhlenbeck process on a manifold with caller-supplied diffusion field.
///
/// Drift: `kappa * log_x(mu_point)` -- Riemannian log toward `mu_point`, scaled by `kappa`.
/// Returns zero tangent on failure (when x == mu_point or log fails at cut locus).
pub fn ou_on_with_diffusion<M, G>(
    manifold: M,
    kappa: f64,
    mu_point: M::Point,
    diffusion: G,
) -> ManifoldSDE<
    M,
    impl Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    G,
>
where
    M: Manifold + Clone + Send + Sync,
    M::Point: Clone + Send + Sync,
    M::Tangent: Clone + Default + std::ops::Mul<f64, Output = M::Tangent> + Send + Sync,
    G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
{
    let mu_clone = mu_point.clone();
    let manifold_clone = manifold.clone();
    ManifoldSDE::new(
        manifold,
        move |x: &M::Point, _t: f64| -> M::Tangent {
            manifold_clone
                .log(x, &mu_clone)
                .map(|v| v * kappa)
                .unwrap_or_default()
        },
        diffusion,
    )
}

/// Placeholder for API compatibility -- panics if the diffusion closure is called.
/// Use `brownian_motion_on_with_diffusion` instead.
#[deprecated(note = "Use brownian_motion_on_with_diffusion; generic Manifold has no frame field")]
pub fn brownian_motion_on<M: Manifold + Clone>(
    manifold: M,
) -> ManifoldSDE<
    M,
    impl Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    impl Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
>
where
    M::Tangent: Clone + Default,
{
    ManifoldSDE::new(
        manifold,
        |_x: &M::Point, _t: f64| M::Tangent::default(),
        |_x: &M::Point, _t: f64| -> M::Tangent {
            panic!("brownian_motion_on: provide explicit diffusion via brownian_motion_on_with_diffusion")
        },
    )
}
