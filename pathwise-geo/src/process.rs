// Process constructors — full implementations in Task 9.
// brownian_motion_on and ou_on require manifold-specific tangent space sampling.
// These stubs compile but panic if called; Task 9 replaces them.
use crate::sde::ManifoldSDE;
use cartan_core::Manifold;

/// Placeholder: returns a ManifoldSDE with unimplemented drift and diffusion.
/// Use Task 9's process constructors instead (they supply explicit closures).
pub fn brownian_motion_on<M: Manifold + Clone>(
    manifold: M,
) -> ManifoldSDE<
    M,
    impl Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    impl Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
>
where
    M::Tangent: Clone,
{
    ManifoldSDE::new(
        manifold,
        |_x: &M::Point, _t: f64| -> M::Tangent { unimplemented!("provide explicit drift") },
        |_x: &M::Point, _t: f64| -> M::Tangent { unimplemented!("provide explicit diffusion") },
    )
}
