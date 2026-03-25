/// A stochastic differential equation on a Riemannian manifold.
pub struct ManifoldSDE<M, D, G> {
    pub manifold: M,
    pub drift: D,
    pub diffusion: G,
}

impl<M, D, G> ManifoldSDE<M, D, G> {
    pub fn new(manifold: M, drift: D, diffusion: G) -> Self {
        Self { manifold, drift, diffusion }
    }
}
