// pathwise-core/src/state.rs
use rand::Rng;
use std::ops::{Add, Mul};

/// Both Brownian increments for one time step.
/// `dw` = dW = z1 * sqrt(dt)
/// `dz` = integral_0^dt W(s) ds = (dt/2)*dw - sqrt(dt^3/12)*z2
///
/// Euler and Milstein ignore `dz`. SRI uses both.
/// Derivation: dZ = dt*dW - I_{(0,1)}, conditional mean of I_{(0,1)} given dW
/// is (dt/2)*dW, conditional variance is dt^3/12. Negative sign is correct.
/// Verified: E[dZ]=0, Var[dZ]=dt^3/3, Cov(dW,dZ)=dt^2/2.
#[derive(Clone, Debug)]
pub struct Increment<B: Clone> {
    pub dw: B,
    pub dz: B,
}

/// Algebraic requirements for SDE state types.
/// `f64` and `nalgebra::SVector<f64, N>` both satisfy this automatically.
pub trait State:
    Clone + Send + Sync + 'static + Add<Output = Self> + Mul<f64, Output = Self>
{
    fn zero() -> Self;
}

impl State for f64 {
    fn zero() -> Self {
        0.0
    }
}

impl<const N: usize> State for nalgebra::SVector<f64, N> {
    fn zero() -> Self {
        nalgebra::SVector::zeros()
    }
}

/// Types that can sample a Brownian increment for a given dt.
///
/// # Object safety
/// This trait is not object-safe because `sample` is generic over `R: Rng`.
/// All uses are monomorphic — `dyn NoiseIncrement` will not compile.
pub trait NoiseIncrement: Clone + Send + Sync + 'static {
    fn sample<R: Rng>(rng: &mut R, dt: f64) -> Increment<Self>;
}

impl NoiseIncrement for f64 {
    fn sample<R: Rng>(rng: &mut R, dt: f64) -> Increment<Self> {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0_f64, 1.0).unwrap();
        let z1 = normal.sample(rng);
        let z2 = normal.sample(rng);
        let dw = z1 * dt.sqrt();
        let dz = (dt / 2.0) * dw - (dt.powi(3) / 12.0).sqrt() * z2;
        Increment { dw, dz }
    }
}

impl<const M: usize> NoiseIncrement for nalgebra::SVector<f64, M> {
    fn sample<R: Rng>(rng: &mut R, dt: f64) -> Increment<Self> {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0_f64, 1.0).unwrap();
        let mut dw = nalgebra::SVector::<f64, M>::zeros();
        let mut dz = nalgebra::SVector::<f64, M>::zeros();
        for i in 0..M {
            let z1 = normal.sample(rng);
            let z2 = normal.sample(rng);
            dw[i] = z1 * dt.sqrt();
            dz[i] = (dt / 2.0) * dw[i] - (dt.powi(3) / 12.0).sqrt() * z2;
        }
        Increment { dw, dz }
    }
}

/// Diffusion coefficient interface.
/// `apply(x, t, dw)` returns the diffusion contribution `g(x,t) * dw` directly.
///
/// Blanket impl for scalar closures: `f(x,t) -> f64` satisfies `Diffusion<f64, f64>`
/// by computing `f(x,t) * dw`.
///
/// Blanket impl for nD diagonal closures: `f(x,t) -> SVector<N>` satisfies
/// `Diffusion<SVector<N>, SVector<N>>` by element-wise (Hadamard) product.
///
/// Full-matrix processes (Heston, CorrOU) provide concrete struct impls.
///
/// # Calling convention
/// Scalar diffusions (`B = f64`) receive `x` by value because `f64: Copy`.
/// Vector diffusions (`B = SVector<f64, N>`) receive `x` by reference.
/// Concrete struct impls (e.g. `HestonDiffusion`) use whichever is appropriate for their state type.
pub trait Diffusion<S: State, B: NoiseIncrement>: Send + Sync {
    fn apply(&self, x: &S, t: f64, dw: &B) -> S;
}

// Scalar blanket impl: closure returns g(x,t); apply multiplies by dw.
impl<F: Fn(f64, f64) -> f64 + Send + Sync> Diffusion<f64, f64> for F {
    fn apply(&self, x: &f64, t: f64, dw: &f64) -> f64 {
        self(*x, t) * dw
    }
}

// nD diagonal blanket impl: closure returns sigma vector; apply component-multiplies with dw.
impl<const N: usize, F> Diffusion<nalgebra::SVector<f64, N>, nalgebra::SVector<f64, N>> for F
where
    F: Fn(&nalgebra::SVector<f64, N>, f64) -> nalgebra::SVector<f64, N> + Send + Sync,
{
    fn apply(
        &self,
        x: &nalgebra::SVector<f64, N>,
        t: f64,
        dw: &nalgebra::SVector<f64, N>,
    ) -> nalgebra::SVector<f64, N> {
        self(x, t).component_mul(dw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn increment_f64_variance() {
        // Var[dW] ≈ dt; Var[dZ] ≈ dt^3/3; Cov(dW,dZ) ≈ dt^2/2
        let dt = 0.01_f64;
        let n = 100_000_usize;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let mut dws = vec![0.0_f64; n];
        let mut dzs = vec![0.0_f64; n];
        for i in 0..n {
            let inc = <f64 as NoiseIncrement>::sample(&mut rng, dt);
            dws[i] = inc.dw;
            dzs[i] = inc.dz;
        }
        let var_dw: f64 = dws.iter().map(|x| x * x).sum::<f64>() / n as f64;
        let var_dz: f64 = dzs.iter().map(|x| x * x).sum::<f64>() / n as f64;
        let cov: f64 = dws.iter().zip(&dzs).map(|(w, z)| w * z).sum::<f64>() / n as f64;
        // Var[dW] = dt = 0.01
        assert!(
            (var_dw - dt).abs() / dt < 0.02,
            "Var[dW]={:.6} expected {:.6}",
            var_dw,
            dt
        );
        // Var[dZ] = dt^3/3
        let expected_var_dz = dt.powi(3) / 3.0;
        assert!(
            (var_dz - expected_var_dz).abs() / expected_var_dz < 0.03,
            "Var[dZ]={:.8} expected {:.8}",
            var_dz,
            expected_var_dz
        );
        // Cov(dW,dZ) = dt^2/2
        let expected_cov = dt.powi(2) / 2.0;
        assert!(
            (cov - expected_cov).abs() / expected_cov < 0.02,
            "Cov(dW,dZ)={:.8} expected {:.8}",
            cov,
            expected_cov
        );
    }

    #[test]
    fn increment_svector_has_correct_length() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let inc = <nalgebra::SVector<f64, 3> as NoiseIncrement>::sample(&mut rng, 0.01);
        assert_eq!(inc.dw.len(), 3);
        assert_eq!(inc.dz.len(), 3);
    }
}
