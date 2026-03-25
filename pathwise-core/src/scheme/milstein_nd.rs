use super::Scheme;
use crate::process::markov::Drift;
use crate::state::{Diffusion, Increment};
use nalgebra::SVector;

/// N-dimensional Milstein scheme with diagonal Levy area terms only.
///
/// For each component i:
///   correction[i] = 0.5 * g_ii * (dg_ii/dx_i) * (dW_i^2 - dt)
///
/// where g_ii = diffusion.apply(x, t, e_i)[i] and e_i is the unit vector.
/// Derivative via central finite difference.
///
/// Strong order 1.0 when commutativity holds; 0.5 for non-commutative systems.
pub struct MilsteinNd<const N: usize> {
    pub h: f64,
}

impl<const N: usize> MilsteinNd<N> {
    pub fn new(h: f64) -> Self {
        Self { h }
    }
}

impl<const N: usize> Scheme<SVector<f64, N>> for MilsteinNd<N> {
    type Noise = SVector<f64, N>;

    fn step<D, G>(
        &self,
        drift: &D,
        diffusion: &G,
        x: &SVector<f64, N>,
        t: f64,
        dt: f64,
        inc: &Increment<SVector<f64, N>>,
    ) -> SVector<f64, N>
    where
        D: Drift<SVector<f64, N>>,
        G: Diffusion<SVector<f64, N>, SVector<f64, N>>,
    {
        let dw = &inc.dw;
        let f_dt = drift(x, t) * dt;
        let g_dw = diffusion.apply(x, t, dw);

        let mut correction = SVector::<f64, N>::zeros();
        let h = self.h;
        for i in 0..N {
            let mut ei = SVector::<f64, N>::zeros();
            ei[i] = 1.0;
            let g_ii = diffusion.apply(x, t, &ei)[i];
            let mut xp = *x;
            xp[i] += h;
            let mut xm = *x;
            xm[i] -= h;
            let g_ii_plus = diffusion.apply(&xp, t, &ei)[i];
            let g_ii_minus = diffusion.apply(&xm, t, &ei)[i];
            let dg_ii = (g_ii_plus - g_ii_minus) / (2.0 * h);
            correction[i] = 0.5 * g_ii * dg_ii * (dw[i] * dw[i] - dt);
        }

        x + f_dt + g_dw + correction
    }
}

pub fn milstein_nd<const N: usize>() -> MilsteinNd<N> {
    MilsteinNd::new(1e-5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Increment;
    use nalgebra::SVector;

    #[test]
    fn milstein_nd_equals_euler_nd_for_constant_diffusion() {
        let m: MilsteinNd<2> = milstein_nd();
        let e = crate::scheme::euler::EulerMaruyama;
        let drift = |_x: &SVector<f64, 2>, _t: f64| SVector::zeros();
        let diff = |_x: &SVector<f64, 2>, _t: f64| SVector::from([1.0_f64, 1.0]);
        let x = SVector::from([0.5_f64, -0.5]);
        let dw = SVector::from([0.1_f64, -0.2]);
        let inc = Increment {
            dw,
            dz: SVector::zeros(),
        };
        let xe = e.step(&drift, &diff, &x, 0.0, 0.01, &inc);
        let xm = m.step(&drift, &diff, &x, 0.0, 0.01, &inc);
        assert!(
            (xe - xm).norm() < 1e-8,
            "should be equal for constant diffusion"
        );
    }
}
