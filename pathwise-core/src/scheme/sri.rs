// pathwise-core/src/scheme/sri.rs
use super::Scheme;
use crate::state::{Diffusion, Increment};
use crate::process::markov::Drift;

/// Kloeden-Platen strong order 1.5 Taylor scheme for scalar autonomous SDEs.
///
/// For the Ito SDE `dX = f(X) dt + g(X) dW`, the step is:
///
///   x_{n+1} = x + f*dt + g*dW
///             + 0.5*g*g'*(dW^2 - dt)                         [Milstein: L^1 g * I_{(1,1)}]
///             + g*f' * (dt*dW - dZ)                           [L^1 f * I_{(1,0)} term]
///             + (f*g' + 0.5*g^2*g'') * dZ                    [L^0 g * I_{(0,1)} term]
///             + 0.5*f*f' * dt^2                               [L^0 f * dt^2/2 term]
///             + (g*g'^2 + g^2*g'') * (dW^3 - 3*dt*dW)/6     [(L^1)^2 g * I_{(1,1,1)}]
///
/// where:
///   `dZ = integral_0^dt W(s) ds = I_{(0,1)}`
///   `I_{(1,0)} = dt*dW - dZ`
///   `I_{(1,1)} = (dW^2 - dt)/2`
///   `I_{(1,1,1)} = (dW^3 - 3*dt*dW)/6`
///
/// All spatial derivatives are approximated by central finite differences with step `h`.
///
/// # Reference
/// Kloeden & Platen, "Numerical Solution of Stochastic Differential Equations", 1992,
/// Chapter 5.5, the strong Taylor 1.5 scheme (Theorem 5.5.1 / eq. 5.5.4).
///
/// # Notes
/// - The `I_{(1,1,1)}` triple iterated integral is expressible in terms of `dW` alone:
///   `I_{(1,1,1)} = (dW^3 - 3*dt*dW)/6`. This is the key 1.5-order correction beyond Milstein.
/// - For BM (constant diffusion), all correction terms vanish, recovering Euler.
/// - For GBM (`g = sigma*x`, `g' = sigma`, `g'' = 0`):
///   - `d111 = g*g'^2 + g^2*g'' = sigma^2*x*sigma = sigma^3*x`
///   - `I_{(1,1,1)} = (dW^3 - 3*dt*dW)/6`
///   - Combined with the other 1.5 terms, local error reduces to O(dt^2), giving O(dt^{1.5}) globally.
/// - Requires scalar (1D) SDE with diagonal noise.
pub struct Sri {
    h: f64,
}

impl Sri {
    pub fn new(h: f64) -> Self { Self { h } }
}

impl Scheme<f64> for Sri {
    type Noise = f64;

    fn step<D, G>(
        &self,
        drift: &D,
        diffusion: &G,
        x: &f64,
        t: f64,
        dt: f64,
        inc: &Increment<f64>,
    ) -> f64
    where
        D: Drift<f64>,
        G: Diffusion<f64, f64>,
    {
        let dw = inc.dw;
        let dz = inc.dz;  // I_{(0,1)} = integral_0^dt W(s) ds
        let h = self.h;

        // f(x,t) and its central finite difference
        let f = drift(x, t);
        let f_plus  = drift(&(x + h), t);
        let f_minus = drift(&(x - h), t);
        let df_dx   = (f_plus - f_minus) / (2.0 * h);

        // g(x,t) and its finite-difference derivatives (unit noise via blanket impl)
        let g       = diffusion.apply(x, t, &1.0_f64);
        let g_plus  = diffusion.apply(&(x + h), t, &1.0_f64);
        let g_minus = diffusion.apply(&(x - h), t, &1.0_f64);
        let dg_dx   = (g_plus - g_minus) / (2.0 * h);
        let d2g_dx2 = (g_plus - 2.0 * g + g_minus) / (h * h);

        // I_{(1,0)} = dt*dW - I_{(0,1)} = dt*dW - dZ
        let i10 = dt * dw - dz;
        // I_{(1,1)} = (dW^2 - dt)/2  (Milstein)
        let i11 = (dw * dw - dt) * 0.5;
        // I_{(1,1,1)} = (dW^3 - 3*dt*dW)/6  (triple iterated Ito integral, expressed via dW)
        let i111 = (dw * dw * dw - 3.0 * dt * dw) / 6.0;

        // Milstein correction: L^1 g * I_{(1,1)}
        let milstein = g * dg_dx * i11;

        // L^1 f * I_{(1,0)}: g * f' * (dt*dW - dZ)
        let term_l1f = g * df_dx * i10;

        // L^0 g * I_{(0,1)}: (f * g' + 0.5 * g^2 * g'') * dZ
        let term_l0g = (f * dg_dx + 0.5 * g * g * d2g_dx2) * dz;

        // L^0 f * dt^2/2: f * f' * dt^2/2
        let term_l0f = 0.5 * f * df_dx * dt * dt;

        // (L^1)^2 g * I_{(1,1,1)}: (g*g'^2 + g^2*g'') * (dW^3 - 3*dt*dW)/6
        let d111 = g * dg_dx * dg_dx + g * g * d2g_dx2;
        let term_l11g = d111 * i111;

        x + f * dt + g * dw + milstein + term_l1f + term_l0g + term_l0f + term_l11g
    }
}

pub fn sri() -> Sri { Sri::new(1e-4) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::markov::bm;
    use crate::state::Increment;

    #[test]
    fn sri_equals_euler_for_constant_diffusion() {
        // For constant g: g'=0, g''=0, so all derivative-based corrections vanish.
        // For BM: f=0, so L^0f and L^1f terms also vanish. SRI == Euler for BM.
        let b = bm();
        let s = sri();
        let e = crate::scheme::euler::euler();
        let x = 1.0_f64;
        let inc = Increment { dw: 0.3, dz: 0.001 };
        let x_euler = e.step(&b.drift, &b.diffusion, &x, 0.0, 0.01, &inc);
        let x_sri = s.step(&b.drift, &b.diffusion, &x, 0.0, 0.01, &inc);
        assert!((x_euler - x_sri).abs() < 1e-6,
            "BM: euler={} sri={}", x_euler, x_sri);
    }

    #[test]
    fn sri_differs_from_milstein_for_state_dependent_diffusion() {
        // GBM: f'=mu, g'=sigma, g''=0.
        // SRI adds L^1 f (drift-noise cross), L^0 g (noise-drift), and I_{(1,1,1)} terms.
        let gbm = crate::process::markov::gbm(0.05, 0.3);
        let s = sri();
        let m = crate::scheme::milstein::milstein();
        let x = 1.0_f64;
        let inc = Increment { dw: 0.05, dz: 0.0001 };
        let dt = 0.01;
        let x_sri = s.step(&gbm.drift, &gbm.diffusion, &x, 0.0, dt, &inc);
        let x_mil = m.step(&gbm.drift, &gbm.diffusion, &x, 0.0, dt, &inc);
        // They differ because SRI includes the I_{(1,0)}, I_{(0,1)}, and I_{(1,1,1)} terms
        assert!((x_sri - x_mil).abs() > 1e-10, "SRI and Milstein should differ");
    }
}
