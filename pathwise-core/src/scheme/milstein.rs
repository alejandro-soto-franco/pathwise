use super::Scheme;
use crate::process::markov::Drift;
use crate::state::{Diffusion, Increment};

pub struct Milstein {
    h: f64,
}

impl Milstein {
    pub fn new(h: f64) -> Self {
        Self { h }
    }
}

/// Scalar Milstein scheme (strong order 1.0 for state-dependent diffusion).
/// Uses central finite difference to approximate dg/dx.
/// x_{n+1} = x_n + f*dt + g*dw + 0.5*g*(dg/dx)*(dw^2 - dt)
impl Scheme<f64> for Milstein {
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
        let f = drift(x, t);
        // g(x,t) via apply with unit noise
        let g = diffusion.apply(x, t, &1.0_f64);
        let g_plus = diffusion.apply(&(x + self.h), t, &1.0_f64);
        let g_minus = diffusion.apply(&(x - self.h), t, &1.0_f64);
        let dg_dx = (g_plus - g_minus) / (2.0 * self.h);
        x + f * dt + g * dw + 0.5 * g * dg_dx * (dw * dw - dt)
    }
}

pub fn milstein() -> Milstein {
    Milstein::new(1e-5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::markov::bm;
    use crate::state::Increment;

    #[test]
    fn milstein_equals_euler_for_constant_diffusion() {
        let b = bm();
        let m = milstein();
        let e = crate::scheme::euler::euler();
        let x = 1.0_f64;
        let inc = Increment {
            dw: 0.3_f64,
            dz: 0.0,
        };
        let x_euler = e.step(&b.drift, &b.diffusion, &x, 0.0, 0.01, &inc);
        let x_milstein = m.step(&b.drift, &b.diffusion, &x, 0.0, 0.01, &inc);
        assert!(
            (x_euler - x_milstein).abs() < 1e-8,
            "euler={} milstein={}",
            x_euler,
            x_milstein
        );
    }

    #[test]
    fn milstein_differs_from_euler_for_state_dependent_diffusion() {
        let gbm = crate::process::markov::gbm(0.0, 0.2);
        let m = milstein();
        let e = crate::scheme::euler();
        let x = 1.0_f64;
        let inc = Increment {
            dw: 0.5_f64,
            dz: 0.0,
        };
        let dt = 0.01;
        let x_euler = e.step(&gbm.drift, &gbm.diffusion, &x, 0.0, dt, &inc);
        let x_milstein = m.step(&gbm.drift, &gbm.diffusion, &x, 0.0, dt, &inc);
        assert!(
            (x_euler - x_milstein).abs() > 1e-4,
            "should differ by correction term"
        );
    }
}
