use crate::process::markov::{Drift, Diffusion};
use super::Scheme;

pub struct Milstein {
    h: f64, // finite difference step for d(sigma)/dx
}

impl Milstein {
    pub fn new(h: f64) -> Self { Self { h } }
}

impl Scheme for Milstein {
    fn step<D: Drift<f64>, G: Diffusion<f64>>(
        &self,
        drift: &D,
        diffusion: &G,
        x: f64,
        t: f64,
        dt: f64,
        dw: f64,
    ) -> f64 {
        let f = drift(&x, t);
        let g = diffusion(&x, t);
        // central difference: dg/dx ~ (g(x+h) - g(x-h)) / (2h)
        let dg_dx = (diffusion(&(x + self.h), t) - diffusion(&(x - self.h), t)) / (2.0 * self.h);
        x + f * dt + g * dw + 0.5 * g * dg_dx * (dw * dw - dt)
    }
}

pub fn milstein() -> Milstein { Milstein::new(1e-5) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::markov::bm;

    #[test]
    fn milstein_equals_euler_for_constant_diffusion() {
        // For constant sigma, dg/dx = 0, so Milstein == Euler
        let b = bm(); // diffusion = 1.0 (constant)
        let m = milstein();
        let e = crate::scheme::euler::euler();
        let x = 1.0;
        let dw = 0.3;
        let dt = 0.01;
        let x_euler = e.step(&b.drift, &b.diffusion, x, 0.0, dt, dw);
        let x_milstein = m.step(&b.drift, &b.diffusion, x, 0.0, dt, dw);
        assert!((x_euler - x_milstein).abs() < 1e-8, "euler={} milstein={}", x_euler, x_milstein);
    }

    #[test]
    fn milstein_differs_from_euler_for_state_dependent_diffusion() {
        // GBM: sigma(x) = sigma*x => dg/dx = sigma != 0
        let gbm = crate::process::markov::gbm(0.0, 0.2); // zero drift, sigma=0.2
        let m = milstein();
        let e = crate::scheme::euler();
        let x = 1.0;
        let dw = 0.5;
        let dt = 0.01;
        let x_euler = e.step(&gbm.drift, &gbm.diffusion, x, 0.0, dt, dw);
        let x_milstein = m.step(&gbm.drift, &gbm.diffusion, x, 0.0, dt, dw);
        assert!((x_euler - x_milstein).abs() > 1e-4, "should differ by correction term");
    }
}
