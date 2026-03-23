use super::Scheme;
use crate::process::markov::{Diffusion, Drift};

pub struct EulerMaruyama;

impl Scheme for EulerMaruyama {
    fn step<D: Drift<f64>, G: Diffusion<f64>>(
        &self,
        drift: &D,
        diffusion: &G,
        x: f64,
        t: f64,
        dt: f64,
        dw: f64,
    ) -> f64 {
        x + drift(&x, t) * dt + diffusion(&x, t) * dw
    }
}

pub fn euler() -> EulerMaruyama {
    EulerMaruyama
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::markov::ou;

    #[test]
    fn euler_step_on_ou_is_deterministic_with_zero_noise() {
        let scheme = euler();
        let o = ou(1.0, 0.0, 0.5);
        // x=1.0, dt=0.01, dw=0 => x_new = 1.0 + (-1.0)*0.01 + 0.5*0 = 0.99
        let x_new = scheme.step(&o.drift, &o.diffusion, 1.0, 0.0, 0.01, 0.0);
        assert!((x_new - 0.99).abs() < 1e-12);
    }

    #[test]
    fn euler_step_adds_diffusion_term_with_positive_noise() {
        let scheme = euler();
        let o = ou(1.0, 0.0, 0.5);
        // dw = 1.0 => x_new = 1.0 + (-1.0)*0.01 + 0.5*1.0 = 1.49
        let x_new = scheme.step(&o.drift, &o.diffusion, 1.0, 0.0, 0.01, 1.0);
        assert!((x_new - 1.49).abs() < 1e-12);
    }
}
