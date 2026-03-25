use super::Scheme;
use crate::process::markov::Drift;
use crate::state::{Diffusion, Increment, NoiseIncrement, State};

pub struct EulerMaruyama;

/// Generic Euler-Maruyama: works for any State S with Noise = S.
/// x_{n+1} = x_n + f(x_n, t)*dt + g(x_n, t)*dW
impl<S: State + NoiseIncrement> Scheme<S> for EulerMaruyama {
    type Noise = S;

    fn step<D, G>(&self, drift: &D, diffusion: &G, x: &S, t: f64, dt: f64, inc: &Increment<S>) -> S
    where
        D: Drift<S>,
        G: Diffusion<S, S>,
    {
        let f_dt = drift(x, t) * dt;
        let g_dw = diffusion.apply(x, t, &inc.dw);
        x.clone() + f_dt + g_dw
    }
}

pub fn euler() -> EulerMaruyama {
    EulerMaruyama
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::markov::{bm, ou};
    use crate::state::Increment;

    #[test]
    fn euler_step_on_ou_is_deterministic_with_zero_noise() {
        let scheme = euler();
        let o = ou(1.0, 0.0, 0.5);
        // x=1.0, dt=0.01, dw=0 => x_new = 1.0 + (-1.0)*0.01 + 0.5*0 = 0.99
        let inc = Increment {
            dw: 0.0_f64,
            dz: 0.0_f64,
        };
        let x_new = scheme.step(&o.drift, &o.diffusion, &1.0_f64, 0.0, 0.01, &inc);
        assert!((x_new - 0.99).abs() < 1e-12);
    }

    #[test]
    fn euler_step_adds_diffusion_term_with_positive_noise() {
        let scheme = euler();
        let o = ou(1.0, 0.0, 0.5);
        // dw = 1.0 => x_new = 1.0 + (-1.0)*0.01 + 0.5*1.0 = 1.49
        let inc = Increment {
            dw: 1.0_f64,
            dz: 0.0_f64,
        };
        let x_new = scheme.step(&o.drift, &o.diffusion, &1.0_f64, 0.0, 0.01, &inc);
        assert!((x_new - 1.49).abs() < 1e-12);
    }

    #[test]
    fn euler_step_bm_is_x_plus_dw() {
        let b = bm();
        let e = euler();
        let inc = Increment {
            dw: 0.3_f64,
            dz: 0.0,
        };
        let x_new = e.step(&b.drift, &b.diffusion, &1.0_f64, 0.0, 0.01, &inc);
        // BM drift=0, diffusion=1: x_new = 1.0 + 0.0*0.01 + 1.0*0.3 = 1.3
        assert!((x_new - 1.3).abs() < 1e-12, "got {}", x_new);
    }

    #[test]
    fn euler_step_via_increment() {
        let e = euler();
        let b = bm();
        let inc = Increment {
            dw: 0.3_f64,
            dz: 0.0_f64,
        };
        let x_new: f64 = e.step(&b.drift, &b.diffusion, &0.0_f64, 0.0, 0.01, &inc);
        assert!(x_new.is_finite());
    }
}
