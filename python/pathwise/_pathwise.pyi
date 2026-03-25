from collections.abc import Callable
from typing import Union

import numpy as np
import numpy.typing as npt

# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class NumericalDivergence(ArithmeticError):
    """Raised when a simulation path produces non-finite values."""

class ConvergenceError(RuntimeError):
    """Raised on inference/solver failures (reserved for future use)."""

# --------------------------------------------------------------------------
# SDE objects
# --------------------------------------------------------------------------

class SDE:
    """Opaque handle to a stochastic differential equation.

    Do not construct directly; use the factory functions below.
    """

# --------------------------------------------------------------------------
# Scheme objects
# --------------------------------------------------------------------------

class Euler:
    """Euler-Maruyama scheme (strong order 0.5, weak order 1)."""
    def __repr__(self) -> str: ...

class Milstein:
    """Milstein scheme (strong order 1, weak order 1)."""
    def __repr__(self) -> str: ...

class Sri:
    """SRI strong-order 1.5 scheme (Kloeden-Platen Taylor 1.5).

    Not supported for Heston or corr_ou; raises ValueError if used with them.
    """
    def __repr__(self) -> str: ...

# --------------------------------------------------------------------------
# Process constructors
# --------------------------------------------------------------------------

def bm() -> SDE:
    """Standard Brownian motion: dX = dW."""
    ...

def gbm(mu: float, sigma: float) -> SDE:
    """Geometric Brownian motion: dX = mu*X dt + sigma*X dW."""
    ...

def ou(theta: float, mu: float, sigma: float) -> SDE:
    """Ornstein-Uhlenbeck: dX = theta*(mu - X) dt + sigma dW."""
    ...

def cir(kappa: float, theta: float, sigma: float) -> SDE:
    """Cox-Ingersoll-Ross: dX = kappa*(theta - X) dt + sigma*sqrt(X) dW.

    Raises ValueError if the Feller condition 2*kappa*theta > sigma^2 is violated.
    """
    ...

def heston(mu: float, kappa: float, theta: float, xi: float, rho: float) -> SDE:
    """Heston stochastic volatility model.

    State: [log S, V].
    simulate() returns shape (n_paths, n_steps+1, 2):
      - index 0: log-price. x0 is log(S0), NOT S0.
      - index 1: variance. Initial variance defaults to theta.

    d(log S) = (mu - V/2) dt + sqrt(V) dW1
    dV       = kappa*(theta - V) dt + xi*sqrt(V)*(rho*dW1 + sqrt(1-rho^2)*dW2)
    """
    ...

def corr_ou(theta: float, mu: list[float], sigma_flat: list[float]) -> SDE:
    """Correlated Ornstein-Uhlenbeck (N-dimensional).

    mu: mean vector of length N.
    sigma_flat: flattened N*N covariance matrix in row-major order.
    Python bindings support N=2 only; use the Rust API for higher dimensions.

    Raises ValueError if len(sigma_flat) != len(mu)**2.
    """
    ...

def sde(
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
) -> SDE:
    """Construct a custom SDE from Python drift and diffusion callables.

    drift(x, t) -> float
    diffusion(x, t) -> float

    Custom SDEs run serially (GIL cannot be released for Python callables).
    """
    ...

# --------------------------------------------------------------------------
# Scheme constructors
# --------------------------------------------------------------------------

def euler() -> Euler:
    """Euler-Maruyama scheme."""
    ...

def milstein() -> Milstein:
    """Milstein scheme (requires differentiable diffusion)."""
    ...

def sri() -> Sri:
    """SRI strong-order 1.5 scheme (Kloeden-Platen Taylor 1.5)."""
    ...

# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate(
    sde: SDE,
    scheme: Union[Euler, Milstein, Sri],
    n_paths: int,
    n_steps: int,
    t1: float,
    x0: float = 0.0,
    t0: float = 0.0,
    device: str = "cpu",
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Simulate SDE paths.

    Parameters
    ----------
    sde:
        SDE object (bm, gbm, ou, cir, heston, corr_ou, or custom sde()).
    scheme:
        Numerical scheme (euler(), milstein(), or sri()).
    n_paths:
        Number of independent paths to simulate.
    n_steps:
        Number of time steps per path.
    t1:
        End time.
    x0:
        Initial value. For Heston, this is log(S0), not S0.
    t0:
        Start time (default 0.0).
    device:
        Compute device. Only "cpu" is supported; "cuda" raises NotImplementedError.
    seed:
        RNG seed for reproducibility. Same seed produces identical paths.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_steps + 1) for scalar SDEs (bm, gbm, ou, cir, custom).
        Shape (n_paths, n_steps + 1, N) for N-dimensional SDEs (heston N=2, corr_ou).

    Raises
    ------
    ValueError
        If n_paths == 0, n_steps == 0, t1 <= t0, or sri() is used with heston/corr_ou.
    NotImplementedError
        If device != "cpu".
    """
    ...
