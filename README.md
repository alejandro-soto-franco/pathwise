# pathwise

High-performance SDE simulation and calibration toolkit — Rust core, Python API.

## Features (v0.1)

- Euler-Maruyama and Milstein schemes
- Standard processes: Brownian motion, GBM, Ornstein-Uhlenbeck
- Composable functional API: define SDEs from any Python callable
- Parallel CPU simulation via Rayon
- NumPy output

## Install

```bash
pip install pathwise
```

Or build from source:
```bash
pip install maturin
maturin develop
```

## Quick Start

```python
import pathwise as pw

# Simulate 10,000 GBM paths
paths = pw.simulate(
    pw.gbm(mu=0.05, sigma=0.2),
    scheme=pw.euler(),
    n_paths=10_000,
    n_steps=252,
    t1=1.0,
    x0=100.0,
)
# paths: np.ndarray shape (10_000, 253)

# Custom SDE from callables
custom = pw.sde(
    drift=lambda x, t: -0.5 * x,
    diffusion=lambda x, t: 1.0,
)
paths = pw.simulate(custom, pw.milstein(), n_paths=1000, n_steps=500, t1=1.0)
```

## Roadmap

- v0.2: fractional BM, Volterra processes, Bayer-Friz-Gatheral scheme
- v0.3: CUDA GPU acceleration
- v0.4: MLE inference, particle filter
- v0.5: rough Heston / rough Bergomi surface calibration
