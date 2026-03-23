//! High-performance SDE simulation library.
//!
//! # Quick start
//!
//! ```
//! use pathwise_core::{simulate, euler, gbm};
//!
//! let g = gbm(0.05, 0.2);
//! let scheme = euler();
//! let paths = simulate(
//!     &g.drift,
//!     &g.diffusion,
//!     &scheme,
//!     100.0, // x0
//!     0.0,   // t0
//!     1.0,   // t1
//!     10,    // n_paths
//!     252,   // n_steps
//!     42,    // seed
//! ).expect("simulate failed");
//! assert_eq!(paths.shape(), &[10, 253]);
//! ```

// pathwise-core: pure Rust SDE simulation library
pub mod error;
pub use error::PathwiseError;

pub mod process;
pub use process::{bm, gbm, ou, Diffusion, Drift, State, SDE};

pub mod scheme;
pub use scheme::{euler, milstein, Scheme};

pub mod simulate;
pub use simulate::simulate;
