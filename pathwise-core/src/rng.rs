/// splitmix64 hash function for deriving per-path seeds from a base seed.
///
/// This function is used by `simulate` and `simulate_nd` to produce independent
/// per-path RNG seeds from a single user-supplied seed. The same derivation is
/// used in `pathwise-geo` and `pathwise-py` to ensure cross-crate seed
/// reproducibility.
#[inline]
pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
