pub struct GeodesicSRI { pub eps: f64 }
impl GeodesicSRI {
    pub fn new() -> Self { Self { eps: 1e-4 } }
}
impl Default for GeodesicSRI { fn default() -> Self { Self::new() } }
