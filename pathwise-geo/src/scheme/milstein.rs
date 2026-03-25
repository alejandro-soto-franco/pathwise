pub struct GeodesicMilstein { pub eps: f64 }
impl GeodesicMilstein {
    pub fn new() -> Self { Self { eps: 1e-4 } }
}
impl Default for GeodesicMilstein { fn default() -> Self { Self::new() } }
