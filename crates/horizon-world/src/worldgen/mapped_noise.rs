//! A noise read at a level's coordinate map (`horizon_core::density::coordinate_map`): for the
//! surface's noises and surface rules' `noise_threshold`, which read whole block columns.
use horizon_core::{
    density::CoordinateMap,
    noise::{NanRange, Noise},
};
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct MappedNoise {
    input: Arc<dyn Noise>,
    map: Arc<dyn CoordinateMap>,
}
impl MappedNoise {
    pub(crate) fn new(input: Arc<dyn Noise>, map: Arc<dyn CoordinateMap>) -> Self {
        Self { input, map }
    }
    fn column(&self, x: f64, z: f64) -> [f64; 2] {
        // Callers pass whole block coordinates.
        let [x, z] = self.map.map(x.floor() as i32, z.floor() as i32);
        [f64::from(x), f64::from(z)]
    }
}
impl Noise for MappedNoise {
    fn range(&self) -> Result<[f32; 2], NanRange> {
        self.input.range()
    }
    fn get2(&self, x: f64, z: f64) -> f32 {
        let [x, z] = self.column(x, z);
        self.input.get2(x, z)
    }
    fn get3(&self, x: f64, y: f64, z: f64) -> f32 {
        let [x, z] = self.column(x, z);
        self.input.get3(x, y, z)
    }
}
