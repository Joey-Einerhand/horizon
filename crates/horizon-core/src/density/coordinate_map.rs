//! A level's horizontal coordinate map: where its noises are read for a block column.
//!
//! Vanilla reads every noise at the block's own X and Z. A level with a map reads its X/Z noise
//! leaves (noise, shifted noise, shifts, blended noise, end islands, X and Z gradients) at the
//! map's answer instead, while everything above them (caches, interpolation, splines, the
//! router's fields) keeps working on the block's real position. A world whose X wraps, such as
//! a ringworld, maps each column onto a circle of the noise plane, so its terrain, climate and
//! biomes join where X wraps.
//!
//! A level without a map compiles exactly the samplers it did before.

use super::{DensitySampler, SampleError, SamplerContext};
use std::sync::Arc;

/// Where a level's noises read block column (`x`, `z`). Pure in its inputs: the same column gives
/// the same answer on every thread, run and machine.
pub trait CoordinateMap: std::fmt::Debug + Send + Sync {
    fn map(&self, x: i32, z: i32) -> [i32; 2];
    /// The chunk whose seed worldgen started in chunk (`x`, `z`) draws from: carvers seed
    /// each start by its chunk. A world whose X wraps answers the canonical chunk, so a cave
    /// started across the seam is the one the canonical chunk starts. The chunk itself by default.
    fn seed_chunk(&self, x: i32, z: i32) -> [i32; 2] {
        [x, z]
    }
}

/// One X/Z-reading leaf, read at the mapped position. Inside one (a shifted noise's own shift
/// noises), positions are already mapped and pass through as they are.
#[derive(Debug)]
pub(crate) struct MappedSampler {
    pub(crate) input: Arc<dyn DensitySampler>,
    pub(crate) map: Arc<dyn CoordinateMap>,
}

impl DensitySampler for MappedSampler {
    fn sample_value(
        &self,
        context: &mut SamplerContext,
        [x, y, z]: [i32; 3],
    ) -> Result<f32, SampleError> {
        if context.mapped {
            return self.input.sample_value(context, [x, y, z]);
        }
        let [mx, mz] = self.map.map(x, z);
        context.mapped = true;
        let value = self.input.sample_value(context, [mx, y, mz]);
        context.mapped = false;
        value
    }
    // `sample_volume` keeps the trait's point-by-point default: a mapped volume is no longer a
    // box of the input's. `pointwise_inputs` stays `None`: the value is the input's at another
    // position, not at this one.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::density::TilingMode;
    use crate::density::graph::{DensityCompiler, DensityGraph, GraphNode};
    use crate::geometry::Axis;

    /// Swaps X and Z, which no gradient along X can miss.
    #[derive(Debug)]
    struct Swap;
    impl CoordinateMap for Swap {
        fn map(&self, x: i32, z: i32) -> [i32; 2] {
            [z, x]
        }
    }

    fn gradient_x(graph: &mut DensityGraph) -> crate::density::graph::FunctionId {
        graph
            .insert(GraphNode::Gradient {
                axis: Axis::X,
                tiling: TilingMode::ClampToEdge,
                from: 0,
                to: 100,
                low: 0.0,
                high: 1.0,
            })
            .unwrap()
    }

    #[test]
    fn a_mapped_level_reads_its_x_leaves_at_the_mapped_column() {
        let mut graph = DensityGraph::new();
        let id = gradient_x(&mut graph);
        let mut flat = DensityCompiler::new(graph.clone());
        let mut mapped = DensityCompiler::new(graph).with_coordinate_map(Some(Arc::new(Swap)));
        let mut context = SamplerContext::default();
        let flat = flat.get_sampler(id).unwrap();
        let mapped = mapped.get_sampler(id).unwrap();
        // Column (10, 90): flat reads x = 10, mapped reads x = 90.
        let a = flat.sample_value(&mut context, [10, 0, 90]).unwrap();
        let b = mapped.sample_value(&mut context, [10, 0, 90]).unwrap();
        let c = flat.sample_value(&mut context, [90, 0, 10]).unwrap();
        assert!((a - 0.1).abs() < 0.02, "{a}");
        assert_eq!(b, c);
    }

    #[test]
    fn without_a_map_nothing_is_wrapped() {
        let mut graph = DensityGraph::new();
        let id = gradient_x(&mut graph);
        let mut plain = DensityCompiler::new(graph).with_coordinate_map(None);
        let sampler = plain.get_sampler(id).unwrap();
        assert!(!format!("{sampler:?}").contains("MappedSampler"));
    }
}
