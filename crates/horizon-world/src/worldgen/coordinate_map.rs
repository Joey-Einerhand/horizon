//! The level coordinate map hook, alone in its module: a `[[native.mixin]]` copy of a replaceable
//! function takes its module's imports with it, so this one imports nothing a mod's copy can't
//! reach.

use std::sync::Arc;

/// Where level `dimension`'s noises read each block column (`horizon_core::density::coordinate_map`):
/// in place, in vanilla. A mod whose world wraps, such as a ringworld, replaces this with a
/// `[[native.mixin]]` that answers its map for the Overworld, so terrain, climate, biomes,
/// aquifers, far terrain and spawn finding join where its X wraps. Asked once each time a
/// level's generator is built.
#[horizon_mod_api_macros::replaceable]
pub fn coordinate_map_for(
    dimension: &str,
    seed: i64,
) -> Option<Arc<dyn horizon_core::density::CoordinateMap>> {
    let _ = (dimension, seed);
    None
}
