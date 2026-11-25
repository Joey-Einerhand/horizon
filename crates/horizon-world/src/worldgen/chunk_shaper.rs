//! The chunk shaper hook, alone in its module: a `[[native.mixin]]` copy of a replaceable
//! function takes its module's imports with it, so this one imports nothing a mod's copy can't
//! reach.

use std::sync::Arc;

/// What level `dimension`'s generator lays over each chunk as it finishes it, before the chunk
/// is lit, saved or sent (`horizon_core::chunk_shape::ChunkShaper`): nothing, in vanilla. A mod
/// that reshapes its terrain, such as a ringworld's void and rim walls, replaces this with a
/// `[[native.mixin]]`, so a player never sees a chunk before its shape. Asked once each time a
/// level's generator is built.
#[horizon_mod_api_macros::replaceable]
pub fn chunk_shaper_for(
    dimension: &str,
    seed: i64,
) -> Option<Arc<dyn horizon_core::chunk_shape::ChunkShaper>> {
    let _ = (dimension, seed);
    None
}
