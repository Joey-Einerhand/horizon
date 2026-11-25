//! Chunk shapers: what a mod lays over each chunk the generator finishes, before it is lit,
//! saved or sent (a ring world's void outside its band and its rim walls).

/// A column of one block a shaper fills: `height` blocks of `block` (a block id, its default
/// state) from `y` up at absolute block column (`x`, `z`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapedRun<'a> {
    pub x: i32,
    pub z: i32,
    pub y: i32,
    pub height: u32,
    pub block: &'a str,
}

/// What a level's generator lays over each chunk as it finishes it.
pub trait ChunkShaper: Send + Sync + std::fmt::Debug {
    /// The runs to fill in chunk (`chunk_x`, `chunk_z`), each handed to `fill`, later ones over
    /// earlier ones. A run outside the chunk or the level's height is cut to it.
    fn shape(&self, chunk_x: i32, chunk_z: i32, fill: &mut dyn FnMut(ShapedRun<'_>));
}
