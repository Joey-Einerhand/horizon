//! Resource decoding for the pinned Java Edition client.
// Raw trig is fine in this crate: model baking, not simulation; `math::joml_sin` is JOML's `Math.sin` (#18123).
#![allow(clippy::disallowed_methods)]
#![forbid(unsafe_code)]

pub mod animation;
pub mod bake;
pub mod blockstate;
pub mod component_codec;
pub mod item_bake;
pub mod item_model;
mod json;
pub mod math;
pub mod model;
pub mod model_loaders;
pub mod obj;
pub mod palette;
pub mod resolve;
pub mod resource_source;
#[cfg(not(target_arch = "wasm32"))]
pub mod zip;

pub use json::DecodeError;
