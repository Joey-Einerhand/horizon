// Frame generation's motion vectors for what moves on its own (mobs, dropped items): each
// vertex drawn where the world pass drew it, with where it was last frame; only the pixels
// the world shows it at keep its vector: where the depth is its own, or, for translucent
// items (which don't write depth), where its texture shows and nothing is in front.

struct Params {
    // The world's projection this frame (TAA's jitter in it), the same without the jitter,
    // and last frame's projection from this frame's camera.
    jittered: mat4x4<f32>,
    current: mat4x4<f32>,
    previous: mat4x4<f32>,
    // The world warp (`Globals.world_warp`): moving things are drawn bent, so their motion is
    // measured bent too, with this frame's warp for both positions.
    world_warp: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
// A depth view bound as unfilterable float: naga's GLSL backend cannot
// `textureLoad` a depth texture, and the texel is the same value.
@group(0) @binding(1) var depth: texture_2d<f32>;
@group(0) @binding(2) var atlas: texture_2d<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) current: vec4<f32>,
    @location(1) previous: vec4<f32>,
    @location(2) uv: vec2<f32>,
    // TRANSLUCENT, NEW.
    @location(3) @interpolate(flat) flags: u32,
};

// A translucent item, whose texture says where it shows.
const TRANSLUCENT: u32 = 1u;
// Nothing to match last frame: drawn as w = 1 only.
const NEW: u32 = 2u;

// Motion is measured where things are drawn: the world warp's bend (gfx/world_warp.rs; terrain.wgsl's `world_warp_point`).
// x 0 is no warp.
fn motion_warp(p: vec3<f32>, w: vec4<f32>) -> vec3<f32> {
    if w.x == 0.0 { return p; }
    let a = p.x / w.y;
    let s = w.z - p.y;
    return vec3(s * sin(a), w.z - s * cos(a), p.z);
}
@vertex fn vs_motion(@location(0) position: vec3<f32>, @location(1) previous: vec3<f32>,
    @location(2) uv: vec2<f32>, @location(3) flags: u32) -> Out {
    var out: Out;
    out.uv = uv;
    out.flags = flags;
    let now = motion_warp(position, params.world_warp);
    out.clip = params.jittered * vec4(now, 1.0);
    out.current = params.current * vec4(now, 1.0);
    out.previous = params.previous * vec4(motion_warp(previous, params.world_warp), 1.0);
    return out;
}

// xy: the motion vector in UV (to where the pixel was last frame, FSR's convention), z: 1;
// w: 1 alone for NEW.
@fragment fn fs_motion(in: Out) -> @location(0) vec4<f32> {
    let stored = textureLoad(depth, vec2<i32>(in.clip.xy), 0).r;
    let tolerance = stored * 1e-3 + 1e-7;
    // Not what the world shows here: a transparent texel, or behind something.
    if (in.flags & TRANSLUCENT) != 0u {
        let alpha = textureSampleLevel(atlas, atlas_sampler, in.uv, 0.0).a;
        if alpha < 0.1 || in.clip.z < stored - tolerance {
            discard;
        }
    } else if abs(in.clip.z - stored) > tolerance {
        discard;
    }
    if (in.flags & NEW) != 0u {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    if in.previous.w <= 0.0 {
        discard;
    }
    let now = in.current.xy / in.current.w;
    let then = in.previous.xy / in.previous.w;
    return vec4((then - now) * vec2(0.5, -0.5), 1.0, 0.0);
}
