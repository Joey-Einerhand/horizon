//! A world warp: the camera-relative bend the world shaders give every world vertex before the
//! projection, and the envelope the CPU culls a box by once it is bent.
//!
//! The one kind is a cylinder whose axis runs along Z above the camera: a block `x` blocks
//! along X from the camera turns `x / radius` radians about the axis, as on the inside of a
//! ring (a ringworld mod's world). `eye_radius` is the camera's distance from the axis. The
//! world's own coordinates, collision and simulation are untouched: only what is drawn bends.
//! With no warp, every shader and test behaves as before.

use glam::{DMat4, DVec3, Mat4, Vec3, Vec4};

/// A cylinder bend of camera-relative positions (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldWarp {
    /// Blocks along X per radian: the circumference over 2π.
    pub radius: f32,
    /// The camera's distance below the axis, in blocks. Positive: the world curves up,
    /// away from the camera, as inside a ring.
    pub eye_radius: f32,
}

impl WorldWarp {
    /// The warp as `Globals.world_warp`: x 1 with a warp, y its radius, z the eye's radius.
    /// All zero without one, which the shaders read as no warp.
    pub fn uniform(warp: Option<WorldWarp>) -> [f32; 4] {
        match warp {
            Some(w) if w.is_valid() => [1.0, w.radius, w.eye_radius, 0.0],
            _ => [0.0; 4],
        }
    }

    /// Finite, with a radius a shader can divide by.
    pub fn is_valid(&self) -> bool {
        self.radius.is_finite() && self.eye_radius.is_finite() && self.radius >= 1.0
    }

    /// Where the camera-relative point `p` is drawn: terrain.wgsl's `world_warp_point`.
    pub fn point(&self, p: Vec3) -> Vec3 {
        let angle = p.x / self.radius;
        // eye_radius - (eye_radius - y)·cos, without the cancellation that loses y at large radii.
        let half = (angle * 0.5).sin();
        Vec3::new(
            (self.eye_radius - p.y) * angle.sin(),
            2.0 * self.eye_radius * half * half + p.y * angle.cos(),
            p.z,
        )
    }

    /// The smallest box holding every point of the camera-relative box `min..max` once bent.
    ///
    /// A bent point is `s·sin a`, `e − s·cos a` with `s = e − y` and `a = x / radius`: linear in
    /// `s`, so its extremes over the box lie at `s`'s two ends, and over `a` at its ends or at
    /// the quarter turns between them, where sine or cosine peaks.
    pub fn envelope(&self, min: Vec3, max: Vec3) -> (Vec3, Vec3) {
        if !min.is_finite() || !max.is_finite() {
            return (min, max);
        }
        let (a0, a1) = (min.x / self.radius, max.x / self.radius);
        let (s0, s1) = (self.eye_radius - max.y, self.eye_radius - min.y);
        let mut angles = [a0, a1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut count = 2;
        let quarter = std::f32::consts::FRAC_PI_2;
        let first = (a0 / quarter).ceil() as i64;
        let last = (a1 / quarter).floor() as i64;
        // Past a full turn every value is reached: four quarter turns cover them all.
        for k in first..=last.min(first + 7) {
            angles[count] = k as f32 * quarter;
            count += 1;
        }
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for &a in &angles[..count] {
            let (sin, cos) = a.sin_cos();
            for s in [s0, s1] {
                let x = s * sin;
                let y = self.eye_radius - s * cos;
                lo = lo.min(Vec3::new(x, y, min.z));
                hi = hi.max(Vec3::new(x, y, max.z));
            }
        }
        (lo, hi)
    }
}

impl WorldWarp {
    /// Where the camera-relative point drawn at `q` lies in the flat world: `point`'s inverse.
    /// `q` is a bend of a point below the axis (`eye_radius - y > 0`), as every point drawn is.
    pub fn unpoint(&self, q: Vec3) -> Vec3 {
        let c = self.eye_radius - q.y;
        let s = q.x.hypot(c);
        Vec3::new(q.x.atan2(c) * self.radius, self.eye_radius - s, q.z)
    }
}

/// How a pixel drawn this frame finds where it was drawn last frame, for TAA's history and
/// frame generation's motion vectors, when a world warp bends either frame. A flat world's
/// camera motion is one matrix (`world_post::reprojection`); a bend is not linear, so the
/// shaders unbend this frame's point (`unproject`, then `unpoint` with `warp`), move it to the
/// previous camera (`offset`) and bend and project it as the previous frame drew it
/// (`previous_warp`, `previous`). Each shader's copy of this is `warped_reproject`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WarpedMotion {
    /// This frame's unjittered clip space to its camera-relative (bent) positions.
    pub unproject: [[f32; 4]; 4],
    /// The previous frame's unjittered projection, relative to the previous camera.
    pub previous: [[f32; 4]; 4],
    /// xyz: this frame's camera minus the previous frame's.
    pub offset: [f32; 4],
    /// This frame's warp and the previous frame's (`WorldWarp::uniform`).
    pub warp: [f32; 4],
    pub previous_warp: [f32; 4],
}

impl WarpedMotion {
    /// The motion from the previous frame's projection `previous` at `previous_camera` with
    /// `previous_warp` to this frame's `current` at `camera` with `warp`.
    pub fn new(
        (previous, previous_camera, previous_warp): (DMat4, DVec3, Option<WorldWarp>),
        (current, camera, warp): (DMat4, DVec3, Option<WorldWarp>),
    ) -> Self {
        Self {
            unproject: current.inverse().as_mat4().to_cols_array_2d(),
            previous: previous.as_mat4().to_cols_array_2d(),
            offset: (camera - previous_camera).as_vec3().extend(0.0).to_array(),
            warp: WorldWarp::uniform(warp),
            previous_warp: WorldWarp::uniform(previous_warp),
        }
    }

    /// Whether either frame is bent; when neither is, the flat matrix reprojects exactly.
    pub fn is_warped(&self) -> bool {
        self.warp[0] != 0.0 || self.previous_warp[0] != 0.0
    }

    /// The previous frame's clip position of the point this frame draws at `ndc` (x, y in
    /// -1..1) and reversed `depth`: the shaders' `warped_reproject`.
    pub fn reproject(&self, ndc: [f32; 2], depth: f32) -> Vec4 {
        let q = Mat4::from_cols_array_2d(&self.unproject) * Vec4::new(ndc[0], ndc[1], depth, 1.0);
        let flat = unbend(q.truncate() / q.w, self.warp) + Vec4::from(self.offset).truncate();
        Mat4::from_cols_array_2d(&self.previous) * bend(flat, self.previous_warp).extend(1.0)
    }

    /// The previous frame's clip position of a point that was at `previous`, relative to this
    /// frame's camera (frame generation's moving things): `motion.wgsl`'s previous position.
    pub fn previous_clip(&self, previous: Vec3) -> Vec4 {
        let there = previous + Vec4::from(self.offset).truncate();
        Mat4::from_cols_array_2d(&self.previous) * bend(there, self.previous_warp).extend(1.0)
    }
}

/// `WorldWarp::point` from a uniform (`WorldWarp::uniform`): x 0 is no warp.
fn bend(p: Vec3, w: [f32; 4]) -> Vec3 {
    if w[0] == 0.0 {
        return p;
    }
    WorldWarp {
        radius: w[1],
        eye_radius: w[2],
    }
    .point(p)
}

/// `WorldWarp::unpoint` from a uniform.
fn unbend(q: Vec3, w: [f32; 4]) -> Vec3 {
    if w[0] == 0.0 {
        return q;
    }
    WorldWarp {
        radius: w[1],
        eye_radius: w[2],
    }
    .unpoint(q)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 16384-block ring seen from 64 blocks above its reference surface.
    const RING: WorldWarp = WorldWarp {
        radius: 2607.594,
        eye_radius: 2607.594,
    };

    #[test]
    fn unpoint_undoes_point() {
        for p in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(300.0, -20.0, 7.0),
            Vec3::new(-2000.0, 40.0, -3.0),
            Vec3::new(5000.0, -64.0, 100.0),
        ] {
            let back = RING.unpoint(RING.point(p));
            assert!((back - p).length() < 0.05, "{p} came back as {back}");
        }
    }

    /// A block seen last frame from one camera and warp, and this frame from another: its
    /// pixel this frame reprojects to where last frame drew it, and so does a moving thing
    /// that was there. Without a warp the result is the flat matrix's.
    #[test]
    fn warped_motion_finds_where_a_block_was_drawn_last_frame() {
        let projection = DMat4::perspective_rh(1.2, 16.0 / 9.0, 1024.0, 0.05);
        let previous = projection * DMat4::from_rotation_y(0.3);
        let current = projection * DMat4::from_rotation_y(0.35);
        let (previous_camera, camera) = (DVec3::new(10.0, 64.0, 5.0), DVec3::new(13.0, 64.5, 4.0));
        let block = DVec3::new(40.0, 50.0, -60.0);
        let other = WorldWarp {
            radius: 2000.0,
            eye_radius: 2010.0,
        };
        for (then, now) in [
            (Some(RING), Some(RING)),
            (Some(other), Some(RING)),
            (None, Some(RING)),
            (Some(RING), None),
            (None, None),
        ] {
            let drawn = |m: DMat4, eye: DVec3, w: Option<WorldWarp>| {
                let p = (block - eye).as_vec3();
                let q = w.map_or(p, |w| w.point(p));
                m.as_mat4() * q.extend(1.0)
            };
            let here = drawn(current, camera, now);
            let ndc = here.truncate() / here.w;
            let motion =
                WarpedMotion::new((previous, previous_camera, then), (current, camera, now));
            let found = motion.reproject([ndc.x, ndc.y], ndc.z);
            let expected = drawn(previous, previous_camera, then);
            let (found, expected) = (found.truncate() / found.w, expected.truncate() / expected.w);
            assert!(
                (found.truncate() - expected.truncate()).length() < 1e-3,
                "{then:?} -> {now:?}: {found} vs {expected}"
            );
            let moving = motion.previous_clip((block - camera).as_vec3());
            let moving = moving.truncate() / moving.w;
            assert!((moving.truncate() - expected.truncate()).length() < 1e-3);
            assert_eq!(motion.is_warped(), then.is_some() || now.is_some());
        }
    }

    #[test]
    fn the_camera_and_points_straight_below_it_stay_put() {
        assert_eq!(RING.point(Vec3::ZERO), Vec3::ZERO);
        let below = RING.point(Vec3::new(0.0, -20.0, 5.0));
        assert!((below - Vec3::new(0.0, -20.0, 5.0)).length() < 1e-3);
    }

    #[test]
    fn far_ground_rises_as_on_the_inside_of_a_ring() {
        // Half a lap away the ground is overhead: the ring's diameter above the camera's feet.
        let half = std::f32::consts::PI * RING.radius;
        let opposite = RING.point(Vec3::new(half, 0.0, 0.0));
        assert!(
            (opposite.y - 2.0 * RING.eye_radius).abs() < 0.5,
            "{opposite}"
        );
        assert!(opposite.x.abs() < 0.5, "{opposite}");
        // A quarter lap away it is level with the axis, one eye radius up and across.
        let quarter = RING.point(Vec3::new(half / 2.0, 0.0, 0.0));
        assert!((quarter.y - RING.eye_radius).abs() < 0.5, "{quarter}");
        assert!((quarter.x - RING.eye_radius).abs() < 0.5, "{quarter}");
    }

    #[test]
    fn the_envelope_holds_every_bent_point_of_the_box() {
        let boxes = [
            (Vec3::new(100.0, -64.0, -8.0), Vec3::new(116.0, -48.0, 8.0)),
            (
                Vec3::new(-3000.0, -10.0, 0.0),
                Vec3::new(5000.0, 30.0, 16.0),
            ),
            (
                Vec3::new(-20000.0, -64.0, 0.0),
                Vec3::new(20000.0, 300.0, 16.0),
            ),
        ];
        for (min, max) in boxes {
            let (lo, hi) = RING.envelope(min, max);
            for i in 0..=40 {
                for j in 0..=8 {
                    let p = Vec3::new(
                        min.x + (max.x - min.x) * i as f32 / 40.0,
                        min.y + (max.y - min.y) * j as f32 / 8.0,
                        min.z,
                    );
                    let q = RING.point(p);
                    let slack = Vec3::splat(1e-2 * (1.0 + q.abs().max_element()));
                    assert!(
                        q.cmpge(lo - slack).all() && q.cmple(hi + slack).all(),
                        "{p} -> {q} not in {lo}..{hi}"
                    );
                }
            }
        }
    }

    /// A radius far past the world's height still tells heights a block apart: the height is
    /// not `e − (e − y)·cos a`, which rounds `y` away once `e` dwarfs it.
    #[test]
    fn a_huge_radius_keeps_heights_a_block_apart() {
        let warp = WorldWarp {
            radius: 134_217_728.0,
            eye_radius: 134_217_728.0,
        };
        assert_eq!(warp.point(Vec3::new(0.0, 1.0, 0.0)).y, 1.0);
        assert_eq!(warp.point(Vec3::new(0.0, 0.0, 0.0)).y, 0.0);
        let near = warp.point(Vec3::new(16.0, 1.0, 0.0));
        assert!((near.y - 1.0).abs() < 1e-3, "{near}");
    }

    #[test]
    fn no_warp_is_all_zero_in_the_uniform() {
        assert_eq!(WorldWarp::uniform(None), [0.0; 4]);
        let tiny = WorldWarp {
            radius: 0.0,
            eye_radius: 1.0,
        };
        assert_eq!(WorldWarp::uniform(Some(tiny)), [0.0; 4]);
        assert_eq!(WorldWarp::uniform(Some(RING))[0], 1.0);
    }
}
