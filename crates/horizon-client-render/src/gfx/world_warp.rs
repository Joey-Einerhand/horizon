//! A world warp: the camera-relative bend the world shaders give every world vertex before the
//! projection, and the envelope the CPU culls a box by once it is bent.
//!
//! The one kind is a cylinder whose axis runs along Z above the camera: a block `x` blocks
//! along X from the camera turns `x / radius` radians about the axis, as on the inside of a
//! ring (a ringworld mod's world). `eye_radius` is the camera's distance from the axis. The
//! world's own coordinates, collision and simulation are untouched: only what is drawn bends.
//! With no warp, every shader and test behaves as before.

use glam::Vec3;

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
        let s = self.eye_radius - p.y;
        Vec3::new(s * angle.sin(), self.eye_radius - s * angle.cos(), p.z)
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

#[cfg(test)]
mod tests {
    use super::*;

    // A 16384-block ring seen from 64 blocks above its reference surface.
    const RING: WorldWarp = WorldWarp {
        radius: 2607.594,
        eye_radius: 2607.594,
    };

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
