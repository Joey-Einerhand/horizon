//! Far terrain of a periodic level (`ChunkMap::set_period_x`): the level stores each column once,
//! at `0 <= x < period`, but far terrain is drawn where the player is, which past the seam is
//! another copy. Every request goes to the stored copy and comes back where it was asked.
//!
//! Terrain noise may repeat along X (a mod's coordinate map), but decoration and structure
//! placement are seeded by real chunk positions: asked at a copy, the generator decorated chunks
//! that no real chunk matches (villages and ruined portals in the far view that were gone once the
//! real chunks loaded), and saved chunks, kept at their stored columns, were never read there.
use super::{LodDetail, LodSampler, LodSource};
use crate::fast_hash::{FxHashMap, FxHashSet};
use crate::lod::{Column, Surface};
use std::sync::{
    Arc,
    atomic::{AtomicI32, Ordering},
};

/// The most stored chunks a sampler remembers the copies of (`PeriodicSampler::asked`): each detail
/// request replaces the last, so what one left out is never answered.
const ASKED: usize = 8192;

/// `inner` with every request moved to its stored copy while `period` (chunks, read live) is not 0.
pub(super) struct PeriodicLodSource {
    pub(super) inner: Arc<dyn LodSource>,
    pub(super) period: Arc<AtomicI32>,
}

impl LodSource for PeriodicLodSource {
    fn sampler(self: Arc<Self>) -> Box<dyn LodSampler> {
        Box::new(PeriodicSampler {
            inner: self.inner.clone().sampler(),
            period: self.period.clone(),
            asked: FxHashMap::default(),
            centre_x: 0,
        })
    }
    fn keeps_observed(&self) -> bool {
        self.inner.keeps_observed()
    }
}

struct PeriodicSampler {
    inner: Box<dyn LodSampler>,
    period: Arc<AtomicI32>,
    /// The chunk X of each copy asked for, by stored chunk (`request_detail`).
    asked: FxHashMap<(i32, i32), Vec<i32>>,
    /// The chunk X the last detail request centred on.
    centre_x: i32,
}

impl PeriodicSampler {
    /// The period in chunks, 0 for none.
    fn period(&self) -> i32 {
        self.period.load(Ordering::Relaxed).max(0)
    }
    /// How far block X `x` is from its stored copy: a whole number of periods.
    fn offset(&self, x: i32) -> i32 {
        match self.period() {
            0 => 0,
            p => x.div_euclid(p * 16) * p * 16,
        }
    }
}

impl LodSampler for PeriodicSampler {
    fn fork(&self) -> Option<Box<dyn LodSampler>> {
        Some(Box::new(PeriodicSampler {
            inner: self.inner.fork()?,
            period: self.period.clone(),
            asked: FxHashMap::default(),
            centre_x: self.centre_x,
        }))
    }
    fn request_detail(&mut self, chunks: &[(i32, i32)], center: [i32; 2]) {
        let p = self.period();
        if p == 0 {
            return self.inner.request_detail(chunks, center);
        }
        self.centre_x = center[0].div_euclid(16);
        if self.asked.len() > ASKED {
            self.asked.clear();
        }
        let mut seen = FxHashSet::default();
        let mut stored = Vec::with_capacity(chunks.len());
        for &(cx, cz) in chunks {
            let key = (cx.rem_euclid(p), cz);
            let copies = self.asked.entry(key).or_default();
            if !copies.contains(&cx) {
                copies.push(cx);
            }
            if seen.insert(key) {
                stored.push(key);
            }
        }
        // Nearest first is judged in the stored frame: the centre moves there too.
        self.inner
            .request_detail(&stored, [center[0].rem_euclid(p * 16), center[1]]);
    }
    fn drain_details(&mut self) -> Vec<LodDetail> {
        let details = self.inner.drain_details();
        let p = self.period();
        if p == 0 {
            return details;
        }
        let mut out = Vec::with_capacity(details.len());
        for (key, columns) in details {
            // The copy nearest the player as well: a generation batch also brings neighbours
            // nobody asked for.
            let nearest = key.0 + (self.centre_x - key.0 + p / 2).div_euclid(p) * p;
            let mut copies = self.asked.remove(&key).unwrap_or_default();
            if !copies.contains(&nearest) {
                copies.push(nearest);
            }
            for cx in copies {
                out.push(((cx, key.1), columns.clone()));
            }
        }
        out
    }
    fn observed(&mut self, (cx, cz): (i32, i32), columns: &Arc<[Column; 256]>) {
        let cx = cx - self.offset(cx * 16) / 16;
        self.inner.observed((cx, cz), columns);
    }
    fn set_approximate(&mut self, approximate: bool) {
        self.inner.set_approximate(approximate);
    }
    fn sample(&mut self, x: i32, z: i32) -> Option<Surface> {
        self.inner.sample(x - self.offset(x), z)
    }
    fn column(&mut self, x: i32, z: i32) -> Column {
        self.inner.column(x - self.offset(x), z)
    }
    fn reduced_column(&mut self, x: i32, z: i32, width: i32) -> Column {
        self.inner.reduced_column(x - self.offset(x), z, width)
    }
    fn grid(&mut self, x: i32, z: i32, stride: i32, side: usize) -> Vec<Column> {
        // From the stored copy of its first column: a grid over the seam reads on past the
        // period there, as the copies continue.
        self.inner.grid(x - self.offset(x), z, stride, side)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Records what it was asked and answers each detail request at once.
    struct Recorder {
        log: Arc<Mutex<Vec<String>>>,
        details: Vec<LodDetail>,
    }
    impl LodSampler for Recorder {
        fn request_detail(&mut self, chunks: &[(i32, i32)], center: [i32; 2]) {
            self.log
                .lock()
                .unwrap()
                .push(format!("detail {chunks:?} {center:?}"));
            let empty = Arc::new(std::array::from_fn(|_| Column::default()));
            self.details
                .extend(chunks.iter().map(|&key| (key, empty.clone())));
        }
        fn drain_details(&mut self) -> Vec<LodDetail> {
            std::mem::take(&mut self.details)
        }
        fn sample(&mut self, x: i32, z: i32) -> Option<Surface> {
            self.log.lock().unwrap().push(format!("sample {x} {z}"));
            None
        }
        fn grid(&mut self, x: i32, z: i32, stride: i32, side: usize) -> Vec<Column> {
            self.log
                .lock()
                .unwrap()
                .push(format!("grid {x} {z} {stride} {side}"));
            vec![Column::default(); side * side]
        }
    }
    struct RecorderSource(Arc<Mutex<Vec<String>>>);
    impl LodSource for RecorderSource {
        fn sampler(self: Arc<Self>) -> Box<dyn LodSampler> {
            Box::new(Recorder {
                log: self.0.clone(),
                details: Vec::new(),
            })
        }
    }

    fn sampler(period: i32) -> (Box<dyn LodSampler>, Arc<Mutex<Vec<String>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let source = Arc::new(PeriodicLodSource {
            inner: Arc::new(RecorderSource(log.clone())),
            period: Arc::new(AtomicI32::new(period)),
        });
        (source.sampler(), log)
    }

    #[test]
    fn far_terrain_past_the_seam_is_asked_of_the_stored_copy_and_drawn_where_asked() {
        // A 64-chunk (1024-block) period, the player just west of the seam.
        let (mut sampler, log) = sampler(64);
        sampler.request_detail(&[(-2, 3), (62, 3), (1, 0)], [-40, 50]);
        assert_eq!(log.lock().unwrap()[0], "detail [(62, 3), (1, 0)] [984, 50]");
        let mut drawn: Vec<_> = sampler
            .drain_details()
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        drawn.sort_unstable();
        // Both copies of (62, 3) asked for get it; (1, 0) is drawn where it was asked.
        assert_eq!(drawn, vec![(-2, 3), (1, 0), (62, 3)]);

        sampler.grid(-1040, 16, 4, 16);
        sampler.sample(1030, 5);
        let log = log.lock().unwrap();
        assert_eq!(log[1], "grid 1008 16 4 16");
        assert_eq!(log[2], "sample 6 5");
    }

    #[test]
    fn without_a_period_far_terrain_is_asked_where_it_is_drawn() {
        let (mut sampler, log) = sampler(0);
        sampler.request_detail(&[(-2, 3)], [-40, 50]);
        sampler.grid(-1040, 16, 4, 16);
        assert_eq!(sampler.drain_details()[0].0, (-2, 3));
        let log = log.lock().unwrap();
        assert_eq!(log[0], "detail [(-2, 3)] [-40, 50]");
        assert_eq!(log[1], "grid -1040 16 4 16");
    }
}
