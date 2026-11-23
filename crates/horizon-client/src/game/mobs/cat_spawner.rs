//! `CatSpawner`: every 1200 ticks a cat may appear near a random player,
//! in a village with enough occupied beds or inside a swamp hut.
use super::Game;
use super::global_tick::SpawnerKind;
use horizon_gameplay::cat_spawner::CatLevel;

pub(super) use horizon_gameplay::cat_spawner::CatSpawner;

impl Game<'_> {
    /// `CatSpawner.tick` (`horizon_gameplay::cat_spawner`) on the game level.
    pub(super) fn tick_cat_spawner(&mut self) {
        let mut cat = std::mem::take(&mut self.creatures.cat);
        if self.global_mode() {
            // The countdown is the level's, counted here once; a region makes the attempt.
            if self.dim == super::Dim::Overworld && cat.due() {
                let _ = self.grant_spawn(SpawnerKind::Cat);
            }
        } else {
            cat.tick(self);
        }
        self.creatures.cat = cat;
    }
    /// `spawnCat`: a natural spawn, then `snapTo(spawnPos, 0.0F, 0.0F)`, so
    /// the cat faces yaw 0 and its facing takes no draw.
    fn spawn_cat(&mut self, pos: [i32; 3], persistent: bool) {
        horizon_gameplay::cat_spawner::spawn_cat(self, pos, persistent);
    }
}

impl CatLevel for Game<'_> {
    fn spawn_cat(&mut self, pos: [i32; 3], persistent: bool) {
        Game::spawn_cat(self, pos, persistent);
    }
}

#[cfg(test)]
#[path = "tests/cat_spawner_tests.rs"]
mod cat_spawner_tests;
