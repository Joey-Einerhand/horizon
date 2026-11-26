//! Periodic levels: a level whose X axis repeats, such as a ring world's.
//!
//! The server stores such a level once, at `0 <= x < period`. A chunk, block or block entity
//! named at any other X is the stored one `x.rem_euclid(period)` away, so walking, flowing and
//! wiring across the seam need nothing of their own. Entities keep whichever copy of the world
//! they are in, and each player's connection sees the copy nearest that player
//! (`horizon_protocol::chart`).

/// How far `dimension`'s level repeats along X, in blocks: never, in vanilla. A mod whose world
/// closes on itself, such as a ringworld, replaces this with a `[[native.mixin]]`. The answer
/// must be a multiple of 16 of at least 48 blocks; any other answer is taken as no period.
/// The server asks once, when it opens the level.
pub fn level_period_x(dimension: &str, seed: i64) -> Option<i32> {
    let _ = (dimension, seed);
    None
}

/// [`level_period_x`] in chunks, when it is a usable period. The server's connections are
/// told (`horizon_protocol::chart::set_level_period`), so each sends its player the copy of
/// the level nearest them.
pub(in crate::game) fn level_period_chunks(dimension: &str, seed: i64) -> Option<i32> {
    let blocks = level_period_x(dimension, seed).filter(|blocks| *blocks >= 48 && blocks % 16 == 0);
    horizon_protocol::chart::set_level_period(dimension, blocks);
    blocks.map(|blocks| blocks / 16)
}

/// Where the players of a periodic level stand along X: what puts a thing kept at its stored
/// position (a block entity) or wandered off (an entity) in the copy of the world they are in.
pub(in crate::game) struct PeriodicView {
    period: f64,
    players: Vec<f64>,
}

impl PeriodicView {
    /// The whole periods that take `x` nearest the nearest player.
    pub(in crate::game) fn shift(&self, x: f64) -> f64 {
        self.players
            .iter()
            .map(|px| {
                let s = ((px - x) / self.period).round() * self.period;
                (s, (x + s - px).abs())
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(0.0, |(s, _)| s)
    }
    /// The players' X.
    pub(in crate::game) fn players(&self) -> &[f64] {
        &self.players
    }
    /// Block `pos` in the copy nearest the nearest player.
    pub(in crate::game) fn block(&self, pos: (i32, i32, i32)) -> (i32, i32, i32) {
        let s = self.shift(f64::from(pos.0) + 0.5);
        (pos.0 + s as i32, pos.1, pos.2)
    }
}

impl crate::game::Game<'_> {
    /// The level's period and its players' X, when it repeats and has players.
    pub(in crate::game) fn periodic_view(&self) -> Option<PeriodicView> {
        let period = f64::from(self.world.chunks.period_x()? * 16);
        let mut players: Vec<f64> = self.player.iter().map(|p| p.position.x).collect();
        if let Some(host) = self.lan.as_ref() {
            players.extend(
                host.hub
                    .guests
                    .simulation_iter()
                    .filter(|g| host.hub.watches(g))
                    .map(|g| g.position[0]),
            );
        }
        (!players.is_empty()).then_some(PeriodicView { period, players })
    }

    /// Keep every entity of a periodic level in the copy of the world its players are in: one
    /// more than half a period from every player moves a whole number of periods to the copy
    /// nearest the nearest one. Such an entity is too far for anyone to see it go, and around a
    /// player everything is then in one copy, so reach, pickup, targeting and pathing need
    /// nothing of their own. Things the level makes at a stored position (a spawner's mob, a
    /// block's drop) join the player's copy here, a tick later.
    pub(in crate::game) fn rebase_periodic_entities(&mut self) {
        use crate::game::commands::Target;
        let Some(view) = self.periodic_view() else {
            return;
        };
        for t in self.level_candidates() {
            if matches!(t, Target::Player | Target::Guest(_)) {
                continue;
            }
            let Some([x, y, z]) = self.target_position(t) else {
                continue;
            };
            let s = view.shift(x);
            if s == 0.0 {
                continue;
            }
            match t {
                // Moved, not teleported: a teleport makes a mob leave the boat or minecart it
                // rides. What it remembers by block goes with it (a bee a period from its hive
                // drops it), and its path, a period off, is given up.
                Target::Mob(id) => {
                    if let Some(mob) = self.mobs.iter_mut().find(|m| m.id == id) {
                        mob.position.x += s;
                        mob.previous.x += s;
                        mob.ai.nav.stop();
                        mob.needs_sync = true;
                        shift_remembered(mob, s as i32);
                    }
                }
                _ => self.teleport_to(t, [x + s, y, z], None, false),
            }
        }
        // A knot is held to its fence, which a teleport doesn't move: the knots and the fences
        // mobs are leashed to move by the same whole periods, or a mob's leash snapped a period
        // from its knot (and the knot, holding nothing, went).
        let shift = |x: i32| view.shift(f64::from(x) + 0.5) as i32;
        self.resident_entities
            .knots
            .for_each_mut(|knot| knot.fence[0] += shift(knot.fence[0]));
        for mob in self.mobs.iter_mut() {
            if let Some(horizon_client::mobs::leash::Holder::Knot(fence)) = &mut mob.extra.leash {
                fence[0] += shift(fence[0]);
            }
        }
        self.boats.for_each_mut(|boat| {
            if let Some(horizon_client::mobs::leash::Holder::Knot(fence)) = &mut boat.leash {
                fence[0] += shift(fence[0]);
            }
        });
    }
}

/// The blocks `mob` remembers, moved `dx` along X with it: its home, claimed bed and meeting
/// bell, a copper golem's opened chest, the jukebox it dances to, the repellent it shuns, a bee's
/// hive and flower, a villager's bed it lies in and an allay's jukebox.
fn shift_remembered(mob: &mut horizon_client::mobs::Mob, dx: i32) {
    let moved = |pos: &mut Option<[i32; 3]>| {
        if let Some(pos) = pos {
            pos[0] += dx;
        }
    };
    let extra = &mut mob.extra;
    for pos in [
        &mut extra.home,
        &mut extra.opened_chest,
        &mut extra.bed,
        &mut extra.meeting,
        &mut extra.party_jukebox,
        &mut extra.repellent,
    ] {
        moved(pos);
    }
    if let Some(bee) = extra.bee.as_mut() {
        moved(&mut bee.hive);
        moved(&mut bee.flower);
    }
    if let Some(villager) = extra.villager.as_mut() {
        moved(&mut villager.sleeping_pos);
    }
    if let Some(allay) = extra.allay.as_mut() {
        moved(&mut allay.jukebox);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_stored_block_ticks_in_the_copy_its_players_are_in() {
        // A 64-block period; a spawner stored at x 61 beside a player at x 2 is at x -3.
        let view = super::PeriodicView {
            period: 64.0,
            players: vec![2.0],
        };
        assert_eq!(view.block((61, 70, 5)), (-3, 70, 5));
        assert_eq!(view.block((10, 70, 5)), (10, 70, 5));
        // The nearest of two players a lap apart takes it.
        let view = super::PeriodicView {
            period: 64.0,
            players: vec![2.0, 124.0],
        };
        assert_eq!(view.block((61, 70, 5)), (125, 70, 5));
    }

    #[test]
    fn vanilla_levels_do_not_repeat() {
        assert_eq!(super::level_period_chunks("minecraft:overworld", 1), None);
    }
}
