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

impl crate::game::Game<'_> {
    /// Keep every entity of a periodic level in the copy of the world its players are in: one
    /// more than half a period from every player moves a whole number of periods to the copy
    /// nearest the nearest one. Such an entity is too far for anyone to see it go, and around a
    /// player everything is then in one copy, so reach, pickup, targeting and pathing need
    /// nothing of their own. Things the level makes at a stored position (a spawner's mob, a
    /// block's drop) join the player's copy here, a tick later.
    pub(in crate::game) fn rebase_periodic_entities(&mut self) {
        use crate::game::commands::Target;
        let Some(period) = self.world.chunks.period_x() else {
            return;
        };
        let period = f64::from(period * 16);
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
        if players.is_empty() {
            return;
        }
        // The whole periods that take `x` nearest the nearest player.
        let shift = |x: f64| {
            players
                .iter()
                .map(|px| {
                    let s = ((px - x) / period).round() * period;
                    (s, (x + s - px).abs())
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map_or(0.0, |(s, _)| s)
        };
        for t in self.level_candidates() {
            if matches!(t, Target::Player | Target::Guest(_)) {
                continue;
            }
            let Some([x, y, z]) = self.target_position(t) else {
                continue;
            };
            let s = shift(x);
            if s != 0.0 {
                self.teleport_to(t, [x + s, y, z], None, false);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn vanilla_levels_do_not_repeat() {
        assert_eq!(super::level_period_chunks("minecraft:overworld", 1), None);
    }
}
