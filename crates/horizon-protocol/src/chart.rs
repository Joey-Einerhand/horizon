//! A player's chart of a periodic level: the copy of the world nearest them.
//!
//! A periodic level (one whose X axis repeats every `period` blocks, such as a ring world)
//! is stored once on the server, at `0 <= x < period`, and its entities live wherever they
//! walked to. A client knows nothing of that: it sees an endless flat level. Every position the
//! server sends one player is moved by a whole number of periods to the copy nearest that
//! player, so the chunks, blocks and entities around them meet without a seam, whichever copy
//! the server names them in.
//!
//! The player's own position (`PlayerPosition`, `MoveVehicle`) is theirs and is never moved:
//! it is where the chart is centred, and the server keeps players where they walked.
//!
//! The chart follows its connection's own packets, so it can never be a packet behind: the
//! level the player is in comes from `Login` and `Respawn` (and [`set_level_period`]'s table
//! of which levels repeat), where they stand from `PlayerPosition` and, as they walk, from
//! the game ([`Chart::set_viewer`]).
use crate::{
    Result,
    buf::Reader,
    packets::play::{Clientbound, PositionPath},
};
use std::sync::{
    RwLock,
    atomic::{AtomicBool, AtomicI32, Ordering},
};

/// The levels of this process's server that repeat along X, by dimension id, with their periods
/// in blocks.
static PERIODS: RwLock<Vec<(String, i32)>> = RwLock::new(Vec::new());
/// Whether any entry of [`PERIODS`] is set: until one is, no packet is looked at.
static ANY: AtomicBool = AtomicBool::new(false);

/// The server opened `dimension`'s level, which repeats every `period` blocks along X (`None`:
/// it doesn't). The period is a multiple of 16 of at least 48 blocks; any other is none.
pub fn set_level_period(dimension: &str, period: Option<i32>) {
    let period = period.filter(|p| *p >= 48 && p % 16 == 0);
    let mut table = PERIODS.write().unwrap_or_else(|e| e.into_inner());
    table.retain(|(d, _)| d != dimension);
    if let Some(p) = period {
        table.push((dimension.to_owned(), p));
    }
    ANY.store(!table.is_empty(), Ordering::Relaxed);
}

/// The period of `dimension`'s level in blocks, 0 when it doesn't repeat.
fn level_period(dimension: &str) -> i32 {
    PERIODS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|(d, _)| d == dimension)
        .map_or(0, |(_, p)| *p)
}

/// One connection's chart: the period of the level its player is in and where they stand.
#[derive(Debug, Default)]
pub struct Chart {
    /// The period along X in blocks; 0: the player's level does not repeat.
    period: AtomicI32,
    /// The player's block X in their own chart.
    viewer_x: AtomicI32,
}

impl Chart {
    /// The player stands at block X `x` (where the server has them: their own chart).
    pub fn set_viewer(&self, x: i32) {
        self.viewer_x.store(x, Ordering::Relaxed);
    }
    /// The block X its player last stood at (`set_viewer`).
    pub fn viewer(&self) -> i32 {
        self.viewer_x.load(Ordering::Relaxed)
    }
    /// Whether packets are looked at: some level of this server repeats.
    pub fn active(&self) -> bool {
        ANY.load(Ordering::Relaxed)
    }
    /// The packet body (VarInt id, then the payload) as this player is sent it: `None` when
    /// it goes as it is (nothing in it is moved, or it names no place).
    pub fn rewrite(&self, body: &[u8]) -> Result<Option<Vec<u8>>> {
        if !self.active() {
            return Ok(None);
        }
        let mut r = Reader::new(body);
        let id = r.varint()?;
        let payload = &body[body.len() - r.remaining()..];
        {
            use crate::ids::play::clientbound::{LOGIN, PLAYER_POSITION, RESPAWN};
            if id == LOGIN || id == RESPAWN || id == PLAYER_POSITION {
                match Clientbound::decode(id, payload)? {
                    Clientbound::Login(p) => self.enter(&p.spawn.dimension),
                    Clientbound::Respawn(p) => self.enter(&p.spawn.dimension),
                    Clientbound::PlayerPosition(p) if p.relatives & RELATIVE_X == 0 => {
                        self.set_viewer(p.position[0].floor() as i32)
                    }
                    _ => {}
                }
                return Ok(None);
            }
        }
        let period = self.period.load(Ordering::Relaxed);
        if period == 0 || !MOVED.contains(&id) {
            return Ok(None);
        }
        let shift = Shift {
            period,
            viewer: self.viewer_x.load(Ordering::Relaxed),
        };
        let mut packet = Clientbound::decode(id, payload)?;
        if !shift.packet(&mut packet) {
            return Ok(None);
        }
        Ok(Some(packet.encode()))
    }
    fn enter(&self, dimension: &str) {
        self.period
            .store(level_period(dimension), Ordering::Relaxed);
    }
}

/// The clientbound play packets that name a place in the level.
const MOVED: &[i32] = {
    use crate::ids::play::clientbound::*;
    &[
        ADD_ENTITY,
        ADD_TRANSIENT_BLOCK,
        BLOCK_DESTRUCTION,
        BLOCK_ENTITY_DATA,
        BLOCK_EVENT,
        BLOCK_UPDATE,
        CHUNKS_BIOMES,
        DAMAGE_EVENT,
        ENTITY_POSITION_SYNC,
        EXPLODE,
        FORGET_LEVEL_CHUNK,
        LEVEL_CHUNK_WITH_LIGHT,
        LEVEL_EVENT,
        LEVEL_PARTICLES,
        LIGHT_UPDATE,
        MOVE_MINECART_ALONG_TRACK,
        OPEN_SIGN_EDITOR,
        PLAYER_LOOK_AT,
        SECTION_BLOCKS_UPDATE,
        SET_CHUNK_CACHE_CENTER,
        SET_DEFAULT_SPAWN_POSITION,
        SOUND,
        TELEPORT_ENTITY,
    ]
};

/// `ClientboundTeleportEntityPacket`'s and `ClientboundPlayerPositionPacket`'s relative X flag.
const RELATIVE_X: i32 = 1;

/// How far to move a place to its copy nearest the viewer.
#[derive(Clone, Copy)]
struct Shift {
    /// In blocks.
    period: i32,
    /// The viewer's block X.
    viewer: i32,
}

impl Shift {
    /// The whole periods (in blocks) that take block X `x` nearest the viewer.
    fn blocks(self, x: i64) -> i64 {
        let p = i64::from(self.period);
        let d = i64::from(self.viewer) - x;
        (d + p / 2).div_euclid(p) * p
    }
    fn block(self, x: &mut i32) -> bool {
        let s = self.blocks(i64::from(*x));
        *x = (i64::from(*x) + s) as i32;
        s != 0
    }
    fn exact(self, x: &mut f64) -> bool {
        let s = self.blocks(x.floor() as i64);
        *x += s as f64;
        s != 0
    }
    /// A chunk (or section) X: moved as its first block is.
    fn chunk(self, cx: &mut i32) -> bool {
        let s = self.blocks(i64::from(*cx) * 16) / 16;
        *cx = (i64::from(*cx) + s) as i32;
        s != 0
    }
    /// `ClientboundSoundPacket`'s X: eighths of a block.
    fn eighths(self, x: &mut i32) -> bool {
        let s = self.blocks(i64::from(*x).div_euclid(8)) * 8;
        *x = (i64::from(*x) + s) as i32;
        s != 0
    }
    fn path(self, path: &mut PositionPath) -> bool {
        match path {
            PositionPath::Linear(p) => self.exact(&mut p[0]),
            PositionPath::Stepped(steps) => {
                // One shift for the whole path, so its steps stay together.
                let Some(first) = steps.first() else {
                    return false;
                };
                let s = self.blocks(first.0[0].floor() as i64) as f64;
                for (p, _) in steps.iter_mut() {
                    p[0] += s;
                }
                s != 0.0
            }
        }
    }
    /// Move every place in `packet`; whether any moved.
    fn packet(self, packet: &mut Clientbound) -> bool {
        match packet {
            Clientbound::AddEntity(p) => self.exact(&mut p.position[0]),
            Clientbound::AddTransientBlock(p) => self.block(&mut p.pos[0]),
            Clientbound::BlockDestruction(p) => self.block(&mut p.pos[0]),
            Clientbound::BlockEntityData(p) => self.block(&mut p.pos[0]),
            Clientbound::BlockEvent(p) => self.block(&mut p.pos[0]),
            Clientbound::BlockUpdate(p) => self.block(&mut p.pos[0]),
            Clientbound::ChunksBiomes(p) => p
                .chunks
                .iter_mut()
                .fold(false, |moved, c| self.chunk(&mut c.pos[0]) | moved),
            Clientbound::DamageEvent(p) => p
                .position
                .as_mut()
                .is_some_and(|pos| self.exact(&mut pos[0])),
            Clientbound::EntityPositionSync(p) => self.path(&mut p.path),
            Clientbound::Explode(p) => self.exact(&mut p.center[0]),
            Clientbound::ForgetLevelChunk(p) => self.chunk(&mut p.pos[0]),
            Clientbound::LevelChunkWithLight(p) => self.chunk(&mut p.x),
            Clientbound::LevelEvent(p) => self.block(&mut p.pos[0]),
            Clientbound::LevelParticles(p) => self.exact(&mut p.pos[0]),
            Clientbound::LightUpdate(p) => self.chunk(&mut p.x),
            Clientbound::MoveMinecartAlongTrack(p) => {
                let Some(first) = p.steps.first() else {
                    return false;
                };
                let s = self.blocks(first.position[0].floor() as i64) as f64;
                for step in &mut p.steps {
                    step.position[0] += s;
                }
                s != 0.0
            }
            Clientbound::OpenSignEditor(p) => self.block(&mut p.pos[0]),
            Clientbound::PlayerLookAt(p) => self.exact(&mut p.target[0]),
            Clientbound::SectionBlocksUpdate(p) => self.chunk(&mut p.section[0]),
            Clientbound::SetChunkCacheCenter(p) => self.chunk(&mut p.x),
            Clientbound::SetDefaultSpawnPosition(p) => self.block(&mut p.pos.pos[0]),
            Clientbound::Sound(p) => self.eighths(&mut p.x),
            Clientbound::TeleportEntity(p) => {
                p.relatives & RELATIVE_X == 0 && self.exact(&mut p.position[0])
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{ChunkData, LightData};
    use crate::packets::play::{BlockUpdate, LevelChunkWithLight, PlayerPosition, TeleportEntity};

    const C: i32 = 16384;

    /// A chart in a level of period `C` (the tests' own dimension, so they don't meet
    /// another test's table entry), its player at `viewer`.
    fn chart(viewer: i32) -> Chart {
        set_level_period("chart_test:ring", Some(C));
        let chart = Chart::default();
        chart.enter("chart_test:ring");
        chart.set_viewer(viewer);
        chart
    }
    fn moved(chart: &Chart, packet: Clientbound) -> Clientbound {
        let body = packet.encode();
        let out = chart.rewrite(&body).unwrap().unwrap_or(body);
        let mut r = Reader::new(&out);
        let id = r.varint().unwrap();
        Clientbound::decode(id, &out[out.len() - r.remaining()..]).unwrap()
    }

    #[test]
    fn a_stored_block_is_sent_at_the_copy_nearest_the_player() {
        // The player walked a little past the seam: block 6 is the one 16390 names.
        let p = moved(
            &chart(16388),
            Clientbound::BlockUpdate(BlockUpdate {
                pos: [6, 64, 2],
                state: 1,
            }),
        );
        assert!(matches!(p, Clientbound::BlockUpdate(b) if b.pos == [16390, 64, 2]));
        // Near the stored copy nothing moves, nor across the seam the other way.
        let p = moved(
            &chart(3),
            Clientbound::BlockUpdate(BlockUpdate {
                pos: [16380, 64, 2],
                state: 1,
            }),
        );
        assert!(matches!(p, Clientbound::BlockUpdate(b) if b.pos == [-4, 64, 2]));
        let body = Clientbound::BlockUpdate(BlockUpdate {
            pos: [10, 64, 2],
            state: 1,
        })
        .encode();
        assert_eq!(chart(3).rewrite(&body).unwrap(), None);
    }

    #[test]
    fn chunks_move_by_whole_columns() {
        let p = moved(
            &chart(16400),
            Clientbound::LevelChunkWithLight(LevelChunkWithLight {
                x: 1,
                z: 5,
                chunk: ChunkData {
                    heightmaps: vec![(4, vec![0; 37])],
                    buffer: vec![0; 8],
                    block_entities: vec![],
                },
                light: LightData {
                    sky_mask: vec![1],
                    block_mask: vec![],
                    empty_sky_mask: vec![],
                    empty_block_mask: vec![2],
                    sky: vec![vec![0xFF; 2048]],
                    block: vec![],
                },
            }),
        );
        assert!(matches!(p, Clientbound::LevelChunkWithLight(c) if (c.x, c.z) == (1025, 5)));
    }

    #[test]
    fn the_players_own_position_and_relative_moves_stay() {
        let own = Clientbound::PlayerPosition(PlayerPosition {
            teleport: 1,
            position: [5.0, 64.0, 0.0],
            delta: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            relatives: 0,
        });
        assert_eq!(chart(16388).rewrite(&own.encode()).unwrap(), None);
        let relative = Clientbound::TeleportEntity(TeleportEntity {
            id: 3,
            position: [1.0, 0.0, 0.0],
            delta: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            relatives: RELATIVE_X,
            on_ground: false,
        });
        assert_eq!(chart(16388).rewrite(&relative.encode()).unwrap(), None);
    }

    #[test]
    fn a_level_without_a_period_sends_everything_as_it_is() {
        set_level_period("chart_test:flat", None);
        let chart = Chart::default();
        chart.enter("chart_test:flat");
        chart.set_viewer(16388);
        let body = Clientbound::BlockUpdate(BlockUpdate {
            pos: [6, 64, 2],
            state: 1,
        })
        .encode();
        assert_eq!(chart.rewrite(&body).unwrap(), None);
    }
}
