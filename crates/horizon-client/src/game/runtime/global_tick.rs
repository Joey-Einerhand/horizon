//! The global tick: what a level owns that no chunk owns (docs/server-levels.md#the-global-tick).
//!
//! The clock, the weather, the world border, the game rules, the level data and the level-wide spawners'
//! counters belong to the level, not to a region. With regions on, the global tick holds them, and a region:
//!
//! - **reads a snapshot** ([`LevelState`]) at the start of its tick: one `Arc` per level, swapped under a lock held
//!   for the swap, so a region never waits for the global tick;
//! - **sends messages** ([`Message`]) for what it changes (a `/time`, a `/weather`, a `/gamerule`, a census of its
//!   mobs, whether a spawn it was granted came off), appended to an inbox the global tick empties. The global tick
//!   never waits for a region either: nothing answers a message.
//!
//! A spawner that counts down once per level tick in vanilla counts down once here, then sends the attempt to one
//! region as a *grant* ([`Ticket`]): a region is picked with odds in proportion to its living players (so a
//! player's odds are vanilla's), the region claims the ticket once and runs the rest of vanilla's function with its
//! own players and chunks. Today the server thread's own game holds the global tick's fields; a region's game never
//! counts a level-wide spawner.
use super::Dim;
use super::Game;
use super::region_map::RegionId;
use super::region_tasks::SERVER;
use horizon_client::gamerules::{GameRules, Value};
use horizon_core::fast_hash::FxHashMap;
use horizon_core::random::RandomSource;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Server ticks a grant waits for its region before it is given to another: the region merged, or came home.
const GRANT_TICKS: u64 = 40;
/// Server ticks a region's census is believed: a region that stopped reporting is gone.
const CENSUS_TICKS: u64 = 40;
/// Server ticks between two copies of the level data (level.dat) into the snapshot.
const LEVEL_DATA_TICKS: u64 = 20;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A level-wide spawner whose attempt is for one region (or, for the phantoms, a pass for each).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SpawnerKind {
    Trader,
    Cat,
    Patrol,
    Phantom,
}

/// A grant: the spawner it is for, and the region that may claim it.
#[derive(Clone, Copy, Debug)]
pub(super) struct Ticket {
    pub kind: SpawnerKind,
    pub dim: Dim,
    pub region: RegionId,
    since: u64,
}

/// What a region changed in the level's own state. A region applies it to its copy at once ([`Change::apply`]) and
/// the global tick applies it to the level ([`Change::apply_to_level`]).
#[derive(Clone, Debug)]
pub(super) enum Change {
    ClockTime {
        clock: String,
        total: i64,
    },
    ClockPaused {
        clock: String,
        paused: bool,
    },
    ClockRate {
        clock: String,
        rate: f32,
    },
    ClockPartial {
        clock: String,
    },
    Weather {
        clear: i32,
        time: i32,
        raining: bool,
        thundering: bool,
    },
    Rule {
        name: String,
        value: Value,
    },
}

/// What one region's census holds: its living players, and what vanilla's mob caps count in its chunks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Census {
    pub alive: u32,
    pub chunks: i32,
    pub counts: [i32; 8],
}

/// What a region tells the global tick.
pub(super) enum Message {
    /// `change` at the sender's `seq`-th message.
    Change {
        dim: Dim,
        region: RegionId,
        seq: u64,
        change: Change,
    },
    Census {
        dim: Dim,
        region: RegionId,
        census: Census,
    },
    /// A granted trader attempt put a trader in the level: its chance starts over.
    Spawned { dim: Dim, kind: SpawnerKind },
}

/// The level's siege night: decided by the global tick, run by the region that claims it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Night {
    pub tonight: bool,
    pub claimed_by: Option<RegionId>,
    /// The claimant spawned its last zombie: tonight's siege is over.
    pub finished: bool,
}

impl Night {
    /// A region may run the night.
    pub(super) fn open_to(&self, region: RegionId) -> bool {
        self.tonight && !self.finished && self.claimed_by.is_none_or(|c| c == region)
    }
}

/// What vanilla's mob caps count over every region of a level: the spawnable chunks and the mobs per category.
#[derive(Clone, Debug, Default)]
pub(super) struct Caps {
    pub chunks: i32,
    pub counts: [i32; 8],
    by_region: FxHashMap<RegionId, (i32, [i32; 8])>,
}

impl Caps {
    /// The caps' counts without `region`'s own, as of the census they were summed from.
    pub(super) fn others(&self, region: RegionId) -> (i32, [i32; 8]) {
        let (c, counts) = self.by_region.get(&region).copied().unwrap_or_default();
        let mut out = self.counts;
        for (o, n) in out.iter_mut().zip(counts) {
            *o = (*o - n).max(0);
        }
        ((self.chunks - c).max(0), out)
    }
}

/// What a region reads of its level at the start of its tick. Everything is the global tick's: a region changes
/// none of it except through a [`Message`].
#[derive(Clone)]
pub(super) struct LevelState {
    /// The server tick it was published in.
    pub tick: u64,
    pub game_time: i64,
    pub day_time: i64,
    pub end_clock: i64,
    pub weather: horizon_client::world::Weather,
    pub step: horizon_client::clock::FixedStep,
    pub ticks: u64,
    pub rules: Arc<GameRules>,
    pub difficulty: u8,
    pub difficulty_locked: bool,
    pub border: Arc<horizon_client::worldborder::WorldBorder>,
    pub clock: super::chat::WorldClock,
    pub end_clock_state: super::chat::WorldClock,
    pub command_clocks: Arc<std::collections::BTreeMap<String, (i64, super::chat::WorldClock)>>,
    pub active_clock: Arc<str>,
    /// Level data (level.dat), copied once a second.
    pub level: Arc<horizon_client::save::Level>,
    pub sky_light: bool,
    pub caps: Arc<Caps>,
    /// The last message of each region the global tick has applied.
    pub acked: Arc<FxHashMap<RegionId, u64>>,
}

/// What every game of a server shares of the global tick: the snapshots, the inbox, the tickets and the siege night.
#[derive(Default)]
pub(super) struct GlobalTick {
    levels: RwLock<FxHashMap<Dim, Arc<LevelState>>>,
    inbox: Mutex<Vec<Message>>,
    tickets: Mutex<FxHashMap<u64, Ticket>>,
    next_ticket: AtomicU64,
    nights: Mutex<FxHashMap<Dim, Night>>,
}

impl GlobalTick {
    /// The level's snapshot as last published.
    pub(super) fn snapshot(&self, dim: Dim) -> Option<Arc<LevelState>> {
        self.levels
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&dim)
            .cloned()
    }

    pub(super) fn publish(&self, dim: Dim, state: LevelState) {
        self.levels
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(dim, Arc::new(state));
    }

    /// A region's message for the global tick. Never blocks on it: the inbox is a list the sender appends to.
    pub(super) fn send(&self, message: Message) {
        lock(&self.inbox).push(message);
    }

    /// Everything sent since the last time, oldest first.
    pub(super) fn drain(&self) -> Vec<Message> {
        std::mem::take(&mut *lock(&self.inbox))
    }

    /// A grant of `kind` for `region`, queued by the global tick at server tick `now`.
    pub(super) fn grant(&self, kind: SpawnerKind, dim: Dim, region: RegionId, now: u64) -> u64 {
        let id = self.next_ticket.fetch_add(1, Ordering::Relaxed) + 1;
        lock(&self.tickets).insert(
            id,
            Ticket {
                kind,
                dim,
                region,
                since: now,
            },
        );
        id
    }

    /// The region takes its grant: exactly one caller gets it, and none once it was given to another region.
    pub(super) fn claim(&self, ticket: u64) -> Option<Ticket> {
        lock(&self.tickets).remove(&ticket)
    }

    /// The grants no region claimed in [`GRANT_TICKS`], taken back.
    pub(super) fn expired(&self, now: u64) -> Vec<Ticket> {
        let mut out = Vec::new();
        lock(&self.tickets).retain(|_, t| {
            let stale = now.saturating_sub(t.since) >= GRANT_TICKS;
            if stale {
                out.push(*t);
            }
            !stale
        });
        out
    }

    /// Grants waiting for their region.
    #[cfg(test)]
    pub(super) fn open_grants(&self) -> usize {
        lock(&self.tickets).len()
    }

    pub(super) fn night(&self, dim: Dim) -> Night {
        lock(&self.nights).get(&dim).copied().unwrap_or_default()
    }

    /// The global tick's decision: tonight is a siege night, or is not (which also clears the claim).
    pub(super) fn set_night(&self, dim: Dim, tonight: bool) {
        let mut nights = lock(&self.nights);
        if !tonight {
            nights.remove(&dim);
        } else {
            nights.entry(dim).or_default().tonight = true;
        }
    }

    /// A region sets the night up: true for the first, and for that region again.
    pub(super) fn claim_night(&self, dim: Dim, region: RegionId) -> bool {
        let mut nights = lock(&self.nights);
        let night = nights.entry(dim).or_default();
        if night.open_to(region) {
            night.claimed_by = Some(region);
            return true;
        }
        false
    }

    /// The claimant has no zombies left.
    pub(super) fn finish_night(&self, dim: Dim) {
        if let Some(n) = lock(&self.nights).get_mut(&dim) {
            n.finished = true;
        }
    }
}

/// What only the global tick keeps: the censuses it was sent, and what it has applied.
#[derive(Default)]
pub(super) struct GlobalState {
    census: FxHashMap<(Dim, RegionId), (u64, Census)>,
    acked: FxHashMap<RegionId, u64>,
    /// Messages for a level that is not the one in place; applied at that level's turn.
    deferred: Vec<Message>,
    /// The tick the level data was last copied into a snapshot.
    level_copied: u64,
}

impl GlobalState {
    pub(super) fn record(&mut self, dim: Dim, region: RegionId, census: Census, now: u64) {
        self.census.insert((dim, region), (now, census));
    }

    /// Forgets the regions that stopped reporting.
    pub(super) fn forget_stale(&mut self, now: u64) {
        self.census
            .retain(|_, (at, _)| now.saturating_sub(*at) < CENSUS_TICKS);
    }

    /// The regions of `dim` with a living player, and how many, in region order.
    pub(super) fn alive(&self, dim: Dim) -> Vec<(RegionId, u32)> {
        let mut out: Vec<(RegionId, u32)> = self
            .census
            .iter()
            .filter(|((d, _), (_, c))| *d == dim && c.alive > 0)
            .map(|((_, r), (_, c))| (*r, c.alive))
            .collect();
        out.sort_unstable();
        out
    }

    /// The mob caps' sums over the regions of `dim`.
    pub(super) fn caps(&self, dim: Dim) -> Caps {
        let mut caps = Caps::default();
        for ((d, region), (_, c)) in &self.census {
            if *d != dim {
                continue;
            }
            caps.chunks += c.chunks;
            for (sum, n) in caps.counts.iter_mut().zip(c.counts) {
                *sum += n;
            }
            caps.by_region.insert(*region, (c.chunks, c.counts));
        }
        caps
    }

    /// The caps' counts of every region of `dim` but `region`.
    pub(super) fn others(&self, dim: Dim, region: RegionId) -> (i32, [i32; 8]) {
        self.caps(dim).others(region)
    }
}

/// `ServerLevel.getRandomPlayer` over regions: a region with odds in proportion to its living players, then the
/// region picks one of its own, so each living player has the same odds as in one level. None when nobody lives.
pub(super) fn pick_region(
    alive: &[(RegionId, u32)],
    random: &mut impl RandomSource,
) -> Option<RegionId> {
    let total: u32 = alive.iter().map(|(_, n)| n).sum();
    if total == 0 {
        return None;
    }
    let mut at = random.next_int_bound(total as i32).unwrap_or(0) as u32;
    for &(region, n) in alive {
        if at < n {
            return Some(region);
        }
        at -= n;
    }
    alive.last().map(|&(r, _)| r)
}

impl Change {
    /// The change in a region's copy of the level: what the command wrote there, and again on a snapshot that
    /// does not hold it yet.
    pub(super) fn apply(&self, g: &mut Game<'_>) {
        match self {
            Change::ClockTime { clock, total } => g.set_command_clock_time(clock, *total),
            Change::ClockPaused { clock, paused } => g.command_clock_state(clock).paused = *paused,
            Change::ClockRate { clock, rate } => g.command_clock_state(clock).rate = *rate,
            Change::ClockPartial { clock } => g.command_clock_state(clock).partial = 0.0,
            Change::Weather {
                clear,
                time,
                raining,
                thundering,
            } => {
                let w = &mut g.world.weather;
                w.clear_time = *clear;
                w.rain_time = *time;
                w.thunder_time = *time;
                w.raining = *raining;
                w.thundering = *thundering;
            }
            Change::Rule { name, value } => {
                g.ui.rules.set(name, *value);
                g.sync_rules();
            }
        }
    }

    /// The change in the level itself, on the global tick: through the paths the commands take with no regions,
    /// so the mods' hooks and the dedicated server's events fire once.
    fn apply_to_level(&self, g: &mut Game<'_>) {
        match self {
            Change::Weather {
                clear,
                time,
                raining,
                thundering,
            } => g.set_weather_parameters(*clear, *time, *raining, *thundering),
            Change::Rule { name, value } => {
                use horizon_gameplay::commands::gamerule::GameRuleHost;
                if let Some(rule) = g.find_rule(name) {
                    g.set_game_rule(&rule, *value, None);
                }
            }
            other => other.apply(g),
        }
    }
}

impl LevelState {
    /// The level in place as regions will read it, reusing what did not change from `prev`.
    fn of(
        g: &Game<'_>,
        prev: Option<&LevelState>,
        now: u64,
        caps: Caps,
        acked: &FxHashMap<RegionId, u64>,
        copy_level: bool,
    ) -> Self {
        let rules = match prev {
            Some(p) if *p.rules == g.ui.rules => Arc::clone(&p.rules),
            _ => Arc::new(g.ui.rules.clone()),
        };
        let border = match prev {
            Some(p) if *p.border == g.border => Arc::clone(&p.border),
            _ => Arc::new(g.border.clone()),
        };
        let level = match prev {
            Some(p) if !copy_level => Arc::clone(&p.level),
            _ => Arc::new(g.level.clone()),
        };
        Self {
            tick: now,
            game_time: g.world.game_time,
            day_time: g.world.day_time,
            end_clock: g.world.end_clock,
            weather: g.world.weather,
            step: g.clock.clone(),
            ticks: g.ticks,
            rules,
            difficulty: g.ui.difficulty,
            difficulty_locked: g.level.difficulty_locked,
            border,
            clock: g.ui.clock,
            end_clock_state: g.ui.end_clock,
            command_clocks: Arc::new(g.ui.command_clocks.clone()),
            active_clock: Arc::from(g.ui.active_clock.as_str()),
            level,
            sky_light: g.streaming.sky_light,
            caps: Arc::new(caps),
            acked: Arc::new(acked.clone()),
        }
    }
}

/// The global tick's fields of a game, and what a region does with them.
impl Game<'_> {
    /// This game is a region's: it reads the global tick and never ticks a level-wide counter.
    pub(super) fn region_id(&self) -> Option<RegionId> {
        self.regions.link.as_ref().map(|l| l.id)
    }

    /// The level in place has regions out on their threads: the server's own tick is the global tick for it, and
    /// level-wide spawners are granted to regions instead of being run here.
    pub(super) fn global_mode(&self) -> bool {
        if self.regions.worker {
            return false;
        }
        self.level_has_residents(self.dim)
    }

    /// What the global tick does at the start of the level's pass on the server's thread: it takes in what the
    /// regions sent (applying their changes to the level), asks again for the grants nobody claimed, and counts the
    /// siege night. Never waits for a region.
    pub(super) fn tick_global(&mut self) {
        if self.regions.worker {
            return;
        }
        let global = Arc::clone(&self.regions.global);
        let now = self.regions.tasks.now();
        let dim = self.dim;
        let mut messages = std::mem::take(&mut self.regions.gstate.deferred);
        messages.extend(global.drain());
        for message in messages {
            match message {
                Message::Census {
                    dim: d,
                    region,
                    census,
                } => self.regions.gstate.record(d, region, census, now),
                Message::Change {
                    dim: d,
                    region,
                    seq,
                    change,
                } if d == dim => {
                    change.apply_to_level(self);
                    let acked = self.regions.gstate.acked.entry(region).or_default();
                    *acked = (*acked).max(seq);
                }
                Message::Spawned { dim: d, kind } if d == dim => {
                    if kind == SpawnerKind::Trader {
                        self.creatures.trader.spawned();
                    }
                }
                other => self.regions.gstate.deferred.push(other),
            }
        }
        self.regions.gstate.forget_stale(now);
        for ticket in global.expired(now) {
            // Its region never came for it: another gets it. A phantom pass is not asked again, the next one is
            // near.
            if ticket.dim == dim && ticket.kind != SpawnerKind::Phantom {
                self.grant_spawn(ticket.kind);
            } else if ticket.dim != dim {
                global.grant(ticket.kind, ticket.dim, ticket.region, now);
            }
        }
    }

    /// Publishes the level in place for its regions to read at the start of their next tick.
    pub(super) fn publish_global(&mut self) {
        if self.regions.worker {
            return;
        }
        let global = Arc::clone(&self.regions.global);
        let now = self.regions.tasks.now();
        let dim = self.dim;
        let prev = global.snapshot(dim);
        let copy_level = prev.is_none()
            || now.saturating_sub(self.regions.gstate.level_copied) >= LEVEL_DATA_TICKS;
        if copy_level {
            self.regions.gstate.level_copied = now;
        }
        let caps = self.regions.gstate.caps(dim);
        let state = LevelState::of(
            self,
            prev.as_deref(),
            now,
            caps,
            &self.regions.gstate.acked,
            copy_level,
        );
        global.publish(dim, state);
    }

    /// A region reads the global tick's snapshot of its level at the start of its tick: the clock, the weather, the
    /// border, the rules and the level data. What it changed itself and the global tick has not applied yet is
    /// applied again on top, so its own players never see the old value come back.
    pub(super) fn read_global(&mut self) {
        let Some(id) = self.region_id() else {
            return;
        };
        let Some(s) = self.regions.global.snapshot(self.dim) else {
            return;
        };
        if self.regions.global_seen == Some(s.tick) && self.regions.unacked.is_empty() {
            return;
        }
        self.regions.global_seen = Some(s.tick);
        self.world.game_time = s.game_time;
        self.world.day_time = s.day_time;
        self.world.end_clock = s.end_clock;
        self.world.weather = s.weather;
        self.clock.clone_from(&s.step);
        self.ticks = s.ticks;
        if self.ui.rules != *s.rules {
            self.ui.rules.clone_from(&s.rules);
            self.sync_rules();
        }
        self.ui.difficulty = s.difficulty;
        if self.border != *s.border {
            self.border.clone_from(&s.border);
        }
        self.ui.clock = s.clock;
        self.ui.end_clock = s.end_clock_state;
        self.ui.command_clocks.clone_from(&s.command_clocks);
        if *self.ui.active_clock != *s.active_clock {
            self.ui.active_clock = s.active_clock.to_string();
        }
        if self
            .regions
            .level_seen
            .as_ref()
            .is_none_or(|l| !Arc::ptr_eq(l, &s.level))
        {
            self.level.clone_from(&s.level);
            self.regions.level_seen = Some(Arc::clone(&s.level));
        }
        self.level.difficulty_locked = s.difficulty_locked;
        self.streaming.sky_light = s.sky_light;
        let acked = s.acked.get(&id).copied().unwrap_or(0);
        self.regions.unacked.retain(|(seq, _)| *seq > acked);
        let again = self.regions.unacked.clone();
        for (_, change) in &again {
            change.apply(self);
        }
    }

    /// A command in a region changed the level's own state: the region keeps its copy, and tells the global tick.
    /// On the server's own game (regions off, or the level whole) the command already changed the level.
    pub(super) fn global_change(&mut self, change: Change) {
        let Some(region) = self.region_id() else {
            return;
        };
        self.regions.change_seq += 1;
        let seq = self.regions.change_seq;
        self.regions.global.send(Message::Change {
            dim: self.dim,
            region,
            seq,
            change: change.clone(),
        });
        self.regions.unacked.push((seq, change));
    }

    /// The end of a region's tick: its living players and what the mob caps count in its chunks.
    pub(super) fn send_census(&mut self) {
        let Some(region) = self.region_id() else {
            return;
        };
        let alive = self
            .level_players_iter()
            .filter(|p| p.alive && !p.spectator)
            .count() as u32;
        let (chunks, counts) = self.regions.last_caps.take().unwrap_or_default();
        self.regions.global.send(Message::Census {
            dim: self.dim,
            region,
            census: Census {
                alive,
                chunks,
                counts,
            },
        });
    }

    /// The mob caps' sums over the level without this game's own chunks, for vanilla's `spawnable chunks × limit ÷
    /// 289` against the level's count. None when the level is not in regions.
    pub(super) fn level_wide_caps(&self) -> Option<(i32, [i32; 8])> {
        if let Some(id) = self.region_id() {
            let s = self.regions.global.snapshot(self.dim)?;
            return Some(s.caps.others(id));
        }
        self.global_mode()
            .then(|| self.regions.gstate.others(self.dim, SERVER))
    }

    /// The server's own game takes part in the census of a level it shares with regions: the players and chunks it
    /// still holds.
    pub(super) fn record_home_census(&mut self) {
        if !self.global_mode() {
            return;
        }
        let alive = self
            .level_players_iter()
            .filter(|p| p.alive && !p.spectator)
            .count() as u32;
        let (chunks, counts) = self.regions.last_caps.take().unwrap_or_default();
        let now = self.regions.tasks.now();
        let dim = self.dim;
        self.regions.gstate.record(
            dim,
            SERVER,
            Census {
                alive,
                chunks,
                counts,
            },
            now,
        );
    }

    /// Where a level-wide attempt goes: a region, picked with odds in proportion to its living players, the
    /// server's own game among them. None when nobody is alive.
    fn pick_spawn_target(&mut self) -> Option<RegionId> {
        let mut alive = self.regions.gstate.alive(self.dim);
        alive.retain(|(r, _)| *r != SERVER);
        let home = self.level_players_iter().filter(|p| p.alive).count() as u32;
        if home > 0 {
            alive.insert(0, (SERVER, home));
        }
        pick_region(&alive, &mut self.world.random)
    }

    /// A spawner's countdown ran out on the global tick: its attempt goes to one region. True when the attempt was
    /// run here, on the server's own game, and what it came to (a trader's chance starts over when it spawned).
    /// `None` when it was sent to a region, or nobody is alive to spawn near (vanilla's `spawn` then counts as
    /// spawned, for the trader).
    pub(super) fn grant_spawn(&mut self, kind: SpawnerKind) -> Option<bool> {
        let Some(target) = self.pick_spawn_target() else {
            return (kind == SpawnerKind::Trader).then_some(true);
        };
        if target == SERVER {
            return Some(self.run_spawn_attempt(kind));
        }
        self.send_grant(kind, target);
        None
    }

    fn send_grant(&mut self, kind: SpawnerKind, region: RegionId) {
        let global = Arc::clone(&self.regions.global);
        let now = self.regions.tasks.now();
        let dim = self.dim;
        let ticket = global.grant(kind, dim, region, now);
        self.regions.tasks.region(region, move |g| {
            if global.claim(ticket).is_some() {
                let spawned = g.run_spawn_attempt(kind);
                if spawned && kind == SpawnerKind::Trader {
                    global.send(Message::Spawned { dim, kind });
                }
            }
        });
    }

    /// The phantoms' countdown ran out: every region with a living player makes its pass over them.
    pub(super) fn grant_phantom_passes(&mut self) {
        let mut regions = self.regions.gstate.alive(self.dim);
        regions.retain(|(r, _)| *r != SERVER);
        for (region, _) in regions {
            self.send_grant(SpawnerKind::Phantom, region);
        }
        self.spawn_phantom_pass();
    }

    /// The rest of vanilla's spawner function, here, for the players and chunks this game holds. True when a
    /// granted trader attempt spawned (or had nobody to spawn near).
    pub(super) fn run_spawn_attempt(&mut self, kind: SpawnerKind) -> bool {
        match kind {
            SpawnerKind::Trader => self.spawn_trader_attempt(),
            SpawnerKind::Cat => {
                horizon_gameplay::cat_spawner::attempt(self);
                false
            }
            SpawnerKind::Patrol => {
                self.patrol_attempt();
                false
            }
            SpawnerKind::Phantom => {
                self.spawn_phantom_pass();
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_client::patrol::PatrolSpawner;
    use horizon_core::random::{LegacyRandom, LevelRandom};
    use horizon_gameplay::{cat_spawner::CatSpawner, wandering_trader::Spawner};

    fn level_random(seed: i64) -> LevelRandom {
        LevelRandom::for_level(LegacyRandom::new(seed))
    }

    /// Two hundred regions, one living player each, ticked for `ticks` level ticks by the global tick's own
    /// counters; every grant is claimed by the region it was sent to. Counts the attempts that were due and the
    /// grants each region ran.
    struct Run {
        trader_due: u32,
        patrol_due: u32,
        cat_due: u32,
        ran: FxHashMap<RegionId, u32>,
        claimed_twice: u32,
    }

    fn two_hundred_regions(ticks: u64, seed: i64) -> Run {
        let dim = Dim::Overworld;
        let global = GlobalTick::default();
        let mut state = GlobalState::default();
        for region in 1..=200u64 {
            state.record(
                dim,
                region,
                Census {
                    alive: 1,
                    ..Census::default()
                },
                0,
            );
        }
        let mut random = level_random(seed);
        let mut trader = Spawner::default();
        let mut patrol = PatrolSpawner::default();
        let mut cat = CatSpawner::default();
        let mut run = Run {
            trader_due: 0,
            patrol_due: 0,
            cat_due: 0,
            ran: FxHashMap::default(),
            claimed_twice: 0,
        };
        let mut sent: Vec<(u64, RegionId)> = Vec::new();
        for now in 1..=ticks {
            // The regions keep reporting; the counters below are the global tick's.
            if now % 20 == 0 {
                state.forget_stale(now);
                for region in 1..=200u64 {
                    state.record(
                        dim,
                        region,
                        Census {
                            alive: 1,
                            ..Census::default()
                        },
                        now,
                    );
                }
            }
            let mut due = Vec::new();
            if trader.attempt_due(&mut random) {
                run.trader_due += 1;
                due.push(SpawnerKind::Trader);
            }
            if patrol.due(true, true, || true, &mut random) {
                run.patrol_due += 1;
                due.push(SpawnerKind::Patrol);
            }
            if cat.due() {
                run.cat_due += 1;
                due.push(SpawnerKind::Cat);
            }
            for kind in due {
                let target =
                    pick_region(&state.alive(dim), &mut random).expect("players are alive");
                sent.push((global.grant(kind, dim, target, now), target));
            }
            // Every region ticks and claims what is for it, twice over: only one claim of a ticket counts.
            for (ticket, region) in sent.drain(..) {
                for _ in 0..2 {
                    match global.claim(ticket) {
                        Some(t) => {
                            assert_eq!(t.region, region);
                            *run.ran.entry(region).or_default() += 1;
                        }
                        None => run.claimed_twice += 1,
                    }
                }
            }
        }
        run
    }

    /// Vanilla's trader spawner rolls once every 24000 ticks (20 countdowns of 1200), its patrols once every
    /// 12000 to 13199 ticks of the countdown, its cats once every 1200: however many regions there are. Each
    /// attempt goes to one region, claimed once.
    #[test]
    fn two_hundred_regions_attempt_the_spawners_at_vanilla_s_rate() {
        const DAY: u64 = 24000;
        let ticks = 10 * DAY;
        let run = two_hundred_regions(ticks, 7);
        // 240000 ticks: 10 trader rolls, of which the chance lets through each at its chance; at most 10.
        assert!(
            run.trader_due <= 10,
            "{} trader attempts in ten days",
            run.trader_due
        );
        // Each roll passes at 25, 50, 75, 75... percent; over ten rolls the seed's draws pass at least one.
        assert!(run.trader_due >= 1);
        // The patrol's countdown is 12000 + 0..1199 ticks, and a due attempt passes one time in five: 240000 ticks
        // hold 18 to 20 countdowns.
        assert!(
            (0..=20).contains(&run.patrol_due),
            "{} patrol attempts",
            run.patrol_due
        );
        // Cats: every 1200 ticks, 200 of them in ten days, the first at tick 1.
        assert_eq!(run.cat_due, 200);
        // Every attempt ran in exactly one region.
        let ran: u32 = run.ran.values().sum();
        assert_eq!(ran, run.trader_due + run.patrol_due + run.cat_due);
        assert_eq!(run.claimed_twice, ran, "a ticket is claimed once");
        // 200 regions each counting down their own would have attempted 200 times as often: 40000 cats.
    }

    /// A region is picked in proportion to its living players, so each player has vanilla's odds.
    #[test]
    fn a_region_is_picked_in_proportion_to_its_living_players() {
        let alive = vec![(1u64, 1u32), (2, 3), (4, 6)];
        let mut random = level_random(42);
        let mut hits: FxHashMap<RegionId, u32> = FxHashMap::default();
        for _ in 0..100_000 {
            *hits
                .entry(pick_region(&alive, &mut random).unwrap())
                .or_default() += 1;
        }
        // 1 : 3 : 6 of 100000 draws, within a percent of the total.
        for (region, share) in [(1u64, 10_000i64), (2, 30_000), (4, 60_000)] {
            let got = i64::from(hits[&region]);
            assert!(
                (got - share).abs() < 1_000,
                "region {region}: {got} of 100000"
            );
        }
        assert!(pick_region(&[], &mut random).is_none());
    }

    /// A grant nobody claims is taken back after 40 ticks, so a region that merged does not lose the attempt.
    #[test]
    fn a_grant_no_region_claims_is_taken_back() {
        let global = GlobalTick::default();
        global.grant(SpawnerKind::Cat, Dim::Overworld, 5, 100);
        assert!(global.expired(120).is_empty());
        assert_eq!(global.expired(140).len(), 1);
        assert_eq!(global.open_grants(), 0);
    }

    /// Mob caps: a region adds the others' counts to its own, once.
    #[test]
    fn a_region_reads_the_caps_of_the_other_regions() {
        let dim = Dim::Overworld;
        let mut state = GlobalState::default();
        for region in 1..=3u64 {
            state.record(
                dim,
                region,
                Census {
                    alive: 1,
                    chunks: 100,
                    counts: [region as i32, 0, 0, 0, 0, 0, 0, 0],
                },
                0,
            );
        }
        let (chunks, counts) = state.others(dim, 2);
        assert_eq!(chunks, 200);
        assert_eq!(counts[0], 1 + 3);
    }

    /// The siege night is claimed by the first region that asks, and by no other.
    #[test]
    fn one_region_runs_the_siege_night() {
        let global = GlobalTick::default();
        let dim = Dim::Overworld;
        global.set_night(dim, true);
        assert!(global.claim_night(dim, 3));
        assert!(!global.claim_night(dim, 4));
        assert!(global.claim_night(dim, 3));
        global.finish_night(dim);
        assert!(!global.night(dim).open_to(3));
        global.set_night(dim, false);
        assert!(!global.night(dim).tonight);
    }
}
