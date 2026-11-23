//! Work that crosses regions: the task queues, and the questions one region asks another.
//!
//! A region never reaches into another region's game. What it needs done there it sends as a task, and the owner
//! runs it at the start of its next tick, on its own thread, as Folia's schedulers do:
//!
//! - a **region queue** (one per [`RegionId`](super::region_map::RegionId); [`SERVER`] is the server thread's own
//!   game): what is for the region and not for one thing in it;
//! - an **entity queue** (one per entity UUID): follows the entity wherever it goes. The queue is not kept in
//!   any region, so a region that hands the entity to another, or merges into it, takes nothing with it and
//!   loses nothing: whichever game holds the entity at its next tick runs the task. A task whose entity is gone
//!   for good (it left the server, or died) runs its `retired` half instead, on the server thread.
//!
//! A task is a closure over the owner's [`Game`]. It may queue more tasks, send an answer back through a channel
//! it captured, or hand a player to another region ([`HandOff`], delivered by `region_handoff.rs`).
use super::region_map::RegionId;
use super::{Dim, Game};
use horizon_client::command::args::Selector;
use horizon_core::fast_hash::FxHashMap;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// The region queue of the server thread's own game: what no region owns.
pub(super) const SERVER: RegionId = 0;

/// Server ticks an entity's task waits for its owner before it is retired: a player who left, an entity that
/// died or sits in a level nobody holds. Five seconds.
const ENTITY_TASK_TICKS: u64 = 100;

/// Server ticks a question to another region may take before it is taken as unanswered.
const QUESTION_TICKS: u64 = 100;

/// What a task does with its owner's game.
pub(super) type Work = Box<dyn for<'a, 'b> FnOnce(&'a mut Game<'b>) + Send + 'static>;

/// One queued task: what to run, and what to run instead when its owner is gone.
pub(super) struct Task {
    run: Work,
    retired: Option<Work>,
    /// The server tick it was queued in.
    since: u64,
}

/// A player on their way to the region that holds `arrival`'s place.
pub(super) struct HandOff {
    pub kit: super::lan::PlayerKit,
    /// The level the player left.
    pub from: Dim,
    pub arrival: Arrival,
}

/// Where a handed-off player appears: a place in a level, with the rotation they keep (None: their own).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Arrival {
    pub dim: Dim,
    pub position: [f64; 3],
    pub rotation: Option<[f32; 2]>,
}

/// Every queue of a server, shared by all its games.
#[derive(Default)]
pub(super) struct Tasks {
    regions: Mutex<FxHashMap<RegionId, VecDeque<Task>>>,
    entities: Mutex<FxHashMap<u128, VecDeque<Task>>>,
    /// Players handed off by a region's task, for the server thread to deliver at a tick boundary.
    handoffs: Mutex<Vec<HandOff>>,
    /// Entity tasks queued: a tick with none does not look at who its game holds.
    queued: AtomicUsize,
    /// Server ticks so far (for the age of a task).
    clock: AtomicU64,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Tasks {
    /// A task for `region`: it runs at that region's next tick.
    pub(super) fn region(
        &self,
        region: RegionId,
        run: impl for<'a, 'b> FnOnce(&'a mut Game<'b>) + Send + 'static,
    ) {
        let since = self.now();
        lock(&self.regions)
            .entry(region)
            .or_default()
            .push_back(Task {
                run: Box::new(run),
                retired: None,
                since,
            });
    }

    /// A task for the entity with `uuid`: it runs at the next tick of the game that holds it, whichever that is
    /// by then. `retired` runs instead, on the server thread, when no game holds the entity for good.
    pub(super) fn entity(
        &self,
        uuid: u128,
        run: impl for<'a, 'b> FnOnce(&'a mut Game<'b>) + Send + 'static,
        retired: Option<Work>,
    ) {
        let since = self.now();
        lock(&self.entities)
            .entry(uuid)
            .or_default()
            .push_back(Task {
                run: Box::new(run),
                retired,
                since,
            });
        self.queued.fetch_add(1, Ordering::Relaxed);
    }

    /// The region's queued tasks, oldest first.
    pub(super) fn take_region(&self, region: RegionId) -> Vec<Task> {
        lock(&self.regions)
            .remove(&region)
            .map_or_else(Vec::new, Vec::from)
    }

    /// Whether any entity has a task queued.
    pub(super) fn has_entity_tasks(&self) -> bool {
        self.queued.load(Ordering::Relaxed) > 0
    }

    /// The queued tasks of the entities `holds` says the asking game holds, oldest first.
    pub(super) fn take_entities(&self, holds: impl Fn(u128) -> bool) -> Vec<Task> {
        let mut out = Vec::new();
        let mut entities = lock(&self.entities);
        entities.retain(|&uuid, queue| {
            if !holds(uuid) {
                return true;
            }
            out.extend(queue.drain(..));
            false
        });
        self.queued.fetch_sub(out.len(), Ordering::Relaxed);
        out
    }

    /// The tasks of `from`, now `to`'s: the region merged into it.
    pub(super) fn merge_region(&self, from: RegionId, to: RegionId) {
        let mut regions = lock(&self.regions);
        if let Some(mut queue) = regions.remove(&from) {
            regions.entry(to).or_default().append(&mut queue);
        }
    }

    /// The entity tasks nobody has run in [`ENTITY_TASK_TICKS`]: their owner is gone.
    pub(super) fn expired(&self) -> Vec<Task> {
        let now = self.now();
        let mut out = Vec::new();
        let mut entities = lock(&self.entities);
        entities.retain(|_, queue| {
            while queue
                .front()
                .is_some_and(|t| now.saturating_sub(t.since) >= ENTITY_TASK_TICKS)
            {
                out.extend(queue.pop_front());
            }
            !queue.is_empty()
        });
        self.queued.fetch_sub(out.len(), Ordering::Relaxed);
        out
    }

    /// How many tasks wait in the queue of `region`.
    #[cfg(test)]
    pub(super) fn region_len(&self, region: RegionId) -> usize {
        lock(&self.regions).get(&region).map_or(0, VecDeque::len)
    }

    /// How many tasks wait in the queue of the entity `uuid`.
    #[cfg(test)]
    pub(super) fn entity_len(&self, uuid: u128) -> usize {
        lock(&self.entities).get(&uuid).map_or(0, VecDeque::len)
    }

    /// One more server tick has begun.
    pub(super) fn tick(&self) {
        self.clock.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn now(&self) -> u64 {
        self.clock.load(Ordering::Relaxed)
    }

    /// A player a region's task handed off.
    pub(super) fn hand_off(&self, hand_off: HandOff) {
        lock(&self.handoffs).push(hand_off);
    }

    /// The handed-off players `wanted` says are for the caller, in the order they were handed off.
    pub(super) fn take_handoffs(&self, wanted: impl Fn(&HandOff) -> bool) -> Vec<HandOff> {
        let mut all = lock(&self.handoffs);
        if all.is_empty() {
            return Vec::new();
        }
        let (take, keep): (Vec<_>, Vec<_>) =
            std::mem::take(&mut *all).into_iter().partition(wanted);
        *all = keep;
        take
    }
}

/// Where a player another region holds stands, as that region reports it.
pub(super) struct Place {
    pub name: String,
    pub dim: Dim,
    pub position: [f64; 3],
    pub rotation: [f32; 2],
}

/// What asking another region about a player has come to.
pub(super) enum Answer {
    Ready(Place),
    /// Asked, no answer yet: the command that wants it tries again next tick.
    Asked,
    /// The player is gone, or their region never answered.
    Gone,
}

/// A question waiting for its answer: the channel the region answers on, and the server tick it was asked in.
pub(super) type Question = (std::sync::mpsc::Receiver<Option<Place>>, u64);

impl Game<'_> {
    /// Runs what is queued for this game at the start of its tick: the tasks of its region, then those of the
    /// entities it holds (a player in its hub, an entity of its level).
    pub(super) fn run_region_tasks(&mut self) {
        let tasks = std::sync::Arc::clone(&self.regions.tasks);
        let id = self.regions.link.as_ref().map_or(SERVER, |l| l.id);
        for task in tasks.take_region(id) {
            (task.run)(self);
        }
        if tasks.has_entity_tasks() {
            for task in tasks.take_entities(|uuid| self.holds_entity(uuid)) {
                (task.run)(self);
            }
        }
    }

    /// The server thread's tick: counts it, and retires the entity tasks whose entity nobody holds any more.
    pub(super) fn run_server_tasks(&mut self) {
        self.regions.tasks.tick();
        for task in self.regions.tasks.expired() {
            if let Some(retired) = task.retired {
                retired(self);
            }
        }
        self.run_region_tasks();
        // The commands that asked another region something, run again with what it answered since.
        self.retry_deferred_commands();
    }

    /// Whether this game holds the entity `uuid`: a player placed in its hub, or an entity of its level.
    fn holds_entity(&self, uuid: u128) -> bool {
        self.lan
            .as_ref()
            .is_some_and(|h| h.hub.guests.iter().any(|g| g.uuid == uuid && g.placed))
            || self.target_by_uuid(uuid).is_some()
    }

    /// Where the player `uuid` stands, when this game holds them.
    pub(super) fn place_of_player(&self, uuid: u128) -> Option<Place> {
        let g = self
            .lan
            .as_ref()?
            .hub
            .guests
            .iter()
            .find(|g| g.uuid == uuid && g.placed)?;
        Some(Place {
            name: g.name.clone(),
            dim: Dim::from_id(&g.dimension),
            position: g.position,
            rotation: [g.yaw, g.pitch],
        })
    }

    /// The place of a player in another region, asked of the region that holds them: the answer comes with a
    /// later call, once their region has run the question at its next tick. `Gone` when they left or their region
    /// does not answer.
    pub(super) fn place_of_remote(&mut self, uuid: u128) -> Answer {
        let now = self.regions.tasks.now();
        let Some((rx, asked)) = self.regions.questions.get(&uuid) else {
            let (tx, rx) = std::sync::mpsc::channel();
            let gone = tx.clone();
            self.regions.tasks.entity(
                uuid,
                move |owner| {
                    let _ = tx.send(owner.place_of_player(uuid));
                },
                Some(Box::new(move |_| {
                    let _ = gone.send(None);
                })),
            );
            self.regions.questions.insert(uuid, (rx, now));
            return Answer::Asked;
        };
        let answer = match rx.try_recv() {
            Ok(Some(place)) => Answer::Ready(place),
            Ok(None) | Err(std::sync::mpsc::TryRecvError::Disconnected) => Answer::Gone,
            Err(std::sync::mpsc::TryRecvError::Empty)
                if now.saturating_sub(*asked) >= QUESTION_TICKS =>
            {
                Answer::Gone
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => return Answer::Asked,
        };
        self.regions.questions.remove(&uuid);
        answer
    }

    /// A selector that names a player who is in another region (by name or UUID) and is not here: their UUID and
    /// name.
    pub(super) fn remote_player_of(&self, selector: &Selector) -> Option<(u128, String)> {
        if selector.player_name.is_none() && selector.uuid.is_none() {
            return None;
        }
        self.lan
            .as_ref()?
            .hub
            .roster_other_player(selector.player_name.as_deref(), selector.uuid)
    }

    /// The command being run waits for another region's answer: it is held and run again next tick, without
    /// the error its failed run would tell. Only a command typed by a player and made of nothing but the
    /// teleport is held (`deferrable`): one run again from the start must not repeat what it did before.
    pub(super) fn defer_command(&mut self) -> bool {
        std::mem::take(&mut self.regions.deferrable) && {
            self.regions.command_deferred = true;
            true
        }
    }

    /// A command that asked another region something, held to run again.
    pub(super) fn hold_command(&mut self, c: super::lan_actions::GuestCommand) {
        let now = self.regions.tasks.now();
        let first = self.regions.deferred_since.unwrap_or(now);
        if let Some(host) = self.lan.as_mut() {
            host.deferred_commands.push((c, first, now));
        }
    }

    /// The commands held for another region's answer, run again: once per server tick each.
    pub(super) fn retry_deferred_commands(&mut self) {
        let now = self.regions.tasks.now();
        let Some(host) = self.lan.as_mut() else {
            return;
        };
        if host.deferred_commands.is_empty() {
            return;
        }
        let (due, wait): (Vec<_>, Vec<_>) = std::mem::take(&mut host.deferred_commands)
            .into_iter()
            .partition(|(_, _, last)| *last < now);
        host.deferred_commands = wait;
        for (c, first, _) in due {
            self.regions.deferred_since = Some(first);
            self.run_guest_command(c);
        }
        self.regions.deferred_since = None;
    }

    /// Teleports the player `uuid`, a player of this game, to `arrival`:
    ///
    /// - at once when the place is in this game's own part of the level;
    /// - in the part of another region of the level: handed to that region, as a task it runs at its next tick
    ///   (`adopt_player`), without the server thread;
    /// - in no region's part, or in another level: handed to the server thread ([`HandOff`]), which makes the
    ///   region or the level teleport.
    pub(super) fn teleport_player_task(&mut self, uuid: u128, arrival: Arrival) {
        let Some(conn) = self
            .lan
            .as_ref()
            .and_then(|h| h.hub.guests.iter().find(|g| g.uuid == uuid && g.placed))
            .map(|g| g.conn)
        else {
            return;
        };
        let chunk = (
            (arrival.position[0].floor() as i32) >> 4,
            (arrival.position[2].floor() as i32) >> 4,
        );
        let target = super::commands::Target::Guest(conn);
        let Some(link) = self.regions.link.clone() else {
            // The server thread's game holds the level it ticks; a place in another level is a level
            // teleport.
            self.arrive_player(target, arrival);
            return;
        };
        let owner = (arrival.dim == self.dim)
            .then(|| link.shared.map.owner_of_chunk(chunk))
            .flatten();
        if owner.is_some_and(|o| o.id == link.id) && self.world.chunks.contains_key(&chunk) {
            self.arrive_player(target, arrival);
            return;
        }
        let dim = self.dim;
        let Some(kit) = self.lan.as_mut().and_then(|h| h.take_player_kit(conn, dim)) else {
            return;
        };
        self.bind_lan_residents();
        self.regions.conns_dirty = true;
        match owner {
            Some(o) if o.id != link.id => {
                self.regions
                    .tasks
                    .region(o.id, move |region| region.adopt_player(kit, arrival));
            }
            _ => self.regions.tasks.hand_off(HandOff {
                kit,
                from: dim,
                arrival,
            }),
        }
    }

    /// A player another region handed to this one, put at `arrival`. Run by the region that owns the place at
    /// its next tick; when that region has gone home meanwhile, by the server thread's game.
    fn adopt_player(&mut self, kit: super::lan::PlayerKit, arrival: Arrival) {
        let conn = kit.guest.conn;
        let dim = self.dim;
        let on_server = self.regions.link.is_none();
        let Some(host) = self.lan.as_mut() else {
            return;
        };
        host.give_player_kit(kit, dim);
        if on_server {
            host.hub.release_away_conns(&[conn]);
            self.forget_away_conn(conn);
        }
        self.bind_lan_residents();
        self.regions.conns_dirty = true;
        self.arrive_player(super::commands::Target::Guest(conn), arrival);
    }

    /// Puts the player `target` (held by this game) at `arrival`: within the level in place, or into another
    /// level.
    pub(super) fn arrive_player(&mut self, target: super::commands::Target, arrival: Arrival) {
        let rotation = arrival.rotation;
        let moved = self
            .teleport_across(
                target,
                arrival.dim,
                arrival.position,
                rotation,
                true,
                super::vehicle_teleport::TeleportMotion::Stop,
            )
            .unwrap_or(false);
        if !moved {
            self.teleport_to(target, arrival.position, rotation, true);
        }
    }
}

/// Whether a command line is a teleport (`/tp`, `/teleport`, with or without the namespace).
pub(super) fn is_teleport_command(line: &str) -> bool {
    let word = line
        .trim_start_matches('/')
        .split(' ')
        .next()
        .unwrap_or_default();
    matches!(
        word,
        "tp" | "teleport" | "minecraft:tp" | "minecraft:teleport"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    /// Counts how often a closure ran, the one way to see a task's effect without a game.
    fn counter() -> (Arc<AtomicU32>, impl Fn() -> u32) {
        let n = Arc::new(AtomicU32::new(0));
        let read = Arc::clone(&n);
        (n, move || read.load(Ordering::Relaxed))
    }

    fn hit(n: &Arc<AtomicU32>) -> impl for<'a, 'b> FnOnce(&'a mut Game<'b>) + Send + 'static {
        let n = Arc::clone(n);
        move |_| {
            n.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn a_region_queue_hands_its_tasks_over_once_in_order() {
        let tasks = Tasks::default();
        let (n, _read) = counter();
        tasks.region(4, hit(&n));
        tasks.region(4, hit(&n));
        tasks.region(5, hit(&n));
        assert_eq!(tasks.region_len(4), 2);
        assert_eq!(tasks.take_region(4).len(), 2);
        assert!(tasks.take_region(4).is_empty(), "a task is taken once");
        assert_eq!(
            tasks.region_len(5),
            1,
            "another region's queue is untouched"
        );
    }

    #[test]
    fn merged_regions_keep_the_tasks_queued_for_the_one_that_went() {
        let tasks = Tasks::default();
        let (n, _read) = counter();
        tasks.region(4, hit(&n));
        tasks.region(7, hit(&n));
        tasks.merge_region(7, 4);
        assert_eq!(tasks.region_len(7), 0);
        assert_eq!(tasks.region_len(4), 2);
    }

    #[test]
    fn an_entity_queue_follows_the_entity_to_whichever_game_holds_it() {
        let tasks = Tasks::default();
        let (n, _read) = counter();
        tasks.entity(0xA, hit(&n), None);
        tasks.entity(0xA, hit(&n), None);
        tasks.entity(0xB, hit(&n), None);
        assert!(tasks.has_entity_tasks());
        // A game that holds neither takes nothing and leaves everything queued.
        assert!(tasks.take_entities(|_| false).is_empty());
        assert_eq!(tasks.entity_len(0xA), 2);
        // The entity is handed to another game: that one holds it now, and takes its tasks.
        assert_eq!(tasks.take_entities(|uuid| uuid == 0xA).len(), 2);
        assert_eq!(tasks.entity_len(0xA), 0);
        assert_eq!(tasks.entity_len(0xB), 1);
        assert_eq!(tasks.take_entities(|_| true).len(), 1);
        assert!(!tasks.has_entity_tasks(), "nothing is left queued");
    }

    #[test]
    fn an_entity_task_nobody_runs_is_retired_after_its_time() {
        let tasks = Tasks::default();
        let (n, _read) = counter();
        tasks.entity(0xA, hit(&n), Some(Box::new(hit(&n))));
        for _ in 0..ENTITY_TASK_TICKS - 1 {
            tasks.tick();
        }
        assert!(tasks.expired().is_empty(), "not yet");
        tasks.tick();
        let gone = tasks.expired();
        assert_eq!(gone.len(), 1);
        assert!(gone[0].retired.is_some());
        assert!(!tasks.has_entity_tasks());
    }
}
