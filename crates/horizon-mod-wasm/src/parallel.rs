//! Independent startup work fanned out over a few threads, results kept in input order.
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Calls `f` on every item on up to `threads` threads, the calling thread among them, and returns
/// the results in `items`' order. An item whose call panicked on a spawned thread gets `None`; a
/// panic on the calling thread panics this call, as a serial loop would.
pub(crate) fn map<T: Sync, R: Send + Sync>(
    items: &[T],
    threads: usize,
    f: impl Fn(&T) -> R + Sync,
) -> Vec<Option<R>> {
    let slots: Vec<OnceLock<R>> = items.iter().map(|_| OnceLock::new()).collect();
    let next = AtomicUsize::new(0);
    let work = || {
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(item) = items.get(index) else { break };
            let _ = slots[index].set(f(item));
        }
    };
    std::thread::scope(|scope| {
        // A thread that cannot be spawned leaves its share to the others.
        let workers: Vec<_> = (1..threads.min(items.len()))
            .filter_map(|_| {
                std::thread::Builder::new()
                    .name("wasm load".into())
                    .stack_size(horizon_guest_stack::HOST_HEADROOM)
                    .spawn_scoped(scope, &work)
                    .ok()
            })
            .collect();
        work();
        for worker in workers {
            let _ = worker.join();
        }
    });
    slots.into_iter().map(OnceLock::into_inner).collect()
}
