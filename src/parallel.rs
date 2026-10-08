//! How many threads the work on a thread may use, and fanning work out
//! within that.
//!
//! A routine that fans out cannot see who called it. A product tree asked
//! for by one of eight threads, each of which was given a sixteenth of the
//! machine, has a sixteenth of the machine to spend and not the whole of it;
//! asking the operating system how many processors there are answers a
//! different question. So each thread carries a budget: the machine's count
//! until something narrows it, and a fan-out's share inside a fan-out. The
//! budgets of the threads running at any moment then sum to no more than the
//! budget the outermost caller had.
//!
//! The budget is a ceiling on threads running, not a reservation: a routine
//! that has nothing to fan out spends one.

use std::cell::Cell;

thread_local! {
    /// This thread's budget; zero until it is set, which reads as the
    /// machine's count.
    static BUDGET: Cell<usize> = const { Cell::new(0) };
}

/// The machine's reported parallelism, asked once.
///
/// [`std::thread::available_parallelism`] reads `/proc/self/cgroup` and the
/// cgroup's CPU limits on every call on Linux, several file syscalls each
/// time. The NTT asks on every large multiplication, where that cost
/// dominates under many threads. The answer does not change within a process
/// in any way this crate should react to, so it is taken once and kept.
fn machine() -> usize {
    static MACHINE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MACHINE
        .get_or_init(|| std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get))
}

/// The threads the calling thread's work may use: what the machine reports,
/// unless the work is inside [`with_budget`] or inside one of this crate's
/// own fan-outs, each of whose threads has its share of the budget of the
/// thread that fanned out. Never more than the machine reports, and never
/// zero.
///
/// ```
/// use rump::parallelism::{budget, with_budget};
///
/// let whole = budget();
/// assert!(whole >= 1);
/// assert_eq!(with_budget(1, budget), 1);
/// assert_eq!(budget(), whole);
/// ```
#[must_use]
pub fn budget() -> usize {
    match BUDGET.with(Cell::get) {
        0 => machine(),
        set => set.min(machine()),
    }
}

/// Runs `work` with the calling thread's budget set to `threads`, and sets it
/// back afterwards, also when `work` panics. Zero is taken as one.
///
/// A caller that fans out on threads of its own gives each its share this
/// way, so that what this crate fans out inside them stays within it. The
/// budget can be raised as well as narrowed; [`budget`] never reports more
/// than the machine has.
pub fn with_budget<R>(threads: usize, work: impl FnOnce() -> R) -> R {
    /// Puts the budget back when dropped.
    struct Restore(usize);

    impl Drop for Restore {
        fn drop(&mut self) {
            BUDGET.with(|budget| budget.set(self.0));
        }
    }

    let _restore = Restore(BUDGET.with(|budget| budget.replace(threads.max(1))));
    work()
}

/// The work a thread of a fan-out should have to itself, in products of two
/// limbs: a product of `a` limbs by `b` counts `a·b`, which is its cost
/// until the operands are wide enough that a fan-out is not in question.
///
/// A thread costs its making and its joining whatever it carries, so a level
/// of a tree is fanned out over as many threads as it has this much work
/// for, and not over one for each node: two threads from twice the grain.
/// `fan_out_grain_timing` in `number_theory.rs` times a level on one thread
/// and on two, four and eight; at 2¹⁹ limb products two were ahead of one on
/// an M4 and on an EPYC 7452 at every leaf width tried, and at 2¹⁸ one was
/// ahead on the EPYC (PERFORMANCE.md, "Fanning out a tree's level").
pub(crate) const GRAIN_LIMB_PRODUCTS: usize = 1 << 18;

/// The threads worth fanning `work` out over, `work` in products of two
/// limbs.
pub(crate) fn threads_for(work: usize) -> usize {
    (work / GRAIN_LIMB_PRODUCTS).max(1)
}

/// The blocks a thread's share of the items is taken in. A thread that took
/// its whole share at once would leave the others idle when its items were
/// the costly ones, and one that took an item at a time would pay the cursor
/// for each; eight blocks a thread bounds the idling at an eighth of a
/// share.
const BLOCKS_PER_WORKER: usize = 8;

/// `map` over `items`, the results in the order of the items, on as many
/// threads as `workers` names, the budget allows and there are items for.
/// Each thread's budget is its share of the caller's, so what `map` fans out
/// in its turn stays within the whole.
///
/// The threads take the items from a shared cursor a block at a time,
/// [`BLOCKS_PER_WORKER`] blocks to a thread's share: the items of a tree's
/// level can differ in cost by the width of their operands. One thread, or
/// one item, runs on the caller's.
pub(crate) fn map_ordered<T, R, F>(items: &[T], workers: usize, map: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &T) -> R + Sync,
{
    let whole = budget();
    let workers = workers.min(whole).min(items.len());
    if workers <= 1 {
        return items
            .iter()
            .enumerate()
            .map(|(index, item)| map(index, item))
            .collect();
    }
    let share = (whole / workers).max(1);
    let block = items.len().div_ceil(workers * BLOCKS_PER_WORKER);
    let cursor = std::sync::atomic::AtomicUsize::new(0);
    let mut blocks: Vec<(usize, Vec<R>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let (cursor, map) = (&cursor, &map);
                scope.spawn(move || {
                    with_budget(share, || {
                        let mut mine = Vec::new();
                        loop {
                            let start =
                                cursor.fetch_add(block, std::sync::atomic::Ordering::Relaxed);
                            if start >= items.len() {
                                break;
                            }
                            let end = (start + block).min(items.len());
                            let mapped = (start..end)
                                .map(|index| map(index, &items[index]))
                                .collect();
                            mine.push((start, mapped));
                        }
                        mine
                    })
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("a fan-out's worker panicked"))
            .collect()
    });
    blocks.sort_unstable_by_key(|&(start, _)| start);
    blocks.into_iter().flat_map(|(_, mapped)| mapped).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_budget_is_the_machine_until_it_is_set() {
        assert_eq!(budget(), machine());
        assert!(budget() >= 1);
    }

    #[test]
    fn a_budget_holds_inside_and_is_put_back_outside() {
        let whole = budget();
        let inside = with_budget(1, || {
            let nested = with_budget(usize::MAX, budget);
            (budget(), nested)
        });
        // One inside; raised past the machine, the machine.
        assert_eq!(inside, (1, machine()));
        assert_eq!(budget(), whole);
        // Zero is one.
        assert_eq!(with_budget(0, budget), 1);
    }

    #[test]
    fn a_budget_is_put_back_when_the_work_panics() {
        let whole = budget();
        let caught = std::panic::catch_unwind(|| with_budget(1, || panic!("the work fails")));
        assert!(caught.is_err());
        assert_eq!(budget(), whole);
    }

    #[test]
    fn a_budget_is_a_threads_own() {
        with_budget(1, || {
            let elsewhere = std::thread::scope(|scope| {
                scope
                    .spawn(budget)
                    .join()
                    .expect("the thread does not panic")
            });
            assert_eq!(elsewhere, machine());
            assert_eq!(budget(), 1);
        });
    }

    #[test]
    fn map_ordered_is_the_serial_map_at_any_width() {
        let items: Vec<u64> = (0..97).map(|v| v * v + 1).collect();
        let serial: Vec<u64> = items
            .iter()
            .map(|v| v.wrapping_mul(2_654_435_761))
            .collect();
        for workers in [0usize, 1, 2, 3, 12, 200] {
            let mapped = map_ordered(&items, workers, |_, v| v.wrapping_mul(2_654_435_761));
            assert_eq!(mapped, serial, "workers = {workers}");
        }
        let empty: Vec<u64> = map_ordered(&[], 4, |_, v: &u64| *v);
        assert!(empty.is_empty());
    }

    /// The budget seen on each thread that mapped an item, keyed by the
    /// thread: the budgets of a fan-out's threads, each once, whichever
    /// items each took from the cursor.
    fn budgets_by_thread(items: usize, workers: usize) -> std::collections::BTreeMap<u64, usize> {
        let items = vec![(); items];
        let seen = map_ordered(&items, workers, |_, ()| (thread_key(), budget()));
        let mut by_thread = std::collections::BTreeMap::new();
        for (thread, share) in seen {
            let known = by_thread.entry(thread).or_insert(share);
            assert_eq!(*known, share, "a thread's budget holds for all its items");
        }
        by_thread
    }

    /// The calling thread, as a key: `ThreadId` has no order, and the
    /// `as_u64` form is unstable.
    fn thread_key() -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn a_fan_outs_threads_share_its_budget() {
        // Eight to share among four: two each, and the shares of the threads
        // that ran sum to no more than the whole. On a machine of fewer than
        // eight the whole is the machine's, the workers are no more than
        // that, and the shares are of it; a share is counted once a thread,
        // however many items the thread took from the cursor, which is what
        // a machine of three ran into when the items were summed instead.
        let (whole, by_thread) = with_budget(8, || (budget(), budgets_by_thread(4, 4)));
        let workers = 4.min(whole);
        assert!(!by_thread.is_empty());
        assert!(by_thread.len() <= workers);
        for share in by_thread.values() {
            assert_eq!(
                *share,
                if workers <= 1 {
                    whole
                } else {
                    (whole / workers).max(1)
                }
            );
        }
        assert!(by_thread.values().sum::<usize>() <= whole);
        // Nothing to share: the fan-out runs on the caller's thread, with the
        // caller's budget.
        assert_eq!(
            with_budget(8, || map_ordered(&[()], 4, |_, ()| budget())),
            [whole]
        );
    }

    #[test]
    fn the_shares_are_within_the_whole_at_every_budget_and_width() {
        // Every budget up to the machine's and past it, workers fewer than
        // the budget, as many and more, and more items than workers so that
        // a thread takes several: the threads that ran number no more than
        // the budget or the workers, each has the share, and the shares sum
        // to no more than the whole. A budget of three with four workers is
        // the shape of a three-core runner; a budget of one is a fan-out
        // that does not fan out.
        for asked in (1..=machine()).chain([machine() + 1, 2 * machine() + 3]) {
            for workers in [1usize, 2, 3, 4, 5, 8, 64] {
                for items in [1usize, 2, 4, 7, 50] {
                    let (whole, by_thread) =
                        with_budget(asked, || (budget(), budgets_by_thread(items, workers)));
                    assert_eq!(whole, asked.min(machine()));
                    let running = workers.min(whole).min(items);
                    let context = format!("budget {asked}, workers {workers}, items {items}");
                    if running <= 1 {
                        assert_eq!(by_thread.len(), 1, "{context}: on the caller's thread");
                        assert_eq!(by_thread.values().next(), Some(&whole), "{context}");
                        continue;
                    }
                    assert!(
                        by_thread.len() <= running,
                        "{context}: too many threads ran"
                    );
                    for share in by_thread.values() {
                        assert_eq!(*share, (whole / running).max(1), "{context}");
                        assert!(*share >= 1, "{context}: a budget is never zero");
                    }
                    assert!(
                        by_thread.values().sum::<usize>() <= whole,
                        "{context}: the shares exceed the whole"
                    );
                }
            }
        }
    }

    #[test]
    fn a_nested_fan_out_stays_within_the_outer_share() {
        // A fan-out inside a fan-out's thread shares that thread's share,
        // not the whole: the innermost budgets, summed over the threads of
        // one outer thread, are within the outer thread's share, so the
        // budgets of the threads running at once are within the whole.
        let outer_items = [(); 2];
        let (whole, leaves) = with_budget(usize::MAX, || {
            let whole = budget();
            let leaves = map_ordered(&outer_items, 2, |_, ()| {
                let share = budget();
                let inner = budgets_by_thread(6, 4);
                (share, inner)
            });
            (whole, leaves)
        });
        assert_eq!(whole, machine());
        let outer_running = 2.min(whole);
        for (share, inner) in &leaves {
            assert_eq!(*share, (whole / outer_running).max(1));
            for inner_share in inner.values() {
                assert!(
                    *inner_share <= *share,
                    "an inner budget exceeds its outer share"
                );
            }
            assert!(
                inner.values().sum::<usize>() <= *share,
                "the inner shares {inner:?} exceed the outer share {share}"
            );
        }
    }

    #[test]
    fn a_fan_out_leaves_the_callers_budget_as_it_was() {
        // The workers' shares are the workers'; the caller's budget is the
        // same after the fan-out as before, and so is the machine's default.
        let items: Vec<u32> = (0..40).collect();
        let (before, after) = with_budget(3, || {
            let before = budget();
            let mapped: Vec<u32> = map_ordered(&items, 3, |_, v| v + 1);
            assert_eq!(mapped, (1..=40).collect::<Vec<u32>>());
            (before, budget())
        });
        assert_eq!(before, after);
        let unset = budget();
        let mapped: Vec<u32> = map_ordered(&items, 3, |_, v| v + 1);
        assert_eq!(mapped.len(), 40);
        assert_eq!(budget(), unset);
        assert_eq!(unset, machine());
    }

    #[test]
    fn map_ordered_keeps_the_order_when_the_items_cost_unevenly() {
        // Items that take unequal time are taken from the cursor out of
        // step; the results come back in the items' order all the same.
        let items: Vec<u64> = (0..64).collect();
        let mapped = map_ordered(&items, 4, |index, v| {
            // The last items spin longest, so a thread that took an early
            // block finishes first and takes more.
            let spins = if index % 7 == 0 { 20_000 } else { 10 };
            let mut acc = *v;
            for _ in 0..spins {
                acc = acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            }
            (index as u64, acc)
        });
        for (index, (seen, _)) in mapped.iter().enumerate() {
            assert_eq!(*seen, index as u64);
        }
    }
}
