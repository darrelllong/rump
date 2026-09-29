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

    #[test]
    fn a_fan_outs_threads_share_its_budget() {
        // Eight to share among four: two each, and the shares sum to no more
        // than the whole. On a machine of fewer than eight the whole is the
        // machine's and the shares are of that.
        let items = [(); 4];
        let (whole, shares) =
            with_budget(8, || (budget(), map_ordered(&items, 4, |_, ()| budget())));
        let workers = 4.min(whole);
        for share in &shares {
            assert_eq!(
                *share,
                if workers <= 1 {
                    whole
                } else {
                    (whole / workers).max(1)
                }
            );
        }
        assert!(workers <= 1 || shares.iter().sum::<usize>() <= whole.max(workers));
        // Nothing to share: the fan-out runs on the caller's thread, with the
        // caller's budget.
        assert_eq!(
            with_budget(8, || map_ordered(&[()], 4, |_, ()| budget())),
            [whole]
        );
    }
}
