//! What a stage says it holds is what it requests. A stage that states
//! its peak bytes from its inputs, for a caller to admit it against a
//! memory before the work, is run here on one thread under a counting
//! allocator, and the peak of the requests live on that thread is within
//! the statement, whose enumerated terms, `RECORDS` below it, are within
//! `RECORDS` of the peak either way on a 64-bit word, the word the
//! statements count by; on a narrower word they stay bounds.
//!
//! Requests are counted, not resident pages: an allocation counts from its
//! request, a reallocation by its change and a release by its size, on
//! the thread that makes it and only while that thread is measuring; a
//! release there of what was allocated before the measure began is not
//! let take the count below zero.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use rump::gf2::{
    filter_floor, filter_merge_within, wiedemann_sequence, wiedemann_sequence_bytes,
    wiedemann_terms_most_of, SparseMatrix, RECORDS,
};
use rump::parallelism::with_budget;

struct Counting;

#[global_allocator]
static ALLOCATOR: Counting = Counting;

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    static LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
}

fn note(change: isize) {
    let _ = MEASURING.try_with(|measuring| {
        if !measuring.get() {
            return;
        }
        let _ = LIVE.try_with(|live| {
            let now = live.get().saturating_add_signed(change);
            live.set(now);
            let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
        });
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            note(layout.size() as isize);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        note(-(layout.size() as isize));
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            note(new_size as isize - layout.size() as isize);
        }
        moved
    }
}

/// `work`'s result, and the peak of the bytes requested and not yet
/// released on this thread while it ran, over what it held at the start.
fn peak_while<R>(work: impl FnOnce() -> R) -> (R, u64) {
    LIVE.with(|live| live.set(0));
    PEAK.with(|peak| peak.set(0));
    MEASURING.with(|measuring| measuring.set(true));
    let result = work();
    MEASURING.with(|measuring| measuring.set(false));
    (result, PEAK.with(Cell::get) as u64)
}

/// A matrix as a sieve's is: `rows` relations of up to `weight` columns
/// each among `columns`, from a xorshift state.
fn sieve_like(rows: usize, columns: usize, weight: usize, mut state: u64) -> SparseMatrix {
    let mut draw = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let lists = (0..rows)
        .map(|_| {
            let mut row: Vec<u32> = (0..weight)
                .map(|_| (draw() % columns as u64) as u32)
                .collect();
            row.sort_unstable();
            row.dedup();
            row
        })
        .collect();
    SparseMatrix::new(columns, lists)
}

/// Whether the statements are tight here: they count a word as eight
/// bytes and a vector's header as three, which is what a 64-bit word
/// gives; on a narrower word the headers are smaller and the statements
/// stay bounds, looser.
const TIGHT: bool = usize::BITS == 64;

/// The peak within the statement; and, on a 64-bit word, the statement's
/// enumerated terms, which are `RECORDS` below it, within `RECORDS` of
/// the peak either way.
fn within(peak: u64, stated: u64, what: &str) {
    eprintln!(
        "{what}: {peak} requested, {stated} stated, {}",
        stated as i64 - peak as i64
    );
    assert!(peak <= stated, "{what}: {peak} requested, {stated} stated");
    assert!(
        !TIGHT || stated - peak <= 2 * RECORDS,
        "{what}: {peak} requested, {stated} stated"
    );
}

/// The sequence stage, from the start and on from a prior, on matrices
/// with and without dense columns and with more columns than rows.
#[test]
fn the_sequence_stage_holds_what_it_states() {
    const OF: usize = 2;
    for (rows, columns, weight) in [(700usize, 600usize, 12usize), (300, 1_000, 5), (64, 40, 3)] {
        let matrix = sieve_like(rows, columns, weight, 0x5eed_0000 + rows as u64);
        let whole = wiedemann_terms_most_of(rows.max(columns), OF);
        let first = whole / 3;
        let stated = |terms| wiedemann_sequence_bytes(&matrix, OF, terms, 1).expect("fits");
        let (prior, peak) = with_budget(1, || {
            peak_while(|| wiedemann_sequence(&matrix, 7, 1, OF, first, None, 1).expect("a slice"))
        });
        within(
            peak,
            stated(first),
            &format!("{rows}×{columns} from the start"),
        );
        let (_, peak) = with_budget(1, || {
            peak_while(|| {
                wiedemann_sequence(&matrix, 7, 1, OF, whole, Some(&prior), 1).expect("a slice")
            })
        });
        within(
            peak,
            stated(whole),
            &format!("{rows}×{columns} on from a prior"),
        );
    }
}

/// Refusals come before any request: a prior of another seed with a slab
/// past `usize` is refused, not a capacity overflow; a thread count past
/// the arithmetic is no statement; and a query's own scratch is a weight
/// a column, stated.
#[test]
fn refusals_come_before_requests() {
    let matrix = sieve_like(64, 40, 3, 0x5eed_0040);
    let prior = wiedemann_sequence(&matrix, 7, 0, 2, 4, None, 1).expect("a slice");
    assert!(wiedemann_sequence(&matrix, 8, 0, 2, usize::MAX / 64, Some(&prior), 1).is_none());
    assert!(wiedemann_sequence(&matrix, 7, 0, 2, usize::MAX / 64, Some(&prior), 1).is_none());
    let one = wiedemann_sequence(&matrix, 7, 0, 1, 4, None, 1).expect("a slice of one");
    assert!(wiedemann_sequence(&matrix, 7, 0, 1, usize::MAX / 64, Some(&one), 1).is_none());
    let (_, peak) = with_budget(1, || {
        peak_while(|| wiedemann_sequence(&matrix, 8, 0, 2, 4, Some(&prior), 1))
    });
    assert_eq!(peak, 0, "a foreign prior is refused before any request");
    // A thread count whose runs pass `u64` is no statement; on a word
    // narrower than sixty-four bits no `usize` count does.
    if usize::BITS >= 64 {
        assert!(wiedemann_sequence_bytes(&matrix, 2, 8, usize::MAX).is_none());
        assert!(filter_floor(&matrix, 8, 32, usize::MAX).is_none());
    }
    assert!(filter_floor(&matrix, usize::MAX, 32, 1).is_some());
    let (_, peak) = with_budget(1, || {
        peak_while(|| wiedemann_sequence_bytes(&matrix, 2, 8, 1))
    });
    assert_eq!(peak, 4 * 40, "the query's scratch");
    let (_, peak) = with_budget(1, || peak_while(|| filter_floor(&matrix, 8, 32, 1)));
    assert_eq!(peak, 4 * 40, "the query's scratch");
}

/// Two matrices of the reviewer's: two hundred rows of three entries over
/// three hundred columns, every column held by two rows, at a cap of
/// two, where the merge's lists were made before they were admitted; and
/// a chain of a thousand and twenty-four rows at a cap of one, where the
/// pruning's rounds went uncounted. At the floor as the limit the peak is
/// within the floor and the report; unlimited, within the report.
#[test]
fn the_filter_keeps_within_its_limit_on_the_reviewers_matrices() {
    let two_by_two = || {
        SparseMatrix::new(
            300,
            (0..200u32)
                .map(|i| {
                    let mut row = vec![(i + 199) % 200, i, 200 + i % 100];
                    row.sort_unstable();
                    row
                })
                .collect(),
        )
    };
    let chain = || SparseMatrix::new(1_025, (0..1_024u32).map(|i| vec![i, i + 1]).collect());
    for (make, cap, excess, what) in [
        (
            &two_by_two as &dyn Fn() -> SparseMatrix,
            2usize,
            32usize,
            "two by two",
        ),
        (&chain, 1, 32, "chain"),
    ] {
        let floor = filter_floor(&make(), cap, excess, 1).expect("fits");
        let (at_floor, peak) = with_budget(1, || {
            peak_while(|| filter_merge_within(make(), cap, excess, floor).expect("at the floor"))
        });
        eprintln!(
            "{what} at the floor: {peak} requested, {floor} stated, {} reported",
            at_floor.peak_bytes()
        );
        assert!(
            peak <= floor,
            "{what} at the floor: {peak} requested, {floor} stated"
        );
        assert!(
            peak <= at_floor.peak_bytes()
                && (!TIGHT || at_floor.peak_bytes() - peak <= 2 * RECORDS),
            "{what} at the floor: {peak} requested, {} reported",
            at_floor.peak_bytes()
        );
        let (merged, peak) = with_budget(1, || {
            peak_while(|| filter_merge_within(make(), cap, excess, u64::MAX).expect("no limit"))
        });
        eprintln!(
            "{what} unlimited: {peak} requested, {} reported",
            merged.peak_bytes()
        );
        assert!(
            peak <= merged.peak_bytes() && (!TIGHT || merged.peak_bytes() - peak <= 2 * RECORDS),
            "{what} unlimited: {peak} requested, {} reported",
            merged.peak_bytes()
        );
    }
}

/// The filter: at its floor as the limit the requests' peak is within the
/// floor and within what it reports it held; with no limit, within what it
/// reports, and that within `RECORDS` above the requests; below the floor
/// it is refused with the floor, and where the whole merge needed more
/// than the floor, the merge at the floor stopped short. The matrix is
/// made inside the measure, since the filter takes it.
#[test]
fn the_filter_holds_what_it_states() {
    const CAP: usize = 8;
    for (rows, columns, weight, excess) in [
        (2_000usize, 4_000usize, 10usize, 32usize),
        (2_000, 1_200, 4, 400),
    ] {
        let seed = 0xf11e_0000 + rows as u64;
        let floor =
            filter_floor(&sieve_like(rows, columns, weight, seed), CAP, excess, 1).expect("fits");
        let (at_floor, peak) = with_budget(1, || {
            peak_while(|| {
                let matrix = sieve_like(rows, columns, weight, seed);
                filter_merge_within(matrix, CAP, excess, floor).expect("at the floor")
            })
        });
        eprintln!(
            "{rows}×{columns} at the floor: {peak} requested, {floor} stated, {} reported",
            at_floor.peak_bytes()
        );
        assert!(
            peak <= floor,
            "{rows}×{columns} at the floor: {peak} requested, {floor} stated"
        );
        assert!(
            peak <= at_floor.peak_bytes()
                && (!TIGHT || at_floor.peak_bytes() - peak <= 2 * RECORDS),
            "{rows}×{columns} at the floor: {peak} requested, {} reported",
            at_floor.peak_bytes()
        );
        let (merged, peak) = with_budget(1, || {
            peak_while(|| {
                let matrix = sieve_like(rows, columns, weight, seed);
                filter_merge_within(matrix, CAP, excess, u64::MAX).expect("no limit")
            })
        });
        eprintln!(
            "{rows}×{columns} unlimited: {peak} requested, {} reported",
            merged.peak_bytes()
        );
        assert!(
            peak <= merged.peak_bytes() && (!TIGHT || merged.peak_bytes() - peak <= 2 * RECORDS),
            "{rows}×{columns} unlimited: {peak} requested, {} reported",
            merged.peak_bytes()
        );
        // A limit binds: below the floor the filter is refused with it,
        // and where the whole merge needed more than the floor, the merge
        // at the floor stopped short of it.
        let refused = with_budget(1, || {
            filter_merge_within(
                sieve_like(rows, columns, weight, seed),
                CAP,
                excess,
                floor - 1,
            )
            .map(|_| ())
            .expect_err("below the floor")
        });
        assert_eq!(refused, floor);
        assert!(merged.rows().len() <= at_floor.rows().len());
        if merged.peak_bytes() > floor {
            assert!(
                merged.rows().len() < at_floor.rows().len(),
                "the limit did not bind"
            );
        }
    }
}
