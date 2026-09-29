//! What the machine is, asked of the machine: what a word read from a place
//! chosen at random costs, by the size of the block it is read from and the
//! threads reading at once.
//!
//! How many processors a machine reports does not say what its threads can
//! do together. Two threads of one core share its caches, the cores of a
//! socket share another, and a block that one thread reads from cheaply is
//! read from dearly when every thread has one: on two EPYC 7452, 64 cores
//! and 128 threads, a word of a 4 MB block costs 1.2 ns read by one thread
//! and 24 ns read by each of 128. A routine that reads at random from
//! something large, a sparse matrix's product or a sieve's buckets, can cut
//! its work into blocks that are cheap to read from, if it knows how large
//! those are here and for how many threads.
//!
//! The sizes of the caches could be read from the operating system where it
//! tells them, but not how they are shared out under load, nor what a
//! container or a virtual machine has been given of them; and the question
//! is what a read costs, which a measurement answers on any machine.
//!
//! # A block is measured against a small one, on the core it is read on
//!
//! A machine's cores need not be alike, and a thread is where the
//! operating system puts it: of a Cortex-X925 and a Cortex-A725 in one
//! package the second reads at half the speed, and a thread timed on one
//! for a block and on the other for the next says that the caches ended
//! where the cores changed. So a thread reads by turns from its block and
//! from a block of the smallest size, and what is kept of the two is how
//! many times dearer the block's reads were: both from the same core, in
//! the same moments. The costs given are those ratios in the nanoseconds of
//! the machine's median core.

use std::sync::Barrier;
use std::time::{Duration, Instant};

/// The smallest block measured, in bytes: within the first-level cache of
/// every machine measured, the smallest of them 32 KB a core and two
/// threads to the core.
const FIRST_BLOCK: usize = 1 << 14;

/// The largest block measured, in bytes. Past the last cache a read costs
/// what memory costs, whatever the block's size; the largest cache to a
/// thread among the machines measured was 16 MB.
const LAST_BLOCK: usize = 1 << 25;

/// The most memory a measurement takes between its threads, in bytes. A
/// block to each of 128 threads at [`LAST_BLOCK`] would be 4 GB, to learn
/// what the blocks of a quarter the size had said; at this ceiling 128
/// threads are measured as far as 8 MB.
const MOST_MEMORY: usize = 1 << 30;

/// The measurement stops at the first block whose reads cost this many
/// times the cheapest: the caches have ended, and what is wanted of the
/// blocks beyond is known. Four is a policy, past any slack a caller has
/// asked of [`GatherCosts::block_within`]; on the machines measured
/// (PERFORMANCE.md, "What a read at random costs") the block a measurement
/// stopped at cost from 4.0 to 31 times the cheapest.
const DEAR: f64 = 4.0;

/// The reads of one pass. A pass is timed whole, so its clock's cost, some
/// tens of nanoseconds, is a thousandth of a nanosecond a read.
const READS: usize = 1 << 16;

/// How long each block is read from, by all the threads at once, by turns
/// with the small block. The shortest pass in it is the block's cost: what
/// else the machine does makes a pass longer and never shorter. 20 ms is a
/// hundred turns and more in a cache and some tens out of one; a policy.
const LONG: Duration = Duration::from_millis(20);

/// What a read at random costs on this machine, by the size of the block
/// read from, with a number of threads reading at once, each from a block of
/// its own. Made by [`gather_costs`].
#[derive(Clone, Debug, PartialEq)]
pub struct GatherCosts {
    threads: usize,
    costs: Vec<(usize, f64)>,
}

impl GatherCosts {
    /// The threads that read at once.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// The blocks measured, each with what a read from it cost: bytes and
    /// nanoseconds, the blocks doubling from 16 KB. The last is the first
    /// whose reads cost four times the cheapest, if there was one within
    /// 32 MB and a gigabyte between the threads.
    ///
    /// A cost includes choosing the place, one multiplication, which is the
    /// same for every block: the costs are for comparing with each other.
    /// Each is how many times dearer than a read from 16 KB the threads
    /// found the block, their median, in the nanoseconds the median thread
    /// read 16 KB in.
    #[must_use]
    pub fn costs(&self) -> &[(usize, f64)] {
        &self.costs
    }

    /// The largest block, in bytes, such that a read from it and from every
    /// smaller block measured costs no more than `slack` times the
    /// cheapest; the smallest block measured if `slack` is less than one.
    #[must_use]
    pub fn block_within(&self, slack: f64) -> usize {
        let cheapest = self
            .costs
            .iter()
            .map(|&(_, cost)| cost)
            .fold(f64::INFINITY, f64::min);
        self.costs
            .iter()
            .take_while(|&&(_, cost)| cost <= slack * cheapest)
            .last()
            .or(self.costs.first())
            .map_or(FIRST_BLOCK, |&(bytes, _)| bytes)
    }
}

/// Measures what a read at random costs on this machine with `threads`
/// threads reading at once, each from a block of its own, for blocks
/// doubling from 16 KB until the reads are dear.
///
/// `threads` is narrowed to the caller's [`budget`](crate::parallel::budget),
/// and zero is one. A measurement takes 20 ms a block and the making of the
/// blocks, a fifth of a second or so, and memory to a gigabyte while it
/// runs. Nothing is kept: a caller that wants it once keeps what it is
/// given, and one that doubts it measures again.
///
/// ```
/// use rump::parallelism::gather_costs;
///
/// let costs = gather_costs(2);
/// assert_eq!(costs.costs()[0].0, 16 << 10);
/// // A block within twice the cheapest is no smaller than one within
/// // once and a half.
/// assert!(costs.block_within(2.0) >= costs.block_within(1.5));
/// ```
#[must_use]
pub fn gather_costs(threads: usize) -> GatherCosts {
    let threads = threads.clamp(1, crate::parallel::budget());
    let mut dearness: Vec<(usize, f64)> = Vec::new();
    let mut small = Vec::new();
    let mut cheapest = f64::INFINITY;
    let mut bytes = FIRST_BLOCK;
    while bytes <= LAST_BLOCK && bytes.saturating_mul(threads) <= MOST_MEMORY {
        let readers = read_costs(bytes, threads);
        let dearer = median(readers.iter().map(|&(block, small)| block / small));
        small.extend(readers.iter().map(|&(_, small)| small));
        dearness.push((bytes, dearer));
        cheapest = cheapest.min(dearer);
        if dearer > DEAR * cheapest {
            break;
        }
        bytes *= 2;
    }
    let small = median(small.into_iter());
    let costs = dearness
        .into_iter()
        .map(|(bytes, dearer)| (bytes, dearer * small))
        .collect();
    GatherCosts { threads, costs }
}

/// The median of `values`, of which there is one at least.
fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut values: Vec<f64> = values.collect();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Nanoseconds a read from a block of `bytes` bytes and from one of
/// [`FIRST_BLOCK`], for each of `threads` threads reading at once for
/// [`LONG`], every thread from blocks of its own: its shortest pass over
/// each.
fn read_costs(bytes: usize, threads: usize) -> Vec<(f64, f64)> {
    let words = |bytes: usize| (bytes / core::mem::size_of::<u64>()) as u64;
    let together = Barrier::new(threads);
    std::thread::scope(|scope| {
        let readers: Vec<_> = (0..threads)
            .map(|thread| {
                let together = &together;
                scope.spawn(move || {
                    // Written, every word: a block left as the allocator
                    // zeroed it may be one page of zeros mapped throughout.
                    let block: Vec<u64> = (0..words(bytes)).map(mix).collect();
                    let small: Vec<u64> = (0..words(FIRST_BLOCK)).map(mix).collect();
                    let seed = mix((thread as u64) << 32 | words(bytes));
                    let mut places: [u64; CHAINS] =
                        core::array::from_fn(|chain| mix(seed ^ chain as u64) | 1);
                    let mut read = pass(&block, &mut places);
                    together.wait();
                    let began = Instant::now();
                    let mut shortest = (Duration::MAX, Duration::MAX);
                    while began.elapsed() < LONG {
                        let clock = Instant::now();
                        read ^= pass(&block, &mut places);
                        shortest.0 = shortest.0.min(clock.elapsed());
                        // The block's pass has put the small block out of
                        // the core's cache: a pass to bring it back, and
                        // the pass that is timed.
                        read ^= pass(&small, &mut places);
                        let clock = Instant::now();
                        read ^= pass(&small, &mut places);
                        shortest.1 = shortest.1.min(clock.elapsed());
                    }
                    core::hint::black_box(read);
                    let each = |passed: Duration| passed.as_secs_f64() * 1e9 / READS as f64;
                    (each(shortest.0), each(shortest.1))
                })
            })
            .collect();
        readers
            .into_iter()
            .map(|reader| reader.join().expect("a reader does not panic"))
            .collect()
    })
}

/// The chains of reads a pass runs side by side. A read waits on memory and
/// the next place does not wait on the read, so a core has several in
/// flight; four is what the solver's products run, and the body names them.
const CHAINS: usize = 4;

/// One pass: the XOR of the block's words at [`READS`] places that no pass
/// repeats. The places are made as they are used and not listed, so that
/// what is read from is the block, and not the part of it a list names.
fn pass(block: &[u64], places: &mut [u64; CHAINS]) -> u64 {
    let words = block.len() as u64;
    let [p0, p1, p2, p3] = places;
    let (mut a, mut b, mut c, mut d) = (0u64, 0u64, 0u64, 0u64);
    for _ in 0..READS / CHAINS {
        a ^= block[place(p0, words)];
        b ^= block[place(p1, words)];
        c ^= block[place(p2, words)];
        d ^= block[place(p3, words)];
    }
    a ^ b ^ c ^ d
}

/// Where in a block of `words` words a chain's next read falls: a step of
/// Lehmer's generator with the multiplier of Steele and Vigna (*Computationally
/// easy, spectrally good multipliers for congruential pseudorandom number
/// generators*, Software: Practice and Experience 52 (2022), 443–458), and
/// its high half scaled to the block.
fn place(state: &mut u64, words: u64) -> usize {
    *state = state.wrapping_mul(0xda94_2042_e4dd_58b5);
    (((*state >> 32) * words) >> 32) as usize
}

/// The `fmix64` finalizer of MurmurHash3 (Appleby): a word of a block, or a
/// chain's first place, from a count.
fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_blocks_double_from_the_first_and_every_cost_is_a_cost() {
        let costs = gather_costs(2);
        assert_eq!(costs.threads(), 2.min(crate::parallel::budget()));
        assert!(!costs.costs().is_empty());
        for (index, &(bytes, cost)) in costs.costs().iter().enumerate() {
            assert_eq!(bytes, FIRST_BLOCK << index);
            assert!(cost.is_finite() && cost > 0.0, "{bytes} bytes: {cost} ns");
        }
        let &(last, _) = costs.costs().last().expect("non-empty");
        assert!(last <= LAST_BLOCK && last * costs.threads() <= MOST_MEMORY);
    }

    #[test]
    fn the_threads_are_the_budgets_at_most_and_one_at_least() {
        crate::parallel::with_budget(1, || {
            assert_eq!(gather_costs(8).threads(), 1);
            assert_eq!(gather_costs(0).threads(), 1);
        });
    }

    #[test]
    fn a_block_within_a_slack_is_the_last_before_the_first_beyond_it() {
        // Costs as 128 threads of two EPYC 7452 had them from 16 KB, each
        // block timed by itself.
        let measured = [0.81, 0.84, 0.80, 0.87, 1.00, 1.37, 1.28, 3.43, 30.63];
        let costs = GatherCosts {
            threads: 128,
            costs: measured
                .iter()
                .enumerate()
                .map(|(index, &cost)| (FIRST_BLOCK << index, cost))
                .collect(),
        };
        assert_eq!(costs.block_within(2.0), 1 << 20);
        assert_eq!(costs.block_within(5.0), 2 << 20);
        assert_eq!(costs.block_within(100.0), 4 << 20);
        // 512 KB is beyond once and a half, and the megabyte after it is
        // not within it for being cheaper.
        assert_eq!(costs.block_within(1.5), 256 << 10);
        // The cheapest is the third block, and the first is not as cheap.
        assert_eq!(costs.block_within(1.0), FIRST_BLOCK);
        assert_eq!(costs.block_within(0.5), FIRST_BLOCK);
    }

    #[test]
    fn a_place_is_in_the_block_and_the_places_cover_it() {
        /// Words of the block; not a power of two.
        const WORDS: u64 = 1_000;
        /// Places drawn: enough that each word is expected sixty-four
        /// times.
        const DRAWN: usize = 64_000;
        let mut state = mix(1) | 1;
        let mut seen = vec![false; WORDS as usize];
        for _ in 0..DRAWN {
            seen[place(&mut state, WORDS)] = true;
        }
        assert!(seen.iter().all(|&seen| seen));
    }

    /// The costs on this machine, for the record: one thread, a quarter of
    /// the budget, half, and the whole.
    #[test]
    #[ignore = "timing probe: what a read at random costs here"]
    fn gather_costs_timing() {
        let whole = crate::parallel::budget();
        let mut counts = vec![1, whole.div_ceil(4), whole.div_ceil(2), whole];
        counts.dedup();
        for threads in counts {
            let began = Instant::now();
            let costs = gather_costs(threads);
            let took = began.elapsed();
            let table: Vec<String> = costs
                .costs()
                .iter()
                .map(|&(bytes, cost)| format!("{} KB {cost:.2}", bytes >> 10))
                .collect();
            eprintln!(
                "{threads} threads, measured in {took:.2?}: within 1.5 {} KB, within 2 {} KB: {}",
                costs.block_within(1.5) >> 10,
                costs.block_within(2.0) >> 10,
                table.join(", ")
            );
        }
    }
}
