//! Linear algebra over GF(2): dense null space, singleton pruning,
//! structured elimination on sparse rows ([`filter`]), and Block Lanczos.
//!
//! Solving `Mx = 0` over GF(2) for a large sparse `M` arises in integer
//! factoring, index-calculus discrete logarithms, coding theory, and anywhere
//! a parity system gets large. The matrix belongs to the caller's problem;
//! the solver does not.
//!
//! Addition is XOR, so a whole row combines in one pass of 64-bit words, and
//! there is no pivoting for numerical stability: any non-zero entry will do.
//!
//! # The packing contract
//!
//! A row is a `&[u64]` holding one bit per column: column `c` lives at bit
//! `c % 64` of word `c / 64`, least significant bit first. Bits at or beyond
//! the declared `columns` are ignored, so a stray high bit cannot masquerade
//! as a column. The type system does not enforce this.
//!
//! This module is distinct from [`finite_field`](crate::finite_field), which
//! is arithmetic *in* the field GF(2^m); here GF(2) is the field the linear
//! algebra happens over.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, RwLock};
use std::thread::Thread;
use std::time::{Duration, Instant};

use crate::random::RandomSource;

#[path = "gf2/filter.rs"]
mod filter;
pub use filter::{filter_merge, FilteredMatrix, SparseMatrix};

/// Bits per storage word.
const WORD: usize = 64;

/// Words needed to hold `bits` bits.
const fn words_for(bits: usize) -> usize {
    bits.div_ceil(WORD)
}

/// The column indices a packed row sets, below `columns`.
fn set_bits(row: &[u64], matrix_words: usize, columns: usize) -> impl Iterator<Item = usize> + '_ {
    (0..matrix_words).flat_map(move |word| {
        let mut bits = row.get(word).copied().unwrap_or(0);
        // A row may carry bits above the declared width; they are not columns
        // and must not be counted as occupants.
        if word == matrix_words - 1 && !columns.is_multiple_of(WORD) {
            bits &= (1u64 << (columns % WORD)) - 1;
        }
        core::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let bit = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            Some(word * WORD + bit)
        })
    })
}

/// Disjoint mutable views of two distinct rows, so one can be XORed into the
/// other without cloning either.
fn borrow_two(rows: &mut [Vec<u64>], first: usize, second: usize) -> (&mut [u64], &mut [u64]) {
    debug_assert_ne!(first, second, "rows must be distinct");
    if first < second {
        let (head, tail) = rows.split_at_mut(second);
        (&mut head[first], &mut tail[0])
    } else {
        let (head, tail) = rows.split_at_mut(first);
        (&mut tail[0], &mut head[second])
    }
}

/// Every linear dependence among `rows`, as lists of row indices.
///
/// A returned dependency is a non-empty set of row indices whose vectors XOR
/// to zero. The dependencies are independent as a set — there are exactly
/// `rows.len() − rank` of them — so a caller whose first dependency is useless
/// has genuinely different ones to try next.
///
/// Gauss–Jordan with an appended identity block: each working row records
/// which original rows have been folded into it, so when a row's matrix part
/// reaches zero the identity part names the dependent set. Full reduction
/// rather than echelon form, which is the same order of work and leaves the
/// dependent rows exactly zero.
///
/// Cost is `O(columns · rows · (rows + columns)/64)` word operations, cubic
/// and blind to sparsity. [`prune_singletons`] first, and
/// [`block_lanczos_dependencies`] instead once the matrix is large.
#[must_use]
pub fn dense_null_space(rows: &[Vec<u64>], columns: usize) -> Vec<Vec<usize>> {
    let count = rows.len();
    if count == 0 {
        return Vec::new();
    }

    let matrix_words = words_for(columns);
    let identity_words = words_for(count);
    let total_words = matrix_words + identity_words;

    // Each working row is its matrix part followed by an identity part; the
    // identity part accumulates exactly which original rows were XORed in.
    let mut work: Vec<Vec<u64>> = Vec::with_capacity(count);
    for (index, row) in rows.iter().enumerate() {
        let mut augmented = vec![0u64; total_words];
        let take = matrix_words.min(row.len());
        augmented[..take].copy_from_slice(&row[..take]);
        // Mask off anything above the declared width, so a stray high bit
        // cannot masquerade as a column.
        if !columns.is_multiple_of(WORD) && matrix_words > 0 {
            augmented[matrix_words - 1] &= (1u64 << (columns % WORD)) - 1;
        }
        augmented[matrix_words + index / WORD] |= 1u64 << (index % WORD);
        work.push(augmented);
    }

    // Forward elimination: one pivot per column, cleared from every other row.
    let mut pivot = 0usize;
    for column in 0..columns {
        let word = column / WORD;
        let mask = 1u64 << (column % WORD);

        let Some(found) = (pivot..count).find(|&row| work[row][word] & mask != 0) else {
            continue; // A free column: no pivot, and a dependency lives here.
        };
        work.swap(pivot, found);

        for row in 0..count {
            if row != pivot && work[row][word] & mask != 0 {
                let (source, target) = borrow_two(&mut work, pivot, row);
                for (t, s) in target.iter_mut().zip(source.iter()) {
                    *t ^= *s;
                }
            }
        }
        pivot += 1;
        if pivot == count {
            break;
        }
    }

    // A row whose matrix part vanished is a combination of original rows
    // summing to zero; the identity part says which.
    work.iter()
        .filter(|row| row[..matrix_words].iter().all(|word| *word == 0))
        .map(|row| {
            (0..count)
                .filter(|index| row[matrix_words + index / WORD] & (1u64 << (index % WORD)) != 0)
                .collect()
        })
        .filter(|indices: &Vec<usize>| !indices.is_empty())
        .collect()
}

/// A matrix with the rows and columns that cannot matter removed.
///
/// See [`prune_singletons`]. The three parts must agree — one original index
/// per surviving row, every row packed for [`Self::columns`] — so they are
/// read through accessors rather than exposed as fields a caller could put
/// out of step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrunedMatrix {
    rows: Vec<Vec<u64>>,
    columns: usize,
    original: Vec<usize>,
}

impl PrunedMatrix {
    /// Rows of the smaller matrix, bit-packed over [`Self::columns`].
    #[must_use]
    pub fn rows(&self) -> &[Vec<u64>] {
        &self.rows
    }

    /// Width of the smaller matrix.
    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// `original()[i]` is the caller's index for surviving row `i`, so a
    /// dependency over the pruned matrix maps straight back.
    #[must_use]
    pub fn original(&self) -> &[usize] {
        &self.original
    }
}

/// Drops the rows that can appear in no dependency, and the columns left
/// empty behind them.
///
/// A column with exactly one row set in it pins that row out of every
/// dependency: a dependency sums to zero, and nothing else carries that column
/// to cancel it. Delete the row — which can leave another column with a single
/// occupant — and repeat to a fixpoint. Then any column no surviving row
/// touches constrains nothing, so it goes too, and the remaining columns are
/// renumbered to close the gaps.
///
/// The null spaces correspond exactly: nothing removed could have been in a
/// dependency, so every dependency of the original survives, and every
/// dependency of the pruned matrix is one of the original's under
/// [`PrunedMatrix::original`].
///
/// Worth doing because [`dense_null_space`] is cubic in the width and blind to
/// sparsity, while a sieve matrix is mostly columns that cannot participate: a
/// prime `p` divides a sieve value about once in `p`, so past a certain size
/// every prime in the base appears in less than one relation, and those
/// columns are singletons and empties the solver would pay for cubically.
///
/// The pass itself is linear in the set bits. Per column it keeps how many
/// live rows set it and the XOR of their indices; while the count is one that
/// XOR *is* the surviving index, which makes finding a singleton's row a
/// lookup rather than a scan.
#[must_use]
pub fn prune_singletons(rows: &[Vec<u64>], columns: usize) -> PrunedMatrix {
    let count = rows.len();
    let matrix_words = words_for(columns);

    let mut occupants = vec![0usize; columns];
    let mut which = vec![0usize; columns];
    for (index, row) in rows.iter().enumerate() {
        for column in set_bits(row, matrix_words, columns) {
            occupants[column] += 1;
            which[column] ^= index;
        }
    }

    let mut live = vec![true; count];
    let mut pending: Vec<usize> = (0..columns).filter(|&c| occupants[c] == 1).collect();
    while let Some(column) = pending.pop() {
        if occupants[column] != 1 {
            continue; // already resolved by an earlier removal
        }
        let victim = which[column];
        if !core::mem::replace(&mut live[victim], false) {
            continue;
        }
        for touched in set_bits(&rows[victim], matrix_words, columns) {
            occupants[touched] -= 1;
            which[touched] ^= victim;
            if occupants[touched] == 1 {
                pending.push(touched);
            }
        }
    }

    // Renumber the columns anything still touches.
    let mut mapping = vec![usize::MAX; columns];
    let mut width = 0usize;
    for column in 0..columns {
        if occupants[column] > 0 {
            mapping[column] = width;
            width += 1;
        }
    }

    let mut original = Vec::new();
    let mut reduced = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        if !live[index] {
            continue;
        }
        let mut packed = vec![0u64; words_for(width)];
        for column in set_bits(row, matrix_words, columns) {
            let target = mapping[column];
            debug_assert_ne!(target, usize::MAX, "a live row touches a dropped column");
            packed[target / WORD] |= 1u64 << (target % WORD);
        }
        original.push(index);
        reduced.push(packed);
    }

    PrunedMatrix {
        rows: reduced,
        columns: width,
        original,
    }
}

// ─── Block Lanczos ─────────────────────────────────────────────────────
// Block Lanczos over GF(2): the null space without the cube.
//
// [`dense_null_space`] is Gauss–Jordan, which costs
// `O(columns · rows · (rows + columns)/64)` and is blind to sparsity. Relation
// matrices, such as a sieve's, carry a few dozen set bits per row across many
// thousands of columns, and at that size elimination dominates.
//
// Block Lanczos costs `O(iterations · nonzeros)` with `iterations ≈ rows/64`,
// because sixty-four vectors ride in the bits of one machine word and every
// iteration advances all of them. For ten thousand rows that is about 160
// iterations where a scalar method needs `2·rows ≈ 20 000`.
//
// # Equations and state
//
// Montgomery, *A Block Lanczos Algorithm for Finding Dependencies over
// GF(2)*, EUROCRYPT '95, LNCS 921, 106–120: the recurrence (18)–(19), the
// solution (20) and the subspace selection of figure 1. A wrong coefficient
// yields no dependencies rather than wrong ones, so the mapping is stated
// here and its invariants are tested step by step.
//
// Representation: `N = 64`. An `n × N` block is a `Vec<u64>` whose word `r`
// holds row `r`, so lane `j` is column `j`. An `N × N` matrix is a `Small`
// whose word `l` holds row `l`. `dot(P, Q)` is `PᵀQ`, `mul(P, Q)` is `PQ`,
// and masking every row of a product by `mask` multiplies it on the right by
// `SᵢSᵢᵀ`. Over GF(2) subtraction is addition, so the paper's signs vanish.
//
// | Paper                                   | Here              | Formed                         |
// |-----------------------------------------|-------------------|--------------------------------|
// | `A = MᵀM`                               | `matrix.apply`    | applied, never stored          |
// | `Y`, `V₀ = AY`                          | `x` (start), `q`  | once                           |
// | `Vᵢ`, `Vᵢ₋₁`, `Vᵢ₋₂`                    | `v0`, `v1`, `v2`  | shifted after (18)             |
// | `AVᵢ`                                   | `av0`             | after `Vᵢ`                     |
// | `Tᵢ = VᵢᵀAVᵢ`, `Tᵢ₋₁`                   | `t0`, `t1`        | after `AVᵢ`                    |
// | `Sᵢ` (lanes)                            | `mask`            | `invert(Tᵢ, Sᵢ₋₁)`, figure 1   |
// | `Winvᵢ = Sᵢ(SᵢᵀTᵢSᵢ)⁻¹Sᵢᵀ`, `i−1`, `i−2` | `w0i`, `w1i`, `w2i` | `invert`                     |
// | `Gᵢ = Vᵢ₋₁ᵀA²Vᵢ₋₁Sᵢ₋₁Sᵢ₋₁ᵀ + Tᵢ₋₁`       | `g` before update | read by `F`                    |
// | `Gᵢ₊₁ = VᵢᵀA²VᵢSᵢSᵢᵀ + Tᵢ`                | `g` after update  | `(AVᵢ)ᵀ(AVᵢ)` masked, plus `Tᵢ` |
// | `Dᵢ₊₁ = I − Winvᵢ Gᵢ₊₁`                   | `d`               | (19)                           |
// | `Eᵢ₊₁ = −Winvᵢ₋₁ TᵢSᵢSᵢᵀ`                 | `e`               | (19)                           |
// | `Fᵢ₊₁ = −Winvᵢ₋₂(I − Tᵢ₋₁Winvᵢ₋₁)GᵢSᵢSᵢᵀ`  | `f`               | (19)                           |
// | `Vᵢ₊₁ = AVᵢSᵢSᵢᵀ + VᵢDᵢ₊₁ + Vᵢ₋₁Eᵢ₊₁ + Vᵢ₋₂Fᵢ₊₁` | `recurrence` | (18)                      |
// | `X = Y + Σ VᵢWinvᵢVᵢᵀV₀`                  | `x`               | (20), so `AX = AY + V₀ = 0`    |
//
// Invariants, tested on small matrices against dense arithmetic
// (`the_recurrence_keeps_montgomerys_invariants`): `WᵢᵀAWⱼ = 0` for `i ≠ j`
// with `Wᵢ = VᵢSᵢ`; `Winvᵢ` inverts `Tᵢ` on `Sᵢ`; `WⱼᵀAVᵢ₊₁ = 0` for every
// `j ≤ i`; and the dependencies returned span the null space found by dense
// elimination.
//
// # The two things that make it delicate
//
// `Vᵢᵀ A Vᵢ` is usually singular over `GF(2)`, which is why a block method is
// needed at all: [`invert`] selects a subspace `Sᵢ` on which it is not, and
// gives the pseudo-inverse `Winvᵢ = Sᵢ (Sᵢᵀ Vᵢᵀ A Vᵢ Sᵢ)⁻¹ Sᵢᵀ`. Indices left
// out of `Sᵢ` get precedence next time; if one sits out two rounds running
// while `V` is non-zero there, the run has stalled and is abandoned.
//
// And `A = MᵀM` is symmetric, but its kernel is *larger* than `M`'s: over
// `GF(2)` a non-zero vector can be self-orthogonal, so `Ax = 0` does not give
// `Mx = 0`. The iteration therefore ends with 128 candidates — the columns of
// `X` and of the final `V` — and a small elimination picks out the
// combinations that `M` really does annihilate.
//
// # Safety
//
// Nothing here is trusted. Every vector returned is checked to be a genuine
// dependency of the caller's rows, and [`block_lanczos_dependencies`] returns `None` rather
// than anything doubtful, leaving the caller on the exact solver. A wrong
// answer is not among the outcomes.

/// Bits per word, and the block width: sixty-four vectors advance together.
const WIDTH: usize = 64;

/// A dense `64 × 64` matrix over `GF(2)`, one word per row.
type Small = [u64; WIDTH];

/// Lists of indices stored flat: the indices of list `i` are
/// `indices[offsets[i]..offsets[i + 1]]`. One allocation read in order, so
/// the matrix product, which dominates a Lanczos run, takes no cache miss per
/// list.
struct Lists {
    offsets: Vec<usize>,
    indices: Vec<u32>,
}

impl Lists {
    fn from_lists(lists: &[Vec<u32>]) -> Self {
        let mut offsets = Vec::with_capacity(lists.len() + 1);
        let mut indices = Vec::with_capacity(lists.iter().map(Vec::len).sum());
        offsets.push(0);
        for list in lists {
            indices.extend_from_slice(list);
            offsets.push(indices.len());
        }
        Self { offsets, indices }
    }

    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn list(&self, index: usize) -> &[u32] {
        &self.indices[self.offsets[index]..self.offsets[index + 1]]
    }

    /// Where to cut the lists into at most `parts` runs of about as many
    /// indices each: the bounds of the runs, from zero to [`Self::len`],
    /// climbing. A fold costs what its list is long, and a sieve's matrix
    /// has most of its entries in the columns of its smallest primes, so
    /// runs of as many lists each would give one worker most of the work.
    fn cuts(&self, parts: usize) -> Vec<usize> {
        let total = self.indices.len();
        let parts = parts.clamp(1, self.len().max(1));
        let mut cuts = vec![0];
        for part in 1..parts {
            // The first list that ends past this part's share.
            let share = total / parts * part + total % parts * part / parts;
            let cut = self.offsets.partition_point(|&offset| offset < share);
            let cut = cut.min(self.len());
            if cut > *cuts.last().expect("non-empty") {
                cuts.push(cut);
            }
        }
        if self.len() > *cuts.last().expect("non-empty") {
            cuts.push(self.len());
        }
        cuts
    }
}

/// The relation matrix `M`, held once by rows and once by columns.
///
/// Both orientations are needed every iteration — `A = MᵀM` is two products —
/// and each is a gather over the side it is indexed by, so storing both costs
/// one extra copy of the indices and saves a scatter with random writes.
struct Sparse {
    /// For each relation, the columns it sets.
    by_relation: Arc<Lists>,
    /// For each column, the relations that set it.
    by_column: Arc<Lists>,
    /// The bounds of the runs of relations a pass is taken in, each of
    /// about as many entries, and of the runs of columns.
    relation_runs: Arc<Vec<usize>>,
    column_runs: Arc<Vec<usize>>,
    /// Threads kept for the whole Lanczos recurrence, so no `A·x` spawns
    /// any.
    folds: FoldPool,
    /// How many of them each kind of pass runs on: `M·x`, `Mᵀ·y` with the
    /// inner products, and the recurrence.
    forward_pace: Mutex<Pace>,
    backward_pace: Mutex<Pace>,
    step_pace: Mutex<Pace>,
}

impl Sparse {
    fn from_packed(rows: &[Vec<u64>], columns: usize, threads: usize) -> Self {
        let words = columns.div_ceil(WIDTH);
        let by_relation = rows
            .iter()
            .map(|row| {
                set_bits(row, words, columns)
                    .map(|column| column as u32)
                    .collect()
            })
            .collect();
        Self::from_lists(by_relation, columns, threads)
    }

    /// From each relation's ascending column list, of `columns` columns.
    ///
    /// The columns are numbered again as those some relation sets, in their
    /// order. A column none sets is a zero of `M·x` that `Mᵀ` never reads,
    /// so `A` is the same `A`; and a matrix that has been filtered has ten
    /// of them to each that is set, which made the block of `M·x` eleven
    /// times as long and gave the fold of the empty lists to one worker.
    fn from_lists(mut by_relation: Vec<Vec<u32>>, columns: usize, threads: usize) -> Self {
        assert!(
            u32::try_from(by_relation.len()).is_ok() && u32::try_from(columns).is_ok(),
            "{} rows by {columns} columns: the solver indexes both in thirty-two bits",
            by_relation.len()
        );
        let mut number = vec![0u32; columns];
        for &column in by_relation.iter().flatten() {
            number[column as usize] = 1;
        }
        let mut set = 0;
        for entry in &mut number {
            set += core::mem::replace(entry, set);
        }
        for column in by_relation.iter_mut().flatten() {
            *column = number[*column as usize];
        }
        let columns = set as usize;
        let mut by_column = vec![Vec::new(); columns];
        for (index, row) in by_relation.iter().enumerate() {
            for &column in row {
                by_column[column as usize].push(index as u32);
            }
        }
        let useful = (by_relation.len().max(columns) / MINIMUM_FOLDS_PER_WORKER).max(1);
        let threads = threads.min(crate::parallel::budget()).min(useful).max(1);
        let by_relation = Lists::from_lists(&by_relation);
        let by_column = Lists::from_lists(&by_column);
        let runs = if threads == 1 {
            1
        } else {
            threads * RUNS_PER_THREAD
        };
        Self {
            relation_runs: Arc::new(by_relation.cuts(runs)),
            column_runs: Arc::new(by_column.cuts(runs)),
            by_relation: Arc::new(by_relation),
            by_column: Arc::new(by_column),
            folds: FoldPool::new(threads),
            forward_pace: Mutex::new(Pace::new(threads)),
            backward_pace: Mutex::new(Pace::new(threads)),
            step_pace: Mutex::new(Pace::new(threads)),
        }
    }

    fn relations(&self) -> usize {
        self.by_relation.len()
    }

    fn columns(&self) -> usize {
        self.by_column.len()
    }

    /// `M·x` into `image`: a block over the relations becomes one over the
    /// columns.
    fn forward(&self, x: &Arc<Block>, image: &Arc<Block>) {
        let (lists, runs) = (&self.by_column, &self.column_runs);
        self.folds.fold(&self.forward_pace, lists, runs, x, image);
    }

    /// `Mᵀ·y` into `out`: a block over the columns becomes one over the
    /// relations.
    fn backward(&self, y: &Arc<Block>, out: &Arc<Block>) {
        let (lists, runs) = (&self.by_relation, &self.relation_runs);
        self.folds.fold(&self.backward_pace, lists, runs, y, out);
    }

    /// `A·x` with `A = MᵀM`, the symmetric operator the iteration runs on.
    fn apply(&self, x: &Arc<Block>) -> Arc<Block> {
        let image = Block::zeroed(self.columns());
        let out = Block::zeroed(self.relations());
        self.forward(x, &image);
        self.backward(&image, &out);
        out
    }

    /// `A·v` into `av`, by way of `M·v` into `image`, and with it the three
    /// inner products an iteration takes of `V`: `VᵀAV`, `(AV)ᵀAV` and
    /// `VᵀQ`.
    ///
    /// Each is a sum over the relations, so the thread that folds a run of
    /// `A·v` takes the run's share of all three while the words are in its
    /// cache, and the shares are XORed. Taken apart, on one thread, the
    /// three cost more than the two products they follow.
    fn apply_with_products(
        &self,
        v: &Arc<Block>,
        q: &Arc<Block>,
        image: &Arc<Block>,
        av: &Arc<Block>,
    ) -> Products {
        self.forward(v, image);
        let shares: Arc<Vec<Mutex<Products>>> = Arc::new(
            (0..self.folds.threads())
                .map(|_| Mutex::default())
                .collect(),
        );
        let (lists, runs) = (
            Arc::clone(&self.by_relation),
            Arc::clone(&self.relation_runs),
        );
        let (v, q) = (Arc::clone(v), Arc::clone(q));
        let (image, av) = (Arc::clone(image), Arc::clone(av));
        let next = AtomicUsize::new(0);
        let made = Arc::clone(&shares);
        self.folds.pass(&self.backward_pace, move |thread| {
            let (mut t, mut squared, mut projection) = (Dot::new(), Dot::new(), Dot::new());
            while let Some((start, end)) = run(&runs, &next) {
                let folded = fold_range(&lists, start, end, &image);
                for ((&folded, v), q) in folded
                    .iter()
                    .zip(v.range(start, end))
                    .zip(q.range(start, end))
                {
                    t.add(v, folded);
                    squared.add(folded, folded);
                    projection.add(v, q);
                }
                av.write(start, &folded);
            }
            let share = Products {
                t: t.product(),
                squared: squared.product(),
                projection: projection.product(),
            };
            *made[thread].lock().unwrap_or_else(PoisonError::into_inner) = share;
        });
        let mut products = Products::default();
        for share in shares.iter() {
            products.add(&share.lock().unwrap_or_else(PoisonError::into_inner));
        }
        products
    }

    /// Equations (20) and (18) in one pass over the relations: `X + V·P`
    /// into `x`, where it stood, and the next `V` into `next`.
    fn step(
        &self,
        x: &Arc<Block>,
        projected: &Small,
        av: &Arc<Block>,
        terms: [&Arc<Block>; 3],
        advance: Recurrence,
        next: &Arc<Block>,
    ) {
        let runs = Arc::clone(&self.relation_runs);
        let (x, av, next) = (Arc::clone(x), Arc::clone(av), Arc::clone(next));
        let [v0, v1, v2] = terms.map(Arc::clone);
        let solve = SmallProduct::new(projected);
        let taken = AtomicUsize::new(0);
        self.folds.pass(&self.step_pace, move |_| {
            while let Some((start, end)) = run(&runs, &taken) {
                let words = av
                    .range(start, end)
                    .zip(v0.range(start, end))
                    .zip(v1.range(start, end))
                    .zip(v2.range(start, end));
                for (index, (((av, v0), v1), v2)) in (start..end).zip(words) {
                    x.set(index, x.word(index) ^ solve.apply(v0));
                    next.set(index, advance.apply(av, v0, v1, v2));
                }
            }
        });
    }
}

/// The inner products an iteration takes of `V` and `A·V`.
struct Products {
    /// `T = VᵀAV`.
    t: Small,
    /// `(AV)ᵀAV`.
    squared: Small,
    /// `VᵀQ`, `Q` the block the run started from.
    projection: Small,
}

impl Default for Products {
    fn default() -> Self {
        Self {
            t: [0; WIDTH],
            squared: [0; WIDTH],
            projection: [0; WIDTH],
        }
    }
}

impl Products {
    /// An inner product is a sum over the relations, and a range's share
    /// of it is added by XOR.
    fn add(&mut self, share: &Self) {
        xor_into(&mut self.t, &share.t);
        xor_into(&mut self.squared, &share.squared);
        xor_into(&mut self.projection, &share.projection);
    }
}

/// One block word per relation or column.
///
/// The words are atomic so that the threads of a pass write one block
/// between them, each its runs, where the next pass reads it: a block
/// gathered on the calling thread from the ranges the workers had made was a
/// third of an iteration. No word is written by two threads, none is read
/// in the pass that writes it but by its writer, and a pass has ended
/// before the next begins, so the ordering is relaxed.
struct Block(Vec<AtomicU64>);

impl Block {
    fn zeroed(len: usize) -> Arc<Self> {
        Self::of((0..len).map(|_| 0))
    }

    fn of(words: impl Iterator<Item = u64>) -> Arc<Self> {
        Arc::new(Self(words.map(AtomicU64::new).collect()))
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn word(&self, index: usize) -> u64 {
        self.0[index].load(Ordering::Relaxed)
    }

    fn set(&self, index: usize, word: u64) {
        self.0[index].store(word, Ordering::Relaxed);
    }

    /// The words from `start` up to `end`.
    fn range(&self, start: usize, end: usize) -> impl Iterator<Item = u64> + '_ {
        self.0[start..end]
            .iter()
            .map(|word| word.load(Ordering::Relaxed))
    }

    /// Every word.
    fn words(&self) -> impl Iterator<Item = u64> + '_ {
        self.range(0, self.len())
    }

    /// `words` written from `start` on.
    fn write(&self, start: usize, words: &[u64]) {
        for (slot, &word) in self.0[start..start + words.len()].iter().zip(words) {
            slot.store(word, Ordering::Relaxed);
        }
    }
}

/// The run `taken` counts to, of those `bounds` bounds, which is then
/// taken; none once they all are.
fn run(bounds: &[usize], taken: &AtomicUsize) -> Option<(usize, usize)> {
    let run = taken.fetch_add(1, Ordering::Relaxed);
    Some((*bounds.get(run)?, *bounds.get(run + 1)?))
}

/// The fewest output folds a thread of the pool is kept for.
///
/// A fold is a short XOR gather, and a pass wakes every thread of the pool,
/// so the pool is limited to give each at least this many, and a matrix too
/// small for two runs on the caller's thread alone. The value is a policy,
/// a power of two large enough that the waking is a small fraction of the
/// work; the crossover has not been measured, and a timing of one fold
/// against one waking on the build hosts would fix it.
const MINIMUM_FOLDS_PER_WORKER: usize = 4_096;

/// The runs a pass is cut into for each thread of the pool. The threads
/// take the runs from one count, so a thread that started late, or shares
/// its core, takes fewer, and the pass waits for the last run and not for
/// the slowest thread's share. Measured on the matrix of a 120-digit sieve,
/// 673 757 rows, on two EPYC 7452 and 128 threads: an iteration was 12.8 ms
/// at one run a thread, 11.3 at eight and 12.0 at thirty-two.
const RUNS_PER_THREAD: usize = 8;

/// The passes timed on one count of threads before the count is judged,
/// by the shortest of them: what else the machine is doing makes a pass
/// longer and never shorter. As many are run before any is timed, while a
/// solve's blocks are first written: on the matrix of a 120-digit sieve
/// the shortest of a solve's first sixteen passes was 6.1 ms and of later
/// sixteens 4.2 to 5.0. Sixteen is a policy, a power of two; a search of
/// three counts is then 48 passes of the ten thousand a solve of 670 000
/// rows takes.
const PACE_WINDOW: usize = 16;

/// The passes run on a count once it is settled on, before the counts are
/// tried again from the pool's whole. A search is cheap, and one that is
/// repeated need not be right every time: on the matrix of a 120-digit
/// sieve a search is 48 passes, sixteen of them on a count a fifth slower,
/// and of twelve searches eleven settled on 64 threads and one, a solve's
/// first, on 32. The solve took 120.0 s so, and 120.4 s on the pool's
/// whole with no search at all. A policy, a power of two.
const PACE_HOLD: usize = 1_024;

/// Half the threads are taken for the whole when their pass is no longer
/// than the shortest seen by more than this part of it. Measured on the
/// matrix of a 120-digit sieve, two EPYC 7452, 64 cores and 128 threads:
/// `M·x` took 4.89 ms on 128 threads, 4.95 on 64 and 5.78 on 32, a
/// hundredth more and then a sixth, and a thirty-second lies between.
const PACE_SLACK_DIVISOR: u32 = 32;

/// The workers each thread wakes as a pass begins: the caller the first of
/// them, and each of those the ones after it, so the last of 128 is woken
/// fourth in a chain and not last of 128 by the caller. On the same matrix
/// an iteration was 11.3 ms at four and 11.7 at sixteen.
const WAKES: usize = 4;

/// Independent XOR accumulators in [`fold_range`]: the gathers are what the
/// product waits on, and a single XOR chain would serialize them. Four is a
/// policy; the count that saturates the load ports has not been measured,
/// and the loop body names its accumulators, so changing it means changing
/// the body too.
const FOLD_ACCUMULATORS: usize = 4;

/// Rows per spare iteration allowed beyond `relations / WIDTH`.
///
/// Every iteration spans at most `WIDTH` new dimensions, so a converging
/// run takes at least `relations / WIDTH` of them, and Montgomery's
/// expected count is `n / (N − 0.76)` for `N = 64`: over `n / N` by
/// `0.76·n / (N·(N − 0.76))`, one iteration per 5,325 rows. The cap exists
/// to catch a run that never drives `T` to zero, and must never cut off a
/// converging one, so its spare grows with the matrix: one iteration per
/// 4,096 rows is the expected excess with a third to spare, and
/// `SPARE_ITERATIONS_FLOOR` covers the rounds a selection of fewer than
/// `WIDTH` lanes costs on any matrix at all.
const ROWS_PER_SPARE_ITERATION: usize = 4096;

/// Spare iterations granted to every run regardless of size: one full block
/// of rounds, plus sixteen for the lanes the last selections leave unused.
const SPARE_ITERATIONS_FLOOR: usize = WIDTH + 16;

/// How many threads a kind of pass runs on, found by timing it.
///
/// The threads that pay are the machine's to say: the products of a solve
/// are gathers from a block no core's cache holds, and two EPYC 7452 make
/// as many of them a second on 64 threads as on 128, where a machine of
/// more cores to its threads, or of faster memory, would not. So a pass is
/// run on the pool's whole, then on half, and on half again while it is no
/// slower, and keeps the fewest threads that were as fast as any; and
/// after [`PACE_HOLD`] passes it looks again, so that a search misled is
/// a search corrected. What a pass makes does not depend on its threads,
/// so nothing but its time does.
struct Pace {
    /// The pool's threads.
    most: usize,
    /// The threads the next pass runs on.
    threads: usize,
    /// What the passes are run for.
    looking: Looking,
    /// The passes left of this window, or of this hold.
    left: usize,
    /// The shortest pass of this window.
    shortest: Duration,
    /// The fewest threads found as fast as any, and the shortest pass
    /// seen.
    best: Option<(usize, Duration)>,
}

/// What a [`Pace`] runs its passes for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Looking {
    /// For the blocks to be written once, untimed.
    Warming,
    /// To time a count of threads.
    Trying,
    /// For the solve, on the count settled on.
    Settled,
}

impl Pace {
    fn new(most: usize) -> Self {
        Self {
            most,
            threads: most,
            looking: if most > 1 {
                Looking::Warming
            } else {
                Looking::Settled
            },
            left: PACE_WINDOW,
            shortest: Duration::MAX,
            best: None,
        }
    }

    /// A pass on [`Self::threads`] took `pass`.
    fn timed(&mut self, pass: Duration) {
        if self.most == 1 {
            return;
        }
        self.shortest = self.shortest.min(pass);
        self.left -= 1;
        if self.left > 0 {
            return;
        }
        let shortest = std::mem::replace(&mut self.shortest, Duration::MAX);
        self.left = PACE_WINDOW;
        if self.looking != Looking::Trying {
            self.looking = Looking::Trying;
            self.threads = self.most;
            self.best = None;
            return;
        }
        let as_fast = self
            .best
            .is_none_or(|(_, best)| shortest <= best + best / PACE_SLACK_DIVISOR);
        if as_fast {
            let shortest = self.best.map_or(shortest, |(_, best)| best.min(shortest));
            self.best = Some((self.threads, shortest));
        }
        if as_fast && self.threads > 1 {
            self.threads = self.threads.div_ceil(2);
        } else {
            self.threads = self.best.map_or(self.most, |(threads, _)| threads);
            self.looking = Looking::Settled;
            self.left = PACE_HOLD;
        }
    }
}

/// What a pass has each thread that runs it do, given the thread's number
/// among the pool's.
type Pass = Arc<dyn Fn(usize) + Send + Sync>;

/// The bits of [`Shared::begun`] that count the workers of a pass; the
/// bits above them count the passes.
const WORKERS_BITS: u32 = 32;

/// What the threads of a pool share.
struct Shared {
    /// The pass in hand.
    pass: RwLock<Option<Pass>>,
    /// The passes begun, and below [`WORKERS_BITS`] the workers the one in
    /// hand runs on, the first so many: one word, so that a worker reads
    /// the two of one pass. A worker runs the pass in hand when this has
    /// moved on from what it last read.
    begun: AtomicU64,
    /// The workers that have not finished the pass in hand.
    running: AtomicUsize,
    /// Set for the workers to return.
    stopping: AtomicBool,
    /// The thread that waits for the pass in hand.
    caller: Mutex<Option<Thread>>,
    /// The workers, for one to wake those after it.
    workers: OnceLock<Vec<Thread>>,
    /// What a worker's pass panicked with, for the caller to resume.
    panic: Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

impl Shared {
    /// A worker's life: each pass it is of as it begins, until the pool
    /// stops.
    fn work(&self, number: usize) {
        let mut read = 0;
        loop {
            let begun = self.begun.load(Ordering::Acquire);
            if begun == read {
                std::thread::park();
                continue;
            }
            read = begun;
            if self.stopping.load(Ordering::Acquire) {
                return;
            }
            let of_the_pass = (begun & ((1 << WORKERS_BITS) - 1)) as usize;
            if number >= of_the_pass {
                continue;
            }
            let workers = self.workers.get().expect("the workers are named first");
            let after = workers[..of_the_pass].iter().skip(WAKES * (number + 1));
            for worker in after.take(WAKES) {
                worker.unpark();
            }
            let pass = self
                .pass
                .read()
                .unwrap_or_else(PoisonError::into_inner)
                .clone();
            if let Some(pass) = pass {
                let ended = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pass(number)));
                if let Err(payload) = ended {
                    self.panic
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .get_or_insert(payload);
                }
            }
            if self.running.fetch_sub(1, Ordering::AcqRel) == 1 {
                let caller = self.caller.lock().unwrap_or_else(PoisonError::into_inner);
                if let Some(caller) = caller.as_ref() {
                    caller.unpark();
                }
            }
        }
    }
}

/// Threads kept for the lifetime of one sparse solve, the caller's among
/// them.
///
/// A pass is one closure that every thread of the pass runs, each taking
/// runs of the pass from a count they share until none is left. The
/// workers sleep between passes and are woken down a tree, [`WAKES`] by
/// each. What a pass makes it writes where it belongs, a word to a run's
/// thread, so the result is the result of the pass on one thread; only its
/// schedule changes, and how many threads it is run on is its [`Pace`]'s
/// to find.
struct FoldPool {
    shared: Arc<Shared>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl FoldPool {
    /// A pool of `threads` threads: the caller's, and a worker for each of
    /// the rest.
    fn new(threads: usize) -> Self {
        assert!(
            threads as u64 >> WORKERS_BITS == 0,
            "{threads} threads: a pool counts them in thirty-two bits"
        );
        let shared = Arc::new(Shared {
            pass: RwLock::new(None),
            begun: AtomicU64::new(0),
            running: AtomicUsize::new(0),
            stopping: AtomicBool::new(false),
            caller: Mutex::new(None),
            workers: OnceLock::new(),
            panic: Mutex::new(None),
        });
        let handles: Vec<_> = (0..threads.saturating_sub(1))
            .map(|number| {
                let shared = Arc::clone(&shared);
                std::thread::spawn(move || shared.work(number))
            })
            .collect();
        let workers = handles.iter().map(|handle| handle.thread().clone());
        shared
            .workers
            .set(workers.collect())
            .expect("the workers are named once");
        Self { shared, handles }
    }

    /// The pool's threads, the caller's among them.
    fn threads(&self) -> usize {
        self.handles.len() + 1
    }

    /// `pass` on as many threads of the pool as `pace` has it run on, each
    /// given its number among the pool's, the caller's the last; and
    /// `pace` told how long it took. A panic in it is resumed here once
    /// every thread has left it.
    fn pass(&self, pace: &Mutex<Pace>, pass: impl Fn(usize) + Send + Sync + 'static) {
        let began = Instant::now();
        let pace = || pace.lock().unwrap_or_else(PoisonError::into_inner);
        let workers = pace().threads.clamp(1, self.threads()) - 1;
        let caller = self.handles.len();
        if workers == 0 {
            pass(caller);
            pace().timed(began.elapsed());
            return;
        }
        let pass: Pass = Arc::new(pass);
        let shared = &self.shared;
        *shared.pass.write().unwrap_or_else(PoisonError::into_inner) = Some(Arc::clone(&pass));
        *shared.caller.lock().unwrap_or_else(PoisonError::into_inner) =
            Some(std::thread::current());
        shared.running.store(workers, Ordering::Release);
        let passes = (shared.begun.load(Ordering::Acquire) >> WORKERS_BITS) + 1;
        shared
            .begun
            .store(passes << WORKERS_BITS | workers as u64, Ordering::Release);
        for handle in self.handles[..workers].iter().take(WAKES) {
            handle.thread().unpark();
        }
        let ended = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pass(caller)));
        while shared.running.load(Ordering::Acquire) != 0 {
            std::thread::park();
        }
        *shared.pass.write().unwrap_or_else(PoisonError::into_inner) = None;
        if let Err(payload) = ended {
            std::panic::resume_unwind(payload);
        }
        let panicked = shared
            .panic
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(payload) = panicked {
            std::panic::resume_unwind(payload);
        }
        pace().timed(began.elapsed());
    }

    /// One output word per index list: the XOR-fold of `input` at those
    /// indices, written to `output`, the lists taken in the runs `runs`
    /// bounds.
    ///
    /// Each output word is an independent fold, so the result is the
    /// inline fold's whatever the threads and whichever took each run.
    fn fold(
        &self,
        pace: &Mutex<Pace>,
        lists: &Arc<Lists>,
        runs: &Arc<Vec<usize>>,
        input: &Arc<Block>,
        output: &Arc<Block>,
    ) {
        let (lists, runs) = (Arc::clone(lists), Arc::clone(runs));
        let (input, output) = (Arc::clone(input), Arc::clone(output));
        let taken = AtomicUsize::new(0);
        self.pass(pace, move |_| {
            while let Some((start, end)) = run(&runs, &taken) {
                output.write(start, &fold_range(&lists, start, end, &input));
            }
        });
    }
}

impl Drop for FoldPool {
    fn drop(&mut self) {
        self.shared.stopping.store(true, Ordering::Release);
        self.shared
            .begun
            .fetch_add(1 << WORKERS_BITS, Ordering::Release);
        for handle in &self.handles {
            handle.thread().unpark();
        }
        for handle in self.handles.drain(..) {
            if let Err(payload) = handle.join() {
                if !std::thread::panicking() {
                    std::panic::resume_unwind(payload);
                }
            }
        }
    }
}

fn fold_range(lists: &Lists, start: usize, end: usize, input: &Block) -> Vec<u64> {
    let mut out = Vec::with_capacity(end - start);
    for index in start..end {
        let indices = lists.list(index);
        let mut chunks = indices.chunks_exact(FOLD_ACCUMULATORS);
        let (mut a, mut b, mut c, mut d) = (0u64, 0u64, 0u64, 0u64);
        for chunk in &mut chunks {
            a ^= input.word(chunk[0] as usize);
            b ^= input.word(chunk[1] as usize);
            c ^= input.word(chunk[2] as usize);
            d ^= input.word(chunk[3] as usize);
        }
        let mut total = a ^ b ^ c ^ d;
        for &index in chunks.remainder() {
            total ^= input.word(index as usize);
        }
        out.push(total);
    }
    out
}

/// `leftᵀ · right` for two blocks, the `64 × 64` matrix of inner products,
/// taken a word of each at a time.
///
/// Eight tables, one per byte of the left word: entry `e` of table `b`
/// accumulates the right words whose left word has byte `b` equal to `e`.
/// Eight table updates per word, against a loop over the word's set bits —
/// thirty-two on average — and the tables then combine into the sixty-four
/// lanes in a fixed sixteen thousand operations. Each Lanczos iteration
/// takes three dot products over the whole block.
struct Dot {
    tables: Box<[[u64; 256]; 8]>,
}

impl Dot {
    fn new() -> Self {
        Self {
            tables: Box::new([[0; 256]; 8]),
        }
    }

    /// A word of the left block and the word of the right beside it.
    fn add(&mut self, mut left: u64, right: u64) {
        for table in self.tables.iter_mut() {
            table[(left & 0xff) as usize] ^= right;
            left >>= 8;
        }
    }

    fn product(&self) -> Small {
        let mut out = [0u64; WIDTH];
        for (byte, table) in self.tables.iter().enumerate() {
            for bit in 0..8 {
                let mut lane = 0u64;
                for (entry, &value) in table.iter().enumerate() {
                    if (entry >> bit) & 1 == 1 {
                        lane ^= value;
                    }
                }
                out[byte * 8 + bit] = lane;
            }
        }
        out
    }
}

/// `P·Q` for two `64 × 64` matrices.
fn mul(p: &Small, q: &Small) -> Small {
    let mut out = [0u64; WIDTH];
    for (slot, row) in out.iter_mut().zip(p.iter()) {
        let mut bits = *row;
        let mut total = 0u64;
        while bits != 0 {
            let lane = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            total ^= q[lane];
        }
        *slot = total;
    }
    out
}

/// Byte-sliced lookup table for right multiplication by one `64 × 64`
/// matrix.
///
/// A block word selects rows of the matrix to XOR. Walking its set bits costs
/// about 32 dependent iterations for the dense words Block Lanczos produces.
/// Split the selector into eight bytes instead: each byte indexes the XOR of
/// its eight possible rows, so applying the matrix is exactly eight lookups
/// and XORs. Building the 16 KiB table costs 2,040 XORs, amortized over one
/// word per relation.
struct SmallProduct {
    by_byte: [[u64; 256]; 8],
}

impl SmallProduct {
    fn new(matrix: &Small) -> Self {
        let mut by_byte = [[0u64; 256]; 8];
        for (byte, table) in by_byte.iter_mut().enumerate() {
            let rows = &matrix[byte * 8..byte * 8 + 8];
            for selector in 1usize..256 {
                let without_lowest = selector & (selector - 1);
                let lane = selector.trailing_zeros() as usize;
                table[selector] = table[without_lowest] ^ rows[lane];
            }
        }
        Self { by_byte }
    }

    fn apply(&self, mut value: u64) -> u64 {
        let mut total = 0u64;
        for table in &self.by_byte {
            total ^= table[(value & 0xff) as usize];
            value >>= 8;
        }
        total
    }
}

/// Equation (18), a word of each of the four blocks at a time.
///
/// The matrices are tiny and fixed for the whole pass, so their byte tables
/// are built once for it.
struct Recurrence {
    mask: u64,
    d: SmallProduct,
    e: SmallProduct,
    f: SmallProduct,
}

impl Recurrence {
    fn new(mask: u64, d: &Small, e: &Small, f: &Small) -> Self {
        Self {
            mask,
            d: SmallProduct::new(d),
            e: SmallProduct::new(e),
            f: SmallProduct::new(f),
        }
    }

    /// The word of `Vᵢ₊₁` from those of `AVᵢ`, `Vᵢ`, `Vᵢ₋₁` and `Vᵢ₋₂`.
    fn apply(&self, av: u64, v0: u64, v1: u64, v2: u64) -> u64 {
        (av & self.mask) ^ self.d.apply(v0) ^ self.e.apply(v1) ^ self.f.apply(v2)
    }
}

/// `P + I`.
fn plus_identity(p: &mut Small) {
    for (lane, row) in p.iter_mut().enumerate() {
        *row ^= 1u64 << lane;
    }
}

/// `a ^= b`, elementwise.
fn xor_into(a: &mut [u64], b: &[u64]) {
    for (slot, value) in a.iter_mut().zip(b.iter()) {
        *slot ^= *value;
    }
}

/// Montgomery figure 1: choose `Sᵢ` and form `Winvᵢ`.
///
/// `previous` marks the lanes that were in `Sᵢ₋₁`; the ones that were *not*
/// are tried first, because a lane must be used in `Wᵢ` or `Wᵢ₊₁` for the
/// iteration to keep spanning new space. Returns `Winvᵢ` and the mask naming
/// `Sᵢ`.
///
/// This is Gauss–Jordan on `[T | I]` in which the row and the column are
/// chosen together — the selection has to be symmetric, since what must come
/// out invertible is `Sᵀ T S` — and lanes that find no pivot are struck from
/// both halves.
fn invert(t: &Small, previous: u64) -> (Small, u64) {
    // Lanes not previously selected first, previously selected last.
    let mut order = [0usize; WIDTH];
    let (mut head, mut tail) = (0usize, WIDTH - 1);
    for lane in 0..WIDTH {
        if previous >> lane & 1 == 1 {
            order[tail] = lane;
            tail = tail.wrapping_sub(1);
        } else {
            order[head] = lane;
            head += 1;
        }
    }

    let mut left = *t;
    let mut right: Small = std::array::from_fn(|lane| 1u64 << lane);
    let mut selected = 0u64;

    for step in 0..WIDTH {
        let column = order[step];
        let pivot = (step..WIDTH).find(|&k| left[order[k]] >> column & 1 == 1);
        if let Some(k) = pivot {
            if order[k] != column {
                left.swap(column, order[k]);
                right.swap(column, order[k]);
            }
            selected |= 1u64 << column;
            for row in 0..WIDTH {
                if row != column && left[row] >> column & 1 == 1 {
                    left[row] ^= left[column];
                    right[row] ^= right[column];
                }
            }
        } else {
            // No pivot in the matrix half: this lane cannot join S. Clear it
            // out of the inverse half as well, so it contributes nothing.
            let k = (step..WIDTH)
                .find(|&k| right[order[k]] >> column & 1 == 1)
                .unwrap_or(step);
            if order[k] != column {
                left.swap(column, order[k]);
                right.swap(column, order[k]);
            }
            for row in 0..WIDTH {
                if row != column && right[row] >> column & 1 == 1 {
                    left[row] ^= left[column];
                    right[row] ^= right[column];
                }
            }
            left[column] = 0;
            right[column] = 0;
        }
    }

    for (lane, row) in right.iter_mut().enumerate() {
        if selected >> lane & 1 == 0 {
            *row = 0;
        }
        *row &= selected;
    }
    (right, selected)
}

/// Whether any word is set.
fn any(block: &[u64]) -> bool {
    block.iter().any(|word| *word != 0)
}

/// Lane `which` of a block, as a packed bit vector.
fn lane(block: &Block, which: usize) -> Vec<u64> {
    let mut out = vec![0u64; block.len().div_ceil(WIDTH)];
    for (index, word) in block.words().enumerate() {
        if word >> which & 1 == 1 {
            out[index / WIDTH] |= 1u64 << (index % WIDTH);
        }
    }
    out
}

/// Dependencies among `rows`, or `None` when the iteration did not produce
/// any.
///
/// `None` is not a failure to work around; it is the signal to fall back to
/// the exact solver. Every dependency returned has been checked to sum to zero
/// over the caller's own rows.
///
/// `threads` is a ceiling on retained sparse-fold workers, not a promise to
/// create that many. Zero and one run inline; larger requests are narrowed to
/// the caller's [`budget`](crate::parallelism::budget), and so that
/// every worker receives at least `MINIMUM_FOLDS_PER_WORKER` output folds.
/// Workers live only for this call, and the dependency set is bit-identical
/// at every count for the same rows and random source.
#[must_use]
pub fn block_lanczos_dependencies<R: RandomSource + ?Sized>(
    rows: &[Vec<u64>],
    columns: usize,
    rng: &mut R,
    threads: usize,
) -> Option<Vec<Vec<usize>>> {
    if rows.is_empty() || columns == 0 {
        return None;
    }
    let matrix = Sparse::from_packed(rows, columns, threads);
    lanczos(&matrix, rng, |indices| {
        // Checked against the caller's own rows, not against anything this
        // module computed.
        let mut total = vec![0u64; columns.div_ceil(WIDTH)];
        for &index in indices {
            xor_into(&mut total, &rows[index]);
        }
        if !columns.is_multiple_of(WIDTH) {
            let last = total.len() - 1;
            total[last] &= (1u64 << (columns % WIDTH)) - 1;
        }
        !any(&total)
    })
}

/// [`block_lanczos_dependencies`] over a [`SparseMatrix`]: the same
/// iteration, the same checks, and the same result for the same rows and
/// random source, without packing the rows one bit per column first.
#[must_use]
pub fn block_lanczos_dependencies_sparse<R: RandomSource + ?Sized>(
    matrix: &SparseMatrix,
    rng: &mut R,
    threads: usize,
) -> Option<Vec<Vec<usize>>> {
    if matrix.rows().is_empty() || matrix.columns() == 0 {
        return None;
    }
    let sparse = Sparse::from_lists(matrix.rows().to_vec(), matrix.columns(), threads);
    lanczos(&sparse, rng, |indices| {
        // The XOR of the chosen rows is zero exactly when every column they
        // touch is touched an even number of times: a bit a column, turned
        // over at each touch.
        let mut parity = vec![0u64; matrix.columns().div_ceil(WIDTH)];
        for &index in indices {
            for &column in &matrix.rows()[index] {
                parity[column as usize / WIDTH] ^= 1u64 << (column as usize % WIDTH);
            }
        }
        !any(&parity)
    })
}

/// Montgomery's iteration over `A = MᵀM`, ending with the candidates
/// filtered by `is_null`, the caller's own test that a set of row indices
/// XORs to zero.
fn lanczos<R: RandomSource + ?Sized>(
    matrix: &Sparse,
    rng: &mut R,
    is_null: impl Fn(&[usize]) -> bool + Sync,
) -> Option<Vec<Vec<usize>>> {
    lanczos_observed(matrix, rng, is_null, &mut |_| {})
}

/// One iteration's state as the tests read it: `Vᵢ`, `Tᵢ`, `Winvᵢ` and `Sᵢ`.
#[cfg_attr(not(test), allow(dead_code))]
struct LanczosStep<'a> {
    v: &'a Block,
    t: &'a Small,
    winv: &'a Small,
    selected: u64,
}

/// [`lanczos`], showing `observe` each iteration's state once `Sᵢ` is chosen.
fn lanczos_observed<R: RandomSource + ?Sized>(
    matrix: &Sparse,
    rng: &mut R,
    is_null: impl Fn(&[usize]) -> bool + Sync,
    observe: &mut dyn FnMut(LanczosStep),
) -> Option<Vec<Vec<usize>>> {
    let count = matrix.relations();

    // The starting block is random; rump chooses no entropy source, so the
    // words come from the caller's generator.
    let mut draw = move || {
        let mut bytes = [0u8; 8];
        rng.fill_bytes(&mut bytes);
        u64::from_le_bytes(bytes)
    };

    // X starts as Y and accumulates the solution; Q = V[0] = A·Y never moves.
    let x = Block::of((0..count).map(|_| draw()));
    let q = matrix.apply(&x);

    // The blocks are made once and written over: `next` takes V[i+1] and
    // is then V[i], and the block V[i-2] leaves is the next `next`.
    let mut v0 = Block::of(q.words());
    let mut v1 = Block::zeroed(count);
    let mut v2 = Block::zeroed(count);
    let mut next = Block::zeroed(count);
    let image = Block::zeroed(matrix.columns());
    let av0 = Block::zeroed(count);
    let mut products = matrix.apply_with_products(&v0, &q, &image, &av0);
    let mut t0 = products.t;
    let mut t1 = [0u64; WIDTH];
    let (mut w1i, mut w2i) = ([0u64; WIDTH], [0u64; WIDTH]);
    let mut g = [0u64; WIDTH];
    let mut mask = u64::MAX;

    // The iterations a run needs if every one spans a full block, plus the
    // slack for the lanes the selection leaves out.
    let full_blocks = count / WIDTH;
    let ceiling = full_blocks + count / ROWS_PER_SPARE_ITERATION + SPARE_ITERATIONS_FLOOR;
    let mut iterations = 0usize;
    while any(&t0) {
        iterations += 1;
        if iterations > ceiling {
            return None; // not converging: hand back to the exact solver
        }

        let previous = mask;
        let (next_w0i, next_mask) = invert(&t0, mask);
        // A lane in neither S[i] nor S[i-1] has sat out two rounds. If V[i-1]
        // is non-zero there the iteration has stalled without spanning it,
        // and Montgomery's guarantee is gone.
        let stranded = !(next_mask | previous);
        if stranded != 0 && v1.words().any(|word| word & stranded != 0) {
            return None;
        }
        let w0i = next_w0i;
        mask = next_mask;
        observe(LanczosStep {
            v: &v0,
            t: &t0,
            winv: &w0i,
            selected: mask,
        });
        if mask == 0 {
            break;
        }

        // (20): X += V[i] Winv[i] V[i]ᵀ V[0], with (18) below.
        let projected = mul(&w0i, &products.projection);

        // (19) F[i+1] = Winv[i-2] (I + T[i-1] Winv[i-1]) G[i] S[i]S[i]ᵀ.
        let mut inner = mul(&t1, &w1i);
        plus_identity(&mut inner);
        let mut f = mul(&mul(&w2i, &inner), &g);
        for row in &mut f {
            *row &= mask;
        }

        // (19) E[i+1] = Winv[i-1] T[i] S[i]S[i]ᵀ.
        let mut e = mul(&w1i, &t0);
        for row in &mut e {
            *row &= mask;
        }

        // G[i+1] = A V[i]ᵀ A V[i] S[i]S[i]ᵀ + T[i]. Computed after F, which
        // needs the old one, and before D, which needs the new one.
        let mut squared = products.squared;
        for row in &mut squared {
            *row &= mask;
        }
        for (slot, value) in g.iter_mut().zip(squared.iter().zip(t0.iter())) {
            *slot = value.0 ^ value.1;
        }

        // (19) D[i+1] = I + Winv[i] G[i+1].
        let mut d = mul(&w0i, &g);
        plus_identity(&mut d);

        // (18) V[i+1] = A V[i] S[i]S[i]ᵀ + V[i] D + V[i-1] E + V[i-2] F.
        let advance = Recurrence::new(mask, &d, &e, &f);
        matrix.step(&x, &projected, &av0, [&v0, &v1, &v2], advance, &next);

        next = std::mem::replace(
            &mut v2,
            std::mem::replace(&mut v1, std::mem::replace(&mut v0, next)),
        );
        products = matrix.apply_with_products(&v0, &q, &image, &av0);
        w2i = w1i;
        w1i = w0i;
        t1 = t0;
        t0 = products.t;
    }

    // The kernel of A = MᵀM contains M's but is not equal to it: over GF(2) a
    // vector can be orthogonal to itself. So take the 128 candidates that came
    // out — the columns of X and of the last V — and ask a small elimination
    // which of their combinations M actually annihilates.
    let images = [&x, &v0].map(|block| {
        let image = Block::zeroed(matrix.columns());
        matrix.forward(block, &image);
        image
    });
    let candidates: Vec<Vec<u64>> = (0..2 * WIDTH)
        .map(|index| lane(&images[index / WIDTH], index % WIDTH))
        .collect();
    let sources = [x, v0];

    // Each combination is formed and checked apart from the others, a walk
    // over every relation it holds, so they share the solve's workers.
    let combinations = dense_null_space(&candidates, matrix.columns());
    let found: Vec<Vec<usize>> =
        crate::parallel::map_ordered(&combinations, matrix.folds.threads(), |_, combination| {
            let mut vector = vec![0u64; count.div_ceil(WIDTH)];
            for &index in combination {
                xor_into(&mut vector, &lane(&sources[index / WIDTH], index % WIDTH));
            }
            let indices: Vec<usize> = (0..count)
                .filter(|&r| vector[r / WIDTH] >> (r % WIDTH) & 1 == 1)
                .collect();
            (!indices.is_empty() && is_null(&indices)).then_some(indices)
        })
        .into_iter()
        .flatten()
        .collect();
    (!found.is_empty()).then_some(found)
}

#[cfg(test)]
mod tests {
    use super::{
        block_lanczos_dependencies, borrow_two, dense_null_space, fold_range, lanczos_observed,
        prune_singletons, words_for, Block, FoldPool, Looking, Pace, Recurrence, Small,
        SmallProduct, Sparse, PACE_HOLD, PACE_WINDOW, WIDTH, WORD,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::time::Duration;

    /// Pack a list of column indices into a row.
    fn pack(columns: usize, set: &[usize]) -> Vec<u64> {
        let mut row = vec![0u64; words_for(columns)];
        for &c in set {
            row[c / WORD] |= 1u64 << (c % WORD);
        }
        row
    }

    /// Does this set of row indices XOR to zero over the given columns?
    fn sums_to_zero(rows: &[Vec<u64>], columns: usize, set: &[usize]) -> bool {
        let mut acc = vec![0u64; words_for(columns)];
        for &i in set {
            for (a, r) in acc.iter_mut().zip(rows[i].iter()) {
                *a ^= *r;
            }
        }
        if !columns.is_multiple_of(WORD) {
            let last = words_for(columns) - 1;
            acc[last] &= (1u64 << (columns % WORD)) - 1;
        }
        acc.iter().all(|w| *w == 0)
    }

    /// Every dependency by exhaustive search over subsets — the oracle, valid
    /// only for a handful of rows, which is why the sweep below stays small.
    fn all_dependencies_by_search(rows: &[Vec<u64>], columns: usize) -> usize {
        /// The most rows the oracle accepts: `2^12` subsets, each a few
        /// word XORs, is a moment; every doubling doubles it.
        const ORACLE_ROW_LIMIT: usize = 12;
        let n = rows.len();
        assert!(n <= ORACLE_ROW_LIMIT, "exhaustive oracle is exponential");
        (1u32..(1 << n))
            .filter(|mask| {
                let set: Vec<usize> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
                sums_to_zero(rows, columns, &set)
            })
            .count()
    }

    /// Knuth's MMIX linear congruential generator (TAOCP vol. 2, §3.3.4).
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    #[test]
    fn byte_sliced_small_product_matches_set_bit_multiplication() {
        /// Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x8b10_c4a7_d35e_29f1;
        /// One random matrix per lane of the block.
        const MATRICES: usize = WIDTH;
        let mut state = SEED;
        for _ in 0..MATRICES {
            let matrix: Small = std::array::from_fn(|_| lcg(&mut state));
            let product = SmallProduct::new(&matrix);
            // Selectors: no lane, the lowest lane, every lane, both end
            // lanes, and a random word.
            for value in [0, 1, u64::MAX, 0x8000_0000_0000_0001, lcg(&mut state)] {
                let mut bits = value;
                let mut expected = 0u64;
                while bits != 0 {
                    let lane = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    expected ^= matrix[lane];
                }
                assert_eq!(product.apply(value), expected, "selector {value:#018x}");
            }
        }
    }

    #[test]
    fn the_recurrence_of_a_word_matches_the_scalar_equation() {
        fn scalar(value: u64, matrix: &Small) -> u64 {
            let mut bits = value;
            let mut total = 0u64;
            while bits != 0 {
                let lane = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                total ^= matrix[lane];
            }
            total
        }

        /// Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x65f4_26b8_91de_0ca3;
        /// Words tried; arbitrary.
        const WORDS: usize = 4 * WIDTH + 1;
        let mut state = SEED;
        let d: Small = std::array::from_fn(|_| lcg(&mut state));
        let e: Small = std::array::from_fn(|_| lcg(&mut state));
        let f: Small = std::array::from_fn(|_| lcg(&mut state));
        let mask = lcg(&mut state);
        let advance = Recurrence::new(mask, &d, &e, &f);
        for _ in 0..WORDS {
            let [av, v0, v1, v2]: [u64; 4] = std::array::from_fn(|_| lcg(&mut state));
            assert_eq!(
                advance.apply(av, v0, v1, v2),
                (av & mask) ^ scalar(v0, &d) ^ scalar(v1, &e) ^ scalar(v2, &f)
            );
        }
    }

    #[test]
    fn kept_threads_fold_as_one_thread_folds_pass_after_pass() {
        /// Input words; arbitrary.
        const INPUT_WORDS: u64 = 2_003;
        /// Output lists; odd, so the runs are unequal.
        const LISTS: usize = 10_003;
        /// Threads of the pool, and the runs their pass is cut into: more
        /// runs than threads, so a thread takes several.
        const THREADS: usize = 8;
        const RUNS: usize = 3 * THREADS + 1;
        /// Passes of the same pool, to show the threads survive reuse.
        const REPEATS: usize = 16;
        // Three arbitrary affine index patterns per list, distinct in
        // stride so the lists do not repeat.
        let input = Block::of((0..INPUT_WORDS).map(|value| value.rotate_left(17)));
        let lists: Vec<Vec<u32>> = (0..LISTS)
            .map(|row| {
                vec![
                    (row % input.len()) as u32,
                    ((row * 17 + 3) % input.len()) as u32,
                    ((row * 101 + 29) % input.len()) as u32,
                ]
            })
            .collect();
        let lists = Arc::new(super::Lists::from_lists(&lists));
        let runs = Arc::new(lists.cuts(RUNS));
        let expected = fold_range(&lists, 0, lists.len(), &input);
        let pool = FoldPool::new(THREADS);
        assert_eq!(pool.threads(), THREADS);
        let pace = Mutex::new(Pace::new(THREADS));
        for _ in 0..REPEATS {
            let output = Block::zeroed(lists.len());
            pool.fold(&pace, &lists, &runs, &input, &output);
            assert_eq!(output.words().collect::<Vec<_>>(), expected);
        }
    }

    /// A pass that panics on one thread panics on the caller's, once the
    /// others have left it, and the pool runs the next.
    #[test]
    fn a_pass_that_panics_is_resumed_on_the_caller() {
        /// Threads of the pool; arbitrary, more than [`super::WAKES`] so
        /// that workers wake workers.
        const THREADS: usize = 9;
        /// The thread that panics: a worker, and not one the caller wakes.
        const PANICS: usize = 6;
        let pool = FoldPool::new(THREADS);
        let pace = Mutex::new(Pace::new(THREADS));
        let ran = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&ran);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.pass(&pace, move |thread| {
                counted.fetch_add(1, Ordering::Relaxed);
                assert_ne!(thread, PANICS, "the pass panics on this thread");
            });
        }));
        assert!(caught.is_err());
        assert_eq!(ran.load(Ordering::Relaxed), THREADS);
        let counted = Arc::clone(&ran);
        pool.pass(&pace, move |_| {
            counted.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(ran.load(Ordering::Relaxed), 2 * THREADS);
    }

    /// The count a pace settles on, told that a pass on `threads` threads
    /// takes `pass(threads)` microseconds, and the passes it took to
    /// settle.
    fn settled(pace: &mut Pace, pass: impl Fn(usize) -> u64) -> (usize, usize) {
        let mut passes = 0;
        while pace.looking != Looking::Settled {
            pace.timed(Duration::from_micros(pass(pace.threads)));
            passes += 1;
        }
        (pace.threads, passes)
    }

    #[test]
    fn a_pace_keeps_the_fewest_threads_that_are_as_fast_as_any() {
        /// The pool's threads: a machine of 64 cores and two threads to
        /// each.
        const MOST: usize = 128;
        // Threads that all pay: a pass takes half as long on twice as
        // many, and the whole is kept after one look at the half.
        let mut pace = Pace::new(MOST);
        let halved = |threads: usize| 640_000 / threads as u64;
        assert_eq!(settled(&mut pace, halved), (MOST, 3 * PACE_WINDOW));
        // Memory that gives out at 64 threads, as measured: a hundredth
        // slower on 64 than on 128, and a sixth slower again on 32.
        let mut pace = Pace::new(MOST);
        let bound = |threads: usize| match threads {
            128 => 4_890,
            64 => 4_950,
            _ => 5_780 * 32 / threads as u64,
        };
        assert_eq!(settled(&mut pace, bound), (64, 4 * PACE_WINDOW));
        // Counts each a little slower than the last, and none by as much
        // as the slack: they are measured against the shortest pass seen,
        // so the slowness does not add up unseen.
        let mut pace = Pace::new(MOST);
        let creeping = |threads: usize| 5_000 + 100 * u64::from((MOST / threads).ilog2());
        assert_eq!(settled(&mut pace, creeping).0, 64);
        // One thread has nothing to find.
        let mut pace = Pace::new(1);
        pace.timed(Duration::from_micros(1));
        assert_eq!(pace.threads, 1);
    }

    /// The passes before the first window are not timed: a first pass that
    /// wrote its blocks for the first time, and took long over it, does
    /// not make the half look as fast as the whole.
    #[test]
    fn a_pace_does_not_time_the_passes_that_warm_the_blocks() {
        /// The pool's threads; as above.
        const MOST: usize = 128;
        let mut pace = Pace::new(MOST);
        for _ in 0..PACE_WINDOW {
            assert_eq!(pace.threads, MOST);
            pace.timed(Duration::from_micros(1));
        }
        let halved = |threads: usize| 640_000 / threads as u64;
        assert_eq!(settled(&mut pace, halved), (MOST, 2 * PACE_WINDOW));
    }

    /// A search misled is corrected by the next: a pace that settled on
    /// one thread, the machine's other work making every count as slow,
    /// holds it and then finds that the threads all pay.
    #[test]
    fn a_pace_looks_again_when_it_has_held() {
        /// The pool's threads; as above.
        const MOST: usize = 128;
        let mut pace = Pace::new(MOST);
        assert_eq!(settled(&mut pace, |_| 5_000).0, 1);
        let halved = |threads: usize| 640_000 / threads as u64;
        for _ in 0..PACE_HOLD {
            assert_eq!((pace.threads, pace.looking), (1, Looking::Settled));
            pace.timed(Duration::from_micros(halved(1)));
        }
        assert_eq!((pace.threads, pace.looking), (MOST, Looking::Trying));
        assert_eq!(settled(&mut pace, halved), (MOST, 2 * PACE_WINDOW));
    }

    /// A pass runs on the threads its pace has, the first so many of the
    /// pool's workers and the caller, and the pool runs the next on
    /// others.
    #[test]
    fn a_pass_runs_on_the_threads_its_pace_has() {
        /// Threads of the pool; more than [`super::WAKES`] and one, so
        /// that workers wake workers.
        const THREADS: usize = 12;
        let pool = FoldPool::new(THREADS);
        let ran = Arc::new(Mutex::new(Vec::new()));
        // The whole, the caller alone, some, and the whole again.
        for threads in [THREADS, 1, 5, 2, THREADS, 7] {
            let mut pace = Pace::new(THREADS);
            pace.threads = threads;
            pace.looking = Looking::Settled;
            let pace = Mutex::new(pace);
            let numbers = Arc::clone(&ran);
            pool.pass(&pace, move |thread| numbers.lock().unwrap().push(thread));
            let mut numbers = std::mem::take(&mut *ran.lock().unwrap());
            numbers.sort_unstable();
            let mut expected: Vec<usize> = (0..threads - 1).collect();
            expected.push(THREADS - 1);
            assert_eq!(numbers, expected, "{threads} threads");
        }
    }

    /// Lists as a sieve's columns are: the first few hold most of the
    /// indices. The cuts give each run about its share of the indices, and
    /// the fold over them is the inline fold.
    #[test]
    fn a_fold_is_cut_by_its_indices_and_not_by_its_lists() {
        /// Input words; arbitrary.
        const INPUT_WORDS: usize = 2_003;
        /// Lists enough for eight workers at `MINIMUM_FOLDS_PER_WORKER`.
        const LISTS: usize = 8 * super::MINIMUM_FOLDS_PER_WORKER;
        /// Runs asked for: eight threads' worth.
        const RUNS: usize = WORKERS * super::RUNS_PER_THREAD;
        /// The heavy lists, and how long each is: together four fifths of
        /// the indices, in a hundredth of a run of even length.
        const HEAVY: usize = 40;
        const HEAVY_LENGTH: usize = 4 * LISTS / HEAVY;
        const WORKERS: usize = 8;
        let input = Block::of(
            (0..INPUT_WORDS as u64).map(|value| value.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
        );
        let lists: Vec<Vec<u32>> = (0..LISTS)
            .map(|list| {
                let length = if list < HEAVY { HEAVY_LENGTH } else { 1 };
                (0..length)
                    .map(|place| ((list * 31 + place * 7) % INPUT_WORDS) as u32)
                    .collect()
            })
            .collect();
        let lists = Arc::new(super::Lists::from_lists(&lists));
        let total: usize = (0..lists.len()).map(|list| lists.list(list).len()).sum();

        for parts in [1usize, 2, 3, WORKERS, 5 * LISTS] {
            let cuts = lists.cuts(parts);
            assert_eq!(cuts.first(), Some(&0), "{parts} parts");
            assert_eq!(cuts.last(), Some(&lists.len()), "{parts} parts");
            assert!(
                cuts.windows(2).all(|pair| pair[0] < pair[1]),
                "{parts} parts"
            );
            assert!(cuts.len() - 1 <= parts.min(lists.len()), "{parts} parts");
        }
        // No run of the eight holds more than its share and one list over.
        let cuts = lists.cuts(WORKERS);
        for bounds in cuts.windows(2) {
            let held: usize = (bounds[0]..bounds[1])
                .map(|list| lists.list(list).len())
                .sum();
            assert!(
                held <= total / WORKERS + HEAVY_LENGTH,
                "a run of {held} indices of {total}"
            );
        }

        let expected = fold_range(&lists, 0, lists.len(), &input);
        let pool = FoldPool::new(WORKERS);
        let pace = Mutex::new(Pace::new(WORKERS));
        let output = Block::zeroed(lists.len());
        pool.fold(&pace, &lists, &Arc::new(lists.cuts(RUNS)), &input, &output);
        assert_eq!(output.words().collect::<Vec<_>>(), expected);
    }

    /// Where the sparse solver's time goes on a large random matrix, timed,
    /// for a profiler to look at. `LANCZOS_SIZE` and `LANCZOS_THREADS`
    /// override the defaults.
    #[test]
    #[ignore = "timing probe for the sparse solver at a large matrix's size"]
    fn lanczos_cost_probe() {
        /// Default rows: a round number of the order of a filtered sieve
        /// matrix. Policy; the row count of a filtered matrix from a
        /// hundred-digit factorization, recorded here, would fix it.
        const PROBE_ROWS: usize = 200_000;
        /// Nonzeros drawn per row (fewer after deduplication): a round
        /// number of the order of a filtered sieve row. Policy, as above.
        const PROBE_WEIGHT: usize = 100;
        /// Default worker ceiling. The pool is narrowed to what
        /// `MINIMUM_FOLDS_PER_WORKER` allows, so this only has to be no
        /// smaller than the host's core count; a power of two, as policy.
        const PROBE_THREADS: usize = 128;
        /// Rows beyond the columns: the candidate count the iteration ends
        /// with, the lanes of `X` and of the last `V`, so the null space is
        /// at least as wide as one run can return.
        const PROBE_EXCESS: usize = 2 * WIDTH;
        /// Arbitrary, fixed so a run reproduces.
        const SEED: u64 = 0x5eed_1234_abcd_ef01;
        let size: usize = std::env::var("LANCZOS_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(PROBE_ROWS);
        let weight: usize = PROBE_WEIGHT;
        let threads: usize = std::env::var("LANCZOS_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(PROBE_THREADS);
        let mut rng = TestRng(SEED);
        let columns = size - PROBE_EXCESS;
        let rows: Vec<Vec<u32>> = (0..size)
            .map(|_| {
                let mut row: Vec<u32> = (0..weight)
                    .map(|_| {
                        let mut bytes = [0u8; 8];
                        crate::random::RandomSource::fill_bytes(&mut rng, &mut bytes);
                        (u64::from_le_bytes(bytes) % columns as u64) as u32
                    })
                    .collect();
                row.sort_unstable();
                row.dedup();
                row
            })
            .collect();
        let matrix = super::filter::SparseMatrix::new(columns, rows);
        let started = std::time::Instant::now();
        let dependencies = super::block_lanczos_dependencies_sparse(&matrix, &mut rng, threads);
        eprintln!(
            "lanczos {size} x {columns}, {weight} per row, {threads} threads: {:?}, {} dependencies",
            started.elapsed(),
            dependencies.map_or(0, |d| d.len())
        );
    }

    /// A matrix whose columns lie among empty ones is solved as the matrix
    /// without them: the same dependencies from the same random words.
    #[test]
    fn empty_columns_change_nothing_of_a_solve() {
        /// Arbitrary, fixed so a failure reproduces.
        const MATRIX_SEED: u64 = 0x0123_4567_89ab_cdef;
        /// Arbitrary, fixed so a failure reproduces.
        const SOLVE_SEED: u64 = 0xfedc_ba98_7654_3210;
        /// Rows enough beyond the columns that the solve has dependencies
        /// to return by the block.
        const ROWS: usize = 2_000;
        const COLUMNS: usize = ROWS - 2 * WIDTH;
        /// Nonzeros drawn per row; arbitrary.
        const WEIGHT: usize = 12;
        /// Columns of the wide matrix to each that is set: what filtering
        /// leaves of a sieve's.
        const SPREAD: usize = 11;
        /// Where in each run of `SPREAD` the set column lies; arbitrary.
        const PLACE: usize = 4;
        let mut rng = TestRng(MATRIX_SEED);
        let close: Vec<Vec<u32>> = (0..ROWS)
            .map(|_| {
                let mut row: Vec<u32> = (0..WEIGHT)
                    .map(|_| {
                        let mut bytes = [0u8; 8];
                        crate::random::RandomSource::fill_bytes(&mut rng, &mut bytes);
                        (u64::from_le_bytes(bytes) % COLUMNS as u64) as u32
                    })
                    .collect();
                row.sort_unstable();
                row.dedup();
                row
            })
            .collect();
        let wide: Vec<Vec<u32>> = close
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&column| column * SPREAD as u32 + PLACE as u32)
                    .collect()
            })
            .collect();
        let close = super::filter::SparseMatrix::new(COLUMNS, close);
        let wide = super::filter::SparseMatrix::new(COLUMNS * SPREAD, wide);
        let solved =
            |matrix| super::block_lanczos_dependencies_sparse(matrix, &mut TestRng(SOLVE_SEED), 4);
        let dependencies = solved(&close).expect("the solve finds dependencies");
        assert!(!dependencies.is_empty());
        assert_eq!(solved(&wide), Some(dependencies));
    }

    /// A deterministic `RandomSource` for the tests, so a failure
    /// reproduces: Marsaglia's xorshift64 with the shift triple (13, 7, 17)
    /// (*Xorshift RNGs*, J. Stat. Software 8 (2003), no. 14).
    struct TestRng(u64);
    impl crate::random::RandomSource for TestRng {
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for chunk in dest.chunks_mut(8) {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                let word = self.0.to_le_bytes();
                chunk.copy_from_slice(&word[..chunk.len()]);
            }
        }
    }

    fn sparse_rows(relations: usize, columns: usize, weight: usize, seed: u64) -> Vec<Vec<u64>> {
        let mut state = seed | 1;
        (0..relations)
            .map(|_| {
                let mut row = vec![0u64; words_for(columns)];
                for _ in 0..weight {
                    let column = (lcg(&mut state) as usize) % columns;
                    row[column / WORD] ^= 1u64 << (column % WORD);
                }
                row
            })
            .collect()
    }

    #[test]
    fn dependencies_do_not_depend_on_the_thread_count() {
        // The applies split by output ranges and concatenate in order, so the
        // whole iteration -- and therefore the dependency sets -- must be
        // bit-identical at any thread count, given the same starting block.
        /// One past two fold thresholds, so the eight-worker arm uses two
        /// retained workers rather than the inline path.
        const RELATIONS: usize = 2 * super::MINIMUM_FOLDS_PER_WORKER + 1;
        /// Arbitrary, fixed so a failure reproduces.
        const FIXTURE_SEED: u64 = 0x00c0_ffee;
        /// Arbitrary, fixed so a failure reproduces; the same for both arms.
        const RNG_SEED: u64 = 7;
        let rows = sparse_rows(RELATIONS, 96, 8, FIXTURE_SEED);
        let one = block_lanczos_dependencies(&rows, 96, &mut TestRng(RNG_SEED), 1);
        let eight = block_lanczos_dependencies(&rows, 96, &mut TestRng(RNG_SEED), 8);
        assert_eq!(one, eight);
    }

    /// Every dependency Block Lanczos returns is a genuine one.
    ///
    /// The method is randomized, so `None` is accepted here; a returned set
    /// that is not dependent never is. Convergence is pinned separately by
    /// `block_lanczos_recovers_a_known_subspace_on_fixed_input`; this test
    /// cannot detect a solver that always gives up.
    #[test]
    fn block_lanczos_returns_only_genuine_dependencies() {
        /// Arbitrary, fixed so a failure reproduces.
        const FIXTURE_SEED: u64 = 0x1234_5678;
        /// Arbitrary, fixed so a failure reproduces.
        const RNG_SEED: u64 = 0xdead_beef;
        // Shapes: rows exceed columns by at least a block, so dependencies
        // exist, over widths of a word and a half, two words, and more than
        // three.
        for &(relations, columns, weight) in
            &[(160usize, 96usize, 8usize), (200, 128, 10), (300, 200, 12)]
        {
            let rows = sparse_rows(relations, columns, weight, FIXTURE_SEED);
            let mut rng = TestRng(RNG_SEED);
            if let Some(deps) = block_lanczos_dependencies(&rows, columns, &mut rng, 1) {
                assert!(!deps.is_empty(), "Some(...) must not be an empty set");
                for dep in &deps {
                    assert!(!dep.is_empty());
                    assert!(
                        sums_to_zero(&rows, columns, dep),
                        "returned a set that does not sum to zero"
                    );
                }
            }
        }
    }

    /// GF(2) rank of a set of row-index sets, viewed as indicator vectors.
    fn rank_of(sets: &[Vec<usize>], relations: usize) -> usize {
        let mut vectors: Vec<Vec<u64>> = sets
            .iter()
            .map(|set| {
                let mut v = vec![0u64; words_for(relations)];
                for &i in set {
                    v[i / WORD] ^= 1u64 << (i % WORD);
                }
                v
            })
            .collect();
        let mut rank = 0usize;
        for bit in 0..relations {
            let (word, mask) = (bit / WORD, 1u64 << (bit % WORD));
            let Some(p) = (rank..vectors.len()).find(|&r| vectors[r][word] & mask != 0) else {
                continue;
            };
            vectors.swap(rank, p);
            for r in 0..vectors.len() {
                if r != rank && vectors[r][word] & mask != 0 {
                    let (src, dst) = borrow_two(&mut vectors, rank, r);
                    for (d, s) in dst.iter_mut().zip(src.iter()) {
                        *d ^= *s;
                    }
                }
            }
            rank += 1;
        }
        rank
    }

    /// On fixed input with a fixed generator, the solver must converge and
    /// recover a specific subspace.
    ///
    /// Both halves matter. `expect` rather than `if let Some`, so an
    /// implementation that always returns `None` fails here rather than
    /// passing vacuously. And the rank is pinned to a measured value rather
    /// than bounded, because `rank <= exact` follows automatically from every
    /// returned vector being a genuine dependency — it would hold for a solver
    /// that returned a single dependency and nothing else.
    ///
    /// The pinned ranks are *below* the exact null space's dimension, and that
    /// is the method rather than a defect: sixty-four vectors ride in the bits
    /// of one word, so one run recovers a subspace bounded by that block width
    /// and not the whole space. A caller wanting more re-runs with a different
    /// generator or falls back to [`dense_null_space`]. These numbers are
    /// therefore a regression pin, not a target: a change in them means the
    /// iteration changed, which is exactly what this test is for.
    #[test]
    fn block_lanczos_recovers_a_known_subspace_on_fixed_input() {
        /// Arbitrary, fixed so a failure reproduces; the ranks below are
        /// pinned to it.
        const FIXTURE_SEED: u64 = 0x0bad_c0de;
        /// Arbitrary, fixed so a failure reproduces; the ranks below are
        /// pinned to it.
        const RNG_SEED: u64 = 0x5eed_1234;
        // Shapes: rows exceed columns by one block and by more, with the
        // exact dimension above the block width in both, so one run cannot
        // recover the whole space.
        for &(relations, columns, weight, expected_rank, exact_dimension) in &[
            (160usize, 96usize, 8usize, 60usize, 92usize),
            (192, 120, 9, 64, 72),
        ] {
            let rows = sparse_rows(relations, columns, weight, FIXTURE_SEED);
            assert_eq!(
                dense_null_space(&rows, columns).len(),
                exact_dimension,
                "the fixture's null space changed"
            );

            let mut rng = TestRng(RNG_SEED);
            let dependencies = block_lanczos_dependencies(&rows, columns, &mut rng, 1)
                .expect("this fixture must converge");
            for dep in &dependencies {
                assert!(
                    sums_to_zero(&rows, columns, dep),
                    "returned a set that does not sum to zero"
                );
            }
            assert_eq!(
                rank_of(&dependencies, relations),
                expected_rank,
                "{relations}x{columns} recovered a different subspace"
            );
        }
    }

    /// Montgomery's invariants, step by step, against dense arithmetic built
    /// here from the rows (`A[r][s]` is the parity of rows `r` and `s`'s
    /// common columns), not from the sparse code under test:
    ///
    /// - `Winvᵢ` inverts `Tᵢ = VᵢᵀAVᵢ` on the selected lanes `Sᵢ`;
    /// - `WⱼᵀAVᵢ₊₁ = 0` for every `j ≤ i`, with `Wⱼ = VⱼSⱼ`, and so
    ///   `WᵢᵀAWⱼ = 0` for `i ≠ j`;
    /// - the dependencies returned span the null space dense elimination
    ///   finds, when that space is narrow enough that one block holds it with
    ///   room to spare — half a block, so the comparison is not made where a
    ///   wider space could have been truncated to the block width.
    #[test]
    fn the_recurrence_keeps_montgomerys_invariants() {
        /// Fixture seeds. Each also sets the width: `330 + 3·seed` columns
        /// against 360 rows, so the excess of rows over columns runs from 27
        /// down past zero and the null space narrows from under half a block
        /// to nothing.
        const SEEDS: std::ops::RangeInclusive<u64> = 1..=12;
        /// Arbitrary, fixed so a failure reproduces; XORed with the seed so
        /// each fixture starts from a different block.
        const RNG_SEED_BASE: u64 = 0x9e37_79b9_7f4a_7c15;
        /// Nonzeros per row; arbitrary.
        const WEIGHT: usize = 10;
        /// The recurrence (18) reaches back two blocks, so its `F` term is
        /// first formed from real blocks at the third step.
        const FEWEST_STEPS: usize = 3;
        /// Null spaces at most this wide are compared with dense elimination:
        /// half a block, so a space one run could have truncated to the
        /// block width is never compared.
        const COMPARABLE_DIMENSION: usize = WIDTH / 2;
        /// Seeds whose span must be compared. These fixtures give nine, so a
        /// solver that stops converging fails while one seed drifting out
        /// of range does not.
        const FEWEST_COMPARED: usize = 8;
        // (VᵀAU)[l][m] for n × 64 blocks, by definition.
        fn form(v: &[u64], a_u: &[u64]) -> Small {
            let mut out = [0u64; WIDTH];
            for (&row_v, &row_au) in v.iter().zip(a_u) {
                for (l, slot) in out.iter_mut().enumerate() {
                    if row_v >> l & 1 == 1 {
                        *slot ^= row_au;
                    }
                }
            }
            out
        }
        fn rank(mut vectors: Vec<Vec<u64>>) -> usize {
            let mut rank = 0;
            let bits = vectors.first().map_or(0, |v| v.len() * WORD);
            for bit in 0..bits {
                let Some(pivot) = (rank..vectors.len())
                    .find(|&i| vectors[i][bit / WORD] >> (bit % WORD) & 1 == 1)
                else {
                    continue;
                };
                vectors.swap(rank, pivot);
                let row = vectors[rank].clone();
                for (i, other) in vectors.iter_mut().enumerate() {
                    if i != rank && other[bit / WORD] >> (bit % WORD) & 1 == 1 {
                        for (a, b) in other.iter_mut().zip(&row) {
                            *a ^= b;
                        }
                    }
                }
                rank += 1;
            }
            rank
        }
        let mut compared = 0;
        for seed in SEEDS {
            let (relations, columns) = (360, 330 + 3 * seed as usize);
            let rows = sparse_rows(relations, columns, WEIGHT, seed);
            let dense_a: Vec<Vec<usize>> = (0..relations)
                .map(|r| {
                    (0..relations)
                        .filter(|&t| {
                            rows[r]
                                .iter()
                                .zip(&rows[t])
                                .map(|(x, y)| (x & y).count_ones())
                                .sum::<u32>()
                                % 2
                                == 1
                        })
                        .collect()
                })
                .collect();
            let apply = |v: &[u64]| -> Vec<u64> {
                dense_a
                    .iter()
                    .map(|row| row.iter().fold(0u64, |acc, &t| acc ^ v[t]))
                    .collect()
            };
            let mut steps: Vec<(Vec<u64>, Small, Small, u64)> = Vec::new();
            let matrix = Sparse::from_packed(&rows, columns, 1);
            let found = lanczos_observed(
                &matrix,
                &mut TestRng(RNG_SEED_BASE ^ seed),
                |_| true,
                &mut |step| {
                    steps.push((step.v.words().collect(), *step.t, *step.winv, step.selected));
                },
            );
            assert!(
                steps.len() >= FEWEST_STEPS,
                "seed {seed}: only {} iterations",
                steps.len()
            );
            let lanes = |mask: u64| (0..WIDTH).filter(move |l| mask >> l & 1 == 1);
            let a_v: Vec<Vec<u64>> = steps.iter().map(|(v, ..)| apply(v)).collect();
            for (i, (v, t, winv, selected)) in steps.iter().enumerate() {
                assert_eq!(&form(v, &a_v[i]), t, "seed {seed}, step {i}: Tᵢ");
                let product: Small = std::array::from_fn(|l| {
                    (0..WIDTH)
                        .filter(|&k| winv[l] >> k & 1 == 1)
                        .fold(0u64, |acc, k| acc ^ t[k])
                });
                for l in lanes(*selected) {
                    assert_eq!(
                        product[l] & selected,
                        1u64 << l,
                        "seed {seed}, step {i}: Winv·T on S, lane {l}"
                    );
                }
                // Vᵢ is step i − 1's Vᵢ₊₁: Wⱼᵀ A Vᵢ = 0 for every j < i, which
                // includes Wⱼᵀ A Wᵢ = 0.
                for (j, (v_j, _, _, selected_j)) in steps.iter().enumerate().take(i) {
                    let w_j_a_v = form(v_j, &a_v[i]);
                    for l in lanes(*selected_j) {
                        assert_eq!(w_j_a_v[l], 0, "seed {seed}: W{j}ᵀAV{i}, lane {l}");
                    }
                }
            }
            let null_space = dense_null_space(&rows, columns);
            if let Some(found) = found {
                if null_space.len() < COMPARABLE_DIMENSION {
                    let indicator = |set: &[usize]| {
                        let mut bits = vec![0u64; words_for(relations)];
                        for &r in set {
                            bits[r / WORD] |= 1 << (r % WORD);
                        }
                        bits
                    };
                    let found_rank = rank(found.iter().map(|d| indicator(d)).collect());
                    let dense_rank = rank(null_space.iter().map(|d| indicator(d)).collect());
                    assert_eq!(
                        found_rank, dense_rank,
                        "seed {seed}: the span differs from dense elimination"
                    );
                    compared += 1;
                }
            }
        }
        assert!(
            compared >= FEWEST_COMPARED,
            "only {compared} spans compared"
        );
    }

    /// Degenerate shapes are refused rather than guessed at.
    #[test]
    fn block_lanczos_refuses_the_degenerate_shapes() {
        /// Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 1;
        let mut rng = TestRng(SEED);
        assert!(block_lanczos_dependencies(&[], 8, &mut rng, 1).is_none());
        assert!(block_lanczos_dependencies(&[vec![0u64]], 0, &mut rng, 1).is_none());
    }

    /// The generator is the caller's: the same source gives the same answer,
    /// and a different one is still only ever asked for genuine dependencies.
    #[test]
    fn block_lanczos_is_driven_by_the_callers_generator() {
        /// Arbitrary, fixed so a failure reproduces.
        const FIXTURE_SEED: u64 = 0xfeed_face;
        /// Arbitrary, fixed so a failure reproduces; used twice.
        const RNG_SEED: u64 = 7;
        /// Arbitrary, distinct from `RNG_SEED`.
        const OTHER_RNG_SEED: u64 = 99;
        let rows = sparse_rows(160, 96, 8, FIXTURE_SEED);
        let first = block_lanczos_dependencies(&rows, 96, &mut TestRng(RNG_SEED), 1);
        let again = block_lanczos_dependencies(&rows, 96, &mut TestRng(RNG_SEED), 1);
        assert_eq!(first, again, "the same source must give the same answer");

        if let Some(deps) = block_lanczos_dependencies(&rows, 96, &mut TestRng(OTHER_RNG_SEED), 1) {
            for dep in &deps {
                assert!(sums_to_zero(&rows, 96, dep));
            }
        }
    }

    #[test]
    fn dense_null_space_returns_only_genuine_dependencies() {
        /// Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x51ed_0001;
        /// Random matrices per shape; arbitrary.
        const DRAWS: usize = 40;
        let mut seed = SEED;
        // Shapes: more rows than columns, square, and fewer rows than
        // columns at exactly one word of width and just past it.
        for &(rows_n, columns) in &[
            (6usize, 4usize),
            (9, 5),
            (10, 10),
            (12, 3),
            (5, WORD),
            (7, WORD + 6),
        ] {
            for _ in 0..DRAWS {
                // Each entry set with probability one half.
                let rows: Vec<Vec<u64>> = (0..rows_n)
                    .map(|_| {
                        let set: Vec<usize> =
                            (0..columns).filter(|_| lcg(&mut seed) & 1 == 0).collect();
                        pack(columns, &set)
                    })
                    .collect();
                for dep in dense_null_space(&rows, columns) {
                    assert!(!dep.is_empty());
                    assert!(
                        sums_to_zero(&rows, columns, &dep),
                        "returned a set that does not sum to zero: {dep:?}"
                    );
                }
            }
        }
    }

    /// The count is `rows − rank`, which the exhaustive oracle confirms: a
    /// null space of dimension `d` has exactly `2^d − 1` non-empty dependent
    /// subsets.
    #[test]
    fn dense_null_space_has_the_full_dimension() {
        /// Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x9e37_0002;
        /// Random matrices per shape; arbitrary.
        const DRAWS: usize = 25;
        let mut seed = SEED;
        // Shapes: more rows than columns, within the oracle's row limit, so
        // the null space is non-trivial and its subsets can be counted.
        for &(rows_n, columns) in &[(5usize, 3usize), (6, 4), (8, 5), (7, 2)] {
            for _ in 0..DRAWS {
                // Each entry set with probability one half.
                let rows: Vec<Vec<u64>> = (0..rows_n)
                    .map(|_| {
                        let set: Vec<usize> =
                            (0..columns).filter(|_| lcg(&mut seed) & 1 == 0).collect();
                        pack(columns, &set)
                    })
                    .collect();
                let dimension = dense_null_space(&rows, columns).len();
                let expected = all_dependencies_by_search(&rows, columns);
                assert_eq!(
                    (1usize << dimension) - 1,
                    expected,
                    "dimension {dimension} does not account for {expected} dependent subsets"
                );
            }
        }
    }

    /// Bits above the declared width are not columns and must not be counted.
    #[test]
    fn bits_above_the_declared_width_are_ignored() {
        // Three columns declared, but each row carries junk in bits 3..64.
        let rows = vec![vec![0b1111_1101u64], vec![0b1111_1101u64]];
        let deps = dense_null_space(&rows, 3);
        // Over three columns both rows are 101, so they are dependent.
        assert_eq!(deps.len(), 1);
        assert!(sums_to_zero(&rows, 3, &deps[0]) || deps[0] == vec![0, 1]);
    }

    #[test]
    fn dense_null_space_handles_the_degenerate_shapes() {
        assert!(dense_null_space(&[], 5).is_empty());
        // A single zero row is itself a dependency.
        assert_eq!(dense_null_space(&[vec![0u64]], 4), vec![vec![0]]);
        // Zero columns: every row is the zero vector, so every row is
        // dependent and there are `rows` independent dependencies.
        assert_eq!(dense_null_space(&[vec![0u64], vec![0u64]], 0).len(), 2);
    }

    /// Pruning must not change the answer: the dependencies of the pruned
    /// matrix, mapped back, are dependencies of the original.
    #[test]
    fn pruning_preserves_the_null_space() {
        /// Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xfeed_0003;
        /// Random matrices per shape; arbitrary.
        const DRAWS: usize = 30;
        /// Each entry is set with probability one in this: sparse enough
        /// that singleton columns occur at these widths.
        const SPARSITY: u64 = 5;
        let mut seed = SEED;
        // Shapes: rows fewer than columns and more, and one wide enough
        // that most columns are empty or singletons.
        for &(rows_n, columns) in &[(8usize, 12usize), (10, 20), (12, 9), (6, 40)] {
            for _ in 0..DRAWS {
                let rows: Vec<Vec<u64>> = (0..rows_n)
                    .map(|_| {
                        let set: Vec<usize> = (0..columns)
                            .filter(|_| lcg(&mut seed).is_multiple_of(SPARSITY))
                            .collect();
                        pack(columns, &set)
                    })
                    .collect();

                let pruned = prune_singletons(&rows, columns);
                assert_eq!(pruned.rows().len(), pruned.original().len());
                for row in pruned.rows() {
                    assert_eq!(row.len(), words_for(pruned.columns()));
                }

                // Every dependency of the pruned matrix is one of the
                // original's, under `original()`.
                for dep in dense_null_space(pruned.rows(), pruned.columns()) {
                    let mapped: Vec<usize> = dep.iter().map(|&i| pruned.original()[i]).collect();
                    assert!(
                        sums_to_zero(&rows, columns, &mapped),
                        "a pruned dependency is not one of the original's"
                    );
                }

                // And the dimension is unchanged: pruning removes only rows
                // that could appear in no dependency.
                assert_eq!(
                    dense_null_space(pruned.rows(), pruned.columns()).len(),
                    dense_null_space(&rows, columns).len(),
                    "pruning changed the dimension of the null space"
                );
            }
        }
    }

    #[test]
    fn pruning_removes_a_singleton_cascade() {
        // Column 0 is set only by row 0, so row 0 goes; that leaves column 1
        // set only by row 1, which goes in turn, and so on.
        let columns = 4;
        let rows = vec![
            pack(columns, &[0, 1]),
            pack(columns, &[1, 2]),
            pack(columns, &[2, 3]),
        ];
        let pruned = prune_singletons(&rows, columns);
        assert!(
            pruned.rows().is_empty(),
            "the whole chain should peel: {:?}",
            pruned.original()
        );
        assert_eq!(pruned.columns(), 0);
    }

    #[test]
    fn pruning_keeps_a_matrix_with_no_singletons() {
        // Every column has two occupants, so nothing peels.
        let columns = 2;
        let rows = vec![
            pack(columns, &[0, 1]),
            pack(columns, &[0, 1]),
            pack(columns, &[0, 1]),
        ];
        let pruned = prune_singletons(&rows, columns);
        assert_eq!(pruned.rows().len(), 3);
        assert_eq!(pruned.columns(), 2);
        assert_eq!(pruned.original(), &[0, 1, 2]);
    }
}
