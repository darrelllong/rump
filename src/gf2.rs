//! Linear algebra over GF(2): dense null space and singleton pruning.
//!
//! Solving `Mx = 0` over GF(2) for a large sparse `M` is not a factoring
//! problem, though factoring is where it is most often met: it is the same
//! computation in index-calculus discrete logarithms, in coding theory, and
//! anywhere a parity system gets large. Only the *matrix* belongs to the
//! caller's problem; the solver does not.
//!
//! Addition is XOR, so a whole row combines in one pass of 64-bit words and
//! there is no pivoting for numerical stability to worry about — any non-zero
//! entry will do.
//!
//! # The packing contract
//!
//! A row is a `&[u64]` holding one bit per column: column `c` lives at bit
//! `c % 64` of word `c / 64`, least significant bit first. Bits at or beyond
//! the declared `columns` are ignored rather than trusted, so a stray high bit
//! cannot masquerade as a column. Nothing in the type system enforces this,
//! which is why it is stated here and in `NAMES.md`.
//!
//! This module is distinct from [`finite_field`](crate::finite_field), which
//! is arithmetic *in* the field GF(2^m); here GF(2) is the field the linear
//! algebra happens over.

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
/// and blind to sparsity. [`prune_singletons`] first; a sparse-time
/// solver belongs here too and is the next transfer.
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

#[cfg(test)]
mod tests {
    use super::{dense_null_space, prune_singletons, words_for, WORD};

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
        let n = rows.len();
        assert!(n <= 12, "exhaustive oracle is exponential");
        (1u32..(1 << n))
            .filter(|mask| {
                let set: Vec<usize> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
                sums_to_zero(rows, columns, &set)
            })
            .count()
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    #[test]
    fn dense_null_space_returns_only_genuine_dependencies() {
        let mut seed = 0x51ed_0001u64;
        for &(rows_n, columns) in &[
            (6usize, 4usize),
            (9, 5),
            (10, 10),
            (12, 3),
            (5, 64),
            (7, 70),
        ] {
            for _ in 0..40 {
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
        let mut seed = 0x9e37_0002u64;
        for &(rows_n, columns) in &[(5usize, 3usize), (6, 4), (8, 5), (7, 2)] {
            for _ in 0..25 {
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
        let mut seed = 0xfeed_0003u64;
        for &(rows_n, columns) in &[(8usize, 12usize), (10, 20), (12, 9), (6, 40)] {
            for _ in 0..30 {
                // Sparse rows, so singletons actually occur.
                let rows: Vec<Vec<u64>> = (0..rows_n)
                    .map(|_| {
                        let set: Vec<usize> = (0..columns)
                            .filter(|_| lcg(&mut seed).is_multiple_of(5))
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
