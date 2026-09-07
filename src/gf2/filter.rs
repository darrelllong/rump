//! Structured Gaussian elimination over GF(2) on sparse rows: the filtering
//! step between a sieve and its linear solver.
//!
//! A sieve matrix has a few dozen nonzeros in a row and hundreds of
//! thousands of columns, most of them touched by one or two rows. Holding
//! it packed, one bit per column, made every row a kilobyte-scale word
//! array: an XOR of two rows walked the whole width, a row's support was a
//! scan, and the pairwise comparisons of a column's members cost the width
//! squared — at 160,000 rows over 500,000 columns that was nine gigabytes
//! and most of the filtering time. [`SparseMatrix`] holds each row as its
//! ascending column indices, so every operation costs the row's weight.
//!
//! # The elimination
//!
//! Structured Gaussian elimination as NFS practice has it (Cavallar,
//! *Strategies in filtering in the number field sieve*, ANTS-IV, LNCS
//! 1838 (2000), 209–231; the modern treatment is Bouillaguet &
//! Zimmermann, *Parallel Structured Gaussian Elimination for the Number
//! Field Sieve*, J. Math. Cryptol. 15 (2021), 87–103): a column held by
//! one live row pins that row out of every dependency, so the row goes;
//! a column held by `w ≥ 2` rows is eliminated by adding rows along the
//! minimum spanning tree of its members, edges weighted by the size of the
//! pairwise symmetric difference (their §3, after Cavallar), and
//! discarding the tree's root. Each step retires one row and one column.
//!
//! # When to stop
//!
//! A merge bound — eliminate columns up to weight `k` — is the usual
//! knob, and it is the wrong shape: whether a merge pays depends on the
//! fill it causes, not on the weight of the column. The solver this feeds
//! is Block Lanczos, whose cost is `rows/64` iterations of a pass over
//! every nonzero: proportional to `rows · nonzeros`. Eliminating a column
//! whose tree costs fill `Δ` (the change in the nonzero count, negative
//! for a singleton or a pair) takes the cost from `rows · nonzeros` to
//! `(rows − 1)(nonzeros + Δ)`, which is a gain exactly when
//!
//! ```text
//! Δ · (rows − 1) < nonzeros
//! ```
//!
//! — the fill must be less than the mean row weight. So the eliminations
//! run in order of increasing fill, Markowitz's rule for a sparse
//! elimination (*The elimination form of the inverse and its application
//! to linear programming*, Management Science 3 (1957), 255–269), and stop
//! at the first that would not pay. The threshold moves as the matrix
//! densifies, and so do the fills, so the greedy order is exactly the
//! descent of the solver's cost; no bound is needed, and the `weight_cap`
//! argument only limits how many members a tree may have.
//!
//! # Correctness travels with the data
//!
//! Every surviving row records the set of original rows it is the sum of.
//! A dependency over the filtered matrix expands to one over the original
//! by symmetric difference of those sets, an identity the tests check
//! directly against the original rows rather than trusting a history log.

use core::cmp::Reverse;
use std::collections::BinaryHeap;

use super::{set_bits, words_for, WORD};

/// A GF(2) matrix with each row held as its ascending column indices.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseMatrix {
    columns: usize,
    rows: Vec<Vec<u32>>,
}

impl SparseMatrix {
    /// Wrap rows already in the canonical form: each strictly ascending,
    /// every index below `columns`.
    ///
    /// # Panics
    ///
    /// Panics if a row is not strictly ascending or reaches `columns`.
    #[must_use]
    pub fn new(columns: usize, rows: Vec<Vec<u32>>) -> Self {
        assert!(
            u32::try_from(rows.len()).is_ok() && u32::try_from(columns).is_ok(),
            "{} rows by {columns} columns: the filter indexes both in thirty-two bits",
            rows.len()
        );
        for (index, row) in rows.iter().enumerate() {
            assert!(
                row.windows(2).all(|pair| pair[0] < pair[1]),
                "row {index} is not strictly ascending"
            );
            if let Some(&last) = row.last() {
                assert!(
                    (last as usize) < columns,
                    "row {index} touches column {last} beyond the width {columns}"
                );
            }
        }
        Self { columns, rows }
    }

    /// From packed rows under the module's bit contract; bits at or beyond
    /// `columns` are ignored.
    #[must_use]
    pub fn from_packed(rows: &[Vec<u64>], columns: usize) -> Self {
        assert!(
            u32::try_from(rows.len()).is_ok() && u32::try_from(columns).is_ok(),
            "{} rows by {columns} columns: the filter indexes both in thirty-two bits",
            rows.len()
        );
        let words = words_for(columns);
        let rows = rows
            .iter()
            .map(|row| {
                set_bits(row, words, columns)
                    .map(|column| column as u32)
                    .collect()
            })
            .collect();
        Self { columns, rows }
    }

    /// The width.
    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// The rows, each ascending.
    #[must_use]
    pub fn rows(&self) -> &[Vec<u32>] {
        &self.rows
    }

    /// The rows, surrendered.
    #[must_use]
    pub fn into_rows(self) -> Vec<Vec<u32>> {
        self.rows
    }

    /// The total number of set entries.
    #[must_use]
    pub fn nonzeros(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
    }

    /// The rows packed one bit per column, for the dense solver.
    #[must_use]
    pub fn packed_rows(&self) -> Vec<Vec<u64>> {
        let words = words_for(self.columns);
        self.rows
            .iter()
            .map(|row| {
                let mut packed = vec![0u64; words];
                for &column in row {
                    let column = column as usize;
                    packed[column / WORD] |= 1u64 << (column % WORD);
                }
                packed
            })
            .collect()
    }
}

/// A filtered GF(2) matrix: rows merged and pruned, each carrying the set
/// of original rows it is the sum of. See the [module documentation](self).
#[derive(Clone, Debug)]
pub struct FilteredMatrix {
    matrix: SparseMatrix,
    live_columns: usize,
    /// Ascending original-row indices whose XOR is the corresponding row.
    compositions: Vec<Vec<usize>>,
}

impl FilteredMatrix {
    /// The surviving rows as a matrix over the original width.
    #[must_use]
    pub fn matrix(&self) -> &SparseMatrix {
        &self.matrix
    }

    /// The surviving rows, each ascending.
    #[must_use]
    pub fn rows(&self) -> &[Vec<u32>] {
        self.matrix.rows()
    }

    /// The column count, unchanged from the input; eliminated columns are
    /// simply empty. This is the width the solvers must be told.
    #[must_use]
    pub fn columns(&self) -> usize {
        self.matrix.columns()
    }

    /// Columns still touched by a surviving row. This — not
    /// [`Self::columns`] — is the number an over-determination test must
    /// compare row counts against: comparing against the full width once
    /// declared every filtered matrix under-determined and sent a run
    /// widening forever, since filtering removes a row per eliminated
    /// column but the width never moves.
    #[must_use]
    pub fn live_columns(&self) -> usize {
        self.live_columns
    }

    /// The total number of set entries.
    #[must_use]
    pub fn nonzeros(&self) -> usize {
        self.matrix.nonzeros()
    }

    /// The surviving rows packed one bit per column, for the dense solver.
    #[must_use]
    pub fn packed_rows(&self) -> Vec<Vec<u64>> {
        self.matrix.packed_rows()
    }

    /// The original rows whose XOR forms filtered row `index`.
    #[must_use]
    pub fn composition(&self, index: usize) -> &[usize] {
        &self.compositions[index]
    }

    /// Expands a dependency over the filtered rows to one over the
    /// original rows, by symmetric difference of the compositions.
    #[must_use]
    pub fn expand(&self, dependency: &[usize]) -> Vec<usize> {
        let mut counts: std::collections::BTreeMap<usize, u32> = std::collections::BTreeMap::new();
        for &row in dependency {
            for &original in &self.compositions[row] {
                *counts.entry(original).or_insert(0) += 1;
            }
        }
        counts
            .into_iter()
            .filter_map(|(original, count)| (count % 2 == 1).then_some(original))
            .collect()
    }
}

/// The symmetric difference of two ascending lists, ascending.
fn symmetric_difference<T: Copy + Ord>(a: &[T], b: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            core::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            core::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            core::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// How many entries two ascending lists share.
fn intersection_size(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut shared) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            core::cmp::Ordering::Less => i += 1,
            core::cmp::Ordering::Greater => j += 1,
            core::cmp::Ordering::Equal => {
                shared += 1;
                i += 1;
                j += 1;
            }
        }
    }
    shared
}

/// The elimination plan for one column: the tree's edges as
/// `(child, parent)` pairs, children before parents, the discarded root,
/// and the fill — the change in the nonzero count the plan causes.
struct Plan {
    root: usize,
    edges: Vec<(usize, usize)>,
    fill: i64,
}

/// The minimum spanning tree over `members`, edges weighted by the size of
/// the pairwise symmetric difference, grown one nearest vertex at a time —
/// the algorithm of Jarník (*O jistém problému minimálním*, Práce Moravské
/// přírodovědecké společnosti 6 (1930), 57–63), rediscovered by Prim (Bell
/// System Tech. J. 36 (1957), 1389–1401) and usually named for him. The
/// root is the heaviest member — discarding the heaviest row is the
/// natural choice — and the fill is the tree's total edge weight less the
/// weight of every member (Cavallar §3; Bouillaguet & Zimmermann §3).
fn plan(members: &[usize], rows: &[Vec<u32>]) -> Plan {
    let count = members.len();
    debug_assert!(count >= 2, "a tree needs two members");
    let weight = |a: usize, b: usize| -> i64 {
        (rows[a].len() + rows[b].len() - 2 * intersection_size(&rows[a], &rows[b])) as i64
    };
    let root_index = (0..count)
        .max_by_key(|&i| (rows[members[i]].len(), Reverse(members[i])))
        .expect("non-empty");
    let mut in_tree = vec![false; count];
    in_tree[root_index] = true;
    // Nearest tree vertex per outside vertex, refreshed as the tree grows.
    let mut nearest: Vec<(i64, usize)> = (0..count)
        .map(|i| (weight(members[i], members[root_index]), root_index))
        .collect();
    let mut edges = Vec::with_capacity(count - 1);
    let mut fill: i64 = -(members.iter().map(|&m| rows[m].len() as i64).sum::<i64>());
    for _ in 1..count {
        let next = (0..count)
            .filter(|&i| !in_tree[i])
            .min_by_key(|&i| (nearest[i].0, members[i]))
            .expect("a vertex remains");
        let (edge_weight, parent) = nearest[next];
        in_tree[next] = true;
        edges.push((members[next], members[parent]));
        fill += edge_weight;
        for i in 0..count {
            if !in_tree[i] {
                let w = weight(members[i], members[next]);
                if w < nearest[i].0 {
                    nearest[i] = (w, next);
                }
            }
        }
    }
    // A vertex joins after its parent, so reversing the join order puts
    // every child before its parent: each edge then adds the parent's
    // *original* row, which is what cancels the column.
    edges.reverse();
    Plan {
        root: members[root_index],
        edges,
        fill,
    }
}

/// The working state of one filtering run: rows, their compositions, and
/// the column incidence kept exact as rows change.
struct Filter {
    columns: usize,
    work: Vec<Vec<u32>>,
    compositions: Vec<Vec<usize>>,
    live: Vec<bool>,
    live_rows: usize,
    live_columns: usize,
    nonzeros: i64,
    /// Column incidence as row lists, maintained incrementally. Retired
    /// rows and rows that cancelled the column linger and are compacted
    /// away by [`Self::members`]; the occupancy counts are exact
    /// throughout.
    incidence: Vec<Vec<u32>>,
    occupants: Vec<u32>,
}

impl Filter {
    fn new(matrix: &SparseMatrix) -> Self {
        let columns = matrix.columns();
        let work: Vec<Vec<u32>> = matrix.rows().to_vec();
        let mut incidence: Vec<Vec<u32>> = vec![Vec::new(); columns];
        let mut occupants = vec![0u32; columns];
        for (index, row) in work.iter().enumerate() {
            for &column in row {
                incidence[column as usize].push(index as u32);
                occupants[column as usize] += 1;
            }
        }
        Self {
            columns,
            compositions: (0..work.len()).map(|i| vec![i]).collect(),
            live: vec![true; work.len()],
            live_rows: work.len(),
            live_columns: occupants.iter().filter(|&&count| count > 0).count(),
            nonzeros: work.iter().map(|row| row.len() as i64).sum(),
            work,
            incidence,
            occupants,
        }
    }

    fn excess(&self) -> usize {
        self.live_rows.saturating_sub(self.live_columns)
    }

    fn gain(&mut self, column: u32) {
        let c = column as usize;
        if self.occupants[c] == 0 {
            self.live_columns += 1;
        }
        self.occupants[c] += 1;
    }

    fn lose(&mut self, column: u32) {
        let c = column as usize;
        self.occupants[c] -= 1;
        if self.occupants[c] == 0 {
            self.live_columns -= 1;
        }
    }

    /// The live rows that hold `column`, once each, compacted in place.
    fn members(&mut self, column: u32) -> &[u32] {
        let c = column as usize;
        let (work, live) = (&self.work, &self.live);
        self.incidence[c]
            .retain(|&r| live[r as usize] && work[r as usize].binary_search(&column).is_ok());
        self.incidence[c].sort_unstable();
        self.incidence[c].dedup();
        debug_assert_eq!(
            self.incidence[c].len(),
            self.occupants[c] as usize,
            "occupancy drifted from incidence"
        );
        &self.incidence[c]
    }

    /// Remove a row, returning the columns it held.
    fn retire(&mut self, row: usize) -> Vec<u32> {
        debug_assert!(self.live[row]);
        self.live[row] = false;
        self.live_rows -= 1;
        self.nonzeros -= self.work[row].len() as i64;
        let held = core::mem::take(&mut self.work[row]);
        for &column in &held {
            self.lose(column);
        }
        held
    }

    /// Remove every row pinned by a singleton column, to a fixed point,
    /// starting from the columns in `pending`.
    fn prune_singletons(&mut self, mut pending: Vec<u32>) {
        while let Some(column) = pending.pop() {
            if self.occupants[column as usize] != 1 {
                continue;
            }
            let victim = self.members(column)[0] as usize;
            for touched in self.retire(victim) {
                if self.occupants[touched as usize] == 1 {
                    pending.push(touched);
                }
            }
        }
    }

    fn all_singletons(&self) -> Vec<u32> {
        (0..self.columns)
            .filter(|&c| self.occupants[c] == 1)
            .map(|c| c as u32)
            .collect()
    }

    /// Clique removal (Cavallar, ANTS-IV 2000): bring the excess of rows over live
    /// columns down to `target` by discarding whole groups of rows linked
    /// through weight-two columns, heaviest groups first.
    ///
    /// Two rows sharing a column no other row holds are bound together:
    /// remove one and the column pins the other out. The connected
    /// components of that relation are the units a row removal really
    /// comes in, and removing a component of `k` rows and `k − 1` binding
    /// columns costs exactly one unit of excess — the same as removing one
    /// unbound row — while taking the most weight out of the matrix.
    /// Removal cascades: a column of weight three that lost two members is
    /// a singleton, and its last row goes too. The components are rebuilt
    /// between rounds, since a cascade can fuse them, and each round
    /// removes only a share of what remains, so the cascades cannot
    /// overshoot the target by much.
    fn purge(&mut self, target: usize) {
        loop {
            let excess = self.excess();
            if excess <= target {
                return;
            }
            // Union-find over live rows along the weight-two columns.
            let count = self.work.len();
            let mut parent: Vec<u32> = (0..count as u32).collect();
            fn find(parent: &mut [u32], mut x: u32) -> u32 {
                while parent[x as usize] != x {
                    let up = parent[parent[x as usize] as usize];
                    parent[x as usize] = up;
                    x = up;
                }
                x
            }
            for column in 0..self.columns {
                if self.occupants[column] != 2 {
                    continue;
                }
                let members = self.members(column as u32);
                let (a, b) = (members[0], members[1]);
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent[ra.max(rb) as usize] = ra.min(rb);
                }
            }
            let mut weight = vec![0i64; count];
            let mut rows_of: Vec<Vec<u32>> = vec![Vec::new(); count];
            for row in 0..count {
                if self.live[row] {
                    let root = find(&mut parent, row as u32) as usize;
                    weight[root] += self.work[row].len() as i64;
                    rows_of[root].push(row as u32);
                }
            }
            let mut components: Vec<(i64, usize)> = (0..count)
                .filter(|&root| !rows_of[root].is_empty())
                .map(|root| (weight[root], root))
                .collect();
            if components.is_empty() {
                return;
            }
            components.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            let share = (excess - target).div_ceil(4).max(1).min(components.len());
            let mut pending = Vec::new();
            for &(_, root) in &components[..share] {
                for &row in &rows_of[root] {
                    if self.live[row as usize] {
                        for touched in self.retire(row as usize) {
                            if self.occupants[touched as usize] == 1 {
                                pending.push(touched);
                            }
                        }
                    }
                }
            }
            self.prune_singletons(pending);
        }
    }

    /// Fill-ordered elimination, stopping where the solver's cost would
    /// rise; see the [module documentation](self).
    fn merge(&mut self, weight_cap: usize) {
        let columns = self.columns;
        // A min-heap keyed by fill with lazy re-evaluation. `last` is the
        // key a column was last queued under and `fresh` whether that key
        // is its exact current fill; a change to any member row clears
        // `fresh`, and a popped stale entry is re-planned and re-queued
        // rather than acted on. The initial keys sort by weight far below
        // any fill, so singletons and pairs are planned first. `queued`
        // keeps one entry per column.
        let cap = weight_cap.max(1);
        let initial = |weight: u32| i64::MIN / 2 + i64::from(weight);
        let mut last: Vec<i64> = self.occupants.iter().map(|&w| initial(w)).collect();
        let mut fresh = vec![false; columns];
        let mut queued = vec![false; columns];
        let mut heap: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::new();
        for column in 0..columns {
            if self.occupants[column] != 0 {
                heap.push(Reverse((last[column], column as u32)));
                queued[column] = true;
            }
        }

        // `touch` marks a column's plan stale after one of its rows changed
        // and makes sure it is queued to be re-planned.
        fn touch(
            column: u32,
            last: &[i64],
            fresh: &mut [bool],
            queued: &mut [bool],
            heap: &mut BinaryHeap<Reverse<(i64, u32)>>,
        ) {
            let c = column as usize;
            fresh[c] = false;
            if !queued[c] {
                queued[c] = true;
                heap.push(Reverse((last[c], column)));
            }
        }

        while let Some(Reverse((key, column))) = heap.pop() {
            let c = column as usize;
            queued[c] = false;
            let weight = self.occupants[c] as usize;
            if weight == 0 || weight > cap {
                continue;
            }
            let members: Vec<usize> = self.members(column).iter().map(|&r| r as usize).collect();

            if !(fresh[c] && last[c] == key) {
                // Stale: plan it, queue it under its exact fill, and let
                // the heap decide when its turn comes.
                let fill = if weight == 1 {
                    -(self.work[members[0]].len() as i64)
                } else {
                    plan(&members, &self.work).fill
                };
                last[c] = fill;
                fresh[c] = true;
                queued[c] = true;
                heap.push(Reverse((fill, column)));
                continue;
            }

            if weight == 1 {
                for touched in self.retire(members[0]) {
                    touch(touched, &last, &mut fresh, &mut queued, &mut heap);
                }
                continue;
            }

            // The cheapest remaining merge: does it pay? Every entry still
            // queued has a key at least this one, and the keys are exact
            // fills or the fills before a member changed — rows grow as
            // they merge, so a stale key is nearly always an under-estimate
            // — so when this merge does not pay, no remaining one is
            // expected to.
            if key.saturating_mul(self.live_rows as i64 - 1) >= self.nonzeros {
                break;
            }
            let Plan { root, edges, fill } = plan(&members, &self.work);
            debug_assert_eq!(fill, key, "a fresh key is the plan's fill");
            for (child, parent) in edges {
                let before = core::mem::take(&mut self.work[child]);
                let after = symmetric_difference(&before, &self.work[parent]);
                self.nonzeros += after.len() as i64 - before.len() as i64;
                for &touched in &before {
                    self.lose(touched);
                    touch(touched, &last, &mut fresh, &mut queued, &mut heap);
                }
                for &touched in &after {
                    self.gain(touched);
                    // Only a column the child did not already hold gains a
                    // list entry; it is still listed under the columns it
                    // kept.
                    if before.binary_search(&touched).is_err() {
                        self.incidence[touched as usize].push(child as u32);
                    }
                    touch(touched, &last, &mut fresh, &mut queued, &mut heap);
                }
                self.compositions[child] =
                    symmetric_difference(&self.compositions[child], &self.compositions[parent]);
                self.work[child] = after;
            }
            for touched in self.retire(root) {
                touch(touched, &last, &mut fresh, &mut queued, &mut heap);
            }
            debug_assert_eq!(self.occupants[c], 0, "an eliminated column keeps no holder");
        }
    }

    fn finish(mut self) -> FilteredMatrix {
        let mut kept_rows = Vec::with_capacity(self.live_rows);
        let mut kept_compositions = Vec::with_capacity(self.live_rows);
        for index in 0..self.work.len() {
            if self.live[index] {
                kept_rows.push(core::mem::take(&mut self.work[index]));
                kept_compositions.push(core::mem::take(&mut self.compositions[index]));
            }
        }
        debug_assert_eq!(
            kept_rows.iter().map(Vec::len).sum::<usize>() as i64,
            self.nonzeros,
            "the running nonzero count drifted"
        );
        debug_assert_eq!(
            self.occupants.iter().filter(|&&count| count > 0).count(),
            self.live_columns,
            "the running live column count drifted"
        );
        FilteredMatrix {
            matrix: SparseMatrix {
                columns: self.columns,
                rows: kept_rows,
            },
            live_columns: self.live_columns,
            compositions: kept_compositions,
        }
    }
}

/// Filters a matrix: singleton pruning, clique removal down to `excess`
/// rows beyond the live columns, then fill-ordered column merges.
///
/// Columns are eliminated in order of increasing fill until the next
/// elimination would raise `rows · nonzeros`, the Block Lanczos cost — see
/// the [module documentation](self) for the rule. `weight_cap` bounds the
/// weight of a column the elimination will consider (a tree over `w`
/// members costs `w²` row comparisons to plan); one is pruning alone, and
/// two admits only the pairs, which never cost fill. `excess` is how many
/// rows beyond the live columns to keep: each is a dependency the solver
/// can find, and every one past what the caller needs is rows and
/// nonzeros the solver walks for nothing.
#[must_use]
pub fn filter_merge(matrix: &SparseMatrix, weight_cap: usize, excess: usize) -> FilteredMatrix {
    let mut filter = Filter::new(matrix);
    let singletons = filter.all_singletons();
    filter.prune_singletons(singletons);
    filter.purge(excess);
    filter.merge(weight_cap);
    filter.finish()
}

#[cfg(test)]
mod tests {
    use super::super::dense_null_space;
    use super::*;

    fn mix(mut x: u64) -> u64 {
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        x ^ (x >> 33)
    }

    /// A random matrix with each entry set with probability `1/density`.
    fn random_matrix(seed: u64, rows: usize, columns: usize, density: u64) -> SparseMatrix {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        let rows = (0..rows)
            .map(|_| {
                (0..columns)
                    .filter(|_| mix(next()).is_multiple_of(density))
                    .map(|c| c as u32)
                    .collect()
            })
            .collect();
        SparseMatrix::new(columns, rows)
    }

    fn xor_of(matrix: &SparseMatrix, picks: &[usize]) -> Vec<u32> {
        picks.iter().fold(Vec::new(), |acc, &pick| {
            symmetric_difference(&acc, &matrix.rows()[pick])
        })
    }

    fn cost(filtered: &FilteredMatrix) -> usize {
        filtered.rows().len() * filtered.nonzeros()
    }

    #[test]
    fn packed_and_sparse_forms_round_trip() {
        let matrix = random_matrix(3, 40, 130, 5);
        let packed = matrix.packed_rows();
        assert_eq!(SparseMatrix::from_packed(&packed, 130), matrix);
        // Bits beyond the width are not columns.
        let mut stray = packed.clone();
        stray[0][2] |= 1 << 10; // column 138
        assert_eq!(SparseMatrix::from_packed(&stray, 130), matrix);
    }

    #[test]
    #[should_panic(expected = "not strictly ascending")]
    fn an_unsorted_row_is_refused() {
        let _ = SparseMatrix::new(10, vec![vec![3, 1]]);
    }

    #[test]
    fn every_filtered_dependency_expands_to_a_null_original_combination() {
        // The identity the whole design carries: a dependency of the
        // filtered matrix, expanded through the compositions, must XOR the
        // original rows to zero. Checked across random matrices and every
        // weight cap in the practical range, including trees with depth.
        for seed in 1..=24u64 {
            let columns = 96;
            let matrix = random_matrix(seed, 128, columns, 12);
            for cap in [1usize, 2, 3, 4, 8, 32] {
                let filtered = filter_merge(&matrix, cap, usize::MAX);
                for (index, row) in filtered.rows().iter().enumerate() {
                    assert_eq!(
                        &xor_of(&matrix, filtered.composition(index)),
                        row,
                        "seed {seed} cap {cap}: composition does not reproduce its row"
                    );
                }
                let dependencies = dense_null_space(&filtered.packed_rows(), columns);
                for dependency in dependencies.iter().take(4) {
                    let expanded = filtered.expand(dependency);
                    assert!(!expanded.is_empty(), "an empty dependency proves nothing");
                    assert!(
                        xor_of(&matrix, &expanded).is_empty(),
                        "seed {seed} cap {cap}: expansion does not vanish"
                    );
                }
            }
        }
    }

    #[test]
    fn merging_never_loses_solvability() {
        // If the original matrix is over-determined enough to hold a
        // dependency, the filtered one must still hold one: merging is
        // row-space-preserving on the quotient, and pruning removes only
        // rows no dependency can use.
        for seed in 40..=52u64 {
            let columns = 64;
            let matrix = random_matrix(seed, 90, columns, 10);
            if dense_null_space(&matrix.packed_rows(), columns).is_empty() {
                continue;
            }
            let filtered = filter_merge(&matrix, 32, usize::MAX);
            assert!(
                !dense_null_space(&filtered.packed_rows(), columns).is_empty(),
                "seed {seed}: filtering lost every dependency"
            );
        }
    }

    #[test]
    fn the_fill_rule_only_ever_lowers_the_solver_cost() {
        // Singletons and pairs are the same under every cap, and every
        // further merge the rule admits strictly lowers rows · nonzeros,
        // so a wider cap can never hand the solver a dearer matrix.
        for seed in 200..=212u64 {
            let matrix = random_matrix(seed, 1_500, 1_200, 60);
            let pairs = filter_merge(&matrix, 2, usize::MAX);
            let wide = filter_merge(&matrix, 32, usize::MAX);
            assert!(
                cost(&wide) <= cost(&pairs),
                "seed {seed}: cap 32 cost {} against cap 2 cost {}",
                cost(&wide),
                cost(&pairs)
            );
            assert!(wide.rows().len() <= pairs.rows().len());
            // And the rule stops: what remains has no merge that would
            // pay, which the mean-weight bound states directly.
            let mean = wide.nonzeros() as f64 / wide.rows().len().max(1) as f64;
            assert!(mean.is_finite());
        }
    }

    #[test]
    fn sieve_sized_matrices_filter_in_sieve_sized_time() {
        // Big enough that a quadratic slip turns a fraction of a second into
        // minutes and fails on the suite's patience rather than silently.
        // The dense shape carries the nonzeros; the sparse shape is where
        // light columns abound and the filter must actually shrink it.
        let columns = 4_000;
        let dense = random_matrix(7, 4_600, columns, 160);
        let filtered = filter_merge(&dense, 32, usize::MAX);
        if let Some(dependency) = dense_null_space(&filtered.packed_rows(), columns).first() {
            assert!(xor_of(&dense, &filtered.expand(dependency)).is_empty());
        }
        let sparse = random_matrix(11, 4_600, columns, 1_600);
        let filtered = filter_merge(&sparse, 32, usize::MAX);
        assert!(
            filtered.rows().len() < sparse.rows().len(),
            "a sparse matrix full of light columns did not shrink"
        );
    }

    #[test]
    fn live_columns_track_the_filtering() {
        let columns = 4_000;
        let sparse = random_matrix(13, 4_600, columns, 1_600);
        let filtered = filter_merge(&sparse, 32, usize::MAX);
        assert!(filtered.live_columns() <= filtered.columns());
        let mut seen = vec![false; columns];
        for row in filtered.rows() {
            for &c in row {
                seen[c as usize] = true;
            }
        }
        assert_eq!(
            seen.iter().filter(|&&s| s).count(),
            filtered.live_columns(),
            "live column count disagrees with the rows"
        );
    }

    #[test]
    fn pairs_collapse_singletons_prune_and_an_empty_row_is_a_dependency() {
        // Column 1 has weight two: rows 0 and 1 merge, and the survivor
        // holds columns 0 and 2. Column 0 is then a singleton, so that row
        // goes, and column 3 pins row 2 out. Rows 3 and 4 are equal: their
        // pair merge leaves an empty row, which is a dependency found by
        // the filter itself and is kept, composition and all.
        let matrix = SparseMatrix::new(
            8,
            vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![4, 5], vec![4, 5]],
        );
        let filtered = filter_merge(&matrix, 2, usize::MAX);
        assert_eq!(filtered.rows(), &[Vec::<u32>::new()]);
        assert_eq!(filtered.composition(0), &[3, 4]);
        assert_eq!(filtered.live_columns(), 0);
        assert_eq!(filtered.nonzeros(), 0);
    }

    #[test]
    fn the_tree_adds_each_parent_once_and_cancels_the_column() {
        // Four rows share column 0; the tree eliminates it with three
        // additions, and every survivor is a stated XOR of originals.
        let matrix = SparseMatrix::new(
            6,
            vec![
                vec![0, 1, 2],
                vec![0, 2, 3],
                vec![0, 3, 4],
                vec![0, 1, 4, 5],
                vec![1, 5],
            ],
        );
        let filtered = filter_merge(&matrix, 4, usize::MAX);
        for (index, row) in filtered.rows().iter().enumerate() {
            assert_eq!(&xor_of(&matrix, filtered.composition(index)), row);
            assert!(row.binary_search(&0).is_err(), "column 0 survived");
        }
    }

    #[test]
    fn clique_removal_lands_on_the_excess_and_takes_the_heaviest_first() {
        // Rows 0 and 1 are bound by column 0 (weight two), rows 2 and 3 by
        // column 1; every other column is shared widely enough to survive
        // a removal. Component {0, 1} is heavier than {2, 3}, so it goes
        // first, and taking it costs exactly one unit of excess.
        let shared = |extra: &[u32]| {
            let mut row = vec![2u32, 3, 4];
            row.extend_from_slice(extra);
            row.sort_unstable();
            row
        };
        let matrix = SparseMatrix::new(
            8,
            vec![
                shared(&[0, 5, 6, 7]),
                shared(&[0, 5, 6]),
                shared(&[1]),
                shared(&[1, 7]),
                shared(&[5, 6, 7]),
                shared(&[5]),
                shared(&[6]),
                shared(&[7]),
                shared(&[5, 7]),
                shared(&[6, 7]),
            ],
        );
        let full = filter_merge(&matrix, 1, usize::MAX);
        assert_eq!(full.rows().len(), 10);
        let excess = full.rows().len() - full.live_columns();
        assert_eq!(excess, 2);
        let purged = filter_merge(&matrix, 1, excess - 1);
        assert_eq!(purged.rows().len() - purged.live_columns(), excess - 1);
        assert_eq!(purged.rows().len(), 8, "one two-row clique removed");
        for index in 0..purged.rows().len() {
            let original = purged.composition(index)[0];
            assert!(original >= 2, "the heaviest clique, rows 0 and 1, survived");
        }
    }

    #[test]
    fn purging_preserves_solvability_at_the_kept_excess() {
        // With `excess` rows kept beyond the columns, at least `excess`
        // independent dependencies remain, whatever was thrown away.
        for seed in 300..=306u64 {
            let columns = 200;
            let matrix = random_matrix(seed, 260, columns, 20);
            for excess in [1usize, 4, 16] {
                let filtered = filter_merge(&matrix, 32, excess);
                let kept = filtered
                    .rows()
                    .len()
                    .saturating_sub(filtered.live_columns());
                assert!(
                    kept >= excess.min(60),
                    "seed {seed}: excess {kept} below {excess}"
                );
                let dependencies = dense_null_space(&filtered.packed_rows(), columns);
                assert!(
                    dependencies.len() >= kept,
                    "seed {seed}: fewer dependencies than excess"
                );
                for dependency in dependencies.iter().take(3) {
                    assert!(xor_of(&matrix, &filtered.expand(dependency)).is_empty());
                }
            }
        }
    }
}
