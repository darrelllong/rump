//! Structured Gaussian elimination over GF(2) on sparse rows: shrinking a
//! very sparse matrix before an iterative solver looks for its dependencies,
//! as between a sieve and its linear algebra.
//!
//! A sieve matrix has a few dozen nonzeros in a row and hundreds of
//! thousands of columns, most of them touched by one or two rows. Packed one
//! bit per column, every row is a kilobyte-scale word array and every XOR or
//! comparison walks the whole width. [`SparseMatrix`] holds each row as its
//! ascending column indices, so every operation costs the row's weight.
//!
//! # The elimination
//!
//! Structured Gaussian elimination as NFS practice has it (Cavallar,
//! *Strategies in filtering in the number field sieve*, ANTS-IV, LNCS
//! 1838 (2000), 209–231; the modern treatment is Bouillaguet &
//! Zimmermann, *Parallel structured Gaussian elimination for the number
//! field sieve*, Mathematical Cryptology 1 (2021), 22–39): a column held by
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
//! is Block Lanczos (Montgomery, *A block Lanczos algorithm for finding
//! dependencies over GF(2)*, EUROCRYPT '95, LNCS 921, 106–120), whose cost
//! is `rows/64` iterations of a pass over
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
//! directly against the original rows.

use core::cmp::Reverse;
use std::collections::BinaryHeap;

use super::{set_bits, words_for, WORD};

/// A purge round removes this fraction of the remaining excess, rounded
/// up. No share can take the excess below the target: a component costs at
/// most one unit of it and a row the cascade takes costs none. What the
/// rounds buy is that the components the cascades fused are weighed again,
/// fused, before more are chosen. Measured on the matrix of a 120-digit
/// sieve, 2 313 268 rows with 81 696 of excess: all of it in one round left
/// 677 903 rows and 60 778 608 nonzeros after merging, and by halves or by
/// quarters 673 757 and 60 223 415, the same matrix.
const PURGE_SHARE_DIVISOR: usize = 4;

/// In [`Cliques`], the row after a component's last.
const LAST: u32 = u32::MAX;

/// The entries of the merge's queue read past a stale column for others to
/// plan with it. A span too short makes batches too small to pay for their
/// threads, and one too long is overtaken: a column touched inside the span
/// already read is planned alone when its turn comes. Measured on the
/// relations of a 120-digit sieve, two EPYC 7452, the merge took 37.4 s at
/// 2 048 with 8 threads a batch, 34.7 at 4 096 and 34.8 at 8 192 with 16,
/// 36.6 at 16 384 with 32, and 74.0 planning every column alone. At 8 192
/// one plan in forty was of a column overtaken; at 65 536, one in seven.
const PLANS_AHEAD: usize = 8_192;

/// The plans a thread of a batch should have to itself. On the same
/// relations the 517 batches of a merge, 6 500 plans each, took 4.8 s on 16
/// threads, 4.0 on 32, 4.8 on 64 and 7.5 on 128: about two hundred plans a
/// thread.
const PLANS_PER_WORKER: usize = 256;

/// Base of the key a column is first queued under: its weight added to
/// this sorts every column below any fill, in weight order, so singletons
/// and pairs are planned first. The key only ever sorts; a stale entry is
/// re-planned before its key reaches the cost test. Half the minimum rather
/// than the minimum leaves headroom, so that a key can be negated or doubled
/// without overflow should the arithmetic on keys ever grow.
const SENTINEL_KEY_BASE: i64 = i64::MIN / 2;

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

    /// The matrix as bytes, to keep: `GF2M`, the row count, the width, and
    /// each row's length and columns, every number a `u32` little-endian.
    ///
    /// A sieve's filtered matrix kept so is solved again without its
    /// relations or its filtering, which is how a solver is measured.
    ///
    /// # Panics
    ///
    /// Panics if the row count or the width does not fit `u32`, which the
    /// solvers do not take either.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let word = |n: usize| u32::try_from(n).expect("the solvers index in thirty-two bits");
        let mut bytes = Vec::with_capacity(12 + 4 * (self.rows.len() + self.nonzeros()));
        bytes.extend_from_slice(MATRIX_MAGIC);
        bytes.extend_from_slice(&word(self.rows.len()).to_le_bytes());
        bytes.extend_from_slice(&word(self.columns).to_le_bytes());
        for row in &self.rows {
            bytes.extend_from_slice(&word(row.len()).to_le_bytes());
            for &column in row {
                bytes.extend_from_slice(&column.to_le_bytes());
            }
        }
        bytes
    }

    /// The matrix [`Self::to_bytes`] made `bytes` of.
    ///
    /// # Errors
    ///
    /// The bytes are not such a matrix: no header, fewer bytes than the
    /// rows claim, or a row not strictly ascending below the width.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MatrixBytesError> {
        let mut words = bytes
            .strip_prefix(MATRIX_MAGIC)
            .ok_or(MatrixBytesError::Header)?
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")) as usize);
        let count = words.next().ok_or(MatrixBytesError::Header)?;
        let columns = words.next().ok_or(MatrixBytesError::Header)?;
        let mut rows = Vec::with_capacity(count);
        for index in 0..count {
            let length = words.next().ok_or(MatrixBytesError::Short)?;
            let mut row = Vec::with_capacity(length);
            for _ in 0..length {
                let column = words.next().ok_or(MatrixBytesError::Short)?;
                if column >= columns || row.last().is_some_and(|&last| column <= last as usize) {
                    return Err(MatrixBytesError::Row(index));
                }
                row.push(column as u32);
            }
            rows.push(row);
        }
        Ok(Self { columns, rows })
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

/// The first four bytes of a matrix kept by [`SparseMatrix::to_bytes`].
const MATRIX_MAGIC: &[u8; 4] = b"GF2M";

/// Bytes handed to [`SparseMatrix::from_bytes`] that are not a matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MatrixBytesError {
    /// No `GF2M`, row count and width to begin with.
    Header,
    /// Fewer bytes than the rows claim.
    Short,
    /// The row at this index is not strictly ascending below the width.
    Row(usize),
}

impl core::fmt::Display for MatrixBytesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Header => f.write_str("not a GF(2) matrix: no header"),
            Self::Short => f.write_str("not a GF(2) matrix: fewer bytes than its rows claim"),
            Self::Row(index) => write!(
                f,
                "not a GF(2) matrix: row {index} is not ascending below the width"
            ),
        }
    }
}

impl std::error::Error for MatrixBytesError {}

/// A filtered GF(2) matrix: rows merged and pruned, each carrying the set
/// of original rows it is the sum of. [`filter_merge`] builds it.
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
    /// compare row counts against. Filtering removes a row per eliminated
    /// column but never changes the width, so against the full width every
    /// filtered matrix looks under-determined.
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

/// The rows bound together through columns of weight two, kept while the
/// purge removes them: a union-find over the rows whose root is a
/// component's least row, with the component's rows threaded from the root
/// and its weight held there.
///
/// A component goes whole or not at all, since a row retired leaves every
/// column that bound it a singleton, which pins the row at its other end.
/// So from one round to the next components only fuse, where a column has
/// come down to weight two, and nothing is built again.
struct Cliques {
    parent: Vec<u32>,
    /// The next row of the same component, or [`LAST`].
    next: Vec<u32>,
    /// At a root, the last row of its component.
    tail: Vec<u32>,
    /// At a root, the nonzeros in its component's rows.
    weight: Vec<i64>,
    /// The components, heaviest first and among equals the least root
    /// first. An entry is stale once its row is retired, is no longer a
    /// root, or roots a component of another weight.
    heap: BinaryHeap<(i64, Reverse<u32>)>,
}

impl Cliques {
    /// Every live row a component of its own.
    fn new(rows: &[Vec<u32>], live: &[bool]) -> Self {
        let count = rows.len();
        assert!(
            u32::try_from(count).is_ok_and(|count| count < LAST),
            "{count} rows: the purge indexes them in thirty-two bits"
        );
        let weight: Vec<i64> = rows.iter().map(|row| row.len() as i64).collect();
        let heap = (0..count)
            .filter(|&row| live[row])
            .map(|row| (weight[row], Reverse(row as u32)))
            .collect();
        Self {
            parent: (0..count as u32).collect(),
            next: vec![LAST; count],
            tail: (0..count as u32).collect(),
            weight,
            heap,
        }
    }

    fn root(&mut self, mut row: u32) -> u32 {
        while self.parent[row as usize] != row {
            let up = self.parent[self.parent[row as usize] as usize];
            self.parent[row as usize] = up;
            row = up;
        }
        row
    }

    /// Two rows a column binds are of one component.
    fn bind(&mut self, a: u32, b: u32) {
        let (a, b) = (self.root(a), self.root(b));
        if a == b {
            return;
        }
        let (root, joined) = (a.min(b) as usize, a.max(b) as usize);
        self.parent[joined] = root as u32;
        self.next[self.tail[root] as usize] = joined as u32;
        self.tail[root] = self.tail[joined];
        self.weight[root] += self.weight[joined];
        self.heap.push((self.weight[root], Reverse(root as u32)));
    }

    /// The root of the heaviest component not yet taken, which is taken.
    fn heaviest(&mut self, live: &[bool]) -> Option<u32> {
        while let Some((weight, Reverse(root))) = self.heap.pop() {
            let row = root as usize;
            if live[row] && self.parent[row] == root && self.weight[row] == weight {
                return Some(root);
            }
        }
        None
    }
}

/// The columns waiting to be merged, cheapest first, and what is known of
/// their fills.
///
/// A min-heap keyed by fill with lazy re-evaluation: a change to any row
/// that holds a column makes what is known of its fill stale, and a stale
/// entry popped is planned and queued again under its exact fill rather
/// than acted on. The keys columns start under sort by weight below any
/// fill (see [`SENTINEL_KEY_BASE`]).
struct Queue {
    /// The heaviest column the merge takes.
    cap: usize,
    /// The key each column was last queued under.
    last: Vec<i64>,
    /// Whether that key is the column's exact fill.
    fresh: Vec<bool>,
    /// Whether the column has an entry in the heap; it has at most one.
    queued: Vec<bool>,
    /// The fills planned ahead of their columns' turns.
    ahead: Vec<i64>,
    /// Whether a column's fill planned ahead is its exact fill.
    known: Vec<bool>,
    /// The last entry read in looking ahead.
    horizon: Option<(i64, u32)>,
    heap: BinaryHeap<Reverse<(i64, u32)>>,
}

impl Queue {
    /// Every column of `occupants` the merge takes, under its weight.
    fn new(occupants: &[u32], cap: usize) -> Self {
        let columns = occupants.len();
        let mut queue = Self {
            cap,
            last: occupants
                .iter()
                .map(|&weight| SENTINEL_KEY_BASE + i64::from(weight))
                .collect(),
            fresh: vec![false; columns],
            queued: vec![false; columns],
            ahead: vec![0; columns],
            known: vec![false; columns],
            horizon: None,
            heap: BinaryHeap::new(),
        };
        for (column, &weight) in occupants.iter().enumerate() {
            if queue.merges(weight) {
                queue
                    .heap
                    .push(Reverse((queue.last[column], column as u32)));
                queue.queued[column] = true;
            }
        }
        queue
    }

    /// Whether the merge takes a column of `weight`.
    fn merges(&self, weight: u32) -> bool {
        weight != 0 && weight as usize <= self.cap
    }

    /// Whether `key` is the exact fill of `column`.
    fn exact(&self, key: i64, column: u32) -> bool {
        self.fresh[column as usize] && self.last[column as usize] == key
    }

    /// A row that holds `column`, now of `weight`, has changed: what is
    /// known of its fill is stale, and if the merge takes its weight it is
    /// queued to be planned again. A column of another weight would be
    /// dropped when its turn came, and every change of a column's weight
    /// touches it, so it is queued when it comes to one: most of the
    /// columns a merge touches are heavier than the cap.
    fn touch(&mut self, column: u32, weight: u32) {
        let c = column as usize;
        self.fresh[c] = false;
        self.known[c] = false;
        if self.merges(weight) && !self.queued[c] {
            self.queued[c] = true;
            self.heap.push(Reverse((self.last[c], column)));
        }
    }

    /// The cheapest entry, taken from the queue.
    fn pop(&mut self) -> Option<(i64, u32)> {
        let Reverse((key, column)) = self.heap.pop()?;
        self.queued[column as usize] = false;
        Some((key, column))
    }

    /// `column` queued under `fill`, its exact fill.
    fn plan(&mut self, column: u32, fill: i64) {
        let c = column as usize;
        self.last[c] = fill;
        self.fresh[c] = true;
        self.known[c] = false;
        self.queued[c] = true;
        self.heap.push(Reverse((fill, column)));
    }

    /// The stale columns among the next [`PLANS_AHEAD`] entries that the
    /// merge takes and whose fills are not known, if nothing has been read
    /// as far as `entry`. The entries stay queued.
    fn stale_ahead(&mut self, entry: (i64, u32), occupants: &[u32]) -> Vec<u32> {
        if self.horizon.is_some_and(|horizon| entry <= horizon) {
            return Vec::new();
        }
        let mut read = Vec::with_capacity(PLANS_AHEAD);
        let mut stale = Vec::new();
        while read.len() < PLANS_AHEAD {
            let Some(Reverse((key, column))) = self.heap.pop() else {
                break;
            };
            let known = self.known[column as usize] || self.exact(key, column);
            if self.merges(occupants[column as usize]) && !known {
                stale.push(column);
            }
            read.push(Reverse((key, column)));
        }
        self.horizon = read.last().map(|&Reverse(last)| last);
        self.heap.extend(read);
        stale
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

    /// The live rows that hold `column` and the fill of eliminating it,
    /// from the lists as they stand.
    fn planned(&self, column: u32) -> (Vec<u32>, i64) {
        let (work, live) = (&self.work, &self.live);
        let mut members: Vec<u32> = self.incidence[column as usize]
            .iter()
            .copied()
            .filter(|&r| live[r as usize] && work[r as usize].binary_search(&column).is_ok())
            .collect();
        members.sort_unstable();
        members.dedup();
        debug_assert_eq!(
            members.len(),
            self.occupants[column as usize] as usize,
            "occupancy drifted from incidence"
        );
        let rows: Vec<usize> = members.iter().map(|&r| r as usize).collect();
        let fill = match rows[..] {
            [only] => -(work[only].len() as i64),
            _ => plan(&rows, work).fill,
        };
        (members, fill)
    }

    /// The fill of `column`, stale and at the head of `queue` under `key`,
    /// and with it the fills of the stale columns behind it, planned side
    /// by side when they are enough to share out.
    ///
    /// A plan reads the rows and changes nothing, and a fill planned ahead
    /// is used only if no row of its column has changed since, so the
    /// merge is the merge of one thread.
    fn plan_ahead(&mut self, queue: &mut Queue, key: i64, column: u32) {
        let threads = crate::parallel::budget();
        let mut columns = vec![column];
        if threads > 1 {
            columns.extend(queue.stale_ahead((key, column), &self.occupants));
        }
        let workers = (columns.len() / PLANS_PER_WORKER).min(threads);
        if workers < 2 {
            columns.truncate(1);
        }
        let filter = &*self;
        let plans =
            crate::parallel::map_ordered(&columns, workers, |_, &column| filter.planned(column));
        for (column, (members, fill)) in columns.into_iter().zip(plans) {
            self.incidence[column as usize] = members;
            queue.ahead[column as usize] = fill;
            queue.known[column as usize] = true;
        }
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
    /// starting from the columns in `pending`. `lightened` is shown each
    /// column a row removed had held.
    fn prune_singletons(&mut self, mut pending: Vec<u32>, mut lightened: impl FnMut(u32)) {
        while let Some(column) = pending.pop() {
            if self.occupants[column as usize] != 1 {
                continue;
            }
            let victim = self.members(column)[0] as usize;
            for touched in self.retire(victim) {
                if self.occupants[touched as usize] == 1 {
                    pending.push(touched);
                }
                lightened(touched);
            }
        }
    }

    /// Every column of `columns` that two rows hold binds them.
    fn bind(&mut self, cliques: &mut Cliques, columns: impl Iterator<Item = u32>) {
        for column in columns {
            if self.occupants[column as usize] == 2 {
                let members = self.members(column);
                cliques.bind(members[0], members[1]);
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
    /// a singleton, and its last row goes too. A cascade can fuse
    /// components, by bringing a column down to weight two, so each round
    /// removes only a share of what remains and the next chooses among the
    /// components as the round left them.
    fn purge(&mut self, target: usize) {
        if self.excess() <= target {
            return;
        }
        let mut cliques = Cliques::new(&self.work, &self.live);
        self.bind(&mut cliques, 0..self.columns as u32);
        loop {
            let excess = self.excess();
            if excess <= target {
                return;
            }
            let share = (excess - target).div_ceil(PURGE_SHARE_DIVISOR);
            let (mut pending, mut lightened) = (Vec::new(), Vec::new());
            let mut taken = 0;
            while taken < share {
                let Some(root) = cliques.heaviest(&self.live) else {
                    break;
                };
                taken += 1;
                let mut row = root;
                while row != LAST {
                    for touched in self.retire(row as usize) {
                        if self.occupants[touched as usize] == 1 {
                            pending.push(touched);
                        }
                        lightened.push(touched);
                    }
                    row = cliques.next[row as usize];
                }
            }
            if taken == 0 {
                return;
            }
            self.prune_singletons(pending, |column| lightened.push(column));
            self.bind(&mut cliques, lightened.into_iter());
        }
    }

    /// Fill-ordered elimination, stopping where the solver's cost would
    /// rise; see the [module documentation](self).
    fn merge(&mut self, weight_cap: usize) {
        let mut queue = Queue::new(&self.occupants, weight_cap.max(1));
        while let Some((key, column)) = queue.pop() {
            let c = column as usize;
            let weight = self.occupants[c];
            if !queue.merges(weight) {
                continue;
            }
            if !queue.exact(key, column) {
                // Stale: plan it, queue it under its exact fill, and let
                // the heap decide when its turn comes.
                if !queue.known[c] {
                    self.plan_ahead(&mut queue, key, column);
                }
                queue.plan(column, queue.ahead[c]);
                continue;
            }
            let members: Vec<usize> = self.members(column).iter().map(|&r| r as usize).collect();

            if weight == 1 {
                for touched in self.retire(members[0]) {
                    queue.touch(touched, self.occupants[touched as usize]);
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
                    queue.touch(touched, self.occupants[touched as usize]);
                }
                for &touched in &after {
                    self.gain(touched);
                    // Only a column the child did not already hold gains a
                    // list entry; it is still listed under the columns it
                    // kept.
                    if before.binary_search(&touched).is_err() {
                        self.incidence[touched as usize].push(child as u32);
                    }
                    queue.touch(touched, self.occupants[touched as usize]);
                }
                self.compositions[child] =
                    symmetric_difference(&self.compositions[child], &self.compositions[parent]);
                self.work[child] = after;
            }
            for touched in self.retire(root) {
                queue.touch(touched, self.occupants[touched as usize]);
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
/// The trees of the merges are planned on the calling thread's budget of
/// threads, ahead of their turns; the matrix is the same at any budget.
///
/// Columns are eliminated in order of increasing fill until the next
/// elimination would raise `rows · nonzeros`, the Block Lanczos cost: one
/// that changes the nonzero count by `Δ` takes that cost to
/// `(rows − 1)(nonzeros + Δ)`, a gain exactly when
/// `Δ · (rows − 1) < nonzeros`. `weight_cap` bounds the
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
    filter.prune_singletons(singletons, |_| {});
    filter.purge(excess);
    filter.merge(weight_cap);
    filter.finish()
}

#[cfg(test)]
mod tests {
    use super::super::dense_null_space;

    #[test]
    fn a_matrix_kept_as_bytes_is_the_matrix() {
        let rows = vec![vec![0, 5, 9], vec![], vec![2, 3], vec![9]];
        let matrix = SparseMatrix::new(10, rows.clone());
        let bytes = matrix.to_bytes();
        assert_eq!(&bytes[..4], b"GF2M");
        assert_eq!(bytes.len(), 12 + 4 * (4 + 6));
        let back = SparseMatrix::from_bytes(&bytes).expect("the bytes are the matrix's");
        assert_eq!(back.columns(), 10);
        assert_eq!(back.rows(), &rows[..]);
    }

    #[test]
    fn bytes_that_are_not_a_matrix_are_refused() {
        let matrix = SparseMatrix::new(10, vec![vec![1, 4], vec![7]]);
        let bytes = matrix.to_bytes();
        assert_eq!(
            SparseMatrix::from_bytes(b"GF2X"),
            Err(MatrixBytesError::Header)
        );
        assert_eq!(
            SparseMatrix::from_bytes(&bytes[..8]),
            Err(MatrixBytesError::Header)
        );
        assert_eq!(
            SparseMatrix::from_bytes(&bytes[..bytes.len() - 4]),
            Err(MatrixBytesError::Short)
        );
        let mut descending = bytes.clone();
        descending[16..20].copy_from_slice(&4u32.to_le_bytes());
        descending[20..24].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(
            SparseMatrix::from_bytes(&descending),
            Err(MatrixBytesError::Row(0))
        );
        let mut wide = bytes;
        wide[28..32].copy_from_slice(&10u32.to_le_bytes());
        assert_eq!(
            SparseMatrix::from_bytes(&wide),
            Err(MatrixBytesError::Row(1))
        );
    }
    use super::*;

    /// The `fmix64` finalizer of MurmurHash3 (Appleby), so the low bits of
    /// the generator's word below are as well mixed as the high ones.
    fn mix(mut x: u64) -> u64 {
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        x ^ (x >> 33)
    }

    /// A random matrix with each entry set with probability `1/density`,
    /// from Knuth's MMIX linear congruential generator (TAOCP vol. 2,
    /// §3.3.4) at `seed`.
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
        /// Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 3;
        /// Two full words and two bits, so the last packed word is partial.
        const COLUMNS: usize = 2 * WORD + 2;
        let matrix = random_matrix(SEED, 40, COLUMNS, 5);
        let packed = matrix.packed_rows();
        assert_eq!(SparseMatrix::from_packed(&packed, COLUMNS), matrix);
        // A bit beyond the width, in the partial word, is not a column.
        let mut stray = packed.clone();
        stray[0][2] |= 1 << 10;
        assert_eq!(SparseMatrix::from_packed(&stray, COLUMNS), matrix);
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
        /// Arbitrary, fixed so a failure reproduces.
        const SEEDS: std::ops::RangeInclusive<u64> = 1..=24;
        /// Dependencies expanded per matrix and cap; arbitrary.
        const EXPANDED: usize = 4;
        for seed in SEEDS {
            // 128 rows over 96 columns, eight nonzeros a row on average:
            // over-determined, with columns of every small weight.
            let columns = 96;
            let matrix = random_matrix(seed, 128, columns, 12);
            // Caps: pruning alone, pairs, the two smallest trees with
            // depth, and two wider, the last the widest these tests use.
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
                for dependency in dependencies.iter().take(EXPANDED) {
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
        /// Arbitrary, fixed so a failure reproduces.
        const SEEDS: std::ops::RangeInclusive<u64> = 40..=52;
        for seed in SEEDS {
            // 90 rows over 64 columns at about six nonzeros a row: usually
            // over-determined, and the dense check skips the draws that
            // are not.
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
        /// Arbitrary, fixed so a failure reproduces.
        const SEEDS: std::ops::RangeInclusive<u64> = 200..=212;
        for seed in SEEDS {
            // 1500 rows over 1200 columns at twenty nonzeros a row: wide
            // enough that merges of every weight up to the cap occur.
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
        /// Arbitrary, fixed so a failure reproduces.
        const DENSE_SEED: u64 = 7;
        /// Arbitrary, fixed so a failure reproduces.
        const SPARSE_SEED: u64 = 11;
        // 4600 rows over 4000 columns: twenty-five nonzeros a row in the
        // dense shape, two and a half in the sparse.
        let columns = 4_000;
        let dense = random_matrix(DENSE_SEED, 4_600, columns, 160);
        let filtered = filter_merge(&dense, 32, usize::MAX);
        if let Some(dependency) = dense_null_space(&filtered.packed_rows(), columns).first() {
            assert!(xor_of(&dense, &filtered.expand(dependency)).is_empty());
        }
        let sparse = random_matrix(SPARSE_SEED, 4_600, columns, 1_600);
        let filtered = filter_merge(&sparse, 32, usize::MAX);
        assert!(
            filtered.rows().len() < sparse.rows().len(),
            "a sparse matrix full of light columns did not shrink"
        );
    }

    #[test]
    fn live_columns_track_the_filtering() {
        /// Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 13;
        // The sparse shape above: two and a half nonzeros a row, so most
        // columns are emptied by the filter.
        let columns = 4_000;
        let sparse = random_matrix(SEED, 4_600, columns, 1_600);
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

    /// The purge by its definition, the components found afresh each round
    /// from every column of weight two. Returns the rounds it took.
    fn purge_afresh(filter: &mut Filter, target: usize) -> usize {
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                x = parent[x];
            }
            x
        }
        let count = filter.work.len();
        let mut rounds = 0;
        loop {
            let excess = filter.excess();
            if excess <= target {
                return rounds;
            }
            let mut parent: Vec<usize> = (0..count).collect();
            for column in 0..filter.columns {
                if filter.occupants[column] == 2 {
                    let members = filter.members(column as u32);
                    let (a, b) = (members[0] as usize, members[1] as usize);
                    let (a, b) = (find(&mut parent, a), find(&mut parent, b));
                    parent[a.max(b)] = a.min(b);
                }
            }
            let mut weight = vec![0usize; count];
            let mut rows_of = vec![Vec::new(); count];
            for row in (0..count).filter(|&row| filter.live[row]) {
                let root = find(&mut parent, row);
                weight[root] += filter.work[row].len();
                rows_of[root].push(row);
            }
            let mut components: Vec<usize> = (0..count)
                .filter(|&root| !rows_of[root].is_empty())
                .collect();
            if components.is_empty() {
                return rounds;
            }
            rounds += 1;
            components.sort_unstable_by_key(|&root| (Reverse(weight[root]), root));
            let share = (excess - target)
                .div_ceil(PURGE_SHARE_DIVISOR)
                .min(components.len());
            let mut pending = Vec::new();
            for &root in &components[..share] {
                for &row in &rows_of[root] {
                    for touched in filter.retire(row) {
                        if filter.occupants[touched as usize] == 1 {
                            pending.push(touched);
                        }
                    }
                }
            }
            filter.prune_singletons(pending, |_| {});
        }
    }

    #[test]
    fn the_components_kept_up_are_the_components_found_afresh() {
        /// Arbitrary, fixed so a failure reproduces.
        const SEEDS: std::ops::RangeInclusive<u64> = 500..=507;
        // 1300 rows over 1000 columns, at five nonzeros a row and at three:
        // some three hundred of excess once the singletons are pruned, so a
        // purge is eighteen rounds or so. The columns of weight two are
        // about 27 at five a row and 170 at three.
        for density in [200, 330] {
            for seed in SEEDS {
                let matrix = random_matrix(seed, 1_300, 1_000, density);
                let pruned = || {
                    let mut filter = Filter::new(&matrix);
                    let singletons = filter.all_singletons();
                    filter.prune_singletons(singletons, |_| {});
                    filter
                };
                // Targets: all but one dependency purged, and two between.
                for target in [1usize, 16, 64] {
                    let (mut kept, mut afresh) = (pruned(), pruned());
                    kept.purge(target);
                    let rounds = purge_afresh(&mut afresh, target);
                    assert!(rounds > 1, "density {density} seed {seed}: one round");
                    let case = format!("density {density} seed {seed} target {target}");
                    assert_eq!(kept.live, afresh.live, "{case}");
                    assert_eq!(kept.work, afresh.work, "{case}");
                    assert_eq!(kept.occupants, afresh.occupants, "{case}");
                }
            }
        }
    }

    #[test]
    fn the_merge_is_the_merge_of_one_thread_at_any_budget() {
        /// Arbitrary, fixed so a failure reproduces.
        const SEEDS: std::ops::RangeInclusive<u64> = 600..=603;
        for seed in SEEDS {
            // 6000 rows over 5000 columns at eight nonzeros a row: every
            // column is of a weight the merge takes, so the first plans
            // are five thousand side by side, and the merges that follow
            // touch columns planned ahead.
            let matrix = random_matrix(seed, 6_000, 5_000, 625);
            let alone = crate::parallel::with_budget(1, || filter_merge(&matrix, 32, 64));
            assert!(alone.rows().len() < matrix.rows().len(), "seed {seed}");
            // Budgets: the fewest that share, and more than a batch of
            // five thousand is cut for.
            for threads in [2usize, 32] {
                let shared =
                    crate::parallel::with_budget(threads, || filter_merge(&matrix, 32, 64));
                assert_eq!(
                    shared.rows(),
                    alone.rows(),
                    "seed {seed}, {threads} threads"
                );
                assert_eq!(
                    shared.compositions, alone.compositions,
                    "seed {seed}, {threads} threads"
                );
            }
        }
    }

    #[test]
    fn purging_preserves_solvability_at_the_kept_excess() {
        // With `excess` rows kept beyond the columns, at least `excess`
        // independent dependencies remain, whatever was thrown away.
        /// Arbitrary, fixed so a failure reproduces.
        const SEEDS: std::ops::RangeInclusive<u64> = 300..=306;
        /// The most excess the assertion demands be kept. Every tested
        /// excess is below it, so the demand is the full excess each time.
        const DEMANDED_EXCESS_CAP: usize = 60;
        /// Dependencies expanded per matrix and excess; arbitrary.
        const EXPANDED: usize = 3;
        for seed in SEEDS {
            // 260 rows over 200 columns at ten nonzeros a row: sixty rows
            // of excess before filtering, and the excesses asked for
            // below are well under that.
            let columns = 200;
            let matrix = random_matrix(seed, 260, columns, 20);
            // Excesses: one dependency, a few, and many.
            for excess in [1usize, 4, 16] {
                let filtered = filter_merge(&matrix, 32, excess);
                let kept = filtered
                    .rows()
                    .len()
                    .saturating_sub(filtered.live_columns());
                assert!(
                    kept >= excess.min(DEMANDED_EXCESS_CAP),
                    "seed {seed}: excess {kept} below {excess}"
                );
                let dependencies = dense_null_space(&filtered.packed_rows(), columns);
                assert!(
                    dependencies.len() >= kept,
                    "seed {seed}: fewer dependencies than excess"
                );
                for dependency in dependencies.iter().take(EXPANDED) {
                    assert!(xor_of(&matrix, &filtered.expand(dependency)).is_empty());
                }
            }
        }
    }
}
