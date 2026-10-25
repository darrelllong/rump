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
//! Column weight limits planning work; fill determines the estimated benefit
//! of a merge. The solver this feeds
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
//! The merge is in passes (Bouillaguet & Zimmermann, §4): each takes, the
//! cheapest first in the spirit of Markowitz's rule (*The elimination form
//! of the inverse and its application to linear programming*, Management
//! Science 3 (1957), 255–269), the columns whose fill lowers this cost
//! and whose rows no column already taken holds, and merges them side by
//! side; the passes end when no column lowers it. Neither the order nor
//! the cost proxy establishes a global optimum. `weight_cap` bounds the
//! number of members in a planned tree.
//!
//! # Dependency reconstruction
//!
//! Every surviving row records the set of original rows it is the sum of.
//! A dependency over the filtered matrix expands to one over the original
//! by symmetric difference of those sets, an identity the tests check
//! directly against the original rows.

use core::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

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

/// In a pass of the merge, a column the merge does not take.
const NOT_LIGHT: u32 = u32::MAX;

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
    /// How many rows the original matrix had.
    originals: usize,
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
    /// original rows, by symmetric difference of the compositions: each
    /// turns over its originals' bits, one bit an original row, and the
    /// bits left set are the dependency, ascending.
    #[must_use]
    pub fn expand(&self, dependency: &[usize]) -> Vec<usize> {
        let words = words_for(self.originals);
        let mut odd = vec![0u64; words];
        for &row in dependency {
            for &original in &self.compositions[row] {
                odd[original / WORD] ^= 1 << (original % WORD);
            }
        }
        set_bits(&odd, words, self.originals).collect()
    }
}

/// The original rows `row` is the sum of: its set in `sets`, or `alone`, the
/// row itself, where the set is empty because it has not been merged.
fn composition<'a>(sets: &'a [Vec<usize>], row: usize, alone: &'a [usize; 1]) -> &'a [usize] {
    if sets[row].is_empty() {
        alone
    } else {
        &sets[row]
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
    fn new(rows: &[Vec<u32>], live: &[AtomicBool]) -> Self {
        let count = rows.len();
        assert!(
            u32::try_from(count).is_ok_and(|count| count < LAST),
            "{count} rows: the purge indexes them in thirty-two bits"
        );
        let weight: Vec<i64> = rows.iter().map(|row| row.len() as i64).collect();
        let heap = (0..count)
            .filter(|&row| live[row].load(Ordering::Relaxed))
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
    fn heaviest(&mut self, live: &[AtomicBool]) -> Option<u32> {
        while let Some((weight, Reverse(root))) = self.heap.pop() {
            let row = root as usize;
            if live[row].load(Ordering::Relaxed)
                && self.parent[row] == root
                && self.weight[row] == weight
            {
                return Some(root);
            }
        }
        None
    }
}

/// A column held by more than this share of the rows, and by this many
/// times the merge's cap, is given no list of its rows at the start: the
/// characters and the sign and count columns, held by about half, and the
/// smallest primes, two fifths of all the entries at RSA-155 together. A
/// column's density stays about what it is as rows go, so one this dense
/// comes down to the cap only once the rows are fewer by as much again;
/// if one does, its list is made then ([`Filter::track`]).
const UNTRACKED_SHARE: usize = 64;

/// The runs of rows a thread takes in turn where the rows are dealt out, so
/// that one run slower than the rest is an eighth of a thread's share.
const CHUNKS_PER_THREAD: usize = 8;

/// The rows of each column, as the filter keeps them: the rows the matrix
/// held at the start, in one array a column's slot after another, each
/// slot as long as the column's first weight; and the rows merges have
/// added since, beside. Rows retired, or that have cancelled the column,
/// linger and are compacted away when the column's members are asked for.
///
/// The slots are made side by side: the rows are counted into each
/// column, the slots laid out from the counts, and the rows written into
/// them, each by a worker, at a place a column's cursor hands out. The
/// order a slot is filled in is not the rows' order, and every reader
/// sorts what it reads.
#[derive(Default)]
struct Lists {
    /// Each column's slot: where it starts, how much of it is in use, and
    /// how long it is, one record a column, so that a column's state is one
    /// reach and not four.
    slots_of: Vec<Slot>,
    slots: Vec<AtomicU32>,
    /// Rows added to a column whose slot is full; empty unless it is.
    added: Vec<Vec<u32>>,
}

/// A column's place in [`Lists`]. A column given no list has a room of
/// [`UNTRACKED`].
#[derive(Clone, Copy)]
struct Slot {
    start: usize,
    used: u32,
    room: u32,
}

/// The room of a column given no list; see [`UNTRACKED_SHARE`].
const UNTRACKED: u32 = u32::MAX;

impl Lists {
    /// The lists of `rows` over `columns`, and each column's weight.
    fn new(rows: &[Vec<u32>], columns: usize, cap: usize) -> (Self, Vec<u32>) {
        let threads = crate::parallel::budget();
        let length = rows.len().div_ceil(threads * CHUNKS_PER_THREAD).max(1);
        let chunks: Vec<&[Vec<u32>]> = rows.chunks(length).collect();
        let counts: Vec<AtomicU32> = (0..columns).map(|_| AtomicU32::new(0)).collect();
        crate::parallel::map_ordered(&chunks, threads, |_, chunk| {
            for row in *chunk {
                for &column in row {
                    counts[column as usize].fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        let occupants: Vec<u32> = counts.into_iter().map(AtomicU32::into_inner).collect();
        let mut slots_of = Vec::with_capacity(columns);
        let mut end = 0;
        for &weight in &occupants {
            let dense = weight as usize * UNTRACKED_SHARE > rows.len()
                && weight as usize > UNTRACKED_SHARE * cap;
            slots_of.push(Slot {
                start: end,
                used: 0,
                room: if dense { UNTRACKED } else { weight },
            });
            if !dense {
                end += weight as usize;
            }
        }
        let slots: Vec<AtomicU32> = (0..end).map(|_| AtomicU32::new(0)).collect();
        let cursors: Vec<AtomicU32> = (0..columns).map(|_| AtomicU32::new(0)).collect();
        crate::parallel::map_ordered(&chunks, threads, |number, chunk| {
            let offset = number * length;
            for (index, row) in chunk.iter().enumerate() {
                for &column in row {
                    let slot = slots_of[column as usize];
                    if slot.room != UNTRACKED {
                        let place = slot.start
                            + cursors[column as usize].fetch_add(1, Ordering::Relaxed) as usize;
                        slots[place].store((offset + index) as u32, Ordering::Relaxed);
                    }
                }
            }
        });
        for (slot, cursor) in slots_of.iter_mut().zip(cursors) {
            slot.used = cursor.into_inner();
        }
        let lists = Self {
            slots_of,
            slots,
            added: vec![Vec::new(); columns],
        };
        (lists, occupants)
    }

    /// Whether `column` was given no list and has not been given one since.
    fn untracked(&self, column: u32) -> bool {
        self.slots_of[column as usize].room == UNTRACKED
    }

    /// The rows listed under `column`, in no order, some perhaps no longer
    /// holding it.
    fn listed(&self, column: u32) -> impl Iterator<Item = u32> + '_ {
        let c = column as usize;
        let slot = self.slots_of[c];
        let full = slot.used == slot.room;
        self.slots[slot.start..slot.start + slot.used as usize]
            .iter()
            .map(|slot| slot.load(Ordering::Relaxed))
            .chain(
                full.then(|| self.added[c].iter().copied())
                    .into_iter()
                    .flatten(),
            )
    }

    /// `members` as the list of `column`, in its slot as far as it fits. A
    /// column given no list is given one, the whole of it beside.
    fn set(&mut self, column: u32, members: &[u32]) {
        let c = column as usize;
        let slot = &mut self.slots_of[c];
        let was_full = slot.used == slot.room;
        if slot.room == UNTRACKED {
            slot.room = 0;
        }
        let kept = members.len().min(slot.room as usize);
        for (place, &row) in self.slots[slot.start..slot.start + kept]
            .iter()
            .zip(members)
        {
            place.store(row, Ordering::Relaxed);
        }
        slot.used = kept as u32;
        if was_full || kept < members.len() {
            self.added[c].clear();
            self.added[c].extend_from_slice(&members[kept..]);
        }
    }
}

/// One column's elimination in a pass of the merge: each child's new row
/// and composition, the root retired, the change in nonzeros, and the
/// columns it emptied and brought back.
struct Merged {
    rows: Vec<(usize, Vec<u32>, Vec<usize>)>,
    root: usize,
    entries: i64,
    emptied: usize,
    revived: usize,
}

/// What a run of rows retired side by side did.
#[derive(Default)]
struct Retired {
    rows: usize,
    entries: i64,
    emptied: usize,
    singletons: Vec<u32>,
    twos: Vec<u32>,
}

/// The working state of one filtering run: rows, their compositions, and
/// the column incidence kept exact as rows change.
struct Filter {
    columns: usize,
    work: Vec<Vec<u32>>,
    /// The original rows each row is the sum of; empty for a row not yet
    /// merged, which is itself alone. A merged row's set is never empty:
    /// the live rows' sets stay independent, so no two are equal.
    compositions: Vec<Vec<usize>>,
    /// Whether each row is live; atomic, so that rows can be retired side
    /// by side, a row claimed by turning its flag.
    live: Vec<AtomicBool>,
    live_rows: usize,
    live_columns: usize,
    nonzeros: i64,
    /// Column incidence as row lists, maintained incrementally; the
    /// occupancy counts are exact throughout.
    lists: Lists,
    /// Each column's weight; atomic for the same reason.
    occupants: Vec<AtomicU32>,
}

impl Filter {
    fn new(matrix: SparseMatrix, cap: usize) -> Self {
        let columns = matrix.columns();
        let work: Vec<Vec<u32>> = matrix.into_rows();
        let (lists, occupants) = Lists::new(&work, columns, cap);
        Self {
            columns,
            compositions: vec![Vec::new(); work.len()],
            live: (0..work.len()).map(|_| AtomicBool::new(true)).collect(),
            live_rows: work.len(),
            live_columns: occupants.iter().filter(|&&count| count > 0).count(),
            nonzeros: work.iter().map(|row| row.len() as i64).sum(),
            work,
            lists,
            occupants: occupants.into_iter().map(AtomicU32::new).collect(),
        }
    }

    /// The live rows that hold `column`.
    fn weight(&self, column: u32) -> u32 {
        self.occupants[column as usize].load(Ordering::Relaxed)
    }

    fn is_live(&self, row: usize) -> bool {
        self.live[row].load(Ordering::Relaxed)
    }

    fn excess(&self) -> usize {
        self.live_rows.saturating_sub(self.live_columns)
    }

    /// A list for `column`, given none at the start because it was dense,
    /// made from the live rows that hold it: the rows are scanned side by
    /// side, once for each such column that comes to need a list.
    fn track(&mut self, column: u32) {
        if !self.lists.untracked(column) {
            return;
        }
        let (work, live) = (&self.work, &self.live);
        let threads = crate::parallel::budget();
        let length = work.len().div_ceil(threads * CHUNKS_PER_THREAD).max(1);
        let chunks: Vec<usize> = (0..work.len().div_ceil(length)).collect();
        let found = crate::parallel::map_ordered(&chunks, threads, |_, &chunk| {
            (chunk * length..((chunk + 1) * length).min(work.len()))
                .filter(|&r| {
                    live[r].load(Ordering::Relaxed) && work[r].binary_search(&column).is_ok()
                })
                .map(|r| r as u32)
                .collect::<Vec<u32>>()
        });
        let members: Vec<u32> = found.into_iter().flatten().collect();
        self.lists.set(column, &members);
    }

    /// The live rows that hold `column`, once each and ascending, from the
    /// lists as they stand.
    fn holders(&self, column: u32) -> Vec<u32> {
        let (work, live) = (&self.work, &self.live);
        let mut members: Vec<u32> = self
            .lists
            .listed(column)
            .filter(|&r| {
                live[r as usize].load(Ordering::Relaxed)
                    && work[r as usize].binary_search(&column).is_ok()
            })
            .collect();
        members.sort_unstable();
        members.dedup();
        debug_assert_eq!(
            members.len(),
            self.weight(column) as usize,
            "occupancy drifted from incidence"
        );
        members
    }

    /// The one live row that holds `column`, a singleton, of a column that
    /// has a list: the first such row the list names, with nothing
    /// gathered, sorted or written back.
    fn holder(&self, column: u32) -> usize {
        debug_assert_eq!(self.weight(column), 1, "a singleton");
        let (work, live) = (&self.work, &self.live);
        self.lists
            .listed(column)
            .find(|&r| {
                live[r as usize].load(Ordering::Relaxed)
                    && work[r as usize].binary_search(&column).is_ok()
            })
            .expect("a singleton's row is listed") as usize
    }

    /// Retire the live rows among `rows`, side by side: a row is claimed
    /// by turning its flag, so one named twice goes once, and its columns'
    /// weights are counted down where they stand. A column's weight before
    /// says what became of it: two, a singleton; three, held by two; one,
    /// empty. Returns the columns the rows left singletons and those they
    /// left at two. A row retired so keeps its entries until
    /// [`Self::release`]; nothing reads a retired row.
    fn retire_all(&mut self, rows: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let threads = crate::parallel::budget();
        let length = rows.len().div_ceil(threads * CHUNKS_PER_THREAD).max(1);
        let chunks: Vec<&[u32]> = rows.chunks(length).collect();
        let (work, live, occupants) = (&self.work, &self.live, &self.occupants);
        let retired = crate::parallel::map_ordered(&chunks, threads, |_, chunk| {
            let mut retired = Retired::default();
            for &row in *chunk {
                if !live[row as usize].swap(false, Ordering::Relaxed) {
                    continue;
                }
                retired.rows += 1;
                retired.entries += work[row as usize].len() as i64;
                for &column in &work[row as usize] {
                    match occupants[column as usize].fetch_sub(1, Ordering::Relaxed) {
                        1 => retired.emptied += 1,
                        2 => retired.singletons.push(column),
                        3 => retired.twos.push(column),
                        _ => {}
                    }
                }
            }
            retired
        });
        let (mut singletons, mut twos) = (Vec::new(), Vec::new());
        for part in retired {
            self.live_rows -= part.rows;
            self.nonzeros -= part.entries;
            self.live_columns -= part.emptied;
            singletons.extend(part.singletons);
            twos.extend(part.twos);
        }
        (singletons, twos)
    }

    /// The entries of the rows retired side by side, given back.
    fn release(&mut self) {
        let threads = crate::parallel::budget();
        let length = self.work.len().div_ceil(threads * CHUNKS_PER_THREAD).max(1);
        let live = &self.live;
        std::thread::scope(|scope| {
            for (number, rows) in self.work.chunks_mut(length).enumerate() {
                scope.spawn(move || {
                    for (index, row) in rows.iter_mut().enumerate() {
                        if !live[number * length + index].load(Ordering::Relaxed) {
                            *row = Vec::new();
                        }
                    }
                });
            }
        });
    }

    /// The live rows that hold each of `columns` still held by one, once
    /// each, found side by side; a column given no list is listed first.
    fn pinned(&mut self, columns: &[u32]) -> Vec<u32> {
        for &column in columns {
            if self.weight(column) == 1 && self.lists.untracked(column) {
                self.track(column);
            }
        }
        let threads = crate::parallel::budget();
        let length = columns.len().div_ceil(threads * CHUNKS_PER_THREAD).max(1);
        let chunks: Vec<&[u32]> = columns.chunks(length).collect();
        let filter = &*self;
        crate::parallel::map_ordered(&chunks, threads, |_, chunk| {
            chunk
                .iter()
                .filter(|&&column| filter.weight(column) == 1)
                .map(|&column| filter.holder(column) as u32)
                .collect::<Vec<u32>>()
        })
        .concat()
    }

    /// Remove every row pinned by a singleton column, to a fixed point,
    /// starting from the columns in `singletons`: in rounds, each the rows
    /// the singletons pin, retired side by side, and the columns they left
    /// singletons the next. Which rows go does not depend on the order they
    /// go in. Returns the columns left at two along the way.
    fn prune(&mut self, mut singletons: Vec<u32>) -> Vec<u32> {
        let mut twos = Vec::new();
        while !singletons.is_empty() {
            let rows = self.pinned(&singletons);
            let (next, two) = self.retire_all(&rows);
            twos.extend(two);
            singletons = next;
        }
        twos
    }

    /// Every column of `columns` that two rows hold binds them. The pairs
    /// are found side by side and bound in the order of `columns`.
    fn bind(&mut self, cliques: &mut Cliques, columns: &[u32]) {
        for &column in columns {
            if self.weight(column) == 2 && self.lists.untracked(column) {
                self.track(column);
            }
        }
        let threads = crate::parallel::budget();
        let length = columns.len().div_ceil(threads * CHUNKS_PER_THREAD).max(1);
        let chunks: Vec<&[u32]> = columns.chunks(length).collect();
        let filter = &*self;
        let pairs = crate::parallel::map_ordered(&chunks, threads, |_, chunk| {
            chunk
                .iter()
                .filter(|&&column| filter.weight(column) == 2)
                .map(|&column| {
                    let held = filter.holders(column);
                    (held[0], held[1])
                })
                .collect::<Vec<(u32, u32)>>()
        });
        for (a, b) in pairs.into_iter().flatten() {
            cliques.bind(a, b);
        }
    }

    fn all_singletons(&self) -> Vec<u32> {
        (0..self.columns as u32)
            .filter(|&c| self.weight(c) == 1)
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
        let every: Vec<u32> = (0..self.columns as u32).collect();
        self.bind(&mut cliques, &every);
        loop {
            let excess = self.excess();
            if excess <= target {
                break;
            }
            let share = (excess - target).div_ceil(PURGE_SHARE_DIVISOR);
            let mut rows = Vec::new();
            let mut taken = 0;
            while taken < share {
                let Some(root) = cliques.heaviest(&self.live) else {
                    break;
                };
                taken += 1;
                let mut row = root;
                while row != LAST {
                    rows.push(row);
                    row = cliques.next[row as usize];
                }
            }
            if taken == 0 {
                break;
            }
            let (singletons, mut twos) = self.retire_all(&rows);
            twos.extend(self.prune(singletons));
            self.bind(&mut cliques, &twos);
        }
    }

    /// Eliminate columns in passes (Bouillaguet & Zimmermann, §4). A pass
    /// lists the rows of every column held by one to `weight_cap` rows,
    /// side by side; plans each one's tree, side by side; takes, cheapest
    /// first, the columns whose fill lowers the cost proxy and whose rows no
    /// column already taken holds; and merges them side by side, no two
    /// touching one row. The passes end when no column lowers the proxy.
    ///
    /// A pass plans and merges against the rows as it began, and takes its
    /// columns in the order of their fills and then their numbers, so the
    /// matrix is the same at any number of threads; it is not the matrix of
    /// merging one column at a time, every merge planned against the last.
    fn merge(&mut self, weight_cap: usize) {
        let cap = weight_cap.max(1) as u32;
        // The lists were the pruning's and the purge's; a pass lists what
        // it needs afresh.
        self.lists = Lists::default();
        let threads = crate::parallel::budget();
        let mut place = vec![NOT_LIGHT; self.columns];
        let mut busy = vec![false; self.work.len()];
        loop {
            let light = self.light(cap, threads);
            if light.is_empty() {
                break;
            }
            let (start, held) = self.transpose(&light, &mut place, threads);
            let members = |k: usize| -> Vec<usize> {
                held[start[k]..start[k + 1]]
                    .iter()
                    .map(|&r| r as usize)
                    .collect()
            };
            let indices: Vec<usize> = (0..light.len()).collect();
            let fills: Vec<i64> =
                crate::parallel::map_ordered(&indices, threads, |_, &k| match members(k)[..] {
                    [only] => -(self.work[only].len() as i64),
                    ref rows => plan(rows, &self.work).fill,
                });
            let gains = |fill: i64| fill.saturating_mul(self.live_rows as i64 - 1) < self.nonzeros;
            let mut order: Vec<usize> = indices.into_iter().filter(|&k| gains(fills[k])).collect();
            order.sort_unstable_by_key(|&k| (fills[k], light[k]));
            let mut chosen: Vec<Vec<usize>> = Vec::new();
            for k in order {
                let rows = members(k);
                if rows.iter().any(|&r| busy[r]) {
                    continue;
                }
                for &r in &rows {
                    busy[r] = true;
                }
                chosen.push(rows);
            }
            for &column in &light {
                place[column as usize] = NOT_LIGHT;
            }
            if chosen.is_empty() {
                break;
            }
            for rows in &chosen {
                for &r in rows {
                    busy[r] = false;
                }
            }
            let merged =
                crate::parallel::map_ordered(&chosen, threads, |_, rows| self.eliminate(rows));
            for one in merged {
                for (row, after, sum) in one.rows {
                    self.work[row] = after;
                    self.compositions[row] = sum;
                }
                self.work[one.root] = Vec::new();
                self.compositions[one.root] = Vec::new();
                self.live_rows -= 1;
                self.nonzeros += one.entries;
                self.live_columns = self.live_columns + one.revived - one.emptied;
            }
        }
    }

    /// The columns held by one to `cap` rows, ascending, found side by side.
    fn light(&self, cap: u32, threads: usize) -> Vec<u32> {
        let length = self.columns.div_ceil(threads * CHUNKS_PER_THREAD).max(1);
        let chunks: Vec<usize> = (0..self.columns.div_ceil(length)).collect();
        crate::parallel::map_ordered(&chunks, threads, |_, &chunk| {
            (chunk * length..((chunk + 1) * length).min(self.columns))
                .map(|c| c as u32)
                .filter(|&c| (1..=cap).contains(&self.weight(c)))
                .collect::<Vec<u32>>()
        })
        .concat()
    }

    /// The live rows of each of `light`, ascending, as one array cut where
    /// each column's rows start: counted from the weights, laid out, and
    /// written side by side through each column's cursor, then sorted side
    /// by side. `place` is each light column's index for the pass, and is
    /// left so for the caller to clear.
    fn transpose(
        &self,
        light: &[u32],
        place: &mut [u32],
        threads: usize,
    ) -> (Vec<usize>, Vec<u32>) {
        let mut start = Vec::with_capacity(light.len() + 1);
        let mut end = 0;
        for (k, &column) in light.iter().enumerate() {
            place[column as usize] = k as u32;
            start.push(end);
            end += self.weight(column) as usize;
        }
        start.push(end);
        let slots: Vec<AtomicU32> = (0..end).map(|_| AtomicU32::new(0)).collect();
        let cursors: Vec<AtomicU32> = (0..light.len()).map(|_| AtomicU32::new(0)).collect();
        let length = self.work.len().div_ceil(threads * CHUNKS_PER_THREAD).max(1);
        let chunks: Vec<usize> = (0..self.work.len().div_ceil(length)).collect();
        let place = &*place;
        crate::parallel::map_ordered(&chunks, threads, |_, &chunk| {
            for row in chunk * length..((chunk + 1) * length).min(self.work.len()) {
                if !self.is_live(row) {
                    continue;
                }
                for &column in &self.work[row] {
                    let k = place[column as usize];
                    if k != NOT_LIGHT {
                        let at = start[k as usize]
                            + cursors[k as usize].fetch_add(1, Ordering::Relaxed) as usize;
                        slots[at].store(row as u32, Ordering::Relaxed);
                    }
                }
            }
        });
        let mut held: Vec<u32> = slots.into_iter().map(AtomicU32::into_inner).collect();
        let mut pieces: Vec<&mut [u32]> = Vec::with_capacity(light.len());
        let mut rest = &mut held[..];
        for k in 0..light.len() {
            let (piece, after) = rest.split_at_mut(start[k + 1] - start[k]);
            pieces.push(piece);
            rest = after;
        }
        let length = pieces.len().div_ceil(threads * CHUNKS_PER_THREAD).max(1);
        std::thread::scope(|scope| {
            for run in pieces.chunks_mut(length) {
                scope.spawn(move || {
                    for piece in run {
                        piece.sort_unstable();
                    }
                });
            }
        });
        (start, held)
    }

    /// The elimination of the column `rows` hold, against the rows as they
    /// stand, written to none of them: each child's new row and its
    /// composition, the root retired, and the weights counted where they
    /// stand, which no other column of the pass shares a row with.
    fn eliminate(&self, rows: &[usize]) -> Merged {
        let tree = match rows[..] {
            [only] => Plan {
                root: only,
                edges: Vec::new(),
                fill: -(self.work[only].len() as i64),
            },
            _ => plan(rows, &self.work),
        };
        let mut merged = Merged {
            rows: Vec::with_capacity(tree.edges.len()),
            root: tree.root,
            entries: 0,
            emptied: 0,
            revived: 0,
        };
        for &(child, parent) in &tree.edges {
            let (mine, theirs) = (&self.work[child], &self.work[parent]);
            let mut after = Vec::with_capacity(mine.len() + theirs.len());
            let (mut i, mut j) = (0, 0);
            while i < mine.len() || j < theirs.len() {
                match (mine.get(i), theirs.get(j)) {
                    (Some(&a), Some(&b)) if a == b => {
                        i += 1;
                        j += 1;
                        if self.occupants[a as usize].fetch_sub(1, Ordering::Relaxed) == 1 {
                            merged.emptied += 1;
                        }
                    }
                    (Some(&a), Some(&b)) if a < b => {
                        i += 1;
                        after.push(a);
                    }
                    (Some(&a), None) => {
                        i += 1;
                        after.push(a);
                    }
                    (_, Some(&b)) => {
                        j += 1;
                        after.push(b);
                        if self.occupants[b as usize].fetch_add(1, Ordering::Relaxed) == 0 {
                            merged.revived += 1;
                        }
                    }
                    (None, None) => unreachable!("the walk ends when both rows do"),
                }
            }
            merged.entries += after.len() as i64 - mine.len() as i64;
            let (alone, beside) = ([child], [parent]);
            let sum = symmetric_difference(
                composition(&self.compositions, child, &alone),
                composition(&self.compositions, parent, &beside),
            );
            merged.rows.push((child, after, sum));
        }
        self.live[tree.root].store(false, Ordering::Relaxed);
        merged.entries -= self.work[tree.root].len() as i64;
        for &column in &self.work[tree.root] {
            if self.occupants[column as usize].fetch_sub(1, Ordering::Relaxed) == 1 {
                merged.emptied += 1;
            }
        }
        merged
    }

    fn finish(mut self) -> FilteredMatrix {
        let mut kept_rows = Vec::with_capacity(self.live_rows);
        let mut kept_compositions = Vec::with_capacity(self.live_rows);
        for index in 0..self.work.len() {
            if self.is_live(index) {
                kept_rows.push(core::mem::take(&mut self.work[index]));
                let set = core::mem::take(&mut self.compositions[index]);
                kept_compositions.push(if set.is_empty() { vec![index] } else { set });
            }
        }
        debug_assert_eq!(
            kept_rows.iter().map(Vec::len).sum::<usize>() as i64,
            self.nonzeros,
            "the running nonzero count drifted"
        );
        debug_assert_eq!(
            self.occupants
                .iter()
                .filter(|count| count.load(Ordering::Relaxed) > 0)
                .count(),
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
            originals: self.work.len(),
        }
    }
}

/// Filters a matrix: singleton pruning, clique removal down to `excess`
/// rows beyond the live columns, then fill-ordered column merges in passes.
///
/// Every step runs on the calling thread's budget of threads, and the
/// matrix is the same at any budget.
///
/// Columns are eliminated in order of increasing fill, each pass taking
/// those whose rows no cheaper column of the pass holds, while an
/// elimination lowers `rows · nonzeros`, the Block Lanczos cost: one
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
pub fn filter_merge(matrix: SparseMatrix, weight_cap: usize, excess: usize) -> FilteredMatrix {
    let mut filter = Filter::new(matrix, weight_cap.max(1));
    let singletons = filter.all_singletons();
    filter.prune(singletons);
    filter.purge(excess);
    filter.release();
    filter.merge(weight_cap);
    filter.finish()
}

#[cfg(test)]
mod tests {
    use super::super::dense_null_space;

    /// A column too dense for a list at the start that comes down to a
    /// singleton is given one then, and pruned like any other.
    #[test]
    fn a_dense_column_brought_down_is_listed_and_pruned() {
        // Column 0 is held by 200 of 10 200 rows, past both a sixty-fourth
        // of them and sixty-four times a cap of one. Rows 1 to 199 hold a
        // singleton each beside it; row 0 holds it and a column it shares
        // with one filler row. The fillers pair up over columns of their
        // own, so nothing else is a singleton at the start.
        const DENSE_ROWS: u32 = 200;
        const FILLERS: u32 = 10_000;
        let pair_base = DENSE_ROWS + 1;
        let mut rows: Vec<Vec<u32>> = vec![vec![0, pair_base]];
        rows.extend((1..DENSE_ROWS).map(|i| vec![0, i]));
        rows.push(vec![pair_base, pair_base + 1]);
        rows.extend((1..FILLERS).map(|k| vec![pair_base + k / 2 + 1, pair_base + k / 2 + 2]));
        let columns = (pair_base + FILLERS + 2) as usize;
        let matrix = SparseMatrix::new(columns, rows);
        assert!(
            Filter::new(matrix.clone(), 1).lists.untracked(0),
            "column 0 starts without a list"
        );
        let filtered = filter_merge(matrix, 1, usize::MAX);
        assert!(
            filtered
                .rows()
                .iter()
                .all(|row| row.binary_search(&0).is_err()),
            "the dense column came down to one row and that row went"
        );
    }

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
                let filtered = filter_merge(matrix.clone(), cap, usize::MAX);
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
            let filtered = filter_merge(matrix.clone(), 32, usize::MAX);
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
            let pairs = filter_merge(matrix.clone(), 2, usize::MAX);
            let wide = filter_merge(matrix.clone(), 32, usize::MAX);
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
        let filtered = filter_merge(dense.clone(), 32, usize::MAX);
        if let Some(dependency) = dense_null_space(&filtered.packed_rows(), columns).first() {
            assert!(xor_of(&dense, &filtered.expand(dependency)).is_empty());
        }
        let sparse = random_matrix(SPARSE_SEED, 4_600, columns, 1_600);
        let filtered = filter_merge(sparse.clone(), 32, usize::MAX);
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
        let filtered = filter_merge(sparse.clone(), 32, usize::MAX);
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
        let filtered = filter_merge(matrix.clone(), 2, usize::MAX);
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
        let filtered = filter_merge(matrix.clone(), 4, usize::MAX);
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
        let full = filter_merge(matrix.clone(), 1, usize::MAX);
        assert_eq!(full.rows().len(), 10);
        let excess = full.rows().len() - full.live_columns();
        assert_eq!(excess, 2);
        let purged = filter_merge(matrix.clone(), 1, excess - 1);
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
                if filter.weight(column as u32) == 2 {
                    let members = filter.holders(column as u32);
                    let (a, b) = (members[0] as usize, members[1] as usize);
                    let (a, b) = (find(&mut parent, a), find(&mut parent, b));
                    parent[a.max(b)] = a.min(b);
                }
            }
            let mut weight = vec![0usize; count];
            let mut rows_of = vec![Vec::new(); count];
            for row in (0..count).filter(|&row| filter.is_live(row)) {
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
            let taken: Vec<u32> = components[..share]
                .iter()
                .flat_map(|&root| rows_of[root].iter().map(|&row| row as u32))
                .collect();
            let (pending, _) = filter.retire_all(&taken);
            filter.prune(pending);
        }
    }

    /// Pruning in rounds, side by side, leaves the rows the naive fixed
    /// point leaves: every column scanned again and again, a row holding a
    /// singleton removed, until none does.
    #[test]
    fn pruning_in_rounds_is_the_fixed_point() {
        const SEEDS: std::ops::RangeInclusive<u64> = 600..=607;
        for density in [200, 330, 500] {
            for seed in SEEDS {
                let matrix = random_matrix(seed, 1_300, 1_000, density);
                let mut live = vec![true; matrix.rows().len()];
                loop {
                    let mut holders = vec![Vec::new(); matrix.columns()];
                    for (row, columns) in matrix.rows().iter().enumerate() {
                        if live[row] {
                            for &column in columns {
                                holders[column as usize].push(row);
                            }
                        }
                    }
                    let pinned: Vec<usize> = holders
                        .iter()
                        .filter(|held| held.len() == 1)
                        .map(|held| held[0])
                        .collect();
                    if pinned.is_empty() {
                        break;
                    }
                    for row in pinned {
                        live[row] = false;
                    }
                }
                for threads in [1, 4] {
                    let mut filter = Filter::new(matrix.clone(), 32);
                    let singletons = filter.all_singletons();
                    crate::parallel::with_budget(threads, || filter.prune(singletons));
                    let kept: Vec<bool> = (0..live.len()).map(|row| filter.is_live(row)).collect();
                    assert_eq!(
                        kept, live,
                        "density {density} seed {seed}, {threads} threads"
                    );
                    assert_eq!(
                        filter.live_rows,
                        live.iter().filter(|&&row| row).count(),
                        "the live count"
                    );
                }
            }
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
                    let mut filter = Filter::new(matrix.clone(), 32);
                    let singletons = filter.all_singletons();
                    filter.prune(singletons);
                    filter
                };
                // Targets: all but one dependency purged, and two between.
                for target in [1usize, 16, 64] {
                    let (mut kept, mut afresh) = (pruned(), pruned());
                    kept.purge(target);
                    let rounds = purge_afresh(&mut afresh, target);
                    assert!(rounds > 1, "density {density} seed {seed}: one round");
                    let case = format!("density {density} seed {seed} target {target}");
                    kept.release();
                    afresh.release();
                    let live = |filter: &Filter| -> Vec<bool> {
                        (0..filter.work.len())
                            .map(|row| filter.is_live(row))
                            .collect()
                    };
                    let weights = |filter: &Filter| -> Vec<u32> {
                        (0..filter.columns as u32)
                            .map(|c| filter.weight(c))
                            .collect()
                    };
                    assert_eq!(live(&kept), live(&afresh), "{case}");
                    assert_eq!(kept.work, afresh.work, "{case}");
                    assert_eq!(weights(&kept), weights(&afresh), "{case}");
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
            // column is of a weight the merge takes, so a pass plans five
            // thousand side by side and merges hundreds of them so.
            let matrix = random_matrix(seed, 6_000, 5_000, 625);
            let alone = crate::parallel::with_budget(1, || filter_merge(matrix.clone(), 32, 64));
            assert!(alone.rows().len() < matrix.rows().len(), "seed {seed}");
            // Budgets: the fewest that share, and more than a pass's
            // columns are cut for.
            for threads in [2usize, 32] {
                let shared =
                    crate::parallel::with_budget(threads, || filter_merge(matrix.clone(), 32, 64));
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
                let filtered = filter_merge(matrix.clone(), 32, excess);
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
