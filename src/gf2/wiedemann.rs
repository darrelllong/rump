//! Block Wiedemann over GF(2): the null space from sequences that can be
//! taken apart.
//!
//! Block Lanczos is a chain, each iteration made of the last. Coppersmith's
//! block Wiedemann (*Solving homogeneous linear equations over GF(2) via
//! block Wiedemann algorithm*, Math. Comp. 62 (1994), 333–350) has three
//! stages instead, and two of them are products of the matrix with blocks
//! that do not depend on one another:
//!
//! 1. **The sequence.** For random blocks `x` and `z` of `n = 64·J` lanes
//!    and `y = B·z`, the small matrices `aₖ = xᵀ·Bᵏ·y`, `k < L`. Lane
//!    slice `j` of `y` makes its own `64·J` by `64` slab of every `aₖ` with
//!    no word from any other slice, so `J` hosts take a slice each
//!    ([`wiedemann_sequence`]).
//! 2. **The generator.** Vectors of polynomials `f(λ) = Σ fᵢλⁱ` of degree
//!    at most `δ` with `Σᵢ aₖ₋ᵢ·fᵢ = 0` for `δ ≤ k < L`, by Coppersmith's
//!    iteration: his generalisation of Berlekamp–Massey, made by halves as
//!    Thomé gives it (*Subquadratic computation of vector generating
//!    polynomials and improvement of the block Wiedemann algorithm*, J.
//!    Symbolic Comput. 33 (2002), 757–775), in `L^{log₂3}` products by
//!    Karatsuba's split, on one host ([`wiedemann_generators`]).
//! 3. **The solution.** `w = Σᵢ B^{δ−i}·z·fᵢ`, by Horner's rule, which is
//!    the sum over the slices of `z` of what each slice makes with its own
//!    words of the coefficients ([`wiedemann_solution`], a host each); then
//!    `xᵀ·Bⁱ·(B·w) = 0` for every `i` the sequence reaches, which leaves
//!    `B·w = 0` unless the projection by `x` hides it, and a `w` that is
//!    not yet a null vector becomes one, or zero, under a few more `B`
//!    ([`wiedemann_finish`]).
//!
//! `B` here is square, of dimension `N`, the greater of the relations and
//! the live columns: the relations' matrix `M` taken as the map from a
//! block over the relations to one over the columns, padded with zero rows
//! or zero columns to square. A null vector of `B` restricted to the
//! relations' coordinates is a dependency; the padding coordinates are
//! not returned.
//!
//! The sequence is `N/m + N/n` products and the solution `N/n`, against
//! `2N/64` for Lanczos, so `J` hosts each take about `2/J` of what one
//! host of Lanczos takes; the generator stage costs the same whatever `J`
//! is, and holds `E` and `π`, `2 KiB · J²` a term, beside its products'
//! working space, which grows with the threads they run on. Each stage reads
//! what the last wrote ([`Sequence::to_bytes`], [`Generators::to_bytes`])
//! and is restartable from it. Nothing here is trusted either: every vector
//! returned is checked against the caller's rows.

use std::sync::Arc;

use super::generator::{halves_from, product, steps, transpose, Degrees, Polynomials};
use super::{any, lane, Block, SmallProduct, Sparse, SparseMatrix, WIDTH};
use crate::random::RandomSource;

/// The coordinates each vector of `x` sums: a row of `aₖ` is then the XOR
/// of that many words of `Bᵏ·y`, and a product by `x` costs nothing beside
/// a product by `B`. One would do in general; a few keep a coordinate that
/// happens to carry little of the sequence from costing a row its rank.
/// Set by hand.
const X_WEIGHT: usize = 4;

/// Steps without a term of its own after which a column is taken for a
/// generator. A stopping rule and not a bound: the terms are projections
/// of one Krylov sequence by one `x`, not independent draws, so no
/// probability is claimed for a column that is not a generator passing
/// this many. What the rule lets through is caught by the check of every
/// vector returned. Set by hand, at a block's width.
const SPARE_TERMS: usize = WIDTH;

/// The most terms taken, in units of `N/64`, before the solve is given up
/// with `None`: two are needed where `x` and `z` see all of `B`, and the
/// third is for where they do not. Set by hand.
const TERMS_MOST: usize = 3;

/// Products by `B` tried on a solution that is not yet a null vector. `w`
/// lies in the null space of a power of `B`, and the power is small where
/// `B`'s nilpotent part is; a lane not finished by then is dropped. Set by
/// hand.
const POWERS: usize = 16;

/// The magic that begins a stage's bytes, and the stage's own letter after
/// it.
const MAGIC: &[u8; 3] = b"GFW";

// `WIDTH` is the machine word, `u64::BITS`: a block is a word a coordinate
// and a bit a vector, so one XOR advances sixty-four vectors. Nothing here
// is tuned to it; a wider block is `of` words.

/// Dependencies of `matrix` by block Wiedemann on one host, as
/// [`block_lanczos_dependencies_sparse`](super::block_lanczos_dependencies_sparse)
/// returns them: sets of row indices, each checked to sum to zero over the
/// caller's rows, or `None` when none was found. The matrix may have more
/// columns than rows, or fewer. The stages below, with one slice.
///
/// `None` is any of: no rows or no columns; the sequence reaching three
/// times `N/64` terms (`TERMS_MOST`) without its generators settling; no
/// lane of the solution a null vector within sixteen products (`POWERS`);
/// or every candidate failing the check. Another random source may do
/// better; the exact solver always does.
///
/// `threads` is a ceiling on the workers a product by the matrix runs on,
/// as for Lanczos.
#[must_use]
pub fn block_wiedemann_dependencies_sparse<R: RandomSource + ?Sized>(
    matrix: &SparseMatrix,
    rng: &mut R,
    threads: usize,
) -> Option<Vec<Vec<usize>>> {
    if matrix.rows().is_empty() || matrix.columns() == 0 {
        return None;
    }
    let mut bytes = [0u8; 8];
    rng.fill_bytes(&mut bytes);
    let seed = u64::from_le_bytes(bytes);
    // The terms the generators should settle within, and more if they do
    // not, up to the most.
    let most = wiedemann_terms_most(matrix, 1);
    let mut terms = wiedemann_terms(matrix, 1).min(most);
    let mut sequence: Option<Sequence> = None;
    let mut iterated: Option<Iterated> = None;
    let trace = std::env::var_os("WIEDEMANN_TRACE").is_some();
    loop {
        let began = std::time::Instant::now();
        let next = wiedemann_sequence(matrix, seed, 0, 1, terms, sequence.as_ref(), threads)?;
        if trace {
            eprintln!("trace: sequence to {terms} terms in {:?}", began.elapsed());
        }
        let began = std::time::Instant::now();
        let generated =
            wiedemann_generators(std::slice::from_ref(&next), iterated.take(), threads)?;
        if trace {
            eprintln!(
                "trace: generators to {terms} terms in {:?}",
                began.elapsed()
            );
        }
        let generators = match generated {
            Generated::Settled(generators) => Some(generators),
            Generated::Short {
                terms: wanted,
                iterated: stopped,
            } => {
                if terms >= most {
                    return None;
                }
                terms = wanted.min(most);
                iterated = Some(stopped);
                None
            }
        };
        if let Some(generators) = generators {
            let began = std::time::Instant::now();
            let partial = wiedemann_solution(matrix, &generators, 0, threads)?;
            if trace {
                eprintln!("trace: solution in {:?}", began.elapsed());
            }
            let began = std::time::Instant::now();
            let found = wiedemann_finish(matrix, &[partial], threads);
            if trace {
                eprintln!("trace: finish in {:?}", began.elapsed());
            }
            return found;
        }
        sequence = Some(next);
    }
}

/// The terms a slice of the sequence should have first for `of` slices over
/// `matrix`: the fewest with which the generators can settle. From `t = 1`
/// each step multiplies `m` of the `m + n` columns by `X`, so their gap
/// `t − δ` grows by `n/(m + n)`, a half, a step; a column is a generator
/// once its gap reaches `λ = ⌈N/m⌉` (Thomé's habilitation, §12.1.3), at
/// `t = 2λ + 1`, and settled `SPARE_TERMS` steps after. A column of `F`
/// that goes to zero takes its share of the steps from the rest; how many
/// do is known only from the iteration, and the generator stage then names
/// the terms it wants ([`Generated::Short`]).
#[must_use]
pub fn wiedemann_terms(matrix: &SparseMatrix, of: usize) -> usize {
    2 * dimension_of(matrix).div_ceil(WIDTH * of.max(1)) + SPARE_TERMS + 1
}

/// The most terms a slice of `of` is taken to before a solve is given up:
/// `TERMS_MOST` times `N/(64·of)`, and the settling.
#[must_use]
pub fn wiedemann_terms_most(matrix: &SparseMatrix, of: usize) -> usize {
    TERMS_MOST * dimension_of(matrix).div_ceil(WIDTH * of.max(1)) + 2 * SPARE_TERMS
}

/// `N`: the greater of the rows and the live columns, those some row sets.
/// A kept matrix's width can be ten times its live columns, and a length
/// taken from it would be ten times what the generators need.
fn dimension_of(matrix: &SparseMatrix) -> usize {
    let mut touched = vec![0u64; matrix.columns().div_ceil(WIDTH)];
    for row in matrix.rows() {
        for &column in row {
            touched[column as usize / WIDTH] |= 1u64 << (column as usize % WIDTH);
        }
    }
    let live: u32 = touched.iter().map(|word| word.count_ones()).sum();
    matrix.rows().len().max(live as usize)
}

/// One slice of the sequence `aₖ = xᵀ·Bᵏ·y`: for lane slice `j` of `J`,
/// the `64·J` by `64` slab of each term that slice `j` of `y` makes, and
/// the block `Bᵏ·y_j` after the last term, from which more terms follow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sequence {
    seed: u64,
    slice: usize,
    of: usize,
    dimension: usize,
    columns: usize,
    /// [`matrix_fingerprint`] of the matrix the terms are of: a slice kept from
    /// one matrix is never continued against another.
    matrix: u64,
    /// `terms · 64 · of` words: term by term, the slab's rows.
    terms: Vec<u64>,
    /// `dimension` words.
    last: Vec<u64>,
}

impl Sequence {
    /// How many terms the slice holds.
    #[must_use]
    pub fn terms(&self) -> usize {
        self.terms.len() / (WIDTH * self.of)
    }

    /// Which slice this is, of how many.
    #[must_use]
    pub fn slice(&self) -> (usize, usize) {
        (self.slice, self.of)
    }

    /// The slice as bytes: `GFWS`, then the fields and the words, each
    /// eight bytes little-endian.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 * (9 + self.terms.len() + self.last.len()));
        bytes.extend_from_slice(MAGIC);
        bytes.push(b'S');
        for word in [
            self.seed,
            self.slice as u64,
            self.of as u64,
            self.dimension as u64,
            self.columns as u64,
            self.matrix,
            self.terms.len() as u64,
            self.last.len() as u64,
        ] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for word in self.terms.iter().chain(&self.last) {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// A slice from its bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WiedemannBytesError> {
        let mut words = Words::of(bytes, b'S')?;
        let seed = words.next()?;
        let slice = words.index()?;
        let of = words.index()?;
        let dimension = words.index()?;
        let columns = words.index()?;
        let matrix = words.next()?;
        let terms = words.index()?;
        let last = words.index()?;
        if of == 0 || slice >= of || !terms.is_multiple_of(WIDTH * of) || last != dimension {
            return Err(WiedemannBytesError::Shape);
        }
        let terms = words.take(terms)?;
        let last = words.take(last)?;
        Ok(Self {
            seed,
            slice,
            of,
            dimension,
            columns,
            matrix,
            terms,
            last,
        })
    }
}

/// The generators of a sequence: up to 64 lanes, each the coefficients of
/// one vector polynomial from its leading one down, `J` words a
/// coefficient, with the seed and the slicing the sequence had, so that a
/// solution's slice knows its `z`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generators {
    seed: u64,
    of: usize,
    dimension: usize,
    columns: usize,
    /// Each lane's coefficients, `(degree + 1) · of` words, the leading
    /// coefficient first.
    lanes: Vec<Vec<u64>>,
}

impl Generators {
    /// How many lanes, which is how many solutions there are to build.
    #[must_use]
    pub fn lanes(&self) -> usize {
        self.lanes.len()
    }

    /// The greatest degree among the lanes.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.lanes
            .iter()
            .map(|lane| lane.len() / self.of)
            .max()
            .unwrap_or(1)
            - 1
    }

    /// The generators as bytes: `GFWG`, the fields, each lane's length and
    /// its words.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.push(b'G');
        for word in [
            self.seed,
            self.of as u64,
            self.dimension as u64,
            self.columns as u64,
            self.lanes.len() as u64,
        ] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for lane in &self.lanes {
            bytes.extend_from_slice(&(lane.len() as u64).to_le_bytes());
            for word in lane {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        bytes
    }

    /// The generators from their bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WiedemannBytesError> {
        let mut words = Words::of(bytes, b'G')?;
        let seed = words.next()?;
        let of = words.index()?;
        let dimension = words.index()?;
        let columns = words.index()?;
        let count = words.index()?;
        if of == 0 || count > WIDTH {
            return Err(WiedemannBytesError::Shape);
        }
        let mut lanes = Vec::with_capacity(count);
        for _ in 0..count {
            let len = words.index()?;
            if len == 0 || !len.is_multiple_of(of) {
                return Err(WiedemannBytesError::Shape);
            }
            lanes.push(words.take(len)?);
        }
        Ok(Self {
            seed,
            of,
            dimension,
            columns,
            lanes,
        })
    }
}

/// What a stage's bytes could not be read as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WiedemannBytesError {
    /// Not the magic of this stage.
    Header,
    /// Fewer bytes than the fields claim.
    Short,
    /// Fields that do not describe a slice or a set of generators.
    Shape,
}

impl std::fmt::Display for WiedemannBytesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Header => write!(f, "not a Wiedemann stage's bytes"),
            Self::Short => write!(f, "the bytes end before their fields say"),
            Self::Shape => write!(f, "the fields do not describe a stage"),
        }
    }
}

impl std::error::Error for WiedemannBytesError {}

/// Words read off a stage's bytes.
struct Words<'a> {
    bytes: &'a [u8],
}

impl<'a> Words<'a> {
    fn of(bytes: &'a [u8], stage: u8) -> Result<Self, WiedemannBytesError> {
        let rest = bytes
            .strip_prefix(MAGIC)
            .and_then(|rest| rest.strip_prefix(&[stage]))
            .ok_or(WiedemannBytesError::Header)?;
        Ok(Self { bytes: rest })
    }

    fn next(&mut self) -> Result<u64, WiedemannBytesError> {
        let (word, rest) = self
            .bytes
            .split_first_chunk::<8>()
            .ok_or(WiedemannBytesError::Short)?;
        self.bytes = rest;
        Ok(u64::from_le_bytes(*word))
    }

    fn index(&mut self) -> Result<usize, WiedemannBytesError> {
        usize::try_from(self.next()?).map_err(|_| WiedemannBytesError::Shape)
    }

    fn take(&mut self, count: usize) -> Result<Vec<u64>, WiedemannBytesError> {
        if self.bytes.len() / 8 < count {
            return Err(WiedemannBytesError::Short);
        }
        (0..count).map(|_| self.next()).collect()
    }
}

/// SplitMix64 (Steele, Lea & Flood, *Fast splittable pseudorandom number
/// generators*, OOPSLA 2014), so that every host derives the same `x` and
/// its own `z` from one seed.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// The rows of `xᵀ`, `64·of` of them: each the coordinates among the
/// columns it sums. From the seed alone, so every slice has the same `x`.
fn supports(seed: u64, of: usize, columns: usize) -> Vec<[usize; X_WEIGHT]> {
    let mut draw = SplitMix(seed);
    (0..WIDTH * of)
        .map(|_| std::array::from_fn(|_| (draw.next() % columns as u64) as usize))
        .collect()
}

/// Lane slice `slice` of `z`: `dimension` words from the seed and the
/// slice's number.
fn z_slice(seed: u64, slice: usize, dimension: usize) -> Arc<Block> {
    let mut draw = SplitMix(seed ^ (slice as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    Block::of((0..dimension).map(|_| draw.next()))
}

/// `B·v` as the next `v`: `M·v` over the columns, which are the first of the
/// block's coordinates, into `image`, its words past the columns zero; then
/// the two change places, the old `v` the next image. Each block is `N`
/// words, the greater of the relations and the columns; `M` reads the
/// first of them up to the relations. The words past the columns are the
/// excess of relations, a few hundred, where copying the image into `v`
/// was every word.
fn advance(matrix: &Sparse, v: &mut Arc<Block>, image: &mut Arc<Block>) {
    matrix.forward(v, image);
    for index in matrix.columns()..image.len() {
        image.set(index, 0);
    }
    core::mem::swap(v, image);
}

/// The matrix's identity for a kept slice: FNV-1a over its shape and
/// every entry, so that a slice of one matrix is not continued against
/// another of the same shape, which would give terms of neither.
fn matrix_fingerprint(matrix: &SparseMatrix) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0100_0000_01b3;
    let mut hash = OFFSET;
    let mut mix = |word: u64| {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(PRIME);
        }
    };
    mix(matrix.rows().len() as u64);
    mix(matrix.columns() as u64);
    for row in matrix.rows() {
        mix(row.len() as u64);
        for &column in row {
            mix(u64::from(column));
        }
    }
    hash
}

/// Slice `slice` of `of` of the sequence, `terms` terms long: from the
/// start, or on from `from`, a shorter slice of the same matrix, seed and
/// place, whose terms are kept; a slice of another matrix is not continued
/// and the terms start afresh. `None` for a matrix with no rows or no
/// columns, or a slice out of range.
#[must_use]
pub fn wiedemann_sequence(
    matrix: &SparseMatrix,
    seed: u64,
    slice: usize,
    of: usize,
    terms: usize,
    from: Option<&Sequence>,
    threads: usize,
) -> Option<Sequence> {
    if matrix.rows().is_empty() || matrix.columns() == 0 || of == 0 || slice >= of {
        return None;
    }
    let sparse = Sparse::forward_only(matrix.rows().to_vec(), matrix.columns(), threads);
    let (count, columns) = (sparse.relations(), sparse.columns());
    let dimension = count.max(columns);
    let fingerprint = matrix_fingerprint(matrix);
    let supports = supports(seed, of, columns);
    let mut image = Block::zeroed(dimension);
    let (mut v, mut slabs, done) = match from {
        Some(from)
            if from.matrix == fingerprint
                && from.seed == seed
                && from.slice == slice
                && from.of == of
                && from.dimension == dimension =>
        {
            (
                Block::of(from.last.iter().copied()),
                from.terms.clone(),
                from.terms(),
            )
        }
        _ => {
            let mut v = z_slice(seed, slice, dimension);
            advance(&sparse, &mut v, &mut image);
            (v, Vec::new(), 0)
        }
    };
    slabs.reserve(terms.saturating_sub(done) * WIDTH * of);
    for term in done..terms {
        if term > 0 {
            advance(&sparse, &mut v, &mut image);
        }
        for support in &supports {
            slabs.push(support.iter().fold(0, |sum, &at| sum ^ v.word(at)));
        }
    }
    Some(Sequence {
        seed,
        slice,
        of,
        dimension,
        columns,
        matrix: fingerprint,
        terms: slabs,
        last: v.words().collect(),
    })
}

/// Where the generator's iteration stopped: `A·F = G + Xᵗ·E` at `t`, with
/// `F` of `2·64·of` columns of polynomials over `64·of` bits, each column's
/// `δ`, and the steps each has gone without being multiplied by `X`. More
/// terms resume it from here, for `E`'s coefficients are those of `A·F`
/// from `t` on: what the iteration did over the terms it had is not done
/// again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Iterated {
    seed: u64,
    of: usize,
    dimension: usize,
    columns: usize,
    /// The sequence's first term, folded: what ties the iteration to the
    /// sequence and not only to its seed and shape.
    fingerprint: u64,
    t: usize,
    degrees: Vec<usize>,
    quiet: Vec<usize>,
    f: Polynomials,
}

impl Iterated {
    /// At `t = 1`, for the sequence of the slices: `F = [I | X·I]`, every
    /// `δ` one.
    fn start(by_slice: &[&Sequence]) -> Self {
        let first = by_slice[0];
        let n = WIDTH * first.of;
        let mut f = Polynomials::zero(first.of, 2 * n, 2);
        for c in 0..n {
            f.column_mut(0, c)[c / WIDTH] = 1u64 << (c % WIDTH);
            f.column_mut(1, n + c)[c / WIDTH] = 1u64 << (c % WIDTH);
        }
        Self {
            seed: first.seed,
            of: first.of,
            dimension: first.dimension,
            columns: first.columns,
            fingerprint: fingerprint(by_slice),
            t: 1,
            degrees: vec![1; 2 * n],
            quiet: vec![0; 2 * n],
            f,
        }
    }

    /// The terms of the sequence the iteration has taken.
    #[must_use]
    pub fn terms(&self) -> usize {
        self.t
    }

    /// The iteration as bytes: `GFWI`, the fields, `F`'s length, each
    /// column's `δ` and quiet steps, and `F`'s words.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 * (8 + 2 * self.degrees.len() + self.f.words().len()));
        bytes.extend_from_slice(MAGIC);
        bytes.push(b'I');
        for word in [
            self.seed,
            self.of as u64,
            self.dimension as u64,
            self.columns as u64,
            self.fingerprint,
            self.t as u64,
            self.f.length() as u64,
        ] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let counts = self
            .degrees
            .iter()
            .chain(&self.quiet)
            .map(|&count| count as u64);
        for word in counts.chain(self.f.words().iter().copied()) {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// An iteration from its bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WiedemannBytesError> {
        let mut words = Words::of(bytes, b'I')?;
        let seed = words.next()?;
        let of = words.index()?;
        let dimension = words.index()?;
        let columns = words.index()?;
        let fingerprint = words.next()?;
        let t = words.index()?;
        let length = words.index()?;
        let count = of
            .checked_mul(2 * WIDTH)
            .ok_or(WiedemannBytesError::Shape)?;
        let f_words = length
            .checked_mul(count)
            .and_then(|words| words.checked_mul(of))
            .ok_or(WiedemannBytesError::Shape)?;
        if of == 0 || t == 0 || length == 0 {
            return Err(WiedemannBytesError::Shape);
        }
        let mut counts = || -> Result<Vec<usize>, WiedemannBytesError> {
            (0..count)
                .map(|_| {
                    let count = words.index()?;
                    if count > t {
                        return Err(WiedemannBytesError::Shape);
                    }
                    Ok(count)
                })
                .collect()
        };
        let degrees = counts()?;
        let quiet = counts()?;
        let f = words.take(f_words)?;
        if !words.bytes.is_empty() {
            return Err(WiedemannBytesError::Shape);
        }
        Ok(Self {
            seed,
            of,
            dimension,
            columns,
            fingerprint,
            t,
            degrees,
            quiet,
            f: Polynomials::from_words(of, count, f),
        })
    }
}

/// What the generator stage made of the terms it had.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Generated {
    /// The generators.
    Settled(Generators),
    /// Not yet.
    Short {
        /// The terms each slice should reach, worked out from where the
        /// columns stand.
        terms: usize,
        /// The iteration, to resume over them.
        iterated: Iterated,
    },
}

/// The generators of the sequence the slices make together, every slice of
/// one seed and length, from `prior` where an earlier pass stopped or from
/// the start: the 64 lanes of least degree once 64 columns of the iteration
/// have settled, or as many as there are columns that can. `None` for
/// slices that are not one sequence's, or a `prior` not theirs: of another
/// seed, shape or first term, or past their length.
#[must_use]
pub fn wiedemann_generators(
    slices: &[Sequence],
    prior: Option<Iterated>,
    threads: usize,
) -> Option<Generated> {
    let first = slices.first()?;
    let of = first.of;
    if slices.len() != of
        || slices.iter().any(|slice| {
            slice.seed != first.seed
                || slice.of != of
                || slice.dimension != first.dimension
                || slice.terms() != first.terms()
        })
    {
        return None;
    }
    let mut by_slice = vec![None; of];
    for slice in slices {
        by_slice[slice.slice] = Some(slice);
    }
    let by_slice: Vec<&Sequence> = by_slice.into_iter().collect::<Option<_>>()?;
    let iterated = match prior {
        Some(prior) => {
            if (
                prior.seed,
                prior.of,
                prior.dimension,
                prior.columns,
                prior.fingerprint,
            ) != (
                first.seed,
                of,
                first.dimension,
                first.columns,
                fingerprint(&by_slice),
            ) || prior.t > first.terms()
            {
                return None;
            }
            prior
        }
        None => Iterated::start(&by_slice),
    };
    let iterated = crate::parallel::with_budget(threads, || take_terms(&by_slice, iterated));
    Some(settle(iterated))
}

/// The iteration taken on over the slices' terms from where it stopped.
fn take_terms(by_slice: &[&Sequence], mut iterated: Iterated) -> Iterated {
    let of = iterated.of;
    let n = WIDTH * of;
    let terms = by_slice[0].terms();
    if terms <= iterated.t {
        return iterated;
    }
    // E's coefficient i is A·F's at t + i, which reaches back over F's
    // degree.
    let low = iterated.t.saturating_sub(iterated.f.length() - 1);
    let a = sequence_terms(by_slice, of, low..terms);
    let e = product(&a, &iterated.f).coefficients(iterated.t - low..terms - low);
    let mut state = Degrees {
        degrees: std::mem::take(&mut iterated.degrees),
        quiet: std::mem::take(&mut iterated.quiet),
    };
    let pi = steps(&e, &mut state, halves_from(2 * n, of));
    Iterated {
        t: terms,
        degrees: state.degrees,
        quiet: state.quiet,
        f: product(&iterated.f, &pi).trimmed(),
        ..iterated
    }
}

/// The generators where enough columns have settled, or the terms wanted.
///
/// A column of `F` gone to zero never takes a step again and generates
/// nothing; `r` of them leave `n − r` columns that can settle, the other
/// `m` taking every step's `m` multiplications by `X`. The columns that can
/// settle then close their gap at `(n − r)/(m + n − r)` a step, so one whose
/// gap at its last step was `g` reaches the gap `λ` at which columns stop
/// after `(λ − g)·(m + n − r)/(n − r)` more, and settles `SPARE_TERMS`
/// after that; the terms wanted are those that bring the 64th of them
/// there. For a generic sequence `λ` is `⌈N/m⌉` and a little less (Thomé's
/// habilitation, §12.1.3); it is taken as the greater of that and the
/// greatest gap at which any column has been seen to take a step, so that
/// where `⌈N/m⌉` is short, what the columns did corrects it.
fn settle(iterated: Iterated) -> Generated {
    let of = iterated.of;
    let (m, n) = (WIDTH * of, WIDTH * of);
    let live: Vec<usize> = (0..m + n)
        .filter(|&column| !iterated.f.zero_column(column))
        .collect();
    let r = m + n - live.len();
    let need = WIDTH.min(n.saturating_sub(r)).max(1);
    let settled: Vec<usize> = live
        .iter()
        .copied()
        .filter(|&column| iterated.quiet[column] >= SPARE_TERMS)
        .collect();
    if settled.len() >= need {
        // Each from its leading coefficient: a column's δ can pass its
        // degree, and the powers of B between would be applied to a vector
        // that may already be a null vector, and take it to zero.
        let mut lanes: Vec<Vec<u64>> = settled
            .iter()
            .map(|&column| {
                let f = &iterated.f;
                let lead = (0..f.length())
                    .rev()
                    .find(|&d| any(f.column(d, column)))
                    .expect("a live column");
                (0..=lead)
                    .rev()
                    .flat_map(|d| f.column(d, column).iter().copied())
                    .collect()
            })
            .collect();
        lanes.sort_by_key(Vec::len);
        lanes.truncate(WIDTH);
        return Generated::Settled(Generators {
            seed: iterated.seed,
            of,
            dimension: iterated.dimension,
            columns: iterated.columns,
            lanes,
        });
    }
    // Each live column's gap at its last step.
    let gaps: Vec<usize> = live
        .iter()
        .map(|&column| {
            (iterated.t - iterated.quiet[column]).saturating_sub(iterated.degrees[column])
        })
        .collect();
    let lambda = gaps
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(iterated.dimension.div_ceil(m));
    let open = n.saturating_sub(r).max(1);
    let mut more: Vec<usize> = gaps
        .iter()
        .map(|&gap| ((lambda - gap) * (m + n - r)).div_ceil(open) + SPARE_TERMS)
        .collect();
    more.sort_unstable();
    let more = more.get(need - 1).copied().unwrap_or(SPARE_TERMS).max(1);
    Generated::Short {
        terms: iterated.t + more,
        iterated,
    }
}

/// The slices' first term, folded by FNV-1a a word at a time (Fowler, Noll
/// and Vo): sequences of one seed and shape from different matrices differ
/// in it.
fn fingerprint(by_slice: &[&Sequence]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let rows = WIDTH * by_slice[0].of;
    by_slice
        .iter()
        .flat_map(|slice| slice.terms.iter().take(rows))
        .fold(OFFSET, |hash, &word| (hash ^ word).wrapping_mul(PRIME))
}

/// Terms `range` of the sequence the slices make, a column an `m`-bit
/// vector: term `k`'s rows are each slab's words, so a 64-row block of slab
/// `j`, transposed, is that block of the rows of columns `64·j` onwards.
fn sequence_terms(by_slice: &[&Sequence], of: usize, range: std::ops::Range<usize>) -> Polynomials {
    let n = WIDTH * of;
    let mut a = Polynomials::zero(of, n, range.len());
    for (index, k) in range.enumerate() {
        for (j, slice) in by_slice.iter().enumerate() {
            for w in 0..of {
                let mut block = [0u64; WIDTH];
                block.copy_from_slice(&slice.terms[k * n + w * WIDTH..][..WIDTH]);
                transpose(&mut block);
                for (b, &word) in block.iter().enumerate() {
                    a.column_mut(index, j * WIDTH + b)[w] = word;
                }
            }
        }
    }
    a
}

/// Slice `slice`'s part of the solutions: `w_j = Σᵢ B^{δ−i}·z_j·fᵢ⁽ʲ⁾`, a
/// lane a generator, by Horner's rule from `f₀` up, so that the leading
/// coefficient meets `B` the fewest times, `fᵢ⁽ʲ⁾` being word `j` of each
/// lane's coefficient `i`. The sum over the
/// slices is `w`. `dimension` words, or `None` for a slice out of range or a
/// matrix not the generators'.
#[must_use]
pub fn wiedemann_solution(
    matrix: &SparseMatrix,
    generators: &Generators,
    slice: usize,
    threads: usize,
) -> Option<Vec<u64>> {
    if slice >= generators.of || matrix.rows().is_empty() || matrix.columns() == 0 {
        return None;
    }
    let sparse = Sparse::forward_only(matrix.rows().to_vec(), matrix.columns(), threads);
    let (count, columns) = (sparse.relations(), sparse.columns());
    let dimension = count.max(columns);
    if dimension != generators.dimension || columns != generators.columns {
        return None;
    }
    let (of, degree) = (generators.of, generators.degree());
    let z = z_slice(generators.seed, slice, dimension);
    let mut image = Block::zeroed(dimension);
    let mut w = Block::zeroed(dimension);
    let runs = word_runs(dimension, threads);
    for power in (0..=degree).rev() {
        if power < degree {
            advance(&sparse, &mut w, &mut image);
        }
        // Row k of the table's matrix: the lanes whose coefficient at this
        // power, in this slice's word, has bit k. Coefficient `p` of the
        // reversed lane is that of `λ^{δ−p}`, the one `B^p` meets; a lane of
        // lower degree joins when the rule reaches its own.
        let mut by_bit = [0u64; WIDTH];
        for (lane, coefficients) in generators.lanes.iter().enumerate() {
            let lane_degree = coefficients.len() / of - 1;
            if power > lane_degree {
                continue;
            }
            let mut bits = coefficients[power * of + slice];
            while bits != 0 {
                by_bit[bits.trailing_zeros() as usize] |= 1u64 << lane;
                bits &= bits - 1;
            }
        }
        if !any(&by_bit) {
            continue;
        }
        let product = SmallProduct::new(&by_bit);
        let (z, w) = (&z, &w);
        crate::parallel::map_ordered(&runs, threads, |_, run| {
            for index in run.clone() {
                let add = product.apply(z.word(index));
                if add != 0 {
                    w.set(index, w.word(index) ^ add);
                }
            }
        });
    }
    Some(w.words().collect())
}

/// The words of a block of `length` in runs, a few a thread, for passes over
/// every word that the threads take in turn; one run below a thread's
/// share of [`MINIMUM_WORDS_PER_RUN`].
fn word_runs(length: usize, threads: usize) -> Vec<std::ops::Range<usize>> {
    let run = length
        .div_ceil(threads.max(1) * RUNS_PER_THREAD)
        .max(MINIMUM_WORDS_PER_RUN);
    (0..length)
        .step_by(run)
        .map(|start| start..(start + run).min(length))
        .collect()
}

/// The runs a pass over a block's words is cut into a thread, so that a
/// slow run is an eighth of a thread's share.
const RUNS_PER_THREAD: usize = 8;

/// The fewest words a run is given: a run is a thread's wake and its
/// joining, some microseconds, and a word's product a few nanoseconds, so a
/// run of 64 Ki words is a thousand times what starting it costs.
const MINIMUM_WORDS_PER_RUN: usize = 1 << 16;

/// The dependencies from the slices' parts of the solutions, summed: a
/// lane is a null vector when `B` takes it to zero, one that is not yet is
/// given `B` again and is one, or zero, a few powers on, and every lane that
/// is one is checked against the matrix's rows. `None` when none survives,
/// or the parts are not of this matrix.
#[must_use]
pub fn wiedemann_finish(
    matrix: &SparseMatrix,
    partials: &[Vec<u64>],
    threads: usize,
) -> Option<Vec<Vec<usize>>> {
    if matrix.rows().is_empty() || matrix.columns() == 0 || partials.is_empty() {
        return None;
    }
    let sparse = Sparse::forward_only(matrix.rows().to_vec(), matrix.columns(), threads);
    let (count, columns) = (sparse.relations(), sparse.columns());
    let dimension = count.max(columns);
    if partials.iter().any(|partial| partial.len() != dimension) {
        return None;
    }
    let w = Block::zeroed(dimension);
    for partial in partials {
        for (index, &word) in partial.iter().enumerate() {
            w.set(index, w.word(index) ^ word);
        }
    }
    let image = Block::zeroed(columns);
    // A lane is alive on its relations' coordinates: the padding beyond
    // them is no part of a dependency.
    let mut candidates: Vec<Vec<u64>> = Vec::new();
    for _ in 0..POWERS {
        sparse.forward(&w, &image);
        let unfinished = image.words().fold(0u64, |lanes, word| lanes | word);
        let alive = w.range(0, count).fold(0u64, |lanes, word| lanes | word);
        let mut done = alive & !unfinished;
        while done != 0 {
            let mut vector = lane(&w, done.trailing_zeros() as usize);
            vector.truncate(count.div_ceil(WIDTH));
            if !count.is_multiple_of(WIDTH) {
                if let Some(last) = vector.last_mut() {
                    *last &= (1u64 << (count % WIDTH)) - 1;
                }
            }
            candidates.push(vector);
            done &= done - 1;
        }
        if unfinished == 0 {
            break;
        }
        // On with the lanes B has not finished, and those alone.
        for index in 0..dimension {
            let word = if index < columns {
                image.word(index) & unfinished
            } else {
                0
            };
            w.set(index, word);
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    let is_null = |indices: &[usize]| {
        let mut parity = vec![0u64; matrix.columns().div_ceil(WIDTH)];
        for &index in indices {
            for &column in &matrix.rows()[index] {
                parity[column as usize / WIDTH] ^= 1u64 << (column as usize % WIDTH);
            }
        }
        !any(&parity)
    };
    let found: Vec<Vec<usize>> = crate::parallel::map_ordered(&candidates, threads, |_, vector| {
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
    use super::*;
    use crate::gf2_impl::block_lanczos_dependencies_sparse;

    /// Knuth's MMIX linear congruential generator (TAOCP vol. 2, §3.3.4),
    /// its high bytes.
    struct Lcg(u64);

    impl RandomSource for Lcg {
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for byte in dest {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *byte = (self.0 >> 56) as u8;
            }
        }
    }

    /// A matrix as a sieve's is: `rows` relations of `weight` columns each
    /// among `columns`, fewer columns than rows.
    fn sieve_like(rows: usize, columns: usize, weight: usize, seed: u64) -> SparseMatrix {
        let mut rng = Lcg(seed);
        let mut draw = || {
            let mut bytes = [0u8; 8];
            rng.fill_bytes(&mut bytes);
            u64::from_le_bytes(bytes)
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

    fn sums_to_zero(matrix: &SparseMatrix, set: &[usize]) -> bool {
        let mut parity = vec![false; matrix.columns()];
        for &index in set {
            for &column in &matrix.rows()[index] {
                parity[column as usize] ^= true;
            }
        }
        parity.iter().all(|odd| !odd)
    }

    /// The stages run apart, `of` slices, through their bytes: what one
    /// host would do with `of` hosts' files.
    fn staged(
        matrix: &SparseMatrix,
        seed: u64,
        of: usize,
        threads: usize,
    ) -> Option<Vec<Vec<usize>>> {
        let mut terms = wiedemann_terms(matrix, of);
        let mut slices: Vec<Option<Sequence>> = vec![None; of];
        let mut iterated: Option<Iterated> = None;
        loop {
            let mut written = Vec::new();
            for (slice, prior) in slices.iter_mut().enumerate() {
                let next =
                    wiedemann_sequence(matrix, seed, slice, of, terms, prior.as_ref(), threads)?;
                let bytes = next.to_bytes();
                written.push(Sequence::from_bytes(&bytes).expect("the slice reads back"));
                *prior = Some(next);
            }
            assert!(written.iter().all(|slice| slice.terms() == terms));
            match wiedemann_generators(&written, iterated.take(), threads)? {
                Generated::Settled(generators) => {
                    let generators =
                        Generators::from_bytes(&generators.to_bytes()).expect("read back");
                    let partials: Vec<Vec<u64>> = (0..of)
                        .map(|slice| wiedemann_solution(matrix, &generators, slice, threads))
                        .collect::<Option<_>>()?;
                    return wiedemann_finish(matrix, &partials, threads);
                }
                Generated::Short {
                    terms: wanted,
                    iterated: stopped,
                } => {
                    if wanted > wiedemann_terms_most(matrix, of) {
                        return None;
                    }
                    terms = wanted;
                    iterated = Some(Iterated::from_bytes(&stopped.to_bytes()).expect("read back"));
                }
            }
        }
    }

    /// The slices of one seed at `terms`, each on from its prior.
    fn slices_to(
        matrix: &SparseMatrix,
        seed: u64,
        of: usize,
        terms: usize,
        prior: Option<&[Sequence]>,
    ) -> Vec<Sequence> {
        (0..of)
            .map(|slice| {
                let prior = prior.map(|prior| &prior[slice]);
                wiedemann_sequence(matrix, seed, slice, of, terms, prior, 1).expect("a slice")
            })
            .collect()
    }

    /// The slices and generators of one seed, from the first length and on
    /// by the terms each short pass wants, with the count of those passes.
    fn settled(matrix: &SparseMatrix, seed: u64, of: usize) -> (Vec<Sequence>, Generators, usize) {
        let mut slices = slices_to(matrix, seed, of, wiedemann_terms(matrix, of), None);
        let mut iterated = None;
        for short in 0.. {
            match wiedemann_generators(&slices, iterated.take(), 1).expect("one sequence's") {
                Generated::Settled(generators) => return (slices, generators, short),
                Generated::Short {
                    terms,
                    iterated: stopped,
                } => {
                    assert!(terms > slices[0].terms(), "a short pass wants more terms");
                    slices = slices_to(matrix, seed, of, terms, Some(&slices));
                    iterated = Some(stopped);
                }
            }
        }
        unreachable!()
    }

    /// Passes resumed over more terms, once or twice, make what one pass
    /// over them all makes; a pass given no new terms wants what it wanted;
    /// the iteration reads back from its bytes; and it is refused for a
    /// sequence of another matrix.
    #[test]
    fn another_matrix_is_not_continued() {
        let matrix = sieve_like(96, 96, 7, 31);
        let other = sieve_like(96, 96, 7, 32);
        let three = wiedemann_sequence(&matrix, 42, 0, 1, 3, None, 1).expect("a slice");
        let resumed = wiedemann_sequence(&other, 42, 0, 1, 8, Some(&three), 1).expect("a slice");
        let fresh = wiedemann_sequence(&other, 42, 0, 1, 8, None, 1).expect("a slice");
        assert_eq!(resumed, fresh, "another matrix's terms are not kept");
        let same = wiedemann_sequence(&matrix, 42, 0, 1, 8, Some(&three), 1).expect("a slice");
        assert_eq!(
            same,
            wiedemann_sequence(&matrix, 42, 0, 1, 8, None, 1).expect("a slice"),
            "the same matrix's are"
        );
        assert_eq!(Sequence::from_bytes(&same.to_bytes()), Ok(same));
    }

    #[test]
    fn a_resumed_pass_is_one_pass() {
        let matrix = sieve_like(1500, 1400, 7, 31);
        let other = sieve_like(1500, 1400, 7, 32);
        for of in [1usize, 2] {
            for seed in [3u64, 4] {
                let first = wiedemann_terms(&matrix, of) - SPARE_TERMS;
                let short = slices_to(&matrix, seed, of, first, None);
                let Some(Generated::Short { terms, iterated }) =
                    wiedemann_generators(&short, None, 1)
                else {
                    panic!("{of} slices, seed {seed}: {first} terms settle");
                };
                assert_eq!(
                    Iterated::from_bytes(&iterated.to_bytes()),
                    Ok(iterated.clone())
                );
                assert_eq!(
                    wiedemann_generators(&short, Some(iterated.clone()), 1),
                    Some(Generated::Short {
                        terms,
                        iterated: iterated.clone()
                    }),
                    "{of} slices, seed {seed}: no new terms"
                );
                let elsewhere = slices_to(&other, seed, of, first, None);
                assert_eq!(
                    wiedemann_generators(&elsewhere, Some(iterated.clone()), 1),
                    None,
                    "{of} slices, seed {seed}: another matrix's sequence"
                );
                let whole = slices_to(&matrix, seed, of, terms, None);
                let one_pass = wiedemann_generators(&whole, None, 1);
                let longer = slices_to(&matrix, seed, of, terms, Some(&short));
                assert_eq!(
                    wiedemann_generators(&longer, Some(iterated.clone()), 1),
                    one_pass,
                    "{of} slices, seed {seed}: one resume"
                );
                let middle = slices_to(&matrix, seed, of, (first + terms) / 2, Some(&short));
                let between = match wiedemann_generators(&middle, Some(iterated), 1) {
                    Some(Generated::Short { iterated, .. }) => iterated,
                    made => panic!("{of} slices, seed {seed}: halfway made {made:?}"),
                };
                let longer = slices_to(&matrix, seed, of, terms, Some(&middle));
                assert_eq!(
                    wiedemann_generators(&longer, Some(between), 1),
                    one_pass,
                    "{of} slices, seed {seed}: two resumes"
                );
            }
        }
    }

    /// Dependencies are found at sizes from under a block to many blocks,
    /// and each is one: the check here shares nothing with the solver's.
    #[test]
    fn wiedemann_finds_dependencies_of_sieve_like_matrices() {
        for (rows, columns, weight, seed) in [
            (40, 30, 3, 1),
            (200, 180, 5, 2),
            (700, 650, 8, 3),
            (2_500, 2_400, 12, 4),
            (6_000, 5_900, 20, 5),
        ] {
            let matrix = sieve_like(rows, columns, weight, seed);
            for threads in [1, 4] {
                let found =
                    block_wiedemann_dependencies_sparse(&matrix, &mut Lcg(seed ^ 0xdead), threads)
                        .unwrap_or_else(|| panic!("{rows} by {columns}: none found"));
                assert!(!found.is_empty());
                for set in &found {
                    assert!(!set.is_empty());
                    assert!(
                        sums_to_zero(&matrix, set),
                        "{rows} by {columns}: a set returned is not a dependency"
                    );
                }
            }
        }
    }

    /// The stages in slices, through their bytes, find dependencies too,
    /// with two and four slices, and fewer terms a slice the more slices.
    #[test]
    fn wiedemann_in_slices_finds_dependencies() {
        for (rows, columns, weight, seed) in [(900, 850, 6, 21), (3_000, 2_900, 12, 22)] {
            let matrix = sieve_like(rows, columns, weight, seed);
            for of in [1, 2, 4] {
                let found = staged(&matrix, seed, of, 2)
                    .unwrap_or_else(|| panic!("{rows} by {columns} in {of} slices: none found"));
                assert!(
                    found.len() >= WIDTH / 4,
                    "{of} slices found {}",
                    found.len()
                );
                for set in &found {
                    assert!(sums_to_zero(&matrix, set), "{of} slices: not a dependency");
                }
            }
        }
    }

    /// Shapes the interface admits: more columns than rows, a full-rank
    /// rectangle with no dependency, rows with no entries, and columns no
    /// row touches. Each is solved or declined, never left to panic.
    #[test]
    fn wiedemann_takes_every_shape_of_sparse_matrix() {
        // Two equal rows over three columns: the dependency is both rows.
        let wide = SparseMatrix::new(3, vec![vec![0, 1, 2], vec![0, 1, 2]]);
        let found = block_wiedemann_dependencies_sparse(&wide, &mut Lcg(3), 1)
            .expect("the equal rows are a dependency");
        assert!(found.iter().all(|set| sums_to_zero(&wide, set)));
        assert!(found.contains(&vec![0, 1]));
        // Full rank, no dependency: declined.
        let full = SparseMatrix::new(4, vec![vec![0], vec![1, 2], vec![3]]);
        assert_eq!(
            block_wiedemann_dependencies_sparse(&full, &mut Lcg(3), 1),
            None
        );
        // An empty row is a dependency of itself; a column no row touches
        // is no column.
        let holes = SparseMatrix::new(
            6,
            vec![vec![0, 5], vec![], vec![0, 5], vec![2], vec![2, 5], vec![5]],
        );
        let found = block_wiedemann_dependencies_sparse(&holes, &mut Lcg(9), 2)
            .expect("the empty row alone is a dependency");
        assert!(found.iter().all(|set| sums_to_zero(&holes, set)));
        assert!(found.iter().all(|set| set.iter().all(|&row| row < 6)));
        // A sieve-like rectangle with more columns than rows.
        let tall = sieve_like(300, 360, 4, 17);
        if let Some(found) = block_wiedemann_dependencies_sparse(&tall, &mut Lcg(17), 2) {
            assert!(found.iter().all(|set| sums_to_zero(&tall, set)));
            assert!(found.iter().all(|set| set.iter().all(|&row| row < 300)));
        }
    }

    /// The same matrix by both solvers: each returns dependencies, and
    /// Wiedemann's are as many as its block's lanes allow where the null
    /// space is that large.
    #[test]
    fn wiedemann_and_lanczos_both_solve_a_matrix_with_a_wide_null_space() {
        let matrix = sieve_like(3_000, 2_800, 10, 7);
        let lanczos = block_lanczos_dependencies_sparse(&matrix, &mut Lcg(11), 2)
            .expect("Lanczos finds dependencies");
        let wiedemann = block_wiedemann_dependencies_sparse(&matrix, &mut Lcg(11), 2)
            .expect("Wiedemann finds dependencies");
        assert!(lanczos.iter().all(|set| sums_to_zero(&matrix, set)));
        assert!(wiedemann.iter().all(|set| sums_to_zero(&matrix, set)));
        assert!(
            wiedemann.len() >= WIDTH / 2,
            "a null space of two hundred and Wiedemann found {}",
            wiedemann.len()
        );
    }

    /// A kept matrix (`SparseMatrix::to_bytes`) solved by both, timed:
    /// `WIEDEMANN_MATRIX=path WIEDEMANN_THREADS=n WIEDEMANN_SLICES=J cargo
    /// test --release --lib wiedemann_cost_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement, on a matrix the caller keeps"]
    fn wiedemann_cost_probe() {
        let Some(path) = std::env::var_os("WIEDEMANN_MATRIX") else {
            return;
        };
        let bytes = std::fs::read(&path).expect("the kept matrix is read");
        let matrix = SparseMatrix::from_bytes(&bytes).expect("a kept matrix");
        let threads: usize = std::env::var("WIEDEMANN_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let slices: usize = std::env::var("WIEDEMANN_SLICES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let only = std::env::var("WIEDEMANN_ONLY").ok();
        for (name, which) in [("wiedemann", 0), ("lanczos", 1), ("sliced", 2)] {
            if only.as_deref().is_some_and(|only| only != name) {
                continue;
            }
            let started = std::time::Instant::now();
            let found = match which {
                0 => block_wiedemann_dependencies_sparse(&matrix, &mut Lcg(1), threads),
                1 => block_lanczos_dependencies_sparse(&matrix, &mut Lcg(1), threads),
                _ => staged(&matrix, 1, slices, threads),
            };
            let found = found.unwrap_or_default();
            assert!(found.iter().all(|set| sums_to_zero(&matrix, set)));
            eprintln!(
                "{name} {} x {}, {} nonzeros, {threads} threads, {slices} slices: {:?}, {} dependencies",
                matrix.rows().len(),
                matrix.columns(),
                matrix.nonzeros(),
                started.elapsed(),
                found.len()
            );
        }
    }

    /// The generators' defining property, on the sequence the slices make:
    /// every lane annihilates the terms from its degree to the end.
    #[test]
    fn the_lanes_generate_the_sequence() {
        let matrix = sieve_like(900, 850, 6, 21);
        for of in [1, 3] {
            let (slices, generators, _) = settled(&matrix, 5, of);
            let terms = slices[0].terms();
            let rows = WIDTH * of;
            let term = |k: usize| -> Vec<u64> {
                let mut term = vec![0u64; rows * of];
                for (j, slice) in slices.iter().enumerate() {
                    for (r, &word) in slice.terms[k * rows..(k + 1) * rows].iter().enumerate() {
                        term[r * of + j] = word;
                    }
                }
                term
            };
            // aₖ·v the plain way: row r of the product is the parity of
            // row r of aₖ against v.
            let apply = |term: &[u64], v: &[u64]| -> Vec<u64> {
                let mut out = vec![0u64; of];
                for r in 0..rows {
                    let row = &term[r * of..(r + 1) * of];
                    let parity = row
                        .iter()
                        .zip(v)
                        .map(|(a, b)| (a & b).count_ones())
                        .sum::<u32>()
                        & 1;
                    out[r / WIDTH] |= u64::from(parity) << (r % WIDTH);
                }
                out
            };
            // A lane is cut at its leading coefficient, and holds from its
            // nominal degree, which it no longer carries: the last terms,
            // which every settled column has passed, are the check here.
            for lane in &generators.lanes {
                let degree = lane.len() / of - 1;
                assert!(
                    degree < terms - SPARE_TERMS,
                    "a generator of degree {degree} proves nothing"
                );
                for k in (terms - SPARE_TERMS)..terms {
                    let mut sum = vec![0u64; of];
                    for (i, coefficient) in lane.chunks(of).enumerate() {
                        for (s, a) in sum
                            .iter_mut()
                            .zip(apply(&term(k - degree + i), coefficient))
                        {
                            *s ^= a;
                        }
                    }
                    assert!(
                        !any(&sum),
                        "{of} slices, degree {degree}: the term at {k} is not zero"
                    );
                }
            }
        }
    }

    /// The iteration's own invariant, at one word and at two: every settled
    /// column of `F`, at its `δ`, annihilates the terms from `δ` to the end.
    #[test]
    fn a_settled_column_annihilates_the_sequence_from_its_nominal_degree() {
        let matrix = sieve_like(900, 850, 6, 21);
        for of in [1usize, 2] {
            let (slices, _, _) = settled(&matrix, 5, of);
            let terms = slices[0].terms();
            let rows = WIDTH * of;
            let term = |k: usize| -> Vec<u64> {
                let mut term = vec![0u64; rows * of];
                for (j, slice) in slices.iter().enumerate() {
                    for (r, &word) in slice.terms[k * rows..(k + 1) * rows].iter().enumerate() {
                        term[r * of + j] = word;
                    }
                }
                term
            };
            let by_slice: Vec<&Sequence> = slices.iter().collect();
            let iterated = take_terms(&by_slice, Iterated::start(&by_slice));
            let settled: Vec<(usize, Vec<Vec<u64>>)> = (0..2 * rows)
                .filter(|&column| {
                    iterated.quiet[column] >= SPARE_TERMS && !iterated.f.zero_column(column)
                })
                .map(|column| {
                    let f = (0..iterated.f.length())
                        .map(|d| iterated.f.column(d, column).to_vec())
                        .collect();
                    (iterated.degrees[column], f)
                })
                .collect();
            let dead = (0..2 * rows)
                .filter(|&column| iterated.f.zero_column(column))
                .count();
            assert!(
                settled.len() >= WIDTH.min(rows - dead),
                "{of} words: {} settled, {dead} columns of F zero",
                settled.len()
            );
            let apply = |term: &[u64], v: &[u64]| -> Vec<u64> {
                let mut out = vec![0u64; of];
                for r in 0..rows {
                    let row = &term[r * of..(r + 1) * of];
                    let parity = row
                        .iter()
                        .zip(v)
                        .map(|(a, b)| (a & b).count_ones())
                        .sum::<u32>()
                        & 1;
                    out[r / WIDTH] |= u64::from(parity) << (r % WIDTH);
                }
                out
            };
            for (degree, f) in settled {
                assert!(
                    f.iter()
                        .skip(degree + 1)
                        .all(|coefficient| !any(coefficient)),
                    "{of} words: a column of F past its δ {degree}"
                );
                let mut failed = Vec::new();
                for k in degree..terms {
                    let mut sum = vec![0u64; of];
                    for (i, coefficient) in f.iter().enumerate().take(degree + 1) {
                        for (s, a) in sum.iter_mut().zip(apply(&term(k - i), coefficient)) {
                            *s ^= a;
                        }
                    }
                    if any(&sum) {
                        failed.push(k);
                    }
                }
                assert!(
                    failed.is_empty(),
                    "{of} words, degree {degree} of {terms} terms: nonzero at {failed:?}"
                );
            }
        }
    }

    /// Bytes that are not a stage's are refused for their reason.
    #[test]
    fn a_stage_refuses_bytes_that_are_not_its_own() {
        let matrix = sieve_like(200, 180, 5, 2);
        let slice = wiedemann_sequence(&matrix, 9, 0, 2, 20, None, 1).expect("a slice");
        let bytes = slice.to_bytes();
        assert_eq!(Sequence::from_bytes(&bytes).as_ref(), Ok(&slice));
        assert_eq!(
            Generators::from_bytes(&bytes),
            Err(WiedemannBytesError::Header)
        );
        assert_eq!(
            Sequence::from_bytes(&bytes[..bytes.len() - 8]),
            Err(WiedemannBytesError::Short)
        );
        let mut wrong = bytes.clone();
        wrong[4 + 8..4 + 16].copy_from_slice(&7u64.to_le_bytes());
        assert_eq!(
            Sequence::from_bytes(&wrong),
            Err(WiedemannBytesError::Shape)
        );
    }
}
