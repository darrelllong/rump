//! Block Wiedemann over GF(2): the null space from sequences that can be
//! taken apart.
//!
//! Block Lanczos is a chain, each iteration made of the last. Coppersmith's
//! block Wiedemann (*Solving homogeneous linear equations over GF(2) via
//! block Wiedemann algorithm*, Math. Comp. 62 (1994), 333–350) has three
//! stages instead, and two of them are products of the matrix with blocks
//! that do not depend on one another:
//!
//! 1. **The sequence.** For random blocks `x` and `z` and `y = B·z`, the
//!    small matrices `aₖ = xᵀ·Bᵏ·y`, `k < L`.
//! 2. **The generator.** Vectors of polynomials `f(λ) = Σ fⱼλʲ` of degree
//!    at most `δ` with `Σⱼ aₖ₋ⱼ·fⱼ = 0` for `δ ≤ k < L`, by Coppersmith's
//!    iteration: his generalisation of Berlekamp–Massey, here in the form
//!    Thomé gives it (*Subquadratic computation of vector generating
//!    polynomials and improvement of the block Wiedemann algorithm*, J.
//!    Symbolic Comput. 33 (2002), 757–775, §3), quadratic in `L`.
//! 3. **The solution.** `w = Σⱼ B^{δ−j}·z·fⱼ`, by Horner's rule. Then
//!    `xᵀ·Bⁱ·(B·w) = 0` for every `i` the sequence reaches, which leaves
//!    `B·w = 0` unless the projection by `x` hides it, and a `w` that is not
//!    yet a null vector becomes one, or zero, under a few more `B`.
//!
//! `B` here is square, of dimension `N`, the greater of the relations and
//! the live columns: the relations' matrix `M` taken as the map from a
//! block over the relations to one over the columns, padded with zero rows
//! or zero columns to square. A null vector of `B` restricted to the
//! relations' coordinates is a dependency; the padding coordinates are
//! not returned.
//!
//! The price against Lanczos is half as many products again: `N/m + N/n`
//! for the sequence and `N/n` for the solution, where an iteration of
//! Lanczos is two and it takes `N/64`. Nothing here is trusted either:
//! every vector returned is checked against the caller's rows. The three
//! stages run on one host as yet; the two of them that are products are
//! independent sequences and could be spread, which is not done here.

use std::sync::Arc;

use super::{any, lane, Block, Small, SmallProduct, Sparse, SparseMatrix, WIDTH};
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

/// The columns of the generator matrix: `n` for the generators sought and
/// `m` more that carry the iteration's rank.
const COLUMNS: usize = 2 * WIDTH;

/// The term from which a step of the iteration runs on the threads: a step
/// is `128` columns of as many table lookups as the term, and below this
/// many the threads cost more than they do. Set by hand; not measured.
const PARALLEL_FROM_TERMS: usize = 2_048;

/// Dependencies of `matrix` by block Wiedemann, as
/// [`block_lanczos_dependencies_sparse`](super::block_lanczos_dependencies_sparse)
/// returns them: sets of row indices, each checked to sum to zero over the
/// caller's rows, or `None` when none was found. The matrix may have more
/// columns than rows, or fewer.
///
/// `None` is any of: no rows or no columns; the sequence passing
/// three times `N/64` terms (`TERMS_MOST`) without its generators settling; no
/// lane of the solution a null vector within sixteen products (`POWERS`); or every
/// candidate failing the check. Another random source may do better; the
/// exact solver always does.
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
    let sparse = Sparse::forward_only(matrix.rows().to_vec(), matrix.columns(), threads);
    wiedemann(&sparse, rng, threads, |indices| {
        let mut parity = vec![0u64; matrix.columns().div_ceil(WIDTH)];
        for &index in indices {
            for &column in &matrix.rows()[index] {
                parity[column as usize / WIDTH] ^= 1u64 << (column as usize % WIDTH);
            }
        }
        !any(&parity)
    })
}

/// `B·v` in place: `M·v` over the columns, which are the first of the
/// block's coordinates, and zero below. The block is `N` words, the
/// greater of the relations and the columns; `M` reads the first of them
/// up to the relations.
fn advance(matrix: &Sparse, v: &Arc<Block>, image: &Arc<Block>) {
    matrix.forward(v, image);
    for (index, word) in image.words().enumerate() {
        v.set(index, word);
    }
    for index in image.len()..v.len() {
        v.set(index, 0);
    }
}

/// The three stages over `matrix`, the candidates filtered by `is_null`.
fn wiedemann<R: RandomSource + ?Sized>(
    matrix: &Sparse,
    rng: &mut R,
    threads: usize,
    is_null: impl Fn(&[usize]) -> bool + Sync,
) -> Option<Vec<Vec<usize>>> {
    let (count, columns) = (matrix.relations(), matrix.columns());
    if columns == 0 {
        return None;
    }
    // The square operator's dimension: a block has a word a coordinate.
    let dimension = count.max(columns);
    let mut draw = move || {
        let mut bytes = [0u8; 8];
        rng.fill_bytes(&mut bytes);
        u64::from_le_bytes(bytes)
    };

    // x: each of its vectors the sum of a few coordinates among the columns,
    // where every power of B past the zeroth is supported.
    let supports: Vec<[usize; X_WEIGHT]> = (0..WIDTH)
        .map(|_| std::array::from_fn(|_| (draw() % columns as u64) as usize))
        .collect();
    let z = Block::of((0..dimension).map(|_| draw()));

    // The sequence aₖ = xᵀ·Bᵏ·y, y = B·z, and the generators as it grows:
    // each term is given to the iteration, which is done when n columns
    // have had no term of their own for a block's width of steps. That is
    // near N/m + N/n terms where x and z see all of B; where they do not,
    // a column settles early, the others carry more, and the count is not
    // known beforehand.
    let image = Block::zeroed(columns);
    let v = Block::of(z.words());
    advance(matrix, &v, &image);
    let most = TERMS_MOST * dimension.div_ceil(WIDTH) + 2 * SPARE_TERMS;
    let mut iteration = Iteration::new();
    while iteration.settled() < WIDTH {
        if iteration.terms() == most {
            return None;
        }
        if iteration.terms() > 0 {
            advance(matrix, &v, &image);
        }
        let mut a = [0u64; WIDTH];
        for (row, support) in a.iter_mut().zip(&supports) {
            *row = support.iter().fold(0, |sum, &at| sum ^ v.word(at));
        }
        iteration.take(&a, threads);
    }
    // Each generator from its leading coefficient: a column's nominal degree
    // can pass its own, and the powers of B between would be applied to a
    // vector that may already be a null vector, and take it to zero.
    let mut generators: Vec<&[u64]> = iteration
        .generators()
        .filter_map(|column| {
            let lead = column.reversed.iter().position(|&word| word != 0)?;
            Some(&column.reversed[lead..])
        })
        .collect();
    generators.sort_by_key(|reversed| reversed.len());
    generators.truncate(WIDTH);
    let degree = generators.iter().map(|reversed| reversed.len()).max()? - 1;

    // The solutions, a lane a generator, by Horner's rule from the leading
    // power of B down: w ← B·w + z·fⱼ. `reversed[i]` is the coefficient of
    // λ^{δ−i}, the one that meets B^i; a generator of lower degree joins
    // when the rule reaches its own.
    let w = Block::zeroed(dimension);
    for power in (0..=degree).rev() {
        if power < degree {
            advance(matrix, &w, &image);
        }
        // Row k of the table's matrix: the lanes whose coefficient has bit k.
        let mut by_bit = [0u64; WIDTH];
        for (lane, reversed) in generators.iter().enumerate() {
            let mut bits = reversed.get(power).copied().unwrap_or(0);
            while bits != 0 {
                by_bit[bits.trailing_zeros() as usize] |= 1u64 << lane;
                bits &= bits - 1;
            }
        }
        if !any(&by_bit) {
            continue;
        }
        let product = SmallProduct::new(&by_bit);
        for index in 0..dimension {
            let add = product.apply(z.word(index));
            if add != 0 {
                w.set(index, w.word(index) ^ add);
            }
        }
    }

    // A lane is a null vector when B takes it to zero; one that is not yet
    // is given B again, and is one, or zero, a few powers on. A lane is
    // alive on its relations' coordinates: the padding beyond them is no
    // part of a dependency.
    let mut candidates: Vec<Vec<u64>> = Vec::new();
    for _ in 0..POWERS {
        matrix.forward(&w, &image);
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

/// A column of the generator matrix: a vector of `n` polynomials, a word a
/// coefficient, held from the top down. `reversed[i]` is the coefficient of
/// `λ^{δ−i}` for the nominal degree `δ = reversed.len() − 1`, so that
/// multiplying by `λ` is a zero pushed on the end.
#[derive(Clone)]
struct Column {
    reversed: Vec<u64>,
}

/// Coppersmith's iteration over the sequence `a`, a term at a time: the
/// `m + n` columns, of which those that have settled are generators.
///
/// At step `t` every column `f` of nominal degree `δ` has
/// `Σⱼ aₖ₋ⱼ·fⱼ = 0` for `δ ≤ k < t`. The step takes each column's term at
/// `k = t`, an `m`-bit word, and eliminates among the columns in order of
/// degree, a column only ever receiving columns of no greater degree, so
/// that its degree stands. The `m` columns the elimination leaves non-zero
/// are multiplied by `λ`, which moves their terms out of the way; the `n`
/// it clears hold at `t` too.
struct Iteration {
    /// aₖ·v for a word v of n bits, by bytes of v: the tables hold aₖ's
    /// columns, the XOR of those v selects.
    products: Vec<SmallProduct>,
    columns: Vec<Column>,
    /// The steps each column has gone without a term of its own.
    quiet: Vec<usize>,
}

impl Iteration {
    /// At `t = 1`, with every degree one: the unit vectors, whose terms are
    /// `a₁`'s columns, and `λ` times them, whose terms are `a₀`'s.
    fn new() -> Self {
        let columns = (0..COLUMNS)
            .map(|index| {
                let unit = 1u64 << (index % WIDTH);
                Column {
                    reversed: if index < WIDTH {
                        vec![0, unit]
                    } else {
                        vec![unit, 0]
                    },
                }
            })
            .collect();
        Self {
            products: Vec::new(),
            columns,
            quiet: vec![0; COLUMNS],
        }
    }

    /// The terms taken so far.
    fn terms(&self) -> usize {
        self.products.len()
    }

    /// The columns that have settled.
    fn settled(&self) -> usize {
        self.quiet
            .iter()
            .filter(|&&steps| steps >= SPARE_TERMS)
            .count()
    }

    /// The settled columns, which generate the sequence.
    fn generators(&self) -> impl Iterator<Item = &Column> {
        self.columns
            .iter()
            .zip(&self.quiet)
            .filter(|(_, &steps)| steps >= SPARE_TERMS)
            .map(|(column, _)| column)
    }

    /// The next term of the sequence, and from the second the step at it.
    fn take(&mut self, a: &Small, threads: usize) {
        let mut transposed = [0u64; WIDTH];
        for (row, &word) in a.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                transposed[bits.trailing_zeros() as usize] |= 1u64 << row;
                bits &= bits - 1;
            }
        }
        self.products.push(SmallProduct::new(&transposed));
        let t = self.products.len() - 1;
        if t == 0 {
            return;
        }
        let (products, columns) = (&self.products, &self.columns);

        // Each column's term at t: Σᵢ a_{t−δ+i}·reversed[i]. Threads are
        // spawned for a step only once a step has the work to repay them:
        // the parallel map makes its workers afresh each call.
        let term_of = |column: &Column| {
            let first = t + 1 - column.reversed.len();
            column
                .reversed
                .iter()
                .enumerate()
                .filter(|(_, &word)| word != 0)
                .fold(0u64, |sum, (i, &word)| {
                    sum ^ products[first + i].apply(word)
                })
        };
        let wide = t >= PARALLEL_FROM_TERMS;
        let mut terms: Vec<u64> = if wide {
            crate::parallel::map_ordered(columns, threads, |_, column| term_of(column))
        } else {
            columns.iter().map(term_of).collect()
        };

        // The elimination, on the terms alone: which columns each column
        // receives, as a set of the columns the step began with.
        let mut order: Vec<usize> = (0..COLUMNS).collect();
        order.sort_by_key(|&index| (columns[index].reversed.len(), index));
        let mut received: Vec<u128> = (0..COLUMNS).map(|index| 1u128 << index).collect();
        let mut pivots: Vec<(u32, usize)> = Vec::with_capacity(WIDTH);
        let mut pivot = [false; COLUMNS];
        for &index in &order {
            let mut term = terms[index];
            let mut set = received[index];
            for &(bit, other) in &pivots {
                if term >> bit & 1 == 1 {
                    term ^= terms[other];
                    set ^= received[other];
                }
            }
            terms[index] = term;
            received[index] = set;
            if term != 0 {
                pivots.push((term.trailing_zeros(), index));
                pivot[index] = true;
            }
        }

        // The columns themselves, each from the step's own, aligned at the
        // top: a column of lower degree enters below its receiver's.
        let combine = |&index: &usize| {
            let mut reversed = columns[index].reversed.clone();
            let mut others = received[index] & !(1u128 << index);
            while others != 0 {
                let other = &columns[others.trailing_zeros() as usize].reversed;
                others &= others - 1;
                let offset = reversed.len() - other.len();
                for (word, &add) in reversed[offset..].iter_mut().zip(other) {
                    *word ^= add;
                }
            }
            if pivot[index] {
                reversed.push(0);
            }
            Column { reversed }
        };
        let next: Vec<Column> = if wide {
            crate::parallel::map_ordered(&order, threads, |_, index| combine(index))
        } else {
            order.iter().map(combine).collect()
        };
        for (&index, column) in order.iter().zip(next) {
            self.columns[index] = column;
            self.quiet[index] = if pivot[index] {
                0
            } else {
                self.quiet[index] + 1
            };
        }
    }
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
    /// `WIEDEMANN_MATRIX=path WIEDEMANN_THREADS=n cargo test --release --lib
    /// wiedemann_cost_probe -- --ignored --nocapture`.
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
        for (name, lanczos) in [("wiedemann", false), ("lanczos", true)] {
            let started = std::time::Instant::now();
            let found = if lanczos {
                block_lanczos_dependencies_sparse(&matrix, &mut Lcg(1), threads)
            } else {
                block_wiedemann_dependencies_sparse(&matrix, &mut Lcg(1), threads)
            };
            let found = found.unwrap_or_default();
            assert!(found.iter().all(|set| sums_to_zero(&matrix, set)));
            eprintln!(
                "{name} {} x {}, {} nonzeros, {threads} threads: {:?}, {} dependencies",
                matrix.rows().len(),
                matrix.columns(),
                matrix.nonzeros(),
                started.elapsed(),
                found.len()
            );
        }
    }

    /// The generator's invariant, on a sequence that has one: after the
    /// iteration every column of least degree annihilates the sequence from
    /// its degree to the end.
    #[test]
    fn the_columns_of_least_degree_generate_the_sequence() {
        let matrix = sieve_like(900, 850, 6, 21);
        let sparse = Sparse::from_lists(matrix.rows().to_vec(), matrix.columns(), 1);
        let mut rng = Lcg(5);
        let mut draw = || {
            let mut bytes = [0u8; 8];
            rng.fill_bytes(&mut bytes);
            u64::from_le_bytes(bytes)
        };
        let (count, columns) = (sparse.relations(), sparse.columns());
        let coordinates: Vec<usize> = (0..WIDTH)
            .map(|_| (draw() % columns as u64) as usize)
            .collect();
        let z = Block::of((0..count).map(|_| draw()));
        let image = Block::zeroed(columns);
        let v = Block::of(z.words());
        advance(&sparse, &v, &image);
        let terms = 2 * count.div_ceil(WIDTH) + 2 * SPARE_TERMS;
        let mut sequence = Vec::new();
        let mut iteration = Iteration::new();
        for term in 0..terms {
            if term > 0 {
                advance(&sparse, &v, &image);
            }
            let mut a = [0u64; WIDTH];
            for (row, &at) in a.iter_mut().zip(&coordinates) {
                *row = v.word(at);
            }
            iteration.take(&a, 1);
            sequence.push(a);
        }
        assert!(
            iteration.settled() >= WIDTH,
            "the generators have not settled"
        );
        let columns: Vec<&Column> = iteration.generators().collect();
        let apply = |a: &Small, word: u64| {
            a.iter().enumerate().fold(0u64, |out, (row, &bits)| {
                out | u64::from((bits & word).count_ones() & 1) << row
            })
        };
        for column in columns {
            let degree = column.reversed.len() - 1;
            assert!(
                degree < terms - count.div_ceil(WIDTH),
                "a generator of degree {degree} in {terms} terms proves nothing"
            );
            for k in degree..terms {
                let term = column
                    .reversed
                    .iter()
                    .enumerate()
                    .fold(0u64, |sum, (i, &word)| {
                        sum ^ apply(&sequence[k - degree + i], word)
                    });
                assert_eq!(term, 0, "degree {degree}: the term at {k} is not zero");
            }
        }
    }
}
