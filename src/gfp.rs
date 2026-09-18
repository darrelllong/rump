//! Linear algebra over a large prime field `GF(l)`: sparse matrices whose
//! entries are small integers, and Wiedemann's algorithm for a kernel vector.
//!
//! Index-calculus discrete logarithms end in a homogeneous system over the
//! order `l` of the group: the relation matrix holds exponent counts, the
//! solution holds the logarithms. That matrix is sparse and its entries are
//! tiny — overwhelmingly `±1`, a short tail of small values — while the
//! vectors are residues hundreds of bits wide. Both facts shape everything
//! here.
//!
//! # Why not [`crate::gf2`]
//!
//! Over `GF(2)` a row combines in one pass of 64-bit words and sixty-four
//! right-hand sides ride in the same word; that is what makes Block Lanczos
//! worth its bookkeeping there. Over `GF(l)` a vector entry is several limbs
//! and addition carries, so nothing of that packing survives. The two
//! modules share a shape and no code.
//!
//! # Why Wiedemann and not Lanczos
//!
//! Montgomery's Block Lanczos exists because `GF(2)` has self-orthogonal
//! vectors, which a large prime field does not. Wiedemann needs only matrix
//! products and a linear recurrence, and its sequences are independent of
//! each other, so a blocked form distributes across machines — the property
//! that matters at the sizes where this stops being cheap. This module has
//! the scalar form: one sequence, one recurrence, correctness first.
//!
//! Wiedemann, *Solving sparse linear equations over finite fields*, IEEE
//! Trans. Inform. Theory 32 (1986), 54–62. The recurrence is found by
//! Berlekamp–Massey (Massey, *Shift-register synthesis and BCH decoding*,
//! IEEE Trans. Inform. Theory 15 (1969), 122–127).

use crate::bigint::{BarrettContext, BigUint};
use crate::number_theory_impl::mod_inverse;
use crate::random::{random_below, RandomSource};

/// A prime field, and the reduction its arithmetic runs on.
///
/// The modulus is the caller's to establish: nothing here tests it for
/// primality, and a composite modulus makes the inverses in
/// [`minimal_polynomial`] fail rather than lie — [`kernel_vector`] returns
/// `None` when that happens.
#[derive(Clone, Debug)]
pub struct Field {
    modulus: BigUint,
    /// Limbs in the modulus: the width of a reduced residue, and the base
    /// width of an accumulator.
    limbs: usize,
    barrett: BarrettContext,
}

/// Spare limbs carried above the modulus's own width while a row accumulates.
///
/// A row adds one term per nonzero, each below `|coefficient|·l`, so an
/// accumulator holds less than `nonzeros · max|coefficient| · l`. One spare
/// limb therefore suffices until a row has `2^64` entries counted with their
/// coefficients — at a hundred nonzeros and coefficients in the hundreds the
/// factor is under `2^15`. The debug assertions in [`add_into`] and
/// [`mul_add_into`] fail loudly if a caller ever approaches it.
const ACCUMULATOR_HEADROOM_LIMBS: usize = 1;

impl Field {
    /// The field of residues modulo `modulus`, or `None` for a modulus below
    /// two.
    #[must_use]
    pub fn new(modulus: BigUint) -> Option<Self> {
        let barrett = BarrettContext::new(&modulus).ok()?;
        let limbs = modulus.limbs().len();
        Some(Self {
            modulus,
            limbs,
            barrett,
        })
    }

    /// The modulus.
    #[must_use]
    pub fn modulus(&self) -> &BigUint {
        &self.modulus
    }

    /// `x mod l`.
    #[must_use]
    pub fn reduce(&self, x: &BigUint) -> BigUint {
        self.barrett.reduce(x)
    }

    /// `a + b` in the field.
    #[must_use]
    pub fn add(&self, a: &BigUint, b: &BigUint) -> BigUint {
        let sum = a.add(b);
        if sum >= self.modulus {
            sum.sub(&self.modulus)
        } else {
            sum
        }
    }

    /// `a − b` in the field.
    #[must_use]
    pub fn sub(&self, a: &BigUint, b: &BigUint) -> BigUint {
        if a >= b {
            a.sub(b)
        } else {
            a.add(&self.modulus).sub(b)
        }
    }

    /// `a · b` in the field.
    #[must_use]
    pub fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        self.reduce(&a.mul(b))
    }

    /// `a⁻¹` in the field, or `None` when `a` is zero or shares a factor with
    /// the modulus — which for a prime modulus is zero alone.
    #[must_use]
    pub fn inverse(&self, a: &BigUint) -> Option<BigUint> {
        mod_inverse(a, &self.modulus)
    }

    /// A uniform residue.
    pub fn random<R: RandomSource + ?Sized>(&self, rng: &mut R) -> BigUint {
        random_below(rng, &self.modulus).expect("a modulus above one is a non-empty range")
    }

    /// A vector of `length` uniform residues.
    pub fn random_vector<R: RandomSource + ?Sized>(
        &self,
        rng: &mut R,
        length: usize,
    ) -> Vec<BigUint> {
        (0..length).map(|_| self.random(rng)).collect()
    }
}

/// Index lists stored flat: list `i` is `values[offsets[i]..offsets[i + 1]]`.
struct Lists<T> {
    offsets: Vec<usize>,
    values: Vec<T>,
}

impl<T> Lists<T> {
    fn with_rows(rows: usize) -> Self {
        let mut offsets = Vec::with_capacity(rows + 1);
        offsets.push(0);
        Self {
            offsets,
            values: Vec::new(),
        }
    }

    fn push_row(&mut self, row: impl IntoIterator<Item = T>) {
        self.values.extend(row);
        self.offsets.push(self.values.len());
    }

    fn list(&self, index: usize) -> &[T] {
        &self.values[self.offsets[index]..self.offsets[index + 1]]
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

/// A sparse matrix over `GF(l)` whose entries are small integers.
///
/// The entries of an index-calculus matrix are exponent counts, so almost
/// every one is `+1` or `−1`. Those two cases are held as bare column
/// indices in their own lists and cost an addition or a subtraction with no
/// multiplication at all; the short tail of other values is held beside them
/// as `(column, coefficient)` pairs.
pub struct SparseMatrix {
    rows: usize,
    columns: usize,
    /// Per row, the columns whose entry is `+1`.
    plus: Lists<u32>,
    /// Per row, the columns whose entry is `−1`.
    minus: Lists<u32>,
    /// Per row, the remaining entries as `(column, coefficient)`.
    weighted: Lists<(u32, i64)>,
}

impl SparseMatrix {
    /// A matrix from its rows, each a list of `(column, coefficient)` pairs.
    ///
    /// `None` when a column index is at or past `columns`, when a row repeats
    /// a column, or when the matrix is not square: Wiedemann's sequence is
    /// built from powers of the matrix, which only a square matrix has. A
    /// caller with surplus rows drops them.
    ///
    /// Zero coefficients are dropped; they are not entries.
    #[must_use]
    pub fn new(columns: usize, rows: Vec<Vec<(u32, i64)>>) -> Option<Self> {
        if rows.len() != columns {
            return None;
        }
        let mut plus = Lists::with_rows(rows.len());
        let mut minus = Lists::with_rows(rows.len());
        let mut weighted = Lists::with_rows(rows.len());
        let mut seen = vec![usize::MAX; columns];
        for (index, row) in rows.iter().enumerate() {
            for &(column, _) in row {
                let column = column as usize;
                if column >= columns || seen[column] == index {
                    return None;
                }
                seen[column] = index;
            }
            plus.push_row(
                row.iter()
                    .filter(|&&(_, c)| c == 1)
                    .map(|&(column, _)| column),
            );
            minus.push_row(
                row.iter()
                    .filter(|&&(_, c)| c == -1)
                    .map(|&(column, _)| column),
            );
            weighted.push_row(
                row.iter()
                    .filter(|&&(_, c)| c != 0 && c != 1 && c != -1)
                    .copied(),
            );
        }
        Some(Self {
            rows: rows.len(),
            columns,
            plus,
            minus,
            weighted,
        })
    }

    /// The number of rows, which equals the number of columns.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// The number of entries.
    #[must_use]
    pub fn nonzeros(&self) -> usize {
        self.plus.len() + self.minus.len() + self.weighted.len()
    }

    /// `self · vector` over the field.
    ///
    /// Each row sums into two wide accumulators, one for the terms it adds
    /// and one for those it subtracts, and reduces once at the end: the
    /// entries are small integers, so the inner loop is limb addition with
    /// carries and no modular reduction per entry.
    ///
    /// # Panics
    ///
    /// Panics when `vector` is not as long as the matrix has columns.
    #[must_use]
    pub fn multiply(&self, field: &Field, vector: &[BigUint]) -> Vec<BigUint> {
        assert_eq!(
            vector.len(),
            self.columns,
            "the vector must have one entry per column"
        );
        let width = field.limbs + ACCUMULATOR_HEADROOM_LIMBS;
        let mut positive = vec![0u64; width];
        let mut negative = vec![0u64; width];
        let mut out = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            positive.fill(0);
            negative.fill(0);
            for &column in self.plus.list(row) {
                add_into(&mut positive, vector[column as usize].limbs());
            }
            for &column in self.minus.list(row) {
                add_into(&mut negative, vector[column as usize].limbs());
            }
            for &(column, coefficient) in self.weighted.list(row) {
                let target = if coefficient < 0 {
                    &mut negative
                } else {
                    &mut positive
                };
                mul_add_into(
                    target,
                    vector[column as usize].limbs(),
                    coefficient.unsigned_abs(),
                );
            }
            let sum = field.reduce(&BigUint::from_limbs(positive.clone()));
            let taken = field.reduce(&BigUint::from_limbs(negative.clone()));
            out.push(field.sub(&sum, &taken));
        }
        out
    }
}

/// `accumulator += value`, in place: the `±1` entries, which are most of the
/// matrix and carry no multiplication.
fn add_into(accumulator: &mut [u64], value: &[u64]) {
    mul_add_into(accumulator, value, 1);
}

/// `accumulator += multiplier · value`, in place, with the carry walking up
/// through the headroom, where it dies; see [`ACCUMULATOR_HEADROOM_LIMBS`].
fn mul_add_into(accumulator: &mut [u64], value: &[u64], multiplier: u64) {
    let mut carry = 0u128;
    for (slot, &limb) in accumulator.iter_mut().zip(value) {
        let product = u128::from(limb) * u128::from(multiplier) + u128::from(*slot) + carry;
        *slot = product as u64;
        carry = product >> 64;
    }
    for slot in accumulator.iter_mut().skip(value.len()) {
        if carry == 0 {
            return;
        }
        let sum = u128::from(*slot) + carry;
        *slot = sum as u64;
        carry = sum >> 64;
    }
    debug_assert_eq!(carry, 0, "the accumulator's headroom absorbs every carry");
}

/// The minimal polynomial of a linearly recurrent sequence, low coefficient
/// first and monic: `Σ fᵢ·s(k + i) = 0` for every `k` the sequence reaches.
///
/// Berlekamp–Massey finds the shortest recurrence, as its connection
/// polynomial `c` with `c(0) = 1` and `Σ cᵢ·s(k − i) = 0`; reading the same
/// recurrence forwards rather than backwards is its reversal, which is the
/// minimal polynomial and what a matrix's Krylov sequence is asked for. The
/// reversal is taken over the recurrence's full length, trailing zeros
/// included: a zero constant term is exactly what says the matrix has zero
/// as an eigenvalue, so trimming it would throw away the kernel.
///
/// `None` when the field rejects an inverse, which for a prime modulus
/// cannot happen and for a composite one is how a wrong modulus shows
/// itself. A sequence from an `n × n` matrix has a minimal polynomial of
/// degree at most `n`, and `2n` terms determine it.
#[must_use]
pub fn minimal_polynomial(field: &Field, sequence: &[BigUint]) -> Option<Vec<BigUint>> {
    let zero = BigUint::zero();
    let one = BigUint::one();
    let mut current = vec![one.clone()];
    let mut previous = vec![one.clone()];
    let mut length = 0usize;
    let mut since = 1usize;
    let mut previous_discrepancy = one.clone();

    for (index, term) in sequence.iter().enumerate() {
        // The discrepancy: what the current recurrence predicts against what
        // the sequence does.
        let mut discrepancy = field.reduce(term);
        for offset in 1..=length {
            if offset < current.len() {
                let contribution = field.mul(&current[offset], &sequence[index - offset]);
                discrepancy = field.add(&discrepancy, &contribution);
            }
        }
        if discrepancy.is_zero() {
            since += 1;
            continue;
        }
        let scale = field.mul(&discrepancy, &field.inverse(&previous_discrepancy)?);
        let mut updated = current.clone();
        if updated.len() < previous.len() + since {
            updated.resize(previous.len() + since, zero.clone());
        }
        for (offset, coefficient) in previous.iter().enumerate() {
            let term = field.mul(&scale, coefficient);
            let slot = &mut updated[offset + since];
            *slot = field.sub(slot, &term);
        }
        if 2 * length <= index {
            previous = current;
            previous_discrepancy = discrepancy;
            length = index + 1 - length;
            since = 1;
        } else {
            since += 1;
        }
        current = updated;
    }
    // The recurrence has `length` taps; the connection polynomial carries
    // one coefficient more than that, padded with the zeros that trailing
    // taps of zero produce. Reversed, it is the minimal polynomial.
    current.resize(length + 1, zero.clone());
    current.reverse();
    Some(current)
}

/// What one Wiedemann draw settled.
///
/// The two failures want opposite responses from a caller — draw again, or
/// go back and change the matrix — so they are not the same answer.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Kernel {
    /// A non-zero `x` with `M·x = 0`.
    Vector(Vec<BigUint>),
    /// The matrix has no kernel: this draw's recurrence has a non-zero
    /// constant term, so zero is not a root of it.
    ///
    /// Certain when the recurrence reached the matrix's dimension, since it
    /// is then the matrix's own minimal polynomial. Otherwise the recurrence
    /// is a divisor of that polynomial and could have divided a zero root
    /// away, which happens with probability about `n/l` — the same risk
    /// every other answer here carries, and negligible at a field of
    /// cryptographic size. Either way the response is to change the matrix,
    /// not to draw again.
    NoKernel,
    /// This draw settled nothing: its sequence gave a proper divisor of the
    /// matrix's minimal polynomial, which happens about `n/l` of the time.
    /// Draw again. Repeated failures are evidence about the matrix — a caller
    /// that has exhausted a sensible bound should suspect its own rows rather
    /// than its luck.
    Inconclusive,
}

/// A non-zero `x` with `matrix · x = 0`, or why this attempt found none.
///
/// *Some* element of the kernel: when the kernel is more than a line, which
/// of it comes back is not the caller's to choose, and a caller that wants a
/// particular one — the logarithms, say, which are the line whose constant
/// coordinate is not zero — must make the kernel one-dimensional before
/// asking. A wide kernel does not fail here and does not look like a
/// failure: it returns vectors, all of them genuine and none of them the one
/// wanted (factoring, 2026-09-17, on a matrix of 62 columns with a
/// 16-dimensional kernel: ten draws, ten kernel vectors, every one with zero
/// in the coordinate that had to be scaled to one).
///
/// Wiedemann's algorithm: the scalars `u·Aᵏv` obey a linear recurrence whose
/// minimal polynomial divides the matrix's, and `2n` of them determine it.
/// Writing that polynomial as `xᵐ·g(x)` with `g(0) ≠ 0`, the vector `g(A)v`
/// is killed by some power of `A`, and the last non-zero iterate before it
/// dies is a kernel vector.
///
/// [`Kernel::Inconclusive`] is this draw failing, and the answer is another
/// draw; [`Kernel::NoKernel`] is about the matrix, and the answer is different
/// rows. A caller that cannot tell them apart retries for ever on a matrix
/// that needed more rows, or collects more rows for a matrix that needed
/// another draw.
pub fn kernel_vector<R: RandomSource + ?Sized>(
    matrix: &SparseMatrix,
    field: &Field,
    rng: &mut R,
) -> Kernel {
    let n = matrix.columns();
    if n == 0 {
        return Kernel::NoKernel;
    }
    let projection = field.random_vector(rng, n);
    let start = field.random_vector(rng, n);
    kernel_vector_from(matrix, field, &projection, &start)
}

/// [`kernel_vector`] with the two draws supplied, so a test can put the
/// algorithm in the positions a random draw reaches only with probability
/// about `n/l`.
fn kernel_vector_from(
    matrix: &SparseMatrix,
    field: &Field,
    projection: &[BigUint],
    start: &[BigUint],
) -> Kernel {
    let n = matrix.columns();

    // The scalar sequence u·A^k v, for k = 0 … 2n − 1.
    let mut iterate = start.to_vec();
    let mut sequence = Vec::with_capacity(2 * n);
    for step in 0..2 * n {
        if step > 0 {
            iterate = matrix.multiply(field, &iterate);
        }
        let mut dot = BigUint::zero();
        for (left, right) in projection.iter().zip(&iterate) {
            dot = field.add(&dot, &field.mul(left, right));
        }
        sequence.push(dot);
    }

    let Some(polynomial) = minimal_polynomial(field, &sequence) else {
        return Kernel::Inconclusive;
    };
    // A non-zero constant term says zero is not a root of this draw's
    // recurrence, so there is no kernel to walk towards.
    // The polynomial is monic, so its last coefficient is one and the
    // valuation is below its length.
    let valuation = polynomial.iter().take_while(|c| c.is_zero()).count();
    if valuation == 0 {
        return Kernel::NoKernel;
    }
    let shifted = &polynomial[valuation..];

    // g(A)·v by Horner, from the top coefficient down.
    let mut accumulated = vec![BigUint::zero(); n];
    for coefficient in shifted.iter().rev() {
        accumulated = matrix.multiply(field, &accumulated);
        for (slot, entry) in accumulated.iter_mut().zip(start) {
            *slot = field.add(slot, &field.mul(coefficient, entry));
        }
    }

    // Some power of A kills it, and the last non-zero iterate before that is
    // the answer.
    //
    // Only if the polynomial was the whole minimal polynomial of the start
    // vector, though. When the draw finds a proper divisor of it — about
    // `n/l` of the time — no power of A kills the result, so the walk is
    // bounded: a vector an `n × n` matrix drives to zero at all reaches zero
    // within `n` steps, and one that has not is not going to. Either way the
    // answer returned is in the kernel, because it is the iterate before a
    // zero one.
    for _ in 0..n {
        let next = matrix.multiply(field, &accumulated);
        if next.iter().all(BigUint::is_zero) {
            // A zero iterate here would say the shifted polynomial already
            // annihilates the start vector, which would make the recurrence
            // it came from longer than it needed to be; the check is a guard
            // on that reasoning, not a path, so nothing exercises it.
            return if accumulated.iter().any(|entry| !entry.is_zero()) {
                Kernel::Vector(accumulated)
            } else {
                Kernel::Inconclusive
            };
        }
        accumulated = next;
    }
    Kernel::Inconclusive
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Arbitrary, fixed so a failure reproduces.
    const SEED: u64 = 0x6766_705f_7769_6564; // "gfp_wied"

    /// A deterministic source for the tests: xorshift64 (Marsaglia, *Xorshift
    /// RNGs*, J. Statistical Software 8 (2003), with the shift triple
    /// 13/7/17 from its table).
    struct TestRng(u64);

    impl RandomSource for TestRng {
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

    impl TestRng {
        fn next_u64(&mut self) -> u64 {
            let mut bytes = [0u8; 8];
            self.fill_bytes(&mut bytes);
            u64::from_le_bytes(bytes)
        }
    }

    /// 2⁶¹ − 1, prime: small enough to read in a failure and wide enough that
    /// a residue is more than one limb after multiplication.
    fn small_field() -> Field {
        Field::new(BigUint::from_u64((1 << 61) - 1)).expect("a prime above one")
    }

    /// 2¹²⁷ − 1, prime: two limbs, so an accumulator carries across limbs and
    /// the vectors are the width discrete logarithms actually use.
    fn wide_field() -> Field {
        let mut power = BigUint::one();
        power.shl_bits(127);
        Field::new(power.sub(&BigUint::one())).expect("a prime above one")
    }

    /// The product computed one entry at a time in field arithmetic: no
    /// accumulator, no lazy reduction, no sign split.
    fn naive_multiply(
        field: &Field,
        columns: usize,
        rows: &[Vec<(u32, i64)>],
        vector: &[BigUint],
    ) -> Vec<BigUint> {
        rows.iter()
            .map(|row| {
                let mut total = BigUint::zero();
                for &(column, coefficient) in row {
                    assert!((column as usize) < columns);
                    let entry = field.reduce(&BigUint::from_u64(coefficient.unsigned_abs()));
                    let term = field.mul(&entry, &vector[column as usize]);
                    total = if coefficient < 0 {
                        field.sub(&total, &term)
                    } else {
                        field.add(&total, &term)
                    };
                }
                total
            })
            .collect()
    }

    /// Random sparse rows with small entries, as an index-calculus matrix
    /// has them: mostly ±1, a tail of small values.
    fn random_rows(rng: &mut TestRng, columns: usize, weight: usize) -> Vec<Vec<(u32, i64)>> {
        // Entries above one in magnitude, as a share of the whole: exponent
        // counts past one are the tail, not the body.
        const WEIGHTED_SHARE: u64 = 8;
        // The largest coefficient drawn, which stands for the tail's reach.
        const LARGEST_COEFFICIENT: u64 = 5;
        (0..columns)
            .map(|_| {
                let mut used = Vec::new();
                let mut row = Vec::new();
                for _ in 0..weight {
                    let column = (rng.next_u64() % columns as u64) as u32;
                    if used.contains(&column) {
                        continue;
                    }
                    used.push(column);
                    let magnitude = if rng.next_u64().is_multiple_of(WEIGHTED_SHARE) {
                        2 + rng.next_u64() % (LARGEST_COEFFICIENT - 1)
                    } else {
                        1
                    } as i64;
                    let sign = if rng.next_u64() & 1 == 0 { 1 } else { -1 };
                    row.push((column, sign * magnitude));
                }
                row
            })
            .collect()
    }

    /// A kernel vector by dense Gaussian elimination over the field, or
    /// `None` when the matrix has full rank: the oracle Wiedemann is checked
    /// against, sharing no code with it.
    fn dense_kernel_vector(
        field: &Field,
        columns: usize,
        rows: &[Vec<(u32, i64)>],
    ) -> Option<Vec<BigUint>> {
        let mut dense = vec![vec![BigUint::zero(); columns]; rows.len()];
        for (index, row) in rows.iter().enumerate() {
            for &(column, coefficient) in row {
                let entry = field.reduce(&BigUint::from_u64(coefficient.unsigned_abs()));
                let slot = &mut dense[index][column as usize];
                *slot = if coefficient < 0 {
                    field.sub(slot, &entry)
                } else {
                    field.add(slot, &entry)
                };
            }
        }
        // Row-reduce, recording which column each pivot sits in.
        let mut pivot_of_column = vec![None; columns];
        let mut pivot_row = 0usize;
        for column in 0..columns {
            let Some(found) = (pivot_row..dense.len()).find(|&r| !dense[r][column].is_zero())
            else {
                continue;
            };
            dense.swap(pivot_row, found);
            let inverse = field.inverse(&dense[pivot_row][column])?;
            for entry in &mut dense[pivot_row] {
                *entry = field.mul(entry, &inverse);
            }
            for other in 0..dense.len() {
                if other == pivot_row || dense[other][column].is_zero() {
                    continue;
                }
                let factor = dense[other][column].clone();
                let pivot = dense[pivot_row].clone();
                for (entry, above) in dense[other].iter_mut().zip(&pivot) {
                    let term = field.mul(&factor, above);
                    *entry = field.sub(entry, &term);
                }
            }
            pivot_of_column[column] = Some(pivot_row);
            pivot_row += 1;
        }
        // A column without a pivot is free: set it to one, read the pivots off.
        let free = (0..columns).find(|&c| pivot_of_column[c].is_none())?;
        let mut solution = vec![BigUint::zero(); columns];
        solution[free] = BigUint::one();
        for column in 0..columns {
            if let Some(row) = pivot_of_column[column] {
                solution[column] = field.sub(&BigUint::zero(), &dense[row][free]);
            }
        }
        Some(solution)
    }

    #[test]
    fn the_product_agrees_with_one_entry_at_a_time() {
        // Widths either side of a limb, and a weight around what a filtered
        // sieve matrix carries.
        const SIZES: [usize; 4] = [1, 8, 37, 64];
        const WEIGHT: usize = 9;
        let mut rng = TestRng(SEED);
        for field in [small_field(), wide_field()] {
            for columns in SIZES {
                let rows = random_rows(&mut rng, columns, WEIGHT);
                let matrix = SparseMatrix::new(columns, rows.clone()).expect("square and in range");
                let vector = field.random_vector(&mut rng, columns);
                assert_eq!(
                    matrix.multiply(&field, &vector),
                    naive_multiply(&field, columns, &rows, &vector),
                    "{columns} columns modulo {}",
                    field.modulus()
                );
            }
        }
    }

    #[test]
    fn the_product_carries_across_every_limb() {
        // Every vector entry at the modulus's ceiling and every column
        // present in every row, so each accumulator reaches its widest and
        // every limb of it carries into the next.
        const COLUMNS: usize = 16;
        // The largest coefficient the tail draws, and the ±1 body: the two
        // accumulate by different routes, and each has its own carry walk.
        const SHAPES: [i64; 3] = [5, 1, -1];
        let field = wide_field();
        let largest = field.modulus().sub(&BigUint::one());
        let vector = vec![largest; COLUMNS];
        for coefficient in SHAPES {
            let rows: Vec<Vec<(u32, i64)>> = (0..COLUMNS)
                .map(|_| (0..COLUMNS as u32).map(|c| (c, coefficient)).collect())
                .collect();
            let matrix = SparseMatrix::new(COLUMNS, rows.clone()).expect("square and in range");
            assert_eq!(
                matrix.multiply(&field, &vector),
                naive_multiply(&field, COLUMNS, &rows, &vector),
                "coefficient {coefficient} at the modulus's ceiling"
            );
        }
        // A row of alternating signs at the ceiling: the two accumulators
        // fill together and the difference is taken across limbs.
        let rows: Vec<Vec<(u32, i64)>> = (0..COLUMNS)
            .map(|_| {
                (0..COLUMNS as u32)
                    .map(|c| (c, if c % 2 == 0 { 1 } else { -1 }))
                    .collect()
            })
            .collect();
        let matrix = SparseMatrix::new(COLUMNS, rows.clone()).expect("square and in range");
        assert_eq!(
            matrix.multiply(&field, &vector),
            naive_multiply(&field, COLUMNS, &rows, &vector)
        );
    }

    #[test]
    fn a_matrix_is_refused_unless_it_is_square_and_in_range() {
        assert!(
            SparseMatrix::new(2, vec![vec![(0, 1)]]).is_none(),
            "not square"
        );
        assert!(
            SparseMatrix::new(1, vec![vec![(1, 1)]]).is_none(),
            "column past the width"
        );
        assert!(
            SparseMatrix::new(1, vec![vec![(0, 1), (0, 1)]]).is_none(),
            "a column twice in one row"
        );
        assert!(
            SparseMatrix::new(1, vec![vec![(0, 0)]]).is_some(),
            "a zero is no entry"
        );
        assert_eq!(
            SparseMatrix::new(1, vec![vec![(0, 0)]])
                .expect("valid")
                .nonzeros(),
            0
        );
    }

    #[test]
    fn the_minimal_polynomial_is_the_recurrence_the_sequence_obeys() {
        // Degrees either side of one, and enough terms that 2·degree of them
        // determine the recurrence.
        const DEGREES: [usize; 4] = [1, 2, 5, 12];
        let field = small_field();
        let mut rng = TestRng(SEED ^ 0x1111);
        for degree in DEGREES {
            // A random monic recurrence s(k) = Σ a_i·s(k − 1 − i), run out
            // from random initial terms.
            let coefficients = field.random_vector(&mut rng, degree);
            let mut sequence = field.random_vector(&mut rng, degree);
            for index in degree..4 * degree + 4 {
                let mut next = BigUint::zero();
                for (offset, coefficient) in coefficients.iter().enumerate() {
                    let term = field.mul(coefficient, &sequence[index - 1 - offset]);
                    next = field.add(&next, &term);
                }
                sequence.push(next);
            }
            let polynomial = minimal_polynomial(&field, &sequence).expect("a prime field inverts");
            assert!(
                polynomial.len() <= degree + 1,
                "degree {} exceeds {degree}",
                polynomial.len() - 1
            );
            assert_eq!(
                *polynomial.last().expect("a non-empty polynomial"),
                BigUint::one(),
                "the minimal polynomial is monic"
            );
            // The recurrence it found must annihilate the sequence.
            for start in 0..sequence.len() - polynomial.len() + 1 {
                let mut total = BigUint::zero();
                for (offset, coefficient) in polynomial.iter().enumerate() {
                    let term = field.mul(coefficient, &sequence[start + offset]);
                    total = field.add(&total, &term);
                }
                assert!(total.is_zero(), "the recurrence fails from term {start}");
            }
        }
    }

    #[test]
    fn a_singular_matrix_gives_up_a_kernel_vector() {
        // Sizes small enough for the dense oracle, weights around a sieve
        // matrix's density.
        const SIZES: [usize; 3] = [8, 21, 40];
        const WEIGHT: usize = 7;
        // Wiedemann's draw can find a proper divisor of the minimal
        // polynomial; the documented answer is to draw again.
        const ATTEMPTS: usize = 8;
        let mut rng = TestRng(SEED ^ 0x2222);
        let mut found = 0;
        for field in [small_field(), wide_field()] {
            for columns in SIZES {
                let mut rows = random_rows(&mut rng, columns, WEIGHT);
                // Two equal rows: the matrix loses rank, so a kernel exists.
                rows[columns - 1] = rows[0].clone();
                let matrix = SparseMatrix::new(columns, rows.clone()).expect("square and in range");
                assert!(
                    dense_kernel_vector(&field, columns, &rows).is_some(),
                    "the oracle finds no kernel to look for"
                );
                let mut solution = None;
                for _ in 0..ATTEMPTS {
                    if let Kernel::Vector(candidate) = kernel_vector(&matrix, &field, &mut rng) {
                        solution = Some(candidate);
                        break;
                    }
                }
                let solution = solution.expect("a kernel vector within the attempts");
                assert!(
                    solution.iter().any(|entry| !entry.is_zero()),
                    "the zero vector is not a kernel vector"
                );
                assert!(
                    matrix
                        .multiply(&field, &solution)
                        .iter()
                        .all(BigUint::is_zero),
                    "{columns} columns modulo {}: the vector is not in the kernel",
                    field.modulus()
                );
                found += 1;
            }
        }
        assert_eq!(found, 6, "every shape was exercised");
    }

    /// A matrix whose kernel is many dimensions wide still gives up a vector,
    /// and an invertible one is told apart from a draw that failed.
    ///
    /// The distinction is what a caller acts on: more rows against another
    /// draw. A rank-deficient matrix that reported bad luck would have the
    /// caller retrying for ever.
    #[test]
    fn a_wide_kernel_is_found_and_an_invertible_matrix_is_named() {
        // A shape and density near a small index-calculus matrix, with the
        // rank cut by repeating rows.
        const COLUMNS: usize = 62;
        const WEIGHT: usize = 6;
        const RANK_DEFICIT: usize = 15;
        const DRAWS: usize = 10;
        let field = small_field();
        let mut rng = TestRng(SEED ^ 0x9999);
        let mut rows = random_rows(&mut rng, COLUMNS, WEIGHT);
        for index in 0..RANK_DEFICIT {
            rows[COLUMNS - 1 - index] = rows[index].clone();
        }
        let matrix = SparseMatrix::new(COLUMNS, rows.clone()).expect("square and in range");
        assert!(
            dense_kernel_vector(&field, COLUMNS, &rows).is_some(),
            "the oracle finds no kernel to look for"
        );
        for _ in 0..DRAWS {
            match kernel_vector(&matrix, &field, &mut rng) {
                Kernel::Vector(solution) => {
                    assert!(solution.iter().any(|entry| !entry.is_zero()));
                    assert!(matrix
                        .multiply(&field, &solution)
                        .iter()
                        .all(BigUint::is_zero));
                }
                Kernel::NoKernel => panic!("a rank-deficient matrix reported as having no kernel"),
                Kernel::Inconclusive => {}
            }
        }
    }

    #[test]
    fn an_invertible_matrix_has_no_kernel_to_report() {
        // The identity, and a permutation with a sign: both invertible.
        const COLUMNS: usize = 12;
        const ATTEMPTS: usize = 4;
        let field = small_field();
        let mut rng = TestRng(SEED ^ 0x3333);
        for shift in [0usize, 1] {
            let rows: Vec<Vec<(u32, i64)>> = (0..COLUMNS)
                .map(|row| {
                    vec![(
                        ((row + shift) % COLUMNS) as u32,
                        if shift == 0 { 1 } else { -1 },
                    )]
                })
                .collect();
            let matrix = SparseMatrix::new(COLUMNS, rows).expect("square and in range");
            for _ in 0..ATTEMPTS {
                assert_eq!(
                    kernel_vector(&matrix, &field, &mut rng),
                    Kernel::NoKernel,
                    "an invertible matrix must be named as having no kernel, not as bad luck"
                );
            }
        }
    }

    /// A draw whose sequence has a proper divisor of the start vector's
    /// minimal polynomial: the walk finds nothing and must say so rather
    /// than iterate forever.
    ///
    /// `diag(0, 1)` with the projection `(1, 0)` sees the sequence
    /// `1, 0, 0, …`, whose recurrence is `x`, but `(1, 1)` is not driven to
    /// zero by any power of the matrix — its second coordinate is fixed.
    #[test]
    fn a_draw_that_cannot_reach_the_kernel_returns_nothing() {
        let field = small_field();
        let matrix =
            SparseMatrix::new(2, vec![Vec::new(), vec![(1, 1)]]).expect("square and in range");
        let projection = vec![BigUint::one(), BigUint::zero()];
        let start = vec![BigUint::one(), BigUint::one()];
        assert_eq!(
            kernel_vector_from(&matrix, &field, &projection, &start),
            Kernel::Inconclusive,
            "a failed draw is not a statement about the matrix"
        );
    }

    /// The fixture: a real index-calculus matrix, 62 columns over
    /// `GF(524351)`, with the kernel vector the run that produced it
    /// verified — factoring's `src/dlog.rs`, a discrete logarithm modulo
    /// `p = 1048703` whose answers were checked by hand.
    ///
    /// Constructed matrices exercise the arithmetic; this one is the shape
    /// the module exists for, with the entry distribution a factor base
    /// actually produces: mostly ±1, a tail of small counts, and one dense
    /// column of large constants.
    #[test]
    fn the_index_calculus_fixture_is_solved() {
        const FIXTURE: &str = include_str!("../tests/data/index_calculus_62.txt");
        // Wiedemann's draw fails about n/l of the time; the field here is
        // 19 bits, so a handful of draws is the sensible bound.
        const ATTEMPTS: usize = 8;
        let field = Field::new(BigUint::from_u64(524_351)).expect("a prime above one");
        let mut width = 0usize;
        let mut rows: Vec<Vec<(u32, i64)>> = Vec::new();
        let mut expected: Vec<BigUint> = Vec::new();
        for line in FIXTURE.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("width ") {
                width = rest.trim().parse().expect("a width");
            } else if let Some(rest) = line.strip_prefix("row ") {
                let mut row = Vec::new();
                for entry in rest.trim_matches(['[', ']']).split("), (") {
                    let entry = entry.trim_matches(['(', ')', ' ']);
                    let (column, coefficient) = entry.split_once(',').expect("a pair");
                    row.push((
                        column.trim().parse().expect("a column"),
                        coefficient.trim().parse().expect("a coefficient"),
                    ));
                }
                rows.push(row);
            } else if let Some(rest) = line.strip_prefix("kernel ") {
                expected = rest
                    .trim_matches(['[', ']'])
                    .split(',')
                    .map(|entry| {
                        BigUint::from_str_radix(entry.trim().trim_matches('"'), 10)
                            .expect("a residue")
                    })
                    .collect();
            }
        }
        assert_eq!(rows.len(), width, "one row per column");
        assert_eq!(expected.len(), width, "one kernel entry per column");
        let matrix = SparseMatrix::new(width, rows).expect("square and in range");

        // The vector the run verified is in the kernel, as its producer said.
        assert!(
            matrix
                .multiply(&field, &expected)
                .iter()
                .all(BigUint::is_zero),
            "the fixture's own kernel vector is not in the kernel"
        );

        // And the solver finds that line: the kernel is one-dimensional here,
        // so whatever comes back is a multiple of it, which is checked by
        // cross-multiplying against the first coordinate that is not zero in
        // both — no division, no assumption about which multiple.
        let mut rng = TestRng(SEED ^ 0x6666);
        let mut solution = None;
        for _ in 0..ATTEMPTS {
            if let Kernel::Vector(candidate) = kernel_vector(&matrix, &field, &mut rng) {
                solution = Some(candidate);
                break;
            }
        }
        let solution = solution.expect("a kernel vector within the attempts");
        assert!(matrix
            .multiply(&field, &solution)
            .iter()
            .all(BigUint::is_zero));
        let pivot = (0..width)
            .find(|&i| !solution[i].is_zero() && !expected[i].is_zero())
            .expect("the two vectors share a non-zero coordinate");
        for index in 0..width {
            assert_eq!(
                field.mul(&solution[index], &expected[pivot]),
                field.mul(&expected[index], &solution[pivot]),
                "coordinate {index} is off the fixture's line"
            );
        }
    }

    /// A matrix with no columns has no kernel vector to give: the empty
    /// vector is not a non-zero one.
    #[test]
    fn a_matrix_with_no_columns_has_no_kernel() {
        let field = small_field();
        let mut rng = TestRng(SEED ^ 0x5555);
        let matrix = SparseMatrix::new(0, Vec::new()).expect("empty is square");
        assert_eq!(matrix.nonzeros(), 0);
        assert_eq!(
            kernel_vector(&matrix, &field, &mut rng),
            Kernel::NoKernel,
            "an empty matrix is settled, not unlucky"
        );
    }

    #[test]
    fn the_zero_matrix_is_all_kernel() {
        const COLUMNS: usize = 6;
        let field = small_field();
        let mut rng = TestRng(SEED ^ 0x4444);
        let matrix =
            SparseMatrix::new(COLUMNS, vec![Vec::new(); COLUMNS]).expect("square and in range");
        let Kernel::Vector(solution) = kernel_vector(&matrix, &field, &mut rng) else {
            panic!("every vector is a kernel vector of the zero matrix");
        };
        assert!(solution.iter().any(|entry| !entry.is_zero()));
        assert!(matrix
            .multiply(&field, &solution)
            .iter()
            .all(BigUint::is_zero));
    }
}
