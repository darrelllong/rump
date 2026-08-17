//! Linear algebra over GF(2): dense null space, singleton pruning, and
//! Block Lanczos.
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

use crate::random::RandomSource;

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
// `O(columns · rows · (rows + columns)/64)` and is blind to sparsity. A
// number field sieve matrix is very sparse — measured at 0.30 % full at 39
// digits, about thirty set bits across ten thousand columns — and past about
// fifty digits the elimination is the larger half of the bill: at 44 digits
// with a 128 000-prime base the sieve takes 82 s and the matrix 429 s.
//
// Block Lanczos costs `O(iterations · nonzeros)` with `iterations ≈ rows/64`,
// because sixty-four vectors ride in the bits of one machine word and every
// iteration advances all of them. That is about 160 iterations where a scalar
// method needs `2·rows ≈ 20 000`, and the blocking is the whole difference.
//
// # Provenance
//
// The recurrence is Montgomery's, and it is transcribed rather than
// remembered: equations (18)–(20) and the subspace selection of figure 1 from
// *A Block Lanczos Algorithm for Finding Dependencies over GF(2)*, EUROCRYPT
// '95, pages 106–120. The correspondence was checked against Sebastian
// Wouters' BSD-licensed C++ implementation (`github.com/SebWouters/blanczos`),
// which annotates its variables with Montgomery's names. The coefficients are
// easy to misremember and a wrong one yields *no* dependencies rather than
// wrong ones — a silent failure — so none of them is reconstructed here.
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

/// The relation matrix `M`, held once by rows and once by columns.
///
/// Both orientations are needed every iteration — `A = MᵀM` is two products —
/// and each is a gather over the side it is indexed by, so storing both costs
/// one extra copy of the indices and saves a scatter with random writes.
struct Sparse {
    /// For each relation, the columns it sets.
    by_relation: Vec<Vec<u32>>,
    /// For each column, the relations that set it.
    by_column: Vec<Vec<u32>>,
}

impl Sparse {
    fn from_packed(rows: &[Vec<u64>], columns: usize) -> Self {
        let words = columns.div_ceil(WIDTH);
        let mut by_relation = Vec::with_capacity(rows.len());
        let mut by_column = vec![Vec::new(); columns];
        for (index, row) in rows.iter().enumerate() {
            let mut set = Vec::new();
            for word in 0..words {
                let mut bits = row.get(word).copied().unwrap_or(0);
                if word == words - 1 && !columns.is_multiple_of(WIDTH) {
                    bits &= (1u64 << (columns % WIDTH)) - 1;
                }
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let column = word * WIDTH + bit;
                    set.push(column as u32);
                    by_column[column].push(index as u32);
                }
            }
            by_relation.push(set);
        }
        Self {
            by_relation,
            by_column,
        }
    }

    fn relations(&self) -> usize {
        self.by_relation.len()
    }

    fn columns(&self) -> usize {
        self.by_column.len()
    }

    /// `M·x`: a block over the relations becomes one over the columns.
    fn forward(&self, x: &[u64]) -> Vec<u64> {
        self.by_column
            .iter()
            .map(|relations| {
                relations
                    .iter()
                    .fold(0u64, |total, &index| total ^ x[index as usize])
            })
            .collect()
    }

    /// `Mᵀ·y`: a block over the columns becomes one over the relations.
    fn backward(&self, y: &[u64]) -> Vec<u64> {
        self.by_relation
            .iter()
            .map(|columns| {
                columns
                    .iter()
                    .fold(0u64, |total, &column| total ^ y[column as usize])
            })
            .collect()
    }

    /// `A·x` with `A = MᵀM`, the symmetric operator the iteration runs on.
    fn apply(&self, x: &[u64]) -> Vec<u64> {
        self.backward(&self.forward(x))
    }
}

/// `leftᵀ · right` for two blocks: the `64 × 64` matrix of inner products.
fn dot(left: &[u64], right: &[u64]) -> Small {
    let mut out = [0u64; WIDTH];
    for (a, b) in left.iter().zip(right.iter()) {
        let mut bits = *a;
        while bits != 0 {
            let lane = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            out[lane] ^= *b;
        }
    }
    out
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

/// `V·P` for a block and a `64 × 64` matrix.
fn mul_block(v: &[u64], p: &Small) -> Vec<u64> {
    v.iter()
        .map(|value| {
            let mut bits = *value;
            let mut total = 0u64;
            while bits != 0 {
                let lane = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                total ^= p[lane];
            }
            total
        })
        .collect()
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
fn lane(block: &[u64], which: usize) -> Vec<u64> {
    let mut out = vec![0u64; block.len().div_ceil(WIDTH)];
    for (index, word) in block.iter().enumerate() {
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
#[must_use]
pub fn block_lanczos_dependencies<R: RandomSource + ?Sized>(
    rows: &[Vec<u64>],
    columns: usize,
    rng: &mut R,
) -> Option<Vec<Vec<usize>>> {
    if rows.is_empty() || columns == 0 {
        return None;
    }
    let matrix = Sparse::from_packed(rows, columns);
    let count = matrix.relations();

    // The starting block is random; rump chooses no entropy source, so the
    // words come from the caller's generator rather than an internal xorshift
    // over a seed argument.
    let mut draw = move || {
        let mut bytes = [0u8; 8];
        rng.fill_bytes(&mut bytes);
        u64::from_le_bytes(bytes)
    };

    // X starts as Y and accumulates the solution; Q = V[0] = A·Y never moves.
    let mut x: Vec<u64> = (0..count).map(|_| draw()).collect();
    let q = matrix.apply(&x);

    let mut v0 = q.clone();
    let mut v1 = vec![0u64; count];
    let mut v2 = vec![0u64; count];
    let mut av0 = matrix.apply(&v0);
    let mut t0 = dot(&v0, &av0);
    let mut t1 = [0u64; WIDTH];
    let (mut w1i, mut w2i) = ([0u64; WIDTH], [0u64; WIDTH]);
    let mut g = [0u64; WIDTH];
    let mut mask = u64::MAX;

    // One block spans up to sixty-four dimensions, so the count is bounded;
    // the slack covers iterations where the selection takes fewer lanes.
    let ceiling = count / WIDTH + WIDTH + 16;
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
        if stranded != 0 && v1.iter().any(|word| word & stranded != 0) {
            return None;
        }
        let w0i = next_w0i;
        mask = next_mask;
        if mask == 0 {
            break;
        }

        // (20): X += V[i] Winv[i] V[i]ᵀ V[0].
        let step = mul_block(&v0, &mul(&w0i, &dot(&v0, &q)));
        xor_into(&mut x, &step);

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
        let mut squared = dot(&av0, &av0);
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
        let mut next = av0.clone();
        for slot in &mut next {
            *slot &= mask;
        }
        xor_into(&mut next, &mul_block(&v0, &d));
        xor_into(&mut next, &mul_block(&v1, &e));
        xor_into(&mut next, &mul_block(&v2, &f));

        v2 = std::mem::replace(&mut v1, std::mem::replace(&mut v0, next));
        av0 = matrix.apply(&v0);
        w2i = w1i;
        w1i = w0i;
        t1 = t0;
        t0 = dot(&v0, &av0);
    }

    // The kernel of A = MᵀM contains M's but is not equal to it: over GF(2) a
    // vector can be orthogonal to itself. So take the 128 candidates that came
    // out — the columns of X and of the last V — and ask a small elimination
    // which of their combinations M actually annihilates.
    let images = [matrix.forward(&x), matrix.forward(&v0)];
    let candidates: Vec<Vec<u64>> = (0..2 * WIDTH)
        .map(|index| lane(&images[index / WIDTH], index % WIDTH))
        .collect();
    let sources = [x, v0];

    let mut found = Vec::new();
    for combination in dense_null_space(&candidates, matrix.columns()) {
        let mut vector = vec![0u64; count.div_ceil(WIDTH)];
        for index in combination {
            xor_into(&mut vector, &lane(&sources[index / WIDTH], index % WIDTH));
        }
        let indices: Vec<usize> = (0..count)
            .filter(|&r| vector[r / WIDTH] >> (r % WIDTH) & 1 == 1)
            .collect();
        if indices.is_empty() {
            continue;
        }
        // Checked against the caller's own rows, not against anything this
        // module computed.
        let mut total = vec![0u64; columns.div_ceil(WIDTH)];
        for &index in &indices {
            xor_into(&mut total, &rows[index]);
        }
        if !columns.is_multiple_of(WIDTH) {
            let last = total.len() - 1;
            total[last] &= (1u64 << (columns % WIDTH)) - 1;
        }
        if !any(&total) {
            found.push(indices);
        }
    }
    (!found.is_empty()).then_some(found)
}

#[cfg(test)]
mod tests {
    use super::{
        block_lanczos_dependencies, borrow_two, dense_null_space, prune_singletons, words_for, WORD,
    };

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

    /// A deterministic `RandomSource` for the tests, so a failure reproduces.
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

    /// Every dependency Block Lanczos returns is a genuine one.
    ///
    /// This is the property that matters: the method is randomized and may
    /// find fewer than the full null space, but it must never return a set
    /// that is not dependent. `None` is an acceptable outcome and a wrong
    /// answer is not.
    #[test]
    fn block_lanczos_returns_only_genuine_dependencies() {
        for &(relations, columns, weight) in
            &[(160usize, 96usize, 8usize), (200, 128, 10), (300, 200, 12)]
        {
            let rows = sparse_rows(relations, columns, weight, 0x1234_5678);
            let mut rng = TestRng(0xdead_beef);
            if let Some(deps) = block_lanczos_dependencies(&rows, columns, &mut rng) {
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

    /// Whatever it finds lies inside the exact solver's null space.
    ///
    /// The comparison is on *rank*, not on count. `dense_null_space` returns a
    /// basis — `rows − rank` independent vectors — while Block Lanczos returns
    /// candidate combinations that need not be independent of one another, so
    /// it can hand back more sets than the null space has dimensions. An
    /// earlier version of this test compared the counts and failed on correct
    /// output: 114 genuine dependencies spanning a 92-dimensional space.
    #[test]
    fn block_lanczos_agrees_with_the_exact_solver() {
        for &(relations, columns, weight) in &[(160usize, 96usize, 8usize), (192, 120, 9)] {
            let rows = sparse_rows(relations, columns, weight, 0x0bad_c0de);
            let exact = dense_null_space(&rows, columns).len();
            let mut rng = TestRng(0x5eed_1234);
            if let Some(deps) = block_lanczos_dependencies(&rows, columns, &mut rng) {
                for dep in &deps {
                    assert!(sums_to_zero(&rows, columns, dep));
                }
                let spanned = rank_of(&deps, relations);
                assert!(
                    spanned <= exact,
                    "spans {spanned} dimensions where the null space has {exact}"
                );
            }
        }
    }

    /// Degenerate shapes are refused rather than guessed at.
    #[test]
    fn block_lanczos_refuses_the_degenerate_shapes() {
        let mut rng = TestRng(1);
        assert!(block_lanczos_dependencies(&[], 8, &mut rng).is_none());
        assert!(block_lanczos_dependencies(&[vec![0u64]], 0, &mut rng).is_none());
    }

    /// The generator is the caller's: the same source gives the same answer,
    /// and a different one is still only ever asked for genuine dependencies.
    #[test]
    fn block_lanczos_is_driven_by_the_callers_generator() {
        let rows = sparse_rows(160, 96, 8, 0xfeed_face);
        let first = block_lanczos_dependencies(&rows, 96, &mut TestRng(7));
        let again = block_lanczos_dependencies(&rows, 96, &mut TestRng(7));
        assert_eq!(first, again, "the same source must give the same answer");

        if let Some(deps) = block_lanczos_dependencies(&rows, 96, &mut TestRng(99)) {
            for dep in &deps {
                assert!(sums_to_zero(&rows, 96, dep));
            }
        }
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
