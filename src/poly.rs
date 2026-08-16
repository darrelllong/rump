//! Dense univariate polynomials over ℤ and over ℤ/pℤ.
//!
//! Two concrete named types, in the crate's style — a documented
//! representation rather than a tower of coefficient-ring traits:
//! [`PolyZ`] over the integers and [`PolyModP`] over a fixed modulus. They
//! are the substrate the general number field sieve is built on (the
//! algebraic side is a degree-5 or -6 polynomial over ℤ and its behaviour
//! modulo small primes), and more broadly the polynomial layer a computer
//! algebra surface provides.
//!
//! Both store coefficients low-to-high (`coeffs[i]` multiplies `xⁱ`) and
//! stay normalized: no trailing zero coefficient, so the zero polynomial is
//! the empty coefficient list and the degree is the last index.

use crate::bigint::{BigInt, BigUint, Sign};
use crate::number_theory;

/// Coefficient count at or above which [`PolyZ`] multiplication splits
/// Karatsuba-style instead of running the schoolbook convolution.
///
/// The recurrence is the classical one — three half-length convolutions
/// instead of four, `T(n) = 3·T(n/2) + O(n)` coefficient operations, so
/// `O(n^{lg 3}) ≈ O(n^{1.585})` against schoolbook's `O(n²)`. The crossover
/// is not a property of `n` alone: the split trades coefficient
/// *multiplications* for coefficient *additions*, both multiprecision, so
/// it pays where a multiplication is dear relative to an addition. Over ℤ
/// it is, increasingly so as the recursion proceeds — the coefficients grow
/// as partial products accumulate while the additions stay linear in that
/// same width.
///
/// Measured by `poly_karatsuba_crossover_timing` (run with `--ignored`),
/// which calibrates its repetition count rather than assuming one, over a
/// sweep of operand *ratios*: the split takes half the longer operand, so
/// a shorter one that barely reaches past that point saves almost nothing
/// while paying the whole recombination.
///
/// Medians of two runs, five passes each, alternating which side is timed
/// first:
///
/// | shorter operand | 1:1 | 5:4 | 3:2 | (2s−1):s |
/// |---|---|---|---|---|
/// | 32 | +12% | +8% | +1% | **−8%** |
/// | 64 | +18% | +11% | +5% | **−4%** |
/// | 96 | +21% | +14% | +7% | **−3%** |
/// | 128 | +21% | +15% | +24% | +17% |
/// | 192 | +37% | +29% | +26% | +30% |
/// | 256 | +39% | +30% | +38% | +37% |
/// | 384 | +51% | +45% | +40% | +49% |
/// | 512 | +52% | +46% | +49% | +52% |
/// | 768 | +62% | +58% | +52% | +61% |
///
/// So 96 is where the *balanced* split turns clearly profitable, and it is
/// the threshold; it is not where every shape does. At 96 the near-2:1
/// column still trails by 3%, ten of ten passes negative across the two
/// runs, and does not turn until 128 — hence
/// [`POLY_KARATSUBA_ANY_RATIO_Z`] and the two-clause gate in
/// [`poly_split_admitted`], which both rings now share — the structure the
/// modular side already had, for the same reason.
///
/// Two earlier revisions of this table were wrong in ways worth recording,
/// since both were caught by review rather than by a test. The first said
/// 64 and cited +19% at `(2s−1):s`, a figure from a four-repetition sample
/// that did not survive calibrated repetitions. The second said +22% and
/// +16% for the 96 row's 3:2 and 2:1 cells — numbers transcribed from the
/// *modular* table's 512 row, which inverted the sign of the one cell that
/// decides whether a ratio clause is needed at all.
const POLY_KARATSUBA_THRESHOLD_Z: usize = 96;

/// The size at which every operand ratio the dispatcher can reach becomes
/// profitable for the integer split, so the balance clause lifts.
///
/// From the table on [`POLY_KARATSUBA_THRESHOLD_Z`]: the near-2:1 column
/// reads −3% at 96 and +17% at 128, and every column is positive from 128
/// up. The 128 row is not monotone in the ratio — 3:2 (+24%) and 2:1
/// (+17%) both beat 5:4 (+15%) — because the split point is half the
/// *longer* operand and lands differently against each length, so the row
/// is read for its sign rather than its ordering.
///
/// End to end, against a build that gates on length alone: `PolyZ::mul` on
/// a 96×191 pair is 1.19× faster for declining the split, and the shapes
/// the clause still admits (128×255, 192×383, 256×511) are unchanged to
/// within a point. Medians of six and seven runs.
const POLY_KARATSUBA_ANY_RATIO_Z: usize = 128;

/// The same threshold for [`PolyModP`], with a balance rule attached.
///
/// Two things move it out. The coefficients do not grow — every one stays
/// reduced below the modulus — so the multiplication a split saves never
/// becomes dear relative to the additions it adds. And each recombination
/// addition is a *modular* one, a compare and a conditional subtraction
/// over a fresh allocation, where the ℤ case adds in place. Against that,
/// the schoolbook path here is unusually cheap: it defers its reductions
/// (see [`convolve_schoolbook_modp`]), so its inner loop is a bare
/// multiply-accumulate.
///
/// Measured at a 20-bit and a 256-bit modulus; the two columns are
/// averaged here, and they agree to within about four points (the widest
/// gap is the 64 row's 2:1 cell, −26% against −22%):
///
/// | shorter operand | 1:1 | 5:4 | 3:2 | (2s−1):s |
/// |---|---|---|---|---|
/// | 32 | −30% | −32% | −35% | −42% |
/// | 64 | −7% | −10% | −14% | −24% |
/// | 96 | +3% | −1% | −7% | −17% |
/// | 128 | +8% | +3% | −3% | −13% |
/// | 192 | +12% | +7% | +9% | +4% |
/// | 256 | +21% | +17% | +14% | +12% |
/// | 384 | +28% | +22% | +24% | +23% |
/// | 512 | +36% | +31% | +30% | +33% |
/// | 768 | +43% | +37% | +38% | +40% |
///
/// So the admissible ratio *widens with size*, which a fixed ratio gate
/// cannot express: 1:1 turns positive at 96 and 5:4 at 128, while 3:2 and
/// `(2s−1):s` do not until 192. [`poly_split_admitted`] encodes both
/// clauses.
const POLY_KARATSUBA_THRESHOLD_MODP: usize = 128;

/// The size at which every operand ratio the dispatcher can reach becomes
/// profitable for the modular split, so the balance clause lifts.
///
/// 192, from the table on [`POLY_KARATSUBA_THRESHOLD_MODP`]: +12%
/// balanced, +7% at 5:4, +9% at 3:2 and +4% at `(2s−1):s`, with all twenty
/// passes of the two marginal cells positive across two runs. An earlier
/// revision set this to 512 — a size at which the clause had been
/// *measured* rather than the size at which it turns — and so refused
/// splits that measure +23% and +24% at 384.
///
/// End to end against that revision, `PolyModP::mul` gains 1.31× at 384×576
/// and 1.26× at 384×767, 1.15× and 1.12× at 256, and 1.08× at 192×288;
/// 192×383 is the marginal cell and does not move outside noise, which is
/// what a threshold placed exactly at the turn should look like. Shapes
/// already admitted at 512 and above are unchanged. Medians of six and
/// seven runs.
const POLY_KARATSUBA_ANY_RATIO_MODP: usize = 192;

/// Whether an operand shape admits the Karatsuba split, given the size
/// threshold and the any-ratio threshold for its coefficient ring.
///
/// Two measured clauses, because one alone is wrong. Below `any_ratio`
/// only near-balanced shapes pay — the split point is half the *longer*
/// operand, so at a ratio approaching 2:1 the shorter contributes almost
/// nothing above it, and the three sub-convolutions cost about `2s² − 2s`
/// coefficient products against schoolbook's `2s² − s`, a vanishing saving
/// against the whole recombination. From `any_ratio` up, every ratio the
/// split can reach wins, and holding the ratio clause above that size
/// refuses shapes measured 20% and more ahead.
///
/// The 5:4 cut-off in the balance clause is where the ratio columns turn:
/// 5:4 is positive from the size threshold in both rings, 3:2 is not.
fn poly_split_admitted(short: usize, long: usize, threshold: usize, any_ratio: usize) -> bool {
    if short < threshold {
        return false;
    }
    short >= any_ratio || 4 * long <= 5 * short
}

/// The coefficient count at or above which a modular *squaring* splits.
///
/// Its own constant, because a squaring's split is not a product's: the
/// three sub-problems are squares, so both sides of the comparison shift
/// and the curves do not coincide. Measured against the cross-terms-once
/// square at a 20-bit and a 256-bit modulus: −31%/−23% at 64 coefficients,
/// −14%/−10% at 96, −5%/−3% at 128, +3%/+5% at 192, +8%/+8% at 256,
/// +17%/+17% at 384, +23%/+23% at 512, +33%/+32% at 768. The turn is at
/// 192 — above the product's balanced crossover of 128, not below it,
/// because a square already saves the cross terms and so leaves the split
/// less to win. An earlier revision borrowed the product's constant on the
/// assumption the two curves matched.
const POLY_SQUARE_SPLIT_THRESHOLD_MODP: usize = 192;

/// Coefficient multiplications the Karatsuba split would perform on
/// *dense* operands of these two lengths — the recursion itself, counted,
/// rather than a closed form fitted to it.
///
/// It mirrors [`convolve_z`]'s and [`convolve_modp`]'s own dispatch: the
/// same admission gate, the same `split = max/2`, and the same three
/// sub-shapes — `(split, split)`, `(a−split, b−split)`, and the two sums'
/// `(max(split, a−split), max(split, b−split))`. Counting is worth the few
/// hundred integer operations because a closed form is not close enough. The
/// obvious one, `3^d · ⌈long/2^d⌉²`, charges the split for a `long × long`
/// problem it never solves and halves until the *longer* side falls below
/// the threshold where the real recursion stops as soon as the *shorter*
/// one does; across the shapes in the tables above it runs from 0.88× to
/// 1.50× of the true count, which moves the break-even density by half and
/// pins it at 1.0 over a wide band of admitted shapes.
///
/// Density is deliberately not consulted here. This is the dense reference
/// the density decision is *made against*, so consulting it would be
/// circular.
///
/// The count is not monotone in the lengths, and cannot be: at 96 and 191
/// coefficients the halvings land such that one more level of splitting is
/// taken, so a slightly longer operand genuinely does more or less work
/// than its neighbour. The trend across sizes is what falls; the
/// neighbour-to-neighbour steps wobble by a few percent, and that is a
/// property of the algorithm rather than of this estimate.
///
/// # Panics
///
/// Panics if `threshold` is below two. Halving has a fixed point at one, so
/// a threshold of zero or one would not terminate.
fn karatsuba_products_estimate(
    a_len: usize,
    b_len: usize,
    threshold: usize,
    any_ratio: usize,
) -> usize {
    assert!(
        threshold >= 2,
        "a split threshold below two never terminates"
    );
    let dense = a_len.saturating_mul(b_len);
    if !poly_split_admitted(a_len.min(b_len), a_len.max(b_len), threshold, any_ratio) {
        return dense;
    }
    let split = a_len.max(b_len) / 2;
    if a_len <= split || b_len <= split {
        return dense;
    }
    let (a_high, b_high) = (a_len - split, b_len - split);
    karatsuba_products_estimate(split, split, threshold, any_ratio)
        .saturating_add(karatsuba_products_estimate(
            a_high, b_high, threshold, any_ratio,
        ))
        .saturating_add(karatsuba_products_estimate(
            split.max(a_high),
            split.max(b_high),
            threshold,
            any_ratio,
        ))
}

/// The same count for a *square*, whose recursion is its own: three
/// sub-squares rather than three sub-products, split at half the single
/// length, and gated on length alone — a square has no operand ratio.
///
/// # Panics
///
/// Panics if `threshold` is below two, for the reason given on
/// [`karatsuba_products_estimate`].
fn karatsuba_square_products_estimate(len: usize, threshold: usize) -> usize {
    assert!(
        threshold >= 2,
        "a split threshold below two never terminates"
    );
    if len < threshold {
        return len.saturating_mul(len);
    }
    let split = len / 2;
    let high = len - split;
    karatsuba_square_products_estimate(split, threshold)
        .saturating_add(karatsuba_square_products_estimate(high, threshold))
        .saturating_add(karatsuba_square_products_estimate(
            split.max(high),
            threshold,
        ))
}

/// Whether the operands are dense enough for the split to be worth taking.
///
/// The schoolbook convolution skips a zero coefficient's whole inner pass
/// and puts the sparser operand outside, so its real cost is
/// `nnz(sparser) · len(denser)` coefficient products, not `len · len`. The
/// split destroys that saving: `a₀ + a₁` is dense even when `a` is not, so
/// every recursive sub-convolution pays full freight. Dispatching on
/// length alone made `(x^{n−1} + 1) · dense` **eight times slower** than
/// the schoolbook path it replaced at 1024 coefficients, and half as fast
/// at 64.
///
/// So the two counts are compared directly, against
/// [`karatsuba_products_estimate`]. The density a shape must reach is that
/// count over `len · len`, which *trends down as the operands grow*
/// because the split's exponent is the better one: three quarters at 96
/// coefficients, about three eighths at 512, under a quarter at 2048. A
/// fixed cut cannot express that, and the fixed three-quarters cut this
/// replaces — correct only at the crossover, where the two agree exactly —
/// put a 2.2× cliff at 75% density on 2048-coefficient operands, so that
/// reducing an operand's non-zero count by a quarter *doubled* the time it
/// took.
///
/// The estimate is floored at the dense schoolbook count so that a fully
/// dense pair is never refused on density grounds; whether it should split
/// at all is the length and ratio gate's question, not this one's. With the
/// counted estimate the floor almost never binds — with the closed form it
/// bound over a band of thousands of admitted shapes and cost 1.18× for a
/// single zero coefficient, which is how the closed form was caught.
///
/// Measured against the fixed cut, `PolyZ::mul` on 2048-coefficient
/// operands gains 2.12× at 74% density, 1.72× at 60% and 1.27× at 76%;
/// `PolyModP::mul` gains 1.40× at 74% on 1024. Densities on the far side
/// of the old cut in either direction — 80% and above, 30% and below, and
/// the two-term sparse shapes the fixed cut was introduced to protect —
/// are unchanged to within a point, so nothing was traded for it. Medians
/// of six and seven runs.
/// Whether a *square* is dense enough for the split to be worth taking.
///
/// The same argument as [`poly_split_dense_enough`], with the square's own
/// counts. The cross-terms-once schoolbook skips a zero coefficient in
/// *both* loops, so its cost is `nnz(nnz+1)/2` rather than the product's
/// one-sided `nnz · len`; the split's cost carries the same factor of one
/// half at every leaf, so the two are compared without it.
///
/// A square is where the omission bites hardest, because squaring is what
/// [`PolyModP::pow_mod`] does at every step and the ladder's early values
/// are the sparsest there are — `x`, `x²`, `x⁴` — staying sparse until
/// reduction densifies them. Splitting a two-term value of 1024
/// coefficients measured 13× slower than the schoolbook square it
/// replaced, 10× at 512, 9× at 384 and 3× at 192: the split's
/// recombination is linear per node and there are `3^d` nodes, so it pays
/// `O(n^{lg 3})` for an answer schoolbook reaches in `O(nnz²)`.
fn poly_square_split_dense_enough(nonzero: usize, len: usize) -> bool {
    let schoolbook = nonzero.saturating_mul(nonzero);
    let dense = len.saturating_mul(len);
    schoolbook
        >= karatsuba_square_products_estimate(len, POLY_SQUARE_SPLIT_THRESHOLD_MODP).min(dense)
}

fn poly_split_dense_enough(
    a_nonzero: usize,
    a_len: usize,
    b_nonzero: usize,
    b_len: usize,
    threshold: usize,
    any_ratio: usize,
) -> bool {
    let (outer_nonzero, outer_len, inner_len) = if a_nonzero <= b_nonzero {
        (a_nonzero, a_len, b_len)
    } else {
        (b_nonzero, b_len, a_len)
    };
    let schoolbook = outer_nonzero.saturating_mul(inner_len);
    let dense = outer_len.saturating_mul(inner_len);
    let split = karatsuba_products_estimate(a_len, b_len, threshold, any_ratio).min(dense);
    schoolbook >= split
}

/// The widest a level of the Hensel lift, or its final answer, is allowed
/// to become before [`PolyZ::roots_mod_prime_power`] gives up.
///
/// Two things can widen a lift without bound. A root where the derivative
/// vanishes splits into `p` candidates at once — all of which may die at
/// the next level, so a wide intermediate does not imply a wide answer —
/// and a content divisible by `pᵛ` multiplies the final count by `pᵛ`.
/// Neither is bounded by the degree, so neither is bounded by anything the
/// caller can see from the polynomial alone.
///
/// The cap is on the count, which is what actually has to fit in memory,
/// not on the prime. An earlier revision guarded only `p ≥ 2⁶⁴`, on the
/// reasoning that a larger prime "has more lifts than can be listed"; that
/// left every prime between `2³²` and `2⁶⁴` to exhaust memory silently
/// while claiming in its documentation to have refused them. The consumer
/// this routine was written for caps its own lift at 4096 for the same
/// reason, and drops the tail; here the caller is told instead, because a
/// root-finder that silently returns some of the roots is worse than one
/// that refuses.
pub const MAX_ROOT_LEVEL: usize = 1 << 20;

/// The number of non-zero coefficients — one linear pass against the
/// quadratic work the answer decides.
fn count_nonzero<T>(coeffs: &[T], is_zero: impl Fn(&T) -> bool) -> usize {
    coeffs.iter().filter(|c| !is_zero(c)).count()
}

/// A univariate polynomial over ℤ, coefficients low-to-high, normalized to
/// drop trailing zeros.
///
/// Normalization is a type invariant, re-established by every constructor
/// and every operation that can cancel a leading term. It is what makes
/// [`Self::degree`] the last index and the derived [`PartialEq`] a decision
/// procedure for polynomial equality: without it `[1, 0]` and `[1]` would be
/// the same polynomial held in two unequal representations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolyZ {
    coeffs: Vec<BigInt>,
}

impl PolyZ {
    /// Build from coefficients low-to-high, normalizing away trailing
    /// zeros. `[3, 0, 2]` is `2x² + 3`.
    #[must_use]
    pub fn new(coeffs: Vec<BigInt>) -> Self {
        let mut poly = Self { coeffs };
        poly.normalize();
        poly
    }

    /// The zero polynomial.
    #[must_use]
    pub fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }

    /// The constant polynomial `c`.
    #[must_use]
    pub fn constant(c: BigInt) -> Self {
        Self::new(vec![c])
    }

    /// Build from small integer coefficients low-to-high, for callers with
    /// literal polynomials.
    #[must_use]
    pub fn from_i64_slice(coeffs: &[i64]) -> Self {
        Self::new(coeffs.iter().map(|&c| BigInt::from_i64(c)).collect())
    }

    /// Restore the trailing-zero invariant by popping high-order zeros. The
    /// loop runs from the top, so it stops at the first non-zero coefficient
    /// and interior zeros are untouched.
    fn normalize(&mut self) {
        while self.coeffs.last().is_some_and(BigInt::is_zero) {
            self.coeffs.pop();
        }
    }

    /// Whether this is the zero polynomial.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// The degree, or `None` for the zero polynomial (whose degree is
    /// conventionally `−∞`).
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        self.coeffs.len().checked_sub(1)
    }

    /// The coefficients, low-to-high; empty for the zero polynomial.
    #[must_use]
    pub fn coefficients(&self) -> &[BigInt] {
        &self.coeffs
    }

    /// The leading coefficient, or zero for the zero polynomial.
    #[must_use]
    pub fn leading_coefficient(&self) -> BigInt {
        self.coeffs.last().cloned().unwrap_or_else(BigInt::zero)
    }

    /// `self + other`, added coefficient-wise (shorter operand zero-extended).
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(BigInt::zero);
            let b = other.coeffs.get(i).cloned().unwrap_or_else(BigInt::zero);
            coeffs.push(a.add_ref(&b));
        }
        Self::new(coeffs)
    }

    /// `self − other`, subtracted coefficient-wise (shorter operand
    /// zero-extended). Equal leading terms cancel, so the result is
    /// renormalized and the degree drops accordingly.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(BigInt::zero);
            let b = other.coeffs.get(i).cloned().unwrap_or_else(BigInt::zero);
            coeffs.push(a.sub_ref(&b));
        }
        Self::new(coeffs)
    }

    /// `−self`, negated coefficient-wise. Negation cannot create a trailing
    /// zero, so the normalized form carries over and no renormalization pass
    /// is needed.
    #[must_use]
    pub fn negated(&self) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(BigInt::negated).collect(),
        }
    }

    /// `self · other`, by the schoolbook coefficient convolution below
    /// 96 coefficients (`POLY_KARATSUBA_THRESHOLD_Z`, a measured crossover
    /// taken over operand ratios, not balanced pairs alone) and by the
    /// Karatsuba split above it — which is additionally refused to a
    /// sparse operand, whose zeros the schoolbook path skips and the split
    /// does not. The result buffer is sized `deg self + deg other + 1`
    /// because ℤ is an integral domain — the leading coefficients cannot
    /// cancel — so the product of two non-zero polynomials has exactly that
    /// degree, and no renormalization can be needed. That is also why the
    /// split needs no sign care that the integer Karatsuba does: polynomial
    /// coefficients carry no borrows between positions, so the middle term
    /// is a plain coefficient-wise difference.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        Self {
            coeffs: convolve_z(&self.coeffs, &other.coeffs),
        }
    }

    /// `self · c` for a scalar. The units ±1 short-circuit to a clone or a
    /// coefficient-wise negation: the division loops scale by a leading
    /// coefficient that is very often ±1 (every monic divisor), and a
    /// multiplication by a unit across the whole coefficient list would be
    /// pure waste.
    #[must_use]
    pub fn scale(&self, c: &BigInt) -> Self {
        if c.is_zero() {
            return Self::zero();
        }
        if c.is_one() {
            return self.clone();
        }
        // −1 is detected from the parts rather than via `negated()`, which
        // would clone the magnitude on every call that is *not* −1.
        if c.sign() == Sign::Negative && c.magnitude().is_one() {
            return self.negated();
        }
        Self::new(self.coeffs.iter().map(|a| a.mul_ref(c)).collect())
    }

    /// Evaluate at `x` by Horner's method: fold `acc ← acc·x + cᵢ` from the
    /// leading coefficient down. No power of `x` is ever materialized, so
    /// the evaluation is one multiplication and one addition per
    /// coefficient.
    #[must_use]
    pub fn evaluate(&self, x: &BigInt) -> BigInt {
        let mut acc = BigInt::zero();
        for coeff in self.coeffs.iter().rev() {
            acc = acc.mul_ref(x).add_ref(coeff);
        }
        acc
    }

    /// The formal derivative `∑ i·cᵢ·xⁱ⁻¹` — formal in that it is defined by
    /// the coefficient rule rather than by a limit, so it exists over any
    /// coefficient ring and obeys the product rule algebraically. Computed
    /// by dropping the constant term and scaling each remaining coefficient
    /// by its old index; a constant differentiates to zero.
    #[must_use]
    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero();
        }
        let coeffs = self.coeffs[1..]
            .iter()
            .enumerate()
            .map(|(i, c)| c.mul_ref(&BigInt::from_i64(i as i64 + 1)))
            .collect();
        Self::new(coeffs)
    }

    /// The content: the gcd of the coefficients, non-negative, zero for the
    /// zero polynomial. Accumulated by folding `BigInt::gcd` over the
    /// coefficients from a seed of zero, which is the identity for gcd.
    #[must_use]
    pub fn content(&self) -> BigInt {
        let mut g = BigInt::zero();
        for c in &self.coeffs {
            g = g.gcd(c);
        }
        g
    }

    /// The primitive part: `self` with its content divided out, so the
    /// coefficients have gcd 1. Every coefficient is divisible by the
    /// content by definition, so the per-coefficient division is exact and
    /// the result stays in ℤ. [`Self::content`] is non-negative, so the sign
    /// stays with the primitive part and its leading coefficient keeps
    /// `self`'s sign — the convention here, not the one that forces a
    /// positive leading coefficient. The zero polynomial has content zero
    /// and is its own primitive part.
    #[must_use]
    pub fn primitive_part(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let content = self.content();
        Self::new(self.coeffs.iter().map(|c| c.div_exact(&content)).collect())
    }

    /// Pseudo-division: returns `(quotient, remainder)` satisfying
    /// `ℓ·self = quotient·divisor + remainder` with `deg remainder <
    /// deg divisor` — staying in ℤ by premultiplying by the leading
    /// coefficient rather than dividing by it (Knuth, *TAOCP* vol. 2,
    /// §4.6.1, Algorithm R). When `deg self ≥ deg divisor`, `ℓ =
    /// lc(divisor)` raised to `deg self − deg divisor + 1`; when
    /// `deg self < deg divisor` there is nothing to divide, the result is
    /// `(zero, self)`, and `ℓ = 1` (the formula's exponent would be
    /// non-positive there and does not apply).
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is the zero polynomial.
    #[must_use]
    pub fn pseudo_div_rem(&self, divisor: &Self) -> (Self, Self) {
        assert!(!divisor.is_zero(), "pseudo-division by the zero polynomial");
        let Some(divisor_degree) = divisor.degree() else {
            unreachable!("non-zero divisor has a degree")
        };
        if self.degree().is_none_or(|d| d < divisor_degree) {
            // Nothing to divide. The exponent deg self − deg divisor + 1 is
            // non-positive here (and undefined for the zero polynomial), so
            // the identity is taken at ℓ = 1: self = 0·divisor + self.
            return (Self::zero(), self.clone());
        }
        let self_degree = self.degree().expect("degree checked above");
        let lc = divisor.leading_coefficient();
        let lc_is_one = lc.is_one();
        let mut rem = self.coeffs.clone();
        let mut quotient = vec![BigInt::zero(); self_degree - divisor_degree + 1];
        // Repeatedly cancel the remainder's leading term; each step scales
        // the whole working state by lc so the coefficients stay integral.
        // The invariant after t steps is lc^t·self = quotient·divisor +
        // remainder, and each step strictly lowers deg remainder (the two
        // leading terms are both rem_lc·lc·x^rem_degree and cancel), so the
        // loop runs at most deg self − deg divisor + 1 times. `top` tracks
        // the remainder's degree by hand so the subtraction runs in place
        // over the window it touches, `[shift, top]`, instead of
        // materializing a shifted, scaled copy of the divisor at full
        // remainder length.
        let mut steps = 0usize;
        let mut top = self_degree;
        loop {
            let shift = top - divisor_degree;
            let rem_lc = rem[top].clone();
            // remainder ← lc·remainder − rem_lc·xˢʰⁱᶠᵗ·divisor. A monic
            // divisor skips the scaling outright — it would multiply every
            // live coefficient of both remainder and quotient by unity.
            if !lc_is_one {
                for c in rem.iter_mut() {
                    if !c.is_zero() {
                        *c = c.mul_ref(&lc);
                    }
                }
                for q in quotient.iter_mut() {
                    if !q.is_zero() {
                        *q = q.mul_ref(&lc);
                    }
                }
            }
            for (k, d) in divisor.coeffs.iter().enumerate() {
                if !d.is_zero() {
                    rem[shift + k] = rem[shift + k].sub_ref(&rem_lc.mul_ref(d));
                }
            }
            debug_assert!(rem[top].is_zero(), "leading terms cancel by construction");
            // quotient accumulates rem_lc at the shift position, itself
            // scaled by lc (above) for the steps still to come.
            quotient[shift] = quotient[shift].add_ref(&rem_lc);
            steps += 1;
            // Step down to the next non-zero coefficient — the new degree —
            // and stop once it falls below the divisor's or the remainder
            // vanishes.
            while top > 0 && rem[top].is_zero() {
                top -= 1;
            }
            if rem[top].is_zero() || top < divisor_degree {
                break;
            }
        }
        // The identity carries lc^(steps) on the left; the required
        // exponent is deg self − deg divisor + 1, and steps ≤ that (fewer
        // when the remainder degree falls by more than one, or reaches zero
        // early). Scale quotient and remainder up to the full exponent so
        // the documented ℓ holds regardless. For ℓ = 1 the scaling is the
        // identity and is skipped.
        let required = self_degree - divisor_degree + 1;
        let mut quotient = Self::new(quotient);
        let mut remainder = Self::new(rem);
        if !lc_is_one {
            for _ in steps..required {
                quotient = quotient.scale(&lc);
                remainder = remainder.scale(&lc);
            }
        }
        (quotient, remainder)
    }

    /// Exact division over ℤ: `Some((quotient, remainder))` satisfying
    /// `self = quotient·divisor + remainder` with `deg remainder <
    /// deg divisor`.
    ///
    /// Why it is fallible: ℤ is not a field, so unlike division over `𝔽_p` a
    /// quotient with integer coefficients need not exist. Schoolbook long
    /// division cancels the remainder's leading term each step by dividing its
    /// coefficient by the divisor's leading coefficient; that division is
    /// exact only when the latter divides the former. It always does for a
    /// monic divisor (leading coefficient `±1`), so those never fail. When a
    /// step does not divide evenly there is no integer quotient and this
    /// returns `None` — reach for [`Self::pseudo_div_rem`], which sidesteps the
    /// obstruction by premultiplying (`ℓ·self = quotient·divisor + remainder`)
    /// and so is always defined; it is what the resultant path uses.
    ///
    /// Deciding and dividing are one operation, not two: each step calls
    /// `BigInt::div_exact_checked`, a single Knuth Algorithm D division
    /// (*TAOCP* vol. 2, §4.3.1) that yields the quotient only when the
    /// remainder vanishes. Testing divisibility and then taking the quotient
    /// separately would run Algorithm D twice per step.
    ///
    /// The quotient is unique when one exists — the leading coefficient of a
    /// non-zero divisor is not a zero divisor in ℤ, so the coefficients are
    /// forced in order from the top — which is why a single failing step is
    /// conclusive rather than an artefact of this particular schedule.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is the zero polynomial.
    #[must_use]
    pub fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        assert!(!divisor.is_zero(), "division by the zero polynomial");
        let divisor_degree = divisor.degree().expect("non-zero divisor has a degree");
        if self.degree().is_none_or(|d| d < divisor_degree) {
            // Nothing to divide: self = 0·divisor + self.
            return Some((Self::zero(), self.clone()));
        }
        let self_degree = self.degree().expect("degree checked above");
        let lc = divisor.leading_coefficient();
        let mut rem = self.coeffs.clone();
        let mut quotient = vec![BigInt::zero(); self_degree - divisor_degree + 1];
        // Cancel the remainder's leading term each step, dividing exactly by
        // the divisor's leading coefficient; a step that does not divide has
        // no integer quotient, so the whole division fails. `top` tracks the
        // remainder's degree by hand so each step subtracts in place over
        // the window it touches, `[shift, top]`, instead of materializing a
        // shifted, scaled copy of the divisor at full remainder length.
        let mut top = self_degree;
        loop {
            // One division decides and delivers: an indivisible leading
            // coefficient means no integer quotient exists, and otherwise the
            // same Algorithm D call yields the coefficient.
            let q_coeff = rem[top].div_exact_checked(&lc)?;
            let shift = top - divisor_degree;
            // rem ← rem − q_coeff·xˢʰⁱᶠᵗ·divisor; the leading terms cancel
            // exactly because the division above was exact.
            for (k, d) in divisor.coeffs.iter().enumerate() {
                if !d.is_zero() {
                    rem[shift + k] = rem[shift + k].sub_ref(&q_coeff.mul_ref(d));
                }
            }
            debug_assert!(rem[top].is_zero(), "leading term cancels by construction");
            quotient[shift] = q_coeff;
            // Step down to the next non-zero coefficient — the new degree —
            // and stop once it falls below the divisor's or the remainder
            // vanishes.
            while top > 0 && rem[top].is_zero() {
                top -= 1;
            }
            if rem[top].is_zero() || top < divisor_degree {
                break;
            }
        }
        Some((Self::new(quotient), Self::new(rem)))
    }

    /// The resultant `res(self, other)` — the determinant of the two
    /// polynomials' Sylvester matrix, zero exactly when they share a
    /// non-constant factor over ℚ. Computed by Bareiss fraction-free
    /// elimination, which keeps every intermediate entry an integer minor
    /// determinant, so no rational arithmetic is needed (Bareiss,
    /// *Sylvester's identity and multistep integer-preserving Gaussian
    /// elimination*, Math. Comp. 22 (1968), 565–578; Cohen, §3.3.1).
    ///
    /// Conventions at the degenerate ends: the resultant of two non-zero
    /// constants is `1`; `res(f, c)` for a non-zero constant `c` is
    /// `c^deg f`; and the resultant is `0` if either argument is the zero
    /// polynomial.
    #[must_use]
    pub fn resultant(&self, other: &Self) -> BigInt {
        if self.is_zero() || other.is_zero() {
            return BigInt::zero();
        }
        let m = self.degree().expect("non-zero");
        let n = other.degree().expect("non-zero");
        if m == 0 && n == 0 {
            return BigInt::one();
        }
        if n == 0 {
            // res(f, c) = c^deg f.
            return other.leading_coefficient().pow_u64(m as u64);
        }
        if m == 0 {
            return self.leading_coefficient().pow_u64(n as u64);
        }
        let matrix = sylvester_matrix(self, other);
        bareiss_determinant(matrix)
    }

    /// The discriminant `disc(self) = (−1)^(d(d−1)/2) · res(self, self') /
    /// lc(self)`, where `d = deg self`. Zero exactly when `self` has a
    /// repeated factor over ℚ, since the repeated factor is then common to
    /// `self` and `self'` and the resultant detects it.
    ///
    /// The division by the leading coefficient is exact for an integer
    /// polynomial: `res(f, f') = lc(f)·disc(f)` with `disc` a polynomial in
    /// the coefficients with integer coefficients (Cohen, §3.3.2). Over ℤ
    /// the derivative of a polynomial of degree `d ≥ 1` is non-zero, so the
    /// resultant call is never the degenerate `res(f, 0) = 0`; at `d = 1`
    /// it takes the constant-argument branch and the result is `1`. The
    /// discriminant of a constant or the zero polynomial is `0` by
    /// convention.
    #[must_use]
    pub fn discriminant(&self) -> BigInt {
        let Some(d) = self.degree() else {
            return BigInt::zero();
        };
        if d == 0 {
            return BigInt::zero();
        }
        let res = self.resultant(&self.derivative());
        // sign = (−1)^(d(d−1)/2)
        let signed = if (d * (d - 1) / 2) % 2 == 0 {
            res
        } else {
            res.negated()
        };
        signed.div_exact(&self.leading_coefficient())
    }

    /// The balanced base-`m` expansion of `n`: the polynomial
    /// `c₀ + c₁x + ⋯ + c_d xᵈ` with `n = Σ cₖ mᵏ` and every digit below the
    /// top one in the symmetric range `(−m/2, m/2]`.
    ///
    /// The ordinary base-`m` expansion takes digits in `[0, m)`; the balanced
    /// one takes the representative of least absolute value instead, which
    /// halves the coefficient bound at no cost. That matters wherever the
    /// polynomial is later evaluated somewhere other than `m` and the size of
    /// the result is what one is paying for — the number-field sieve's
    /// polynomial selection is the standard example, where `f(m) = n` by
    /// construction and the norms one wants small are governed by
    /// `max |cₖ|`.
    ///
    /// Only the first `degree` digits are reduced; the last coefficient
    /// carries whatever remains, so it obeys no bound and the identity
    /// `Σ cₖ mᵏ = n` holds exactly for every `degree`. Choosing `degree` too
    /// small therefore does not lose information, it merely puts all of `n`
    /// into the top coefficient. The result is trimmed like any other
    /// polynomial, so a leading digit that lands on zero lowers the degree.
    ///
    /// The digit recurrence is the schoolbook one with the balanced choice:
    /// `cₖ = symrem(rₖ, m)` and `r_{k+1} = (rₖ − cₖ)/m`, where the division is
    /// exact because `rₖ − cₖ ≡ 0 (mod m)`.
    ///
    /// # Panics
    ///
    /// Panics if `base` is less than two.
    #[must_use]
    pub fn balanced_base_expansion(n: &BigInt, base: &BigUint, degree: usize) -> Self {
        assert!(
            *base >= BigUint::from_u64(2),
            "a base expansion needs a base of at least two"
        );
        let signed_base = BigInt::from_biguint(base.clone());
        let mut coeffs = Vec::with_capacity(degree + 1);
        let mut remaining = n.clone();
        for _ in 0..degree {
            let digit = remaining.symmetric_remainder(base);
            let carried = remaining.sub_ref(&digit);
            let (quotient, check) = carried.div_rem(&signed_base);
            debug_assert!(
                check.is_zero(),
                "subtracting the residue leaves a multiple of the base"
            );
            coeffs.push(digit);
            remaining = quotient;
        }
        coeffs.push(remaining);
        Self::new(coeffs)
    }

    /// The remainder of `self` modulo a monic `divisor`, without forming the
    /// quotient.
    ///
    /// Monicity is what makes this cheap. [`div_rem`](Self::div_rem) must ask
    /// at every step whether the divisor's leading coefficient divides the
    /// remainder's — a full Algorithm D call that can also answer "no", which
    /// is why it returns an `Option`. Against a monic divisor the answer is
    /// always yes and the quotient coefficient *is* the remainder's leading
    /// coefficient, so each step is one multiply-subtract pass over the
    /// divisor's window and nothing else. Reduction becomes a ring
    /// homomorphism `ℤ[x] → ℤ[x]/(divisor)` that never fails.
    ///
    /// A divisor with leading coefficient `−1` generates the same ideal, so
    /// negate it first and the remainder is unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is the zero polynomial or its leading coefficient
    /// is not `1`.
    #[must_use]
    pub fn rem_monic(&self, divisor: &Self) -> Self {
        let divisor_degree = divisor
            .degree()
            .expect("division by the zero polynomial is undefined");
        assert!(
            divisor.leading_coefficient().is_one(),
            "rem_monic requires a monic divisor"
        );
        let Some(self_degree) = self.degree() else {
            return Self::zero();
        };
        if self_degree < divisor_degree {
            return self.clone();
        }
        let mut rem = self.coeffs.clone();
        // Cancel from the top down. The quotient coefficient at each step is
        // exactly `rem[top]` — no division — and the leading term it cancels
        // is set to zero rather than computed, since `factor·1 − factor` is
        // zero by construction.
        for top in (divisor_degree..=self_degree).rev() {
            if rem[top].is_zero() {
                continue;
            }
            let factor = rem[top].clone();
            let shift = top - divisor_degree;
            for (k, d) in divisor.coeffs[..divisor_degree].iter().enumerate() {
                if !d.is_zero() {
                    rem[shift + k].sub_assign_ref(&factor.mul_ref(d));
                }
            }
            rem[top] = BigInt::zero();
        }
        rem.truncate(divisor_degree);
        Self::new(rem)
    }

    /// The product of `factors` in `ℤ[x]/(divisor)` for a monic `divisor`,
    /// by a product tree.
    ///
    /// A fold multiplies a running product whose coefficients grow with every
    /// term by a fresh small one, so the work at step `k` is proportional to
    /// `k` and the total is quadratic in the number of factors. Pairing
    /// instead keeps both operands the same size at every level, which is the
    /// shape [`mul`](Self::mul)'s Karatsuba and Toom kernels want, and the
    /// depth is logarithmic.
    ///
    /// Reducing at every level rather than once at the end is both sound and
    /// the point: reduction modulo a monic polynomial is a ring homomorphism
    /// (see [`rem_monic`](Self::rem_monic)), so the reduced product equals the
    /// reduction of the product, and holding every *reduced* intermediate to
    /// degree below `deg divisor` stops the degree growing with the number of
    /// factors. Each `mul` still reaches twice that degree before its
    /// reduction; what is bounded is the level's output, not the transient.
    /// The coefficients grow with the number of factors, and only track the
    /// answer's own height asymptotically — a list whose product cancels has
    /// intermediates far larger than its result.
    ///
    /// The empty product is `1`, reduced — which is the zero polynomial when
    /// `divisor` is the constant `1`, the ring having collapsed.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is the zero polynomial or is not monic.
    #[must_use]
    pub fn product_mod_monic(factors: &[Self], divisor: &Self) -> Self {
        let mut level: Vec<Self> = factors.iter().map(|f| f.rem_monic(divisor)).collect();
        if level.is_empty() {
            return Self::constant(BigInt::one()).rem_monic(divisor);
        }
        while level.len() > 1 {
            let mut above = Vec::with_capacity(level.len().div_ceil(2));
            for pair in level.chunks(2) {
                above.push(match pair {
                    [left, right] => left.mul(right).rem_monic(divisor),
                    [only] => only.clone(),
                    _ => unreachable!("chunks(2) yields one or two"),
                });
            }
            level = above;
        }
        level.pop().expect("a non-empty level has a root")
    }

    /// The homogeneous substitution `Σ cₖ·aᵏ·b^(d−k)`, where the `cₖ` are the
    /// coefficients of `self` and `d` is its degree.
    ///
    /// This evaluates the *homogenization* `F(X, Y) = Yᵈ·f(X/Y)` at
    /// `(a, b)` — the unique degree-`d` form in two variables that restricts
    /// to `f` on `Y = 1`. Where [`evaluate`](Self::evaluate) answers "what is
    /// `f` at a point", this answers "what does `f` become under a projective
    /// change of variables", and the two agree when `b` is the constant `1`.
    ///
    /// The homogeneous form is the one to use whenever the argument is a
    /// ratio: the sieve's algebraic norm `bᵈ·f(a/b)` is exactly this at
    /// `a = a`, `b = b`, and composing a linear change of coordinates into
    /// `f` — following a lattice, rotating a polynomial — is this at linear
    /// `a` and `b`. It keeps everything in `ℤ[x]` with no division.
    ///
    /// Both power ladders are built once and indexed, so a degree-`d`
    /// substitution costs `2d` multiplications to build them and `d + 1` to
    /// combine, not `O(d²)` repeated powering. Zero coefficients are skipped,
    /// which is what makes a sparse `f` cheap.
    #[must_use]
    pub fn homogeneous_substitution(&self, a: &Self, b: &Self) -> Self {
        let Some(degree) = self.degree() else {
            return Self::zero();
        };
        let one = Self::constant(BigInt::one());
        let mut a_powers = Vec::with_capacity(degree + 1);
        let mut b_powers = Vec::with_capacity(degree + 1);
        a_powers.push(one.clone());
        b_powers.push(one);
        for k in 1..=degree {
            a_powers.push(a_powers[k - 1].mul(a));
            b_powers.push(b_powers[k - 1].mul(b));
        }
        let mut total = Self::zero();
        for (k, c) in self.coeffs.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            total = total.add(&a_powers[k].mul(&b_powers[degree - k]).scale(c));
        }
        total
    }

    /// Every root of `self` modulo `primeᵉ`, in increasing order, by Hensel
    /// lifting from the roots modulo `prime`.
    ///
    /// A root `r` modulo `pᵏ` lifts to candidates `r + t·pᵏ`, and Taylor
    /// truncated after one term is exact modulo `p^(k+1)`:
    ///
    /// ```text
    /// f(r + t·pᵏ) ≡ f(r) + t·pᵏ·f′(r)   (mod p^(k+1)).
    /// ```
    ///
    /// Write `f(r) = s·pᵏ`, which is legitimate because `r` is a root modulo
    /// `pᵏ`. Where `f′(r) ≢ 0 (mod p)` the congruence is linear in `t` with an
    /// invertible coefficient, so there is exactly one lift, `t ≡ −s·f′(r)⁻¹`.
    /// Where `f′(r) ≡ 0 (mod p)` the `t` term vanishes and the congruence no
    /// longer mentions `t` at all: either `p^(k+1) | f(r)` and all `p` lifts
    /// are roots, or none is. That branching is why the count of roots modulo
    /// `pᵏ` is not in general the count modulo `p` — the multiple roots, the
    /// ones that also kill the derivative, are where the tree widens, and they
    /// are exactly the ones lying over primes that divide `disc f`.
    ///
    /// The simple case is the textbook Hensel lemma (Cohen, §1.6); the
    /// branching case is what a general root-finder must add to it.
    ///
    /// A polynomial whose coefficients share a factor of `p` is handled
    /// rather than refused. If `pᵛ` is the largest power of `p` dividing
    /// every coefficient, then `f(x) ≡ 0 (mod pᵉ)` says exactly
    /// `(f/pᵛ)(x) ≡ 0 (mod p^{e−v})`, so the roots are those of `f/pᵛ` to
    /// the reduced precision, each standing for the `pᵛ` residues congruent
    /// to it modulo `p^{e−v}`. Only when `v ≥ e` does the condition become
    /// vacuous and the root set really is everything.
    ///
    /// # Panics
    ///
    /// Panics if `exponent` is zero, if `prime` is less than two, or if
    /// `self` is the zero polynomial or has content divisible by `pᵉ` — in
    /// those two cases every residue is a root and returning `pᵉ` of them as
    /// a list is not useful.
    ///
    /// Panics, too, when the answer or an intermediate level of the lift
    /// would be too wide to hold: a branching root splits into `p`
    /// candidates at once, and a common factor of `pᵛ` multiplies the final
    /// count by `pᵛ`. Both are refused above [`MAX_ROOT_LEVEL`] rather than
    /// left to exhaust memory. That bound is on the *level*, not on the
    /// prime: a branching root modulo a 40-bit prime is refused even though
    /// the prime is far below `2⁶⁴`, which is the case an earlier revision
    /// of this documentation got wrong.
    ///
    /// `prime` must be prime; over a composite the result is unspecified.
    #[must_use]
    pub fn roots_mod_prime_power<R: crate::random::Rng + ?Sized>(
        &self,
        prime: &BigUint,
        exponent: u32,
        rng: &mut R,
    ) -> Vec<BigUint> {
        assert!(exponent >= 1, "a prime power needs a positive exponent");
        assert!(*prime >= BigUint::from_u64(2), "the base must be a prime");
        debug_assert!(
            number_theory::is_probable_prime_bpsw(prime),
            "Hensel lifting and the base-level root finder both need a prime"
        );
        assert!(
            !self.is_zero(),
            "every residue is a root of the zero polynomial"
        );
        // Divide out the largest power of `p` common to every coefficient.
        // It does not change the root set, it changes the precision the
        // root set is taken at: `pᵛ·g(x) ≡ 0 (mod pᵉ)` is `g(x) ≡ 0
        // (mod p^{e−v})`.
        let content_valuation = self
            .coeffs
            .iter()
            .filter(|c| !c.is_zero())
            .map(|c| number_theory::valuation(&c.abs(), prime))
            .min()
            .expect("a non-zero polynomial has a non-zero coefficient");
        assert!(
            content_valuation < exponent as usize,
            "every coefficient is divisible by p^e, so every residue is a root"
        );
        let reduced_exponent = exponent - content_valuation as u32;
        let stripped;
        let target = if content_valuation == 0 {
            self
        } else {
            let divisor = BigInt::from_biguint(prime.pow_u64(content_valuation as u64));
            stripped = Self::new(self.coeffs.iter().map(|c| c.div_rem(&divisor).0).collect());
            &stripped
        };
        let exponent = reduced_exponent;
        let base = PolyModP::from_poly_z(target, prime);
        debug_assert!(
            !base.is_zero(),
            "dividing out the content leaves a polynomial not divisible by p"
        );
        // The derivative is only ever consulted modulo p — the case split
        // above turns on `f′(r) mod p`, not on any higher power — so it is
        // reduced once here and evaluated per root.
        let derivative = PolyModP::from_poly_z(&target.derivative(), prime);
        let mut level = base.roots(rng);
        let mut modulus = prime.clone();

        for _ in 1..exponent {
            let next_modulus = modulus.mul_ref(prime);
            let reduced = PolyModP::from_poly_z(target, &next_modulus);
            let mut next = Vec::with_capacity(level.len());
            for r in &level {
                let value = reduced.evaluate(r);
                let slope = derivative.evaluate(&r.modulo(prime));
                if slope.is_zero() {
                    if value.is_zero() {
                        let span = prime.to_u64().unwrap_or(u64::MAX);
                        assert!(
                            span <= MAX_ROOT_LEVEL as u64
                                && next.len() as u64 + span <= MAX_ROOT_LEVEL as u64,
                            "a branching root would widen the lift past {MAX_ROOT_LEVEL} candidates"
                        );
                        for t in 0..span {
                            next.push(r.add_ref(&BigUint::from_u64(t).mul_ref(&modulus)));
                        }
                    }
                    // Otherwise no lift of this root survives, and the branch
                    // dies here.
                } else {
                    let (s, check) = value.div_rem(&modulus);
                    debug_assert!(check.is_zero(), "a root modulo pᵏ has f(r) divisible by pᵏ");
                    let inverse = number_theory::mod_inverse(&slope, prime)
                        .expect("a non-zero residue is invertible modulo a prime");
                    let negated = BigUint::mod_sub(&BigUint::zero(), &s.modulo(prime), prime);
                    let t = BigUint::mod_mul(&negated, &inverse, prime);
                    next.push(r.add_ref(&t.mul_ref(&modulus)));
                }
            }
            level = next;
            modulus = next_modulus;
        }
        if content_valuation > 0 {
            // Each root modulo `p^{e−v}` stands for the `pᵛ` residues
            // congruent to it modulo `p^{e−v}`; `modulus` is `p^{e−v}` here.
            let span = prime
                .pow_u64(content_valuation as u64)
                .to_u64()
                .unwrap_or(u64::MAX);
            assert!(
                span
                    .checked_mul(level.len() as u64)
                    .is_some_and(|total| total <= MAX_ROOT_LEVEL as u64),
                "a common factor of p^{content_valuation} multiplies the root count past {MAX_ROOT_LEVEL}"
            );
            let mut expanded = Vec::with_capacity(level.len() * span as usize);
            for r in &level {
                for t in 0..span {
                    expanded.push(r.add_ref(&BigUint::from_u64(t).mul_ref(&modulus)));
                }
            }
            level = expanded;
        }
        level.sort();
        level
    }
}

/// The convolution `a ⋆ b` over ℤ, returning `a.len() + b.len() - 1`
/// coefficients — the coefficient-vector core behind [`PolyZ::mul`].
///
/// Schoolbook below [`POLY_KARATSUBA_THRESHOLD_Z`], Karatsuba above it:
/// splitting both operands at `k` coefficients as `a = a₁xᵏ + a₀` and
/// `b = b₁xᵏ + b₀`,
///
/// ```text
/// z₀ = a₀⋆b₀,  z₂ = a₁⋆b₁,  z₁ = (a₀+a₁)⋆(b₀+b₁) − z₀ − z₂ = a₀⋆b₁ + a₁⋆b₀,
/// a ⋆ b = z₂·x²ᵏ + z₁·xᵏ + z₀,
/// ```
///
/// three half-length convolutions where schoolbook needs four quarter-area
/// passes. The sub-convolutions recurse through this function, so a large
/// operand splits repeatedly until a factor drops below the threshold.
/// Unlike the integer Karatsuba this needs no underflow reasoning: the
/// coefficients are independent ring elements with no carries between
/// positions, so `z₁` is a plain coefficient-wise difference and is exactly
/// the cross-term sum.
///
/// A split is taken only when both operands actually reach across it; a
/// lopsided pair whose shorter side lies entirely below the split point has
/// an empty high half and gains nothing, so it falls back to schoolbook.
fn convolve_z(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    if !poly_split_admitted(
        a.len().min(b.len()),
        a.len().max(b.len()),
        POLY_KARATSUBA_THRESHOLD_Z,
        POLY_KARATSUBA_ANY_RATIO_Z,
    ) || !poly_split_dense_enough(
        count_nonzero(a, BigInt::is_zero),
        a.len(),
        count_nonzero(b, BigInt::is_zero),
        b.len(),
        POLY_KARATSUBA_THRESHOLD_Z,
        POLY_KARATSUBA_ANY_RATIO_Z,
    ) {
        return convolve_schoolbook_z(a, b);
    }
    let split = a.len().max(b.len()) / 2;
    if a.len() <= split || b.len() <= split {
        return convolve_schoolbook_z(a, b);
    }
    let (a0, a1) = a.split_at(split);
    let (b0, b1) = b.split_at(split);

    let z0 = convolve_z(a0, b0);
    let z2 = convolve_z(a1, b1);
    let a_sum = add_coeffs_z(a0, a1);
    let b_sum = add_coeffs_z(b0, b1);
    let mut z1 = convolve_z(&a_sum, &b_sum);
    sub_assign_coeffs_z(&mut z1, &z0);
    sub_assign_coeffs_z(&mut z1, &z2);

    let mut out = vec![BigInt::zero(); a.len() + b.len() - 1];
    add_into_at_z(&mut out, &z0, 0);
    add_into_at_z(&mut out, &z1, split);
    add_into_at_z(&mut out, &z2, 2 * split);
    out
}

/// The schoolbook convolution: every `out[i + j]` accumulates `aᵢ·bⱼ`. Zero
/// coefficients skip their inner pass, which is what makes a sparse operand
/// cheap here.
fn convolve_schoolbook_z(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    // Only the outer operand's zeros are skipped, so the sparser one goes
    // outside; the convolution is symmetric, which makes the swap free.
    let (a, b) = if count_nonzero(b, BigInt::is_zero) < count_nonzero(a, BigInt::is_zero) {
        (b, a)
    } else {
        (a, b)
    };
    let mut out = vec![BigInt::zero(); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        if x.is_zero() {
            continue;
        }
        for (j, y) in b.iter().enumerate() {
            let term = x.mul_ref(y);
            out[i + j].add_assign_ref(&term);
        }
    }
    out
}

/// Coefficient-wise sum of two slices, zero-extending the shorter.
fn add_coeffs_z(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = long.to_vec();
    for (slot, value) in out.iter_mut().zip(short) {
        slot.add_assign_ref(value);
    }
    out
}

/// `acc ← acc − sub`, coefficient-wise. `sub` is never longer than `acc`
/// where this is used (the middle term of a Karatsuba split dominates both
/// terms removed from it), which the debug assertion records.
fn sub_assign_coeffs_z(acc: &mut [BigInt], sub: &[BigInt]) {
    debug_assert!(sub.len() <= acc.len(), "Karatsuba middle term dominates");
    for (slot, value) in acc.iter_mut().zip(sub) {
        slot.sub_assign_ref(value);
    }
}

/// `acc[offset..] += addend`, coefficient-wise — the recomposition step.
fn add_into_at_z(acc: &mut [BigInt], addend: &[BigInt], offset: usize) {
    for (slot, value) in acc[offset..].iter_mut().zip(addend) {
        slot.add_assign_ref(value);
    }
}

/// The convolution `a ⋆ b` modulo `m`, returning `a.len() + b.len() - 1`
/// coefficients — the coefficient-vector core behind [`PolyModP::mul`].
/// Shape and split rule are [`convolve_z`]'s, at
/// [`POLY_KARATSUBA_THRESHOLD_MODP`] (far higher, for the reasons recorded
/// on that constant); the differences are where the modular reductions sit
/// and that the recombination steps work in the ring, so intermediates
/// never grow beyond it.
fn convolve_modp(a: &[BigUint], b: &[BigUint], m: &BigUint) -> Vec<BigUint> {
    if !poly_split_admitted(
        a.len().min(b.len()),
        a.len().max(b.len()),
        POLY_KARATSUBA_THRESHOLD_MODP,
        POLY_KARATSUBA_ANY_RATIO_MODP,
    ) || !poly_split_dense_enough(
        count_nonzero(a, BigUint::is_zero),
        a.len(),
        count_nonzero(b, BigUint::is_zero),
        b.len(),
        POLY_KARATSUBA_THRESHOLD_MODP,
        POLY_KARATSUBA_ANY_RATIO_MODP,
    ) {
        return convolve_schoolbook_modp(a, b, m);
    }
    let split = a.len().max(b.len()) / 2;
    if a.len() <= split || b.len() <= split {
        return convolve_schoolbook_modp(a, b, m);
    }
    let (a0, a1) = a.split_at(split);
    let (b0, b1) = b.split_at(split);

    let z0 = convolve_modp(a0, b0, m);
    let z2 = convolve_modp(a1, b1, m);
    let a_sum = add_coeffs_modp(a0, a1, m);
    let b_sum = add_coeffs_modp(b0, b1, m);
    let mut z1 = convolve_modp(&a_sum, &b_sum, m);
    sub_assign_coeffs_modp(&mut z1, &z0, m);
    sub_assign_coeffs_modp(&mut z1, &z2, m);

    let mut out = vec![BigUint::zero(); a.len() + b.len() - 1];
    add_into_at_modp(&mut out, &z0, 0, m);
    add_into_at_modp(&mut out, &z1, split, m);
    add_into_at_modp(&mut out, &z2, 2 * split, m);
    out
}

/// The schoolbook convolution modulo `m`, with the reduction **deferred**:
/// partial products accumulate as ordinary integers and each output
/// coefficient is reduced once, at the end.
///
/// This is the difference between `a.len()·b.len()` divisions and
/// `a.len() + b.len()` of them. Reducing every partial product — the
/// obvious transcription of the ring operation — makes `BigUint::mod_mul`'s
/// Knuth division the inner loop of the convolution, and measurement put
/// that at roughly 2.4× the cost of the same convolution over ℤ.
///
/// The accumulator is bounded and small: every partial product is below
/// `m²`, and an output coefficient sums at most `min(a.len(), b.len())` of
/// them, so it stays under `min(len)·m²` — about `2·bits(m) + lg(len)` bits,
/// two limbs' worth of headroom over the modulus at any size this layer
/// sees.
fn convolve_schoolbook_modp(a: &[BigUint], b: &[BigUint], m: &BigUint) -> Vec<BigUint> {
    // As in the ℤ case: the sparser operand goes outside, where its zeros
    // skip whole inner passes.
    let (a, b) = if count_nonzero(b, BigUint::is_zero) < count_nonzero(a, BigUint::is_zero) {
        (b, a)
    } else {
        (a, b)
    };
    let mut acc = vec![BigUint::zero(); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        if x.is_zero() {
            continue;
        }
        for (j, y) in b.iter().enumerate() {
            let term = x.mul_ref(y);
            acc[i + j].add_assign_ref(&term);
        }
    }
    for slot in &mut acc {
        *slot = slot.modulo(m);
    }
    acc
}

/// The convolution `a ⋆ a` modulo `m` — a squaring, which forms each
/// distinct cross term once and doubles it rather than computing both
/// orderings:
///
/// ```text
/// (Σ aᵢxⁱ)² = Σᵢ aᵢ²x²ⁱ + 2·Σ_{i<j} aᵢaⱼx^{i+j}.
/// ```
///
/// That is `n(n+1)/2` coefficient multiplications against the general
/// convolution's `n²`, and the doubling is a one-bit shift rather than a
/// second product. Reduction is deferred exactly as in
/// [`convolve_schoolbook_modp`]; the doubled terms keep every accumulator
/// below `2n·m²`, one bit above the general case's bound.
///
/// Two dispatches sit in front of it. In characteristic 2 every cross term
/// is doubled and therefore vanishes, leaving the Frobenius map: squaring
/// is a coefficient spread with no arithmetic at all, and the `n(n−1)/2`
/// products the general form would compute are all discarded by the final
/// reduction. Above [`POLY_SQUARE_SPLIT_THRESHOLD_MODP`] the quadratic
/// term count outgrows the split's overhead and the square splits too —
/// three sub-*squarings* rather than three sub-products, since
/// `(a₀+a₁)² − a₀² − a₁²` is the cross term. That threshold is its own
/// constant, half again the product's, and deliberately so: the square
/// already saves the cross terms, so the split has less left to win and
/// has to reach further before it does. Reading the product's crossover
/// across to the square is the mistake the separate constant exists to
/// prevent.
///
/// The split is also refused to a square too sparse to pay for it, by
/// [`poly_square_split_dense_enough`] — the same clause the product has,
/// for the same reason, and the case where it matters most.
fn convolve_square_modp(a: &[BigUint], m: &BigUint) -> Vec<BigUint> {
    let n = a.len();
    if *m == BigUint::from_u64(2) {
        // Frobenius: (Σ aᵢxⁱ)² = Σ aᵢ²x²ⁱ = Σ aᵢx²ⁱ over 𝔽₂, every
        // coefficient being its own square there. Unlike the other two
        // branches this one performs no arithmetic and so has no closing
        // reduction: it requires its input already reduced, which every
        // caller inside the type supplies and which the type's own
        // invariant guarantees.
        let mut acc = vec![BigUint::zero(); 2 * n - 1];
        for (i, coeff) in a.iter().enumerate() {
            acc[2 * i] = coeff.clone();
        }
        return acc;
    }
    if n >= POLY_SQUARE_SPLIT_THRESHOLD_MODP
        && poly_square_split_dense_enough(count_nonzero(a, BigUint::is_zero), n)
    {
        let split = n / 2;
        let (a0, a1) = a.split_at(split);
        let z0 = convolve_square_modp(a0, m);
        let z2 = convolve_square_modp(a1, m);
        let sum = add_coeffs_modp(a0, a1, m);
        let mut z1 = convolve_square_modp(&sum, m);
        sub_assign_coeffs_modp(&mut z1, &z0, m);
        sub_assign_coeffs_modp(&mut z1, &z2, m);
        let mut out = vec![BigUint::zero(); 2 * n - 1];
        add_into_at_modp(&mut out, &z0, 0, m);
        add_into_at_modp(&mut out, &z1, split, m);
        add_into_at_modp(&mut out, &z2, 2 * split, m);
        return out;
    }
    let mut acc = vec![BigUint::zero(); 2 * n - 1];
    for i in 0..n {
        if a[i].is_zero() {
            continue;
        }
        acc[2 * i].add_assign_ref(&a[i].square_ref());
        for j in (i + 1)..n {
            if a[j].is_zero() {
                continue;
            }
            let mut cross = a[i].mul_ref(&a[j]);
            cross.shl1();
            acc[i + j].add_assign_ref(&cross);
        }
    }
    for slot in &mut acc {
        *slot = slot.modulo(m);
    }
    acc
}

/// Coefficient-wise modular sum of two slices, zero-extending the shorter.
fn add_coeffs_modp(a: &[BigUint], b: &[BigUint], m: &BigUint) -> Vec<BigUint> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = long.to_vec();
    for (slot, value) in out.iter_mut().zip(short) {
        *slot = BigUint::mod_add(slot, value, m);
    }
    out
}

/// `acc ← acc − sub` modulo `m`, coefficient-wise.
fn sub_assign_coeffs_modp(acc: &mut [BigUint], sub: &[BigUint], m: &BigUint) {
    debug_assert!(sub.len() <= acc.len(), "Karatsuba middle term dominates");
    for (slot, value) in acc.iter_mut().zip(sub) {
        *slot = BigUint::mod_sub(slot, value, m);
    }
}

/// `acc[offset..] += addend` modulo `m`, coefficient-wise.
fn add_into_at_modp(acc: &mut [BigUint], addend: &[BigUint], offset: usize, m: &BigUint) {
    for (slot, value) in acc[offset..].iter_mut().zip(addend) {
        *slot = BigUint::mod_add(slot, value, m);
    }
}

/// The Sylvester matrix of two non-constant polynomials `a` (degree `m`)
/// and `b` (degree `n`): an `(m+n)×(m+n)` matrix whose top `n` rows are
/// shifted copies of `a`'s coefficients (high-to-low) and bottom `m` rows
/// shifted copies of `b`'s. Row-major.
fn sylvester_matrix(a: &PolyZ, b: &PolyZ) -> Vec<Vec<BigInt>> {
    let m = a.degree().expect("non-constant");
    let n = b.degree().expect("non-constant");
    let size = m + n;
    let mut matrix = vec![vec![BigInt::zero(); size]; size];
    // a's coefficients high-to-low, length m+1.
    let a_hi: Vec<BigInt> = a.coefficients().iter().rev().cloned().collect();
    let b_hi: Vec<BigInt> = b.coefficients().iter().rev().cloned().collect();
    for i in 0..n {
        for (j, coeff) in a_hi.iter().enumerate() {
            matrix[i][i + j] = coeff.clone();
        }
    }
    for i in 0..m {
        for (j, coeff) in b_hi.iter().enumerate() {
            matrix[n + i][i + j] = coeff.clone();
        }
    }
    matrix
}

/// The determinant of an integer matrix by Bareiss fraction-free Gaussian
/// elimination: each elimination step divides exactly by the previous
/// pivot, so all intermediates stay integral, and the last pivot is the
/// determinant (up to the sign accumulated by row swaps).
///
/// Exactness rests on Sylvester's identity — after step `k` the entry at
/// `(i, j)` is the `(k+1)×(k+1)` minor built from rows `0..=k, i` and
/// columns `0..=k, j`, and dividing by the previous pivot recovers exactly
/// that minor from the cross-product. A zero pivot is repaired by
/// exchanging the whole row with one below, which negates the determinant
/// and leaves every surviving entry a minor of the exchanged matrix, so the
/// division stays exact. A column with no non-zero entry on or below the
/// diagonal makes the matrix singular and the determinant zero.
///
/// `previous` is never zero: it starts at 1 and is thereafter the pivot of
/// the preceding step, which the exchange-or-return above guaranteed
/// non-zero.
fn bareiss_determinant(mut matrix: Vec<Vec<BigInt>>) -> BigInt {
    let n = matrix.len();
    if n == 0 {
        return BigInt::one();
    }
    let mut sign_negative = false;
    let mut previous = BigInt::one();
    for k in 0..n - 1 {
        if matrix[k][k].is_zero() {
            // Find a row below with a non-zero entry in this column.
            let Some(swap) = (k + 1..n).find(|&r| !matrix[r][k].is_zero()) else {
                return BigInt::zero(); // singular
            };
            matrix.swap(k, swap);
            sign_negative = !sign_negative;
        }
        let pivot = matrix[k][k].clone();
        for i in k + 1..n {
            for j in k + 1..n {
                // matrix[i][j] ← (matrix[i][j]·pivot − matrix[i][k]·matrix[k][j]) / previous
                let cross = matrix[i][j]
                    .mul_ref(&pivot)
                    .sub_ref(&matrix[i][k].mul_ref(&matrix[k][j]));
                matrix[i][j] = cross.div_exact(&previous);
            }
            matrix[i][k] = BigInt::zero();
        }
        previous = pivot;
    }
    let det = matrix[n - 1][n - 1].clone();
    if sign_negative {
        det.negated()
    } else {
        det
    }
}

/// A univariate polynomial over ℤ/mℤ for a fixed modulus `m ≥ 2`,
/// coefficients low-to-high and reduced, normalized to drop trailing
/// zeros.
///
/// The modulus travels with the polynomial, and every binary operation
/// asserts that the two moduli agree — a **hard `assert_eq!` in release as
/// well as debug**, not a `debug_assert!`. The whole point of carrying the
/// modulus in the value is that it cannot drift out of step with the
/// coefficients; a debug-only check would let an optimized build combine an
/// 𝔽ₚ element with an 𝔽_q one and hand back a result tagged with whichever
/// of the two moduli the receiver happened to hold. The check runs before
/// any short-circuit, so mismatched zero operands panic too. Affected:
/// [`Self::add`], [`Self::sub`], [`Self::mul`], [`Self::div_rem`],
/// [`Self::rem`], [`Self::gcd`], and [`Self::pow_mod`].
///
/// Addition, subtraction, multiplication, scaling, and evaluation work for
/// any `m ≥ 2`. The division-based operations — [`Self::div_rem`],
/// [`Self::rem`], [`Self::gcd`], [`Self::make_monic`], and
/// [`Self::pow_mod`] — invert a leading coefficient modulo `m`, so they
/// require `m` **prime** (precisely, the relevant leading coefficient
/// invertible modulo `m`) and panic on a non-invertible pivot. This is the
/// field ℤ/pℤ the factorization routines are built over.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolyModP {
    coeffs: Vec<BigUint>,
    modulus: BigUint,
}

impl PolyModP {
    /// Build from coefficients low-to-high, reducing each modulo `modulus`
    /// and normalizing away trailing zeros. Both steps establish the type's
    /// invariants: coefficients are the least non-negative residues, so two
    /// representations of one residue class cannot compare unequal, and a
    /// coefficient that reduces to zero at the top does not inflate the
    /// reported degree. `modulus < 2` is rejected because ℤ/1ℤ is the zero
    /// ring, where degree carries no information.
    ///
    /// # Panics
    ///
    /// Panics if `modulus < 2`.
    #[must_use]
    pub fn new(coeffs: Vec<BigUint>, modulus: &BigUint) -> Self {
        assert!(
            *modulus >= BigUint::from_u64(2),
            "polynomial modulus must be at least 2"
        );
        let coeffs = coeffs.into_iter().map(|c| c.modulo(modulus)).collect();
        let mut poly = Self {
            coeffs,
            modulus: modulus.clone(),
        };
        poly.normalize();
        poly
    }

    /// The zero polynomial over `modulus`: the empty coefficient list, whose
    /// degree is `None`.
    ///
    /// # Panics
    ///
    /// Panics if `modulus < 2`, as [`Self::new`] does.
    #[must_use]
    pub fn zero(modulus: &BigUint) -> Self {
        Self::new(Vec::new(), modulus)
    }

    /// Reduce an integer polynomial modulo `modulus`, the ring homomorphism
    /// `ℤ[x] → (ℤ/mℤ)[x]`. Each coefficient goes through
    /// [`BigInt::modulo_positive`], which maps a negative integer to its
    /// least non-negative residue rather than to a negative remainder, so
    /// the result satisfies this type's reduced-coefficient invariant.
    /// Reduction can lower the degree, when the leading coefficient is a
    /// multiple of `modulus`.
    ///
    /// # Panics
    ///
    /// Panics if `modulus < 2`, as [`Self::new`] does.
    #[must_use]
    pub fn from_poly_z(poly: &PolyZ, modulus: &BigUint) -> Self {
        let coeffs = poly
            .coefficients()
            .iter()
            .map(|c| c.modulo_positive(modulus))
            .collect();
        Self::new(coeffs, modulus)
    }

    /// Build from coefficients the caller guarantees are already reduced
    /// modulo `modulus`, skipping the per-coefficient reduction that
    /// [`Self::new`] pays. The trailing-zero invariant is still
    /// re-established here; the reduced-coefficient invariant is the
    /// caller's to uphold, which is why this is private — every internal
    /// caller hands over residues that are reduced by construction: the
    /// `mod_add`/`mod_sub`/`mod_mul` results of the ring operations, the
    /// closing reduction pass of the convolutions, and the closing pass of
    /// the division's working remainder.
    fn from_reduced(coeffs: Vec<BigUint>, modulus: &BigUint) -> Self {
        let mut poly = Self {
            coeffs,
            modulus: modulus.clone(),
        };
        poly.normalize();
        poly
    }

    /// Restore the trailing-zero invariant by popping high-order zeros.
    /// A coefficient becomes zero here through reduction as well as through
    /// cancellation, so this runs after every construction.
    fn normalize(&mut self) {
        while self.coeffs.last().is_some_and(BigUint::is_zero) {
            self.coeffs.pop();
        }
    }

    /// The modulus.
    #[must_use]
    pub fn modulus(&self) -> &BigUint {
        &self.modulus
    }

    /// Whether this is the zero polynomial.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// The degree, or `None` for the zero polynomial.
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        self.coeffs.len().checked_sub(1)
    }

    /// The coefficients, low-to-high.
    #[must_use]
    pub fn coefficients(&self) -> &[BigUint] {
        &self.coeffs
    }

    /// The leading coefficient, or zero for the zero polynomial.
    #[must_use]
    pub fn leading_coefficient(&self) -> BigUint {
        self.coeffs.last().cloned().unwrap_or_else(BigUint::zero)
    }

    fn check_modulus(&self, other: &Self) {
        // A hard assert in every build: the whole point of the type is that the
        // modulus cannot drift out of step with the coefficients, and a
        // debug-only check would let a release build silently combine an 𝔽ₚ
        // element with an 𝔽_q one and tag the result with one of the two moduli.
        assert_eq!(
            self.modulus, other.modulus,
            "PolyModP operands must share a modulus"
        );
    }

    /// `self + other` (mod m), added coefficient-wise (shorter operand
    /// zero-extended).
    ///
    /// # Panics
    ///
    /// Panics if the two moduli differ (see the type documentation): a hard
    /// assertion in every build, checked before anything else.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        self.check_modulus(other);
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(BigUint::zero);
            let b = other.coeffs.get(i).cloned().unwrap_or_else(BigUint::zero);
            coeffs.push(BigUint::mod_add(&a, &b, &self.modulus));
        }
        Self::new(coeffs, &self.modulus)
    }

    /// `self − other` (mod m), subtracted coefficient-wise (shorter operand
    /// zero-extended). Equal leading terms cancel, so the result is
    /// renormalized and the degree drops accordingly.
    ///
    /// # Panics
    ///
    /// Panics if the two moduli differ (see the type documentation): a hard
    /// assertion in every build, checked before anything else.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        self.check_modulus(other);
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(BigUint::zero);
            let b = other.coeffs.get(i).cloned().unwrap_or_else(BigUint::zero);
            coeffs.push(BigUint::mod_sub(&a, &b, &self.modulus));
        }
        Self::new(coeffs, &self.modulus)
    }

    /// `self · other` (mod m) by the coefficient convolution — schoolbook
    /// below 128 coefficients (`POLY_KARATSUBA_THRESHOLD_MODP`), Karatsuba
    /// above it for shapes the measured admission rule accepts — near
    /// balance up to 512 coefficients, any ratio above that, and dense
    /// operands throughout. The threshold is higher than the ℤ one because
    /// the coefficients here stay bounded by the modulus rather than
    /// growing, so a saved multiplication never becomes dear relative to
    /// the modular additions a split adds; both constants carry their
    /// measurements. The reduction is deferred to one per output
    /// coefficient rather than one per partial product, which is what makes
    /// the schoolbook path here as cheap as it is.
    ///
    /// Unlike the ℤ case, `deg self + deg other` is only an upper bound on
    /// the degree of the product: ℤ/mℤ has zero divisors for composite `m`,
    /// so the leading coefficients can multiply to zero, and the
    /// renormalization then drops the top entries. The coefficients come
    /// back reduced, so only that renormalization is needed.
    ///
    /// # Panics
    ///
    /// Panics if the two moduli differ (see the type documentation): a hard
    /// assertion in every build, checked before the zero short-circuit, so
    /// a mismatch involving the zero polynomial panics as well.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        self.check_modulus(other);
        if self.is_zero() || other.is_zero() {
            return Self::zero(&self.modulus);
        }
        Self::from_reduced(
            convolve_modp(&self.coeffs, &other.coeffs, &self.modulus),
            &self.modulus,
        )
    }

    /// `self · c` for a scalar (mod m). `c` is a bare [`BigUint`] carrying no
    /// modulus of its own, so there is nothing to cross-check; it need not
    /// arrive reduced, since `BigUint::mod_mul` reduces it. The literal
    /// value 1 short-circuits to a clone — the coefficients are already
    /// reduced, so multiplying by unity cannot change them. (A `c` that is
    /// merely congruent to 1 takes the general path; recognizing it would
    /// cost the reduction the fast path exists to avoid.)
    #[must_use]
    pub fn scale(&self, c: &BigUint) -> Self {
        if c.is_one() {
            return self.clone();
        }
        let coeffs = self
            .coeffs
            .iter()
            .map(|a| BigUint::mod_mul(a, c, &self.modulus))
            .collect();
        Self::new(coeffs, &self.modulus)
    }

    /// Evaluate at `x` (mod m) by Horner's method, folding
    /// `acc ← acc·x + cᵢ` from the leading coefficient down with every step
    /// reduced, so no intermediate exceeds `m²`. `x` is reduced first and so
    /// need not arrive in `[0, m)`.
    #[must_use]
    pub fn evaluate(&self, x: &BigUint) -> BigUint {
        let x = x.modulo(&self.modulus);
        let mut acc = BigUint::zero();
        for coeff in self.coeffs.iter().rev() {
            acc = BigUint::mod_add(
                &BigUint::mod_mul(&acc, &x, &self.modulus),
                coeff,
                &self.modulus,
            );
        }
        acc
    }

    /// The monic associate of `self`: `self` scaled by the inverse of its
    /// leading coefficient, so the result generates the same ideal and has
    /// leading coefficient 1. This is the canonical representative the gcd
    /// and factorization routines return, since over a field the gcd is
    /// unique only up to a unit. The zero polynomial has no leading
    /// coefficient to invert and is returned unchanged; an already-monic
    /// polynomial short-circuits without an inversion.
    ///
    /// # Panics
    ///
    /// Panics if the leading coefficient is not invertible modulo `m`
    /// (always invertible when `m` is prime and the polynomial is
    /// non-zero).
    #[must_use]
    pub fn make_monic(&self) -> Self {
        if self.is_zero() {
            return self.clone();
        }
        let lc = self.leading_coefficient();
        if lc.is_one() {
            return self.clone();
        }
        let inv = number_theory::mod_inverse(&lc, &self.modulus)
            .expect("leading coefficient is invertible modulo a prime");
        self.scale(&inv)
    }

    /// Division with remainder by a divisor whose leading coefficient is
    /// invertible modulo `m`: `self = quotient·divisor + remainder` with
    /// `deg remainder < deg divisor`.
    ///
    /// Why the invertibility requirement, and why this is total where
    /// [`PolyZ::div_rem`] is not: schoolbook long division cancels the
    /// remainder's leading term by multiplying the divisor by
    /// `lc(remainder)·lc(divisor)⁻¹`. Over the field ℤ/pℤ that inverse
    /// always exists for a non-zero divisor, so no step can fail and no
    /// `Option` is needed. The inverse is computed once, before the loop,
    /// and reused at every step — and not computed at all for a monic
    /// divisor; each step lowers `deg remainder` by at least one, which is
    /// what terminates the loop.
    ///
    /// # Panics
    ///
    /// Panics if the two moduli differ (see the type documentation), if
    /// `divisor` is zero, or if the divisor's leading coefficient is not
    /// invertible modulo `m` *and there is anything to divide* — a dividend
    /// of lower degree returns `(0, self)` without touching the
    /// coefficient. For composite `m` the invertibility panic is reachable,
    /// not an internal invariant.
    #[must_use]
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        let (quotient, remainder) = self.divide(divisor, true);
        (quotient.expect("quotient was requested"), remainder)
    }

    /// `self mod divisor` — the remainder alone. This runs the same
    /// long-division loop as [`Self::div_rem`] but skips the quotient
    /// bookkeeping entirely, which matters because remainders are the hot
    /// operation here: [`Self::gcd`] — and through it the whole
    /// factorization pipeline — discards every quotient it would have paid
    /// to build.
    ///
    /// # Panics
    ///
    /// Panics exactly as [`Self::div_rem`] does: differing moduli, a zero
    /// divisor, or a leading coefficient not invertible modulo `m` when
    /// there is anything to divide (a dividend of lower degree comes back
    /// unchanged without touching the coefficient).
    #[must_use]
    pub fn rem(&self, divisor: &Self) -> Self {
        self.divide(divisor, false).1
    }

    /// The schoolbook long-division core behind [`Self::div_rem`] and
    /// [`Self::rem`]: quotient accumulation is optional so the
    /// remainder-only callers do not pay for coefficients they discard.
    ///
    /// Two deliberate economies, both on the pipeline's hottest path
    /// (`gcd` → `squarefree_factorization` / `distinct_degree`):
    ///
    /// - A monic divisor — which is every divisor the factorization
    ///   routines produce — skips the leading-coefficient inversion
    ///   outright, since [`number_theory::mod_inverse`] has no shortcut for
    ///   an argument of 1 and would run a full Euclid loop to invert it.
    /// - The remainder's degree is tracked by hand (`top`) so each step
    ///   subtracts `factor·xˢʰⁱᶠᵗ·divisor` in place over the window it
    ///   touches, `[shift, top]`, instead of materializing a shifted,
    ///   scaled copy of the divisor at full remainder length.
    fn divide(&self, divisor: &Self, want_quotient: bool) -> (Option<Self>, Self) {
        self.check_modulus(divisor);
        assert!(!divisor.is_zero(), "division by the zero polynomial");
        let divisor_degree = divisor.degree().expect("non-zero divisor");
        if self.degree().is_none_or(|d| d < divisor_degree) {
            // Nothing to divide: self = 0·divisor + self.
            let quotient = want_quotient.then(|| Self::zero(&self.modulus));
            return (quotient, self.clone());
        }
        let self_degree = self.degree().expect("degree checked above");
        let lc = divisor.leading_coefficient();
        let lc_inv = if lc.is_one() {
            None
        } else {
            Some(
                number_theory::mod_inverse(&lc, &self.modulus)
                    .expect("divisor's leading coefficient is invertible"),
            )
        };
        // The working remainder carries *unreduced* coefficients, for the
        // same reason the convolution does (see `convolve_schoolbook_modp`):
        // reducing every coefficient the window touches makes a Knuth
        // division the inner loop, one per coefficient per step, where the
        // whole division needs only one per step.
        //
        // Subtraction is what makes that awkward over ℕ, so the window adds
        // instead: `m² − factor·d` is non-negative (both factors are below
        // `m`, so the product is below `m²`) and congruent to `−factor·d`,
        // `m²` being a multiple of `m`. Only the leading position is
        // reduced each step — it has to be, since it decides both the
        // cancellation test and the next quotient coefficient — and the
        // survivors are reduced once at the end.
        //
        // Growth is bounded: a position lies inside the window for at most
        // `deg divisor + 1` steps, so it accumulates that many offsets
        // below `m²`, staying under `2·bits(m) + lg(deg divisor)` bits.
        let modulus_squared = self.modulus.square_ref();
        let mut rem = self.coeffs.clone();
        let mut quotient =
            want_quotient.then(|| vec![BigUint::zero(); self_degree - divisor_degree + 1]);
        let mut top = self_degree;
        loop {
            // Reduce the leading position: cheap when it is already reduced
            // (`div_rem` short-circuits below the divisor), and this is
            // simultaneously the cancellation test, since the step below
            // leaves a multiple of `m` behind rather than a literal zero.
            rem[top] = rem[top].modulo(&self.modulus);
            if rem[top].is_zero() {
                if top == 0 {
                    break;
                }
                top -= 1;
                continue;
            }
            if top < divisor_degree {
                break;
            }
            let shift = top - divisor_degree;
            let factor = match &lc_inv {
                Some(inv) => BigUint::mod_mul(&rem[top], inv, &self.modulus),
                None => rem[top].clone(),
            };
            // rem ← rem − factor·xˢʰⁱᶠᵗ·divisor over the window, carried as
            // the congruent addition described above.
            for (k, d) in divisor.coeffs.iter().enumerate() {
                if !d.is_zero() {
                    let offset = modulus_squared.sub_ref(&factor.mul_ref(d));
                    rem[shift + k].add_assign_ref(&offset);
                }
            }
            // The step must have cancelled the leading position — that is
            // what the choice of `factor` is for — leaving a multiple of
            // the modulus rather than a literal zero, since the window
            // added `m² − factor·d` instead of subtracting. Without this
            // check a failure would not raise: the loop would take a second
            // step at the same `top`, and so at the same `shift`, and
            // *overwrite* the quotient coefficient it had already written,
            // then terminate normally with a silently wrong quotient.
            debug_assert!(
                rem[top].modulo(&self.modulus).is_zero(),
                "the leading term cancels by construction"
            );
            if let Some(q) = quotient.as_mut() {
                q[shift] = factor;
            }
        }
        // The quotient coefficients came from `mod_mul` or from an
        // already-reduced leading position, so only the remainder needs the
        // closing pass.
        let quotient = quotient.map(|q| Self::from_reduced(q, &self.modulus));
        for slot in &mut rem {
            *slot = slot.modulo(&self.modulus);
        }
        (quotient, Self::from_reduced(rem, &self.modulus))
    }

    /// The monic greatest common divisor, by the Euclidean algorithm:
    /// repeatedly replace `(a, b)` with `(b, a mod b)` until `b` is zero,
    /// then take the monic associate of `a`. Each step strictly lowers
    /// `deg b`, so the recursion is finite; the gcd is normalized because
    /// over a field it is unique only up to a unit, and the monic
    /// representative is the canonical choice.
    ///
    /// The zero polynomial is the identity for gcd, so `gcd(0, q)` and
    /// `gcd(q, 0)` are both the monic form of `q`, and `gcd(0, 0)` is zero.
    ///
    /// # Panics
    ///
    /// Panics if the two moduli differ (see the type documentation), or if
    /// some remainder reached during the descent has a leading coefficient
    /// that is not invertible modulo `m`. Over a prime modulus the latter
    /// cannot occur, since every non-zero residue is a unit.
    #[must_use]
    pub fn gcd(&self, other: &Self) -> Self {
        self.check_modulus(other);
        let mut a = self.clone();
        let mut b = other.clone();
        while !b.is_zero() {
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        a.make_monic()
    }

    /// `self^exponent mod modulus_poly` by left-to-right binary
    /// exponentiation — the primitive behind distinct-degree
    /// factorization's `x^(p^d)`.
    ///
    /// Scanning the exponent from its top bit down, each step squares the
    /// accumulator and multiplies in the base when the bit is set, reducing
    /// modulo `modulus_poly` after every product. Reducing at every step,
    /// rather than at the end, is what keeps the degree bounded by
    /// `deg modulus_poly`: the exponent is `p^d` in the factorization
    /// routines, so the unreduced power is not representable.
    ///
    /// # Panics
    ///
    /// Panics if the two moduli differ (see the type documentation), if
    /// `modulus_poly` is the zero polynomial, or if a leading coefficient
    /// reached during reduction is not invertible modulo `m` (`m` prime
    /// avoids the last).
    #[must_use]
    pub fn pow_mod(&self, exponent: &BigUint, modulus_poly: &Self) -> Self {
        self.check_modulus(modulus_poly);
        if exponent.is_zero() {
            let one = Self::new(vec![BigUint::one()], &self.modulus);
            return one.rem(modulus_poly);
        }
        // Seeded from the top set bit, which a non-zero exponent always
        // has: starting from the constant 1 would spend a squaring, a
        // multiplication, and two polynomial reductions arriving at the
        // same state. Every squaring goes through `square`, which forms
        // each cross term once.
        let base = self.rem(modulus_poly);
        let mut result = base.clone();
        for bit in (0..exponent.bits() - 1).rev() {
            result = result.square().rem(modulus_poly);
            if exponent.bit(bit) {
                result = result.mul(&base).rem(modulus_poly);
            }
        }
        result
    }

    /// `self²` (mod m), by the dedicated squaring convolution
    /// ([`convolve_square_modp`]) rather than `mul(self, self)`: half the
    /// coefficient multiplications, since each distinct cross term is
    /// formed once and doubled. This is the operation the exponentiation
    /// ladder spends most of its time in.
    #[must_use]
    fn square(&self) -> Self {
        if self.is_zero() {
            return Self::zero(&self.modulus);
        }
        Self::from_reduced(
            convolve_square_modp(&self.coeffs, &self.modulus),
            &self.modulus,
        )
    }

    /// The monic polynomial `x` over this modulus — the argument of the
    /// Frobenius map, needed by both [`Self::distinct_degree`] and
    /// [`Self::roots`].
    fn monomial_x(modulus: &BigUint) -> Self {
        Self::new(vec![BigUint::zero(), BigUint::one()], modulus)
    }

    /// Squarefree factorization over 𝔽ₚ: the distinct squarefree factors
    /// with their multiplicities, `(factor, e)` with `∏ factorᵉ = ` the
    /// monic form of `self` (Cohen, *A Course in Computational Algebraic
    /// Number Theory*, §3.4.2, Squarefree Factorization). The factors are
    /// monic, of degree ≥ 1, and pairwise coprime; a factor with
    /// multiplicity divisible by `p` is invisible to the derivative and is
    /// recovered instead through the `p`-th root the characteristic forces.
    /// The list is returned in ascending order of multiplicity. Each
    /// multiplicity occurs at most once, so the ordering is total: the
    /// multiplicities reached directly are coprime to `p` and those reached
    /// through the `p`-th root are divisible by it.
    ///
    /// # Panics
    ///
    /// Panics if `self` is constant or zero (no factorization), or on a
    /// non-invertible pivot during division. A prime modulus is required
    /// (see the type documentation): over a composite modulus a division
    /// may instead find every pivot a unit, and then the result is
    /// unspecified rather than a panic.
    #[must_use]
    pub fn squarefree_factorization(&self) -> Vec<(Self, usize)> {
        assert!(
            self.degree().is_some_and(|d| d >= 1),
            "squarefree factorization needs a non-constant polynomial"
        );
        let mut factors: Vec<(Self, usize)> = Vec::new();
        self.squarefree_into(1, &mut factors);
        factors.sort_by_key(|(_, e)| *e);
        factors
    }

    /// The recursive worker of [`Self::squarefree_factorization`]: extract
    /// the squarefree factors of `self` whose multiplicity is not a
    /// multiple of `p`, then recurse on the `p`-th root of what remains,
    /// scaling the multiplicity by `p`.
    ///
    /// Writing `f = ∏ aᵢⁱ` with the `aᵢ` squarefree and pairwise coprime,
    /// the loop maintains, at the top of iteration `e`,
    ///
    /// ```text
    /// v = ∏_{p∤i, i≥e} aᵢ           t = ∏_{p∤i, i≥e} aᵢ^(i−e) · ∏_{p|i} aᵢⁱ
    /// ```
    ///
    /// so `gcd(t, v)` advances `v` to `e+1` and the quotient `v / gcd(t, v)`
    /// is exactly `a_e`. Each iteration strictly shrinks `v`, which is what
    /// terminates the loop; on exit `v` is a unit and `t` has collapsed to
    /// `∏_{p|i} aᵢⁱ`, the `p`-th power the recursion consumes.
    fn squarefree_into(&self, mult_shift: usize, out: &mut Vec<(Self, usize)>) {
        let f = self.make_monic();
        let derivative = f.derivative_modp();
        // gcd(f, f') keeps each factor of multiplicity i to the power i−1
        // when p ∤ i, and to the full power i when p | i — differentiating
        // aᵢⁱ gives i·aᵢ^(i−1)·aᵢ', which vanishes exactly when p | i. So
        // f / gcd(f, f') is the product of the distinct factors whose
        // multiplicity is not divisible by p.
        let mut t = f.gcd(&derivative);
        let mut v = f.div_rem(&t).0;
        let mut e = 1usize;
        while v.degree().is_some_and(|d| d >= 1) {
            let w = t.gcd(&v);
            let factor = v.div_rem(&w).0;
            if factor.degree().is_some_and(|d| d >= 1) {
                out.push((factor.make_monic(), e * mult_shift));
            }
            v = w;
            t = t.div_rem(&v).0;
            e += 1;
        }
        // What remains in t is a p-th power: g(x)^p = g(xᵖ). Take its p-th
        // root and recurse with the multiplicity scaled by p. Under the
        // type's prime-modulus precondition, reaching here with a
        // positive-degree t means deg t ≥ p, so p fits a machine word (deg
        // is bounded by memory) and the first `expect` is unreachable; a
        // *composite* modulus above 2^64 can reach it, because t then need
        // not be a p-th power at all. The second conversion is infallible
        // on the crate's supported 64-bit hosts, where usize and u64 have
        // the same width.
        if t.degree().is_some_and(|d| d >= 1) {
            let p = usize::try_from(
                self.modulus
                    .to_u64()
                    .expect("unreachable under a prime modulus: the modulus was composite"),
            )
            .expect("u64 fits usize on the supported 64-bit hosts");
            let root = t.pth_root(p);
            root.squarefree_into(mult_shift * p, out);
        }
    }

    /// The `p`-th root of a polynomial that is known to be a `p`-th power:
    /// `g(xᵖ)` has `g`'s coefficients at positions that are multiples of
    /// `p`, and in 𝔽ₚ each coefficient is its own `p`-th root (Frobenius).
    ///
    /// Concretely, `(∑ gᵢxⁱ)ᵖ = ∑ gᵢᵖx^(ip) = ∑ gᵢx^(ip)` over 𝔽ₚ — the
    /// cross terms carry a binomial coefficient divisible by `p` and the
    /// coefficients are fixed by Fermat's little theorem — so the root is
    /// read off by taking every `p`-th coefficient with no arithmetic at
    /// all. The precondition is unchecked: applied to a polynomial that is
    /// not a `p`-th power this silently discards the coefficients at
    /// positions not divisible by `p`. The only caller is
    /// `squarefree_into`, where the residual `t` is a `p`-th power by
    /// construction.
    fn pth_root(&self, p: usize) -> Self {
        let coeffs = (0..self.coeffs.len())
            .step_by(p)
            .map(|i| self.coeffs[i].clone())
            .collect();
        Self::new(coeffs, &self.modulus)
    }

    /// The formal derivative over 𝔽ₚ, `∑ i·cᵢ·xⁱ⁻¹` with the scaling done
    /// modulo `m`. Unlike the ℤ case it can vanish on a non-constant
    /// polynomial: the scaling kills every term whose index is a multiple of
    /// the characteristic, so the derivative is zero exactly when the
    /// polynomial is `g(xᵖ) = g(x)ᵖ`. That vanishing is what drives the
    /// recursion in `squarefree_into`.
    fn derivative_modp(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero(&self.modulus);
        }
        let coeffs = self.coeffs[1..]
            .iter()
            .enumerate()
            .map(|(i, c)| BigUint::mod_mul(c, &BigUint::from_u64(i as u64 + 1), &self.modulus))
            .collect();
        Self::new(coeffs, &self.modulus)
    }

    /// Distinct-degree factorization of a squarefree monic polynomial:
    /// the pairs `(d, g_d)` where `g_d` is the product of all the degree-`d`
    /// irreducible factors of `self` (Cohen, §3.4.3, Distinct Degree
    /// Factorization). The `g_d` are monic; a `d` with no factor of that
    /// degree is omitted.
    ///
    /// The mechanism is that `x^(pᵈ) − x` is the product of every monic
    /// irreducible over 𝔽ₚ whose degree *divides* `d`, so
    /// `gcd(remaining, x^(pᵈ) − x)` would capture those lower degrees too.
    /// It does not, because `d` ascends by one from 1 with no gaps and each
    /// captured block is divided out of `remaining` before the next round,
    /// so every proper divisor of `d` has already been removed. The
    /// Frobenius power is advanced in place, `x^(pᵈ) = (x^(pᵈ⁻¹))ᵖ`, one
    /// [`Self::pow_mod`] per round rather than a fresh exponentiation.
    ///
    /// The loop stops once `deg remaining < 2(d+1)`: every irreducible factor
    /// still present has degree greater than `d`, so two of them would give
    /// degree at least `2(d+1)`. There is therefore at most one, and the
    /// residue is emitted whole at its own degree without a further gcd.
    ///
    /// Both preconditions are unchecked. A non-squarefree argument yields
    /// blocks that are not products of *distinct* irreducibles, which the
    /// equal-degree split then cannot separate; a non-monic argument
    /// propagates its leading coefficient into the quotients. Both callers
    /// ([`Self::factor`] via [`Self::squarefree_factorization`], and
    /// [`Self::is_irreducible`] after its own gcd test) establish them.
    fn distinct_degree(&self) -> Vec<(usize, Self)> {
        let mut factors = Vec::new();
        let mut remaining = self.clone();
        // xqi holds x^(pᵈ) mod remaining, advanced one Frobenius power per d.
        let mut xqi = Self::monomial_x(&self.modulus);
        let x = Self::monomial_x(&self.modulus);
        let mut d = 0usize;
        while remaining.degree().is_some_and(|r| r >= 2 * (d + 1)) {
            d += 1;
            // x^(pᵈ) = (x^(pᵈ⁻¹))ᵖ mod remaining.
            xqi = xqi.pow_mod(&self.modulus, &remaining);
            // The degree-d factors divide x^(pᵈ) − x; lower degrees dividing
            // d were captured and divided out in earlier rounds.
            // pow_mod reduces after every product, so deg(xqi) < deg(remaining),
            // and the loop guard keeps deg(remaining) ≥ 2 ≥ deg(x) + 1. The
            // difference is therefore already reduced; a rem here would be a
            // full division that cannot change it.
            let diff = xqi.sub(&x);
            let g = remaining.gcd(&diff);
            if g.degree().is_some_and(|dg| dg >= 1) {
                factors.push((d, g.clone()));
                remaining = remaining.div_rem(&g).0;
                xqi = xqi.rem(&remaining);
            }
        }
        if remaining.degree().is_some_and(|r| r >= 1) {
            let dr = remaining.degree().expect("non-constant");
            factors.push((dr, remaining));
        }
        factors
    }

    /// Split a monic product of *distinct* degree-`d` irreducibles into its
    /// irreducible factors — the equal-degree step (Cantor & Zassenhaus,
    /// *A new algorithm for factoring polynomials over finite fields*,
    /// Math. Comp. 36 (1981), 587–592; Cohen, §3.4.4, Final Splitting).
    ///
    /// By the Chinese remainder theorem `𝔽ₚ[x]/(self)` is the product of the
    /// residue fields `𝔽_(pᵈ)`, one per factor. For odd `p`, raising a random
    /// element to `(pᵈ−1)/2` lands each component independently on `±1`, so
    /// `gcd(self, a^((pᵈ−1)/2) − 1)` collects the components that landed on
    /// `1` — a non-trivial split unless every component agreed. Over 𝔽₂
    /// there is no such square-root character and the trace map
    /// `a + a² + a⁴ + … + a^(2^(d−1))` takes its place, mapping each
    /// component onto 𝔽₂ independently; this is the standard
    /// characteristic-2 instance, not a special case of the odd formula.
    ///
    /// The number of factors is known in advance — `deg self / d`, since the
    /// factors all have degree `d` — so the loop terminates on a count
    /// rather than on a fixed point. That count is only correct under the
    /// stated precondition, which is unchecked: given a `self` that is not a
    /// product of distinct degree-`d` irreducibles the target is
    /// unreachable and the stall guard below fires, attributing the failure
    /// to the `Rng`.
    fn equal_degree_split<R: crate::random::Rng + ?Sized>(
        &self,
        d: usize,
        rng: &mut R,
    ) -> Vec<Self> {
        let total_degree = self.degree().expect("non-zero");
        if total_degree == d {
            return vec![self.make_monic()];
        }
        let target = total_degree / d;
        let mut factors = vec![self.make_monic()];
        let two = BigUint::from_u64(2);
        // Cantor–Zassenhaus is Las Vegas: each draw splits a given pair of
        // factors with probability ≥ 1/2, so the loop finishes quickly for any
        // Rng that produces entropy. Bound the consecutive draws that make no
        // progress so a dead or all-zero Rng fails loudly instead of spinning
        // forever — 256 fruitless draws has probability ≈ 2⁻²⁵⁶ for a working
        // source, so this can only fire on a broken one.
        const MAX_STALLED_DRAWS: usize = 256;
        let mut stalled = 0usize;
        // Neither the splitting exponent nor the constant 1 depends on the
        // draw, so both are built once rather than per attempt. The exponent
        // costs a binary exponentiation to a d·log₂p-bit integer, which is the
        // most expensive thing in the loop that is not a polynomial operation.
        let one_poly = Self::new(vec![BigUint::one()], &self.modulus);
        let cz_exponent = if self.modulus == two {
            None
        } else {
            // (pᵈ − 1)/2. The dividend is even (pᵈ is odd for odd p), and
            // halving is a one-bit shift — no reason to spend a full
            // multiprecision division on it.
            let pd = self.modulus.pow_u64(u64::try_from(d).expect("d fits u64"));
            let mut half = pd.sub_ref(&BigUint::one());
            half.shr1();
            Some(half)
        };
        while factors.len() < target {
            assert!(
                stalled < MAX_STALLED_DRAWS,
                "equal-degree split made no progress in {MAX_STALLED_DRAWS} draws: \
                 either the Rng yields no entropy, or the input is not a product \
                 of distinct degree-{d} irreducibles as this step requires"
            );
            let before = factors.len();
            // A random splitter of degree < deg self.
            let a = self.random_below_degree(rng);
            if a.is_zero() {
                stalled += 1;
                continue;
            }
            // g captures roughly half the factors: over 𝔽₂ by the trace map,
            // otherwise by a^((pᵈ−1)/2) − 1.
            let g = if self.modulus == two {
                // `a` comes from random_below_degree, which fills exactly
                // deg(self) coefficients, so it is already reduced.
                let mut trace = a.clone();
                let mut term = trace.clone();
                for _ in 1..d {
                    term = term.mul(&term).rem(self);
                    trace = trace.add(&term);
                }
                self.gcd(&trace)
            } else {
                let exponent = cz_exponent
                    .as_ref()
                    .expect("odd modulus takes the exponent branch");
                let ae = a.pow_mod(exponent, self);
                let ae_minus_one = ae.sub(&one_poly);
                self.gcd(&ae_minus_one)
            };
            // Refine every current factor by g.
            let mut refined = Vec::with_capacity(factors.len());
            for factor in factors {
                let piece = factor.gcd(&g);
                if piece.degree().is_some_and(|dp| dp >= 1) && piece.degree() < factor.degree() {
                    let other = factor.div_rem(&piece).0;
                    refined.push(piece.make_monic());
                    refined.push(other.make_monic());
                } else {
                    refined.push(factor);
                }
            }
            factors = refined;
            if factors.len() > before {
                stalled = 0;
            } else {
                stalled += 1;
            }
        }
        factors
    }

    /// A random polynomial of degree below `self`'s, over the modulus:
    /// `deg self` coefficients drawn uniformly from `[0, m)`, filling
    /// positions `0 .. deg self − 1`.
    ///
    /// The draw may be the zero polynomial, and may have degree strictly
    /// below `deg self − 1` when high coefficients come up zero; both are
    /// legitimate elements of the residue ring and the caller treats a zero
    /// draw as a stalled round. A `random_below` failure — reachable only
    /// for a zero bound, which `m ≥ 2` excludes — is folded to zero for the
    /// same reason. `random_below` itself panics on a generator whose
    /// output is confined to `[m, 2^bits)` (its own stall guard), which is
    /// the one way a broken `Rng` aborts here instead of reaching the
    /// stall assertion in `equal_degree_split`.
    fn random_below_degree<R: crate::random::Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let deg = self.degree().expect("non-zero");
        let coeffs = (0..deg)
            .map(|_| crate::random::random_below(rng, &self.modulus).unwrap_or_else(BigUint::zero))
            .collect();
        Self::new(coeffs, &self.modulus)
    }

    /// Whether `self` is irreducible over 𝔽ₚ. A non-constant polynomial is
    /// irreducible exactly when its distinct-degree factorization is a
    /// single block whose degree equals its own — deterministic, no
    /// randomness: only `distinct_degree` is used, never the randomized
    /// equal-degree split, since separating the factors within a block is
    /// unnecessary when only their count is in question.
    ///
    /// Squarefreeness is tested first and separately, because
    /// distinct-degree factorization presumes it: a repeated factor makes
    /// `gcd(f, f')` non-constant, and such an `f` is reducible (its repeated
    /// factor is a proper divisor). The zero polynomial and the constants
    /// are not irreducible by definition — units and zero are excluded from
    /// the notion — and return `false`.
    ///
    /// # Panics
    ///
    /// Panics on a non-invertible pivot during division. A prime modulus is
    /// required (see the type documentation): over a composite modulus the
    /// answer is unspecified and may be returned without a panic — `x² + 1`,
    /// for one, is reported irreducible modulo 15.
    #[must_use]
    pub fn is_irreducible(&self) -> bool {
        let Some(degree) = self.degree() else {
            return false; // zero polynomial
        };
        if degree == 0 {
            return false; // units and constants are not irreducible
        }
        let monic = self.make_monic();
        // Squarefree is necessary; a repeated factor makes gcd(f, f') > 1.
        if monic.gcd(&monic.derivative_modp()).degree() != Some(0) {
            return false;
        }
        let dd = monic.distinct_degree();
        dd.len() == 1 && dd[0].0 == degree
    }

    /// Complete factorization over 𝔽ₚ into monic irreducibles with
    /// multiplicities: `(factor, e)` with `∏ factorᵉ` equal to the monic
    /// form of `self`. The leading coefficient is not recoverable from the
    /// factor list, which is why the identity is stated against the monic
    /// form.
    ///
    /// Three stages, each narrowing what the next must handle: squarefree
    /// decomposition separates the multiplicities, distinct-degree groups
    /// the remaining factors by degree, and the randomized equal-degree
    /// split separates the factors within one group. That split is Las
    /// Vegas: it draws until each factor separates, so `rng` must yield
    /// entropy. A source that never does (e.g. all-zero bytes) makes no
    /// progress, and the split panics after a bounded number of fruitless
    /// draws rather than looping forever.
    ///
    /// # Panics
    ///
    /// Panics if `self` is constant or zero, on a non-invertible pivot during
    /// division, or if `rng` produces no entropy (the split stalls). A prime
    /// modulus is required (see the type documentation); over a composite
    /// modulus the result is unspecified and may be returned without a panic.
    #[must_use]
    pub fn factor<R: crate::random::Rng + ?Sized>(&self, rng: &mut R) -> Vec<(Self, usize)> {
        let mut result = Vec::new();
        for (squarefree, mult) in self.squarefree_factorization() {
            for (degree, block) in squarefree.distinct_degree() {
                for irreducible in block.equal_degree_split(degree, rng) {
                    result.push((irreducible, mult));
                }
            }
        }
        result
    }

    /// The roots of `self` in 𝔽ₚ, ascending and without repetition, each a
    /// residue `r` with `self(r) ≡ 0`.
    ///
    /// Computed as the linear factors rather than by trial evaluation, whose
    /// cost would be proportional to `p`. Since `xᵖ − x` is the product of
    /// `(x − r)` over every `r ∈ 𝔽ₚ`, `gcd(self, xᵖ − x)` is the product of
    /// `(x − r)` over exactly the roots of `self`, squarefree because
    /// `xᵖ − x` is. That product is then broken into linear factors by the
    /// same Las Vegas equal-degree split as [`Self::factor`] at `d = 1`, and
    /// each root read off as the negated constant term. Multiplicity is
    /// discarded: a repeated root is reported once.
    ///
    /// A constant has no roots and yields the empty list. So does the zero
    /// polynomial, which is the one place the return value is a convention
    /// rather than an answer — every residue is a root of it.
    ///
    /// # Panics
    ///
    /// Panics if `rng` produces no entropy: the equal-degree split cannot
    /// make progress and gives up after a bounded number of fruitless draws
    /// rather than looping forever. Also panics on a non-invertible pivot
    /// during division. A prime modulus is required (see the type
    /// documentation); over a composite modulus the result is unspecified
    /// and may be returned without a panic.
    #[must_use]
    pub fn roots<R: crate::random::Rng + ?Sized>(&self, rng: &mut R) -> Vec<BigUint> {
        if self.degree().is_none_or(|d| d == 0) {
            return Vec::new();
        }
        let monic = self.make_monic();
        // x^p − x mod f, then gcd with f: the product of (x − r) over roots r.
        let x = Self::monomial_x(&self.modulus);
        let xp = x.pow_mod(&self.modulus, &monic);
        let linear_product = monic.gcd(&xp.sub(&x));
        if linear_product.degree().is_none_or(|d| d == 0) {
            return Vec::new();
        }
        let mut roots = Vec::new();
        for factor in linear_product.equal_degree_split(1, rng) {
            // factor = x − r (monic linear); the root is −(constant term).
            let constant = factor
                .coefficients()
                .first()
                .cloned()
                .unwrap_or_else(BigUint::zero);
            let root = BigUint::mod_sub(&BigUint::zero(), &constant, &self.modulus);
            roots.push(root);
        }
        roots.sort();
        roots
    }

    /// The lift to `ℤ[x]` taking each coefficient to its representative of
    /// least absolute value, in `(−m/2, m/2]`.
    ///
    /// [`coefficients`](Self::coefficients) exposes the canonical
    /// representatives in `[0, m)`, which is the right normal form for
    /// arithmetic and the wrong one for size. A residue class near `m` is a
    /// small negative number wearing a large positive costume: the balanced
    /// representative is the one whose height reflects the element. That is
    /// what makes this the lift to use when a modular computation is meant to
    /// have recovered an integer answer — one lifts, and checks the answer
    /// over `ℤ`, and a modulus wide enough for the true coefficients to fit in
    /// the symmetric range returns them exactly.
    ///
    /// The same balanced convention as
    /// [`PolyZ::balanced_base_expansion`](PolyZ::balanced_base_expansion), for
    /// the same reason.
    #[must_use]
    pub fn symmetric_lift(&self) -> PolyZ {
        PolyZ::new(
            self.coeffs
                .iter()
                .map(|c| BigInt::from_biguint(c.clone()).symmetric_remainder(&self.modulus))
                .collect(),
        )
    }

    /// The same coefficient *representatives*, read modulo a different
    /// modulus.
    ///
    /// It takes each class to its canonical representative in `[0, m)`, an
    /// integer, and reduces that integer modulo the new modulus. Which of
    /// the two divisibility directions holds decides what survives, and
    /// they are not symmetric:
    ///
    /// - **Narrowing**, `m′ | m`: this is the projection `ℤ/m → ℤ/m′`, a
    ///   ring homomorphism. Sums and products agree computed before or
    ///   after.
    /// - **Widening**, `m | m′`: this is the canonical *section* of that
    ///   projection, and a section between rings of different size is never
    ///   additive. Concretely at `m = 7`, `m′ = 49`: `5 + 4 = 2` in `ℤ/7`
    ///   widens to `2`, while widening first gives `5 + 4 = 9`. Only
    ///   equality survives — two elements equal before are equal after.
    ///
    /// Widening is nonetheless what this exists for. It re-reads a solution
    /// modulo `m` as a starting point modulo `m′`, correct to the old
    /// precision and arbitrary beyond it, which is the seeding step of a
    /// Newton lift: each round squares the modulus and the correction term
    /// repairs the newly exposed digits. The arithmetic that follows the
    /// seeding is done at `m′` throughout, so the section's failure to
    /// commute with `+` is not something the lift relies on — but a caller
    /// who assumes it does will get a silently wrong answer rather than a
    /// panic, which is why the direction is spelled out.
    #[must_use]
    pub fn with_modulus(&self, modulus: &BigUint) -> Self {
        Self::new(self.coeffs.clone(), modulus)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        convolve_modp, convolve_schoolbook_modp, convolve_schoolbook_z, convolve_z, PolyModP,
        PolyZ, POLY_KARATSUBA_THRESHOLD_MODP, POLY_KARATSUBA_THRESHOLD_Z,
    };
    use crate::bigint::{BigInt, BigUint, Sign};

    struct SplitMix64 {
        state: u64,
    }
    impl crate::random::Rng for SplitMix64 {
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            let mut i = 0;
            while i < dest.len() {
                let word = self.next_u64().to_le_bytes();
                let take = (dest.len() - i).min(8);
                dest[i..i + take].copy_from_slice(&word[..take]);
                i += take;
            }
        }
    }

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        // A coefficient in [-range, range].
        fn coeff(&mut self, range: i64) -> BigInt {
            let m = 2 * range as u64 + 1;
            let v = (self.next_u64() % m) as i64 - range;
            BigInt::from_i64(v)
        }
        fn poly_z(&mut self, max_deg: usize, range: i64) -> PolyZ {
            let len = (self.next_u64() as usize % (max_deg + 1)) + 1;
            PolyZ::new((0..len).map(|_| self.coeff(range)).collect())
        }
        // Monic of exactly the given degree — the shape `rem_monic` and the
        // quotient-ring routines require.
        fn monic_z(&mut self, degree: usize, range: i64) -> PolyZ {
            let mut coeffs: Vec<BigInt> = (0..degree).map(|_| self.coeff(range)).collect();
            coeffs.push(BigInt::one());
            PolyZ::new(coeffs)
        }
        // An unsigned integer of the given number of 64-bit words.
        fn biguint(&mut self, words: usize) -> BigUint {
            let radix = BigUint::from_u64(1u64 << 32);
            let mut v = BigUint::zero();
            for _ in 0..2 * words {
                v = v
                    .mul_ref(&radix)
                    .add_ref(&BigUint::from_u64(self.next_u64() >> 32));
            }
            v
        }
    }

    // Naive schoolbook multiply as an independent oracle.
    fn naive_mul(a: &PolyZ, b: &PolyZ) -> PolyZ {
        if a.is_zero() || b.is_zero() {
            return PolyZ::zero();
        }
        let mut coeffs = vec![BigInt::zero(); a.coefficients().len() + b.coefficients().len() - 1];
        for (i, x) in a.coefficients().iter().enumerate() {
            for (j, y) in b.coefficients().iter().enumerate() {
                coeffs[i + j] = coeffs[i + j].add_ref(&x.mul_ref(y));
            }
        }
        PolyZ::new(coeffs)
    }

    #[test]
    fn karatsuba_convolution_matches_schoolbook_across_shapes() {
        // The split kernel against the quadratic one it replaces, at the
        // shapes where a split rule can go wrong: exactly at the threshold,
        // one on either side of it, lopsided pairs (where the shorter
        // operand may not reach across the split point), odd lengths that
        // make the halves unequal, and sizes deep enough to recurse several
        // levels. Both coefficient rings, and a modulus small enough that
        // reductions actually fire.
        let mut rng = SplitMix64 {
            state: 0x4b17_5aba_0001,
        };
        let t = POLY_KARATSUBA_THRESHOLD_Z;
        let lens = [
            1usize,
            2,
            t - 1,
            t,
            t + 1,
            2 * t - 1,
            2 * t,
            2 * t + 1,
            3 * t + 7,
            4 * t,
            129,
        ];
        let modulus = BigUint::from_u64(97);
        for &la in &lens {
            for &lb in &lens {
                let a: Vec<BigInt> = (0..la).map(|_| rng.coeff(1_000_000)).collect();
                let b: Vec<BigInt> = (0..lb).map(|_| rng.coeff(1_000_000)).collect();
                assert_eq!(
                    convolve_z(&a, &b),
                    convolve_schoolbook_z(&a, &b),
                    "PolyZ convolution differs at {la}x{lb}"
                );

                // The modular split logic gets the same shape sweep, forced
                // one level at a time: its own dispatch threshold is an
                // order of magnitude higher (a measured property of the
                // ring, not of the algebra), so waiting for dispatch would
                // leave the split rule untested at every shape that can
                // break it.
                let am: Vec<BigUint> = (0..la)
                    .map(|_| BigUint::from_u64(rng.next_u64() % 97))
                    .collect();
                let bm: Vec<BigUint> = (0..lb)
                    .map(|_| BigUint::from_u64(rng.next_u64() % 97))
                    .collect();
                assert_eq!(
                    karatsuba_forced_modp(&am, &bm, &modulus),
                    convolve_schoolbook_modp(&am, &bm, &modulus),
                    "PolyModP forced split differs at {la}x{lb}"
                );
            }
        }

        // And the real modular dispatch at its own threshold, where the
        // recursion actually engages: three shapes only, because these are
        // genuinely large convolutions.
        let tm = POLY_KARATSUBA_THRESHOLD_MODP;
        // The balance guard admits up to 3:2 and rejects beyond it; a
        // rejected shape still has to compute the right answer, so both
        // sides of the guard appear here.
        let any = super::POLY_KARATSUBA_ANY_RATIO_MODP;
        let admitted = |short, long| super::poly_split_admitted(short, long, tm, any);
        assert!(admitted(tm, tm));
        assert!(admitted(tm, 5 * tm / 4));
        assert!(!admitted(tm, 2 * tm - 1));
        // Above the any-ratio size the balance clause lifts.
        assert!(admitted(any, 2 * any - 1));
        assert!(!admitted(tm - 1, tm - 1));
        for &(la, lb) in &[(tm, tm), (tm + 1, tm), (3 * tm / 2, tm), (2 * tm - 1, tm)] {
            let am: Vec<BigUint> = (0..la)
                .map(|_| BigUint::from_u64(rng.next_u64() % 97))
                .collect();
            let bm: Vec<BigUint> = (0..lb)
                .map(|_| BigUint::from_u64(rng.next_u64() % 97))
                .collect();
            assert_eq!(
                convolve_modp(&am, &bm, &modulus),
                convolve_schoolbook_modp(&am, &bm, &modulus),
                "PolyModP dispatched convolution differs at {la}x{lb}"
            );
        }
    }

    #[test]
    fn squaring_convolution_matches_the_general_one() {
        // The dedicated squaring against the general convolution it
        // replaces, including the shapes where the doubling could go wrong:
        // a single coefficient (no cross terms at all), two (one cross
        // term), interior zeros (whose inner passes are skipped), and a
        // modulus of 2, where doubling annihilates every cross term and the
        // result is the Frobenius image.
        let mut rng = SplitMix64 {
            state: 0x5111_1ee0_0001,
        };
        for &m in &[2u64, 3, 97, 1_000_003] {
            let modulus = BigUint::from_u64(m);
            for len in 1usize..24 {
                let mut a: Vec<BigUint> = (0..len)
                    .map(|_| BigUint::from_u64(rng.next_u64() % m))
                    .collect();
                if len > 3 {
                    a[len / 2] = BigUint::zero();
                }
                assert_eq!(
                    super::convolve_square_modp(&a, &modulus),
                    convolve_schoolbook_modp(&a, &a, &modulus),
                    "squaring differs at length {len}, modulus {m}"
                );
            }
        }

        // The split path, which only dispatches above its own threshold,
        // and the characteristic-2 spread at a length that reaches it —
        // both against the general convolution.
        let ts = POLY_KARATSUBA_THRESHOLD_MODP;
        for &m in &[2u64, 97] {
            let modulus = BigUint::from_u64(m);
            for &len in &[ts, ts + 1, 2 * ts + 3] {
                let a: Vec<BigUint> = (0..len)
                    .map(|_| BigUint::from_u64(rng.next_u64() % m))
                    .collect();
                assert_eq!(
                    super::convolve_square_modp(&a, &modulus),
                    convolve_schoolbook_modp(&a, &a, &modulus),
                    "split squaring differs at length {len}, modulus {m}"
                );
            }
        }
    }

    #[test]
    #[ignore = "timing probe for the polynomial thresholds; run with --ignored"]
    fn poly_karatsuba_crossover_timing() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        // Repetitions are calibrated, not guessed: an earlier revision of
        // this probe fixed a floor of four repetitions, which at 384
        // coefficients meant four samples of a millisecond-scale operation
        // and put the modular crossover four times too high. Each chunk is
        // now sized to a target duration from a measured single run.
        fn calibrate(target: Duration, f: &mut dyn FnMut()) -> u32 {
            let t = Instant::now();
            f();
            let once = t.elapsed().as_secs_f64().max(1e-9);
            ((target.as_secs_f64() / once).ceil() as u64).clamp(1, 1_000_000) as u32
        }

        // Paired interleaved chunks with the order alternated between
        // passes, so neither kernel systematically runs on a warmer cache.
        fn paired_saving(chunk: u32, flip: bool, a: &mut dyn FnMut(), b: &mut dyn FnMut()) -> f64 {
            let (mut ta, mut tb) = (0f64, 0f64);
            for _ in 0..3 {
                if flip {
                    let t = Instant::now();
                    for _ in 0..chunk {
                        b();
                    }
                    tb += t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    for _ in 0..chunk {
                        a();
                    }
                    ta += t.elapsed().as_secs_f64();
                } else {
                    let t = Instant::now();
                    for _ in 0..chunk {
                        a();
                    }
                    ta += t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    for _ in 0..chunk {
                        b();
                    }
                    tb += t.elapsed().as_secs_f64();
                }
            }
            (ta - tb) / ta * 100.0
        }

        fn sweep(label: &str, a: &mut dyn FnMut(), b: &mut dyn FnMut()) {
            let chunk = calibrate(Duration::from_millis(20), a);
            let mut passes = [0f64; 5];
            for (k, slot) in passes.iter_mut().enumerate() {
                *slot = paired_saving(chunk, k % 2 == 1, a, b);
            }
            let mut sorted = passes;
            sorted.sort_by(f64::total_cmp);
            eprintln!(
                "{label:<34} {chunk:>7} {:>+7.1}%   {:+6.1} {:+6.1} {:+6.1} {:+6.1} {:+6.1}",
                sorted[2], passes[0], passes[1], passes[2], passes[3], passes[4]
            );
        }

        let mut rng = SplitMix64 {
            state: 0x0c0f_fee0_0bad_0001,
        };
        let small = BigUint::from_u64(1_000_003);
        let mut wide = BigUint::one();
        wide.shl_bits(255);
        let wide = wide.add_ref(&BigUint::from_u64(235));
        let draw_wide = |rng: &mut SplitMix64, m: &BigUint| {
            let mut v = BigUint::zero();
            for k in 0..4 {
                let mut w = BigUint::from_u64(rng.next_u64());
                w.shl_bits(64 * k);
                v = v.add_ref(&w);
            }
            v.modulo(m)
        };

        eprintln!("Karatsuba saving over schoolbook (positive = split wins)");
        eprintln!("{:<34} {:>7} {:>8}   passes", "shape", "reps", "median");
        for &short in &[32usize, 64, 96, 128, 192, 256, 384, 512, 768] {
            // Ratios the dispatcher can actually admit. Exactly 2:1 is
            // omitted: the split point lands on the shorter operand's
            // length, both sides bail to schoolbook, and the row would
            // compare a kernel with itself.
            for &(num, den) in &[(1usize, 1usize), (5, 4), (3, 2)] {
                let long = short * num / den;
                for &(rlabel, long) in &[("", long), ("(2s-1):s", 2 * short - 1)] {
                    if !rlabel.is_empty() && num != 1 {
                        continue; // draw the lopsided shape once per size
                    }
                    let za: Vec<BigInt> = (0..long).map(|_| rng.coeff(i64::MAX / 4)).collect();
                    let zb: Vec<BigInt> = (0..short).map(|_| rng.coeff(i64::MAX / 4)).collect();
                    let ratio = long as f64 / short as f64;
                    sweep(
                        &format!("Z    {short:>5}x{long:<5} {ratio:>4.2}"),
                        &mut || {
                            black_box(convolve_schoolbook_z(black_box(&za), black_box(&zb)));
                        },
                        &mut || {
                            black_box(karatsuba_forced_z(black_box(&za), black_box(&zb)));
                        },
                    );
                    for (mlabel, m) in [("20b", &small), ("256b", &wide)] {
                        let ma: Vec<BigUint> = (0..long).map(|_| draw_wide(&mut rng, m)).collect();
                        let mb: Vec<BigUint> = (0..short).map(|_| draw_wide(&mut rng, m)).collect();
                        sweep(
                            &format!("modp/{mlabel:<4}{short:>5}x{long:<5} {ratio:>4.2}"),
                            &mut || {
                                black_box(convolve_schoolbook_modp(
                                    black_box(&ma),
                                    black_box(&mb),
                                    m,
                                ));
                            },
                            &mut || {
                                black_box(karatsuba_forced_modp(black_box(&ma), black_box(&mb), m));
                            },
                        );
                    }
                }
            }
        }

        // The squaring split has its own crossover: its sub-problems are
        // squares, so the curve is not the product's and cannot be
        // borrowed from it.
        eprintln!();
        eprintln!("split squaring saving over the cross-terms-once square");
        eprintln!("{:<34} {:>7} {:>8}   passes", "shape", "reps", "median");
        for &len in &[64usize, 96, 128, 192, 256, 384, 512, 768] {
            for (mlabel, m) in [("20b", &small), ("256b", &wide)] {
                let a: Vec<BigUint> = (0..len).map(|_| draw_wide(&mut rng, m)).collect();
                sweep(
                    &format!("sqr/{mlabel:<5}{len:>5}"),
                    &mut || {
                        black_box(square_forced_schoolbook_modp(black_box(&a), m));
                    },
                    &mut || {
                        black_box(square_forced_split_modp(black_box(&a), m));
                    },
                );
            }
        }
    }

    /// The cross-terms-once square without the dispatch in front of it, so
    /// the probe can compare the two kernels at any size.
    #[cfg(test)]
    fn square_forced_schoolbook_modp(a: &[BigUint], m: &BigUint) -> Vec<BigUint> {
        let n = a.len();
        let mut acc = vec![BigUint::zero(); 2 * n - 1];
        for i in 0..n {
            if a[i].is_zero() {
                continue;
            }
            acc[2 * i].add_assign_ref(&a[i].square_ref());
            for j in (i + 1)..n {
                if a[j].is_zero() {
                    continue;
                }
                let mut cross = a[i].mul_ref(&a[j]);
                cross.shl1();
                acc[i + j].add_assign_ref(&cross);
            }
        }
        for slot in &mut acc {
            *slot = slot.modulo(m);
        }
        acc
    }

    /// One forced split level of the squaring, likewise.
    #[cfg(test)]
    fn square_forced_split_modp(a: &[BigUint], m: &BigUint) -> Vec<BigUint> {
        let n = a.len();
        let split = n / 2;
        if split == 0 {
            return square_forced_schoolbook_modp(a, m);
        }
        let (a0, a1) = a.split_at(split);
        let z0 = super::convolve_square_modp(a0, m);
        let z2 = super::convolve_square_modp(a1, m);
        let sum = super::add_coeffs_modp(a0, a1, m);
        let mut z1 = super::convolve_square_modp(&sum, m);
        super::sub_assign_coeffs_modp(&mut z1, &z0, m);
        super::sub_assign_coeffs_modp(&mut z1, &z2, m);
        let mut out = vec![BigUint::zero(); 2 * n - 1];
        super::add_into_at_modp(&mut out, &z0, 0, m);
        super::add_into_at_modp(&mut out, &z1, split, m);
        super::add_into_at_modp(&mut out, &z2, 2 * split, m);
        out
    }

    /// One forced Karatsuba level regardless of the threshold, so the probe
    /// above compares the two kernels at sizes where dispatch would not
    /// normally choose the split.
    #[cfg(test)]
    #[test]
    fn split_gates_never_change_the_answer() {
        // The dispatch gates decide *which* kernel runs, never what it
        // computes. This sweeps the matrix each gate turns on — lengths
        // straddling both size thresholds, ratios straddling the balance
        // clause, densities straddling the computed density cut — and
        // demands the dispatched product equal the schoolbook one every
        // time. Both rings, and for the modular side several moduli, since
        // the deferred reduction makes the accumulator bound depend on the
        // modulus width.
        let mut rng = SplitMix64 {
            state: 0x9051_7a7e_0011,
        };
        let tz = POLY_KARATSUBA_THRESHOLD_Z;
        let az = super::POLY_KARATSUBA_ANY_RATIO_Z;
        let tm = POLY_KARATSUBA_THRESHOLD_MODP;
        let am = super::POLY_KARATSUBA_ANY_RATIO_MODP;
        // Lengths just under, at, and just over each threshold, plus sizes
        // deep enough for the density cut to have fallen well below 3/4.
        let lengths = [
            tz - 1,
            tz,
            tz + 1,
            az - 1,
            az,
            az + 1,
            tm - 1,
            tm,
            tm + 1,
            am - 1,
            am,
            am + 1,
            300,
            513,
        ];
        // Densities as percentages, straddling the retired fixed 3/4 cut
        // and the computed cut at each size.
        let densities = [100usize, 80, 76, 75, 74, 50, 34, 32, 24, 10, 2];
        let moduli = [
            BigUint::from_u64(2),
            BigUint::from_u64(97),
            BigUint::from_u64(1_048_573),
            BigUint::from_u64(2).pow_u64(200).sub_ref(&BigUint::one()),
        ];

        let sparse_z = |len: usize, pct: usize, rng: &mut SplitMix64| -> Vec<BigInt> {
            (0..len)
                .map(|i| {
                    if i + 1 == len || (rng.next_u64() % 100) < pct as u64 {
                        rng.coeff(1_000_000)
                    } else {
                        BigInt::zero()
                    }
                })
                .collect()
        };
        for &la in &lengths {
            for ratio in [1.0f64, 1.25, 1.5, 1.97] {
                let lb = ((la as f64) * ratio) as usize;
                for &pct in &densities {
                    let a = sparse_z(la, pct, &mut rng);
                    let b = sparse_z(lb, pct, &mut rng);
                    assert_eq!(
                        convolve_z(&a, &b),
                        convolve_schoolbook_z(&a, &b),
                        "Z dispatch differs at {la}x{lb}, density {pct}%"
                    );
                    // The kernel itself, on shapes the gate refuses — so a
                    // gate that wrongly admits one is still caught by the
                    // arithmetic, not only by the timing.
                    assert_eq!(
                        karatsuba_forced_z(&a, &b),
                        convolve_schoolbook_z(&a, &b),
                        "Z forced split differs at {la}x{lb}, density {pct}%"
                    );
                }
            }
        }
        for &la in &lengths {
            for ratio in [1.0f64, 1.5, 1.97] {
                let lb = ((la as f64) * ratio) as usize;
                for &pct in &[100usize, 76, 74, 40, 2] {
                    for m in &moduli {
                        // The widest modulus at the longest lengths is the
                        // one expensive corner of the sweep and adds no
                        // shape the narrower moduli have not already
                        // covered; the accumulator bound it exercises is
                        // covered at every length below 300.
                        if m.bits() > 64 && la >= 300 {
                            continue;
                        }
                        let a: Vec<BigUint> = (0..la)
                            .map(|i| {
                                if i + 1 == la || (rng.next_u64() % 100) < pct as u64 {
                                    BigUint::from_u64(rng.next_u64()).modulo(m)
                                } else {
                                    BigUint::zero()
                                }
                            })
                            .collect();
                        let b: Vec<BigUint> = (0..lb)
                            .map(|i| {
                                if i + 1 == lb || (rng.next_u64() % 100) < pct as u64 {
                                    BigUint::from_u64(rng.next_u64()).modulo(m)
                                } else {
                                    BigUint::zero()
                                }
                            })
                            .collect();
                        assert_eq!(
                            convolve_modp(&a, &b, m),
                            convolve_schoolbook_modp(&a, &b, m),
                            "modp dispatch differs at {la}x{lb}, density {pct}%, {} bits",
                            m.bits()
                        );
                        assert_eq!(
                            karatsuba_forced_modp(&a, &b, m),
                            convolve_schoolbook_modp(&a, &b, m),
                            "modp forced split differs at {la}x{lb}, density {pct}%, {} bits",
                            m.bits()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn split_thresholds_are_the_values_that_were_measured() {
        // Pinned literals, deliberately. Every one of these is a
        // measurement, recorded in the tables on the constants themselves,
        // and a threshold can drift to a wrong value without changing a
        // single computed answer — so nothing else in the suite can notice.
        // Four of them have already been wrong once each. Changing one here
        // is meant to be work: re-run `poly_karatsuba_crossover_timing`
        // with `--ignored`, update the table it feeds, then update this.
        //
        // An earlier version of this test asserted only that
        // `poly_split_admitted` was self-consistent, threading each constant
        // in as both the shape and the parameter. That is a tautology in the
        // constant and caught none of the four.
        assert_eq!(POLY_KARATSUBA_THRESHOLD_Z, 96);
        assert_eq!(super::POLY_KARATSUBA_ANY_RATIO_Z, 128);
        assert_eq!(POLY_KARATSUBA_THRESHOLD_MODP, 128);
        assert_eq!(super::POLY_KARATSUBA_ANY_RATIO_MODP, 192);
        assert_eq!(super::POLY_SQUARE_SPLIT_THRESHOLD_MODP, 192);
    }

    #[test]
    fn split_gate_decisions_are_the_measured_ones() {
        use super::{
            karatsuba_products_estimate, karatsuba_square_products_estimate, poly_split_admitted,
            poly_split_dense_enough, poly_square_split_dense_enough,
            POLY_KARATSUBA_ANY_RATIO_MODP as AM, POLY_KARATSUBA_ANY_RATIO_Z as AZ,
            POLY_KARATSUBA_THRESHOLD_MODP as TM, POLY_KARATSUBA_THRESHOLD_Z as TZ,
        };

        // Balance clause: 5:4 admitted at the size threshold, 3:2 and 2:1
        // not, in both rings — the measured shape of both tables. Written
        // against literal shapes so the assertions are not tautologies in
        // the constants they exercise.
        assert!(poly_split_admitted(96, 96, TZ, AZ));
        assert!(poly_split_admitted(96, 120, TZ, AZ));
        assert!(!poly_split_admitted(96, 144, TZ, AZ));
        assert!(!poly_split_admitted(96, 191, TZ, AZ)); // measured -3%
        assert!(!poly_split_admitted(95, 95, TZ, AZ));
        assert!(poly_split_admitted(128, 255, TZ, AZ)); // measured +17%
        assert!(poly_split_admitted(128, 128, TM, AM));
        assert!(poly_split_admitted(128, 160, TM, AM));
        assert!(!poly_split_admitted(128, 192, TM, AM)); // measured -3%
        assert!(!poly_split_admitted(128, 255, TM, AM)); // measured -13%
        assert!(!poly_split_admitted(127, 127, TM, AM));
        assert!(poly_split_admitted(192, 288, TM, AM)); // measured +9%
        assert!(poly_split_admitted(192, 383, TM, AM)); // measured +4%

        // The estimate counts the dispatcher's own recursion. These two
        // are hand-checked against a separate walk of `convolve_z`'s rules
        // — the shape, the split point, and the three sub-problems — so
        // they pin the count against an independent derivation rather than
        // against a restatement of the function.
        assert_eq!(karatsuba_products_estimate(128, 255, TZ, AZ), 24_513);
        assert_eq!(karatsuba_products_estimate(192, 288, TZ, AZ), 38_016);
        // A shape the gate refuses is charged the schoolbook count, since
        // that is what the dispatcher would actually run.
        assert_eq!(karatsuba_products_estimate(96, 191, TZ, AZ), 96 * 191);
        assert_eq!(karatsuba_products_estimate(50, 50, TZ, AZ), 2_500);

        // The density a shape must reach trends *down* with size — the
        // property the retired fixed cut could not express. It is not
        // monotone, and must not be asserted to be: where the halvings
        // land decides whether one more level of splitting is taken, so
        // neighbouring lengths genuinely differ. The trend is the claim,
        // so the trend is what is checked, on a grid that deliberately
        // includes sizes that are not multiples of the threshold.
        let required = |n: usize| -> f64 {
            karatsuba_products_estimate(n, n, TZ, AZ).min(n * n) as f64 / (n * n) as f64
        };
        assert!(
            (required(96) - 0.75).abs() < 1e-9,
            "3/4 exactly at the threshold"
        );
        for &(small, large) in &[(96usize, 384usize), (97, 385), (150, 1000), (191, 1537)] {
            assert!(
                required(large) < required(small) * 0.8,
                "the required density must fall with size: {small} vs {large}"
            );
        }
        assert!(required(2048) < 0.25, "2048 must ask far less than 3/4");

        // The dense floor must almost never bind. It bound over thousands
        // of admitted shapes when the estimate was a closed form, which
        // cost 18% for a single zero coefficient at 128x255; with the
        // counted recursion that shape asks for three quarters.
        let pinned: Vec<(usize, usize)> = (TZ..=260)
            .flat_map(|short| (short..=2 * short - 1).map(move |long| (short, long)))
            .filter(|&(short, long)| poly_split_admitted(short, long, TZ, AZ))
            .filter(|&(short, long)| {
                karatsuba_products_estimate(short, long, TZ, AZ) >= short * long
            })
            .collect();
        assert!(
            pinned.is_empty(),
            "{} admitted shapes are pinned at full density, e.g. {:?}",
            pinned.len(),
            &pinned[..pinned.len().min(3)]
        );
        // And concretely: one zero coefficient must not flip 128x255 to
        // schoolbook.
        assert!(poly_split_dense_enough(127, 128, 255, 255, TZ, AZ));
        assert!(poly_split_dense_enough(100, 128, 200, 255, TZ, AZ));

        // Full density is never refused on density grounds, at any shape.
        for &n in &[TZ, TM, 300, 1024, 2048] {
            assert!(poly_split_dense_enough(n, n, n, n, TZ, AZ));
            assert!(poly_split_dense_enough(n, n, 2 * n, 2 * n, TZ, AZ));
            assert!(poly_square_split_dense_enough(n, n));
        }
        // A two-term operand is refused at every size — the regression the
        // density clause exists to prevent, for products and for squares.
        for &n in &[64usize, 96, 128, 1024, 2048] {
            assert!(!poly_split_dense_enough(2, n, n, n, TZ, AZ));
        }
        for &n in &[192usize, 256, 512, 1024, 2048] {
            assert!(
                !poly_square_split_dense_enough(2, n),
                "a two-term square must not split at {n}"
            );
        }
        // And the band that the fixed cut cut in half is now admitted.
        for &(n, pct) in &[(1024usize, 40usize), (2048, 30), (2048, 74), (1024, 74)] {
            let nnz = n * pct / 100;
            assert!(
                poly_split_dense_enough(nnz, n, nnz, n, TZ, AZ),
                "{pct}% density at {n} must still split"
            );
        }
        // The square estimate is the square recursion, not the product's.
        assert_eq!(karatsuba_square_products_estimate(96, 192), 9_216);
        assert_eq!(karatsuba_square_products_estimate(192, 192), 3 * 96 * 96);
    }

    #[test]
    #[should_panic(expected = "monic")]
    fn rem_monic_rejects_a_non_monic_divisor() {
        let _ = PolyZ::from_i64_slice(&[1, 2, 3]).rem_monic(&PolyZ::from_i64_slice(&[1, 2]));
    }

    #[test]
    #[should_panic(expected = "zero polynomial")]
    fn rem_monic_rejects_the_zero_divisor() {
        let _ = PolyZ::from_i64_slice(&[1, 2, 3]).rem_monic(&PolyZ::zero());
    }

    #[test]
    #[should_panic(expected = "monic")]
    fn product_mod_monic_rejects_a_non_monic_divisor() {
        let f = PolyZ::from_i64_slice(&[1, 1]);
        let _ = PolyZ::product_mod_monic(&[f.clone(), f], &PolyZ::from_i64_slice(&[1, 2]));
    }

    #[test]
    #[should_panic(expected = "base of at least two")]
    fn balanced_expansion_rejects_a_base_below_two() {
        let _ = PolyZ::balanced_base_expansion(&BigInt::one(), &BigUint::one(), 3);
    }

    #[test]
    #[should_panic(expected = "positive exponent")]
    fn roots_mod_prime_power_rejects_a_zero_exponent() {
        let mut rng = SplitMix64 { state: 1 };
        let _ = PolyZ::from_i64_slice(&[1, 0, 1]).roots_mod_prime_power(
            &BigUint::from_u64(7),
            0,
            &mut rng,
        );
    }

    #[test]
    #[should_panic(expected = "must be a prime")]
    fn roots_mod_prime_power_rejects_a_base_below_two() {
        let mut rng = SplitMix64 { state: 1 };
        let _ =
            PolyZ::from_i64_slice(&[1, 0, 1]).roots_mod_prime_power(&BigUint::one(), 2, &mut rng);
    }

    fn karatsuba_forced_z(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
        let split = a.len().max(b.len()) / 2;
        if a.len() <= split || b.len() <= split || split == 0 {
            return convolve_schoolbook_z(a, b);
        }
        let (a0, a1) = a.split_at(split);
        let (b0, b1) = b.split_at(split);
        let z0 = convolve_z(a0, b0);
        let z2 = convolve_z(a1, b1);
        let a_sum = super::add_coeffs_z(a0, a1);
        let b_sum = super::add_coeffs_z(b0, b1);
        let mut z1 = convolve_z(&a_sum, &b_sum);
        super::sub_assign_coeffs_z(&mut z1, &z0);
        super::sub_assign_coeffs_z(&mut z1, &z2);
        let mut out = vec![BigInt::zero(); a.len() + b.len() - 1];
        super::add_into_at_z(&mut out, &z0, 0);
        super::add_into_at_z(&mut out, &z1, split);
        super::add_into_at_z(&mut out, &z2, 2 * split);
        out
    }

    #[cfg(test)]
    fn karatsuba_forced_modp(a: &[BigUint], b: &[BigUint], m: &BigUint) -> Vec<BigUint> {
        let split = a.len().max(b.len()) / 2;
        if a.len() <= split || b.len() <= split || split == 0 {
            return convolve_schoolbook_modp(a, b, m);
        }
        let (a0, a1) = a.split_at(split);
        let (b0, b1) = b.split_at(split);
        let z0 = convolve_modp(a0, b0, m);
        let z2 = convolve_modp(a1, b1, m);
        let a_sum = super::add_coeffs_modp(a0, a1, m);
        let b_sum = super::add_coeffs_modp(b0, b1, m);
        let mut z1 = convolve_modp(&a_sum, &b_sum, m);
        super::sub_assign_coeffs_modp(&mut z1, &z0, m);
        super::sub_assign_coeffs_modp(&mut z1, &z2, m);
        let mut out = vec![BigUint::zero(); a.len() + b.len() - 1];
        super::add_into_at_modp(&mut out, &z0, 0, m);
        super::add_into_at_modp(&mut out, &z1, split, m);
        super::add_into_at_modp(&mut out, &z2, 2 * split, m);
        out
    }

    #[test]
    fn poly_z_ring_axioms_and_evaluation() {
        let mut rng = SplitMix64 {
            state: 0x9017_0000_0001,
        };
        for _ in 0..500 {
            let a = rng.poly_z(6, 20);
            let b = rng.poly_z(6, 20);
            let c = rng.poly_z(6, 20);
            // Commutativity and the naive-multiply oracle.
            assert_eq!(a.mul(&b), b.mul(&a));
            assert_eq!(a.mul(&b), naive_mul(&a, &b));
            // Distributivity.
            assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
            // add/sub inverse.
            assert_eq!(a.add(&b).sub(&b), a);
            // The evaluation homomorphism: eval(a·b) = eval(a)·eval(b).
            for x in [-3i64, 0, 1, 7] {
                let xv = BigInt::from_i64(x);
                assert_eq!(
                    a.mul(&b).evaluate(&xv),
                    a.evaluate(&xv).mul_ref(&b.evaluate(&xv))
                );
                assert_eq!(
                    a.add(&b).evaluate(&xv),
                    a.evaluate(&xv).add_ref(&b.evaluate(&xv))
                );
            }
        }
    }

    #[test]
    fn poly_z_derivative_and_content() {
        // d/dx (3x^2 + 2x + 5) = 6x + 2.
        let p = PolyZ::from_i64_slice(&[5, 2, 3]);
        assert_eq!(p.derivative(), PolyZ::from_i64_slice(&[2, 6]));
        // Product rule on random polynomials.
        let mut rng = SplitMix64 {
            state: 0xde01_0000_0007,
        };
        for _ in 0..200 {
            let a = rng.poly_z(5, 15);
            let b = rng.poly_z(5, 15);
            // (a·b)' = a'·b + a·b'.
            assert_eq!(
                a.mul(&b).derivative(),
                a.derivative().mul(&b).add(&a.mul(&b.derivative()))
            );
        }
        // content(6x^2 + 9x + 15) = 3, primitive part = 2x^2 + 3x + 5.
        let q = PolyZ::from_i64_slice(&[15, 9, 6]);
        assert_eq!(q.content(), BigInt::from_i64(3));
        assert_eq!(q.primitive_part(), PolyZ::from_i64_slice(&[5, 3, 2]));
        // content · primitive_part = original.
        assert_eq!(q.primitive_part().scale(&q.content()), q);
    }

    #[test]
    fn poly_z_pseudo_division_identity() {
        let mut rng = SplitMix64 {
            state: 0x9500_0000_0001,
        };
        for _ in 0..1000 {
            let a = rng.poly_z(8, 12);
            let mut b = rng.poly_z(5, 12);
            if b.is_zero() {
                b = PolyZ::from_i64_slice(&[1]);
            }
            let (q, r) = a.pseudo_div_rem(&b);
            // deg r < deg b.
            if let (Some(dr), Some(db)) = (r.degree(), b.degree()) {
                assert!(dr < db, "remainder degree");
            }
            // lc(b)^(deg a − deg b + 1) · a = q·b + r, when deg a ≥ deg b.
            if let (Some(da), Some(db)) = (a.degree(), b.degree()) {
                if da >= db {
                    let exp = (da - db + 1) as u64;
                    let lc = b.leading_coefficient();
                    let mut lc_pow = BigInt::one();
                    for _ in 0..exp {
                        lc_pow = lc_pow.mul_ref(&lc);
                    }
                    assert_eq!(
                        a.scale(&lc_pow),
                        q.mul(&b).add(&r),
                        "pseudo-division identity"
                    );
                }
            }
        }
    }

    #[test]
    fn poly_z_exact_division_worked_cases() {
        // (x^2 + 3x + 2) = (x + 1)(x + 2): exact, zero remainder.
        let a = PolyZ::from_i64_slice(&[2, 3, 1]);
        let b = PolyZ::from_i64_slice(&[1, 1]); // x + 1
        let (q, r) = a.div_rem(&b).expect("exact division");
        assert_eq!(q, PolyZ::from_i64_slice(&[2, 1])); // x + 2
        assert!(r.is_zero());
        // (x^2 + 1) / (x + 1) = (x - 1) remainder 2 — monic divisor, always Some.
        let a = PolyZ::from_i64_slice(&[1, 0, 1]);
        let (q, r) = a.div_rem(&b).expect("monic divisor divides");
        assert_eq!(q, PolyZ::from_i64_slice(&[-1, 1])); // x - 1
        assert_eq!(r, PolyZ::from_i64_slice(&[2]));
        // 2x + 1 has leading coefficient 2, which does not divide 1: no ℤ
        // quotient for x^2 + 1.
        let two_x_plus_one = PolyZ::from_i64_slice(&[1, 2]);
        assert_eq!(
            PolyZ::from_i64_slice(&[1, 0, 1]).div_rem(&two_x_plus_one),
            None
        );
        // Dividend of smaller degree: quotient 0, remainder the dividend.
        let (q, r) = b.div_rem(&a).expect("smaller dividend");
        assert!(q.is_zero());
        assert_eq!(r, b);
    }

    #[test]
    #[should_panic(expected = "zero polynomial")]
    fn poly_z_div_rem_panics_on_zero_divisor() {
        let _ = PolyZ::from_i64_slice(&[1, 2, 3]).div_rem(&PolyZ::zero());
    }

    #[test]
    fn poly_z_div_rem_identity_and_agrees_with_pseudo_on_monic() {
        let mut rng = SplitMix64 {
            state: 0x0d1f_0000_0001,
        };
        for _ in 0..1000 {
            let a = rng.poly_z(8, 12);
            let b = rng.poly_z(5, 12);
            // When exact division exists, the identity must hold with a
            // remainder of lower degree.
            if !b.is_zero() {
                if let Some((q, r)) = a.div_rem(&b) {
                    assert_eq!(q.mul(&b).add(&r), a, "exact division identity");
                    if let (Some(dr), Some(db)) = (r.degree(), b.degree()) {
                        assert!(dr < db, "remainder degree");
                    }
                }
            }
            // A monic divisor always divides over ℤ, and there exact division
            // coincides with pseudo-division (ℓ = 1). The two are separate
            // code paths, so agreement cross-checks both.
            let deg = 1 + (rng.next_u64() % 4) as usize;
            let mut coeffs: Vec<i64> = (0..deg).map(|_| (rng.next_u64() % 13) as i64 - 6).collect();
            coeffs.push(1); // monic leading coefficient
            let monic = PolyZ::from_i64_slice(&coeffs);
            let exact = a.div_rem(&monic).expect("monic divisor divides");
            let pseudo = a.pseudo_div_rem(&monic);
            assert_eq!(
                exact, pseudo,
                "monic: exact division equals pseudo-division"
            );
        }
    }

    #[test]
    fn poly_mod_p_division_and_gcd() {
        let p = BigUint::from_u64(101);
        let mut rng = SplitMix64 {
            state: 0x4001_0000_0001,
        };
        let to_mod = |poly: &PolyZ| PolyModP::from_poly_z(poly, &p);
        for _ in 0..500 {
            let a = to_mod(&rng.poly_z(8, 200));
            let mut b = to_mod(&rng.poly_z(5, 200));
            if b.is_zero() {
                b = PolyModP::new(vec![BigUint::one()], &p);
            }
            let (q, r) = a.div_rem(&b);
            // a = q·b + r, deg r < deg b.
            assert_eq!(q.mul(&b).add(&r), a, "division identity mod p");
            if let (Some(dr), Some(db)) = (r.degree(), b.degree()) {
                assert!(dr < db);
            }
            // gcd divides both and is monic.
            let g = a.gcd(&b);
            if !g.is_zero() {
                assert!(g.leading_coefficient().is_one(), "gcd is monic");
                assert!(a.rem(&g).is_zero(), "gcd divides a");
                assert!(b.rem(&g).is_zero(), "gcd divides b");
            }
        }
    }

    #[test]
    fn composite_modulus_verdicts_are_unspecified_not_proofs() {
        // The type requires a prime modulus; over a composite one the
        // division-based results are unspecified and need not panic. This
        // test pins the *current* unspecified behaviour on the canonical
        // example so a later change cannot quietly promote it to a
        // correctness claim: x² + 1 factors as (x+2)(x+3) modulo 5, so no
        // sound irreducibility test over ℤ/15ℤ could call it irreducible —
        // yet the Frobenius argument this routine leans on assumes a field,
        // and modulo 15 it reports `true` without noticing. If this
        // assertion ever fails, the behaviour changed: re-document it,
        // do not "fix" the test.
        let m = BigUint::from_u64(15);
        let f = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[1, 0, 1]), &m);
        assert!(
            f.is_irreducible(),
            "unspecified composite-modulus verdict drifted; update the contract notes"
        );
    }

    #[test]
    fn poly_mod_p_gcd_of_known_factors() {
        let p = BigUint::from_u64(7);
        // (x-1)(x-2) and (x-2)(x-3) share (x-2).
        let x_minus = |a: i64| PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[-a, 1]), &p);
        let f = x_minus(1).mul(&x_minus(2));
        let g = x_minus(2).mul(&x_minus(3));
        assert_eq!(f.gcd(&g), x_minus(2), "shared linear factor");
    }

    #[test]
    #[should_panic(expected = "invertible")]
    fn poly_mod_p_div_rem_panics_on_noninvertible_pivot() {
        // Composite modulus with a divisor whose leading coefficient shares
        // a factor: 2x + 1 mod 6, leading coeff 2 not invertible.
        let m = BigUint::from_u64(6);
        let a = PolyModP::new(vec![BigUint::one(), BigUint::one(), BigUint::one()], &m);
        let b = PolyModP::new(vec![BigUint::one(), BigUint::from_u64(2)], &m);
        let _ = a.div_rem(&b);
    }

    // An independent resultant oracle: the determinant of the Sylvester
    // matrix by rational (fraction) Gaussian elimination — no shared code
    // with the production Bareiss path. The elimination reads and writes
    // distinct rows of one matrix, so it indexes by row rather than
    // iterating.
    #[allow(clippy::needless_range_loop)]
    fn resultant_rational_oracle(a: &PolyZ, b: &PolyZ) -> BigInt {
        use crate::bigint::Sign;
        if a.is_zero() || b.is_zero() {
            return BigInt::zero();
        }
        let (m, n) = (a.degree().unwrap(), b.degree().unwrap());
        if m == 0 && n == 0 {
            return BigInt::one();
        }
        let size = m + n;
        // Rationals as (num, den) BigInt pairs, den > 0.
        type Q = (BigInt, BigInt);
        let q = |v: BigInt| -> Q { (v, BigInt::one()) };
        let qmul = |x: &Q, y: &Q| -> Q { (x.0.mul_ref(&y.0), x.1.mul_ref(&y.1)) };
        let qsub = |x: &Q, y: &Q| -> Q {
            (
                x.0.mul_ref(&y.1).sub_ref(&y.0.mul_ref(&x.1)),
                x.1.mul_ref(&y.1),
            )
        };
        let qdiv = |x: &Q, y: &Q| -> Q {
            let mut num = x.0.mul_ref(&y.1);
            let mut den = x.1.mul_ref(&y.0);
            if den.sign() == Sign::Negative {
                num = num.negated();
                den = den.negated();
            }
            (num, den)
        };
        let qzero = |x: &Q| -> bool { x.0.is_zero() };
        let mut mat = vec![vec![q(BigInt::zero()); size]; size];
        let a_hi: Vec<BigInt> = a.coefficients().iter().rev().cloned().collect();
        let b_hi: Vec<BigInt> = b.coefficients().iter().rev().cloned().collect();
        for i in 0..n {
            for (j, c) in a_hi.iter().enumerate() {
                mat[i][i + j] = q(c.clone());
            }
        }
        for i in 0..m {
            for (j, c) in b_hi.iter().enumerate() {
                mat[n + i][i + j] = q(c.clone());
            }
        }
        let mut det = q(BigInt::one());
        for col in 0..size {
            let Some(piv) = (col..size).find(|&r| !qzero(&mat[r][col])) else {
                return BigInt::zero();
            };
            if piv != col {
                mat.swap(col, piv);
                det = (det.0.negated(), det.1);
            }
            det = qmul(&det, &mat[col][col]);
            let inv = mat[col][col].clone();
            let pivot_row = mat[col].clone();
            for r in col + 1..size {
                let factor = qdiv(&mat[r][col], &inv);
                for (cell, pivot_cell) in mat[r].iter_mut().zip(&pivot_row).skip(col) {
                    let prod = qmul(&factor, pivot_cell);
                    *cell = qsub(cell, &prod);
                }
            }
        }
        // det is an integer; num/den divides evenly.
        det.0.div_exact(&det.1)
    }

    #[test]
    fn resultant_matches_rational_oracle_and_known_values() {
        let mut rng = SplitMix64 {
            state: 0x8e50_0000_0001,
        };
        for _ in 0..500 {
            let a = rng.poly_z(5, 8);
            let b = rng.poly_z(5, 8);
            assert_eq!(
                a.resultant(&b),
                resultant_rational_oracle(&a, &b),
                "resultant vs rational oracle"
            );
            // Symmetry up to sign: res(a,b) = (-1)^(deg a · deg b) res(b,a).
            if let (Some(da), Some(db)) = (a.degree(), b.degree()) {
                let expected = if (da * db) % 2 == 0 {
                    b.resultant(&a)
                } else {
                    b.resultant(&a).negated()
                };
                assert_eq!(a.resultant(&b), expected, "resultant symmetry");
            }
        }
        // Known values.
        let xm = |r: i64| PolyZ::from_i64_slice(&[-r, 1]); // x - r
                                                           // res(x-2, x-3) = -1 (Sylvester convention).
        assert_eq!(xm(2).resultant(&xm(3)), BigInt::from_i64(-1));
        // Shares root 1 → 0.
        assert_eq!(
            PolyZ::from_i64_slice(&[-1, 0, 1]).resultant(&xm(1)),
            BigInt::zero()
        );
        // res(2x, 3) = 3^1.
        assert_eq!(
            PolyZ::from_i64_slice(&[0, 2]).resultant(&PolyZ::from_i64_slice(&[3])),
            BigInt::from_i64(3)
        );
        // Two nonzero constants → 1.
        assert_eq!(
            PolyZ::from_i64_slice(&[5]).resultant(&PolyZ::from_i64_slice(&[7])),
            BigInt::one()
        );
        // Zero polynomial → 0.
        assert_eq!(xm(1).resultant(&PolyZ::zero()), BigInt::zero());
    }

    #[test]
    fn resultant_multiplicativity() {
        let mut rng = SplitMix64 {
            state: 0x3711_0000_0001,
        };
        for _ in 0..300 {
            let f = rng.poly_z(4, 6);
            let g = rng.poly_z(3, 6);
            let h = rng.poly_z(3, 6);
            if f.is_zero() || g.is_zero() || h.is_zero() {
                continue;
            }
            // res(f, g·h) = res(f, g)·res(f, h).
            assert_eq!(
                f.resultant(&g.mul(&h)),
                f.resultant(&g).mul_ref(&f.resultant(&h)),
                "resultant multiplicativity"
            );
        }
    }

    #[test]
    fn discriminant_known_forms() {
        // disc(x^2 + bx + c) = b^2 - 4c.
        for b in -4i64..=4 {
            for c in -4i64..=4 {
                assert_eq!(
                    PolyZ::from_i64_slice(&[c, b, 1]).discriminant(),
                    BigInt::from_i64(b * b - 4 * c),
                    "disc quadratic b={b} c={c}"
                );
            }
        }
        // disc(x^3 + px + q) = -4p^3 - 27q^2.
        for p in -3i64..=3 {
            for q in -3i64..=3 {
                assert_eq!(
                    PolyZ::from_i64_slice(&[q, p, 0, 1]).discriminant(),
                    BigInt::from_i64(-4 * p * p * p - 27 * q * q),
                    "disc depressed cubic p={p} q={q}"
                );
            }
        }
        // A polynomial with a repeated root has discriminant 0:
        // (x-1)^2(x-2) = x^3 - 4x^2 + 5x - 2.
        let repeated = PolyZ::from_i64_slice(&[-2, 5, -4, 1]);
        assert_eq!(repeated.discriminant(), BigInt::zero());
        // Constant and zero → 0.
        assert_eq!(PolyZ::from_i64_slice(&[7]).discriminant(), BigInt::zero());
        assert_eq!(PolyZ::zero().discriminant(), BigInt::zero());
    }

    // Multiply a slice of factors-with-multiplicity back into one poly.
    fn reassemble(factors: &[(PolyModP, usize)], p: &BigUint) -> PolyModP {
        let mut acc = PolyModP::new(vec![BigUint::one()], p);
        for (f, e) in factors {
            for _ in 0..*e {
                acc = acc.mul(f);
            }
        }
        acc
    }

    #[test]
    fn factorization_reconstructs_and_is_irreducible() {
        let mut rng = SplitMix64 {
            state: 0xfac0_0000_0001,
        };
        for &p in &[2u64, 3, 5, 7, 11, 13] {
            let pm = BigUint::from_u64(p);
            for _ in 0..200 {
                // Build a random product of small factors with multiplicities.
                let mut f = PolyModP::new(vec![BigUint::one()], &pm);
                let parts = 1 + (rng.next_u64() % 3) as usize;
                for _ in 0..parts {
                    let deg = 1 + (rng.next_u64() % 3) as usize;
                    let coeffs: Vec<BigUint> = (0..=deg)
                        .map(|_| BigUint::from_u64(rng.next_u64() % p))
                        .collect();
                    let mut base = PolyModP::new(coeffs, &pm);
                    // Ensure non-constant.
                    if base.degree().unwrap_or(0) < 1 {
                        base = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[0, 1]), &pm);
                    }
                    let mult = 1 + (rng.next_u64() % 3) as usize;
                    for _ in 0..mult {
                        f = f.mul(&base);
                    }
                }
                if f.degree().unwrap_or(0) < 1 {
                    continue;
                }
                let factors = f.factor(&mut rng);
                // Product of factors^mult equals the monic form of f.
                assert_eq!(
                    reassemble(&factors, &pm),
                    f.make_monic(),
                    "reconstruct p={p}"
                );
                // Every returned factor is monic and irreducible.
                for (fac, e) in &factors {
                    assert!(*e >= 1);
                    assert!(fac.leading_coefficient().is_one(), "monic factor p={p}");
                    assert!(fac.is_irreducible(), "factor is irreducible p={p}");
                }
            }
        }
    }

    #[test]
    fn irreducibility_matches_brute_force() {
        // Over small p, check is_irreducible against exhaustive trial
        // division by all lower-degree monics.
        for &p in &[2u64, 3, 5] {
            let pm = BigUint::from_u64(p);
            // Every monic polynomial of degree 1..=3.
            for deg in 1..=3usize {
                let mut coeffs = vec![0u64; deg + 1];
                coeffs[deg] = 1;
                loop {
                    let poly =
                        PolyModP::new(coeffs.iter().map(|&c| BigUint::from_u64(c)).collect(), &pm);
                    if poly.degree() == Some(deg) {
                        let brute = brute_irreducible(&poly, p);
                        assert_eq!(
                            poly.is_irreducible(),
                            brute,
                            "is_irreducible mismatch p={p} coeffs={coeffs:?}"
                        );
                    }
                    // Increment the low `deg` coefficients (leading stays 1);
                    // when they all overflow, the enumeration is done.
                    let mut i = 0;
                    while i < deg {
                        coeffs[i] += 1;
                        if coeffs[i] < p {
                            break;
                        }
                        coeffs[i] = 0;
                        i += 1;
                    }
                    if i == deg {
                        break;
                    }
                }
            }
        }
    }

    // Trial-division irreducibility oracle: divisible by no lower-degree
    // monic of degree 1..deg.
    fn brute_irreducible(poly: &PolyModP, p: u64) -> bool {
        let pm = BigUint::from_u64(p);
        let deg = poly.degree().unwrap();
        if deg < 1 {
            return false;
        }
        for d in 1..deg {
            let mut coeffs = vec![0u64; d + 1];
            coeffs[d] = 1;
            loop {
                let divisor =
                    PolyModP::new(coeffs.iter().map(|&c| BigUint::from_u64(c)).collect(), &pm);
                if divisor.degree() == Some(d) && poly.rem(&divisor).is_zero() {
                    return false;
                }
                let mut i = 0;
                while i < d {
                    coeffs[i] += 1;
                    if coeffs[i] < p {
                        break;
                    }
                    coeffs[i] = 0;
                    i += 1;
                }
                if i == d {
                    break;
                }
            }
        }
        true
    }

    #[test]
    fn roots_match_evaluation() {
        let mut rng = SplitMix64 {
            state: 0x9007_0000_0001,
        };
        for &p in &[2u64, 3, 5, 7, 11, 13, 101] {
            let pm = BigUint::from_u64(p);
            for _ in 0..100 {
                let deg = 1 + (rng.next_u64() % 5) as usize;
                let coeffs: Vec<BigUint> = (0..=deg)
                    .map(|_| BigUint::from_u64(rng.next_u64() % p))
                    .collect();
                let f = PolyModP::new(coeffs, &pm);
                if f.degree().unwrap_or(0) < 1 {
                    continue;
                }
                let roots = f.roots(&mut rng);
                // Every returned root is a genuine root, ascending, distinct.
                let mut previous: Option<BigUint> = None;
                for r in &roots {
                    assert!(f.evaluate(r).is_zero(), "claimed root at p={p}");
                    if let Some(prev) = &previous {
                        assert!(prev < r, "roots ascending and distinct");
                    }
                    previous = Some(r.clone());
                }
                // Brute force: every residue that is a root is returned.
                for a in 0..p {
                    let av = BigUint::from_u64(a);
                    if f.evaluate(&av).is_zero() {
                        assert!(roots.contains(&av), "missed root {a} at p={p}");
                    }
                }
            }
        }
    }

    #[test]
    fn factor_of_known_polynomial() {
        // x^2 - 1 = (x-1)(x+1) mod 7.
        let p = BigUint::from_u64(7);
        let f = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[-1, 0, 1]), &p);
        let mut rng = SplitMix64 { state: 0x1234 };
        let factors = f.factor(&mut rng);
        assert_eq!(factors.len(), 2);
        assert_eq!(reassemble(&factors, &p), f.make_monic());
        // x^2 + 1 is irreducible mod 7 (no root: -1 is a non-residue mod 7).
        let g = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[1, 0, 1]), &p);
        assert!(g.is_irreducible());
        assert!(g.roots(&mut rng).is_empty());
    }

    #[test]
    fn factor_p2_trace_map_splits_equal_cubics() {
        // Φ₇ = x⁶+x⁵+x⁴+x³+x²+x+1 = (x³+x+1)(x³+x²+1) over 𝔽₂: two distinct
        // irreducible cubics, so distinct-degree yields one degree-6 block of
        // two degree-3 factors and the equal-degree split runs at p = 2, d = 3.
        // The p = 2 trace-map loop body executes only for d ≥ 2, so this is the
        // case that exercises it (the odd-p exponent path is never taken here).
        let p = BigUint::from_u64(2);
        let f = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[1, 1, 1, 1, 1, 1, 1]), &p);
        let mut rng = SplitMix64 { state: 0xfac0_0002 };
        let mut factors = f.factor(&mut rng);
        assert_eq!(factors.len(), 2, "two cubic factors over F_2");
        for (fac, e) in &factors {
            assert_eq!(*e, 1, "each cubic is simple");
            assert_eq!(fac.degree(), Some(3), "degree-3 factor");
            assert!(fac.is_irreducible(), "irreducible cubic");
        }
        assert_eq!(
            reassemble(&factors, &p),
            f.make_monic(),
            "product reconstructs"
        );
        // The two cubics are exactly x³+x+1 and x³+x²+1.
        factors.sort_by(|a, b| a.0.coefficients().cmp(b.0.coefficients()));
        let c_small = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[1, 0, 1, 1]), &p); // x³+x²+1
        let c_big = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[1, 1, 0, 1]), &p); // x³+x+1
        assert_eq!(factors[0].0, c_small);
        assert_eq!(factors[1].0, c_big);
    }

    #[test]
    #[should_panic(expected = "share a modulus")]
    fn poly_mod_p_rejects_mixed_moduli() {
        // Combining an 𝔽₅ element with an 𝔽₇ one must panic in every build, not
        // silently emit a value tagged with one modulus (review §2.1).
        let a = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[2]), &BigUint::from_u64(5));
        let b = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[3]), &BigUint::from_u64(7));
        let _ = a.add(&b);
    }

    #[test]
    #[should_panic(expected = "made no progress")]
    fn factor_with_dead_rng_panics_rather_than_hangs() {
        // An Rng that never yields entropy cannot drive the equal-degree split;
        // it must panic after a bounded number of fruitless draws, not loop
        // forever (review §2.4 / §5.4). x² − 1 = (x−1)(x+1) mod 7 leaves a
        // degree-1 block of two factors, so the split is actually entered.
        struct ZeroRng;
        impl crate::random::Rng for ZeroRng {
            fn fill_bytes(&mut self, dest: &mut [u8]) {
                dest.fill(0);
            }
        }
        let p = BigUint::from_u64(7);
        let f = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[-1, 0, 1]), &p);
        let _ = f.factor(&mut ZeroRng);
    }

    #[test]
    fn poly_mod_p_pow_mod_matches_repeated_multiply() {
        let p = BigUint::from_u64(13);
        let mut rng = SplitMix64 {
            state: 0x7011_0000_0001,
        };
        for _ in 0..100 {
            let base = PolyModP::from_poly_z(&rng.poly_z(4, 50), &p);
            let modulus = {
                let mut m = PolyModP::from_poly_z(&rng.poly_z(4, 50), &p);
                if m.degree().unwrap_or(0) < 1 {
                    m = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[1, 0, 1]), &p);
                }
                m
            };
            let e = rng.next_u64() % 20;
            let by_pow = base.pow_mod(&BigUint::from_u64(e), &modulus);
            let mut by_mul = PolyModP::new(vec![BigUint::one()], &p).rem(&modulus);
            for _ in 0..e {
                by_mul = by_mul.mul(&base).rem(&modulus);
            }
            assert_eq!(by_pow, by_mul, "pow_mod vs repeated multiply");
        }
    }

    #[test]
    fn balanced_expansion_reconstructs_n_and_bounds_its_digits() {
        // The two things the expansion promises: `f(m) = n` exactly, for
        // every requested degree, and every digit below the top one lands in
        // the symmetric range. The second is what the expansion is *for*, and
        // it is the half a caller cannot check cheaply for itself.
        let mut rng = SplitMix64 {
            state: 0x0ba1_a2ce_d169,
        };
        for _ in 0..200 {
            let n = BigInt::from_biguint(rng.biguint(4));
            let base = rng.biguint(1).add_ref(&BigUint::from_u64(2));
            let degree = 1 + (rng.next_u64() as usize % 6);
            let f = PolyZ::balanced_base_expansion(&n, &base, degree);
            assert_eq!(
                f.evaluate(&BigInt::from_biguint(base.clone())),
                n,
                "the expansion must reconstruct n exactly"
            );
            for c in f.coefficients().iter().take(degree) {
                // |c| ≤ m/2, stated without division: 2|c| ≤ m.
                assert!(
                    c.abs().mul_ref(&BigUint::from_u64(2)) <= base,
                    "a digit below the top must be balanced"
                );
            }
        }
        // A negative n is expanded on the same terms, and a base larger than
        // n puts everything in the constant term.
        let n = BigInt::from_i64(-1_000_003);
        let base = BigUint::from_u64(1_000_000_007);
        let f = PolyZ::balanced_base_expansion(&n, &base, 3);
        assert_eq!(f.evaluate(&BigInt::from_biguint(base)), n);
        assert_eq!(f.degree(), Some(0));

        // The generated cases above are all non-negative, use a ~64-bit
        // base, and never ask for degree 0, so the small-base, negative,
        // and degenerate corners are swept exhaustively instead. A negative
        // `n` with a small base is what actually drives the digit loop
        // through a negative quotient.
        for base in 2u64..=12 {
            let m = BigUint::from_u64(base);
            let signed = BigInt::from_biguint(m.clone());
            let half = BigInt::from_biguint(m.clone());
            for n in -200i64..=200 {
                for degree in 0..=5 {
                    let value = BigInt::from_i64(n);
                    let f = PolyZ::balanced_base_expansion(&value, &m, degree);
                    assert_eq!(
                        f.evaluate(&signed),
                        value,
                        "n = {n}, base = {base}, degree = {degree}"
                    );
                    for c in f.coefficients().iter().take(degree) {
                        assert!(
                            c.abs().mul_ref(&BigUint::from_u64(2)) <= m,
                            "digit out of range at n = {n}, base = {base}"
                        );
                    }
                    let _ = &half;
                }
            }
        }

        // The half-open end of `(−m/2, m/2]`: with an even base a digit can
        // land exactly on `m/2`, and the convention says it stays positive.
        // Reconstruction and the size bound are both satisfied by either
        // sign, so only the coefficients themselves pin the tie-break.
        assert_eq!(
            PolyZ::balanced_base_expansion(&BigInt::from_i64(5), &BigUint::from_u64(10), 1),
            PolyZ::from_i64_slice(&[5])
        );
        assert_eq!(
            PolyZ::balanced_base_expansion(&BigInt::from_i64(15), &BigUint::from_u64(10), 2),
            PolyZ::from_i64_slice(&[5, 1])
        );
        assert_eq!(
            PolyZ::balanced_base_expansion(&BigInt::from_i64(-5), &BigUint::from_u64(10), 1),
            PolyZ::from_i64_slice(&[5, -1])
        );
    }

    #[test]
    fn rem_monic_agrees_with_the_general_division() {
        // The fast path against the routine it specializes, including the
        // boundary shapes: a dividend below the divisor's degree, a dividend
        // exactly at it, and the zero dividend.
        let mut rng = SplitMix64 {
            state: 0x5e11_0c2d_1137,
        };
        for _ in 0..300 {
            let divisor_degree = 1 + (rng.next_u64() as usize % 6);
            let f = rng.monic_z(divisor_degree, 40);
            let a = rng.poly_z(14, 40);
            let expected = a
                .div_rem(&f)
                .expect("a monic divisor always divides over ℤ")
                .1;
            assert_eq!(a.rem_monic(&f), expected, "rem_monic vs div_rem");
        }
        let f = PolyZ::from_i64_slice(&[7, -3, 1]);
        assert!(PolyZ::zero().rem_monic(&f).is_zero());
        assert_eq!(
            PolyZ::from_i64_slice(&[5, 2]).rem_monic(&f),
            PolyZ::from_i64_slice(&[5, 2]),
            "a dividend below the divisor's degree is its own remainder"
        );
        // Against the constant 1 the quotient ring is trivial.
        assert!(PolyZ::from_i64_slice(&[9, 4, 1])
            .rem_monic(&PolyZ::from_i64_slice(&[1]))
            .is_zero());
    }

    #[test]
    fn rem_monic_is_a_ring_homomorphism() {
        // The property the product tree leans on: reducing at every level
        // gives the same answer as reducing once at the end.
        let mut rng = SplitMix64 {
            state: 0x1707_ab0a_7711,
        };
        for _ in 0..200 {
            let degree = 1 + (rng.next_u64() as usize % 5);
            let f = rng.monic_z(degree, 30);
            let a = rng.poly_z(9, 30);
            let b = rng.poly_z(9, 30);
            assert_eq!(
                a.mul(&b).rem_monic(&f),
                a.rem_monic(&f).mul(&b.rem_monic(&f)).rem_monic(&f),
                "reduction commutes with multiplication"
            );
            assert_eq!(
                a.add(&b).rem_monic(&f),
                a.rem_monic(&f).add(&b.rem_monic(&f)).rem_monic(&f),
                "reduction commutes with addition"
            );
        }
    }

    #[test]
    fn product_mod_monic_agrees_with_an_unreduced_fold() {
        // The tree, which reduces at every level, against the honest thing:
        // multiply everything out over ℤ and reduce once. Sizes chosen to
        // straddle the pairing — odd counts leave a factor unpaired at some
        // level, which is where a tree gets its off-by-one.
        let mut rng = SplitMix64 {
            state: 0x9a3c_11de_bb05,
        };
        for count in [1usize, 2, 3, 4, 5, 7, 8, 9, 16, 17] {
            let f = rng.monic_z(3, 25);
            let factors: Vec<PolyZ> = (0..count).map(|_| rng.poly_z(4, 25)).collect();
            let mut folded = PolyZ::constant(BigInt::one());
            for g in &factors {
                folded = folded.mul(g);
            }
            assert_eq!(
                PolyZ::product_mod_monic(&factors, &f),
                folded.rem_monic(&f),
                "product tree vs a fold, {count} factors"
            );
        }
        // A single factor is the only input for which the per-factor
        // pre-reduction is the *sole* reduction — with two or more, every
        // element passes through a `mul(..).rem_monic(..)` anyway. So it
        // needs a factor that genuinely reduces, not the degree-0 one the
        // random draw happened to give.
        let quad = PolyZ::from_i64_slice(&[1, 0, 1]); // x^2 + 1
        let cubic = PolyZ::from_i64_slice(&[4, 3, 2, 1]); // x^3 + 2x^2 + 3x + 4
        assert_eq!(
            PolyZ::product_mod_monic(std::slice::from_ref(&cubic), &quad),
            PolyZ::from_i64_slice(&[2, 2]),
        );
        assert_ne!(
            PolyZ::product_mod_monic(std::slice::from_ref(&cubic), &quad),
            cubic
        );

        // The empty product is the ring's identity.
        let f = PolyZ::from_i64_slice(&[1, 0, 1]);
        assert_eq!(
            PolyZ::product_mod_monic(&[], &f),
            PolyZ::constant(BigInt::one())
        );
        assert!(PolyZ::product_mod_monic(&[], &PolyZ::from_i64_slice(&[1])).is_zero());
    }

    #[test]
    fn homogeneous_substitution_is_the_homogenisation_evaluated() {
        let mut rng = SplitMix64 {
            state: 0x40e5_1c7b_022d,
        };
        let one = PolyZ::constant(BigInt::one());
        let x = PolyZ::from_i64_slice(&[0, 1]);
        for _ in 0..150 {
            let f = rng.poly_z(6, 30);
            let a = rng.poly_z(3, 15);
            let b = rng.poly_z(3, 15);
            let Some(degree) = f.degree() else { continue };

            // Independent check one: with Y = 1 the form restricts to `f`, so
            // substituting X = x must give `f` back.
            assert_eq!(f.homogeneous_substitution(&x, &one), f);

            // Independent check two: with Y = 1 and X = a it is composition,
            // computed here by Horner rather than by the power ladder.
            let mut composed = PolyZ::zero();
            for c in f.coefficients().iter().rev() {
                composed = composed.mul(&a).add(&PolyZ::constant(c.clone()));
            }
            assert_eq!(f.homogeneous_substitution(&a, &one), composed);

            // Independent check three: at constants A = q·B and B the value
            // is Bᵈ·f(q), which needs no homogeneous reasoning at all.
            let q = rng.coeff(12);
            let bconst = rng.coeff(12);
            if !bconst.is_zero() {
                let aconst = q.mul_ref(&bconst);
                let got = f.homogeneous_substitution(
                    &PolyZ::constant(aconst),
                    &PolyZ::constant(bconst.clone()),
                );
                let want = PolyZ::constant(
                    BigInt::from_biguint(bconst.abs().pow_u64(degree as u64))
                        .mul_ref(&if degree % 2 == 1 && bconst.sign() == Sign::Negative {
                            BigInt::from_i64(-1)
                        } else {
                            BigInt::one()
                        })
                        .mul_ref(&f.evaluate(&q)),
                );
                assert_eq!(got, want, "Bᵈ·f(q) at A = q·B");
            }

            // And the general shape, against the defining sum.
            let mut expect = PolyZ::zero();
            for (k, c) in f.coefficients().iter().enumerate() {
                let mut term = PolyZ::constant(c.clone());
                for _ in 0..k {
                    term = term.mul(&a);
                }
                for _ in 0..(degree - k) {
                    term = term.mul(&b);
                }
                expect = expect.add(&term);
            }
            assert_eq!(f.homogeneous_substitution(&a, &b), expect);
        }
        assert!(PolyZ::zero().homogeneous_substitution(&x, &one).is_zero());
    }

    #[test]
    fn roots_mod_prime_power_matches_exhaustive_search() {
        // Hensel lifting against trying every residue. The squared
        // polynomials in the second half exist to force the branching case:
        // a repeated root kills the derivative, which is the path that does
        // not follow from the textbook lemma.
        let mut rng = SplitMix64 {
            state: 0xc001_d00d_5a11,
        };
        for &(p, e) in &[(2u64, 6u32), (3, 4), (5, 3), (7, 3), (11, 2), (13, 2)] {
            let prime = BigUint::from_u64(p);
            let power = BigUint::from_u64(p.pow(e));
            for round in 0..40 {
                let g = rng.poly_z(3, 20);
                let f = match round % 3 {
                    0 => g.clone(),
                    1 => g.mul(&g),
                    // A content divisible by p: the root set is the same as
                    // for `g` at one lower precision, spread back out. This
                    // is the family an earlier revision refused outright,
                    // with a panic message asserting — falsely — that every
                    // residue was a root.
                    _ => g.scale(&BigInt::from_biguint(prime.clone())),
                };
                if f.is_zero() {
                    continue;
                }
                let content_valuation = f
                    .coefficients()
                    .iter()
                    .filter(|c| !c.is_zero())
                    .map(|c| crate::number_theory::valuation(&c.abs(), &prime))
                    .min()
                    .expect("non-zero");
                if content_valuation >= e as usize {
                    continue;
                }
                let found: Vec<u64> = f
                    .roots_mod_prime_power(&prime, e, &mut rng)
                    .iter()
                    .map(|r| r.to_u64().expect("a residue below p^e fits"))
                    .collect();
                let reduced = PolyModP::from_poly_z(&f, &power);
                let expected: Vec<u64> = (0..p.pow(e))
                    .filter(|&r| reduced.evaluate(&BigUint::from_u64(r)).is_zero())
                    .collect();
                assert_eq!(found, expected, "roots of {f:?} mod {p}^{e}");
            }
        }
    }

    #[test]
    fn roots_mod_prime_power_lifts_a_known_branching_root() {
        // x² modulo 3³: the lifts of the single root 0 mod 3 are the
        // multiples of 9, since 27 | x² needs 3 | x and then 3 | x/3.
        let mut rng = SplitMix64 {
            state: 0x2b2b_2b2b_0001,
        };
        let roots = PolyZ::from_i64_slice(&[0, 0, 1]).roots_mod_prime_power(
            &BigUint::from_u64(3),
            3,
            &mut rng,
        );
        assert_eq!(
            roots,
            vec![BigUint::zero(), BigUint::from_u64(9), BigUint::from_u64(18)]
        );
        // A simple root lifts uniquely and stays alone: x² − 2 modulo 7³ has
        // the two roots ±√2, and no more.
        let roots = PolyZ::from_i64_slice(&[-2, 0, 1]).roots_mod_prime_power(
            &BigUint::from_u64(7),
            3,
            &mut rng,
        );
        assert_eq!(roots.len(), 2, "a simple root does not branch");
        for r in &roots {
            let square = BigUint::mod_mul(r, r, &BigUint::from_u64(343));
            assert_eq!(square, BigUint::from_u64(2));
        }
    }

    #[test]
    fn symmetric_lift_recovers_the_integer_it_came_from() {
        // A modulus wider than twice the height is the whole contract: the
        // lift then returns the original polynomial, coefficient for
        // coefficient, sign and all.
        let mut rng = SplitMix64 {
            state: 0x7e5e_a11f_0009,
        };
        let modulus = BigUint::from_u64(2).pow_u64(96);
        let half = modulus.div_rem(&BigUint::from_u64(2)).0;
        for _ in 0..200 {
            let f = rng.poly_z(8, 1_000_000);
            let lifted = PolyModP::from_poly_z(&f, &modulus).symmetric_lift();
            assert_eq!(lifted, f, "the lift recovers a polynomial that fits");
            for c in lifted.coefficients() {
                assert!(c.abs() <= half, "the lift is balanced");
            }
        }
    }

    #[test]
    fn with_modulus_reads_the_same_representatives_wider() {
        let mut rng = SplitMix64 {
            state: 0x3ec0_de11_0077,
        };
        let narrow = BigUint::from_u64(1_000_003);
        let wide = narrow.mul_ref(&narrow);
        for _ in 0..100 {
            let f = PolyModP::from_poly_z(&rng.poly_z(6, 10_000), &narrow);
            let widened = f.with_modulus(&wide);
            assert_eq!(widened.modulus(), &wide);
            // The representatives carried are the *canonical* ones, in
            // `[0, m)` — not the symmetric ones. Asserting the round trip
            // alone cannot tell the two apart, since both are sections of
            // the same projection and both round-trip; only the values do.
            assert_eq!(widened.coefficients(), f.coefficients());
            assert_eq!(widened.with_modulus(&narrow), f);
        }
        // Concretely, and away from any polynomial the generator draws: 5
        // and 4 modulo 7 widen to 5 and 4, not to 47 and 46.
        let small = BigUint::from_u64(7);
        let big = BigUint::from_u64(49);
        let f = PolyModP::new(vec![BigUint::from_u64(5), BigUint::from_u64(4)], &small);
        assert_eq!(
            f.with_modulus(&big).coefficients(),
            &[BigUint::from_u64(5), BigUint::from_u64(4)]
        );
        // Narrowing is the ring projection and commutes with addition;
        // widening is its section and does not. Both directions asserted,
        // because the documentation distinguishes them.
        let g = PolyModP::new(vec![BigUint::from_u64(4)], &small);
        let h = PolyModP::new(vec![BigUint::from_u64(5)], &small);
        assert_ne!(
            g.add(&h).with_modulus(&big),
            g.with_modulus(&big).add(&h.with_modulus(&big)),
            "widening is not additive"
        );
        let wide_g = PolyModP::new(vec![BigUint::from_u64(40)], &big);
        let wide_h = PolyModP::new(vec![BigUint::from_u64(45)], &big);
        assert_eq!(
            wide_g.add(&wide_h).with_modulus(&small),
            wide_g
                .with_modulus(&small)
                .add(&wide_h.with_modulus(&small)),
            "narrowing is the projection, so it is additive"
        );
    }

    #[test]
    fn the_pieces_compose_into_a_newton_square_root_in_a_quotient_ring() {
        // What the downstream use actually needs: recover β from δ = β² in
        // ℤ[x]/(f) by lifting a square root out of 𝔽_q[x]/(f) with the
        // modulus squaring each round, then reading the answer back over ℤ.
        // `with_modulus` seeds each round, `rem_monic` keeps the degree down
        // and settles the check over ℤ, and `symmetric_lift` is what turns a
        // residue into the integer answer.
        let f = PolyZ::from_i64_slice(&[1, 0, 1]); // x² + 1, irreducible mod 7
        let beta = PolyZ::from_i64_slice(&[9_413, -7_121]);
        let delta = beta.mul(&beta).rem_monic(&f);

        let q = BigUint::from_u64(7);
        let field = PolyModP::from_poly_z(&f, &q);
        let residue = PolyModP::from_poly_z(&delta, &q).rem(&field);

        // Base root by exhaustion over the 49 elements of 𝔽_49 — deliberately
        // not the field square-root routine, so this test does not depend on
        // it.
        let mut root = None;
        for a in 0..7u64 {
            for b in 0..7u64 {
                let candidate = PolyModP::new(vec![BigUint::from_u64(a), BigUint::from_u64(b)], &q);
                if candidate.mul(&candidate).rem(&field) == residue {
                    root = Some(candidate);
                    break;
                }
            }
        }
        let mut current = root.expect("δ is a square in 𝔽_49");
        let mut modulus = q.clone();

        // u ≈ (2β)⁻¹ in the field, by Fermat: the inverse Newton also lifts.
        let order = q.pow_u64(2);
        let two = PolyModP::new(vec![BigUint::from_u64(2)], &q);
        let mut inverse = two
            .mul(&current)
            .rem(&field)
            .pow_mod(&order.sub_ref(&BigUint::from_u64(2)), &field);

        for _ in 0..6 {
            if current.symmetric_lift().rem_monic(&f) == PolyZ::zero() {
                unreachable!("δ is not zero");
            }
            let candidate = current.symmetric_lift();
            if candidate.mul(&candidate).rem_monic(&f) == delta {
                assert!(candidate == beta || candidate == beta.negated());
                return;
            }
            // Square the precision: q^k → q^{2k}.
            let next = modulus.mul_ref(&modulus);
            let f_next = PolyModP::from_poly_z(&f, &next);
            let delta_next = PolyModP::from_poly_z(&delta, &next).rem(&f_next);
            let mut b = current.with_modulus(&next);
            let u = inverse.with_modulus(&next);

            // β ← β − (β² − δ)·u
            let error = b.mul(&b).rem(&f_next).sub(&delta_next);
            b = b.sub(&error.mul(&u).rem(&f_next));

            // u ← u·(2 − 2βu), Newton for the inverse.
            let two_next = PolyModP::new(vec![BigUint::from_u64(2)], &next);
            let product = two_next.mul(&b).rem(&f_next).mul(&u).rem(&f_next);
            inverse = u.mul(&two_next.sub(&product)).rem(&f_next);

            current = b;
            modulus = next;
        }
        panic!("the lift did not converge");
    }
}
