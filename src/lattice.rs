//! Lattice reduction over the integers.
//!
//! [`lll_reduce`](crate::lattice::lll_reduce) applies the Lenstra–Lenstra–Lovász algorithm (A. K. Lenstra,
//! H. W. Lenstra Jr. & L. Lovász, *Factoring polynomials with rational
//! coefficients*, Math. Ann. 261 (1982), 515–534) to an ordered
//! basis of a lattice in `ℤ^m`, replacing it in place with an LLL-reduced basis
//! of the same lattice. The implementation is the integral variant of Cohen,
//! *A Course in Computational Algebraic Number Theory*, Algorithm 2.6.3: the
//! Gram–Schmidt data are carried as the exact integer Gram determinants `d_i`
//! and the integers `λ_{i,j} = d_j · μ_{i,j}`, so no rational or floating-point
//! arithmetic enters and the reduced basis is exact.
//!
//! A basis `b_1, …, b_n` is LLL-reduced (for parameter `δ`) when it is
//! size-reduced, `|μ_{i,j}| ≤ 1/2` for `j < i`, and satisfies the Lovász
//! condition `‖b*_k‖² ≥ (δ − μ_{k,k-1}²)‖b*_{k-1}‖²` for every `k`, where the
//! `b*_i` are the Gram–Schmidt vectors.

use crate::bigint::{BigInt, BigUint, Sign};
use core::num::NonZeroU64;

/// Reduce `basis` in place with the Lovász parameter `δ = 3/4` of the
/// original Lenstra–Lenstra–Lovász paper, the value for which the reduced
/// basis satisfies the classical bounds `‖b_1‖ ≤ 2^((n−1)/4)·det(L)^(1/n)`
/// and `‖b_1‖ ≤ 2^((n−1)/2)·λ_1(L)`, with `λ_1(L)` the length of a shortest
/// non-zero vector of the lattice.
///
/// See [`lll_reduce_delta`] for the panics and preconditions.
pub fn lll_reduce(basis: &mut [Vec<BigInt>]) {
    lll_reduce_delta(basis, 3, 4);
}

/// Reduce `basis` in place with Lovász parameter `δ = delta_num / delta_den`.
///
/// The vectors are the rows; each spans `ℤ^m` for a common `m`. On return the
/// rows are an LLL-reduced basis of the same lattice: size-reduced, and
/// satisfying the Lovász condition at every index. A larger `δ` (nearer 1)
/// makes that condition harder to satisfy, so it tightens the guarantee on
/// the output — the bound `‖b_1‖ ≤ (4/(4δ − 1))^((n−1)/4)·det(L)^(1/n)`
/// improves as `δ → 1` — while admitting more swaps, since the decrease each
/// swap forces on `∏ d_i` shrinks with `δ`. The classical choice is `3/4`.
///
/// The Gram–Schmidt norms `‖b*_i‖²` are *not* nondecreasing on return. The
/// Lovász condition bounds their decay from below —
/// `‖b*_k‖² ≥ (δ − 1/4)‖b*_{k−1}‖²`, a factor of `1/2` at `δ = 3/4` — but it
/// permits a decrease. The basis `[1, 3], [3, 0]` is reduced at `δ = 3/4`
/// and is returned unchanged, with `‖b*_1‖² = 10` and `‖b*_2‖² = 81/10`.
///
/// The mechanism is Cohen's Algorithm 2.6.3, driven by an index `k` that
/// walks up the basis. Reaching a row for the first time extends the exact
/// integer Gram–Schmidt data — the determinants `d_i` and the
/// `λ_{i,j} = d_j·μ_{i,j}` — to that row; each row is then size-reduced
/// against its predecessor and tested against the Lovász condition, a swap
/// sending `k` back down and a pass sending it up after size-reducing
/// against the rest. Every division in the recurrences is exact, so nothing
/// leaves ℤ. Termination is the standard argument: each swap strictly
/// decreases the positive integer `∏ d_i`.
///
/// # Panics
///
/// - if `delta_den` is zero, or `δ ∉ (1/4, 1)` — the range in which Cohen's
///   Algorithm 2.6.3 both terminates and yields a reduced basis. The
///   comparison is carried out in `u128` so that a numerator above
///   `u64::MAX / 4` cannot overflow the `4·delta_num` term and reject a
///   valid `δ`;
/// - if the rows do not all have the same, nonzero length;
/// - if the rows are linearly dependent, i.e. do not form a lattice basis.
///   Dependence is detected as a vanishing Gram determinant `d_k`, and every
///   `d_k` for `k = 1..=n` is computed, so no dependent input escapes.
///
/// An empty basis is reduced vacuously and returns without panicking.
pub fn lll_reduce_delta(basis: &mut [Vec<BigInt>], delta_num: u64, delta_den: u64) {
    lll_reduce_with(basis, delta_num, delta_den, &dot);
}

/// Reduce `basis` in place under the quadratic form `form`: the inner
/// product is `⟨u, v⟩ = uᵀ·form·v` in place of the dot product, so the
/// reduced basis is short in the norm the form induces rather than the
/// Euclidean one.
///
/// The same Algorithm 2.6.3 of Cohen: LLL never looks at coordinates, only
/// at inner products, and an integral form keeps every Gram–Schmidt
/// quantity in `ℤ` exactly as the dot product does. What it is for: a
/// lattice whose vectors are measured by something other than their
/// coordinates — the coefficients of a polynomial measured by the `L²`
/// integral of the polynomial over a region, say, whose Gram matrix over
/// the monomials is a dense form (Herrmann, May & Ritzenhofen, *Polynomial
/// selection using lattices*, Factoring 2009 workshop, for that use).
///
/// `form` must be square with the rows' length, symmetric, and positive
/// definite; the first two are checked, and the third shows as a Gram
/// determinant failing to be positive, which panics as a dependent basis
/// would.
///
/// # Panics
///
/// As [`lll_reduce_delta`], and if `form` is not square of the rows'
/// length or not symmetric.
pub fn lll_reduce_form(
    basis: &mut [Vec<BigInt>],
    form: &[Vec<BigInt>],
    delta_num: u64,
    delta_den: u64,
) {
    if let Some(first) = basis.first() {
        let m = first.len();
        assert!(
            form.len() == m && form.iter().all(|row| row.len() == m),
            "the form must be square of the vectors' length"
        );
        for (i, row) in form.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                assert!(*entry == form[j][i], "the form must be symmetric");
            }
        }
    }
    let inner = |u: &[BigInt], v: &[BigInt]| -> BigInt {
        let mut acc = BigInt::zero();
        for (i, a) in u.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            let mut row = BigInt::zero();
            for (j, b) in v.iter().enumerate() {
                if !b.is_zero() && !form[i][j].is_zero() {
                    row = row.add(&form[i][j].mul(b));
                }
            }
            acc = acc.add(&a.mul(&row));
        }
        acc
    };
    lll_reduce_with(basis, delta_num, delta_den, &inner);
}

/// The reduction, with the inner product supplied.
fn lll_reduce_with(
    basis: &mut [Vec<BigInt>],
    delta_num: u64,
    delta_den: u64,
    dot: &dyn Fn(&[BigInt], &[BigInt]) -> BigInt,
) {
    assert!(delta_den > 0, "delta denominator must be positive");
    // 1/4 < δ < 1. Compare in u128 so a large numerator cannot overflow the
    // `4·delta_num` term and turn a valid δ into a false rejection.
    assert!(
        4u128 * u128::from(delta_num) > u128::from(delta_den) && delta_num < delta_den,
        "LLL parameter delta must lie in (1/4, 1)"
    );

    let n = basis.len();
    if n == 0 {
        return;
    }
    let m = basis[0].len();
    assert!(m > 0, "lattice vectors must be non-empty");
    assert!(
        basis.iter().all(|v| v.len() == m),
        "lattice vectors must all share one length"
    );

    let p = big_u64(delta_num);
    let q = big_u64(delta_den);

    // Gram determinants d[0..=n] (Cohen's d_i), d_0 = 1. The integers
    // lam[i][j] = d_j · μ_{i,j} are stored for 1 ≤ j < i ≤ n; the array is
    // (n+1)×(n+1) so Cohen's 1-based indices are used verbatim.
    let mut d = vec![BigInt::zero(); n + 1];
    d[0] = BigInt::one();
    d[1] = dot(&basis[0], &basis[0]);
    assert!(
        d[1].sign() == crate::bigint::Sign::Positive,
        "linearly dependent basis (zero vector), or a form that is not positive definite"
    );
    let mut lam = vec![vec![BigInt::zero(); n + 1]; n + 1];

    let mut k = 2usize;
    let mut k_max = 1usize;
    while k <= n {
        // Incremental Gram–Schmidt: extend the d_i and λ_{k,j} to row k the
        // first time it is reached (Cohen 2.6.3, step 2). Rows revisited
        // after a swap skip this, their data having been repaired in place.
        // k advances one at a time, so every k in 2..=n passes through here
        // once and every d_k is therefore computed and checked.
        if k > k_max {
            k_max = k;
            for j in 1..=k {
                let mut u = dot(&basis[k - 1], &basis[j - 1]);
                for i in 1..j {
                    // u ← (d_i · u − λ_{k,i} · λ_{j,i}) / d_{i-1}, exact.
                    let num = d[i].mul(&u).sub(&lam[k][i].mul(&lam[j][i]));
                    u = num.div_exact(&d[i - 1]);
                }
                if j < k {
                    lam[k][j] = u;
                } else {
                    // u is now the Gram determinant of b_1..b_k, which
                    // vanishes exactly when those rows are dependent. The
                    // divisions above were by d_1..d_{k-1}, already checked.
                    assert!(
                        u.sign() == crate::bigint::Sign::Positive,
                        "linearly dependent basis, or a form that is not positive definite"
                    );
                    d[k] = u;
                }
            }
        }

        // Test the Lovász condition at k, size-reducing against b_{k-1} first
        // (Cohen 2.6.3, step 3). A swap lowers k and repeats; otherwise b_k is
        // fully size-reduced and k advances. Control returns to step 3 rather
        // than step 2 after a swap: the Gram–Schmidt data for rows up to k_max
        // are kept current by SWAP itself, so no row is recomputed.
        loop {
            red(basis, &mut lam, &d, k, k - 1);

            // The Lovász condition ‖b*_k‖² ≥ (δ − μ_{k,k-1}²)‖b*_{k-1}‖², in
            // integers: substitute ‖b*_i‖² = d_i/d_{i-1} and μ = λ/d_{k-1},
            // then clear the positive denominators d_{k-1}·d_{k-2} and q.
            // Swap iff q·d_k·d_{k-2} < p·d_{k-1}² − q·λ_{k,k-1}²  (δ = p/q).
            let lhs = q.mul(&d[k].mul(&d[k - 2]));
            let lam_sq = lam[k][k - 1].mul(&lam[k][k - 1]);
            let rhs = p.mul(&d[k - 1].mul(&d[k - 1])).sub(&q.mul(&lam_sq));

            if lhs < rhs {
                // The swap replaces d_{k-1} by a strictly smaller positive
                // integer, so the product of the d_i strictly decreases and
                // only finitely many swaps can occur.
                swap_step(basis, &mut lam, &mut d, k, k_max);
                k = core::cmp::max(2, k - 1);
            } else {
                // Size-reduce b_k against the remaining rows, descending so
                // that each RED(k, l) sees the λ_{k,i}, i < l, that later
                // steps will consume. Empty when k = 2.
                for l in (1..=k - 2).rev() {
                    red(basis, &mut lam, &d, k, l);
                }
                k += 1;
                break;
            }
        }
    }
}

/// Inner product `⟨u, v⟩` over `ℤ`.
///
/// The zip stops at the shorter operand rather than panicking on a length
/// mismatch; the caller has already asserted that all rows share one length,
/// so a truncation here would be a silently wrong Gram entry.
fn dot(u: &[BigInt], v: &[BigInt]) -> BigInt {
    let mut acc = BigInt::zero();
    for (a, b) in u.iter().zip(v.iter()) {
        acc = acc.add(&a.mul(b));
    }
    acc
}

/// `BigInt` from a δ component, accepting the full `u64` range (converting
/// through `BigUint` rather than `i64`, so no component is out of reach).
fn big_u64(x: u64) -> BigInt {
    BigInt::from_biguint(BigUint::from_u64(x))
}

/// Nearest integer to `a / b` for `b > 0`, ties toward `+∞`.
///
/// Equal to `⌊(2a + b) / (2b)⌋` (floor toward `−∞`), which realises Cohen's
/// `⌊x + 1/2⌋` rounding used by the size-reduction step. Formed from the two
/// integers rather than from a quotient, so no rational or floating-point
/// value appears; `b > 0` is a precondition of [`floor_div`] and holds here
/// because `b` is always a Gram determinant `d_l`, which is positive for an
/// independent basis.
fn nearest_int(a: &BigInt, b: &BigInt) -> BigInt {
    let two_a_plus_b = a.add(a).add(b);
    let two_b = b.add(b);
    floor_div(&two_a_plus_b, &two_b)
}

/// `⌊num / den⌋` (floor toward `−∞`) for `den > 0`.
///
/// `BigInt` exposes no signed division, so the quotient is formed from the
/// unsigned magnitudes and corrected for a negative dividend: truncating
/// division rounds toward zero, which for a negative dividend is one too
/// large unless the division was exact.
///
/// `den > 0` is unchecked, and the sign of `den` is not consulted — a
/// negative denominator would yield `⌊num / |den|⌋` with the wrong sign.
/// Every call arrives through [`nearest_int`] with a positive Gram
/// determinant.
fn floor_div(num: &BigInt, den: &BigInt) -> BigInt {
    // The caller's contract: only `nearest_int` calls this, always with a Gram
    // determinant, which is positive for an independent basis. The sign of
    // `den` is never inspected below, so a negative one would silently yield
    // ⌊num/|den|⌋ — check rather than trust.
    debug_assert!(
        den.sign() == Sign::Positive,
        "floor_div requires a positive divisor"
    );
    let (quotient, remainder) = num.magnitude().div_rem(den.magnitude());
    if num.sign() != Sign::Negative {
        BigInt::from_biguint(quotient)
    } else if remainder.is_zero() {
        BigInt::from_biguint(quotient).negated()
    } else {
        BigInt::from_biguint(quotient.add(&BigUint::from_u64(1))).negated()
    }
}

/// `RED(k, l)` for `l < k`: subtract the nearest integer multiple of row `l`
/// from row `k` so that `|μ_{k,l}| ≤ 1/2`, and carry the change through the
/// λ bookkeeping (Cohen 2.6.3, sub-algorithm RED).
///
/// Subtracting an integer multiple of `b_l` from `b_k` is a unimodular
/// column operation, so the lattice is unchanged; and since `l < k` it does
/// not disturb `b*_k` or any `d_i`, only the `λ_{k,i}` for `i ≤ l`. Those are
/// updated in the same pass: `λ_{k,l}` loses `q·d_l` and each `λ_{k,i}`,
/// `i < l`, loses `q·λ_{l,i}` — the integral image of `μ_{k,·} ← μ_{k,·} −
/// q·μ_{l,·}`.
fn red(basis: &mut [Vec<BigInt>], lam: &mut [Vec<BigInt>], d: &[BigInt], k: usize, l: usize) {
    // Nothing to do when |2·λ_{k,l}| ≤ d_l, i.e. |μ_{k,l}| ≤ 1/2: d_l > 0, so
    // the magnitude comparison is the comparison of the values.
    let two_lam = lam[k][l].add(&lam[k][l]);
    if *two_lam.magnitude() <= *d[l].magnitude() {
        return;
    }
    let qnt = nearest_int(&lam[k][l], &d[l]);

    // b_k ← b_k − q·b_l. Split the slice so row l is read while row k is
    // written (l < k, so l lands in the left half).
    let (left, right) = basis.split_at_mut(k - 1);
    let bl = &left[l - 1];
    let bk = &mut right[0];
    for c in 0..bk.len() {
        bk[c] = bk[c].sub(&qnt.mul(&bl[c]));
    }

    // λ_{k,l} ← λ_{k,l} − q·d_l; λ_{k,i} ← λ_{k,i} − q·λ_{l,i} for i < l.
    let (lleft, lright) = lam.split_at_mut(k);
    let lam_l = &lleft[l];
    let lam_k = &mut lright[0];
    lam_k[l] = lam_k[l].sub(&qnt.mul(&d[l]));
    for i in 1..l {
        lam_k[i] = lam_k[i].sub(&qnt.mul(&lam_l[i]));
    }
}

/// `SWAP(k)`: exchange rows `k` and `k-1` and restore the integral
/// Gram–Schmidt data (Cohen 2.6.3, sub-algorithm SWAP). Every division is
/// exact.
///
/// Two quantities are deliberately left alone, and the correctness of the
/// routine depends on it:
///
/// - `d_k` is the Gram determinant of `b_1, …, b_k`, which is invariant
///   under a permutation of those very vectors, so exchanging rows `k` and
///   `k−1` cannot change it. Only `d_{k−1}` moves, to `B`.
/// - `λ_{k,k−1}` is likewise invariant. Writing the new `b*_{k−1}` as
///   `b*_k + μ_{k,k−1}·b*_{k−1}` and using `⟨b_{k−1}, b*_k⟩ = 0`, the new
///   `λ_{k,k−1} = d_{k−2}·⟨b_{k−1}, b*_k + μ_{k,k−1} b*_{k−1}⟩` collapses to
///   `d_{k−1}·μ_{k,k−1}`, the old value. Hence `lam[k][k-1]` is read as
///   `lambda` and never written.
///
/// The rows above `k` do move: for each `i` in `k+1 ..= k_max` the pair
/// `(λ_{i,k−1}, λ_{i,k})` is rewritten, and the second assignment uses the
/// `λ_{i,k}` produced by the first, not the saved `t`.
fn swap_step(
    basis: &mut [Vec<BigInt>],
    lam: &mut [Vec<BigInt>],
    d: &mut [BigInt],
    k: usize,
    k_max: usize,
) {
    basis.swap(k - 1, k - 2);
    // Exchange the already-computed λ_{k,j} and λ_{k-1,j} for j < k-1. Row k-1
    // sits in the left half of the split, row k as the first of the right.
    if k > 2 {
        let (lo, hi) = lam.split_at_mut(k);
        for j in 1..=k - 2 {
            core::mem::swap(&mut lo[k - 1][j], &mut hi[0][j]);
        }
    }

    let lambda = lam[k][k - 1].clone();
    // B = (d_{k-2}·d_k + λ²) / d_{k-1}, the new d_{k-1}. The swap condition
    // that brought us here is exactly d_{k-2}·d_k + λ² < δ·d_{k-1}², so
    // B < δ·d_{k-1} < d_{k-1}: the strict decrease that bounds the number of
    // swaps. B > 0 because d_{k-2} and d_k are.
    let b_new = d[k - 2]
        .mul(&d[k])
        .add(&lambda.mul(&lambda))
        .div_exact(&d[k - 1]);

    for row in lam.iter_mut().take(k_max + 1).skip(k + 1) {
        let t = row[k].clone();
        // λ_{i,k} ← (d_k·λ_{i,k-1} − λ·t) / d_{k-1}.
        row[k] = d[k]
            .mul(&row[k - 1])
            .sub(&lambda.mul(&t))
            .div_exact(&d[k - 1]);
        // λ_{i,k-1} ← (B·t + λ·λ_{i,k}) / d_k, using the updated λ_{i,k}.
        row[k - 1] = b_new.mul(&t).add(&lambda.mul(&row[k])).div_exact(&d[k]);
    }
    d[k - 1] = b_new;
}

// ─── Two-dimensional reduction under a diagonal form ───────────────────────

/// Why [`gauss_reduce_weighted`] could not reduce.
///
/// There is no weight variant: the weights are [`NonZeroU64`], so a
/// non-positive weight cannot be expressed in the first place.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ReductionError {
    /// The two vectors are linearly dependent, so they are not a basis.
    DependentBasis,
    /// The arithmetic left `i128`. The rounding step forms `2⟨u,v⟩ + ‖u‖²`
    /// over `2‖u‖²`, so it is twice the norm that must be representable:
    /// every vector the reduction visits needs `(w₀·x)² + (w₁·y)² < 2¹²⁶`.
    /// A wrapped norm compares wrongly and would return an unreduced basis
    /// with no indication, so this is refused instead.
    OutOfRange,
}

impl core::fmt::Display for ReductionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::DependentBasis => "the two vectors are linearly dependent",
            Self::OutOfRange => "the weighted arithmetic does not fit i128",
        })
    }
}

impl std::error::Error for ReductionError {}

/// The squared length of `v` under the diagonal form, `(w₀·v₀)² + (w₁·v₁)²`.
///
/// `None` on overflow rather than a wrapped answer: a wrapped norm compares
/// wrongly and would return a basis that is not reduced, silently.
fn weighted_norm_sq(v: [i128; 2], weights: [i128; 2]) -> Option<i128> {
    let x = weights[0].checked_mul(v[0])?;
    let y = weights[1].checked_mul(v[1])?;
    x.checked_mul(x)?.checked_add(y.checked_mul(y)?)
}

/// The inner product matching [`weighted_norm_sq`]: `w₀²·u₀·v₀ + w₁²·u₁·v₁`,
/// formed as `(w₀u₀)(w₀v₀) + (w₁u₁)(w₁v₁)` so the intermediates stay the same
/// size as the norm's.
fn weighted_dot(u: [i128; 2], v: [i128; 2], weights: [i128; 2]) -> Option<i128> {
    let ux = weights[0].checked_mul(u[0])?;
    let vx = weights[0].checked_mul(v[0])?;
    let uy = weights[1].checked_mul(u[1])?;
    let vy = weights[1].checked_mul(v[1])?;
    ux.checked_mul(vx)?.checked_add(uy.checked_mul(vy)?)
}

/// `round(numerator / denominator)` for a positive `denominator`, exactly.
///
/// Ties go to the larger quotient. Which way ties break does not affect
/// correctness — both choices leave `|⟨u,v⟩| ≤ ‖u‖²/2` — only which of two
/// equally reduced bases comes back.
fn round_div(numerator: i128, denominator: i128) -> Option<i128> {
    debug_assert!(denominator > 0);
    // Every one of these must fit, which is why the documented bound is on
    // twice the norm rather than the norm: a basis whose norms reach the top
    // of `i128` fails here, not in `weighted_norm_sq`.
    let doubled = numerator.checked_mul(2)?;
    let shifted = doubled.checked_add(denominator)?;
    Some(shifted.div_euclid(denominator.checked_mul(2)?))
}

/// Lagrange–Gauss reduction of a two-dimensional basis under the diagonal
/// form `‖(x, y)‖² = (w₀·x)² + (w₁·y)²`, exactly, in machine integers.
///
/// Returns the two vectors in non-decreasing order of that norm. The first is
/// a shortest non-zero vector of the lattice and the second is shortest among
/// those independent of it — in two dimensions reduction is not a heuristic,
/// as it is for [`lll_reduce`](crate::lattice::lll_reduce) in general dimension, but solves the shortest
/// vector problem outright (Lagrange 1773; Gauss, *Disquisitiones
/// Arithmeticae* 1801, art. 171; the modern analysis is Vallée, *Gauss'
/// algorithm revisited*, J. Algorithms 12 (1991), 556–572).
///
/// The weights make the metric anisotropic, which is what a skewed lattice
/// wants. To reduce under the skewed form `(x/√s)² + (y·√s)²` for a **rational**
/// skew `s = p/q`, multiply through by `pq` — scaling a quadratic form by a
/// positive constant changes no comparison and no rounding — which clears the
/// square roots and gives `(q·x)² + (p·y)²`, so pass `weights = [q, p]`.
///
/// An *integer* skew `s` is the case `q = 1`, `weights = [1, s]`. A skew that
/// is not rational has to be approximated first, and then this reduces exactly
/// under the approximating form rather than under the intended one: the
/// arithmetic is exact, the *form* is only as faithful as `p/q`. Choosing that
/// approximation is the caller's, and it is a real choice — the norms grow
/// like the square of the weights, so a denominator bought for precision is
/// paid for out of the range below.
///
/// Within the form actually given there is no accuracy cliff, which is the
/// point: a floating-point metric loses the ordering once the weighted
/// coordinates pass `2⁵³` and degrades quietly to a poor basis, where this
/// either answers exactly or refuses.
///
/// # Termination
///
/// Each iteration replaces the shorter vector with a strictly shorter one, so
/// the sequence of squared norms is strictly decreasing in the positive
/// integers and the loop runs at most `log` many times. No iteration cap is
/// needed and none is imposed — a cap here could only turn a correct answer
/// into a wrong one.
///
/// `None` rather than a panic in every rejecting case, so a caller can test
/// rather than guard: the vectors are linearly dependent (a zero determinant
/// is not a basis), a weight is not positive, or the arithmetic leaves
/// `i128`.
///
/// That last is a real restriction and tighter than it looks. Write `S` for
/// the largest weighted squared norm the reduction visits. The rounding step
/// forms `2·⟨u,v⟩ + ‖u‖²`, and `|⟨u,v⟩| ≤ S` by Cauchy–Schwarz, so the
/// largest intermediate is `3S` and the working condition is
///
/// ```text
/// 3·S ≤ i128::MAX,   i.e.   S ≤ (2¹²⁷ − 1)/3 ≈ 2^125.415
/// ```
///
/// Per coordinate: `|w₀·x| ≤ M` and `|w₁·y| ≤ M` give `S ≤ 2M²` and an
/// intermediate of at most `6M²`, so `M ≤ 2^62.2075`. Keeping each of
/// `|w₀·x|` and `|w₁·y|` below `2⁶²` therefore holds comfortably — but `2⁶³`
/// does not, since `6·(2⁶³)² = 2^128.585`.
///
/// This bound was previously stated as `(w₀·x)² + (w₁·y)² < 2¹²⁶`, which is
/// one and a half times too generous: a basis with `S` between `2^125.415`
/// and `2¹²⁶` satisfies it and is still refused, and a downstream reader who
/// derived a per-coordinate rail from it arrived at `2⁶³`, which overflows.
/// The refusal was always correct; only the sentence was not.
///
/// A basis whose norms fill `i128` to the top yields an error rather than a
/// wrapped answer — a wrapped norm compares wrongly and would return an
/// unreduced basis with no indication.
pub fn gauss_reduce_weighted(
    basis: [[i128; 2]; 2],
    weights: [NonZeroU64; 2],
) -> Result<[[i128; 2]; 2], ReductionError> {
    let weights = [i128::from(weights[0].get()), i128::from(weights[1].get())];
    let range = || ReductionError::OutOfRange;

    let determinant = basis[0][0]
        .checked_mul(basis[1][1])
        .and_then(|a| a.checked_sub(basis[0][1].checked_mul(basis[1][0])?))
        .ok_or_else(range)?;
    if determinant == 0 {
        return Err(ReductionError::DependentBasis);
    }

    let mut u = basis[0];
    let mut v = basis[1];
    let mut norm_u = weighted_norm_sq(u, weights).ok_or_else(range)?;
    if norm_u > weighted_norm_sq(v, weights).ok_or_else(range)? {
        core::mem::swap(&mut u, &mut v);
        norm_u = weighted_norm_sq(u, weights).ok_or_else(range)?;
    }

    loop {
        // `norm_u > 0` throughout: the determinant is non-zero, so neither
        // vector is zero, and the weights are positive by construction.
        let dot = weighted_dot(u, v, weights).ok_or_else(range)?;
        let q = round_div(dot, norm_u).ok_or_else(range)?;
        let r = [
            v[0].checked_sub(q.checked_mul(u[0]).ok_or_else(range)?)
                .ok_or_else(range)?,
            v[1].checked_sub(q.checked_mul(u[1]).ok_or_else(range)?)
                .ok_or_else(range)?,
        ];
        let norm_r = weighted_norm_sq(r, weights).ok_or_else(range)?;
        if norm_r >= norm_u {
            // `u` is a shortest vector; `r` is reduced against it.
            return Ok([u, r]);
        }
        v = u;
        u = r;
        norm_u = norm_r;
    }
}

/// The lattice vectors of squared norm at most `bound` under the
/// positive-definite integral form `form`, shortest first, at most
/// `limit` of them, each as a lattice vector in the basis's coordinates
/// (not as coefficients).
///
/// Schnorr–Euchner enumeration (Schnorr & Euchner, *Lattice basis
/// reduction: improved practical algorithms and solving subset sum
/// problems*, Math. Programming 66 (1994), 181–199, the enumeration of
/// Fincke & Pohst, *Improved methods for calculating vectors of short
/// length in a lattice*, Math. Comp. 44 (1985), 463–471, with the
/// zig-zag order of coefficients): the basis's Gram matrix under the
/// form is taken exactly, brought to doubles by a common shift, and
/// Cholesky-decomposed; the search runs over integer coefficient vectors
/// from the last coordinate down, pruning by the partial norm; and every
/// vector the floating-point search admits has its norm recomputed
/// exactly before it is kept, so rounding can only ever offer an extra
/// candidate, never lose one within the slack. The basis should be
/// LLL-reduced first — the enumeration's cost is what reduction buys.
///
/// What it is for: LLL returns a reduced basis, and a reduced basis's
/// rows are not the lattice's shortest vectors, only vectors within a
/// factor of them; where the shortest vectors matter — the kernel
/// lattice of a number field sieve polynomial search, whose vectors are
/// the polynomials with a given root — the rows are a sample of what the
/// lattice holds, and this is the rest of it.
///
/// # Panics
///
/// As [`lll_reduce_form`] on a malformed basis or form, and if `bound`
/// is negative.
#[must_use]
pub fn short_vectors_form(
    basis: &[Vec<BigInt>],
    form: &[Vec<BigInt>],
    bound: &BigInt,
    limit: usize,
) -> Vec<Vec<BigInt>> {
    let n = basis.len();
    if n == 0 || limit == 0 || bound.sign() == Sign::Negative {
        return Vec::new();
    }
    let m = basis[0].len();
    assert!(
        form.len() == m && form.iter().all(|row| row.len() == m),
        "the form must be square of the vectors' length"
    );
    // The Gram matrix under the form, exactly.
    let form_dot = |u: &[BigInt], v: &[BigInt]| -> BigInt {
        let mut total = BigInt::zero();
        for (i, a) in u.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in v.iter().enumerate() {
                if !form[i][j].is_zero() && !b.is_zero() {
                    total = total.add(&a.mul(b).mul(&form[i][j]));
                }
            }
        }
        total
    };
    let gram: Vec<Vec<BigInt>> = (0..n)
        .map(|i| (0..n).map(|j| form_dot(&basis[i], &basis[j])).collect())
        .collect();
    // To doubles by a common shift: the Cholesky factors want relative
    // precision, and the entries can be thousands of bits.
    let widest = gram
        .iter()
        .flatten()
        .map(|g| g.magnitude().bits())
        .max()
        .unwrap_or(0)
        .max(bound.magnitude().bits());
    let shift = widest.saturating_sub(900);
    let to_double = |x: &BigInt| -> f64 {
        let mut magnitude = x.magnitude().clone();
        magnitude.shr_bits(shift);
        let value = magnitude.to_f64_lossy();
        if x.sign() == Sign::Negative {
            -value
        } else {
            value
        }
    };
    let g: Vec<Vec<f64>> = gram
        .iter()
        .map(|row| row.iter().map(to_double).collect())
        .collect();
    let radius = to_double(bound) * (1.0 + 1e-9) + f64::MIN_POSITIVE;
    // Gram–Schmidt from the Gram matrix: mu[i][j] for j < i and the
    // squared norms r[i] of the orthogonalised vectors.
    let mut mu = vec![vec![0.0f64; n]; n];
    let mut r = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = g[i][j];
            for k in 0..j {
                value -= mu[i][k] * mu[j][k] * r[k];
            }
            if j < i {
                mu[i][j] = if r[j] > 0.0 { value / r[j] } else { 0.0 };
            } else {
                r[i] = value;
            }
        }
        assert!(r[i] > 0.0, "the form is not positive definite on the basis");
    }
    // The enumeration: coefficient vector x, from coordinate n−1 down.
    let mut found: Vec<(BigInt, Vec<BigInt>)> = Vec::new();
    let mut x = vec![0i64; n];
    let mut centre = vec![0.0f64; n];
    let mut partial = vec![0.0f64; n + 1];
    let mut step = vec![0i64; n];
    // The top coordinate starts at its centre, zero, and steps up first.
    step[n - 1] = 1;
    let mut level = n - 1;
    let mut visited = 0u64;
    loop {
        // The partial norm at this level with the current x[level].
        let deviation = x[level] as f64 - centre[level];
        let here = partial[level + 1] + deviation * deviation * r[level];
        if here <= radius {
            if level == 0 {
                // A full vector; the zero vector is skipped.
                if x.iter().any(|&c| c != 0) {
                    let coefficients: Vec<BigInt> =
                        x.iter().map(|&c| BigInt::from_i64(c)).collect();
                    let mut vector = vec![BigInt::zero(); m];
                    for (c, b) in coefficients.iter().zip(basis) {
                        if c.is_zero() {
                            continue;
                        }
                        for (slot, entry) in vector.iter_mut().zip(b) {
                            *slot = slot.add(&c.mul(entry));
                        }
                    }
                    let norm = form_dot(&vector, &vector);
                    if norm <= *bound {
                        found.push((norm, vector));
                    }
                }
                // Next sibling at level 0.
                x[0] += step[0];
                step[0] = -step[0] - step[0].signum();
                if step[0] == 0 {
                    step[0] = 1;
                }
                visited += 1;
            } else {
                // Descend: the centre of the next coordinate.
                partial[level] = here;
                level -= 1;
                let mut c = 0.0;
                for k in (level + 1)..n {
                    c -= mu[k][level] * x[k] as f64;
                }
                centre[level] = c;
                x[level] = c.round() as i64;
                step[level] = if c >= x[level] as f64 { 1 } else { -1 };
            }
        } else {
            // Exhausted this level's zig-zag: back up.
            if level == n - 1 {
                break;
            }
            level += 1;
            x[level] += step[level];
            step[level] = -step[level] - step[level].signum();
            if step[level] == 0 {
                step[level] = 1;
            }
            visited += 1;
        }
        if visited > 50_000_000 {
            break;
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    found.truncate(limit);
    found.into_iter().map(|(_, v)| v).collect()
}

#[cfg(test)]
mod short_vector_tests {
    use super::*;

    fn big(x: i64) -> BigInt {
        BigInt::from_i64(x)
    }

    /// Every vector the enumeration returns is a lattice vector within
    /// the bound, and it returns every such vector: checked against an
    /// exhaustive search over a box of coefficients large enough to hold
    /// them, on random small bases under the identity and a diagonal
    /// form.
    #[test]
    fn the_enumeration_agrees_with_an_exhaustive_search() {
        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as i64
        };
        for trial in 0..40 {
            let n = 3 + (trial % 2);
            let basis: Vec<Vec<BigInt>> = (0..n)
                .map(|_| (0..n).map(|_| big(next() % 9 - 4)).collect())
                .collect();
            let form: Vec<Vec<BigInt>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            if i == j {
                                big(1 + (trial as i64 * (i as i64 + 1)) % 5)
                            } else {
                                big(0)
                            }
                        })
                        .collect()
                })
                .collect();
            let mut reduced = basis.clone();
            // A dependent random basis is possible; skip those.
            let gram_det_zero = {
                let mut m: Vec<Vec<f64>> = basis
                    .iter()
                    .map(|r| r.iter().map(|x| x.to_f64_lossy()).collect())
                    .collect();
                let mut det = 1.0;
                for i in 0..n {
                    let pivot = (i..n)
                        .max_by(|&a, &b| m[a][i].abs().partial_cmp(&m[b][i].abs()).unwrap())
                        .unwrap();
                    m.swap(i, pivot);
                    if m[i][i].abs() < 1e-9 {
                        det = 0.0;
                        break;
                    }
                    det *= m[i][i];
                    for k in (i + 1)..n {
                        let factor = m[k][i] / m[i][i];
                        let (pivot, below) = m.split_at_mut(k);
                        for (target, &source) in below[0][i..n].iter_mut().zip(&pivot[i][i..n]) {
                            *target -= factor * source;
                        }
                    }
                }
                det.abs() < 0.5
            };
            if gram_det_zero {
                continue;
            }
            lll_reduce_form(&mut reduced, &form, 3, 4);
            let norm = |v: &[BigInt]| -> BigInt {
                let mut total = BigInt::zero();
                for (i, a) in v.iter().enumerate() {
                    total = total.add(&a.mul(a).mul(&form[i][i]));
                }
                total
            };
            let bound = norm(&reduced[0]).mul(&big(4));
            let found = short_vectors_form(&reduced, &form, &bound, usize::MAX);
            // Exhaustive: coefficients in a box; the reduced basis's vectors are
            // short, so a box of radius six around zero holds every vector of
            // twice the shortest's length in these dimensions.
            let radius = 6i64;
            let mut expected: Vec<Vec<BigInt>> = Vec::new();
            let count = (2 * radius + 1).pow(n as u32);
            for code in 0..count {
                let mut c = code;
                let coefficients: Vec<i64> = (0..n)
                    .map(|_| {
                        let value = c % (2 * radius + 1) - radius;
                        c /= 2 * radius + 1;
                        value
                    })
                    .collect();
                if coefficients.iter().all(|&x| x == 0) {
                    continue;
                }
                let mut v = vec![BigInt::zero(); n];
                for (k, &coefficient) in coefficients.iter().enumerate() {
                    for (slot, entry) in v.iter_mut().zip(&reduced[k]) {
                        *slot = slot.add(&big(coefficient).mul(entry));
                    }
                }
                if norm(&v) <= bound {
                    expected.push(v);
                }
            }
            let key = |v: &Vec<BigInt>| {
                v.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let mut found_keys: Vec<String> = found.iter().map(key).collect();
            let mut expected_keys: Vec<String> = expected.iter().map(key).collect();
            found_keys.sort();
            expected_keys.sort();
            assert_eq!(found_keys, expected_keys, "trial {trial}: dimension {n}");
            // And in order of norm.
            for pair in found.windows(2) {
                assert!(norm(&pair[0]) <= norm(&pair[1]));
            }
        }
    }

    #[test]
    fn the_limit_keeps_the_shortest() {
        let basis = vec![vec![big(7), big(0)], vec![big(3), big(1)]];
        let form = vec![vec![big(1), big(0)], vec![big(0), big(1)]];
        let mut reduced = basis.clone();
        lll_reduce_form(&mut reduced, &form, 3, 4);
        let found = short_vectors_form(&reduced, &form, &big(100), 2);
        assert_eq!(found.len(), 2);
        // The lattice { (7a + 3b, b) }: its shortest vectors are ±(1, 2)... (7·(-1) + 3·... ) — check by norm only.
        let norm = |v: &[BigInt]| v[0].mul(&v[0]).add(&v[1].mul(&v[1]));
        assert_eq!(norm(&found[0]), norm(&found[1]));
        let negated: Vec<BigInt> = found[1].iter().map(BigInt::negated).collect();
        assert_eq!(found[0], negated);
    }
}

#[cfg(test)]
mod tests {
    use super::lll_reduce_form;

    use super::{gauss_reduce_weighted, lll_reduce, lll_reduce_delta, weighted_norm_sq};
    use crate::bigint::{BigInt, Sign};
    use core::num::NonZeroU64;

    /// Weights as the signature now takes them.
    fn w(a: u64, b: u64) -> [NonZeroU64; 2] {
        [
            NonZeroU64::new(a).expect("test weight is non-zero"),
            NonZeroU64::new(b).expect("test weight is non-zero"),
        ]
    }

    fn det(basis: [[i128; 2]; 2]) -> i128 {
        basis[0][0] * basis[1][1] - basis[0][1] * basis[1][0]
    }

    /// Is `v` an integer combination of `basis`? Cramer's rule, with the
    /// solution required to be exact rather than merely close.
    fn in_lattice(basis: [[i128; 2]; 2], v: [i128; 2]) -> bool {
        let d = det(basis);
        assert!(d != 0);
        let a = v[0] * basis[1][1] - v[1] * basis[1][0];
        let b = basis[0][0] * v[1] - basis[0][1] * v[0];
        a % d == 0 && b % d == 0
    }

    /// Nothing in a small window around a *reduced* basis is shorter than its
    /// first vector.
    ///
    /// This is the minimality check, and it is stated over the reduced basis
    /// on purpose. Searching combinations of the *input* basis is not a valid
    /// oracle: two nearly parallel generators need large coefficients to
    /// express the short vectors, so a fixed window silently misses them and
    /// the test then reports the reduction wrong when it is right. Over a
    /// reduced basis no window is needed beyond ±2 — with `2|⟨u,v⟩| ≤ ‖u‖²`
    /// and `‖u‖ ≤ ‖v‖`, the norm of `a·u + b·v` is at least
    /// `(a² − |ab| + b²)‖u‖²`, and `a² − |ab| + b²` exceeds 1 for every
    /// integer pair outside `{(±1,0), (0,±1), ±(1,1), ±(1,−1)}`. ±4 is taken
    /// for margin.
    fn nothing_shorter_nearby(reduced: [[i128; 2]; 2], weights: [i128; 2], best: i128) {
        for a in -4i128..=4 {
            for b in -4i128..=4 {
                if a == 0 && b == 0 {
                    continue;
                }
                let v = [
                    a * reduced[0][0] + b * reduced[1][0],
                    a * reduced[0][1] + b * reduced[1][1],
                ];
                if let Some(n) = weighted_norm_sq(v, weights) {
                    assert!(
                        n >= best,
                        "combination ({a},{b}) of {reduced:?} is shorter: {n} < {best}"
                    );
                }
            }
        }
    }

    #[test]
    fn gauss_reduce_finds_the_shortest_vector_under_a_weighted_norm() {
        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as i64) as i128
        };
        for weights in [w(1, 1), w(1, 2), w(1, 7), w(3, 5), w(1, 1000), w(64, 1)] {
            for _ in 0..200 {
                let basis = [
                    [next() % 4096, next() % 4096],
                    [next() % 4096, next() % 4096],
                ];
                if det(basis) == 0 {
                    continue;
                }
                let reduced = gauss_reduce_weighted(basis, weights).expect("a valid basis");
                let weights = [i128::from(weights[0].get()), i128::from(weights[1].get())];

                // The reduction returns a basis of the *same* lattice: the
                // determinant is preserved up to sign.
                assert_eq!(det(reduced).abs(), det(basis).abs(), "lattice changed");

                // Both vectors come from the original lattice, and the equal
                // determinant above rules out a proper sublattice.
                assert!(in_lattice(basis, reduced[0]), "left the lattice");
                assert!(in_lattice(basis, reduced[1]), "left the lattice");

                let n0 = weighted_norm_sq(reduced[0], weights).expect("fits");
                let n1 = weighted_norm_sq(reduced[1], weights).expect("fits");
                assert!(n0 <= n1, "returned out of order: {n0} > {n1}");
                nothing_shorter_nearby(reduced, weights, n0);

                // Reduced means the projection is at most a half step: this
                // is the defining property, independent of the search above.
                let dot = super::weighted_dot(reduced[0], reduced[1], weights).expect("fits");
                assert!(2 * dot.abs() <= n0, "not size-reduced");
            }
        }
    }

    /// A skewed sieve metric `(x/√s)² + (y·√s)²` for a *rational* `s = p/q`
    /// is `weights = [q, p]`, after multiplying the form through by `pq`.
    ///
    /// The integer skew this test used to take is the easy case (`q = 1`) and
    /// the one that does not occur: a sieve's skew is the argmin of a search
    /// and is not an integer, so rounding it to one reduces under a different
    /// form and can return a vector that is longer under the metric actually
    /// wanted. That is what this checks — against the float metric the caller
    /// means, not against the integer one the reduction was handed.
    #[test]
    fn gauss_reduce_weights_encode_a_rational_skew() {
        // A non-integer skew of the shape `skew_for` produces.
        let (p, q) = (2_113_745_839u64, 10_000_000u64); // s ≈ 211.3745839
        let s = p as f64 / q as f64;
        let float_norm = |v: [i128; 2]| {
            let a = v[0] as f64 / s.sqrt();
            let b = v[1] as f64 * s.sqrt();
            a * a + b * b
        };
        for basis in [
            [[20_003i128, 0], [12_577, 1]],
            [[65_537, 0], [4_099, 1]],
            [[1024, 0], [37, 1]],
        ] {
            let reduced = gauss_reduce_weighted(basis, w(q, p)).expect("a valid basis");
            assert_eq!(det(reduced).abs(), det(basis).abs());
            // Ordered under the metric the caller actually means.
            assert!(
                float_norm(reduced[0]) <= float_norm(reduced[1]) * (1.0 + 1e-12),
                "out of order under the intended metric"
            );
            // And no nearby combination is shorter under that metric either.
            for a in -3i128..=3 {
                for b in -3i128..=3 {
                    if a == 0 && b == 0 {
                        continue;
                    }
                    let v = [
                        a * reduced[0][0] + b * reduced[1][0],
                        a * reduced[0][1] + b * reduced[1][1],
                    ];
                    assert!(
                        float_norm(v) >= float_norm(reduced[0]) * (1.0 - 1e-12),
                        "({a},{b}) beats the answer under the intended metric"
                    );
                }
            }
        }
    }

    /// Twice the norm must be representable, not the norm: the rounding step
    /// forms `2⟨u,v⟩ + ‖u‖²` over `2‖u‖²`. This basis is already reduced and
    /// its norms fit `i128` with room to spare, so it must come back
    /// unchanged rather than panic — the case the documented bound used to
    /// admit and the code used to refuse.
    #[test]
    fn gauss_reduce_accepts_norms_that_fill_half_the_range() {
        let a = 1i128 << 62;
        let reduced = gauss_reduce_weighted([[a, a], [a, -a]], w(1, 1)).expect("norms fit");
        assert_eq!(det(reduced).abs(), 2 * a * a);
        assert_eq!(
            weighted_norm_sq(reduced[0], [1, 1]).expect("fits"),
            2 * a * a
        );
    }

    #[test]
    fn gauss_reduce_leaves_an_already_reduced_basis_alone() {
        // The standard basis is reduced under any weights.
        let basis = [[1i128, 0], [0, 1]];
        assert_eq!(gauss_reduce_weighted(basis, w(1, 1)), Ok([[1, 0], [0, 1]]));
        // Under a heavy y-weight the x-axis vector is the shorter one.
        assert_eq!(
            gauss_reduce_weighted(basis, w(1, 100)),
            Ok([[1, 0], [0, 1]])
        );
        // And under a heavy x-weight the order flips.
        assert_eq!(
            gauss_reduce_weighted(basis, w(100, 1)),
            Ok([[0, 1], [1, 0]])
        );
    }

    /// Every rejection is a typed error, so a caller matches rather than
    /// guards. There is no weight case: `NonZeroU64` makes it unrepresentable.
    #[test]
    fn gauss_reduce_reports_bad_input_as_an_error() {
        use super::ReductionError;
        // Dependent: the second row is half the first, so no basis.
        assert_eq!(
            gauss_reduce_weighted([[2, 4], [1, 2]], w(1, 1)),
            Err(ReductionError::DependentBasis)
        );
        // Past the range: the weighted coordinate squares.
        let big = 1i128 << 100;
        assert_eq!(
            gauss_reduce_weighted([[big, 0], [0, 1]], w(1, 1)),
            Err(ReductionError::OutOfRange)
        );
        // And a valid basis still comes back.
        assert!(gauss_reduce_weighted([[1, 0], [0, 1]], w(1, 1)).is_ok());
    }

    fn rows(data: &[&[i64]]) -> Vec<Vec<BigInt>> {
        data.iter()
            .map(|r| r.iter().map(|&x| BigInt::from_i64(x)).collect())
            .collect()
    }

    // --- Independent exact-rational Gram–Schmidt oracle (BigInt fractions).
    // Shares no code with the integral d/λ recurrence under test: it computes
    // μ_{i,j} and ‖b*_i‖² directly from the definition and checks the two
    // LLL properties against them.
    #[derive(Clone)]
    struct Frac {
        n: BigInt,
        d: BigInt, // always > 0, reduced
    }

    impl Frac {
        fn int(a: BigInt) -> Self {
            Self {
                n: a,
                d: BigInt::one(),
            }
        }
        fn reduced(mut n: BigInt, mut d: BigInt) -> Self {
            assert!(!d.is_zero(), "zero denominator");
            if d.sign() == Sign::Negative {
                n = n.negated();
                d = d.negated();
            }
            if n.is_zero() {
                return Self::int(BigInt::zero());
            }
            let g = n.gcd(&d); // non-negative
            Self {
                n: n.div_exact(&g),
                d: d.div_exact(&g),
            }
        }
        fn add(&self, o: &Self) -> Self {
            Self::reduced(self.n.mul(&o.d).add(&o.n.mul(&self.d)), self.d.mul(&o.d))
        }
        fn sub(&self, o: &Self) -> Self {
            Self::reduced(self.n.mul(&o.d).sub(&o.n.mul(&self.d)), self.d.mul(&o.d))
        }
        fn mul(&self, o: &Self) -> Self {
            Self::reduced(self.n.mul(&o.n), self.d.mul(&o.d))
        }
        fn div(&self, o: &Self) -> Self {
            assert!(!o.n.is_zero(), "division by zero fraction");
            Self::reduced(self.n.mul(&o.d), self.d.mul(&o.n))
        }
        // self ≥ o, both denominators positive.
        fn ge(&self, o: &Self) -> bool {
            self.n.mul(&o.d) >= o.n.mul(&self.d)
        }
        // |self| ≤ 1/2  ⟺  2|n| ≤ d.
        fn abs_le_half(&self) -> bool {
            let two_n = self.n.add(&self.n);
            *two_n.magnitude() <= *self.d.magnitude()
        }
    }

    fn dot_frac(u: &[Frac], v: &[Frac]) -> Frac {
        let mut acc = Frac::int(BigInt::zero());
        for (a, b) in u.iter().zip(v.iter()) {
            acc = acc.add(&a.mul(b));
        }
        acc
    }

    // Gram–Schmidt of an integer basis: returns (‖b*_i‖², μ_{i,j}). Index
    // loops here mirror the textbook recurrence and cross-index one matrix.
    #[allow(clippy::needless_range_loop)]
    fn gram_schmidt(basis: &[Vec<BigInt>]) -> (Vec<Frac>, Vec<Vec<Frac>>) {
        let n = basis.len();
        let m = basis[0].len();
        let bi: Vec<Vec<Frac>> = basis
            .iter()
            .map(|r| r.iter().map(|x| Frac::int(x.clone())).collect())
            .collect();
        let mut bstar = vec![vec![Frac::int(BigInt::zero()); m]; n];
        let mut bnorm = vec![Frac::int(BigInt::zero()); n];
        let mut mu = vec![vec![Frac::int(BigInt::zero()); n]; n];
        for i in 0..n {
            bstar[i] = bi[i].clone();
            for j in 0..i {
                mu[i][j] = dot_frac(&bi[i], &bstar[j]).div(&bnorm[j]);
                for c in 0..m {
                    bstar[i][c] = bstar[i][c].sub(&mu[i][j].mul(&bstar[j][c]));
                }
            }
            bnorm[i] = dot_frac(&bstar[i], &bstar[i]);
        }
        (bnorm, mu)
    }

    fn is_reduced(basis: &[Vec<BigInt>], dn: u64, dd: u64) -> bool {
        let n = basis.len();
        if n <= 1 {
            return true;
        }
        let (bnorm, mu) = gram_schmidt(basis);
        for (i, row) in mu.iter().enumerate() {
            if row[..i].iter().any(|muij| !muij.abs_le_half()) {
                return false;
            }
        }
        let delta = Frac::reduced(BigInt::from_i64(dn as i64), BigInt::from_i64(dd as i64));
        for k in 1..n {
            // ‖b*_k‖² ≥ (δ − μ_{k,k-1}²)‖b*_{k-1}‖².
            let mu2 = mu[k][k - 1].mul(&mu[k][k - 1]);
            let rhs = delta.sub(&mu2).mul(&bnorm[k - 1]);
            if !bnorm[k].ge(&rhs) {
                return false;
            }
        }
        true
    }

    // Fraction-free (Bareiss) determinant of the Gram matrix G = B·Bᵀ, the
    // squared covolume — a lattice invariant, so it must survive reduction.
    fn gram_det(basis: &[Vec<BigInt>]) -> BigInt {
        let n = basis.len();
        let mut g: Vec<Vec<BigInt>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let mut s = BigInt::zero();
                        for (a, b) in basis[i].iter().zip(&basis[j]) {
                            s = s.add(&a.mul(b));
                        }
                        s
                    })
                    .collect()
            })
            .collect();
        let mut sign = 1i64;
        let mut prev = BigInt::one();
        for kk in 0..n {
            if g[kk][kk].is_zero() {
                let piv = (kk + 1..n).find(|&r| !g[r][kk].is_zero());
                match piv {
                    Some(r) => {
                        g.swap(kk, r);
                        sign = -sign;
                    }
                    None => return BigInt::zero(),
                }
            }
            for r in (kk + 1)..n {
                for c in (kk + 1)..n {
                    let num = g[kk][kk].mul(&g[r][c]).sub(&g[r][kk].mul(&g[kk][c]));
                    g[r][c] = num.div_exact(&prev);
                }
                g[r][kk] = BigInt::zero();
            }
            prev = g[kk][kk].clone();
        }
        let det = g[n - 1][n - 1].clone();
        if sign < 0 {
            det.negated()
        } else {
            det
        }
    }

    // Deterministic LCG for random small integer bases.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn int(&mut self, lo: i64, hi: i64) -> i64 {
            let span = (hi - lo + 1) as u64;
            lo + (self.next() % span) as i64
        }
    }

    // (name, input, expected reduced basis).
    type Case = (&'static str, Vec<Vec<BigInt>>, Vec<Vec<BigInt>>);

    // The seven locked oracle cases, produced by an independent rational-Fraction
    // LLL (scripts/lll_oracle.py, δ = 3/4) and there verified size-reduced ∧
    // Lovász ∧ determinant-invariant.
    fn oracle_cases() -> Vec<Case> {
        vec![
            (
                "eye3",
                rows(&[&[1, 0, 0], &[0, 1, 0], &[0, 0, 1]]),
                rows(&[&[1, 0, 0], &[0, 1, 0], &[0, 0, 1]]),
            ),
            (
                "cohen_ex",
                rows(&[&[1, 1, 1], &[-1, 0, 2], &[3, 5, 6]]),
                rows(&[&[0, 1, 0], &[1, 0, 1], &[-1, 0, 2]]),
            ),
            (
                "skew",
                rows(&[&[201, 37], &[1648, 297]]),
                rows(&[&[1, 32], &[40, 1]]),
            ),
            (
                "hard4",
                rows(&[
                    &[1, 0, 0, 1345],
                    &[0, 1, 0, 3571],
                    &[0, 0, 1, 8765],
                    &[0, 0, 0, 10007],
                ]),
                rows(&[
                    &[-6, 4, 5, 4],
                    &[-1, 6, -8, -4],
                    &[3, 2, 9, -1],
                    &[-6, -3, 1, -11],
                ]),
            ),
            (
                "neg",
                rows(&[&[-2, 7, 3], &[5, -1, 4], &[0, 6, -8]]),
                rows(&[&[5, -1, 4], &[-2, 7, 3], &[5, 5, -4]]),
            ),
            (
                "collinear_free",
                rows(&[&[2, 4], &[3, 1]]),
                rows(&[&[3, 1], &[-1, 3]]),
            ),
            (
                "big",
                rows(&[
                    &[123456789, 0, 0],
                    &[0, 987654321, 0],
                    &[111111111, 222222222, 333333333],
                ]),
                rows(&[
                    &[123456789, 0, 0],
                    &[-12345678, 222222222, 333333333],
                    &[12345678, 765432099, -333333333],
                ]),
            ),
        ]
    }

    #[test]
    fn lll_matches_rational_oracle_on_fixed_lattices() {
        for (name, input, expected) in oracle_cases() {
            let mut basis = input;
            lll_reduce(&mut basis);
            assert_eq!(basis, expected, "reduced basis for {name}");
        }
    }

    #[test]
    fn lll_preserves_the_lattice_determinant() {
        for (name, input, _) in oracle_cases() {
            let before = gram_det(&input);
            let mut basis = input.clone();
            lll_reduce(&mut basis);
            let after = gram_det(&basis);
            assert_eq!(before, after, "Gram determinant changed for {name}");
        }
    }

    #[test]
    fn lll_output_is_reduced_on_fixed_lattices() {
        for (name, input, _) in oracle_cases() {
            let mut basis = input;
            lll_reduce(&mut basis);
            assert!(is_reduced(&basis, 3, 4), "not LLL-reduced: {name}");
        }
    }

    #[test]
    fn lll_is_idempotent_on_fixed_lattices() {
        for (name, input, expected) in oracle_cases() {
            let mut once = expected.clone();
            lll_reduce(&mut once);
            assert_eq!(once, expected, "already-reduced basis moved: {name}");
            let _ = input;
        }
    }

    #[test]
    fn lll_random_full_rank_bases_reduce_and_preserve_the_lattice() {
        let mut rng = Lcg(0x1234_5678_9abc_def1);
        let mut tested = 0;
        for _ in 0..2000 {
            let n = 2 + (rng.next() % 3) as usize; // 2..=4 vectors
            let m = n + (rng.next() % 2) as usize; // ambient ≥ n
            let input: Vec<Vec<BigInt>> = (0..n)
                .map(|_| (0..m).map(|_| BigInt::from_i64(rng.int(-9, 9))).collect())
                .collect();
            // Skip singular (dependent) draws — lll_reduce requires a basis.
            if gram_det(&input).is_zero() {
                continue;
            }
            let before = gram_det(&input);
            let mut basis = input.clone();
            lll_reduce(&mut basis);
            assert_eq!(gram_det(&basis), before, "determinant changed: {input:?}");
            assert!(is_reduced(&basis, 3, 4), "not reduced: {input:?}");
            // Idempotence: a second pass is a fixed point.
            let mut twice = basis.clone();
            lll_reduce(&mut twice);
            assert_eq!(twice, basis, "not idempotent: {input:?}");
            tested += 1;
        }
        assert!(tested > 1000, "too few non-singular draws: {tested}");
    }

    #[test]
    fn lll_settable_delta_still_reduces() {
        // A small δ (3/5, a loose reduction) and a large one (99/100, near the
        // tight end): each must return a basis reduced for its own δ, with the
        // lattice determinant preserved.
        for &(dn, dd) in &[(3u64, 5u64), (99, 100)] {
            for (_name, input, _) in oracle_cases() {
                let before = gram_det(&input);
                let mut basis = input;
                lll_reduce_delta(&mut basis, dn, dd);
                assert_eq!(gram_det(&basis), before, "det changed at δ={dn}/{dd}");
                assert!(is_reduced(&basis, dn, dd), "not reduced at δ={dn}/{dd}");
            }
        }
    }

    #[test]
    fn lll_accepts_large_delta_components() {
        // δ = 5/6 with a numerator above u64::MAX/4: the range check must not
        // overflow, nor falsely reject a valid δ (rung-D review, objection 1).
        let (dn, dd) = (5_000_000_000_000_000_000u64, 6_000_000_000_000_000_000u64);
        let input = rows(&[&[201, 37], &[1648, 297]]);
        let before = gram_det(&input);
        let mut basis = input;
        lll_reduce_delta(&mut basis, dn, dd);
        assert_eq!(gram_det(&basis), before, "determinant preserved at δ=5/6");
        assert!(is_reduced(&basis, dn, dd), "reduced for δ=5/6");
    }

    #[test]
    fn lll_handles_single_vector_and_empty() {
        let mut one = rows(&[&[3, 4]]);
        lll_reduce(&mut one);
        assert_eq!(one, rows(&[&[3, 4]]), "single vector unchanged");
        let mut none: Vec<Vec<BigInt>> = Vec::new();
        lll_reduce(&mut none); // must not panic
        assert!(none.is_empty());
    }

    #[test]
    #[should_panic(expected = "delta")]
    fn lll_rejects_delta_at_or_above_one() {
        let mut basis = rows(&[&[1, 0], &[0, 1]]);
        lll_reduce_delta(&mut basis, 1, 1);
    }

    #[test]
    #[should_panic(expected = "delta")]
    fn lll_rejects_delta_at_or_below_quarter() {
        let mut basis = rows(&[&[1, 0], &[0, 1]]);
        lll_reduce_delta(&mut basis, 1, 4);
    }

    #[test]
    #[should_panic(expected = "share one length")]
    fn lll_rejects_ragged_rows() {
        let mut basis = vec![
            vec![BigInt::from_i64(1), BigInt::from_i64(0)],
            vec![BigInt::from_i64(0)],
        ];
        lll_reduce(&mut basis);
    }

    #[test]
    #[should_panic(expected = "dependent")]
    fn lll_rejects_a_dependent_basis() {
        // Second row is twice the first: rank 1, not a basis of ℤ².
        let mut basis = rows(&[&[1, 2], &[2, 4]]);
        lll_reduce(&mut basis);
    }

    /// The identity form is the dot product, and a diagonal form is the
    /// reduction of the scaled basis: `lll_reduce_form` agrees with
    /// `lll_reduce` on both.
    #[test]
    fn the_form_reduction_agrees_with_the_dot_product_on_diagonal_forms() {
        let rows = |entries: &[[i64; 3]]| -> Vec<Vec<BigInt>> {
            entries
                .iter()
                .map(|row| row.iter().map(|&x| BigInt::from_i64(x)).collect())
                .collect()
        };
        let original = rows(&[[1, 1, 1], [-1, 0, 2], [3, 5, 6]]);
        let identity = rows(&[[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        let mut plain = original.clone();
        lll_reduce(&mut plain);
        let mut formed = original.clone();
        lll_reduce_form(&mut formed, &identity, 3, 4);
        assert_eq!(plain, formed);

        // Diagonal weights (1, 4, 9) = the columns scaled by (1, 2, 3).
        let diagonal = rows(&[[1, 0, 0], [0, 4, 0], [0, 0, 9]]);
        let mut formed = original.clone();
        lll_reduce_form(&mut formed, &diagonal, 3, 4);
        let mut scaled: Vec<Vec<BigInt>> = original
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(i, x)| x.mul(&BigInt::from_i64(i as i64 + 1)))
                    .collect()
            })
            .collect();
        lll_reduce(&mut scaled);
        let unscaled: Vec<Vec<BigInt>> = scaled
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(i, x)| x.div_rem(&BigInt::from_i64(i as i64 + 1)).0)
                    .collect()
            })
            .collect();
        assert_eq!(formed, unscaled);
    }

    /// Under a dense form the shortest vector is not the Euclidean one:
    /// the form `[[2, 1], [1, 2]]` (norm² = 2x² + 2xy + 2y²) makes
    /// `(1, −1)` shorter (norm² 2) than `(1, 0)` (norm² 2) — equal — and
    /// `(1, 1)` (norm² 6) longer than `(2, −1)` (norm² 6)... so take the
    /// basis `{(3, 0), (1, 1)}`, whose Euclidean reduction is
    /// `{(1, 1), (2, −1)}` and whose form reduction must lead with a vector
    /// of form-norm² 6 or less, and verify every returned vector's
    /// form-norm is what the form says, with the first the shortest.
    #[test]
    fn the_form_reduction_measures_by_the_form() {
        let rows = |entries: &[[i64; 2]]| -> Vec<Vec<BigInt>> {
            entries
                .iter()
                .map(|row| row.iter().map(|&x| BigInt::from_i64(x)).collect())
                .collect()
        };
        let form = rows(&[[2, 1], [1, 2]]);
        let norm = |v: &[BigInt]| -> BigInt {
            let (x, y) = (&v[0], &v[1]);
            let two = BigInt::from_i64(2);
            two.mul(&x.mul(x))
                .add(&two.mul(&x.mul(y)))
                .add(&two.mul(&y.mul(y)))
        };
        let mut basis = rows(&[[3, 0], [1, 1]]);
        lll_reduce_form(&mut basis, &form, 3, 4);
        // The lattice is index-3 in ℤ²; its shortest form-norm² is 6,
        // attained by (1, 1) and (2, −1) among others, and LLL at δ = 3/4
        // in dimension two returns a shortest vector first.
        assert_eq!(norm(&basis[0]), BigInt::from_i64(6));
        assert!(norm(&basis[1]) >= norm(&basis[0]));
        // Still a basis of the same lattice: determinant ±3.
        let det = basis[0][0]
            .mul(&basis[1][1])
            .sub(&basis[0][1].mul(&basis[1][0]));
        assert_eq!(det.magnitude(), &crate::BigUint::from_u64(3));
    }
}
