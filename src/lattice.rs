//! Lattice reduction over the integers.
//!
//! [`lll_reduce`] applies the Lenstra–Lenstra–Lovász algorithm (A. K. Lenstra,
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
    assert!(!d[1].is_zero(), "linearly dependent basis (zero vector)");
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
                    let num = d[i].mul_ref(&u).sub_ref(&lam[k][i].mul_ref(&lam[j][i]));
                    u = num.div_exact(&d[i - 1]);
                }
                if j < k {
                    lam[k][j] = u;
                } else {
                    // u is now the Gram determinant of b_1..b_k, which
                    // vanishes exactly when those rows are dependent. The
                    // divisions above were by d_1..d_{k-1}, already checked.
                    assert!(!u.is_zero(), "linearly dependent basis");
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
            let lhs = q.mul_ref(&d[k].mul_ref(&d[k - 2]));
            let lam_sq = lam[k][k - 1].mul_ref(&lam[k][k - 1]);
            let rhs = p
                .mul_ref(&d[k - 1].mul_ref(&d[k - 1]))
                .sub_ref(&q.mul_ref(&lam_sq));

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
        acc = acc.add_ref(&a.mul_ref(b));
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
    let two_a_plus_b = a.add_ref(a).add_ref(b);
    let two_b = b.add_ref(b);
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
        BigInt::from_biguint(quotient.add_ref(&BigUint::from_u64(1))).negated()
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
    let two_lam = lam[k][l].add_ref(&lam[k][l]);
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
        bk[c] = bk[c].sub_ref(&qnt.mul_ref(&bl[c]));
    }

    // λ_{k,l} ← λ_{k,l} − q·d_l; λ_{k,i} ← λ_{k,i} − q·λ_{l,i} for i < l.
    let (lleft, lright) = lam.split_at_mut(k);
    let lam_l = &lleft[l];
    let lam_k = &mut lright[0];
    lam_k[l] = lam_k[l].sub_ref(&qnt.mul_ref(&d[l]));
    for i in 1..l {
        lam_k[i] = lam_k[i].sub_ref(&qnt.mul_ref(&lam_l[i]));
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
        .mul_ref(&d[k])
        .add_ref(&lambda.mul_ref(&lambda))
        .div_exact(&d[k - 1]);

    for row in lam.iter_mut().take(k_max + 1).skip(k + 1) {
        let t = row[k].clone();
        // λ_{i,k} ← (d_k·λ_{i,k-1} − λ·t) / d_{k-1}.
        row[k] = d[k]
            .mul_ref(&row[k - 1])
            .sub_ref(&lambda.mul_ref(&t))
            .div_exact(&d[k - 1]);
        // λ_{i,k-1} ← (B·t + λ·λ_{i,k}) / d_k, using the updated λ_{i,k}.
        row[k - 1] = b_new
            .mul_ref(&t)
            .add_ref(&lambda.mul_ref(&row[k]))
            .div_exact(&d[k]);
    }
    d[k - 1] = b_new;
}

// ─── Two-dimensional reduction under a diagonal form ───────────────────────

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
/// as it is for [`lll_reduce`] in general dimension, but solves the shortest
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
/// # Panics
///
/// Panics if the two vectors are linearly dependent (a zero determinant is
/// not a basis), if either weight is not positive, if the basis determinant
/// overflows `i128`, or if the weighted arithmetic does.
///
/// That last bound is a real restriction and tighter than it first looks. The
/// rounding step forms `2·⟨u,v⟩ + ‖u‖²` over `2‖u‖²`, so it is *twice* the
/// norm that must be representable, not the norm: the working condition is
/// `(w₀·x)² + (w₁·y)² < 2¹²⁶` for every vector the reduction visits, which
/// holds comfortably when each of `|w₀·x|` and `|w₁·y|` stays below `2⁶²`.
/// A basis whose norms fill `i128` to the top is refused rather than
/// silently wrapped — a wrapped norm compares wrongly and would return an
/// unreduced basis with no indication.
#[must_use]
pub fn gauss_reduce_weighted(basis: [[i128; 2]; 2], weights: [i128; 2]) -> [[i128; 2]; 2] {
    assert!(
        weights[0] > 0 && weights[1] > 0,
        "a diagonal form needs positive weights"
    );
    let determinant = basis[0][0]
        .checked_mul(basis[1][1])
        .and_then(|a| {
            basis[0][1]
                .checked_mul(basis[1][0])
                .and_then(|b| a.checked_sub(b))
        })
        .expect("the basis determinant overflowed i128");
    assert!(
        determinant != 0,
        "a two-dimensional basis needs independent vectors"
    );

    let overflow = "the weighted arithmetic overflowed i128";
    let mut u = basis[0];
    let mut v = basis[1];
    let mut norm_u = weighted_norm_sq(u, weights).expect(overflow);
    if norm_u > weighted_norm_sq(v, weights).expect(overflow) {
        core::mem::swap(&mut u, &mut v);
        norm_u = weighted_norm_sq(u, weights).expect(overflow);
    }

    loop {
        // `norm_u > 0` throughout: the determinant is non-zero, so neither
        // vector is zero, and the weights are positive.
        let dot = weighted_dot(u, v, weights).expect(overflow);
        // Distinguished from the norm overflow above: this is the rounding
        // step, which needs twice the norm to be representable.
        let q = round_div(dot, norm_u)
            .expect("the weighted arithmetic overflowed i128 while rounding; norms must fit 2^126");
        let r = [
            v[0].checked_sub(q.checked_mul(u[0]).expect(overflow))
                .expect(overflow),
            v[1].checked_sub(q.checked_mul(u[1]).expect(overflow))
                .expect(overflow),
        ];
        let norm_r = weighted_norm_sq(r, weights).expect(overflow);
        if norm_r >= norm_u {
            // `u` is a shortest vector; `r` is reduced against it.
            return [u, r];
        }
        v = u;
        u = r;
        norm_u = norm_r;
    }
}

#[cfg(test)]
mod tests {
    use super::{gauss_reduce_weighted, lll_reduce, lll_reduce_delta, weighted_norm_sq};
    use crate::bigint::{BigInt, Sign};

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
        for weights in [[1i128, 1], [1, 2], [1, 7], [3, 5], [1, 1000], [64, 1]] {
            for _ in 0..200 {
                let basis = [
                    [next() % 4096, next() % 4096],
                    [next() % 4096, next() % 4096],
                ];
                if det(basis) == 0 {
                    continue;
                }
                let reduced = gauss_reduce_weighted(basis, weights);

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
        let (p, q) = (2_113_745_839i128, 10_000_000i128); // s ≈ 211.3745839
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
            let reduced = gauss_reduce_weighted(basis, [q, p]);
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
        let reduced = gauss_reduce_weighted([[a, a], [a, -a]], [1, 1]);
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
        assert_eq!(gauss_reduce_weighted(basis, [1, 1]), [[1, 0], [0, 1]]);
        // Under a heavy y-weight the x-axis vector is the shorter one.
        assert_eq!(gauss_reduce_weighted(basis, [1, 100]), [[1, 0], [0, 1]]);
        // And under a heavy x-weight the order flips.
        assert_eq!(gauss_reduce_weighted(basis, [100, 1]), [[0, 1], [1, 0]]);
    }

    #[test]
    #[should_panic(expected = "independent vectors")]
    fn gauss_reduce_refuses_a_dependent_pair() {
        let _ = gauss_reduce_weighted([[2, 4], [1, 2]], [1, 1]);
    }

    #[test]
    #[should_panic(expected = "positive weights")]
    fn gauss_reduce_refuses_a_zero_weight() {
        let _ = gauss_reduce_weighted([[1, 0], [0, 1]], [1, 0]);
    }

    #[test]
    #[should_panic(expected = "overflowed")]
    fn gauss_reduce_refuses_an_unrepresentable_norm() {
        // The weighted coordinate squares, so this is past the bound the
        // documentation names rather than an arbitrary large value.
        let big = 1i128 << 100;
        let _ = gauss_reduce_weighted([[big, 0], [0, 1]], [1, 1]);
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
            Self::reduced(
                self.n.mul_ref(&o.d).add_ref(&o.n.mul_ref(&self.d)),
                self.d.mul_ref(&o.d),
            )
        }
        fn sub(&self, o: &Self) -> Self {
            Self::reduced(
                self.n.mul_ref(&o.d).sub_ref(&o.n.mul_ref(&self.d)),
                self.d.mul_ref(&o.d),
            )
        }
        fn mul(&self, o: &Self) -> Self {
            Self::reduced(self.n.mul_ref(&o.n), self.d.mul_ref(&o.d))
        }
        fn div(&self, o: &Self) -> Self {
            assert!(!o.n.is_zero(), "division by zero fraction");
            Self::reduced(self.n.mul_ref(&o.d), self.d.mul_ref(&o.n))
        }
        // self ≥ o, both denominators positive.
        fn ge(&self, o: &Self) -> bool {
            self.n.mul_ref(&o.d) >= o.n.mul_ref(&self.d)
        }
        // |self| ≤ 1/2  ⟺  2|n| ≤ d.
        fn abs_le_half(&self) -> bool {
            let two_n = self.n.add_ref(&self.n);
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
                            s = s.add_ref(&a.mul_ref(b));
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
                    let num = g[kk][kk]
                        .mul_ref(&g[r][c])
                        .sub_ref(&g[r][kk].mul_ref(&g[kk][c]));
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
}
