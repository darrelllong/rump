//! Division by Newton's reciprocal, for operands wide enough that Knuth's
//! Algorithm D — quadratic in the divisor's width — is the wrong tool.
//!
//! The quotient of `n` by `d` is `⌊n·(1/d)⌋`, and `1/d` to `k` limbs of
//! precision is a fixed point of `x ↦ x·(2 − d·x)`, Newton's iteration for
//! the reciprocal, which doubles its correct limbs at every step and so
//! costs a constant number of multiplications at the target width plus the
//! same again at half the width, and so on: `O(M(k))` in all, against the
//! `O(k²)` of the schoolbook division. The quotient is then two
//! multiplications and a short correction, exactly as Barrett reduction
//! spends its precomputed `μ` — this *is* Barrett's `μ = ⌊b^{2k}/d⌋`,
//! computed by Newton instead of by long division.
//!
//! Brent & Zimmermann, *Modern Computer Arithmetic*, Cambridge, 2010,
//! §4.2.2 (Newton's reciprocal) and §2.4 (Barrett's division); the
//! reciprocal's recursive half-precision start is their Algorithm
//! `ApproximateReciprocal` in spirit, with every truncation replaced by a
//! full product and an exact correction at the end, which costs a constant
//! factor and buys a proof: the returned `μ` is exact whatever the
//! iteration's rounding did, because the last thing done to it is to check
//! `d·μ ≤ b^{2k} < d·(μ + 1)` and adjust until it holds.
//!
//! What this is not: the divide-and-conquer division of Burnikel and
//! Ziegler (1998), which is `O(M(k) log k)` and has the better constant at
//! moderate widths. The reciprocal wins where the products are already
//! NTT-fast, which is the regime the threshold below selects; a
//! Burnikel–Ziegler middle band is a measurement this crate has not made.
//!
//! Found wanting by measurement: a 70-digit number field sieve's algebraic
//! square root, lifting a root modulo `q^k` with `q^k` near fifteen million
//! bits, spent 558 of its 737 seconds in coefficient reductions that each
//! ran Algorithm D over 234 000-limb operands.

use super::{bit_span, BigUint};

/// Divisor limb count from which [`BigUint::div_rem`] takes this route.
///
/// Measured on M4 (`newton_division_crossover_timing`, ignored test), a
/// `2k`-limb dividend by `k` limbs:
///
/// | limbs | Algorithm D | Newton |
/// |---|---:|---:|
/// | 1 024 | 1.7 ms | 2.3 ms |
/// | 2 048 | 7.3 ms | 7.6 ms |
/// | 4 096 | 29 ms | 20 ms |
/// | 16 384 | 429 ms | 116 ms |
/// | 65 536 | 6.53 s | 0.84 s |
///
/// Parity near 2 048; the threshold sits a little above it, where the
/// products are already Toom-4's and the reciprocal's constant is paid
/// back. Below it the quadratic division's small constant wins.
pub(crate) const NEWTON_DIVISION_THRESHOLD_LIMBS: usize = 3072;

/// Widths at or below which the reciprocal is taken by long division.
///
/// The recursion halves the width until this floor, where a single
/// Algorithm D division of a `2k`-limb power of the base by `d` is cheap.
const RECIPROCAL_BASE_LIMBS: usize = 32;

/// `⌊b^{2k}/d⌋` for a normalized `d` of `k` limbs (top bit set), exact.
///
/// Newton from a half-width reciprocal: with `h = ⌈k/2⌉` and `d_h` the top
/// `h` limbs of `d`, `x₀ = ⌊b^{2h}/d_h⌋·b^{k−h}` is within about `5·b^{k−h}`
/// of the answer (a relative error of `5·b^{−h}`), one step of
/// `x ← x + x·(b^{2k} − d·x)/b^{2k}` squares that relative error to under
/// `b^{−k}`, which is a few units absolute, and the correction loop takes
/// the last few units exactly. Three full products per level.
pub(super) fn reciprocal(d: &BigUint) -> BigUint {
    let k = d.limbs().len();
    debug_assert!(
        k > 0 && d.limbs()[k - 1] >> 63 == 1,
        "the reciprocal wants a normalized divisor"
    );
    let two_k = power_of_base(2 * k);
    if k <= RECIPROCAL_BASE_LIMBS {
        // Algorithm D directly; `div_rem` would route a one-limb divisor
        // to the Horner path, which is also fine, so go through it.
        return two_k.div_rem(d).0;
    }
    let h = k.div_ceil(2);
    let low = k - h;
    let mut d_h = d.clone();
    d_h.shr_bits(bit_span(low, 64));
    let mut x = reciprocal(&d_h);
    x.shl_bits(bit_span(low, 64));

    // One Newton step at full width: x ← x + ⌊x·(b^{2k} − d·x) / b^{2k}⌋,
    // with the sign of the residual handled by which side is larger.
    let product = d.mul(&x);
    if product <= two_k {
        let residual = two_k.sub(&product);
        let mut step = x.mul(&residual);
        step.shr_bits(bit_span(2 * k, 64));
        x = x.add(&step);
    } else {
        let residual = product.sub(&two_k);
        let mut step = x.mul(&residual);
        step.shr_bits(bit_span(2 * k, 64));
        x = x.sub(&step).sub(&BigUint::one());
    }

    // Exact: d·x ≤ b^{2k} < d·(x + 1), by adjusting until it holds. The
    // Newton step leaves a handful of units at most; the loop is bounded
    // in every build so a regression in the analysis above shows up as a
    // failure and not as a slow division.
    let mut product = d.mul(&x);
    let mut corrections = 0u32;
    while product > two_k {
        x = x.sub(&BigUint::one());
        product = product.sub(d);
        corrections += 1;
        assert!(corrections < 256, "the Newton reciprocal did not converge");
    }
    let mut next = product.add(d);
    while next <= two_k {
        x = x.add(&BigUint::one());
        product = next;
        next = product.add(d);
        corrections += 1;
        assert!(corrections < 256, "the Newton reciprocal did not converge");
    }
    x
}

/// `b^limbs`, as a value.
fn power_of_base(limbs: usize) -> BigUint {
    let mut value = BigUint::zero();
    value.set_bit(bit_span(limbs, 64));
    value
}

/// Euclidean division through the reciprocal: `(quotient, remainder)` with
/// the remainder in `[0, d)`.
///
/// The divisor is normalized by a bit shift (which leaves the quotient
/// alone and shifts the remainder), its reciprocal `μ` is taken once, and
/// the dividend is consumed a block of `k` limbs at a time from the top:
/// each block's `2k`-limb window `x = r·b^k + block` has `x < d·b^k`, so
/// Barrett's estimate `⌊⌊x/b^{k−1}⌋·μ/b^{k+1}⌋` is short of the true
/// quotient digit by at most two (HAC Note 14.44), and two subtractions
/// finish it. Two products per block.
///
/// Requires a non-zero divisor and a dividend no smaller than it; the
/// caller ([`BigUint::div_rem`]) settles the trivial cases.
pub(super) fn div_rem(dividend: &BigUint, divisor: &BigUint) -> (BigUint, BigUint) {
    let shift = divisor.limbs()[divisor.limbs().len() - 1].leading_zeros() as usize;
    let mut d = divisor.clone();
    d.shl_bits(shift);
    let mut n = dividend.clone();
    n.shl_bits(shift);
    let k = d.limbs().len();
    let mu = reciprocal(&d);
    let n_limbs = n.limbs();
    let m = n_limbs.len();

    // The top k limbs, reduced once by hand: they are below b^k ≤ 2d, so at
    // most one subtraction.
    let mut r = BigUint::from_limbs(n_limbs[m - k..].to_vec());
    let mut quotient_limbs: Vec<u64> = Vec::with_capacity(m - k + 1);
    let mut top_digit = 0u64;
    if r >= d {
        r = r.sub(&d);
        top_digit = 1;
    }
    // Blocks of up to k limbs, high to low; the quotient digits come out
    // high to low and are collected in reverse.
    let mut blocks: Vec<Vec<u64>> = Vec::new();
    let mut end = m - k;
    while end > 0 {
        let width = end.min(k);
        let start = end - width;
        // x = r·b^width + block.
        let mut x_limbs = n_limbs[start..end].to_vec();
        x_limbs.extend_from_slice(r.limbs());
        let x = BigUint::from_limbs(x_limbs);
        let (q, rem) = barrett_digit(&x, &d, &mu, k);
        let mut q_limbs = q.limbs().to_vec();
        q_limbs.resize(width, 0);
        blocks.push(q_limbs);
        r = rem;
        end = start;
    }
    for block in blocks.into_iter().rev() {
        quotient_limbs.extend_from_slice(&block);
    }
    quotient_limbs.push(top_digit);
    let quotient = BigUint::from_limbs(quotient_limbs);
    r.shr_bits(shift);
    (quotient, r)
}

/// One Barrett step: `x < d·b^k` with `d` normalized of `k` limbs and
/// `μ = ⌊b^{2k}/d⌋`, returning `(⌊x/d⌋, x mod d)`.
fn barrett_digit(x: &BigUint, d: &BigUint, mu: &BigUint, k: usize) -> (BigUint, BigUint) {
    let mut q = x.clone();
    q.shr_bits(bit_span(k - 1, 64));
    q = q.mul(mu);
    q.shr_bits(bit_span(k + 1, 64));
    let mut r = x.sub(&q.mul(d));
    let mut corrections = 0u32;
    while r >= *d {
        r = r.sub(d);
        q = q.add(&BigUint::one());
        corrections += 1;
        debug_assert!(corrections <= 2, "HAC Note 14.44 bounds the corrections");
    }
    (q, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed-seed splitmix64 stream for reproducible operand shapes.
    struct Stream(u64);
    impl Stream {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn limbs(&mut self, count: usize, top_bit: bool) -> BigUint {
            let mut limbs: Vec<u64> = (0..count).map(|_| self.next()).collect();
            if top_bit {
                limbs[count - 1] |= 1 << 63;
            } else {
                limbs[count - 1] |= 1; // non-zero top limb, any normalization
            }
            BigUint::from_limbs(limbs)
        }
    }

    fn check(dividend: &BigUint, divisor: &BigUint) {
        let (q, r) = div_rem(dividend, divisor);
        let (q_knuth, r_knuth) = if divisor.limbs().len() == 1 {
            dividend.div_rem(divisor)
        } else {
            BigUint::div_rem_knuth(dividend.limbs(), divisor.limbs())
        };
        assert_eq!(q, q_knuth, "quotient disagrees with Algorithm D");
        assert_eq!(r, r_knuth, "remainder disagrees with Algorithm D");
        assert!(r < *divisor);
        assert_eq!(q.mul(divisor).add(&r), *dividend);
    }

    #[test]
    fn the_reciprocal_is_exact_at_every_width() {
        let mut stream = Stream(1);
        for k in [1usize, 2, 3, 32, 33, 64, 65, 100, 255, 256, 257, 1000] {
            let d = stream.limbs(k, true);
            let mu = reciprocal(&d);
            let two_k = power_of_base(2 * k);
            assert!(d.mul(&mu) <= two_k, "μ too large at {k} limbs");
            assert!(
                d.mul(&mu.add(&BigUint::one())) > two_k,
                "μ too small at {k} limbs"
            );
        }
        // The extremes of the normalized range: b^k/2 exactly, and b^k − 1.
        for k in [40usize, 100] {
            let mut half = BigUint::zero();
            half.set_bit(bit_span(k, 64) - 1);
            let mu = reciprocal(&half);
            let mut expected = BigUint::zero();
            expected.set_bit(bit_span(k, 64) + 1);
            assert_eq!(mu, expected, "1/(b^k/2) = 2·b^k");
            let ones = BigUint::from_limbs(vec![u64::MAX; k]);
            let mu = reciprocal(&ones);
            let two_k = power_of_base(2 * k);
            assert!(ones.mul(&mu) <= two_k && ones.mul(&mu.add(&BigUint::one())) > two_k);
        }
    }

    #[test]
    fn division_agrees_with_algorithm_d_over_shapes() {
        let mut stream = Stream(2);
        for &(n_limbs, d_limbs) in &[
            (2usize, 2usize),
            (3, 2),
            (40, 40),
            (41, 40),
            (80, 40),
            (81, 40),
            (200, 40),
            (203, 40),
            (300, 100),
            (1000, 300),
            (1024, 512),
            (2047, 1024),
        ] {
            for top_bit in [true, false] {
                let d = stream.limbs(d_limbs, top_bit);
                let n = stream.limbs(n_limbs, top_bit);
                if n < d {
                    continue;
                }
                check(&n, &d);
            }
        }
    }

    #[test]
    fn division_handles_the_edges() {
        let mut stream = Stream(3);
        let d = stream.limbs(64, true);
        // An exact multiple, a multiple less one, a multiple plus one.
        let q = stream.limbs(70, false);
        let exact = q.mul(&d);
        check(&exact, &d);
        check(&exact.sub(&BigUint::one()), &d);
        check(&exact.add(&BigUint::one()), &d);
        // Dividend equal to the divisor, and to the divisor squared.
        check(&d, &d);
        check(&d.mul(&d), &d);
        // A divisor of all ones and a dividend of all ones, wider.
        let ones_d = BigUint::from_limbs(vec![u64::MAX; 50]);
        let ones_n = BigUint::from_limbs(vec![u64::MAX; 175]);
        check(&ones_n, &ones_d);
        // A power of two divisor, unnormalized (top limb = 1).
        let mut pow = BigUint::zero();
        pow.set_bit(64 * 60);
        check(&ones_n, &pow);
    }

    #[test]
    fn the_public_division_takes_this_route_above_the_threshold() {
        // Through `BigUint::div_rem`, on operands past the threshold, against
        // the invariant q·d + r = n: whichever route the dispatch takes must
        // hold it, and the sizes here are the ones the dispatch sends here.
        let mut stream = Stream(4);
        let d = stream.limbs(NEWTON_DIVISION_THRESHOLD_LIMBS + 5, false);
        let n = stream.limbs(3 * NEWTON_DIVISION_THRESHOLD_LIMBS, false);
        let (q, r) = n.div_rem(&d);
        assert!(r < d);
        assert_eq!(q.mul(&d).add(&r), n);
        assert_eq!(n.rem(&d), r);
    }

    #[test]
    #[ignore = "timing probe for the division crossover; run with --ignored --nocapture"]
    fn newton_division_crossover_timing() {
        let mut stream = Stream(5);
        for k in [256usize, 512, 768, 1024, 2048, 4096, 16_384, 65_536] {
            let d = stream.limbs(k, false);
            let n = stream.limbs(2 * k, false);
            let started = std::time::Instant::now();
            let knuth = BigUint::div_rem_knuth(n.limbs(), d.limbs());
            let knuth_time = started.elapsed();
            let started = std::time::Instant::now();
            let newton = div_rem(&n, &d);
            let newton_time = started.elapsed();
            assert_eq!(knuth, newton);
            eprintln!(
                "{k:6} limbs: knuth {:?}  newton {:?}",
                knuth_time, newton_time
            );
        }
    }
}
