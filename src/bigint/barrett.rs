//! Barrett reduction: a fixed-modulus context for moduli of either parity.
//!
//! Barrett, *Implementing the Rivest Shamir and Adleman Public Key Encryption
//! Algorithm on a Standard Digital Signal Processor*, CRYPTO '86; the shape
//! here follows HAC Algorithm 14.42 with Note 14.44's bound of two on the
//! corrections.
//!
//! Separate from `montgomery` because the preconditions differ: Montgomery
//! requires an odd modulus, Barrett takes either parity, which is the reason
//! this context exists at all.
//!
//! The test module lives in the parent, which is why the threshold constant
//! and `last_corrections` are visible there.

use super::BigUint;

// Modulus width up to which Barrett's second multiplication is taken as a
// schoolbook half-product rather than a dispatched full product. The half
// costs `k²/2` limb products against the ladder's `O(k^{1.585})`, so it
// wins while `k` is small and loses once the better exponent tells:
// measured on M4, `reduce` is 1.44× ahead at 2 kbit (32 limbs) and 1.33× at
// 8 kbit (128 limbs), at parity near 32 kbit (512 limbs), and behind by
// 1.19× at 64 kbit and 1.32× at 128 kbit. The handoff sits at the measured
// parity point.
pub(super) const BARRETT_HALF_PRODUCT_MAX_LIMBS: usize = 512;

/// Barrett reduction context for a fixed modulus of either parity — the
/// complement to [`MontgomeryContext`](super::MontgomeryContext), which requires an odd
/// modulus.
///
/// Precomputes `μ = ⌊b^{2k} / n⌋` for `b = 2⁶⁴` and `k` the modulus's limb
/// count; each reduction then costs two multiplications of roughly the
/// modulus's width instead of a division (*Handbook of Applied
/// Cryptography*, Algorithm 14.42; Barrett, CRYPTO '86). The estimate
/// `q̂ = ⌊⌊x/b^{k−1}⌋·μ / b^{k+1}⌋` undershoots the true quotient by at
/// most two (HAC Note 14.44), so at most two corrective subtractions
/// follow.
///
/// The products here are computed in full where the algorithm needs only
/// a high and a low half. The second of the two products needs only its
/// low `k+1` limbs, and forms only those — the half-product of HAC Note
/// 14.45(ii), exact rather than approximate, since a partial product at or
/// above the window cannot influence a limb below it.
///
/// Measured on M4 against a plain division of the same operands, twelve
/// random modulus/dividend pairs per width, each timed nine times with the
/// order alternated, three runs. A figure is quoted only where every
/// sampled pair falls on the same side of parity; where the distribution
/// straddles 1.0 the width is named as parity and left without one, since
/// a headline number there is a report of which draw was taken:
///
/// ```text
///   512 bits   1.4×      12/12 pairs ahead
///  1024 bits   1.26×     12/12 pairs ahead
///  2048 bits   parity    per-pair medians 0.98–1.11, 1–2 of 12 behind
///  4096 bits   parity    per-pair medians 0.99–1.10, 0–4 of 12 behind
///  8192 bits   1.31×     12/12 pairs ahead
///   256 bits   parity    per-pair medians 0.96–1.32, a fifth behind
/// ```
///
/// Those intervals are the spread *observed*, not a bound on it. A sample
/// minimum and maximum over `n` runs is exceeded by run `n + 1` with
/// probability about `2/n`, so quoting one as though it were a bound is
/// the same error as quoting a median as though it were the answer, one
/// level down. What the table is for is the classification in its second
/// column; the intervals are there to show how close the parity rows are
/// to their neighbours, not to bound anything.
///
/// Before the half-product, `reduce` *trailed* a division by up to a third
/// at 2–4 kbit, so parity there is the gain. The series is not monotone —
/// the win is large at 512, erodes through 2–4 kbit, and returns at 8192 —
/// because the division it is measured against has its own crossovers.
///
/// This comment has been wrong three times in the same way: 0.96× at 256
/// bits, then 1.23× at 256 bits, then 1.03× and 1.01× at 2048 and 4096.
/// Each was a true reading of an under-sampled draw from a distribution
/// sitting on 1.0. The rule above is the fix; quoting a fourth number
/// would not be.
/// The half-product is itself quadratic, so it is taken only up to
/// `BARRETT_HALF_PRODUCT_MAX_LIMBS` (32 kbit, the measured parity point);
/// above that the dispatched full product's better exponent wins and the
/// window is taken from it. The first product's high half is always formed
/// in full; refining it (HAC Note 14.45(i)) trades exactness for a wider
/// correction bound and is not taken here. The context's other value is the capability: a
/// fixed-modulus context on *even* moduli, where Montgomery cannot
/// operate.
///
/// Like the rest of the crate, variable-time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BarrettContext {
    modulus: BigUint,
    mu: BigUint,
    limb_count: usize,
}

#[cfg(test)]
thread_local! {
    /// Corrections taken by the last `BarrettContext::reduce` on this thread.
    static CORRECTIONS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

impl BarrettContext {
    /// Build the context. The single division here computes
    /// `μ = ⌊b^{2k}/n⌋` for `b = 2⁶⁴` and `k` the modulus's limb count; every
    /// later [`Self::reduce`] spends two multiplications and at most two
    /// subtractions instead of a division, which is the whole point of the
    /// precomputation.
    ///
    /// `None` for a modulus below 2: zero has no residues, and rem one
    /// every residue is zero — neither needs a context.
    #[must_use]
    pub fn new(modulus: &BigUint) -> Option<Self> {
        if modulus.bits() < 2 {
            return None;
        }
        let limb_count = modulus.limbs().len();
        let mut numerator = BigUint::zero();
        numerator.set_bit(128 * limb_count);
        let (mu, _) = numerator.div_rem(modulus);
        Some(Self {
            modulus: modulus.clone(),
            mu,
            limb_count,
        })
    }

    /// The modulus this context reduces by.
    #[must_use]
    pub fn modulus(&self) -> &BigUint {
        &self.modulus
    }

    /// `x mod n` for `x < b^{2k}` — every product of two reduced values
    /// qualifies. Wider inputs fall back to the division this context
    /// exists to avoid, so callers keep their operands reduced.
    #[must_use]
    pub fn reduce(&self, x: &BigUint) -> BigUint {
        let k = self.limb_count;
        if x.bits() > 128 * k {
            return x.rem(&self.modulus);
        }
        if *x < self.modulus {
            return x.clone();
        }
        // q̂ = ⌊⌊x/b^(k−1)⌋·μ/b^(k+1)⌋ — the two shifts are limb-aligned.
        let mut q = x.clone();
        q.shr_bits(64 * (k - 1));
        q = q.mul(&self.mu);
        q.shr_bits(64 * (k + 1));
        // r = (x − q̂·n) mod b^(k+1); the difference of the two low windows,
        // lifted by b^(k+1) when it wraps.
        let window = 64 * (k + 1);
        let x_low = x.low_bits(window);
        // Only the low k+1 limbs of q̂·n survive the window, so only those
        // are formed (HAC Note 14.45(ii)); the half-product is exact.
        //
        // It is also schoolbook, and therefore quadratic, where the full
        // product would dispatch to Karatsuba and Toom. Half of `k²` beats
        // `k^{1.585}` only up to a point: measured, `reduce` gains 1.44× at
        // 2 kbit and 1.33× at 8 kbit, reaches parity near 32 kbit, and
        // *loses* 1.19× at 64 kbit and 1.32× at 128 kbit. Past
        // `BARRETT_HALF_PRODUCT_MAX_LIMBS` the full product's better
        // exponent wins and the window is taken from it instead.
        let qn_low = if k <= BARRETT_HALF_PRODUCT_MAX_LIMBS {
            BigUint::mul_low_ref(&q, &self.modulus, k + 1)
        } else {
            q.mul(&self.modulus).low_bits(window)
        };
        let mut r = if x_low >= qn_low {
            x_low.sub(&qn_low)
        } else {
            let mut lift = BigUint::zero();
            lift.set_bit(window);
            lift.add(&x_low).sub(&qn_low)
        };
        // The estimate is short by at most two.
        let mut corrections = 0u32;
        while r >= self.modulus {
            r = r.sub(&self.modulus);
            corrections += 1;
            debug_assert!(corrections <= 2, "HAC Note 14.44 bounds the corrections");
        }
        #[cfg(test)]
        CORRECTIONS.with(|cell| cell.set(corrections));
        r
    }

    /// The correction count of the last [`reduce`](Self::reduce) on this
    /// thread, for the test that shows HAC Note 14.44's bound of two is
    /// attained rather than merely respected.
    ///
    /// Two corrections happen about once in five hundred reductions and
    /// only on particular modulus shapes, so a test that does not look at
    /// the count cannot tell whether it ever reached the bound — and an
    /// earlier measurement, taken with a counter that tracked a running
    /// maximum rather than the per-call value, concluded wrongly that two
    /// was unreachable. The count is observable so that conclusion can be
    /// checked rather than assumed.
    #[cfg(test)]
    pub(crate) fn last_corrections() -> u32 {
        CORRECTIONS.with(std::cell::Cell::get)
    }

    /// `(a · b) mod n`: both operands reduced, then the double-width product
    /// reduced again. Three [`Self::reduce`] calls, of which the first two
    /// collapse to a comparison and a copy when the operands already lie in
    /// `[0, n)` — the case in an exponentiation loop, where every operand is
    /// a previous result.
    #[must_use]
    pub fn mod_mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        let a = self.reduce(a);
        let b = self.reduce(b);
        self.reduce(&a.mul(&b))
    }

    /// `a² mod n`. The square comes from [`BigUint::square`], whose
    /// specialized kernels form each cross term once between 8 and 256
    /// limbs, so this costs a squaring plus one Barrett reduction. The
    /// Montgomery domain's [`MontgomeryContext::square_mont`](super::MontgomeryContext::square_mont) goes further
    /// still, fusing the reduction into the kernel.
    #[must_use]
    pub fn mod_square(&self, a: &BigUint) -> BigUint {
        let a = self.reduce(a);
        self.reduce(&a.square())
    }

    /// `base^exponent mod n` by left-to-right binary exponentiation (Knuth,
    /// *TAOCP* vol. 2, §4.6.3) with one [`Self::reduce`] after each step —
    /// the exponentiation route for even moduli, where Montgomery cannot
    /// operate. The accumulator is seeded from the exponent's top set bit
    /// (as the Montgomery ladder in this file does), so the cost is one
    /// squaring per remaining exponent bit and one multiplication per
    /// remaining set bit; there is no window table here, unlike
    /// [`MontgomeryContext::pow`](super::MontgomeryContext::pow). `0^0 = 1` by the usual convention.
    ///
    /// Variable-time, like the rest of the crate: a clear exponent bit skips
    /// its multiplication.
    #[must_use]
    pub fn mod_pow(&self, base: &BigUint, exponent: &BigUint) -> BigUint {
        if exponent.is_zero() {
            return self.reduce(&BigUint::one());
        }
        let base = self.reduce(base);
        // The top bit of a non-zero exponent is set by definition, so the
        // ladder starts at `base` and scans the bits below it. Seeding from
        // 1 would spend a squaring, a multiplication and two reductions
        // arriving at the same state.
        let mut result = base.clone();
        for bit in (0..exponent.bits() - 1).rev() {
            result = self.reduce(&result.square());
            if exponent.bit(bit) {
                result = self.reduce(&result.mul(&base));
            }
        }
        result
    }
}
