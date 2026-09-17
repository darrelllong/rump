//! Barrett reduction: a fixed-modulus context for moduli of either parity.
//!
//! Barrett, *Implementing the Rivest Shamir and Adleman Public Key Encryption
//! Algorithm on a Standard Digital Signal Processor*, CRYPTO '86; the shape
//! here follows HAC Algorithm 14.42 with Note 14.44's bound of two on the
//! corrections.
//!
//! Montgomery requires an odd modulus; Barrett takes either parity, which is
//! why this context exists.
//!
//! The tests live in the parent module, so the threshold constant and
//! `last_corrections` are visible there.

use super::{bit_span, BigUint, ModulusError};

// Modulus width up to which Barrett's second multiplication is a schoolbook
// half-product rather than a dispatched full product. The half costs `k²/2`
// limb products against the ladder's `O(k^{1.585})`: measured on M4,
// `reduce` is 1.44× ahead at 2 kbit and 1.33× at 8 kbit, at parity near
// 32 kbit (512 limbs), and 1.19× behind at 64 kbit.
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
/// The second product needs only its low `k+1` limbs and forms only those:
/// the half-product of HAC Note 14.45(ii), exact because a partial product
/// at or above the window cannot influence a limb below it. The half-product
/// is quadratic, so it is used only up to `BARRETT_HALF_PRODUCT_MAX_LIMBS`
/// (32 kbit); above that the window is taken from the dispatched full
/// product. The first product's high half is always formed in full; refining
/// it (HAC Note 14.45(i)) would widen the correction bound.
///
/// Speed against a plain division of the same operands, measured on M4 over
/// twelve random modulus/dividend pairs per width. A ratio is given only
/// where every pair falls on the same side of parity:
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
/// The intervals are the observed spread, not a bound. The series is not
/// monotone because the division it is measured against has its own
/// crossovers. The ratios are the M4's: on an AMD EPYC 7452 a prepared
/// context's [`Self::mod_mul`] on reduced operands trails the one-shot
/// [`BigUint::mod_mul`] by 23% at 256 bits and 5% at 2048
/// (`bench/modes_dennard.md`).
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
    /// subtractions instead of a division.
    ///
    /// # Errors
    ///
    /// [`ModulusError::Zero`] or [`ModulusError::One`] for a modulus below 2:
    /// zero has no residues, and modulo one every residue is zero.
    pub fn new(modulus: &BigUint) -> Result<Self, ModulusError> {
        if modulus.is_zero() {
            return Err(ModulusError::Zero);
        }
        if modulus.is_one() {
            return Err(ModulusError::One);
        }
        let limb_count = modulus.limbs().len();
        let mut numerator = BigUint::zero();
        numerator.set_bit(bit_span(limb_count, 128));
        let (mu, _) = numerator.div_rem(modulus);
        Ok(Self {
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
        if x.bits() > bit_span(k, 128) {
            return x.rem(&self.modulus);
        }
        if *x < self.modulus {
            return x.clone();
        }
        // q̂ = ⌊⌊x/b^(k−1)⌋·μ/b^(k+1)⌋ — the two shifts are limb-aligned.
        let mut q = x.clone();
        q.shr_bits(bit_span(k - 1, 64));
        q = q.mul(&self.mu);
        q.shr_bits(bit_span(k + 1, 64));
        // r = (x − q̂·n) mod b^(k+1); the difference of the two low windows,
        // lifted by b^(k+1) when it wraps.
        let window = bit_span(k + 1, 64);
        let x_low = x.low_bits(window);
        // Only the low k+1 limbs of q̂·n survive the window, so only those
        // are formed (HAC Note 14.45(ii)); the half-product is exact.
        //
        // It is schoolbook, so past `BARRETT_HALF_PRODUCT_MAX_LIMBS` the
        // dispatched full product is faster and the window is taken from it.
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
    /// the count cannot tell whether the bound is ever reached.
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

    /// `a² mod n`: one [`BigUint::square`] (whose specialized kernels form
    /// each cross term once from 8 to 447 limbs) plus one Barrett reduction.
    /// [`MontgomeryContext::square_residue`](crate::modular::MontgomeryContext::square_residue)
    /// goes further, fusing the reduction into the kernel.
    #[must_use]
    pub fn mod_square(&self, a: &BigUint) -> BigUint {
        let a = self.reduce(a);
        self.reduce(&a.square())
    }

    /// `base^exponent mod n` by left-to-right binary exponentiation (Knuth,
    /// *TAOCP* vol. 2, §4.6.3) with one [`Self::reduce`] after each step —
    /// the exponentiation route for even moduli, where Montgomery cannot
    /// operate. The accumulator is seeded from the exponent's top set bit,
    /// so the cost is one
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
        // The top bit of a non-zero exponent is set, so the ladder starts at
        // `base` and scans the bits below it.
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
