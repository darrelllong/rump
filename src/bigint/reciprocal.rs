//! Division by a divisor that does not change, via a precomputed reciprocal.
//!
//! Möller & Granlund, *Improved Division by Invariant Integers*, IEEE
//! Transactions on Computers 60 (2011), Algorithm 4 (`div2by1`) over the
//! reciprocal of Algorithm 2; the approach descends from Granlund &
//! Montgomery, *Division by Invariant Integers using Multiplication*, PLDI
//! 1994. The precomputed reciprocal replaces each division with a
//! multiplication, a correction, and a rare second correction.
//!
//! **This is a multi-limb win and a word-sized loss**, at least on the
//! hardware it has been measured on. Divisor table of the 2 262 primes below
//! 20 000, M4, `--release`, minimum of nine rounds, against the hardware-divide
//! path it replaces:
//!
//! ```text
//!   width          hardware   WordReciprocal
//!   one word         0.84 ns     2.07 ns    0.41x  — slower
//!   two limbs        1.37x faster
//!   four limbs       1.52x faster
//!   sixteen limbs    2.38x faster
//! ```
//!
//! Apple's divider retires a 64-bit division in a few cycles, so a word-sized
//! remainder has nothing to gain and pays for the setup and the correction
//! branches. The often-quoted "20–40 cycles" for a hardware divide is an x86
//! figure and does not describe this machine; it was in this comment,
//! unmeasured, until the numbers above were taken. Reach for this type when
//! the dividend is a `BigUint` of two limbs or more, not for reducing sieve
//! positions that already fit a word.
//!
//! One reciprocal serves every entry point here. Dividing a multi-limb value
//! by a word is Horner's recurrence in base `2⁶⁴`, and each step of it is a
//! two-word-by-one-word division — exactly what Algorithm 4 computes — so the
//! word and multi-limb paths are the same kernel rather than two.

use super::BigUint;
use core::num::NonZeroU64;

/// A `u64` divisor with its reciprocal precomputed, for division repeated
/// enough times that the setup is free.
///
/// Build one per divisor and keep it. Construction costs a single hardware
/// division; every use afterwards costs a multiplication and a correction.
///
/// Worth it for `BigUint` dividends of two limbs or more, where the division
/// this replaces is paid *per limb*. Not worth it for word-sized dividends:
/// see the module documentation for measurements, which show the hardware
/// divide ahead there.
///
/// # Examples
///
/// ```
/// use rump::{BigUint, WordReciprocal};
///
/// let r = WordReciprocal::new(1_000_003);
/// assert_eq!(r.rem_u64(2_000_006), 0);
/// assert_eq!(r.div_rem_u64(2_000_007), (2, 1));
///
/// // Sieve positions are signed and want a non-negative residue.
/// assert_eq!(r.rem_euclid_i64(-1), 1_000_002);
///
/// let n = BigUint::from_u64(12_345_678_901_234_567_890);
/// assert_eq!(n.rem_reciprocal(&r), n.rem_u64(1_000_003));
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WordReciprocal {
    divisor: u64,
    /// `divisor << shift`, so the top bit is set — Algorithm 4 requires it.
    normalized: u64,
    /// `⌊(2¹²⁸ − 1) / normalized⌋ − 2⁶⁴`, which fits a word exactly because
    /// `normalized ≥ 2⁶³` puts the quotient in `[2⁶⁴, 2⁶⁵)`.
    reciprocal: u64,
    /// `divisor.leading_zeros()`, at most 63 since the divisor is non-zero.
    shift: u32,
}

impl WordReciprocal {
    /// Precompute the reciprocal of `divisor`.
    ///
    /// Total: non-zero is the entire precondition, so it is carried by
    /// [`NonZeroU64`] rather than reported as a panic or as an unexplained
    /// `None`. There is no failure left for the return type to describe.
    #[must_use]
    pub fn new(divisor: NonZeroU64) -> Self {
        let divisor = divisor.get();
        let shift = divisor.leading_zeros();
        let normalized = divisor << shift;
        // `u128::MAX` is `2¹²⁸ − 1`; forming `2¹²⁸` directly would overflow.
        let reciprocal = (u128::MAX / u128::from(normalized) - (1u128 << 64)) as u64;
        Self {
            divisor,
            normalized,
            reciprocal,
            shift,
        }
    }

    /// The divisor this reciprocal was built for.
    #[must_use]
    pub fn divisor(&self) -> u64 {
        self.divisor
    }

    /// `(u1·2⁶⁴ + u0) / normalized`, as (quotient, remainder).
    ///
    /// Möller–Granlund Algorithm 4. The additions and the product that forms
    /// `r` are deliberately wrapping: the algorithm works rem `2⁶⁴` and
    /// corrects afterwards, which is the whole trick.
    ///
    /// `u1 < self.normalized` is the precondition. It is not checkable at the
    /// public boundary because no public entry point takes `u1` — the callers
    /// below either pass a normalized high word or carry a running remainder
    /// that is smaller than the divisor by induction — so it is a
    /// `debug_assert` rather than a release check.
    #[inline]
    fn div2by1(&self, u1: u64, u0: u64) -> (u64, u64) {
        debug_assert!(
            u1 < self.normalized,
            "Algorithm 4 needs a reduced high word"
        );
        let d = self.normalized;
        let product = u128::from(self.reciprocal) * u128::from(u1);
        let sum = product.wrapping_add((u128::from(u1) << 64) | u128::from(u0));
        let q0 = sum as u64;
        let mut q1 = (sum >> 64) as u64;

        q1 = q1.wrapping_add(1);
        let mut r = u0.wrapping_sub(q1.wrapping_mul(d));

        // The estimate is short by at most one here, and the branch is taken
        // about as often as not.
        if r > q0 {
            q1 = q1.wrapping_sub(1);
            r = r.wrapping_add(d);
        }
        // Rare: the paper bounds the remaining error at one.
        if r >= d {
            q1 = q1.wrapping_add(1);
            r -= d;
        }
        (q1, r)
    }

    /// The `i`-th limb of `limbs << shift`, where index `limbs.len()` is the
    /// word shifted off the top. Normalizing the dividend costs nothing extra
    /// this way: the shifted stream is produced a word at a time rather than
    /// materialized.
    #[inline]
    fn normalized_limb(&self, limbs: &[u64], index: usize) -> u64 {
        let shift = self.shift;
        let high = if index < limbs.len() {
            limbs[index] << shift
        } else {
            0
        };
        // A shift by 64 is undefined, so a zero normalization shift — a
        // divisor that already has its top bit set — takes no low part.
        if shift == 0 || index == 0 {
            return high;
        }
        high | (limbs[index - 1] >> (64 - shift))
    }

    /// `value mod divisor`, discarding the quotient.
    ///
    /// Not `rem_u64`: [`BigUint::rem_u64`] takes the *divisor* as its `u64`,
    /// and this takes the *dividend*. One name for two argument roles is a
    /// trap, so the divisor-carrying type spells it without the suffix.
    #[inline]
    #[must_use]
    pub fn rem(&self, value: u64) -> u64 {
        // Directly, not through `rem_limbs`: a one-word dividend is one
        // `div2by1`, and routing it through the slice loop was measurably
        // worse on the path where this type is already behind the hardware.
        let high = self.normalized_limb(&[value], 1);
        let low = self.normalized_limb(&[value], 0);
        self.div2by1(high, low).1 >> self.shift
    }

    /// `(value / divisor, value mod divisor)`. See [`Self::rem`] on the name.
    #[inline]
    #[must_use]
    pub fn div_rem(&self, value: u64) -> (u64, u64) {
        let high = self.normalized_limb(&[value], 1);
        let low = self.normalized_limb(&[value], 0);
        let (quotient, remainder) = self.div2by1(high, low);
        (quotient, remainder >> self.shift)
    }

    /// The non-negative residue of a signed `value`, in `0..divisor`.
    ///
    /// This is the shape a sieve wants: positions are signed and the residue
    /// indexes a table, so a truncating remainder is the wrong answer for
    /// half the inputs and every caller that re-derives this gets a chance to
    /// be wrong. `i64::MIN` is handled — the magnitude is taken as `u64`.
    #[inline]
    #[must_use]
    pub fn rem_euclid_i64(&self, value: i64) -> u64 {
        let magnitude = self.rem(value.unsigned_abs());
        if value < 0 && magnitude != 0 {
            self.divisor - magnitude
        } else {
            magnitude
        }
    }

    /// `limbs mod divisor` over a little-endian limb slice.
    pub(super) fn rem_limbs(&self, limbs: &[u64]) -> u64 {
        if limbs.is_empty() {
            return 0;
        }
        // The word shifted off the top starts the recurrence. It is below
        // `2^shift ≤ 2⁶³ ≤ normalized`, so Algorithm 4's precondition holds
        // on the first step as it does on every later one.
        let mut remainder = self.normalized_limb(limbs, limbs.len());
        for index in (0..limbs.len()).rev() {
            let low = self.normalized_limb(limbs, index);
            remainder = self.div2by1(remainder, low).1;
        }
        remainder >> self.shift
    }

    /// `(limbs / divisor, limbs mod divisor)` over a little-endian limb slice.
    ///
    /// Scaling dividend and divisor by `2^shift` leaves the quotient
    /// unchanged and multiplies the remainder by the same factor, so the
    /// quotient words need no correction and the remainder is shifted back.
    pub(super) fn div_rem_limbs(&self, limbs: &[u64]) -> (Vec<u64>, u64) {
        if limbs.is_empty() {
            return (Vec::new(), 0);
        }
        let mut quotient = vec![0u64; limbs.len()];
        let mut remainder = self.normalized_limb(limbs, limbs.len());
        for index in (0..limbs.len()).rev() {
            let low = self.normalized_limb(limbs, index);
            let (digit, next) = self.div2by1(remainder, low);
            quotient[index] = digit;
            remainder = next;
        }
        (quotient, remainder >> self.shift)
    }
}

impl BigUint {
    /// `self mod r.divisor()`, using a precomputed reciprocal.
    ///
    /// The answer is [`Self::rem_u64`]'s; this trades a hardware division per
    /// limb for a multiplication per limb. Since [`WordReciprocal::new`] costs one
    /// division in total, that pays back within the first call on a dividend
    /// of two limbs or more, and by a widening margin above — not, as an
    /// earlier version of this sentence claimed, only after the divisor has
    /// been reused a number of times.
    #[must_use]
    pub fn rem_reciprocal(&self, r: &WordReciprocal) -> u64 {
        r.rem_limbs(self.limbs())
    }

    /// `(self / r.divisor(), self mod r.divisor())`, using a precomputed
    /// reciprocal. The answer is [`Self::div_rem_u64`]'s.
    #[must_use]
    pub fn div_rem_reciprocal(&self, r: &WordReciprocal) -> (Self, u64) {
        let (limbs, remainder) = r.div_rem_limbs(self.limbs());
        (Self::from_limbs(limbs), remainder)
    }
}
