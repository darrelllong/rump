//! Multiprecision unsigned and signed integers on `u64` limbs.
//!
//! The representation uses little-endian `u64` limbs because the algorithms
//! are naturally word-oriented. The kernels come straight from the literature
//! so they are auditable against their sources: schoolbook (Knuth's
//! Algorithm M), Karatsuba, and Toom–Cook three- and four-way multiplication,
//! and Knuth's Algorithm D for division — all fully in Rust with no external
//! arithmetic backend.
//!
//! References for the multiplication and division kernels:
//! - Knuth, *TAOCP* vol. 2, §4.3.1, Algorithm M (schoolbook multiply) and
//!   Algorithm D (long division); §4.3.3 ("How Fast Can We Multiply?") for the
//!   Karatsuba and Toom–Cook methods.
//! - Karatsuba & Ofman, *Multiplication of Multidigit Numbers on Automata*,
//!   Soviet Physics–Doklady 7 (1963).
//! - Bodrato, *Towards Optimal Toom–Cook Multiplication…*, WAIFI 2007, for the
//!   optimized Toom evaluation/interpolation sequences.

use core::cmp::Ordering;

// Heuristic crossover where the recursive split starts beating schoolbook in
// this pure-Rust implementation on our benchmark hardware.
const KARATSUBA_THRESHOLD_LIMBS: usize = 32;
// Limit highly lopsided splits; beyond this ratio the extra recursion/temporary
// cost usually outweighs Karatsuba's multiplication count reduction.
const KARATSUBA_MAX_IMBALANCE: usize = 2;
// Toom-3 (three-way Toom–Cook) crossover: above this many limbs in the shorter
// operand, the five sub-multiplications of size n/3 overtake Karatsuba's three
// of size n/2, despite the heavier evaluate/interpolate pass. Measured crossover
// on this pure-Rust implementation is ~120 limbs (Karatsuba still wins at and
// below 4096-bit crypto sizes); see PERFORMANCE.md.
const TOOM3_THRESHOLD_LIMBS: usize = 128;
// Toom-4 (four-way Toom–Cook) crossover. Its exponent (log 7 / log 4 ≈ 1.404)
// beats Toom-3's (1.465), but the seven-point interpolation carries a much
// larger constant here, so it only overtakes Toom-3 for very large operands —
// measured near ~3000 limbs (~190 kbit). Set there as headroom; the practical
// range stays on Toom-3. See PERFORMANCE.md.
const TOOM4_THRESHOLD_LIMBS: usize = 3072;
// Block-decomposition crossover for lopsided products (long ≥ 2·short): the
// shorter length above which cutting the longer operand into short-sized
// digits and multiplying each pair through the balanced kernels beats one
// flat schoolbook pass. This is deliberately far above the Karatsuba
// crossover: at 32-limb digits each block is only marginally sub-quadratic
// while the per-block dispatch, allocation, and scrub overhead is paid in
// full. Measured on M4 (`unbalanced_crossover_timing`): the decomposition
// loses 2x at 32-limb digits, breaks even near 128, still trails slightly
// at 192, and wins 25-35% at 256 rising toward 2x at 512. Set at the first
// size that wins across every measured ratio.
const UNBALANCED_THRESHOLD_LIMBS: usize = 256;
// Width at or above which squaring runs its own kernel rather than the
// general multiplication. The kernel forms each cross term once, so its
// ceiling is a saving of (n−1)/2n — half the limb products,
// asymptotically — but it reaches that ceiling only once the products
// dominate its three passes and their carry walks.
//
// Measured on M4 (`squaring_crossover_timing`, run with `--ignored`) at the
// widths this constant actually gates, which is 8 up to the Karatsuba
// threshold: +12% at 8 limbs, +29% at 12, +36% at 16, +29% at 24. Below 8
// the measurement is inconclusive — the five passes at 1–6 limbs straddle
// zero and one width reads a loss — so the floor sits where the win is
// legible rather than where the arithmetic first favours it.
const SQR_SCHOOLBOOK_MIN_LIMBS: usize = 8;
// Width at or above which squaring stops splitting Karatsuba-style and
// hands over to the multiplication ladder's Toom kernels. Asymptotics say
// only that Toom eventually wins (its 1.465 against Karatsuba's 1.585);
// they do not say where, and the ordinary multiplication crossover is the
// wrong place to guess, because a Karatsuba *square* carries a constant
// factor an ordinary Karatsuba product does not.
//
// Measured against `mul_toom3_ref` on the same operands
// (`squaring_crossover_timing`, run with `--ignored`), quoting only widths
// at which `mul_ref` would actually reach Toom-3 — below its own 128-limb
// threshold that comparison measures a kernel production never calls:
// as the range observed rather than a single figure, because the spread
// between runs is wider than the precision a single figure implies. Across
// thirteen runs on M4: +6.2 to +10.1% at 128 limbs, +12.6 to +20.3% at
// 160, +36 to +40% at 192, +3 to +6% at 256, +19 to +25% at 384, and −12
// to −15% at 512.
//
// These are sample extremes, not bounds, and two revisions of this comment
// have now been written as though they were — the fourteenth run will
// probably widen them again. Only the sign and its persistence across runs
// carry the threshold; the magnitudes are here to show how far from zero
// each row sits, which is why the 512 row matters and the 256 row does
// not. What carries the threshold
// is the sign and its persistence across runs, not any one magnitude. The series is not monotone either, because
// Toom-3's three-way split lands differently on each width (192 = 3·64
// divides exactly, 256 does not), so the threshold is the last width the
// squaring is consistently ahead at rather than a crossing point read off
// a curve. Raising it from an earlier, wrongly-signed reading of 256 was
// confirmed end to end against a build of the previous revision, six
// passes alternating order: public `square_ref` is 6.5% faster at 288
// limbs, 23% at 384, 25% at 447, and at parity at 512 where the handoff
// takes effect.
const SQR_KARATSUBA_MAX_LIMBS: usize = 448;
// Modulus width up to which Barrett's second multiplication is taken as a
// schoolbook half-product rather than a dispatched full product. The half
// costs `k²/2` limb products against the ladder's `O(k^{1.585})`, so it
// wins while `k` is small and loses once the better exponent tells:
// measured on M4, `reduce` is 1.44× ahead at 2 kbit (32 limbs) and 1.33× at
// 8 kbit (128 limbs), at parity near 32 kbit (512 limbs), and behind by
// 1.19× at 64 kbit and 1.32× at 128 kbit. The handoff sits at the measured
// parity point.
const BARRETT_HALF_PRODUCT_MAX_LIMBS: usize = 512;

/// Bitset of the 44 quadratic residues modulo 256, one bit per residue
/// across four words, derived by enumeration.
const SQUARES_MOD_256: [u64; 4] = [
    0x0202_0212_0203_0213,
    0x0202_0212_0202_0213,
    0x0202_0212_0203_0212,
    0x0202_0212_0202_0212,
];

/// Digit alphabet for radix rendering: `0-9` then `a-z`.
const RADIX_DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";

/// Digit count at or above which parsing dispatches to divide and conquer
/// (`RADIX_FROM_DC_THRESHOLD_DIGITS`), and the recursion floor below which
/// sub-problems convert classically (`RADIX_FROM_DC_BASE_DIGITS`).
/// Measured on M4 with the ignored `radix_dc_crossover_timing` probe over
/// repeated runs: with the 512-digit floor the ladder engine ties
/// classical parsing at ~600 decimal digits, leads from ~1,200 (1.4×),
/// and reaches 3× at ~40,000 digits. Correctness does not depend on
/// either value: the recursion's hard base case is the ladder's first
/// entry, and the suite drives both engines over the same vectors.
const RADIX_FROM_DC_THRESHOLD_DIGITS: usize = 1024;
const RADIX_FROM_DC_BASE_DIGITS: usize = 512;

/// Bit width at or above which rendering dispatches to divide and conquer
/// (`RADIX_TO_DC_THRESHOLD_BITS`), and the recursion floor below which
/// sub-values render classically (`RADIX_TO_DC_BASE_BITS`). Measured with
/// the same probe: with the 512-bit floor the ladder render is 1.8× ahead
/// of repeated division at 2,048 bits (~600 decimal digits), 6× at
/// 16 kbit, 11× at 128 kbit — the quadratic curve falling away exactly as
/// its complexity requires. Structurally safe at any values, as above.
const RADIX_TO_DC_THRESHOLD_BITS: usize = 2048;
const RADIX_TO_DC_BASE_BITS: usize = 512;

/// Sign of a [`BigInt`], carried beside an unsigned magnitude.
///
/// Zero is a variant of its own rather than a convention over the magnitude:
/// a sign-magnitude representation otherwise admits `+0` and `−0`, and the
/// derived `Eq` would then disagree with the arithmetic. [`BigInt::from_parts`]
/// enforces the pairing — `Zero` exactly when the magnitude is empty.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sign {
    /// Strictly positive value.
    Positive,
    /// Strictly negative value.
    Negative,
    /// Zero.
    Zero,
}

/// Unsigned multiprecision integer stored as little-endian `u64` limbs.
///
/// Limb 0 holds the least-significant 64 bits, the order the word-oriented
/// kernels want: carries and borrows run from index 0 upward. The
/// representation is canonical — zero is the empty vector, and every other
/// value ends in a non-zero limb — which is what lets the derived `Eq`
/// compare limb vectors directly and lets [`Ord`] decide on limb count before
/// looking at any limb. Every operation that can strand a zero at the top
/// restores the invariant through `normalize`.
#[derive(Debug, Eq, PartialEq)]
pub struct BigUint {
    limbs: Vec<u64>,
}

impl Clone for BigUint {
    fn clone(&self) -> Self {
        Self {
            limbs: self.limbs.clone(),
        }
    }

    /// Copy `source`'s value into `self`'s existing limb buffer
    /// (`Vec::clone_from`): no allocation when the buffer's capacity covers
    /// `source`. The derived implementation would discard the buffer and
    /// allocate a fresh one, which is the cost this type exists to avoid on
    /// its cheapest operations.
    fn clone_from(&mut self, source: &Self) {
        // Scrub the limbs this copy abandons before the truncation strands
        // them: `Drop` covers only the initialized prefix, and the crate
        // promises that values do not linger. The volatile scrub, not a
        // plain fill — stores to memory nothing can legally read again are
        // stores the optimizer may otherwise delete.
        let n = source.limbs.len();
        if self.limbs.len() > n {
            crate::scrub::zeroize_slice(&mut self.limbs[n..]);
        }
        self.limbs.clone_from(&source.limbs);
    }
}

/// Signed multiprecision integer: a [`Sign`] joined to a [`BigUint`]
/// magnitude.
///
/// Sign-magnitude rather than two's complement, because an arbitrary-width
/// value has no fixed sign bit to borrow and every kernel in the crate is
/// written for unsigned limbs. The canonical pairing is `Sign::Zero` exactly
/// when the magnitude is zero — established by [`Self::from_parts`] and
/// preserved by every operation — so the derived `Eq` agrees with the [`Ord`]
/// implementation below.
#[derive(Debug, Eq, PartialEq)]
pub struct BigInt {
    sign: Sign,
    magnitude: BigUint,
}

impl Clone for BigInt {
    fn clone(&self) -> Self {
        Self {
            sign: self.sign,
            magnitude: self.magnitude.clone(),
        }
    }

    /// Copy `source`'s value, reusing the magnitude's limb buffer — see
    /// [`BigUint::clone_from`].
    fn clone_from(&mut self, source: &Self) {
        self.sign = source.sign;
        self.magnitude.clone_from(&source.magnitude);
    }
}

/// Montgomery arithmetic context for a fixed odd modulus.
///
/// Long computations — exponentiation ladders, field arithmetic — spend
/// most of their time doing repeated modular multiplication under one
/// long-lived odd modulus. Precomputing the Montgomery constants once avoids
/// paying the setup cost on every multiply, and the explicit context lets
/// callers stay in the Montgomery domain across whole computations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MontgomeryCtx {
    modulus: BigUint,
    // n0_inv = -n^{-1} mod 2^64 (Montgomery reduction coefficient).
    n0_inv: u64,
    // R^2 mod n with R = 2^(64 * limbs(n)): conversion factor into Montgomery form.
    r2_mod: BigUint,
    // 1 encoded in Montgomery form, i.e. R mod n.
    one_mont: BigUint,
}

/// Numeric order, decided on limb count first and then on limbs from the top
/// down. Length can settle the comparison only because the representation is
/// canonical: with no leading zero limbs, a longer vector is a strictly larger
/// value. Consistent with the derived `Eq` for the same reason — equal values
/// have identical limb vectors.
impl Ord for BigUint {
    fn cmp(&self, other: &Self) -> Ordering {
        // Ordering assumes normalized limb vectors (no most-significant zero
        // limbs). All constructors/arithmetic paths call `normalize()`.
        debug_assert!(
            self.limbs.last().copied() != Some(0),
            "BigUint invariant: no leading zero limbs",
        );
        debug_assert!(
            other.limbs.last().copied() != Some(0),
            "BigUint invariant: no leading zero limbs",
        );
        match self.limbs.len().cmp(&other.limbs.len()) {
            Ordering::Equal => {}
            ord => return ord,
        }

        for (&lhs, &rhs) in self.limbs.iter().rev().zip(other.limbs.iter().rev()) {
            match lhs.cmp(&rhs) {
                Ordering::Equal => {}
                ord => return ord,
            }
        }

        Ordering::Equal
    }
}

impl PartialOrd for BigUint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl BigUint {
    /// Construct zero: the empty limb vector, which is the canonical form and
    /// allocates nothing.
    #[must_use]
    pub fn zero() -> Self {
        Self { limbs: Vec::new() }
    }

    /// Construct one: a single limb.
    #[must_use]
    pub fn one() -> Self {
        Self { limbs: vec![1] }
    }

    /// Construct from a machine word. Zero becomes the empty vector rather
    /// than a single zero limb, which is what keeps the representation
    /// canonical for every value this constructor can produce.
    #[must_use]
    pub fn from_u64(value: u64) -> Self {
        if value == 0 {
            Self::zero()
        } else {
            Self { limbs: vec![value] }
        }
    }

    /// Construct from a `u128`, split into its low and high halves. The high
    /// limb is dropped when it is zero, so the result is canonical without a
    /// `normalize` pass.
    ///
    /// # Panics
    ///
    /// Does not panic in normal use; an internal `expect` guards the limb-split
    /// invariant and would trip only on a corrupt value.
    #[must_use]
    pub fn from_u128(value: u128) -> Self {
        if value == 0 {
            return Self::zero();
        }

        let lo =
            u64::try_from(value & u128::from(u64::MAX)).expect("low 64 bits always fit into u64");
        let hi = u64::try_from(value >> 64).expect("high 64 bits always fit into u64");
        if hi == 0 {
            Self { limbs: vec![lo] }
        } else {
            Self {
                limbs: vec![lo, hi],
            }
        }
    }

    /// Decode big-endian bytes.
    ///
    /// Internally, limb 0 always stores the least-significant 64 bits.
    #[must_use]
    pub fn from_be_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self::zero();
        }

        let mut limbs = Vec::with_capacity(bytes.len().div_ceil(8));
        let mut acc = 0u64;
        let mut shift = 0u32;

        // Walk bytes from least-significant (last byte of the big-endian input)
        // to most-significant, packing eight bytes at a time into a 64-bit limb.
        // When `shift` reaches 64, the current limb is full — push it and start
        // the next one.  Any remaining bytes at the end form a partial limb.
        for &byte in bytes.iter().rev() {
            acc |= u64::from(byte) << shift;
            shift += 8;
            if shift == 64 {
                limbs.push(acc);
                acc = 0;
                shift = 0;
            }
        }

        if shift != 0 {
            limbs.push(acc);
        }

        let mut out = Self { limbs };
        out.normalize();
        out
    }

    /// Encode as big-endian bytes without leading zero bytes.
    ///
    /// Internally, limb 0 stores the least-significant 64 bits, so encoding
    /// walks the limbs in reverse order and strips only the leading zero bytes
    /// introduced by the fixed-width `u64` representation.
    ///
    /// # Panics
    ///
    /// Does not panic in normal use; the internal `expect` would trip only on a
    /// corrupt representation (a non-zero value with no non-zero bytes).
    #[must_use]
    pub fn to_be_bytes(&self) -> Vec<u8> {
        if self.is_zero() {
            return vec![0];
        }

        let mut out = Vec::with_capacity(self.limbs.len() * 8);
        for &limb in self.limbs.iter().rev() {
            out.extend_from_slice(&limb.to_be_bytes());
        }

        let first_nonzero = out
            .iter()
            .position(|&byte| byte != 0)
            .expect("non-zero bigint must encode to at least one non-zero byte");
        out.drain(0..first_nonzero);
        out
    }

    /// View the little-endian limbs (crate-internal: the GF(2^m) kernels
    /// work word-level).
    pub(crate) fn limbs(&self) -> &[u64] {
        &self.limbs
    }

    /// Take the little-endian limb buffer by value (crate-internal: lets a
    /// consumer that already owns the [`BigUint`] mutate the buffer in place
    /// rather than copying it, as [`Gf2m::reduce`](crate::gf2m::Gf2m) does).
    ///
    /// The buffer is extracted with `mem::take` because [`BigUint`]
    /// implements [`Drop`] (the scrub) and a plain field move is forbidden;
    /// the drop that follows scrubs the empty vector. The live buffer's
    /// scrub-on-drop guarantee travels with the caller, who is expected to
    /// hand the buffer back to [`Self::from_limbs`].
    pub(crate) fn into_limbs(mut self) -> Vec<u64> {
        core::mem::take(&mut self.limbs)
    }

    /// Build from little-endian limbs, normalizing leading zeros
    /// (crate-internal counterpart of [`Self::limbs`]).
    pub(crate) fn from_limbs(limbs: Vec<u64>) -> Self {
        let mut out = Self { limbs };
        out.normalize();
        out
    }

    /// Encode as big-endian bytes at a fixed width, zero-padded on the
    /// left — the shape wire formats and share serializations want.
    ///
    /// # Panics
    ///
    /// Panics if the value does not fit in `byte_width` bytes.
    /// `byte_width = 0` is legal only for zero (and yields an empty vector).
    #[must_use]
    pub fn to_be_bytes_padded(&self, byte_width: usize) -> Vec<u8> {
        if self.is_zero() {
            return vec![0u8; byte_width];
        }
        let minimal = self.bits().div_ceil(8);
        assert!(
            minimal <= byte_width,
            "value does not fit in {byte_width} bytes"
        );
        let mut out = vec![0u8; byte_width];
        let bytes = self.to_be_bytes();
        out[byte_width - bytes.len()..].copy_from_slice(&bytes);
        out
    }

    /// Parse from a digit string in the given radix (2 through 36, digits
    /// `0-9a-z`, upper case accepted), `None` on an empty string or an
    /// invalid digit. Leading zeros are accepted; no sign, no whitespace,
    /// no `0x` prefix — this is the value, not a literal.
    ///
    /// Below a measured digit-count crossover the conversion is the
    /// classical method with a word-sized base: digits are consumed in
    /// groups of the largest power of the radix that fits a limb, each
    /// group folded in by one limb multiply-add — O(n²/64) limb work.
    /// Above it, divide and conquer: the string splits against a ladder of
    /// squared radix powers built once per conversion,
    /// `high · radix^k + low`, for O(M(n)·log n) total (Knuth, TAOCP
    /// vol. 2, §4.4; Brent and Zimmermann, *Modern Computer Arithmetic*,
    /// §1.7). Power-of-two radices bypass both paths and pack bits
    /// directly, O(n).
    ///
    /// # Panics
    ///
    /// Panics when `radix` is outside `2..=36`, matching the standard
    /// library's `from_str_radix` contract.
    #[must_use]
    pub fn from_str_radix(text: &str, radix: u32) -> Option<Self> {
        assert!((2..=36).contains(&radix), "radix must be in 2..=36");
        if text.is_empty() {
            return None;
        }
        let mut digits = Vec::with_capacity(text.len());
        for c in text.chars() {
            digits.push(u8::try_from(c.to_digit(radix)?).expect("digit below 36 fits u8"));
        }
        if radix.is_power_of_two() {
            return Some(Self::from_digits_pow2(&digits, radix));
        }
        Some(Self::from_digits_dc(&digits, radix))
    }

    /// Render as a digit string in the given radix (2 through 36, lower
    /// case, no sign, zero as `"0"`), by the mirror of the
    /// [`Self::from_str_radix`] dispatch: bit extraction for power-of-two
    /// radices, classical word-sized division below the divide-and-conquer
    /// threshold, remainder-tree splitting against the radix-power ladder
    /// above it.
    ///
    /// # Panics
    ///
    /// Panics when `radix` is outside `2..=36`.
    #[must_use]
    pub fn to_str_radix(&self, radix: u32) -> String {
        assert!((2..=36).contains(&radix), "radix must be in 2..=36");
        if self.is_zero() {
            return "0".to_string();
        }
        let digits = if radix.is_power_of_two() {
            self.to_digits_pow2(radix)
        } else {
            self.to_digits_dc(radix)
        };
        digits
            .iter()
            .map(|&d| char::from(RADIX_DIGITS[usize::from(d)]))
            .collect()
    }

    /// The largest power of `radix` that fits a `u64`, with its digit count —
    /// the "big base" both classical conversions work in, so that a whole
    /// group of digits costs one limb-sized multiply-add instead of one per
    /// digit. Found by repeated `checked_mul`, which stops at the last power
    /// below `2^64` (`10^19` for decimal, `3^40` for radix 3, `2^63` for
    /// radix 2). The count is the number of digits that power spans.
    fn limb_radix_power(radix: u32) -> (u64, usize) {
        let unit = u64::from(radix);
        let mut power = unit;
        let mut count = 1usize;
        while let Some(next) = power.checked_mul(unit) {
            power = next;
            count += 1;
        }
        (power, count)
    }

    /// Bit-pack digits of a power-of-two radix, least significant first.
    ///
    /// When the radix is `2^b` a digit *is* a `b`-bit field of the value, so
    /// the conversion is a re-slicing of the bit string and needs no
    /// arithmetic at all — no multiply-add per group, no division, and no
    /// crossover to a subquadratic method. A digit straddles a limb boundary
    /// whenever `b` does not divide 64 (radices 8 and 32), which is what the
    /// spill into `limbs[limb + 1]` handles.
    fn from_digits_pow2(digits: &[u8], radix: u32) -> Self {
        let bits_per = radix.trailing_zeros() as usize;
        let total_bits = digits.len() * bits_per;
        let mut limbs = vec![0u64; total_bits.div_ceil(64)];
        let mut position = 0usize;
        for &digit in digits.iter().rev() {
            let limb = position / 64;
            let offset = position % 64;
            limbs[limb] |= u64::from(digit) << offset;
            if offset + bits_per > 64 {
                limbs[limb + 1] |= u64::from(digit) >> (64 - offset);
            }
            position += bits_per;
        }
        let mut value = Self { limbs };
        value.normalize();
        value
    }

    /// Extract digits of a power-of-two radix, most significant first — the
    /// inverse of [`Self::from_digits_pow2`], reading each `b`-bit field out
    /// of the limbs and stitching across a limb boundary where one straddles
    /// it. The digit count comes from the bit width, so the leading digit
    /// carries no padding zeros.
    fn to_digits_pow2(&self, radix: u32) -> Vec<u8> {
        let bits_per = radix.trailing_zeros() as usize;
        let digit_count = self.bits().div_ceil(bits_per);
        let mask = (1u64 << bits_per) - 1;
        let mut digits = Vec::with_capacity(digit_count);
        for index in (0..digit_count).rev() {
            let position = index * bits_per;
            let limb = position / 64;
            let offset = position % 64;
            let mut field = self.limbs[limb] >> offset;
            if offset + bits_per > 64 && limb + 1 < self.limbs.len() {
                field |= self.limbs[limb + 1] << (64 - offset);
            }
            digits.push(u8::try_from(field & mask).expect("masked field is below the radix"));
        }
        digits
    }

    /// Classical parse: fold digit groups in against the word-sized base.
    ///
    /// Horner's rule with the big base `radix^chunk` in place of the radix
    /// itself, so one limb-sized multiply-add absorbs `chunk` digits rather
    /// than one. The leading group takes the remainder `len mod chunk`, which
    /// leaves every later group full and able to scale by the precomputed
    /// `big_base` instead of a freshly exponentiated `radix.pow(take)`. Each
    /// step multiplies a value that grows with the input by a single limb, so
    /// the whole conversion is quadratic in the digit count.
    fn from_digits_classical(digits: &[u8], radix: u32) -> Self {
        let (big_base, chunk) = Self::limb_radix_power(radix);
        let mut value = Self::zero();
        let mut index = 0usize;
        while index < digits.len() {
            let take = (digits.len() - index).min(if index == 0 {
                let rem = digits.len() % chunk;
                if rem == 0 {
                    chunk
                } else {
                    rem
                }
            } else {
                chunk
            });
            let mut group = 0u64;
            for &digit in &digits[index..index + take] {
                group = group * u64::from(radix) + u64::from(digit);
            }
            let base = if take == chunk {
                big_base
            } else {
                u64::from(radix).pow(u32::try_from(take).expect("group fits u32"))
            };
            value = value.mul_ref(&Self::from_u64(base));
            value.add_assign_ref(&Self::from_u64(group));
            index += take;
        }
        value
    }

    /// The squared-power ladder `radix^(chunk·2^i)`, built once per
    /// conversion and shared down the recursion (Brent and Zimmermann
    /// build it exactly once; rebuilding it per level is what turns the
    /// subquadratic method back into a quadratic one). `chunk` is the
    /// digit count of the first entry; entry `i` spans `chunk·2^i` digits.
    fn radix_power_ladder(radix: u32, digit_count: usize) -> (Vec<Self>, usize) {
        let (big_base, chunk) = Self::limb_radix_power(radix);
        let mut ladder = vec![Self::from_u64(big_base)];
        let mut span = chunk;
        while span * 2 < digit_count {
            let top = ladder.last().expect("ladder starts non-empty").square_ref();
            ladder.push(top);
            span *= 2;
        }
        (ladder, chunk)
    }

    /// The render-side ladder, capped by the value's bit width — sizing it
    /// by a digit proxy overshoots by up to two entries, and the extra
    /// entries are the most expensive squarings in the whole conversion.
    fn radix_power_ladder_bits(radix: u32, bit_width: usize) -> (Vec<Self>, usize) {
        let (big_base, chunk) = Self::limb_radix_power(radix);
        let mut ladder = vec![Self::from_u64(big_base)];
        while ladder.last().expect("non-empty").bits() * 2 < bit_width {
            let top = ladder.last().expect("non-empty").square_ref();
            ladder.push(top);
        }
        (ladder, chunk)
    }

    /// Divide-and-conquer parse: split against the shared power ladder.
    fn from_digits_dc(digits: &[u8], radix: u32) -> Self {
        if digits.len() < RADIX_FROM_DC_THRESHOLD_DIGITS {
            return Self::from_digits_classical(digits, radix);
        }
        let (ladder, chunk) = Self::radix_power_ladder(radix, digits.len());
        Self::from_digits_ladder(digits, radix, &ladder, chunk, RADIX_FROM_DC_BASE_DIGITS)
    }

    /// The recursion itself: pick the ladder entry `radix^span` with `span`
    /// below the digit count, convert the leading `len − span` digits and the
    /// trailing `span` digits separately, and recombine as
    /// `high · radix^span + low`. Choosing the largest such entry keeps the
    /// two halves within a factor of two of each other, which is what makes
    /// the recursion depth logarithmic and the multiply at each level a
    /// balanced one.
    ///
    /// `span` is tracked alongside `index` rather than recomputed: entry `i`
    /// spans `chunk·2^i` digits by construction of the ladder.
    fn from_digits_ladder(
        digits: &[u8],
        radix: u32,
        ladder: &[Self],
        chunk: usize,
        base_digits: usize,
    ) -> Self {
        // The first clause is the structural base case, independent of any
        // tuning constant: with no ladder entry spanning fewer digits than
        // the input, there is nothing to split. The second is the measured
        // floor below which classical conversion wins.
        if digits.len() <= chunk || digits.len() < base_digits {
            return Self::from_digits_classical(digits, radix);
        }
        // The largest entry spanning fewer digits than the input.
        let mut index = 0usize;
        let mut span = chunk;
        while index + 1 < ladder.len() && span * 2 < digits.len() {
            index += 1;
            span *= 2;
        }
        let split = digits.len() - span;
        let high = Self::from_digits_ladder(&digits[..split], radix, ladder, chunk, base_digits);
        let low = Self::from_digits_ladder(&digits[split..], radix, ladder, chunk, base_digits);
        let mut value = high.mul_ref(&ladder[index]);
        value.add_assign_ref(&low);
        value
    }

    /// Classical render: divide out the word-sized base, emitting groups.
    /// Callers route power-of-two radices to bit extraction first; radix 2
    /// would pack 63 digits per limb and overrun the group buffer below.
    fn to_digits_classical(&self, radix: u32) -> Vec<u8> {
        debug_assert!(!radix.is_power_of_two(), "powers of two take the bit path");
        let (big_base, chunk) = Self::limb_radix_power(radix);
        let mut groups = Vec::new();
        let mut rest = self.clone();
        while !rest.is_zero() {
            let (quotient, remainder) = Self::div_rem_limb(rest.limbs(), big_base);
            groups.push(remainder);
            rest = quotient;
        }
        let mut digits = Vec::with_capacity(groups.len() * chunk);
        for (index, &group) in groups.iter().rev().enumerate() {
            // Among the radices that reach this path — the dispatch sends
            // powers of two to bit extraction — radix 3 packs the most
            // digits per limb: 3^40 < 2^64. The assert above records the
            // precondition the buffer size relies on.
            let mut buffer = [0u8; 40];
            let mut value = group;
            for slot in buffer[..chunk].iter_mut().rev() {
                *slot = u8::try_from(value % u64::from(radix)).expect("digit below radix");
                value /= u64::from(radix);
            }
            // The most significant group drops its leading zeros; interior
            // groups keep them — they are positional.
            let start = if index == 0 {
                buffer[..chunk]
                    .iter()
                    .position(|&d| d != 0)
                    .unwrap_or(chunk - 1)
            } else {
                0
            };
            digits.extend_from_slice(&buffer[start..chunk]);
        }
        digits
    }

    /// Divide-and-conquer render: split by the shared power ladder, the
    /// low half zero-padded to the split's exact digit span.
    fn to_digits_dc(&self, radix: u32) -> Vec<u8> {
        if self.bits() < RADIX_TO_DC_THRESHOLD_BITS {
            return self.to_digits_classical(radix);
        }
        let (ladder, chunk) = Self::radix_power_ladder_bits(radix, self.bits());
        self.to_digits_ladder(radix, &ladder, chunk, RADIX_TO_DC_BASE_BITS)
    }

    /// The mirror of [`Self::from_digits_ladder`]: one division by the
    /// largest ladder entry `radix^span` below the value splits it into a
    /// quotient and a remainder that render independently, and the remainder
    /// occupies exactly `span` positional digits — hence the zero padding
    /// before the low half is appended. Without that padding a remainder with
    /// fewer significant digits than its span would silently shift the whole
    /// low half left.
    fn to_digits_ladder(
        &self,
        radix: u32,
        ladder: &[Self],
        chunk: usize,
        base_bits: usize,
    ) -> Vec<u8> {
        // The first clause is the structural base case — a value no wider
        // than the first ladder entry splits into nothing; the second is
        // the measured floor below which classical division wins.
        if self <= &ladder[0] || self.bits() < base_bits {
            return self.to_digits_classical(radix);
        }
        // The largest entry below the value keeps the halves balanced.
        let mut index = 0usize;
        let mut span = chunk;
        while index + 1 < ladder.len() && ladder[index + 1] < *self {
            index += 1;
            span *= 2;
        }
        let (high, low) = self.div_rem(&ladder[index]);
        debug_assert!(
            !high.is_zero(),
            "the chosen ladder entry is below the value"
        );
        let mut digits = high.to_digits_ladder(radix, ladder, chunk, base_bits);
        let low_digits = low.to_digits_ladder(radix, ladder, chunk, base_bits);
        digits.resize(digits.len() + span - low_digits.len(), 0);
        digits.extend_from_slice(&low_digits);
        digits
    }

    /// The top (up to) 64 significant bits packed into a `u64`, with the
    /// count of those bits — the shared mantissa of [`Self::to_f64_lossy`]
    /// and [`Self::ln_approx`]. Below `2^64` this is the value and its bit
    /// length; above it, the top limb's significant bits filled from the
    /// limb below.
    fn top_64_bits(&self) -> (u64, usize) {
        let bits = self.bits();
        if bits <= 64 {
            return (self.limbs.first().copied().unwrap_or(0), bits);
        }
        // bits > 64 guarantees at least two limbs.
        let hi = self.limbs[self.limbs.len() - 1];
        let lo = self.limbs[self.limbs.len() - 2];
        let top_bits = 64 - hi.leading_zeros() as usize; // 1..=64
        let shift = 64 - top_bits;
        let mantissa = if shift == 0 {
            hi
        } else {
            (hi << shift) | (lo >> top_bits)
        };
        (mantissa, 64)
    }

    /// The value as an `f64` — the lossy narrowing the parameter heuristics
    /// of factoring and lattice work are written in terms of. The result is
    /// within one unit in the last place of the true value (the top 64 bits
    /// are taken as the mantissa, then rounded to `f64`'s 53; the direction
    /// is unspecified), and saturates to `f64::INFINITY` above the
    /// double-precision range (~2^1024).
    #[must_use]
    pub fn to_f64_lossy(&self) -> f64 {
        let bits = self.bits();
        if bits == 0 {
            return 0.0;
        }
        let (mantissa, mantissa_bits) = self.top_64_bits();
        let exponent = bits - mantissa_bits;
        mantissa as f64 * 2f64.powi(i32::try_from(exponent).unwrap_or(i32::MAX))
    }

    /// A natural logarithm of the value as an `f64`, for the size-driven
    /// tuning heuristics stated in terms of `ln n` (the smoothness bound
    /// `exp(½√(ln n · ln ln n))`, for one). Computed as
    /// `ln(mantissa) + (bits − mantissa_bits)·ln 2` so it stays finite far
    /// past the point where the value itself overflows `f64`.
    ///
    /// # Panics
    ///
    /// Panics when the value is zero, whose logarithm is undefined.
    #[must_use]
    pub fn ln_approx(&self) -> f64 {
        assert!(!self.is_zero(), "ln is undefined at zero");
        let bits = self.bits();
        let (mantissa, mantissa_bits) = self.top_64_bits();
        (mantissa as f64).ln() + ((bits - mantissa_bits) as f64) * core::f64::consts::LN_2
    }

    /// How many digits the value has in `radix`, without writing them out.
    ///
    /// [`Self::to_str_radix`] answers this too, by producing the whole
    /// expansion — repeated division, quadratic in the limbs — when the caller
    /// wanted a single number. Size-driven tuning asks for the length and
    /// throws the digits away, and at that point the expansion is the cost.
    ///
    /// The logarithm decides every value but one class: `log_radix(n)` is an
    /// integer exactly at the powers of `radix`, and there the floor can land
    /// either side of it. So the estimate is corrected by comparison, which
    /// runs at most once in each direction and costs about one exponentiation
    /// by squaring rather than one division per digit.
    ///
    /// Zero has one digit, by the convention that writes it `0`.
    ///
    /// # Panics
    ///
    /// Panics when `radix` is below two, which names no positional system.
    #[must_use]
    pub fn digit_count(&self, radix: u32) -> usize {
        assert!(radix >= 2, "radix must be at least two");
        if self.is_zero() {
            return 1;
        }
        let radix_value = BigUint::from_u64(u64::from(radix));
        let estimate = self.ln_approx() / f64::from(radix).ln();
        let mut digits = if estimate.is_finite() && estimate > 0.0 {
            estimate as usize + 1
        } else {
            1
        };
        while digits > 1 && *self < radix_value.pow_u64(digits as u64 - 1) {
            digits -= 1;
        }
        while *self >= radix_value.pow_u64(digits as u64) {
            digits += 1;
        }
        digits
    }

    /// The low 128 bits as a `u128`; bits above position 127 are silently
    /// dropped. For callers that have already pinned their operand range
    /// (fixed-field reductions and the like).
    #[must_use]
    pub fn low_u128(&self) -> u128 {
        let lo = self.limbs.first().copied().unwrap_or(0);
        let hi = self.limbs.get(1).copied().unwrap_or(0);
        u128::from(lo) | (u128::from(hi) << 64)
    }

    /// The low `k` bits as a fresh value — `self mod 2^k`, splitting at any
    /// bit boundary, limb-aligned or not. Truncation to `⌈k/64⌉` limbs
    /// handles the whole-limb part; a mask clears the surplus bits of the
    /// boundary limb when `k` is not a multiple of 64. Reduction modulo a
    /// power of two is a truncation, not a division, which is why
    /// [`BarrettCtx::reduce`] can take its `mod b^{k+1}` windows this way.
    #[must_use]
    pub fn low_bits(&self, k: usize) -> Self {
        let full_limbs = k / 64;
        let partial_bits = k % 64;
        let take = self.limbs.len().min(k.div_ceil(64));
        let mut limbs: Vec<u64> = self.limbs[..take].to_vec();
        if partial_bits != 0 && limbs.len() > full_limbs {
            limbs[full_limbs] &= (1u64 << partial_bits) - 1;
        }
        Self::from_limbs(limbs)
    }

    /// Whether the value is zero — an emptiness test, because the canonical
    /// form of zero is the empty limb vector and no other representation of
    /// it exists.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Whether the value is odd: bit 0 of limb 0, with zero handled first
    /// because it has no limb to read. This is the predicate
    /// [`MontgomeryCtx::new`] gates on — REDC needs `gcd(2^64, n) = 1` — and
    /// the one the binary gcd and Jacobi recursions branch on.
    #[must_use]
    pub fn is_odd(&self) -> bool {
        !self.is_zero() && (self.limbs[0] & 1) == 1
    }

    /// Whether the value is exactly one: a single limb holding 1. The
    /// canonical form makes this a two-word test rather than a comparison
    /// against a freshly built [`Self::one`].
    #[must_use]
    pub fn is_one(&self) -> bool {
        self.limbs.len() == 1 && self.limbs[0] == 1
    }

    /// Number of significant bits: `64` per full limb below the top one, plus
    /// the top limb's width from its leading-zero count. Zero has zero bits.
    ///
    /// # Panics
    ///
    /// Does not panic in normal use; the internal `expect` would trip only on a
    /// corrupt representation (a non-zero value with no limbs).
    #[must_use]
    pub fn bits(&self) -> usize {
        if self.is_zero() {
            return 0;
        }

        let top = *self
            .limbs
            .last()
            .expect("non-zero bigint has at least one limb");
        let top_bits = (u64::BITS - top.leading_zeros()) as usize;
        (self.limbs.len() - 1) * 64 + top_bits
    }

    /// Integer square root: the largest `r` with `r² ≤ self` — the root
    /// half of [`Self::sqrt_rem`], which documents the Newton iteration
    /// both share. Callers that do not need the remainder use this and skip
    /// the full-width squaring and subtraction that produce it.
    #[must_use]
    pub fn sqrt_floor(&self) -> Self {
        if self.is_zero() || self.is_one() {
            return self.clone();
        }
        self.sqrt_newton()
    }

    /// Integer square root with remainder: `(r, self − r²)` for the largest
    /// `r` with `r² ≤ self`.
    ///
    /// Newton's iteration on `x ↦ (x + self/x)/2` from a one-bit seed above
    /// the root. For any `x > 0` the iterate is at least `⌊√self⌋` (AM–GM:
    /// `(x + n/x)/2 ≥ √n`, and the floor of the average cannot drop below
    /// the floor of the root), and from a starting point above the root the
    /// sequence decreases strictly until it reaches it, so the first
    /// non-decrease certifies the answer (Cohen, *A Course in Computational
    /// Algebraic Number Theory*, Algorithm 1.7.1). Each step costs one
    /// division at the operand's width, and convergence is quadratic — the
    /// error's bit count roughly halves per step — so the step count is
    /// about log₂ of the bit width: a dozen iterations at 8,192 bits,
    /// measured. This replaced a bisection whose every probe was a
    /// full-width square: 13.4 ms fell to 89 µs at 8,192 bits on M4, 150×
    /// (the ignored `sqrt_newton_vs_bisection_timing` probe reproduces
    /// both numbers against the bisection retained in the test module).
    #[must_use]
    pub fn sqrt_rem(&self) -> (Self, Self) {
        if self.is_zero() || self.is_one() {
            return (self.clone(), Self::zero());
        }
        let root = self.sqrt_newton();
        let square = root.square_ref();
        (root, self.sub_ref(&square))
    }

    /// The Newton core shared by [`Self::sqrt_floor`] and
    /// [`Self::sqrt_rem`] (the latter documents the iteration and its
    /// certificate). Requires `self ≥ 2`, which both callers establish; the
    /// returned root is certified by the first non-decrease, so no squaring
    /// happens here — the remainder is the caller's business.
    fn sqrt_newton(&self) -> Self {
        debug_assert!(!self.is_zero() && !self.is_one(), "callers handle 0 and 1");
        // Seed: 2^⌈bits/2⌉ ≥ ⌈√self⌉, one bit above the root's width.
        let mut current = Self::zero();
        current.set_bit(self.bits().div_ceil(2));
        loop {
            // next = (current + self/current) / 2
            let (quotient, _) = self.div_rem(&current);
            let mut next = current.add_ref(&quotient);
            next.shr1();
            if next >= current {
                debug_assert!(
                    current.square_ref() <= *self,
                    "certified root is not above the value"
                );
                return current;
            }
            current = next;
        }
    }

    /// Population count: the number of set bits, summed limb by limb through
    /// `u64::count_ones`. The Hamming weight of the binary expansion, which
    /// is exactly the number of multiplications a binary exponentiation
    /// ladder performs for this exponent.
    #[must_use]
    pub fn popcount(&self) -> usize {
        self.limbs
            .iter()
            .map(|limb| limb.count_ones() as usize)
            .sum()
    }

    /// The number of trailing zero bits — the 2-adic valuation — or `None`
    /// for zero, which has no well-defined valuation.
    #[must_use]
    pub fn trailing_zeros(&self) -> Option<usize> {
        self.limbs
            .iter()
            .position(|&limb| limb != 0)
            .map(|index| index * 64 + self.limbs[index].trailing_zeros() as usize)
    }

    /// `self^exponent` for a machine-word exponent, by binary
    /// exponentiation — the small-power helper the root routines need.
    #[must_use]
    pub fn pow_u64(&self, exponent: u64) -> Self {
        let mut result = Self::one();
        let mut base = self.clone();
        let mut remaining = exponent;
        while remaining > 0 {
            if remaining & 1 == 1 {
                result = result.mul_ref(&base);
            }
            remaining >>= 1;
            if remaining > 0 {
                base = base.square_ref();
            }
        }
        result
    }

    /// Floor of the `k`-th root: the largest `r` with `r^k ≤ self`.
    ///
    /// Newton's iteration on `x ↦ ((k−1)·x + self/x^(k−1))/k` from a
    /// one-bit seed above the root; as with [`Self::sqrt_rem`], every
    /// iterate stays at or above the true floor and the sequence decreases
    /// strictly until it certifies itself (Cohen, Algorithm 1.7.1 for the
    /// square case; the general `k` is the same argument through the
    /// arithmetic–geometric mean inequality on `k` terms).
    ///
    /// # Panics
    ///
    /// Panics when `k` is zero — the zeroth root does not exist.
    #[must_use]
    pub fn nth_root_floor(&self, k: u64) -> Self {
        assert!(k > 0, "the zeroth root does not exist");
        if k == 1 || self.is_zero() || self.is_one() {
            return self.clone();
        }
        if u64::try_from(self.bits()).expect("bit count fits u64") <= k {
            // 2^k > self for self < 2^k, so the root is 1.
            return Self::one();
        }
        let k_value = Self::from_u64(k);
        let k_minus_one = Self::from_u64(k - 1);
        // Seed: 2^⌈bits/k⌉ ≥ ⌈self^(1/k)⌉.
        let mut current = Self::zero();
        current.set_bit(
            self.bits()
                .div_ceil(usize::try_from(k).expect("k fits usize")),
        );
        loop {
            let (quotient, _) = self.div_rem(&current.pow_u64(k - 1));
            let mut next = current.mul_ref(&k_minus_one);
            next.add_assign_ref(&quotient);
            let (next, _) = next.div_rem(&k_value);
            if next >= current {
                debug_assert!(
                    current.pow_u64(k) <= *self,
                    "certified root is not above the value"
                );
                return current;
            }
            current = next;
        }
    }

    /// Whether the value is a perfect square, by residue filters and one
    /// certified square root. The filters reject most non-squares without
    /// arithmetic: squares occupy 44 of 256 residues modulo 256, and the
    /// modulus 9·5·7·13·17 = 69 615 folds five more character tests into a
    /// single word remainder (the classical filter set, as in GMP's
    /// `mpz_perfect_square_p`).
    #[must_use]
    pub fn is_square(&self) -> bool {
        if self.is_zero() {
            return true;
        }
        let low_byte = self.limbs[0] & 0xff;
        if SQUARES_MOD_256[(low_byte / 64) as usize] >> (low_byte % 64) & 1 == 0 {
            return false;
        }
        let folded = self.rem_u64(69_615);
        // Bit masks of the quadratic residues, derived by enumeration
        // (k² mod m for k in 0..m) rather than transcription.
        for &(modulus, residue_mask) in &[
            (9u64, 0x93u64), // {0,1,4,7}
            (5, 0x13),       // {0,1,4}
            (7, 0x17),       // {0,1,2,4}
            (13, 0x161b),    // {0,1,3,4,9,10,12}
            (17, 0x1a317),   // {0,1,2,4,8,9,13,15,16}
        ] {
            if residue_mask >> (folded % modulus) & 1 == 0 {
                return false;
            }
        }
        let (_, remainder) = self.sqrt_rem();
        remainder.is_zero()
    }

    /// Whether the value is `m^k` for some `m` and some `k ≥ 2`. Checks one
    /// certified root per prime exponent up to the bit length — a composite
    /// exponent `k = a·b` implies a perfect `a`-th power, so primes
    /// suffice — with the 2-adic valuation as a fast filter: any `k` must
    /// divide the valuation when it is non-zero. Zero and one are perfect
    /// powers by convention (`0^2`, `1^2`).
    ///
    /// On odd operands the valuation filter is inert and every prime
    /// exponent below the bit width pays a full root: measured on M4,
    /// 0.9 ms at 1,024 bits and 21.9 ms at 4,096, growing roughly
    /// cubically. Residue-based exponent filters would trim this and are
    /// a candidate refinement.
    #[must_use]
    pub fn is_perfect_power(&self) -> bool {
        if self.is_zero() || self.is_one() {
            return true;
        }
        let valuation = self.trailing_zeros().expect("non-zero value");
        if valuation == 1 {
            // 2 divides the value exactly once; no k ≥ 2 divides 1.
            return false;
        }
        let bits = self.bits();
        let mut k = 2u64;
        while u64::try_from(bits).expect("bit count fits u64") > k {
            let k_is_prime = {
                let mut prime = true;
                let mut d = 2;
                while d * d <= k {
                    if k.is_multiple_of(d) {
                        prime = false;
                        break;
                    }
                    d += 1;
                }
                prime
            };
            let divides_valuation = valuation == 0
                || valuation.is_multiple_of(usize::try_from(k).expect("k fits usize"));
            if k_is_prime && divides_valuation {
                let root = self.nth_root_floor(k);
                if root.pow_u64(k) == *self {
                    return true;
                }
            }
            k += 1;
        }
        false
    }

    /// Test bit `index`, counted from the least-significant bit of limb 0.
    /// Indices at or above the value's width read as `false`: the value is
    /// conceptually zero-extended, so exponentiation ladders may scan a fixed
    /// window past the top set bit without a bound check.
    #[must_use]
    pub fn bit(&self, index: usize) -> bool {
        let limb = index / 64;
        let shift = index % 64;
        if limb >= self.limbs.len() {
            false
        } else {
            ((self.limbs[limb] >> shift) & 1) == 1
        }
    }

    /// Set bit `index`, growing the limb vector with zero limbs when the
    /// index lies above the current width. Setting a bit can only raise the
    /// top limb above zero, so the canonical form survives without a
    /// `normalize` pass. This is how the crate materializes a power of two —
    /// `R² = 2^(128w)` in [`MontgomeryCtx::new`], the Newton seeds in
    /// [`Self::sqrt_rem`] — without building and shifting a value.
    pub fn set_bit(&mut self, index: usize) {
        let limb = index / 64;
        let shift = index % 64;
        if self.limbs.len() <= limb {
            self.limbs.resize(limb + 1, 0);
        }
        self.limbs[limb] |= 1u64 << shift;
    }

    /// Add another bigint in place: `self` grows to `other`'s width, then one
    /// carry pass runs across the overlap and the carry ripples through the
    /// remaining limbs, pushing a new top limb only if it escapes. The
    /// accumulator is a `u128` so the sum of two limbs and a carry cannot
    /// overflow. The result stays canonical without a `normalize` pass —
    /// adding to a non-zero top limb cannot zero it.
    ///
    /// # Panics
    ///
    /// Does not panic in normal use; an internal `expect` guards the
    /// limb-packing invariant (a `u128` accumulator splitting back into `u64`
    /// limbs) and would trip only on a logic error.
    pub fn add_assign_ref(&mut self, other: &Self) {
        if other.is_zero() {
            return;
        }

        if self.limbs.len() < other.limbs.len() {
            self.limbs.resize(other.limbs.len(), 0);
        }

        let mut carry = 0u128;
        for i in 0..other.limbs.len() {
            let sum = u128::from(self.limbs[i]) + u128::from(other.limbs[i]) + carry;
            self.limbs[i] = low_u64(sum);
            carry = sum >> 64;
        }

        let mut i = other.limbs.len();
        while carry != 0 && i < self.limbs.len() {
            let sum = u128::from(self.limbs[i]) + carry;
            self.limbs[i] = low_u64(sum);
            carry = sum >> 64;
            i += 1;
        }

        if carry != 0 {
            self.limbs
                .push(u64::try_from(carry).expect("final carry from u64 addition is at most 1"));
        }
    }

    /// Return `self + other`, leaving both operands intact: a clone of `self`
    /// followed by [`Self::add_assign_ref`]. The clone is the price of the
    /// functional form; [`Self::assign_add`] avoids it when the caller
    /// already owns a destination buffer.
    #[must_use]
    pub fn add_ref(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.add_assign_ref(other);
        out
    }

    /// Write `lhs + rhs` into `self`, reusing its limb buffer — the
    /// three-operand form (the shape of GMP's `mpz_add`) for callers that
    /// hold the result's storage across calls. One carry pass over the
    /// operands; no allocation once the buffer's capacity covers the result.
    /// Contrast [`Self::add_assign_ref`], which *accumulates* into `self`;
    /// this form replaces it.
    ///
    /// # Panics
    ///
    /// Does not panic in normal use; an internal `expect` guards the
    /// limb-packing invariant (a `u128` accumulator splitting back into `u64`
    /// limbs) and would trip only on a logic error.
    pub fn assign_add(&mut self, lhs: &Self, rhs: &Self) {
        debug_assert!(
            lhs.limbs.last() != Some(&0) && rhs.limbs.last() != Some(&0),
            "operands arrive canonical; the result's canonical form relies on it"
        );
        let (long, short) = if lhs.limbs.len() >= rhs.limbs.len() {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        };
        // Shape the buffer to the working width. Any tail beyond it is
        // scrubbed (volatile, as in `Drop`) before the shrink strands it —
        // `Drop` covers only the initialized prefix.
        let n = long.limbs.len();
        if self.limbs.len() > n {
            crate::scrub::zeroize_slice(&mut self.limbs[n..]);
        }
        self.limbs.resize(n, 0);
        let mut carry = 0u128;
        for i in 0..n {
            let rhs_limb = if i < short.limbs.len() {
                u128::from(short.limbs[i])
            } else {
                0
            };
            let sum = u128::from(long.limbs[i]) + rhs_limb + carry;
            self.limbs[i] = low_u64(sum);
            carry = sum >> 64;
        }
        // Canonical without a normalize pass: `long`'s top limb is
        // non-zero, so the top result limb can be zero only when the sum
        // carried out — and that carry is pushed.
        if carry != 0 {
            self.limbs
                .push(u64::try_from(carry).expect("final carry from u64 addition is at most 1"));
        }
    }

    /// Write `lhs - rhs` into `self`, reusing its limb buffer — the
    /// three-operand counterpart of [`Self::assign_add`]. One borrow pass;
    /// no allocation once the buffer's capacity covers the result.
    /// Contrast [`Self::sub_assign_ref`], which subtracts *from* `self`;
    /// this form replaces it.
    ///
    /// # Panics
    ///
    /// Panics if `lhs < rhs`.
    pub fn assign_sub(&mut self, lhs: &Self, rhs: &Self) {
        assert!(lhs.cmp(rhs) != Ordering::Less, "BigUint underflow");
        // Shape the buffer as in `assign_add`, scrubbing any abandoned
        // tail before it shrinks. The normalize below pops only zeros, so
        // the shrink it performs strands nothing.
        let n = lhs.limbs.len();
        if self.limbs.len() > n {
            crate::scrub::zeroize_slice(&mut self.limbs[n..]);
        }
        self.limbs.resize(n, 0);
        let mut borrow = 0u128;
        for i in 0..n {
            let minuend = u128::from(lhs.limbs[i]);
            let subtrahend = if i < rhs.limbs.len() {
                u128::from(rhs.limbs[i])
            } else {
                0
            } + borrow;
            if minuend >= subtrahend {
                self.limbs[i] = low_u64(minuend - subtrahend);
                borrow = 0;
            } else {
                self.limbs[i] = low_u64((1u128 << 64) + minuend - subtrahend);
                borrow = 1;
            }
        }
        self.normalize();
    }

    /// `self ← minuend - self`, in place — the reversed subtraction the
    /// signed in-place operations need when the result's magnitude is the
    /// *other* operand's minus this one's. Panics if `minuend < self`.
    fn rsub_assign_ref(&mut self, minuend: &Self) {
        assert!(minuend.cmp(self) != Ordering::Less, "BigUint underflow");
        self.limbs.resize(minuend.limbs.len(), 0);
        let mut borrow = 0u128;
        for i in 0..self.limbs.len() {
            let lhs = u128::from(minuend.limbs[i]);
            let subtrahend = u128::from(self.limbs[i]) + borrow;
            if lhs >= subtrahend {
                self.limbs[i] = low_u64(lhs - subtrahend);
                borrow = 0;
            } else {
                self.limbs[i] = low_u64((1u128 << 64) + lhs - subtrahend);
                borrow = 1;
            }
        }
        self.normalize();
    }

    /// Subtract another bigint in place: one borrow pass, each limb difference
    /// taken in `u128` and biased by `2^64` when it would go negative so the
    /// borrow is carried explicitly rather than inferred from a wrap. A
    /// cancellation can empty the top limbs, so the pass ends in `normalize`.
    ///
    /// # Panics
    ///
    /// Panics if `self < other`. ℕ is closed under addition but not under
    /// subtraction, and this type has no sign in which to record a negative
    /// difference; [`BigInt::sub_assign_ref`] is the total operation.
    pub fn sub_assign_ref(&mut self, other: &Self) {
        assert!((*self).cmp(other) != Ordering::Less, "BigUint underflow");
        if other.is_zero() {
            return;
        }

        let mut borrow = 0u128;
        for i in 0..self.limbs.len() {
            let lhs = u128::from(self.limbs[i]);
            let rhs = if i < other.limbs.len() {
                u128::from(other.limbs[i])
            } else {
                0
            };

            let subtrahend = rhs + borrow;
            if lhs >= subtrahend {
                self.limbs[i] = low_u64(lhs - subtrahend);
                borrow = 0;
            } else {
                self.limbs[i] = low_u64((1u128 << 64) + lhs - subtrahend);
                borrow = 1;
            }
        }

        self.normalize();
    }

    /// Return `self - other`: a clone of `self` followed by
    /// [`Self::sub_assign_ref`].
    ///
    /// # Panics
    ///
    /// Panics if `self < other`, for the reason given on
    /// [`Self::sub_assign_ref`].
    #[must_use]
    pub fn sub_ref(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.sub_assign_ref(other);
        out
    }

    /// Multiply two big integers, choosing the multiplication kernel by
    /// operand size: schoolbook (Knuth's Algorithm M) by default, Karatsuba
    /// above 32 limbs, three-way Toom–Cook above 128, and four-way Toom–Cook
    /// above 3072 (`KARATSUBA_/TOOM3_/TOOM4_THRESHOLD_LIMBS`). Each successive
    /// kernel is asymptotically cheaper but splits the operands more, so its
    /// overhead only pays off past the crossover — small products stay
    /// schoolbook. A lopsided pair (`long ≥ 2·short`) whose shorter operand
    /// is past `UNBALANCED_THRESHOLD_LIMBS` — 256, this fourth kernel's own
    /// measured crossover, well above Karatsuba's — takes
    /// `mul_unbalanced_ref`, block decomposition into balanced products;
    /// lopsided pairs below that threshold stay schoolbook, which
    /// measurement favors there. The module header cites each algorithm. A
    /// zero operand short-circuits to zero.
    ///
    /// # Panics
    ///
    /// Does not panic in normal use: an internal `expect` guards a limb-packing
    /// invariant (`u128` accumulators splitting back into `u64` limbs) and can
    /// trip only on a logic error in a kernel.
    #[must_use]
    pub fn mul_ref(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        if Self::should_use_toom4(self, other) {
            return self.mul_toom4_ref(other);
        }

        if Self::should_use_toom3(self, other) {
            return self.mul_toom3_ref(other);
        }

        if Self::should_use_karatsuba(self, other) {
            return self.mul_karatsuba_ref(other);
        }

        if Self::should_use_unbalanced(self, other) {
            return self.mul_unbalanced_ref(other);
        }

        Self::mul_schoolbook_ref(self, other)
    }

    /// Square a value, exploiting the symmetry that lets a squaring form
    /// each distinct cross term once instead of twice.
    ///
    /// Three regimes, each measured against the kernel it displaces rather
    /// than against a proxy. Below `SQR_SCHOOLBOOK_MIN_LIMBS` (8) the
    /// specialized passes do not repay themselves and this is
    /// [`Self::mul_ref`]. From there to the Karatsuba crossover it is
    /// `sqr_schoolbook_ref`, forming each cross term once. From there to
    /// `SQR_KARATSUBA_MAX_LIMBS` (448) it is `sqr_karatsuba_ref`, the same
    /// split with all three sub-products squarings, measured ahead of the
    /// Toom-3 multiplication it would otherwise defer to. At 448 limbs and
    /// above Toom wins and this hands back to [`Self::mul_ref`].
    ///
    /// End to end, against the multiplication a caller would otherwise
    /// write: +12% at 8 limbs, +36% at 16, +32% at 32, +27% at 64, +26% at
    /// 127. Outside the specialized range the two are the same code, less
    /// this function's dispatch — which at one and two limbs is a
    /// reproducible 3–5% of an operation that costs some fifteen
    /// nanoseconds, the price of choosing between four kernels rather than
    /// calling one.
    ///
    /// The Montgomery domain keeps its own in-domain squaring
    /// ([`MontgomeryCtx::square_mont`]), which fuses the reduction and is
    /// not reached from here.
    #[must_use]
    pub fn square_ref(&self) -> Self {
        // Zero needs no special case: its width is below every threshold,
        // so it takes the multiplication, which short-circuits it.
        // Ordered so the narrowest operands — the commonest, and the ones
        // whose absolute cost leaves least room for dispatch — decide on a
        // single comparison.
        let width = self.limbs.len();
        if width < SQR_SCHOOLBOOK_MIN_LIMBS {
            // Too narrow for the specialized kernel's three passes to repay
            // themselves. (Zero lands here too, and the multiplication
            // short-circuits it.)
            return self.mul_ref(self);
        }
        if width < KARATSUBA_THRESHOLD_LIMBS {
            return Self::sqr_schoolbook_ref(self);
        }
        if width >= SQR_KARATSUBA_MAX_LIMBS {
            // Wide enough that the multiplication ladder's Toom kernels
            // beat a Karatsuba square outright.
            return self.mul_ref(self);
        }
        self.sqr_karatsuba_ref()
    }

    /// Schoolbook squaring: `n(n+1)/2` limb products against the general
    /// multiplication's `n²`, by forming each distinct cross term once
    /// (*Handbook of Applied Cryptography*, Algorithm 14.16).
    ///
    /// Three passes rather than the obvious one. The strict upper triangle
    /// `Σ_{i<j} aᵢaⱼB^{i+j}` accumulates first; doubling it then accounts
    /// for the lower triangle, which is its mirror image; and the diagonal
    /// `Σ aᵢ²B^{2i}` is added last. Doubling the accumulated sum once, as a
    /// single shift over the buffer, is what avoids the `2·aᵢ·aⱼ` term that
    /// would otherwise overflow a `u128` accumulator and force a wider
    /// carry discipline.
    ///
    /// The doubling cannot overflow the buffer: twice the strict upper
    /// triangle is at most the whole square, and `a < B^n` gives
    /// `a² < B^{2n}`, the buffer's width. The debug assertion records that.
    fn sqr_schoolbook_ref(value: &Self) -> Self {
        let n = value.limbs.len();
        let mut out = vec![0u64; 2 * n];

        // Pass one: the strict upper triangle, each cross term once.
        for i in 0..n {
            let a_i = u128::from(value.limbs[i]);
            if a_i == 0 {
                continue;
            }
            let mut carry = 0u128;
            for j in (i + 1)..n {
                let idx = i + j;
                let acc = u128::from(out[idx]) + a_i * u128::from(value.limbs[j]) + carry;
                out[idx] = low_u64(acc);
                carry = acc >> 64;
            }
            let mut idx = i + n;
            while carry != 0 {
                let acc = u128::from(out[idx]) + carry;
                out[idx] = low_u64(acc);
                carry = acc >> 64;
                idx += 1;
            }
        }

        // Pass two: double, accounting for the lower triangle.
        let mut carry = 0u64;
        for limb in &mut out {
            let next = *limb >> 63;
            *limb = (*limb << 1) | carry;
            carry = next;
        }
        debug_assert!(
            carry == 0,
            "twice the cross terms is at most the square, which the buffer holds"
        );

        // Pass three: the diagonal. A term at `2i` occupies two limbs, and
        // its carry lands on `2i + 2` — which is the next iteration's own
        // position, so the ripple is carried in the loop variable rather
        // than re-walked.
        let mut carry = 0u128;
        for i in 0..n {
            let a_i = u128::from(value.limbs[i]);
            let acc = u128::from(out[2 * i]) + a_i * a_i + carry;
            out[2 * i] = low_u64(acc);
            let acc = u128::from(out[2 * i + 1]) + (acc >> 64);
            out[2 * i + 1] = low_u64(acc);
            carry = acc >> 64;
        }
        debug_assert!(carry == 0, "the square fits the buffer");

        let mut result = Self { limbs: out };
        result.normalize();
        result
    }

    /// Karatsuba squaring: the same split as [`Self::mul_karatsuba_ref`],
    /// but every sub-product is itself a square, so the three recursive
    /// calls are squarings and the middle term needs no separate operand.
    /// Writing `a = a₁B + a₀` for `B = 2^{64·split}`,
    ///
    /// ```text
    /// a² = a₁²B² + ((a₀+a₁)² − a₀² − a₁²)·B + a₀².
    /// ```
    ///
    /// The subtractions cannot underflow: `(a₀+a₁)²` dominates both squares
    /// removed from it, their cross term being non-negative.
    fn sqr_karatsuba_ref(&self) -> Self {
        // Unreachable from `square_ref`, which only routes widths of at
        // least `KARATSUBA_THRESHOLD_LIMBS` here, so `split >= 16`; kept
        // because the function is meaningful on its own terms and a zero
        // split would recurse forever.
        let split = self.limbs.len() / 2;
        if split == 0 {
            return Self::sqr_schoolbook_ref(self);
        }
        let (low, high) = self.split_at_limb(split);
        // Unlike the multiplication's split, which halves the *longer* of
        // two operands and so can leave the shorter one's high half empty,
        // this halves the operand's own width: `split = len/2 <= len − 1`
        // for every `len >= 2`, so the high half retains `limbs[len − 1]`,
        // which normalization guarantees is non-zero. No bail-out is
        // reachable here.
        debug_assert!(
            !high.is_zero(),
            "a normalized operand split at half its own width has a non-zero high half"
        );

        let z0 = low.square_ref();
        let z2 = high.square_ref();
        let sum = low.add_ref(&high);
        let mut z1 = sum.square_ref();
        z1.sub_assign_ref(&z0);
        z1.sub_assign_ref(&z2);

        let mut out = z0;
        z1.shl_bits(split * 64);
        out.add_assign_ref(&z1);
        let mut z2_shifted = z2;
        z2_shifted.shl_bits(split * 128);
        out.add_assign_ref(&z2_shifted);
        out
    }

    /// Split into low `[0, split)` and high `[split, len)` limb halves, each
    /// normalized so the recursive multiplications see canonical operands. A
    /// `split` at or above the width yields the whole value and zero.
    fn split_at_limb(&self, split: usize) -> (Self, Self) {
        let low_end = split.min(self.limbs.len());
        let mut low = Self {
            limbs: self.limbs[..low_end].to_vec(),
        };
        low.normalize();

        if split >= self.limbs.len() {
            return (low, Self::zero());
        }

        let mut high = Self {
            limbs: self.limbs[split..].to_vec(),
        };
        high.normalize();
        (low, high)
    }

    /// Both operands past the crossover, and within `KARATSUBA_MAX_IMBALANCE`
    /// of each other in length. The ratio bound is strict because it is also
    /// the kernel's structural floor: the split below is taken at half the
    /// *longer* operand, so at `long = 2·short` exactly the shorter operand
    /// fits entirely below the split, its high half is empty, and the kernel
    /// would reject the pair back to schoolbook. That shape goes to
    /// `mul_unbalanced_ref` above its threshold and to schoolbook below it.
    fn should_use_karatsuba(lhs: &Self, rhs: &Self) -> bool {
        let short = lhs.limbs.len().min(rhs.limbs.len());
        let long = lhs.limbs.len().max(rhs.limbs.len());
        short >= KARATSUBA_THRESHOLD_LIMBS && long < short * KARATSUBA_MAX_IMBALANCE
    }

    /// Unbalanced admission: the shorter operand alone is past
    /// `UNBALANCED_THRESHOLD_LIMBS`, but the pair is too lopsided for any
    /// balanced kernel (`long ≥ 2·short`). The threshold is this kernel's
    /// own measured crossover, not Karatsuba's: below it the block
    /// decomposition's per-block overhead loses to one flat schoolbook
    /// pass, even though each block is nominally past the Karatsuba
    /// crossover.
    fn should_use_unbalanced(lhs: &Self, rhs: &Self) -> bool {
        let short = lhs.limbs.len().min(rhs.limbs.len());
        let long = lhs.limbs.len().max(rhs.limbs.len());
        short >= UNBALANCED_THRESHOLD_LIMBS && long >= short * KARATSUBA_MAX_IMBALANCE
    }

    /// Unbalanced multiplication by block decomposition: cut the longer
    /// operand into base-`B = 2^{64k}` digits of the shorter one's length
    /// `k`, so that `long · short = Σᵢ digitᵢ·short·Bⁱ` — a sum of balanced
    /// `k × k` products, each accumulated into the output at its limb
    /// offset. Each digit product re-enters [`Self::mul_ref`] and lands on
    /// a balanced sub-quadratic kernel; a long × short product previously
    /// failed every balanced ratio test and ran schoolbook at full width.
    ///
    /// The accumulation is in place — `add_into_at` adds each product into
    /// the preallocated output window `[i·k, i·k + len)` with carries
    /// rippling upward — because the obvious recomposition (shift each
    /// product by `i·k` limbs, then add full-width) copies
    /// `Σᵢ i·k ≈ long²/(2k)` limbs and is quadratic in the *long* length,
    /// which measurement showed losing to flat schoolbook at every shape.
    /// An all-zero digit (a run of zero limbs in the longer operand)
    /// contributes nothing and is skipped.
    fn mul_unbalanced_ref(&self, other: &Self) -> Self {
        let (long, short) = if self.limbs.len() >= other.limbs.len() {
            (self, other)
        } else {
            (other, self)
        };
        let k = short.limbs.len();
        let mut out = vec![0u64; long.limbs.len() + k];
        for (i, digit_limbs) in long.limbs.chunks(k).enumerate() {
            let mut digit = Self {
                limbs: digit_limbs.to_vec(),
            };
            digit.normalize();
            if digit.is_zero() {
                continue;
            }
            let part = digit.mul_ref(short);
            // The window fits: this digit spans limbs [i·k, i·k + d) of
            // `long` with d = digit_limbs.len(), so the product has at most
            // d + k limbs and i·k + d + k ≤ long.len() + k = out.len().
            Self::add_into_at(&mut out, part.limbs(), i * k);
        }
        Self::from_limbs(out)
    }

    /// `acc += addend · β^offset`, in place over a raw limb buffer: the
    /// recomposition primitive of [`Self::mul_unbalanced_ref`]. The addend
    /// is added limb-wise into `acc[offset..]` and the final carry ripples
    /// upward; the caller guarantees the true sum fits in `acc`, which is
    /// what bounds the ripple (the running total never reaches
    /// `β^acc.len()`, so the carry dies before the buffer ends).
    fn add_into_at(acc: &mut [u64], addend: &[u64], offset: usize) {
        let mut carry = 0u64;
        for (j, &limb) in addend.iter().enumerate() {
            let (sum, c1) = acc[offset + j].overflowing_add(limb);
            let (sum, c2) = sum.overflowing_add(carry);
            acc[offset + j] = sum;
            carry = u64::from(c1) + u64::from(c2);
        }
        let mut index = offset + addend.len();
        while carry > 0 {
            let (sum, c) = acc[index].overflowing_add(carry);
            acc[index] = sum;
            carry = u64::from(c);
            index += 1;
        }
    }

    /// Karatsuba multiplication (Karatsuba & Ofman 1963; Knuth, *TAOCP*
    /// vol. 2, §4.3.3).
    ///
    /// Writing `a = a1·B + a0` and `b = b1·B + b0` for `B = 2^{64·split}`,
    /// the product needs only three half-width multiplications instead of
    /// four, because the middle coefficient is recovered by subtraction:
    /// `z0 = a0·b0`, `z2 = a1·b1`, and
    /// `z1 = (a0+a1)(b0+b1) − z0 − z2 = a0·b1 + a1·b0`. Recomposition is
    /// `z2·B² + z1·B + z0`, and both shifts are limb-aligned. The three
    /// sub-products recurse through [`Self::mul_ref`], so a large operand
    /// re-enters the dispatch and may take a different kernel on the way down.
    ///
    /// The subtractions cannot underflow: `z1`'s product dominates both terms
    /// removed from it. An empty high half on either side (possible when the
    /// operands differ in length) leaves nothing to save, so those fall back
    /// to schoolbook.
    fn mul_karatsuba_ref(&self, other: &Self) -> Self {
        let split = self.limbs.len().max(other.limbs.len()) / 2;
        if split == 0 {
            return Self::mul_schoolbook_ref(self, other);
        }

        let (a0, a1) = self.split_at_limb(split);
        let (b0, b1) = other.split_at_limb(split);
        if a1.is_zero() || b1.is_zero() {
            return Self::mul_schoolbook_ref(self, other);
        }

        let z0 = a0.mul_ref(&b0);
        let z2 = a1.mul_ref(&b1);

        let a_sum = a0.add_ref(&a1);
        let b_sum = b0.add_ref(&b1);
        let mut z1 = a_sum.mul_ref(&b_sum);
        z1.sub_assign_ref(&z0);
        z1.sub_assign_ref(&z2);

        let mut out = z0;
        z1.shl_bits(split * 64);
        out.add_assign_ref(&z1);

        let mut z2_shifted = z2;
        z2_shifted.shl_bits(split * 128);
        out.add_assign_ref(&z2_shifted);
        out
    }

    /// Toom-3 admission: both operands past `TOOM3_THRESHOLD_LIMBS` and
    /// within 1.5× of each other in length.
    fn should_use_toom3(lhs: &Self, rhs: &Self) -> bool {
        let short = lhs.limbs.len().min(rhs.limbs.len());
        let long = lhs.limbs.len().max(rhs.limbs.len());
        // Both operands large, and close enough in length that all three parts
        // of each carry weight (a lopsided split wastes the five-way machinery).
        short >= TOOM3_THRESHOLD_LIMBS && long <= short + short / 2
    }

    /// Split into three little-endian chunks of `k` limbs — low `[0, k)`, mid
    /// `[k, 2k)`, high `[2k, len)`. The high chunk holds whatever remains and
    /// may be shorter than `k`, or empty.
    fn split3_at(&self, k: usize) -> (Self, Self, Self) {
        let n = self.limbs.len();
        let piece = |lo: usize, hi: usize| {
            if lo >= n {
                Self::zero()
            } else {
                let mut part = Self {
                    limbs: self.limbs[lo..hi.min(n)].to_vec(),
                };
                part.normalize();
                part
            }
        };
        (piece(0, k), piece(k, 2 * k), piece(2 * k, n))
    }

    /// Toom–Cook three-way multiplication (Knuth, *TAOCP* vol. 2, §4.3.3, the
    /// generalization of Karatsuba; interpolation sequence after Bodrato,
    /// *Towards Optimal Toom–Cook Multiplication…*, WAIFI 2007).
    ///
    /// Split both operands into three base-`B = 2^{64k}` digits, evaluate each
    /// as a degree-2 polynomial at `{0, 1, -1, 2, ∞}`, multiply the five pairs
    /// (recursively — this is where the sub-quadratic saving lives: five
    /// products of a third the size, versus schoolbook's nine or Karatsuba's
    /// three of a half), then interpolate the five product digits and
    /// recompose. Evaluation and interpolation run in signed arithmetic; the
    /// interpolation's divisions by 2, 3, 6 are exact.
    fn mul_toom3_ref(&self, other: &Self) -> Self {
        let n = self.limbs.len().max(other.limbs.len());
        let k = n.div_ceil(3);
        let (a0, a1, a2) = self.split3_at(k);
        let (b0, b1, b2) = other.split3_at(k);

        // Evaluate a and b at 0, 1, -1, 2, ∞. The value at -1 can go negative,
        // so those points live in signed arithmetic.
        let eval = |c0: &Self, c1: &Self, c2: &Self| {
            let even = c0.add_ref(c2); // c0 + c2
            let at_1 = even.add_ref(c1); // c(1)
            let at_m1 = BigInt::from_biguint(even).sub_ref(&BigInt::from_biguint(c1.clone()));
            let mut twice_c1 = c1.clone();
            twice_c1.shl_bits(1);
            let mut four_c2 = c2.clone();
            four_c2.shl_bits(2);
            let at_2 = c0.add_ref(&twice_c1).add_ref(&four_c2); // c(2)
            (at_1, at_m1, at_2)
        };
        let (a_1, a_m1, a_2) = eval(&a0, &a1, &a2);
        let (b_1, b_m1, b_2) = eval(&b0, &b1, &b2);

        // Pointwise products (each a recursive multiplication).
        let v0 = BigInt::from_biguint(a0.mul_ref(&b0)); // W(0)
        let v_inf = BigInt::from_biguint(a2.mul_ref(&b2)); // W(∞)
        let v1 = BigInt::from_biguint(a_1.mul_ref(&b_1)); // W(1)
        let vm1 = bigint_mul(&a_m1, &b_m1); // W(-1)
        let v2 = BigInt::from_biguint(a_2.mul_ref(&b_2)); // W(2)

        // Interpolate the product digits c0..c4. Derivation: with
        // W(x) = Σ cᵢ xⁱ, the points give c0 = W(0), c4 = W(∞), and
        //   s = (W(1)+W(-1))/2 = c0 + c2 + c4,   t = (W(1)-W(-1))/2 = c1 + c3,
        //   u = (W(2) - c0 - 4c2 - 16c4)/2 = c1 + 4c3,
        // whence c2 = s - c0 - c4, c3 = (u - t)/3, c1 = t - c3. Every quotient
        // is exact.
        let c0 = v0;
        let c4 = v_inf;
        let s = bigint_div_exact(&v1.add_ref(&vm1), 2);
        let t = bigint_div_exact(&v1.sub_ref(&vm1), 2);
        let c2 = s.sub_ref(&c0).sub_ref(&c4);
        let four_c2 = bigint_shl_exact(&c2, 2);
        let sixteen_c4 = bigint_shl_exact(&c4, 4);
        let u = bigint_div_exact(&v2.sub_ref(&c0).sub_ref(&four_c2).sub_ref(&sixteen_c4), 2);
        let c3 = bigint_div_exact(&u.sub_ref(&t), 3);
        let c1 = t.sub_ref(&c3);

        // Recompose Σ cᵢ·B^{ik} by Horner. The product's digits are all
        // non-negative, so this returns to unsigned.
        let shift = 64 * k;
        let mut acc = BigUint::zero();
        for coefficient in [&c4, &c3, &c2, &c1, &c0] {
            debug_assert!(
                coefficient.sign() != Sign::Negative,
                "Toom-3 product digits are non-negative"
            );
            acc.shl_bits(shift);
            acc.add_assign_ref(coefficient.magnitude());
        }
        acc
    }

    /// Toom-4 admission, on the same shape as [`Self::should_use_toom3`]:
    /// both operands past `TOOM4_THRESHOLD_LIMBS` and within 1.5× in length.
    fn should_use_toom4(lhs: &Self, rhs: &Self) -> bool {
        let short = lhs.limbs.len().min(rhs.limbs.len());
        let long = lhs.limbs.len().max(rhs.limbs.len());
        short >= TOOM4_THRESHOLD_LIMBS && long <= short + short / 2
    }

    /// Split into four little-endian chunks of `k` limbs; the top chunk holds
    /// whatever remains and may be shorter than `k`, or empty.
    fn split4_at(&self, k: usize) -> (Self, Self, Self, Self) {
        let n = self.limbs.len();
        let piece = |lo: usize, hi: usize| {
            if lo >= n {
                Self::zero()
            } else {
                let mut part = Self {
                    limbs: self.limbs[lo..hi.min(n)].to_vec(),
                };
                part.normalize();
                part
            }
        };
        (
            piece(0, k),
            piece(k, 2 * k),
            piece(2 * k, 3 * k),
            piece(3 * k, n),
        )
    }

    /// Toom–Cook four-way multiplication: split into four base-`B = 2^{64k}`
    /// digits (degree-3 polynomials), evaluate at `{0, 1, -1, 2, -2, 3, ∞}`,
    /// multiply the seven pairs recursively, then interpolate the seven product
    /// digits. Sub-quadratic exponent `log 7 / log 4 ≈ 1.404`, below Toom-3's
    /// `1.465`, so it overtakes Toom-3 once the seven-point interpolation
    /// (divisions by 2, 3, 4, 5, 8, 12, all exact) is amortized. Same
    /// interpolation shape as Toom-3, one order up.
    fn mul_toom4_ref(&self, other: &Self) -> Self {
        let n = self.limbs.len().max(other.limbs.len());
        let k = n.div_ceil(4);
        let (a0, a1, a2, a3) = self.split4_at(k);
        let (b0, b1, b2, b3) = other.split4_at(k);

        // Evaluate a degree-3 digit polynomial at 1, -1, 2, -2, 3 (the ∞ and 0
        // points are the top and bottom digits themselves). Points -1, -2 can
        // go negative, so those live in signed arithmetic.
        let eval4 = |c0: &Self, c1: &Self, c2: &Self, c3: &Self| {
            let mut two_c1 = c1.clone();
            two_c1.shl_bits(1);
            let mut four_c2 = c2.clone();
            four_c2.shl_bits(2);
            let mut eight_c3 = c3.clone();
            eight_c3.shl_bits(3);

            let at_1 = c0.add_ref(c1).add_ref(c2).add_ref(c3); // c(1)
            let even = c0.add_ref(c2); // c0 + c2
            let odd = c1.add_ref(c3); // c1 + c3
            let at_m1 = BigInt::from_biguint(even).sub_ref(&BigInt::from_biguint(odd)); // c(-1)
            let at_2 = c0.add_ref(&two_c1).add_ref(&four_c2).add_ref(&eight_c3); // c(2)
            let even2 = c0.add_ref(&four_c2); // c0 + 4c2
            let odd2 = two_c1.add_ref(&eight_c3); // 2c1 + 8c3
            let at_m2 = BigInt::from_biguint(even2).sub_ref(&BigInt::from_biguint(odd2)); // c(-2)

            // c(3) = c0 + 3c1 + 9c2 + 27c3, by Horner at x = 3.
            let three = BigUint::from_u64(3);
            let mut at_3 = c3.mul_ref(&three);
            at_3.add_assign_ref(c2);
            at_3 = at_3.mul_ref(&three);
            at_3.add_assign_ref(c1);
            at_3 = at_3.mul_ref(&three);
            at_3.add_assign_ref(c0);
            (at_1, at_m1, at_2, at_m2, at_3)
        };
        let (a_1, a_m1, a_2, a_m2, a_3) = eval4(&a0, &a1, &a2, &a3);
        let (b_1, b_m1, b_2, b_m2, b_3) = eval4(&b0, &b1, &b2, &b3);

        // Seven pointwise products (each a recursive multiplication).
        let w0 = BigInt::from_biguint(a0.mul_ref(&b0)); // W(0)
        let w1 = BigInt::from_biguint(a_1.mul_ref(&b_1)); // W(1)
        let w2 = bigint_mul(&a_m1, &b_m1); // W(-1)
        let w3 = BigInt::from_biguint(a_2.mul_ref(&b_2)); // W(2)
        let w4 = bigint_mul(&a_m2, &b_m2); // W(-2)
        let w5 = BigInt::from_biguint(a_3.mul_ref(&b_3)); // W(3)
        let w6 = BigInt::from_biguint(a3.mul_ref(&b3)); // W(∞)

        // Powers of two shift; the odd weights (9, 81, 729, 5, 3) go through
        // the general multiply.
        let scale = |x: &BigInt, m: u64| {
            if m.is_power_of_two() {
                bigint_shl_exact(x, m.trailing_zeros() as usize)
            } else {
                x.mul_biguint_ref(&BigUint::from_u64(m))
            }
        };
        let c0 = w0;
        let c6 = w6;

        // Even coefficients c2, c4 from the symmetric sums.
        let e1 = bigint_div_exact(&w1.add_ref(&w2), 2); // c2 + c4 + c0 + c6
        let e2 = bigint_div_exact(&w3.add_ref(&w4), 2); // 4c2 + 16c4 + c0 + 64c6
        let sum24 = e1.sub_ref(&c0).sub_ref(&c6); // c2 + c4
        let weighted24 = e2.sub_ref(&c0).sub_ref(&scale(&c6, 64)); // 4c2 + 16c4
        let c4 = bigint_div_exact(&weighted24.sub_ref(&scale(&sum24, 4)), 12);
        let c2 = sum24.sub_ref(&c4);

        // Odd coefficients c1, c3, c5 from the antisymmetric sums and W(3).
        let o1 = bigint_div_exact(&w1.sub_ref(&w2), 2); // c1 + c3 + c5
        let o2 = bigint_div_exact(&w3.sub_ref(&w4), 4); // c1 + 4c3 + 16c5
        let o3 = bigint_div_exact(
            &w5.sub_ref(&c0)
                .sub_ref(&scale(&c2, 9))
                .sub_ref(&scale(&c4, 81))
                .sub_ref(&scale(&c6, 729)),
            3,
        ); // c1 + 9c3 + 81c5
        let p = bigint_div_exact(&o2.sub_ref(&o1), 3); // c3 + 5c5
        let q = bigint_div_exact(&o3.sub_ref(&o1), 8); // c3 + 10c5
        let c5 = bigint_div_exact(&q.sub_ref(&p), 5);
        let c3 = p.sub_ref(&scale(&c5, 5));
        let c1 = o1.sub_ref(&c3).sub_ref(&c5);

        // Recompose Σ cᵢ·B^{ik}; the product digits are all non-negative.
        let shift = 64 * k;
        let mut acc = BigUint::zero();
        for coefficient in [&c6, &c5, &c4, &c3, &c2, &c1, &c0] {
            debug_assert!(
                coefficient.sign() != Sign::Negative,
                "Toom-4 product digits are non-negative"
            );
            acc.shl_bits(shift);
            acc.add_assign_ref(coefficient.magnitude());
        }
        acc
    }

    /// The low `limit` limbs of `lhs · rhs` — the product modulo
    /// `2^{64·limit}`, computed without forming the rest of it.
    ///
    /// Every partial product lands at a fixed position, so one whose
    /// position is at or above `limit` cannot influence any limb below it
    /// and is simply never computed; the same is true of a carry walking
    /// off the top of the window. That makes the result *exact* rather than
    /// approximate, and costs about half the limb products of the full
    /// multiplication when `limit` is half the product's width.
    ///
    /// This is the half-product of *Handbook of Applied Cryptography*, Note
    /// 14.45(ii), which observes that Barrett reduction's second
    /// multiplication needs only the low `k+1` limbs of `q̂·n`.
    fn mul_low_ref(lhs: &Self, rhs: &Self, limit: usize) -> Self {
        let mut out = vec![0u64; limit];
        for (i, &lhs_limb) in lhs.limbs.iter().enumerate() {
            if i >= limit {
                break;
            }
            let mut carry = 0u128;
            for (j, &rhs_limb) in rhs.limbs.iter().enumerate() {
                let idx = i + j;
                if idx >= limit {
                    break;
                }
                let acc =
                    u128::from(out[idx]) + u128::from(lhs_limb) * u128::from(rhs_limb) + carry;
                out[idx] = low_u64(acc);
                carry = acc >> 64;
            }
            // A carry leaving the window belongs to a limb the caller
            // discards, so it is dropped rather than propagated.
            let mut idx = i + rhs.limbs.len();
            while carry != 0 && idx < limit {
                let acc = u128::from(out[idx]) + carry;
                out[idx] = low_u64(acc);
                carry = acc >> 64;
                idx += 1;
            }
        }
        let mut result = Self { limbs: out };
        result.normalize();
        result
    }

    /// Classic operand-scanning long multiplication (Knuth, *TAOCP* vol. 2,
    /// §4.3.1, Algorithm M): for each limb of `lhs`, multiply-accumulate it
    /// across `rhs` into the running product with a `u128` carry.
    fn mul_schoolbook_ref(lhs: &Self, rhs: &Self) -> Self {
        let mut out = vec![0u64; lhs.limbs.len() + rhs.limbs.len()];
        for (i, &lhs_limb) in lhs.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &rhs_limb) in rhs.limbs.iter().enumerate() {
                let idx = i + j;
                let acc =
                    u128::from(out[idx]) + u128::from(lhs_limb) * u128::from(rhs_limb) + carry;
                out[idx] = low_u64(acc);
                carry = acc >> 64;
            }

            let mut idx = i + rhs.limbs.len();
            while carry != 0 {
                let acc = u128::from(out[idx]) + carry;
                out[idx] = low_u64(acc);
                carry = acc >> 64;
                idx += 1;
            }
        }

        let mut result = Self { limbs: out };
        // A normalized non-zero multiplicand and multiplier cannot produce a
        // spuriously zero high limb except through the carry chain itself, so
        // one post-pass normalization is enough.
        result.normalize();
        result
    }

    /// Double the value: one bit carried from limb to limb, low to high, with
    /// a new top limb pushed when it escapes. The single-bit case of
    /// [`Self::shl_bits`] earns its own loop because it needs neither a
    /// whole-limb move nor a `normalize` — a doubling cannot zero the top
    /// limb.
    pub fn shl1(&mut self) {
        if self.is_zero() {
            return;
        }

        let mut carry = 0u64;
        for limb in &mut self.limbs {
            let next = *limb >> 63;
            *limb = (*limb << 1) | carry;
            carry = next;
        }

        if carry != 0 {
            self.limbs.push(carry);
        }
        // A left shift on an already-normalized value cannot introduce a
        // leading zero limb, so no normalize() pass is required here.
    }

    /// Halve the value, discarding the low bit — `⌊self/2⌋`: one bit carried
    /// from limb to limb, high to low, then a `normalize`, because unlike
    /// doubling a halving can empty the top limb. This is the averaging step
    /// of [`Self::sqrt_rem`]'s Newton iteration, where a division by two
    /// would otherwise cost a full Algorithm D pass.
    pub fn shr1(&mut self) {
        if self.is_zero() {
            return;
        }

        let mut carry = 0u64;
        for limb in self.limbs.iter_mut().rev() {
            let next = (*limb & 1) << 63;
            *limb = (*limb >> 1) | carry;
            carry = next;
        }

        self.normalize();
    }

    /// XOR another bigint into `self` in place (GF(2^m) field addition).
    ///
    /// Extends `self.limbs` with zeros if shorter than `other.limbs`, then
    /// XORs each corresponding limb pair.  The result is normalized to strip
    /// any leading zero limbs produced by XOR cancellation.
    pub fn bitxor_assign(&mut self, other: &BigUint) {
        if self.limbs.len() < other.limbs.len() {
            self.limbs.resize(other.limbs.len(), 0);
        }
        for (s, &o) in self.limbs.iter_mut().zip(other.limbs.iter()) {
            *s ^= o;
        }
        self.normalize();
    }

    /// Left-shift by `n` bits — multiplication by `2^n`.
    ///
    /// Split into a whole-limb move of `n / 64` positions (zero limbs
    /// prepended at the low end) and one pass shifting each limb by the
    /// remaining `n % 64` bits with the displaced high bits carried into the
    /// next limb. The split is what keeps every shift amount below 64:
    /// shifting a `u64` by 64 or more is undefined, and `64 - bit_shifts`
    /// appears in the carry expression.
    pub fn shl_bits(&mut self, n: usize) {
        if self.is_zero() || n == 0 {
            return;
        }
        let limb_shifts = n / 64;
        let bit_shifts = n % 64;
        // Full-limb shift: prepend zeros at the low (index 0) end. The
        // displaced buffer is dropped unscrubbed, deliberately: this sits on
        // the recomposition path of every Karatsuba and Toom multiplication,
        // and scrubbing it here was measured at 3-5% on large balanced
        // products (an extra full write pass per shift). That keeps it
        // within the crate's stated scope — buffers freed on reallocation
        // are not wiped.
        if limb_shifts > 0 {
            let mut new_limbs = vec![0u64; limb_shifts];
            new_limbs.extend_from_slice(&self.limbs);
            self.limbs = new_limbs;
        }
        // Remaining bit-level shift (0 < bit_shifts < 64, so 64 - bit_shifts is safe).
        if bit_shifts > 0 {
            let mut carry = 0u64;
            for limb in &mut self.limbs {
                let next_carry = *limb >> (64 - bit_shifts);
                *limb = (*limb << bit_shifts) | carry;
                carry = next_carry;
            }
            if carry != 0 {
                self.limbs.push(carry);
            }
        }
        // A left-shift on a normalized value cannot introduce a leading zero
        // limb, so no normalize() pass is needed here.
    }

    /// Right-shift by `n` bits, discarding the shifted-out low bits.
    ///
    /// The mirror of [`Self::shl_bits`]: `n / 64` whole-limb drops plus up to
    /// 63 bit positions within limbs, avoiding undefined behaviour from
    /// shifting a `u64` by 64 or more. Equivalent to dividing by `2^n`.
    pub fn shr_bits(&mut self, n: usize) {
        if self.is_zero() || n == 0 {
            return;
        }
        let limb_shifts = n / 64;
        let bit_shifts = (n % 64) as u32;

        if limb_shifts >= self.limbs.len() {
            // Everything shifts out. Wipe rather than truncate so the old
            // limbs do not linger beyond the vector's length. A volatile
            // scrub, not a plain `fill(0)`: the store is immediately followed
            // by `clear`, so an ordinary write to memory about to leave the
            // slice is exactly the dead store the optimizer may elide.
            crate::scrub::zeroize_slice(self.limbs.as_mut_slice());
            self.limbs.clear();
            return;
        }

        // Whole-limb shift: move the high limbs down, then wipe the vacated
        // top slots before truncating for the same reason as above.
        if limb_shifts > 0 {
            let kept = self.limbs.len() - limb_shifts;
            self.limbs.copy_within(limb_shifts.., 0);
            crate::scrub::zeroize_slice(&mut self.limbs[kept..]);
            self.limbs.truncate(kept);
        }

        // Remaining bit-level shift (0 < bit_shifts < 64, so 64 - bit_shifts
        // is a defined shift amount).
        if bit_shifts > 0 {
            let mut carry = 0u64;
            for limb in self.limbs.iter_mut().rev() {
                let next_carry = *limb << (64 - bit_shifts);
                *limb = (*limb >> bit_shifts) | carry;
                carry = next_carry;
            }
        }

        self.normalize();
    }

    /// The remainder `self mod modulus`, in `[0, modulus)`: [`Self::div_rem`]
    /// with the quotient discarded. Algorithm D produces both halves in one
    /// pass, so a caller that needs the quotient as well should take it from
    /// `div_rem` rather than calling this and dividing a second time.
    ///
    /// # Panics
    ///
    /// Panics if `modulus == 0`.
    #[must_use]
    pub fn modulo(&self, modulus: &Self) -> Self {
        let (_, remainder) = self.div_rem(modulus);
        remainder
    }

    /// Divide by a machine word, returning `(quotient, remainder)` in one
    /// pass — the word-sized companion to [`Self::div_rem`], without the
    /// heap-allocated divisor a `BigUint` division would need. This is the
    /// shape a trial-division inner loop wants: recover the quotient and
    /// the remainder together, per word-sized prime, per candidate.
    ///
    /// # Panics
    ///
    /// Panics if `divisor == 0`.
    #[must_use]
    pub fn div_rem_u64(&self, divisor: u64) -> (Self, u64) {
        assert!(divisor != 0, "division by zero");
        Self::div_rem_limb(&self.limbs, divisor)
    }

    /// The value as a `u64` when it fits, `None` otherwise — the checked
    /// narrowing that [`Self::low_u128`] leaves to the caller. Use this
    /// where a value is *expected* to fit a word and the expectation
    /// should be verified rather than assumed.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        match self.limbs.as_slice() {
            [] => Some(0),
            [single] => Some(*single),
            _ => None,
        }
    }

    /// Compute the remainder modulo a machine word.
    ///
    /// # Panics
    ///
    /// Panics if `modulus == 0`.
    #[must_use]
    pub fn rem_u64(&self, modulus: u64) -> u64 {
        assert!(modulus != 0, "division by zero");
        if self.is_zero() {
            return 0;
        }

        let mut remainder = 0u128;
        // Horner's method in base `2^64`: carry the remainder of the already
        // processed high limbs, then append the next limb as the next base
        // digit before reducing again.
        for &limb in self.limbs.iter().rev() {
            let acc = (remainder << 64) | u128::from(limb);
            remainder = acc % u128::from(modulus);
        }

        u64::try_from(remainder).expect("remainder modulo u64 fits into u64")
    }

    /// Compute `(lhs * rhs) mod modulus`.
    ///
    /// Multiply, then reduce once. This used to build a throwaway
    /// [`MontgomeryCtx`] for odd moduli and fall back to a double-and-add
    /// reducer for even ones, both to dodge a division. With Algorithm D doing
    /// the reduction that trade no longer pays: a Montgomery context costs a
    /// division to construct (`R² mod n`) and then three Montgomery multiplies
    /// plus a reduction to encode both operands, multiply, and decode, where
    /// this costs one multiply and one division — and it needs no odd-modulus
    /// special case.
    ///
    /// Callers that perform many multiplications under one modulus should still
    /// build a [`MontgomeryCtx`] once and reuse it; this is the one-shot path.
    ///
    /// # Panics
    ///
    /// Panics if `modulus == 0`.
    #[must_use]
    pub fn mod_mul(lhs: &Self, rhs: &Self, modulus: &Self) -> Self {
        assert!(!modulus.is_zero(), "modulus must be non-zero");
        if modulus.is_one() {
            return Self::zero();
        }
        lhs.mul_ref(rhs).modulo(modulus)
    }

    /// One-shot modular addition, on [`Self::mod_mul`]'s contract: any
    /// operands, non-zero modulus (panic otherwise). Reduced operands take
    /// one compare-and-correct; the domain contexts' `add` operations are
    /// the reduced-only fast paths of this.
    ///
    /// # Panics
    ///
    /// Panics if `modulus == 0`.
    #[must_use]
    pub fn mod_add(lhs: &Self, rhs: &Self, modulus: &Self) -> Self {
        assert!(!modulus.is_zero(), "modulus must be non-zero");
        let lhs = if lhs < modulus {
            lhs.clone()
        } else {
            lhs.modulo(modulus)
        };
        let rhs = if rhs < modulus {
            rhs.clone()
        } else {
            rhs.modulo(modulus)
        };
        let sum = lhs.add_ref(&rhs);
        if sum >= *modulus {
            sum.sub_ref(modulus)
        } else {
            sum
        }
    }

    /// One-shot modular subtraction, on the same contract as
    /// [`Self::mod_add`]; the wrap adds the modulus back.
    ///
    /// # Panics
    ///
    /// Panics if `modulus == 0`.
    #[must_use]
    pub fn mod_sub(lhs: &Self, rhs: &Self, modulus: &Self) -> Self {
        assert!(!modulus.is_zero(), "modulus must be non-zero");
        let lhs = if lhs < modulus {
            lhs.clone()
        } else {
            lhs.modulo(modulus)
        };
        let rhs = if rhs < modulus {
            rhs.clone()
        } else {
            rhs.modulo(modulus)
        };
        if lhs >= rhs {
            lhs.sub_ref(&rhs)
        } else {
            modulus.add_ref(&lhs).sub_ref(&rhs)
        }
    }

    /// Return `(quotient, remainder)` for Euclidean division, with the
    /// remainder in `[0, divisor)`.
    ///
    /// Dispatches on the divisor's width: a single-limb divisor takes a
    /// base-2⁶⁴ Horner division (one pass, no quotient estimation needed),
    /// while a multi-limb divisor uses Knuth's Algorithm D (*TAOCP* vol. 2,
    /// §4.3.1) — operand normalization, the two-limb quotient estimate, and the
    /// occasional add-back correction. A dividend smaller than the divisor
    /// returns `(0, self)` without either path.
    ///
    /// # Panics
    ///
    /// Panics if `divisor == 0`.
    #[must_use]
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        assert!(!divisor.is_zero(), "division by zero");
        if self.cmp(divisor) == Ordering::Less {
            return (Self::zero(), self.clone());
        }

        // One limb of quotient at a time, not one bit: both paths below produce
        // 64 quotient bits per pass over the divisor.
        if divisor.limbs.len() == 1 {
            let (quotient, remainder) = Self::div_rem_limb(&self.limbs, divisor.limbs[0]);
            return (quotient, Self::from_u64(remainder));
        }

        Self::div_rem_knuth(&self.limbs, &divisor.limbs)
    }

    /// Divide by a single limb by Horner's method in base `2^64`, the same
    /// recurrence [`Self::rem_u64`] uses, keeping the quotient digits.
    fn div_rem_limb(dividend: &[u64], divisor: u64) -> (Self, u64) {
        let divisor = u128::from(divisor);
        let mut quotient = vec![0u64; dividend.len()];
        let mut remainder = 0u128;
        for (slot, &limb) in quotient.iter_mut().zip(dividend.iter()).rev() {
            let acc = (remainder << 64) | u128::from(limb);
            *slot = low_u64(acc / divisor);
            remainder = acc % divisor;
        }

        let mut quotient = Self { limbs: quotient };
        quotient.normalize();
        (quotient, low_u64(remainder))
    }

    /// Knuth's Algorithm D — long division in base `b = 2^64`.
    ///
    /// Reference: Knuth, *TAOCP* vol. 2, §4.3.1, Algorithm D; the borrow and
    /// add-back mechanics follow Warren, *Hacker's Delight*, §9-2 (`divmnu`).
    /// Step labels D1–D8 in the comments are Knuth's.
    ///
    /// Requires `dividend >= divisor` and at least two divisor limbs; both
    /// slices are normalized (non-zero top limb). Costs
    /// `O(quotient_limbs * divisor_limbs)` limb operations, against
    /// `O(bits * limbs)` for the bit-serial long division it replaced: one
    /// pass over the divisor now yields 64 quotient bits instead of one.
    ///
    /// Like the rest of the crate this is variable-time: the quotient-digit
    /// corrections below are data-dependent.
    fn div_rem_knuth(dividend: &[u64], divisor: &[u64]) -> (Self, Self) {
        /// Knuth's `b`, the digit base.
        const BASE: u128 = 1u128 << 64;

        let n = divisor.len();
        debug_assert!(n >= 2, "single-limb divisors take the Horner path");
        debug_assert!(dividend.len() >= n, "caller screens dividend < divisor");
        let m = dividend.len() - n;

        // D1. Scale both operands so the divisor's top limb has its high bit
        // set (the quotient is unchanged; the remainder is scaled back in D8).
        // Normalization is what bounds the D3 estimate to at most two over the
        // true digit, so a single conditional add-back in D6 suffices.
        let shift = divisor[n - 1].leading_zeros();
        let divisor = shl_into(divisor, shift, n);
        // One limb of headroom: the shift can carry out, and the estimate step
        // reads `rem[j + n]` for the top window.
        let mut rem = shl_into(dividend, shift, dividend.len() + 1);
        let divisor_hi = u128::from(divisor[n - 1]);
        let divisor_next = u128::from(divisor[n - 2]);

        let mut quotient = vec![0u64; m + 1];

        // D2/D7. One quotient digit per pass, most significant first. The
        // window `rem[j..=j + n]` always holds less than `divisor * b`, so
        // each true digit fits in one limb.
        for j in (0..=m).rev() {
            // D3. Estimate the digit from the window's top two limbs:
            // `q_hat = numerator / divisor_hi`, remainder `r_hat` (Knuth's
            // q-hat and r-hat). Normalization guarantees `q_hat <= q + 2`.
            //
            // The loop's second test rules the estimate against the divisor's
            // *third*-from-top limb; each firing lowers `q_hat` by one, and
            // when it stops `q_hat <= q + 1` (TAOCP §4.3.1, exercise 20),
            // leaving at most the one overshoot D6 can repair. Skipping this
            // correction is not an option: for divisors like
            // `[v0, d, d, ...]` with `d >= b/2` the raw estimate reaches
            // `b + 1` — two over the true digit `b - 1` — which no single
            // add-back can fix.
            //
            // The `q_hat >= BASE` arm is Knuth's `min(q_hat, b - 1)` clamp.
            // Because `q_hat` stays in `u128` all the way into D4, the clamp
            // is provably redundant here — an estimate of `b` or `b + 1` is
            // always caught by the second test or repaired by D6 — but it is
            // kept both to match the algorithm as published and to skip a
            // predictably doomed full-width subtraction.
            //
            // Termination: each round adds `divisor_hi >= b/2` to `r_hat`, so
            // the `r_hat >= BASE` break bounds the loop at two corrections
            // beyond the clamp.
            let numerator = (u128::from(rem[j + n]) << 64) | u128::from(rem[j + n - 1]);
            let mut q_hat = numerator / divisor_hi;
            let mut r_hat = numerator % divisor_hi;
            while q_hat >= BASE || q_hat * divisor_next > (r_hat << 64) | u128::from(rem[j + n - 2])
            {
                q_hat -= 1;
                r_hat += divisor_hi;
                if r_hat >= BASE {
                    break;
                }
            }

            // D4. Subtract `q_hat * divisor` from the window. Each step biases
            // the difference by `BASE` so it stays unsigned; bit 64 of the
            // biased result is 1 exactly when no borrow was needed.
            let mut borrow = 0u128;
            let mut carry = 0u128;
            for i in 0..n {
                let product = q_hat * u128::from(divisor[i]) + carry;
                carry = product >> 64;
                let diff = BASE + u128::from(rem[i + j]) - u128::from(low_u64(product)) - borrow;
                rem[i + j] = low_u64(diff);
                borrow = 1 - (diff >> 64);
            }
            let diff = BASE + u128::from(rem[j + n]) - carry - borrow;
            rem[j + n] = low_u64(diff);

            // D5/D6. A borrow out of the top means `q_hat` was one too large
            // (probability about `2/b` on random input); add the divisor back
            // once. The carry out of the add-back cancels the borrow D4 left
            // in the top limb, restoring the invariant checked below.
            if diff >> 64 == 0 {
                q_hat -= 1;
                let mut carry = 0u128;
                for i in 0..n {
                    let sum = u128::from(rem[i + j]) + u128::from(divisor[i]) + carry;
                    rem[i + j] = low_u64(sum);
                    carry = sum >> 64;
                }
                rem[j + n] = rem[j + n].wrapping_add(low_u64(carry));
            }

            // After a correct step the remaining value fits below `b^n`, so
            // the window's top limb must be clean. Release builds never read
            // `rem[j + n]` again (the next window sits one limb lower), but
            // the store above keeps this invariant true and checkable.
            debug_assert!(rem[j + n] == 0, "quotient digit left residue");

            quotient[j] = low_u64(q_hat);
        }

        let mut quotient = Self { limbs: quotient };
        quotient.normalize();

        // D8. The remainder is the final window, still scaled by `2^shift`
        // from D1; the true remainder's shifted-out low bits are zero.
        debug_assert!(
            shift == 0 || rem[0].trailing_zeros() >= shift,
            "denormalized remainder must be a multiple of 2^shift"
        );
        let mut remainder = Self {
            limbs: shr_limbs(&rem[..n], shift),
        };
        remainder.normalize();

        (quotient, remainder)
    }

    /// Restore the canonical representation by popping zero limbs off the
    /// top:
    ///
    /// - zero has `limbs.is_empty()`
    /// - every non-zero value has a non-zero top limb
    ///
    /// Every path that can strand a zero above the significant limbs — a
    /// subtraction that cancels, a right shift, a slice copied out of a
    /// wider value, a product whose top limb did not carry — must end here,
    /// because `Eq`, `Ord`, [`Self::bits`] and the kernel dispatch all read
    /// the limb count as the value's width. `pop` only shortens the vector,
    /// so the capacity survives for reuse.
    fn normalize(&mut self) {
        while self.limbs.last().copied() == Some(0) {
            self.limbs.pop();
        }
    }

    /// The `BigUint`-facing wrapper around [`mont_mul`]: the kernels work on
    /// fixed-width limb slices, while a canonical `BigUint` is only as wide as
    /// its value, so this pads both operands to the modulus width and carves
    /// scratch, operand, and output windows out of one reusable workspace.
    /// The workspace is threaded through by the caller so a sequence of
    /// domain operations allocates once rather than per multiply.
    ///
    /// Operands must be reduced residues; an operand *wider* than the modulus
    /// panics in [`copy_padded`] rather than silently producing a wrong
    /// residue.
    fn montgomery_mul_odd_with_workspace(
        lhs: &Self,
        rhs: &Self,
        modulus: &Self,
        n0_inv: u64,
        workspace: &mut Vec<u64>,
    ) -> Self {
        debug_assert!(modulus.is_odd(), "Montgomery path requires an odd modulus");
        let width = modulus.limbs.len();
        debug_assert!(
            lhs < modulus && rhs < modulus,
            "Montgomery operands must be reduced residues"
        );

        // Layout: `[scratch 2w+1 | lhs w | rhs w | out w]`.
        let needed = mont_scratch_limbs(width) + 3 * width;
        if workspace.len() < needed {
            workspace.resize(needed, 0);
        }
        let (scratch, rest) = workspace.split_at_mut(mont_scratch_limbs(width));
        let (lhs_pad, rest) = rest.split_at_mut(width);
        let (rhs_pad, out) = rest.split_at_mut(width);
        copy_padded(lhs_pad, &lhs.limbs);
        copy_padded(rhs_pad, &rhs.limbs);

        mont_mul(
            &mut out[..width],
            lhs_pad,
            rhs_pad,
            &modulus.limbs,
            n0_inv,
            scratch,
        );

        let mut result = Self {
            limbs: out[..width].to_vec(),
        };
        result.normalize();
        result
    }

    /// Squaring companion to [`Self::montgomery_mul_odd_with_workspace`]:
    /// pads the single operand to the modulus width and defers to the
    /// dedicated squaring kernel [`mont_sqr`], which forms each cross term
    /// once rather than the full schoolbook product `mont_mul` would.
    fn montgomery_sqr_odd_with_workspace(
        value: &Self,
        modulus: &Self,
        n0_inv: u64,
        workspace: &mut Vec<u64>,
    ) -> Self {
        debug_assert!(modulus.is_odd(), "Montgomery path requires an odd modulus");
        let width = modulus.limbs.len();
        debug_assert!(
            value < modulus,
            "Montgomery operands must be reduced residues"
        );

        // Layout: `[scratch 2w+1 | value w | out w]`.
        let needed = mont_scratch_limbs(width) + 2 * width;
        if workspace.len() < needed {
            workspace.resize(needed, 0);
        }
        let (scratch, rest) = workspace.split_at_mut(mont_scratch_limbs(width));
        let (value_pad, out) = rest.split_at_mut(width);
        copy_padded(value_pad, &value.limbs);

        mont_sqr(
            &mut out[..width],
            value_pad,
            &modulus.limbs,
            n0_inv,
            scratch,
        );

        let mut result = Self {
            limbs: out[..width].to_vec(),
        };
        result.normalize();
        result
    }
}

/// Scratch limbs the `mont_*` kernels need for a `width`-limb modulus: a
/// `2 * width` product plus one limb so the reduction's final carry has a
/// home.
#[inline]
fn mont_scratch_limbs(width: usize) -> usize {
    width * 2 + 1
}

/// Copy `src` into `dst`, zero-padding the (little-endian) high limbs.
///
/// The `mont_*` kernels index fixed `width`-limb windows, so a shorter
/// canonical operand must be widened and the surplus explicitly zeroed —
/// the buffer is reused across calls and may still hold a previous residue.
/// A `src` wider than `dst` violates the reduced-residue contract and panics
/// on the slice range; the `debug_assert` names the invariant first.
#[inline]
fn copy_padded(dst: &mut [u64], src: &[u64]) {
    debug_assert!(src.len() <= dst.len(), "operand wider than the modulus");
    dst[..src.len()].copy_from_slice(src);
    dst[src.len()..].fill(0);
}

/// Montgomery multiplication on fixed-width limb slices:
/// `out = lhs * rhs * R^-1 mod n` with `R = 2^(64 * width)`, canonical
/// (`out < n`).
///
/// Reference: Montgomery, *Modular Multiplication Without Trial Division*,
/// Math. Comp. 44 (1985). This is the "separated operand scanning" shape from
/// Koç, Acar & Kaliski, *Analyzing and Comparing Montgomery Multiplication
/// Algorithms* (IEEE Micro 16(3), 1996): a full schoolbook product followed by
/// one reduction pass, which keeps each phase auditable on its own.
///
/// `lhs` and `rhs` are `width`-limb reduced residues; `scratch` holds the
/// double-width product ([`mont_scratch_limbs`]). `out` may not alias the
/// inputs (enforced by borrow rules at every call site).
fn mont_mul(
    out: &mut [u64],
    lhs: &[u64],
    rhs: &[u64],
    modulus: &[u64],
    n0_inv: u64,
    scratch: &mut [u64],
) {
    let width = modulus.len();
    debug_assert!(lhs.len() == width && rhs.len() == width && out.len() == width);
    let scratch = &mut scratch[..mont_scratch_limbs(width)];
    scratch.fill(0);

    // Schoolbook product `lhs * rhs` into the low `2 * width` limbs. The
    // carry out of each row lands one limb past the row end and cannot ripple
    // further: that limb was last touched as the previous row's carry, so
    // adding a fresh carry to it stays below `2^64`.
    for i in 0..width {
        let lhs_limb = u128::from(lhs[i]);
        let mut carry = 0u128;
        for j in 0..width {
            let acc = u128::from(scratch[i + j]) + lhs_limb * u128::from(rhs[j]) + carry;
            scratch[i + j] = low_u64(acc);
            carry = acc >> 64;
        }
        scratch[i + width] = low_u64(carry);
    }

    mont_redc(out, modulus, n0_inv, scratch);
}

/// Montgomery squaring: `out = value^2 * R^-1 mod n`, canonical.
///
/// The squaring kernel is the classic multiple-precision squaring of
/// *Handbook of Applied Cryptography*, Algorithm 14.16 (cross terms once, then
/// double, then add the diagonal), followed by one Montgomery reduction.
///
/// The product pass computes each cross term `value[i] * value[j]` (`i < j`)
/// once, doubles the whole partial sum with a single shift pass, then adds
/// the `value[i]^2` diagonal. Doubling as a separate pass sidesteps the
/// overflow in accumulating `2 * a_i * a_j` directly — that product can
/// exceed `u128` once the running carry joins it — and cuts the
/// multiplication count from `width^2` to `width * (width + 1) / 2`, which
/// matters because squarings are the bulk of an exponentiation ladder.
fn mont_sqr(out: &mut [u64], value: &[u64], modulus: &[u64], n0_inv: u64, scratch: &mut [u64]) {
    let width = modulus.len();
    debug_assert!(value.len() == width && out.len() == width);
    let scratch = &mut scratch[..mont_scratch_limbs(width)];
    scratch.fill(0);

    // Cross terms, each pair once: rows shorten as `i` rises.
    for i in 0..width {
        let value_limb = u128::from(value[i]);
        let mut carry = 0u128;
        for j in (i + 1)..width {
            let acc = u128::from(scratch[i + j]) + value_limb * u128::from(value[j]) + carry;
            scratch[i + j] = low_u64(acc);
            carry = acc >> 64;
        }
        scratch[i + width] = low_u64(carry);
    }

    // Double the cross-term sum: one bit shifted through `2 * width` limbs.
    let mut carry = 0u64;
    for limb in scratch[..width * 2].iter_mut() {
        let next = *limb >> 63;
        *limb = (*limb << 1) | carry;
        carry = next;
    }
    debug_assert!(carry == 0, "doubled cross terms stay under 2^(128w - 1)");

    // Diagonal `value[i]^2` terms, rippling each carry only as far as it
    // reaches.
    for (i, &limb) in value.iter().enumerate() {
        let mut carry = u128::from(limb) * u128::from(limb);
        let mut idx = i * 2;
        while carry != 0 {
            let acc = u128::from(scratch[idx]) + (carry & u128::from(u64::MAX));
            scratch[idx] = low_u64(acc);
            carry = (carry >> 64) + (acc >> 64);
            idx += 1;
        }
    }

    mont_redc(out, modulus, n0_inv, scratch);
}

/// Montgomery reduction (REDC): fold the double-width value in `scratch`
/// down to `out = scratch * R^-1 mod n`, canonical.
///
/// Each round picks `m = scratch[i] * (-n^-1) mod 2^64` so adding
/// `m * modulus` zeroes limb `i`; after `width` rounds the low half is all
/// zero and discarding it is the division by `R`. The result before the
/// final subtraction lies in `[0, 2n)` — `scratch < R*n` guarantees it — so
/// one conditional subtract restores the canonical range.
fn mont_redc(out: &mut [u64], modulus: &[u64], n0_inv: u64, scratch: &mut [u64]) {
    let width = modulus.len();

    // Carry out of each round's row, accumulated at `scratch[i + width]`.
    // Unlike the product pass this can ripple, so `overflow` tracks the bit
    // that escapes past the end of the double-width value.
    let mut overflow = 0u64;
    for i in 0..width {
        let m = u128::from(scratch[i].wrapping_mul(n0_inv));
        let mut carry = 0u128;
        for j in 0..width {
            let acc = u128::from(scratch[i + j]) + m * u128::from(modulus[j]) + carry;
            scratch[i + j] = low_u64(acc);
            carry = acc >> 64;
        }
        debug_assert!(scratch[i] == 0, "REDC round must clear its low limb");

        let acc = u128::from(scratch[i + width]) + u128::from(overflow) + carry;
        scratch[i + width] = low_u64(acc);
        overflow = low_u64(acc >> 64);
    }

    // For reduced operands the SOS product is below `R·n`, so the pre-subtract
    // value is below `2n` and the escaped bit is 0 or 1; a value of 2 would
    // make the single conditional subtract below insufficient.
    debug_assert!(overflow <= 1, "REDC overflow bit stays in {{0, 1}}");

    // The reduced value is the high half plus the escaped bit; it is below
    // `2n`, so at most one subtraction of `n` is needed. Subtract when the
    // escaped bit is set (the value is at least `R > n`) or the high half
    // reaches `n`.
    let high = &scratch[width..width * 2];
    if overflow != 0 || cmp_limbs(high, modulus) != Ordering::Less {
        let mut borrow = 0u128;
        for i in 0..width {
            let diff = (1u128 << 64) + u128::from(high[i]) - u128::from(modulus[i]) - borrow;
            out[i] = low_u64(diff);
            borrow = 1 - (diff >> 64);
        }
        debug_assert!(
            u128::from(overflow) == borrow,
            "conditional subtract must consume the escaped bit"
        );
    } else {
        out.copy_from_slice(high);
    }
}

/// Compare two equal-width little-endian limb slices, most significant limb
/// first. Equal width is the caller's obligation — both are modulus-width
/// windows — so unlike `BigUint`'s `Ord` there is no length test to settle
/// the ordering before the scan, and a shorter value must arrive zero-padded.
fn cmp_limbs(lhs: &[u64], rhs: &[u64]) -> Ordering {
    debug_assert!(lhs.len() == rhs.len());
    for (&l, &r) in lhs.iter().rev().zip(rhs.iter().rev()) {
        match l.cmp(&r) {
            Ordering::Equal => {}
            other => return other,
        }
    }
    Ordering::Equal
}

impl MontgomeryCtx {
    /// Modulus width in limbs; every kernel buffer is sized from this.
    fn width(&self) -> usize {
        self.modulus.limbs.len()
    }

    /// Grow `workspace` to at least the kernels' scratch size and return the
    /// whole buffer as a slice. It may be longer than
    /// [`mont_scratch_limbs`] — the buffer is shared with the wider layouts
    /// of the `*_with_workspace` helpers — so the kernels re-slice it to the
    /// exact width they need rather than trusting its length.
    fn scratch<'a>(&self, workspace: &'a mut Vec<u64>) -> &'a mut [u64] {
        let needed = mont_scratch_limbs(self.width());
        if workspace.len() < needed {
            workspace.resize(needed, 0);
        }
        workspace
    }

    /// Enter the Montgomery domain, reusing a caller-held workspace. Zero
    /// encodes to zero (`0·R ≡ 0`), which also keeps the reduced-residue
    /// precondition of the kernel satisfied for a modulus of one.
    fn encode_with_workspace(&self, value: &BigUint, workspace: &mut Vec<u64>) -> BigUint {
        if value.is_zero() {
            return BigUint::zero();
        }

        // Multiplying by `R^2 mod n` inside the reduction yields
        // `value * R mod n`, the Montgomery form. The reduction also brings
        // an unreduced `value` into range first.
        BigUint::montgomery_mul_odd_with_workspace(
            &value.modulo(&self.modulus),
            &self.r2_mod,
            &self.modulus,
            self.n0_inv,
            workspace,
        )
    }

    /// Convert back from Montgomery form: a bare REDC, since
    /// `REDC(x) = x * R^-1 mod n` and decoding is exactly multiplication by
    /// `R^-1`. No product pass needed — the double-width input is just the
    /// value itself, zero-extended.
    fn decode_with_workspace(&self, value: &BigUint, workspace: &mut Vec<u64>) -> BigUint {
        let width = self.width();
        debug_assert!(
            value.limbs.len() <= width,
            "Montgomery residues never exceed the modulus width"
        );

        let mut out = vec![0u64; width];
        let scratch = self.scratch(workspace);
        copy_padded(scratch, &value.limbs);
        mont_redc(&mut out, &self.modulus.limbs, self.n0_inv, scratch);

        let mut result = BigUint { limbs: out };
        result.normalize();
        result
    }

    /// The exponentiation ladder shared by [`Self::pow`] and
    /// [`Self::pow_encoded`]: two engines selected by exponent width — binary
    /// square-and-multiply for exponents inside one word, a fixed 4-bit
    /// window above that — followed by a single decoding REDC. The result
    /// leaves the Montgomery domain here, so both public entry points return
    /// an ordinary residue.
    ///
    /// Both engines run on `width`-limb buffers swapped in place, so the
    /// ladder allocates nothing after the table and every buffer that held an
    /// exponent-dependent intermediate is wiped on the way out.
    fn pow_encoded_with_workspace(
        &self,
        base_mont: &BigUint,
        exponent: &BigUint,
        workspace: &mut Vec<u64>,
    ) -> BigUint {
        if self.modulus.is_one() {
            return BigUint::zero();
        }

        let bits = exponent.bits();
        if bits == 0 {
            // `x^0 = 1`, and the modulus exceeds one here.
            return BigUint::one();
        }

        let width = self.width();
        let modulus = &self.modulus.limbs;
        let scratch = self.scratch(workspace);

        // The ladder runs on fixed-width buffers with a swap after each step,
        // so the whole exponentiation performs no allocation and no
        // intermediate wipes; every buffer that touched secret-derived state
        // is scrubbed once, on exit.
        let mut acc = vec![0u64; width];
        let mut tmp = vec![0u64; width];

        let result = if bits <= 64 {
            // Short exponents (e.g. F4 public exponents): right-to-left
            // binary square-and-multiply (Knuth, TAOCP vol. 2, §4.6.3)
            // avoids the window table setup. The accumulator starts at the
            // first set bit's power rather than at one, and the final
            // squaring — whose result no later bit consumes — is skipped.
            let exponent_word = exponent.limbs[0];
            let mut power = vec![0u64; width];
            copy_padded(&mut power, &base_mont.limbs);
            let mut seeded = false;

            for bit in 0..bits {
                if exponent_word >> bit & 1 == 1 {
                    if seeded {
                        mont_mul(&mut tmp, &acc, &power, modulus, self.n0_inv, scratch);
                        core::mem::swap(&mut acc, &mut tmp);
                    } else {
                        acc.copy_from_slice(&power);
                        seeded = true;
                    }
                }
                if bit + 1 < bits {
                    mont_sqr(&mut tmp, &power, modulus, self.n0_inv, scratch);
                    core::mem::swap(&mut power, &mut tmp);
                }
            }

            crate::scrub::zeroize_slice(&mut power);
            debug_assert!(seeded, "bits counts up to a set bit");
            acc
        } else {
            // Fixed 4-bit window, scanned left to right (the k-ary method:
            // Knuth, TAOCP vol. 2, §4.6.3; HAC algorithm 14.82). Per window:
            // four squarings plus at most one multiply out of a 16-entry
            // power table, ~1.23 multiplies per exponent bit against ~1.5
            // for binary; the 15-step table amortizes over any exponent long
            // enough to reach this path. A sliding window would shave a few
            // percent more at the cost of variable-length window parsing;
            // the fixed window keeps the scan trivially auditable.
            //
            // Like the rest of the crate this is variable-time: zero
            // windows skip their multiply.
            const WINDOW: usize = 4;
            const TABLE_LEN: usize = 1 << WINDOW;

            // table[i] holds `base^i` in Montgomery form, contiguously:
            // entry `i` at limbs `i * width..(i + 1) * width`. Even entries
            // are squares of earlier entries, odd entries one multiply away.
            let mut table = vec![0u64; TABLE_LEN * width];
            copy_padded(&mut table[..width], &self.one_mont.limbs);
            copy_padded(&mut table[width..2 * width], &base_mont.limbs);
            for i in 2..TABLE_LEN {
                let (built, rest) = table.split_at_mut(i * width);
                let entry = &mut rest[..width];
                if i % 2 == 0 {
                    mont_sqr(
                        entry,
                        &built[(i / 2) * width..(i / 2 + 1) * width],
                        modulus,
                        self.n0_inv,
                        scratch,
                    );
                } else {
                    mont_mul(
                        entry,
                        &built[(i - 1) * width..i * width],
                        &built[width..2 * width],
                        modulus,
                        self.n0_inv,
                        scratch,
                    );
                }
            }

            let windows = bits.div_ceil(WINDOW);
            let mut seeded = false;
            for w in (0..windows).rev() {
                if seeded {
                    for _ in 0..WINDOW {
                        mont_sqr(&mut tmp, &acc, modulus, self.n0_inv, scratch);
                        core::mem::swap(&mut acc, &mut tmp);
                    }
                }

                let mut idx = 0usize;
                for j in (0..WINDOW).rev() {
                    idx = (idx << 1) | usize::from(exponent.bit(w * WINDOW + j));
                }

                let entry = &table[idx * width..(idx + 1) * width];
                if !seeded {
                    // Top window: seed the accumulator directly instead of
                    // squaring up from one (it is non-zero because `bits`
                    // counts up to the most significant set bit).
                    acc.copy_from_slice(entry);
                    seeded = true;
                } else if idx != 0 {
                    // Skipping `idx == 0` merely skips a multiply by one;
                    // performing it would be correct, just wasted work.
                    mont_mul(&mut tmp, &acc, entry, modulus, self.n0_inv, scratch);
                    core::mem::swap(&mut acc, &mut tmp);
                }
            }

            crate::scrub::zeroize_slice(&mut table);
            acc
        };

        // Decode with a bare REDC (see `decode_with_workspace`), reusing
        // `tmp` as the double-width input.
        let mut acc = result;
        tmp.resize(mont_scratch_limbs(width), 0);
        copy_padded(&mut tmp, &acc);
        mont_redc(&mut acc, modulus, self.n0_inv, &mut tmp);

        crate::scrub::zeroize_slice(&mut tmp);
        let mut result = BigUint { limbs: acc };
        result.normalize();
        result
    }

    /// Build a Montgomery context for a modulus, or `None` if the modulus is
    /// zero or even.
    ///
    /// Montgomery reduction works in the residue system of `R = 2^(64·limbs)`,
    /// and REDC requires `R` and the modulus to be coprime; since `R` is a
    /// power of two, that holds exactly when the modulus is odd. An even (or
    /// zero) modulus has no Montgomery form, hence the `None` — reach for
    /// [`BarrettCtx`], which reduces modulo either parity. Construction also
    /// precomputes the inverse `−n⁻¹ mod 2⁶⁴` (Hensel lifting) and `R² mod n`
    /// once, so later encode/multiply steps do not repeat that work.
    #[must_use]
    pub fn new(modulus: &BigUint) -> Option<Self> {
        if modulus.is_zero() || !modulus.is_odd() {
            return None;
        }

        let n0_inv = montgomery_n0_inv(modulus.limbs[0]);

        // With `w` limbs, Montgomery arithmetic uses `R = 2^(64w)`. `R^2 mod
        // n` is the standard conversion factor for entering the Montgomery
        // domain because `montgomery_mul(a, R^2) = a * R^2 * R^-1 = aR`, the
        // Montgomery encoding of the ordinary residue `a`.
        let mut r2 = BigUint::zero();
        r2.set_bit(modulus.limbs.len() * 128);
        let r2_mod = r2.modulo(modulus);

        // `R mod n`, the Montgomery encoding of 1, seeds exponentiation
        // accumulators. One REDC derives it from the constant above —
        // `REDC(R^2 mod n) = R mod n` — instead of a second division.
        let width = modulus.limbs.len();
        let mut one_limbs = vec![0u64; width];
        let mut scratch = vec![0u64; mont_scratch_limbs(width)];
        copy_padded(&mut scratch, &r2_mod.limbs);
        mont_redc(&mut one_limbs, &modulus.limbs, n0_inv, &mut scratch);
        let mut one_mont = BigUint { limbs: one_limbs };
        one_mont.normalize();

        Some(Self {
            modulus: modulus.clone(),
            n0_inv,
            r2_mod,
            one_mont,
        })
    }

    /// Return the odd modulus this context was built for.
    #[must_use]
    pub fn modulus(&self) -> &BigUint {
        &self.modulus
    }

    /// Convert an ordinary residue into Montgomery form (the inverse of
    /// [`Self::decode`]).
    ///
    /// Encoding is multiplication by `R = 2^(64·limbs)` modulo `n`, done as one
    /// Montgomery multiplication by the precomputed `R² mod n`: REDC of
    /// `value · R²` returns `value · R mod n`. The argument is reduced modulo
    /// `n` first, so any representative is accepted.
    #[must_use]
    pub fn encode(&self, value: &BigUint) -> BigUint {
        let mut workspace = Vec::new();
        let result = self.encode_with_workspace(value, &mut workspace);
        crate::scrub::zeroize_slice(workspace.as_mut_slice());
        result
    }

    /// Convert a Montgomery residue back to the ordinary representation.
    ///
    /// Accepts any representative: an operand at or above the modulus (or wider
    /// than it) is reduced first, so the result is always the canonical value
    /// in `[0, modulus)`. This is the domain's exit boundary, called once per
    /// computation rather than in the inner loop, so the reduction is free in
    /// practice and removes any way to get a non-canonical answer.
    #[must_use]
    pub fn decode(&self, value: &BigUint) -> BigUint {
        let reduced = if value >= &self.modulus {
            value.modulo(&self.modulus)
        } else {
            value.clone()
        };
        let mut workspace = Vec::new();
        let result = self.decode_with_workspace(&reduced, &mut workspace);
        crate::scrub::zeroize_slice(workspace.as_mut_slice());
        result
    }

    /// Multiply two ordinary residues modulo the context modulus — the
    /// one-shot encode → multiply → decode round trip.
    ///
    /// Convenient for a single product. Callers doing many operations should
    /// encode once and stay in the domain with [`Self::mul_mont`] /
    /// [`Self::square_mont`], decoding only at the end, rather than re-paying
    /// the encode and decode conversions on every step.
    #[must_use]
    pub fn mul(&self, lhs: &BigUint, rhs: &BigUint) -> BigUint {
        let mut workspace = Vec::new();
        let lhs_mont = self.encode_with_workspace(lhs, &mut workspace);
        let rhs_mont = self.encode_with_workspace(rhs, &mut workspace);
        let product_mont = BigUint::montgomery_mul_odd_with_workspace(
            &lhs_mont,
            &rhs_mont,
            &self.modulus,
            self.n0_inv,
            &mut workspace,
        );
        let result = self.decode_with_workspace(&product_mont, &mut workspace);
        crate::scrub::zeroize_slice(workspace.as_mut_slice());
        result
    }

    /// Square one ordinary residue modulo the context modulus — the one-shot
    /// encode → square → decode round trip.
    ///
    /// Unlike [`Self::mul`] the in-domain step is the dedicated squaring kernel
    /// (`mont_sqr`), which forms each cross term once instead of the full
    /// `width²` products. The same encode-once, stay-in-domain advice as
    /// [`Self::mul`] applies to repeated work.
    #[must_use]
    pub fn square(&self, value: &BigUint) -> BigUint {
        let mut workspace = Vec::new();
        let value_mont = self.encode_with_workspace(value, &mut workspace);
        let square_mont = BigUint::montgomery_sqr_odd_with_workspace(
            &value_mont,
            &self.modulus,
            self.n0_inv,
            &mut workspace,
        );
        let result = self.decode_with_workspace(&square_mont, &mut workspace);
        crate::scrub::zeroize_slice(workspace.as_mut_slice());
        result
    }

    /// Multiply two residues that are **already in Montgomery form**, staying
    /// in Montgomery form.
    ///
    /// One Montgomery reduction instead of the encode/multiply/decode round
    /// trip of [`Self::mul`]; the workhorse for callers (such as elliptic
    /// curve point arithmetic) that keep whole computations in the Montgomery
    /// domain and convert only at the boundaries.
    ///
    /// Operands must be reduced residues, below the modulus — the domain
    /// contract shared by every in-domain operation, as produced by
    /// [`Self::encode`] and returned by the domain operations themselves. The
    /// single conditional subtraction in the reduction relies on it; debug
    /// builds assert it.
    ///
    /// Unlike [`Self::mul`]/[`Self::pow`] this does **not** scrub its
    /// workspace: it is the innermost field-multiply, called in tight loops.
    /// Memory scrubbing is out of scope for this variable-time crate (see the
    /// crate-level note); the product `BigUint` is wiped by its own `Drop`
    /// regardless, and callers who want the scratch wiped can route through
    /// [`Self::mul`], whose encode/decode path already does.
    ///
    /// # Panics
    ///
    /// Panics if either operand occupies more limbs than the modulus: the
    /// kernel pads operands into modulus-width windows, and a wider one does
    /// not fit. An operand of the modulus's width but at or above it does not
    /// panic in release builds — it returns a non-canonical result, which is
    /// what the debug assertion exists to catch.
    #[must_use]
    pub fn mul_mont(&self, lhs: &BigUint, rhs: &BigUint) -> BigUint {
        debug_assert!(
            lhs < &self.modulus && rhs < &self.modulus,
            "domain operands arrive reduced"
        );
        let mut workspace = Vec::new();
        BigUint::montgomery_mul_odd_with_workspace(
            lhs,
            rhs,
            &self.modulus,
            self.n0_inv,
            &mut workspace,
        )
    }

    /// [`Self::mul_mont`] with a caller-supplied workspace, for the loops
    /// where the product *is* the loop: the kernel pads its operands and
    /// carves scratch out of one buffer, and threading that buffer through
    /// a sequence of calls allocates it once instead of per multiply.
    /// Pass the same `Vec` (empty on the first call — it is sized on
    /// demand) to every domain operation in the loop. Measured
    /// per-operation at exact modulus widths by `mont_workspace_timing` in
    /// this file (run with `--ignored`; it interleaves the two forms in
    /// paired chunks so drift cancels, and prints every pass so a figure
    /// inside the spread reads as the noise it is): about 43% saved at one
    /// limb, roughly 25–33% at four, and ~20% at eight — the 64–512-bit
    /// moduli where Pollard rho actually runs — falling to 2–3% at 32
    /// limbs and to the edge of measurement (~1%) at 64, where the kernel
    /// dominates the allocator.
    ///
    /// The workspace holds unscrubbed Montgomery intermediates, exactly as
    /// [`Self::mul_mont`]'s discarded buffer does (see the scope note
    /// there); a caller that wants it wiped scrubs it after the loop.
    ///
    /// # Panics
    ///
    /// Panics if either operand occupies more limbs than the modulus, as
    /// in [`Self::mul_mont`].
    #[must_use]
    pub fn mul_mont_with_workspace(
        &self,
        lhs: &BigUint,
        rhs: &BigUint,
        workspace: &mut Vec<u64>,
    ) -> BigUint {
        debug_assert!(
            lhs < &self.modulus && rhs < &self.modulus,
            "domain operands arrive reduced"
        );
        BigUint::montgomery_mul_odd_with_workspace(lhs, rhs, &self.modulus, self.n0_inv, workspace)
    }

    /// Square a residue that is already in Montgomery form, staying in
    /// Montgomery form.
    ///
    /// Uses the dedicated squaring kernel (`mont_sqr`, each cross term formed
    /// once) rather than `mul_mont(value, value)`: it does `width·(width+1)/2`
    /// multiplications instead of `width²`, so it is the faster kernel at the
    /// exponentiation-sized moduli where squarings dominate. At very small
    /// widths its fixed doubling and diagonal passes cost more than the saved
    /// multiplications, a property of the separated-squaring construction; the
    /// crossover favors `mont_sqr` where it matters. Like [`Self::mul_mont`]
    /// it does not scrub its workspace, for the same reason. Operands must be
    /// reduced residues (the shared domain contract; debug builds assert it).
    ///
    /// # Panics
    ///
    /// Panics if `value` occupies more limbs than the modulus, as in
    /// [`Self::mul_mont`].
    #[must_use]
    pub fn square_mont(&self, value: &BigUint) -> BigUint {
        debug_assert!(value < &self.modulus, "domain operand arrives reduced");
        let mut workspace = Vec::new();
        BigUint::montgomery_sqr_odd_with_workspace(
            value,
            &self.modulus,
            self.n0_inv,
            &mut workspace,
        )
    }

    /// [`Self::square_mont`] with a caller-supplied workspace — the
    /// squaring companion to [`Self::mul_mont_with_workspace`], sharing its
    /// contract, its scrubbing posture, and its buffer (the two size their
    /// windows independently from the same `Vec`).
    ///
    /// # Panics
    ///
    /// Panics if `value` occupies more limbs than the modulus, as in
    /// [`Self::mul_mont`].
    #[must_use]
    pub fn square_mont_with_workspace(&self, value: &BigUint, workspace: &mut Vec<u64>) -> BigUint {
        debug_assert!(value < &self.modulus, "domain operand arrives reduced");
        BigUint::montgomery_sqr_odd_with_workspace(value, &self.modulus, self.n0_inv, workspace)
    }

    /// Add two residues in Montgomery form, staying in Montgomery form:
    /// the reduced-operand fast path of [`BigUint::mod_add`], one
    /// compare-and-subtract with no reduction machinery. The encoding is
    /// linear — `x̃ + ỹ ≡ (x + y)·R (mod n)` — so modular addition acts on
    /// domain residues exactly as on ordinary ones. Operands must be
    /// reduced (below the modulus), the same precondition every domain
    /// operation carries; it is checked in debug builds only, and a
    /// release caller violating it gets a non-canonical result.
    #[must_use]
    pub fn add_mont(&self, lhs: &BigUint, rhs: &BigUint) -> BigUint {
        debug_assert!(
            lhs < &self.modulus && rhs < &self.modulus,
            "domain operands arrive reduced"
        );
        let sum = lhs.add_ref(rhs);
        if sum >= self.modulus {
            sum.sub_ref(&self.modulus)
        } else {
            sum
        }
    }

    /// Subtract one Montgomery-form residue from another, staying in
    /// Montgomery form: the reduced-operand fast path of
    /// [`BigUint::mod_sub`], the wrap adding the modulus back. Linear for
    /// the same reason as [`Self::add_mont`], under the same
    /// debug-checked reduced-operand contract.
    #[must_use]
    pub fn sub_mont(&self, lhs: &BigUint, rhs: &BigUint) -> BigUint {
        debug_assert!(
            lhs < &self.modulus && rhs < &self.modulus,
            "domain operands arrive reduced"
        );
        if lhs >= rhs {
            lhs.sub_ref(rhs)
        } else {
            self.modulus.add_ref(lhs).sub_ref(rhs)
        }
    }

    /// The Montgomery encoding of one, `R mod n` — the multiplicative
    /// identity of the domain, precomputed at construction. A ladder that
    /// starts from an identity needs this and not [`BigUint::one`], which is
    /// not a domain element; it is also entry 0 of the window table in
    /// [`Self::pow`].
    #[must_use]
    pub fn one_mont(&self) -> &BigUint {
        &self.one_mont
    }

    /// Compute `base^exponent mod modulus` inside the context.
    ///
    /// Encodes `base` once, runs the exponentiation entirely in the
    /// Montgomery domain — where each squaring and product is one REDC
    /// rather than a division — and decodes the result. Exponents above 64
    /// bits take a fixed 4-bit-window left-to-right ladder (the k-ary
    /// method, HAC Algorithm 14.82), seeded from the top window; exponents
    /// of 64 bits or fewer (e.g. the F4 public exponent) take right-to-left
    /// binary square-and-multiply instead, which skips the window-table
    /// setup that would dominate so short a ladder. Keeping the whole
    /// ladder in-domain is the point of the context: the per-step cost is
    /// a Montgomery multiply, not a modular reduction. The workspace holding
    /// the (possibly secret) intermediates is wiped before it is freed.
    ///
    /// `base` may be unreduced (encoding reduces it); `exponent == 0` yields
    /// one, and a modulus of one yields zero.
    #[must_use]
    pub fn pow(&self, base: &BigUint, exponent: &BigUint) -> BigUint {
        let mut workspace = Vec::new();
        let base_mont = self.encode_with_workspace(base, &mut workspace);
        let result = self.pow_encoded_with_workspace(&base_mont, exponent, &mut workspace);
        // The workspace held Montgomery intermediates of a (possibly secret)
        // exponentiation; wipe it before the buffer is freed.
        crate::scrub::zeroize_slice(workspace.as_mut_slice());
        result
    }

    /// Compute `base^exponent mod modulus` with `base` already in Montgomery form.
    ///
    /// Saves the encoding multiplication when a caller reuses one base across
    /// many exponentiations and can cache its encoded form. Otherwise
    /// identical to [`Self::pow`], including the decoding step: the result
    /// comes back as an ordinary residue, not a domain element.
    ///
    /// `base_mont` must be a reduced residue (below the modulus), the shared
    /// domain contract; debug builds assert it.
    ///
    /// # Panics
    ///
    /// Panics if `base_mont` occupies more limbs than the modulus, as in
    /// [`Self::mul_mont`].
    #[must_use]
    pub fn pow_encoded(&self, base_mont: &BigUint, exponent: &BigUint) -> BigUint {
        debug_assert!(base_mont < &self.modulus, "domain operand arrives reduced");
        let mut workspace = Vec::new();
        let result = self.pow_encoded_with_workspace(base_mont, exponent, &mut workspace);
        crate::scrub::zeroize_slice(workspace.as_mut_slice());
        result
    }
}

/// Sign-aware total order: every negative value is below zero, zero below
/// every positive. Two positives compare by magnitude; two negatives compare
/// by magnitude reversed. Consistent with `Eq` because representations are
/// canonical — zero is exactly `(Sign::Zero, empty magnitude)`.
impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.sign, other.sign) {
            (Sign::Negative, Sign::Negative) => other.magnitude.cmp(&self.magnitude),
            (Sign::Negative, _) => Ordering::Less,
            (_, Sign::Negative) => Ordering::Greater,
            (Sign::Zero, Sign::Zero) => Ordering::Equal,
            (Sign::Zero, Sign::Positive) => Ordering::Less,
            (Sign::Positive, Sign::Zero) => Ordering::Greater,
            (Sign::Positive, Sign::Positive) => self.magnitude.cmp(&other.magnitude),
        }
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Barrett reduction context for a fixed modulus of either parity — the
/// complement to [`MontgomeryCtx`], which requires an odd modulus.
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
pub struct BarrettCtx {
    modulus: BigUint,
    mu: BigUint,
    limb_count: usize,
}

#[cfg(test)]
thread_local! {
    /// Corrections taken by the last `BarrettCtx::reduce` on this thread.
    static CORRECTIONS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

impl BarrettCtx {
    /// Build the context. The single division here computes
    /// `μ = ⌊b^{2k}/n⌋` for `b = 2⁶⁴` and `k` the modulus's limb count; every
    /// later [`Self::reduce`] spends two multiplications and at most two
    /// subtractions instead of a division, which is the whole point of the
    /// precomputation.
    ///
    /// `None` for a modulus below 2: zero has no residues, and modulo one
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
            return x.modulo(&self.modulus);
        }
        if *x < self.modulus {
            return x.clone();
        }
        // q̂ = ⌊⌊x/b^(k−1)⌋·μ/b^(k+1)⌋ — the two shifts are limb-aligned.
        let mut q = x.clone();
        q.shr_bits(64 * (k - 1));
        q = q.mul_ref(&self.mu);
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
            q.mul_ref(&self.modulus).low_bits(window)
        };
        let mut r = if x_low >= qn_low {
            x_low.sub_ref(&qn_low)
        } else {
            let mut lift = BigUint::zero();
            lift.set_bit(window);
            lift.add_ref(&x_low).sub_ref(&qn_low)
        };
        // The estimate is short by at most two.
        let mut corrections = 0u32;
        while r >= self.modulus {
            r = r.sub_ref(&self.modulus);
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

    /// `(a + b) mod n` — the additive counterpart of [`Self::mul_mod`], so a
    /// caller holding a context need not reach past it for one operation.
    /// `μ` plays no part: [`BigUint::mod_add`] reduces the operands and then
    /// corrects the sum by at most one subtraction, since a sum of two
    /// residues is below `2n`.
    #[must_use]
    pub fn add_mod(&self, a: &BigUint, b: &BigUint) -> BigUint {
        BigUint::mod_add(a, b, &self.modulus)
    }

    /// `(a − b) mod n`, through [`BigUint::mod_sub`]: the wrap adds the
    /// modulus back rather than borrowing, ℕ having no negative values.
    #[must_use]
    pub fn sub_mod(&self, a: &BigUint, b: &BigUint) -> BigUint {
        BigUint::mod_sub(a, b, &self.modulus)
    }

    /// `(a · b) mod n`: both operands reduced, then the double-width product
    /// reduced again. Three [`Self::reduce`] calls, of which the first two
    /// collapse to a comparison and a copy when the operands already lie in
    /// `[0, n)` — the case in an exponentiation loop, where every operand is
    /// a previous result.
    #[must_use]
    pub fn mul_mod(&self, a: &BigUint, b: &BigUint) -> BigUint {
        let a = self.reduce(a);
        let b = self.reduce(b);
        self.reduce(&a.mul_ref(&b))
    }

    /// `a² mod n`. The square comes from [`BigUint::square_ref`], whose
    /// specialized kernels form each cross term once between 8 and 256
    /// limbs, so this costs a squaring plus one Barrett reduction. The
    /// Montgomery domain's [`MontgomeryCtx::square_mont`] goes further
    /// still, fusing the reduction into the kernel.
    #[must_use]
    pub fn square_mod(&self, a: &BigUint) -> BigUint {
        let a = self.reduce(a);
        self.reduce(&a.square_ref())
    }

    /// `base^exponent mod n` by left-to-right binary exponentiation (Knuth,
    /// *TAOCP* vol. 2, §4.6.3) with one [`Self::reduce`] after each step —
    /// the exponentiation route for even moduli, where Montgomery cannot
    /// operate. The accumulator is seeded from the exponent's top set bit
    /// (as the Montgomery ladder in this file does), so the cost is one
    /// squaring per remaining exponent bit and one multiplication per
    /// remaining set bit; there is no window table here, unlike
    /// [`MontgomeryCtx::pow`]. `0^0 = 1` by the usual convention.
    ///
    /// Variable-time, like the rest of the crate: a clear exponent bit skips
    /// its multiplication.
    #[must_use]
    pub fn pow_mod(&self, base: &BigUint, exponent: &BigUint) -> BigUint {
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
            result = self.reduce(&result.square_ref());
            if exponent.bit(bit) {
                result = self.reduce(&result.mul_ref(&base));
            }
        }
        result
    }
}

/// Error from parsing a [`BigUint`] or [`BigInt`] out of a string: the
/// input was empty (or a bare sign), or held a character that is not a
/// digit of the requested radix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ParseBigIntError;

impl core::fmt::Display for ParseBigIntError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("empty string or invalid digit")
    }
}

impl std::error::Error for ParseBigIntError {}

impl core::fmt::Display for BigUint {
    /// Decimal rendering, through [`BigUint::to_str_radix`]. The digits go to
    /// `pad_integral` rather than the formatter directly, so width, fill,
    /// zero-padding and the `+` flag behave as they do for the primitive
    /// integers.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.pad_integral(true, "", &self.to_str_radix(10))
    }
}

impl core::str::FromStr for BigUint {
    type Err = ParseBigIntError;

    /// Decimal parsing, through [`BigUint::from_str_radix`]: leading zeros
    /// are accepted, a sign or surrounding whitespace is not. Any rejected
    /// input yields [`ParseBigIntError`], which carries no position — the
    /// error exists to distinguish failure from a valid parse, not to
    /// diagnose the input.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::from_str_radix(text, 10).ok_or(ParseBigIntError)
    }
}

impl core::fmt::Display for BigInt {
    /// Decimal rendering with a leading `-` for negative values. The sign is
    /// passed to `pad_integral` rather than prepended to the digits, so a
    /// requested width pads between the sign and the digits, as for the
    /// primitive integers.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.pad_integral(
            self.sign != Sign::Negative,
            "",
            &self.magnitude.to_str_radix(10),
        )
    }
}

impl core::str::FromStr for BigInt {
    type Err = ParseBigIntError;

    /// Decimal parsing with an optional leading `-`. A leading `+` is *not*
    /// accepted, unlike `i64::from_str`: the magnitude is parsed by
    /// [`BigUint::from_str_radix`], which treats any non-digit as invalid.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::from_str_radix(text, 10).ok_or(ParseBigIntError)
    }
}

impl Drop for BigUint {
    fn drop(&mut self) {
        // BigUint values may hold secrets — private exponents, prime
        // factors, nonces. Clear the limb buffer on drop so they do not
        // linger in freed heap memory.
        crate::scrub::zeroize_slice(self.limbs.as_mut_slice());
    }
}

/// The low 64 bits of a `u128` accumulator as a limb. Every kernel here
/// accumulates in `u128` and splits the result into a stored limb and a
/// carry; this is the stored half, written as a masked `try_from` so the
/// truncation is a checked operation rather than an `as` cast.
#[inline]
fn low_u64(value: u128) -> u64 {
    u64::try_from(value & u128::from(u64::MAX)).expect("masked low 64 bits always fit into u64")
}

/// Copy `value` into a fresh `len`-limb buffer, shifted left by `shift` bits.
///
/// `shift` is below 64 and `len` is at least `value.len()`; this is the
/// Algorithm D normalization step, which never needs a whole-limb shift.
fn shl_into(value: &[u64], shift: u32, len: usize) -> Vec<u64> {
    debug_assert!(shift < 64, "normalization shift stays within one limb");
    debug_assert!(len >= value.len(), "destination must hold the source");

    let mut out = vec![0u64; len];
    if shift == 0 {
        out[..value.len()].copy_from_slice(value);
        return out;
    }

    // `shift` is in `1..64`, so `64 - shift` is also a defined shift amount.
    let mut carry = 0u64;
    for (slot, &limb) in out.iter_mut().zip(value.iter()) {
        *slot = (limb << shift) | carry;
        carry = limb >> (64 - shift);
    }
    if value.len() < len {
        out[value.len()] = carry;
    } else {
        debug_assert!(carry == 0, "shift carried out of the destination");
    }
    out
}

/// Return `value` shifted right by `shift` bits (below 64) in a fresh buffer —
/// the inverse of [`shl_into`], undoing Algorithm D's D1 normalization on the
/// remainder in step D8. The width is unchanged: the shift is by less than one
/// limb, so only the top limb can lose significance, and the caller
/// normalizes.
fn shr_limbs(value: &[u64], shift: u32) -> Vec<u64> {
    debug_assert!(shift < 64, "normalization shift stays within one limb");
    if shift == 0 {
        return value.to_vec();
    }

    let mut out = vec![0u64; value.len()];
    for (i, slot) in out.iter_mut().enumerate() {
        let high = value.get(i + 1).map_or(0, |&next| next << (64 - shift));
        *slot = (value[i] >> shift) | high;
    }
    out
}

/// Compute `-n0^-1 mod 2^64`, the reduction coefficient REDC multiplies by
/// each round (Dussé & Kaliski, *A Cryptographic Library for the Motorola
/// DSP56000*, EUROCRYPT '90, where the word-level variant of Montgomery
/// reduction was introduced).
fn montgomery_n0_inv(n0: u64) -> u64 {
    debug_assert!(n0 & 1 == 1, "Montgomery path requires an odd modulus");
    // Newton/Hensel iteration in Z_(2^64): `inv = 1` inverts `n0` modulo 2
    // (both are odd), and each step doubles the correct low bits —
    // 1, 2, 4, 8, 16, 32, 64 — so six steps reach the full word. Montgomery
    // reduction wants the negation.
    let mut inv = 1u64;
    for _ in 0..6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(n0.wrapping_mul(inv)));
    }
    inv.wrapping_neg()
}

/// Signed product `a · b`, named at the point of use so the Toom
/// evaluate/interpolate sequences read as ordinary arithmetic: evaluating at
/// the points `−1` and `−2` makes those pointwise multiplications signed,
/// even though the multiplicands and the final product are not.
fn bigint_mul(a: &BigInt, b: &BigInt) -> BigInt {
    a.mul_ref(b)
}

/// `x / divisor` where `divisor` is known to divide `x` — the interpolation
/// steps of Toom-3 and Toom-4 (dividing by 2, 3, 4, 5, 8, 12).
///
/// Exactness is a property of the interpolation, not of the inputs: each
/// quotient in the Vandermonde solve is an integer because the product
/// polynomial's coefficients are integers, so the remainder is discarded and
/// only checked in debug builds. The sign rides along unchanged, the divisor
/// being positive, so a single-limb Horner division of the magnitude suffices.
fn bigint_div_exact(x: &BigInt, divisor: u64) -> BigInt {
    debug_assert!(divisor > 0, "Toom interpolation never divides by zero");
    // Most of the interpolation's divisors are 2, 4 or 8. An exact division by
    // a power of two is a right shift, so take it: the Horner path below runs a
    // `u128` division for every limb, and at Toom widths (Toom-3 dispatches at
    // 128 limbs, Toom-4 at 3072) that is thousands of multi-cycle divisions
    // standing in for a single-cycle shift per word. The evaluation side of
    // this same algorithm already scales by shifting; this is the interpolation
    // side catching up.
    if divisor.is_power_of_two() {
        let shift = divisor.trailing_zeros() as usize;
        debug_assert!(
            x.is_zero() || x.magnitude().trailing_zeros().unwrap_or(0) >= shift,
            "Toom interpolation divides evenly by {divisor}"
        );
        let mut magnitude = x.magnitude().clone();
        magnitude.shr_bits(shift);
        return BigInt::from_parts(x.sign(), magnitude);
    }
    let (quotient, remainder) = BigUint::div_rem_limb(x.magnitude().limbs(), divisor);
    debug_assert!(
        remainder == 0,
        "Toom interpolation divides evenly by {divisor}"
    );
    BigInt::from_parts(x.sign(), quotient)
}

/// Multiply by `2^shift`, for the interpolation's power-of-two weights.
///
/// The general path, `mul_biguint_ref(&BigUint::from_u64(1 << shift))`,
/// allocates a constant and enters the full multiplication dispatch to apply a
/// weight that is one shift of the limb buffer.
fn bigint_shl_exact(x: &BigInt, shift: usize) -> BigInt {
    let mut magnitude = x.magnitude().clone();
    magnitude.shl_bits(shift);
    BigInt::from_parts(x.sign(), magnitude)
}

impl BigInt {
    /// Construct zero: `Sign::Zero` over an empty magnitude, the one
    /// representation of zero this type admits.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            sign: Sign::Zero,
            magnitude: BigUint::zero(),
        }
    }

    /// Construct from an explicit sign and magnitude, canonicalizing the pair.
    ///
    /// The representation admits exactly one zero, so an inconsistent argument
    /// is normalized rather than stored: any sign with a zero magnitude becomes
    /// canonical zero (`Sign::Zero`), and `Sign::Zero` with a non-zero
    /// magnitude becomes `Positive`. This keeps `Eq`/`Ord` total and
    /// well-defined, at the cost of silently accepting a contradictory `(sign,
    /// magnitude)` pair — callers constructing from untrusted parts should not
    /// rely on the sign surviving unchanged.
    #[must_use]
    pub fn from_parts(sign: Sign, magnitude: BigUint) -> Self {
        if magnitude.is_zero() {
            return Self::zero();
        }

        let canonical_sign = match sign {
            Sign::Zero => Sign::Positive,
            other => other,
        };

        Self {
            sign: canonical_sign,
            magnitude,
        }
    }

    /// Construct a non-negative signed integer from an unsigned magnitude.
    /// Routed through [`Self::from_parts`], so a zero magnitude yields
    /// canonical zero rather than a positive zero. This is the lift the Toom
    /// evaluation uses to put an unsigned operand into signed arithmetic.
    #[must_use]
    pub fn from_biguint(magnitude: BigUint) -> Self {
        Self::from_parts(Sign::Positive, magnitude)
    }

    /// Construct from a signed double word. Total for the same reason as
    /// [`Self::from_i64`]: `i128::MIN` has no `i128` negation, and its
    /// magnitude `2^127` is an ordinary `u128`.
    #[must_use]
    pub fn from_i128(value: i128) -> Self {
        let sign = if value < 0 {
            Sign::Negative
        } else {
            Sign::Positive
        };
        Self::from_parts(sign, BigUint::from_u128(value.unsigned_abs()))
    }

    /// Construct from a machine-word signed value. The magnitude is taken
    /// with `unsigned_abs`, which is total: `i64::MIN` has no `i64` negation
    /// but its magnitude `2^63` is an ordinary `u64`.
    #[must_use]
    pub fn from_i64(value: i64) -> Self {
        let sign = if value < 0 {
            Sign::Negative
        } else {
            Sign::Positive
        };
        Self::from_parts(sign, BigUint::from_u64(value.unsigned_abs()))
    }

    /// Return the sign. `Sign::Zero` identifies zero exactly, so this is also
    /// the fastest zero test.
    #[must_use]
    pub fn sign(&self) -> Sign {
        self.sign
    }

    /// Borrow the absolute value. Sign and magnitude are stored apart, so
    /// `|self|` is a borrow rather than a computation, and the unsigned
    /// kernels can be applied to it directly.
    #[must_use]
    pub fn magnitude(&self) -> &BigUint {
        &self.magnitude
    }

    /// Return `-self`: the sign flips and the magnitude is copied. Zero
    /// negates to zero, which is exactly what the separate `Sign::Zero`
    /// variant buys — a sign convention over the magnitude would produce a
    /// second, unequal zero here.
    #[must_use]
    pub fn negated(&self) -> Self {
        let sign = match self.sign {
            Sign::Positive => Sign::Negative,
            Sign::Negative => Sign::Positive,
            Sign::Zero => Sign::Zero,
        };
        Self {
            sign,
            magnitude: self.magnitude.clone(),
        }
    }

    /// Return `self + other`: a clone of `self` followed by
    /// [`Self::add_assign_ref`].
    #[must_use]
    pub fn add_ref(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.add_assign_ref(other);
        out
    }

    /// Add another integer in place, reusing the magnitude's limb buffer in
    /// every sign combination. The sign case analysis is `combine_assign`,
    /// which this enters with `other`'s own sign.
    pub fn add_assign_ref(&mut self, other: &Self) {
        self.combine_assign(other.sign, &other.magnitude);
    }

    /// Return `self - other`: a clone of `self` followed by
    /// [`Self::sub_assign_ref`]. Total on ℤ, unlike [`BigUint::sub_ref`].
    #[must_use]
    pub fn sub_ref(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.sub_assign_ref(other);
        out
    }

    /// Subtract another integer in place, reusing the magnitude's limb
    /// buffer in every sign combination. Full signed semantics — the sign
    /// follows the result, and nothing panics — unlike
    /// [`BigUint::sub_assign_ref`], whose domain has no negative values.
    pub fn sub_assign_ref(&mut self, other: &Self) {
        let negated = match other.sign {
            Sign::Positive => Sign::Negative,
            Sign::Negative => Sign::Positive,
            Sign::Zero => Sign::Zero,
        };
        self.combine_assign(negated, &other.magnitude);
    }

    /// The shared core of the signed in-place operations:
    /// `self ← self + s·m`, where `(s, m)` is `other`'s decomposition for
    /// addition and its negation for subtraction. Like signs grow this
    /// magnitude in place; unlike signs cancel — the smaller magnitude
    /// leaves the larger (reversed in place when the larger is `m`), and the
    /// sign follows the survivor. Zero results clear the buffer without
    /// releasing it, preserving the type's canonical form (`Sign::Zero`
    /// with an empty magnitude).
    fn combine_assign(&mut self, sign: Sign, magnitude: &BigUint) {
        debug_assert!(
            (sign == Sign::Zero) == magnitude.is_zero(),
            "operand arrives in canonical form: Sign::Zero iff zero magnitude"
        );
        if sign == Sign::Zero {
            return;
        }
        if self.sign == Sign::Zero {
            self.magnitude.clone_from(magnitude);
            self.sign = sign;
            return;
        }
        if self.sign == sign {
            self.magnitude.add_assign_ref(magnitude);
            return;
        }
        match self.magnitude.cmp(magnitude) {
            Ordering::Greater => self.magnitude.sub_assign_ref(magnitude),
            Ordering::Less => {
                self.magnitude.rsub_assign_ref(magnitude);
                self.sign = sign;
            }
            Ordering::Equal => {
                // Scrub before clearing: `Drop` covers only the initialized
                // prefix, and these limbs may be Bézout coefficients of
                // secret operands. The capacity is kept for reuse.
                crate::scrub::zeroize_slice(self.magnitude.limbs.as_mut_slice());
                self.magnitude.limbs.clear();
                self.sign = Sign::Zero;
            }
        }
    }

    /// Return `self * factor` for a non-negative factor: the magnitudes
    /// multiply and the sign is unchanged, a positive factor being unable to
    /// flip it. The form the Toom interpolation wants when it scales a signed
    /// coefficient by a small positive constant, since the constant then
    /// needs no sign of its own.
    #[must_use]
    pub fn mul_biguint_ref(&self, factor: &BigUint) -> Self {
        if factor.is_zero() || self.sign == Sign::Zero {
            return Self::zero();
        }

        Self::from_parts(self.sign, self.magnitude.mul_ref(factor))
    }

    /// Parse from a digit string with an optional leading `-`, in the given
    /// radix (2 through 36); `None` on an empty string, a bare sign, or an
    /// invalid digit. `-0` parses to canonical zero.
    ///
    /// # Panics
    ///
    /// Panics when `radix` is outside `2..=36`.
    #[must_use]
    pub fn from_str_radix(text: &str, radix: u32) -> Option<Self> {
        let (negative, digits) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text),
        };
        let magnitude = BigUint::from_str_radix(digits, radix)?;
        let sign = if negative {
            Sign::Negative
        } else {
            Sign::Positive
        };
        Some(Self::from_parts(sign, magnitude))
    }

    /// Render as a digit string in the given radix, a leading `-` for
    /// negative values; the mirror of [`Self::from_str_radix`].
    ///
    /// # Panics
    ///
    /// Panics when `radix` is outside `2..=36`.
    #[must_use]
    pub fn to_str_radix(&self, radix: u32) -> String {
        let digits = self.magnitude.to_str_radix(radix);
        match self.sign {
            Sign::Negative => format!("-{digits}"),
            _ => digits,
        }
    }

    /// Signed product `self · other`: the magnitudes multiply through the
    /// full [`BigUint::mul_ref`] kernel ladder and the sign follows the
    /// usual rule (like signs positive, unlike negative, zero absorbing).
    /// Inside the crate this is the Half-GCD matrix arithmetic and the
    /// `PolyZ` coefficient ring; outside it, one third of the signed ring
    /// the number field sieve's balanced base-`m` expansion works in,
    /// together with [`Self::div_rem`] and [`Self::abs`].
    #[must_use]
    pub fn mul_ref(&self, other: &Self) -> Self {
        let sign = match (self.sign, other.sign) {
            (Sign::Zero, _) | (_, Sign::Zero) => Sign::Zero,
            (lhs, rhs) if lhs == rhs => Sign::Positive,
            _ => Sign::Negative,
        };
        Self::from_parts(sign, self.magnitude.mul_ref(&other.magnitude))
    }

    /// Signed division with remainder, **truncated toward zero** — the
    /// convention of C and of Rust's primitive `/` and `%`, not Python's
    /// floored one: the quotient is `self / divisor` rounded toward zero,
    /// and the remainder takes the dividend's sign (or is zero), so that
    ///
    /// ```text
    /// self = quotient·divisor + remainder,    |remainder| < |divisor|.
    /// ```
    ///
    /// Concretely, `(-7).div_rem(2) = (-3, -1)` where the floored
    /// convention would give `(-4, 1)`. A caller who wants the least
    /// non-negative residue instead uses [`Self::modulo_positive`], which
    /// is the floored remainder against an unsigned modulus.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is zero, matching [`BigUint::div_rem`].
    #[must_use]
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        let (quotient, remainder) = self.magnitude.div_rem(&divisor.magnitude);
        // Truncation is what magnitude division already does; only the
        // signs need assigning. `from_parts` canonicalizes a zero quotient
        // or remainder to `Sign::Zero`.
        (
            Self::from_parts(Self::quotient_sign(self.sign, divisor.sign), quotient),
            Self::from_parts(self.sign, remainder),
        )
    }

    /// The absolute value as an owned [`BigUint`] — a clone of what
    /// [`Self::magnitude`] lends. Reach for `magnitude()` whenever a borrow
    /// suffices (comparisons against a bound, feeding an unsigned kernel);
    /// this owned form exists for the callers that go on to consume or
    /// store `|self|` independently of `self`.
    #[must_use]
    pub fn abs(&self) -> BigUint {
        self.magnitude.clone()
    }

    /// Construct one: positive sign over a single-limb magnitude.
    #[must_use]
    pub fn one() -> Self {
        Self::from_parts(Sign::Positive, BigUint::one())
    }

    /// Whether the value is zero — a sign test, since the canonical form
    /// pairs `Sign::Zero` with an empty magnitude and admits no other zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.sign == Sign::Zero
    }

    /// Whether the value is exactly one: positive sign and a unit magnitude.
    #[must_use]
    pub fn is_one(&self) -> bool {
        self.sign == Sign::Positive && self.magnitude.is_one()
    }

    /// Exact signed quotient `self / divisor` where the division is known
    /// to leave no remainder — the interpolation and primitive-part steps
    /// of polynomial arithmetic. The sign follows the usual rule; the
    /// magnitudes divide evenly.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is zero, or (in debug) if the division is
    /// inexact.
    #[must_use]
    pub(crate) fn div_exact(&self, divisor: &Self) -> Self {
        let (quotient, remainder) = self.div_rem(divisor);
        debug_assert!(remainder.is_zero(), "div_exact requires an exact division");
        quotient
    }

    /// Exact division when it divides, `None` when it does not — the checked
    /// companion to [`Self::div_exact`], costing a single division.
    ///
    /// Callers that must *decide* divisibility (polynomial division over `ℤ`,
    /// where a step with an indivisible leading coefficient means no integer
    /// quotient exists) would otherwise divide twice: once to inspect the
    /// remainder and again to take the quotient. Knuth's Algorithm D already
    /// produces both, so this returns them together and the second division
    /// disappears.
    #[must_use]
    pub(crate) fn div_exact_checked(&self, divisor: &Self) -> Option<Self> {
        let (quotient, remainder) = self.div_rem(divisor);
        remainder.is_zero().then_some(quotient)
    }

    /// The sign of a quotient: zero numerator gives zero, like signs give a
    /// positive, unlike signs a negative.
    fn quotient_sign(numerator: Sign, divisor: Sign) -> Sign {
        match (numerator, divisor) {
            (Sign::Zero, _) => Sign::Zero,
            (lhs, rhs) if lhs == rhs => Sign::Positive,
            _ => Sign::Negative,
        }
    }

    /// Greatest common divisor of two signed integers, returned non-negative.
    ///
    /// A gcd over ℤ is only defined up to sign — `d` and `−d` are associates
    /// and divide the same set — so the convention is to name the
    /// non-negative representative. The computation is therefore a function
    /// of the magnitudes alone, and defers to [`crate::gcd`]; `gcd(0, 0)` is
    /// zero.
    #[must_use]
    pub(crate) fn gcd(&self, other: &Self) -> Self {
        Self::from_biguint(crate::number_theory::gcd(&self.magnitude, &other.magnitude))
    }

    /// `self^exponent` for a machine-word exponent, by binary
    /// exponentiation — the signed counterpart of [`BigUint::pow_u64`],
    /// carrying the sign through (a negative base to an odd power stays
    /// negative). `self^0 = 1`.
    #[must_use]
    pub(crate) fn pow_u64(&self, exponent: u64) -> Self {
        let magnitude = self.magnitude.pow_u64(exponent);
        // The sign is negative iff the base is negative and the exponent odd.
        let sign = if self.sign == Sign::Negative && exponent % 2 == 1 {
            Sign::Negative
        } else {
            Sign::Positive
        };
        Self::from_parts(sign, magnitude)
    }

    /// Reduce modulo a positive modulus and return the least non-negative
    /// residue, in `[0, modulus)`.
    ///
    /// Rust's `%` on the primitive integers truncates toward zero and gives
    /// the remainder the dividend's sign, which is not a residue: residue
    /// arithmetic needs the canonical representative of the class. A negative
    /// value is therefore folded as `modulus − (|self| mod modulus)`, with
    /// the exactly-divisible case held at zero so the result is never
    /// `modulus` itself.
    ///
    /// # Panics
    ///
    /// Panics if `modulus == 0`.
    #[must_use]
    pub fn modulo_positive(&self, modulus: &BigUint) -> BigUint {
        assert!(!modulus.is_zero(), "modulus must be non-zero");
        match self.sign {
            Sign::Zero => BigUint::zero(),
            Sign::Positive => self.magnitude.modulo(modulus),
            Sign::Negative => {
                let rem = self.magnitude.modulo(modulus);
                if rem.is_zero() {
                    BigUint::zero()
                } else {
                    modulus.sub_ref(&rem)
                }
            }
        }
    }

    /// The representative of `self` modulo `modulus` in the *symmetric* range
    /// `(−modulus/2, modulus/2]`.
    ///
    /// [`Self::modulo_positive`]'s companion, and the other canonical choice
    /// of representative. Where that one is what residue arithmetic wants,
    /// this is what *size* wants: it is the smallest representative in
    /// absolute value, which halves the magnitude of a reduced coefficient and
    /// so of everything built from one.
    ///
    /// The range is half-open at the upper end, so an exact half — possible
    /// only for even `modulus` — stays positive: `5 mod 10` is `5`, not `−5`.
    ///
    /// # Panics
    ///
    /// Panics if `modulus == 0`.
    ///
    /// ```
    /// use rump::{BigInt, BigUint};
    ///
    /// let ten = BigUint::from_u64(10);
    /// let reduced = |value: i64| BigInt::from_i64(value).symmetric_remainder(&ten);
    /// assert_eq!(reduced(7), BigInt::from_i64(-3));
    /// assert_eq!(reduced(-7), BigInt::from_i64(3));
    /// assert_eq!(reduced(5), BigInt::from_i64(5));
    /// ```
    #[must_use]
    pub fn symmetric_remainder(&self, modulus: &BigUint) -> BigInt {
        assert!(!modulus.is_zero(), "modulus must be non-zero");
        let reduced = self.modulo_positive(modulus);
        // reduced ∈ [0, modulus). Anything strictly above the midpoint belongs
        // on the negative side: subtracting the modulus lands it in
        // (−modulus/2, 0).
        if reduced.mul_ref(&BigUint::from_u64(2)) > *modulus {
            BigInt::from_biguint(reduced).sub_ref(&BigInt::from_biguint(modulus.clone()))
        } else {
            BigInt::from_biguint(reduced)
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn symmetric_remainder_is_congruent_and_smallest() {
        // The two properties that define it, checked against the other
        // representative rather than against a table: the result is congruent
        // to `modulo_positive`, and no other representative of the class is
        // smaller in absolute value.
        for m in [1u64, 2, 7, 8, 97, 1_000, 1_001] {
            let modulus = BigUint::from_u64(m);
            for value in -60i64..=60 {
                let signed = BigInt::from_i64(value);
                let symmetric = signed.symmetric_remainder(&modulus);
                assert_eq!(
                    symmetric.modulo_positive(&modulus),
                    signed.modulo_positive(&modulus),
                    "value {value} mod {m}: not the same residue"
                );
                // (−m/2, m/2]: doubled magnitude at most m, and equal only on
                // the positive side.
                let doubled = symmetric.magnitude().mul_ref(&BigUint::from_u64(2));
                assert!(doubled <= modulus, "value {value} mod {m}: not reduced");
                if doubled == modulus {
                    assert!(
                        symmetric.sign() != Sign::Negative,
                        "value {value} mod {m}: the half belongs on the positive side"
                    );
                }
            }
        }
    }

    #[test]
    fn digit_count_matches_writing_the_digits_out() {
        // The oracle is the expansion it exists to avoid producing.
        let check = |value: &BigUint, radix: u32| {
            assert_eq!(
                value.digit_count(radix),
                value.to_str_radix(radix).len(),
                "radix {radix} on {}",
                value.to_str_radix(10)
            );
        };
        for radix in [2u32, 3, 8, 10, 16, 36] {
            let base = BigUint::from_u64(u64::from(radix));
            check(&BigUint::zero(), radix);
            for value in [1u64, 2, 7, 63, 64, 65, u64::MAX] {
                check(&BigUint::from_u64(value), radix);
            }
            // The powers are the whole reason for the correction: the
            // logarithm is an integer there and its floor can fall either way.
            for exponent in 0..40u64 {
                let power = base.pow_u64(exponent);
                check(&power, radix);
                check(&power.add_ref(&BigUint::one()), radix);
                if exponent > 0 {
                    check(&power.sub_ref(&BigUint::one()), radix);
                }
            }
        }
        // And far past anything a machine word reaches.
        check(&BigUint::from_u64(10).pow_u64(3_000), 10);
    }

    #[test]
    #[should_panic(expected = "radix must be at least two")]
    fn digit_count_refuses_a_radix_below_two() {
        let _ = BigUint::from_u64(5).digit_count(1);
    }

    #[test]
    fn from_i128_is_total_including_the_minimum() {
        for value in [
            0i128,
            1,
            -1,
            i128::from(i64::MAX),
            i128::from(i64::MIN),
            i128::MAX,
        ] {
            let made = BigInt::from_i128(value);
            let expected = if value < 0 {
                BigInt::from_biguint(BigUint::from_u128(value.unsigned_abs())).negated()
            } else {
                BigInt::from_biguint(BigUint::from_u128(value.unsigned_abs()))
            };
            assert_eq!(made, expected, "from_i128({value})");
        }
        // i128::MIN has no i128 negation; its magnitude is an ordinary u128.
        let least = BigInt::from_i128(i128::MIN);
        assert_eq!(least.sign(), Sign::Negative);
        assert_eq!(*least.magnitude(), BigUint::from_u128(1u128 << 127));
    }
    use super::{
        BigInt, BigUint, MontgomeryCtx, Sign, KARATSUBA_THRESHOLD_LIMBS, SQR_KARATSUBA_MAX_LIMBS,
        SQR_SCHOOLBOOK_MIN_LIMBS, TOOM3_THRESHOLD_LIMBS, UNBALANCED_THRESHOLD_LIMBS,
    };

    fn lcg_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    }

    fn seeded_biguint(words: usize, state: &mut u64) -> BigUint {
        let mut limbs = Vec::with_capacity(words);
        for _ in 0..words {
            limbs.push(lcg_next(state));
        }
        if words > 0 && limbs[words - 1] == 0 {
            limbs[words - 1] = 1;
        }
        BigUint { limbs }
    }

    #[test]
    fn bytes_roundtrip() {
        let value =
            BigUint::from_be_bytes(&[0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22]);
        assert_eq!(
            value.to_be_bytes(),
            vec![0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22]
        );
    }

    #[test]
    fn add_sub_mul_small_values() {
        let a = BigUint::from_u128(1_000_000_000_000);
        let b = BigUint::from_u128(777_777_777_777);
        assert_eq!(a.add_ref(&b), BigUint::from_u128(1_777_777_777_777));
        assert_eq!(
            a.sub_ref(&BigUint::from_u64(1)),
            BigUint::from_u128(999_999_999_999)
        );
        assert_eq!(
            a.mul_ref(&b),
            BigUint::from_u128(777_777_777_777_000_000_000_000)
        );
    }

    /// The specification for the signed in-place operations: the flattened
    /// composition of the previous `add_ref`/`sub_ref` case analysis over
    /// the unsigned primitives, retained as a structural oracle.
    fn signed_add_oracle(a: &BigInt, b: &BigInt) -> BigInt {
        use core::cmp::Ordering;
        match (a.sign(), b.sign()) {
            (Sign::Zero, _) => b.clone(),
            (_, Sign::Zero) => a.clone(),
            (sa, sb) if sa == sb => BigInt::from_parts(sa, a.magnitude().add_ref(b.magnitude())),
            (sa, sb) => match a.magnitude().cmp(b.magnitude()) {
                Ordering::Greater => BigInt::from_parts(sa, a.magnitude().sub_ref(b.magnitude())),
                Ordering::Less => BigInt::from_parts(sb, b.magnitude().sub_ref(a.magnitude())),
                Ordering::Equal => BigInt::zero(),
            },
        }
    }

    /// The bisection sqrt_floor this crate shipped before Newton — kept as
    /// the independent oracle for the replacement.
    fn sqrt_floor_bisection(n: &BigUint) -> BigUint {
        if n.is_zero() || n.is_one() {
            return n.clone();
        }
        let mut low = BigUint::one();
        let mut high = BigUint::zero();
        high.set_bit(n.bits().div_ceil(2));
        while low.add_ref(&BigUint::one()) < high {
            let mut middle = low.add_ref(&high);
            middle.shr1();
            if middle.square_ref() <= *n {
                low = middle;
            } else {
                high = middle;
            }
        }
        low
    }

    #[test]
    fn float_estimates_match_reference() {
        // Exact on values within f64's integer range.
        let mut seed = 0xf10a_7000_0000_0001;
        for _ in 0..2000 {
            let v = lcg_next(&mut seed) >> (lcg_next(&mut seed) % 40);
            let n = BigUint::from_u64(v);
            assert_eq!(n.to_f64_lossy(), v as f64, "to_f64_lossy at {v}");
            if v != 0 {
                let approx = n.ln_approx();
                let exact = (v as f64).ln();
                assert!(
                    (approx - exact).abs() < 1e-9,
                    "ln at {v}: {approx} vs {exact}"
                );
            }
        }
        assert_eq!(BigUint::zero().to_f64_lossy(), 0.0);
        // Powers of two land exactly, well past u64.
        for bits in [64usize, 100, 200, 500, 1000] {
            let mut p = BigUint::zero();
            p.set_bit(bits);
            let expect = 2f64.powi(bits as i32);
            assert_eq!(p.to_f64_lossy(), expect, "2^{bits}");
            let ln = p.ln_approx();
            assert!(
                (ln - (bits as f64) * core::f64::consts::LN_2).abs() < 1e-6,
                "ln(2^{bits})"
            );
        }
        // Saturation above the f64 range.
        let mut huge = BigUint::zero();
        huge.set_bit(2000);
        assert_eq!(huge.to_f64_lossy(), f64::INFINITY, "2^2000 saturates");
        // ln stays finite where the value does not.
        assert!((huge.ln_approx() - 2000.0 * core::f64::consts::LN_2).abs() < 1e-3);
        // A wide non-power: 3^500, checked against ln in log space.
        let three_pow = BigUint::from_u64(3).pow_u64(500);
        let ln = three_pow.ln_approx();
        assert!((ln - 500.0 * 3f64.ln()).abs() < 1e-3, "ln(3^500)");
    }

    #[test]
    fn div_rem_u64_and_to_u64() {
        let mut seed = 0xd10e_5eed_0000_0001;
        for _ in 0..2000 {
            let words = 1 + (lcg_next(&mut seed) % 20) as usize;
            let n = seeded_biguint(words, &mut seed);
            let d = (lcg_next(&mut seed) | 1).max(2);
            let (q, r) = n.div_rem_u64(d);
            let (q_ref, r_ref) = n.div_rem(&BigUint::from_u64(d));
            assert_eq!(q, q_ref, "quotient");
            assert_eq!(BigUint::from_u64(r), r_ref, "remainder");
            // The defining identity.
            assert_eq!(
                q.mul_ref(&BigUint::from_u64(d))
                    .add_ref(&BigUint::from_u64(r)),
                n
            );
        }
        assert_eq!(
            BigUint::from_u64(100).div_rem_u64(7),
            (BigUint::from_u64(14), 2)
        );
        assert_eq!(BigUint::zero().div_rem_u64(5), (BigUint::zero(), 0));
        // to_u64: exact below 2^64, None above.
        assert_eq!(BigUint::zero().to_u64(), Some(0));
        assert_eq!(BigUint::from_u64(u64::MAX).to_u64(), Some(u64::MAX));
        let mut over = BigUint::zero();
        over.set_bit(64);
        assert_eq!(over.to_u64(), None);
        assert_eq!(
            BigUint::from_u128((1u128 << 64) - 1).to_u64(),
            Some(u64::MAX)
        );
        assert_eq!(BigUint::from_u128(1u128 << 64).to_u64(), None);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn div_rem_u64_rejects_zero() {
        let _ = BigUint::from_u64(5).div_rem_u64(0);
    }

    #[test]
    fn mod_add_sub_match_machine_arithmetic() {
        let mut seed = 0x30d5_0bad_0000_0001;
        for _ in 0..2000 {
            let m = (lcg_next(&mut seed) >> (lcg_next(&mut seed) % 32)) | 1;
            let a = lcg_next(&mut seed) >> (lcg_next(&mut seed) % 32);
            let b = lcg_next(&mut seed) >> (lcg_next(&mut seed) % 32);
            let (bm, ba, bb) = (
                BigUint::from_u64(m),
                BigUint::from_u64(a),
                BigUint::from_u64(b),
            );
            // Unreduced operands are within the contract.
            assert_eq!(
                BigUint::mod_add(&ba, &bb, &bm),
                BigUint::from_u64(((u128::from(a) + u128::from(b)) % u128::from(m)) as u64)
            );
            let expect_sub =
                (u128::from(a % m) + u128::from(m) - u128::from(b % m)) % u128::from(m);
            assert_eq!(
                BigUint::mod_sub(&ba, &bb, &bm),
                BigUint::from_u64(expect_sub as u64)
            );
        }
        // The Barrett pair delegates to the same operations.
        let ctx = super::BarrettCtx::new(&BigUint::from_u64(1000)).expect("modulus >= 2");
        assert_eq!(
            ctx.add_mod(&BigUint::from_u64(999), &BigUint::from_u64(2)),
            BigUint::from_u64(1)
        );
        assert_eq!(
            ctx.sub_mod(&BigUint::from_u64(2), &BigUint::from_u64(999)),
            BigUint::from_u64(3)
        );
        // from_i64 round-trips both signs into the canonical range.
        assert_eq!(
            BigInt::from_i64(-3).modulo_positive(&BigUint::from_u64(11)),
            BigUint::from_u64(8)
        );
        assert_eq!(
            BigInt::from_i64(3).modulo_positive(&BigUint::from_u64(11)),
            BigUint::from_u64(3)
        );
        assert_eq!(BigInt::from_i64(0), BigInt::zero());
        assert_eq!(
            BigInt::from_i64(i64::MIN),
            BigInt::from_parts(Sign::Negative, BigUint::from_u64(1u64 << 63))
        );
    }

    #[test]
    fn montgomery_domain_add_sub_match_plain_arithmetic() {
        use super::MontgomeryCtx;
        let mut seed = 0x0a0d_50b7_0000_0001;
        for &words in &[1usize, 4, 32] {
            let mut n = seeded_biguint(words, &mut seed);
            n.limbs[0] |= 1;
            let ctx = MontgomeryCtx::new(&n).expect("odd modulus");
            for _ in 0..16 {
                let a = seeded_biguint(words, &mut seed).modulo(&n);
                let b = seeded_biguint(words, &mut seed).modulo(&n);
                let (am, bm) = (ctx.encode(&a), ctx.encode(&b));
                // Linearity: the domain sum decodes to the plain sum.
                let sum = a.add_ref(&b).modulo(&n);
                assert_eq!(ctx.decode(&ctx.add_mont(&am, &bm)), sum);
                let diff = if a >= b {
                    a.sub_ref(&b)
                } else {
                    n.add_ref(&a).sub_ref(&b)
                };
                assert_eq!(ctx.decode(&ctx.sub_mont(&am, &bm)), diff);
            }
            // Boundaries: zero, self-cancellation, both identities, the
            // wrap, and the add correction's own boundary x + (n − x) = n.
            let zero = BigUint::zero();
            let top = ctx.encode(&n.sub_ref(&BigUint::one()));
            assert_eq!(ctx.sub_mont(&top, &top), zero);
            assert_eq!(ctx.add_mont(&top, &zero), top);
            assert_eq!(ctx.sub_mont(&top, &zero), top);
            assert!(
                ctx.add_mont(&top, &ctx.sub_mont(&zero, &top)).is_zero(),
                "x + (n - x) folds to exactly zero"
            );
            assert_eq!(
                ctx.decode(&ctx.add_mont(&top, &top)),
                n.sub_ref(&BigUint::from_u64(2)),
                "(n-1) + (n-1) wraps to n-2"
            );
            assert_eq!(
                ctx.decode(&ctx.sub_mont(&zero, &top)),
                BigUint::one(),
                "0 - (n-1) wraps to 1"
            );
        }
    }

    #[test]
    fn barrett_matches_division_reduction() {
        use super::BarrettCtx;
        let mut seed = 0xba22_e77e_0000_0001;
        for &words in &[1usize, 4, 16, 64] {
            for parity_even in [false, true] {
                let mut n = seeded_biguint(words, &mut seed);
                if parity_even {
                    n.limbs[0] &= !1;
                } else {
                    n.limbs[0] |= 1;
                }
                if n.bits() < 2 {
                    continue;
                }
                let ctx = BarrettCtx::new(&n).expect("modulus is at least 2");
                for _ in 0..8 {
                    // Products of reduced values — the advertised domain.
                    let a = seeded_biguint(words, &mut seed).modulo(&n);
                    let b = seeded_biguint(words, &mut seed).modulo(&n);
                    let wide = a.mul_ref(&b);
                    assert_eq!(ctx.reduce(&wide), wide.modulo(&n));
                    assert_eq!(ctx.mul_mod(&a, &b), BigUint::mod_mul(&a, &b, &n));
                    assert_eq!(ctx.square_mod(&a), BigUint::mod_mul(&a, &a, &n));
                }
                // The full-width boundary of the contract: b^2k − 1.
                let mut edge = BigUint::zero();
                edge.set_bit(128 * words);
                let edge = edge.sub_ref(&BigUint::one());
                assert_eq!(ctx.reduce(&edge), edge.modulo(&n));
                // Beyond the contract the fallback still answers.
                let mut wide = seeded_biguint(3 * words, &mut seed);
                wide.set_bit(3 * words * 64 - 1);
                assert_eq!(ctx.reduce(&wide), wide.modulo(&n));
                assert_eq!(ctx.reduce(&BigUint::zero()), BigUint::zero());
            }
        }
        // Tiny and structured moduli.
        for n_small in [2u64, 3, 4, 16, 255, 256, 257, u64::MAX] {
            let n = BigUint::from_u64(n_small);
            let ctx = BarrettCtx::new(&n).expect("at least 2");
            for x in [
                Some(0u64),
                Some(1),
                Some(n_small - 1),
                Some(n_small),
                n_small.checked_add(1),
            ]
            .into_iter()
            .flatten()
            {
                let x2 = BigUint::from_u64(x).square_ref();
                assert_eq!(ctx.reduce(&x2), x2.modulo(&n), "n = {n_small}, x = {x}");
            }
        }
        assert!(BarrettCtx::new(&BigUint::one()).is_none());
        assert!(BarrettCtx::new(&BigUint::zero()).is_none());
        // The tightest shapes for the quotient estimate: moduli at the
        // limb-boundary edges, b^(k-1) and b^k - 1.
        for k in [2usize, 3, 8] {
            for n in [
                {
                    let mut v = BigUint::zero();
                    v.set_bit(64 * (k - 1));
                    v
                },
                {
                    let mut v = BigUint::zero();
                    v.set_bit(64 * (k - 1));
                    v.add_ref(&BigUint::one())
                },
                {
                    let mut v = BigUint::zero();
                    v.set_bit(64 * k);
                    v.sub_ref(&BigUint::one())
                },
            ] {
                let ctx = BarrettCtx::new(&n).expect("at least 2");
                let mut seed2 = 0x0b0b_0b0b_0000_0001 ^ (k as u64);
                for _ in 0..6 {
                    let a = seeded_biguint(k, &mut seed2).modulo(&n);
                    let wide = a.square_ref();
                    assert_eq!(ctx.reduce(&wide), wide.modulo(&n), "edge modulus, k = {k}");
                }
            }
        }
    }

    #[test]
    fn barrett_reduce_straddles_the_half_product_cutoff() {
        // Above `BARRETT_HALF_PRODUCT_MAX_LIMBS`, `reduce` takes the window
        // from a dispatched full product instead of the schoolbook half
        // product. Nothing else in the crate builds a modulus that wide, so
        // without this test the branch is unexecuted by every gate and a
        // corrupted arm passes the entire suite.
        //
        // What this cannot catch, and no value-based test can: the two arms
        // are exactly equal by construction, so deleting the cutoff,
        // inverting its predicate, or moving it to the wrong width all
        // still compute the right answer. Only the timing changes, and only
        // above 32 kbit. The constant is guarded by the comment on it and
        // by `--ignored` measurement, not by this.
        use super::{BarrettCtx, BARRETT_HALF_PRODUCT_MAX_LIMBS};
        let cutoff = BARRETT_HALF_PRODUCT_MAX_LIMBS;
        let mut seed = 0x5a11_b0bb_0000_0001;
        for k in [cutoff - 1, cutoff, cutoff + 1, cutoff + 2] {
            // Four modulus shapes at each width, not one. `reduce` itself
            // is parity-blind — comparisons, shifts, two products and the
            // correction loop, with nothing that branches on the low bit —
            // so the even shape is not closing a class of parity defect;
            // the parity-sensitive code is in `MontgomeryCtx::new` and in
            // `mod_pow`'s routing, neither of which this test reaches. It
            // is here because an even modulus is the case this type exists
            // to serve and ought to be exercised at width somewhere, and
            // because it is another independent μ. The other two shapes do
            // carry a specific argument: a top limb of 1 maximizes μ and
            // all-ones maximizes the quotient estimate's error, so between
            // them they stress `q̂` from both ends.
            let mut shapes = Vec::new();
            let mut odd = seeded_biguint(k, &mut seed);
            odd.limbs[0] |= 1;
            odd.set_bit(64 * k - 1); // exactly k limbs, so the branch is the k tested
            let mut even = odd.clone();
            even.limbs[0] &= !1;
            shapes.push(("random odd", odd));
            shapes.push(("random even", even));
            let mut small_top = seeded_biguint(k - 1, &mut seed);
            small_top.set_bit(64 * (k - 1)); // top limb exactly 1
            shapes.push(("top limb 1", small_top));
            let mut ones = BigUint::zero();
            ones.set_bit(64 * k);
            shapes.push(("all ones", ones.sub_ref(&BigUint::one())));

            for (label, n) in shapes {
                assert_eq!(n.bits().div_ceil(64), k, "{label}: k = {k} as intended");
                let ctx = BarrettCtx::new(&n).expect("a modulus of at least 2");
                for _ in 0..2 {
                    let a = seeded_biguint(k, &mut seed).modulo(&n);
                    let b = seeded_biguint(k, &mut seed).modulo(&n);
                    let wide = a.mul_ref(&b);
                    assert_eq!(ctx.reduce(&wide), wide.modulo(&n), "{label}, k = {k}");
                }
                // The ends of the range, where an off-by-one in the
                // correction loop shows up: 0, n − 1, n itself, the largest
                // square, and the top of the accepted input range.
                assert!(ctx.reduce(&BigUint::zero()).is_zero(), "{label}, zero");
                let below = n.sub_ref(&BigUint::one());
                assert_eq!(ctx.reduce(&below), below, "{label}, n - 1");
                assert!(ctx.reduce(&n).is_zero(), "{label}, n");
                let square = below.square_ref();
                assert_eq!(ctx.reduce(&square), square.modulo(&n), "{label}, (n-1)^2");
                let mut widest = BigUint::zero();
                widest.set_bit(128 * k);
                let widest = widest.sub_ref(&BigUint::one());
                assert_eq!(
                    ctx.reduce(&widest),
                    widest.modulo(&n),
                    "{label}, the widest accepted input"
                );
            }
        }
    }

    #[test]
    #[ignore = "search for a two-correction witness; run with --ignored"]
    fn barrett_correction_search() {
        use super::BarrettCtx;
        let mut seed = 0x7777_0000_0000_0001;
        let mut seen = [0usize; 4];
        let mut witness: Option<(String, String)> = None;
        for k in 1usize..=4 {
            let mut shapes: Vec<(String, BigUint)> = Vec::new();
            for d in [1u64, 3, 5, 7, 9, 17, 33, 65, 257, 1025] {
                let mut n = BigUint::zero();
                n.set_bit(64 * (k - 1));
                shapes.push((format!("b^{}+{d}", k - 1), n.add_ref(&BigUint::from_u64(d))));
                let mut m = BigUint::zero();
                m.set_bit(64 * k);
                shapes.push((format!("b^{k}-{d}"), m.sub_ref(&BigUint::from_u64(d))));
                shapes.push((format!("b^{k}+{d}"), m.add_ref(&BigUint::from_u64(d))));
            }
            for _ in 0..40 {
                let mut r = seeded_biguint(k, &mut seed);
                r.set_bit(64 * k - 1);
                shapes.push(("random".into(), r));
            }
            for (label, n) in shapes {
                if n.bits() < 2 {
                    continue;
                }
                let ctx = BarrettCtx::new(&n).expect("ok");
                let kk = n.bits().div_ceil(64);
                let mut top = BigUint::zero();
                top.set_bit(128 * kk);
                let top = top.sub_ref(&BigUint::one());
                for _ in 0..600 {
                    let x = match seed % 3 {
                        0 => seeded_biguint(2 * kk, &mut seed).modulo(&top),
                        1 => top.sub_ref(&seeded_biguint(kk, &mut seed).modulo(&top)),
                        _ => n.mul_ref(&seeded_biguint(kk, &mut seed)).modulo(&top),
                    };
                    assert_eq!(ctx.reduce(&x), x.modulo(&n));
                    let t = BarrettCtx::last_corrections() as usize;
                    seen[t.min(3)] += 1;
                    if t == 2 && witness.is_none() {
                        witness = Some((
                            format!("{label} (k={kk}) n={}", n.to_str_radix(16)),
                            x.to_str_radix(16),
                        ));
                    }
                }
            }
        }
        println!("corrections histogram: {seen:?}");
        // The gating test cites this probe for the claim that three
        // corrections never occur, so the probe enforces it rather than
        // merely printing it — a sweep that only reports cannot support a
        // claim, which is the shape of error that let a running-maximum
        // counter conclude two was unreachable.
        assert_eq!(seen[3], 0, "HAC Note 14.44's bound of two was exceeded");
        assert!(seen[2] > 0, "the sweep must reach the bound: {seen:?}");
        if let Some((n, x)) = &witness {
            println!("witness modulus: {n}");
            println!("witness x: {x}");
        } else {
            println!("no two-correction witness found");
        }
    }

    #[test]
    fn barrett_correction_bound_is_attained_and_not_exceeded() {
        // HAC Note 14.44 bounds `q̂`'s shortfall at two, and the reduction
        // loop carries a `debug_assert` for it. A bound that is never
        // reached is indistinguishable from a bound that is wrong, so this
        // demands the tight case exist rather than merely not be exceeded.
        //
        // Finding it needs the right shapes. Two corrections are
        // concentrated on moduli just above a power of the base — `b² + 1`
        // is the readiest witness — and are missed entirely by a sweep over
        // random moduli, or by dividends drawn only from near the top of the
        // range. `barrett_correction_search`, run with `--ignored`, is the
        // wider sweep this was cut down from: 678 two-correction reductions
        // in 168 000, and no three-correction reduction at any width.
        use super::BarrettCtx;
        let mut seed = 0xc0de_1044_0000_0001;
        let mut seen = [0usize; 3];
        for k in 2usize..=4 {
            let mut shapes = Vec::new();
            for d in [1u64, 3, 5, 17, 257] {
                let mut n = BigUint::zero();
                n.set_bit(64 * (k - 1));
                shapes.push(n.add_ref(&BigUint::from_u64(d)));
            }
            let mut random = seeded_biguint(k, &mut seed);
            random.set_bit(64 * k - 1);
            shapes.push(random);

            for n in shapes {
                if n.bits() < 2 {
                    continue;
                }
                let width = n.bits().div_ceil(64);
                let ctx = BarrettCtx::new(&n).expect("a modulus of at least 2");
                let mut top = BigUint::zero();
                top.set_bit(128 * width);
                let top = top.sub_ref(&BigUint::one());
                for step in 0..250u64 {
                    // Uniform over the accepted range, and multiples of `n`
                    // near it — the two draws the witnesses come from.
                    let x = if step % 2 == 0 {
                        seeded_biguint(2 * width, &mut seed).modulo(&top)
                    } else {
                        n.mul_ref(&seeded_biguint(width, &mut seed)).modulo(&top)
                    };
                    assert_eq!(ctx.reduce(&x), x.modulo(&n), "k = {k}, step {step}");
                    let taken = BarrettCtx::last_corrections();
                    assert!(taken <= 2, "k = {k}: {taken} corrections exceeds the bound");
                    seen[taken as usize] += 1;
                }
            }
        }
        assert!(seen[2] > 0, "two corrections never occurred: {seen:?}");
        assert!(
            seen[0] > 0 && seen[1] > 0,
            "the easy cases must occur too: {seen:?}"
        );
    }

    #[test]
    fn barrett_pow_matches_mod_pow() {
        use super::BarrettCtx;
        use crate::number_theory::mod_pow;

        // A deliberately slow reference: square-and-multiply with a full
        // product and a direct division at every step, sharing no code with
        // either context. It exists because `mod_pow` now *delegates* even
        // moduli to `BarrettCtx::pow_mod` — comparing the two against each
        // other would be an identity, not a test, and the odd branch's
        // independence (Montgomery) would have quietly become the only
        // real coverage.
        fn reference_pow(base: &BigUint, exponent: &BigUint, modulus: &BigUint) -> BigUint {
            if modulus.is_one() {
                return BigUint::zero();
            }
            let mut result = BigUint::one().modulo(modulus);
            let mut power = base.modulo(modulus);
            for bit in 0..exponent.bits() {
                if exponent.bit(bit) {
                    result = result.mul_ref(&power).modulo(modulus);
                }
                power = power.mul_ref(&power).modulo(modulus);
            }
            result
        }

        let mut seed = 0xba22_e77e_0000_0002;
        for &words in &[1usize, 4, 16] {
            for parity_even in [false, true] {
                let mut n = seeded_biguint(words, &mut seed);
                if parity_even {
                    n.limbs[0] &= !1;
                } else {
                    n.limbs[0] |= 1;
                }
                if n.is_zero() || n.is_one() {
                    continue;
                }
                let ctx = BarrettCtx::new(&n).expect("at least 2");
                let base = seeded_biguint(words, &mut seed);
                let exponent = seeded_biguint(2, &mut seed);
                let expected = reference_pow(&base, &exponent, &n);
                // Both public routes against the independent ladder.
                assert_eq!(
                    ctx.pow_mod(&base, &exponent),
                    expected,
                    "BarrettCtx::pow_mod at {words} words, even = {parity_even}"
                );
                assert_eq!(
                    mod_pow(&base, &exponent, &n),
                    expected,
                    "mod_pow at {words} words, even = {parity_even}"
                );
                assert_eq!(
                    ctx.pow_mod(&base, &BigUint::zero()),
                    BigUint::one().modulo(&n)
                );
            }
        }

        // The corners the random sweep will not reach, every one an even
        // modulus so they exercise the delegated path: the smallest
        // modulus, powers of two, a non-power-of-two even modulus, a
        // multi-limb even modulus, exponents 0 and 1, and bases far wider
        // than the modulus.
        let mut wide = BigUint::one();
        wide.shl_bits(300);
        let wide_even = wide.add_ref(&BigUint::from_u64(2));
        let mut base_wider = BigUint::one();
        base_wider.shl_bits(700);
        base_wider = base_wider.add_ref(&BigUint::from_u64(12_345));
        for modulus in [
            BigUint::from_u64(2),
            BigUint::from_u64(4),
            BigUint::from_u64(1024),
            BigUint::from_u64(30),
            BigUint::from_u64(u64::MAX - 1),
            wide_even,
        ] {
            for base in [
                BigUint::zero(),
                BigUint::one(),
                BigUint::from_u64(7),
                modulus.clone(),
                base_wider.clone(),
            ] {
                for exponent in [
                    BigUint::zero(),
                    BigUint::one(),
                    BigUint::from_u64(2),
                    BigUint::from_u64(65_537),
                ] {
                    let expected = reference_pow(&base, &exponent, &modulus);
                    assert_eq!(
                        mod_pow(&base, &exponent, &modulus),
                        expected,
                        "mod_pow corner: base {base}, exponent {exponent}, modulus {modulus}"
                    );
                    let ctx = BarrettCtx::new(&modulus).expect("at least 2");
                    assert_eq!(
                        ctx.pow_mod(&base, &exponent),
                        expected,
                        "BarrettCtx corner: base {base}, exponent {exponent}, modulus {modulus}"
                    );
                }
            }
        }
    }

    #[test]
    fn sqrt_rem_matches_bisection_and_certifies() {
        let mut seed = 0x5eed_0006_0001_0001;
        for &words in &[1usize, 2, 8, 32, 128] {
            for _ in 0..6 {
                let n = seeded_biguint(words, &mut seed);
                let (root, remainder) = n.sqrt_rem();
                assert_eq!(
                    root,
                    sqrt_floor_bisection(&n),
                    "root diverged at {words} words"
                );
                assert_eq!(remainder, n.sub_ref(&root.square_ref()));
                assert!(
                    root.add_ref(&BigUint::one()).square_ref() > n,
                    "floor certificate"
                );
            }
        }
        // Exact squares and their neighbours.
        let mut seed2 = 0x0bad_cafe_0000_0007;
        for &words in &[2usize, 16, 64] {
            let r = seeded_biguint(words, &mut seed2);
            let square = r.square_ref();
            assert_eq!(square.sqrt_rem(), (r.clone(), BigUint::zero()));
            let below = square.sub_ref(&BigUint::one());
            let r_minus_one = r.sub_ref(&BigUint::one());
            assert_eq!(
                below.sqrt_rem().0,
                r_minus_one,
                "just below a square roots to r - 1"
            );
        }
    }

    #[test]
    fn predicates_match_machine_arithmetic() {
        let mut seed = 0x1234_5678_0000_0001;
        for _ in 0..3000 {
            let v = lcg_next(&mut seed) >> (lcg_next(&mut seed) % 40);
            let n = BigUint::from_u64(v);
            assert_eq!(n.popcount(), v.count_ones() as usize, "popcount at {v}");
            if v != 0 {
                assert_eq!(
                    n.trailing_zeros(),
                    Some(v.trailing_zeros() as usize),
                    "trailing_zeros at {v}"
                );
                let isqrt = v.isqrt();
                assert_eq!(n.sqrt_rem().0, BigUint::from_u64(isqrt), "sqrt at {v}");
                assert_eq!(n.is_square(), isqrt * isqrt == v, "is_square at {v}");
            }
        }
        assert_eq!(BigUint::zero().trailing_zeros(), None);
        assert_eq!(BigUint::zero().popcount(), 0);
        assert!(BigUint::zero().is_square());
    }

    #[test]
    fn nth_root_and_perfect_power_brute_force() {
        // Exhaustive over a small range: every n and k against direct search.
        for v in 1u64..2000 {
            let n = BigUint::from_u64(v);
            for k in [2u64, 3, 5, 7] {
                let mut r = 0u64;
                while (r + 1).pow(u32::try_from(k).expect("small")) <= v {
                    r += 1;
                }
                assert_eq!(n.nth_root_floor(k), BigUint::from_u64(r), "root {k} of {v}");
            }
            let mut is_power = v == 1; // 1 = 1^k, below the search's floor
            for k in 2u64..64 {
                let mut m = 2u64;
                while let Some(p) = m.checked_pow(u32::try_from(k).expect("small")) {
                    if p == v {
                        is_power = true;
                    }
                    if p >= v {
                        break;
                    }
                    m += 1;
                }
            }
            assert_eq!(n.is_perfect_power(), is_power, "perfect power at {v}");
        }
        assert!(BigUint::zero().is_perfect_power());
        assert!(BigUint::one().is_perfect_power());
    }

    #[test]
    fn wide_roots_and_powers() {
        let mut seed = 0x0f0f_0f0f_5eed_0001;
        for &(words, k) in &[(8usize, 3u64), (16, 5), (40, 2), (24, 7)] {
            let m = seeded_biguint(words, &mut seed);
            let power = m.pow_u64(k);
            assert_eq!(power.nth_root_floor(k), m, "exact {k}-th root");
            assert!(power.is_perfect_power());
            let bumped = power.add_ref(&BigUint::one());
            // m^k + 1 is a perfect power only at 8, 9: Catalan's
            // conjecture, proved by Mihăilescu (J. reine angew. Math. 572,
            // 2004) — the only consecutive perfect powers are 8 and 9 —
            // and these operands are far beyond that pair.
            assert!(!bumped.is_perfect_power(), "power + 1 at {words} words");
            assert_eq!(bumped.nth_root_floor(k), m, "root of power + 1");
        }
        // A square of a square: detected through either exponent route.
        let base = seeded_biguint(6, &mut seed);
        assert!(base.square_ref().square_ref().is_perfect_power());
    }

    #[test]
    #[should_panic(expected = "the zeroth root does not exist")]
    fn nth_root_rejects_zeroth_root() {
        let _ = BigUint::from_u64(5).nth_root_floor(0);
    }

    #[test]
    #[ignore = "timing probe for the Newton/bisection square-root comparison; run with --ignored"]
    fn sqrt_newton_vs_bisection_timing() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut seed = 0x0bad_5eed_0000_0001;
        eprintln!("{:>8} {:>12} {:>12}", "bits", "newton_us", "bisect_us");
        for &words in &[16usize, 64, 128, 1024] {
            let n = seeded_biguint(words, &mut seed);
            let time = |f: &dyn Fn()| {
                let mut best = f64::INFINITY;
                for _ in 0..9 {
                    let t0 = Instant::now();
                    f();
                    best = best.min(t0.elapsed().as_secs_f64() * 1e6);
                }
                best
            };
            let newton = time(&|| {
                black_box(n.sqrt_rem());
            });
            let bisect = time(&|| {
                black_box(sqrt_floor_bisection(&n));
            });
            eprintln!("{:>8} {newton:>12.1} {bisect:>12.1}", words * 64);
        }
    }

    #[test]
    fn radix_round_trips_across_bases() {
        let mut seed = 0x5eed_5eed_1234_5678;
        for radix in 2u32..=36 {
            for &words in &[1usize, 5, 32, 200] {
                for _ in 0..2 {
                    let value = seeded_biguint(words, &mut seed);
                    let text = value.to_str_radix(radix);
                    assert_eq!(
                        BigUint::from_str_radix(&text, radix),
                        Some(value.clone()),
                        "round trip failed at radix {radix}, {words} words"
                    );
                    assert!(
                        !text.starts_with('0') || text == "0",
                        "no leading zeros at radix {radix}"
                    );
                    let negative = BigInt::from_parts(Sign::Negative, value);
                    let text = negative.to_str_radix(radix);
                    assert!(text.starts_with('-'));
                    assert_eq!(
                        BigInt::from_str_radix(&text, radix),
                        Some(negative),
                        "signed round trip failed at radix {radix}"
                    );
                }
            }
        }
        assert_eq!(BigUint::zero().to_str_radix(10), "0");
        assert_eq!(BigInt::zero().to_str_radix(10), "0");
    }

    #[test]
    fn radix_matches_std_formatting() {
        let mut seed = 0x0123_4567_89ab_cdef;
        for _ in 0..200 {
            let v = u128::from(lcg_next(&mut seed)) << 64 | u128::from(lcg_next(&mut seed));
            let value = BigUint::from_u128(v);
            assert_eq!(value.to_str_radix(10), v.to_string());
            assert_eq!(value.to_str_radix(16), format!("{v:x}"));
            assert_eq!(value.to_str_radix(8), format!("{v:o}"));
            assert_eq!(value.to_str_radix(2), format!("{v:b}"));
            assert_eq!(value.to_string(), v.to_string());
            assert_eq!(v.to_string().parse::<BigUint>().ok(), Some(value));
        }
        // A fixed vector in the highest base: "rump" in base 36.
        assert_eq!(
            BigUint::from_str_radix("rump", 36),
            Some(BigUint::from_u64(1_299_409))
        );
        assert_eq!(BigUint::from_u64(1_299_409).to_str_radix(36), "rump");
        // A wide external vector, generated by CPython's integer formatter:
        // 10^100 in base 36 — an oracle beyond the u128 range that shares
        // nothing with this crate's engines.
        let googol =
            BigUint::from_str_radix(&format!("1{}", "0".repeat(100)), 10).expect("valid decimal");
        let base36 = "2hqbczu2ow52bala8lgc3s5y9mm5tiy0vo9tke25466gfi6ax8gs22x7kuu8l1tds";
        assert_eq!(googol.to_str_radix(36), base36);
        assert_eq!(BigUint::from_str_radix(base36, 36), Some(googol));
    }

    #[test]
    fn radix_divide_and_conquer_matches_classical() {
        let mut seed = 0xfeed_beef_dead_cafe;
        // 256 words is 3,169 digits even in base 36 — above the 1,024-digit
        // dispatch threshold for every radix here — so the production path
        // is divide and conquer throughout, with the classical engines as
        // the oracle.
        for &radix in &[3u32, 10, 36] {
            let value = seeded_biguint(256, &mut seed);
            let classical = value.to_digits_classical(radix);
            let text = value.to_str_radix(radix);
            let rendered: Vec<u8> = text
                .bytes()
                .map(|b| {
                    u8::try_from(char::from(b).to_digit(radix).expect("own digits are valid"))
                        .expect("digit fits")
                })
                .collect();
            assert_eq!(rendered, classical, "render diverged at radix {radix}");
            assert_eq!(
                BigUint::from_digits_classical(&classical, radix),
                value,
                "classical parse diverged at radix {radix}"
            );
            assert_eq!(
                BigUint::from_str_radix(&text, radix),
                Some(value),
                "dispatched parse diverged at radix {radix}"
            );
        }
        // The big-value sweep for the dispatched path.
        for &words in &[300usize, 500] {
            let value = seeded_biguint(words, &mut seed);
            let text = value.to_str_radix(10);
            assert_eq!(BigUint::from_str_radix(&text, 10), Some(value));
        }
    }

    #[test]
    fn radix_rejects_malformed_input() {
        assert_eq!(BigUint::from_str_radix("", 10), None);
        assert_eq!(BigUint::from_str_radix("12a", 10), None);
        assert_eq!(BigUint::from_str_radix("z", 35), None);
        assert_eq!(
            BigUint::from_str_radix("z", 36),
            Some(BigUint::from_u64(35))
        );
        assert_eq!(BigUint::from_str_radix("+5", 10), None);
        assert_eq!(BigUint::from_str_radix(" 5", 10), None);
        assert_eq!(BigUint::from_str_radix("0x10", 10), None);
        assert_eq!(
            BigUint::from_str_radix("0007", 10),
            Some(BigUint::from_u64(7))
        );
        assert_eq!(
            BigUint::from_str_radix("FF", 16),
            Some(BigUint::from_u64(255))
        );
        assert_eq!(BigInt::from_str_radix("-", 10), None);
        assert_eq!(BigInt::from_str_radix("-0", 10), Some(BigInt::zero()));
        assert_eq!(
            BigInt::from_str_radix("-7", 10),
            Some(BigInt::from_parts(Sign::Negative, BigUint::from_u64(7)))
        );
        assert_eq!(
            "-42".parse::<BigInt>().map(|v| v.to_string()),
            Ok("-42".into())
        );
        assert!("".parse::<BigUint>().is_err());
    }

    #[test]
    #[should_panic(expected = "radix must be in 2..=36")]
    fn radix_rejects_radix_one() {
        let _ = BigUint::from_str_radix("0", 1);
    }

    #[test]
    #[should_panic(expected = "radix must be in 2..=36")]
    fn radix_rejects_radix_thirty_seven() {
        let _ = BigUint::from_u64(1).to_str_radix(37);
    }

    #[test]
    #[ignore = "timing probe for the classical/divide-and-conquer radix crossover; run with --ignored"]
    fn radix_dc_crossover_timing() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut seed = 0x7157_ab1e_5eed_0001;
        eprintln!(
            "{:>8} {:>8} {:>12} {:>12} {:>12} {:>12}",
            "words", "digits", "to_cl_ms", "to_dc_ms", "from_cl_ms", "from_dc_ms"
        );
        for &words in &[32usize, 64, 128, 256, 512, 1024, 2048] {
            let value = seeded_biguint(words, &mut seed);
            let digits = value.to_digits_classical(10);
            let time = |f: &dyn Fn()| {
                let mut best = f64::INFINITY;
                for _ in 0..9 {
                    let t0 = Instant::now();
                    f();
                    best = best.min(t0.elapsed().as_secs_f64() * 1e3);
                }
                best
            };
            let to_cl = time(&|| {
                black_box(value.to_digits_classical(10));
            });
            let to_dc = time(&|| {
                black_box(value.to_digits_dc(10));
            });
            let from_cl = time(&|| {
                black_box(BigUint::from_digits_classical(&digits, 10));
            });
            let from_dc = time(&|| {
                black_box(BigUint::from_digits_dc(&digits, 10));
            });
            // Base-case sweeps for both recursions, bypassing the dispatch
            // thresholds so the floors' own effects are visible.
            let (ladder, chunk) = BigUint::radix_power_ladder(10, digits.len());
            let bases: Vec<f64> = [512usize, 1024, 2048, 4096]
                .iter()
                .map(|&b| {
                    time(&|| {
                        black_box(BigUint::from_digits_ladder(&digits, 10, &ladder, chunk, b));
                    })
                })
                .collect();
            let (rladder, rchunk) = BigUint::radix_power_ladder_bits(10, value.bits());
            let rbases: Vec<f64> = [512usize, 1024, 2048, 4096]
                .iter()
                .map(|&b| {
                    time(&|| {
                        black_box(value.to_digits_ladder(10, &rladder, rchunk, b));
                    })
                })
                .collect();
            eprintln!(
                "{words:>8} {:>8} {to_cl:>12.3} {to_dc:>12.3} {from_cl:>12.3} {from_dc:>12.3}  parse[.5k,1k,2k,4k]={bases:.3?} render[.5k,1k,2k,4k]={rbases:.3?}",
                digits.len()
            );
        }
    }

    #[test]
    fn assign_add_sub_match_two_operand_forms() {
        let mut seed = 0x5851_f42d_4c95_7f2d;
        let mut out = BigUint::zero();
        for &(wa, wb) in &[(0usize, 0usize), (1, 1), (1, 48), (48, 1), (8, 8), (48, 48)] {
            for _ in 0..12 {
                let a = seeded_biguint(wa, &mut seed);
                let b = seeded_biguint(wb, &mut seed);
                out.assign_add(&a, &b);
                assert_eq!(out, a.add_ref(&b));
                assert!(out.limbs.last() != Some(&0), "canonical form");
                let (hi, lo) = if a >= b { (&a, &b) } else { (&b, &a) };
                out.assign_sub(hi, lo);
                assert_eq!(out, hi.sub_ref(lo));
                assert!(out.limbs.last() != Some(&0), "canonical form");
            }
        }
        // A full carry ripple: (2^(64k) - 1) + 1 = 2^(64k).
        let ones = BigUint {
            limbs: vec![u64::MAX; 5],
        };
        let one = BigUint::from_u64(1);
        out.assign_add(&ones, &one);
        let mut expect = BigUint::zero();
        expect.set_bit(320);
        assert_eq!(out, expect);
        // And the borrow ripple back down.
        out.assign_sub(&expect, &one);
        assert_eq!(out, ones);
    }

    #[test]
    fn assign_add_reuses_the_buffer() {
        let mut seed = 0x0123_4567_89ab_cdef;
        let a = seeded_biguint(32, &mut seed);
        let b = seeded_biguint(32, &mut seed);
        // The first call may grow the buffer once (the result width plus the
        // carry slot); from then on the no-allocation contract holds.
        let mut out = BigUint::zero();
        out.assign_add(&a, &b);
        let ptr = out.limbs.as_ptr();
        for _ in 0..8 {
            out.assign_add(&a, &b);
            assert_eq!(out.limbs.as_ptr(), ptr, "assign_add must not reallocate");
            out.assign_sub(&a, &b.sub_ref(&b)); // a - 0 = a, exercising short rhs
            assert_eq!(out.limbs.as_ptr(), ptr, "assign_sub must not reallocate");
        }
    }

    #[test]
    #[should_panic(expected = "BigUint underflow")]
    fn assign_sub_panics_on_underflow() {
        let mut out = BigUint::zero();
        out.assign_sub(&BigUint::from_u64(3), &BigUint::from_u64(5));
    }

    /// The verification counterpart of the crate's audited scrub
    /// exception: reading a buffer's abandoned tail cannot be expressed in
    /// safe Rust, so proving the shrink paths scrub it requires one raw
    /// read-back. Confined to this test; the pointers are captured while
    /// the limbs are live and the buffer's identity is asserted unchanged
    /// before each read.
    #[test]
    #[allow(unsafe_code)]
    fn shrinking_paths_scrub_abandoned_limbs() {
        let read8 =
            |p: *const u64| -> Vec<u64> { (0..8).map(|i| unsafe { p.add(i).read() }).collect() };
        let wide = BigUint {
            limbs: vec![0xdead_beef_0bad_cafe; 8],
        };
        let narrow = BigUint::from_u64(1);

        let mut x = wide.clone();
        let p = x.limbs.as_ptr();
        x.clone_from(&narrow);
        assert_eq!(x.limbs.as_ptr(), p, "clone_from reuses the buffer");
        assert!(
            read8(p)[1..].iter().all(|&w| w == 0),
            "clone_from stranded live limbs"
        );

        let mut out = wide.clone();
        let p = out.limbs.as_ptr();
        out.assign_add(&narrow, &narrow);
        assert_eq!(out.limbs.as_ptr(), p, "assign_add reuses the buffer");
        assert!(
            read8(p)[1..].iter().all(|&w| w == 0),
            "assign_add stranded live limbs"
        );

        let mut out2 = wide.clone();
        let p = out2.limbs.as_ptr();
        out2.assign_sub(&narrow, &narrow);
        assert_eq!(out2.limbs.as_ptr(), p, "assign_sub reuses the buffer");
        assert!(
            read8(p).iter().all(|&w| w == 0),
            "assign_sub stranded live limbs"
        );

        let a = BigInt::from_parts(Sign::Positive, wide.clone());
        let mut z = a.clone();
        let p = z.magnitude.limbs.as_ptr();
        z.sub_assign_ref(&a);
        assert_eq!(
            z.magnitude.limbs.as_ptr(),
            p,
            "cancellation keeps the buffer"
        );
        assert!(
            read8(p).iter().all(|&w| w == 0),
            "cancellation stranded live limbs"
        );
    }

    #[test]
    fn shrinking_paths_stay_canonical_and_keep_capacity() {
        let mut seed = 0xdead_beef_0bad_cafe;
        let wide = seeded_biguint(8, &mut seed);
        let narrow = seeded_biguint(2, &mut seed);
        let mut x = wide.clone();
        x.clone_from(&narrow);
        assert_eq!(x, narrow);
        let mut out = wide.clone();
        out.assign_add(&narrow, &narrow);
        assert_eq!(out, narrow.add_ref(&narrow));
        let mut out2 = wide.clone();
        out2.assign_sub(&narrow, &narrow);
        assert!(out2.is_zero());
        assert!(out2.limbs.last() != Some(&0), "canonical zero is empty");
        // Cancellation clears to canonical zero with capacity kept.
        let mut z = BigInt::from_parts(Sign::Positive, wide.clone());
        z.sub_assign_ref(&BigInt::from_parts(Sign::Positive, wide.clone()));
        assert_eq!(z, BigInt::zero());
        assert!(
            z.magnitude().limbs.capacity() >= 8,
            "capacity kept for reuse"
        );
    }

    #[test]
    fn clone_from_reuses_and_matches() {
        let mut seed = 0xfeed_face_cafe_beef;
        let big = seeded_biguint(48, &mut seed);
        let small = seeded_biguint(3, &mut seed);
        let mut x = big.clone();
        let capacity = x.limbs.capacity();
        let ptr = x.limbs.as_ptr();
        x.clone_from(&small);
        assert_eq!(x, small);
        assert_eq!(x.limbs.capacity(), capacity, "shrinking keeps the buffer");
        assert_eq!(x.limbs.as_ptr(), ptr);
        x.clone_from(&big);
        assert_eq!(x, big, "regrowing within capacity restores the value");
        assert_eq!(
            x.limbs.as_ptr(),
            ptr,
            "regrowth within capacity must not reallocate"
        );
        let mut y = BigInt::from_parts(Sign::Negative, big.clone());
        y.clone_from(&BigInt::from_parts(Sign::Positive, small.clone()));
        assert_eq!(y, BigInt::from_parts(Sign::Positive, small));
    }

    #[test]
    fn signed_in_place_matches_case_analysis_oracle() {
        let mut seed = 0x2545_f491_4f6c_dd1d;
        let signed = |sign, words: usize, seed: &mut u64| {
            if words == 0 {
                BigInt::zero()
            } else {
                BigInt::from_parts(sign, seeded_biguint(words, seed))
            }
        };
        let mut cases: Vec<(BigInt, BigInt)> = Vec::new();
        for &sa in &[Sign::Positive, Sign::Negative] {
            for &sb in &[Sign::Positive, Sign::Negative] {
                for &(wa, wb) in &[(0usize, 6usize), (6, 0), (0, 0), (1, 6), (6, 1), (6, 6)] {
                    for _ in 0..6 {
                        cases.push((signed(sa, wa, &mut seed), signed(sb, wb, &mut seed)));
                    }
                }
                // Exact cancellation: equal magnitudes, opposite signs.
                let m = seeded_biguint(5, &mut seed);
                cases.push((BigInt::from_parts(sa, m.clone()), BigInt::from_parts(sb, m)));
            }
        }
        for (a, b) in &cases {
            let mut sum = a.clone();
            sum.add_assign_ref(b);
            assert_eq!(
                sum,
                signed_add_oracle(a, b),
                "add: {:?} + {:?}",
                a.sign(),
                b.sign()
            );
            let mut diff = a.clone();
            diff.sub_assign_ref(b);
            assert_eq!(
                diff,
                signed_add_oracle(a, &b.negated()),
                "sub: {:?} - {:?}",
                a.sign(),
                b.sign()
            );
            // Canonical zero: Sign::Zero with an empty magnitude.
            if sum.sign() == Sign::Zero {
                assert!(sum.magnitude().is_zero());
            }
            if diff.sign() == Sign::Zero {
                assert!(diff.magnitude().is_zero());
            }
        }
    }

    #[test]
    fn signed_arithmetic_matches_i128() {
        let mut seed = 0x9e37_79b9_7f4a_7c15;
        let to_bigint = |v: i64| {
            let sign = if v > 0 {
                Sign::Positive
            } else if v < 0 {
                Sign::Negative
            } else {
                Sign::Zero
            };
            BigInt::from_parts(sign, BigUint::from_u64(v.unsigned_abs()))
        };
        let to_i128 = |v: &BigInt| -> i128 {
            let mag = v.magnitude().limbs.first().copied().unwrap_or(0);
            match v.sign() {
                Sign::Negative => -i128::from(mag),
                _ => i128::from(mag),
            }
        };
        for _ in 0..4000 {
            let a = lcg_next(&mut seed) as i64 >> 8;
            let b = lcg_next(&mut seed) as i64 >> 8;
            let (ba, bb) = (to_bigint(a), to_bigint(b));
            let mut sum = ba.clone();
            sum.add_assign_ref(&bb);
            assert_eq!(to_i128(&sum), i128::from(a) + i128::from(b));
            let mut diff = ba.clone();
            diff.sub_assign_ref(&bb);
            assert_eq!(to_i128(&diff), i128::from(a) - i128::from(b));
        }
    }

    #[test]
    fn signed_ring_matches_i128() {
        // The public signed ring (mul_ref, div_rem, abs) against i128, with
        // the division convention pinned: truncated toward zero, remainder
        // taking the dividend's sign — i128's own convention, so the oracle
        // is the primitive operators.
        let mut seed = 0x517e_d00d_0000_0001;
        let to_bigint = |v: i64| {
            let sign = if v > 0 {
                Sign::Positive
            } else if v < 0 {
                Sign::Negative
            } else {
                Sign::Zero
            };
            BigInt::from_parts(sign, BigUint::from_u64(v.unsigned_abs()))
        };
        let to_i128 = |v: &BigInt| -> i128 {
            let mag = v.magnitude().limbs.first().copied().unwrap_or(0);
            match v.sign() {
                Sign::Negative => -i128::from(mag),
                _ => i128::from(mag),
            }
        };
        for _ in 0..4000 {
            let a = lcg_next(&mut seed) as i64 >> 34;
            let b = lcg_next(&mut seed) as i64 >> 34;
            let (ba, bb) = (to_bigint(a), to_bigint(b));
            assert_eq!(
                to_i128(&ba.mul_ref(&bb)),
                i128::from(a) * i128::from(b),
                "mul {a} * {b}"
            );
            assert_eq!(ba.abs(), BigUint::from_u64(a.unsigned_abs()), "abs {a}");
            if b != 0 {
                let (q, r) = ba.div_rem(&bb);
                assert_eq!(to_i128(&q), i128::from(a / b), "quotient {a} / {b}");
                assert_eq!(to_i128(&r), i128::from(a % b), "remainder {a} % {b}");
            }
        }
        // The named corner from the documentation: truncated, not floored.
        let minus_seven = to_bigint(-7);
        let two = to_bigint(2);
        let (q, r) = minus_seven.div_rem(&two);
        assert_eq!(to_i128(&q), -3);
        assert_eq!(to_i128(&r), -1);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn signed_div_rem_panics_on_zero_divisor() {
        let _ = BigInt::one().div_rem(&BigInt::zero());
    }

    #[test]
    fn square_ref_matches_mul_ref() {
        // The squaring ladder against the multiplication it specializes, at
        // every width where its dispatch changes hands — one limb either
        // side of `SQR_SCHOOLBOOK_MIN_LIMBS` (multiplication vs schoolbook
        // squaring), of the Karatsuba threshold (schoolbook vs Karatsuba
        // squaring), and of `SQR_KARATSUBA_MAX_LIMBS` (Karatsuba squaring
        // vs handing back to the multiply ladder) — plus odd widths, which
        // make the split halves unequal, and operands with interior zeros
        // and all-ones limbs.
        let mut seed = 0x9e37_79b9_7f4a_7c15;
        let k = KARATSUBA_THRESHOLD_LIMBS;
        let smin = SQR_SCHOOLBOOK_MIN_LIMBS;
        let smax = SQR_KARATSUBA_MAX_LIMBS;
        let widths = [
            1usize,
            2,
            3,
            smin - 1,
            smin,
            smin + 1,
            31,
            k - 1,
            k,
            k + 1,
            2 * k + 1,
            TOOM3_THRESHOLD_LIMBS,
            smax - 1,
            smax,
            smax + 1,
        ];
        for words in widths {
            for _ in 0..4 {
                let value = seeded_biguint(words, &mut seed);
                assert_eq!(
                    value.square_ref(),
                    value.mul_ref(&value),
                    "square != mul at {words} limbs"
                );
                // And against the schoolbook kernel directly, so a defect
                // shared by both dispatched paths cannot hide.
                assert_eq!(
                    value.square_ref(),
                    BigUint::mul_schoolbook_ref(&value, &value),
                    "square != schoolbook at {words} limbs"
                );
            }
        }
        // Interior zero limbs, which make whole rows of the cross-term
        // pass vanish. (A *trailing* run of zeros cannot be built: the
        // constructor normalizes it away, which is also why the Karatsuba
        // squaring needs no empty-high-half bail-out — see its assertion.)
        let mut limbs = seeded_biguint(k + 4, &mut seed).limbs().to_vec();
        for limb in &mut limbs[2..(k + 4) / 2] {
            *limb = 0;
        }
        limbs[0] |= 1;
        let holed = BigUint::from_limbs(limbs);
        assert_eq!(holed.square_ref(), holed.mul_ref(&holed));
        // All-ones operands: the worst case for every carry chain in the
        // three passes, and the shape that would expose a doubling that
        // overflowed its buffer.
        for words in [1usize, 2, 8, k, k + 1, 2 * k] {
            let ones = BigUint::from_limbs(vec![u64::MAX; words]);
            assert_eq!(
                ones.square_ref(),
                BigUint::mul_schoolbook_ref(&ones, &ones),
                "all-ones square at {words} limbs"
            );
        }
        assert!(BigUint::zero().square_ref().is_zero());
        assert_eq!(BigUint::one().square_ref(), BigUint::one());
    }

    #[test]
    fn karatsuba_dispatch_matches_schoolbook() {
        let mut seed = 0x243f_6a88_85a3_08d3;
        for words in [32usize, 40, 64] {
            for _ in 0..6 {
                let lhs = seeded_biguint(words, &mut seed);
                let rhs = seeded_biguint(words, &mut seed);
                let dispatched = lhs.mul_ref(&rhs);
                let schoolbook = BigUint::mul_schoolbook_ref(&lhs, &rhs);
                assert_eq!(dispatched, schoolbook);
            }
        }
    }

    #[test]
    fn toom3_matches_schoolbook_across_shapes() {
        let mut seed = 0x1357_9bdf_2468_ace0;
        // Exercise the Toom-3 kernel directly — including well below the
        // dispatch threshold, at sizes not divisible by three, and with heavy
        // imbalance (one operand collapsing to a single Toom part) — against
        // the schoolbook oracle it must reproduce exactly.
        let sizes = [
            3usize, 4, 5, 7, 8, 9, 16, 31, 33, 48, 64, 65, 96, 127, 130, 200,
        ];
        for &la in &sizes {
            for &lb in &sizes {
                for _ in 0..3 {
                    let a = seeded_biguint(la, &mut seed);
                    let b = seeded_biguint(lb, &mut seed);
                    assert_eq!(
                        a.mul_toom3_ref(&b),
                        BigUint::mul_schoolbook_ref(&a, &b),
                        "toom3 != schoolbook for {la}x{lb} words"
                    );
                }
            }
        }
        // Full dispatch (Toom-3 for large balanced operands) and squaring.
        for &words in &[64usize, 96, 150, 256] {
            for _ in 0..4 {
                let a = seeded_biguint(words, &mut seed);
                let b = seeded_biguint(words, &mut seed);
                assert_eq!(a.mul_ref(&b), BigUint::mul_schoolbook_ref(&a, &b));
                assert_eq!(a.square_ref(), BigUint::mul_schoolbook_ref(&a, &a));
            }
        }
    }

    #[test]
    fn toom4_matches_schoolbook_across_shapes() {
        let mut seed = 0x0f0f_1e1e_2d2d_3c3c;
        // Direct Toom-4 kernel exercise: sizes not divisible by four, heavy
        // imbalance (a short operand collapsing to fewer Toom parts), and sizes
        // straddling its dispatch threshold — all against the schoolbook oracle.
        let sizes = [4usize, 5, 6, 7, 9, 13, 16, 33, 64, 128, 256, 260, 384, 500];
        for &la in &sizes {
            for &lb in &sizes {
                for _ in 0..2 {
                    let a = seeded_biguint(la, &mut seed);
                    let b = seeded_biguint(lb, &mut seed);
                    assert_eq!(
                        a.mul_toom4_ref(&b),
                        BigUint::mul_schoolbook_ref(&a, &b),
                        "toom4 != schoolbook for {la}x{lb} words"
                    );
                }
            }
        }
        // Full dispatch at Toom-4 sizes, plus squaring, plus a Toom-4 call that
        // recurses (its n/4 parts themselves crossing the Toom-3 threshold).
        for &words in &[256usize, 300, 512, 768] {
            for _ in 0..3 {
                let a = seeded_biguint(words, &mut seed);
                let b = seeded_biguint(words, &mut seed);
                assert_eq!(a.mul_ref(&b), BigUint::mul_schoolbook_ref(&a, &b));
                assert_eq!(a.square_ref(), BigUint::mul_schoolbook_ref(&a, &a));
            }
        }
    }

    #[test]
    fn unbalanced_matches_schoolbook_across_shapes() {
        let mut seed = 0x9e37_79b9_7f4a_7c15;
        // The block-decomposition kernel directly, below and above its
        // dispatch threshold. 64×32 is the exact boundary the balanced
        // admission excludes; 100×32 leaves a short final digit; 129×32 a
        // one-limb one; the larger shapes recurse through several balanced
        // kernels.
        for &(la, lb) in &[
            (64usize, 32usize),
            (100, 32),
            (129, 32),
            (320, 40),
            (96, 48),
            (256, 128),
            (300, 130),
        ] {
            for _ in 0..3 {
                let a = seeded_biguint(la, &mut seed);
                let b = seeded_biguint(lb, &mut seed);
                assert_eq!(
                    a.mul_unbalanced_ref(&b),
                    BigUint::mul_schoolbook_ref(&a, &b),
                    "unbalanced != schoolbook for {la}x{lb} words"
                );
                // Commutativity of the dispatch: the same pair in either order.
                assert_eq!(b.mul_ref(&a), BigUint::mul_schoolbook_ref(&a, &b));
            }
        }
        // The dispatch boundary, table-driven around the threshold: the
        // exact 2:1 ratio must go to the block decomposition (Karatsuba's
        // admission is strict, and its kernel would find an empty high half
        // there), one limb under 2:1 must go to Karatsuba, and one limb
        // under the threshold must fall back to schoolbook — checked by the
        // predicates AND by value, so admission cannot silently regress.
        let t = UNBALANCED_THRESHOLD_LIMBS;
        for (long_len, short_len, unbal, kara) in [
            (2 * t, t, true, false),      // exact 2:1 at the threshold
            (2 * t - 1, t, false, true),  // one limb under 2:1
            (2 * t, t - 1, false, false), // one limb under the threshold
            (2 * (t - 1), t - 1, false, false),
        ] {
            let a = seeded_biguint(long_len, &mut seed);
            let b = seeded_biguint(short_len, &mut seed);
            assert_eq!(
                BigUint::should_use_unbalanced(&a, &b),
                unbal,
                "unbalanced admission at {long_len}x{short_len}"
            );
            assert_eq!(
                BigUint::should_use_karatsuba(&a, &b),
                kara,
                "karatsuba admission at {long_len}x{short_len}"
            );
            assert_eq!(a.mul_ref(&b), BigUint::mul_schoolbook_ref(&a, &b));
        }
        // A longer operand containing an all-zero digit block, which the
        // kernel skips: build it by clearing the middle limbs.
        let a = seeded_biguint(96, &mut seed);
        let b = seeded_biguint(32, &mut seed);
        let mut limbs = a.limbs().to_vec();
        for limb in &mut limbs[32..64] {
            *limb = 0;
        }
        let a = BigUint::from_limbs(limbs);
        assert_eq!(
            a.mul_unbalanced_ref(&b),
            BigUint::mul_schoolbook_ref(&a, &b)
        );
    }

    /// An odd modulus of exactly `limbs` limbs, for the Montgomery tests.
    fn seeded_odd_modulus(limbs: usize, state: &mut u64) -> BigUint {
        let mut n = seeded_biguint(limbs, state);
        n.limbs[0] |= 1;
        n
    }

    #[test]
    fn mont_workspace_variants_match_allocating_forms() {
        // The with_workspace wrappers against their allocating twins and the
        // plain modular product, with ONE buffer shared across widths and
        // across both methods in both orders — so the resize-only-grow path,
        // the smaller-window-after-larger path, and stale contents from a
        // previous width are all exercised. The width order is deliberately
        // non-monotonic: a narrower modulus following a wider one is the
        // only shape that hands the kernels an over-long buffer (an
        // ascending sweep always resizes to an exact fit), and the 16 → 2
        // and 8 → 1 steps force that shape for both kernels. copy_padded
        // zero-fills and the kernels clear their scratch on entry, so none
        // of it may show.
        let mut seed = 0x0dd5_eed0_0000_0001;
        let mut ws: Vec<u64> = Vec::new();
        for &limbs in &[16usize, 2, 3, 8, 1, 5] {
            let n = seeded_odd_modulus(limbs, &mut seed);
            let ctx = MontgomeryCtx::new(&n).expect("odd modulus");
            let mut x = ctx.encode(&seeded_biguint(limbs, &mut seed));
            let y = ctx.encode(&seeded_biguint(limbs, &mut seed));
            for round in 0..200 {
                // Alternate the call order so each window size follows the
                // other's leftovers.
                let next = if round % 2 == 0 {
                    let m = ctx.mul_mont_with_workspace(&x, &y, &mut ws);
                    assert_eq!(m, ctx.mul_mont(&x, &y), "mul at {limbs} limbs");
                    let s = ctx.square_mont_with_workspace(&m, &mut ws);
                    assert_eq!(s, ctx.square_mont(&m), "sqr at {limbs} limbs");
                    s
                } else {
                    let s = ctx.square_mont_with_workspace(&x, &mut ws);
                    assert_eq!(s, ctx.square_mont(&x), "sqr at {limbs} limbs");
                    let m = ctx.mul_mont_with_workspace(&s, &y, &mut ws);
                    assert_eq!(m, ctx.mul_mont(&s, &y), "mul at {limbs} limbs");
                    m
                };
                x = next;
            }
            // Decoded agreement with the reduction-based product: the domain
            // arithmetic and the ordinary arithmetic name the same value.
            let product = ctx.mul_mont_with_workspace(&x, &y, &mut ws);
            assert_eq!(
                ctx.decode(&product),
                BigUint::mod_mul(&ctx.decode(&x), &ctx.decode(&y), &n),
                "domain product decodes to the modular product at {limbs} limbs"
            );
        }
    }

    #[test]
    #[ignore = "timing probe for the with_workspace docs; run with --ignored"]
    fn mont_workspace_timing() {
        use std::hint::black_box;
        use std::time::Instant;

        /// One pass: alternate the two forms in short chunks across the
        /// whole pass, so slow drift (thermal, frequency scaling) hits both
        /// sides equally instead of landing on whichever ran second; the
        /// naive measure-A-then-measure-B design was shown to report up to
        /// 20% on a function compared against itself. Returns the saving
        /// of `b` over `a` in percent.
        fn paired_saving(chunks: u32, chunk: u32, a: &mut dyn FnMut(), b: &mut dyn FnMut()) -> f64 {
            let (mut total_a, mut total_b) = (0f64, 0f64);
            for _ in 0..chunks {
                let t = Instant::now();
                for _ in 0..chunk {
                    a();
                }
                total_a += t.elapsed().as_secs_f64();
                let t = Instant::now();
                for _ in 0..chunk {
                    b();
                }
                total_b += t.elapsed().as_secs_f64();
            }
            (total_a - total_b) / total_a * 100.0
        }

        /// Five interleaved passes; print every pass so the spread is
        /// visible, and report the median as the figure. A claimed saving
        /// smaller than the printed spread is noise and must be quoted as
        /// noise.
        fn report(label: &str, limbs: usize, mut passes: [f64; 5]) {
            passes.sort_by(f64::total_cmp);
            eprintln!(
                "{limbs:>6} {label} median {:+6.1}%  passes {:+5.1} {:+5.1} {:+5.1} {:+5.1} {:+5.1}",
                passes[2], passes[0], passes[1], passes[2], passes[3], passes[4]
            );
        }

        let mut seed = 0x0dd5_eed0_0000_0002;
        for &limbs in &[1usize, 4, 8, 32, 64] {
            let n = seeded_odd_modulus(limbs, &mut seed);
            let ctx = MontgomeryCtx::new(&n).expect("odd modulus");
            let a = ctx.encode(&seeded_biguint(limbs, &mut seed));
            let b = ctx.encode(&seeded_biguint(limbs, &mut seed));
            let chunk = 256u32;
            let chunks = ((2_000_000 / (limbs * limbs)).max(4_000) as u32 / chunk).max(8);

            let mut ws = Vec::new();
            let mut sqr_passes = [0f64; 5];
            for pass in &mut sqr_passes {
                *pass = paired_saving(
                    chunks,
                    chunk,
                    &mut || {
                        black_box(ctx.square_mont(black_box(&a)));
                    },
                    &mut || {
                        black_box(ctx.square_mont_with_workspace(black_box(&a), &mut ws));
                    },
                );
            }
            report("sqr", limbs, sqr_passes);

            let mut ws2 = Vec::new();
            let mut mul_passes = [0f64; 5];
            for pass in &mut mul_passes {
                *pass = paired_saving(
                    chunks,
                    chunk,
                    &mut || {
                        black_box(ctx.mul_mont(black_box(&a), black_box(&b)));
                    },
                    &mut || {
                        black_box(ctx.mul_mont_with_workspace(
                            black_box(&a),
                            black_box(&b),
                            &mut ws2,
                        ));
                    },
                );
            }
            report("mul", limbs, mul_passes);
        }
    }

    #[test]
    #[ignore = "timing probe for tuning UNBALANCED_THRESHOLD_LIMBS; run with --ignored"]
    fn unbalanced_crossover_timing() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut seed = 0x5eed_5eed_5eed_5eed;
        eprintln!(
            "{:>6} {:>6} {:>12} {:>12}  best",
            "long", "short", "school", "unbal"
        );
        for &short in &[32usize, 48, 64, 96, 128, 192, 256, 384, 512] {
            for &ratio in &[2usize, 4, 16] {
                let long = short * ratio;
                let a = seeded_biguint(long, &mut seed);
                let b = seeded_biguint(short, &mut seed);
                let reps = (200_000 / (long * short / 64)).max(3);
                let time = |f: &dyn Fn() -> BigUint| {
                    let mut best = f64::INFINITY;
                    for _ in 0..5 {
                        let t = Instant::now();
                        for _ in 0..reps {
                            black_box(f());
                        }
                        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
                    }
                    best
                };
                let school = time(&|| BigUint::mul_schoolbook_ref(&a, &b));
                let unbal = time(&|| a.mul_unbalanced_ref(&b));
                eprintln!(
                    "{:>6} {:>6} {:>10.3}us {:>10.3}us  {}",
                    long,
                    short,
                    school * 1e6,
                    unbal * 1e6,
                    if unbal < school { "unbal" } else { "school" }
                );
            }
        }
    }

    #[test]
    fn mul_low_ref_matches_the_truncated_full_product() {
        // The half-product against the full product it replaces, truncated
        // to the same window: limits below, at, and above each operand's
        // width, and the degenerate limit of zero.
        let mut seed = 0x1010_7ef7_0000_0001;
        for &(la, lb) in &[(1usize, 1usize), (2, 3), (4, 4), (8, 5), (17, 16), (32, 32)] {
            let a = seeded_biguint(la, &mut seed);
            let b = seeded_biguint(lb, &mut seed);
            let full = a.mul_ref(&b);
            for limit in 0..=(la + lb + 1) {
                let expected = if limit == 0 {
                    BigUint::zero()
                } else {
                    full.low_bits(64 * limit)
                };
                assert_eq!(
                    BigUint::mul_low_ref(&a, &b, limit),
                    expected,
                    "mul_low_ref({la} limbs, {lb} limbs, limit {limit})"
                );
            }
        }
    }

    #[test]
    #[ignore = "timing probe for the squaring thresholds; run with --ignored"]
    fn squaring_crossover_timing() {
        use std::hint::black_box;
        use std::time::Instant;

        // Paired interleaved chunks with the order alternated between
        // passes, and every pass printed: a median alone cannot show
        // whether a claimed win clears the run-to-run spread.
        fn paired_saving(
            passes: usize,
            chunk: u32,
            flip: bool,
            a: &mut dyn FnMut(),
            b: &mut dyn FnMut(),
        ) -> f64 {
            let (mut ta, mut tb) = (0f64, 0f64);
            for _ in 0..passes {
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

        fn report(label: &str, w: usize, p: [f64; 5]) {
            let mut sorted = p;
            sorted.sort_by(f64::total_cmp);
            eprintln!(
                "{label:<22} {w:>5} {:>+8.1}%   {:+6.1} {:+6.1} {:+6.1} {:+6.1} {:+6.1}",
                sorted[2], p[0], p[1], p[2], p[3], p[4]
            );
        }

        let mut seed = 0x59ea_5011_0000_0001;
        eprintln!("saving of the second kernel over the first; median then every pass");
        eprintln!(
            "{:<22} {:>5} {:>9}   passes",
            "comparison", "limbs", "median"
        );
        for &w in &[
            1usize, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 112, 127, 128, 160, 192, 256, 384, 512,
        ] {
            let v = seeded_biguint(w, &mut seed);
            let chunk = (2_000_000 / (w * w)).max(50) as u32;

            // The lower handoff: general schoolbook against schoolbook
            // squaring, the pair `SQR_SCHOOLBOOK_MIN_LIMBS` sits between.
            if w <= 64 {
                let mut p = [0f64; 5];
                for (k, slot) in p.iter_mut().enumerate() {
                    *slot = paired_saving(
                        3,
                        chunk,
                        k % 2 == 1,
                        &mut || {
                            black_box(BigUint::mul_schoolbook_ref(black_box(&v), black_box(&v)));
                        },
                        &mut || {
                            black_box(BigUint::sqr_schoolbook_ref(black_box(&v)));
                        },
                    );
                }
                report("schoolbook sqr", w, p);
            }

            // The middle regime: the general Karatsuba against Karatsuba
            // squaring, both forced, so the comparison is the kernels' and
            // not the dispatcher's.
            if (KARATSUBA_THRESHOLD_LIMBS / 2..=512).contains(&w) {
                let mut p = [0f64; 5];
                for (k, slot) in p.iter_mut().enumerate() {
                    *slot = paired_saving(
                        3,
                        chunk,
                        k % 2 == 1,
                        &mut || {
                            black_box(black_box(&v).mul_karatsuba_ref(black_box(&v)));
                        },
                        &mut || {
                            black_box(black_box(&v).sqr_karatsuba_ref());
                        },
                    );
                }
                report("karatsuba sqr", w, p);
            }

            // The upper handoff: Karatsuba squaring against the Toom-3
            // multiplication it hands over to, which is the comparison the
            // 128-limb boundary actually rests on.
            if w >= 64 {
                let mut p = [0f64; 5];
                for (k, slot) in p.iter_mut().enumerate() {
                    *slot = paired_saving(
                        3,
                        chunk,
                        k % 2 == 1,
                        &mut || {
                            black_box(black_box(&v).mul_toom3_ref(black_box(&v)));
                        },
                        &mut || {
                            black_box(black_box(&v).sqr_karatsuba_ref());
                        },
                    );
                }
                report("karatsuba sqr vs toom3", w, p);
            }

            // And the public entry points, which is what a caller sees.
            let mut p = [0f64; 5];
            for (k, slot) in p.iter_mut().enumerate() {
                *slot = paired_saving(
                    3,
                    chunk,
                    k % 2 == 1,
                    &mut || {
                        black_box(black_box(&v).mul_ref(black_box(&v)));
                    },
                    &mut || {
                        black_box(black_box(&v).square_ref());
                    },
                );
            }
            report("square_ref vs mul_ref", w, p);
        }
    }

    #[test]
    #[ignore = "timing probe for tuning the Toom thresholds; run with --ignored"]
    fn toom_crossover_timing() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut seed = 0xC0FF_EE00_1234_5678;
        eprintln!(
            "{:>6} {:>11} {:>11} {:>11}  best",
            "words", "kara_us", "toom3_us", "toom4_us"
        );
        for &words in &[
            96usize, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072,
        ] {
            let reps = (2_000_000 / words).max(20);
            // Average each kernel over several independent operand pairs, each
            // measured as the min of three runs, to shed operand- and
            // scheduler-specific noise.
            let operands: Vec<(BigUint, BigUint)> = (0..4)
                .map(|_| {
                    (
                        seeded_biguint(words, &mut seed),
                        seeded_biguint(words, &mut seed),
                    )
                })
                .collect();
            let time = |f: &dyn Fn(&BigUint, &BigUint) -> BigUint| {
                let mut total = 0.0;
                for (a, b) in &operands {
                    let mut best = f64::INFINITY;
                    for _ in 0..3 {
                        black_box(f(a, b));
                        let t = Instant::now();
                        for _ in 0..reps {
                            black_box(f(a, b));
                        }
                        best = best.min(t.elapsed().as_secs_f64() / reps as f64 * 1e6);
                    }
                    total += best;
                }
                total / operands.len() as f64
            };
            let kara = time(&|a, b| BigUint::mul_karatsuba_ref(a, b));
            let toom3 = time(&|a, b| a.mul_toom3_ref(b));
            let toom4 = time(&|a, b| a.mul_toom4_ref(b));
            let best = if kara <= toom3 && kara <= toom4 {
                "kara"
            } else if toom3 <= toom4 {
                "toom3"
            } else {
                "toom4"
            };
            eprintln!("{words:6} {kara:11.4} {toom3:11.4} {toom4:11.4}  {best}");
        }
    }

    #[test]
    fn shr_bits_inverts_shl_bits_and_matches_division() {
        let mut seed = 0x6a09_e667_f3bc_c908;
        let shifts = [0usize, 1, 7, 63, 64, 65, 127, 128, 200];
        for words in [1usize, 2, 4, 9] {
            for _ in 0..8 {
                let value = seeded_biguint(words, &mut seed);
                for &n in &shifts {
                    // Round trip through the left shift.
                    let mut widened = value.clone();
                    widened.shl_bits(n);
                    widened.shr_bits(n);
                    assert_eq!(widened, value, "(x << {n}) >> {n} != x");

                    // Independent oracle: shifting right by n is dividing by
                    // 2^n, and division goes through Algorithm D, not the
                    // shift code.
                    let mut shifted = value.clone();
                    shifted.shr_bits(n);
                    let mut power_of_two = BigUint::zero();
                    power_of_two.set_bit(n);
                    assert_eq!(shifted, value.div_rem(&power_of_two).0, "x >> {n}");
                }
            }
        }
    }

    #[test]
    fn padded_bytes_and_low_windows() {
        let value = BigUint::from_u64(0x0102);
        assert_eq!(value.to_be_bytes_padded(2), vec![0x01, 0x02]); // exact fit
        assert_eq!(value.to_be_bytes_padded(5), vec![0, 0, 0, 0x01, 0x02]);
        assert_eq!(BigUint::zero().to_be_bytes_padded(3), vec![0, 0, 0]);
        assert!(BigUint::zero().to_be_bytes_padded(0).is_empty());

        let wide = BigUint::from_u128((0xABCD_u128 << 64) | 0x1234);
        assert_eq!(wide.low_u128(), (0xABCD_u128 << 64) | 0x1234);
        // Bits above 127 drop silently.
        let mut tall = BigUint::zero();
        tall.set_bit(200);
        tall.set_bit(3);
        assert_eq!(tall.low_u128(), 8);

        // Limb-aligned and mid-limb splits, and a window wider than the value.
        assert_eq!(wide.low_bits(64), BigUint::from_u64(0x1234));
        assert_eq!(
            wide.low_bits(68),
            BigUint::from_u128((0xD_u128 << 64) | 0x1234)
        );
        assert_eq!(wide.low_bits(4), BigUint::from_u64(4));
        assert!(wide.low_bits(0).is_zero());
        assert_eq!(wide.low_bits(500), wide);
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn padded_bytes_reject_overflow() {
        let _ = BigUint::from_u64(0x0102).to_be_bytes_padded(1);
    }

    #[test]
    fn shr_bits_edge_cases() {
        // Shifting everything out yields zero.
        let mut value = BigUint::from_u128(u128::MAX);
        value.shr_bits(128);
        assert!(value.is_zero());

        let mut value = BigUint::from_u64(1);
        value.shr_bits(1);
        assert!(value.is_zero());

        // Shifting zero and shifting by zero are identities.
        let mut zero = BigUint::zero();
        zero.shr_bits(1_000);
        assert!(zero.is_zero());
        let mut value = BigUint::from_u64(42);
        value.shr_bits(0);
        assert_eq!(value, BigUint::from_u64(42));

        // A shift far past the width is the same as shifting everything out.
        let mut value = BigUint::from_u128(u128::MAX);
        value.shr_bits(100_000);
        assert!(value.is_zero());
    }

    #[test]
    fn division_roundtrip() {
        let dividend = BigUint::from_u128(1_234_567_890_123_456_789);
        let divisor = BigUint::from_u64(37);
        let (q, r) = dividend.div_rem(&divisor);
        assert_eq!(q, BigUint::from_u128(33_366_699_733_066_399));
        assert_eq!(r, BigUint::from_u64(26));
        assert_eq!(q.mul_ref(&divisor).add_ref(&r), dividend);
    }

    /// `(q, r)` with `dividend = q * divisor + r` and `r < divisor` is unique,
    /// so checking the pair is a complete correctness statement for
    /// [`BigUint::div_rem`] and needs no separately computed expected value.
    fn assert_div_rem_invariant(dividend: &BigUint, divisor: &BigUint) {
        let (quotient, remainder) = dividend.div_rem(divisor);
        assert!(
            remainder < *divisor,
            "remainder {remainder:?} not reduced modulo {divisor:?}"
        );
        assert_eq!(
            quotient.mul_ref(divisor).add_ref(&remainder),
            *dividend,
            "q * d + r != n for {dividend:?} / {divisor:?}"
        );
    }

    #[test]
    fn div_rem_invariant_over_limb_shapes() {
        let mut seed = 0x243f_6a88_85a3_08d3;
        // Cover both division paths (one-limb Horner and multi-limb Knuth),
        // every quotient length from one limb up, and — because the leading
        // limb is random — a spread of D1 normalization shifts.
        for dividend_words in 1..=9usize {
            for divisor_words in 1..=dividend_words {
                for _ in 0..12 {
                    let dividend = seeded_biguint(dividend_words, &mut seed);
                    let divisor = seeded_biguint(divisor_words, &mut seed);
                    assert_div_rem_invariant(&dividend, &divisor);
                }
            }
        }
    }

    #[test]
    fn div_rem_handles_quotient_estimate_corrections() {
        // Knuth's D6 add-back runs with probability about 2^-63 on random
        // input, so it needs inputs built to force it. These are the base-2^64
        // analogues of the classic add-back cases from Warren, *Hacker's
        // Delight*, §9-2, plus a divisor whose top limb is already normalized
        // (D1 shift of zero) and one that needs the maximum shift.
        let cases: [(&[u64], &[u64]); 5] = [
            (&[0, 0, 0x8000_0000_0000_0000], &[1, 0x8000_0000_0000_0000]),
            (
                &[0, 0xFFFF_FFFF_FFFF_FFFE, 0x8000_0000_0000_0000],
                &[0xFFFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000],
            ),
            (
                &[0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
                &[0xFFFF_FFFF_FFFF_FFFF, 0x0000_0000_FFFF_FFFF],
            ),
            (&[0, 0, 0, 1], &[1, 1]),
            (&[u64::MAX, u64::MAX, u64::MAX], &[u64::MAX, 1]),
        ];

        for (dividend, divisor) in cases {
            assert_div_rem_invariant(
                &BigUint {
                    limbs: dividend.to_vec(),
                },
                &BigUint {
                    limbs: divisor.to_vec(),
                },
            );
        }
    }

    #[test]
    fn div_rem_exercises_the_add_back_path() {
        // Knuth's D6 add-back cannot happen for a two-limb divisor — there the
        // `v[n-2]` test in D3 is exact — and on random longer input it runs
        // with probability about `2 / 2^64`, so reaching it needs constructed
        // inputs. `dividend = (q + 1) * divisor - 1` is that construction: D3
        // accepts `q + 1` because it cannot see the divisor's low limbs, while
        // the true quotient is `q`, which is precisely what D6 repairs.
        let mut seed = 0xb504_f333_f9de_6484;
        for divisor_words in 3..=6usize {
            for q in [1u64, 2, 12_345, u64::MAX - 1] {
                let mut divisor = seeded_biguint(divisor_words, &mut seed);
                // A D1 shift of zero keeps the construction exact.
                divisor.limbs[divisor_words - 1] |= 1 << 63;

                let scale = BigUint::from_u64(q).add_ref(&BigUint::one());
                let dividend = scale.mul_ref(&divisor).sub_ref(&BigUint::one());

                assert_div_rem_invariant(&dividend, &divisor);
                // `(q + 1) * d - 1 = q * d + (d - 1)`, so the answer is exact.
                let (quotient, remainder) = dividend.div_rem(&divisor);
                assert_eq!(quotient, BigUint::from_u64(q));
                assert_eq!(remainder, divisor.sub_ref(&BigUint::one()));
            }
        }
    }

    #[test]
    fn div_rem_edge_cases() {
        let big = BigUint::from_be_bytes(&[0xFF; 40]);
        assert_div_rem_invariant(&big, &BigUint::one());
        assert_div_rem_invariant(&big, &big);
        assert_eq!(big.div_rem(&big).0, BigUint::one());
        assert!(big.div_rem(&big).1.is_zero());

        // Divisor above the dividend takes the early exit.
        let (quotient, remainder) = BigUint::from_u64(5).div_rem(&BigUint::from_u64(9));
        assert!(quotient.is_zero());
        assert_eq!(remainder, BigUint::from_u64(5));

        assert!(BigUint::zero().div_rem(&BigUint::from_u64(7)).0.is_zero());
    }

    #[test]
    fn sqrt_floor_small_values() {
        assert_eq!(BigUint::from_u64(0).sqrt_floor(), BigUint::from_u64(0));
        assert_eq!(BigUint::from_u64(1).sqrt_floor(), BigUint::from_u64(1));
        assert_eq!(BigUint::from_u64(2).sqrt_floor(), BigUint::from_u64(1));
        assert_eq!(BigUint::from_u64(15).sqrt_floor(), BigUint::from_u64(3));
        assert_eq!(BigUint::from_u64(16).sqrt_floor(), BigUint::from_u64(4));
        assert_eq!(BigUint::from_u64(17).sqrt_floor(), BigUint::from_u64(4));
        assert_eq!(
            BigUint::from_u128(17_184_849_881).sqrt_floor(),
            BigUint::from_u64(131_090)
        );
    }

    #[test]
    fn mod_mul_matches_small_arithmetic() {
        let a = BigUint::from_u64(123_456_789);
        let b = BigUint::from_u64(987_654_321);
        let m = BigUint::from_u64(1_000_000_007);
        assert_eq!(BigUint::mod_mul(&a, &b, &m), BigUint::from_u64(259_106_859));
    }

    #[test]
    fn montgomery_mod_pow_matches_small_arithmetic() {
        let ctx = MontgomeryCtx::new(&BigUint::from_u64(1_000_000_007))
            .expect("odd modulus builds a context");
        let base = BigUint::from_u64(123_456_789);
        let exponent = BigUint::from_u64(65_537);
        assert_eq!(ctx.pow(&base, &exponent), BigUint::from_u64(560_583_526));
    }

    #[test]
    fn montgomery_ctx_mul_matches_small_arithmetic() {
        let ctx = MontgomeryCtx::new(&BigUint::from_u64(1_000_000_007))
            .expect("odd modulus builds a context");
        let a = BigUint::from_u64(123_456_789);
        let b = BigUint::from_u64(987_654_321);
        assert_eq!(ctx.mul(&a, &b), BigUint::from_u64(259_106_859));
    }

    #[test]
    fn mod_mul_handles_even_modulus() {
        // Even moduli have no Montgomery representation, so this used to take a
        // separate double-and-add path; multiply-then-reduce covers both.
        let a = BigUint::from_u64(37);
        let b = BigUint::from_u64(19);
        let modulus = BigUint::from_u64(100);
        assert_eq!(BigUint::mod_mul(&a, &b, &modulus), BigUint::from_u64(3));
    }

    #[test]
    fn mod_mul_matches_montgomery_context() {
        // The one-shot path and the reusable-context path must agree.
        let mut seed = 0x0123_4567_89ab_cdef;
        for words in [1usize, 2, 4, 8, 16] {
            for _ in 0..8 {
                let lhs = seeded_biguint(words, &mut seed);
                let rhs = seeded_biguint(words, &mut seed);
                let mut modulus = seeded_biguint(words, &mut seed);
                modulus.limbs[0] |= 1; // Montgomery needs an odd modulus.

                let ctx = MontgomeryCtx::new(&modulus).expect("odd modulus builds a context");
                assert_eq!(BigUint::mod_mul(&lhs, &rhs, &modulus), ctx.mul(&lhs, &rhs));
            }
        }
    }

    #[test]
    fn bigint_sign_normalization() {
        let zero = BigInt::from_parts(Sign::Negative, BigUint::zero());
        assert_eq!(zero.sign(), Sign::Zero);

        let value = BigInt::from_parts(Sign::Positive, BigUint::from_u64(7));
        assert_eq!(value.negated().sign(), Sign::Negative);
        assert_eq!(value.magnitude(), &BigUint::from_u64(7));
    }

    #[test]
    fn bigint_add_sub_and_modulo() {
        let a = BigInt::from_biguint(BigUint::from_u64(10));
        let b = BigInt::from_parts(Sign::Negative, BigUint::from_u64(3));
        assert_eq!(a.add_ref(&b), BigInt::from_biguint(BigUint::from_u64(7)));
        assert_eq!(
            b.sub_ref(&a),
            BigInt::from_parts(Sign::Negative, BigUint::from_u64(13))
        );
        assert_eq!(
            BigInt::from_parts(Sign::Negative, BigUint::from_u64(3))
                .modulo_positive(&BigUint::from_u64(11)),
            BigUint::from_u64(8)
        );
    }
}
