//! Multiprecision unsigned and signed integers on `u64` limbs.
//!
//! The representation uses little-endian `u64` limbs because the algorithms
//! are naturally word-oriented. The kernels come straight from the literature
//! so they are auditable against their sources: schoolbook (Knuth's
//! Algorithm M), Karatsuba, Toom–Cook three- and four-way multiplication, and
//! exact number-theoretic-transform convolution at very large sizes; division
//! is Knuth's Algorithm D. Everything is Rust with no external arithmetic
//! backend.
//!
//! References for the multiplication and division kernels:
//! - Knuth, *TAOCP* vol. 2, §4.3.1, Algorithm M (schoolbook multiply) and
//!   Algorithm D (long division); §4.3.3 ("How Fast Can We Multiply?") for the
//!   Karatsuba and Toom–Cook methods.
//! - Karatsuba & Ofman, *Multiplication of Multidigit Numbers on Automata*,
//!   Soviet Physics–Doklady 7 (1963).
//! - Bodrato, *Towards Optimal Toom–Cook Multiplication…*, WAIFI 2007, for the
//!   optimized Toom evaluation/interpolation sequences.
//! - Schönhage & Strassen, *Schnelle Multiplikation großer Zahlen*, Computing
//!   7 (1971), 281–292, for modular-transform integer multiplication; the
//!   radix-2 transform is Cooley & Tukey, Math. Comp. 19 (1965), 297–301.

use core::cmp::Ordering;

mod montgomery;
mod ntt;

use montgomery::{copy_padded, mont_mul, mont_scratch_limbs, mont_sqr};
pub use montgomery::{ContextMismatch, MontgomeryContext, MontgomeryResidue, MontgomeryScratch};

mod barrett;
mod newton;
mod reciprocal;

pub use barrett::BarrettContext;
pub(crate) use newton::NEWTON_DIVISION_THRESHOLD_LIMBS;
pub use reciprocal::WordReciprocal;
// Only the test module reads this threshold; the dispatch lives in `barrett`.
#[cfg(test)]
use barrett::BARRETT_HALF_PRODUCT_MAX_LIMBS;

// Width from which one Karatsuba split beats a flat schoolbook pass, by
// `karatsuba_crossover_timing`. The crossover is machine-dependent and this
// is the width at which no measured machine loses: on the Cortex-A76 the
// split is ahead from 48 limbs (+5%, and +22% at 96), while on the EPYC 7452
// schoolbook holds until parity at 96 (-17% at 48, -47% at 64) and the split
// only pulls ahead at 192 (+16%). Below 96 a machine would pay for the
// split; at 96 one gains and the other is level. Correctness does not depend
// on the value.
const KARATSUBA_THRESHOLD_LIMBS: usize = 96;
// Length ratio at which Karatsuba stops: it accepts `long < 2·short`. The
// bound is structural, not tuned. The split is taken at half the longer
// operand, so at `long = 2·short` the shorter operand's high half is empty
// and the kernel degenerates to schoolbook. `should_use_unbalanced` admits
// from the same boundary upward; the two gates partition the shapes between
// them, which is why one constant serves both.
const KARATSUBA_MAX_IMBALANCE: usize = 2;
// Length ratio the balanced kernels — Toom-3, Toom-4 and the NTT — accept,
// as `long ≤ 3/2·short`, taken in integers as `short + short / 2`.
// Toom-3 splits at a third of the longer operand, so past this ratio the
// shorter operand's top third is empty and the five-point evaluation is
// spent on a two-part number. Toom-4 splits at a quarter, where the top
// part is empty already past 4/3, and the NTT does not split at all; both
// keep Toom-3's ceiling as policy, so one shape decision routes a pair
// through the whole ladder. Pairs past it fall to Karatsuba below 2× and to
// block decomposition from 2×. No measurement compares this ceiling with a
// tighter one for Toom-4 or a looser one for the NTT.
fn within_balanced_ratio(short: usize, long: usize) -> bool {
    long <= short + short / 2
}
// Toom-3 crossover: from this many limbs in the shorter operand, the five
// sub-multiplications of size n/3 overtake Karatsuba's three of size n/2,
// despite the heavier evaluate/interpolate pass. `toom_crossover_timing` on
// the EPYC 7452: Karatsuba leads at 96 limbs (12.1 µs against 16.4), Toom-3
// from 128 (21.4 against 25.6).
const TOOM3_THRESHOLD_LIMBS: usize = 128;
// Toom-4 crossover. Its exponent (log 7 / log 4 ≈ 1.404) beats Toom-3's
// (1.465), but the seven-point interpolation carries a much larger constant,
// so it overtakes Toom-3 only in the thousands of limbs.
// `toom_crossover_timing` puts the last width where a machine still prefers
// Toom-3 at 3072 (EPYC 7452: 2.75 ms against Toom-4's 2.97) and has both
// machines on Toom-4 from 4096 (EPYC 4.06 ms against 4.43; Cortex-A76
// 6.46 ms against 6.73) and above.
const TOOM4_THRESHOLD_LIMBS: usize = 4096;
// Exact NTT multiplication crossover on one execution context. The transform
// works in base 2^16 under two 31-bit primes and reconstructs every
// convolution coefficient by CRT. `ntt_crossover_timing` puts the crossover
// at 131,072 limbs on both measured machines and nowhere below it: on the
// EPYC 7452 a serial transform loses to Toom-4 at every narrower width in
// the sweep (180 ms against 156 ms at 65,536) and wins at 131,072 (379 ms
// against 448 ms); on the M4 it wins at 32,768, loses again at 65,536 — the
// padding staircase — and wins from 131,072 on. The threshold is the width
// past which no measured machine prefers Toom-4, not the first width where
// one of them does. Correctness is independent of it.
const NTT_SERIAL_THRESHOLD_LIMBS: usize = 131_072;
// The same kernel crosses earlier when its transform stages run on
// independent execution contexts: `ntt_crossover_timing` measures two
// contexts ahead of Toom-4 from 32,768 limbs on the M4 and from 8,192 on the
// EPYC, and four or more ahead from 8,192 on both, so each threshold is the
// wider of the two machines. `ntt::automatic_worker_count` never exceeds the
// reported machine parallelism and returns one if detection fails.
const NTT_TWO_WORKER_THRESHOLD_LIMBS: usize = 32_768;
const NTT_PARALLEL_THRESHOLD_LIMBS: usize = 8_192;
// Block-decomposition crossover for lopsided products (long ≥ 2·short): the
// shorter length from which cutting the longer operand into short-sized
// digits and multiplying each pair through the balanced kernels beats one
// flat schoolbook pass. It sits far above the Karatsuba crossover because
// per-block dispatch and allocation are paid in full while small blocks are
// barely sub-quadratic. `unbalanced_crossover_timing`: the decomposition
// loses 2x at 32-limb digits, breaks even near 128, trails slightly at 192,
// and wins 25-35% at 256, rising toward 2x at 512.
const UNBALANCED_THRESHOLD_LIMBS: usize = 256;
// Width at or above which squaring runs its own kernel rather than the
// general multiplication. Forming each cross term once saves up to
// (n−1)/2n of the limb products, but only once the products dominate the
// kernel's three passes and their carry walks.
//
// `squaring_crossover_timing` (run with `--ignored`), from 8 limbs up to the
// Karatsuba threshold: +12% at 8 limbs, +29% at 12, +36% at 16, +29% at 24.
// Below 8 the measurement straddles zero, so the floor sits where the win is
// clear.
const SQR_SCHOOLBOOK_MIN_LIMBS: usize = 8;
// Width at or above which squaring stops splitting Karatsuba-style and
// hands over to the multiplication ladder. The ordinary multiplication
// crossover does not carry over, because a Karatsuba square's constant
// factor differs from a Karatsuba product's.
//
// `squaring_crossover_timing` (run with `--ignored`) times Karatsuba
// squaring against what `mul` dispatches to at the same width, on the
// EPYC 7452 and the M4 Pro. Squaring is ahead by 29–44% at every width
// from 128 through 768 limbs on both (768: +29.2% and +31.6%), and behind
// from 1024 on both (1024: −5.4% and −7.0%; 1536: −2.3% and −15.6%; 4096:
// −11.8% and −15.3%). The boundary sits at the first measured loss; the
// widths between 768 and 1024 are unmeasured.
const SQR_KARATSUBA_MAX_LIMBS: usize = 1024;

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

/// Most digits of one radix that fit a `u64`, over the radices the
/// classical render reaches. The power-of-two radices take the bit path,
/// so radix 3 packs the most: 40 digits, since 3^40 < 2^64 < 3^41.
const MAX_DIGITS_PER_LIMB: usize = 40;

/// Digit count at or above which parsing dispatches to divide and conquer
/// (`RADIX_FROM_DC_THRESHOLD_DIGITS`), and the recursion floor below which
/// sub-problems convert classically (`RADIX_FROM_DC_BASE_DIGITS`).
///
/// The floor is where one classical pass, quadratic in the digit count, is
/// cheaper than a further split and its ladder multiply. Measured by
/// `radix_dc_crossover_timing` (run with `--ignored`) on the EPYC 7452
/// over 32 to 2048 limbs with floors of 512, 1024, 2048 and 4096 digits:
/// 512 parses fastest at every width, by 3% at 32 limbs (5 µs) and 20% at
/// 2048 (1.60 ms against 1.91 ms at 4096). The dispatch threshold is twice
/// the floor, so a dispatched input splits at least once; the same run has
/// the classical parse level with the recursion at 32 limbs (617 digits,
/// 6 µs each) and behind it from 64 limbs (1,233 digits: 15 against 12 µs),
/// which brackets the threshold. Correctness does not depend on either: the
/// recursion's hard base case is the ladder's first entry.
const RADIX_FROM_DC_THRESHOLD_DIGITS: usize = 1024;
const RADIX_FROM_DC_BASE_DIGITS: usize = 512;

/// Bit width at or above which rendering dispatches to divide and conquer
/// (`RADIX_TO_DC_THRESHOLD_BITS`), and the recursion floor below which
/// sub-values render classically (`RADIX_TO_DC_BASE_BITS`). The same probe
/// on the same machine: the 512-bit floor renders fastest at every width
/// (12 µs at 32 limbs against 14–15 at the wider floors; 5.14 ms at 2048
/// against 5.69), and the recursion beats classical rendering from the
/// narrowest width probed, 32 limbs (12 against 23 µs), which is the
/// threshold; below it is unmeasured. The two sides do not dispatch at the
/// same value: 2,048 bits is about 617 decimal digits (2048·log₁₀2) and
/// 1,024 digits about 3,400 bits, so parsing stays classical to about 1.7×
/// the width at which rendering splits, and the probe shows why — at 32
/// limbs the recursion already halves rendering while it only matches
/// parsing. Correct at any values, as above.
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
        if self.limbs.len() > source.limbs.len() {
            // The truncation strands the high limbs in spare capacity, out
            // of reach of the drop-time scrub, so wipe them first.
            crate::scrub::zeroize_slice(&mut self.limbs[source.limbs.len()..]);
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
    #[must_use]
    pub fn from_be_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self::zero();
        }

        let mut limbs = Vec::with_capacity(bytes.len().div_ceil(8));
        let mut acc = 0u64;
        let mut shift = 0u32;

        // Pack bytes from the least significant (the last) into limbs, eight
        // to a limb; leftover bytes form a partial top limb.
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
    /// The output is exactly `⌈bits/8⌉` bytes (zero encodes as a single
    /// `0x00`), written directly into one allocation of that length, so no
    /// byte of the value is left behind in a discarded buffer or in spare
    /// capacity.
    #[must_use]
    pub fn to_be_bytes(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.bits().div_ceil(8).max(1)];
        self.write_bytes_least_significant_first(out.iter_mut().rev());
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
    /// The buffer is extracted with `mem::take`, leaving an empty vector
    /// behind; the caller is expected to hand it back to
    /// [`Self::from_limbs`].
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
    /// The limbs are written straight into the returned buffer, so no second
    /// copy of the value's bytes is left in a freed allocation.
    ///
    /// # Panics
    ///
    /// Panics if the value does not fit in `byte_width` bytes.
    /// `byte_width = 0` is legal only for zero (and yields an empty vector).
    #[must_use]
    pub fn to_be_bytes_padded(&self, byte_width: usize) -> Vec<u8> {
        assert!(
            self.bits().div_ceil(8) <= byte_width,
            "value does not fit in {byte_width} bytes"
        );
        let mut out = vec![0u8; byte_width];
        self.write_bytes_least_significant_first(out.iter_mut().rev());
        out
    }

    /// Decode little-endian bytes: the first byte is the least significant,
    /// the mirror of [`Self::from_be_bytes`]. The empty slice decodes to zero,
    /// and zero bytes at the end (the high end) are accepted and ignored.
    #[must_use]
    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut limbs = Vec::with_capacity(bytes.len().div_ceil(8));
        // Eight bytes to a limb, the first byte of each chunk lowest; a short
        // final chunk is the top limb's low bytes.
        for chunk in bytes.chunks(8) {
            let mut limb = 0u64;
            for (&byte, shift) in chunk.iter().zip((0..u64::BITS).step_by(8)) {
                limb |= u64::from(byte) << shift;
            }
            limbs.push(limb);
        }

        let mut out = Self { limbs };
        out.normalize();
        out
    }

    /// Encode as little-endian bytes without trailing zero bytes — the mirror
    /// of [`Self::to_be_bytes`]: exactly `⌈bits/8⌉` bytes, least significant
    /// first, and zero encodes as a single `0x00`, written directly into one
    /// allocation.
    #[must_use]
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.bits().div_ceil(8).max(1)];
        self.write_bytes_least_significant_first(out.iter_mut());
        out
    }

    /// Encode as little-endian bytes at a fixed width, zero-padded on the
    /// right (the high end) — the mirror of [`Self::to_be_bytes_padded`],
    /// written straight into the returned buffer in the same way.
    ///
    /// # Panics
    ///
    /// Panics if the value does not fit in `byte_width` bytes.
    /// `byte_width = 0` is legal only for zero (and yields an empty vector).
    #[must_use]
    pub fn to_le_bytes_padded(&self, byte_width: usize) -> Vec<u8> {
        assert!(
            self.bits().div_ceil(8) <= byte_width,
            "value does not fit in {byte_width} bytes"
        );
        let mut out = vec![0u8; byte_width];
        self.write_bytes_least_significant_first(out.iter_mut());
        out
    }

    /// The byte writer behind the four encoders: the value's bytes, least
    /// significant first, into `slots` in iteration order.
    ///
    /// Writing stops when the slots or the limbs' `8·len` bytes run out.
    /// Slots past the limbs keep their contents (zero, in every caller);
    /// limb bytes past the slots are zero because every caller sizes its
    /// buffer to at least `⌈bits/8⌉`.
    fn write_bytes_least_significant_first<'a>(&self, slots: impl Iterator<Item = &'a mut u8>) {
        let bytes = self.limbs.iter().flat_map(|&limb| {
            // The cast keeps the low eight bits of each shifted limb.
            (0..u64::BITS)
                .step_by(8)
                .map(move |shift| (limb >> shift) as u8)
        });
        for (slot, byte) in slots.zip(bytes) {
            *slot = byte;
        }
    }

    /// Parse from a digit string in the given radix (2 through 36, digits
    /// `0-9a-z`, upper case accepted), `None` on an empty string or an
    /// invalid digit. Leading zeros are accepted; no sign, no whitespace,
    /// no `0x` prefix — this is the value, not a literal.
    ///
    /// Below a measured digit-count crossover the conversion is classical
    /// with a word-sized base: digits are consumed in groups of the largest
    /// power of the radix that fits a limb, each group folded in by one limb
    /// multiply-add, O(n²) in total.
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
        if radix.is_power_of_two() {
            self.to_digits_pow2(radix)
        } else {
            self.to_digits_dc(radix)
        }
        .iter()
        .map(|&d| char::from(RADIX_DIGITS[usize::from(d)]))
        .collect()
    }

    /// The largest power of `radix` that fits a `u64`, with its exponent —
    /// the "big base" both classical conversions work in, so a group of
    /// digits costs one limb-sized multiply-add instead of one per digit
    /// (`10^19` for decimal, `3^40` for radix 3, `2^63` for radix 2).
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
    /// When the radix is `2^b` a digit is a `b`-bit field of the value, so
    /// the conversion re-slices the bit string with no arithmetic. A digit
    /// straddles a limb boundary when `b` does not divide 64 (radices 8 and
    /// 32); the spill into `limbs[limb + 1]` handles that.
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
    /// Horner's rule with the big base `radix^chunk` in place of the radix,
    /// so one limb-sized multiply-add absorbs `chunk` digits. The leading
    /// group takes `len mod chunk` digits, leaving every later group full and
    /// scaled by the precomputed `big_base`. Quadratic in the digit count.
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
            value = value.mul(&Self::from_u64(base));
            value.add_assign_ref(&Self::from_u64(group));
            index += take;
        }
        value
    }

    /// The squared-power ladder `radix^(chunk·2^i)`, built once per
    /// conversion and shared down the recursion; rebuilding it per level
    /// would make the method quadratic again. Entry `i` spans `chunk·2^i`
    /// digits.
    fn radix_power_ladder(radix: u32, digit_count: usize) -> (Vec<Self>, usize) {
        let (big_base, chunk) = Self::limb_radix_power(radix);
        let mut ladder = vec![Self::from_u64(big_base)];
        let mut span = chunk;
        while span * 2 < digit_count {
            let top = ladder.last().expect("ladder starts non-empty").square();
            ladder.push(top);
            span *= 2;
        }
        (ladder, chunk)
    }

    /// The render-side ladder, sized by the value's bit width. A digit-count
    /// estimate would overshoot by up to two entries, the most expensive
    /// squarings in the conversion.
    fn radix_power_ladder_bits(radix: u32, bit_width: usize) -> (Vec<Self>, usize) {
        let (big_base, chunk) = Self::limb_radix_power(radix);
        let mut ladder = vec![Self::from_u64(big_base)];
        while ladder.last().expect("non-empty").bits() * 2 < bit_width {
            let top = ladder.last().expect("non-empty").square();
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
    /// two halves within a factor of two of each other, so the recursion
    /// depth is logarithmic and each multiply is balanced.
    fn from_digits_ladder(
        digits: &[u8],
        radix: u32,
        ladder: &[Self],
        chunk: usize,
        base_digits: usize,
    ) -> Self {
        // The first clause is the structural base case, independent of any
        // tuning constant: with no ladder entry spanning fewer digits than
        // the input, there is nothing to split. The second is the floor
        // below which conversion is classical by policy.
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
        let mut value = high.mul(&ladder[index]);
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
            let mut buffer = [0u8; MAX_DIGITS_PER_LIMB];
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
    /// occupies exactly `span` positional digits, so the low half is
    /// zero-padded to that span before it is appended.
    fn to_digits_ladder(
        &self,
        radix: u32,
        ladder: &[Self],
        chunk: usize,
        base_bits: usize,
    ) -> Vec<u8> {
        // The first clause is the structural base case — a value no wider
        // than the first ladder entry splits into nothing; the second is
        // the floor below which rendering is classical by policy.
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
    /// and [`Self::ln_approx`].
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

    /// The nearest integer to a finite, non-negative double, or `None` for
    /// a negative or non-finite one.
    ///
    /// A double is an integer significand at a power of two, so once the
    /// value is rounded the conversion is exact, however large. The inverse
    /// of [`Self::to_f64_lossy`] for values that round-trip.
    #[must_use]
    pub fn from_f64_lossy(value: f64) -> Option<Self> {
        if !value.is_finite() || value < 0.0 {
            return None;
        }
        let rounded = value.round();
        if rounded == 0.0 {
            return Some(Self::from_u64(0));
        }
        let bits = rounded.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i64;
        let fraction = bits & ((1u64 << 52) - 1);
        let significand = if exponent == 0 {
            fraction
        } else {
            fraction | (1u64 << 52)
        };
        // rounded = significand · 2^(exponent − 1075).
        let shift = exponent - 1075;
        let mut integer = Self::from_u64(significand);
        if shift >= 0 {
            integer.shl_bits(shift as usize);
        } else {
            // An integer below 2⁵³: the bits shifted out are zero.
            integer.shr_bits((-shift) as usize);
        }
        Some(integer)
    }

    /// The value as an `f64`, for size-driven parameter heuristics. The
    /// result is within one unit in the last place of the true value (the top 64 bits
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

    /// The natural logarithm of the value as an `f64`, for size-driven
    /// heuristics stated in terms of `ln n` (such as the smoothness bound
    /// `exp(½√(ln n · ln ln n))`). Computed as
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
    /// A logarithm gives the estimate; near powers of `radix`, where the
    /// floating-point floor can land on either side, comparisons against
    /// powers of `radix` correct it.
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
    /// [`BarrettContext::reduce`] can take its `mod b^{k+1}` windows this way.
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
    /// [`MontgomeryContext::new`] gates on — REDC needs `gcd(2^64, n) = 1` — and
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
        bit_span(self.limbs.len() - 1, 64) + top_bits
    }

    /// Integer square root: the largest `r` with `r² ≤ self`. The root half
    /// of [`Self::sqrt_rem`], which documents the Newton iteration; it skips
    /// the squaring and subtraction that produce the remainder.
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
    /// division at the operand's width, and convergence is quadratic, so the
    /// step count is about log₂ of the bit width.
    #[must_use]
    pub fn sqrt_rem(&self) -> (Self, Self) {
        if self.is_zero() || self.is_one() {
            return (self.clone(), Self::zero());
        }
        let root = self.sqrt_newton();
        let square = root.square();
        (root, self.sub(&square))
    }

    /// The Newton core shared by [`Self::sqrt_floor`] and
    /// [`Self::sqrt_rem`]. Requires `self ≥ 2`.
    fn sqrt_newton(&self) -> Self {
        debug_assert!(!self.is_zero() && !self.is_one(), "callers handle 0 and 1");
        // Seed: 2^⌈bits/2⌉ ≥ ⌈√self⌉, one bit above the root's width.
        let mut current = Self::zero();
        current.set_bit(self.bits().div_ceil(2));
        loop {
            // next = (current + self/current) / 2
            let (quotient, _) = self.div_rem(&current);
            let mut next = current.add(&quotient);
            next.shr1();
            if next >= current {
                debug_assert!(
                    current.square() <= *self,
                    "certified root is not above the value"
                );
                return current;
            }
            current = next;
        }
    }

    /// Population count: the number of set bits.
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
            .map(|index| bit_span(index, 64) + self.limbs[index].trailing_zeros() as usize)
    }

    /// `self^exponent` for a machine-word exponent, by binary
    /// exponentiation.
    #[must_use]
    pub fn pow_u64(&self, exponent: u64) -> Self {
        let mut result = Self::one();
        let mut base = self.clone();
        let mut remaining = exponent;
        while remaining > 0 {
            if remaining & 1 == 1 {
                result = result.mul(&base);
            }
            remaining >>= 1;
            if remaining > 0 {
                base = base.square();
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
            let mut next = current.mul(&k_minus_one);
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
    /// powers by convention (`0^2`, `1^2`). This is the straightforward
    /// method; the essentially linear-time one is Bernstein, *Detecting
    /// perfect powers in essentially linear time*, Math. Comp. 67 (1998),
    /// 1253–1283.
    ///
    /// On odd operands the valuation filter is inert and every prime
    /// exponent below the bit width pays a full root, so the cost grows
    /// roughly as the cube of the width.
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
    /// index lies above the current width. Setting a bit cannot zero the
    /// top limb, so the result stays canonical without `normalize`. It is
    /// the cheap way to build a power of two, such as `R² = 2^(128w)` in
    /// [`MontgomeryContext::new`].
    pub fn set_bit(&mut self, index: usize) {
        let limb = index / 64;
        let shift = index % 64;
        if self.limbs.len() <= limb {
            self.limbs.resize(limb + 1, 0);
        }
        self.limbs[limb] |= 1u64 << shift;
    }

    /// Add another bigint in place: one `u128` carry pass over `other`'s
    /// width, the carry rippling on and pushing a new top limb if it escapes.
    /// Adding cannot zero a non-zero top limb, so no `normalize` is needed.
    ///
    /// # Panics
    ///
    /// Does not panic in normal use; an internal `expect` guards the
    /// limb-packing invariant (a `u128` accumulator splitting back into `u64`
    /// limbs) and would trip only on a logic error.
    pub(crate) fn add_assign_ref(&mut self, other: &Self) {
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

    /// `lhs · rhs`, written into `self`'s buffer.
    ///
    /// The three-operand product, for callers that keep one output alive
    /// across many multiplications — a field's residues, an accumulator —
    /// and want the product's storage to be that output's rather than a
    /// fresh allocation each call. Below `KARATSUBA_THRESHOLD_LIMBS` in the
    /// shorter operand the schoolbook kernel runs straight into `self`, so
    /// once the buffer's capacity covers the product nothing is allocated;
    /// wider operands take the ladder, whose kernels build their own
    /// temporaries, and only the result is copied into `self`'s storage.
    ///
    /// Limbs the result does not cover are wiped before the buffer shrinks,
    /// as every in-place operation here does under the `wipe` feature.
    pub fn mul_into(&mut self, lhs: &Self, rhs: &Self) {
        let (short, long) = if lhs.limbs.len() <= rhs.limbs.len() {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        };
        if short.is_zero() {
            crate::scrub::zeroize_slice(self.limbs.as_mut_slice());
            self.limbs.clear();
            return;
        }
        if short.limbs.len() >= KARATSUBA_THRESHOLD_LIMBS {
            let product = lhs.mul(rhs);
            self.clone_from(&product);
            return;
        }
        let width = short.limbs.len() + long.limbs.len();
        if self.limbs.len() > width {
            crate::scrub::zeroize_slice(&mut self.limbs[width..]);
        }
        self.limbs.resize(width, 0);
        self.limbs.fill(0);
        for (i, &short_limb) in short.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &long_limb) in long.limbs.iter().enumerate() {
                let acc = u128::from(self.limbs[i + j])
                    + u128::from(short_limb) * u128::from(long_limb)
                    + carry;
                self.limbs[i + j] = low_u64(acc);
                carry = acc >> 64;
            }
            // The row's carry lands one limb past the row, which the width
            // has room for: two operands of `a` and `b` limbs multiply to at
            // most `a + b` limbs.
            self.limbs[i + long.limbs.len()] = low_u64(carry);
        }
        self.normalize();
    }

    /// Keep the low `bits` bits of `self`, in place: `self mod 2^bits`.
    ///
    /// The reduction step of arithmetic modulo a Mersenne number `2^k − 1`
    /// is a shift and this mask, and a caller folding a product wants both
    /// without allocating. Limbs the mask discards are wiped before the
    /// buffer shrinks.
    pub fn keep_low_bits(&mut self, bits: usize) {
        if bits >= self.bits() {
            return;
        }
        let kept = bits.div_ceil(64);
        if kept < self.limbs.len() {
            crate::scrub::zeroize_slice(&mut self.limbs[kept..]);
            self.limbs.truncate(kept);
        }
        let partial = bits % 64;
        if partial != 0 {
            if let Some(top) = self.limbs.last_mut() {
                *top &= (1u64 << partial) - 1;
            }
        }
        self.normalize();
    }

    /// Return `self + other`: a clone of `self` plus an in-place add.
    /// [`Self::add_into`] avoids the clone when the caller owns a
    /// destination buffer.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.add_assign_ref(other);
        out
    }

    /// Write `lhs + rhs` into `self`, replacing its value and reusing its
    /// limb buffer — the three-operand form of GMP's `mpz_add`. No
    /// allocation once the buffer's capacity covers the result.
    ///
    /// # Panics
    ///
    /// Does not panic in normal use; an internal `expect` guards the
    /// limb-packing invariant (a `u128` accumulator splitting back into `u64`
    /// limbs) and would trip only on a logic error.
    pub fn add_into(&mut self, lhs: &Self, rhs: &Self) {
        debug_assert!(
            lhs.limbs.last() != Some(&0) && rhs.limbs.last() != Some(&0),
            "operands arrive canonical; the result's canonical form relies on it"
        );
        let (long, short) = if lhs.limbs.len() >= rhs.limbs.len() {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        };
        // Shape the buffer to the working width.
        let n = long.limbs.len();
        if self.limbs.len() > n {
            // The shrinking resize strands the high limbs in spare capacity;
            // wipe them before they leave the drop-time scrub's reach.
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

    /// Write `lhs - rhs` into `self`, replacing its value and reusing its
    /// limb buffer — the counterpart of [`Self::add_into`]. No allocation
    /// once the buffer's capacity covers the result.
    ///
    /// # Panics
    ///
    /// Panics if `lhs < rhs`.
    pub fn sub_into(&mut self, lhs: &Self, rhs: &Self) {
        assert!(lhs.cmp(rhs) != Ordering::Less, "BigUint underflow");
        // Shape the buffer as in `add_into`.
        let n = lhs.limbs.len();
        if self.limbs.len() > n {
            // As in `add_into`: wipe the limbs the shrinking resize strands.
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

    /// `self ← minuend - self`, in place, for signed operations whose
    /// result magnitude is the other operand's minus this one's. Panics if
    /// `minuend < self`.
    fn rsub_assign_ref(&mut self, minuend: &Self) {
        assert!(minuend.cmp(self) != Ordering::Less, "BigUint underflow");
        debug_assert!(
            self.limbs.len() <= minuend.limbs.len(),
            "self <= minuend, so the resize only grows"
        );
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

    /// Subtract another bigint in place: one `u128` borrow pass, then
    /// `normalize`, since cancellation can empty the top limbs.
    ///
    /// # Panics
    ///
    /// Panics if `self < other`; this type cannot represent a negative
    /// difference. `BigInt`'s `-=` is the total operation.
    pub(crate) fn sub_assign_ref(&mut self, other: &Self) {
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

    /// Return `self - other`.
    ///
    /// # Panics
    ///
    /// Panics if `self < other`.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.sub_assign_ref(other);
        out
    }

    /// Multiply two big integers, choosing the kernel by the shorter
    /// operand's length: schoolbook (Knuth's Algorithm M) by default,
    /// Karatsuba from `KARATSUBA_THRESHOLD_LIMBS`, three-way Toom–Cook from
    /// `TOOM3_THRESHOLD_LIMBS`, four-way from `TOOM4_THRESHOLD_LIMBS`, and
    /// an exact number-theoretic transform from `NTT_SERIAL_THRESHOLD_LIMBS`
    /// on one execution context, `NTT_TWO_WORKER_THRESHOLD_LIMBS` with two
    /// and `NTT_PARALLEL_THRESHOLD_LIMBS` with four or more; each constant
    /// carries the measurement that set it. The NTT never uses more contexts than
    /// [`std::thread::available_parallelism`] reports. Toom and NTT require
    /// `long ≤ 1.5·short`, Karatsuba `long < 2·short`. A lopsided pair
    /// (`long ≥ 2·short`) whose shorter operand has at least 256 limbs takes
    /// `mul_unbalanced_ref`, block decomposition into balanced products;
    /// smaller lopsided pairs stay schoolbook. The module header cites each
    /// algorithm.
    ///
    /// # Panics
    ///
    /// Does not panic in normal use: an internal `expect` guards a limb-packing
    /// invariant (`u128` accumulators splitting back into `u64` limbs) and can
    /// trip only on a logic error in a kernel.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        if Self::should_use_ntt(self, other) {
            return self.mul_ntt_ref(other);
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

    /// Exact NTT admission: large, approximately balanced operands whose
    /// base-2^16 convolution fits both transform primes.
    fn should_use_ntt(lhs: &Self, rhs: &Self) -> bool {
        let Some(transform_len) = ntt::transform_len(lhs.limbs.len(), rhs.limbs.len()) else {
            return false;
        };
        Self::should_use_ntt_with_workers(lhs, rhs, ntt::automatic_worker_count(transform_len))
    }

    /// Deterministic form of NTT admission, parameterized for its tests.
    #[cfg(test)]
    fn should_use_ntt_with_contexts(lhs: &Self, rhs: &Self, max_contexts: usize) -> bool {
        let Some(transform_len) = ntt::transform_len(lhs.limbs.len(), rhs.limbs.len()) else {
            return false;
        };
        let workers = ntt::worker_count(transform_len, max_contexts.max(1));
        Self::should_use_ntt_with_workers(lhs, rhs, workers)
    }

    fn should_use_ntt_with_workers(lhs: &Self, rhs: &Self, workers: usize) -> bool {
        let short = lhs.limbs.len().min(rhs.limbs.len());
        let long = lhs.limbs.len().max(rhs.limbs.len());
        let threshold = match workers {
            1 => NTT_SERIAL_THRESHOLD_LIMBS,
            2 => NTT_TWO_WORKER_THRESHOLD_LIMBS,
            _ => NTT_PARALLEL_THRESHOLD_LIMBS,
        };
        // Padding is not a reason to refuse. A radix-2 transform rounds the
        // convolution up to a power of two, so the work per limb depends on
        // where a width falls in that staircase — 16,384 limbs pad to 8
        // coefficients per limb, 20,410 to 12.8 — but `ntt_padding_gate_timing`
        // has the transform ahead at every width and every ratio it measures
        // on both machines: at the worst ratio, 12.84, by 43% on the EPYC 7452
        // and 13% on the Cortex-A76, and never behind at the friendlier ones.
        // The NFS square root lifts at 1.3 Mbit — 20,410 limbs, ratio 12.84 —
        // so a gate on padding would send its widest products to Toom-4.
        short >= threshold && within_balanced_ratio(short, long)
    }

    /// Exact large multiplication through two modular transforms and CRT.
    fn mul_ntt_ref(&self, other: &Self) -> Self {
        ntt::multiply(self, other)
    }

    /// Serial exact NTT reference for differential and crossover measurement.
    #[cfg(test)]
    fn mul_ntt_serial_ref(&self, other: &Self) -> Self {
        ntt::multiply_serial(self, other)
    }

    /// Exact NTT with a forced context ceiling for parallel-scaling probes.
    #[cfg(test)]
    fn mul_ntt_with_contexts_ref(&self, other: &Self, max_contexts: usize) -> Self {
        ntt::multiply_with_contexts(self, other, max_contexts)
    }

    /// Exact NTT with a forced worker count for scaling probes.
    #[cfg(test)]
    fn mul_ntt_with_workers_ref(&self, other: &Self, workers: usize) -> Self {
        ntt::multiply_with_workers(self, other, workers)
    }

    /// Exact large squaring through one modular transform buffer and CRT.
    fn sqr_ntt_ref(&self) -> Self {
        ntt::square(self)
    }

    /// Square a value, exploiting the symmetry that lets a squaring form
    /// each distinct cross term once instead of twice.
    ///
    /// Below `SQR_SCHOOLBOOK_MIN_LIMBS` this is [`Self::mul`]. From there
    /// to `KARATSUBA_THRESHOLD_LIMBS` it is `sqr_schoolbook_ref`; from there
    /// to `SQR_KARATSUBA_MAX_LIMBS` it is `sqr_karatsuba_ref`, whose three
    /// sub-products are themselves squares. Wider operands take
    /// [`Self::mul`]'s Toom kernels, or, once the NTT admits them, an NTT
    /// square that needs one transform array and one forward transform per
    /// prime instead of a general product's two. Each threshold carries the
    /// measurement that set it; PERFORMANCE.md's `sqr` rows carry the cost.
    ///
    /// Montgomery residues have their own squaring
    /// ([`MontgomeryContext::square_residue`](crate::modular::MontgomeryContext::square_residue)),
    /// which fuses the reduction.
    #[must_use]
    pub fn square(&self) -> Self {
        // Narrowest first: the commonest operands decide on one comparison.
        let width = self.limbs.len();
        if width < SQR_SCHOOLBOOK_MIN_LIMBS {
            // Too narrow for the specialized kernel to pay; zero lands here.
            return self.mul(self);
        }
        if width < KARATSUBA_THRESHOLD_LIMBS {
            return Self::sqr_schoolbook_ref(self);
        }
        if width >= SQR_KARATSUBA_MAX_LIMBS {
            if Self::should_use_ntt(self, self) {
                return self.sqr_ntt_ref();
            }
            // Wide enough that the Toom kernels beat a Karatsuba square.
            return self.mul(self);
        }
        self.sqr_karatsuba_ref()
    }

    /// Schoolbook squaring: `n(n+1)/2` limb products against the general
    /// multiplication's `n²`, by forming each distinct cross term once
    /// (*Handbook of Applied Cryptography*, Algorithm 14.16).
    ///
    /// Three passes: accumulate the strict upper triangle
    /// `Σ_{i<j} aᵢaⱼB^{i+j}`, double it with one shift over the buffer, then
    /// add the diagonal `Σ aᵢ²B^{2i}`. Doubling once, rather than per term,
    /// avoids a `2·aᵢ·aⱼ` term that would overflow the `u128` accumulator.
    ///
    /// The doubling cannot overflow the buffer: twice the strict upper
    /// triangle is at most the whole square, and `a² < B^{2n}`.
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

        // Pass three: the diagonal. A term at `2i` occupies two limbs and
        // its carry lands on `2i + 2`, the next iteration's position.
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
        // Unreachable from `square`, which routes only widths of at least
        // `KARATSUBA_THRESHOLD_LIMBS` here; a zero split would recurse
        // forever.
        let split = self.limbs.len() / 2;
        if split == 0 {
            return Self::sqr_schoolbook_ref(self);
        }
        let (low, high) = self.split_at_limb(split);
        // `split = len/2 <= len − 1` for `len >= 2`, so the high half keeps
        // the non-zero top limb.
        debug_assert!(
            !high.is_zero(),
            "a normalized operand split at half its own width has a non-zero high half"
        );

        let z0 = low.square();
        let z2 = high.square();
        let sum = low.add(&high);
        let mut z1 = sum.square();
        z1.sub_assign_ref(&z0);
        z1.sub_assign_ref(&z2);

        let mut out = z0;
        z1.shl_bits(bit_span(split, 64));
        out.add_assign_ref(&z1);
        let mut z2_shifted = z2;
        z2_shifted.shl_bits(bit_span(split, 128));
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

    /// Karatsuba admission: the shorter operand at least
    /// `KARATSUBA_THRESHOLD_LIMBS`, and `long < KARATSUBA_MAX_IMBALANCE ·
    /// short`. The bound is strict because the split is taken at half the
    /// longer operand: at `long = 2·short` the shorter operand's high half is
    /// empty and the kernel would fall back to schoolbook.
    fn should_use_karatsuba(lhs: &Self, rhs: &Self) -> bool {
        let short = lhs.limbs.len().min(rhs.limbs.len());
        let long = lhs.limbs.len().max(rhs.limbs.len());
        short >= KARATSUBA_THRESHOLD_LIMBS && long < short * KARATSUBA_MAX_IMBALANCE
    }

    /// Unbalanced admission: the shorter operand at least
    /// `UNBALANCED_THRESHOLD_LIMBS`, and the pair too lopsided for any
    /// balanced kernel (`long ≥ 2·short`).
    fn should_use_unbalanced(lhs: &Self, rhs: &Self) -> bool {
        let short = lhs.limbs.len().min(rhs.limbs.len());
        let long = lhs.limbs.len().max(rhs.limbs.len());
        short >= UNBALANCED_THRESHOLD_LIMBS && long >= short * KARATSUBA_MAX_IMBALANCE
    }

    /// Unbalanced multiplication by block decomposition: cut the longer
    /// operand into base-`B = 2^{64k}` digits of the shorter one's length
    /// `k`, so that `long · short = Σᵢ digitᵢ·short·Bⁱ` — a sum of balanced
    /// `k × k` products, each accumulated into the output at its limb
    /// offset. Each digit product re-enters [`Self::mul`] and lands on
    /// a balanced sub-quadratic kernel.
    ///
    /// `add_into_at` accumulates each product in place at its window
    /// `[i·k, i·k + len)`. Shifting each product and adding full-width would
    /// copy `Σᵢ i·k ≈ long²/(2k)` limbs, quadratic in the long length. An
    /// all-zero digit is skipped.
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
            let part = digit.mul(short);
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
    /// upward. The caller guarantees the true sum fits in `acc`, so the
    /// carry dies before the buffer ends.
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
    /// sub-products recurse through [`Self::mul`], so a large operand
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

        let z0 = a0.mul(&b0);
        let z2 = a1.mul(&b1);

        let a_sum = a0.add(&a1);
        let b_sum = b0.add(&b1);
        let mut z1 = a_sum.mul(&b_sum);
        z1.sub_assign_ref(&z0);
        z1.sub_assign_ref(&z2);

        let mut out = z0;
        z1.shl_bits(bit_span(split, 64));
        out.add_assign_ref(&z1);

        let mut z2_shifted = z2;
        z2_shifted.shl_bits(bit_span(split, 128));
        out.add_assign_ref(&z2_shifted);
        out
    }

    /// Toom-3 admission: both operands past `TOOM3_THRESHOLD_LIMBS` and
    /// within 1.5× of each other in length.
    fn should_use_toom3(lhs: &Self, rhs: &Self) -> bool {
        let short = lhs.limbs.len().min(rhs.limbs.len());
        let long = lhs.limbs.len().max(rhs.limbs.len());
        short >= TOOM3_THRESHOLD_LIMBS && within_balanced_ratio(short, long)
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
            let even = c0.add(c2); // c0 + c2
            let at_1 = even.add(c1); // c(1)
            let at_m1 = BigInt::from_biguint(even).sub(&BigInt::from_biguint(c1.clone()));
            let mut twice_c1 = c1.clone();
            twice_c1.shl_bits(1);
            let mut four_c2 = c2.clone();
            four_c2.shl_bits(2);
            let at_2 = c0.add(&twice_c1).add(&four_c2); // c(2)
            (at_1, at_m1, at_2)
        };
        let (a_1, a_m1, a_2) = eval(&a0, &a1, &a2);
        let (b_1, b_m1, b_2) = eval(&b0, &b1, &b2);

        // Pointwise products (each a recursive multiplication).
        let v0 = BigInt::from_biguint(a0.mul(&b0)); // W(0)
        let v_inf = BigInt::from_biguint(a2.mul(&b2)); // W(∞)
        let v1 = BigInt::from_biguint(a_1.mul(&b_1)); // W(1)
        let vm1 = bigint_mul(&a_m1, &b_m1); // W(-1)
        let v2 = BigInt::from_biguint(a_2.mul(&b_2)); // W(2)

        // Interpolate the product digits c0..c4. Derivation: with
        // W(x) = Σ cᵢ xⁱ, the points give c0 = W(0), c4 = W(∞), and
        //   s = (W(1)+W(-1))/2 = c0 + c2 + c4,   t = (W(1)-W(-1))/2 = c1 + c3,
        //   u = (W(2) - c0 - 4c2 - 16c4)/2 = c1 + 4c3,
        // whence c2 = s - c0 - c4, c3 = (u - t)/3, c1 = t - c3. Every quotient
        // is exact.
        let c0 = v0;
        let c4 = v_inf;
        let s = bigint_div_exact(&v1.add(&vm1), 2);
        let t = bigint_div_exact(&v1.sub(&vm1), 2);
        let c2 = s.sub(&c0).sub(&c4);
        let four_c2 = bigint_shl_exact(&c2, 2);
        let sixteen_c4 = bigint_shl_exact(&c4, 4);
        let u = bigint_div_exact(&v2.sub(&c0).sub(&four_c2).sub(&sixteen_c4), 2);
        let c3 = bigint_div_exact(&u.sub(&t), 3);
        let c1 = t.sub(&c3);

        // Recompose Σ cᵢ·B^{ik} by Horner. The product's digits are all
        // non-negative, so this returns to unsigned.
        let shift = bit_span(k, 64);
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

    /// Toom-4 admission (four-way Toom–Cook: Toom, Soviet Physics–Doklady 3
    /// (1963), 714–716; Cook, *On the minimum computation time of
    /// functions*, Harvard thesis, 1966; the interpolation after Bodrato,
    /// WAIFI 2007), on the same shape as [`Self::should_use_toom3`]:
    /// both operands past `TOOM4_THRESHOLD_LIMBS` and within 1.5× in length.
    fn should_use_toom4(lhs: &Self, rhs: &Self) -> bool {
        let short = lhs.limbs.len().min(rhs.limbs.len());
        let long = lhs.limbs.len().max(rhs.limbs.len());
        short >= TOOM4_THRESHOLD_LIMBS && within_balanced_ratio(short, long)
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
    /// digits (divisions by 2, 3, 4, 5, 8, 12, all exact). Exponent
    /// `log 7 / log 4 ≈ 1.404`, below Toom-3's `1.465`.
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

            let at_1 = c0.add(c1).add(c2).add(c3); // c(1)
            let even = c0.add(c2); // c0 + c2
            let odd = c1.add(c3); // c1 + c3
            let at_m1 = BigInt::from_biguint(even).sub(&BigInt::from_biguint(odd)); // c(-1)
            let at_2 = c0.add(&two_c1).add(&four_c2).add(&eight_c3); // c(2)
            let even2 = c0.add(&four_c2); // c0 + 4c2
            let odd2 = two_c1.add(&eight_c3); // 2c1 + 8c3
            let at_m2 = BigInt::from_biguint(even2).sub(&BigInt::from_biguint(odd2)); // c(-2)

            // c(3) = c0 + 3c1 + 9c2 + 27c3, by Horner at x = 3.
            let three = BigUint::from_u64(3);
            let mut at_3 = c3.mul(&three);
            at_3.add_assign_ref(c2);
            at_3 = at_3.mul(&three);
            at_3.add_assign_ref(c1);
            at_3 = at_3.mul(&three);
            at_3.add_assign_ref(c0);
            (at_1, at_m1, at_2, at_m2, at_3)
        };
        let (a_1, a_m1, a_2, a_m2, a_3) = eval4(&a0, &a1, &a2, &a3);
        let (b_1, b_m1, b_2, b_m2, b_3) = eval4(&b0, &b1, &b2, &b3);

        // Seven pointwise products (each a recursive multiplication).
        let w0 = BigInt::from_biguint(a0.mul(&b0)); // W(0)
        let w1 = BigInt::from_biguint(a_1.mul(&b_1)); // W(1)
        let w2 = bigint_mul(&a_m1, &b_m1); // W(-1)
        let w3 = BigInt::from_biguint(a_2.mul(&b_2)); // W(2)
        let w4 = bigint_mul(&a_m2, &b_m2); // W(-2)
        let w5 = BigInt::from_biguint(a_3.mul(&b_3)); // W(3)
        let w6 = BigInt::from_biguint(a3.mul(&b3)); // W(∞)

        // Powers of two shift; the odd weights (9, 81, 729, 5, 3) go through
        // the general multiply.
        let scale = |x: &BigInt, m: u64| {
            if m.is_power_of_two() {
                bigint_shl_exact(x, m.trailing_zeros() as usize)
            } else {
                x.mul_biguint(&BigUint::from_u64(m))
            }
        };
        let c0 = w0;
        let c6 = w6;

        // Even coefficients c2, c4 from the symmetric sums.
        let e1 = bigint_div_exact(&w1.add(&w2), 2); // c2 + c4 + c0 + c6
        let e2 = bigint_div_exact(&w3.add(&w4), 2); // 4c2 + 16c4 + c0 + 64c6
        let sum24 = e1.sub(&c0).sub(&c6); // c2 + c4
        let weighted24 = e2.sub(&c0).sub(&scale(&c6, 64)); // 4c2 + 16c4
        let c4 = bigint_div_exact(&weighted24.sub(&scale(&sum24, 4)), 12);
        let c2 = sum24.sub(&c4);

        // Odd coefficients c1, c3, c5 from the antisymmetric sums and W(3).
        let o1 = bigint_div_exact(&w1.sub(&w2), 2); // c1 + c3 + c5
        let o2 = bigint_div_exact(&w3.sub(&w4), 4); // c1 + 4c3 + 16c5
        let o3 = bigint_div_exact(
            &w5.sub(&c0)
                .sub(&scale(&c2, 9))
                .sub(&scale(&c4, 81))
                .sub(&scale(&c6, 729)),
            3,
        ); // c1 + 9c3 + 81c5
        let p = bigint_div_exact(&o2.sub(&o1), 3); // c3 + 5c5
        let q = bigint_div_exact(&o3.sub(&o1), 8); // c3 + 10c5
        let c5 = bigint_div_exact(&q.sub(&p), 5);
        let c3 = p.sub(&scale(&c5, 5));
        let c1 = o1.sub(&c3).sub(&c5);

        // Recompose Σ cᵢ·B^{ik}; the product digits are all non-negative.
        let shift = bit_span(k, 64);
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
    /// A partial product or carry at or above position `limit` cannot
    /// affect any limb below it, so it is never computed. The result is
    /// exact, and costs about half the limb products of the full
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
        // The top limb of the buffer can be zero.
        result.normalize();
        result
    }

    /// Double the value: the single-bit case of [`Self::shl_bits`], with no
    /// whole-limb move and no `normalize`.
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
        // A left shift cannot introduce a leading zero limb.
    }

    /// Halve the value, discarding the low bit: `⌊self/2⌋`.
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
    /// The result is normalized, since cancellation can zero the top limbs.
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
    /// A whole-limb move of `n / 64` positions, then one pass shifting by
    /// the remaining `n % 64` bits, which keeps every `u64` shift amount
    /// below 64.
    pub fn shl_bits(&mut self, n: usize) {
        if self.is_zero() || n == 0 {
            return;
        }
        let limb_shifts = n / 64;
        let bit_shifts = n % 64;
        // Full-limb shift: prepend zeros at the low (index 0) end.
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
        // A left shift cannot introduce a leading zero limb.
    }

    /// Right-shift by `n` bits, discarding the shifted-out low bits.
    ///
    /// The mirror of [`Self::shl_bits`]: `⌊self / 2^n⌋`.
    pub fn shr_bits(&mut self, n: usize) {
        if self.is_zero() || n == 0 {
            return;
        }
        let limb_shifts = n / 64;
        let bit_shifts = (n % 64) as u32;

        if limb_shifts >= self.limbs.len() {
            // Everything shifts out. Wipe before clearing: `clear` only
            // shortens the vector, leaving the limbs in spare capacity where
            // the drop-time scrub cannot reach them.
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

        // Remaining bit-level shift (0 < bit_shifts < 64).
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
    /// with the quotient discarded. A caller that also needs the quotient
    /// should call `div_rem` once instead.
    ///
    /// # Panics
    ///
    /// Panics if `modulus == 0`.
    #[must_use]
    pub fn rem(&self, modulus: &Self) -> Self {
        let (_, remainder) = self.div_rem(modulus);
        remainder
    }

    /// Divide by a machine word, returning `(quotient, remainder)` in one
    /// pass — the word-sized companion to [`Self::div_rem`], with no
    /// heap-allocated divisor.
    ///
    /// # Panics
    ///
    /// Panics if `divisor == 0`.
    #[must_use]
    pub fn div_rem_u64(&self, divisor: u64) -> (Self, u64) {
        assert!(divisor != 0, "division by zero");
        Self::div_rem_limb(&self.limbs, divisor)
    }

    /// The value as a `u64` when it fits, `None` otherwise.
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
        // Horner's method in base `2^64`, from the top limb down.
        for &limb in self.limbs.iter().rev() {
            let acc = (remainder << 64) | u128::from(limb);
            remainder = acc % u128::from(modulus);
        }

        u64::try_from(remainder).expect("remainder modulo u64 fits into u64")
    }

    /// Compute `(lhs * rhs) mod modulus`.
    ///
    /// One multiply and one division, for any non-zero modulus. For a
    /// single product this beats a throwaway [`MontgomeryContext`], which
    /// costs a division to build plus encode, multiply and decode steps.
    /// Callers doing many multiplications under one modulus should build a
    /// [`MontgomeryContext`] once and reuse it.
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
        lhs.mul(rhs).rem(modulus)
    }

    /// One-shot modular addition on [`Self::mod_mul`]'s contract: any
    /// operands, non-zero modulus. Reduced operands take one
    /// compare-and-correct.
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
            lhs.rem(modulus)
        };
        let rhs = if rhs < modulus {
            rhs.clone()
        } else {
            rhs.rem(modulus)
        };
        let sum = lhs.add(&rhs);
        if sum >= *modulus {
            sum.sub(modulus)
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
            lhs.rem(modulus)
        };
        let rhs = if rhs < modulus {
            rhs.clone()
        } else {
            rhs.rem(modulus)
        };
        if lhs >= rhs {
            lhs.sub(&rhs)
        } else {
            modulus.add(&lhs).sub(&rhs)
        }
    }

    /// One-shot modular negation, on the same contract as
    /// [`Self::mod_add`]: any operand, non-zero modulus (panic otherwise).
    ///
    /// The result is in `[0, modulus)`: a multiple of the modulus negates to
    /// zero, any other `value` to `modulus − (value mod modulus)`.
    ///
    /// # Panics
    ///
    /// Panics if `modulus == 0`.
    #[must_use]
    pub fn mod_neg(value: &Self, modulus: &Self) -> Self {
        assert!(!modulus.is_zero(), "modulus must be non-zero");
        if value < modulus {
            return if value.is_zero() {
                Self::zero()
            } else {
                modulus.sub(value)
            };
        }
        let reduced = value.rem(modulus);
        if reduced.is_zero() {
            Self::zero()
        } else {
            modulus.sub(&reduced)
        }
    }

    /// Return `(quotient, remainder)` for Euclidean division, with the
    /// remainder in `[0, divisor)`.
    ///
    /// Dispatches on the divisor's width: a single-limb divisor takes a
    /// base-2⁶⁴ Horner division, a multi-limb divisor Knuth's Algorithm D
    /// (*TAOCP* vol. 2, §4.3.1), and a divisor of at least
    /// `NEWTON_DIVISION_THRESHOLD_LIMBS` (3072) limbs the subquadratic
    /// Newton reciprocal (`newton.rs`; Brent & Zimmermann, *Modern Computer
    /// Arithmetic*, §4.2.2). A dividend smaller than the divisor returns
    /// `(0, self)`.
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

        if divisor.limbs.len() == 1 {
            let (quotient, remainder) = Self::div_rem_limb(&self.limbs, divisor.limbs[0]);
            return (quotient, Self::from_u64(remainder));
        }
        // Wide divisors take Newton's reciprocal, which is O(M(k)) where
        // Algorithm D is O(k²); see `newton.rs` for the crossover.
        if divisor.limbs.len() >= newton::NEWTON_DIVISION_THRESHOLD_LIMBS {
            return newton::div_rem(self, divisor);
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
    /// `O(quotient_limbs * divisor_limbs)` limb operations.
    ///
    /// Variable-time: the quotient-digit corrections are data-dependent.
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
            // `q_hat = numerator / divisor_hi`, remainder `r_hat`.
            // Normalization guarantees `q_hat <= q + 2`.
            //
            // The loop's second test checks the estimate against the
            // divisor's second limb from the top; when it stops,
            // `q_hat <= q + 1` (TAOCP §4.3.1, exercise 20), the one overshoot
            // D6 repairs. Without it, divisors like `[v0, d, d, ...]` with
            // `d >= b/2` give an estimate two over the true digit.
            //
            // The `q_hat >= BASE` arm is Knuth's `min(q_hat, b - 1)` clamp.
            // With `q_hat` in `u128` it is redundant for correctness, but it
            // matches the published algorithm and skips a doomed subtraction.
            //
            // Termination: each round adds `divisor_hi >= b/2` to `r_hat`, so
            // the `r_hat >= BASE` break bounds the loop at two corrections.
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

            // After a correct step the window's value is below `b^n`, so
            // its top limb is zero.
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
    /// Every path that can leave a zero top limb must end here, because
    /// `Eq`, `Ord`, [`Self::bits`] and the kernel dispatch read the limb
    /// count as the value's width. The capacity is kept for reuse.
    fn normalize(&mut self) {
        while self.limbs.last().copied() == Some(0) {
            self.limbs.pop();
        }
    }

    /// The `BigUint`-facing wrapper around [`mont_mul`]: the kernels work on
    /// fixed-width limb slices, while a canonical `BigUint` is only as wide as
    /// its value, so this pads both operands to the modulus width and carves
    /// scratch, operand, and output windows out of one caller-owned
    /// workspace, so a sequence of operations allocates once.
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
        crate::scrub::zeroize_slice(workspace.as_mut_slice());
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

/// Why a modulus was refused when building a fixed-modulus context.
///
/// The variants describe the value, not the context that rejected it:
/// [`BarrettContext::new`] returns `Zero` or `One`, and
/// [`MontgomeryContext::new`] returns `Zero` or `Even`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ModulusError {
    /// Zero has no residues.
    Zero,
    /// Modulo one every residue is zero, so a context computes nothing.
    One,
    /// Montgomery reduction requires an odd modulus: `R = 2^(64w)` is
    /// invertible modulo `n` only when `n` is odd.
    Even,
}

impl core::fmt::Display for ModulusError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Zero => "zero is not a modulus",
            Self::One => "modulo one every residue is zero",
            Self::Even => "an even modulus has no Montgomery domain",
        })
    }
}

impl std::error::Error for ModulusError {}

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
    /// input yields [`ParseBigIntError`], which carries no position.
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

// ─── In-place operators ────────────────────────────────────────────────────
//
// The public spelling of in-place arithmetic is `x += &y`. The right-hand side
// is borrowed, so an accumulator reuses its limb buffer across a loop. These
// forward to the crate-private `add_assign_ref` and `sub_assign_ref`.

impl core::ops::AddAssign<&BigUint> for BigUint {
    /// `self += other`, reusing `self`'s limb buffer.
    fn add_assign(&mut self, other: &BigUint) {
        self.add_assign_ref(other);
    }
}

impl core::ops::SubAssign<&BigUint> for BigUint {
    /// `self -= other`, reusing `self`'s limb buffer.
    ///
    /// # Panics
    ///
    /// Panics if `self < other`: ℕ has no sign in which to record a negative
    /// difference. `BigInt`'s implementation is total.
    fn sub_assign(&mut self, other: &BigUint) {
        self.sub_assign_ref(other);
    }
}

impl core::ops::AddAssign<&BigInt> for BigInt {
    /// `self += other`, reusing the magnitude's limb buffer in every sign
    /// combination.
    fn add_assign(&mut self, other: &BigInt) {
        self.add_assign_ref(other);
    }
}

impl core::ops::SubAssign<&BigInt> for BigInt {
    /// `self -= other`, reusing the magnitude's limb buffer in every sign
    /// combination. Total: the sign follows the result.
    fn sub_assign(&mut self, other: &BigInt) {
        self.sub_assign_ref(other);
    }
}

/// `limbs · per_limb` as a bit index, refusing rather than wrapping.
///
/// Every conversion of a limb count into a bit position goes through here.
/// On a 64-bit target overflow is unreachable, but on a 32-bit target
/// `len · 64` wraps at operands of about 537 MB and `len · 128` at about
/// 268 MB, and a wrapped index is a silently wrong answer.
///
/// # Panics
///
/// Panics if the product exceeds `usize`, which means the operand cannot be
/// indexed by bit position on this target at all.
#[inline]
pub(crate) fn bit_span(limbs: usize, per_limb: usize) -> usize {
    limbs
        .checked_mul(per_limb)
        .expect("operand too wide to index by bit on this target")
}

#[cfg(feature = "wipe")]
impl Drop for BigUint {
    fn drop(&mut self) {
        // BigUint values may hold secrets — private exponents, prime
        // factors, nonces. Clear the limb buffer on drop so they do not
        // linger in freed heap memory.
        crate::scrub::zeroize_slice(self.limbs.as_mut_slice());
    }
}

/// The low 64 bits of a `u128` accumulator as a limb, by a masked
/// `try_from` rather than an `as` cast.
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

/// Return `value` shifted right by `shift` bits (below 64) in a fresh buffer
/// of the same width — the inverse of [`shl_into`], undoing Algorithm D's
/// normalization on the remainder in step D8. The caller normalizes.
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

/// Signed product `a · b`, for the Toom pointwise products at `−1` and `−2`.
fn bigint_mul(a: &BigInt, b: &BigInt) -> BigInt {
    a.mul(b)
}

/// `x / divisor` where `divisor` is known to divide `x` — the interpolation
/// steps of Toom-3 and Toom-4 (dividing by 2, 3, 4, 5, 8, 12).
///
/// Each quotient in the interpolation is an integer because the product
/// polynomial's coefficients are integers, so the remainder is checked only
/// in debug builds. The divisor is positive, so the sign is unchanged.
fn bigint_div_exact(x: &BigInt, divisor: u64) -> BigInt {
    debug_assert!(divisor > 0, "Toom interpolation never divides by zero");
    // Most divisors are 2, 4 or 8: a right shift, rather than a `u128`
    // division per limb.
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
/// A shift, rather than `mul_biguint` through the multiplication dispatch.
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
    /// magnitude becomes `Positive`. Callers constructing from untrusted
    /// parts should not rely on the sign surviving unchanged.
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

    /// Construct a non-negative signed integer from an unsigned magnitude;
    /// a zero magnitude yields canonical zero.
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

    /// [`BigUint::from_f64_lossy`] with the sign: the nearest integer to a
    /// finite double, or `None` for a non-finite one.
    #[must_use]
    pub fn from_f64_lossy(value: f64) -> Option<Self> {
        let magnitude = BigUint::from_f64_lossy(value.abs())?;
        let integer = Self::from_biguint(magnitude);
        Some(if value < 0.0 {
            integer.negated()
        } else {
            integer
        })
    }

    /// The nearest `f64`, sign included: [`BigUint::to_f64_lossy`] on the
    /// magnitude, negated for a negative value.
    #[must_use]
    pub fn to_f64_lossy(&self) -> f64 {
        let magnitude = self.magnitude.to_f64_lossy();
        match self.sign {
            Sign::Negative => -magnitude,
            _ => magnitude,
        }
    }

    /// Borrow the absolute value. Sign and magnitude are stored apart, so
    /// `|self|` is a borrow rather than a computation, and the unsigned
    /// kernels can be applied to it directly.
    #[must_use]
    pub fn magnitude(&self) -> &BigUint {
        &self.magnitude
    }

    /// Consume a signed value and return its magnitude without copying
    /// (crate-internal: number-theory workspaces recycle the limb buffer).
    pub(crate) fn into_magnitude(self) -> BigUint {
        self.magnitude
    }

    /// Return `-self`: the sign flips and the magnitude is copied. Zero
    /// negates to zero.
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

    /// Return `self + other`.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.add_assign_ref(other);
        out
    }

    /// Add another integer in place, reusing the magnitude's limb buffer in
    /// every sign combination.
    pub(crate) fn add_assign_ref(&mut self, other: &Self) {
        self.combine_assign(other.sign, &other.magnitude);
    }

    /// Return `self - other`. Total on ℤ, unlike [`BigUint::sub`].
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.sub_assign_ref(other);
        out
    }

    /// Subtract another integer in place, reusing the magnitude's limb
    /// buffer in every sign combination. Never panics.
    pub(crate) fn sub_assign_ref(&mut self, other: &Self) {
        let negated = match other.sign {
            Sign::Positive => Sign::Negative,
            Sign::Negative => Sign::Positive,
            Sign::Zero => Sign::Zero,
        };
        self.combine_assign(negated, &other.magnitude);
    }

    /// The shared core of the signed in-place operations:
    /// `self ← self + s·m`, where `(s, m)` is `other`'s decomposition for
    /// addition and its negation for subtraction. Like signs add
    /// magnitudes; unlike signs subtract the smaller from the larger, and
    /// the sign follows the larger. A zero result clears the buffer without
    /// releasing it.
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
                // Wipe before clearing: `Drop` covers only the initialized
                // prefix, and the capacity is kept for reuse.
                crate::scrub::zeroize_slice(self.magnitude.limbs.as_mut_slice());
                self.magnitude.limbs.clear();
                self.sign = Sign::Zero;
            }
        }
    }

    /// Return `self * factor` for an unsigned factor: the magnitudes
    /// multiply and the sign is unchanged (zero when `factor` is zero).
    #[must_use]
    pub fn mul_biguint(&self, factor: &BigUint) -> Self {
        if factor.is_zero() || self.sign == Sign::Zero {
            return Self::zero();
        }

        Self::from_parts(self.sign, self.magnitude.mul(factor))
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

    /// Signed product `self · other`: the magnitudes multiply through
    /// [`BigUint::mul`] and the sign follows the usual rule.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        let sign = match (self.sign, other.sign) {
            (Sign::Zero, _) | (_, Sign::Zero) => Sign::Zero,
            (lhs, rhs) if lhs == rhs => Sign::Positive,
            _ => Sign::Negative,
        };
        Self::from_parts(sign, self.magnitude.mul(&other.magnitude))
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
    /// convention would give `(-4, 1)`. For the least non-negative residue
    /// use [`Self::rem_euclid`].
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is zero, matching [`BigUint::div_rem`].
    #[must_use]
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        let (quotient, remainder) = self.magnitude.div_rem(&divisor.magnitude);
        // Magnitude division truncates; only the signs need assigning.
        (
            Self::from_parts(Self::quotient_sign(self.sign, divisor.sign), quotient),
            Self::from_parts(self.sign, remainder),
        )
    }

    /// The absolute value as an owned [`BigUint`]; [`Self::magnitude`]
    /// borrows it instead.
    #[must_use]
    pub fn abs(&self) -> BigUint {
        self.magnitude.clone()
    }

    /// Construct one: positive sign over a single-limb magnitude.
    #[must_use]
    pub fn one() -> Self {
        Self::from_parts(Sign::Positive, BigUint::one())
    }

    /// Whether the value is zero.
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
    /// to leave no remainder, as in polynomial interpolation and
    /// primitive-part steps.
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
    /// companion to [`Self::div_exact`], costing a single division. For
    /// callers that must decide divisibility, such as polynomial division
    /// over `ℤ`.
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
    /// A gcd over ℤ is defined up to sign, so this names the non-negative
    /// one, computed from the magnitudes by
    /// [`crate::number_theory::gcd`]; `gcd(0, 0)` is zero.
    #[must_use]
    pub(crate) fn gcd(&self, other: &Self) -> Self {
        Self::from_biguint(crate::number_theory_impl::gcd(
            &self.magnitude,
            &other.magnitude,
        ))
    }

    /// `self^exponent` for a machine-word exponent — the signed counterpart
    /// of [`BigUint::pow_u64`]. `self^0 = 1`.
    #[must_use]
    pub fn pow_u64(&self, exponent: u64) -> Self {
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
    /// Unlike Rust's `%`, whose remainder takes the dividend's sign, a
    /// negative value maps to `modulus − (|self| mod modulus)`, or zero when
    /// the modulus divides it.
    ///
    /// # Panics
    ///
    /// Panics if `modulus == 0`.
    #[must_use]
    pub fn rem_euclid(&self, modulus: &BigUint) -> BigUint {
        assert!(!modulus.is_zero(), "modulus must be non-zero");
        match self.sign {
            Sign::Zero => BigUint::zero(),
            Sign::Positive => self.magnitude.rem(modulus),
            Sign::Negative => {
                let rem = self.magnitude.rem(modulus);
                if rem.is_zero() {
                    BigUint::zero()
                } else {
                    modulus.sub(&rem)
                }
            }
        }
    }

    /// The representative of `self` modulo `modulus` in the *symmetric* range
    /// `(−modulus/2, modulus/2]`.
    ///
    /// The representative smallest in absolute value, where
    /// [`Self::rem_euclid`] gives the least non-negative one. An exact half
    /// (even `modulus` only) stays positive: `5 mod 10` is `5`, not `−5`.
    ///
    /// # Panics
    ///
    /// Panics if `modulus == 0`.
    ///
    /// ```
    /// use rump::{BigInt, BigUint};
    ///
    /// let ten = BigUint::from_u64(10);
    /// let reduced = |value: i64| BigInt::from_i64(value).symmetric_rem(&ten);
    /// assert_eq!(reduced(7), BigInt::from_i64(-3));
    /// assert_eq!(reduced(-7), BigInt::from_i64(3));
    /// assert_eq!(reduced(5), BigInt::from_i64(5));
    /// ```
    #[must_use]
    pub fn symmetric_rem(&self, modulus: &BigUint) -> BigInt {
        assert!(!modulus.is_zero(), "modulus must be non-zero");
        let reduced = self.rem_euclid(modulus);
        // reduced ∈ [0, modulus); above the midpoint, subtracting the modulus
        // lands it in (−modulus/2, 0).
        if reduced.mul(&BigUint::from_u64(2)) > *modulus {
            BigInt::from_biguint(reduced).sub(&BigInt::from_biguint(modulus.clone()))
        } else {
            BigInt::from_biguint(reduced)
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_double_becomes_the_integer_it_is() {
        for value in [
            0.0f64,
            1.0,
            2.5,
            3.5,
            1e15,
            2f64.powi(53),
            2f64.powi(90) * 1.5,
            1e300,
        ] {
            let integer = BigUint::from_f64_lossy(value).expect("finite and non-negative");
            let expected = value.round();
            assert_eq!(
                integer.to_f64_lossy(),
                expected,
                "{value} became {integer:?}"
            );
            if expected >= 2f64.powi(53) {
                // Exact: a double this large is an integer, and its
                // significand sits at its exponent.
                let bits = expected.to_bits();
                let significand = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
                let exponent = ((bits >> 52) & 0x7ff) as usize - 1075;
                let mut exact = BigUint::from_u64(significand);
                exact.shl_bits(exponent);
                assert_eq!(integer, exact);
            }
        }
        assert!(BigUint::from_f64_lossy(-1.0).is_none());
        assert!(BigUint::from_f64_lossy(f64::NAN).is_none());
        assert!(BigUint::from_f64_lossy(f64::INFINITY).is_none());
        assert_eq!(
            BigInt::from_f64_lossy(-2.5).expect("finite"),
            BigInt::from_i64(-3)
        );
        assert_eq!(
            BigInt::from_f64_lossy(-1e20).expect("finite"),
            BigInt::from_i128(-100_000_000_000_000_000_000)
        );
        assert!(BigInt::from_f64_lossy(f64::NEG_INFINITY).is_none());
    }

    #[test]
    fn symmetric_rem_is_congruent_and_smallest() {
        // The defining properties: congruent to `rem_euclid`, and no other
        // representative of the class is smaller in absolute value. Small
        // moduli of both parities, a prime, and a pair either side of a
        // thousand; values over ±60 wrap the small ones several times.
        for m in [1u64, 2, 7, 8, 97, 1_000, 1_001] {
            let modulus = BigUint::from_u64(m);
            for value in -60i64..=60 {
                let signed = BigInt::from_i64(value);
                let symmetric = signed.symmetric_rem(&modulus);
                assert_eq!(
                    symmetric.rem_euclid(&modulus),
                    signed.rem_euclid(&modulus),
                    "value {value} mod {m}: not the same residue"
                );
                // (−m/2, m/2]: doubled magnitude at most m, and equal only on
                // the positive side.
                let doubled = symmetric.magnitude().mul(&BigUint::from_u64(2));
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
        // The oracle is the full expansion.
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
            // At powers of the radix the logarithm is an integer and its
            // floating-point floor can fall either way. Exponents through
            // 39: radix 8 and above cross 2^64, radix 2 and 3 stay within a
            // word.
            for exponent in 0..40u64 {
                let power = base.pow_u64(exponent);
                check(&power, radix);
                check(&power.add(&BigUint::one()), radix);
                if exponent > 0 {
                    check(&power.sub(&BigUint::one()), radix);
                }
            }
        }
        // And far past anything a machine word reaches; the exponent is
        // arbitrary.
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
        BarrettContext, BigInt, BigUint, MontgomeryContext, Sign, KARATSUBA_THRESHOLD_LIMBS,
        NTT_SERIAL_THRESHOLD_LIMBS, RADIX_FROM_DC_THRESHOLD_DIGITS, RADIX_TO_DC_THRESHOLD_BITS,
        SQR_KARATSUBA_MAX_LIMBS, SQR_SCHOOLBOOK_MIN_LIMBS, TOOM3_THRESHOLD_LIMBS,
        TOOM4_THRESHOLD_LIMBS, UNBALANCED_THRESHOLD_LIMBS,
    };
    use super::{ModulusError, MontgomeryScratch};
    use core::num::NonZeroU64;

    // Knuth's MMIX multiplier (TAOCP vol. 2, §3.3.4). With an odd increment
    // and a multiplier that is 1 mod 4 the generator has full period 2^64;
    // 1 is the smallest odd increment.
    const LCG_MULTIPLIER: u64 = 6364136223846793005;
    const LCG_INCREMENT: u64 = 1;

    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LCG_INCREMENT);
        *state
    }

    // Width from which the NTT timing probes take fewer samples and rounds:
    // one product at 65,536 limbs takes long enough that repeated draws cost
    // more wall time than the noise they remove. The rounds are best-of
    // counts, three below the width and two at or above it.
    const NTT_PROBE_WIDE_WORDS: usize = 65_536;
    const NTT_PROBE_ROUNDS: usize = 3;
    const NTT_PROBE_ROUNDS_WIDE: usize = 2;

    /// Divisors that exercise every branch of the normalization: already
    /// normalized, one below a power of two, one above, the extremes, small
    /// primes, and random odd words.
    fn reciprocal_divisor_corners() -> Vec<u64> {
        let mut divisors = vec![
            1,
            2,
            3,
            7,
            (1 << 16) - 1,
            1 << 16,
            (1 << 16) + 1,
            20_011,
            1_000_003,
            (1 << 32) - 1,
            1 << 32,
            (1 << 32) + 1,
            (1 << 63) - 1,
            1 << 63,
            (1 << 63) + 1,
            u64::MAX - 1,
            u64::MAX,
        ];
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x5eed_1234;
        let mut state = SEED;
        // Random odd words beside the corners; the count is arbitrary.
        for _ in 0..32 {
            divisors.push(lcg_next(&mut state) | 1);
        }
        divisors
    }

    /// The reciprocal path must agree with the hardware-division path on
    /// every input, for both the quotient and the remainder. `div_rem_u64`
    /// is the independently tested oracle.
    #[test]
    fn reciprocal_agrees_with_hardware_division_on_words() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xabcd_ef01;
        let mut state = SEED;
        for divisor in reciprocal_divisor_corners() {
            let r = super::WordReciprocal::new(
                NonZeroU64::new(divisor).expect("corner divisors are non-zero"),
            );
            assert_eq!(r.divisor(), divisor);
            let mut values = vec![0u64, 1, divisor.wrapping_sub(1), divisor, u64::MAX];
            if let Some(next) = divisor.checked_add(1) {
                values.push(next);
            }
            // The listed edges plus 64 random words per divisor; the count
            // is arbitrary.
            for _ in 0..64 {
                values.push(lcg_next(&mut state));
            }
            for value in values {
                let oracle = BigUint::from_u64(value);
                let (expected_q, expected_r) = oracle.div_rem_u64(divisor);
                assert_eq!(
                    r.div_rem(value),
                    (expected_q.to_u64().expect("word quotient fits"), expected_r),
                    "div_rem_u64({value}) by {divisor}"
                );
                assert_eq!(r.rem(value), expected_r, "rem_u64({value}) by {divisor}");
            }
        }
    }

    /// The multi-limb path is the same kernel driven by Horner's recurrence,
    /// so it is checked against the same oracle across widths — including one
    /// limb, where the normalization's top word is the only carry.
    #[test]
    fn reciprocal_agrees_with_hardware_division_on_bignums() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x1357_9bdf;
        let mut state = SEED;
        for divisor in reciprocal_divisor_corners() {
            let r = super::WordReciprocal::new(
                NonZeroU64::new(divisor).expect("corner divisors are non-zero"),
            );
            // One limb, small widths either side of a power of two, and a
            // wide one; four random values each.
            for words in [1usize, 2, 3, 5, 8, 17, 64] {
                for _ in 0..4 {
                    let value = seeded_biguint(words, &mut state);
                    let (expected_q, expected_r) = value.div_rem_u64(divisor);
                    let (got_q, got_r) = value.div_rem_reciprocal(&r);
                    assert_eq!(got_q, expected_q, "quotient at {words} words by {divisor}");
                    assert_eq!(got_r, expected_r, "remainder at {words} words by {divisor}");
                    assert_eq!(value.rem_reciprocal(&r), expected_r);
                }
            }
            assert_eq!(BigUint::zero().rem_reciprocal(&r), 0);
            assert_eq!(BigUint::zero().div_rem_reciprocal(&r).0, BigUint::zero());
        }
    }

    /// `rem_euclid_i64` must land in `0..divisor` for negative inputs too.
    /// `i64::rem_euclid` is the oracle
    /// wherever the divisor fits a positive `i64`; `i64::MIN` is included
    /// because its magnitude is not representable as a positive `i64`.
    #[test]
    fn reciprocal_rem_euclid_matches_the_signed_oracle() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x2468_ace0;
        let mut state = SEED;
        for divisor in reciprocal_divisor_corners() {
            let r = super::WordReciprocal::new(
                NonZeroU64::new(divisor).expect("corner divisors are non-zero"),
            );
            let mut values = vec![0i64, 1, -1, i64::MAX, i64::MIN];
            // The signed edges plus 64 random words per divisor, about half
            // of them negative; the count is arbitrary.
            for _ in 0..64 {
                values.push(lcg_next(&mut state) as i64);
            }
            for value in values {
                let got = r.rem_euclid_i64(value);
                assert!(got < divisor, "residue {got} not below {divisor}");
                // The two-word form agrees on every one-word value, and on
                // the value widened by a word.
                assert_eq!(r.rem_euclid_i128(i128::from(value)), got);
                let wide = i128::from(value) << 64 | i128::from(value.unsigned_abs() >> 1);
                let wide_expected = BigInt::from_i128(wide).rem_euclid(&BigUint::from_u64(divisor));
                assert_eq!(
                    BigUint::from_u64(r.rem_euclid_i128(wide)),
                    wide_expected,
                    "rem_euclid_i128({wide}) by {divisor}"
                );
                if let Ok(signed) = i64::try_from(divisor) {
                    assert_eq!(
                        got,
                        value.rem_euclid(signed) as u64,
                        "rem_euclid_i64({value}) by {divisor}"
                    );
                }
                // Independent of the oracle: the residue must differ from the
                // value by a multiple of the divisor.
                let magnitude = BigUint::from_u64(value.unsigned_abs());
                let residue = magnitude.rem_u64(divisor);
                let expected = if value < 0 && residue != 0 {
                    divisor - residue
                } else {
                    residue
                };
                assert_eq!(got, expected);
            }
        }
    }

    /// Zero is excluded by the argument type, so there is nothing to test at
    /// run time; this pins that the boundary divisors do build and work.
    #[test]
    fn reciprocal_accepts_the_boundary_divisors() {
        for d in [1u64, 2, u64::MAX - 1, u64::MAX] {
            let r = super::WordReciprocal::new(NonZeroU64::new(d).expect("non-zero"));
            assert_eq!(r.divisor(), d);
            // An arbitrary dividend, fixed.
            assert_eq!(r.rem(12_345), 12_345 % d);
        }
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
        assert_eq!(a.add(&b), BigUint::from_u128(1_777_777_777_777));
        assert_eq!(
            a.sub(&BigUint::from_u64(1)),
            BigUint::from_u128(999_999_999_999)
        );
        assert_eq!(
            a.mul(&b),
            BigUint::from_u128(777_777_777_777_000_000_000_000)
        );
    }

    /// Oracle for the signed in-place operations: signed addition as a case
    /// analysis over the unsigned primitives.
    fn signed_add_oracle(a: &BigInt, b: &BigInt) -> BigInt {
        use core::cmp::Ordering;
        match (a.sign(), b.sign()) {
            (Sign::Zero, _) => b.clone(),
            (_, Sign::Zero) => a.clone(),
            (sa, sb) if sa == sb => BigInt::from_parts(sa, a.magnitude().add(b.magnitude())),
            (sa, sb) => match a.magnitude().cmp(b.magnitude()) {
                Ordering::Greater => BigInt::from_parts(sa, a.magnitude().sub(b.magnitude())),
                Ordering::Less => BigInt::from_parts(sb, b.magnitude().sub(a.magnitude())),
                Ordering::Equal => BigInt::zero(),
            },
        }
    }

    /// Integer square root by bisection: an independent oracle for the
    /// Newton iteration.
    fn sqrt_floor_bisection(n: &BigUint) -> BigUint {
        if n.is_zero() || n.is_one() {
            return n.clone();
        }
        let mut low = BigUint::one();
        let mut high = BigUint::zero();
        high.set_bit(n.bits().div_ceil(2));
        while low.add(&BigUint::one()) < high {
            let mut middle = low.add(&high);
            middle.shr1();
            if middle.square() <= *n {
                low = middle;
            } else {
                high = middle;
            }
        }
        low
    }

    #[test]
    fn float_estimates_match_reference() {
        // Exact on values within f64's integer range: 2,000 random words
        // shifted down by up to 39 bits, so widths from about 25 to 64 bits
        // appear on both sides of the 53-bit exact range.
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xf10a_7000_0000_0001;
        let mut seed = SEED;
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
        // Powers of two land exactly, from the first past u64 up to below
        // f64's exponent limit at 2^1024.
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
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xd10e_5eed_0000_0001;
        let mut seed = SEED;
        // 2,000 random dividends of 1 to 20 limbs, each against a random odd
        // word divisor.
        for _ in 0..2000 {
            let words = 1 + (lcg_next(&mut seed) % 20) as usize;
            let n = seeded_biguint(words, &mut seed);
            let d = (lcg_next(&mut seed) | 1).max(2);
            let (q, r) = n.div_rem_u64(d);
            let (q_ref, r_ref) = n.div_rem(&BigUint::from_u64(d));
            assert_eq!(q, q_ref, "quotient");
            assert_eq!(BigUint::from_u64(r), r_ref, "remainder");
            // The defining identity.
            assert_eq!(q.mul(&BigUint::from_u64(d)).add(&BigUint::from_u64(r)), n);
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
    fn mod_neg_matches_machine_arithmetic() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x6e6e_9a7e_0000_0001;
        let mut seed = SEED;
        // 2,000 random moduli and values of every width up to a word, the
        // value not reduced.
        for _ in 0..2000 {
            let m = (lcg_next(&mut seed) >> (lcg_next(&mut seed) % 64)).max(1);
            let a = lcg_next(&mut seed) >> (lcg_next(&mut seed) % 64);
            let (bm, ba) = (BigUint::from_u64(m), BigUint::from_u64(a));
            let negated = BigUint::mod_neg(&ba, &bm);
            // Unreduced operands are within the contract; the result is not.
            let expected = (u128::from(m) - u128::from(a % m)) % u128::from(m);
            assert_eq!(negated, BigUint::from_u128(expected));
            assert!(negated < bm);
            assert!(BigUint::mod_add(&ba, &negated, &bm).is_zero());
            assert_eq!(BigUint::mod_sub(&BigUint::zero(), &ba, &bm), negated);
        }

        // Multi-limb, at each edge of the range and past it.
        let mut modulus = BigUint::zero();
        modulus.set_bit(130);
        let modulus = modulus.add(&BigUint::from_u64(27));
        let one = BigUint::one();
        let three = BigUint::from_u64(3);
        assert!(BigUint::mod_neg(&BigUint::zero(), &modulus).is_zero());
        assert!(BigUint::mod_neg(&modulus, &modulus).is_zero());
        assert_eq!(BigUint::mod_neg(&one, &modulus), modulus.sub(&one));
        assert_eq!(BigUint::mod_neg(&modulus.sub(&one), &modulus), one);
        let unreduced = modulus.mul(&BigUint::from_u64(5)).add(&three);
        assert_eq!(BigUint::mod_neg(&unreduced, &modulus), modulus.sub(&three));
        let multiple = modulus.mul(&BigUint::from_u64(7));
        assert!(BigUint::mod_neg(&multiple, &modulus).is_zero());
        // Modulus one: the ring has one element.
        assert!(BigUint::mod_neg(&unreduced, &one).is_zero());
    }

    #[test]
    #[should_panic(expected = "modulus must be non-zero")]
    fn mod_neg_rejects_zero_modulus() {
        let _ = BigUint::mod_neg(&BigUint::one(), &BigUint::zero());
    }

    #[test]
    fn mod_add_sub_match_machine_arithmetic() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x30d5_0bad_0000_0001;
        let mut seed = SEED;
        // 2,000 random odd moduli of 32 to 64 bits with operands of the same
        // range, so sums carry past a word and differences borrow.
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
        // Against a Barrett context's modulus.
        let ctx = super::BarrettContext::new(&BigUint::from_u64(1000)).expect("modulus >= 2");
        assert_eq!(
            BigUint::mod_add(
                &BigUint::from_u64(999),
                &BigUint::from_u64(2),
                ctx.modulus()
            ),
            BigUint::from_u64(1)
        );
        assert_eq!(
            BigUint::mod_sub(
                &BigUint::from_u64(2),
                &BigUint::from_u64(999),
                ctx.modulus()
            ),
            BigUint::from_u64(3)
        );
        // from_i64 round-trips both signs into the canonical range.
        assert_eq!(
            BigInt::from_i64(-3).rem_euclid(&BigUint::from_u64(11)),
            BigUint::from_u64(8)
        );
        assert_eq!(
            BigInt::from_i64(3).rem_euclid(&BigUint::from_u64(11)),
            BigUint::from_u64(3)
        );
        assert_eq!(BigInt::from_i64(0), BigInt::zero());
        assert_eq!(
            BigInt::from_i64(i64::MIN),
            BigInt::from_parts(Sign::Negative, BigUint::from_u64(1u64 << 63))
        );
    }

    /// The gcd of a residue with the modulus is the gcd of the value it
    /// encodes: for zero, one, a unit, each proper factor and their
    /// products, and `n − 1`, under a three-prime modulus and under the
    /// widest odd two-limb modulus, `2¹²⁸ − 1`; a residue from another
    /// context is refused.
    #[test]
    fn a_residue_gcd_with_the_modulus_is_the_gcd_of_its_value() {
        use crate::number_theory_impl::gcd;
        let (p, q, r) = (
            BigUint::from_u64(1_000_000_007),
            BigUint::from_u64(998_244_353),
            BigUint::from_u64(4_294_967_291),
        );
        let three_primes = p.mul(&q).mul(&r);
        // 2^128 - 1 = 3 · 5 · 17 · 257 · 641 · 65537 · 274177 · 6700417 · 67280421310721.
        let widest = BigUint::from_limbs(vec![u64::MAX, u64::MAX]);
        let two_factors = BigUint::from_u64(641 * 65_537);
        for (n, factors) in [
            (
                three_primes.clone(),
                vec![p.clone(), q.clone(), r.clone(), p.mul(&q)],
            ),
            (
                widest.clone(),
                vec![
                    BigUint::from_u64(3),
                    BigUint::from_u64(67_280_421_310_721),
                    two_factors,
                ],
            ),
        ] {
            let context = MontgomeryContext::new(&n).expect("odd modulus");
            let mut values = vec![
                BigUint::zero(),
                BigUint::one(),
                BigUint::from_u64(2),
                n.sub(&BigUint::one()),
            ];
            for factor in &factors {
                values.push(factor.clone());
                // An arbitrary multiplier, fixed: a multiple of the factor
                // that is not the factor itself.
                values.push(factor.mul(&BigUint::from_u64(12_345)).rem(&n));
            }
            for value in values {
                let residue = context.to_residue(&value);
                assert_eq!(
                    context.gcd_with_modulus(&residue).expect("same context"),
                    gcd(&value, &n),
                    "gcd({value}, {n})"
                );
            }
            assert_eq!(
                context.gcd_with_modulus(&context.to_residue(&BigUint::zero())),
                Ok(n.clone())
            );
        }
        let other = MontgomeryContext::new(&three_primes).expect("odd modulus");
        let context = MontgomeryContext::new(&widest).expect("odd modulus");
        assert_eq!(
            context.gcd_with_modulus(&other.to_residue(&BigUint::from_u64(3))),
            Err(crate::modular::ContextMismatch)
        );
    }

    #[test]
    fn montgomery_domain_add_sub_match_plain_arithmetic() {
        use super::MontgomeryContext;
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x0a0d_50b7_0000_0001;
        let mut seed = SEED;
        // One limb, a few, and a few dozen; 16 random pairs each.
        for &words in &[1usize, 4, 32] {
            let mut n = seeded_biguint(words, &mut seed);
            n.limbs[0] |= 1;
            let ctx = MontgomeryContext::new(&n).expect("odd modulus");
            for _ in 0..16 {
                let a = seeded_biguint(words, &mut seed).rem(&n);
                let b = seeded_biguint(words, &mut seed).rem(&n);
                let (am, bm) = (ctx.encode(&a), ctx.encode(&b));
                // Linearity: the domain sum decodes to the plain sum.
                let sum = a.add(&b).rem(&n);
                assert_eq!(ctx.decode(&ctx.add_mont(&am, &bm)), sum);
                let diff = if a >= b { a.sub(&b) } else { n.add(&a).sub(&b) };
                assert_eq!(ctx.decode(&ctx.sub_mont(&am, &bm)), diff);
            }
            // Boundaries: zero, self-cancellation, both identities, the
            // wrap, and the add correction's own boundary x + (n − x) = n.
            let zero = BigUint::zero();
            let top = ctx.encode(&n.sub(&BigUint::one()));
            assert_eq!(ctx.sub_mont(&top, &top), zero);
            assert_eq!(ctx.add_mont(&top, &zero), top);
            assert_eq!(ctx.sub_mont(&top, &zero), top);
            assert!(
                ctx.add_mont(&top, &ctx.sub_mont(&zero, &top)).is_zero(),
                "x + (n - x) folds to exactly zero"
            );
            assert_eq!(
                ctx.decode(&ctx.add_mont(&top, &top)),
                n.sub(&BigUint::from_u64(2)),
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
        use super::BarrettContext;
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xba22_e77e_0000_0001;
        let mut seed = SEED;
        // Widths from one limb up by factors of four, each with an odd and
        // an even modulus; eight random products each.
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
                let ctx = BarrettContext::new(&n).expect("modulus is at least 2");
                for _ in 0..8 {
                    // Products of reduced values — the advertised domain.
                    let a = seeded_biguint(words, &mut seed).rem(&n);
                    let b = seeded_biguint(words, &mut seed).rem(&n);
                    let wide = a.mul(&b);
                    assert_eq!(ctx.reduce(&wide), wide.rem(&n));
                    assert_eq!(ctx.mod_mul(&a, &b), BigUint::mod_mul(&a, &b, &n));
                    assert_eq!(ctx.mod_square(&a), BigUint::mod_mul(&a, &a, &n));
                }
                // The full-width boundary of the contract: b^2k − 1.
                let mut edge = BigUint::zero();
                edge.set_bit(128 * words);
                let edge = edge.sub(&BigUint::one());
                assert_eq!(ctx.reduce(&edge), edge.rem(&n));
                // Beyond the contract the fallback still answers.
                let mut wide = seeded_biguint(3 * words, &mut seed);
                wide.set_bit(3 * words * 64 - 1);
                assert_eq!(ctx.reduce(&wide), wide.rem(&n));
                assert_eq!(ctx.reduce(&BigUint::zero()), BigUint::zero());
            }
        }
        // Tiny and structured moduli.
        for n_small in [2u64, 3, 4, 16, 255, 256, 257, u64::MAX] {
            let n = BigUint::from_u64(n_small);
            let ctx = BarrettContext::new(&n).expect("at least 2");
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
                let x2 = BigUint::from_u64(x).square();
                assert_eq!(ctx.reduce(&x2), x2.rem(&n), "n = {n_small}, x = {x}");
            }
        }
        assert_eq!(BarrettContext::new(&BigUint::one()), Err(ModulusError::One));
        assert_eq!(
            BarrettContext::new(&BigUint::zero()),
            Err(ModulusError::Zero)
        );
        // The tightest shapes for the quotient estimate: moduli at the
        // limb-boundary edges, b^(k-1) and b^k - 1, at the two narrowest
        // multi-limb widths and a wider one; six random residues each.
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
                    v.add(&BigUint::one())
                },
                {
                    let mut v = BigUint::zero();
                    v.set_bit(64 * k);
                    v.sub(&BigUint::one())
                },
            ] {
                let ctx = BarrettContext::new(&n).expect("at least 2");
                // Arbitrary, fixed so a failure reproduces; varied by width.
                const EDGE_SEED: u64 = 0x0b0b_0b0b_0000_0001;
                let mut seed2 = EDGE_SEED ^ (k as u64);
                for _ in 0..6 {
                    let a = seeded_biguint(k, &mut seed2).rem(&n);
                    let wide = a.square();
                    assert_eq!(ctx.reduce(&wide), wide.rem(&n), "edge modulus, k = {k}");
                }
            }
        }
    }

    #[test]
    fn barrett_reduce_straddles_the_half_product_cutoff() {
        // Above `BARRETT_HALF_PRODUCT_MAX_LIMBS`, `reduce` takes the window
        // from a dispatched full product instead of the schoolbook half
        // product. No other test builds a modulus that wide.
        //
        // The two arms compute equal values, so a misplaced cutoff changes
        // only timing; no value-based test can catch that.
        use super::{BarrettContext, BARRETT_HALF_PRODUCT_MAX_LIMBS};
        let cutoff = BARRETT_HALF_PRODUCT_MAX_LIMBS;
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x5a11_b0bb_0000_0001;
        let mut seed = SEED;
        for k in [cutoff - 1, cutoff, cutoff + 1, cutoff + 2] {
            // Four modulus shapes at each width. The even one exercises an
            // even modulus at width and adds an independent μ; a top limb of
            // 1 maximizes μ and all-ones maximizes the quotient estimate's
            // error, stressing `q̂` from both ends.
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
            shapes.push(("all ones", ones.sub(&BigUint::one())));

            for (label, n) in shapes {
                assert_eq!(n.bits().div_ceil(64), k, "{label}: k = {k} as intended");
                let ctx = BarrettContext::new(&n).expect("a modulus of at least 2");
                // Two random pairs per shape; the listed edges below do the
                // stressing.
                for _ in 0..2 {
                    let a = seeded_biguint(k, &mut seed).rem(&n);
                    let b = seeded_biguint(k, &mut seed).rem(&n);
                    let wide = a.mul(&b);
                    assert_eq!(ctx.reduce(&wide), wide.rem(&n), "{label}, k = {k}");
                }
                // The ends of the range, where an off-by-one in the
                // correction loop shows up: 0, n − 1, n itself, the largest
                // square, and the top of the accepted input range.
                assert!(ctx.reduce(&BigUint::zero()).is_zero(), "{label}, zero");
                let below = n.sub(&BigUint::one());
                assert_eq!(ctx.reduce(&below), below, "{label}, n - 1");
                assert!(ctx.reduce(&n).is_zero(), "{label}, n");
                let square = below.square();
                assert_eq!(ctx.reduce(&square), square.rem(&n), "{label}, (n-1)^2");
                let mut widest = BigUint::zero();
                widest.set_bit(128 * k);
                let widest = widest.sub(&BigUint::one());
                assert_eq!(
                    ctx.reduce(&widest),
                    widest.rem(&n),
                    "{label}, the widest accepted input"
                );
            }
        }
    }

    #[test]
    #[ignore = "search for a two-correction witness; run with --ignored"]
    fn barrett_correction_search() {
        use super::BarrettContext;
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x7777_0000_0000_0001;
        let mut seed = SEED;
        let mut seen = [0usize; 4];
        let mut witness: Option<(String, String)> = None;
        // One to four limbs.
        for k in 1usize..=4 {
            let mut shapes: Vec<(String, BigUint)> = Vec::new();
            for d in [1u64, 3, 5, 7, 9, 17, 33, 65, 257, 1025] {
                let mut n = BigUint::zero();
                n.set_bit(64 * (k - 1));
                shapes.push((format!("b^{}+{d}", k - 1), n.add(&BigUint::from_u64(d))));
                let mut m = BigUint::zero();
                m.set_bit(64 * k);
                shapes.push((format!("b^{k}-{d}"), m.sub(&BigUint::from_u64(d))));
                shapes.push((format!("b^{k}+{d}"), m.add(&BigUint::from_u64(d))));
            }
            // Forty random moduli per width beside the structured ones; the
            // count is arbitrary.
            for _ in 0..40 {
                let mut r = seeded_biguint(k, &mut seed);
                r.set_bit(64 * k - 1);
                shapes.push(("random".into(), r));
            }
            for (label, n) in shapes {
                if n.bits() < 2 {
                    continue;
                }
                let ctx = BarrettContext::new(&n).expect("ok");
                let kk = n.bits().div_ceil(64);
                let mut top = BigUint::zero();
                top.set_bit(128 * kk);
                let top = top.sub(&BigUint::one());
                // 600 draws per modulus, cycling three constructions; the
                // count is arbitrary.
                for _ in 0..600 {
                    let x = match seed % 3 {
                        0 => seeded_biguint(2 * kk, &mut seed).rem(&top),
                        1 => top.sub(&seeded_biguint(kk, &mut seed).rem(&top)),
                        _ => n.mul(&seeded_biguint(kk, &mut seed)).rem(&top),
                    };
                    assert_eq!(ctx.reduce(&x), x.rem(&n));
                    let t = BarrettContext::last_corrections() as usize;
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
        // `barrett_correction_bound_is_attained_and_not_exceeded` cites this
        // probe, so it asserts the bound rather than only printing it.
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
        // HAC Note 14.44 bounds `q̂`'s shortfall at two. A bound never
        // reached is indistinguishable from a wrong one, so this demands the
        // tight case occur.
        //
        // Two corrections concentrate on moduli just above a power of the
        // base (`b² + 1` is the readiest witness); random moduli miss them.
        // The wider sweep `barrett_correction_search` (run with `--ignored`)
        // finds 678 two-correction reductions in 168 000, and none with
        // three.
        use super::BarrettContext;
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xc0de_1044_0000_0001;
        let mut seed = SEED;
        let mut seen = [0usize; 3];
        // The narrowest multi-limb widths.
        for k in 2usize..=4 {
            let mut shapes = Vec::new();
            for d in [1u64, 3, 5, 17, 257] {
                let mut n = BigUint::zero();
                n.set_bit(64 * (k - 1));
                shapes.push(n.add(&BigUint::from_u64(d)));
            }
            let mut random = seeded_biguint(k, &mut seed);
            random.set_bit(64 * k - 1);
            shapes.push(random);

            for n in shapes {
                if n.bits() < 2 {
                    continue;
                }
                let width = n.bits().div_ceil(64);
                let ctx = BarrettContext::new(&n).expect("a modulus of at least 2");
                let mut top = BigUint::zero();
                top.set_bit(128 * width);
                let top = top.sub(&BigUint::one());
                // 250 draws per modulus; the count is arbitrary.
                for step in 0..250u64 {
                    // Uniform over the accepted range, and multiples of `n`
                    // near it — the two draws the witnesses come from.
                    let x = if step % 2 == 0 {
                        seeded_biguint(2 * width, &mut seed).rem(&top)
                    } else {
                        n.mul(&seeded_biguint(width, &mut seed)).rem(&top)
                    };
                    assert_eq!(ctx.reduce(&x), x.rem(&n), "k = {k}, step {step}");
                    let taken = BarrettContext::last_corrections();
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
        use super::BarrettContext;
        use crate::number_theory_impl::mod_pow;

        // An independent reference: square-and-multiply with a full product
        // and a division at every step, sharing no code with either context.
        // `mod_pow` routes even moduli to `BarrettContext`, so comparing the
        // two against each other would test nothing.
        fn reference_pow(base: &BigUint, exponent: &BigUint, modulus: &BigUint) -> BigUint {
            if modulus.is_one() {
                return BigUint::zero();
            }
            let mut result = BigUint::one().rem(modulus);
            let mut power = base.rem(modulus);
            for bit in 0..exponent.bits() {
                if exponent.bit(bit) {
                    result = result.mul(&power).rem(modulus);
                }
                power = power.mul(&power).rem(modulus);
            }
            result
        }

        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xba22_e77e_0000_0002;
        let mut seed = SEED;
        // One, four and sixteen limbs, each odd and even, with a two-limb
        // exponent.
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
                let ctx = BarrettContext::new(&n).expect("at least 2");
                let base = seeded_biguint(words, &mut seed);
                let exponent = seeded_biguint(2, &mut seed);
                let expected = reference_pow(&base, &exponent, &n);
                // Both public routes against the independent ladder.
                assert_eq!(
                    ctx.mod_pow(&base, &exponent),
                    expected,
                    "BarrettContext::mod_pow at {words} words, even = {parity_even}"
                );
                assert_eq!(
                    mod_pow(&base, &exponent, &n),
                    expected,
                    "mod_pow at {words} words, even = {parity_even}"
                );
                assert_eq!(ctx.mod_pow(&base, &BigUint::zero()), BigUint::one().rem(&n));
            }
        }

        // Corners the random sweep will not reach, all even moduli: the smallest
        // modulus, powers of two, a non-power-of-two even modulus, a
        // multi-limb even modulus, exponents 0 and 1, and bases far wider
        // than the modulus. 2^300 + 2 is the multi-limb even modulus and
        // 2^700 + 12,345 a base more than twice its width; the widths and
        // the offset are arbitrary and fixed.
        let mut wide = BigUint::one();
        wide.shl_bits(300);
        let wide_even = wide.add(&BigUint::from_u64(2));
        let mut base_wider = BigUint::one();
        base_wider.shl_bits(700);
        base_wider = base_wider.add(&BigUint::from_u64(12_345));
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
                    let ctx = BarrettContext::new(&modulus).expect("at least 2");
                    assert_eq!(
                        ctx.mod_pow(&base, &exponent),
                        expected,
                        "BarrettContext corner: base {base}, exponent {exponent}, modulus {modulus}"
                    );
                }
            }
        }
    }

    #[test]
    fn sqrt_rem_matches_bisection_and_certifies() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x5eed_0006_0001_0001;
        let mut seed = SEED;
        // From one limb to past the Karatsuba threshold; six random values
        // each.
        for &words in &[1usize, 2, 8, 32, 128] {
            for _ in 0..6 {
                let n = seeded_biguint(words, &mut seed);
                let (root, remainder) = n.sqrt_rem();
                assert_eq!(
                    root,
                    sqrt_floor_bisection(&n),
                    "root diverged at {words} words"
                );
                assert_eq!(remainder, n.sub(&root.square()));
                assert!(root.add(&BigUint::one()).square() > n, "floor certificate");
            }
        }
        // Exact squares and their neighbours, from roots of two limbs to
        // roots whose squares reach the Karatsuba squaring regime.
        // Arbitrary, fixed so a failure reproduces.
        const SEED_SQUARES: u64 = 0x0bad_cafe_0000_0007;
        let mut seed2 = SEED_SQUARES;
        for &words in &[2usize, 16, 64] {
            let r = seeded_biguint(words, &mut seed2);
            let square = r.square();
            assert_eq!(square.sqrt_rem(), (r.clone(), BigUint::zero()));
            let below = square.sub(&BigUint::one());
            let r_minus_one = r.sub(&BigUint::one());
            assert_eq!(
                below.sqrt_rem().0,
                r_minus_one,
                "just below a square roots to r - 1"
            );
        }
    }

    #[test]
    fn predicates_match_machine_arithmetic() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x1234_5678_0000_0001;
        let mut seed = SEED;
        // 3,000 random words shifted down by up to 39 bits, so every width
        // from about 25 to 64 bits appears.
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
        // Below 2,000 each listed root index has an exact power in range,
        // 2^7 = 128 the last; the indices are the first four primes, and a
        // composite index follows from its factors.
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
                                       // Every exponent with a base-2 power inside u64 (2^63 fits).
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
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x0f0f_0f0f_5eed_0001;
        let mut seed = SEED;
        // Bases of 8 to 40 limbs; the powers reach 168 limbs (24·7), past
        // the Karatsuba threshold.
        for &(words, k) in &[(8usize, 3u64), (16, 5), (40, 2), (24, 7)] {
            let m = seeded_biguint(words, &mut seed);
            let power = m.pow_u64(k);
            assert_eq!(power.nth_root_floor(k), m, "exact {k}-th root");
            assert!(power.is_perfect_power());
            let bumped = power.add(&BigUint::one());
            // m^k + 1 is a perfect power only at 8, 9: Catalan's
            // conjecture, proved by Mihăilescu (J. reine angew. Math. 572,
            // 2004) — the only consecutive perfect powers are 8 and 9 —
            // and these operands are far beyond that pair.
            assert!(!bumped.is_perfect_power(), "power + 1 at {words} words");
            assert_eq!(bumped.nth_root_floor(k), m, "root of power + 1");
        }
        // A square of a square: detected through either exponent route.
        let base = seeded_biguint(6, &mut seed);
        assert!(base.square().square().is_perfect_power());
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
        // Best of nine runs: the minimum sheds scheduler interference; the
        // count is arbitrary.
        const RUNS: usize = 9;
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x0bad_5eed_0000_0001;
        let mut seed = SEED;
        eprintln!("{:>8} {:>12} {:>12}", "bits", "newton_us", "bisect_us");
        // From 1 kbit to 64 kbit.
        for &words in &[16usize, 64, 128, 1024] {
            let n = seeded_biguint(words, &mut seed);
            let time = |f: &dyn Fn()| {
                let mut best = f64::INFINITY;
                for _ in 0..RUNS {
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
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x5eed_5eed_1234_5678;
        let mut seed = SEED;
        // One limb, a few, a few dozen, and 200 limbs — 12,800 bits, above
        // both divide-and-conquer thresholds in every radix; two random
        // values each.
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
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x0123_4567_89ab_cdef;
        let mut seed = SEED;
        // 200 random two-limb values, the widest the std formatter checks;
        // the count is arbitrary.
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
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xfeed_beef_dead_cafe;
        let mut seed = SEED;
        // 256 words is at least 3,169 digits in every radix here, above the
        // divide-and-conquer thresholds; the classical engines are the oracle.
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
        // Two wider values for the dispatched decimal round trip.
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
        // Best of nine runs: the minimum sheds scheduler interference; the
        // count is arbitrary.
        const RUNS: usize = 9;
        // Recursion floors to sweep: the shipped 512 and its next three
        // doublings.
        const FLOORS: [usize; 4] = [512, 1024, 2048, 4096];
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x7157_ab1e_5eed_0001;
        let mut seed = SEED;
        eprintln!(
            "{:>8} {:>8} {:>12} {:>12} {:>12} {:>12}",
            "words", "digits", "to_cl_ms", "to_dc_ms", "from_cl_ms", "from_dc_ms"
        );
        // By doublings from 2 kbit (about 617 decimal digits) to 128 kbit
        // (about 39,000), so both dispatch thresholds fall inside the sweep.
        for &words in &[32usize, 64, 128, 256, 512, 1024, 2048] {
            let value = seeded_biguint(words, &mut seed);
            let digits = value.to_digits_classical(10);
            let time = |f: &dyn Fn()| {
                let mut best = f64::INFINITY;
                for _ in 0..RUNS {
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
            let bases: Vec<f64> = FLOORS
                .iter()
                .map(|&b| {
                    time(&|| {
                        black_box(BigUint::from_digits_ladder(&digits, 10, &ladder, chunk, b));
                    })
                })
                .collect();
            let (rladder, rchunk) = BigUint::radix_power_ladder_bits(10, value.bits());
            let rbases: Vec<f64> = FLOORS
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
    fn add_into_sub_into_match_two_operand_forms() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x5851_f42d_4c95_7f2d;
        let mut seed = SEED;
        let mut out = BigUint::zero();
        // Zero, one and many limbs on each side, in both orders; twelve
        // random pairs each.
        for &(wa, wb) in &[(0usize, 0usize), (1, 1), (1, 48), (48, 1), (8, 8), (48, 48)] {
            for _ in 0..12 {
                let a = seeded_biguint(wa, &mut seed);
                let b = seeded_biguint(wb, &mut seed);
                out.add_into(&a, &b);
                assert_eq!(out, a.add(&b));
                assert!(out.limbs.last() != Some(&0), "canonical form");
                let (hi, lo) = if a >= b { (&a, &b) } else { (&b, &a) };
                out.sub_into(hi, lo);
                assert_eq!(out, hi.sub(lo));
                assert!(out.limbs.last() != Some(&0), "canonical form");
            }
        }
        // A full carry ripple: (2^(64k) - 1) + 1 = 2^(64k).
        let ones = BigUint {
            limbs: vec![u64::MAX; 5],
        };
        let one = BigUint::from_u64(1);
        out.add_into(&ones, &one);
        let mut expect = BigUint::zero();
        expect.set_bit(320);
        assert_eq!(out, expect);
        // And the borrow ripple back down.
        out.sub_into(&expect, &one);
        assert_eq!(out, ones);
    }

    #[test]
    fn add_into_reuses_the_buffer() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x0123_4567_89ab_cdef;
        let mut seed = SEED;
        let a = seeded_biguint(32, &mut seed);
        let b = seeded_biguint(32, &mut seed);
        // The first call may grow the buffer once (the result width plus the
        // carry slot); from then on the no-allocation contract holds.
        let mut out = BigUint::zero();
        out.add_into(&a, &b);
        let ptr = out.limbs.as_ptr();
        // Repeated calls; the count is arbitrary.
        for _ in 0..8 {
            out.add_into(&a, &b);
            assert_eq!(out.limbs.as_ptr(), ptr, "add_into must not reallocate");
            out.sub_into(&a, &b.sub(&b)); // a - 0 = a, exercising short rhs
            assert_eq!(out.limbs.as_ptr(), ptr, "sub_into must not reallocate");
        }
    }

    #[test]
    #[should_panic(expected = "BigUint underflow")]
    fn sub_into_panics_on_underflow() {
        let mut out = BigUint::zero();
        out.sub_into(&BigUint::from_u64(3), &BigUint::from_u64(5));
    }

    /// The `wipe` feature scrubs abandoned limbs; proving that the shrink
    /// paths do so requires one raw read-back, because reading a buffer's
    /// abandoned tail cannot be expressed in safe Rust. Confined to this
    /// test; the pointers are captured while the limbs are live and the
    /// buffer's identity is asserted unchanged before each read.
    #[test]
    #[cfg(feature = "wipe")]
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
        out.add_into(&narrow, &narrow);
        assert_eq!(out.limbs.as_ptr(), p, "add_into reuses the buffer");
        assert!(
            read8(p)[1..].iter().all(|&w| w == 0),
            "add_into stranded live limbs"
        );

        let mut out2 = wide.clone();
        let p = out2.limbs.as_ptr();
        out2.sub_into(&narrow, &narrow);
        assert_eq!(out2.limbs.as_ptr(), p, "sub_into reuses the buffer");
        assert!(
            read8(p).iter().all(|&w| w == 0),
            "sub_into stranded live limbs"
        );

        let a = BigInt::from_parts(Sign::Positive, wide.clone());
        let mut z = a.clone();
        let p = z.magnitude.limbs.as_ptr();
        z -= &a;
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
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xdead_beef_0bad_cafe;
        let mut seed = SEED;
        let wide = seeded_biguint(8, &mut seed);
        let narrow = seeded_biguint(2, &mut seed);
        let mut x = wide.clone();
        x.clone_from(&narrow);
        assert_eq!(x, narrow);
        let mut out = wide.clone();
        out.add_into(&narrow, &narrow);
        assert_eq!(out, narrow.add(&narrow));
        let mut out2 = wide.clone();
        out2.sub_into(&narrow, &narrow);
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
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xfeed_face_cafe_beef;
        let mut seed = SEED;
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
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x2545_f491_4f6c_dd1d;
        let mut seed = SEED;
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
                // Zero, one and several limbs on each side, in both orders;
                // six random pairs per sign and width combination.
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
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut seed = SEED;
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
        // 4,000 random 56-bit signed pairs, so sums stay within a limb.
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
        // The public signed ring (mul, div_rem, abs) against i128, with
        // the division convention pinned: truncated toward zero, remainder
        // taking the dividend's sign — i128's own convention, so the oracle
        // is the primitive operators.
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x517e_d00d_0000_0001;
        let mut seed = SEED;
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
        // 4,000 random 30-bit signed pairs, so products fit a limb.
        for _ in 0..4000 {
            let a = lcg_next(&mut seed) as i64 >> 34;
            let b = lcg_next(&mut seed) as i64 >> 34;
            let (ba, bb) = (to_bigint(a), to_bigint(b));
            assert_eq!(
                to_i128(&ba.mul(&bb)),
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
        // Truncated, not floored.
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
    fn square_matches_mul() {
        // The squaring ladder against the multiplication it specializes, at
        // every width where its dispatch changes hands — one limb either
        // side of `SQR_SCHOOLBOOK_MIN_LIMBS` (multiplication vs schoolbook
        // squaring), of the Karatsuba threshold (schoolbook vs Karatsuba
        // squaring), and of `SQR_KARATSUBA_MAX_LIMBS` (Karatsuba squaring
        // vs handing back to the multiply ladder) — plus odd widths, which
        // make the split halves unequal, and operands with interior zeros
        // and all-ones limbs.
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut seed = SEED;
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
        // Four random values per width.
        for words in widths {
            for _ in 0..4 {
                let value = seeded_biguint(words, &mut seed);
                assert_eq!(
                    value.square(),
                    value.mul(&value),
                    "square != mul at {words} limbs"
                );
                // And against the schoolbook kernel directly, so a defect
                // shared by both dispatched paths cannot hide.
                assert_eq!(
                    value.square(),
                    BigUint::mul_schoolbook_ref(&value, &value),
                    "square != schoolbook at {words} limbs"
                );
            }
        }
        // Interior zero limbs, which make whole rows of the cross-term
        // pass vanish.
        let mut limbs = seeded_biguint(k + 4, &mut seed).limbs().to_vec();
        for limb in &mut limbs[2..(k + 4) / 2] {
            *limb = 0;
        }
        limbs[0] |= 1;
        let holed = BigUint::from_limbs(limbs);
        assert_eq!(holed.square(), holed.mul(&holed));
        // All-ones operands: the worst case for every carry chain, and for
        // the doubling pass — at one and two limbs (the general multiply),
        // the schoolbook floor, and Karatsuba squaring either side of its
        // threshold and at twice it.
        for words in [1usize, 2, 8, k, k + 1, 2 * k] {
            let ones = BigUint::from_limbs(vec![u64::MAX; words]);
            assert_eq!(
                ones.square(),
                BigUint::mul_schoolbook_ref(&ones, &ones),
                "all-ones square at {words} limbs"
            );
        }
        assert!(BigUint::zero().square().is_zero());
        assert_eq!(BigUint::one().square(), BigUint::one());
    }

    /// The into-storage product against the allocating one, on both sides
    /// of the width where it stops running schoolbook into the buffer,
    /// with the buffer arriving wider and narrower than the product and
    /// with lopsided shapes.
    #[test]
    fn mul_into_matches_mul_at_every_shape() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x6d75_6c5f_696e_746f;
        let mut seed = SEED;
        let shapes = [
            (1usize, 1usize),
            (1, 7),
            (2, 2),
            (3, 40),
            (KARATSUBA_THRESHOLD_LIMBS - 1, KARATSUBA_THRESHOLD_LIMBS - 1),
            (KARATSUBA_THRESHOLD_LIMBS, KARATSUBA_THRESHOLD_LIMBS),
            (
                KARATSUBA_THRESHOLD_LIMBS + 1,
                2 * KARATSUBA_THRESHOLD_LIMBS + 3,
            ),
            (300, 300),
        ];
        let mut out = BigUint::from_limbs(vec![u64::MAX; 1000]); // wider than any product
        for (a, b) in shapes {
            let lhs = seeded_biguint(a, &mut seed);
            let rhs = seeded_biguint(b, &mut seed);
            let expected = lhs.mul(&rhs);
            out.mul_into(&lhs, &rhs);
            assert_eq!(out, expected, "{a}x{b} limbs into a wide buffer");
            out.mul_into(&rhs, &lhs);
            assert_eq!(out, expected, "{b}x{a} limbs, operands swapped");
            let mut narrow = BigUint::zero();
            narrow.mul_into(&lhs, &rhs);
            assert_eq!(narrow, expected, "{a}x{b} limbs into an empty buffer");
        }
        out.mul_into(&BigUint::zero(), &BigUint::from_u64(5));
        assert!(out.is_zero());
        out.mul_into(&BigUint::from_u64(5), &BigUint::zero());
        assert!(out.is_zero());
    }

    /// `keep_low_bits` against the oracle `x − (x ≫ k) ≪ k`, at bit counts
    /// on both sides of every limb boundary and of the value's own width.
    #[test]
    fn keep_low_bits_is_the_residue_modulo_a_power_of_two() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x6b65_6570_5f6c_6f77;
        let mut seed = SEED;
        let value = seeded_biguint(4, &mut seed);
        let width = value.bits();
        for bits in [
            0usize,
            1,
            63,
            64,
            65,
            127,
            128,
            129,
            191,
            width - 1,
            width,
            width + 1,
            300,
        ] {
            let mut high = value.clone();
            high.shr_bits(bits);
            high.shl_bits(bits);
            let expected = value.sub(&high);
            let mut kept = value.clone();
            kept.keep_low_bits(bits);
            assert_eq!(kept, expected, "low {bits} bits of a {width}-bit value");
        }
        let mut zero = BigUint::zero();
        zero.keep_low_bits(10);
        assert!(zero.is_zero());
    }

    #[test]
    fn karatsuba_dispatch_matches_schoolbook() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x243f_6a88_85a3_08d3;
        let mut seed = SEED;
        // Six random pairs per width, either side of and at the Karatsuba
        // crossover, so the dispatcher answers with schoolbook below it and
        // with a split at and above it, and both must agree with schoolbook.
        for words in [
            KARATSUBA_THRESHOLD_LIMBS / 2,
            KARATSUBA_THRESHOLD_LIMBS - 1,
            KARATSUBA_THRESHOLD_LIMBS,
            KARATSUBA_THRESHOLD_LIMBS + 1,
            2 * KARATSUBA_THRESHOLD_LIMBS,
        ] {
            for _ in 0..6 {
                let lhs = seeded_biguint(words, &mut seed);
                let rhs = seeded_biguint(words, &mut seed);
                let dispatched = lhs.mul(&rhs);
                let schoolbook = BigUint::mul_schoolbook_ref(&lhs, &rhs);
                assert_eq!(dispatched, schoolbook);
            }
        }
    }

    #[test]
    fn toom3_matches_schoolbook_across_shapes() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x1357_9bdf_2468_ace0;
        let mut seed = SEED;
        // Exercise the Toom-3 kernel directly — including well below the
        // dispatch threshold, at sizes not divisible by three, and with heavy
        // imbalance (one operand collapsing to a single Toom part) — against
        // the schoolbook oracle it must reproduce exactly.
        let sizes = [
            3usize, 4, 5, 7, 8, 9, 16, 31, 33, 48, 64, 65, 96, 127, 130, 200,
        ];
        // Three random pairs per shape.
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
        // Full dispatch (Toom-3 for large balanced operands) and squaring:
        // below, at and past the Karatsuba threshold, and past Toom-3's;
        // four random pairs each.
        for &words in &[64usize, 96, 150, 256] {
            for _ in 0..4 {
                let a = seeded_biguint(words, &mut seed);
                let b = seeded_biguint(words, &mut seed);
                assert_eq!(a.mul(&b), BigUint::mul_schoolbook_ref(&a, &b));
                assert_eq!(a.square(), BigUint::mul_schoolbook_ref(&a, &a));
            }
        }
    }

    #[test]
    fn toom4_matches_schoolbook_across_shapes() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x0f0f_1e1e_2d2d_3c3c;
        let mut seed = SEED;
        // The Toom-4 kernel directly: sizes not divisible by four and heavy
        // imbalance (a short operand collapsing to fewer Toom parts), against
        // the schoolbook oracle.
        let sizes = [4usize, 5, 6, 7, 9, 13, 16, 33, 64, 128, 256, 260, 384, 500];
        // Two random pairs per shape; the schoolbook oracle at 500 limbs is
        // the cost that caps it.
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
        // Public dispatch at and past the Toom-4 threshold, below the NTT's:
        // equal lengths, the widest admitted imbalance (1.5×), and squaring.
        let t = TOOM4_THRESHOLD_LIMBS;
        for &(la, lb) in &[(t, t), (t + t / 2, t), (t + 77, t + 5)] {
            let a = seeded_biguint(la, &mut seed);
            let b = seeded_biguint(lb, &mut seed);
            assert!(BigUint::should_use_toom4(&a, &b), "{la}x{lb} is Toom-4's");
            assert!(
                !BigUint::should_use_ntt(&a, &b),
                "{la}x{lb} is not the NTT's"
            );
            assert_eq!(a.mul(&b), BigUint::mul_schoolbook_ref(&a, &b), "{la}x{lb}");
            assert_eq!(
                a.square(),
                BigUint::mul_schoolbook_ref(&a, &a),
                "{la} squared"
            );
        }
        // Full dispatch and squaring at Toom-3 sizes, from twice its
        // threshold to below Toom-4's; three random pairs each.
        for &words in &[256usize, 300, 512, 768] {
            for _ in 0..3 {
                let a = seeded_biguint(words, &mut seed);
                let b = seeded_biguint(words, &mut seed);
                assert_eq!(a.mul(&b), BigUint::mul_schoolbook_ref(&a, &b));
                assert_eq!(a.square(), BigUint::mul_schoolbook_ref(&a, &a));
            }
        }
    }

    /// Operands that stress carries, borrows, normalization and zero
    /// digits at a given width: random limbs, all ones, a lone top bit,
    /// alternating zero limbs, and one bit at each end.
    fn boundary_patterns(words: usize, seed: &mut u64) -> Vec<(&'static str, BigUint)> {
        let random = seeded_biguint(words, seed);
        let mut top_bit = vec![0u64; words];
        top_bit[words - 1] = 1 << 63;
        let mut alternating = seeded_biguint(words, seed).limbs().to_vec();
        for limb in alternating.iter_mut().skip(1).step_by(2) {
            *limb = 0;
        }
        alternating[words - 1] |= 1;
        let mut ends = vec![0u64; words];
        ends[0] = 1;
        ends[words - 1] |= 1 << 63;
        vec![
            ("random", random),
            ("all ones", BigUint::from_limbs(vec![u64::MAX; words])),
            ("top bit", BigUint::from_limbs(top_bit)),
            ("alternating zero limbs", BigUint::from_limbs(alternating)),
            ("both ends", BigUint::from_limbs(ends)),
        ]
    }

    /// Every multiplication and squaring threshold, one limb either side,
    /// against schoolbook: each pattern against itself, against the random
    /// operand, and against its predecessor (nearly equal operands); at the
    /// 2:1 unbalanced edge and the 1.5× Toom-4 edge the lopsided shapes too.
    #[test]
    fn products_agree_with_schoolbook_at_every_dispatch_boundary() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x5bd1_e995_2545_f491;
        let mut seed = SEED;
        let check = |a: &BigUint, b: &BigUint, what: &str| {
            let expected = BigUint::mul_schoolbook_ref(a, b);
            assert_eq!(a.mul(b), expected, "{what}");
            assert_eq!(b.mul(a), expected, "{what}, swapped");
        };
        for threshold in [
            KARATSUBA_THRESHOLD_LIMBS,
            TOOM3_THRESHOLD_LIMBS,
            UNBALANCED_THRESHOLD_LIMBS,
        ] {
            for words in [threshold - 1, threshold, threshold + 1] {
                let patterns = boundary_patterns(words, &mut seed);
                let random = patterns[0].1.clone();
                for (name, value) in &patterns {
                    check(value, value, &format!("{name} squared at {words}"));
                    check(value, &random, &format!("{name} × random at {words}"));
                    if !value.is_zero() {
                        let before = value.sub(&BigUint::one());
                        check(
                            value,
                            &before,
                            &format!("{name} × its predecessor at {words}"),
                        );
                    }
                    for long in [2 * words - 1, 2 * words, 2 * words + 1] {
                        let wide = seeded_biguint(long, &mut seed);
                        check(&wide, value, &format!("{long} random × {name} at {words}"));
                    }
                    assert_eq!(
                        value.square(),
                        BigUint::mul_schoolbook_ref(value, value),
                        "square of {name} at {words}"
                    );
                }
            }
        }
        for words in [
            SQR_SCHOOLBOOK_MIN_LIMBS - 1,
            SQR_SCHOOLBOOK_MIN_LIMBS,
            SQR_SCHOOLBOOK_MIN_LIMBS + 1,
            SQR_KARATSUBA_MAX_LIMBS - 1,
            SQR_KARATSUBA_MAX_LIMBS,
            SQR_KARATSUBA_MAX_LIMBS + 1,
        ] {
            for (name, value) in boundary_patterns(words, &mut seed) {
                assert_eq!(
                    value.square(),
                    BigUint::mul_schoolbook_ref(&value, &value),
                    "square of {name} at {words}"
                );
            }
        }
        // Toom-4: schoolbook at these widths is costly, so the all-ones and
        // random patterns, balanced and at the 1.5× admission edge.
        let t = TOOM4_THRESHOLD_LIMBS;
        for words in [t - 1, t, t + 1] {
            let patterns = boundary_patterns(words, &mut seed);
            for (name, value) in patterns.iter().take(2) {
                check(value, value, &format!("{name} squared at {words}"));
                let edge = words + words / 2;
                for long in [edge, edge + 1] {
                    let wide = BigUint::from_limbs(vec![u64::MAX; long]);
                    check(
                        &wide,
                        value,
                        &format!("{long} all ones × {name} at {words}"),
                    );
                }
            }
        }
    }

    /// Division one limb either side of the Newton threshold, and at one
    /// and two limbs: the dividend is built as `q·d + r` by schoolbook with
    /// `r` the largest remainder, so `div_rem` must return exactly `(q, r)`.
    /// Divisors cover all ones, a lone top bit (already normalized) and a top
    /// limb of one (the widest normalization shift).
    #[test]
    fn division_recovers_quotient_and_remainder_at_every_dispatch_boundary() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x2545_f491_4f6c_dd1d;
        let mut seed = SEED;
        let t = super::newton::NEWTON_DIVISION_THRESHOLD_LIMBS;
        for words in [1usize, 2, t - 1, t, t + 1] {
            let mut small_top = seeded_biguint(words, &mut seed).limbs().to_vec();
            small_top[words - 1] = 1;
            let mut top_bit = vec![0u64; words];
            top_bit[words - 1] = 1 << 63;
            let divisors = [
                ("random", seeded_biguint(words, &mut seed)),
                ("all ones", BigUint::from_limbs(vec![u64::MAX; words])),
                ("top bit", BigUint::from_limbs(top_bit)),
                ("top limb one", BigUint::from_limbs(small_top)),
            ];
            for (name, divisor) in &divisors {
                let remainder = divisor.sub(&BigUint::one());
                for quotient_words in [1usize, words, words + 1] {
                    let quotient = seeded_biguint(quotient_words, &mut seed);
                    let dividend = BigUint::mul_schoolbook_ref(&quotient, divisor).add(&remainder);
                    let (q, r) = dividend.div_rem(divisor);
                    assert_eq!(
                        q, quotient,
                        "quotient: {name} divisor of {words}, {quotient_words}-limb quotient"
                    );
                    assert_eq!(
                        r, remainder,
                        "remainder: {name} divisor of {words}, {quotient_words}-limb quotient"
                    );
                    assert_eq!(dividend.rem(divisor), remainder);
                }
            }
        }
    }

    /// Barrett reduction one limb either side of its half-product limit,
    /// against the division remainder, for products of residues including
    /// the largest, `(n − 1)²`.
    #[test]
    fn barrett_reduces_like_division_at_its_half_product_limit() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut seed = SEED;
        let t = super::barrett::BARRETT_HALF_PRODUCT_MAX_LIMBS;
        for words in [t - 1, t, t + 1] {
            for (name, modulus) in boundary_patterns(words, &mut seed) {
                let context = BarrettContext::new(&modulus).expect("above one");
                let top = modulus.sub(&BigUint::one());
                let a = seeded_biguint(words, &mut seed).rem(&modulus);
                for x in [
                    BigUint::mul_schoolbook_ref(&top, &top),
                    BigUint::mul_schoolbook_ref(&a, &top),
                    modulus.clone(),
                    top.clone(),
                ] {
                    assert_eq!(
                        context.reduce(&x),
                        x.rem(&modulus),
                        "{name} modulus of {words}"
                    );
                }
            }
        }
    }

    /// Radix conversion either side of the divide-and-conquer thresholds,
    /// against Horner's rule one decimal digit at a time.
    #[test]
    fn radix_conversion_agrees_with_horner_at_its_thresholds() {
        let ten = BigUint::from_u64(10);
        let horner = |digits: &str| {
            digits.bytes().fold(BigUint::zero(), |value, digit| {
                BigUint::mul_schoolbook_ref(&value, &ten)
                    .add(&BigUint::from_u64(u64::from(digit - b'0')))
            })
        };
        let d = RADIX_FROM_DC_THRESHOLD_DIGITS;
        for length in [d - 1, d, d + 1] {
            for digits in [
                "9".repeat(length),
                format!("1{}", "0".repeat(length - 1)),
                "1234567890".repeat(length / 10 + 1)[..length].to_string(),
            ] {
                let value = BigUint::from_str_radix(&digits, 10).expect("decimal");
                assert_eq!(value, horner(&digits), "parse of {length} digits");
                assert_eq!(
                    value.to_str_radix(10),
                    digits.trim_start_matches('0'),
                    "print of {length} digits"
                );
            }
        }
        let bits = RADIX_TO_DC_THRESHOLD_BITS;
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x0bad_5eed_0bad_5eed;
        let mut seed = SEED;
        for width in [bits - 1, bits, bits + 1] {
            let mut value = seeded_biguint(width.div_ceil(64), &mut seed);
            value.shr_bits(value.bits().saturating_sub(width));
            let text = value.to_str_radix(10);
            assert_eq!(horner(&text), value, "print of {width} bits");
            let mut ones = BigUint::one();
            ones.shl_bits(width);
            let ones = ones.sub(&BigUint::one());
            assert_eq!(
                horner(&ones.to_str_radix(10)),
                ones,
                "print of 2^{width} − 1"
            );
        }
    }

    #[test]
    fn ntt_matches_independent_products_and_carry_extremes() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x6a09_e667_f3bc_c909;
        let mut seed = SEED;

        // Force the NTT kernel far below its dispatch threshold so an error in
        // admission cannot hide it. Odd digit counts, partial top limbs, and
        // unequal operands exercise zero padding and CRT/carry reconstruction.
        for &(lhs_words, rhs_words) in &[
            (1usize, 1usize),
            (1, 7),
            (2, 3),
            (5, 9),
            (17, 31),
            (64, 65),
            (129, 193),
            (257, 384),
        ] {
            // Three random pairs per shape.
            for _ in 0..3 {
                let lhs = seeded_biguint(lhs_words, &mut seed);
                let rhs = seeded_biguint(rhs_words, &mut seed);
                assert_eq!(
                    lhs.mul_ntt_ref(&rhs),
                    BigUint::mul_schoolbook_ref(&lhs, &rhs),
                    "NTT != schoolbook for {lhs_words}x{rhs_words} limbs"
                );
                assert_eq!(
                    lhs.sqr_ntt_ref(),
                    BigUint::mul_schoolbook_ref(&lhs, &lhs),
                    "NTT square != schoolbook for {lhs_words} limbs"
                );
            }
        }

        // The one-coefficient transform and partial top base-2^16 digits.
        for value in [1u64, 2, 65_535, 65_536, 65_537, u32::MAX as u64] {
            let value = BigUint::from_u64(value);
            assert_eq!(
                value.sqr_ntt_ref(),
                BigUint::mul_schoolbook_ref(&value, &value)
            );
        }

        // Every base-2^16 convolution coefficient and every carry is maximal,
        // from one limb to past 512, with 129 and 513 one limb past a
        // doubling of the transform length.
        for words in [1usize, 2, 7, 32, 129, 513] {
            let all_ones = BigUint::from_limbs(vec![u64::MAX; words]);
            assert_eq!(
                all_ones.mul_ntt_ref(&all_ones),
                BigUint::mul_schoolbook_ref(&all_ones, &all_ones),
                "all-ones NTT square at {words} limbs"
            );
            assert_eq!(
                all_ones.sqr_ntt_ref(),
                BigUint::mul_schoolbook_ref(&all_ones, &all_ones),
                "specialized all-ones NTT square at {words} limbs"
            );
        }

        // Drive public dispatch at the actual NTT threshold without using a
        // second fast kernel as the oracle: (B^n - 1)^2 has a closed-form limb
        // representation and is the worst carry chain the transform can see.
        let words = NTT_SERIAL_THRESHOLD_LIMBS;
        let all_ones = BigUint::from_limbs(vec![u64::MAX; words]);
        let mut expected = vec![0u64; 2 * words];
        expected[0] = 1;
        expected[words] = u64::MAX - 1;
        expected[words + 1..].fill(u64::MAX);
        let expected = BigUint::from_limbs(expected);
        assert_eq!(all_ones.mul(&all_ones), expected);
        assert_eq!(all_ones.square(), expected);

        // Admission is by width and balance, not by where the width falls in
        // the radix-2 padding staircase. The NFS square root lifts at
        // 1.3 Mbit — 20,410 limbs, whose convolution pads to 12.8
        // coefficients per limb, the worst ratio the staircase produces —
        // and the transform is ahead there on every machine measured, so
        // the dispatcher must take it.
        const NFS_LIFT_LIMBS: usize = 20_410;
        let at_8k = BigUint::from_limbs(vec![u64::MAX; 8_192]);
        let at_worst_padding = BigUint::from_limbs(vec![u64::MAX; NFS_LIFT_LIMBS]);
        let at_16k = BigUint::from_limbs(vec![u64::MAX; 16_384]);
        let at_98k = BigUint::from_limbs(vec![u64::MAX; 98_304]);
        let at_114k = BigUint::from_limbs(vec![u64::MAX; 114_688]);
        assert!(!BigUint::should_use_ntt_with_contexts(&at_8k, &at_8k, 1));
        assert!(BigUint::should_use_ntt_with_contexts(&at_8k, &at_8k, 4));
        assert!(BigUint::should_use_ntt_with_contexts(
            &at_worst_padding,
            &at_worst_padding,
            4
        ));
        assert!(BigUint::should_use_ntt_with_contexts(&at_16k, &at_16k, 4));
        assert!(!BigUint::should_use_ntt_with_contexts(&at_98k, &at_98k, 1));
        assert!(BigUint::should_use_ntt_with_contexts(&at_98k, &at_98k, 4));
        // One context takes the transform only past its own threshold.
        assert!(!BigUint::should_use_ntt_with_contexts(
            &at_114k, &at_114k, 1
        ));
        assert!(BigUint::should_use_ntt_with_contexts(&at_114k, &at_114k, 4));
        let at_serial_threshold = BigUint::from_limbs(vec![u64::MAX; NTT_SERIAL_THRESHOLD_LIMBS]);
        assert!(BigUint::should_use_ntt_with_contexts(
            &at_serial_threshold,
            &at_serial_threshold,
            1
        ));
    }

    #[test]
    fn unbalanced_matches_schoolbook_across_shapes() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut seed = SEED;
        // The block-decomposition kernel directly, below its dispatch
        // threshold. 64×32 is the exact boundary the balanced admission
        // excludes; 100×32 leaves a short final digit; 129×32 a one-limb one;
        // the larger shapes recurse through several balanced kernels.
        for &(la, lb) in &[
            (64usize, 32usize),
            (100, 32),
            (129, 32),
            (320, 40),
            (96, 48),
            (256, 128),
            (300, 130),
        ] {
            // Three random pairs per shape.
            for _ in 0..3 {
                let a = seeded_biguint(la, &mut seed);
                let b = seeded_biguint(lb, &mut seed);
                assert_eq!(
                    a.mul_unbalanced_ref(&b),
                    BigUint::mul_schoolbook_ref(&a, &b),
                    "unbalanced != schoolbook for {la}x{lb} words"
                );
                // Commutativity of the dispatch: the same pair in either order.
                assert_eq!(b.mul(&a), BigUint::mul_schoolbook_ref(&a, &b));
            }
        }
        // The dispatch boundary, table-driven around the threshold: the
        // exact 2:1 ratio must go to the block decomposition (Karatsuba's
        // admission is strict, and its kernel would find an empty high half
        // there), one limb under 2:1 must go to Karatsuba, and one limb
        // under the threshold must fall back to schoolbook — checked by the
        // predicates and by value.
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
            assert_eq!(a.mul(&b), BigUint::mul_schoolbook_ref(&a, &b));
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
        // plain modular product, with one buffer shared across widths and
        // both methods in both orders. The width order is non-monotonic so a
        // narrower modulus follows a wider one (16 → 2, 8 → 1), handing the
        // kernels an over-long buffer with stale contents.
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x0dd5_eed0_0000_0001;
        let mut seed = SEED;
        let mut ws = MontgomeryScratch::new();
        for &limbs in &[16usize, 2, 3, 8, 1, 5] {
            let n = seeded_odd_modulus(limbs, &mut seed);
            let ctx = MontgomeryContext::new(&n).expect("odd modulus");
            let x_plain = seeded_biguint(limbs, &mut seed);
            let y_plain = seeded_biguint(limbs, &mut seed);
            let mut x = ctx.to_residue(&x_plain);
            let y = ctx.to_residue(&y_plain);
            // 200 rounds, the residue walking so no two rounds share inputs;
            // the count is arbitrary.
            for round in 0..200 {
                // Alternate the call order so each window size follows the
                // other's leftovers.
                let next = if round % 2 == 0 {
                    let m = ctx.mul_residue_with(&x, &y, &mut ws).expect("same context");
                    assert_eq!(
                        m,
                        ctx.mul_residue(&x, &y).expect("same"),
                        "mul at {limbs} limbs"
                    );
                    let s = ctx.square_residue_with(&m, &mut ws).expect("same context");
                    assert_eq!(
                        s,
                        ctx.square_residue(&m).expect("same"),
                        "sqr at {limbs} limbs"
                    );
                    s
                } else {
                    let s = ctx.square_residue_with(&x, &mut ws).expect("same context");
                    assert_eq!(
                        s,
                        ctx.square_residue(&x).expect("same"),
                        "sqr at {limbs} limbs"
                    );
                    let m = ctx.mul_residue_with(&s, &y, &mut ws).expect("same context");
                    assert_eq!(
                        m,
                        ctx.mul_residue(&s, &y).expect("same"),
                        "mul at {limbs} limbs"
                    );
                    m
                };
                x = next;
            }
            // Decoded agreement with the reduction-based product: the domain
            // arithmetic and the ordinary arithmetic name the same value.
            let product = ctx.mul_residue_with(&x, &y, &mut ws).expect("same context");
            let x_out = ctx.from_residue(&x).expect("same context");
            let y_out = ctx.from_residue(&y).expect("same context");
            assert_eq!(
                ctx.from_residue(&product).expect("same context"),
                BigUint::mod_mul(&x_out, &y_out, &n),
                "domain product decodes to the modular product at {limbs} limbs"
            );
        }
    }

    #[test]
    #[ignore = "timing probe for the with_workspace docs; run with --ignored"]
    fn mont_workspace_timing() {
        use std::hint::black_box;
        use std::time::Instant;

        /// One pass: alternate the two forms in short chunks, so slow drift
        /// (thermal, frequency scaling) hits both sides equally. Returns the
        /// saving of `b` over `a` in percent.
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

        /// Print every pass so the spread is visible, and the median. A
        /// saving smaller than the spread is noise.
        fn report(label: &str, limbs: usize, mut passes: [f64; PASSES]) {
            passes.sort_by(f64::total_cmp);
            eprintln!(
                "{limbs:>6} {label} median {:+6.1}%  passes {:+5.1} {:+5.1} {:+5.1} {:+5.1} {:+5.1}",
                passes[2], passes[0], passes[1], passes[2], passes[3], passes[4]
            );
        }

        // Work per pass per side, in limb² units: calls fall as 1/limbs² so
        // a pass does about the same arithmetic at every width.
        const WORK_UNITS: usize = 2_000_000;
        // Floor on calls per pass, so the widest moduli still get thousands
        // of samples.
        const MIN_CALLS: usize = 4_000;
        // Calls per timed chunk, the alternation grain: short enough that
        // slow drift hits both sides alike, long enough that the timer's own
        // cost is small beside the chunk.
        const CHUNK_CALLS: u32 = 256;
        // At least this many chunks, so the sides alternate at all.
        const MIN_CHUNKS: u32 = 8;
        // Passes per comparison: a median with two on each side.
        const PASSES: usize = 5;
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x0dd5_eed0_0000_0002;
        let mut seed = SEED;
        // One limb to 64 (4,096 bits).
        for &limbs in &[1usize, 4, 8, 32, 64] {
            let n = seeded_odd_modulus(limbs, &mut seed);
            let ctx = MontgomeryContext::new(&n).expect("odd modulus");
            let a = ctx.to_residue(&seeded_biguint(limbs, &mut seed));
            let b = ctx.to_residue(&seeded_biguint(limbs, &mut seed));
            let chunk = CHUNK_CALLS;
            let chunks =
                ((WORK_UNITS / (limbs * limbs)).max(MIN_CALLS) as u32 / chunk).max(MIN_CHUNKS);

            let mut ws = MontgomeryScratch::new();
            let mut sqr_passes = [0f64; PASSES];
            for pass in &mut sqr_passes {
                *pass = paired_saving(
                    chunks,
                    chunk,
                    &mut || {
                        let _ = black_box(ctx.square_residue(black_box(&a)));
                    },
                    &mut || {
                        let _ = black_box(ctx.square_residue_with(black_box(&a), &mut ws));
                    },
                );
            }
            report("sqr", limbs, sqr_passes);

            let mut ws2 = MontgomeryScratch::new();
            let mut mul_passes = [0f64; PASSES];
            for pass in &mut mul_passes {
                *pass = paired_saving(
                    chunks,
                    chunk,
                    &mut || {
                        let _ = black_box(ctx.mul_residue(black_box(&a), black_box(&b)));
                    },
                    &mut || {
                        let _ =
                            black_box(ctx.mul_residue_with(black_box(&a), black_box(&b), &mut ws2));
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
        // Work per timed run in units of 64 limb products: about 12.8
        // million limb products at every shape, so wide and narrow shapes
        // are timed over the same arithmetic.
        const WORK_UNITS: usize = 200_000;
        // Floor on repetitions per run, so the widest shapes still average
        // over more than one product.
        const MIN_REPS: usize = 3;
        // Best of five runs: the minimum sheds scheduler interference; the
        // count is arbitrary.
        const RUNS: usize = 5;
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x5eed_5eed_5eed_5eed;
        let mut seed = SEED;
        eprintln!(
            "{:>6} {:>6} {:>12} {:>12}  best",
            "long", "short", "school", "unbal"
        );
        // Short widths from a third of the Karatsuba threshold to twice the
        // shipped unbalanced threshold, at the lopsided ratios 2, 4 and 16.
        for &short in &[32usize, 48, 64, 96, 128, 192, 256, 384, 512] {
            for &ratio in &[2usize, 4, 16] {
                let long = short * ratio;
                let a = seeded_biguint(long, &mut seed);
                let b = seeded_biguint(short, &mut seed);
                let reps = (WORK_UNITS / (long * short / 64)).max(MIN_REPS);
                let time = |f: &dyn Fn() -> BigUint| {
                    let mut best = f64::INFINITY;
                    for _ in 0..RUNS {
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
        // The half-product against the truncated full product: limits
        // below, at, and above each operand's
        // width, and the degenerate limit of zero.
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x1010_7ef7_0000_0001;
        let mut seed = SEED;
        // One limb, unequal and equal small widths, and a few dozen limbs.
        for &(la, lb) in &[(1usize, 1usize), (2, 3), (4, 4), (8, 5), (17, 16), (32, 32)] {
            let a = seeded_biguint(la, &mut seed);
            let b = seeded_biguint(lb, &mut seed);
            let full = a.mul(&b);
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

        fn report(label: &str, w: usize, p: [f64; PASSES]) {
            let mut sorted = p;
            sorted.sort_by(f64::total_cmp);
            eprintln!(
                "{label:<22} {w:>5} {:>+8.1}%   {:+6.1} {:+6.1} {:+6.1} {:+6.1} {:+6.1}",
                sorted[2], p[0], p[1], p[2], p[3], p[4]
            );
        }

        // Limb products per timed chunk (chunk · w²), so every width is
        // timed over the same amount of arithmetic.
        const WORK_LIMB_PRODUCTS: usize = 2_000_000;
        // Floor on calls per chunk, so the widest operands still average
        // over dozens of squarings.
        const MIN_CHUNK: u32 = 50;
        // Interleaved chunk pairs inside one sample.
        const CHUNK_PASSES: usize = 3;
        // Samples per comparison: a median with two on each side.
        const PASSES: usize = 5;
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x59ea_5011_0000_0001;
        let mut seed = SEED;
        eprintln!("saving of the second kernel over the first; median then every pass");
        eprintln!(
            "{:<22} {:>5} {:>9}   passes",
            "comparison", "limbs", "median"
        );
        // One limb to 512, dense around the two handoffs (8 and 96), and
        // every width the `SQR_KARATSUBA_MAX_LIMBS` doc cites.
        for &w in &[
            1usize, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 112, 127, 128, 160, 192, 256, 320, 384,
            416, 448, 480, 512, 640, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
        ] {
            let v = seeded_biguint(w, &mut seed);
            let chunk = (WORK_LIMB_PRODUCTS / (w * w)).max(MIN_CHUNK as usize) as u32;

            // The lower handoff: general schoolbook against schoolbook
            // squaring, the pair `SQR_SCHOOLBOOK_MIN_LIMBS` sits between;
            // widths up to 64 are the schoolbook regime.
            if w <= 64 {
                let mut p = [0f64; PASSES];
                for (k, slot) in p.iter_mut().enumerate() {
                    *slot = paired_saving(
                        CHUNK_PASSES,
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
                let mut p = [0f64; PASSES];
                for (k, slot) in p.iter_mut().enumerate() {
                    *slot = paired_saving(
                        CHUNK_PASSES,
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
            // multiplication it hands over to, the comparison
            // `SQR_KARATSUBA_MAX_LIMBS` rests on.
            if w >= 64 {
                let mut p = [0f64; PASSES];
                for (k, slot) in p.iter_mut().enumerate() {
                    *slot = paired_saving(
                        CHUNK_PASSES,
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

            // Karatsuba squaring against whatever `mul` dispatches to at
            // this width: the comparison `SQR_KARATSUBA_MAX_LIMBS` rests on,
            // since past Toom-3 the ladder changes kernel underneath it.
            if w >= 384 {
                let mut p = [0f64; 5];
                for (k, slot) in p.iter_mut().enumerate() {
                    *slot = paired_saving(
                        CHUNK_PASSES,
                        chunk,
                        k % 2 == 1,
                        &mut || {
                            black_box(black_box(&v).mul(black_box(&v)));
                        },
                        &mut || {
                            black_box(black_box(&v).sqr_karatsuba_ref());
                        },
                    );
                }
                report("karatsuba sqr vs mul", w, p);
            }

            // And the public entry points, which is what a caller sees.
            let mut p = [0f64; PASSES];
            for (k, slot) in p.iter_mut().enumerate() {
                *slot = paired_saving(
                    CHUNK_PASSES,
                    chunk,
                    k % 2 == 1,
                    &mut || {
                        black_box(black_box(&v).mul(black_box(&v)));
                    },
                    &mut || {
                        black_box(black_box(&v).square());
                    },
                );
            }
            report("square vs mul", w, p);
        }
    }

    /// Schoolbook against Karatsuba over the widths around
    /// `KARATSUBA_THRESHOLD_LIMBS`, which is the crossover this measures and
    /// nothing else sets.
    #[test]
    #[ignore = "timing probe for the schoolbook/Karatsuba crossover; run with --ignored"]
    fn karatsuba_crossover_timing() {
        use std::hint::black_box;
        use std::time::Instant;
        // From where a split first has anything to save up to well past the
        // shipped threshold, so the crossover falls inside the sweep.
        const WIDTHS: [usize; 13] = [8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256];
        // Operand pairs per width, each timed as the best of several runs and
        // the row taken as the median over operands: the best run sheds
        // scheduler interference, the median sheds an unlucky operand. A
        // crossover decided by a few percent needs both.
        const OPERANDS: usize = 9;
        const RUNS: usize = 5;
        // Repetitions sized to about a millisecond of work per run at the
        // narrowest width, and held above a floor where the clock's
        // resolution would otherwise show through.
        const WORK_UNITS: usize = 2_000_000;
        const MIN_REPS: usize = 200;
        // Arbitrary ("KARATSUB" in ASCII), fixed so a failure reproduces.
        const SEED: u64 = 0x4b41_5241_5453_5542;
        let mut seed = SEED;
        eprintln!(
            "{:>6} {:>12} {:>12} {:>7}  best",
            "words", "school_us", "kara_us", "saving"
        );
        for words in WIDTHS {
            let reps = (WORK_UNITS / (words * words)).max(MIN_REPS);
            let operands: Vec<(BigUint, BigUint)> = (0..OPERANDS)
                .map(|_| {
                    (
                        seeded_biguint(words, &mut seed),
                        seeded_biguint(words, &mut seed),
                    )
                })
                .collect();
            let time = |f: &dyn Fn(&BigUint, &BigUint) -> BigUint| {
                let mut per_operand: Vec<f64> = operands
                    .iter()
                    .map(|(a, b)| {
                        let mut best = f64::INFINITY;
                        for _ in 0..RUNS {
                            black_box(f(a, b));
                            let start = Instant::now();
                            for _ in 0..reps {
                                black_box(f(a, b));
                            }
                            best = best.min(start.elapsed().as_secs_f64() / reps as f64 * 1e6);
                        }
                        best
                    })
                    .collect();
                per_operand.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
                per_operand[per_operand.len() / 2]
            };
            let school = time(&|a, b| BigUint::mul_schoolbook_ref(a, b));
            let kara = time(&|a, b| a.mul_karatsuba_ref(b));
            let best = if school <= kara {
                "schoolbook"
            } else {
                "karatsuba"
            };
            let saving = (school - kara) / school * 100.0;
            eprintln!("{words:6} {school:12.4} {kara:12.4} {saving:+6.1}%  {best}");
        }
    }

    /// The transform against the ladder across the padding staircase.
    ///
    /// A radix-2 transform rounds the convolution length up to a power of
    /// two, so the cost per limb depends on where a width falls in that
    /// staircase: 16,384 limbs pad to 8 coefficients per limb and 20,410 to
    /// 12.8. The gates admit the NTT only below a measured ratio, which is
    /// what this measures. The awkward widths are not academic — the NFS
    /// square root lifts at 1.3 Mbit, which is 20,410 limbs.
    #[test]
    #[ignore = "timing probe for the NTT padding gates; run with --ignored"]
    fn ntt_padding_gate_timing() {
        use std::hint::black_box;
        use std::time::Instant;
        const RUNS: usize = 3;
        // Arbitrary ("ntt_pad!" in ASCII), fixed so a failure reproduces.
        const SEED: u64 = 0x6e74_745f_7061_6421;
        let mut seed = SEED;
        eprintln!(
            "{:>8} {:>7} {:>11} {:>11} {:>8}  best",
            "words", "ratio", "toom_us", "ntt_us", "saving"
        );
        for words in [
            12_288usize,
            16_384,
            20_410,
            24_576,
            32_768,
            40_960,
            49_152,
            65_536,
            81_920,
            98_304,
        ] {
            let a = seeded_biguint(words, &mut seed);
            let b = seeded_biguint(words, &mut seed);
            // Base-2^16 digits per limb, and the radix-2 length the
            // convolution rounds up to.
            let digits = words * 4;
            let padded = (2 * digits - 1).next_power_of_two();
            let ratio = padded as f64 / words as f64;
            let time = |f: &dyn Fn() -> BigUint| {
                let mut best = f64::INFINITY;
                for _ in 0..RUNS {
                    let start = Instant::now();
                    black_box(f());
                    best = best.min(start.elapsed().as_secs_f64() * 1e6);
                }
                best
            };
            let toom = time(&|| a.mul_toom4_ref(&b));
            let ntt = time(&|| a.mul_ntt_ref(&b));
            let best = if toom <= ntt { "toom4" } else { "ntt" };
            let saving = (toom - ntt) / toom * 100.0;
            eprintln!("{words:8} {ratio:7.2} {toom:11.1} {ntt:11.1} {saving:+7.1}%  {best}");
        }
    }

    #[test]
    #[ignore = "timing probe for tuning the Toom thresholds; run with --ignored"]
    fn toom_crossover_timing() {
        use std::hint::black_box;
        use std::time::Instant;
        // Operand limbs per timed run (reps · words), so every width is
        // timed over the same amount of input.
        const WORK_LIMBS: usize = 2_000_000;
        // Floor on repetitions per run, so the widest operands still average
        // over a score of products.
        const MIN_REPS: usize = 20;
        // Independent operand pairs per width, and best-of runs per pair:
        // the average over pairs sheds an unlucky operand, the minimum over
        // runs sheds scheduler interference. Both counts are arbitrary.
        const OPERANDS: usize = 4;
        const RUNS: usize = 3;
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xC0FF_EE00_1234_5678;
        let mut seed = SEED;
        eprintln!(
            "{:>6} {:>11} {:>11} {:>11}  best",
            "words", "kara_us", "toom3_us", "toom4_us"
        );
        // Past the shipped Toom-4 threshold as well as up to it: a sweep that
        // stops at a crossover cannot show one.
        for &words in &[
            96usize, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288,
        ] {
            let reps = (WORK_LIMBS / words).max(MIN_REPS);
            let operands: Vec<(BigUint, BigUint)> = (0..OPERANDS)
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
                    for _ in 0..RUNS {
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
    #[ignore = "timing probe for tuning the exact NTT crossover; run with --ignored"]
    fn ntt_crossover_timing() {
        use std::hint::black_box;
        use std::time::Instant;

        // Operand pairs per width: three below `NTT_PROBE_WIDE_WORDS`, one
        // at or above it.
        const SAMPLES: usize = 3;
        const SAMPLES_WIDE: usize = 1;
        // Operand limbs per timed run: reps fall as 1/words, clamped so a
        // run is at least one product and at most four.
        const REP_WORK_LIMBS: usize = 16_384;
        const MIN_REPS: usize = 1;
        const MAX_REPS: usize = 4;
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x510e_527f_ade6_82d1;
        let mut seed = SEED;
        let available = crate::available_parallelism();
        eprintln!("available contexts: {available}");
        eprintln!(
            "{:>7} {:>12} {:>12} {:>12} {:>12} {:>12}  best",
            "words", "toom4_us", "serial_us", "two_us", "auto_us", "square_us"
        );
        // From 2,048 limbs to the serial threshold, 131,072: every doubling
        // plus points between them (3/2 of the lower, 7/8 and 15/16 of the
        // upper), so both sides of each padding step are sampled.
        let word_sizes = std::env::var("RUMP_NTT_TIMING_WORDS").map_or_else(
            |_| {
                vec![
                    2048usize, 3072, 4096, 6144, 8192, 12_288, 16_384, 24_576, 32_768, 49_152,
                    57_344, 61_440, 65_536, 98_304, 114_688, 122_880, 131_072,
                ]
            },
            |list| {
                list.split(',')
                    .map(|word| {
                        word.trim()
                            .parse::<usize>()
                            .expect("RUMP_NTT_TIMING_WORDS entries must be limb counts")
                    })
                    .collect()
            },
        );
        for words in word_sizes {
            let wide = words >= NTT_PROBE_WIDE_WORDS;
            let samples = if wide { SAMPLES_WIDE } else { SAMPLES };
            let rounds = if wide {
                NTT_PROBE_ROUNDS_WIDE
            } else {
                NTT_PROBE_ROUNDS
            };
            let operands: Vec<(BigUint, BigUint)> = (0..samples)
                .map(|_| {
                    (
                        seeded_biguint(words, &mut seed),
                        seeded_biguint(words, &mut seed),
                    )
                })
                .collect();
            let reps = (REP_WORK_LIMBS / words).clamp(MIN_REPS, MAX_REPS);
            let time = |f: &dyn Fn(&BigUint, &BigUint) -> BigUint| {
                let mut total = 0.0;
                for (lhs, rhs) in &operands {
                    let mut best = f64::INFINITY;
                    for _ in 0..rounds {
                        black_box(f(lhs, rhs));
                        let started = Instant::now();
                        for _ in 0..reps {
                            black_box(f(lhs, rhs));
                        }
                        best = best.min(started.elapsed().as_secs_f64() / reps as f64 * 1e6);
                    }
                    total += best;
                }
                total / operands.len() as f64
            };
            let toom4 = time(&|lhs, rhs| lhs.mul_toom4_ref(rhs));
            let serial = time(&|lhs, rhs| lhs.mul_ntt_serial_ref(rhs));
            let two = time(&|lhs, rhs| lhs.mul_ntt_with_contexts_ref(rhs, 2));
            let automatic = time(&|lhs, rhs| lhs.mul_ntt_ref(rhs));
            let square = time(&|lhs, _| lhs.sqr_ntt_ref());
            let best = if toom4 <= automatic { "toom4" } else { "ntt" };
            eprintln!(
                "{words:7} {toom4:12.3} {serial:12.3} {two:12.3} {automatic:12.3} {square:12.3}  {best}"
            );
        }
    }

    #[test]
    #[ignore = "timing probe for exact NTT worker scaling; run with --ignored"]
    fn ntt_worker_scaling_timing() {
        use std::hint::black_box;
        use std::time::Instant;

        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xbb67_ae85_84ca_a73b;
        let mut seed = SEED;
        let available = crate::available_parallelism();
        // The parallel threshold to the serial one, by doublings.
        let word_sizes = std::env::var("RUMP_NTT_SCALING_WORDS").map_or_else(
            |_| vec![8_192usize, 16_384, 32_768, 65_536, 131_072],
            |list| {
                list.split(',')
                    .map(|word| {
                        word.trim()
                            .parse::<usize>()
                            .expect("RUMP_NTT_SCALING_WORDS entries must be limb counts")
                    })
                    .collect()
            },
        );
        // Powers of two up to 64, the largest target `ntt::worker_count`
        // selects.
        let worker_counts = std::env::var("RUMP_NTT_SCALING_WORKERS").map_or_else(
            |_| vec![1usize, 2, 4, 8, 16, 32, 64],
            |list| {
                list.split(',')
                    .map(|worker| {
                        let workers = worker
                            .trim()
                            .parse::<usize>()
                            .expect("RUMP_NTT_SCALING_WORKERS entries must be worker counts");
                        assert!(
                            workers.is_power_of_two(),
                            "worker counts must be powers of two"
                        );
                        workers
                    })
                    .collect()
            },
        );
        let configured_rounds = std::env::var("RUMP_NTT_SCALING_ROUNDS").ok().map(|rounds| {
            rounds
                .parse::<usize>()
                .expect("RUMP_NTT_SCALING_ROUNDS must be a positive count")
        });
        assert_ne!(
            configured_rounds,
            Some(0),
            "scaling rounds must be positive"
        );
        eprintln!("available contexts: {available}");
        eprintln!("{:>7} {:>7} {:>12}", "words", "workers", "product_us");
        for words in word_sizes {
            let lhs = seeded_biguint(words, &mut seed);
            let rhs = seeded_biguint(words, &mut seed);
            let expected = lhs.mul_ntt_serial_ref(&rhs);
            for &workers in &worker_counts {
                if workers > available {
                    continue;
                }
                let actual = lhs.mul_ntt_with_workers_ref(&rhs, workers);
                assert_eq!(actual, expected, "NTT product at {workers} workers");
                let rounds = configured_rounds.unwrap_or(if words < NTT_PROBE_WIDE_WORDS {
                    NTT_PROBE_ROUNDS
                } else {
                    NTT_PROBE_ROUNDS_WIDE
                });
                let mut best = f64::INFINITY;
                for _ in 0..rounds {
                    black_box(lhs.mul_ntt_with_workers_ref(&rhs, workers));
                    let started = Instant::now();
                    black_box(lhs.mul_ntt_with_workers_ref(&rhs, workers));
                    best = best.min(started.elapsed().as_secs_f64() * 1e6);
                }
                eprintln!("{words:7} {workers:7} {best:12.3}");
            }
        }
    }

    #[test]
    #[ignore = "wall-clock phase profile for exact NTT; run with --ignored"]
    fn ntt_phase_profile() {
        let words = std::env::var("RUMP_NTT_PROFILE_WORDS").map_or(131_072usize, |words| {
            words
                .parse()
                .expect("RUMP_NTT_PROFILE_WORDS must be a limb count")
        });
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x3c6e_f372_fe94_f82b;
        let mut seed = SEED;
        let lhs = seeded_biguint(words, &mut seed);
        let rhs = seeded_biguint(words, &mut seed);
        let transform_len =
            super::ntt::transform_len(words, words).expect("profile transform is supported");
        let workers = std::env::var("RUMP_NTT_PROFILE_WORKERS").map_or_else(
            |_| super::ntt::automatic_worker_count(transform_len),
            |workers| {
                workers
                    .parse()
                    .expect("RUMP_NTT_PROFILE_WORKERS must be a worker count")
            },
        );
        let expected = lhs.mul_ntt_serial_ref(&rhs);
        let (actual, profile) = super::ntt::multiply_profiled(&lhs, &rhs, workers);
        assert_eq!(actual, expected);

        let phases = [
            ("allocate", profile.allocate),
            ("prepare", profile.prepare_inputs),
            ("forward", profile.forward),
            ("pointwise", profile.pointwise),
            ("inverse", profile.inverse),
            ("residues", profile.residue_copy),
            ("clear", profile.clear),
            ("crt/carry", profile.reconstruct),
        ];
        let total: std::time::Duration = phases.iter().map(|(_, elapsed)| *elapsed).sum();
        eprintln!("words={words} workers={workers} transform={transform_len}");
        eprintln!("{:>12} {:>12} {:>8}", "phase", "microseconds", "percent");
        for (phase, elapsed) in phases {
            eprintln!(
                "{phase:>12} {:12.3} {:7.2}%",
                elapsed.as_secs_f64() * 1e6,
                elapsed.as_secs_f64() / total.as_secs_f64() * 100.0
            );
        }
        eprintln!(
            "{:>12} {:12.3} {:7.2}%",
            "total",
            total.as_secs_f64() * 1e6,
            100.0
        );
    }

    #[test]
    fn shr_bits_inverts_shl_bits_and_matches_division() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x6a09_e667_f3bc_c908;
        let mut seed = SEED;
        // Shifts of zero, within a limb, at and across limb boundaries, and
        // past the value; eight random values at each of four widths.
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
                    // 2^n, and division does not use the shift code.
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
    #[should_panic(expected = "does not fit")]
    fn padded_little_endian_bytes_reject_overflow() {
        let _ = BigUint::from_u64(0x0102).to_le_bytes_padded(1);
    }

    #[test]
    fn little_endian_bytes_mirror_big_endian() {
        // The fixed conventions, spelled out.
        let value = BigUint::from_u64(0x0102);
        assert_eq!(value.to_le_bytes(), vec![0x02, 0x01]);
        assert_eq!(value.to_le_bytes_padded(2), vec![0x02, 0x01]); // exact fit
        assert_eq!(value.to_le_bytes_padded(5), vec![0x02, 0x01, 0, 0, 0]);
        assert_eq!(BigUint::from_le_bytes(&[0x02, 0x01, 0, 0]), value);
        assert!(BigUint::from_le_bytes(&[]).is_zero());
        assert!(BigUint::from_le_bytes(&[0, 0, 0]).is_zero());
        assert_eq!(BigUint::zero().to_le_bytes(), vec![0]);
        assert_eq!(BigUint::zero().to_le_bytes_padded(3), vec![0, 0, 0]);
        assert!(BigUint::zero().to_le_bytes_padded(0).is_empty());

        // Random byte strings across limb boundaries, some with leading
        // (high) zero bytes. The oracle is the input bytes themselves, decoded
        // by the big-endian parser: every encoder must reproduce
        // them, stripped or padded, and in either byte order.
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x1eb1_7e50_0000_0001;
        let mut seed = SEED;
        // Every length from empty to five limbs plus one byte; eight random
        // strings each, about a third with a leading zero byte.
        for byte_len in 0usize..=41 {
            for _ in 0..8 {
                let mut bytes: Vec<u8> = (0..byte_len)
                    .map(|_| (lcg_next(&mut seed) >> 56) as u8)
                    .collect();
                if byte_len > 1 && lcg_next(&mut seed).is_multiple_of(3) {
                    bytes[0] = 0;
                }
                let value = BigUint::from_be_bytes(&bytes);
                let minimal = value.bits().div_ceil(8);
                let significant = &bytes[bytes.len() - minimal..];

                let expected_be = if minimal == 0 {
                    vec![0]
                } else {
                    significant.to_vec()
                };
                let mut expected_le = expected_be.clone();
                expected_le.reverse();
                let be = value.to_be_bytes();
                let le = value.to_le_bytes();
                assert_eq!(be, expected_be);
                assert_eq!(le, expected_le);
                // Allocated at the final length: nothing sits in spare
                // capacity past the bytes a caller can see.
                assert_eq!(be.capacity(), be.len());
                assert_eq!(le.capacity(), le.len());

                let mut reversed = bytes.clone();
                reversed.reverse();
                assert_eq!(BigUint::from_le_bytes(&reversed), value);
                assert_eq!(BigUint::from_le_bytes(&le), value);

                // Padding from an exact fit to nine bytes over, which crosses
                // a limb boundary.
                for width in minimal..minimal + 10 {
                    let mut padded_be = vec![0u8; width - minimal];
                    padded_be.extend_from_slice(significant);
                    let mut padded_le = padded_be.clone();
                    padded_le.reverse();
                    assert_eq!(value.to_be_bytes_padded(width), padded_be);
                    assert_eq!(value.to_le_bytes_padded(width), padded_le);
                }
            }
        }
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
        assert_eq!(q.mul(&divisor).add(&r), dividend);
    }

    /// `(q, r)` with `dividend = q * divisor + r` and `r < divisor` is unique,
    /// so checking the pair is a complete correctness statement for
    /// [`BigUint::div_rem`] and needs no separately computed expected value.
    fn assert_div_rem_invariant(dividend: &BigUint, divisor: &BigUint) {
        let (quotient, remainder) = dividend.div_rem(divisor);
        assert!(
            remainder < *divisor,
            "remainder {remainder:?} not reduced rem {divisor:?}"
        );
        assert_eq!(
            quotient.mul(divisor).add(&remainder),
            *dividend,
            "q * d + r != n for {dividend:?} / {divisor:?}"
        );
    }

    #[test]
    fn div_rem_invariant_over_limb_shapes() {
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x243f_6a88_85a3_08d3;
        let mut seed = SEED;
        // Cover both division paths (one-limb Horner and multi-limb Knuth),
        // every quotient length from one limb up, and — because the leading
        // limb is random — a spread of D1 normalization shifts. Dividends of
        // one to nine limbs, every divisor width up to them, twelve random
        // pairs per shape.
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
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xb504_f333_f9de_6484;
        let mut seed = SEED;
        // Three limbs is the narrowest divisor with an add-back, up to six;
        // quotients at the small end, an arbitrary middle value, and the top
        // of a limb.
        for divisor_words in 3..=6usize {
            for q in [1u64, 2, 12_345, u64::MAX - 1] {
                let mut divisor = seeded_biguint(divisor_words, &mut seed);
                // A D1 shift of zero keeps the construction exact.
                divisor.limbs[divisor_words - 1] |= 1 << 63;

                let scale = BigUint::from_u64(q).add(&BigUint::one());
                let dividend = scale.mul(&divisor).sub(&BigUint::one());

                assert_div_rem_invariant(&dividend, &divisor);
                // `(q + 1) * d - 1 = q * d + (d - 1)`, so the answer is exact.
                let (quotient, remainder) = dividend.div_rem(&divisor);
                assert_eq!(quotient, BigUint::from_u64(q));
                assert_eq!(remainder, divisor.sub(&BigUint::one()));
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
        let ctx = MontgomeryContext::new(&BigUint::from_u64(1_000_000_007))
            .expect("odd modulus builds a context");
        let base = BigUint::from_u64(123_456_789);
        let exponent = BigUint::from_u64(65_537);
        assert_eq!(ctx.pow(&base, &exponent), BigUint::from_u64(560_583_526));
    }

    #[test]
    fn montgomery_ctx_mul_matches_small_arithmetic() {
        let ctx = MontgomeryContext::new(&BigUint::from_u64(1_000_000_007))
            .expect("odd modulus builds a context");
        let a = BigUint::from_u64(123_456_789);
        let b = BigUint::from_u64(987_654_321);
        assert_eq!(ctx.mul(&a, &b), BigUint::from_u64(259_106_859));
    }

    #[test]
    fn mod_mul_handles_even_modulus() {
        // Even moduli have no Montgomery representation.
        let a = BigUint::from_u64(37);
        let b = BigUint::from_u64(19);
        let modulus = BigUint::from_u64(100);
        assert_eq!(BigUint::mod_mul(&a, &b, &modulus), BigUint::from_u64(3));
    }

    #[test]
    fn mod_mul_matches_montgomery_context() {
        // The one-shot path and the reusable-context path must agree.
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x0123_4567_89ab_cdef;
        let mut seed = SEED;
        // Widths by doubling from one limb; eight random triples each.
        for words in [1usize, 2, 4, 8, 16] {
            for _ in 0..8 {
                let lhs = seeded_biguint(words, &mut seed);
                let rhs = seeded_biguint(words, &mut seed);
                let mut modulus = seeded_biguint(words, &mut seed);
                modulus.limbs[0] |= 1; // Montgomery needs an odd modulus.

                let ctx = MontgomeryContext::new(&modulus).expect("odd modulus builds a context");
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
        assert_eq!(a.add(&b), BigInt::from_biguint(BigUint::from_u64(7)));
        assert_eq!(
            b.sub(&a),
            BigInt::from_parts(Sign::Negative, BigUint::from_u64(13))
        );
        assert_eq!(
            BigInt::from_parts(Sign::Negative, BigUint::from_u64(3))
                .rem_euclid(&BigUint::from_u64(11)),
            BigUint::from_u64(8)
        );
    }
}
