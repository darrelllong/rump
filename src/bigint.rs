//! Multiprecision unsigned and signed integers on `u64` limbs.
//!
//! The representation uses little-endian `u64` limbs because the algorithms
//! are naturally word-oriented. The kernels come straight from the literature
//! so they are easy to audit against their sources: schoolbook (Knuth's
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

/// Sign of a [`BigInt`].
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

/// Signed multiprecision integer: a sign joined to a [`BigUint`] magnitude.
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
    /// Construct zero.
    #[must_use]
    pub fn zero() -> Self {
        Self { limbs: Vec::new() }
    }

    /// Construct one.
    #[must_use]
    pub fn one() -> Self {
        Self { limbs: vec![1] }
    }

    /// Construct from a machine word.
    #[must_use]
    pub fn from_u64(value: u64) -> Self {
        if value == 0 {
            Self::zero()
        } else {
            Self { limbs: vec![value] }
        }
    }

    /// Construct from a `u128`.
    ///
    /// # Panics
    ///
    /// Panics only if the internal limb split invariants fail unexpectedly.
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
    /// Panics only if the internal representation is corrupt and a non-zero
    /// value contains no non-zero bytes.
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

    /// Largest power of `radix` fitting a limb, with its digit count.
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

    /// Extract digits of a power-of-two radix, most significant first.
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
    /// bit boundary, limb-aligned or not.
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

    /// Return whether the value is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Return whether the value is odd.
    #[must_use]
    pub fn is_odd(&self) -> bool {
        !self.is_zero() && (self.limbs[0] & 1) == 1
    }

    /// Return whether the value is exactly one.
    #[must_use]
    pub fn is_one(&self) -> bool {
        self.limbs.len() == 1 && self.limbs[0] == 1
    }

    /// Number of significant bits.
    ///
    /// # Panics
    ///
    /// Panics only if the internal representation is corrupt and a non-zero
    /// value contains no limbs.
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
    /// both share.
    #[must_use]
    pub fn sqrt_floor(&self) -> Self {
        self.sqrt_rem().0
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
        // Seed: 2^⌈bits/2⌉ ≥ ⌈√self⌉, one bit above the root's width.
        let mut current = Self::zero();
        current.set_bit(self.bits().div_ceil(2));
        loop {
            // next = (current + self/current) / 2
            let (quotient, _) = self.div_rem(&current);
            let mut next = current.add_ref(&quotient);
            next.shr1();
            if next >= current {
                let square = current.square_ref();
                debug_assert!(square <= *self, "certified root is not above the value");
                return (current.clone(), self.sub_ref(&square));
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

    /// Test bit `index`.
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

    /// Set bit `index`.
    pub fn set_bit(&mut self, index: usize) {
        let limb = index / 64;
        let shift = index % 64;
        if self.limbs.len() <= limb {
            self.limbs.resize(limb + 1, 0);
        }
        self.limbs[limb] |= 1u64 << shift;
    }

    /// Add another bigint in place.
    ///
    /// # Panics
    ///
    /// Panics only if the internal `u128` accumulator cannot be split back
    /// into `u64` limbs, which would indicate a logic error.
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

    /// Return `self + other`.
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
    /// Panics only if the internal `u128` accumulator cannot be split back
    /// into `u64` limbs, which would indicate a logic error.
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

    /// Subtract another bigint in place. Panics if `self < other`.
    ///
    /// # Panics
    ///
    /// Panics if `self < other`.
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

    /// Return `self - other`. Panics if `self < other`.
    #[must_use]
    pub fn sub_ref(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.sub_assign_ref(other);
        out
    }

    /// Multiply two big integers.
    ///
    /// # Panics
    ///
    /// Panics only if the internal `u128` accumulators cannot be split back
    /// into `u64` limbs, which would indicate a logic error.
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

        Self::mul_schoolbook_ref(self, other)
    }

    /// Multiply a value by itself.
    #[must_use]
    pub fn square_ref(&self) -> Self {
        self.mul_ref(self)
    }

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

    fn should_use_karatsuba(lhs: &Self, rhs: &Self) -> bool {
        let short = lhs.limbs.len().min(rhs.limbs.len());
        let long = lhs.limbs.len().max(rhs.limbs.len());
        short >= KARATSUBA_THRESHOLD_LIMBS && long <= short * KARATSUBA_MAX_IMBALANCE
    }

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
        let four_c2 = c2.mul_biguint_ref(&BigUint::from_u64(4));
        let sixteen_c4 = c4.mul_biguint_ref(&BigUint::from_u64(16));
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

        let scale = |x: &BigInt, m: u64| x.mul_biguint_ref(&BigUint::from_u64(m));
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

    /// Shift left by one bit.
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

    /// Shift right by one bit.
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

    /// Left-shift by `n` bits.
    ///
    /// Implemented as `n / 64` full-limb shifts (inserting zero limbs at the
    /// low end) followed by up to 63 single-bit left shifts, which avoids
    /// undefined behaviour from shifting a `u64` by 64 or more positions.
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
            // limbs do not linger beyond the vector's length.
            self.limbs.fill(0);
            self.limbs.clear();
            return;
        }

        // Whole-limb shift: move the high limbs down, then wipe the vacated
        // top slots before truncating for the same reason as above.
        if limb_shifts > 0 {
            let kept = self.limbs.len() - limb_shifts;
            self.limbs.copy_within(limb_shifts.., 0);
            self.limbs[kept..].fill(0);
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

    /// Compute `self mod modulus`.
    #[must_use]
    pub fn modulo(&self, modulus: &Self) -> Self {
        let (_, remainder) = self.div_rem(modulus);
        remainder
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
    /// the reduction that trade no longer pays: a Montgomery context costs two
    /// divisions to construct and then four Montgomery multiplies to encode,
    /// multiply, and decode, where this costs one multiply and one division —
    /// and it needs no odd-modulus special case.
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

    /// Return `(quotient, remainder)` for Euclidean division. Panics on zero divisor.
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

    fn normalize(&mut self) {
        // Canonical representation invariant:
        // - zero has `limbs.is_empty()`
        // - non-zero values have a non-zero top limb
        while self.limbs.last().copied() == Some(0) {
            self.limbs.pop();
        }
    }

    /// Legacy entry point kept for the two callers that still hand in
    /// `BigUint`s of unknown shape ([`MontgomeryCtx::mul_mont`] and friends):
    /// pads the operands to the modulus width and defers to the slice kernels.
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
            lhs.limbs.len() <= width && rhs.limbs.len() <= width,
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
}

/// Scratch limbs the `mont_*` kernels need for a `width`-limb modulus: a
/// `2 * width` product plus one limb so the reduction's final carry has a
/// home.
#[inline]
fn mont_scratch_limbs(width: usize) -> usize {
    width * 2 + 1
}

/// Copy `src` into `dst`, zero-padding the (little-endian) high limbs.
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

/// Compare two equal-width little-endian limb slices.
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

    /// Grow `workspace` to at least the kernels' scratch size and return it
    /// as a slice.
    fn scratch<'a>(&self, workspace: &'a mut Vec<u64>) -> &'a mut [u64] {
        let needed = mont_scratch_limbs(self.width());
        if workspace.len() < needed {
            workspace.resize(needed, 0);
        }
        workspace
    }

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

    /// Build a Montgomery context for a non-zero odd modulus.
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

    /// Convert an ordinary residue into Montgomery form.
    #[must_use]
    pub fn encode(&self, value: &BigUint) -> BigUint {
        let mut workspace = Vec::new();
        let result = self.encode_with_workspace(value, &mut workspace);
        crate::scrub::zeroize_slice(workspace.as_mut_slice());
        result
    }

    /// Convert a Montgomery residue back to the ordinary representation.
    #[must_use]
    pub fn decode(&self, value: &BigUint) -> BigUint {
        let mut workspace = Vec::new();
        let result = self.decode_with_workspace(value, &mut workspace);
        crate::scrub::zeroize_slice(workspace.as_mut_slice());
        result
    }

    /// Multiply two ordinary residues modulo the context modulus.
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

    /// Square one ordinary residue modulo the context modulus.
    #[must_use]
    pub fn square(&self, value: &BigUint) -> BigUint {
        let mut workspace = Vec::new();
        let value_mont = self.encode_with_workspace(value, &mut workspace);
        let square_mont = BigUint::montgomery_mul_odd_with_workspace(
            &value_mont,
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
    /// Unlike [`Self::mul`]/[`Self::pow`] this does not scrub its workspace:
    /// it is the innermost field-multiply, called in tight loops, so the
    /// per-call volatile wipe is omitted for speed. The product is returned as
    /// a `BigUint`, whose own `Drop` wipes it; the caller keeps the value.
    #[must_use]
    pub fn mul_mont(&self, lhs: &BigUint, rhs: &BigUint) -> BigUint {
        let mut workspace = Vec::new();
        BigUint::montgomery_mul_odd_with_workspace(
            lhs,
            rhs,
            &self.modulus,
            self.n0_inv,
            &mut workspace,
        )
    }

    /// Square a residue that is already in Montgomery form, staying in
    /// Montgomery form.
    #[must_use]
    pub fn square_mont(&self, value: &BigUint) -> BigUint {
        self.mul_mont(value, value)
    }

    /// The Montgomery encoding of one (`R mod n`).
    #[must_use]
    pub fn one_mont(&self) -> &BigUint {
        &self.one_mont
    }

    /// Compute `base^exponent mod modulus` inside the context.
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
    /// This is useful when callers reuse the same base and can cache the
    /// encoded value once.
    #[must_use]
    pub fn pow_encoded(&self, base_mont: &BigUint, exponent: &BigUint) -> BigUint {
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
    /// Decimal rendering, through [`BigUint::to_str_radix`].
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.pad_integral(true, "", &self.to_str_radix(10))
    }
}

impl core::str::FromStr for BigUint {
    type Err = ParseBigIntError;

    /// Decimal parsing, through [`BigUint::from_str_radix`].
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::from_str_radix(text, 10).ok_or(ParseBigIntError)
    }
}

impl core::fmt::Display for BigInt {
    /// Decimal rendering with a leading `-` for negative values.
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

    /// Decimal parsing with an optional leading `-`.
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

/// Return `value` shifted right by `shift` bits (below 64) in a fresh buffer.
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

/// Signed product `a · b`, for the Toom-3 evaluate/interpolate arithmetic.
fn bigint_mul(a: &BigInt, b: &BigInt) -> BigInt {
    a.mul_ref(b)
}

/// `x / divisor` where `divisor` divides `x` exactly — the interpolation
/// steps of Toom-3 and Toom-4 (dividing by 2, 3, 4, 5, 8, 12).
fn bigint_div_exact(x: &BigInt, divisor: u64) -> BigInt {
    let (quotient, remainder) = BigUint::div_rem_limb(x.magnitude().limbs(), divisor);
    debug_assert!(
        remainder == 0,
        "Toom interpolation divides evenly by {divisor}"
    );
    BigInt::from_parts(x.sign(), quotient)
}

impl BigInt {
    /// Construct zero.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            sign: Sign::Zero,
            magnitude: BigUint::zero(),
        }
    }

    /// Construct from an explicit sign and magnitude.
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

    /// Construct a non-negative signed integer from an unsigned value.
    #[must_use]
    pub fn from_biguint(magnitude: BigUint) -> Self {
        Self::from_parts(Sign::Positive, magnitude)
    }

    /// Return the sign.
    #[must_use]
    pub fn sign(&self) -> Sign {
        self.sign
    }

    /// Return the absolute value.
    #[must_use]
    pub fn magnitude(&self) -> &BigUint {
        &self.magnitude
    }

    /// Negate the integer.
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
    pub fn add_ref(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.add_assign_ref(other);
        out
    }

    /// Add another integer in place, reusing the magnitude's limb buffer in
    /// every sign combination.
    pub fn add_assign_ref(&mut self, other: &Self) {
        self.combine_assign(other.sign, &other.magnitude);
    }

    /// Return `self - other`.
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

    /// Return `self * factor` for a non-negative factor.
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

    /// Signed product `self * other` (the Half-GCD matrix arithmetic composes
    /// 2×2 matrices of full-size signed entries).
    #[must_use]
    pub(crate) fn mul_ref(&self, other: &Self) -> Self {
        let sign = match (self.sign, other.sign) {
            (Sign::Zero, _) | (_, Sign::Zero) => Sign::Zero,
            (lhs, rhs) if lhs == rhs => Sign::Positive,
            _ => Sign::Negative,
        };
        Self::from_parts(sign, self.magnitude.mul_ref(&other.magnitude))
    }

    /// Reduce modulo a positive modulus and return the least non-negative residue.
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
}

#[cfg(test)]
mod tests {
    use super::{BigInt, BigUint, MontgomeryCtx, Sign};

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
    fn square_ref_matches_mul_ref() {
        let mut seed = 0x9e37_79b9_7f4a_7c15;
        for words in [1usize, 2, 8, 32, 48] {
            for _ in 0..8 {
                let value = seeded_biguint(words, &mut seed);
                assert_eq!(value.square_ref(), value.mul_ref(&value));
            }
        }
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
