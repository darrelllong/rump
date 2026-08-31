//! Montgomery arithmetic: the domain, its slice kernels, and REDC.
//!
//! Montgomery, *Modular Multiplication Without Trial Division*, Mathematics of
//! Computation 44 (1985); the separated-operand-scanning shape follows Koç,
//! Acar & Kaliski, *Analyzing and Comparing Montgomery Multiplication
//! Algorithms*, IEEE Micro 16 (1996).
//!
//! The kernels take `&[u64]` slices rather than `BigUint` so the
//! exponentiation ladder can reuse one workspace across a whole computation.
//!
//! The test module lives in the parent, which is why four of the kernels are
//! `pub(super)` rather than private.

use super::{bit_span, low_u64, BigUint, ModulusError};
use core::cmp::Ordering;
use std::sync::Arc;

/// The identity a context and its residues share.
///
/// Deliberately carries no data: identity *is* the allocation, compared with
/// [`Arc::ptr_eq`]. A context and its clones share one `Arc`, so a clone
/// accepts the original's residues; two separately built contexts never share
/// one, so residues cannot cross between them.
///
/// It replaced an abbreviated tag — the modulus's low limb and limb count —
/// that was not unique: `2⁶⁴ + 3` and `2⁶⁵ + 3` are both odd, both two limbs,
/// both low limb 3, so each context accepted the other's residues and
/// decoded them under the wrong modulus. A wider fingerprint would only move
/// that boundary; sharing one allocation removes it.
#[derive(Debug, Eq, PartialEq)]
struct ContextIdentity;

/// A value in a [`MontgomeryContext`]'s domain.
///
/// Opaque on purpose. The domain invariant — encoded, reduced, belonging to
/// one context — is carried by the type rather than by a debug assertion.
/// There is no way to build one except [`MontgomeryContext::to_residue`] and
/// no way to read one except [`MontgomeryContext::from_residue`], so an
/// unencoded, unreduced, or foreign value cannot reach a kernel at all.
///
/// Belonging is by provenance, not by modulus: a context and its clones share
/// one identity, but a context *rebuilt* from the same modulus is a different
/// one and refuses residues it did not make. That is stricter than the
/// mathematics requires and deliberately so — it is the same rule in every
/// case, with no value to compare and so nothing to collide.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MontgomeryResidue {
    value: BigUint,
    identity: Arc<ContextIdentity>,
}

/// Reusable scratch for the domain operations.
///
/// Opaque, so the limb buffer is not part of the contract. Hand the same one
/// to every call in a loop and the scratch allocates once.
#[derive(Clone, Debug, Default)]
pub struct MontgomeryScratch {
    limbs: Vec<u64>,
}

impl MontgomeryScratch {
    /// A new, empty scratch buffer. It grows to the width the first operation
    /// needs and is reused from then on.
    #[must_use]
    pub fn new() -> Self {
        Self { limbs: Vec::new() }
    }
}

#[cfg(feature = "wipe")]
impl Drop for MontgomeryScratch {
    fn drop(&mut self) {
        // The scratch is reused across residue operations and can hold
        // secret-derived intermediates from the last one; wipe at end of
        // life like a `BigUint`'s limbs.
        crate::scrub::zeroize_slice(self.limbs.as_mut_slice());
    }
}

/// A residue was used with a context that did not produce it.
///
/// The type system cannot express "this residue belongs to that context" for
/// contexts built at run time, so the relationship is checked and reported.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ContextMismatch;

impl core::fmt::Display for ContextMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("residue belongs to a different Montgomery context")
    }
}

impl std::error::Error for ContextMismatch {}

/// Montgomery arithmetic context for a fixed odd modulus.
///
/// Long computations — exponentiation ladders, field arithmetic — spend
/// most of their time doing repeated modular multiplication under one
/// long-lived odd modulus. Precomputing the Montgomery constants once avoids
/// paying the setup cost on every multiply, and the explicit context lets
/// callers stay in the Montgomery domain across whole computations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MontgomeryContext {
    identity: Arc<ContextIdentity>,
    modulus: BigUint,
    // n0_inv = -n^{-1} mod 2^64 (Montgomery reduction coefficient).
    n0_inv: u64,
    // R^2 mod n with R = 2^(64 * limbs(n)): conversion factor into Montgomery form.
    r2_mod: BigUint,
    // 1 encoded in Montgomery form, i.e. R mod n.
    one_mont: BigUint,
}

/// Scratch limbs the `mont_*` kernels need for a `width`-limb modulus: a
/// `2 * width` product plus one limb so the reduction's final carry has a
/// home.
#[inline]
pub(super) fn mont_scratch_limbs(width: usize) -> usize {
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
pub(super) fn copy_padded(dst: &mut [u64], src: &[u64]) {
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
pub(super) fn mont_mul(
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
pub(super) fn mont_sqr(
    out: &mut [u64],
    value: &[u64],
    modulus: &[u64],
    n0_inv: u64,
    scratch: &mut [u64],
) {
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

impl MontgomeryContext {
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
            &value.rem(&self.modulus),
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
    /// The ladder: two engines selected by exponent width — binary
    /// square-and-multiply for exponents inside one word, a fixed 4-bit
    /// window above that — followed by a single decoding REDC. The result
    /// leaves the Montgomery domain here, so both public entry points return
    /// an ordinary residue.
    ///
    /// Both engines run on `width`-limb buffers swapped in place, so the
    /// ladder allocates nothing after the table and every buffer that held an
    /// exponent-dependent intermediate is wiped on the way out.
    /// The exponentiation ladder, returning the result **still encoded**.
    ///
    /// Split from the decoding step because the two callers want different
    /// halves: `pow_encoded_with_workspace` decodes, `pow_residue` keeps the
    /// domain element. Conflating them is exactly the trap HANDOFF records —
    /// a ladder that takes an encoded base and returns an ordinary residue —
    /// and it cost a wrong answer here before the tests caught it.
    fn pow_ladder_encoded(
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
            // `x^0 = 1` encoded is `one_mont`, and the modulus exceeds one.
            return self.one_mont.clone();
        }

        let width = self.width();
        let modulus = &self.modulus.limbs;
        let scratch = self.scratch(workspace);

        // The ladder runs on fixed-width buffers with a swap after each step,
        // so the whole exponentiation performs no allocation and no
        // intermediate wipes; every buffer that touched secret-derived state
        // is wiped once on the way out (under the `wipe` feature; the calls
        // compile to nothing otherwise).
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

            debug_assert!(seeded, "bits counts up to a set bit");
            crate::scrub::zeroize_slice(power.as_mut_slice());
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

            crate::scrub::zeroize_slice(table.as_mut_slice());
            acc
        };

        crate::scrub::zeroize_slice(tmp.as_mut_slice());
        crate::scrub::zeroize_slice(scratch);
        let mut result = BigUint { limbs: result };
        result.normalize();
        result
    }

    /// The ladder followed by the decode: an ordinary residue out.
    fn pow_encoded_with_workspace(
        &self,
        base_mont: &BigUint,
        exponent: &BigUint,
        workspace: &mut Vec<u64>,
    ) -> BigUint {
        if self.modulus.is_one() {
            return BigUint::zero();
        }
        let encoded = self.pow_ladder_encoded(base_mont, exponent, workspace);
        self.decode_with_workspace(&encoded, workspace)
    }

    /// Build a Montgomery context for an odd, non-zero modulus.
    ///
    /// Montgomery reduction works in the residue system of `R = 2^(64·limbs)`,
    /// and REDC requires `R` and the modulus to be coprime; since `R` is a
    /// power of two, that holds exactly when the modulus is odd. An even (or
    /// zero) modulus has no Montgomery form — reach for
    /// [`BarrettContext`](super::BarrettContext), which reduces modulo either parity. Construction also
    /// precomputes the inverse `−n⁻¹ mod 2⁶⁴` (Hensel lifting) and `R² mod n`
    /// once, so later encode/multiply steps do not repeat that work.
    ///
    /// # Errors
    ///
    /// [`ModulusError::Zero`] — zero has no residues.
    /// [`ModulusError::Even`] — `R` is a power of two, so it is coprime to
    /// the modulus only when the modulus is odd.
    pub fn new(modulus: &BigUint) -> Result<Self, ModulusError> {
        if modulus.is_zero() {
            return Err(ModulusError::Zero);
        }
        if !modulus.is_odd() {
            return Err(ModulusError::Even);
        }

        let n0_inv = montgomery_n0_inv(modulus.limbs[0]);

        // With `w` limbs, Montgomery arithmetic uses `R = 2^(64w)`. `R^2 mod
        // n` is the standard conversion factor for entering the Montgomery
        // domain because `montgomery_mul(a, R^2) = a * R^2 * R^-1 = aR`, the
        // Montgomery encoding of the ordinary residue `a`.
        let mut r2 = BigUint::zero();
        r2.set_bit(bit_span(modulus.limbs.len(), 128));
        let r2_mod = r2.rem(modulus);

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

        Ok(Self {
            identity: Arc::new(ContextIdentity),
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
        self.encode_with_workspace(value, &mut workspace)
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
            value.rem(&self.modulus)
        } else {
            value.clone()
        };
        let mut workspace = Vec::new();
        self.decode_with_workspace(&reduced, &mut workspace)
    }

    /// Multiply two ordinary residues modulo the context modulus — the
    /// one-shot encode → multiply → decode round trip.
    ///
    /// Convenient for a single product. Callers doing many operations should
    /// encode once and stay in the domain with [`Self::mul_residue`] /
    /// [`Self::square_residue`], decoding only at the end, rather than re-paying
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
        self.decode_with_workspace(&product_mont, &mut workspace)
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
        self.decode_with_workspace(&square_mont, &mut workspace)
    }

    /// The identity check every residue operation performs.
    ///
    /// Pointer equality on the shared `Arc`, so it cannot collide: a residue
    /// belongs to this context exactly when it was made by this context or a
    /// clone of it.
    fn check(&self, residue: &MontgomeryResidue) -> Result<(), ContextMismatch> {
        if Arc::ptr_eq(&residue.identity, &self.identity) {
            Ok(())
        } else {
            Err(ContextMismatch)
        }
    }

    /// Encode `value` into this context's Montgomery domain.
    ///
    /// The returned [`MontgomeryResidue`] carries the domain invariant — it is
    /// encoded, reduced, and belongs to this context — so no operation has to
    /// re-check it and no caller can violate it. That is the point of the
    /// type: the raw API this replaces took a bare `BigUint` and could only
    /// check in debug builds, so a release build accepted an unencoded or
    /// unreduced value and returned a wrong answer.
    #[must_use]
    pub fn to_residue(&self, value: &BigUint) -> MontgomeryResidue {
        self.to_residue_with(value, &mut MontgomeryScratch::new())
    }

    /// [`Self::to_residue`], reusing a caller-held scratch buffer.
    #[must_use]
    pub fn to_residue_with(
        &self,
        value: &BigUint,
        scratch: &mut MontgomeryScratch,
    ) -> MontgomeryResidue {
        MontgomeryResidue {
            value: self.encode_with_workspace(value, &mut scratch.limbs),
            identity: Arc::clone(&self.identity),
        }
    }

    /// Decode a residue back to an ordinary value in `[0, modulus)`.
    ///
    /// # Errors
    ///
    /// [`ContextMismatch`] if `residue` was produced by a different context.
    pub fn from_residue(&self, residue: &MontgomeryResidue) -> Result<BigUint, ContextMismatch> {
        self.from_residue_with(residue, &mut MontgomeryScratch::new())
    }

    /// [`Self::from_residue`], reusing a caller-held scratch buffer.
    pub fn from_residue_with(
        &self,
        residue: &MontgomeryResidue,
        scratch: &mut MontgomeryScratch,
    ) -> Result<BigUint, ContextMismatch> {
        self.check(residue)?;
        Ok(self.decode_with_workspace(&residue.value, &mut scratch.limbs))
    }

    /// One, encoded in this context's domain.
    #[must_use]
    pub fn one(&self) -> MontgomeryResidue {
        MontgomeryResidue {
            value: self.one_mont.clone(),
            identity: Arc::clone(&self.identity),
        }
    }

    /// `lhs · rhs` in the domain.
    ///
    /// # Errors
    ///
    /// [`ContextMismatch`] if either operand belongs to another context.
    pub fn mul_residue(
        &self,
        lhs: &MontgomeryResidue,
        rhs: &MontgomeryResidue,
    ) -> Result<MontgomeryResidue, ContextMismatch> {
        self.mul_residue_with(lhs, rhs, &mut MontgomeryScratch::new())
    }

    /// [`Self::mul_residue`], reusing a caller-held scratch buffer — the
    /// inner-loop form. The scratch is reused across calls; the returned
    /// residue still owns its own result.
    pub fn mul_residue_with(
        &self,
        lhs: &MontgomeryResidue,
        rhs: &MontgomeryResidue,
        scratch: &mut MontgomeryScratch,
    ) -> Result<MontgomeryResidue, ContextMismatch> {
        self.check(lhs)?;
        self.check(rhs)?;
        Ok(MontgomeryResidue {
            value: BigUint::montgomery_mul_odd_with_workspace(
                &lhs.value,
                &rhs.value,
                &self.modulus,
                self.n0_inv,
                &mut scratch.limbs,
            ),
            identity: Arc::clone(&self.identity),
        })
    }

    /// `value²` in the domain, by the dedicated squaring kernel: each cross
    /// term formed once, `width·(width+1)/2` multiplications against `width²`.
    ///
    /// # Errors
    ///
    /// [`ContextMismatch`] if `value` belongs to another context.
    pub fn square_residue(
        &self,
        value: &MontgomeryResidue,
    ) -> Result<MontgomeryResidue, ContextMismatch> {
        self.square_residue_with(value, &mut MontgomeryScratch::new())
    }

    /// [`Self::square_residue`], reusing a caller-held scratch buffer.
    pub fn square_residue_with(
        &self,
        value: &MontgomeryResidue,
        scratch: &mut MontgomeryScratch,
    ) -> Result<MontgomeryResidue, ContextMismatch> {
        self.check(value)?;
        Ok(MontgomeryResidue {
            value: BigUint::montgomery_sqr_odd_with_workspace(
                &value.value,
                &self.modulus,
                self.n0_inv,
                &mut scratch.limbs,
            ),
            identity: Arc::clone(&self.identity),
        })
    }

    /// `lhs + rhs` in the domain. The encoding is linear —
    /// `x̃ + ỹ ≡ (x + y)·R (mod n)` — so modular addition acts on domain
    /// residues exactly as on ordinary ones.
    ///
    /// # Errors
    ///
    /// [`ContextMismatch`] if either operand belongs to another context.
    pub fn add_residue(
        &self,
        lhs: &MontgomeryResidue,
        rhs: &MontgomeryResidue,
    ) -> Result<MontgomeryResidue, ContextMismatch> {
        self.check(lhs)?;
        self.check(rhs)?;
        Ok(MontgomeryResidue {
            value: BigUint::mod_add(&lhs.value, &rhs.value, &self.modulus),
            identity: Arc::clone(&self.identity),
        })
    }

    /// `lhs − rhs` in the domain.
    ///
    /// # Errors
    ///
    /// [`ContextMismatch`] if either operand belongs to another context.
    pub fn sub_residue(
        &self,
        lhs: &MontgomeryResidue,
        rhs: &MontgomeryResidue,
    ) -> Result<MontgomeryResidue, ContextMismatch> {
        self.check(lhs)?;
        self.check(rhs)?;
        Ok(MontgomeryResidue {
            value: BigUint::mod_sub(&lhs.value, &rhs.value, &self.modulus),
            identity: Arc::clone(&self.identity),
        })
    }

    /// `base^exponent` in the domain, staying encoded throughout.
    ///
    /// # Errors
    ///
    /// [`ContextMismatch`] if `base` belongs to another context.
    pub fn pow_residue(
        &self,
        base: &MontgomeryResidue,
        exponent: &BigUint,
    ) -> Result<MontgomeryResidue, ContextMismatch> {
        self.check(base)?;
        let mut workspace = Vec::new();
        Ok(MontgomeryResidue {
            value: self.pow_ladder_encoded(&base.value, exponent, &mut workspace),
            identity: Arc::clone(&self.identity),
        })
    }

    // ─── Crate-private raw kernels ─────────────────────────────────────
    //
    // The public domain API is the residue type above. These take bare
    // `BigUint`s and carry the reduced-operand contract in a debug
    // assertion, which is only sound because every caller is in this
    // crate and holds the invariant by construction. They are not, and
    // must not become, public.

    pub(crate) fn mul_mont(&self, lhs: &BigUint, rhs: &BigUint) -> BigUint {
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

    pub(crate) fn square_mont(&self, value: &BigUint) -> BigUint {
        debug_assert!(value < &self.modulus, "domain operand arrives reduced");
        let mut workspace = Vec::new();
        BigUint::montgomery_sqr_odd_with_workspace(
            value,
            &self.modulus,
            self.n0_inv,
            &mut workspace,
        )
    }

    pub(crate) fn square_mont_with_workspace(
        &self,
        value: &BigUint,
        workspace: &mut Vec<u64>,
    ) -> BigUint {
        debug_assert!(value < &self.modulus, "domain operand arrives reduced");
        BigUint::montgomery_sqr_odd_with_workspace(value, &self.modulus, self.n0_inv, workspace)
    }

    pub(crate) fn add_mont(&self, lhs: &BigUint, rhs: &BigUint) -> BigUint {
        debug_assert!(
            lhs < &self.modulus && rhs < &self.modulus,
            "domain operands arrive reduced"
        );
        let sum = lhs.add(rhs);
        if sum >= self.modulus {
            sum.sub(&self.modulus)
        } else {
            sum
        }
    }

    pub(crate) fn sub_mont(&self, lhs: &BigUint, rhs: &BigUint) -> BigUint {
        debug_assert!(
            lhs < &self.modulus && rhs < &self.modulus,
            "domain operands arrive reduced"
        );
        if lhs >= rhs {
            lhs.sub(rhs)
        } else {
            self.modulus.add(lhs).sub(rhs)
        }
    }

    pub(crate) fn one_mont(&self) -> &BigUint {
        &self.one_mont
    }

    /// `base^exponent mod modulus` on ordinary values, encoding and decoding
    /// at the boundary.
    #[must_use]
    pub fn pow(&self, base: &BigUint, exponent: &BigUint) -> BigUint {
        let mut workspace = Vec::new();
        let base_mont = self.encode_with_workspace(base, &mut workspace);
        self.pow_encoded_with_workspace(&base_mont, exponent, &mut workspace)
    }
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
