//! Montgomery arithmetic: the domain, its slice kernels, and REDC.
//!
//! Montgomery, *Modular Multiplication Without Trial Division*, Mathematics of
//! Computation 44 (1985); the separated-operand-scanning shape follows Koç,
//! Acar & Kaliski, *Analyzing and Comparing Montgomery Multiplication
//! Algorithms*, IEEE Micro 16 (1996).
//!
//! Split out of `bigint.rs` so the domain's invariant — that a value is either
//! an ordinary residue or an encoded one, never both — is reviewable in one
//! place. The kernels operate on `&[u64]` slices rather than `BigUint` so the
//! exponentiation ladder can reuse one workspace across a whole computation.
//!
//! The test module stays in the parent, where it reaches these and much else.

// `BarrettCtx` is imported for the intra-doc link below, not for code: it is
// the even-modulus counterpart this domain refers the reader to.
#[allow(unused_imports)]
use super::BarrettCtx;
use super::{low_u64, BigUint};
use core::cmp::Ordering;

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
