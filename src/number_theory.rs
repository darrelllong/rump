//! Deterministic number theory over [`BigUint`].
//!
//! Everything here is a pure function of its inputs: gcd and lcm by Euclid,
//! the Jacobi symbol by binary reciprocity, modular exponentiation and
//! inversion, and fixed-base Miller-Rabin. Randomized prime *generation* and
//! adversarially hardened primality testing live with their consumers (the
//! parent cryptography crate), where the entropy source and hash live.

use crate::bigint::{BigInt, BigUint, MontgomeryCtx, Sign};

// ─── Divisibility ──────────────────────────────────────────────────────────────

// Lehmer's gcd machinery (Knuth, TAOCP vol. 2, §4.5.2, Algorithm L).
//
// Classical Euclid does a full multiprecision division per step, and there are
// O(bits) of them. Lehmer's refinement runs Euclid on just the aligned leading
// 64-bit digits, accumulating the 2×2 transform of every step whose quotient
// the leading digits pin down *exactly*, then applies that one transform to the
// full operands with a handful of multiplications. The quotient test — accept
// `q` only when the low and high leading-digit estimates agree — certifies each
// batched quotient equals the true one, so the outcome is bit-for-bit classical
// Euclid, with an order of magnitude fewer big-integer divisions. `gcd`,
// `gcd_extended`, and `mod_inverse` all share this engine.

/// Single-word Euclid, the base case once both operands fit in one limb.
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// The top ≤124 bits of `limbs` starting at bit `shift`, as a non-negative
/// `i128`. Callers pick `shift` so `value >> shift < 2^124`, keeping the result
/// positive and leaving headroom for the transform's corrections. Clone-free:
/// it reads the (up to) three limbs the window spans directly.
fn leading_i128(limbs: &[u64], shift: usize) -> i128 {
    let word = shift / 64;
    let bit = shift % 64;
    let lo = u128::from(limbs.get(word).copied().unwrap_or(0));
    let mid = u128::from(limbs.get(word + 1).copied().unwrap_or(0));
    let window = if bit == 0 {
        lo | (mid << 64)
    } else {
        let hi = u128::from(limbs.get(word + 2).copied().unwrap_or(0));
        (lo >> bit) | (mid << (64 - bit)) | (hi << (128 - bit))
    };
    // The caller's `shift` guarantees `window < 2^124`, so this cast is exact.
    window as i128
}

/// Knuth Algorithm L on the aligned leading digits `u_hat >= v_hat` (each below
/// 2^124): the 2×2 transform `(m00, m01, m10, m11)` collecting every Euclidean
/// step whose quotient the leading digits determine exactly. `m01 == 0` signals
/// that the digits pinned no step, so the caller must take one full division
/// step. Wider leading digits (124 bits, not one 64-bit limb) batch far more
/// steps per call — the difference between a shallow and a useful transform.
///
/// Signs live in the entries, so `m00·u + m01·v` and `m10·u + m11·v` reproduce
/// the post-step operands directly, both provably non-negative.
fn lehmer_transform(
    u_hat: i128,
    v_hat: i128,
    mut quotients: Option<&mut QuotientLog>,
) -> (i128, i128, i128, i128) {
    let (mut m00, mut m01, mut m10, mut m11) = (1i128, 0i128, 0i128, 1i128);
    let (mut u, mut v) = (u_hat, v_hat);
    loop {
        // A full log ends the batch: every logged quotient must correspond to
        // a committed matrix step, so an unloggable step is not taken.
        if let Some(log) = quotients.as_deref_mut() {
            if log.len == log.q_mod_4.len() {
                break;
            }
        }
        // The corrected denominators bracket the true quotient; both must stay
        // positive for the bracket to hold.
        let (denom_low, denom_high) = (v + m10, v + m11);
        if denom_low <= 0 || denom_high <= 0 {
            break;
        }
        let q = (u + m00) / denom_low;
        // A genuine Euclid step has quotient >= 1; accepting only when the low
        // and high estimates agree certifies q is the true quotient.
        if q < 1 || q != (u + m01) / denom_high {
            break;
        }
        // With 124-bit leading digits the accumulated entries can approach the
        // digit size, so these products can genuinely overflow i128 near the
        // end of a long batch. A checked break there is exact: it just ends the
        // batch one step early, and the caller applies what was collected.
        let (Some(q_m10), Some(q_m11), Some(q_v)) =
            (q.checked_mul(m10), q.checked_mul(m11), q.checked_mul(v))
        else {
            break;
        };
        let (Some(new_m10), Some(new_m11)) = (m00.checked_sub(q_m10), m01.checked_sub(q_m11))
        else {
            break;
        };
        (m00, m10) = (m10, new_m10);
        (m01, m11) = (m11, new_m11);
        (u, v) = (v, u - q_v);
        if let Some(log) = quotients.as_deref_mut() {
            log.q_mod_4[log.len] = (q & 3) as u8;
            log.len += 1;
        }
    }
    (m00, m01, m10, m11)
}

/// The applied-quotient log of one Lehmer batch: each entry a quotient's low
/// two bits, in application order — what the Jacobi state replays when the
/// batch commits. Sized for the longest possible batch: all-ones quotients
/// advance the leading digits along the Fibonacci sequence, and F₁₈₀ already
/// exceeds 2¹²⁴, so 184 entries cannot be filled by a 124-bit window.
struct QuotientLog {
    q_mod_4: [u8; 184],
    len: usize,
}

impl QuotientLog {
    fn new() -> Self {
        Self {
            q_mod_4: [0; 184],
            len: 0,
        }
    }
}

/// Replay one committed batch through the symbol state. The batch's steps are
/// swapping remainder steps; in fixed slots the first step reduces whichever
/// slot held the larger value, and the reduced slot alternates thereafter.
fn replay_batch(state: &mut JacobiState, first_reduced_is_a: bool, log: &QuotientLog) {
    let d0 = u8::from(first_reduced_is_a);
    for i in 0..log.len {
        let d = if i.is_multiple_of(2) { d0 } else { 1 - d0 };
        state.update(d, log.q_mod_4[i]);
    }
}

/// Aligned 124-bit leading digits of `a >= b`, both non-zero: the same window
/// (top of the larger) taken from each, ready for [`lehmer_transform`].
fn leading_pair(a: &BigUint, b: &BigUint) -> (i128, i128) {
    let shift = a.bits().saturating_sub(124);
    (
        leading_i128(a.limbs(), shift),
        leading_i128(b.limbs(), shift),
    )
}

// The Lehmer transform is applied to the operands (and, for the extended
// variants, to the Bézout cofactors) as `c0·x0 + c1·x1` with two-word signed
// coefficients. Going through `mul_ref`/`add_ref` would allocate several
// temporaries per application; the transform runs tens of times per gcd, so
// these fused limb-level routines — accumulate `|c|·x` straight into a positive
// or negative bucket by sign, then take the difference once — are what make the
// batching actually pay off.

/// `out += (clo, chi)·x` in place, where `(clo, chi)` is a two-word magnitude
/// and `out` is little-endian with room for the carries (`x.len() + 2` limbs
/// past the write origin).
fn mul_add_2word(out: &mut [u64], x: &[u64], clo: u64, chi: u64) {
    if clo != 0 {
        let mut carry = 0u128;
        for (i, &xi) in x.iter().enumerate() {
            let acc = u128::from(out[i]) + u128::from(xi) * u128::from(clo) + carry;
            out[i] = acc as u64;
            carry = acc >> 64;
        }
        let mut idx = x.len();
        while carry != 0 {
            let acc = u128::from(out[idx]) + carry;
            out[idx] = acc as u64;
            carry = acc >> 64;
            idx += 1;
        }
    }
    if chi != 0 {
        // The high word contributes one limb further up.
        let mut carry = 0u128;
        for (i, &xi) in x.iter().enumerate() {
            let acc = u128::from(out[i + 1]) + u128::from(xi) * u128::from(chi) + carry;
            out[i + 1] = acc as u64;
            carry = acc >> 64;
        }
        let mut idx = x.len() + 1;
        while carry != 0 {
            let acc = u128::from(out[idx]) + carry;
            out[idx] = acc as u64;
            carry = acc >> 64;
            idx += 1;
        }
    }
}

/// Route `|coefficient|·x` into the positive or negative accumulator by the
/// term's overall sign (`sign(coefficient) · sign(x)`).
fn route_term(pos: &mut [u64], neg: &mut [u64], coefficient: i128, x_negative: bool, x: &[u64]) {
    if coefficient == 0 || x.is_empty() {
        return;
    }
    let magnitude = coefficient.unsigned_abs();
    let (clo, chi) = (magnitude as u64, (magnitude >> 64) as u64);
    let term_negative = (coefficient < 0) ^ x_negative;
    let target = if term_negative { neg } else { pos };
    mul_add_2word(target, x, clo, chi);
}

/// Compare equal-length little-endian limb slices.
fn cmp_slices(a: &[u64], b: &[u64]) -> core::cmp::Ordering {
    for i in (0..a.len()).rev() {
        match a[i].cmp(&b[i]) {
            core::cmp::Ordering::Equal => {}
            other => return other,
        }
    }
    core::cmp::Ordering::Equal
}

/// `a - b` for equal-length little-endian slices with `a >= b`.
fn sub_slices(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; a.len()];
    let mut borrow = 0i128;
    for i in 0..a.len() {
        let diff = i128::from(a[i]) - i128::from(b[i]) - borrow;
        if diff < 0 {
            out[i] = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            out[i] = diff as u64;
            borrow = 0;
        }
    }
    debug_assert!(borrow == 0, "sub_slices requires a >= b");
    out
}

/// `c0·x0 + c1·x1` as a `BigUint`, for a result the Lehmer transform guarantees
/// is non-negative (the remainder sequence).
fn combine_unsigned(c0: i128, x0: &BigUint, c1: i128, x1: &BigUint) -> BigUint {
    let width = x0.limbs().len().max(x1.limbs().len()) + 3;
    let mut pos = vec![0u64; width];
    let mut neg = vec![0u64; width];
    route_term(&mut pos, &mut neg, c0, false, x0.limbs());
    route_term(&mut pos, &mut neg, c1, false, x1.limbs());
    debug_assert!(
        cmp_slices(&pos, &neg) != core::cmp::Ordering::Less,
        "Lehmer transform keeps the remainder sequence non-negative"
    );
    BigUint::from_limbs(sub_slices(&pos, &neg))
}

/// `c0·x0 + c1·x1` as a signed `BigInt`, for the Bézout cofactor sequences.
fn combine_signed(c0: i128, x0: &BigInt, c1: i128, x1: &BigInt) -> BigInt {
    let width = x0
        .magnitude()
        .limbs()
        .len()
        .max(x1.magnitude().limbs().len())
        + 3;
    let mut pos = vec![0u64; width];
    let mut neg = vec![0u64; width];
    route_term(
        &mut pos,
        &mut neg,
        c0,
        x0.sign() == Sign::Negative,
        x0.magnitude().limbs(),
    );
    route_term(
        &mut pos,
        &mut neg,
        c1,
        x1.sign() == Sign::Negative,
        x1.magnitude().limbs(),
    );
    match cmp_slices(&pos, &neg) {
        core::cmp::Ordering::Less => {
            BigInt::from_parts(Sign::Negative, BigUint::from_limbs(sub_slices(&neg, &pos)))
        }
        _ => BigInt::from_parts(Sign::Positive, BigUint::from_limbs(sub_slices(&pos, &neg))),
    }
}

// ─── Jacobi state machine ───────────────────────────────────────────────────
//
// The Jacobi symbol can ride the same left-to-right quotient sequence as gcd:
// Schönhage's identities give the symbol's change across one Euclidean step
// r0 = q·r1 + r2 as a sign that depends only on r0 and r1 modulo 4 and on
// q modulo 4 — never on the operands' high bits. Two rules cover everything:
// for odd r0, r1, reciprocity charges a sign exactly when both are ≡ 3
// (mod 4); for even r1 (the remainders of a quotient sequence are not kept
// odd), the accumulated sign across the step is (−1)^((q(r0−1)/2 + r0(q−1)/2))
// when r1 ≡ 2 (mod 4), and nothing otherwise. Möller's formulation carries
// the whole computation as a five-bit state — the sign bit plus thirteen
// reachable classes of (a mod 4, b mod 4, which side is the denominator) —
// advanced by one table lookup per quotient. Even intermediates need no
// two-stripping: the state tracks their low bits symbolically, and a
// remainder sequence beside an odd operand cannot end on an even value.
//
// The design and the table generator are Niels Möller's, as shipped in GMP
// (`mpn_jacobi_n`, `gen-jacobitab.c`); the identities are Schönhage's. The
// published subquadratic-Jacobi reference is Brent and Zimmermann, *An
// O(M(n) log n) algorithm for the Jacobi symbol*, ANTS-IX, 2010 — their
// algorithm takes the binary (2-adic) route; this implementation takes the
// left-to-right route their §1 attributes to Möller, which composes with the
// Half-GCD machinery below.

/// The thirteen reachable `(a mod 4, b mod 4)` classes. At least one side of
/// the pair is always odd; the denominator flag `d` is ambiguous only for
/// `(3, 3)`, which therefore appears twice — index 7 with `d = 1`, index 12
/// with `d = 0`. For indices 0–7 the parity of `b` implies `d = 1`; for
/// 8–11, `d = 0`.
const JACOBI_DECODE: [(u8, u8); 13] = [
    (0, 1),
    (0, 3),
    (1, 1),
    (1, 3),
    (2, 1),
    (2, 3),
    (3, 1),
    (3, 3), // d = 1
    (1, 0),
    (1, 2),
    (3, 0),
    (3, 2),
    (3, 3), // d = 0
];

/// Index of `(a, b)` with denominator flag `d` in [`JACOBI_DECODE`].
const fn jacobi_encode(a: u8, b: u8, d: u8) -> u8 {
    if a == 3 && b == 3 {
        return if d == 1 { 7 } else { 12 };
    }
    let mut i = 0;
    while i < 12 {
        let (da, db) = JACOBI_DECODE[i];
        if da == a && db == b {
            return i as u8;
        }
        i += 1;
    }
    panic!("unreachable (a, b) class: one side must be odd");
}

/// Build the 208-entry transition table from Schönhage's rules, at compile
/// time. Entry `(state << 3) | (d << 2) | (q mod 4)` holds the successor of
/// `state` after one Euclidean step with quotient `q` reducing side `d`
/// (`d = 1`: `a ← a − q·b`; `d = 0`: `b ← b − q·a`). A state is
/// `(class << 1) | e`, the result being `(−1)^e`.
const fn build_jacobi_table() -> [u8; 208] {
    let mut table = [0u8; 208];
    let mut idx = 0;
    while idx < 208 {
        let q = (idx & 3) as u8;
        let d = ((idx >> 2) & 1) as u8;
        let state = (idx >> 3) as u8;
        let mut e = state & 1;
        let class = (state >> 1) as usize;
        let (mut a, mut b) = JACOBI_DECODE[class];
        // d is determinate only for the two (3, 3) classes; elsewhere the
        // reciprocity charge below cannot fire, so any value serves.
        let d_old = if class == 7 { 1 } else { 0 };

        // Reciprocity: exchanging the denominator costs a sign exactly when
        // both sides are ≡ 3 (mod 4).
        if d != d_old && a == 3 && b == 3 {
            e ^= 1;
        }
        // The even-denominator rule, and the symbolic low-bit recurrence for
        // the reduced side.
        if d == 1 {
            if b == 2 {
                e ^= (q & (a >> 1)) ^ (q >> 1);
            }
            a = a.wrapping_sub(q.wrapping_mul(b)) & 3;
        } else {
            if a == 2 {
                e ^= (q & (b >> 1)) ^ (q >> 1);
            }
            b = b.wrapping_sub(q.wrapping_mul(a)) & 3;
        }
        table[idx] = (jacobi_encode(a, b, d) << 1) | e;
        idx += 1;
    }
    table
}

/// The transition table, derived from the rules above by the compiler.
static JACOBI_TABLE: [u8; 208] = build_jacobi_table();

/// The five-bit Jacobi state threaded through a quotient sequence.
#[derive(Clone, Copy)]
struct JacobiState(u8);

impl JacobiState {
    /// Initial state from the operands' low two bits; `b` must be odd. The
    /// large operands' low bits are never consulted again after this.
    fn new(a_low: u8, b_low: u8) -> Self {
        debug_assert!(b_low & 1 == 1, "the initial denominator must be odd");
        Self(((a_low & 3) << 2) + (b_low & 2))
    }

    /// Advance across one applied Euclidean step: side `d` was reduced by
    /// `q` times the other (`d = 1`: `a ← a − q·b`). `q` is the quotient as
    /// applied — after any size-guard back-off — reduced mod 4.
    fn update(&mut self, d: u8, q_mod_4: u8) {
        debug_assert!(self.0 < 26 && d < 2 && q_mod_4 < 4);
        self.0 = JACOBI_TABLE[((self.0 as usize) << 3) | ((d as usize) << 2) | q_mod_4 as usize];
    }

    /// Read out the symbol once the pair has reached `(1, 0)` or `(0, 1)`.
    fn finish(&self) -> i8 {
        1 - 2 * (self.0 & 1) as i8
    }
}

// ─── Half-GCD machinery ─────────────────────────────────────────────────────
//
// Euclid's quotient sequence is the continued-fraction expansion of a/b, and a
// ratio is pinned by its leading bits: the top halves of a and b already
// determine the first half of the quotient sequence. Half-GCD lives on that
// fact. It computes, from the top halves alone, the 2×2 matrix a half-run of
// Euclid would apply, applies it to the full operands in a few multiplications,
// and finds that matrix by recursing on the halves themselves — so reduction
// costs O(M(n)·log n), where Lehmer, re-reading the leading digits after every
// 124-bit batch, stays O(n²).
//
// The treachery is at the boundary. "These quotients are what full-width
// Euclid would do" holds for a half-run's beginning but can fail for its last
// step or two — the discarded low bits can tip a quotient — so every reduction
// here is size-guarded: it stops strictly above the certification boundary,
// and short runs of guarded full-width steps repair each splice. Möller
// exhibits a concrete pair computed wrong without that repair (§6.3); his
// Figure 4 is the algorithm implemented here, and GMP's mpn_hgcd is the same
// design.

/// The 2×2 integer matrix of a partial Euclidean reduction, acting on columns:
/// each row combines `(a, b)` into a later remainder of the sequence. Later
/// remainders are alternating-sign combinations of earlier ones, so the
/// entries are signed even though every remainder they produce is not.
struct Mat2 {
    m00: BigInt,
    m01: BigInt,
    m10: BigInt,
    m11: BigInt,
}

impl Mat2 {
    fn identity() -> Self {
        let one = || BigInt::from_biguint(BigUint::one());
        Self {
            m00: one(),
            m01: BigInt::zero(),
            m10: BigInt::zero(),
            m11: one(),
        }
    }

    /// The matrix product `self · other`.
    fn compose(&self, other: &Self) -> Self {
        Self {
            m00: self
                .m00
                .mul_ref(&other.m00)
                .add_ref(&self.m01.mul_ref(&other.m10)),
            m01: self
                .m00
                .mul_ref(&other.m01)
                .add_ref(&self.m01.mul_ref(&other.m11)),
            m10: self
                .m10
                .mul_ref(&other.m00)
                .add_ref(&self.m11.mul_ref(&other.m10)),
            m11: self
                .m10
                .mul_ref(&other.m01)
                .add_ref(&self.m11.mul_ref(&other.m11)),
        }
    }

    /// Apply to a non-negative column `(a, b)`, returning `(m00·a + m01·b,
    /// m10·a + m11·b)` — both non-negative when the transform is valid for the
    /// pair (used by the invariant tests; the hot path is [`hgcd_adjust`]).
    #[cfg(test)]
    fn apply(&self, a: &BigUint, b: &BigUint) -> (BigUint, BigUint) {
        let combine = |r: &BigInt, s: &BigInt| {
            let sum = r.mul_biguint_ref(a).add_ref(&s.mul_biguint_ref(b));
            debug_assert!(
                sum.sign() != Sign::Negative,
                "Half-GCD transform keeps the pair non-negative"
            );
            sum.magnitude().clone()
        };
        (combine(&self.m00, &self.m01), combine(&self.m10, &self.m11))
    }

    /// Fold in the non-swapping Euclid step `a ← a − q·b` (used when `a > b`):
    /// left-multiply the transform by `[[1, −q], [0, 1]]`, i.e. `row0 −= q·row1`.
    fn reduce_top(&self, q: &BigUint) -> Self {
        Self {
            m00: self.m00.sub_ref(&self.m10.mul_biguint_ref(q)),
            m01: self.m01.sub_ref(&self.m11.mul_biguint_ref(q)),
            m10: self.m10.clone(),
            m11: self.m11.clone(),
        }
    }

    /// Fold in the non-swapping Euclid step `b ← b − q·a` (used when `b > a`):
    /// left-multiply by `[[1, 0], [−q, 1]]`, i.e. `row1 −= q·row0`.
    fn reduce_bottom(&self, q: &BigUint) -> Self {
        Self {
            m00: self.m00.clone(),
            m01: self.m01.clone(),
            m10: self.m10.sub_ref(&self.m00.mul_biguint_ref(q)),
            m11: self.m11.sub_ref(&self.m01.mul_biguint_ref(q)),
        }
    }

    /// Fold in the swapping Euclid step `(a, b) ← (b, a − q·b)` — the shape
    /// the gcd drivers use, which keeps the pair ordered by moving the old
    /// smaller element on top: left-multiply by `[[0, 1], [1, −q]]`.
    fn step_swap(&self, q: &BigUint) -> Self {
        Self {
            m00: self.m10.clone(),
            m01: self.m11.clone(),
            m10: self.m00.sub_ref(&self.m10.mul_biguint_ref(q)),
            m11: self.m01.sub_ref(&self.m11.mul_biguint_ref(q)),
        }
    }

    /// Exchange the rows — the transform-side mirror of swapping the pair.
    fn swap_rows(&mut self) {
        core::mem::swap(&mut self.m00, &mut self.m10);
        core::mem::swap(&mut self.m01, &mut self.m11);
    }
}

/// `x >> bits`, out of place.
fn shr(x: &BigUint, bits: usize) -> BigUint {
    let mut y = x.clone();
    y.shr_bits(bits);
    y
}

/// `#(a, b)` — bit-size of the larger element.
fn pair_size(a: &BigUint, b: &BigUint) -> usize {
    a.bits().max(b.bits())
}

/// Möller's underlined `#(a, b)` — bit-size of the *smaller* element. The
/// distinction matters: hgcd's precondition and both of its recursion guards
/// are conditions on the smaller element, and reading them as the larger is
/// exactly the mistake that produces transforms invalid for the full operands.
fn pair_min_size(a: &BigUint, b: &BigUint) -> usize {
    a.bits().min(b.bits())
}

/// Splice the low bits back after a recursion on the top halves. `T` is
/// linear, so applying it to `a = 2^p·(a≫p) + (a mod 2^p)` splits into the
/// part the sub-call already computed and the part it never saw:
/// `T·(a, b) = 2^p·(α, β) + T·(a mod 2^p, b mod 2^p)` — Möller's Equation 4.
/// Using the right-hand side costs a matrix–vector product on `p`-bit pieces
/// instead of full-width ones, which is most of the reason the recursion is
/// cheap.
///
/// Each spliced total is positive even though the matrix term alone need not
/// be: the sub-call left `α, β` above its boundary, so the shifted term
/// exceeds `2^{p+s}`, while the correction is capped below it by the entry
/// bounds of `T` (his Lemma 6, the size analysis behind this function). Hence
/// signed arithmetic inside, unsigned out.
fn hgcd_adjust(
    t: &Mat2,
    alpha: &BigUint,
    beta: &BigUint,
    a: &BigUint,
    b: &BigUint,
    p: usize,
) -> (BigUint, BigUint) {
    let a_low = a.low_bits(p);
    let b_low = b.low_bits(p);
    let attach = |high: &BigUint, r: &BigInt, s: &BigInt| {
        let mut shifted = high.clone();
        shifted.shl_bits(p);
        let sum = BigInt::from_biguint(shifted)
            .add_ref(&r.mul_biguint_ref(&a_low))
            .add_ref(&s.mul_biguint_ref(&b_low));
        debug_assert!(
            sum.sign() == Sign::Positive,
            "adjusted pair stays positive (Möller Lemma 6)"
        );
        sum.magnitude().clone()
    };
    (attach(alpha, &t.m00, &t.m01), attach(beta, &t.m10, &t.m11))
}

/// `#(a − b)` — bit-size of the absolute difference.
fn abs_diff_bits(a: &BigUint, b: &BigUint) -> usize {
    if a >= b {
        a.sub_ref(b).bits()
    } else {
        b.sub_ref(a).bits()
    }
}

/// One size-guarded division step — Möller's `sdiv`: reduce the larger element
/// by the largest multiple of the smaller that leaves the remainder strictly
/// above `s` bits, folding the step into `t`. Returns `false` if no quotient
/// can respect the guard (only reachable when the caller's invariants are
/// already violated; the loops treat it as "stop here" rather than corrupt
/// the transform).
///
/// The guard is what earns the right to work on high bits alone. Above the
/// boundary these quotients are exactly what full-width Euclid would produce;
/// a remainder allowed to dip to `s` bits or below would depend on low bits
/// the recursion never saw, and every conclusion drawn after it would be
/// unsound. So the step refuses — reduction stalls at the boundary by design,
/// and the caller decides what happens next.
fn sdiv_step(
    a: &mut BigUint,
    b: &mut BigUint,
    t: &mut Mat2,
    s: usize,
    state: Option<&mut JacobiState>,
) -> bool {
    let a_is_larger = *a >= *b;
    let (hi, lo) = if a_is_larger {
        (a.clone(), b.clone())
    } else {
        (b.clone(), a.clone())
    };
    if lo.is_zero() {
        return false;
    }
    // The largest q with `hi − q·lo ≥ 2^s`, i.e. `q = ⌊(hi − 2^s)/lo⌋`. This is
    // the ordinary Euclidean quotient whenever `hi mod lo` already clears the
    // boundary, and backs off by exactly as much as needed otherwise — Möller's
    // sdiv, which never lets the remainder fall to or below `s` bits.
    let threshold = {
        let mut t = BigUint::zero();
        t.set_bit(s);
        t
    };
    if hi < threshold {
        return false; // hi already sits at or below the boundary
    }
    let q = hi.sub_ref(&threshold).div_rem(&lo).0;
    if q.is_zero() {
        return false; // reducing even once would cross the boundary
    }
    let r = hi.sub_ref(&q.mul_ref(&lo));
    if let Some(st) = state {
        // The as-applied quotient, after any size-guard back-off.
        st.update(u8::from(a_is_larger), (q.limbs()[0] & 3) as u8);
    }
    if a_is_larger {
        *a = r;
        *t = t.reduce_top(&q);
    } else {
        *b = r;
        *t = t.reduce_bottom(&q);
    }
    true
}

/// Below this many limbs [`hgcd`] stops recursing and runs [`hgcd_base`]
/// directly — the analogue of GMP's `HGCD_THRESHOLD`. Tuned empirically.
const HGCD_BASE_LIMBS: usize = 96;

/// [`hgcd`]'s workhorse below the recursion threshold — the role GMP's
/// `hgcd2` loop plays. Reduction runs in two regimes: far from the boundary,
/// whole Lehmer batches (the leading 124 bits certify a run of ~35 quotients,
/// replayed against the full operands as one matrix application); near it,
/// single guarded divisions, because a batch commits to its whole run and
/// cannot stop at the boundary mid-way.
///
/// What licenses running a batch *unguarded*: one batch advances the remainder
/// sequence by at most its digit window, so started with both elements above
/// `s + LEHMER_MARGIN` it physically cannot drop either to `s` bits — and
/// everywhere above the boundary, certified quotients are Euclid's, so the
/// batch is the same reduction the guarded steps would have taken.
fn hgcd_base(
    a: &BigUint,
    b: &BigUint,
    s: usize,
    mut state: Option<&mut JacobiState>,
) -> (Mat2, BigUint, BigUint) {
    /// A batch moves each element by at most the 124-bit digit window; the
    /// slack above that covers the window's own imprecision at its last step.
    const LEHMER_MARGIN: usize = 130;

    let mut aa = a.clone();
    let mut bb = b.clone();
    let mut t = Mat2::identity();

    while abs_diff_bits(&aa, &bb) > s {
        if pair_min_size(&aa, &bb) > s + LEHMER_MARGIN {
            let a_is_larger = aa >= bb;
            let (hi, lo) = if a_is_larger { (&aa, &bb) } else { (&bb, &aa) };
            let (u_hat, v_hat) = leading_pair(hi, lo);
            let mut log = QuotientLog::new();
            let (m00, m01, m10, m11) = lehmer_transform(u_hat, v_hat, Some(&mut log));
            // m01 == 0 means the digits pinned no quotient — the pair is too
            // lopsided for its leading windows to overlap — and division is
            // the only way forward.
            if m01 != 0 {
                let next_hi = combine_unsigned(m00, hi, m01, lo);
                let next_lo = combine_unsigned(m10, hi, m11, lo);
                // The batch is linear, so it composes with `t` by acting on
                // t's rows exactly as it acts on the values — with the same
                // care for which row currently plays hi and which lo.
                let (row_hi, row_lo) = if a_is_larger {
                    ((&t.m00, &t.m01), (&t.m10, &t.m11))
                } else {
                    ((&t.m10, &t.m11), (&t.m00, &t.m01))
                };
                let new_hi_row = (
                    combine_signed(m00, row_hi.0, m01, row_lo.0),
                    combine_signed(m00, row_hi.1, m01, row_lo.1),
                );
                let new_lo_row = (
                    combine_signed(m10, row_hi.0, m11, row_lo.0),
                    combine_signed(m10, row_hi.1, m11, row_lo.1),
                );
                if let Some(st) = state.as_deref_mut() {
                    replay_batch(st, a_is_larger, &log);
                }
                // A batch of k swapping steps leaves the slot that held the
                // larger input holding the even-indexed member of the final
                // remainder pair. The transform rows follow the same
                // placement, so the matrix, the values, and the symbol
                // state's fixed slots stay aligned.
                let even_steps = log.len.is_multiple_of(2);
                let (hi_slot_val, lo_slot_val) = if even_steps {
                    (next_hi, next_lo)
                } else {
                    (next_lo, next_hi)
                };
                let (hi_slot_row, lo_slot_row) = if even_steps {
                    (new_hi_row, new_lo_row)
                } else {
                    (new_lo_row, new_hi_row)
                };
                if a_is_larger {
                    (t.m00, t.m01) = hi_slot_row;
                    (t.m10, t.m11) = lo_slot_row;
                    aa = hi_slot_val;
                    bb = lo_slot_val;
                } else {
                    (t.m10, t.m11) = hi_slot_row;
                    (t.m00, t.m01) = lo_slot_row;
                    bb = hi_slot_val;
                    aa = lo_slot_val;
                }
                continue;
            }
        }
        // Within a batch-width of the boundary a batch could sail past it, so
        // the last stretch goes one guarded division at a time.
        if !sdiv_step(&mut aa, &mut bb, &mut t, s, state.as_deref_mut()) {
            break;
        }
    }

    (t, aa, bb)
}

/// Half-GCD: run Euclid's reduction of `(a, b)` halfway — to the boundary
/// `s = ⌊N/2⌋ + 1`, `N` the larger element's bit-size — returning the reduced
/// pair `(α, β) = T·(a, b)` and the transform `T` that got there.
///
/// The contract, from Möller's Lemma 5: the caller supplies both elements
/// above `s` bits, and gets back both elements still above `s` bits with the
/// *difference* at or below — the last moment reduction is still certifiable
/// from the bits above `s`. Asking for one more step is asking for an answer
/// the high bits do not contain; that boundary discipline, not the recursion,
/// is what makes the function composable, because a caller holding `2p + n`
/// bits can run it on the top `n` and trust the result for the whole number.
///
/// The recursion is that composition applied to itself: one sub-call on the
/// top halves takes the pair to ~3N/4 bits, a second on the top halves of the
/// remainder reaches ~N/2. Each splice is followed by a short run of guarded
/// full-width division steps (at most four, Lemma 7) because a sub-call's
/// final quotient can be wrong for the full operands — the low bits it never
/// saw can tip it — and §6.3 constructs a pair computed wrongly when this
/// repair is skipped. Both recursion guards test the *smaller* element: the
/// sub-call's contract needs both of its inputs above its own boundary, and
/// the guard is the parent-level condition that delivers exactly that
/// (Lemma 7). Read the guards against the larger element and the recursion
/// runs on pairs it has no contract for.
///
/// `T` is `M⁻¹` in Möller's notation: unimodular with rows of alternating
/// sign. det is ±1 rather than his +1 — the base case batches swapping Euclid
/// steps, det −1 apiece — which nothing downstream depends on. At full
/// reduction (`s = 0`) a row of `T` is a Bézout cofactor pair.
///
/// Reference: Niels Möller, *On Schönhage's algorithm and subquadratic
/// integer gcd computation*, Math. Comp. 77 (2008), 589–607, Figure 4 — the
/// algorithm behind GMP's `mpn_hgcd`. O(M(n)·log n) with fast multiplication
/// carrying the matrix work.
fn hgcd(a: &BigUint, b: &BigUint, mut state: Option<&mut JacobiState>) -> (Mat2, BigUint, BigUint) {
    let n = pair_size(a, b);
    let s = n / 2 + 1; // Möller's S = ⌊N/2⌋ + 1
    debug_assert!(
        pair_min_size(a, b) > s,
        "hgcd precondition: both elements above the boundary"
    );
    let mut aa = a.clone();
    let mut bb = b.clone();
    let mut t = Mat2::identity();

    // Already straddling the boundary: nothing to do. (Möller's sgcd threads
    // this escape through its Step 5 → Step 8 goto; without it a close pair
    // that is still large recurses on nearly its own size, and the recursion's
    // cost estimate collapses.)
    if abs_diff_bits(&aa, &bb) <= s {
        return (t, aa, bb);
    }

    // Below this size the recursion's matrix multiplications cost more than
    // simply reducing the pair with batched Lehmer steps, so the recursion
    // bottoms out here rather than at trivial sizes.
    if n <= HGCD_BASE_LIMBS * 64 {
        return hgcd_base(&aa, &bb, s, state);
    }

    // First recursive call (Step 2): only when the smaller element clears
    // ⌊3N/4⌋ + 2, which puts both top halves above the sub-call's boundary.
    if pair_min_size(&aa, &bb) > 3 * n / 4 + 2 {
        let p1 = n / 2;
        let (t1, alpha, beta) = hgcd(&shr(&aa, p1), &shr(&bb, p1), state.as_deref_mut());
        let (na, nb) = hgcd_adjust(&t1, &alpha, &beta, &aa, &bb, p1);
        aa = na;
        bb = nb;
        t = t1;
    }

    // Repair the splice (Step 9): the sub-call certified its quotients only
    // against the bits it saw, so its last step or two may be wrong for the
    // full operands. At most four full-width guarded steps make it right.
    while pair_size(&aa, &bb) > 3 * n / 4 + 1 && abs_diff_bits(&aa, &bb) > s {
        if !sdiv_step(&mut aa, &mut bb, &mut t, s, state.as_deref_mut()) {
            break;
        }
    }

    // Second recursive call (Step 12), again guarded on the smaller element —
    // and skipped when the backup steps already met the target (the sgcd
    // Step 5 → Step 8 escape).
    if pair_min_size(&aa, &bb) > s + 2 && abs_diff_bits(&aa, &bb) > s {
        let n2 = pair_size(&aa, &bb);
        let p2 = 2 * s + 1 - n2; // 2S − N2 + 1, positive since N2 ≤ 2S − 1
        let (t2, alpha, beta) = hgcd(&shr(&aa, p2), &shr(&bb, p2), state.as_deref_mut());
        let (na, nb) = hgcd_adjust(&t2, &alpha, &beta, &aa, &bb, p2);
        aa = na;
        bb = nb;
        t = t2.compose(&t); // T ← T′·T: the second reduction acts after the first
    }

    // Repair the second splice and land on the target (Step 20).
    while abs_diff_bits(&aa, &bb) > s {
        if !sdiv_step(&mut aa, &mut bb, &mut t, s, state.as_deref_mut()) {
            break;
        }
    }

    debug_assert!(
        pair_min_size(&aa, &bb) > s && abs_diff_bits(&aa, &bb) <= s,
        "hgcd postcondition: pair straddles the boundary"
    );
    (t, aa, bb)
}

/// Below this many limbs in the smaller operand, gcd runs on Lehmer; at or
/// above it, on the Half-GCD driver — and the driver hands its own tail back
/// to Lehmer at the same line, since below the crossover every round is
/// better spent there. Measured on M4: a tie at 2048 limbs, Half-GCD ahead
/// 1.3× at 4096 and 1.7× at 8192, the gap widening as the subquadratic curve
/// pulls away (PERFORMANCE.md, "GCD at scale"). Correctness does not depend
/// on the value — the suite validates with this set to 2, forcing every size
/// through the recursion.
const HGCD_THRESHOLD_LIMBS: usize = 2048;

/// Greatest common divisor through [`hgcd`]. Each round halves the pair, so
/// the rounds' costs form a geometric series and the whole gcd runs in
/// O(M(n)·log n) — about twice the cost of the first hgcd call.
///
/// hgcd's contract wants both elements past the halfway boundary and a gap
/// wider than it; a pair can leave a round (or arrive) violating either. One
/// ordinary division step is the repair for both, and a sharp one: an
/// unbalanced pair collapses to the smaller element's size, and a close pair
/// drops to its difference — which the previous round just certified small.
/// Below the crossover the tail goes to Lehmer, whose constant wins there.
fn gcd_via_hgcd(a: &BigUint, b: &BigUint) -> BigUint {
    let (mut aa, mut bb) = if a >= b {
        (a.clone(), b.clone())
    } else {
        (b.clone(), a.clone())
    };
    loop {
        if bb.is_zero() {
            return aa;
        }
        if bb.limbs().len() < HGCD_THRESHOLD_LIMBS {
            return gcd_lehmer(&aa, &bb);
        }
        let s = aa.bits() / 2 + 1;
        // hgcd needs the smaller element above the boundary and a gap wide
        // enough to close; otherwise one division step makes sharp progress
        // (an unbalanced pair collapses, a close pair drops to its difference).
        if bb.bits() <= s || abs_diff_bits(&aa, &bb) <= s {
            let r = aa.modulo(&bb);
            aa = core::mem::replace(&mut bb, r);
            continue;
        }
        let (_t, ra, rb) = hgcd(&aa, &bb, None);
        (aa, bb) = if ra >= rb { (ra, rb) } else { (rb, ra) };
    }
}

/// Below this many limbs in the smaller operand the Bézout functions stay on
/// Lehmer; at or above it they ride the Half-GCD driver. The crossover sits
/// *below* plain gcd's: Lehmer's extended form carries full-width signed
/// cofactors through every batch, where the driver folds all cofactor work
/// into one matrix accumulation per round — measured on M4, the driver ties
/// Lehmer near 448 limbs and is 2× ahead by 16384 (PERFORMANCE.md).
/// Correctness does not depend on the value — the suite validates with this
/// set to 2, forcing every size through the driver and its canonicalization.
const HGCD_EXT_THRESHOLD_LIMBS: usize = 512;

/// Extended-gcd counterpart of [`gcd_via_hgcd`]: the same rounds, with the
/// reduce transform accumulated instead of discarded. The loop maintains
/// `(aa, bb) = ACC·(a, b)`; when the pair drops below the crossover the
/// Lehmer engine finishes from `(aa, bb)`, and its Bézout pair converts to
/// one for the original operands through the transform —
/// `g = s'·aa + t'·bb = (s'·ACC₀₀ + t'·ACC₁₀)·a + (s'·ACC₀₁ + t'·ACC₁₁)·b`.
///
/// The returned pair satisfies the Bézout identity but need not be the
/// classical pair: hgcd's size-guarded quotients can leave the true
/// continued-fraction path near each boundary, shifting the cofactors by a
/// multiple of `(b/g, −a/g)`. Callers that promise the classical pair
/// normalize with [`canonicalize_bezout`].
fn gcd_extended_via_hgcd(a: &BigUint, b: &BigUint) -> (BigUint, BigInt, BigInt) {
    let mut acc = Mat2::identity();
    let (mut aa, mut bb) = if a >= b {
        (a.clone(), b.clone())
    } else {
        acc.swap_rows();
        (b.clone(), a.clone())
    };
    loop {
        if bb.is_zero() {
            // aa = ACC₀₀·a + ACC₀₁·b: the top row is already the answer.
            return (aa, acc.m00, acc.m01);
        }
        if bb.limbs().len() < HGCD_EXT_THRESHOLD_LIMBS {
            let (g, s_top, t_top) = gcd_extended_lehmer(&aa, &bb);
            let s = s_top.mul_ref(&acc.m00).add_ref(&t_top.mul_ref(&acc.m10));
            let t = s_top.mul_ref(&acc.m01).add_ref(&t_top.mul_ref(&acc.m11));
            return (g, s, t);
        }
        let boundary = aa.bits() / 2 + 1;
        if bb.bits() <= boundary || abs_diff_bits(&aa, &bb) <= boundary {
            let (q, r) = aa.div_rem(&bb);
            acc = acc.step_swap(&q);
            aa = core::mem::replace(&mut bb, r);
            continue;
        }
        let (round, ra, rb) = hgcd(&aa, &bb, None);
        acc = round.compose(&acc);
        if ra >= rb {
            (aa, bb) = (ra, rb);
        } else {
            acc.swap_rows();
            (aa, bb) = (rb, ra);
        }
    }
}

/// Reduce a valid Bézout pair for `(a, b)` to the classical one. The cofactor
/// of `a` is unique modulo `b/g` — any two valid pairs differ by a multiple
/// of `(b/g, −a/g)` — and classical extended Euclid returns the
/// representative of least absolute value. Select it, then recover the
/// second cofactor exactly from the identity `t = (g − s·a)/b`.
///
/// Precondition: `a, b > 0` (the zero-operand cases return directly from the
/// driver in classical form).
fn canonicalize_bezout(a: &BigUint, b: &BigUint, g: &BigUint, s: &BigInt) -> (BigInt, BigInt) {
    let modulus = b.div_rem(g).0; // b/g, exact by definition of g
    if modulus.is_one() {
        // b divides a; classical Euclid ends in one step with (g, 0, 1).
        return (BigInt::zero(), BigInt::from_biguint(BigUint::one()));
    }
    let residue = s.modulo_positive(&modulus); // s mod (b/g), in [0, b/g)
    let twice = residue.add_ref(&residue);
    let s_min = if twice > modulus {
        BigInt::from_parts(Sign::Negative, modulus.sub_ref(&residue))
    } else {
        BigInt::from_biguint(residue)
    };
    // t = (g − s·a)/b, an exact division by the Bézout identity.
    let numerator = BigInt::from_biguint(g.clone()).sub_ref(&s_min.mul_biguint_ref(a));
    let (quotient, remainder) = numerator.magnitude().div_rem(b);
    debug_assert!(remainder.is_zero(), "g − s·a is divisible by b exactly");
    let t_min = BigInt::from_parts(numerator.sign(), quotient);
    (s_min, t_min)
}

/// Greatest common divisor by Lehmer's algorithm: classical Euclid with each
/// run of steps whose quotient the leading 64-bit digits fix batched into one
/// 2×2 transform of the full operands. Same result as plain Euclid, an order of
/// magnitude fewer multiprecision divisions.
fn gcd_lehmer(lhs: &BigUint, rhs: &BigUint) -> BigUint {
    let (mut a, mut b) = if lhs >= rhs {
        (lhs.clone(), rhs.clone())
    } else {
        (rhs.clone(), lhs.clone())
    };
    loop {
        if b.is_zero() {
            return a;
        }
        // Both operands in a single limb: finish in single-word Euclid.
        if b.limbs().len() == 1 {
            let small = a.rem_u64(b.limbs()[0]);
            return BigUint::from_u64(gcd_u64(b.limbs()[0], small));
        }
        // Leading digits are only comparable at equal length; otherwise one
        // ordinary step brings the operands level.
        let n = b.limbs().len();
        if a.limbs().len() != n {
            let remainder = a.modulo(&b);
            a = b;
            b = remainder;
            continue;
        }
        let (u_hat, v_hat) = leading_pair(&a, &b);
        let (m00, m01, m10, m11) = lehmer_transform(u_hat, v_hat, None);
        if m01 == 0 {
            // The leading digits pinned no step; take one full division step.
            let remainder = a.modulo(&b);
            a = b;
            b = remainder;
        } else {
            let next_a = combine_unsigned(m00, &a, m01, &b);
            let next_b = combine_unsigned(m10, &a, m11, &b);
            a = next_a;
            b = next_b;
            debug_assert!(a >= b, "Lehmer transform preserves a >= b");
        }
    }
}

/// Greatest common divisor.
///
/// Lehmer's algorithm (Knuth, *TAOCP* vol. 2, §4.5.2, Algorithm L) below the
/// crossover; above it, subquadratic Half-GCD (Möller, *On Schönhage's
/// algorithm and subquadratic integer gcd computation*, Math. Comp. 77
/// (2008), 589–607 — the algorithm behind GMP's `mpn_hgcd`), whose
/// O(M(n)·log n) beats Lehmer's O(n²). The dispatch tests the smaller
/// operand: one ordinary division collapses any pair to its smaller
/// element's size, so that size is what the work scales with.
#[must_use]
pub fn gcd(lhs: &BigUint, rhs: &BigUint) -> BigUint {
    if lhs.limbs().len().min(rhs.limbs().len()) >= HGCD_THRESHOLD_LIMBS {
        gcd_via_hgcd(lhs, rhs)
    } else {
        gcd_lehmer(lhs, rhs)
    }
}

/// Extended Euclid: `(g, s, t)` with `g = gcd(a, b) = a·s + b·t` (*Handbook of
/// Applied Cryptography*, Algorithm 2.107; Knuth, *TAOCP* vol. 2, §4.5.2,
/// Algorithm X).
///
/// Tracks both Bézout coefficient pairs, carrying them through the same
/// leading-digit Lehmer transform [`gcd`] uses, so the result is identical to
/// the classical step-by-step recurrence but with the reductions batched.
/// [`mod_inverse`] is the lean variant that keeps only one coefficient.
///
/// ```
/// use rump::{gcd_extended, BigInt, BigUint};
///
/// let (a, b) = (BigUint::from_u64(240), BigUint::from_u64(46));
/// let (g, s, t) = gcd_extended(&a, &b);
/// assert_eq!(g, BigUint::from_u64(2));
/// let bezout = s.mul_biguint_ref(&a).add_ref(&t.mul_biguint_ref(&b));
/// assert_eq!(bezout, BigInt::from_biguint(g));
/// ```
#[must_use]
pub fn gcd_extended(a: &BigUint, b: &BigUint) -> (BigUint, BigInt, BigInt) {
    if a.limbs().len().min(b.limbs().len()) >= HGCD_EXT_THRESHOLD_LIMBS
        && !a.is_zero()
        && !b.is_zero()
    {
        let (g, s, _t) = gcd_extended_via_hgcd(a, b);
        let (s, t) = canonicalize_bezout(a, b, &g, &s);
        return (g, s, t);
    }
    gcd_extended_lehmer(a, b)
}

/// The Lehmer-engine extended Euclid: [`gcd_extended`]'s below-crossover
/// path, and the finishing step of the Half-GCD driver above it.
fn gcd_extended_lehmer(a: &BigUint, b: &BigUint) -> (BigUint, BigInt, BigInt) {
    // Invariant: r0 = s0·a + t0·b and r1 = s1·a + t1·b throughout. The loop
    // reproduces classical extended Euclid step for step (a leading `a < b`
    // enters as one quotient-zero swap), only with runs of steps batched.
    let (mut r0, mut r1) = (a.clone(), b.clone());
    let (mut s0, mut s1) = (BigInt::from_biguint(BigUint::one()), BigInt::zero());
    let (mut t0, mut t1) = (BigInt::zero(), BigInt::from_biguint(BigUint::one()));

    while !r1.is_zero() {
        let n = r1.limbs().len();
        // A Lehmer step only when the operands share a multi-limb length and
        // r0 >= r1, so the 124-bit leading digits are aligned and in range.
        if n >= 2 && r0.limbs().len() == n && r0 >= r1 {
            let (u_hat, v_hat) = leading_pair(&r0, &r1);
            let (m00, m01, m10, m11) = lehmer_transform(u_hat, v_hat, None);
            if m01 != 0 {
                let next_r0 = combine_unsigned(m00, &r0, m01, &r1);
                let next_r1 = combine_unsigned(m10, &r0, m11, &r1);
                let next_s0 = combine_signed(m00, &s0, m01, &s1);
                let next_s1 = combine_signed(m10, &s0, m11, &s1);
                let next_t0 = combine_signed(m00, &t0, m01, &t1);
                let next_t1 = combine_signed(m10, &t0, m11, &t1);
                (r0, r1) = (next_r0, next_r1);
                (s0, s1) = (next_s0, next_s1);
                (t0, t1) = (next_t0, next_t1);
                debug_assert!(r0 >= r1, "Lehmer transform preserves r0 >= r1");
                continue;
            }
        }
        // One ordinary Euclid step: (r0, r1) → (r1, r0 mod r1), each cofactor
        // pair following the same quotient.
        let (quotient, remainder) = r0.div_rem(&r1);
        let next_s1 = s0.sub_ref(&s1.mul_biguint_ref(&quotient));
        let next_t1 = t0.sub_ref(&t1.mul_biguint_ref(&quotient));
        r0 = core::mem::replace(&mut r1, remainder);
        s0 = core::mem::replace(&mut s1, next_s1);
        t0 = core::mem::replace(&mut t1, next_t1);
    }

    (r0, s0, t0)
}

// ─── Batch smoothness (Bernstein) ──────────────────────────────────────────

/// The product tree of `values`: level 0 is the values themselves, each
/// higher level the pairwise products of the level below, the top level a
/// single root equal to the product of all of them. An odd node at any
/// level carries up unpaired. Empty input yields an empty tree.
///
/// The tree is the shared structure of [`remainder_tree`]: building it once
/// and reducing a modulus down it costs O(M(N)·log n) for total input size
/// N, against O(n) full-width divisions done separately. A zero leaf is
/// permitted here (its product is zero), but [`remainder_tree`] cannot
/// divide by it.
///
/// Reference: Bernstein, *How to find smooth parts of integers* (2004);
/// the product/remainder tree is the standard fast multiple-reduction.
#[must_use]
pub fn product_tree(values: &[BigUint]) -> Vec<Vec<BigUint>> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut levels = vec![values.to_vec()];
    while levels.last().expect("non-empty").len() > 1 {
        let below = levels.last().expect("non-empty");
        let mut up = Vec::with_capacity(below.len().div_ceil(2));
        let mut i = 0;
        while i < below.len() {
            if i + 1 < below.len() {
                up.push(below[i].mul_ref(&below[i + 1]));
            } else {
                up.push(below[i].clone());
            }
            i += 2;
        }
        levels.push(up);
    }
    levels
}

/// `modulus mod vᵢ` for every leaf `vᵢ` of a [`product_tree`], by one
/// descent: reduce the modulus against the root, then reduce each running
/// remainder against the two child products below it, so the divisor
/// shrinks toward the leaf rather than the whole modulus dividing each.
/// Returns the leaf remainders in input order; empty tree yields an empty
/// vector.
///
/// # Panics
///
/// Panics if any leaf is zero — reduction divides by it.
#[must_use]
pub fn remainder_tree(tree: &[Vec<BigUint>], modulus: &BigUint) -> Vec<BigUint> {
    let Some(root_level) = tree.last() else {
        return Vec::new();
    };
    // Remainders against the top level.
    let mut remainders: Vec<BigUint> = root_level.iter().map(|v| modulus.modulo(v)).collect();
    // Descend: level index from top−1 down to 0; a node's parent remainder
    // reduces against the node's own value.
    for level in (0..tree.len() - 1).rev() {
        let here = &tree[level];
        let mut next = Vec::with_capacity(here.len());
        for (j, value) in here.iter().enumerate() {
            next.push(remainders[j / 2].modulo(value));
        }
        remainders = next;
    }
    remainders
}

/// The smooth part of each value over the prime set `primes`: the largest
/// divisor composed only of those primes, with multiplicity. A value is
/// fully smooth exactly when its smooth part equals itself.
///
/// The primes' product `z` is reduced against every value at once by a
/// remainder tree. A prime `p ∈ primes` dividing `vᵢ` divides `z` and so
/// divides `z mod vᵢ`; raising that remainder to `2^s` (mod `vᵢ`), with
/// `2^s` past every prime's multiplicity in `vᵢ`, accumulates each such
/// prime to its full exponent, and the gcd with `vᵢ` is the smooth part.
/// No prime outside the base survives the gcd, since any `q` dividing both
/// `vᵢ` and `z mod vᵢ` divides `z` and is therefore in the base. One
/// batched pass replaces per-value trial division (Bernstein, 2004).
///
/// Trivial values pass through: `0` (divisible by every prime) maps to
/// `0`, `1` maps to `1`, and neither joins the tree.
#[must_use]
pub fn smooth_parts(values: &[BigUint], primes: &[u64]) -> Vec<BigUint> {
    if values.is_empty() {
        return Vec::new();
    }
    // z = ∏ primes, itself via a product tree for the near-linear cost.
    let prime_values: Vec<BigUint> = primes.iter().map(|&p| BigUint::from_u64(p)).collect();
    let z = match product_tree(&prime_values).last() {
        Some(root) => root[0].clone(),
        None => BigUint::one(), // no primes: nothing is smooth beyond 1
    };

    // The remainder tree divides by each value, so trivial values (0 has
    // no valid reduction; 1 reduces everything to 0) are handled directly
    // and only the rest form the tree. The smooth part of 0 is 0 — every
    // prime divides it — and of 1 is 1.
    let nontrivial: Vec<BigUint> = values
        .iter()
        .filter(|v| !v.is_zero() && !v.is_one())
        .cloned()
        .collect();
    let tree = product_tree(&nontrivial);
    let residues = remainder_tree(&tree, &z);
    let mut smooth_iter = nontrivial.iter().zip(residues).map(|(value, residue)| {
        // Raise the residue to 2^s mod v with 2^s ≥ ⌊log₂ v⌋, the largest
        // multiplicity any prime can have in v. `s` squarings give exponent
        // 2^s, so s = ⌈log₂(bits)⌉ suffices — a dozen squarings at 4 kbit,
        // not four thousand.
        let bits = value.bits();
        let squarings = bits.ilog2() as usize + 1; // 2^squarings > bits
        let mut acc = residue; // already z mod value from the tree
        for _ in 0..squarings {
            acc = BigUint::mod_mul(&acc, &acc, value);
        }
        // gcd(v, acc) is the product of the smooth prime powers in v.
        gcd(value, &acc)
    });

    values
        .iter()
        .map(|value| {
            if value.is_zero() || value.is_one() {
                value.clone()
            } else {
                smooth_iter
                    .next()
                    .expect("one smooth part per nontrivial value")
            }
        })
        .collect()
}

// ─── Rational reconstruction ───────────────────────────────────────────────

/// Rational reconstruction with explicit bounds: the unique fraction `p/q`
/// with `p ≡ q·x (mod m)`, `|p| ≤ num_bound`, `0 < q ≤ den_bound`, and
/// `gcd(p, q) = 1` — or `None` when no such fraction exists.
///
/// The candidate comes from the extended Euclidean remainder sequence of
/// `(m, x)`, stopped at the first remainder `r ≤ num_bound`: writing
/// `r ≡ t·x (mod m)` for that row's cofactor `t`, the only possible answer
/// is `p = ±r`, `q = |t|` (von zur Gathen and Gerhard, *Modern Computer
/// Algebra*, 3rd ed. (2013), §5.10, rational number reconstruction). The
/// technique is Wang's: *A p-adic algorithm for univariate partial
/// fractions*, SYMSAC '81, 212–217; the explicit rational-number statement
/// is Wang, Guy and Davenport, SIGSAM Bulletin 16(2) (1982), 2–3. See also
/// Collins and Encarnación, *Efficient rational number reconstruction*,
/// J. Symbolic Comput. 20(3) (1995), 287–297, for the accelerated variants
/// this implementation does not need. The final checks `q ≤ den_bound` and
/// `gcd(r, t) = 1` decide existence; uniqueness is the precondition
/// `2·num_bound·den_bound < m`.
///
/// The walk batches Euclid steps with the crate's Lehmer machinery — a
/// batch that would land at or below the stop line is discarded and
/// replayed as single division steps, so the row the theorem names is hit
/// exactly, at batch speed elsewhere.
///
/// # Panics
///
/// Panics unless `2·num_bound·den_bound < m` — the uniqueness precondition
/// is the caller's contract, and violating it silently would conflate
/// "no solution" with misuse.
#[must_use]
pub fn rational_reconstruct_bounded(
    x: &BigUint,
    m: &BigUint,
    num_bound: &BigUint,
    den_bound: &BigUint,
) -> Option<(BigInt, BigUint)> {
    assert!(
        BigUint::from_u64(2).mul_ref(num_bound).mul_ref(den_bound) < *m,
        "rational reconstruction requires 2·N·D < m"
    );
    if den_bound.is_zero() {
        return None;
    }
    let x = x.modulo(m);
    if x.is_zero() {
        return Some((BigInt::zero(), BigUint::one()));
    }

    // Invariant: r0 ≡ t0·x and r1 ≡ t1·x (mod m); r0 > r1 after entry.
    let (mut r0, mut r1) = (m.clone(), x);
    let (mut t0, mut t1) = (BigInt::zero(), BigInt::from_biguint(BigUint::one()));

    while r1 > *num_bound {
        let n = r1.limbs().len();
        if n >= 2 && r0.limbs().len() == n {
            let (u_hat, v_hat) = leading_pair(&r0, &r1);
            let (m00, m01, m10, m11) = lehmer_transform(u_hat, v_hat, None);
            if m01 != 0 {
                let next_r1 = combine_unsigned(m10, &r0, m11, &r1);
                // Commit only while the batch stays strictly above the stop
                // line; a batch that crosses it may skip the exact row the
                // theorem names, so that stretch is walked step by step.
                if next_r1 > *num_bound {
                    let next_r0 = combine_unsigned(m00, &r0, m01, &r1);
                    let next_t0 = combine_signed(m00, &t0, m01, &t1);
                    let next_t1 = combine_signed(m10, &t0, m11, &t1);
                    (r0, r1) = (next_r0, next_r1);
                    (t0, t1) = (next_t0, next_t1);
                    debug_assert!(r0 >= r1, "Lehmer transform preserves r0 >= r1");
                    continue;
                }
            }
        }
        let (quotient, remainder) = r0.div_rem(&r1);
        let next_t1 = t0.sub_ref(&t1.mul_biguint_ref(&quotient));
        r0 = core::mem::replace(&mut r1, remainder);
        t0 = core::mem::replace(&mut t1, next_t1);
    }

    // The candidate row: p = ±r1, q = |t1|.
    let denominator = t1.magnitude().clone();
    if denominator.is_zero() || denominator > *den_bound {
        return None;
    }
    if !gcd(&r1, &denominator).is_one() {
        return None;
    }
    let numerator = BigInt::from_parts(
        match t1.sign() {
            Sign::Negative => Sign::Negative,
            _ => Sign::Positive,
        },
        r1,
    );
    Some((numerator, denominator))
}

/// Rational reconstruction with the symmetric default bounds
/// `N = D = ⌊√((m−1)/2)⌋`, under which `2·N·D < m` holds automatically —
/// the standard choice when numerator and denominator are equally
/// unknown, as in recovering a fraction from its image modulo a large
/// modulus (CRT-lifted linear algebra, p-adic lifting).
///
/// The bound costs about what the walk costs — with Newton's iteration
/// under `sqrt_floor`, 0.3× the walk at 2048 bits and parity at 8192
/// (measured on M4 with generic operands; the 87× that stood here before
/// item 6 replaced the bisection is gone). Callers reconstructing many
/// values under one modulus should still compute the bound once and call
/// [`rational_reconstruct_bounded`]: at scale that halves the total.
///
/// See [`rational_reconstruct_bounded`] for the contract and references.
#[must_use]
pub fn rational_reconstruct(x: &BigUint, m: &BigUint) -> Option<(BigInt, BigUint)> {
    let mut half = m.sub_ref(&BigUint::one());
    half.shr1();
    let bound = half.sqrt_floor();
    rational_reconstruct_bounded(x, m, &bound, &bound)
}

/// Above this ratio of `s²` to the modulus bit width, [`sqrt_mod`] leaves
/// the Tonelli–Shanks descent for [`sqrt_mod_cipolla`]: the descent pays
/// on the order of `s²` base-field multiplications while Cipolla's ladder
/// is flat in `s`, so the crossover sits where `s²` reaches a fixed
/// multiple of the exponentiation cost, itself linear in the bit width.
/// Measured on M4 with constructed primes `k·2^s + 1` by the ignored
/// `cipolla_crossover_timing` probe, which times *both* engines at every
/// `s` on a grid bracketing the crossing: the engines cross at `s ≈ 70`
/// at 1024 bits, `s ≈ 93` at 2048, `s ≈ 124` at 4096 — `s²/bits` of 4.8,
/// 4.2, and 3.8 — and this factor sits near their center. Correctness
/// does not depend on the value: the suite drives both engines over the
/// same primes.
const CIPOLLA_THRESHOLD_FACTOR: usize = 4;

/// The Tonelli–Shanks descent for `p − 1 = q·2^s`, `a` a residue already
/// reduced modulo the odd prime `p` (Cohen, Algorithm 1.5.1). `None` when
/// the bounded non-residue scan exhausts or the descent finds no order to
/// reduce — both composite-modulus signatures.
fn sqrt_mod_descent(
    a: &BigUint,
    p: &BigUint,
    ctx: &MontgomeryCtx,
    q: &BigUint,
    s: usize,
) -> Option<BigUint> {
    let one = BigUint::one();
    // Any quadratic non-residue drives the descent; the scan is bounded so
    // an odd perfect-square modulus — which has no Jacobi non-residue at
    // all — terminates in None instead of scanning forever.
    let mut z = BigUint::from_u64(2);
    let mut attempts = 0u32;
    while jacobi(&z, p) != Some(-1) {
        z = z.add_ref(&one);
        attempts += 1;
        if attempts > NON_RESIDUE_SCAN_BOUND {
            return None;
        }
    }

    let mut m = s;
    let mut c = ctx.pow(&z, q);
    let mut t = ctx.pow(a, q);
    let mut r = {
        let mut half = q.add_ref(&one);
        half.shr1();
        ctx.pow(a, &half)
    };

    while t != one {
        // Least i with t^(2^i) = 1; it exists below m while p is prime.
        let mut i = 0usize;
        let mut probe = t.clone();
        while probe != one && i < m {
            probe = ctx.square(&probe);
            i += 1;
        }
        if i == m {
            // Reachable only for composite p; the caller's final check
            // would also catch it, but there is nothing left to descend.
            return None;
        }

        let mut b = c;
        for _ in 0..(m - i - 1) {
            b = ctx.square(&b);
        }
        m = i;
        c = ctx.square(&b);
        t = ctx.mul(&t, &c);
        r = ctx.mul(&r, &b);
    }
    Some(r)
}

/// Bound on the deterministic non-residue scans in both square-root
/// engines. For a prime modulus each draw fails with probability ~1/2, so
/// 128 consecutive misses has probability 2⁻¹²⁸ — never observed for a
/// prime; for an odd perfect-square modulus, whose Jacobi symbol is never
/// −1, the bound is what turns an unbounded loop into `None`.
const NON_RESIDUE_SCAN_BOUND: u32 = 128;

/// Cipolla's modular square root, for an odd prime `p` and a quadratic
/// residue `a` already reduced modulo `p`.
///
/// Choose `t` with `w = t² − a` a non-residue (each try succeeds with
/// probability ~1/2; consecutive small `t` serve, the character being
/// equidistributed). In `F_p² = F_p[x]/(x² − w)`, the element `t + x` has
/// norm `(t + x)(t + x)^p = (t + x)(t − x) = t² − w = a`, and `(t + x)^((p+1)/2)` is an
/// element of the base field whose square is `a`. The ladder runs
/// left-to-right over `(p+1)/2` as Montgomery-domain pairs `(u, v)`
/// standing for `u + v·x`: squaring is `(u² + w·v², 2uv)` and the
/// multiply by the fixed base `t + x` is `(u·t + w·v, u + v·t)`. Cost is
/// flat in `s = v₂(p−1)` — the property [`sqrt_mod`]'s dispatch buys —
/// against the descent's `s²` growth.
///
/// For composite `p` the returned value need not square to `a`;
/// [`sqrt_mod`]'s final verification is the contract, exactly as for the
/// descent. `None` only when the bounded parameter search exhausts —
/// impossible for a prime modulus in any observable sense, and exactly
/// what an odd perfect-square modulus produces.
///
/// Reference: Cipolla, *Un metodo per la risoluzione della congruenza di
/// secondo grado*, Rend. Accad. Sci. Fis. Mat. Napoli (3) 9 (1903),
/// 153–163.
fn sqrt_mod_cipolla(a: &BigUint, p: &BigUint, ctx: &MontgomeryCtx) -> Option<BigUint> {
    let one = BigUint::one();
    // t = 1, 2, …: the first with t² − a a non-residue, under the same
    // scan bound as the descent's search.
    let mut t = BigUint::one();
    let mut attempts = 0u32;
    let w = loop {
        let t_squared = BigUint::mod_mul(&t, &t, p);
        let difference = BigUint::mod_sub(&t_squared, a, p);
        if jacobi(&difference, p) == Some(-1) {
            break difference;
        }
        t = t.add_ref(&one);
        attempts += 1;
        if attempts > NON_RESIDUE_SCAN_BOUND {
            return None;
        }
    };

    let w_mont = ctx.encode(&w);
    let t_mont = ctx.encode(&t);
    // (p + 1)/2, the exponent taking t + x to the root.
    let mut exponent = p.add_ref(&one);
    exponent.shr1();

    // Ladder state (u, v) = u + v·x, starting at the identity 1 + 0·x.
    let mut u = ctx.encode(&one);
    let mut v = BigUint::zero();
    for bit in (0..exponent.bits()).rev() {
        // Square: (u² + w·v², 2uv).
        let u_squared = ctx.square_mont(&u);
        let cross = ctx.mul_mont(&u, &v);
        u = ctx.add_mont(&u_squared, &ctx.mul_mont(&w_mont, &ctx.square_mont(&v)));
        v = ctx.add_mont(&cross, &cross);
        if exponent.bit(bit) {
            // Multiply by the base t + x: (u·t + w·v, u + v·t).
            let new_u = ctx.add_mont(&ctx.mul_mont(&u, &t_mont), &ctx.mul_mont(&w_mont, &v));
            let new_v = ctx.add_mont(&u, &ctx.mul_mont(&v, &t_mont));
            u = new_u;
            v = new_v;
        }
    }
    // For prime p the result lies in the base field (v ≡ 0); the caller's
    // verification covers the composite case.
    Some(ctx.decode(&u))
}

/// Simultaneous modular inversion — Montgomery's trick: the inverses of
/// `n` values for the price of one inversion and `3(n − 1)`
/// multiplications.
///
/// Prefix products `P_i = x_1···x_i` climb forward; one inversion of
/// `P_n` and a backward unwind recover every `x_i⁻¹`:
/// `x_i⁻¹ = P_n⁻¹·P_{i−1}·(x_{i+1}···x_n)`, the trailing product carried
/// in the accumulator as it walks back. `None` when any element shares a
/// factor with the modulus — the batch learns *that* from the single
/// inversion, but not which element, and identifying the culprit would
/// cost the per-element inversions the trick exists to avoid. The empty
/// batch inverts to the empty vector.
///
/// The gain approaches the ratio of one inversion to three
/// multiplications: measured on M4 at 2048 bits, 1.5× at a batch of two,
/// 3.3× at a hundred, levelling near 3.4× — this crate's Lehmer inversion
/// costs about ten multiplications, which caps the trick's ceiling at
/// that ratio over three.
///
/// A modulus of zero yields `None` (nothing is invertible in no ring); a
/// modulus of one yields the trivial ring's answer, zero for every
/// element, matching [`mod_inverse`].
///
/// Reference: Montgomery, *Speeding the Pollard and elliptic curve
/// methods of factorization*, Math. Comp. 48 (1987), 243–264, where the
/// device batches the curve-arithmetic inversions.
#[must_use]
pub fn mod_inverse_batch(values: &[BigUint], modulus: &BigUint) -> Option<Vec<BigUint>> {
    if modulus.is_zero() {
        return None;
    }
    if modulus.is_one() {
        return Some(vec![BigUint::zero(); values.len()]);
    }
    if values.is_empty() {
        return Some(Vec::new());
    }
    // Forward prefix products, each reduced.
    let mut prefixes = Vec::with_capacity(values.len());
    let mut running = values[0].modulo(modulus);
    prefixes.push(running.clone());
    for value in &values[1..] {
        running = BigUint::mod_mul(&running, value, modulus);
        prefixes.push(running.clone());
    }
    // One inversion pays for the whole batch.
    let mut accumulator = mod_inverse(prefixes.last().expect("non-empty batch"), modulus)?;
    // Backward unwind: at step i the accumulator holds (x_1···x_i)⁻¹.
    let mut inverses = vec![BigUint::zero(); values.len()];
    for i in (1..values.len()).rev() {
        inverses[i] = BigUint::mod_mul(&accumulator, &prefixes[i - 1], modulus);
        accumulator = BigUint::mod_mul(&accumulator, &values[i], modulus);
    }
    inverses[0] = accumulator;
    debug_assert!(
        inverses
            .iter()
            .zip(values)
            .all(|(inv, x)| BigUint::mod_mul(inv, x, modulus).is_one()),
        "every batched inverse verifies against its element"
    );
    Some(inverses)
}

/// The `p`-adic valuation of `n`: the exponent of the largest power of
/// `p` dividing `n`. A thin reading of [`remove_factor`], for call sites
/// that want the exponent alone. `p` need not be prime — any `p ≥ 2` is
/// accepted, PARI's `valuation` convention — but for composite `p` the
/// result is the `mpz_remove` exponent, not a valuation in the
/// field-theoretic sense.
///
/// # Panics
///
/// Panics when `n` is zero (every power divides zero — the valuation is
/// unbounded) or `p < 2`.
#[must_use]
pub fn valuation(n: &BigUint, p: &BigUint) -> usize {
    remove_factor(n, p).1
}

/// Divide every factor of `p` out of `n`: `(n / p^e, e)` with `e` the
/// [`valuation`] — the shape of GMP's `mpz_remove`.
///
/// Factors come out through a ladder of squared powers, so a valuation of
/// `e` costs O(log e) divisions at `n`'s width rather than `e`: climb
/// while `p^(2^i)` divides, then descend re-trying each rung once.
/// `p = 2` is a shift, read off the limbs directly.
///
/// # Panics
///
/// Panics when `n` is zero or `p < 2`.
#[must_use]
pub fn remove_factor(n: &BigUint, p: &BigUint) -> (BigUint, usize) {
    assert!(!n.is_zero(), "zero has unbounded valuation");
    assert!(*p >= BigUint::from_u64(2), "valuation needs p >= 2");
    if *p == BigUint::from_u64(2) {
        let exponent = n.trailing_zeros().expect("n is non-zero");
        let mut cofactor = n.clone();
        cofactor.shr_bits(exponent);
        return (cofactor, exponent);
    }
    // Climb: divide out p^(2^i) while it divides exactly, doubling the
    // rung and keeping the ladder. The climb ends on a failed division —
    // an oversized rung fails in O(1), so no size guard is needed, and the
    // failure is what guarantees the remaining valuation is below the
    // failed rung's exponent.
    let mut cofactor = n.clone();
    let mut exponent = 0usize;
    let mut ladder = vec![p.clone()];
    let mut rung_exponent = 1usize;
    loop {
        let rung = ladder.last().expect("ladder starts non-empty");
        let (quotient, remainder) = cofactor.div_rem(rung);
        if !remainder.is_zero() {
            break;
        }
        cofactor = quotient;
        exponent += rung_exponent;
        ladder.push(rung.square_ref());
        rung_exponent *= 2;
    }
    // Descend: each lower rung divides at most once, because the rung
    // above it just failed.
    ladder.pop();
    rung_exponent /= 2;
    while let Some(rung) = ladder.pop() {
        if rung_exponent == 0 {
            break;
        }
        let (quotient, remainder) = cofactor.div_rem(&rung);
        if remainder.is_zero() {
            cofactor = quotient;
            exponent += rung_exponent;
        }
        rung_exponent /= 2;
    }
    (cofactor, exponent)
}

/// Least common multiple.
///
/// This is the Carmichael-function building block used by the RSA code: the
/// Python reference chooses `lambda = lcm(p - 1, q - 1)` rather than Euler's
/// totient because the private exponent only needs to invert modulo the
/// exponent cycle length.
#[must_use]
pub fn lcm(lhs: &BigUint, rhs: &BigUint) -> BigUint {
    if lhs.is_zero() || rhs.is_zero() {
        return BigUint::zero();
    }

    let divisor = gcd(lhs, rhs);
    let (quotient, remainder) = lhs.div_rem(&divisor);
    debug_assert!(remainder.is_zero(), "gcd divides the left operand exactly");
    quotient.mul_ref(rhs)
}

// ─── Quadratic-residue symbols ─────────────────────────────────────────────────

/// Jacobi symbol `(a/n)` for odd `n`, or `None` when `n` is even or zero.
///
/// Quadratic reciprocity, in the shape of *Handbook of Applied Cryptography*,
/// Algorithm 2.149: strip factors of two using the supplement
/// `(2/n) = (-1)^((n^2 - 1)/8)` — a sign flip exactly when `n ≡ 3, 5 (mod 8)`
/// — then swap the arguments, paying the reciprocity sign flip when both are
/// `≡ 3 (mod 4)`. The reduction, though, is division-free: rather than
/// `a mod n`, subtract-and-halve using the symbol's periodicity in its top
/// argument, `(a/n) = ((a - n)/n)` — the binary gcd this shadows, which is
/// markedly faster here than a full division per step.
///
/// That binary engine serves small operands. Above
/// `JACOBI_LEHMER_THRESHOLD_LIMBS` the computation moves to the Euclidean
/// quotient sequence with Lehmer batching, a state machine replaying each
/// batch's quotients (Möller's design, after Schönhage's identities); above
/// `JACOBI_HGCD_THRESHOLD_LIMBS` that state threads through the Half-GCD
/// recursion and the symbol is subquadratic, O(M(n)·log n), matching the
/// crate's gcd.
///
/// For prime `n` this is the Legendre symbol: `1` for quadratic residues,
/// `-1` for non-residues, `0` when `n` divides `a`. `(a/1) = 1` by the
/// empty-product convention.
///
/// ```
/// use rump::{jacobi, BigUint};
///
/// let nine = BigUint::from_u64(9);
/// assert_eq!(jacobi(&BigUint::from_u64(2), &nine), Some(1)); // 9 ≡ 1 (mod 8)
/// assert_eq!(jacobi(&BigUint::from_u64(3), &nine), Some(0)); // shared factor
/// assert_eq!(jacobi(&nine, &BigUint::from_u64(4)), None); // even modulus
/// ```
#[must_use]
pub fn jacobi(a: &BigUint, n: &BigUint) -> Option<i8> {
    if n.is_zero() || !n.is_odd() {
        return None;
    }
    let reduced = a.modulo(n);
    let inner = n.limbs().len().min(reduced.limbs().len());
    if inner >= JACOBI_HGCD_THRESHOLD_LIMBS {
        return Some(jacobi_hgcd(reduced, n.clone()));
    }
    if inner >= JACOBI_LEHMER_THRESHOLD_LIMBS {
        return Some(jacobi_lehmer(reduced, n.clone()));
    }
    jacobi_binary(reduced, n.clone())
}

/// Below this many limbs in the smaller operand, [`jacobi`] runs the binary
/// algorithm, whose shift-and-subtract steps are cheapest at small sizes; at
/// or above it, the Lehmer-batched quotient sequence, which advances by ~35
/// certified quotients per full-width pass where the binary loop advances by
/// a few bits. Measured on M4: a tie near 32 limbs, the batched engine 1.8×
/// ahead at 64 and 5.5× at 512, the gap compounding with size
/// (PERFORMANCE.md). Correctness does not depend on the value — the suite
/// validates with this set to 2, forcing every size through the new engine.
const JACOBI_LEHMER_THRESHOLD_LIMBS: usize = 64;

/// [`jacobi`]'s below-crossover engine: binary quadratic reciprocity, taking
/// `a` already reduced modulo the odd `n`.
fn jacobi_binary(reduced: BigUint, n: BigUint) -> Option<i8> {
    let mut a = reduced;
    let mut n = n;
    let mut sign = 1i8;

    while !a.is_zero() {
        // Strip a's factors of two; each one contributes the (2/n) supplement,
        // a sign flip exactly when n ≡ 3 or 5 (mod 8).
        let mut twos = 0usize;
        while !a.bit(twos) {
            twos += 1;
        }
        if twos % 2 == 1 && matches!(n.rem_u64(8), 3 | 5) {
            sign = -sign;
        }
        a.shr_bits(twos);

        // Order the (now both odd) arguments so a >= n. The swap is the
        // reciprocity step, paying its sign flip when both are ≡ 3 (mod 4).
        if a < n {
            if a.rem_u64(4) == 3 && n.rem_u64(4) == 3 {
                sign = -sign;
            }
            core::mem::swap(&mut a, &mut n);
        }

        // a >= n and both odd, so a - n is even and non-negative — stripped on
        // the next pass. This is the reduction, division-free: the symbol is
        // periodic in its top argument, so (a/n) = ((a - n)/n). Repeated
        // subtract-and-halve is the binary gcd this loop already shadows.
        a.sub_assign_ref(&n);
    }

    // The loop preserves (a/n) up to the accumulated sign; it ends with the
    // gcd in `n`. A gcd above one means a and the original n share a factor,
    // where the symbol is zero by definition.
    if n.is_one() {
        Some(sign)
    } else {
        Some(0)
    }
}

/// [`jacobi`]'s above-crossover engine: the Euclidean quotient sequence with
/// Lehmer batching, the [`JacobiState`] replaying each batch's applied
/// quotients. Takes `x` already reduced modulo the odd `y`.
///
/// The state's two slots are fixed to `x` and `y`; nothing is ever swapped.
/// A batch of `k` swapping remainder steps ends with the remainder pair
/// distributed by parity — the slot that held `r₀` (the larger input) holds
/// the even-indexed member of `(r_k, r_{k+1})` — and the replayed direction
/// flags alternate from whichever slot was reduced first. Every quotient the
/// state sees is a quotient actually applied, which is what Schönhage's
/// identities require.
fn jacobi_lehmer(x: BigUint, y: BigUint) -> i8 {
    debug_assert!(y.is_odd() && x < y);
    if x.is_zero() {
        return i8::from(y.is_one());
    }
    let state = JacobiState::new((x.limbs()[0] & 3) as u8, (y.limbs()[0] & 3) as u8);
    jacobi_lehmer_with_state(x, y, state)
}

/// The state-carrying core of [`jacobi_lehmer`]: continue the reduction of
/// `(x, y)` — the state's fixed `a` and `b` slots — to completion and read
/// out the symbol. Entered fresh by [`jacobi_lehmer`] and mid-flight by the
/// Half-GCD driver's tail.
fn jacobi_lehmer_with_state(x: BigUint, y: BigUint, state: JacobiState) -> i8 {
    let mut x = x;
    let mut y = y;
    let mut state = state;
    loop {
        if x.is_zero() {
            return if y.is_one() { state.finish() } else { 0 };
        }
        if y.is_zero() {
            return if x.is_one() { state.finish() } else { 0 };
        }
        let x_is_hi = x >= y;
        let (hi, lo) = if x_is_hi { (&x, &y) } else { (&y, &x) };
        // Batch only when the aligned leading digits can certify quotients:
        // both multi-limb and of equal length.
        let n_limbs = hi.limbs().len();
        if n_limbs >= 2 && lo.limbs().len() == n_limbs {
            let (u_hat, v_hat) = leading_pair(hi, lo);
            let mut log = QuotientLog::new();
            let (m00, m01, m10, m11) = lehmer_transform(u_hat, v_hat, Some(&mut log));
            if m01 != 0 {
                let next_hi = combine_unsigned(m00, hi, m01, lo);
                let next_lo = combine_unsigned(m10, hi, m11, lo);
                replay_batch(&mut state, x_is_hi, &log);
                // Parity places the results: the slot that held r₀ now holds
                // the even-indexed remainder of the final pair.
                let even_steps = log.len.is_multiple_of(2);
                let hi_slot_gets = if even_steps { &next_hi } else { &next_lo };
                let lo_slot_gets = if even_steps { &next_lo } else { &next_hi };
                if x_is_hi {
                    x = hi_slot_gets.clone();
                    y = lo_slot_gets.clone();
                } else {
                    y = hi_slot_gets.clone();
                    x = lo_slot_gets.clone();
                }
                continue;
            }
        }
        // The digits certified nothing (unequal lengths, or a boundary case):
        // one exact division step, reducing the larger slot in place.
        let d = u8::from(x_is_hi);
        let (q, r) = if x_is_hi {
            x.div_rem(&y)
        } else {
            y.div_rem(&x)
        };
        state.update(d, (q.limbs().first().copied().unwrap_or(0) & 3) as u8);
        if x_is_hi {
            x = r;
        } else {
            y = r;
        }
    }
}

/// Below this many limbs in the smaller operand, [`jacobi`] stays on the
/// Lehmer-batched engine; at or above it, the symbol state threads through
/// the Half-GCD recursion and the whole computation runs in O(M(n)·log n).
/// Measured on M4: Lehmer ahead 10% at 1536 limbs, the recursion ahead 7% at
/// 2048 and pulling away — 1.5× at 4096, 1.9× at 8192, 2.2× at 16384 limbs
/// (1 Mbit) — the same crossover as plain gcd's, which is also where GMP
/// pins its analogous `JACOBI_DC_THRESHOLD`. Correctness does not depend on
/// the value — the suite threads the state through [`hgcd`] at sizes from
/// 130 bits up and exercises [`jacobi_hgcd_engine`]'s recursion directly.
const JACOBI_HGCD_THRESHOLD_LIMBS: usize = 2048;

/// [`jacobi`]'s subquadratic engine: the reduction of [`gcd_via_hgcd`] with
/// the [`JacobiState`] threaded through every applied quotient. Takes `x`
/// already reduced modulo the odd `y`.
///
/// The state's slots are fixed to `x` and `y`, and [`hgcd`] preserves slot
/// order — its base case places each Lehmer batch's results by step parity,
/// and its guarded divisions reduce one slot in place — so each round's pair
/// drops back into the same slots. Nothing is ever sorted; the slots are
/// semantic, and a swap would silently misdirect every subsequent state
/// update. Each round halves the pair, so the rounds' costs form the same
/// geometric series as gcd's. When the pair falls below the crossover the
/// Lehmer engine finishes mid-flight through [`jacobi_lehmer_with_state`].
///
/// References: the threading design is Möller's, as realized in GMP's
/// `mpn_hgcd_jacobi` (`hgcd_jacobi.c`); the published subquadratic symbol is
/// Brent and Zimmermann, *An O(M(n) log n) algorithm for the Jacobi symbol*,
/// ANTS-IX, LNCS 6197 (2010), 83–95, which reaches the same complexity by
/// the binary route.
fn jacobi_hgcd(x: BigUint, y: BigUint) -> i8 {
    debug_assert!(y.is_odd() && x < y);
    if x.is_zero() {
        return i8::from(y.is_one());
    }
    let state = JacobiState::new((x.limbs()[0] & 3) as u8, (y.limbs()[0] & 3) as u8);
    jacobi_hgcd_engine(x, y, state, JACOBI_HGCD_THRESHOLD_LIMBS)
}

/// The round loop of [`jacobi_hgcd`], with the Lehmer handoff size a
/// parameter so the crossover probe can measure the recursion at sizes the
/// shipped threshold routes elsewhere — the same discipline as the gcd
/// probes, which measure the code as shipped rather than a copy of it.
fn jacobi_hgcd_engine(x: BigUint, y: BigUint, state: JacobiState, tail_limbs: usize) -> i8 {
    let mut state = state;
    let mut x = x;
    let mut y = y;
    loop {
        if x.limbs().len().min(y.limbs().len()) < tail_limbs {
            return jacobi_lehmer_with_state(x, y, state);
        }
        let s = pair_size(&x, &y) / 2 + 1;
        // hgcd needs the smaller element above the boundary and a gap wide
        // enough to close; otherwise one division step makes sharp progress
        // (an unbalanced pair collapses, a close pair drops to its
        // difference), with the state fed the as-applied quotient exactly as
        // in the Lehmer engine's guarded path.
        if pair_min_size(&x, &y) <= s || abs_diff_bits(&x, &y) <= s {
            let x_is_hi = x >= y;
            let (q, r) = if x_is_hi {
                x.div_rem(&y)
            } else {
                y.div_rem(&x)
            };
            state.update(
                u8::from(x_is_hi),
                (q.limbs().first().copied().unwrap_or(0) & 3) as u8,
            );
            if x_is_hi {
                x = r;
            } else {
                y = r;
            }
            continue;
        }
        let (_t, rx, ry) = hgcd(&x, &y, Some(&mut state));
        x = rx;
        y = ry;
    }
}

/// Every square root of `a` modulo `p^e` for a prime `p` and exponent
/// `e ≥ 1`, ascending, empty when `a` is a non-residue.
///
/// Sieving credits a prime power by the count of `x` with `x² ≡ kn`
/// (mod `p^e`); this returns them all, so a value divisible by `p³` is
/// counted at each of its roots. The awkward structure a caller should not
/// have to own lives here: for odd `p` a residue has two roots lifted from
/// one by Hensel's construction; for `p = 2` the count runs 1, 2, 4 as `e`
/// passes 1, 2, 3, and an odd `a` is a residue mod `2^e` (`e ≥ 3`) exactly
/// when `a ≡ 1 (mod 8)`; and an `a` divisible by `p` reduces by its
/// valuation, each unit root fanning into `p^(v/2)` roots of the power.
///
/// The returned count is input-proportional and can be large: `p^⌊v/2⌋`
/// roots when `p^v` exactly divides `a` (`v` even, `v < e`), and
/// `p^⌊e/2⌋` when `p^e` divides `a`. A 32-bit base prime with `v = 2`
/// therefore returns `p` roots — do not call this where `a` may be
/// divisible by a large `p` without expecting the allocation.
///
/// Reference: the odd-prime lift is Hensel's lemma (Cohen, *A Course in
/// Computational Algebraic Number Theory*, §3.5.3, "Factorization Modulo
/// pᵉ: Hensel's Lemma"); the dyadic and valuation cases follow the
/// standard structure of squares in `ℤ/2^eℤ`.
///
/// Primality of `p` is the caller's contract and is not checked. A
/// composite `p` returns a value satisfying no useful guarantee: the set
/// may be incomplete, and unlike [`sqrt_mod`] it is not verified by
/// squaring.
///
/// # Panics
///
/// Panics when `e == 0` or `p < 2`. A composite `p` can also reach an
/// internal inversion that panics (`2·root` must be a unit modulo `p^k`),
/// though a prime `p` never does.
#[must_use]
pub fn sqrt_mod_prime_power(a: &BigUint, p: &BigUint, e: u32) -> Vec<BigUint> {
    assert!(e >= 1, "prime-power exponent must be at least 1");
    assert!(*p >= BigUint::from_u64(2), "p must be a prime at least 2");
    let modulus = p.pow_u64(u64::from(e));
    let a = a.modulo(&modulus);

    // Valuation of a with respect to p, capped at e (a ≡ 0 mod p^e beyond).
    let (unit, v) = if a.is_zero() {
        (BigUint::zero(), e)
    } else {
        let (cofactor, val) = remove_factor(&a, p);
        (
            cofactor,
            u32::try_from(val)
                .expect("valuation below e ≤ u32::MAX")
                .min(e),
        )
    };

    // a ≡ 0 (mod p^e): x² ≡ 0 means x ≡ 0 (mod p^⌈e/2⌉).
    if v >= e {
        let step_exp = e.div_ceil(2);
        let step = p.pow_u64(u64::from(step_exp));
        let mut roots = Vec::new();
        let mut x = BigUint::zero();
        while x < modulus {
            roots.push(x.clone());
            x = x.add_ref(&step);
        }
        return roots;
    }
    // An odd valuation leaves an unsquarable p after the even part comes out.
    if v % 2 == 1 {
        return Vec::new();
    }

    // a = p^v · unit; solve t² ≡ unit (mod p^(e−v)) for the unit, then
    // x = p^(v/2)·t, each unit root fanning into p^(v/2) roots mod p^e.
    let reduced_exp = e - v;
    let unit_roots = sqrt_mod_prime_power_unit(&unit, p, reduced_exp);
    if unit_roots.is_empty() {
        return Vec::new();
    }
    let k = v / 2;
    let scale = p.pow_u64(u64::from(k)); // p^(v/2): the root's leading factor
    let fan_modulus = p.pow_u64(u64::from(reduced_exp)); // the unit roots' modulus
    let fan_count = scale.clone(); // p^(v/2) offsets fan each unit root
    let mut roots = Vec::new();
    for t0 in &unit_roots {
        let mut j = BigUint::zero();
        while j < fan_count {
            // x = scale · (t0 + j·fan_modulus)
            let t = t0.add_ref(&j.mul_ref(&fan_modulus));
            roots.push(scale.mul_ref(&t).modulo(&modulus));
            j = j.add_ref(&BigUint::one());
        }
    }
    roots.sort();
    roots
}

/// Square roots of a `p`-unit `u` modulo `p^e` — the [`sqrt_mod_prime_power`]
/// core, assuming `gcd(u, p) = 1`.
fn sqrt_mod_prime_power_unit(u: &BigUint, p: &BigUint, e: u32) -> Vec<BigUint> {
    let modulus = p.pow_u64(u64::from(e));
    let u = u.modulo(&modulus);
    let two = BigUint::from_u64(2);

    if *p == two {
        // Dyadic units: residue structure by e.
        let r = u.rem_u64(1u64 << e.min(3));
        return match e {
            1 => vec![BigUint::one()], // any odd u ≡ 1 (mod 2)
            2 => {
                if r % 4 == 1 {
                    vec![BigUint::one(), BigUint::from_u64(3)]
                } else {
                    Vec::new()
                }
            }
            _ => {
                if r % 8 != 1 {
                    return Vec::new();
                }
                // Lift a root of u mod 8 (namely 1) up to mod 2^e by
                // Newton correction, then generate the four roots
                // ±root, ±root + 2^(e−1).
                let mut root = BigUint::one();
                let mut k = 3u32;
                while k < e {
                    // root' = root − (root² − u)/(2·root) is ill-defined
                    // dyadically; instead correct bit by bit: if the new
                    // bit is wrong, flip it.
                    k += 1;
                    let modk = p.pow_u64(u64::from(k));
                    let sq = root.mul_ref(&root).modulo(&modk);
                    if sq != u.modulo(&modk) {
                        // Lifting a root from mod 2^(k−1) to mod 2^k: the
                        // discrepancy is at bit k−1, and adding 2^(k−2)
                        // flips exactly it, since (r + 2^(k−2))² differs
                        // from r² by 2^(k−1)·r (≡ 2^(k−1), r odd) plus a
                        // 2^(2k−4) term that vanishes mod 2^k for k ≥ 4.
                        let mut fix = BigUint::zero();
                        fix.set_bit((k - 2) as usize);
                        root = root.add_ref(&fix).modulo(&modk);
                    }
                }
                let neg = modulus.sub_ref(&root);
                let mut half = BigUint::zero();
                half.set_bit((e - 1) as usize);
                let alt = root.add_ref(&half).modulo(&modulus);
                let alt_neg = modulus.sub_ref(&alt);
                let mut roots = vec![root, neg, alt, alt_neg];
                roots.sort();
                roots.dedup();
                roots
            }
        };
    }

    // Odd p: one base root mod p, Hensel-lifted to mod p^e; the two roots
    // are r and p^e − r.
    let Some(base) = sqrt_mod(&u, p) else {
        return Vec::new();
    };
    let mut root = base;
    let mut k = 1u32;
    while k < e {
        // Newton's correction converges quadratically — root² ≡ u holds to
        // twice as many p-adic digits each step — so the precision doubles,
        // capped at e, turning e−1 inversions into ⌈log₂ e⌉.
        let next = (2 * k).min(e);
        let modk = p.pow_u64(u64::from(next));
        // Newton: root ← root − f(root)/f'(root), f = x² − u, f' = 2x.
        let f = root
            .mul_ref(&root)
            .modulo(&modk)
            .add_ref(&modk)
            .sub_ref(&u.modulo(&modk))
            .modulo(&modk);
        let two_root = two.mul_ref(&root).modulo(&modk);
        let inv = mod_inverse(&two_root, &modk).expect("2·root is a unit mod p^k");
        let correction = f.mul_ref(&inv).modulo(&modk);
        root = root.add_ref(&modk).sub_ref(&correction).modulo(&modk);
        k = next;
    }
    let other = modulus.sub_ref(&root);
    let mut roots = vec![root, other];
    roots.sort();
    roots.dedup();
    roots
}

/// Legendre symbol `(a/p)` for an odd prime `p`, or `None` when `p` is even
/// or zero.
///
/// For prime `p` the Jacobi and Legendre symbols coincide, so this delegates
/// to [`jacobi`]; it exists so call sites can say what they mean. Primality
/// of `p` is the caller's contract — for an odd composite the value returned
/// is the Jacobi symbol, which no longer decides quadratic residuosity.
#[must_use]
pub fn legendre(a: &BigUint, p: &BigUint) -> Option<i8> {
    jacobi(a, p)
}

/// Kronecker symbol `(a/n)`, the total extension of [`jacobi`] to every
/// modulus (Cohen, *A Course in Computational Algebraic Number Theory*,
/// Algorithm 1.4.10, restricted to non-negative arguments).
///
/// Factor `n = 2^v · m` with `m` odd; then `(a/n) = (a/2)^v · (a/m)` where
/// the supplement `(a/2)` is `0` for even `a` and `(-1)^((a^2 - 1)/8)`
/// otherwise, and `(a/m)` is the Jacobi symbol. By convention `(a/0)` is `1`
/// when `a = 1` and `0` otherwise. Agrees with [`jacobi`] whenever that is
/// defined.
#[must_use]
pub fn kronecker(a: &BigUint, n: &BigUint) -> i8 {
    if n.is_zero() {
        return i8::from(a.is_one());
    }

    // Strip n's factors of two, paying the (a/2) supplement per factor: zero
    // if a is even, and a sign that depends on a mod 8 — but only the parity
    // of v can matter.
    let mut twos = 0usize;
    while !n.bit(twos) {
        twos += 1;
    }
    let mut sign = 1i8;
    if twos > 0 {
        if !a.is_odd() {
            return 0;
        }
        if twos % 2 == 1 && matches!(a.rem_u64(8), 3 | 5) {
            sign = -sign;
        }
    }

    let mut m = n.clone();
    m.shr_bits(twos);
    let j = jacobi(a, &m).expect("m is odd by construction");
    sign * j
}

// ─── Modular arithmetic ────────────────────────────────────────────────────────

/// `base^exponent mod modulus` by square-and-multiply.
///
/// Dispatches on the modulus parity, because the fast reduction only exists
/// for odd moduli: an odd modulus runs the exponentiation in a
/// [`MontgomeryCtx`] (each step a REDC, no division), while an even modulus —
/// which has no Montgomery form — falls back to a binary square-and-multiply
/// that reduces with [`BigUint::mod_mul`] at every step. Both yield the same
/// value; the split is purely which reduction is available.
///
/// # Panics
///
/// Panics if `modulus == 0`.
#[must_use]
pub fn mod_pow(base: &BigUint, exponent: &BigUint, modulus: &BigUint) -> BigUint {
    assert!(!modulus.is_zero(), "modulus must be non-zero");
    if modulus == &BigUint::one() {
        return BigUint::zero();
    }
    if let Some(ctx) = MontgomeryCtx::new(modulus) {
        return ctx.pow(base, exponent);
    }

    let mut result = BigUint::one();
    let mut power = base.modulo(modulus);
    for bit in 0..exponent.bits() {
        if exponent.bit(bit) {
            result = BigUint::mod_mul(&result, &power, modulus);
        }
        power = BigUint::mod_mul(&power, &power, modulus);
    }
    result
}

/// Multiplicative inverse `a^{-1} mod n`, if it exists (*Handbook of Applied
/// Cryptography*, Algorithm 2.142).
///
/// Extended Euclid — over the shared leading-digit Lehmer engine — tracking
/// only the coefficient of `a`, half the signed bookkeeping of
/// [`gcd_extended`], which measurably matters to callers doing single-shot
/// inversion chains (Lagrange interpolation is one inversion per share). Use
/// [`gcd_extended`] when the full Bézout triple is wanted.
#[must_use]
pub fn mod_inverse(a: &BigUint, n: &BigUint) -> Option<BigUint> {
    if n.is_zero() {
        return None;
    }

    let reduced = a.modulo(n);

    // Above the crossover, ride the Half-GCD driver: the inverse is the
    // cofactor of `a mod n` in the Bézout identity for (n, a mod n), and
    // reducing it mod n makes the result unique — identical to the Lehmer
    // path's, whichever valid Bézout pair the transform produced.
    if n.limbs().len().min(reduced.limbs().len()) >= HGCD_EXT_THRESHOLD_LIMBS {
        let (g, _s, t) = gcd_extended_via_hgcd(n, &reduced);
        if !g.is_one() {
            return None;
        }
        return Some(t.modulo_positive(n));
    }

    // Extended Euclid on (n, a mod n) over the shared Lehmer engine, tracking
    // only the coefficient of `a mod n`: modulo n the modulus contributes
    // nothing, so r0 ≡ u0·(a mod n) (mod n), and when r0 reaches gcd = 1 the
    // coefficient u0 is the inverse.
    let (mut r0, mut r1) = (n.clone(), reduced);
    let (mut u0, mut u1) = (BigInt::zero(), BigInt::from_biguint(BigUint::one()));

    while !r1.is_zero() {
        let m = r1.limbs().len();
        if m >= 2 && r0.limbs().len() == m && r0 >= r1 {
            let (u_hat, v_hat) = leading_pair(&r0, &r1);
            let (m00, m01, m10, m11) = lehmer_transform(u_hat, v_hat, None);
            if m01 != 0 {
                let next_r0 = combine_unsigned(m00, &r0, m01, &r1);
                let next_r1 = combine_unsigned(m10, &r0, m11, &r1);
                let next_u0 = combine_signed(m00, &u0, m01, &u1);
                let next_u1 = combine_signed(m10, &u0, m11, &u1);
                (r0, r1) = (next_r0, next_r1);
                (u0, u1) = (next_u0, next_u1);
                debug_assert!(r0 >= r1, "Lehmer transform preserves r0 >= r1");
                continue;
            }
        }
        let (quotient, remainder) = r0.div_rem(&r1);
        let next_u1 = u0.sub_ref(&u1.mul_biguint_ref(&quotient));
        r0 = core::mem::replace(&mut r1, remainder);
        u0 = core::mem::replace(&mut u1, next_u1);
    }

    if !r0.is_one() {
        return None;
    }
    Some(u0.modulo_positive(n))
}

/// Write `n - 1 = d * 2^s` with `d` odd: the 2-adic split shared by the
/// Miller-Rabin witness test and the Tonelli–Shanks descent.
fn decompose_n_minus_one(n: &BigUint) -> (BigUint, usize) {
    let mut odd_factor = n.sub_ref(&BigUint::one());
    let two_adic_exponent = odd_factor.trailing_zeros().expect("n exceeds one");
    odd_factor.shr_bits(two_adic_exponent);
    (odd_factor, two_adic_exponent)
}

/// Modular square root: some `r` with `r^2 ≡ a (mod p)` for an odd prime
/// `p`, or `None` when `a` is a non-residue.
///
/// Three engines serve by the prime's shape. `p ≡ 3 (mod 4)` takes the
/// `a^((p+1)/4)` shortcut (*Handbook of Applied Cryptography*, §3.5.1).
/// Otherwise, writing `p − 1 = q·2^s`, a shallow 2-adic structure runs the
/// Tonelli–Shanks descent (Cohen, Algorithm 1.5.1), whose cost grows with
/// `s²`; past a measured depth (`s² > 4·bits`) the dispatch switches to
/// Cipolla's algorithm (Cipolla 1903), whose cost is flat in `s`. The non-residue each engine needs is found
/// by a bounded deterministic scan — expected to end within a couple of
/// draws, since half of all residues qualify, and abandoned after 128
/// misses, an event of probability `2^{-128}` for a prime modulus.
///
/// The other root is `p - r`. `p = 2` and `a ≡ 0` return `a mod p` and zero
/// respectively. Primality of `p` is the caller's contract, and the function
/// does not certify it. What it does guarantee on every odd `p` is safety:
/// the candidate is verified to square to `a mod p` before it is returned, so
/// the return is never a value that fails its own check, and the non-residue
/// scan is bounded, so the call always terminates — odd perfect squares
/// included, which admit no non-residue-by-Jacobi at all. On a composite `p`
/// the result is therefore either `None` or a genuine root modulo that
/// composite; it is not a primality verdict. For example `sqrt_mod(1, 15)`
/// returns `Some(1)` (a true root of 1 mod 15), while `sqrt_mod(1, 9)`
/// returns `None` (the bounded scan finds no non-residue mod the square 9).
/// A residue that genuinely is a square can still return `None` this way:
/// `sqrt_mod(4, 9)` is `None` even though `4 = 2² (mod 9)`, because the
/// descent needs a non-residue that a square modulus does not have. For roots
/// modulo a prime power, use [`sqrt_mod_prime_power`], which lifts by Hensel
/// instead of searching for a non-residue.
///
/// ```
/// use rump::{sqrt_mod, BigUint};
///
/// let p = BigUint::from_u64(41); // 41 ≡ 1 (mod 8): the general descent
/// let two = BigUint::from_u64(2);
/// let root = sqrt_mod(&two, &p).expect("2 is a residue mod 41");
/// assert_eq!(BigUint::mod_mul(&root, &root, &p), two);
/// assert_eq!(sqrt_mod(&BigUint::from_u64(3), &p), None); // non-residue
/// ```
#[must_use]
pub fn sqrt_mod(a: &BigUint, p: &BigUint) -> Option<BigUint> {
    if p.is_zero() {
        return None;
    }
    let a = a.modulo(p);
    if !p.is_odd() {
        // The only even prime: 0 and 1 are their own square roots mod 2.
        return if p == &BigUint::from_u64(2) {
            Some(a)
        } else {
            None
        };
    }
    if a.is_zero() {
        return Some(BigUint::zero());
    }
    if jacobi(&a, p) != Some(1) {
        return None;
    }

    let one = BigUint::one();
    let ctx = MontgomeryCtx::new(p).expect("p is odd and non-zero");

    let candidate = if p.rem_u64(4) == 3 {
        // a^((p+1)/4): squaring it gives a^((p+1)/2) = a · a^((p-1)/2) = a
        // by Euler's criterion.
        let mut exponent = p.add_ref(&one);
        exponent.shr_bits(2);
        ctx.pow(&a, &exponent)
    } else {
        // p - 1 = q · 2^s with q odd. A deep 2-adic descent goes to
        // Cipolla, whose cost is flat where Tonelli–Shanks pays
        // quadratically in s.
        let (q, s) = decompose_n_minus_one(p);
        if s * s > CIPOLLA_THRESHOLD_FACTOR * p.bits() {
            sqrt_mod_cipolla(&a, p, &ctx)?
        } else {
            sqrt_mod_descent(&a, p, &ctx, &q, s)?
        }
    };

    // The square is the contract: a candidate that does not square back to
    // `a` is rejected, so a composite `p` yields either None or a value that
    // genuinely squares to `a` — never one that merely looks plausible.
    if ctx.square(&candidate) == a {
        Some(candidate)
    } else {
        None
    }
}

/// Chinese remaindering: the unique `x` below the product of the moduli with
/// `x ≡ rᵢ (mod mᵢ)` for every pair, or `None` when the moduli are not
/// pairwise coprime (or the input is empty or contains a zero modulus).
///
/// Incremental Garner recombination (*Handbook of Applied Cryptography*,
/// Algorithm 14.71): fold each congruence into the solution-so-far by
/// solving `x + M·k ≡ rᵢ (mod mᵢ)` for `k`. Residues may be unreduced.
///
/// Sunzi's classic: what leaves 2 mod 3, 3 mod 5, and 2 mod 7?
///
/// ```
/// use rump::{crt_combine, BigUint};
///
/// let x = crt_combine(&[
///     (BigUint::from_u64(2), BigUint::from_u64(3)),
///     (BigUint::from_u64(3), BigUint::from_u64(5)),
///     (BigUint::from_u64(2), BigUint::from_u64(7)),
/// ])
/// .expect("moduli are pairwise coprime");
/// assert_eq!(x, BigUint::from_u64(23));
/// ```
#[must_use]
pub fn crt_combine(congruences: &[(BigUint, BigUint)]) -> Option<BigUint> {
    let (first_residue, first_modulus) = congruences.first()?;
    if first_modulus.is_zero() {
        return None;
    }

    let mut solution = first_residue.modulo(first_modulus);
    let mut product = first_modulus.clone();

    for (residue, modulus) in &congruences[1..] {
        if modulus.is_zero() {
            return None;
        }
        // k = (residue - solution) · product⁻¹ (mod modulus); a missing
        // inverse is exactly the non-coprime case.
        let inverse = mod_inverse(&product.modulo(modulus), modulus)?;
        let residue = residue.modulo(modulus);
        // The bias by `modulus` keeps the subtraction in range; mod_mul
        // reduces the product, so no further reduction is needed here.
        let difference = residue.add_ref(modulus).sub_ref(&solution.modulo(modulus));
        let k = BigUint::mod_mul(&difference, &inverse, modulus);
        solution = solution.add_ref(&product.mul_ref(&k));
        product = product.mul_ref(modulus);
    }

    Some(solution)
}

// ─── Primality ─────────────────────────────────────────────────────────────────

/// Fixed Miller-Rabin witness set used by the bigint probable-prime test.
///
/// These twelve small prime bases give a deterministic, repeatable witness
/// schedule. They are the first twelve primes, `2` through `37`.
///
/// Notes on determinism:
/// - The first twelve prime bases make Miller-Rabin deterministic for every
///   `n < 3.317 × 10^24` (Sorenson & Webster, *Strong Pseudoprimes to Twelve
///   Prime Bases*, Math. Comp. 86 (2017), 985–1003; arXiv:1509.00864), which
///   covers the whole `n < 2^64 ≈ 1.8 × 10^19` range.
/// - For larger `BigUint` candidates this remains a strong fixed-basis
///   probable-prime test, but not a proof of primality.
const MR_BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// Trial-division sieve primes checked before the Miller-Rabin stage.
///
/// Cheap remainders here discard most composites before the code pays for any
/// modular exponentiation.
const SMALL_TRIAL_PRIMES: [u16; 168] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547,
    557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
    661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797,
    809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929,
    937, 941, 947, 953, 967, 971, 977, 983, 991, 997,
];

fn is_witness(
    base: &BigUint,
    ctx: &MontgomeryCtx,
    odd_factor: &BigUint,
    two_adic_exponent: usize,
) -> bool {
    let one = BigUint::one();
    let n_minus_one = ctx.modulus().sub_ref(&one);
    let mut value = ctx.pow(base, odd_factor);

    // Miller-Rabin witness test (HAC Algorithm 4.24): a non-trivial square
    // root of 1 proves compositeness, and failing to end at 1 is the usual
    // Fermat backstop.
    for _ in 0..two_adic_exponent {
        let next = ctx.square(&value);
        if next == one && value != one && value != n_minus_one {
            return true;
        }
        value = next;
    }

    value != one
}

/// Every prime below `bound` (exclusive), ascending, by the sieve of
/// Eratosthenes. The bulk companion to [`is_probable_prime`]: where that
/// tests one candidate, this enumerates a range, for callers assembling a
/// small-prime table (a factor base, a trial-division wheel, a wheel's
/// spokes).
///
/// A `Vec` is materialized; for a very large bound a segmented or
/// streaming form would cost less memory, but the whole-vector form is
/// what the immediate callers use. Cost is O(bound·log log bound) time and
/// O(bound) *bytes* of sieve (one `bool` per odd number; a bit-packed
/// sieve would use eight times less).
#[must_use]
pub fn primes_below(bound: u64) -> Vec<u64> {
    if bound <= 2 {
        return Vec::new();
    }
    // Odd-only sieve: index i represents the odd number 2i + 3, so the
    // sieve covers 3, 5, 7, … below `bound`.
    let odd_count = usize::try_from((bound - 3) / 2 + 1).expect("sieve fits addressable memory");
    let mut composite = vec![false; odd_count];
    let number = |i: usize| 2 * (i as u64) + 3;
    let mut i = 0usize;
    while number(i) * number(i) < bound {
        if !composite[i] {
            let p = number(i);
            // Cross out p², p²+2p, … (odd multiples of p only).
            let mut multiple = p * p;
            while multiple < bound {
                let index = usize::try_from((multiple - 3) / 2).expect("index within sieve");
                composite[index] = true;
                multiple += 2 * p;
            }
        }
        i += 1;
    }
    let mut primes = vec![2u64];
    for (i, &is_composite) in composite.iter().enumerate() {
        // The sieve slot for an odd number at or above `bound` (when
        // `bound` is itself odd, its own slot exists) is not a prime below
        // the bound.
        if !is_composite && number(i) < bound {
            primes.push(number(i));
        }
    }
    primes
}

/// Miller-Rabin probable-prime test with a fixed witness set.
///
/// Appropriate for the caller's own randomly generated candidates, where a
/// fixed base set already rejects composites with overwhelming probability.
/// For values that arrive from an untrusted source (parsed keys,
/// peer-supplied domain parameters) a fixed base set can be defeated by a
/// purpose-built strong pseudoprime; harden with additional
/// candidate-derived witnesses via [`miller_rabin_witness`], as the parent
/// cryptography crate's `is_probable_prime_untrusted` does.
#[must_use]
pub fn is_probable_prime(n: &BigUint) -> bool {
    mr_probable_prime(n, &MR_BASES)
}

/// Miller-Rabin using explicit witness bases.
///
/// Each base is reduced modulo `candidate` before use; a base congruent to
/// `0`, `1`, or `candidate − 1` is the trivial `±1` witness, testifies to
/// nothing, and does not count as a round. When *no* supplied base yields an
/// effective round — an empty set, or one whose every base is trivial — the
/// result is `false`: an untested candidate is never certified prime.
/// Callers wanting a guaranteed-nontrivial schedule should draw bases in
/// `[2, candidate − 1)`.
#[must_use]
pub fn is_probable_prime_with_bases(candidate: &BigUint, bases: &[u64]) -> bool {
    mr_probable_prime(candidate, bases)
}

/// Run one Miller-Rabin round: does `witness` prove `candidate` composite?
///
/// The reusable primitive behind the probable-prime tests, public so callers
/// can bring their own witness schedule (for example, witnesses derived by
/// hashing an untrusted candidate). Returns `false` — no compositeness
/// proven — for witnesses congruent to `0`, `±1 (mod candidate)`, which can
/// never testify, and treats even or trivial candidates as composite.
#[must_use]
pub fn miller_rabin_witness(candidate: &BigUint, witness: &BigUint) -> bool {
    let Some(ctx) = MontgomeryCtx::new(candidate) else {
        // Even or zero candidates: composite (or degenerate) by inspection.
        return true;
    };
    if candidate.is_one() {
        return true;
    }

    let n_minus_one = candidate.sub_ref(&BigUint::one());
    let witness = witness.modulo(candidate);
    if witness <= BigUint::one() || witness == n_minus_one {
        return false;
    }

    let (odd_factor, two_adic_exponent) = decompose_n_minus_one(candidate);
    is_witness(&witness, &ctx, &odd_factor, two_adic_exponent)
}

/// Trial-division screen shared by the probable-prime tests: `Some(true)`
/// when the candidate *is* one of the sieve primes, `Some(false)` when the
/// sieve proves it composite (or it is zero or one), `None` when it
/// survives to the probabilistic stages.
fn small_prime_screen(candidate: &BigUint) -> Option<bool> {
    if candidate.is_zero() || candidate == &BigUint::one() {
        return Some(false);
    }
    for &prime in &SMALL_TRIAL_PRIMES {
        let prime = u64::from(prime);
        if candidate.rem_u64(prime) == 0 {
            // A small prime divides itself as well as its composite
            // multiples. For candidates below 2^10, the residue modulo 2^10
            // distinguishes the identity case without allocating a temporary
            // BigUint for every sieve entry.
            let is_the_prime = candidate.bits() <= 10 && candidate.rem_u64(1u64 << 10) == prime;
            return Some(is_the_prime);
        }
    }
    None
}

fn mr_probable_prime(candidate: &BigUint, bases: &[u64]) -> bool {
    if let Some(verdict) = small_prime_screen(candidate) {
        return verdict;
    }

    if bases.is_empty() {
        return false;
    }

    let Some(ctx) = MontgomeryCtx::new(candidate) else {
        return false;
    };
    let n_minus_one = candidate.sub_ref(&BigUint::one());
    let (odd_factor, two_adic_exponent) = decompose_n_minus_one(candidate);

    // Count the rounds that could actually testify. A base reducing to 0, 1,
    // or n − 1 modulo the candidate is the trivial ±1 case and proves
    // nothing; skipping it must not be mistaken for passing it.
    let mut effective_rounds = 0usize;
    for &base in bases {
        // Reduce before classifying as trivial: an unreduced u64 that is
        // ≥ n − 1 may still reduce to a non-trivial residue that testifies,
        // so the raw comparison the earlier code used silently dropped real
        // witnesses (and, when every base was so dropped, returned `true`).
        let witness = BigUint::from_u64(base).modulo(candidate);
        if witness <= BigUint::one() || witness == n_minus_one {
            continue;
        }
        effective_rounds += 1;
        if is_witness(&witness, &ctx, &odd_factor, two_adic_exponent) {
            return false;
        }
    }

    // No effective round ran: nothing was tested, so nothing is proven. A
    // composite whose every supplied base was trivial must not be stamped
    // prime.
    effective_rounds > 0
}

/// Selfridge's Method A: the discriminant for the Lucas stage, the first of
/// `5, -7, 9, -11, 13, …` whose Jacobi symbol `(D/n)` is `-1`. Returns
/// `None` when the search itself proves `n` composite — a zero symbol
/// exposing a factor shared with a candidate discriminant, or `n` a perfect
/// square, for which no discriminant has symbol `-1` and which is therefore
/// ruled out directly once three candidates have failed (Baillie and
/// Wagstaff, §6). A zero symbol whose gcd with `n` is `n` itself carries no
/// information (n is 5, 7, 11, …) and the search continues past it.
///
/// The search terminates for every valid input: once squares are excluded,
/// the map `D ↦ (D/n)` is a non-principal character, `-1` on half the
/// residues, so a qualifying discriminant exists and arrives quickly — the
/// maximum |D| over all odd `n < 2·10⁶` is 59, far below the `i64` bound
/// the conversion below asserts.
fn selfridge_discriminant(n: &BigUint) -> Option<i64> {
    debug_assert!(n.is_odd() && !n.is_one());
    let mut d_abs: u64 = 5;
    let mut positive = true;
    let mut attempts = 0u32;
    loop {
        let signed = i64::try_from(d_abs).expect("discriminant search stays far below i64::MAX");
        let residue = BigInt::from_i64(if positive { signed } else { -signed }).modulo_positive(n);
        match jacobi(&residue, n) {
            Some(-1) => {
                let magnitude =
                    i64::try_from(d_abs).expect("discriminant search stays far below i64::MAX");
                return Some(if positive { magnitude } else { -magnitude });
            }
            // A zero symbol means gcd(|D|, n) > 1. A proper divisor proves
            // n composite; gcd equal to n itself (n divides the candidate
            // discriminant) carries no information — n may be the prime 5,
            // 7, 11, … — and the search continues.
            Some(0) if gcd(&BigUint::from_u64(d_abs), n) != *n => return None,
            _ => {}
        }
        attempts += 1;
        if attempts == 3 {
            let root = n.sqrt_floor();
            if root.square_ref() == *n {
                return None;
            }
        }
        d_abs += 2;
        positive = !positive;
    }
}

/// The strong Lucas test proper, for an odd `n > 1` with its Selfridge
/// discriminant already chosen: `P = 1`, `Q = (1 - D)/4`, and
/// `n + 1 = 2^s·d` with `d` odd. Accepts when `U_d ≡ 0 (mod n)`, or
/// `V_{d·2^r} ≡ 0 (mod n)` for some `0 ≤ r < s` — the conditions every
/// prime satisfies (Baillie and Wagstaff, §5; Crandall and Pomerance,
/// Algorithm 3.6.9).
///
/// The sequence runs left to right over `d`'s bits on the triple
/// `(U_k, V_k, Q^k)`, all as Montgomery residues: doubling by
/// `U_{2k} = U_k·V_k`, `V_{2k} = V_k² - 2Q^k`, `Q^{2k} = (Q^k)²`, and
/// stepping by the `P = 1` forms `U_{k+1} = (U_k + V_k)/2`,
/// `V_{k+1} = (D·U_k + V_k)/2`. The halving is exact modulo odd `n`
/// (add `n` first when the residue is odd), and commutes with the
/// Montgomery encoding because dividing by two is multiplication by the
/// constant `2⁻¹ mod n`.
fn strong_lucas_core(n: &BigUint, ctx: &MontgomeryCtx, discriminant: i64) -> bool {
    let to_residue = |value: i64| -> BigUint { BigInt::from_i64(value).modulo_positive(n) };
    let half_mod = |value: &BigUint| -> BigUint {
        let mut halved = if value.is_odd() {
            value.add_ref(n)
        } else {
            value.clone()
        };
        halved.shr1();
        halved
    };

    let d_mont = ctx.encode(&to_residue(discriminant));
    let q_mont = ctx.encode(&to_residue((1 - discriminant) / 4));

    // n + 1 = 2^s · d with d odd; s ≥ 1 since n is odd. One pass over the
    // limbs rather than a shift per bit.
    let mut d = n.add_ref(&BigUint::one());
    let s = d.trailing_zeros().expect("n + 1 is non-zero");
    d.shr_bits(s);

    // (U₁, V₁, Q¹) = (1, P, Q) with P = 1, then one double-and-maybe-step
    // per remaining bit of d, most significant first.
    let mut u = ctx.encode(&BigUint::one());
    let mut v = u.clone();
    let mut q_pow = q_mont.clone();
    for bit in (0..d.bits() - 1).rev() {
        u = ctx.mul_mont(&u, &v);
        v = ctx.sub_mont(&ctx.square_mont(&v), &ctx.add_mont(&q_pow, &q_pow));
        q_pow = ctx.square_mont(&q_pow);
        if d.bit(bit) {
            let stepped_u = half_mod(&ctx.add_mont(&u, &v));
            v = half_mod(&ctx.add_mont(&ctx.mul_mont(&d_mont, &u), &v));
            u = stepped_u;
            q_pow = ctx.mul_mont(&q_pow, &q_mont);
        }
    }

    // A zero Montgomery residue is a zero value: gcd(R, n) = 1.
    if u.is_zero() || v.is_zero() {
        return true;
    }
    for _ in 1..s {
        v = ctx.sub_mont(&ctx.square_mont(&v), &ctx.add_mont(&q_pow, &q_pow));
        q_pow = ctx.square_mont(&q_pow);
        if v.is_zero() {
            return true;
        }
    }
    false
}

/// Strong Lucas probable-prime test with Selfridge's Method A parameters.
///
/// The Lucas-sequence analogue of the strong (Miller–Rabin) test. Every
/// prime passes; a composite that passes is a strong Lucas pseudoprime —
/// the first is 5459, and the list below 10⁵ is disjoint from the strong
/// base-2 pseudoprimes, which is the observation [`is_probable_prime_bpsw`]
/// stakes its power on.
///
/// Reference: Baillie and Wagstaff, *Lucas pseudoprimes*, Math. Comp. 35
/// (1980), 1391–1417.
#[must_use]
pub fn is_strong_lucas_probable_prime(n: &BigUint) -> bool {
    if n.is_zero() || n.is_one() {
        return false;
    }
    if !n.is_odd() {
        return *n == BigUint::from_u64(2);
    }
    let Some(discriminant) = selfridge_discriminant(n) else {
        return false;
    };
    let ctx = MontgomeryCtx::new(n).expect("candidate is odd");
    strong_lucas_core(n, &ctx, discriminant)
}

/// Baillie–PSW probable-prime test: trial division, one strong base-2
/// Miller–Rabin round, then the strong Lucas test with Selfridge's
/// parameters.
///
/// The two probabilistic stages fail on disjoint kinds of composites as far
/// as anyone has found: no composite passing both is known, and none exists
/// below 2⁶⁴ — Feitsma's enumeration of the base-2 Fermat pseudoprimes to
/// that bound (verified independently by Galway), whose strong subset has
/// been checked exhaustively against the Lucas stage — so below 2⁶⁴ the
/// test is deterministic. [`is_probable_prime`]'s twelve fixed bases are
/// deterministic further, to 3.3·10²⁴ (Sorenson and Webster), and the two
/// tests fail differently above their bounds, which is why both exist.
///
/// Above 2⁶⁴ this is a probable-prime test, not a proof, and its
/// parameters are a fixed function of the candidate: as with any fixed
/// schedule, treat values from an untrusted source with additional
/// candidate-derived witnesses ([`miller_rabin_witness`]), as the parent
/// cryptography crate's `is_probable_prime_untrusted` does. No composite
/// passing this test is known at any size — but absence of a known
/// counterexample is not a proof, and the crate does not treat it as one.
///
/// References: Baillie and Wagstaff, *Lucas pseudoprimes*, Math. Comp. 35
/// (1980), 1391–1417; Pomerance, Selfridge and Wagstaff, *The pseudoprimes
/// to 25·10⁹*, Math. Comp. 35 (1980), 1003–1026.
#[must_use]
pub fn is_probable_prime_bpsw(n: &BigUint) -> bool {
    if let Some(verdict) = small_prime_screen(n) {
        return verdict;
    }
    // The screen passed an odd n > 1 with no factor ≤ 997.
    let ctx = MontgomeryCtx::new(n).expect("screened candidate is odd");
    let (odd_factor, two_adic_exponent) = decompose_n_minus_one(n);
    if is_witness(&BigUint::from_u64(2), &ctx, &odd_factor, two_adic_exponent) {
        return false;
    }
    let Some(discriminant) = selfridge_discriminant(n) else {
        return false;
    };
    strong_lucas_core(n, &ctx, discriminant)
}

#[cfg(test)]
mod tests {
    use super::{
        gcd, is_probable_prime, is_probable_prime_with_bases, jacobi, lcm, miller_rabin_witness,
        mod_inverse, mod_pow,
    };
    use crate::bigint::BigUint;

    /// splitmix64 (Steele, Lea & Flood 2014): deterministic scattered draws
    /// with no dependency; not a CSPRNG and not meant to be one.
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    /// Draw a value in `[1, upper)` by rejection from random 64-bit words.
    fn draw_below(rng: &mut SplitMix64, upper: &BigUint) -> BigUint {
        let words = upper.bits().div_ceil(64);
        loop {
            let mut bytes = Vec::with_capacity(words * 8);
            for _ in 0..words {
                bytes.extend_from_slice(&rng.next_u64().to_be_bytes());
            }
            let candidate = BigUint::from_be_bytes(&bytes).modulo(upper);
            if !candidate.is_zero() {
                return candidate;
            }
        }
    }

    fn mersenne(exponent: usize) -> BigUint {
        let mut value = BigUint::zero();
        value.set_bit(exponent);
        value.sub_ref(&BigUint::one())
    }

    fn pow2(bits: usize) -> BigUint {
        let mut v = BigUint::zero();
        v.set_bit(bits);
        v
    }

    /// The Jacobi state machine driven by a plain Euclidean quotient
    /// sequence — the validation harness for the state machine in isolation,
    /// before it is threaded through the Half-GCD reduction.
    fn jacobi_by_quotient_state(a: &BigUint, n: &BigUint) -> Option<i8> {
        use super::JacobiState;
        if n.is_zero() || !n.is_odd() {
            return None;
        }
        let mut x = a.modulo(n); // the state's `a` slot
        let mut y = n.clone(); // the state's `b` slot; odd at initialization
        if x.is_zero() {
            return Some(i8::from(y.is_one()));
        }
        let mut state = JacobiState::new((x.limbs()[0] & 3) as u8, (y.limbs()[0] & 3) as u8);
        loop {
            let d = u8::from(x >= y);
            let (q, r) = if d == 1 { x.div_rem(&y) } else { y.div_rem(&x) };
            state.update(d, (q.limbs().first().copied().unwrap_or(0) & 3) as u8);
            if d == 1 {
                x = r;
            } else {
                y = r;
            }
            let (reduced, other) = if d == 1 { (&x, &y) } else { (&y, &x) };
            if reduced.is_zero() {
                return Some(if other.is_one() { state.finish() } else { 0 });
            }
        }
    }

    #[test]
    #[ignore = "timing probe for the binary/Lehmer jacobi crossover; run with --ignored"]
    fn jacobi_crossover_timing() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut rng = SplitMix64 {
            state: 0x7ac0_b1de_ba7c_4ed5,
        };
        eprintln!(
            "{:>7} {:>12} {:>12}  winner",
            "limbs", "binary_ms", "lehmer_ms"
        );
        for &limbs in &[2usize, 4, 8, 16, 32, 64, 128, 256, 512] {
            let bits = limbs * 64;
            let mut n = draw_below(&mut rng, &pow2(bits));
            n.set_bit(bits - 1);
            if !n.is_odd() {
                n = n.add_ref(&BigUint::one());
            }
            let a = draw_below(&mut rng, &pow2(bits)).modulo(&n);
            let reps = (512 / limbs).max(2);
            let time = |f: &dyn Fn()| {
                let mut best = f64::INFINITY;
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        f();
                    }
                    best = best.min(t0.elapsed().as_secs_f64() / reps as f64 * 1e3);
                }
                best
            };
            let bin = time(&|| {
                black_box(super::jacobi_binary(a.clone(), n.clone()));
            });
            let leh = time(&|| {
                black_box(super::jacobi_lehmer(a.clone(), n.clone()));
            });
            let winner = if bin <= leh { "binary" } else { "lehmer" };
            eprintln!("{limbs:7} {bin:12.4} {leh:12.4}  {winner}");
        }
    }

    #[test]
    fn jacobi_table_matches_gmp() {
        // GMP's shipped jacobitab.h, as produced by Möller's gen-jacobitab.c —
        // an independently generated cross-check of the compile-time
        // derivation from Schönhage's rules.
        #[rustfmt::skip]
        const GMP_TABLE: [u8; 208] = [
             0,  0,  0,  0,  0, 12,  8,  4,  1,  1,  1,  1,  1, 13,  9,  5,
             2,  2,  2,  2,  2,  6, 10, 14,  3,  3,  3,  3,  3,  7, 11, 15,
             4, 16,  6, 18,  4,  0, 12,  8,  5, 17,  7, 19,  5,  1, 13,  9,
             6, 18,  4, 16,  6, 10, 14,  2,  7, 19,  5, 17,  7, 11, 15,  3,
             8, 10,  9, 11,  8,  4,  0, 12,  9, 11,  8, 10,  9,  5,  1, 13,
            10,  9, 11,  8, 10, 14,  2,  6, 11,  8, 10,  9, 11, 15,  3,  7,
            12, 22, 24, 20, 12,  8,  4,  0, 13, 23, 25, 21, 13,  9,  5,  1,
            25, 21, 13, 23, 14,  2,  6, 10, 24, 20, 12, 22, 15,  3,  7, 11,
            16,  6, 18,  4, 16, 16, 16, 16, 17,  7, 19,  5, 17, 17, 17, 17,
            18,  4, 16,  6, 18, 22, 19, 23, 19,  5, 17,  7, 19, 23, 18, 22,
            20, 12, 22, 24, 20, 20, 20, 20, 21, 13, 23, 25, 21, 21, 21, 21,
            22, 24, 20, 12, 22, 19, 23, 18, 23, 25, 21, 13, 23, 18, 22, 19,
            24, 20, 12, 22, 15,  3,  7, 11, 25, 21, 13, 23, 14,  2,  6, 10,
        ];
        assert_eq!(super::JACOBI_TABLE, GMP_TABLE);
    }

    /// The GMP-vector-grounded binary implementation as an oracle, on the
    /// same contract as the public function.
    fn jacobi_binary_oracle(a: &BigUint, n: &BigUint) -> Option<i8> {
        if n.is_zero() || !n.is_odd() {
            return None;
        }
        super::jacobi_binary(a.modulo(n), n.clone())
    }

    #[test]
    fn jacobi_state_machine_matches_binary() {
        // Exhaustive small cases: every (a, odd n) with n < 200, for both the
        // plain quotient driver and the Lehmer-batched engine.
        for n_small in (1u64..200).step_by(2) {
            let n = BigUint::from_u64(n_small);
            for a_small in 0..n_small.min(60) {
                let a = BigUint::from_u64(a_small);
                let oracle = jacobi_binary_oracle(&a, &n);
                assert_eq!(
                    jacobi_by_quotient_state(&a, &n),
                    oracle,
                    "quotient driver diverged at ({a_small}/{n_small})"
                );
                assert_eq!(
                    super::jacobi(&a, &n),
                    oracle,
                    "dispatched jacobi diverged at ({a_small}/{n_small})"
                );
            }
        }
        // Random sweep across sizes; the development threshold routes the
        // public function through the Lehmer-batched engine everywhere.
        let mut rng = SplitMix64 {
            state: 0x0dd5_ba11_5eed_c0de,
        };
        for &bits in &[16usize, 64, 128, 256, 777, 1024, 2048, 4096] {
            for _ in 0..30 {
                let mut n = draw_below(&mut rng, &pow2(bits));
                if !n.is_odd() {
                    n = n.add_ref(&BigUint::one());
                }
                let a = draw_below(&mut rng, &pow2(bits));
                let oracle = jacobi_binary_oracle(&a, &n);
                assert_eq!(
                    jacobi_by_quotient_state(&a, &n),
                    oracle,
                    "quotient driver diverged at {bits} bits"
                );
                assert_eq!(
                    super::jacobi(&a, &n),
                    oracle,
                    "dispatched jacobi diverged at {bits} bits"
                );
            }
        }
    }

    #[test]
    fn jacobi_state_through_hgcd_matches_binary() {
        use super::{hgcd, jacobi_lehmer_with_state, JacobiState};
        let mut rng = SplitMix64 {
            state: 0x7ac0_b1a5_ca55_e77e,
        };
        // Sizes span the batched base case (≤ 6144 bits) and the recursion
        // above it. Each case threads the state through one hgcd call and
        // hands the reduced pair to the Lehmer engine mid-flight — exactly
        // the driver's composition, at sizes the dispatch threshold would
        // never route here.
        for &(bits, reps) in &[
            (130usize, 60),
            (500, 40),
            (1500, 30),
            (4000, 20),
            (8000, 10),
            (16000, 6),
            (40000, 3),
        ] {
            for _ in 0..reps {
                // Both operands full width, satisfying hgcd's precondition
                // #min > ⌊bits/2⌋ + 1; sorted so x < y, and y made odd, which
                // only raises it.
                let mut x = draw_below(&mut rng, &pow2(bits));
                let mut y = draw_below(&mut rng, &pow2(bits));
                x.set_bit(bits - 1);
                y.set_bit(bits - 1);
                if x >= y {
                    core::mem::swap(&mut x, &mut y);
                }
                y.set_bit(0);
                if x >= y {
                    continue;
                }
                let mut state =
                    JacobiState::new((x.limbs()[0] & 3) as u8, (y.limbs()[0] & 3) as u8);
                let (_t, rx, ry) = hgcd(&x, &y, Some(&mut state));
                assert_eq!(
                    jacobi_lehmer_with_state(rx, ry, state),
                    jacobi_binary_oracle(&x, &y).unwrap(),
                    "state through hgcd diverged at {bits} bits"
                );
            }
        }
    }

    #[test]
    fn jacobi_hgcd_driver_matches_lehmer() {
        use super::{jacobi_hgcd, jacobi_lehmer, JACOBI_HGCD_THRESHOLD_LIMBS};
        let mut rng = SplitMix64 {
            state: 0x5eed_7e57_0dd1_7e57,
        };
        let full_width_odd = |rng: &mut SplitMix64, limbs: usize| {
            let bits = limbs * 64;
            let mut y = draw_below(rng, &pow2(bits));
            y.set_bit(bits - 1);
            y.set_bit(0);
            y
        };
        // One driver round at the threshold plus a margin, two rounds at
        // twice it (each round halves the pair). The Lehmer engine — itself
        // pinned to the binary oracle above — is the reference.
        for &limbs in &[
            JACOBI_HGCD_THRESHOLD_LIMBS + 8,
            2 * JACOBI_HGCD_THRESHOLD_LIMBS + 8,
        ] {
            let y = full_width_odd(&mut rng, limbs);
            let x = draw_below(&mut rng, &y);
            assert_eq!(
                jacobi_hgcd(x.clone(), y.clone()),
                jacobi_lehmer(x.clone(), y.clone()),
                "driver diverged at {limbs} limbs"
            );
            assert_eq!(
                super::jacobi(&x, &y),
                Some(jacobi_lehmer(x, y)),
                "public jacobi dispatch diverged at {limbs} limbs"
            );
        }
        // Structured shapes that force the driver's division fallback: a pair
        // too unbalanced for hgcd's boundary (smaller element at or below s
        // bits), and a pair too close (difference at or below s bits).
        let y = full_width_odd(&mut rng, 2 * JACOBI_HGCD_THRESHOLD_LIMBS + 56);
        let mut x = draw_below(&mut rng, &pow2(JACOBI_HGCD_THRESHOLD_LIMBS * 64));
        x.set_bit(JACOBI_HGCD_THRESHOLD_LIMBS * 64 - 1);
        assert_eq!(
            jacobi_hgcd(x.clone(), y.clone()),
            jacobi_lehmer(x, y),
            "unbalanced fallback diverged"
        );
        let y = full_width_odd(&mut rng, JACOBI_HGCD_THRESHOLD_LIMBS + 52);
        let x = y.sub_ref(&BigUint::from_u64(2));
        assert_eq!(
            jacobi_hgcd(x.clone(), y.clone()),
            jacobi_lehmer(x, y),
            "close-pair fallback diverged"
        );
    }

    #[test]
    #[ignore = "timing probe for the Lehmer/HGCD jacobi crossover; run with --ignored"]
    fn jacobi_hgcd_crossover_timing() {
        use super::{
            jacobi_hgcd_engine, jacobi_lehmer, JacobiState, JACOBI_LEHMER_THRESHOLD_LIMBS,
        };
        use std::hint::black_box;
        use std::time::Instant;
        let mut rng = SplitMix64 {
            state: 0xc0de_57a7_e0f0_a11e,
        };
        eprintln!(
            "{:>7} {:>12} {:>12}  winner",
            "limbs", "lehmer_ms", "hgcd_ms"
        );
        for &limbs in &[256usize, 512, 1024, 1536, 2048, 3072, 4096, 8192, 16384] {
            let bits = limbs * 64;
            let mut y = draw_below(&mut rng, &pow2(bits));
            y.set_bit(bits - 1);
            y.set_bit(0);
            let x = draw_below(&mut rng, &y);
            let time = |f: &dyn Fn()| {
                let mut best = f64::INFINITY;
                for _ in 0..3 {
                    let t0 = Instant::now();
                    f();
                    best = best.min(t0.elapsed().as_secs_f64() * 1e3);
                }
                best
            };
            let lehmer = time(&|| {
                black_box(jacobi_lehmer(x.clone(), y.clone()));
            });
            let state = JacobiState::new((x.limbs()[0] & 3) as u8, (y.limbs()[0] & 3) as u8);
            let hgcd_t = time(&|| {
                black_box(jacobi_hgcd_engine(
                    x.clone(),
                    y.clone(),
                    state,
                    JACOBI_LEHMER_THRESHOLD_LIMBS,
                ));
            });
            let winner = if lehmer <= hgcd_t { "lehmer" } else { "hgcd" };
            eprintln!("{limbs:7} {lehmer:12.3} {hgcd_t:12.3}  {winner}");
        }
    }

    #[test]
    fn hgcd_matrix_step_algebra() {
        use super::Mat2;
        let (a, b) = (BigUint::from_u64(240), BigUint::from_u64(46));
        let id = Mat2::identity();
        assert_eq!(id.apply(&a, &b), (a.clone(), b.clone()));
        // reduce_top by 5: a ← a − 5·b = 10, b unchanged.
        let m = id.reduce_top(&BigUint::from_u64(5));
        assert_eq!(
            m.apply(&a, &b),
            (BigUint::from_u64(10), BigUint::from_u64(46))
        );
        // reduce_bottom by 4: b ← b − 4·a' = 46 − 40 = 6.
        let m2 = m.reduce_bottom(&BigUint::from_u64(4));
        assert_eq!(
            m2.apply(&a, &b),
            (BigUint::from_u64(10), BigUint::from_u64(6))
        );
    }

    #[test]
    #[ignore = "timing probe for the Lehmer/HGCD crossover; run with --ignored"]
    fn hgcd_crossover_timing() {
        use super::{gcd_lehmer, gcd_via_hgcd};
        use std::hint::black_box;
        use std::time::Instant;
        let mut rng = SplitMix64 {
            state: 0x7a11_5eed_0dd5_ba11,
        };
        eprintln!(
            "{:>7} {:>12} {:>12}  winner",
            "limbs", "lehmer_ms", "hgcd_ms"
        );
        for &limbs in &[64usize, 128, 256, 512, 1024, 2048, 4096, 8192] {
            let bits = limbs * 64;
            let mut a = draw_below(&mut rng, &pow2(bits));
            let mut b = draw_below(&mut rng, &pow2(bits));
            a.set_bit(bits - 1);
            b.set_bit(bits - 1);
            let reps = (2048 / limbs).max(1);
            let time = |f: &dyn Fn() -> BigUint| {
                let mut best = f64::INFINITY;
                for _ in 0..2 {
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        black_box(f());
                    }
                    best = best.min(t0.elapsed().as_secs_f64() / reps as f64 * 1e3);
                }
                best
            };
            let lehmer = time(&|| gcd_lehmer(&a, &b));
            let hgcd_t = time(&|| gcd_via_hgcd(&a, &b));
            let winner = if lehmer <= hgcd_t { "lehmer" } else { "hgcd" };
            eprintln!("{limbs:7} {lehmer:12.3} {hgcd_t:12.3}  {winner}");
        }
    }

    #[test]
    #[ignore = "timing probe for the Lehmer/HGCD extended-gcd crossover; run with --ignored"]
    fn hgcd_ext_crossover_timing() {
        use super::{canonicalize_bezout, gcd_extended_lehmer, gcd_extended_via_hgcd};
        use std::hint::black_box;
        use std::time::Instant;
        let mut rng = SplitMix64 {
            state: 0xe87e_9d5a_11c0_ffee,
        };
        eprintln!(
            "{:>7} {:>12} {:>12}  winner",
            "limbs", "lehmer_ms", "hgcd_ms"
        );
        for &limbs in &[128usize, 256, 384, 512, 1024, 4096, 16384] {
            let bits = limbs * 64;
            let mut a = draw_below(&mut rng, &pow2(bits));
            let mut b = draw_below(&mut rng, &pow2(bits));
            a.set_bit(bits - 1);
            b.set_bit(bits - 1);
            let time = |f: &dyn Fn()| {
                let mut best = f64::INFINITY;
                for _ in 0..2 {
                    let t0 = Instant::now();
                    f();
                    best = best.min(t0.elapsed().as_secs_f64() * 1e3);
                }
                best
            };
            let lehmer = time(&|| {
                black_box(gcd_extended_lehmer(&a, &b));
            });
            let hgcd_t = time(&|| {
                let (g, s, _t) = gcd_extended_via_hgcd(&a, &b);
                black_box(canonicalize_bezout(&a, &b, &g, &s));
            });
            let winner = if lehmer <= hgcd_t { "lehmer" } else { "hgcd" };
            eprintln!("{limbs:7} {lehmer:12.3} {hgcd_t:12.3}  {winner}");
        }
    }

    #[test]
    fn hgcd_reduction_invariants() {
        use super::{abs_diff_bits, gcd_lehmer, hgcd, pair_min_size};
        use crate::bigint::BigInt;
        let mut rng = SplitMix64 {
            state: 0x5eed_cafe_f00d_babe,
        };
        let one = BigInt::from_biguint(BigUint::one());
        // Sizes span the batched base case (≤ 6144 bits) and the recursion
        // above it; reps taper as the sizes grow.
        for &(bits, reps) in &[
            (8usize, 120),
            (12, 120),
            (16, 120),
            (24, 120),
            (40, 120),
            (64, 120),
            (100, 120),
            (200, 60),
            (400, 60),
            (1000, 40),
            (3000, 20),
            (8000, 10),
            (16000, 6),
            (40000, 3),
        ] {
            for _ in 0..reps {
                // Both operands full width (top bit forced), satisfying hgcd's
                // precondition #min > ⌊bits/2⌋ + 1 — as the driver guarantees.
                let mut a = draw_below(&mut rng, &pow2(bits));
                let mut b = draw_below(&mut rng, &pow2(bits));
                a.set_bit(bits - 1);
                b.set_bit(bits - 1);
                let s = bits / 2 + 1;

                let (t, ra, rb) = hgcd(&a, &b, None);
                // The transform reproduces the reduced pair from the inputs.
                assert_eq!(
                    t.apply(&a, &b),
                    (ra.clone(), rb.clone()),
                    "transform inconsistent at {bits} bits"
                );
                // Postcondition: the pair straddles the boundary.
                assert!(pair_min_size(&ra, &rb) > s, "over-reduced at {bits} bits");
                assert!(abs_diff_bits(&ra, &rb) <= s, "under-reduced at {bits} bits");
                // gcd is preserved and the transform is unimodular. det is ±1,
                // not always +1: Möller's non-swapping steps are det +1, but
                // the Lehmer batches in the base case are products of swapping
                // Euclid steps (det −1 each), so an odd-length batch flips it.
                assert_eq!(gcd_lehmer(&ra, &rb), gcd_lehmer(&a, &b));
                let det = t.m00.mul_ref(&t.m11).sub_ref(&t.m01.mul_ref(&t.m10));
                assert_eq!(
                    det.magnitude(),
                    one.magnitude(),
                    "transform must be unimodular (det = ±1) at {bits} bits"
                );
            }
        }
    }

    #[test]
    fn gcd_via_hgcd_matches_lehmer() {
        use super::{gcd_lehmer, gcd_via_hgcd};
        let mut rng = SplitMix64 {
            state: 0x4859_2b17_ac3f_1d05,
        };
        // Sizes below the driver's Lehmer handoff, across the batched base
        // case, and through the recursion (dispatch is 64 limbs = 4096 bits;
        // the recursion engages above 96 limbs = 6144 bits).
        for &bits in &[
            130usize, 200, 256, 400, 512, 777, 1024, 1500, 2048, 3000, 4096, 5000, 8192, 16000,
            50000,
        ] {
            let bound = pow2(bits);
            for _ in 0..4 {
                let a = draw_below(&mut rng, &bound);
                let b = draw_below(&mut rng, &bound);
                assert_eq!(
                    gcd_via_hgcd(&a, &b),
                    gcd_lehmer(&a, &b),
                    "hgcd != lehmer at {bits} bits"
                );
            }
        }
        // Structured: 2^2000 against 2^2000 − 1 (coprime), and a shared factor.
        let big = pow2(2000);
        let odd = big.sub_ref(&BigUint::one());
        assert_eq!(gcd_via_hgcd(&big, &odd), gcd_lehmer(&big, &odd));
        let m = mersenne(1279);
        let shared = m.mul_ref(&BigUint::from_u64(6));
        let other = m.mul_ref(&BigUint::from_u64(10));
        assert_eq!(gcd_via_hgcd(&shared, &other), gcd_lehmer(&shared, &other));

        // Above the dispatch thresholds the public functions take the
        // Half-GCD paths; large cases pin those routes against the Lehmer
        // engines they must reproduce exactly.
        let bound = pow2(super::HGCD_THRESHOLD_LIMBS * 64 + 512);
        let a = draw_below(&mut rng, &bound);
        let b = draw_below(&mut rng, &bound);
        assert_eq!(
            gcd(&a, &b),
            gcd_lehmer(&a, &b),
            "public gcd dispatch diverged"
        );

        let ext_bound = pow2(super::HGCD_EXT_THRESHOLD_LIMBS * 64 + 512);
        let p = draw_below(&mut rng, &ext_bound);
        let q = draw_below(&mut rng, &ext_bound);
        assert_eq!(
            super::gcd_extended(&p, &q),
            super::gcd_extended_lehmer(&p, &q),
            "public gcd_extended dispatch diverged from the classical triple"
        );
        let mut n = p.clone();
        if !n.is_odd() {
            n = n.add_ref(&BigUint::one());
        }
        match mod_inverse(&q, &n) {
            Some(inverse) => {
                // The inverse is unique mod n; the identity is a complete check.
                assert_eq!(
                    BigUint::mod_mul(&q.modulo(&n), &inverse, &n),
                    BigUint::one(),
                    "public mod_inverse dispatch returned a non-inverse"
                );
            }
            None => assert!(
                !gcd(&q, &n).is_one(),
                "public mod_inverse dispatch missed an existing inverse"
            ),
        }
    }

    fn biguint_from_hex(hex: &str) -> BigUint {
        let padded = if hex.len() % 2 == 1 {
            format!("0{hex}")
        } else {
            hex.to_string()
        };
        let bytes: Vec<u8> = (0..padded.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&padded[i..i + 2], 16).expect("hex vector"))
            .collect();
        BigUint::from_be_bytes(&bytes)
    }

    /// Reference vectors computed by GMP 6.3.0's mpz_jacobi (see
    /// scripts/bench_gmp.sh for the toolchain): an independent oracle for the
    /// binary reciprocity algorithm. Triples are (a, n, (a/n)) with a and n in
    /// hex, spanning 8- to 1024-bit odd moduli, a below/at/above n, shared
    /// factors, and the (2/n) supplement cases.
    const GMP_JACOBI_VECTORS: &[(&str, &str, i8)] = &[
        ("8a", "bf", 1),
        ("f6ad", "bf", -1),
        ("be", "bf", -1),
        ("c0", "c1", 1),
        ("7e10", "c1", -1),
        ("c0", "c1", 1),
        ("53", "a9", 1),
        ("18a3", "a9", 1),
        ("a8", "a9", 1),
        ("32", "fb", -1),
        ("7110", "fb", 1),
        ("fa", "fb", -1),
        ("36", "db", 0),
        ("1bc9", "db", 0),
        ("da", "db", -1),
        ("a5e2", "8d27", 1),
        ("509c55fd", "8d27", -1),
        ("8d26", "8d27", -1),
        ("fd96", "9003", -1),
        ("19e41c1a", "9003", 1),
        ("9002", "9003", -1),
        ("8e6f", "ffbd", 1),
        ("7cfdf0cd", "ffbd", 0),
        ("ffbc", "ffbd", 1),
        ("229a", "a381", 1),
        ("66bcc7a2", "a381", -1),
        ("a380", "a381", 1),
        ("c30a", "a0c7", -1),
        ("e16d0717", "a0c7", 1),
        ("a0c6", "a0c7", -1),
        ("6d873e188a3b18af", "98976396c66cf253", -1),
        ("b81585e184058393b4e70630249faa41", "98976396c66cf253", 1),
        ("98976396c66cf252", "98976396c66cf253", -1),
        ("ec55142e656f893f", "e031880a1a92cd89", -1),
        ("d93ee354868a5658d359a944ff748928", "e031880a1a92cd89", 0),
        ("e031880a1a92cd88", "e031880a1a92cd89", 1),
        ("29ffabdbcb4a054", "d120bc92095cbe79", 0),
        ("c8122140efaf5fc236c6ce77b0b2bf68", "d120bc92095cbe79", 0),
        ("d120bc92095cbe78", "d120bc92095cbe79", 1),
        ("1365749bb30353c7", "a664b4d6936fd78f", 1),
        ("54e5f8197da36f2d88e051f28b6bf16c", "a664b4d6936fd78f", -1),
        ("a664b4d6936fd78e", "a664b4d6936fd78f", -1),
        ("d9c52e47915040ed", "b35a54c1cb764e05", 1),
        ("8802aa2cd2117e21c785a6949cab78e3", "b35a54c1cb764e05", -1),
        ("b35a54c1cb764e04", "b35a54c1cb764e05", 1),
        ("152a452713c13cf28", "11d0822f11e0a04f9", 0),
        ("2e39313c5e003a23ddb0ec24261cdea75", "11d0822f11e0a04f9", -1),
        ("11d0822f11e0a04f8", "11d0822f11e0a04f9", 1),
        ("10c82f8b0beeac785", "186a236eddb59c71d", -1),
        ("275d1f9e5e55cd14860166699856c1cb2", "186a236eddb59c71d", -1),
        ("186a236eddb59c71c", "186a236eddb59c71d", 1),
        ("1c86648969034aad7", "1cdc1ca9241c8c89f", 1),
        ("1cb550f0a9ec2f62bb89ce5ddc855a5b5", "1cdc1ca9241c8c89f", 0),
        ("1cdc1ca9241c8c89e", "1cdc1ca9241c8c89f", -1),
        ("10058c5f107e77202", "1b9fe84737e9ce40d", 1),
        ("3f312533cefb5bf51d382b7d7cd061dc0", "1b9fe84737e9ce40d", 0),
        ("1b9fe84737e9ce40c", "1b9fe84737e9ce40d", 1),
        ("1921ff6d416fb942c", "1c1f064b7a273494d", -1),
        ("1185aff0ecc65131f898a25cc3f5ee120", "1c1f064b7a273494d", -1),
        ("1c1f064b7a273494c", "1c1f064b7a273494d", 1),
        ("a9438fc24555f3ba25098f47a0c9900c", "9f7d286288234afda8fb40e225691723", -1),
        ("e4ec421914dfe4a5640b28bb6881fc7f8e289c18cfd6249db02d6883e765c44b", "9f7d286288234afda8fb40e225691723", -1),
        ("9f7d286288234afda8fb40e225691722", "9f7d286288234afda8fb40e225691723", -1),
        ("2e9c936bbaf896c50b5e6b167b1b9b4e", "aa61f09956d095dbd4b7468ec84e0f09", -1),
        ("986fceb34ef71509b8273d4449310beb4ec320b5c8da56fa93d62dec246e4fe3", "aa61f09956d095dbd4b7468ec84e0f09", -1),
        ("aa61f09956d095dbd4b7468ec84e0f08", "aa61f09956d095dbd4b7468ec84e0f09", 1),
        ("39f0134ade9c857546025fae1e69df0", "df4ea3ac476c5572a46d159b2243e3c3", -1),
        ("b1d0277f524cbac76ced302f001d147380c8c4742b4953bcc127113da2a5654d", "df4ea3ac476c5572a46d159b2243e3c3", 0),
        ("df4ea3ac476c5572a46d159b2243e3c2", "df4ea3ac476c5572a46d159b2243e3c3", -1),
        ("1d73044ce3c70a32aa25c14a375ba74d", "eefd0a7d626f5ad5293715d4703ae73b", -1),
        ("7f5a737dcdc5d2e0bd0fff8160b765080bb3bc5ba396eae348089aee98c35a68", "eefd0a7d626f5ad5293715d4703ae73b", -1),
        ("eefd0a7d626f5ad5293715d4703ae73a", "eefd0a7d626f5ad5293715d4703ae73b", -1),
        ("740b15bc6c4d79966d8a27bfd52bedf", "91ba073343b37c454b8cf8cf7e4f35c9", -1),
        ("a6aae2ae5a925cd4cc66688c84e300324fed8cf4c9206b1dbd5f674a3e070297", "91ba073343b37c454b8cf8cf7e4f35c9", -1),
        ("91ba073343b37c454b8cf8cf7e4f35c8", "91ba073343b37c454b8cf8cf7e4f35c9", 1),
        ("52aaff89c15602f4a88ea950b46abf84e54522d39ce8f27eec3974977e61ba8d", "923b5aa7d3b160ff8e18d442cc210d0ba1ec297bc3692111abc437e3e7b3e0db", -1),
        ("88a12277a1a579d30a67031c3caacf9cdf22ab6582f343320f46902a22bc4cb7376ad797f98f284733a8ccb24fd8e70550e85c2ad3ec06d8fd133637f2b93e65", "923b5aa7d3b160ff8e18d442cc210d0ba1ec297bc3692111abc437e3e7b3e0db", -1),
        ("923b5aa7d3b160ff8e18d442cc210d0ba1ec297bc3692111abc437e3e7b3e0da", "923b5aa7d3b160ff8e18d442cc210d0ba1ec297bc3692111abc437e3e7b3e0db", -1),
        ("789a5350f58ab019a1c9e51a4b0eefe608fe1ce5ce9099bb9e571491251427f1", "faa7fc7723c3c19c7295126de4aa0f813076731e0d96990475a63529ee19dcc5", 0),
        ("c269e9c35bd61340219ab4a1c6b173ae584f51f0756588eb5762a39b9a80d599afea93e3d209dbe35d27154ff743bc0f28bf4f564c6ebce34ddc8b6d56f3993f", "faa7fc7723c3c19c7295126de4aa0f813076731e0d96990475a63529ee19dcc5", -1),
        ("faa7fc7723c3c19c7295126de4aa0f813076731e0d96990475a63529ee19dcc4", "faa7fc7723c3c19c7295126de4aa0f813076731e0d96990475a63529ee19dcc5", 1),
        ("97ef3e2f6910b313fb1de73398a332b2157d9dae447d79cf85d58bf099d5e9ec", "a94549240d2d4ed2779f54ebd3ea0f915920f807abc2b00ab0a4473b9c40e9c9", 0),
        ("6f9f85fc79d6c631cb2dcf18c74eea09975d332f89a056208ae53efb17ee8f7455e28557ed30196c46977bd65e18a359c873750472125974db5d33b537727c41", "a94549240d2d4ed2779f54ebd3ea0f915920f807abc2b00ab0a4473b9c40e9c9", 0),
        ("a94549240d2d4ed2779f54ebd3ea0f915920f807abc2b00ab0a4473b9c40e9c8", "a94549240d2d4ed2779f54ebd3ea0f915920f807abc2b00ab0a4473b9c40e9c9", 1),
        ("c4152c3771950fe29dc8f7459471fed21d6f9c15b94d5211b96940d71e2689b0", "88add2d7cba103dc87c7498dbd8d37e947a0644a907c123f22a2bdcccff02aff", -1),
        ("d094982cf17c83358ae3114b0bbc7c7b9bca1e85708b0a5fbd92479844ca562158abdd0c93f3b0028eeb436021bc064093b881877c366831e827f2b59bf34a4a", "88add2d7cba103dc87c7498dbd8d37e947a0644a907c123f22a2bdcccff02aff", 1),
        ("88add2d7cba103dc87c7498dbd8d37e947a0644a907c123f22a2bdcccff02afe", "88add2d7cba103dc87c7498dbd8d37e947a0644a907c123f22a2bdcccff02aff", -1),
        ("933086656e1af289d42f23adbe850f8495b758176826a20b6cd87f29ace65c91", "a8cb19945ea56c44e14585fe276fa8cef4c2bc27ffa33d1d290a1141783e891b", -1),
        ("9ce2a2929ae038d6ebbd86048cc8fb2599e90ee288a040385fcd6523fd3bdc210572431aba1611d9a991d585e819a4c94f109e64cb8dae5ede9e2acd7d83332d", "a8cb19945ea56c44e14585fe276fa8cef4c2bc27ffa33d1d290a1141783e891b", -1),
        ("a8cb19945ea56c44e14585fe276fa8cef4c2bc27ffa33d1d290a1141783e891a", "a8cb19945ea56c44e14585fe276fa8cef4c2bc27ffa33d1d290a1141783e891b", -1),
        ("17d4de0e6921e6da75d313b33c8b03e79df71f7e99e271115b2d299aa6c5e9953b50321429f39ff48a7abd8489e1781306bebd684276fa1c44319d481d8d70f1688", "1a848bc61fb55346c87fc5a6e01f3ff1a39545f9fe5cfa9f4f2418cb3bceacd9c5e5330e76cf0a1fff115ea3316c538bfcb5a7811f7706ddac73ff72219ed8324cd", 1),
        ("1937a228664f6e2560b0c7b2628854d08ed07663dba0ab47264c70d6c39bf3f391dbdab37db4eb3fb46a7b05399884d9ceb5040a641d83bf7a4785d103bf60de2f46f6bb01520940332ebc227184127d34ff745e273cb30ea6c35bcd24447c41f8c23254891531e7f42ed3a22589cf49a9eb53d283a19149b65d70d51cf256e172fd7", "1a848bc61fb55346c87fc5a6e01f3ff1a39545f9fe5cfa9f4f2418cb3bceacd9c5e5330e76cf0a1fff115ea3316c538bfcb5a7811f7706ddac73ff72219ed8324cd", 1),
        ("1a848bc61fb55346c87fc5a6e01f3ff1a39545f9fe5cfa9f4f2418cb3bceacd9c5e5330e76cf0a1fff115ea3316c538bfcb5a7811f7706ddac73ff72219ed8324cc", "1a848bc61fb55346c87fc5a6e01f3ff1a39545f9fe5cfa9f4f2418cb3bceacd9c5e5330e76cf0a1fff115ea3316c538bfcb5a7811f7706ddac73ff72219ed8324cd", 1),
        ("a29602617c33a9bca435546f7a1b475e691a63f90caffa41a39a9f2b2ed54b535ad1ba14f8eb9075def2932e9b9a36b1b7f5baff5bf3e130c564dce07bbc1de961", "1c7884db9121df2d851c507756c187fab4b1131ee1ee188d8f38a08e6e057654fd0eba10673c0a72d41d8111c2e5d242e9f3db9451b45667a08f60e3e8aa63394ff", 1),
        ("85179c2362148bfb3ce9b0be61475461080349e33696ec7ebf7ad9637b563f7013c58898632a3366abd6f82ba5e26dd512c360942a5943deff8786b56caed03052c1ef55e35ad5d84c7b6dfaf237a68482fc7817ed98234712413f6ad271a349cfa6172b931a66a65ced240f3d87294a26c2e3b78bda9c7e4e730ba83b0f0752e6fc", "1c7884db9121df2d851c507756c187fab4b1131ee1ee188d8f38a08e6e057654fd0eba10673c0a72d41d8111c2e5d242e9f3db9451b45667a08f60e3e8aa63394ff", 1),
        ("1c7884db9121df2d851c507756c187fab4b1131ee1ee188d8f38a08e6e057654fd0eba10673c0a72d41d8111c2e5d242e9f3db9451b45667a08f60e3e8aa63394fe", "1c7884db9121df2d851c507756c187fab4b1131ee1ee188d8f38a08e6e057654fd0eba10673c0a72d41d8111c2e5d242e9f3db9451b45667a08f60e3e8aa63394ff", -1),
        ("b27d4208b8f77f7e1a61b13ee0579a5caf01ff8023bd92369885b0e97d44eb9c21ccfc6247f712b2b8efb191a832730770c5e9d74c7c4ec1d064decb334984c78c", "10bada040e275c63cb510bb620058ff52cd9c0b1984b293de8b27267d017e0118549980c5a14e570f9d2bae091a3ae32101e8890502ba6c762ac052efae0e524ceb", 1),
        ("2fcdec8864cf66f0e83fbcc94ac19e99a2ed4cbb9073aeefbe0e2f538a804e611b9e7b61781727a7ec1eef061026e5dd0b985e12f65c0845f4a45db18fc231931d8c17c432418205d23506dc0ab70edc27d0175b797f3b567551e9d6167c88ee221e8263c579488ec45707b2ab4f472aba0ad4f0c6c15afe525920a86b41e2fcf3114", "10bada040e275c63cb510bb620058ff52cd9c0b1984b293de8b27267d017e0118549980c5a14e570f9d2bae091a3ae32101e8890502ba6c762ac052efae0e524ceb", 1),
        ("10bada040e275c63cb510bb620058ff52cd9c0b1984b293de8b27267d017e0118549980c5a14e570f9d2bae091a3ae32101e8890502ba6c762ac052efae0e524cea", "10bada040e275c63cb510bb620058ff52cd9c0b1984b293de8b27267d017e0118549980c5a14e570f9d2bae091a3ae32101e8890502ba6c762ac052efae0e524ceb", -1),
        ("81aa611e687bb0805671a0343c0a63d848d96785f5cddf4200b31c897a8e35d60a25275bf05232e71d6bc90a12532a042cf3d0d10c6ba3ddaf0fa6726953e27aea", "1bd800115259fdd8d5f446716cd292647332e01cadf5d1b044e663dfa862bdc14e4a80fe76e6ccd62d81a66a920bf1e5072ccfe49d9c10a55ce112e512fe2ce48b5", -1),
        ("68c2528e9b792684f96da6d0a7a4e3fd5bedaf6241708679189699e8d40f378855f2710d136aa4deb549981b3c2e62fdc93d3fd7089ba5391ae47c8ad025707973fa692f1c1344fb9564547a13c3a56bf3f1db749ff7405124fb2b5903617ec4b8509659882be92d9e0ce4ce91468b611c177272471ab317c4d1d7c179dc13495d2c", "1bd800115259fdd8d5f446716cd292647332e01cadf5d1b044e663dfa862bdc14e4a80fe76e6ccd62d81a66a920bf1e5072ccfe49d9c10a55ce112e512fe2ce48b5", -1),
        ("1bd800115259fdd8d5f446716cd292647332e01cadf5d1b044e663dfa862bdc14e4a80fe76e6ccd62d81a66a920bf1e5072ccfe49d9c10a55ce112e512fe2ce48b4", "1bd800115259fdd8d5f446716cd292647332e01cadf5d1b044e663dfa862bdc14e4a80fe76e6ccd62d81a66a920bf1e5072ccfe49d9c10a55ce112e512fe2ce48b5", 1),
        ("16ac28fbeb8e9fa3e408aad1f0687c8b204fa8459265817b9fe05f29c8f4cb3b4a81128e02e4a961a8931a18f742cbf0b7345845609b5cfaa0e6445935d01717d34", "197f69d016a44ac2c2b3f33a9cbcc59f0fc27e0aa59f72ff1c1e4201467eeb251976abe8c60445644eb272ea1d423526a672a28ceb092ee24f08b4f39e712984875", 1),
        ("261e7d6d20ba6b4abc39fae0771a060d37d37bf48482e639ba2f391d006f242a62d0e05c1d28dad24638be1e2e15b3568199b8b07941e901df6ab150bddda43756c60dc88317d03e3969cae4bc14097f0398aa537cf5e70419cbfb295b2616f82c2aa2ee780db6b02dd4f652971a3ef1bad6e2ff7814240c2523f2ae415df4f5bdbeb", "197f69d016a44ac2c2b3f33a9cbcc59f0fc27e0aa59f72ff1c1e4201467eeb251976abe8c60445644eb272ea1d423526a672a28ceb092ee24f08b4f39e712984875", -1),
        ("197f69d016a44ac2c2b3f33a9cbcc59f0fc27e0aa59f72ff1c1e4201467eeb251976abe8c60445644eb272ea1d423526a672a28ceb092ee24f08b4f39e712984874", "197f69d016a44ac2c2b3f33a9cbcc59f0fc27e0aa59f72ff1c1e4201467eeb251976abe8c60445644eb272ea1d423526a672a28ceb092ee24f08b4f39e712984875", 1),
        ("53201e0f5c468d338127426ab27d05defa5f02e918be973a97d6313dd0d25db829e7f5790b9bbf4457b83b1909bb1be0a87a6d9b37cd20f80d57d90e9f84eb90c50c71e1237ad6643da295612f7566c740f4af6b6081692001dc635d1c6f8a96d7b49acc59e8bd64bf9ed5893607f5e5ba0111b427513417ee2b21598ee9fdfb", "9b7b9b81fc01fec3271906924bca2b208706aa7d20d1a7170ecfe68258e6980a4bd9f5a5c7be65fad55b8b9d197bdcc8d3b68c0f4a2f2c68873bf95d21ad185f73163bf57394a9b237be271865c2af8afe72caa22329037de24513510492c063fad5cb2e20cb51d035db09801a74886ccd897ba62139283595ff2255b73cc11f", 1),
        ("4d38bb7b9bb8b762f8ed62417a9685531dba5434d7f4b6c738d8e99d0e8c6496c16b182b5df84ca47248428d37c42b2cf01e8d4002de7b0c75ba59c9014246bae8f9752bf48332f38fd982630cd1e1c37b5cb3ad845d4640d131044066f672ce1bb90527aaa19549240b6779fb6c223bbacbbf381f17cd9260bc795f77cb82c7132fd14bec2f2c4528bfaaa393a178a67004d43c54f97831c98a60e456ebdefaf392dd19112e47c19f78926504f54693c14deb5bb7670df590f271439188deb3ca0d6615a299edae130de3893255c242e5b3faa6d7991dc7ad222ecda559705b3beaf902c6fafc4dee7c1faffe684a474eb955bcb8fd3083bc89740d93c537f2", "9b7b9b81fc01fec3271906924bca2b208706aa7d20d1a7170ecfe68258e6980a4bd9f5a5c7be65fad55b8b9d197bdcc8d3b68c0f4a2f2c68873bf95d21ad185f73163bf57394a9b237be271865c2af8afe72caa22329037de24513510492c063fad5cb2e20cb51d035db09801a74886ccd897ba62139283595ff2255b73cc11f", 1),
        ("9b7b9b81fc01fec3271906924bca2b208706aa7d20d1a7170ecfe68258e6980a4bd9f5a5c7be65fad55b8b9d197bdcc8d3b68c0f4a2f2c68873bf95d21ad185f73163bf57394a9b237be271865c2af8afe72caa22329037de24513510492c063fad5cb2e20cb51d035db09801a74886ccd897ba62139283595ff2255b73cc11e", "9b7b9b81fc01fec3271906924bca2b208706aa7d20d1a7170ecfe68258e6980a4bd9f5a5c7be65fad55b8b9d197bdcc8d3b68c0f4a2f2c68873bf95d21ad185f73163bf57394a9b237be271865c2af8afe72caa22329037de24513510492c063fad5cb2e20cb51d035db09801a74886ccd897ba62139283595ff2255b73cc11f", -1),
        ("98776f863c74a863b859270a498cd465f35fbd549c1c1cadf1f2f01919db2287f94a9b7b2d8f078f03ed80246f7f1cd1218384a8df73432530f139d9e3f141b8cf98a159109f035a83e5384ad37de0e792f669b89a189a43ae150b21a007786067b78e84bb04c689035e1fac0ae7739fa24666a31c9f721c8b89e8bf17ecea09", "cffa14009aafb8653889ea2540d706b76270e9d117f2bf135581ec1c98a2f2515a09c7bdce43beab643ef15d5134bd950f97f26f8d7115e3326ec7489abe780d64c66b847eb3ddad05d76d1f4d9a79740b2ce06847ae562a6ce41913aa5c94eae15b18e35152e269f0bf47c30d8c14d71208af84403946e4d2bb96b317f7c88b", 0),
        ("aaba65380cda8fbff9d381278536de363e4b7513409e9443ddd86a1dd7facc7cfa429d5cd59a7a7b5cb6ad2396bed3ebb7d6c27cf7986c4846086d8b9d3743285f19b66c71d12149ada06504a8a148c633c65191662942b0624cf777fb76517720b096eac25cadc9a745a0949a4b8d7d421b7b4e4c07bdd6e32a056ade9660181e36b661168b1d83021f181b856d576ad64aa068d56e22d85af15be0611814039ee29f4c4df81eafbc82100628549716e839d053684fed1f32f1059b5b78287db19ceb1600073d1cb0b65bc4d77bcef5b02562f5a305a4aae00e676394fe413c9c8e9330f62c9ce6193cbc3fe572cb6f12175d04aafb4637875fc60ca83e2231", "cffa14009aafb8653889ea2540d706b76270e9d117f2bf135581ec1c98a2f2515a09c7bdce43beab643ef15d5134bd950f97f26f8d7115e3326ec7489abe780d64c66b847eb3ddad05d76d1f4d9a79740b2ce06847ae562a6ce41913aa5c94eae15b18e35152e269f0bf47c30d8c14d71208af84403946e4d2bb96b317f7c88b", 1),
        ("cffa14009aafb8653889ea2540d706b76270e9d117f2bf135581ec1c98a2f2515a09c7bdce43beab643ef15d5134bd950f97f26f8d7115e3326ec7489abe780d64c66b847eb3ddad05d76d1f4d9a79740b2ce06847ae562a6ce41913aa5c94eae15b18e35152e269f0bf47c30d8c14d71208af84403946e4d2bb96b317f7c88a", "cffa14009aafb8653889ea2540d706b76270e9d117f2bf135581ec1c98a2f2515a09c7bdce43beab643ef15d5134bd950f97f26f8d7115e3326ec7489abe780d64c66b847eb3ddad05d76d1f4d9a79740b2ce06847ae562a6ce41913aa5c94eae15b18e35152e269f0bf47c30d8c14d71208af84403946e4d2bb96b317f7c88b", -1),
        ("9bc69b5c956a1324ddddd1d93a3e2b7134b3809ec28c2ebb94b935243a3afa9704a6d1ebab260dbcc7cfcbda9f58cc097c9d97aa5e719cd55551e47083d6993a0fb9549aac43a4bfcf7956b4932c83adbb3f5b7229025a345279dd42e184032ec167811d19f5f5c3db16ff063789c586c3e9580d371bd16fdb91967d43559420", "dfd434e12b6c201280b5a2c4a84aa2586e87172873a3d6e0f3590c1ce25b4b5ffd44c3356a69c2f8388851d0db87e7548b16c064cb5a64235d604a9e87f01c93bded1b75891a885723f00e66ab1825f7dbf58c04d7da131f26c9b563d5459ceb0c9b0b110af70c767ab8b47a4029bf138275db973c072e640956aed70624f55f", -1),
        ("8ef67abc4b95368f5788bcc0ce23acb9121d307ce500562b8e48df3f8fc1ccbf87d3113f58ef99a61a2c1d9235193c26dd6a1713ea4d30af6e760a034f4221a5f10ad5faaa55fcd63b7e3d16359075fffc7bd96e327263eda214dd456862292893615a0797db168a7e93ddef75bb0bfea31047ca93e62f08dc3ca5681bf5d242018491b16a1507bd9beb3542dc1fa08130969a2be83c72ce67b3fa37d1a43e53c670a88f0eb7e331cd393c4de0caa897b6bc519c02ed51e57d60666dbc305dac7870588fec7d15a4c0cfbfb48685006a27cefeb06d825a7253a812027e7f5a94b6e81f50784a9a9159406727ac4de65b42d280b0a1e07351d093c894bf0f8d3a", "dfd434e12b6c201280b5a2c4a84aa2586e87172873a3d6e0f3590c1ce25b4b5ffd44c3356a69c2f8388851d0db87e7548b16c064cb5a64235d604a9e87f01c93bded1b75891a885723f00e66ab1825f7dbf58c04d7da131f26c9b563d5459ceb0c9b0b110af70c767ab8b47a4029bf138275db973c072e640956aed70624f55f", 0),
        ("dfd434e12b6c201280b5a2c4a84aa2586e87172873a3d6e0f3590c1ce25b4b5ffd44c3356a69c2f8388851d0db87e7548b16c064cb5a64235d604a9e87f01c93bded1b75891a885723f00e66ab1825f7dbf58c04d7da131f26c9b563d5459ceb0c9b0b110af70c767ab8b47a4029bf138275db973c072e640956aed70624f55e", "dfd434e12b6c201280b5a2c4a84aa2586e87172873a3d6e0f3590c1ce25b4b5ffd44c3356a69c2f8388851d0db87e7548b16c064cb5a64235d604a9e87f01c93bded1b75891a885723f00e66ab1825f7dbf58c04d7da131f26c9b563d5459ceb0c9b0b110af70c767ab8b47a4029bf138275db973c072e640956aed70624f55f", -1),
        ("6d711a6962b72ca27cd7c741d69d69902c0d2ef908f4911b2f0579b175c39f1d648fb796bf3097c316f5a7f42da363ad8efca78b860f780b97f9a6208e8951d5fcdf1b31cb875fbe8bd58f3e50afc54ffe8d06182d5164db93f58749d03cae674d0a3765c366e66e251d0dba7adb1ee30083779c0725e3af1c8d9f591f59a53d", "a20295f47f3d794b692ddee446afbc48389118927fc50f2c058d319df1a2b889ec2617bf16934f2df25a6a491000dc629b89d729e1507b4624ea2ea9d77391e17cffb7067f9974f6971ee2f181832ae5eecd18705696ecc75888196b67e76790254b15c22a26d4400e199860bc712c6dafdf0e325747db884e4c958a98139717", 1),
        ("17963ddf237ce0ecc4da248b44f666dd74bfe36f1f6bc837d75616e9ae0e52eae7e4c051429004ef1c88fea3cf922e0448fde6d48a4c52107035489c8e5cbafbee0cd0873f40aeb51e7586a7e8ca5759e4f61608fb9d456248d6f03414cd42141a02567b0acb6ced83fdd64dd3f06bee6b5ad4a27378ebe69f26b165fa3cac643465ec905145a8612fe10d27763e0bdc7808dccc18b344c90b92e482a8a2ab6912152f45eb34e46e21cfd49438656d68b43ba9a4b9f6d77d6d45a2240ec28dbdacf1421f95f83b7b6fcccd1f52e2049f7a973b64d28fc0d999d2edb572411166e68ba95f55020d7c729e979b745c47b8cfd930afb8fb9fb536f5646213e15c2c", "a20295f47f3d794b692ddee446afbc48389118927fc50f2c058d319df1a2b889ec2617bf16934f2df25a6a491000dc629b89d729e1507b4624ea2ea9d77391e17cffb7067f9974f6971ee2f181832ae5eecd18705696ecc75888196b67e76790254b15c22a26d4400e199860bc712c6dafdf0e325747db884e4c958a98139717", 1),
        ("a20295f47f3d794b692ddee446afbc48389118927fc50f2c058d319df1a2b889ec2617bf16934f2df25a6a491000dc629b89d729e1507b4624ea2ea9d77391e17cffb7067f9974f6971ee2f181832ae5eecd18705696ecc75888196b67e76790254b15c22a26d4400e199860bc712c6dafdf0e325747db884e4c958a98139716", "a20295f47f3d794b692ddee446afbc48389118927fc50f2c058d319df1a2b889ec2617bf16934f2df25a6a491000dc629b89d729e1507b4624ea2ea9d77391e17cffb7067f9974f6971ee2f181832ae5eecd18705696ecc75888196b67e76790254b15c22a26d4400e199860bc712c6dafdf0e325747db884e4c958a98139717", -1),
        ("7c453669df9f75cf0c9ead528695ccdcc15e6002f6b32f76e740fec6925db2e1a3a57495a04e67ec52ae2a8216f7655595a60d6ea0b639c1361552f137056a9684e56e6f3469cabc9ca2a6a0e90ddb4bbac8d95c6271191b939e3c4b590e243c8a78cf96b55e754e5ad691181c42bfda894a7dea065253ed1eea627badcd9d80", "ba00833faf06a90d15f73f79591f000e8db2ccce78b559c36a203e19dac00ffe5f371987ee4daf0b81affddc679a21d9252b6e7cee92f74e6133a1495536cc1414f72cfc8f72cf6b10a9701dfb82582a6ed824cfdda0e5924af94778f098963836c911b565c17da6333960b1d8b353cea4d6b25e379fb5f0ff4bb00781309f2d", 1),
        ("d37c1c26a89e630919e5ec863222e184849fc23b067dcf7fcaff6564d9dd5001d43f294720289285f0407e3ab4fab556d247cb949118191d92c6361513bedd85de1b7f6bb5f51b79a431cbe777dd3ded2466f2207dead1abc6a7a19ef0635fb43ce03c8a4cf0b4a35425baa714b497d5b3974cefe6fdafd71edbed732318b4fb462046f5ac98acc37c8693ec2987f1b773b5e702536c37abeb07b7dc1bda0131f640f2926ac6d39487d4b6f87a262d1083736bd4af80f445386910c61ed634bb9ca10980e3e6c29e5373cbe942a506a471c5f597938358e0fc62381faaa89ae2d6e17d5c8e2f5ebcb3b14096c091c97147aa1d0abe793ac030c78427073969c2", "ba00833faf06a90d15f73f79591f000e8db2ccce78b559c36a203e19dac00ffe5f371987ee4daf0b81affddc679a21d9252b6e7cee92f74e6133a1495536cc1414f72cfc8f72cf6b10a9701dfb82582a6ed824cfdda0e5924af94778f098963836c911b565c17da6333960b1d8b353cea4d6b25e379fb5f0ff4bb00781309f2d", -1),
        ("ba00833faf06a90d15f73f79591f000e8db2ccce78b559c36a203e19dac00ffe5f371987ee4daf0b81affddc679a21d9252b6e7cee92f74e6133a1495536cc1414f72cfc8f72cf6b10a9701dfb82582a6ed824cfdda0e5924af94778f098963836c911b565c17da6333960b1d8b353cea4d6b25e379fb5f0ff4bb00781309f2c", "ba00833faf06a90d15f73f79591f000e8db2ccce78b559c36a203e19dac00ffe5f371987ee4daf0b81affddc679a21d9252b6e7cee92f74e6133a1495536cc1414f72cfc8f72cf6b10a9701dfb82582a6ed824cfdda0e5924af94778f098963836c911b565c17da6333960b1d8b353cea4d6b25e379fb5f0ff4bb00781309f2d", 1),
        ("0", "1", 1),
        ("5", "1", 1),
        ("0", "f", 0),
        ("1", "f", 1),
        ("2", "7", 1),
        ("2", "9", 1),
        ("2", "b", -1),
        ("2", "d", -1),
        ("2", "f", 1),
        ("3", "c9", 0),
        ("ff", "ff", 0),
        ("10001", "fffffffffffffffb", 1),
    ];

    #[test]
    fn jacobi_matches_gmp_vectors() {
        for &(a_hex, n_hex, expected) in GMP_JACOBI_VECTORS {
            let a = biguint_from_hex(a_hex);
            let n = biguint_from_hex(n_hex);
            assert_eq!(
                jacobi(&a, &n),
                Some(expected),
                "jacobi({a_hex}, {n_hex}) != {expected}"
            );
        }
    }

    #[test]
    fn jacobi_matches_euler_criterion_on_primes() {
        // For odd prime p the Jacobi symbol is the Legendre symbol, and
        // Euler's criterion computes it independently: a^((p-1)/2) mod p is
        // 1 for residues, p-1 for non-residues, 0 when p | a. mod_pow rides
        // the Montgomery kernels, a code path disjoint from jacobi's
        // shift-and-reciprocity loop. The large primes are the Mersenne
        // primes M89, M107, M127.
        let mut rng = SplitMix64 { state: 0x3c3c_3c3c };
        let mut primes: Vec<BigUint> = [3u64, 5, 7, 11, 13, 65_537, 2_147_483_647]
            .iter()
            .map(|&p| BigUint::from_u64(p))
            .collect();
        primes.push(mersenne(89));
        primes.push(mersenne(107));
        primes.push(mersenne(127));

        for p in &primes {
            let exponent = p.sub_ref(&BigUint::one());
            let mut half = exponent.clone();
            half.shr_bits(1);
            for _ in 0..12 {
                let a = draw_below(&mut rng, p);
                let euler = mod_pow(&a, &half, p);
                let expected = if euler.is_one() {
                    1
                } else if euler == p.sub_ref(&BigUint::one()) {
                    -1
                } else {
                    assert!(euler.is_zero(), "Euler criterion out of range");
                    0
                };
                assert_eq!(jacobi(&a, p), Some(expected), "disagrees for p={p:?}");
            }
        }
    }

    #[test]
    fn jacobi_is_multiplicative_and_periodic() {
        let mut rng = SplitMix64 { state: 0x5151_5151 };
        let bound = BigUint::from_u128(1 << 80);
        for _ in 0..40 {
            let mut n1 = draw_below(&mut rng, &bound);
            let mut n2 = draw_below(&mut rng, &bound);
            if !n1.is_odd() {
                n1 = n1.add_ref(&BigUint::one());
            }
            if !n2.is_odd() {
                n2 = n2.add_ref(&BigUint::one());
            }
            let a = draw_below(&mut rng, &n1);
            let b = draw_below(&mut rng, &n1);

            // Multiplicative in the top argument: (ab/n) = (a/n)(b/n).
            assert_eq!(
                jacobi(&a.mul_ref(&b), &n1).unwrap(),
                jacobi(&a, &n1).unwrap() * jacobi(&b, &n1).unwrap()
            );
            // Multiplicative in the bottom: (a/n1n2) = (a/n1)(a/n2).
            assert_eq!(
                jacobi(&a, &n1.mul_ref(&n2)).unwrap(),
                jacobi(&a, &n1).unwrap() * jacobi(&a, &n2).unwrap()
            );
            // Periodic in the top argument: (a + n / n) = (a/n).
            assert_eq!(jacobi(&a.add_ref(&n1), &n1), jacobi(&a, &n1));
        }
    }

    #[test]
    fn jacobi_two_supplement_follows_n_mod_8() {
        // (2/n) = +1 for n = 1, 7 (mod 8) and -1 for n = 3, 5 (mod 8).
        let two = BigUint::from_u64(2);
        for n in (3u64..200).step_by(2) {
            let expected = match n % 8 {
                1 | 7 => 1,
                3 | 5 => -1,
                _ => unreachable!("n is odd"),
            };
            assert_eq!(jacobi(&two, &BigUint::from_u64(n)), Some(expected));
        }
    }

    #[test]
    fn jacobi_edge_cases() {
        // Undefined for even or zero moduli.
        assert_eq!(jacobi(&BigUint::from_u64(3), &BigUint::zero()), None);
        assert_eq!(jacobi(&BigUint::from_u64(3), &BigUint::from_u64(10)), None);
        // Empty-product convention and the zero cases.
        assert_eq!(jacobi(&BigUint::zero(), &BigUint::one()), Some(1));
        assert_eq!(jacobi(&BigUint::from_u64(9), &BigUint::one()), Some(1));
        assert_eq!(jacobi(&BigUint::zero(), &BigUint::from_u64(9)), Some(0));
        assert_eq!(
            jacobi(&BigUint::from_u64(21), &BigUint::from_u64(7)),
            Some(0)
        );
    }

    /// Reference vectors computed by GMP 6.3.0's mpz_kronecker: even moduli,
    /// powers of two, n = 0, shared factors, and odd moduli where the symbol
    /// must agree with jacobi. Triples are (a, n, (a/n)) in hex.
    const GMP_KRONECKER_VECTORS: &[(&str, &str, i8)] = &[
        ("6", "17", 1),
        ("cf3", "17", 1),
        ("45", "17", 0),
        ("1e", "3e", 0),
        ("ada", "3e", 0),
        ("ba", "3e", 0),
        ("34", "1c", 0),
        ("fb9", "1c", 0),
        ("54", "1c", 0),
        ("2c", "10", 0),
        ("c51", "10", 1),
        ("30", "10", 0),
        ("8", "3", -1),
        ("70a", "3", -1),
        ("9", "3", 0),
        ("39", "b", -1),
        ("1c0", "b", -1),
        ("21", "b", 0),
        ("6d90", "c736", 0),
        ("bae978da", "c736", 0),
        ("255a2", "c736", 0),
        ("db27", "6fd", 1),
        ("9e967592", "6fd", -1),
        ("14f7", "6fd", 0),
        ("5d42", "6e22", 0),
        ("a5ae927c", "6e22", 0),
        ("14a66", "6e22", 0),
        ("b40d", "555e", 1),
        ("ce98547c", "555e", 0),
        ("1001a", "555e", 0),
        ("9332", "a1fe", 0),
        ("1ca2ec71", "a1fe", 1),
        ("1e5fa", "a1fe", 0),
        ("130f", "31cd", 1),
        ("985d392b", "31cd", -1),
        ("9567", "31cd", 0),
        ("3f685d69a27c4277", "87f2e0f5425bb044", -1),
        ("efeeda2034df7293910e9262ad1e6324", "87f2e0f5425bb044", 0),
        ("197d8a2dfc71310cc", "87f2e0f5425bb044", 0),
        ("7e5b2efcb3eb358a", "94da51efee649565", -1),
        ("8f1206ac54d871b88990a259dc7e637b", "94da51efee649565", 1),
        ("1be8ef5cfcb2dc02f", "94da51efee649565", 0),
        ("e061171c0efb20ce", "63d994a4e44ccc9e", 0),
        ("7f697669a3f64ad1cbb2f5b96b80554f", "63d994a4e44ccc9e", -1),
        ("12b8cbdeeace665da", "63d994a4e44ccc9e", 0),
        ("64a4822aa6413040", "b13284b0dc864c5b", 1),
        ("c61af9eb0a0078a9883c565cf691b38e", "b13284b0dc864c5b", -1),
        ("213978e129592e511", "b13284b0dc864c5b", 0),
        ("b4dac7c239161d25", "dc194c6c3dc9cd56", -1),
        ("1699a669fb123d0a0ab7e8695ff6a80f", "dc194c6c3dc9cd56", -1),
        ("2944be544b95d6802", "dc194c6c3dc9cd56", 0),
        ("49df346299ec61f7", "d77a518a4be94a5b", 0),
        ("47d58925813818a4c762c44a1aab5bf6", "d77a518a4be94a5b", 0),
        ("2866ef49ee3bbdf11", "d77a518a4be94a5b", 0),
        ("101f1993bf2963630", "165cc57e58623a33c", 0),
        ("350d583eb038998ff5fb24ff5ea90b280", "165cc57e58623a33c", 0),
        ("4316507b0926ae9b4", "165cc57e58623a33c", 0),
        ("144934acc99280bb9", "19e0ca6a295dad77b", 1),
        ("2b6402e1b144d3aa194b5ef19bd982f1e", "19e0ca6a295dad77b", 1),
        ("4da25f3e7c1908671", "19e0ca6a295dad77b", 0),
        ("2c3adb65ee0b57a2", "1e9be65f0d28deab", 1),
        ("efdadec2a7c95cdb0678454cabec4511", "1e9be65f0d28deab", -1),
        ("5bd3b31d277a9c01", "1e9be65f0d28deab", 0),
        ("144821bfa9f47fef6", "77ba84608e7afeb7", 1),
        ("391039ac9b2bebeeca24791827afbc976", "77ba84608e7afeb7", 1),
        ("1672f8d21ab70fc25", "77ba84608e7afeb7", 0),
        ("c999c6b3bbe5cbfd", "14f36c9ff5702c155", -1),
        ("18bede79858f97116f826c3cf1334009b", "14f36c9ff5702c155", -1),
        ("3eda45dfe050843ff", "14f36c9ff5702c155", 0),
        ("106e15e9eea842980", "7ef417fe62a04c6b", -1),
        ("37334c3d20667cf8ded312b94bfe17bfd", "7ef417fe62a04c6b", 1),
        ("17cdc47fb27e0e541", "7ef417fe62a04c6b", 0),
        ("3c40763dbac10a14ab013fc08fe74607", "4b308a1718a394139be94394b6afcf1f", 0),
        ("a728a8d5e99ccea6e36ddd4a377aba7db4b71920ee282206d79eca83fe125d2f", "4b308a1718a394139be94394b6afcf1f", 1),
        ("e1919e4549eabc3ad3bbcabe240f6d5d", "4b308a1718a394139be94394b6afcf1f", 0),
        ("58d7c70762808b0384eafc0fd8b1df37", "6c86107922a00457b6ecf1675849a8da", -1),
        ("6e6f35ade10acc0a7bfe5e12c825edff9133479b9463967a540ea9c14e46d2c1", "6c86107922a00457b6ecf1675849a8da", 1),
        ("14592316b67e00d0724c6d43608dcfa8e", "6c86107922a00457b6ecf1675849a8da", 0),
        ("5f4e35fa792c66c80f6f8b99d6dbd9e7", "4393c8b1591eee3e286c4a17fe4a35d4", 0),
        ("c13b98ba7e416750d3a7952a89cf7707ebd0da3b131d9a7b0baefc165c819c7c", "4393c8b1591eee3e286c4a17fe4a35d4", 0),
        ("cabb5a140b5ccaba7944de47fadea17c", "4393c8b1591eee3e286c4a17fe4a35d4", 0),
        ("264daed76f230a44bde7082dd26fff76", "9de164425c4fec5b1f04abe97289691a", 0),
        ("33111102bf3ed8a153bca328dc7d3e14ffdf7009e63684c65970aa3699427757", "9de164425c4fec5b1f04abe97289691a", 1),
        ("1d9a42cc714efc5115d0e03bc579c3b4e", "9de164425c4fec5b1f04abe97289691a", 0),
        ("1a27d42d7859a1825cb61709f903e61", "d0fae7b3a4e0598d85908cb21ef968d", -1),
        ("6a7dce5f8029bae4c30cc1667517a9e3cd01477fdfa124ea4ca2a1b249d78860", "d0fae7b3a4e0598d85908cb21ef968d", -1),
        ("272f0b71aeea10ca890b1a6165cec3a7", "d0fae7b3a4e0598d85908cb21ef968d", 0),
        ("4a791b32495f188533f4114327dbb88a", "ae6232c33d7a6663f1639966a1a46dfa", 0),
        ("6f48a8f0c4375c015e6d5abf2f32da5bec4c260ed57035ebcdcd19657f9c4525", "ae6232c33d7a6663f1639966a1a46dfa", -1),
        ("20b269849b86f332bd42acc33e4ed49ee", "ae6232c33d7a6663f1639966a1a46dfa", 0),
        ("1322b705d4b590c5abe50af1673dc68249aa0b4f569483847fb31d056d2a778b1", "a651be231d2ef7bba2d3d7c8a619cac33a3ea773f5d61ae81c4ae6e9a2161a35", -1),
        ("13953ac042ab20a04dde68f239f91c0be3775eb0d9d6f37418aa195eda4d905ba0a776e1954d76f001041adf453a6db4e53705d533bbd7dd116dfb2398f927af4", "a651be231d2ef7bba2d3d7c8a619cac33a3ea773f5d61ae81c4ae6e9a2161a35", -1),
        ("1f2f53a69578ce732e87b8759f24d6049aebbf65be18250b854e0b4bce6424e9f", "a651be231d2ef7bba2d3d7c8a619cac33a3ea773f5d61ae81c4ae6e9a2161a35", 0),
        ("c4f0b682db1f4f3e7c473fa9c26c22b3f76217b3ac0814d7492c025a90c89abf", "a1810d72fb227e5121ac20967f177a3f2078f99106e8270983165a79a1bf537d", 1),
        ("11fb471b646646d18a8b64ee79ddfb013679759577639668dc62c1cc15faae577a7e61a368982bfe36d743feb457509dfe2ac960240115b98de72a732a46b4a1d", "a1810d72fb227e5121ac20967f177a3f2078f99106e8270983165a79a1bf537d", 0),
        ("1e4832858f1677af3650461c37d466ebd616aecb314b8751c89430f6ce53dfa77", "a1810d72fb227e5121ac20967f177a3f2078f99106e8270983165a79a1bf537d", 0),
        ("ea4e175fa484d3ef5e2951323ffa00abf13f182d9ce3dd50663e6e048b5b600a", "242a2f00ff07e9e6531faaa1ccbed29d8bef2b534bd303b51a60d4b6c38e5933", 0),
        ("2cc0070a42b726bc58f5e7a708ea13541d2186c2b0a5e1c558ce2e70d60b58cdced817a4dd604d04ec37031ab3175c0d095fcd7a92e61100def74087e02fd0746", "242a2f00ff07e9e6531faaa1ccbed29d8bef2b534bd303b51a60d4b6c38e5933", -1),
        ("6c7e8d02fd17bdb2f95effe5663c77d8a3cd81f9e3790b1f4f227e244aab0b99", "242a2f00ff07e9e6531faaa1ccbed29d8bef2b534bd303b51a60d4b6c38e5933", 0),
        ("37600b3cb7f36c8a285f86f11fc1457695c9939d6a1d2b2978cf475f94ac17b0", "976fe790406bf1732bd076106305324ddd3fa3c68cb7a65eb39187c3b4c7aaa4", 0),
        ("3f7a758c9fba12946d8acd8be114391fb5158e80581b553b2060317b6239128576bc9af01c581ad7506f8c3442258517d3abd706512d97582cefe8473ddae9001", "976fe790406bf1732bd076106305324ddd3fa3c68cb7a65eb39187c3b4c7aaa4", 0),
        ("1c64fb6b0c143d45983716231290f96e997beeb53a626f31c1ab4974b1e56ffec", "976fe790406bf1732bd076106305324ddd3fa3c68cb7a65eb39187c3b4c7aaa4", 0),
        ("102362676e6d6bbb24471d49d0f0fe6ae1b85205ccc0421d1933bd3681480b7a0", "14896511af7f6e65ed3704944abd021a603e5528a854c03afa1916db807dfb429", 1),
        ("3509d7c2c54d4680802031fc2faa3d2b8efd2a4ae2072008efc40ac42a939f2518437a694e02bc259687a34cef9b4d422adecaed75990da915aaca0b90e8426de", "14896511af7f6e65ed3704944abd021a603e5528a854c03afa1916db807dfb429", -1),
        ("3d9c2f350e7e4b31c7a50dbce037064f20baff79f8fe40b0ee4b44928179f1c7b", "14896511af7f6e65ed3704944abd021a603e5528a854c03afa1916db807dfb429", 0),
        ("1db8619cf5278e2145603c04f850f3bead657387a78fb8626bd5cdf8e29fc2235", "1bfcabd5e4d1fa9d814d4d4caba3c507311721e7846845beacd65c7091e865ca8", 1),
        ("34c382a8c828fb3680f54c91a856f0763fbf8653464873010a0295b9d86f300deb4de9d734e476b9a8c2e713bca5a00d1281ca73187948bd67bae77eeda556cc", "1bfcabd5e4d1fa9d814d4d4caba3c507311721e7846845beacd65c7091e865ca8", 0),
        ("53f60381ae75efd883e7e7e602eb4f15934565b68d38d13c06831551b5b9315f8", "1bfcabd5e4d1fa9d814d4d4caba3c507311721e7846845beacd65c7091e865ca8", 0),
        ("0", "0", 0),
        ("1", "0", 1),
        ("2", "0", 0),
        ("1", "1", 1),
        ("5", "8", -1),
        ("3", "8", -1),
        ("7", "8", 1),
        ("2", "4", 0),
        ("6", "4", 0),
        ("1", "10", 1),
        ("9", "10", 1),
        ("ff", "100", 1),
        ("3", "c", 0),
        ("5", "c", -1),
    ];

    #[test]
    fn kronecker_matches_gmp_vectors() {
        for &(a_hex, n_hex, expected) in GMP_KRONECKER_VECTORS {
            let a = biguint_from_hex(a_hex);
            let n = biguint_from_hex(n_hex);
            assert_eq!(
                super::kronecker(&a, &n),
                expected,
                "kronecker({a_hex}, {n_hex}) != {expected}"
            );
        }
    }

    #[test]
    fn kronecker_extends_jacobi_and_is_multiplicative() {
        let mut rng = SplitMix64 { state: 0x2718_2818 };
        let bound = BigUint::from_u128(1 << 72);
        for _ in 0..40 {
            let a = draw_below(&mut rng, &bound);
            let mut n_odd = draw_below(&mut rng, &bound);
            if !n_odd.is_odd() {
                n_odd = n_odd.add_ref(&BigUint::one());
            }
            // On odd moduli the Kronecker symbol IS the Jacobi symbol.
            assert_eq!(super::kronecker(&a, &n_odd), jacobi(&a, &n_odd).unwrap());

            // Multiplicative in the bottom argument.
            let n2 = draw_below(&mut rng, &bound);
            if n2.is_zero() {
                continue;
            }
            assert_eq!(
                super::kronecker(&a, &n_odd.mul_ref(&n2)),
                super::kronecker(&a, &n_odd) * super::kronecker(&a, &n2)
            );
        }
    }

    #[test]
    fn legendre_is_jacobi_on_primes() {
        let p = BigUint::from_u64(1_000_000_007);
        for a in [0u64, 1, 2, 3, 5, 999_999_999] {
            let a = BigUint::from_u64(a);
            assert_eq!(super::legendre(&a, &p), jacobi(&a, &p));
        }
        assert_eq!(
            super::legendre(&BigUint::one(), &BigUint::from_u64(4)),
            None
        );
    }

    #[test]
    fn sqrt_mod_roundtrips_on_squares() {
        // Primes covering every residue class the algorithm branches on:
        // 3 mod 4 (shortcut), 5 mod 8 (s = 2), and deep 2-adic descents —
        // 41 and 97 (s = 3, 5) and the NTT primes 15·2^27 + 1 and
        // 17·2^27 + 1 (s = 27). M127 exercises the shortcut at width.
        let mut primes: Vec<BigUint> = [
            3u64,
            5,
            7,
            11,
            13,
            17,
            41,
            73,
            97,
            65_537,
            2_147_483_647,
            2_013_265_921,
            2_281_701_377,
        ]
        .iter()
        .map(|&p| BigUint::from_u64(p))
        .collect();
        primes.push(mersenne(89));
        primes.push(mersenne(127));

        let mut rng = SplitMix64 { state: 0x1414_2135 };
        for p in &primes {
            for _ in 0..8 {
                let x = draw_below(&mut rng, p);
                let square = BigUint::mod_mul(&x, &x, p);
                let root = super::sqrt_mod(&square, p).expect("squares have roots");
                assert_eq!(
                    BigUint::mod_mul(&root, &root, p),
                    square,
                    "root fails its own contract for p={p:?}"
                );
                // The root is x or p - x.
                assert!(
                    root == x || root == p.sub_ref(&x).modulo(p),
                    "root is neither ±x for p={p:?}"
                );
            }

            // Non-residues have no root; zero is its own.
            if p > &BigUint::from_u64(2) {
                let mut z = BigUint::from_u64(2);
                while jacobi(&z, p) != Some(-1) {
                    z = z.add_ref(&BigUint::one());
                }
                assert_eq!(super::sqrt_mod(&z, p), None);
            }
            assert_eq!(super::sqrt_mod(&BigUint::zero(), p), Some(BigUint::zero()));
        }

        // p = 2 and invalid moduli.
        let two = BigUint::from_u64(2);
        assert_eq!(
            super::sqrt_mod(&BigUint::from_u64(5), &two),
            Some(BigUint::one())
        );
        assert_eq!(super::sqrt_mod(&BigUint::one(), &BigUint::zero()), None);
        // Composite p: the final verification refuses to lie.
        assert_eq!(
            super::sqrt_mod(&BigUint::from_u64(2), &BigUint::from_u64(15)),
            None
        );
    }

    #[test]
    fn gcd_extended_satisfies_bezout() {
        use crate::bigint::BigInt;
        let mut rng = SplitMix64 { state: 0x0577_2156 };
        let bound = BigUint::from_u128(1 << 96);
        for _ in 0..60 {
            let a = draw_below(&mut rng, &bound);
            let b = draw_below(&mut rng, &bound);
            let (g, s, t) = super::gcd_extended(&a, &b);
            assert_eq!(g, gcd(&a, &b), "g disagrees with plain gcd");
            // a·s + b·t = g, in signed arithmetic.
            let lhs = s.mul_biguint_ref(&a).add_ref(&t.mul_biguint_ref(&b));
            assert_eq!(lhs, BigInt::from_biguint(g), "Bezout identity fails");
        }

        // Degenerate corners.
        let (g, s, _) = super::gcd_extended(&BigUint::zero(), &BigUint::from_u64(7));
        assert_eq!(g, BigUint::from_u64(7));
        assert!(matches!(s.sign(), crate::bigint::Sign::Zero));
    }

    #[test]
    fn crt_combine_reconstructs_and_rejects() {
        let mut rng = SplitMix64 { state: 0x6931_4718 };
        // Pairwise coprime moduli, including big primes.
        let moduli = [
            BigUint::from_u64(97),
            BigUint::from_u64(1_000_000_007),
            mersenne(89),
            mersenne(107),
        ];
        let product = moduli.iter().fold(BigUint::one(), |acc, m| acc.mul_ref(m));

        for _ in 0..12 {
            let x = draw_below(&mut rng, &product);
            let congruences: Vec<(BigUint, BigUint)> =
                moduli.iter().map(|m| (x.modulo(m), m.clone())).collect();
            assert_eq!(super::crt_combine(&congruences), Some(x.clone()));

            // Order must not matter.
            let mut reversed = congruences.clone();
            reversed.reverse();
            assert_eq!(super::crt_combine(&reversed), Some(x));
        }

        // A single congruence reduces its residue.
        assert_eq!(
            super::crt_combine(&[(BigUint::from_u64(23), BigUint::from_u64(7))]),
            Some(BigUint::from_u64(2))
        );
        // Non-coprime moduli and degenerate inputs.
        assert_eq!(
            super::crt_combine(&[
                (BigUint::one(), BigUint::from_u64(6)),
                (BigUint::from_u64(2), BigUint::from_u64(9)),
            ]),
            None
        );
        assert_eq!(super::crt_combine(&[]), None);
        assert_eq!(
            super::crt_combine(&[(BigUint::one(), BigUint::zero())]),
            None
        );
    }

    /// Sieve of Eratosthenes: the oracle that shares no code with the
    /// functions under test — the probable-prime tests all open with the
    /// same trial-division screen, so their mutual agreement cannot detect
    /// a defect in it.
    fn eratosthenes(limit: usize) -> Vec<bool> {
        let mut is_prime = vec![true; limit];
        is_prime[0] = false;
        if limit > 1 {
            is_prime[1] = false;
        }
        let mut p = 2usize;
        while p * p < limit {
            if is_prime[p] {
                let mut multiple = p * p;
                while multiple < limit {
                    is_prime[multiple] = false;
                    multiple += p;
                }
            }
            p += 1;
        }
        is_prime
    }

    #[test]
    fn remainder_tree_matches_direct_reduction() {
        use super::{product_tree, remainder_tree};
        let mut rng = SplitMix64 {
            state: 0x73ee_0007_0000_0001,
        };
        for &count in &[1usize, 2, 3, 5, 8, 17, 64] {
            let values: Vec<BigUint> = (0..count)
                .map(|_| {
                    let mut v = draw_below(&mut rng, &pow2(256));
                    v.set_bit(0); // non-zero
                    v
                })
                .collect();
            let modulus = draw_below(&mut rng, &pow2(2048));
            let tree = product_tree(&values);
            // The root is the product of all values.
            let mut product = BigUint::one();
            for v in &values {
                product = product.mul_ref(v);
            }
            assert_eq!(tree.last().unwrap()[0], product, "root is the product");
            // Each leaf remainder equals a direct reduction.
            let batched = remainder_tree(&tree, &modulus);
            for (v, r) in values.iter().zip(&batched) {
                assert_eq!(*r, modulus.modulo(v), "batched vs direct reduction");
            }
        }
        assert!(product_tree(&[]).is_empty());
        assert!(remainder_tree(&[], &BigUint::from_u64(5)).is_empty());
    }

    #[test]
    fn smooth_parts_matches_trial_division() {
        use super::{primes_below, smooth_parts};
        let primes = primes_below(50); // 2,3,5,7,11,...,47
                                       // Trial-division oracle for the smooth part of one value.
        let smooth_part_naive = |value: &BigUint| -> BigUint {
            let mut rest = value.clone();
            let mut part = BigUint::one();
            for &p in &primes {
                let pv = BigUint::from_u64(p);
                loop {
                    let (q, r) = rest.div_rem_u64(p);
                    if r != 0 {
                        break;
                    }
                    rest = q;
                    part = part.mul_ref(&pv);
                }
            }
            part
        };
        let mut rng = SplitMix64 {
            state: 0x5000_7000_0000_0001,
        };
        // Mixed batch: some fully smooth (products of small primes), some
        // with a large prime factor, some coprime to the base.
        let mut values = Vec::new();
        // Fully smooth constructions.
        for _ in 0..8 {
            let mut v = BigUint::one();
            for _ in 0..6 {
                let p = primes[(rng.next_u64() as usize) % primes.len()];
                v = v.mul_ref(&BigUint::from_u64(p));
            }
            values.push(v);
        }
        // Smooth core times a big prime.
        for _ in 0..8 {
            let mut core = BigUint::one();
            for _ in 0..4 {
                let p = primes[(rng.next_u64() as usize) % primes.len()];
                core = core.mul_ref(&BigUint::from_u64(p));
            }
            let mut big = draw_below(&mut rng, &pow2(80));
            big.set_bit(79);
            big.set_bit(0);
            while !is_probable_prime(&big) {
                big = big.add_ref(&BigUint::from_u64(2));
            }
            values.push(core.mul_ref(&big));
        }
        // Random values.
        for _ in 0..8 {
            let mut v = draw_below(&mut rng, &pow2(200));
            v.set_bit(0);
            values.push(v);
        }
        let parts = smooth_parts(&values, &primes);
        for (v, part) in values.iter().zip(&parts) {
            assert_eq!(*part, smooth_part_naive(v), "smooth part of a value");
            // The smooth part divides the value.
            assert!(v.modulo(part).is_zero(), "smooth part divides the value");
        }
        // A fully-smooth value has smooth part equal to itself.
        let smooth = BigUint::from_u64(2 * 2 * 3 * 5 * 7);
        assert_eq!(
            smooth_parts(std::slice::from_ref(&smooth), &primes)[0],
            smooth
        );
        // Trivial values pass through the batch: 0 → 0, 1 → 1, in order,
        // interleaved with nontrivial values.
        let mixed = [
            BigUint::zero(),
            BigUint::from_u64(30),
            BigUint::one(),
            BigUint::from_u64(2 * 101),
        ];
        let parts = smooth_parts(&mixed, &primes);
        assert_eq!(parts[0], BigUint::zero());
        assert_eq!(parts[1], BigUint::from_u64(30));
        assert_eq!(parts[2], BigUint::one());
        assert_eq!(parts[3], BigUint::from_u64(2));
        // Empty prime set: nothing is smooth beyond the trivial values.
        assert_eq!(
            smooth_parts(&[BigUint::from_u64(30)], &[]),
            vec![BigUint::one()]
        );
        // Empty inputs.
        assert!(smooth_parts(&[], &primes).is_empty());
    }

    #[test]
    #[should_panic(expected = "prime-power exponent must be at least 1")]
    fn sqrt_mod_prime_power_rejects_zero_exponent() {
        let _ = super::sqrt_mod_prime_power(&BigUint::one(), &BigUint::from_u64(7), 0);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn remainder_tree_rejects_zero_leaf() {
        use super::{product_tree, remainder_tree};
        let tree = product_tree(&[BigUint::from_u64(6), BigUint::zero()]);
        let _ = remainder_tree(&tree, &BigUint::from_u64(30));
    }

    #[test]
    fn primes_below_matches_primality_test() {
        use super::{is_probable_prime, primes_below};
        let primes = primes_below(10_000);
        let mut previous = 0u64;
        let mut set = std::collections::HashSet::new();
        for &p in &primes {
            assert!(p > previous, "ascending and distinct");
            previous = p;
            assert!(is_probable_prime(&BigUint::from_u64(p)), "{p} is prime");
            set.insert(p);
        }
        for n in 2u64..10_000 {
            assert_eq!(
                set.contains(&n),
                is_probable_prime(&BigUint::from_u64(n)),
                "sieve disagrees with test at {n}"
            );
        }
        // Known prime counts (primes strictly below the bound).
        assert_eq!(primes_below(10).len(), 4); // 2,3,5,7
        assert_eq!(primes_below(100).len(), 25);
        assert_eq!(primes_below(1_000).len(), 168);
        assert!(primes_below(2).is_empty());
        assert_eq!(primes_below(3), vec![2]);
    }

    #[test]
    fn sqrt_mod_prime_power_exhaustive_small() {
        use super::sqrt_mod_prime_power;
        for &p in &[2u64, 3, 5, 7, 11] {
            let max_e = if p == 2 { 7 } else { 4 };
            for e in 1u32..=max_e {
                let m = p.pow(e);
                for a in 0..m {
                    let expected: Vec<u64> = (0..m).filter(|x| (x * x) % m == a).collect();
                    let got: Vec<u64> =
                        sqrt_mod_prime_power(&BigUint::from_u64(a), &BigUint::from_u64(p), e)
                            .iter()
                            .map(|r| r.to_u64().expect("root below p^e fits u64"))
                            .collect();
                    assert_eq!(got, expected, "roots of {a} mod {p}^{e}");
                }
            }
        }
    }

    #[test]
    fn sqrt_mod_prime_power_wide() {
        use super::sqrt_mod_prime_power;
        let mut rng = SplitMix64 {
            state: 0x5170_0007_0000_0001,
        };
        let mut p = draw_below(&mut rng, &pow2(128));
        p.set_bit(127);
        p.set_bit(0);
        while !is_probable_prime(&p) {
            p = p.add_ref(&BigUint::from_u64(2));
        }
        let e = 3u32;
        let modulus = p.pow_u64(u64::from(e));
        let x = draw_below(&mut rng, &modulus);
        let a = BigUint::mod_mul(&x, &x, &modulus);
        let roots = sqrt_mod_prime_power(&a, &p, e);
        assert_eq!(
            roots.len(),
            2,
            "unit square mod odd prime power has two roots"
        );
        for r in &roots {
            assert_eq!(BigUint::mod_mul(r, r, &modulus), a, "root squares to a");
        }
        assert!(roots.contains(&x) || roots.contains(&modulus.sub_ref(&x)));
    }

    #[test]
    fn primality_tests_match_the_sieve_below_300k() {
        use super::is_probable_prime_bpsw;
        const LIMIT: usize = 300_000;
        let sieve = eratosthenes(LIMIT);
        for (n, &expected) in sieve.iter().enumerate() {
            let candidate = BigUint::from_u64(n as u64);
            assert_eq!(
                is_probable_prime_bpsw(&candidate),
                expected,
                "bpsw diverged from the sieve at {n}"
            );
            assert_eq!(
                is_probable_prime(&candidate),
                expected,
                "twelve-base Miller–Rabin diverged from the sieve at {n}"
            );
        }
    }

    #[test]
    fn strong_lucas_matches_published_pseudoprime_tables() {
        use super::{is_probable_prime_bpsw, is_strong_lucas_probable_prime};
        // Strong Lucas pseudoprimes with Selfridge's parameters below 10⁵
        // (Baillie & Wagstaff; OEIS A217255), re-derived independently for
        // this suite: composites the Lucas stage alone must pass — and the
        // base-2 Miller–Rabin stage must kill inside the composed test.
        const STRONG_LUCAS_PSEUDOPRIMES: [u64; 12] = [
            5459, 5777, 10877, 16109, 18971, 22499, 24569, 25199, 40309, 58519, 75077, 97439,
        ];
        for &n in &STRONG_LUCAS_PSEUDOPRIMES {
            let value = BigUint::from_u64(n);
            assert!(!is_probable_prime(&value), "{n} is composite");
            assert!(
                is_strong_lucas_probable_prime(&value),
                "{n} must pass the Lucas stage alone"
            );
            assert!(
                !is_probable_prime_bpsw(&value),
                "{n} must fail the composed test"
            );
        }
        // Strong pseudoprimes to base 2 below 10⁵ (OEIS A001262), likewise
        // re-derived: composites the base-2 stage alone must pass — and the
        // Lucas stage must kill.
        const STRONG_BASE2_PSEUDOPRIMES: [u64; 16] = [
            2047, 3277, 4033, 4681, 8321, 15841, 29341, 42799, 49141, 52633, 65281, 74665, 80581,
            85489, 88357, 90751,
        ];
        for &n in &STRONG_BASE2_PSEUDOPRIMES {
            let value = BigUint::from_u64(n);
            assert!(!is_probable_prime(&value), "{n} is composite");
            assert!(
                !miller_rabin_witness(&value, &BigUint::from_u64(2)),
                "base 2 alone must not witness {n}"
            );
            assert!(
                !is_strong_lucas_probable_prime(&value),
                "the Lucas stage must reject {n}"
            );
            assert!(
                !is_probable_prime_bpsw(&value),
                "{n} must fail the composed test"
            );
        }
    }

    #[test]
    fn bpsw_agrees_with_deterministic_miller_rabin_on_random_words() {
        use super::is_probable_prime_bpsw;
        // Above the sieve's reach the twelve-base test is the practical
        // reference (deterministic to 3.3·10²⁴, Sorenson & Webster). The
        // two tests share only the trial-division screen, which the sieve
        // test above checks against an independent oracle.
        let mut rng = SplitMix64 {
            state: 0xb5c0_5e1f_ba11_1e50,
        };
        for _ in 0..4000 {
            let candidate = BigUint::from_u64(rng.next_u64());
            assert_eq!(
                is_probable_prime_bpsw(&candidate),
                is_probable_prime(&candidate),
                "bpsw diverged on a random 64-bit value"
            );
        }
    }

    #[test]
    fn strong_lucas_accepts_primes_and_handles_edges() {
        use super::is_strong_lucas_probable_prime;
        // Positive coverage for the standalone stage: every prime passes.
        for p in [2u64, 3, 5, 7, 13, 101, 1009, 65_537, 4_294_967_291] {
            assert!(
                is_strong_lucas_probable_prime(&BigUint::from_u64(p)),
                "prime {p} must pass the standalone Lucas stage"
            );
        }
        assert!(is_strong_lucas_probable_prime(&mersenne(127)));
        // A 256-bit prime, found with the twelve-base test.
        let mut rng = SplitMix64 {
            state: 0x10c4_ea51_ed00_0001,
        };
        let mut p = draw_below(&mut rng, &pow2(256));
        p.set_bit(255);
        p.set_bit(0);
        while !is_probable_prime(&p) {
            p = p.add_ref(&BigUint::from_u64(2));
        }
        assert!(is_strong_lucas_probable_prime(&p));
        // Wrapper edges the composed test never routes here.
        for n in [0u64, 1, 4, 6] {
            assert!(!is_strong_lucas_probable_prime(&BigUint::from_u64(n)));
        }
    }

    /// Classical single-step reconstruction — the same theorem walked one
    /// division at a time, no Lehmer batches — as the oracle for the
    /// batched production walk.
    fn rational_reconstruct_classical(
        x: &BigUint,
        m: &BigUint,
        num_bound: &BigUint,
        den_bound: &BigUint,
    ) -> Option<(crate::bigint::BigInt, BigUint)> {
        use crate::bigint::{BigInt, Sign};
        let x = x.modulo(m);
        if den_bound.is_zero() {
            return None;
        }
        if x.is_zero() {
            return Some((BigInt::zero(), BigUint::one()));
        }
        let (mut r0, mut r1) = (m.clone(), x);
        let (mut t0, mut t1) = (BigInt::zero(), BigInt::from_biguint(BigUint::one()));
        while r1 > *num_bound {
            let (quotient, remainder) = r0.div_rem(&r1);
            let next_t1 = t0.sub_ref(&t1.mul_biguint_ref(&quotient));
            r0 = core::mem::replace(&mut r1, remainder);
            t0 = core::mem::replace(&mut t1, next_t1);
        }
        let q = t1.magnitude().clone();
        if q.is_zero() || q > *den_bound || !gcd(&r1, &q).is_one() {
            return None;
        }
        let sign = match t1.sign() {
            Sign::Negative => Sign::Negative,
            _ => Sign::Positive,
        };
        Some((BigInt::from_parts(sign, r1), q))
    }

    /// A probable prime `k·2^s + 1` with `k` odd — the constructed
    /// high-valuation primes the Cipolla tests and probe need.
    fn prime_with_two_adic_valuation(bits: usize, s: usize, rng: &mut SplitMix64) -> BigUint {
        loop {
            let mut k = draw_below(rng, &pow2(bits - s - 1));
            k.set_bit(0);
            k.set_bit(bits - s - 1);
            let mut p = k;
            p.shl_bits(s);
            p = p.add_ref(&BigUint::one());
            // A word-sized trial screen rejects most candidates for a few
            // remainders each, keeping the constructor off the suite's
            // critical path.
            if [3u64, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
                .iter()
                .any(|&q| p.rem_u64(q) == 0)
            {
                continue;
            }
            if is_probable_prime(&p) {
                return p;
            }
        }
    }

    #[test]
    fn cipolla_agrees_with_the_descent() {
        use super::{sqrt_mod, sqrt_mod_cipolla};
        use crate::bigint::MontgomeryCtx;
        let mut rng = SplitMix64 {
            state: 0xc1b0_11a0_0000_0001,
        };
        // Low-s primes route through Tonelli–Shanks; Cipolla, called
        // directly on the same operands, must land on the same root pair.
        for &(bits, s) in &[(256usize, 2usize), (256, 5), (512, 3), (1024, 4)] {
            let p = prime_with_two_adic_valuation(bits, s, &mut rng);
            let x = draw_below(&mut rng, &p);
            let a = BigUint::mod_mul(&x, &x, &p);
            if a.is_zero() {
                continue;
            }
            let descent_root = sqrt_mod(&a, &p).expect("a is a residue by construction");
            let ctx = MontgomeryCtx::new(&p).expect("odd prime");
            let cipolla_root =
                sqrt_mod_cipolla(&a, &p, &ctx).expect("prime modulus never exhausts the scan");
            assert!(
                cipolla_root == descent_root || cipolla_root == p.sub_ref(&descent_root),
                "engines disagree at {bits} bits, s = {s}"
            );
            assert!(BigUint::mod_mul(&cipolla_root, &cipolla_root, &p) == a);
        }
        // High-s primes route through Cipolla inside the dispatch; the
        // returned root must verify, and the deepest case must be fast
        // enough to sit in a unit test at all — which is the point.
        for &(bits, s) in &[(512usize, 128usize), (1024, 256), (2048, 512)] {
            let p = prime_with_two_adic_valuation(bits, s, &mut rng);
            assert!(
                s * s > super::CIPOLLA_THRESHOLD_FACTOR * p.bits(),
                "the constructed prime must cross the dispatch"
            );
            let x = draw_below(&mut rng, &p);
            let a = BigUint::mod_mul(&x, &x, &p);
            if a.is_zero() {
                continue;
            }
            let root = sqrt_mod(&a, &p).expect("a is a residue by construction");
            assert_eq!(
                BigUint::mod_mul(&root, &root, &p),
                a,
                "{bits} bits, s = {s}"
            );
        }
        // Non-residues stay None regardless of the engine.
        let p = prime_with_two_adic_valuation(512, 128, &mut rng);
        let mut z = BigUint::from_u64(2);
        while super::jacobi(&z, &p) != Some(-1) {
            z = z.add_ref(&BigUint::one());
        }
        assert_eq!(sqrt_mod(&z, &p), None);
        // The trivial residues, through both engines.
        for &(bits, s) in &[(256usize, 2usize), (512, 128)] {
            let p = prime_with_two_adic_valuation(bits, s, &mut rng);
            // Either square root of one is a correct answer.
            let root_of_one = sqrt_mod(&BigUint::one(), &p).expect("1 is a residue");
            assert!(
                root_of_one.is_one() || root_of_one == p.sub_ref(&BigUint::one()),
                "the root of 1 is ±1"
            );
            let minus_one = p.sub_ref(&BigUint::one());
            let root = sqrt_mod(&minus_one, &p);
            // −1 is a residue exactly when p ≡ 1 (mod 4) — true for every
            // constructed prime with s ≥ 2 — and the root must verify.
            let r = root.expect("-1 is a residue for p = 1 mod 4");
            assert_eq!(BigUint::mod_mul(&r, &r, &p), minus_one);
        }
    }

    #[test]
    fn sqrt_mod_terminates_on_odd_square_modulus() {
        use super::sqrt_mod;
        // An odd perfect square has no Jacobi non-residue, so both engines'
        // parameter scans would once run forever; the bound turns the
        // pathology into None. 4097² routes to the descent's scan; a
        // high-valuation square would route to Cipolla's.
        let square = BigUint::from_u64(4097).square_ref();
        assert_eq!(sqrt_mod(&BigUint::from_u64(4), &square), None);
    }

    #[test]
    fn sqrt_mod_terminates_on_small_odd_squares() {
        use super::sqrt_mod;
        // The first odd composite squares. Every odd square is ≡ 1 (mod 4),
        // so each enters the non-residue scan rather than the ≡ 3 shortcut;
        // the bounded scan returns None instead of looping. 169 = 13² is a
        // p ≡ 1 (mod 4) square with a deeper structure.
        for m in [9u64, 25, 49, 169] {
            assert_eq!(
                sqrt_mod(&BigUint::from_u64(1), &BigUint::from_u64(m)),
                None,
                "sqrt_mod(1, {m}) must terminate as None"
            );
        }
        // Sharpened: an *actual* square residue hangs the unbounded scan too —
        // 4 ≡ 2² ≡ 7² (mod 9), jacobi(4, 9) = 1. Tonelli needs a non-residue,
        // which an odd square modulus does not have, so the bounded scan
        // returns None (no root over a non-prime modulus) rather than looping.
        // Use sqrt_mod_prime_power for roots modulo a prime power.
        assert_eq!(
            sqrt_mod(&BigUint::from_u64(4), &BigUint::from_u64(9)),
            None,
            "sqrt_mod(4, 9) must terminate"
        );
        // a ≡ 0 exits before the scan and is the one residue that resolves.
        assert_eq!(
            sqrt_mod(&BigUint::from_u64(0), &BigUint::from_u64(9)),
            Some(BigUint::zero())
        );
        // Even non-prime moduli: the even branch declines before any scan, so
        // these terminate as None even though 0 and 1 are squares mod 4.
        assert_eq!(sqrt_mod(&BigUint::from_u64(0), &BigUint::from_u64(4)), None);
        assert_eq!(sqrt_mod(&BigUint::from_u64(1), &BigUint::from_u64(4)), None);
    }

    #[test]
    fn sqrt_mod_on_composite_may_return_a_genuine_root() {
        use super::sqrt_mod;
        // The contract is verification, not primality: for a composite modulus
        // the function returns None or a value that genuinely squares to `a`.
        // 15 ≡ 3 (mod 4) takes the shortcut and 1 is a true root of 1 mod 15.
        let root = sqrt_mod(&BigUint::from_u64(1), &BigUint::from_u64(15));
        assert_eq!(root, Some(BigUint::one()));
        let r = root.expect("a verified root");
        assert_eq!(
            BigUint::mod_mul(&r, &r, &BigUint::from_u64(15)),
            BigUint::from_u64(1),
            "the returned value squares back to a"
        );
    }

    #[test]
    #[ignore = "timing probe for the Tonelli-Shanks/Cipolla crossover; run with --ignored"]
    fn cipolla_crossover_timing() {
        use super::{decompose_n_minus_one, sqrt_mod_cipolla, sqrt_mod_descent};
        use crate::bigint::MontgomeryCtx;
        use std::hint::black_box;
        use std::time::Instant;
        let mut rng = SplitMix64 {
            state: 0xc1b0_11a0_dead_beef,
        };
        // Both engines at every s, on a grid bracketing √(4·bits): the
        // crossing is read directly off the two columns.
        eprintln!(
            "{:>6} {:>6} {:>12} {:>12}",
            "bits", "s", "descent_us", "cipolla_us"
        );
        for &bits in &[1024usize, 2048, 4096] {
            let center = (4.0 * bits as f64).sqrt() as usize;
            for step in -3i64..=3 {
                let s = usize::try_from((center as i64 + step * 16).max(2)).expect("positive");
                let p = prime_with_two_adic_valuation(bits, s, &mut rng);
                let (q, s_actual) = decompose_n_minus_one(&p);
                assert_eq!(s_actual, s, "constructed valuation");
                let ctx = MontgomeryCtx::new(&p).expect("odd prime");
                let x = draw_below(&mut rng, &p);
                let a = BigUint::mod_mul(&x, &x, &p);
                let time = |f: &dyn Fn()| {
                    let mut best = f64::INFINITY;
                    for _ in 0..5 {
                        let t0 = Instant::now();
                        f();
                        best = best.min(t0.elapsed().as_secs_f64() * 1e6);
                    }
                    best
                };
                let descent = time(&|| {
                    black_box(sqrt_mod_descent(&a, &p, &ctx, &q, s));
                });
                let cipolla = time(&|| {
                    black_box(sqrt_mod_cipolla(&a, &p, &ctx));
                });
                eprintln!("{bits:>6} {s:>6} {descent:>12.1} {cipolla:>12.1}");
            }
        }
    }

    #[test]
    fn batch_inversion_matches_element_wise() {
        use super::{mod_inverse, mod_inverse_batch};
        let mut rng = SplitMix64 {
            state: 0x7007_ba7c_4000_0001,
        };
        for &bits in &[64usize, 256, 1024, 2048] {
            let mut modulus = draw_below(&mut rng, &pow2(bits));
            modulus.set_bit(bits - 1);
            modulus.set_bit(0);
            for &count in &[1usize, 2, 3, 17, 100] {
                // All-coprime batches: retry elements until invertible.
                let mut values = Vec::new();
                while values.len() < count {
                    let candidate = draw_below(&mut rng, &modulus);
                    if mod_inverse(&candidate, &modulus).is_some() {
                        values.push(candidate);
                    }
                }
                let batch =
                    mod_inverse_batch(&values, &modulus).expect("all elements are invertible");
                for (inverse, value) in batch.iter().zip(&values) {
                    assert_eq!(
                        Some(inverse.clone()),
                        mod_inverse(value, &modulus),
                        "batched inverse diverged at {bits} bits, batch of {count}"
                    );
                }
            }
        }
    }

    #[test]
    fn batch_inversion_edges() {
        use super::mod_inverse_batch;
        let m = BigUint::from_u64(101);
        assert_eq!(mod_inverse_batch(&[], &m), Some(Vec::new()));
        // Degenerate moduli follow mod_inverse's own answers.
        let five = [BigUint::from_u64(5)];
        assert_eq!(mod_inverse_batch(&five, &BigUint::zero()), None);
        assert_eq!(mod_inverse_batch(&[], &BigUint::zero()), None);
        assert_eq!(
            mod_inverse_batch(&five, &BigUint::one()),
            Some(vec![BigUint::zero()])
        );
        // A poisoned batch: 505 shares the factor 101.
        let values = [
            BigUint::from_u64(3),
            BigUint::from_u64(505),
            BigUint::from_u64(7),
        ];
        assert_eq!(mod_inverse_batch(&values, &m), None);
        // Duplicates, ones, and unreduced elements are all fine.
        let values = [
            BigUint::from_u64(1),
            BigUint::from_u64(9),
            BigUint::from_u64(9),
            BigUint::from_u64(102), // ≡ 1
        ];
        let batch = mod_inverse_batch(&values, &m).expect("all coprime to 101");
        assert!(BigUint::mod_mul(&batch[3], &BigUint::from_u64(102), &m).is_one());
        assert_eq!(batch[0], BigUint::one());
        assert_eq!(batch[1], batch[2]);
    }

    #[test]
    fn valuation_and_remove_factor() {
        use super::{remove_factor, valuation};
        // Machine-arithmetic brute force.
        let mut rng0 = SplitMix64 {
            state: 0x0e0e_0e0e_5eed_0006,
        };
        for _ in 0..2000 {
            let v = (rng0.next_u64() >> (rng0.next_u64() % 48)) | 1;
            for p in [2u64, 3, 5, 7, 11, 97] {
                let mut expect_e = 0usize;
                let mut expect_c = v;
                while expect_c.is_multiple_of(p) {
                    expect_c /= p;
                    expect_e += 1;
                }
                let (cofactor, exponent) =
                    remove_factor(&BigUint::from_u64(v), &BigUint::from_u64(p));
                assert_eq!(exponent, expect_e, "exponent of {p} in {v}");
                assert_eq!(
                    cofactor,
                    BigUint::from_u64(expect_c),
                    "cofactor of {p} in {v}"
                );
                assert_eq!(
                    valuation(&BigUint::from_u64(v), &BigUint::from_u64(p)),
                    expect_e
                );
            }
        }
        // Planted wide valuations, including the ladder's descent edges:
        // exponents on and off rung boundaries (2^i, 2^i ± 1, and 12 — the
        // shape a guarded climb once got wrong).
        let mut rng = SplitMix64 {
            state: 0x0dd5_0006_0006_0006,
        };
        let p = draw_below(&mut rng, &pow2(200));
        let p = p.add_ref(&BigUint::from_u64(3)); // any value ≥ 2 serves
        for &e in &[1usize, 2, 3, 4, 7, 8, 9, 12, 31, 32, 33, 100] {
            let mut m = draw_below(&mut rng, &pow2(150));
            loop {
                let (_, r) = m.div_rem(&p);
                if !r.is_zero() {
                    break;
                }
                m = m.add_ref(&BigUint::one());
            }
            let planted = p.pow_u64(u64::try_from(e).expect("small")).mul_ref(&m);
            let (cofactor, exponent) = remove_factor(&planted, &p);
            assert_eq!(exponent, e, "planted exponent {e}");
            assert_eq!(cofactor, m, "planted cofactor at exponent {e}");
        }
        // p = 2 reads the limbs directly.
        let mut v = BigUint::from_u64(0b1011);
        v.shl_bits(777);
        assert_eq!(remove_factor(&v, &BigUint::from_u64(2)).1, 777);
    }

    #[test]
    #[should_panic(expected = "unbounded valuation")]
    fn valuation_rejects_zero() {
        let _ = super::valuation(&BigUint::zero(), &BigUint::from_u64(3));
    }

    #[test]
    #[should_panic(expected = "valuation needs p >= 2")]
    fn remove_factor_rejects_unit_base() {
        let _ = super::remove_factor(&BigUint::from_u64(6), &BigUint::one());
    }

    #[test]
    fn rational_reconstruction_exhaustive_small() {
        use super::rational_reconstruct;
        use crate::bigint::{BigInt, Sign};
        // Against a naive search: under 2·N·D < m the qualifying fraction
        // is unique, so the search finding one pins the function exactly.
        for m_small in 3u64..=120 {
            let m = BigUint::from_u64(m_small);
            let mut half = m.sub_ref(&BigUint::one());
            half.shr1();
            let bound = half.sqrt_floor();
            let b = bound.limbs().first().copied().unwrap_or(0);
            for x_small in 0..m_small {
                let x = BigUint::from_u64(x_small);
                let mut expected: Option<(BigInt, BigUint)> = None;
                'search: for q in 1..=b {
                    for p_abs in 0..=b {
                        for negative in [false, true] {
                            if negative && p_abs == 0 {
                                continue;
                            }
                            // p ≡ q·x (mod m), gcd(p, q) = 1.
                            let residue = (q * x_small) % m_small;
                            let target = if negative {
                                (m_small - p_abs % m_small) % m_small
                            } else {
                                p_abs % m_small
                            };
                            if residue == target
                                && gcd(&BigUint::from_u64(p_abs), &BigUint::from_u64(q)).is_one()
                            {
                                let sign = if negative {
                                    Sign::Negative
                                } else {
                                    Sign::Positive
                                };
                                expected = Some((
                                    BigInt::from_parts(sign, BigUint::from_u64(p_abs)),
                                    BigUint::from_u64(q),
                                ));
                                break 'search;
                            }
                        }
                    }
                }
                assert_eq!(
                    rational_reconstruct(&x, &m),
                    expected,
                    "diverged at x = {x_small}, m = {m_small}"
                );
            }
        }
    }

    #[test]
    fn rational_reconstruction_exhaustive_asymmetric_bounds() {
        use super::rational_reconstruct_bounded;
        use crate::bigint::{BigInt, Sign};
        // Every (m, N, D, x) with m ≤ 40 and 2·N·D < m, against the naive
        // uniqueness search — the asymmetric perimeter the symmetric sweep
        // cannot reach.
        for m_small in 3u64..=40 {
            let m = BigUint::from_u64(m_small);
            for n_bound in 0..m_small {
                for d_bound in 0..m_small {
                    if 2 * n_bound * d_bound >= m_small {
                        continue;
                    }
                    for x_small in 0..m_small {
                        let x = BigUint::from_u64(x_small);
                        let mut expected: Option<(BigInt, BigUint)> = None;
                        'search: for q in 1..=d_bound {
                            for p_abs in 0..=n_bound {
                                for negative in [false, true] {
                                    if negative && p_abs == 0 {
                                        continue;
                                    }
                                    let residue = (q * x_small) % m_small;
                                    let target = if negative {
                                        (m_small - p_abs % m_small) % m_small
                                    } else {
                                        p_abs % m_small
                                    };
                                    if residue == target
                                        && gcd(&BigUint::from_u64(p_abs), &BigUint::from_u64(q))
                                            .is_one()
                                    {
                                        let sign = if negative {
                                            Sign::Negative
                                        } else {
                                            Sign::Positive
                                        };
                                        expected = Some((
                                            BigInt::from_parts(sign, BigUint::from_u64(p_abs)),
                                            BigUint::from_u64(q),
                                        ));
                                        break 'search;
                                    }
                                }
                            }
                        }
                        assert_eq!(
                            rational_reconstruct_bounded(
                                &x,
                                &m,
                                &BigUint::from_u64(n_bound),
                                &BigUint::from_u64(d_bound)
                            ),
                            expected,
                            "diverged at x = {x_small}, m = {m_small}, N = {n_bound}, D = {d_bound}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rational_reconstruction_round_trips() {
        use super::{mod_inverse, rational_reconstruct};
        use crate::bigint::{BigInt, Sign};
        let mut rng = SplitMix64 {
            state: 0x5eed_f00d_ca11_ab1e,
        };
        for &bits in &[64usize, 256, 1024, 4096] {
            let mut recovered = 0usize;
            while recovered < 6 {
                let m = {
                    let mut m = draw_below(&mut rng, &pow2(bits));
                    m.set_bit(bits - 1);
                    m.set_bit(0);
                    m
                };
                let mut half = m.sub_ref(&BigUint::one());
                half.shr1();
                let bound = half.sqrt_floor();
                let p_abs = draw_below(&mut rng, &bound);
                let q = draw_below(&mut rng, &bound);
                if q.is_zero() || !gcd(&p_abs, &q).is_one() {
                    continue;
                }
                let Some(q_inv) = mod_inverse(&q, &m) else {
                    continue;
                };
                let negative = rng.next_u64() & 1 == 1;
                // x = ±p·q⁻¹ mod m.
                let mut x = BigUint::mod_mul(&p_abs, &q_inv, &m);
                if negative && !x.is_zero() {
                    x = m.sub_ref(&x);
                }
                let sign = if negative && !p_abs.is_zero() {
                    Sign::Negative
                } else {
                    Sign::Positive
                };
                let expected = (BigInt::from_parts(sign, p_abs), q);
                assert_eq!(
                    rational_reconstruct(&x, &m).expect("planted fraction must be found"),
                    expected,
                    "round trip failed at {bits} bits"
                );
                recovered += 1;
            }
            // The theorem's tightest points, planted exactly: |p| = N with
            // q = 1, p = 1 with q = D, and p = 0.
            let mut m = draw_below(&mut rng, &pow2(bits));
            m.set_bit(bits - 1);
            m.set_bit(0);
            let mut half = m.sub_ref(&BigUint::one());
            half.shr1();
            let bound = half.sqrt_floor();
            assert_eq!(
                super::rational_reconstruct(&bound, &m),
                Some((BigInt::from_biguint(bound.clone()), BigUint::one())),
                "|p| = N, q = 1 at {bits} bits"
            );
            if let Some(d_inv) = super::mod_inverse(&bound, &m) {
                assert_eq!(
                    super::rational_reconstruct(&d_inv, &m),
                    Some((BigInt::from_biguint(BigUint::one()), bound.clone())),
                    "p = 1, q = D at {bits} bits"
                );
            }
            assert_eq!(
                super::rational_reconstruct(&BigUint::zero(), &m),
                Some((BigInt::zero(), BigUint::one())),
                "p = 0 at {bits} bits"
            );
        }
    }

    #[test]
    fn rational_reconstruction_batched_matches_classical() {
        use super::{rational_reconstruct, rational_reconstruct_bounded};
        let mut rng = SplitMix64 {
            state: 0x0dd5_ba11_ad00_0004,
        };
        for &bits in &[256usize, 1024, 2048, 4096] {
            for _ in 0..6 {
                let mut m = draw_below(&mut rng, &pow2(bits));
                m.set_bit(bits - 1);
                let x = draw_below(&mut rng, &m);
                let mut half = m.sub_ref(&BigUint::one());
                half.shr1();
                let bound = half.sqrt_floor();
                assert_eq!(
                    rational_reconstruct(&x, &m),
                    rational_reconstruct_classical(&x, &m, &bound, &bound),
                    "batched walk diverged from classical at {bits} bits"
                );
                // Asymmetric bounds exercise the den_bound rejection.
                let tight = bound.sqrt_floor();
                assert_eq!(
                    rational_reconstruct_bounded(&x, &m, &bound, &tight),
                    rational_reconstruct_classical(&x, &m, &bound, &tight),
                    "asymmetric bounds diverged at {bits} bits"
                );
            }
        }
    }

    #[test]
    fn rational_reconstruction_edges() {
        use super::{rational_reconstruct, rational_reconstruct_bounded};
        use crate::bigint::BigInt;
        let m = BigUint::from_u64(101);
        // x = 0 is 0/1.
        assert_eq!(
            rational_reconstruct(&BigUint::zero(), &m),
            Some((BigInt::zero(), BigUint::one()))
        );
        // x within the numerator bound is x/1.
        assert_eq!(
            rational_reconstruct(&BigUint::from_u64(5), &m),
            Some((BigInt::from_biguint(BigUint::from_u64(5)), BigUint::one()))
        );
        // A zero denominator bound admits nothing.
        assert_eq!(
            rational_reconstruct_bounded(
                &BigUint::from_u64(5),
                &m,
                &BigUint::from_u64(7),
                &BigUint::zero()
            ),
            None
        );
    }

    #[test]
    #[should_panic(expected = "2·N·D < m")]
    fn rational_reconstruction_rejects_bad_bounds() {
        use super::rational_reconstruct_bounded;
        let m = BigUint::from_u64(100);
        let ten = BigUint::from_u64(10);
        // 2·10·10 = 200 ≥ 100: the uniqueness precondition fails.
        let _ = rational_reconstruct_bounded(&BigUint::from_u64(3), &m, &ten, &ten);
    }

    #[test]
    fn bpsw_structured_cases() {
        use super::{is_probable_prime_bpsw, is_strong_lucas_probable_prime};
        // Perfect squares: the Selfridge search rules them out directly.
        for square_root in [3u64, 5, 101, 1009, 65537] {
            let root = BigUint::from_u64(square_root);
            assert!(!is_probable_prime_bpsw(&root.square_ref()));
            assert!(!is_strong_lucas_probable_prime(&root.square_ref()));
        }
        let big_root = mersenne(89); // 2^89 − 1, prime
        assert!(!is_probable_prime_bpsw(&big_root.square_ref()));
        // Mersenne primes and their composite neighbours at width.
        for exponent in [61usize, 89, 107, 127] {
            assert!(
                is_probable_prime_bpsw(&mersenne(exponent)),
                "M{exponent} is prime"
            );
        }
        assert!(
            !is_probable_prime_bpsw(&mersenne(67)),
            "M67 = 193707721 · 761838257287"
        );
        assert!(!is_probable_prime_bpsw(&mersenne(2047)));
        // Large random primes from the crate's own generator, and their
        // pairwise products.
        let mut rng = SplitMix64 {
            state: 0x1234_5678_9abc_def0,
        };
        for bits in [256usize, 512, 1024] {
            let mut p = draw_below(&mut rng, &pow2(bits));
            p.set_bit(bits - 1);
            p.set_bit(0);
            while !is_probable_prime(&p) {
                p = p.add_ref(&BigUint::from_u64(2));
            }
            assert!(is_probable_prime_bpsw(&p), "random prime at {bits} bits");
            let mut q = draw_below(&mut rng, &pow2(bits));
            q.set_bit(bits - 1);
            q.set_bit(0);
            while !is_probable_prime(&q) {
                q = q.add_ref(&BigUint::from_u64(2));
            }
            assert!(
                !is_probable_prime_bpsw(&p.mul_ref(&q)),
                "semiprime at {} bits",
                2 * bits
            );
        }
    }

    #[test]
    fn miller_rabin_witness_primitive() {
        // 341 = 11 * 31 is the classic Fermat 2-pseudoprime; base 3 is a
        // Miller-Rabin witness for it, while no witness testifies against a
        // genuine prime.
        let n341 = BigUint::from_u64(341);
        assert!(miller_rabin_witness(&n341, &BigUint::from_u64(3)));
        let p = BigUint::from_u64(1_000_000_007);
        for base in [2u64, 3, 5, 7, 11] {
            assert!(!miller_rabin_witness(&p, &BigUint::from_u64(base)));
        }

        // Trivial witnesses (0, ±1 mod n) can never testify.
        assert!(!miller_rabin_witness(&n341, &BigUint::zero()));
        assert!(!miller_rabin_witness(&n341, &BigUint::one()));
        assert!(!miller_rabin_witness(&n341, &BigUint::from_u64(340)));
        assert!(!miller_rabin_witness(&n341, &n341));

        // Even and degenerate candidates are composite by inspection.
        assert!(miller_rabin_witness(
            &BigUint::from_u64(100),
            &BigUint::from_u64(3)
        ));
        assert!(miller_rabin_witness(&BigUint::one(), &BigUint::from_u64(3)));
    }

    #[test]
    fn gcd_small_values() {
        let lhs = BigUint::from_u64(48);
        let rhs = BigUint::from_u64(18);
        assert_eq!(gcd(&lhs, &rhs), BigUint::from_u64(6));
    }

    #[test]
    fn lcm_small_values() {
        let lhs = BigUint::from_u64(60);
        let rhs = BigUint::from_u64(52);
        assert_eq!(lcm(&lhs, &rhs), BigUint::from_u64(780));
    }

    #[test]
    fn modular_exponentiation_small_values() {
        let base = BigUint::from_u64(7);
        let exponent = BigUint::from_u64(560);
        let modulus = BigUint::from_u64(561);
        assert_eq!(mod_pow(&base, &exponent, &modulus), BigUint::from_u64(1));
    }

    #[test]
    fn miller_rabin_rejects_composites() {
        assert!(!is_probable_prime(&BigUint::from_u64(561)));
        assert!(!is_probable_prime(&BigUint::from_u64(341)));
        assert!(!is_probable_prime(&BigUint::from_u64(221)));
    }

    #[test]
    fn miller_rabin_accepts_primes() {
        assert!(is_probable_prime(&BigUint::from_u64(65_537)));
        assert!(is_probable_prime(&BigUint::from_be_bytes(&[
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xc5
        ])));
    }

    #[test]
    fn miller_rabin_rejects_empty_witness_sets() {
        assert!(!is_probable_prime_with_bases(
            &BigUint::from_u64(65_537),
            &[]
        ));
    }

    #[test]
    fn miller_rabin_rejects_all_trivial_witness_sets() {
        // 1_022_117 = 1009 × 1013 clears the trial sieve (no prime factor
        // ≤ 997). Each base is reduced modulo n before it is classified: one
        // reducing to 0, 1, or n − 1 is the trivial ±1 case and testifies to
        // nothing. A non-empty set of only such bases runs zero effective
        // rounds, and the composite must not then be reported prime — that is
        // the effective-rounds guard, which the all-trivial cases below cover
        // (replacing `effective_rounds > 0` with `true` fails them).
        let n = BigUint::from_u64(1_022_117);
        assert!(!is_probable_prime_with_bases(&n, &[1_022_116])); // ≡ n − 1
        assert!(!is_probable_prime_with_bases(&n, &[1_022_117])); // ≡ 0
        assert!(!is_probable_prime_with_bases(&n, &[1, 1_022_116])); // 1 and n − 1
                                                                     // By contrast, a large *unreduced* base that reduces to a genuine
                                                                     // witness (u64::MAX ≡ 807_583) must still expose the composite — the
                                                                     // bug the reduce-first change fixed, where such bases were dropped.
        assert!(!is_probable_prime_with_bases(&n, &[u64::MAX]));
        assert!(!is_probable_prime_with_bases(&n, &[2]));
        assert!(!is_probable_prime(&n));
    }

    #[test]
    fn miller_rabin_wrapper_agrees_with_single_round_on_trivial_bases() {
        // Second-pass §2.3: the batch wrapper and the single-round primitive
        // share one trivial-base rule — reduce modulo n, discard {0, 1, n−1}.
        let prime = BigUint::from_u64(1_000_000_007); // large, reaches MR
        let composite = BigUint::from_u64(1_022_117); // 1009 × 1013, sieve-surviving

        // 0 and 1 are trivial: the single round proves nothing with them.
        assert!(!miller_rabin_witness(&prime, &BigUint::from_u64(0)));
        assert!(!miller_rabin_witness(&prime, &BigUint::from_u64(1)));
        // A leading 0 must not stamp a prime composite (the old bug), and a
        // valid base alongside it still decides.
        assert!(!is_probable_prime_with_bases(&prime, &[0])); // 0 effective rounds
        assert!(is_probable_prime_with_bases(&prime, &[0, 2])); // 0 discarded, 2 decides
        assert!(is_probable_prime_with_bases(&prime, &[2]));
        // 1 never testifies: it must not stamp a composite prime.
        assert!(!is_probable_prime_with_bases(&composite, &[1]));
        // Genuine bases decide correctly on both sides.
        assert!(miller_rabin_witness(&composite, &BigUint::from_u64(2))); // 2 exposes it
        assert!(!is_probable_prime_with_bases(&composite, &[2]));
    }

    #[test]
    fn modular_inverse_small_values() {
        assert_eq!(
            mod_inverse(&BigUint::from_u64(11), &BigUint::from_u64(16)),
            Some(BigUint::from_u64(3))
        );
        assert_eq!(
            mod_inverse(&BigUint::from_u64(23), &BigUint::from_u64(46)),
            None
        );
    }
}
