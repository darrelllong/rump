//! Arithmetic in binary extension fields GF(2^m).
//!
//! Elements of GF(2^m) are represented as [`BigUint`] values whose bit
//! pattern encodes a polynomial over GF(2): bit `i` set means the coefficient
//! of `x^i` is 1. A [`Gf2m`] context holds the field modulus — an irreducible
//! polynomial of degree `m`, stored with the same bit-pattern convention —
//! and derives the degree from it, so the two can never disagree.
//!
//! Binary fields underlie the FIPS binary elliptic curves, but also error
//! correcting codes, CRCs, and LFSR analysis — none of which this crate has
//! opinions about. Irreducibility of the modulus is the caller's contract;
//! for a reducible polynomial the ring has zero divisors and [`Gf2m::inverse`]
//! can fail on non-zero elements.
//!
//! ## Algorithm notes
//!
//! - **Addition** is XOR (no reduction needed: XOR of two polynomials of
//!   degree < m has degree < m).
//! - **Multiplication** uses the left-to-right comb method with 4-bit windows
//!   (Algorithm 2.36 of Hankerson, Menezes, Vanstone — *Guide to ECC*),
//!   followed by polynomial reduction modulo the field polynomial.
//! - **Inversion** uses the extended Euclidean algorithm for polynomials over
//!   GF(2) (Algorithm 2.48 of Hankerson, Menezes, Vanstone — *Guide to ECC*).
//! - **Half-trace** computes HT(c) = Σ c^{2^{2i}} (i = 0...(m−1)/2), which
//!   is a root of z² + z = c for any c in GF(2^m) with Tr(c) = 0. This
//!   solves the quadratic needed for compressed-point decompression on
//!   binary curves with odd m.

use crate::bigint::BigUint;

/// A binary extension field GF(2^m), defined by its irreducible polynomial.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gf2m {
    poly: BigUint,
    degree: usize,
    // Bit positions of the polynomial below the leading term, ascending —
    // the reduction taps. Sparse for every standard modulus (FIPS binary
    // curves are trinomials and pentanomials), and reduction cost scales
    // with this weight.
    taps: Vec<usize>,
}

impl Gf2m {
    /// Build the field GF(2^m) from its modulus polynomial, or `None` when
    /// the polynomial has degree below one (a constant defines no field).
    ///
    /// The degree is `poly.bits() - 1` — derived, never supplied, so it
    /// cannot fall out of step with the polynomial.
    #[must_use]
    pub fn new(poly: BigUint) -> Option<Self> {
        let bits = poly.bits();
        if bits < 2 {
            return None;
        }
        let degree = bits - 1;
        let taps = (0..degree).filter(|&i| poly.bit(i)).collect();
        Some(Self { poly, degree, taps })
    }

    /// The field's modulus polynomial.
    #[must_use]
    pub fn modulus(&self) -> &BigUint {
        &self.poly
    }

    /// The field degree `m`.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Add two field elements: XOR, with no reduction needed.
    ///
    /// An associated function rather than a method because addition does not
    /// depend on the field polynomial — any two GF(2) polynomials add this
    /// way.
    #[inline]
    #[must_use]
    pub fn add(a: &BigUint, b: &BigUint) -> BigUint {
        let mut out = a.clone();
        out.bitxor_assign(b);
        out
    }

    /// Multiply two field elements modulo the field polynomial.
    ///
    /// Left-to-right comb multiplication with 4-bit windows (Hankerson,
    /// Menezes, Vanstone — *Guide to ECC*, Algorithm 2.36): precompute the
    /// sixteen products `u(x)·b(x)` for 4-bit `u`, then sweep `a` one window
    /// at a time, shifting the accumulator four bits between sweeps. Word
    /// arithmetic throughout; the double-width product is then reduced
    /// tap-wise.
    #[must_use]
    pub fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        if a.is_zero() || b.is_zero() {
            return BigUint::zero();
        }

        let a_limbs = a.limbs();
        let b_limbs = b.limbs();
        let stride = b_limbs.len() + 1;

        // table[u] = u(x) · b(x), built incrementally: even entries shift a
        // smaller one, odd entries add b.
        let mut table = vec![0u64; 16 * stride];
        table[stride..stride + b_limbs.len()].copy_from_slice(b_limbs);
        for u in 2..16 {
            let (lo, hi) = table.split_at_mut(u * stride);
            let dst = &mut hi[..stride];
            if u % 2 == 0 {
                let src = &lo[(u / 2) * stride..(u / 2) * stride + stride];
                let mut carry = 0u64;
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = (s << 1) | carry;
                    carry = s >> 63;
                }
            } else {
                let src = &lo[(u - 1) * stride..u * stride];
                for (i, d) in dst.iter_mut().enumerate() {
                    *d = src[i] ^ b_limbs.get(i).copied().unwrap_or(0);
                }
            }
        }

        let mut product = vec![0u64; a_limbs.len() + stride];
        for window in (0..16).rev() {
            for (j, &limb) in a_limbs.iter().enumerate() {
                let u = ((limb >> (4 * window)) & 0xF) as usize;
                if u != 0 {
                    let entry = &table[u * stride..(u + 1) * stride];
                    for (i, &w) in entry.iter().enumerate() {
                        product[j + i] ^= w;
                    }
                }
            }
            if window > 0 {
                let mut carry = 0u64;
                for limb in product.iter_mut() {
                    let next = *limb >> 60;
                    *limb = (*limb << 4) | carry;
                    carry = next;
                }
            }
        }

        self.reduce_limbs(&mut product);
        BigUint::from_limbs(product)
    }

    /// Square a field element (Hankerson, Menezes, Vanstone — *Guide to ECC*,
    /// Algorithm 2.39, "Polynomial squaring").
    ///
    /// Squaring is linear over GF(2): `(Σ aᵢxⁱ)² = Σ aᵢx²ⁱ`, because every
    /// cross term appears twice and cancels. Spreading each bit to twice its
    /// position (via a precomputed byte table) and reducing costs O(m), against
    /// O(m²) for a general multiply — and squaring chains are the backbone of
    /// [`Self::trace`], [`Self::sqrt`], [`Self::half_trace`], and binary-curve
    /// doubling.
    #[must_use]
    pub fn square(&self, a: &BigUint) -> BigUint {
        let limbs = a.limbs();
        let mut spread = Vec::with_capacity(limbs.len() * 2);
        for &limb in limbs {
            spread.push(spread_half(limb as u32));
            spread.push(spread_half((limb >> 32) as u32));
        }
        self.reduce_limbs(&mut spread);
        BigUint::from_limbs(spread)
    }

    /// Raise a field element to an integer power by square-and-multiply.
    ///
    /// `pow(a, 0)` is one for every `a`, matching [`crate::mod_pow`].
    #[must_use]
    pub fn pow(&self, base: &BigUint, exponent: &BigUint) -> BigUint {
        let bits = exponent.bits();
        if bits == 0 {
            return BigUint::one();
        }

        let base = self.reduce(base.clone());
        let mut acc = base.clone();
        for i in (0..bits - 1).rev() {
            acc = self.square(&acc);
            if exponent.bit(i) {
                acc = self.mul(&acc, &base);
            }
        }
        acc
    }

    /// Divide one field element by another: `a · b⁻¹`, or `None` for a zero
    /// divisor.
    #[must_use]
    pub fn div(&self, a: &BigUint, b: &BigUint) -> Option<BigUint> {
        Some(self.mul(a, &self.inverse(b)?))
    }

    /// The unique square root: squaring is a bijection (the Frobenius map)
    /// in GF(2^m), and its inverse is `a ↦ a^{2^{m−1}}` — square `m − 1`
    /// times.
    #[must_use]
    pub fn sqrt(&self, a: &BigUint) -> BigUint {
        let mut root = self.reduce(a.clone());
        for _ in 1..self.degree {
            root = self.square(&root);
        }
        root
    }

    /// Solve `z² + z = c`, at any field degree, or `None` when no solution
    /// exists (exactly when `Tr(c) = 1`; the other root is always `z + 1`).
    ///
    /// Odd degrees use the half-trace. Even degrees use the classic
    /// construction (IEEE P1363, A.4.7): with any `δ` of trace one,
    /// `z = Σ_{i=0}^{m−2} sᵢ δ^{2^i}` where `sᵢ = Σ_{j=i+1}^{m−1} c^{2^j}`
    /// — and `s₀ = c` when `Tr(c) = 0`, so the suffix sums peel off one
    /// squaring at a time.
    #[must_use]
    pub fn solve_quadratic(&self, c: &BigUint) -> Option<BigUint> {
        let c = self.reduce(c.clone());
        if self.trace(&c) == 1 {
            return None;
        }
        if self.degree % 2 == 1 {
            return Some(self.half_trace(&c));
        }

        // Any trace-one element drives the construction; half of the field
        // qualifies, so a scan from small constants ends fast.
        let mut delta = BigUint::one();
        while self.trace(&delta) == 0 {
            delta = delta.add_ref(&BigUint::one());
        }

        let mut suffix = c.clone(); // s_0 = Tr(c) + c = c
        let mut c_power = c.clone(); // c^{2^i}
        let mut delta_power = delta; // δ^{2^i}
        let mut z = BigUint::zero();
        for _ in 0..self.degree - 1 {
            z.bitxor_assign(&self.mul(&suffix, &delta_power));
            c_power = self.square(&c_power);
            suffix.bitxor_assign(&c_power);
            delta_power = self.square(&delta_power);
        }

        debug_assert!(
            Self::add(&self.square(&z), &z) == c,
            "the construction must satisfy its own equation"
        );
        Some(z)
    }

    /// The absolute trace `Tr(c) = Σ_{i=0}^{m−1} c^{2^i}`, always 0 or 1
    /// (IEEE Std 1363-2000, Annex A.4.5, "Trace").
    ///
    /// The trace decides solvability of `z² + z = c`: a solution exists
    /// exactly when `Tr(c) = 0` — the precondition of [`Self::half_trace`],
    /// now checkable through the public API.
    #[must_use]
    pub fn trace(&self, c: &BigUint) -> u8 {
        let mut power = self.reduce(c.clone());
        let mut acc = power.clone();
        for _ in 1..self.degree {
            power = self.square(&power);
            acc.bitxor_assign(&power);
        }
        debug_assert!(
            acc.is_zero() || acc.is_one(),
            "the trace lands in the prime subfield"
        );
        u8::from(acc.is_one())
    }

    /// Test whether a GF(2) polynomial is irreducible (Rabin's test).
    ///
    /// [`Gf2m::new`] trusts its polynomial, the cheap default for the fixed,
    /// published moduli of the FIPS curves; run this when the polynomial
    /// arrives from an untrusted source — the same posture the parent
    /// cryptography crate takes for primality. Rabin's criterion
    /// (*Probabilistic algorithms in finite fields*, 1980, here in its
    /// deterministic GF(2) form): `f` of degree `m` is irreducible iff
    /// `x^{2^m} ≡ x (mod f)` and, for every prime `q` dividing `m`,
    /// `gcd(x^{2^{m/q}} − x, f) = 1`.
    #[must_use]
    pub fn is_irreducible(poly: &BigUint) -> bool {
        let bits = poly.bits();
        if bits < 2 {
            return false; // constants are units or zero, not irreducible
        }
        let degree = bits - 1;
        if degree == 1 {
            return true; // x and x + 1
        }

        // Arithmetic modulo f needs no irreducibility, so the context's own
        // reduction machinery drives the test.
        let ring = Self::new(poly.clone()).expect("degree checked above");
        let x = BigUint::from_u64(2);

        // Frobenius orbit: squaring i times gives x^(2^i) mod f. Walk the
        // checkpoints m/q in ascending order — one forward pass visits each.
        let mut checkpoints: Vec<usize> =
            prime_divisors(degree).iter().map(|&q| degree / q).collect();
        checkpoints.sort_unstable();

        let mut frobenius = x.clone();
        let mut steps = 0usize;
        for target in checkpoints {
            while steps < target {
                frobenius = ring.square(&frobenius);
                steps += 1;
            }
            if !gf2_poly_gcd(Self::add(&frobenius, &x), poly.clone()).is_one() {
                return false;
            }
        }
        while steps < degree {
            frobenius = ring.square(&frobenius);
            steps += 1;
        }
        frobenius == x
    }

    /// Invert a field element via the extended Euclidean algorithm over
    /// `GF(2)[x]`, or `None` for zero (which has no inverse).
    ///
    /// Algorithm 2.48 from Hankerson, Menezes, Vanstone — *Guide to ECC*: each
    /// step cancels the leading term of the higher-degree remainder by adding a
    /// shifted copy of the other (`u ^= v · x^{deg u − deg v}`), carrying the
    /// single cofactor of `a` alongside.
    /// Invariant during the loop: `b ≡ u · a (mod poly)` in the sense that
    /// `b` and `u` are updated in lockstep so that `b = u · a XOR s · poly`
    /// for some polynomial `s` we do not track.
    #[must_use]
    pub fn inverse(&self, a: &BigUint) -> Option<BigUint> {
        if a.is_zero() {
            return None;
        }

        let mut u = a.clone();
        let mut v = self.poly.clone();
        let mut b = BigUint::one();
        let mut c = BigUint::zero();

        // Loop until u = 1 (degree 0 polynomial over GF(2)).
        while !u.is_one() {
            // How many bits separate the leading terms of u and v?
            let deg_u = u.bits(); // deg(u) + 1
            let deg_v = v.bits(); // deg(v) + 1

            // Ensure deg(u) >= deg(v) by swapping if necessary.
            if deg_u < deg_v {
                core::mem::swap(&mut u, &mut v);
                core::mem::swap(&mut b, &mut c);
            }

            // j = deg(u) - deg(v), guaranteed >= 0 after the potential swap.
            // Use saturating_sub defensively (logically it's always exact).
            let j = u.bits().saturating_sub(v.bits());

            // u = u XOR (v * x^j);  b = b XOR (c * x^j).
            let mut sv = v.clone();
            sv.shl_bits(j);
            u.bitxor_assign(&sv);

            let mut sc = c.clone();
            sc.shl_bits(j);
            b.bitxor_assign(&sc);
        }

        // u is now 1; b satisfies b · a ≡ 1 (mod poly), possibly with
        // degree ≥ m.
        Some(self.reduce(b))
    }

    /// Compute the half-trace HT(c) = Σ_{i=0}^{(m−1)/2} c^{2^{2i}}.
    ///
    /// For any `c` with absolute trace Tr(c) = 0, `z = HT(c)` solves
    /// `z² + z = c` — the quadratic behind compressed-point decompression on
    /// binary curves. The field degree must be odd (all FIPS 186-4 binary
    /// curve degrees are); [`Self::solve_quadratic`] is the total form that
    /// handles every degree and checks the trace itself.
    #[must_use]
    pub fn half_trace(&self, c: &BigUint) -> BigUint {
        // HT(c) = c^{2^0} + c^{2^2} + c^{2^4} + ... + c^{2^{degree-1}}
        // Starting from power = c, square twice per iteration to advance by
        // 2 exponent steps: c → c^4 → c^{16} → ...
        let mut t = c.clone(); // accumulator starts at c^{2^0}
        let mut power = c.clone(); // current term

        for _ in 0..(self.degree - 1) / 2 {
            // Advance power from c^{2^{2i}} to c^{2^{2(i+1)}} = c^{2^{2i+2}}.
            power = self.square(&self.square(&power));
            t.bitxor_assign(&power);
        }

        t
    }

    /// Reduce `a` modulo the field polynomial.
    fn reduce(&self, a: BigUint) -> BigUint {
        if a.bits() <= self.degree {
            return a;
        }
        let mut limbs = a.limbs().to_vec();
        self.reduce_limbs(&mut limbs);
        BigUint::from_limbs(limbs)
    }

    /// Reduce a limb buffer modulo the field polynomial, in place.
    ///
    /// The word-at-a-time tap folding of *Guide to ECC* §2.3.5 (fast reduction,
    /// e.g. Algorithms 2.41–2.45 for the specific NIST polynomials), here
    /// generalized to fold at the taps of any reduction polynomial rather than
    /// a hard-coded one.
    ///
    /// One whole word of excess coefficients at a time, top down: a word `w`
    /// whose bits sit at positions `degree + k` folds back as `w << t` at
    /// each reduction tap `t`, and clearing the source word is what the
    /// polynomial's leading term would have done. Cost scales with the
    /// polynomial's weight — constant per word for the trinomial and
    /// pentanomial moduli every standard uses. Small tap gaps can re-raise
    /// bits above the degree inside the boundary word; the outer loop
    /// re-scans until the buffer is clean, and each pass strictly shrinks
    /// the excess.
    fn reduce_limbs(&self, buf: &mut Vec<u64>) {
        let boundary_word = self.degree / 64;
        let boundary_bit = self.degree % 64;

        loop {
            let bits = limbs_bits(buf);
            if bits <= self.degree {
                break;
            }
            let top_word = (bits - 1) / 64;

            let (excess, base_shift) = if top_word > boundary_word {
                let w = buf[top_word];
                buf[top_word] = 0;
                (w, top_word * 64 - self.degree)
            } else {
                let w = buf[top_word] >> boundary_bit;
                buf[top_word] &= (1u64 << boundary_bit) - 1;
                (w, 0)
            };

            for &t in &self.taps {
                xor_shifted_word(buf, excess, base_shift + t);
            }
        }

        while buf.last() == Some(&0) {
            buf.pop();
        }
    }
}

/// Interleave a zero bit after every bit of `half` — the squaring map on
/// one 32-bit word, via an 8-bit spread table.
#[inline]
fn spread_half(half: u32) -> u64 {
    const SPREAD: [u16; 256] = {
        let mut table = [0u16; 256];
        let mut byte = 0usize;
        while byte < 256 {
            let mut spread = 0u16;
            let mut bit = 0;
            while bit < 8 {
                if byte & (1 << bit) != 0 {
                    spread |= 1 << (2 * bit);
                }
                bit += 1;
            }
            table[byte] = spread;
            byte += 1;
        }
        table
    };

    u64::from(SPREAD[(half & 0xFF) as usize])
        | u64::from(SPREAD[((half >> 8) & 0xFF) as usize]) << 16
        | u64::from(SPREAD[((half >> 16) & 0xFF) as usize]) << 32
        | u64::from(SPREAD[(half >> 24) as usize]) << 48
}

/// Significant bits of a little-endian limb buffer (trailing zero words
/// permitted).
fn limbs_bits(buf: &[u64]) -> usize {
    for (i, &limb) in buf.iter().enumerate().rev() {
        if limb != 0 {
            return i * 64 + (64 - limb.leading_zeros() as usize);
        }
    }
    0
}

/// XOR `word` into the buffer at the given bit offset.
fn xor_shifted_word(buf: &mut [u64], word: u64, bit_offset: usize) {
    let index = bit_offset / 64;
    let shift = bit_offset % 64;
    buf[index] ^= word << shift;
    if shift > 0 {
        let high = word >> (64 - shift);
        if high != 0 {
            buf[index + 1] ^= high;
        }
    }
}

/// Polynomial gcd over GF(2): Euclid with XOR-shift reduction steps.
fn gf2_poly_gcd(mut a: BigUint, mut b: BigUint) -> BigUint {
    while !b.is_zero() {
        while !a.is_zero() && a.bits() >= b.bits() {
            let mut shifted = b.clone();
            shifted.shl_bits(a.bits() - b.bits());
            a.bitxor_assign(&shifted);
        }
        core::mem::swap(&mut a, &mut b);
    }
    a
}

/// Distinct prime divisors of `n`, ascending, by trial division — field
/// degrees are small enough that nothing cleverer earns its keep.
fn prime_divisors(mut n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut d = 2usize;
    while d * d <= n {
        if n.is_multiple_of(d) {
            out.push(d);
            while n.is_multiple_of(d) {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        out.push(n);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // GF(2^163) irreducible polynomial: x^163 + x^7 + x^6 + x^3 + 1
    fn gf163() -> Gf2m {
        let mut p = BigUint::zero();
        p.set_bit(163);
        p.set_bit(7);
        p.set_bit(6);
        p.set_bit(3);
        p.set_bit(0);
        Gf2m::new(p).expect("degree 163")
    }

    fn gf4() -> Gf2m {
        // GF(2^4) with poly = x^4 + x + 1 = 0b10011
        Gf2m::new(BigUint::from_u64(0b10011)).expect("degree 4")
    }

    #[test]
    fn new_derives_the_degree_and_rejects_constants() {
        assert_eq!(gf163().degree(), 163);
        assert_eq!(gf4().degree(), 4);
        assert_eq!(*gf4().modulus(), BigUint::from_u64(0b10011));
        assert!(Gf2m::new(BigUint::zero()).is_none());
        assert!(Gf2m::new(BigUint::one()).is_none());
    }

    #[test]
    fn add_is_xor() {
        let a = BigUint::from_u64(0b1010);
        let b = BigUint::from_u64(0b1100);
        assert_eq!(Gf2m::add(&a, &b), BigUint::from_u64(0b0110));

        let a = BigUint::from_u64(0xDEAD_BEEF);
        assert!(Gf2m::add(&a, &a).is_zero(), "a XOR a must be zero");
    }

    #[test]
    fn mul_small() {
        // (x^2 + 1) * (x + 1) = x^3 + x^2 + x + 1 = 0b1111
        let field = gf4();
        let a = BigUint::from_u64(0b0101);
        let b = BigUint::from_u64(0b0011);
        assert_eq!(field.mul(&a, &b), BigUint::from_u64(0b1111));
    }

    #[test]
    fn mul_reduces() {
        // x^3 * x = x^4 ≡ x + 1 in GF(2^4) with x^4 + x + 1.
        let field = gf4();
        let a = BigUint::from_u64(0b1000);
        let b = BigUint::from_u64(0b0010);
        assert_eq!(field.mul(&a, &b), BigUint::from_u64(0b0011));
    }

    #[test]
    fn square_equals_mul_self() {
        let field = gf163();
        let a = BigUint::from_u64(0x0123_4567_89AB_CDEF);
        assert_eq!(field.square(&a), field.mul(&a, &a));
    }

    #[test]
    fn inverse_roundtrip_gf163() {
        let field = gf163();
        let a = BigUint::from_u64(0xDEAD_BEEF_CAFE_F00D);
        let a_inv = field.inverse(&a).expect("non-zero is invertible");
        assert_eq!(field.mul(&a, &a_inv), BigUint::one());
    }

    #[test]
    fn inverse_edges() {
        let field = gf163();
        assert_eq!(field.inverse(&BigUint::one()), Some(BigUint::one()));
        assert!(field.inverse(&BigUint::zero()).is_none());
    }

    #[test]
    fn half_trace_solves_the_quadratic() {
        // c = x has Tr(x) = 0 in GF(2^163) (a fact about this field), so
        // z = HT(c) must satisfy z² + z = c exactly.
        let field = gf163();
        let c = BigUint::from_u64(2); // the polynomial x
        let z = field.half_trace(&c);
        let check = Gf2m::add(&field.square(&z), &z);
        assert_eq!(check, c, "HT(c)² + HT(c) must equal c");
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn random_element(field: &Gf2m, state: &mut u64) -> BigUint {
        let mut bytes = Vec::new();
        for _ in 0..field.degree().div_ceil(64) {
            bytes.extend_from_slice(&splitmix(state).to_be_bytes());
        }
        // Reduce into the field via a multiplication by one.
        field.mul(&BigUint::from_be_bytes(&bytes), &BigUint::one())
    }

    #[test]
    fn sqrt_inverts_the_frobenius_map() {
        // Squaring is a bijection in GF(2^m); sqrt must invert it exactly,
        // in both compositions, for every element.
        let field = gf163();
        let mut state = 0x5157_0163;
        for _ in 0..16 {
            let a = random_element(&field, &mut state);
            assert_eq!(field.sqrt(&field.square(&a)), a);
            assert_eq!(field.square(&field.sqrt(&a)), a);
        }
        assert!(field.sqrt(&BigUint::zero()).is_zero());
        assert!(field.sqrt(&BigUint::one()).is_one());
    }

    #[test]
    fn pow_group_facts() {
        // GF(8)*: x generates the order-7 group.
        let field = Gf2m::new(BigUint::from_u64(0b1011)).expect("degree 3");
        let x = BigUint::from_u64(0b010);
        assert_eq!(field.pow(&x, &BigUint::from_u64(7)), BigUint::one());
        assert_eq!(field.pow(&x, &BigUint::zero()), BigUint::one());
        assert!(field.pow(&BigUint::zero(), &BigUint::from_u64(5)).is_zero());

        // Fermat in GF(2^163): a^(2^m) = a for every a.
        let field = gf163();
        let mut exponent = BigUint::zero();
        exponent.set_bit(163);
        let mut state = 0x0fe2_2163;
        for _ in 0..4 {
            let a = random_element(&field, &mut state);
            assert_eq!(field.pow(&a, &exponent), a);
        }
    }

    #[test]
    fn div_inverts_mul() {
        let field = gf163();
        let mut state = 0xd1_4143;
        for _ in 0..8 {
            let a = random_element(&field, &mut state);
            let mut b = random_element(&field, &mut state);
            if b.is_zero() {
                b = BigUint::one();
            }
            assert_eq!(field.div(&field.mul(&a, &b), &b), Some(a));
        }
        assert_eq!(field.div(&BigUint::one(), &BigUint::zero()), None);
    }

    #[test]
    fn trace_is_linear_and_decides_the_quadratic() {
        let field = gf163();
        // Known values: Tr(x) = 0 in GF(2^163); Tr(1) = m mod 2 = 1.
        assert_eq!(field.trace(&BigUint::from_u64(2)), 0);
        assert_eq!(field.trace(&BigUint::one()), 1);
        assert_eq!(field.trace(&BigUint::zero()), 0);

        let mut state = 0x7ace_0163;
        for _ in 0..12 {
            let a = random_element(&field, &mut state);
            let b = random_element(&field, &mut state);
            // Linearity over GF(2).
            assert_eq!(
                field.trace(&Gf2m::add(&a, &b)),
                field.trace(&a) ^ field.trace(&b)
            );
            // For odd m: HT(c)² + HT(c) = c + Tr(c) — the half-trace solves
            // the quadratic exactly when the trace vanishes.
            let z = field.half_trace(&a);
            let residue = Gf2m::add(&field.square(&z), &z);
            let expected = if field.trace(&a) == 0 {
                a.clone()
            } else {
                Gf2m::add(&a, &BigUint::one())
            };
            assert_eq!(residue, expected);
        }
    }

    #[test]
    fn irreducibility_known_answers() {
        // Irreducible: x, x+1, x²+x+1, x³+x+1, x⁴+x+1, the AES polynomial
        // x⁸+x⁴+x³+x+1, and every FIPS binary-curve modulus.
        for poly in [0b10u64, 0b11, 0b111, 0b1011, 0b10011, 0x11B] {
            assert!(
                Gf2m::is_irreducible(&BigUint::from_u64(poly)),
                "0b{poly:b} is irreducible"
            );
        }
        let fips = [
            (163usize, &[7usize, 6, 3, 0][..]),
            (233, &[74, 0]),
            (283, &[12, 7, 5, 0]),
            (409, &[87, 0]),
            (571, &[10, 5, 2, 0]),
        ];
        for (degree, taps) in fips {
            let mut poly = BigUint::zero();
            poly.set_bit(degree);
            for &t in taps {
                poly.set_bit(t);
            }
            assert!(Gf2m::is_irreducible(&poly), "FIPS degree {degree}");
        }

        // Reducible: x², (x+1)² = x²+1, x³+1 = (x+1)(x²+x+1),
        // (x²+x+1)² = x⁴+x²+1 — the last two have composite structure that a
        // wrong Frobenius checkpoint order would miss. Constants are not
        // irreducible.
        for poly in [0b100u64, 0b101, 0b1001, 0b10101, 0b0, 0b1] {
            assert!(
                !Gf2m::is_irreducible(&BigUint::from_u64(poly)),
                "0b{poly:b} is reducible or degenerate"
            );
        }

        // Composite degree with both checkpoints live: x⁶+x+1 is
        // irreducible; x⁶+x²+x+1 is not.
        assert!(Gf2m::is_irreducible(&BigUint::from_u64(0b1000011)));
        assert!(!Gf2m::is_irreducible(&BigUint::from_u64(0b1000111)));
    }

    /// The pre-comb algorithm, kept as an independent oracle: shift-and-XOR
    /// per set bit, reduced bit-serially with no shared kernel code.
    fn mul_reference(field: &Gf2m, a: &BigUint, b: &BigUint) -> BigUint {
        let mut acc = BigUint::zero();
        let mut temp = a.clone();
        for i in 0..b.bits() {
            if b.bit(i) {
                acc.bitxor_assign(&temp);
            }
            temp.shl1();
        }
        // Bit-serial reduction, structurally unlike the tap-wise word walk.
        while acc.bits() > field.degree() {
            let shift = acc.bits() - 1 - field.degree();
            let mut shifted = field.modulus().clone();
            shifted.shl_bits(shift);
            acc.bitxor_assign(&shifted);
        }
        acc
    }

    #[test]
    fn comb_mul_matches_the_reference() {
        // Sparse FIPS-style moduli, the AES byte field, and a deliberately
        // dense degree-8 modulus whose top tap gap is 1 — the case that
        // forces the word-level reduction through repeated boundary passes.
        let mut dense = None;
        for candidate in 0x100u64..0x200 {
            let poly = BigUint::from_u64(candidate | 0x180); // x^8 + x^7 + ...
            if candidate.count_ones() >= 6 && Gf2m::is_irreducible(&poly) {
                dense = Some(poly);
                break;
            }
        }
        let dense =
            Gf2m::new(dense.expect("a dense degree-8 irreducible exists")).expect("degree 8");

        let mut b571 = BigUint::zero();
        for bit in [571usize, 10, 5, 2, 0] {
            b571.set_bit(bit);
        }

        let fields = [
            gf163(),
            gf4(),
            Gf2m::new(BigUint::from_u64(0x11B)).expect("AES field"),
            Gf2m::new(b571).expect("B-571 field"),
            dense,
        ];

        let mut state = 0xc0b_0236;
        for field in &fields {
            for _ in 0..24 {
                let a = random_element(field, &mut state);
                let b = random_element(field, &mut state);
                assert_eq!(
                    field.mul(&a, &b),
                    mul_reference(field, &a, &b),
                    "comb disagrees with reference in degree {}",
                    field.degree()
                );
            }
        }
    }

    #[test]
    fn solve_quadratic_all_degrees() {
        // Odd degree: must agree with the half-trace on trace-zero input.
        let odd = gf163();
        let mut state = 0x50_1363;
        for _ in 0..8 {
            let a = random_element(&odd, &mut state);
            let c = Gf2m::add(&odd.square(&a), &a); // Tr(c) = 0 by construction
            let z = odd.solve_quadratic(&c).expect("constructed solvable");
            assert_eq!(Gf2m::add(&odd.square(&z), &z), c);
            assert_eq!(z, odd.half_trace(&c));
        }

        // Even degrees: GF(2^4) and the AES byte field GF(2^8).
        for field in [
            gf4(),
            Gf2m::new(BigUint::from_u64(0x11B)).expect("AES field"),
        ] {
            for _ in 0..16 {
                let a = random_element(&field, &mut state);
                let c = Gf2m::add(&field.square(&a), &a);
                let z = field
                    .solve_quadratic(&c)
                    .expect("z^2 + z = a^2 + a is solvable");
                assert_eq!(Gf2m::add(&field.square(&z), &z), c);
                // The two roots are a and a + 1.
                let other = Gf2m::add(&z, &BigUint::one());
                assert!(z == a || other == a, "root must be a or a + 1");
            }

            // A trace-one element has no root.
            let mut witness = BigUint::one();
            while field.trace(&witness) == 0 {
                witness = witness.add_ref(&BigUint::one());
            }
            assert_eq!(field.solve_quadratic(&witness), None);
        }
    }

    #[test]
    fn mul_distributes_over_add() {
        let field = gf163();
        let a = BigUint::from_u64(0xABCD);
        let b = BigUint::from_u64(0x1234);
        let c = BigUint::from_u64(0x5678);
        let a_bc = field.mul(&a, &Gf2m::add(&b, &c));
        let ab_ac = Gf2m::add(&field.mul(&a, &b), &field.mul(&a, &c));
        assert_eq!(a_bc, ab_ac, "multiplication must distribute over addition");
    }

    #[test]
    fn mul_is_associative() {
        let field = gf163();
        let a = BigUint::from_u64(0x1111_2222_3333);
        let b = BigUint::from_u64(0x4444_5555_6666);
        let c = BigUint::from_u64(0x7777_8888_9999);
        assert_eq!(
            field.mul(&field.mul(&a, &b), &c),
            field.mul(&a, &field.mul(&b, &c)),
        );
    }
}
