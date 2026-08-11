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
//! - **Multiplication** uses a schoolbook shift-and-XOR loop followed by
//!   polynomial reduction modulo the field polynomial.
//! - **Inversion** uses the binary extended GCD for polynomials over GF(2)
//!   (Algorithm 2.22 of Hankerson, Menezes, Vanstone — *Guide to ECC*).
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
        Some(Self {
            poly,
            degree: bits - 1,
        })
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
    /// Uses a shift-and-XOR loop: for each set bit `i` of `b`, XOR `a << i`
    /// into an accumulator, then reduce the accumulator.
    #[must_use]
    pub fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        if a.is_zero() || b.is_zero() {
            return BigUint::zero();
        }

        let mut acc = BigUint::zero();
        let mut temp = a.clone();
        let b_bits = b.bits();

        for i in 0..b_bits {
            if b.bit(i) {
                acc.bitxor_assign(&temp);
            }
            temp.shl1();
        }

        self.reduce(acc)
    }

    /// Square a field element.
    #[inline]
    #[must_use]
    pub fn square(&self, a: &BigUint) -> BigUint {
        self.mul(a, a)
    }

    /// Invert a field element via the binary extended GCD algorithm, or
    /// `None` for zero (which has no inverse).
    ///
    /// Algorithm 2.22 from Hankerson, Menezes, Vanstone — *Guide to ECC*.
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
    /// curve degrees are).
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
    ///
    /// Scans from the highest set bit of `a` down to the degree, and for
    /// each set bit at position `i ≥ degree`, XORs in `poly << (i − degree)`.
    /// The leading bit of the shifted polynomial is `i`, which clears bit
    /// `i`; its remaining bits all sit below `i`, so the downward scan stays
    /// valid.
    fn reduce(&self, mut a: BigUint) -> BigUint {
        let mut current_bits = a.bits();
        while current_bits > self.degree {
            let i = current_bits - 1; // position of the highest set bit
            let shift = i - self.degree;
            let mut shifted = self.poly.clone();
            shifted.shl_bits(shift);
            a.bitxor_assign(&shifted);
            current_bits = a.bits();
        }
        a
    }
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
