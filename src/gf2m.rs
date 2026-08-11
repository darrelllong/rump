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
    ///
    /// Squaring is linear over GF(2): `(Σ aᵢxⁱ)² = Σ aᵢx²ⁱ`, because every
    /// cross term appears twice and cancels. Spreading each bit to twice its
    /// position and reducing costs O(m), against O(m²) for a general
    /// multiply — and squaring chains are the backbone of [`Self::trace`],
    /// [`Self::sqrt`], [`Self::half_trace`], and binary-curve doubling.
    #[must_use]
    pub fn square(&self, a: &BigUint) -> BigUint {
        let mut spread = BigUint::zero();
        for i in 0..a.bits() {
            if a.bit(i) {
                spread.set_bit(2 * i);
            }
        }
        self.reduce(spread)
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

    /// The absolute trace `Tr(c) = Σ_{i=0}^{m−1} c^{2^i}`, always 0 or 1.
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
    /// valid. One scratch copy of the polynomial is shifted down in place as
    /// the scan descends — the loop allocates nothing.
    fn reduce(&self, mut a: BigUint) -> BigUint {
        let mut current_bits = a.bits();
        if current_bits <= self.degree {
            return a;
        }

        let mut shift = current_bits - 1 - self.degree;
        let mut shifted = self.poly.clone();
        shifted.shl_bits(shift);

        loop {
            // Invariant: shifted = poly << shift, with its leading bit at
            // a's current highest set bit.
            a.bitxor_assign(&shifted);
            current_bits = a.bits();
            if current_bits <= self.degree {
                return a;
            }
            let next_shift = current_bits - 1 - self.degree;
            shifted.shr_bits(shift - next_shift);
            shift = next_shift;
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
