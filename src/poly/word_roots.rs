//! The roots of an integer polynomial modulo a prime that fits a word.
//!
//! [`PolyMod::roots`](super::PolyMod::roots) finds them over a modulus of
//! any size, with a multiprecision integer for every coefficient of every
//! intermediate polynomial. A sieve's factor base asks for the roots modulo
//! each prime below a bound of some millions, and every one of those primes
//! is a machine word. Here the same computation stays in words: the
//! coefficients are [`Montgomery64`] residues in a few buffers made once a
//! call.
//!
//! The method is the one `PolyMod::roots` uses (Cantor & Zassenhaus, *A New
//! Algorithm for Factoring Polynomials Over Finite Fields*, Mathematics of
//! Computation 36, 1981). `x^p − x` is the product of `x − r` over every
//! `r` in `𝔽_p`, so its greatest common divisor with `f` is the product of
//! `x − r` over the distinct roots of `f`. That product is split by
//! `(x + a)^((p−1)/2) − 1` for random `a`, which vanishes at the roots `r`
//! with `r + a` a nonzero square and at no other.

use super::PolyZ;
use crate::bigint::Sign;
use crate::modular_fixed::{Montgomery64, Residue64};
use crate::number_theory_impl::mod_inverse_u64;
use crate::random_impl::RandomSource;

impl PolyZ {
    /// The distinct roots in `𝔽_p` of the polynomial reduced modulo the
    /// prime `p`, ascending.
    ///
    /// The set is the one [`PolyMod::roots`](super::PolyMod::roots) gives
    /// for the same reduction: a root of any multiplicity is listed once,
    /// and a reduction of degree zero has none, the zero polynomial
    /// included. The draws from `rng` choose how the roots are separated
    /// and never which roots are returned.
    ///
    /// `prime` must be prime; over a composite the result is unspecified.
    ///
    /// # Panics
    ///
    /// When `prime < 2`, or when `rng` yields 256 draws in a row that
    /// separate nothing, which a working source does with probability
    /// below `2⁻²⁵⁶`.
    #[must_use]
    pub fn roots_mod_u64<R: RandomSource + ?Sized>(&self, prime: u64, rng: &mut R) -> Vec<u64> {
        assert!(prime >= 2, "the modulus must be a prime");
        let mut reduced: Vec<u64> = self
            .coeffs
            .iter()
            .map(|coefficient| {
                let residue = coefficient.magnitude().rem_u64(prime);
                if coefficient.sign() == Sign::Negative && residue != 0 {
                    prime - residue
                } else {
                    residue
                }
            })
            .collect();
        while reduced.last() == Some(&0) {
            reduced.pop();
        }
        let degree = reduced.len().saturating_sub(1);
        if degree == 0 {
            return Vec::new();
        }
        if degree == 1 {
            let inverse =
                mod_inverse_u64(reduced[1], prime).expect("a nonzero residue of a prime inverts");
            let negated = (prime - reduced[0]) % prime;
            return vec![((u128::from(negated) * u128::from(inverse)) % u128::from(prime)) as u64];
        }
        if enumeration_is_cheaper(prime, degree) {
            return (0..prime)
                .filter(|&x| value_at(&reduced, x, prime) == 0)
                .collect();
        }
        let field = Montgomery64::new(prime).expect("an odd prime: two is enumerated");
        // The monic associate, without its leading one.
        let inverse = field.enter(
            mod_inverse_u64(reduced[degree], prime).expect("a nonzero residue of a prime inverts"),
        );
        let lower: Vec<Residue64> = reduced[..degree]
            .iter()
            .map(|&coefficient| field.mul(field.enter(coefficient), inverse))
            .collect();
        let linear = product_of_linear_factors(&field, &lower);
        let mut roots = Vec::with_capacity(linear.len());
        split_into_roots(&field, linear, rng, &mut roots);
        roots.sort_unstable();
        roots
    }
}

/// Whether trying every residue costs less than the algebra does.
///
/// Trying them is `p` evaluations of `d` multiplications. The algebra is
/// led by `x^p` modulo `f`: a squaring and its reduction, about `2d²`
/// multiplications, for each of the `bits` bits of `p`. So the residues are
/// tried when `p ≤ 2·d·bits`, which for a quintic is `p < 64`. Two is
/// always tried, as the Montgomery form needs an odd modulus.
fn enumeration_is_cheaper(prime: u64, degree: usize) -> bool {
    let bits = u64::from(u64::BITS - prime.leading_zeros());
    prime <= 2 * degree as u64 * bits
}

/// The value at `x` of the polynomial with these residues for coefficients,
/// low first, by Horner's rule.
fn value_at(coefficients: &[u64], x: u64, prime: u64) -> u64 {
    let (x, prime) = (u128::from(x), u128::from(prime));
    coefficients
        .iter()
        .rev()
        .fold(0u128, |value, &coefficient| {
            (value * x + u128::from(coefficient)) % prime
        }) as u64
}

/// Arithmetic of `𝔽_p[x]` modulo a monic polynomial: an element is a
/// polynomial below its degree, low coefficient first.
struct Quotient<'a> {
    field: &'a Montgomery64,
    /// The modulus without its leading one: as many coefficients as its
    /// degree.
    lower: &'a [Residue64],
    /// Room for a product before it is reduced.
    wide: Vec<Residue64>,
}

impl<'a> Quotient<'a> {
    fn new(field: &'a Montgomery64, lower: &'a [Residue64]) -> Self {
        Self {
            field,
            lower,
            wide: vec![field.zero(); 2 * lower.len()],
        }
    }

    /// `element` squared.
    fn square(&mut self, element: &mut [Residue64]) {
        let (field, degree) = (self.field, self.lower.len());
        self.wide.fill(field.zero());
        for i in 0..degree {
            self.wide[2 * i] = field.add(self.wide[2 * i], field.square(element[i]));
            for j in i + 1..degree {
                let cross = field.mul(element[i], element[j]);
                self.wide[i + j] = field.add(self.wide[i + j], field.add(cross, cross));
            }
        }
        // The top coefficient at each step is cleared by its multiple of
        // the modulus, whose leading term is one.
        for top in (degree..2 * degree - 1).rev() {
            let leading = self.wide[top];
            for (j, &coefficient) in self.lower.iter().enumerate() {
                let place = top - degree + j;
                self.wide[place] = field.sub(self.wide[place], field.mul(leading, coefficient));
            }
        }
        element.copy_from_slice(&self.wide[..degree]);
    }

    /// `element` times `x + shift`.
    fn times_linear(&self, element: &mut [Residue64], shift: Residue64) {
        let (field, degree) = (self.field, self.lower.len());
        let leading = element[degree - 1];
        for i in (0..degree).rev() {
            let below = if i == 0 { field.zero() } else { element[i - 1] };
            let shifted = field.add(below, field.mul(shift, element[i]));
            element[i] = field.sub(shifted, field.mul(leading, self.lower[i]));
        }
    }

    /// `(x + shift)^exponent`, for a positive exponent, by the exponent's
    /// bits from the top.
    fn power_of_linear(&mut self, shift: Residue64, exponent: u64) -> Vec<Residue64> {
        let field = self.field;
        let mut power = vec![field.zero(); self.lower.len()];
        power[0] = field.one();
        self.times_linear(&mut power, shift);
        for bit in (0..exponent.ilog2()).rev() {
            self.square(&mut power);
            if exponent >> bit & 1 == 1 {
                self.times_linear(&mut power, shift);
            }
        }
        power
    }
}

/// The product of `x − r` over the distinct roots `r` of the monic
/// polynomial with these lower coefficients: its greatest common divisor
/// with `x^p − x`. Monic, without its leading one, so its length is the
/// number of roots.
fn product_of_linear_factors(field: &Montgomery64, lower: &[Residue64]) -> Vec<Residue64> {
    let mut quotient = Quotient::new(field, lower);
    let mut difference = quotient.power_of_linear(field.zero(), field.modulus());
    // Less `x`, which is below the modulus's degree of two or more.
    difference[1] = field.sub(difference[1], field.one());
    monic_gcd(field, lower.to_vec(), difference)
}

/// The monic greatest common divisor of a monic polynomial, given without
/// its leading one, and `other`, given whole and below it in degree. The
/// result is without its leading one.
fn monic_gcd(
    field: &Montgomery64,
    mut monic: Vec<Residue64>,
    mut other: Vec<Residue64>,
) -> Vec<Residue64> {
    loop {
        while other.last() == Some(&field.zero()) {
            other.pop();
        }
        let Some(leading) = other.pop() else {
            return monic;
        };
        // `other` made monic, then `monic` reduced by it: the pair is
        // replaced by the divisor and the remainder, as Euclid's algorithm
        // has it.
        let inverse = field.pow(leading, field.modulus() - 2);
        for coefficient in &mut other {
            *coefficient = field.mul(*coefficient, inverse);
        }
        let remainder = remainder_by_monic(field, &monic, &other);
        monic = other;
        other = remainder;
    }
}

/// The remainder of one monic polynomial by another of no greater degree,
/// both without their leading ones; the remainder is whole and may end in
/// zeros.
fn remainder_by_monic(
    field: &Montgomery64,
    dividend: &[Residue64],
    divisor: &[Residue64],
) -> Vec<Residue64> {
    let mut work = dividend.to_vec();
    work.push(field.one());
    let degree = divisor.len();
    for top in (degree..work.len()).rev() {
        let leading = work[top];
        for (j, &coefficient) in divisor.iter().enumerate() {
            let place = top - degree + j;
            work[place] = field.sub(work[place], field.mul(leading, coefficient));
        }
    }
    work.truncate(degree);
    work
}

/// The quotient of one monic polynomial by a monic divisor of it, all three
/// without their leading ones.
fn quotient_by_monic(
    field: &Montgomery64,
    dividend: &[Residue64],
    divisor: &[Residue64],
) -> Vec<Residue64> {
    let mut work = dividend.to_vec();
    work.push(field.one());
    let degree = divisor.len();
    for top in (degree..work.len()).rev() {
        let leading = work[top];
        for (j, &coefficient) in divisor.iter().enumerate() {
            let place = top - degree + j;
            work[place] = field.sub(work[place], field.mul(leading, coefficient));
        }
    }
    // The quotient's coefficients are the leading terms the division
    // cleared: they stand where it left them, above the remainder, and the
    // highest of them is the quotient's leading one.
    work.drain(..degree);
    work.pop();
    work
}

/// The most draws in a row that separate nothing before the source is
/// taken to be broken. A draw separates two given roots with probability
/// about a half, so a working source fails this often with probability
/// below `2⁻²⁵⁶`.
const MOST_STALLED_DRAWS: usize = 256;

/// The roots of a product of distinct monic linear factors, given without
/// its leading one, pushed onto `roots` as plain residues.
///
/// A factor of degree one names its root. A longer one is split by its
/// greatest common divisor with `(x + a)^((p−1)/2) − 1` for a random `a`:
/// that polynomial vanishes at a root `r` exactly when `r + a` is a nonzero
/// square, which about half the roots are, independently enough that two
/// roots part with probability about a half.
fn split_into_roots<R: RandomSource + ?Sized>(
    field: &Montgomery64,
    product: Vec<Residue64>,
    rng: &mut R,
    roots: &mut Vec<u64>,
) {
    let mut pending = vec![product];
    let mut stalled = 0usize;
    while let Some(factor) = pending.pop() {
        match factor.len() {
            0 => {}
            1 => roots.push(field.exit(field.sub(field.zero(), factor[0]))),
            degree => {
                assert!(
                    stalled < MOST_STALLED_DRAWS,
                    "no root was separated in {MOST_STALLED_DRAWS} draws: the random source \
                     yields no entropy, or the modulus is not a prime"
                );
                let mut bytes = [0u8; 8];
                rng.fill_bytes(&mut bytes);
                let shift = field.enter(u64::from_le_bytes(bytes));
                let mut quotient = Quotient::new(field, &factor);
                let mut power = quotient.power_of_linear(shift, (field.modulus() - 1) / 2);
                power[0] = field.sub(power[0], field.one());
                let part = monic_gcd(field, factor.clone(), power);
                if part.is_empty() || part.len() == degree {
                    stalled += 1;
                    pending.push(factor);
                } else {
                    stalled = 0;
                    pending.push(quotient_by_monic(field, &factor, &part));
                    pending.push(part);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::PolyMod;
    use super::*;
    use crate::bigint::{BigInt, BigUint};
    use crate::number_theory_impl::primes_below;

    /// A generator for tests: a linear congruential sequence, its high
    /// bytes taken.
    struct Sequence(u64);

    impl RandomSource for Sequence {
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for byte in dest {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *byte = (self.0 >> 56) as u8;
            }
        }
    }

    fn polynomial(coefficients: &[i64]) -> PolyZ {
        PolyZ::new(coefficients.iter().map(|&c| BigInt::from_i64(c)).collect())
    }

    /// The roots by the multiprecision finder, as words.
    fn roots_by_poly_mod(f: &PolyZ, prime: u64, rng: &mut Sequence) -> Vec<u64> {
        PolyMod::from_poly_z(f, &BigUint::from_u64(prime))
            .roots(rng)
            .iter()
            .map(|root| root.to_u64().expect("a residue of a word"))
            .collect()
    }

    #[test]
    fn the_roots_are_those_of_every_residue_tried() {
        let polynomials = [
            polynomial(&[-6, 11, -6, 1]),
            polynomial(&[1, 0, 1]),
            polynomial(&[7, -3, 0, 2, 5, 9]),
            polynomial(&[0, 0, 3, 1]),
            polynomial(&[-120, 274, -225, 85, -15, 1]),
            polynomial(&[5, 4, 3, 2, 1, 6, 7]),
            polynomial(&[4, 4, 1]),
            polynomial(&[30, 0, 0, 0, 0, 0, 15]),
        ];
        let mut rng = Sequence(1);
        for f in &polynomials {
            for prime in primes_below(600) {
                let reduced: Vec<u64> = f
                    .coefficients()
                    .iter()
                    .map(|c| {
                        c.rem_euclid(&BigUint::from_u64(prime))
                            .to_u64()
                            .expect("a residue")
                    })
                    .collect();
                let constant = reduced.iter().skip(1).all(|&c| c == 0);
                let tried: Vec<u64> = if constant {
                    Vec::new()
                } else {
                    (0..prime)
                        .filter(|&x| value_at(&reduced, x, prime) == 0)
                        .collect()
                };
                assert_eq!(
                    f.roots_mod_u64(prime, &mut rng),
                    tried,
                    "{:?} modulo {prime}",
                    f.coefficients()
                );
            }
        }
    }

    #[test]
    fn the_roots_are_those_of_the_multiprecision_finder() {
        // A quintic and a sextic with coefficients of many words, over
        // primes of every size a word holds.
        let wide = |seed: u64, degree: usize| -> PolyZ {
            let mut rng = Sequence(seed);
            PolyZ::new(
                (0..=degree)
                    .map(|_| {
                        let mut bytes = [0u8; 40];
                        rng.fill_bytes(&mut bytes);
                        let magnitude = BigUint::from_be_bytes(&bytes);
                        if bytes[0] & 1 == 1 {
                            BigInt::from_parts(Sign::Negative, magnitude)
                        } else {
                            BigInt::from_biguint(magnitude)
                        }
                    })
                    .collect(),
            )
        };
        let primes = [
            2u64,
            3,
            61,
            67,
            251,
            65_537,
            1_000_003,
            4_294_967_291,
            1_099_511_627_791,
            18_446_744_073_709_551_557,
        ];
        let mut rng = Sequence(7);
        let mut found = 0usize;
        for (seed, degree) in [(11u64, 5usize), (12, 6), (13, 4), (14, 5)] {
            let f = wide(seed, degree);
            for &prime in &primes {
                let expected = roots_by_poly_mod(&f, prime, &mut rng);
                found += expected.len();
                assert_eq!(f.roots_mod_u64(prime, &mut rng), expected, "modulo {prime}");
            }
            for prime in primes_below(20_000).into_iter().skip(1_800) {
                let expected = roots_by_poly_mod(&f, prime, &mut rng);
                found += expected.len();
                assert_eq!(f.roots_mod_u64(prime, &mut rng), expected, "modulo {prime}");
            }
        }
        // The comparison is of something: a polynomial has a root modulo
        // a prime about once a prime.
        assert!(found > 1_000, "{found} roots compared");
    }

    #[test]
    fn a_polynomial_that_splits_gives_every_root_once() {
        // (x − 1)(x − 2)…(x − 6) and its square have the same six roots.
        let mut f = polynomial(&[1]);
        for root in 1..=6i64 {
            f = f.mul(&polynomial(&[-root, 1]));
        }
        let squared = f.mul(&f);
        let mut rng = Sequence(3);
        for prime in [101u64, 1_000_003, 4_294_967_291] {
            assert_eq!(f.roots_mod_u64(prime, &mut rng), [1, 2, 3, 4, 5, 6]);
            assert_eq!(squared.roots_mod_u64(prime, &mut rng), [1, 2, 3, 4, 5, 6]);
        }
    }

    #[test]
    fn the_roots_do_not_depend_on_the_draws() {
        let f = polynomial(&[-120, 274, -225, 85, -15, 1]);
        for prime in [1_000_003u64, 998_244_353] {
            let first = f.roots_mod_u64(prime, &mut Sequence(1));
            for seed in 2..40 {
                assert_eq!(f.roots_mod_u64(prime, &mut Sequence(seed)), first);
            }
        }
    }

    #[test]
    fn a_reduction_of_degree_zero_has_no_root() {
        let mut rng = Sequence(5);
        // Every coefficient vanishes modulo 5; only the constant survives
        // modulo 7.
        assert!(polynomial(&[5, 10, 15])
            .roots_mod_u64(5, &mut rng)
            .is_empty());
        assert!(polynomial(&[3, 7, 14])
            .roots_mod_u64(7, &mut rng)
            .is_empty());
        assert!(polynomial(&[]).roots_mod_u64(7, &mut rng).is_empty());
    }

    #[test]
    fn a_leading_coefficient_that_vanishes_lowers_the_degree() {
        // 7x³ + x − 3 is x − 3 modulo 7, and 14x² + 2x + 4 is 2x + 4.
        let mut rng = Sequence(9);
        assert_eq!(polynomial(&[-3, 1, 0, 7]).roots_mod_u64(7, &mut rng), [3]);
        assert_eq!(polynomial(&[4, 2, 14]).roots_mod_u64(7, &mut rng), [5]);
    }
}
