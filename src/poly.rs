//! Dense univariate polynomials over ℤ and over ℤ/pℤ.
//!
//! Two concrete named types, in the crate's style — a documented
//! representation rather than a tower of coefficient-ring traits:
//! [`PolyZ`] over the integers and [`PolyModP`] over a fixed modulus. They
//! are the substrate the general number field sieve is built on (the
//! algebraic side is a degree-5 or -6 polynomial over ℤ and its behaviour
//! modulo small primes), and more broadly the polynomial layer a computer
//! algebra surface provides.
//!
//! Both store coefficients low-to-high (`coeffs[i]` multiplies `xⁱ`) and
//! stay normalized: no trailing zero coefficient, so the zero polynomial is
//! the empty coefficient list and the degree is the last index.

use crate::bigint::{BigInt, BigUint};
use crate::number_theory;

/// A univariate polynomial over ℤ, coefficients low-to-high, normalized to
/// drop trailing zeros.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolyZ {
    coeffs: Vec<BigInt>,
}

impl PolyZ {
    /// Build from coefficients low-to-high, normalizing away trailing
    /// zeros. `[3, 0, 2]` is `2x² + 3`.
    #[must_use]
    pub fn new(coeffs: Vec<BigInt>) -> Self {
        let mut poly = Self { coeffs };
        poly.normalize();
        poly
    }

    /// The zero polynomial.
    #[must_use]
    pub fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }

    /// The constant polynomial `c`.
    #[must_use]
    pub fn constant(c: BigInt) -> Self {
        Self::new(vec![c])
    }

    /// Build from small integer coefficients low-to-high, for callers with
    /// literal polynomials.
    #[must_use]
    pub fn from_i64_slice(coeffs: &[i64]) -> Self {
        Self::new(coeffs.iter().map(|&c| BigInt::from_i64(c)).collect())
    }

    fn normalize(&mut self) {
        while self.coeffs.last().is_some_and(BigInt::is_zero) {
            self.coeffs.pop();
        }
    }

    /// Whether this is the zero polynomial.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// The degree, or `None` for the zero polynomial (whose degree is
    /// conventionally `−∞`).
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        self.coeffs.len().checked_sub(1)
    }

    /// The coefficients, low-to-high; empty for the zero polynomial.
    #[must_use]
    pub fn coefficients(&self) -> &[BigInt] {
        &self.coeffs
    }

    /// The leading coefficient, or zero for the zero polynomial.
    #[must_use]
    pub fn leading_coefficient(&self) -> BigInt {
        self.coeffs.last().cloned().unwrap_or_else(BigInt::zero)
    }

    /// `self + other`.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(BigInt::zero);
            let b = other.coeffs.get(i).cloned().unwrap_or_else(BigInt::zero);
            coeffs.push(a.add_ref(&b));
        }
        Self::new(coeffs)
    }

    /// `self − other`.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(BigInt::zero);
            let b = other.coeffs.get(i).cloned().unwrap_or_else(BigInt::zero);
            coeffs.push(a.sub_ref(&b));
        }
        Self::new(coeffs)
    }

    /// `−self`.
    #[must_use]
    pub fn negated(&self) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(BigInt::negated).collect(),
        }
    }

    /// `self · other`, schoolbook.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut coeffs = vec![BigInt::zero(); self.coeffs.len() + other.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in other.coeffs.iter().enumerate() {
                let term = a.mul_ref(b);
                coeffs[i + j] = coeffs[i + j].add_ref(&term);
            }
        }
        Self::new(coeffs)
    }

    /// `self · c` for a scalar.
    #[must_use]
    pub fn scale(&self, c: &BigInt) -> Self {
        if c.is_zero() {
            return Self::zero();
        }
        Self::new(self.coeffs.iter().map(|a| a.mul_ref(c)).collect())
    }

    /// Evaluate at `x` by Horner's method.
    #[must_use]
    pub fn evaluate(&self, x: &BigInt) -> BigInt {
        let mut acc = BigInt::zero();
        for coeff in self.coeffs.iter().rev() {
            acc = acc.mul_ref(x).add_ref(coeff);
        }
        acc
    }

    /// The formal derivative.
    #[must_use]
    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero();
        }
        let coeffs = self.coeffs[1..]
            .iter()
            .enumerate()
            .map(|(i, c)| c.mul_ref(&BigInt::from_i64(i as i64 + 1)))
            .collect();
        Self::new(coeffs)
    }

    /// The content: the gcd of the coefficients, non-negative, zero for the
    /// zero polynomial.
    #[must_use]
    pub fn content(&self) -> BigInt {
        let mut g = BigInt::zero();
        for c in &self.coeffs {
            g = g.gcd(c);
        }
        g
    }

    /// The primitive part: `self` divided by its content (the content's
    /// sign left on the content, so the primitive part's leading
    /// coefficient keeps `self`'s sign). The zero polynomial is its own
    /// primitive part.
    #[must_use]
    pub fn primitive_part(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let content = self.content();
        Self::new(self.coeffs.iter().map(|c| c.div_exact(&content)).collect())
    }

    /// Pseudo-division: returns `(quotient, remainder)` satisfying
    /// `ℓ·self = quotient·divisor + remainder` with `deg remainder <
    /// deg divisor` — staying in ℤ by premultiplying by the leading
    /// coefficient rather than dividing by it (Knuth, *TAOCP* vol. 2,
    /// §4.6.1, Algorithm R). When `deg self ≥ deg divisor`, `ℓ =
    /// lc(divisor)` raised to `deg self − deg divisor + 1`; when
    /// `deg self < deg divisor` there is nothing to divide, the result is
    /// `(zero, self)`, and `ℓ = 1` (the formula's exponent would be
    /// non-positive there and does not apply).
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is the zero polynomial.
    #[must_use]
    pub fn pseudo_div_rem(&self, divisor: &Self) -> (Self, Self) {
        assert!(!divisor.is_zero(), "pseudo-division by the zero polynomial");
        let Some(divisor_degree) = divisor.degree() else {
            unreachable!("non-zero divisor has a degree")
        };
        if self.degree().is_none_or(|d| d < divisor_degree) {
            // Nothing to divide: lc^1·self = 0·divisor + lc^1·self, but the
            // exponent is deg self − deg divisor + 1 which is ≤ 0 here, so
            // the identity holds with exponent 0 and remainder self.
            return (Self::zero(), self.clone());
        }
        let self_degree = self.degree().expect("degree checked above");
        let lc = divisor.leading_coefficient();
        let mut remainder = self.clone();
        let mut quotient = vec![BigInt::zero(); self_degree - divisor_degree + 1];
        // Repeatedly cancel the remainder's leading term; each step scales
        // the whole working state by lc so the coefficients stay integral.
        let mut steps = 0usize;
        while let Some(rem_degree) = remainder.degree() {
            if rem_degree < divisor_degree {
                break;
            }
            let shift = rem_degree - divisor_degree;
            let rem_lc = remainder.leading_coefficient();
            // remainder ← lc·remainder − rem_lc·xˢʰⁱᶠᵗ·divisor
            remainder = remainder.scale(&lc);
            let subtrahend = divisor.shift_up(shift).scale(&rem_lc);
            remainder = remainder.sub(&subtrahend);
            // quotient accumulates rem_lc at the shift position, itself
            // scaled by lc for the steps still to come.
            for q in quotient.iter_mut() {
                *q = q.mul_ref(&lc);
            }
            quotient[shift] = quotient[shift].add_ref(&rem_lc);
            steps += 1;
        }
        // The identity carries lc^(steps) on the left; the required
        // exponent is deg self − deg divisor + 1, and steps ≤ that. Scale
        // quotient and remainder up to the full exponent.
        let required = self_degree - divisor_degree + 1;
        let mut quotient = Self::new(quotient);
        let mut remainder = remainder;
        for _ in steps..required {
            quotient = quotient.scale(&lc);
            remainder = remainder.scale(&lc);
        }
        (quotient, remainder)
    }

    /// `self · x^shift` — shift coefficients up.
    fn shift_up(&self, shift: usize) -> Self {
        if self.is_zero() || shift == 0 {
            return self.clone();
        }
        let mut coeffs = vec![BigInt::zero(); shift];
        coeffs.extend(self.coeffs.iter().cloned());
        Self { coeffs }
    }

    /// The resultant `res(self, other)` — the determinant of the two
    /// polynomials' Sylvester matrix, zero exactly when they share a
    /// non-constant factor over ℚ. Computed by Bareiss fraction-free
    /// elimination, which keeps every intermediate entry an integer minor
    /// determinant, so no rational arithmetic is needed (Bareiss, Math.
    /// Comp. 22 (1968); Cohen, §3.3.1).
    ///
    /// Conventions at the degenerate ends: the resultant of two non-zero
    /// constants is `1`; `res(f, c)` for a non-zero constant `c` is
    /// `c^deg f`; and the resultant is `0` if either argument is the zero
    /// polynomial.
    #[must_use]
    pub fn resultant(&self, other: &Self) -> BigInt {
        if self.is_zero() || other.is_zero() {
            return BigInt::zero();
        }
        let m = self.degree().expect("non-zero");
        let n = other.degree().expect("non-zero");
        if m == 0 && n == 0 {
            return BigInt::one();
        }
        if n == 0 {
            // res(f, c) = c^deg f.
            return other.leading_coefficient().pow_u64(m as u64);
        }
        if m == 0 {
            return self.leading_coefficient().pow_u64(n as u64);
        }
        let matrix = sylvester_matrix(self, other);
        bareiss_determinant(matrix)
    }

    /// The discriminant `disc(self) = (−1)^(d(d−1)/2) · res(self, self') /
    /// lc(self)`, where `d = deg self`. Zero exactly when `self` has a
    /// repeated factor over ℚ. The division by the leading coefficient is
    /// exact for an integer polynomial (Cohen, §3.3.2). The discriminant of
    /// a constant or the zero polynomial is `0` by convention.
    #[must_use]
    pub fn discriminant(&self) -> BigInt {
        let Some(d) = self.degree() else {
            return BigInt::zero();
        };
        if d == 0 {
            return BigInt::zero();
        }
        let res = self.resultant(&self.derivative());
        // sign = (−1)^(d(d−1)/2)
        let signed = if (d * (d - 1) / 2) % 2 == 0 {
            res
        } else {
            res.negated()
        };
        signed.div_exact(&self.leading_coefficient())
    }
}

/// The Sylvester matrix of two non-constant polynomials `a` (degree `m`)
/// and `b` (degree `n`): an `(m+n)×(m+n)` matrix whose top `n` rows are
/// shifted copies of `a`'s coefficients (high-to-low) and bottom `m` rows
/// shifted copies of `b`'s. Row-major.
fn sylvester_matrix(a: &PolyZ, b: &PolyZ) -> Vec<Vec<BigInt>> {
    let m = a.degree().expect("non-constant");
    let n = b.degree().expect("non-constant");
    let size = m + n;
    let mut matrix = vec![vec![BigInt::zero(); size]; size];
    // a's coefficients high-to-low, length m+1.
    let a_hi: Vec<BigInt> = a.coefficients().iter().rev().cloned().collect();
    let b_hi: Vec<BigInt> = b.coefficients().iter().rev().cloned().collect();
    for i in 0..n {
        for (j, coeff) in a_hi.iter().enumerate() {
            matrix[i][i + j] = coeff.clone();
        }
    }
    for i in 0..m {
        for (j, coeff) in b_hi.iter().enumerate() {
            matrix[n + i][i + j] = coeff.clone();
        }
    }
    matrix
}

/// The determinant of an integer matrix by Bareiss fraction-free Gaussian
/// elimination: each elimination step divides exactly by the previous
/// pivot, so all intermediates stay integral, and the last pivot is the
/// determinant (up to the sign accumulated by row swaps).
fn bareiss_determinant(mut matrix: Vec<Vec<BigInt>>) -> BigInt {
    let n = matrix.len();
    if n == 0 {
        return BigInt::one();
    }
    let mut sign_negative = false;
    let mut previous = BigInt::one();
    for k in 0..n - 1 {
        if matrix[k][k].is_zero() {
            // Find a row below with a non-zero entry in this column.
            let Some(swap) = (k + 1..n).find(|&r| !matrix[r][k].is_zero()) else {
                return BigInt::zero(); // singular
            };
            matrix.swap(k, swap);
            sign_negative = !sign_negative;
        }
        let pivot = matrix[k][k].clone();
        for i in k + 1..n {
            for j in k + 1..n {
                // matrix[i][j] ← (matrix[i][j]·pivot − matrix[i][k]·matrix[k][j]) / previous
                let cross = matrix[i][j]
                    .mul_ref(&pivot)
                    .sub_ref(&matrix[i][k].mul_ref(&matrix[k][j]));
                matrix[i][j] = cross.div_exact(&previous);
            }
            matrix[i][k] = BigInt::zero();
        }
        previous = pivot;
    }
    let det = matrix[n - 1][n - 1].clone();
    if sign_negative {
        det.negated()
    } else {
        det
    }
}

/// A univariate polynomial over ℤ/mℤ for a fixed modulus `m ≥ 2`,
/// coefficients low-to-high and reduced, normalized to drop trailing
/// zeros. The modulus travels with the polynomial; operations between two
/// require the same modulus.
///
/// Addition, subtraction, multiplication, scaling, and evaluation work for
/// any `m ≥ 2`. The division-based operations — [`Self::div_rem`],
/// [`Self::rem`], [`Self::gcd`], [`Self::make_monic`], and
/// [`Self::pow_mod`] — invert a leading coefficient modulo `m`, so they
/// require `m` **prime** (precisely, the relevant leading coefficient
/// invertible modulo `m`) and panic on a non-invertible pivot. This is the
/// field ℤ/pℤ the factorization routines are built over.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolyModP {
    coeffs: Vec<BigUint>,
    modulus: BigUint,
}

impl PolyModP {
    /// Build from coefficients low-to-high, reducing each modulo `modulus`
    /// and normalizing away trailing zeros.
    ///
    /// # Panics
    ///
    /// Panics if `modulus < 2`.
    #[must_use]
    pub fn new(coeffs: Vec<BigUint>, modulus: &BigUint) -> Self {
        assert!(
            *modulus >= BigUint::from_u64(2),
            "polynomial modulus must be at least 2"
        );
        let coeffs = coeffs.into_iter().map(|c| c.modulo(modulus)).collect();
        let mut poly = Self {
            coeffs,
            modulus: modulus.clone(),
        };
        poly.normalize();
        poly
    }

    /// The zero polynomial over `modulus`.
    #[must_use]
    pub fn zero(modulus: &BigUint) -> Self {
        Self::new(Vec::new(), modulus)
    }

    /// Reduce an integer polynomial modulo `modulus`.
    #[must_use]
    pub fn from_poly_z(poly: &PolyZ, modulus: &BigUint) -> Self {
        let coeffs = poly
            .coefficients()
            .iter()
            .map(|c| c.modulo_positive(modulus))
            .collect();
        Self::new(coeffs, modulus)
    }

    fn normalize(&mut self) {
        while self.coeffs.last().is_some_and(BigUint::is_zero) {
            self.coeffs.pop();
        }
    }

    /// The modulus.
    #[must_use]
    pub fn modulus(&self) -> &BigUint {
        &self.modulus
    }

    /// Whether this is the zero polynomial.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// The degree, or `None` for the zero polynomial.
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        self.coeffs.len().checked_sub(1)
    }

    /// The coefficients, low-to-high.
    #[must_use]
    pub fn coefficients(&self) -> &[BigUint] {
        &self.coeffs
    }

    /// The leading coefficient, or zero for the zero polynomial.
    #[must_use]
    pub fn leading_coefficient(&self) -> BigUint {
        self.coeffs.last().cloned().unwrap_or_else(BigUint::zero)
    }

    fn check_modulus(&self, other: &Self) {
        debug_assert_eq!(self.modulus, other.modulus, "operands share a modulus");
    }

    /// `self + other` (mod m).
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        self.check_modulus(other);
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(BigUint::zero);
            let b = other.coeffs.get(i).cloned().unwrap_or_else(BigUint::zero);
            coeffs.push(BigUint::mod_add(&a, &b, &self.modulus));
        }
        Self::new(coeffs, &self.modulus)
    }

    /// `self − other` (mod m).
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        self.check_modulus(other);
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(BigUint::zero);
            let b = other.coeffs.get(i).cloned().unwrap_or_else(BigUint::zero);
            coeffs.push(BigUint::mod_sub(&a, &b, &self.modulus));
        }
        Self::new(coeffs, &self.modulus)
    }

    /// `self · other` (mod m), schoolbook.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        self.check_modulus(other);
        if self.is_zero() || other.is_zero() {
            return Self::zero(&self.modulus);
        }
        let mut coeffs = vec![BigUint::zero(); self.coeffs.len() + other.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in other.coeffs.iter().enumerate() {
                let term = BigUint::mod_mul(a, b, &self.modulus);
                coeffs[i + j] = BigUint::mod_add(&coeffs[i + j], &term, &self.modulus);
            }
        }
        Self::new(coeffs, &self.modulus)
    }

    /// `self · c` for a scalar (mod m).
    #[must_use]
    pub fn scale(&self, c: &BigUint) -> Self {
        let coeffs = self
            .coeffs
            .iter()
            .map(|a| BigUint::mod_mul(a, c, &self.modulus))
            .collect();
        Self::new(coeffs, &self.modulus)
    }

    /// Evaluate at `x` (mod m), Horner.
    #[must_use]
    pub fn evaluate(&self, x: &BigUint) -> BigUint {
        let x = x.modulo(&self.modulus);
        let mut acc = BigUint::zero();
        for coeff in self.coeffs.iter().rev() {
            acc = BigUint::mod_add(
                &BigUint::mod_mul(&acc, &x, &self.modulus),
                coeff,
                &self.modulus,
            );
        }
        acc
    }

    /// Divide by a scalar's inverse to make the polynomial monic: `self`
    /// scaled by the inverse of its leading coefficient. The zero
    /// polynomial is returned unchanged.
    ///
    /// # Panics
    ///
    /// Panics if the leading coefficient is not invertible modulo `m`
    /// (always invertible when `m` is prime and the polynomial is
    /// non-zero).
    #[must_use]
    pub fn make_monic(&self) -> Self {
        if self.is_zero() {
            return self.clone();
        }
        let lc = self.leading_coefficient();
        if lc.is_one() {
            return self.clone();
        }
        let inv = number_theory::mod_inverse(&lc, &self.modulus)
            .expect("leading coefficient is invertible modulo a prime");
        self.scale(&inv)
    }

    /// Division with remainder by a divisor whose leading coefficient is
    /// invertible modulo `m`: `self = quotient·divisor + remainder` with
    /// `deg remainder < deg divisor`.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is zero or its leading coefficient is not
    /// invertible modulo `m`.
    #[must_use]
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        self.check_modulus(divisor);
        assert!(!divisor.is_zero(), "division by the zero polynomial");
        let divisor_degree = divisor.degree().expect("non-zero divisor");
        let lc_inv = number_theory::mod_inverse(&divisor.leading_coefficient(), &self.modulus)
            .expect("divisor's leading coefficient is invertible");
        let mut remainder = self.clone();
        let mut quotient = vec![BigUint::zero(); self.degree().map_or(0, |d| d + 1)];
        while let Some(rem_degree) = remainder.degree() {
            if rem_degree < divisor_degree {
                break;
            }
            let shift = rem_degree - divisor_degree;
            let factor = BigUint::mod_mul(&remainder.leading_coefficient(), &lc_inv, &self.modulus);
            quotient[shift] = factor.clone();
            // remainder ← remainder − factor·xˢʰⁱᶠᵗ·divisor
            let subtrahend = divisor.shift_up(shift).scale(&factor);
            remainder = remainder.sub(&subtrahend);
        }
        (Self::new(quotient, &self.modulus), remainder)
    }

    /// `self mod divisor` — the remainder of [`Self::div_rem`].
    #[must_use]
    pub fn rem(&self, divisor: &Self) -> Self {
        self.div_rem(divisor).1
    }

    /// Monic greatest common divisor by the Euclidean algorithm; the zero
    /// polynomial's gcd with `q` is the monic form of `q`.
    #[must_use]
    pub fn gcd(&self, other: &Self) -> Self {
        self.check_modulus(other);
        let mut a = self.clone();
        let mut b = other.clone();
        while !b.is_zero() {
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        a.make_monic()
    }

    /// `self^exponent mod modulus_poly` by square-and-multiply — the
    /// primitive behind distinct-degree factorization's `x^(p^d)`.
    ///
    /// # Panics
    ///
    /// Panics if `modulus_poly` is the zero polynomial, or if a leading
    /// coefficient reached during reduction is not invertible modulo `m`
    /// (see the type-level note; `m` prime avoids the latter).
    #[must_use]
    pub fn pow_mod(&self, exponent: &BigUint, modulus_poly: &Self) -> Self {
        self.check_modulus(modulus_poly);
        let one = Self::new(vec![BigUint::one()], &self.modulus);
        if exponent.is_zero() {
            return one.rem(modulus_poly);
        }
        let mut result = one;
        let base = self.rem(modulus_poly);
        for bit in (0..exponent.bits()).rev() {
            result = result.mul(&result).rem(modulus_poly);
            if exponent.bit(bit) {
                result = result.mul(&base).rem(modulus_poly);
            }
        }
        result
    }

    /// `self · x^shift`.
    fn shift_up(&self, shift: usize) -> Self {
        if self.is_zero() || shift == 0 {
            return self.clone();
        }
        let mut coeffs = vec![BigUint::zero(); shift];
        coeffs.extend(self.coeffs.iter().cloned());
        Self::new(coeffs, &self.modulus)
    }

    /// The monic polynomial `x` over this modulus.
    fn monomial_x(modulus: &BigUint) -> Self {
        Self::new(vec![BigUint::zero(), BigUint::one()], modulus)
    }

    /// Squarefree factorization over 𝔽ₚ: the distinct squarefree factors
    /// with their multiplicities, `(factor, e)` with `∏ factorᵉ = ` the
    /// monic form of `self` (Cohen, *A Course in Computational Algebraic
    /// Number Theory*, §3.4.2, Squarefree Factorization). The factors are
    /// monic, of degree ≥ 1, and pairwise coprime; a factor with
    /// multiplicity divisible by `p` is recovered through the `p`-th root
    /// the characteristic forces.
    ///
    /// # Panics
    ///
    /// Panics if `self` is constant or zero (no factorization), or on a
    /// non-invertible pivot during division. A prime modulus is required
    /// (see the type documentation): over a composite modulus a division
    /// may instead find every pivot a unit, and then the result is
    /// unspecified rather than a panic.
    #[must_use]
    pub fn squarefree_factorization(&self) -> Vec<(Self, usize)> {
        assert!(
            self.degree().is_some_and(|d| d >= 1),
            "squarefree factorization needs a non-constant polynomial"
        );
        let mut factors: Vec<(Self, usize)> = Vec::new();
        self.squarefree_into(1, &mut factors);
        factors.sort_by_key(|(_, e)| *e);
        factors
    }

    /// The recursive worker of [`Self::squarefree_factorization`]: extract
    /// the squarefree factors of `self` whose multiplicity is not a
    /// multiple of `p`, then recurse on the `p`-th root of what remains,
    /// scaling the multiplicity by `p`.
    fn squarefree_into(&self, mult_shift: usize, out: &mut Vec<(Self, usize)>) {
        let f = self.make_monic();
        let derivative = f.derivative_modp();
        // gcd(f, f') collects every factor to one less than its multiplicity;
        // f / that is the product of the distinct factors (radical) whose
        // multiplicity is coprime to p.
        let mut t = f.gcd(&derivative);
        let mut v = f.div_rem(&t).0;
        let mut e = 1usize;
        while v.degree().is_some_and(|d| d >= 1) {
            let w = t.gcd(&v);
            let factor = v.div_rem(&w).0;
            if factor.degree().is_some_and(|d| d >= 1) {
                out.push((factor.make_monic(), e * mult_shift));
            }
            v = w;
            t = t.div_rem(&v).0;
            e += 1;
        }
        // What remains in t is a p-th power: g(x)^p = g(xᵖ). Take its p-th
        // root and recurse with the multiplicity scaled by p. Reaching here
        // with a positive-degree t means deg t ≥ p, so p fits a machine
        // word (deg is bounded by memory).
        if t.degree().is_some_and(|d| d >= 1) {
            let p = usize::try_from(
                self.modulus
                    .to_u64()
                    .expect("a p-th power of positive degree bounds p by its degree"),
            )
            .expect("characteristic fits usize");
            let root = t.pth_root(p);
            root.squarefree_into(mult_shift * p, out);
        }
    }

    /// The `p`-th root of a polynomial that is known to be a `p`-th power:
    /// `g(xᵖ)` has `g`'s coefficients at positions that are multiples of
    /// `p`, and in 𝔽ₚ each coefficient is its own `p`-th root (Frobenius).
    fn pth_root(&self, p: usize) -> Self {
        let coeffs = (0..self.coeffs.len())
            .step_by(p)
            .map(|i| self.coeffs[i].clone())
            .collect();
        Self::new(coeffs, &self.modulus)
    }

    /// The formal derivative over 𝔽ₚ.
    fn derivative_modp(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero(&self.modulus);
        }
        let coeffs = self.coeffs[1..]
            .iter()
            .enumerate()
            .map(|(i, c)| BigUint::mod_mul(c, &BigUint::from_u64(i as u64 + 1), &self.modulus))
            .collect();
        Self::new(coeffs, &self.modulus)
    }

    /// Distinct-degree factorization of a squarefree monic polynomial:
    /// the pairs `(d, gₐ)` where `gₐ` is the product of all the degree-`d`
    /// irreducible factors of `self` (Cohen, §3.4.3, Distinct Degree
    /// Factorization). The `gₐ` are monic; a `d` with no factor of that
    /// degree is omitted.
    fn distinct_degree(&self) -> Vec<(usize, Self)> {
        let mut factors = Vec::new();
        let mut remaining = self.clone();
        // xqi holds x^(pᵈ) mod remaining, advanced one Frobenius power per d.
        let mut xqi = Self::monomial_x(&self.modulus);
        let x = Self::monomial_x(&self.modulus);
        let mut d = 0usize;
        while remaining.degree().is_some_and(|r| r >= 2 * (d + 1)) {
            d += 1;
            // x^(pᵈ) = (x^(pᵈ⁻¹))ᵖ mod remaining.
            xqi = xqi.pow_mod(&self.modulus, &remaining);
            // The degree-d factors divide x^(pᵈ) − x.
            let diff = xqi.sub(&x).rem(&remaining);
            let g = remaining.gcd(&diff);
            if g.degree().is_some_and(|dg| dg >= 1) {
                factors.push((d, g.clone()));
                remaining = remaining.div_rem(&g).0;
                xqi = xqi.rem(&remaining);
            }
        }
        if remaining.degree().is_some_and(|r| r >= 1) {
            let dr = remaining.degree().expect("non-constant");
            factors.push((dr, remaining));
        }
        factors
    }

    /// Split a monic product of degree-`d` irreducibles into its
    /// irreducible factors — the equal-degree step (Cantor & Zassenhaus,
    /// *A new algorithm for factoring polynomials over finite fields*,
    /// Math. Comp. 36 (1981), 587–592; Cohen, §3.4.4, Final Splitting).
    /// Draws random splitting polynomials from `rng` until each factor is
    /// isolated; over 𝔽₂ the split uses the trace map, the standard
    /// characteristic-2 instance.
    fn equal_degree_split<R: crate::random::Rng + ?Sized>(
        &self,
        d: usize,
        rng: &mut R,
    ) -> Vec<Self> {
        let total_degree = self.degree().expect("non-zero");
        if total_degree == d {
            return vec![self.make_monic()];
        }
        let target = total_degree / d;
        let mut factors = vec![self.make_monic()];
        let two = BigUint::from_u64(2);
        while factors.len() < target {
            // A random splitter of degree < deg self.
            let a = self.random_below_degree(rng);
            if a.is_zero() {
                continue;
            }
            // g captures roughly half the factors: over 𝔽₂ by the trace map,
            // otherwise by a^((pᵈ−1)/2) − 1.
            let g = if self.modulus == two {
                let mut trace = a.rem(self);
                let mut term = trace.clone();
                for _ in 1..d {
                    term = term.mul(&term).rem(self);
                    trace = trace.add(&term);
                }
                self.gcd(&trace)
            } else {
                let exponent = {
                    let pd = self.modulus.pow_u64(u64::try_from(d).expect("d fits u64"));
                    pd.sub_ref(&BigUint::one()).div_rem(&two).0
                };
                let ae = a.pow_mod(&exponent, self);
                let ae_minus_one = ae.sub(&Self::new(vec![BigUint::one()], &self.modulus));
                self.gcd(&ae_minus_one)
            };
            // Refine every current factor by g.
            let mut refined = Vec::with_capacity(factors.len());
            for factor in factors {
                let piece = factor.gcd(&g);
                if piece.degree().is_some_and(|dp| dp >= 1) && piece.degree() < factor.degree() {
                    let other = factor.div_rem(&piece).0;
                    refined.push(piece.make_monic());
                    refined.push(other.make_monic());
                } else {
                    refined.push(factor);
                }
            }
            factors = refined;
        }
        factors
    }

    /// A random polynomial of degree below `self`'s, over the modulus.
    fn random_below_degree<R: crate::random::Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let deg = self.degree().expect("non-zero");
        let coeffs = (0..deg)
            .map(|_| crate::random::random_below(rng, &self.modulus).unwrap_or_else(BigUint::zero))
            .collect();
        Self::new(coeffs, &self.modulus)
    }

    /// Whether `self` is irreducible over 𝔽ₚ. A non-constant polynomial is
    /// irreducible exactly when its distinct-degree factorization is a
    /// single block whose degree equals its own — deterministic, no
    /// randomness.
    ///
    /// # Panics
    ///
    /// Panics on a non-invertible pivot during division. A prime modulus is
    /// required (see the type documentation): over a composite modulus the
    /// answer is unspecified and may be returned without a panic — `x² + 1`,
    /// for one, is reported irreducible modulo 15.
    #[must_use]
    pub fn is_irreducible(&self) -> bool {
        let Some(degree) = self.degree() else {
            return false; // zero polynomial
        };
        if degree == 0 {
            return false; // units and constants are not irreducible
        }
        let monic = self.make_monic();
        // Squarefree is necessary; a repeated factor makes gcd(f, f') > 1.
        if monic.gcd(&monic.derivative_modp()).degree() != Some(0) {
            return false;
        }
        let dd = monic.distinct_degree();
        dd.len() == 1 && dd[0].0 == degree
    }

    /// Complete factorization over 𝔽ₚ into monic irreducibles with
    /// multiplicities: `(factor, e)` with `∏ factorᵉ` equal to the monic
    /// form of `self`. Squarefree decomposition, then distinct-degree, then
    /// the randomized equal-degree split drawn from `rng`. That split is Las
    /// Vegas: it draws until each factor separates, so `rng` must yield
    /// entropy — a source returning only zero bytes does not terminate.
    ///
    /// # Panics
    ///
    /// Panics if `self` is constant or zero, or on a non-invertible pivot
    /// during division. A prime modulus is required (see the type
    /// documentation); over a composite modulus the result is unspecified
    /// and may be returned without a panic.
    #[must_use]
    pub fn factor<R: crate::random::Rng + ?Sized>(&self, rng: &mut R) -> Vec<(Self, usize)> {
        let mut result = Vec::new();
        for (squarefree, mult) in self.squarefree_factorization() {
            for (degree, block) in squarefree.distinct_degree() {
                for irreducible in block.equal_degree_split(degree, rng) {
                    result.push((irreducible, mult));
                }
            }
        }
        result
    }

    /// The roots of `self` in 𝔽ₚ, ascending, each a residue `r` with
    /// `self(r) ≡ 0`. Computed as the linear factors: intersect with
    /// `xᵖ − x` (whose roots are all of 𝔽ₚ), then split into linears by the
    /// same Las Vegas split as [`Self::factor`], so `rng` must yield entropy.
    ///
    /// # Panics
    ///
    /// Panics on a non-invertible pivot during division. A prime modulus is
    /// required (see the type documentation); over a composite modulus the
    /// result is unspecified and may be returned without a panic.
    #[must_use]
    pub fn roots<R: crate::random::Rng + ?Sized>(&self, rng: &mut R) -> Vec<BigUint> {
        if self.degree().is_none_or(|d| d == 0) {
            return Vec::new();
        }
        let monic = self.make_monic();
        // x^p − x mod f, then gcd with f: the product of (x − r) over roots r.
        let x = Self::monomial_x(&self.modulus);
        let xp = x.pow_mod(&self.modulus, &monic);
        let linear_product = monic.gcd(&xp.sub(&x));
        if linear_product.degree().is_none_or(|d| d == 0) {
            return Vec::new();
        }
        let mut roots = Vec::new();
        for factor in linear_product.equal_degree_split(1, rng) {
            // factor = x − r (monic linear); the root is −(constant term).
            let constant = factor
                .coefficients()
                .first()
                .cloned()
                .unwrap_or_else(BigUint::zero);
            let root = BigUint::mod_sub(&BigUint::zero(), &constant, &self.modulus);
            roots.push(root);
        }
        roots.sort();
        roots
    }
}

#[cfg(test)]
mod tests {
    use super::{PolyModP, PolyZ};
    use crate::bigint::{BigInt, BigUint};

    struct SplitMix64 {
        state: u64,
    }
    impl crate::random::Rng for SplitMix64 {
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            let mut i = 0;
            while i < dest.len() {
                let word = self.next_u64().to_le_bytes();
                let take = (dest.len() - i).min(8);
                dest[i..i + take].copy_from_slice(&word[..take]);
                i += take;
            }
        }
    }

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        // A coefficient in [-range, range].
        fn coeff(&mut self, range: i64) -> BigInt {
            let m = 2 * range as u64 + 1;
            let v = (self.next_u64() % m) as i64 - range;
            BigInt::from_i64(v)
        }
        fn poly_z(&mut self, max_deg: usize, range: i64) -> PolyZ {
            let len = (self.next_u64() as usize % (max_deg + 1)) + 1;
            PolyZ::new((0..len).map(|_| self.coeff(range)).collect())
        }
    }

    // Naive schoolbook multiply as an independent oracle.
    fn naive_mul(a: &PolyZ, b: &PolyZ) -> PolyZ {
        if a.is_zero() || b.is_zero() {
            return PolyZ::zero();
        }
        let mut coeffs = vec![BigInt::zero(); a.coefficients().len() + b.coefficients().len() - 1];
        for (i, x) in a.coefficients().iter().enumerate() {
            for (j, y) in b.coefficients().iter().enumerate() {
                coeffs[i + j] = coeffs[i + j].add_ref(&x.mul_ref(y));
            }
        }
        PolyZ::new(coeffs)
    }

    #[test]
    fn poly_z_ring_axioms_and_evaluation() {
        let mut rng = SplitMix64 {
            state: 0x9017_0000_0001,
        };
        for _ in 0..500 {
            let a = rng.poly_z(6, 20);
            let b = rng.poly_z(6, 20);
            let c = rng.poly_z(6, 20);
            // Commutativity and the naive-multiply oracle.
            assert_eq!(a.mul(&b), b.mul(&a));
            assert_eq!(a.mul(&b), naive_mul(&a, &b));
            // Distributivity.
            assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
            // add/sub inverse.
            assert_eq!(a.add(&b).sub(&b), a);
            // The evaluation homomorphism: eval(a·b) = eval(a)·eval(b).
            for x in [-3i64, 0, 1, 7] {
                let xv = BigInt::from_i64(x);
                assert_eq!(
                    a.mul(&b).evaluate(&xv),
                    a.evaluate(&xv).mul_ref(&b.evaluate(&xv))
                );
                assert_eq!(
                    a.add(&b).evaluate(&xv),
                    a.evaluate(&xv).add_ref(&b.evaluate(&xv))
                );
            }
        }
    }

    #[test]
    fn poly_z_derivative_and_content() {
        // d/dx (3x^2 + 2x + 5) = 6x + 2.
        let p = PolyZ::from_i64_slice(&[5, 2, 3]);
        assert_eq!(p.derivative(), PolyZ::from_i64_slice(&[2, 6]));
        // Product rule on random polynomials.
        let mut rng = SplitMix64 {
            state: 0xde01_0000_0007,
        };
        for _ in 0..200 {
            let a = rng.poly_z(5, 15);
            let b = rng.poly_z(5, 15);
            // (a·b)' = a'·b + a·b'.
            assert_eq!(
                a.mul(&b).derivative(),
                a.derivative().mul(&b).add(&a.mul(&b.derivative()))
            );
        }
        // content(6x^2 + 9x + 15) = 3, primitive part = 2x^2 + 3x + 5.
        let q = PolyZ::from_i64_slice(&[15, 9, 6]);
        assert_eq!(q.content(), BigInt::from_i64(3));
        assert_eq!(q.primitive_part(), PolyZ::from_i64_slice(&[5, 3, 2]));
        // content · primitive_part = original.
        assert_eq!(q.primitive_part().scale(&q.content()), q);
    }

    #[test]
    fn poly_z_pseudo_division_identity() {
        let mut rng = SplitMix64 {
            state: 0x9500_0000_0001,
        };
        for _ in 0..1000 {
            let a = rng.poly_z(8, 12);
            let mut b = rng.poly_z(5, 12);
            if b.is_zero() {
                b = PolyZ::from_i64_slice(&[1]);
            }
            let (q, r) = a.pseudo_div_rem(&b);
            // deg r < deg b.
            if let (Some(dr), Some(db)) = (r.degree(), b.degree()) {
                assert!(dr < db, "remainder degree");
            }
            // lc(b)^(deg a − deg b + 1) · a = q·b + r, when deg a ≥ deg b.
            if let (Some(da), Some(db)) = (a.degree(), b.degree()) {
                if da >= db {
                    let exp = (da - db + 1) as u64;
                    let lc = b.leading_coefficient();
                    let mut lc_pow = BigInt::one();
                    for _ in 0..exp {
                        lc_pow = lc_pow.mul_ref(&lc);
                    }
                    assert_eq!(
                        a.scale(&lc_pow),
                        q.mul(&b).add(&r),
                        "pseudo-division identity"
                    );
                }
            }
        }
    }

    #[test]
    fn poly_mod_p_division_and_gcd() {
        let p = BigUint::from_u64(101);
        let mut rng = SplitMix64 {
            state: 0x4001_0000_0001,
        };
        let to_mod = |poly: &PolyZ| PolyModP::from_poly_z(poly, &p);
        for _ in 0..500 {
            let a = to_mod(&rng.poly_z(8, 200));
            let mut b = to_mod(&rng.poly_z(5, 200));
            if b.is_zero() {
                b = PolyModP::new(vec![BigUint::one()], &p);
            }
            let (q, r) = a.div_rem(&b);
            // a = q·b + r, deg r < deg b.
            assert_eq!(q.mul(&b).add(&r), a, "division identity mod p");
            if let (Some(dr), Some(db)) = (r.degree(), b.degree()) {
                assert!(dr < db);
            }
            // gcd divides both and is monic.
            let g = a.gcd(&b);
            if !g.is_zero() {
                assert!(g.leading_coefficient().is_one(), "gcd is monic");
                assert!(a.rem(&g).is_zero(), "gcd divides a");
                assert!(b.rem(&g).is_zero(), "gcd divides b");
            }
        }
    }

    #[test]
    fn poly_mod_p_gcd_of_known_factors() {
        let p = BigUint::from_u64(7);
        // (x-1)(x-2) and (x-2)(x-3) share (x-2).
        let x_minus = |a: i64| PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[-a, 1]), &p);
        let f = x_minus(1).mul(&x_minus(2));
        let g = x_minus(2).mul(&x_minus(3));
        assert_eq!(f.gcd(&g), x_minus(2), "shared linear factor");
    }

    #[test]
    #[should_panic(expected = "invertible")]
    fn poly_mod_p_div_rem_panics_on_noninvertible_pivot() {
        // Composite modulus with a divisor whose leading coefficient shares
        // a factor: 2x + 1 mod 6, leading coeff 2 not invertible.
        let m = BigUint::from_u64(6);
        let a = PolyModP::new(vec![BigUint::one(), BigUint::one(), BigUint::one()], &m);
        let b = PolyModP::new(vec![BigUint::one(), BigUint::from_u64(2)], &m);
        let _ = a.div_rem(&b);
    }

    // An independent resultant oracle: the determinant of the Sylvester
    // matrix by rational (fraction) Gaussian elimination — no shared code
    // with the production Bareiss path. The elimination reads and writes
    // distinct rows of one matrix, so it indexes by row rather than
    // iterating.
    #[allow(clippy::needless_range_loop)]
    fn resultant_rational_oracle(a: &PolyZ, b: &PolyZ) -> BigInt {
        use crate::bigint::Sign;
        if a.is_zero() || b.is_zero() {
            return BigInt::zero();
        }
        let (m, n) = (a.degree().unwrap(), b.degree().unwrap());
        if m == 0 && n == 0 {
            return BigInt::one();
        }
        let size = m + n;
        // Rationals as (num, den) BigInt pairs, den > 0.
        type Q = (BigInt, BigInt);
        let q = |v: BigInt| -> Q { (v, BigInt::one()) };
        let qmul = |x: &Q, y: &Q| -> Q { (x.0.mul_ref(&y.0), x.1.mul_ref(&y.1)) };
        let qsub = |x: &Q, y: &Q| -> Q {
            (
                x.0.mul_ref(&y.1).sub_ref(&y.0.mul_ref(&x.1)),
                x.1.mul_ref(&y.1),
            )
        };
        let qdiv = |x: &Q, y: &Q| -> Q {
            let mut num = x.0.mul_ref(&y.1);
            let mut den = x.1.mul_ref(&y.0);
            if den.sign() == Sign::Negative {
                num = num.negated();
                den = den.negated();
            }
            (num, den)
        };
        let qzero = |x: &Q| -> bool { x.0.is_zero() };
        let mut mat = vec![vec![q(BigInt::zero()); size]; size];
        let a_hi: Vec<BigInt> = a.coefficients().iter().rev().cloned().collect();
        let b_hi: Vec<BigInt> = b.coefficients().iter().rev().cloned().collect();
        for i in 0..n {
            for (j, c) in a_hi.iter().enumerate() {
                mat[i][i + j] = q(c.clone());
            }
        }
        for i in 0..m {
            for (j, c) in b_hi.iter().enumerate() {
                mat[n + i][i + j] = q(c.clone());
            }
        }
        let mut det = q(BigInt::one());
        for col in 0..size {
            let Some(piv) = (col..size).find(|&r| !qzero(&mat[r][col])) else {
                return BigInt::zero();
            };
            if piv != col {
                mat.swap(col, piv);
                det = (det.0.negated(), det.1);
            }
            det = qmul(&det, &mat[col][col]);
            let inv = mat[col][col].clone();
            let pivot_row = mat[col].clone();
            for r in col + 1..size {
                let factor = qdiv(&mat[r][col], &inv);
                for (cell, pivot_cell) in mat[r].iter_mut().zip(&pivot_row).skip(col) {
                    let prod = qmul(&factor, pivot_cell);
                    *cell = qsub(cell, &prod);
                }
            }
        }
        // det is an integer; num/den divides evenly.
        det.0.div_exact(&det.1)
    }

    #[test]
    fn resultant_matches_rational_oracle_and_known_values() {
        let mut rng = SplitMix64 {
            state: 0x8e50_0000_0001,
        };
        for _ in 0..500 {
            let a = rng.poly_z(5, 8);
            let b = rng.poly_z(5, 8);
            assert_eq!(
                a.resultant(&b),
                resultant_rational_oracle(&a, &b),
                "resultant vs rational oracle"
            );
            // Symmetry up to sign: res(a,b) = (-1)^(deg a · deg b) res(b,a).
            if let (Some(da), Some(db)) = (a.degree(), b.degree()) {
                let expected = if (da * db) % 2 == 0 {
                    b.resultant(&a)
                } else {
                    b.resultant(&a).negated()
                };
                assert_eq!(a.resultant(&b), expected, "resultant symmetry");
            }
        }
        // Known values.
        let xm = |r: i64| PolyZ::from_i64_slice(&[-r, 1]); // x - r
                                                           // res(x-2, x-3) = -1 (Sylvester convention).
        assert_eq!(xm(2).resultant(&xm(3)), BigInt::from_i64(-1));
        // Shares root 1 → 0.
        assert_eq!(
            PolyZ::from_i64_slice(&[-1, 0, 1]).resultant(&xm(1)),
            BigInt::zero()
        );
        // res(2x, 3) = 3^1.
        assert_eq!(
            PolyZ::from_i64_slice(&[0, 2]).resultant(&PolyZ::from_i64_slice(&[3])),
            BigInt::from_i64(3)
        );
        // Two nonzero constants → 1.
        assert_eq!(
            PolyZ::from_i64_slice(&[5]).resultant(&PolyZ::from_i64_slice(&[7])),
            BigInt::one()
        );
        // Zero polynomial → 0.
        assert_eq!(xm(1).resultant(&PolyZ::zero()), BigInt::zero());
    }

    #[test]
    fn resultant_multiplicativity() {
        let mut rng = SplitMix64 {
            state: 0x3711_0000_0001,
        };
        for _ in 0..300 {
            let f = rng.poly_z(4, 6);
            let g = rng.poly_z(3, 6);
            let h = rng.poly_z(3, 6);
            if f.is_zero() || g.is_zero() || h.is_zero() {
                continue;
            }
            // res(f, g·h) = res(f, g)·res(f, h).
            assert_eq!(
                f.resultant(&g.mul(&h)),
                f.resultant(&g).mul_ref(&f.resultant(&h)),
                "resultant multiplicativity"
            );
        }
    }

    #[test]
    fn discriminant_known_forms() {
        // disc(x^2 + bx + c) = b^2 - 4c.
        for b in -4i64..=4 {
            for c in -4i64..=4 {
                assert_eq!(
                    PolyZ::from_i64_slice(&[c, b, 1]).discriminant(),
                    BigInt::from_i64(b * b - 4 * c),
                    "disc quadratic b={b} c={c}"
                );
            }
        }
        // disc(x^3 + px + q) = -4p^3 - 27q^2.
        for p in -3i64..=3 {
            for q in -3i64..=3 {
                assert_eq!(
                    PolyZ::from_i64_slice(&[q, p, 0, 1]).discriminant(),
                    BigInt::from_i64(-4 * p * p * p - 27 * q * q),
                    "disc depressed cubic p={p} q={q}"
                );
            }
        }
        // A polynomial with a repeated root has discriminant 0:
        // (x-1)^2(x-2) = x^3 - 4x^2 + 5x - 2.
        let repeated = PolyZ::from_i64_slice(&[-2, 5, -4, 1]);
        assert_eq!(repeated.discriminant(), BigInt::zero());
        // Constant and zero → 0.
        assert_eq!(PolyZ::from_i64_slice(&[7]).discriminant(), BigInt::zero());
        assert_eq!(PolyZ::zero().discriminant(), BigInt::zero());
    }

    // Multiply a slice of factors-with-multiplicity back into one poly.
    fn reassemble(factors: &[(PolyModP, usize)], p: &BigUint) -> PolyModP {
        let mut acc = PolyModP::new(vec![BigUint::one()], p);
        for (f, e) in factors {
            for _ in 0..*e {
                acc = acc.mul(f);
            }
        }
        acc
    }

    #[test]
    fn factorization_reconstructs_and_is_irreducible() {
        let mut rng = SplitMix64 {
            state: 0xfac0_0000_0001,
        };
        for &p in &[2u64, 3, 5, 7, 11, 13] {
            let pm = BigUint::from_u64(p);
            for _ in 0..200 {
                // Build a random product of small factors with multiplicities.
                let mut f = PolyModP::new(vec![BigUint::one()], &pm);
                let parts = 1 + (rng.next_u64() % 3) as usize;
                for _ in 0..parts {
                    let deg = 1 + (rng.next_u64() % 3) as usize;
                    let coeffs: Vec<BigUint> = (0..=deg)
                        .map(|_| BigUint::from_u64(rng.next_u64() % p))
                        .collect();
                    let mut base = PolyModP::new(coeffs, &pm);
                    // Ensure non-constant.
                    if base.degree().unwrap_or(0) < 1 {
                        base = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[0, 1]), &pm);
                    }
                    let mult = 1 + (rng.next_u64() % 3) as usize;
                    for _ in 0..mult {
                        f = f.mul(&base);
                    }
                }
                if f.degree().unwrap_or(0) < 1 {
                    continue;
                }
                let factors = f.factor(&mut rng);
                // Product of factors^mult equals the monic form of f.
                assert_eq!(
                    reassemble(&factors, &pm),
                    f.make_monic(),
                    "reconstruct p={p}"
                );
                // Every returned factor is monic and irreducible.
                for (fac, e) in &factors {
                    assert!(*e >= 1);
                    assert!(fac.leading_coefficient().is_one(), "monic factor p={p}");
                    assert!(fac.is_irreducible(), "factor is irreducible p={p}");
                }
            }
        }
    }

    #[test]
    fn irreducibility_matches_brute_force() {
        // Over small p, check is_irreducible against exhaustive trial
        // division by all lower-degree monics.
        for &p in &[2u64, 3, 5] {
            let pm = BigUint::from_u64(p);
            // Every monic polynomial of degree 1..=3.
            for deg in 1..=3usize {
                let mut coeffs = vec![0u64; deg + 1];
                coeffs[deg] = 1;
                loop {
                    let poly =
                        PolyModP::new(coeffs.iter().map(|&c| BigUint::from_u64(c)).collect(), &pm);
                    if poly.degree() == Some(deg) {
                        let brute = brute_irreducible(&poly, p);
                        assert_eq!(
                            poly.is_irreducible(),
                            brute,
                            "is_irreducible mismatch p={p} coeffs={coeffs:?}"
                        );
                    }
                    // Increment the low `deg` coefficients (leading stays 1);
                    // when they all overflow, the enumeration is done.
                    let mut i = 0;
                    while i < deg {
                        coeffs[i] += 1;
                        if coeffs[i] < p {
                            break;
                        }
                        coeffs[i] = 0;
                        i += 1;
                    }
                    if i == deg {
                        break;
                    }
                }
            }
        }
    }

    // Trial-division irreducibility oracle: divisible by no lower-degree
    // monic of degree 1..deg.
    fn brute_irreducible(poly: &PolyModP, p: u64) -> bool {
        let pm = BigUint::from_u64(p);
        let deg = poly.degree().unwrap();
        if deg < 1 {
            return false;
        }
        for d in 1..deg {
            let mut coeffs = vec![0u64; d + 1];
            coeffs[d] = 1;
            loop {
                let divisor =
                    PolyModP::new(coeffs.iter().map(|&c| BigUint::from_u64(c)).collect(), &pm);
                if divisor.degree() == Some(d) && poly.rem(&divisor).is_zero() {
                    return false;
                }
                let mut i = 0;
                while i < d {
                    coeffs[i] += 1;
                    if coeffs[i] < p {
                        break;
                    }
                    coeffs[i] = 0;
                    i += 1;
                }
                if i == d {
                    break;
                }
            }
        }
        true
    }

    #[test]
    fn roots_match_evaluation() {
        let mut rng = SplitMix64 {
            state: 0x9007_0000_0001,
        };
        for &p in &[2u64, 3, 5, 7, 11, 13, 101] {
            let pm = BigUint::from_u64(p);
            for _ in 0..100 {
                let deg = 1 + (rng.next_u64() % 5) as usize;
                let coeffs: Vec<BigUint> = (0..=deg)
                    .map(|_| BigUint::from_u64(rng.next_u64() % p))
                    .collect();
                let f = PolyModP::new(coeffs, &pm);
                if f.degree().unwrap_or(0) < 1 {
                    continue;
                }
                let roots = f.roots(&mut rng);
                // Every returned root is a genuine root, ascending, distinct.
                let mut previous: Option<BigUint> = None;
                for r in &roots {
                    assert!(f.evaluate(r).is_zero(), "claimed root at p={p}");
                    if let Some(prev) = &previous {
                        assert!(prev < r, "roots ascending and distinct");
                    }
                    previous = Some(r.clone());
                }
                // Brute force: every residue that is a root is returned.
                for a in 0..p {
                    let av = BigUint::from_u64(a);
                    if f.evaluate(&av).is_zero() {
                        assert!(roots.contains(&av), "missed root {a} at p={p}");
                    }
                }
            }
        }
    }

    #[test]
    fn factor_of_known_polynomial() {
        // x^2 - 1 = (x-1)(x+1) mod 7.
        let p = BigUint::from_u64(7);
        let f = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[-1, 0, 1]), &p);
        let mut rng = SplitMix64 { state: 0x1234 };
        let factors = f.factor(&mut rng);
        assert_eq!(factors.len(), 2);
        assert_eq!(reassemble(&factors, &p), f.make_monic());
        // x^2 + 1 is irreducible mod 7 (no root: -1 is a non-residue mod 7).
        let g = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[1, 0, 1]), &p);
        assert!(g.is_irreducible());
        assert!(g.roots(&mut rng).is_empty());
    }

    #[test]
    fn factor_p2_trace_map_splits_equal_cubics() {
        // Φ₇ = x⁶+x⁵+x⁴+x³+x²+x+1 = (x³+x+1)(x³+x²+1) over 𝔽₂: two distinct
        // irreducible cubics, so distinct-degree yields one degree-6 block of
        // two degree-3 factors and the equal-degree split runs at p = 2, d = 3.
        // The p = 2 trace-map loop body executes only for d ≥ 2, so this is the
        // case that exercises it (the odd-p exponent path is never taken here).
        let p = BigUint::from_u64(2);
        let f = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[1, 1, 1, 1, 1, 1, 1]), &p);
        let mut rng = SplitMix64 { state: 0xfac0_0002 };
        let mut factors = f.factor(&mut rng);
        assert_eq!(factors.len(), 2, "two cubic factors over F_2");
        for (fac, e) in &factors {
            assert_eq!(*e, 1, "each cubic is simple");
            assert_eq!(fac.degree(), Some(3), "degree-3 factor");
            assert!(fac.is_irreducible(), "irreducible cubic");
        }
        assert_eq!(
            reassemble(&factors, &p),
            f.make_monic(),
            "product reconstructs"
        );
        // The two cubics are exactly x³+x+1 and x³+x²+1.
        factors.sort_by(|a, b| a.0.coefficients().cmp(b.0.coefficients()));
        let c_small = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[1, 0, 1, 1]), &p); // x³+x²+1
        let c_big = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[1, 1, 0, 1]), &p); // x³+x+1
        assert_eq!(factors[0].0, c_small);
        assert_eq!(factors[1].0, c_big);
    }

    #[test]
    fn poly_mod_p_pow_mod_matches_repeated_multiply() {
        let p = BigUint::from_u64(13);
        let mut rng = SplitMix64 {
            state: 0x7011_0000_0001,
        };
        for _ in 0..100 {
            let base = PolyModP::from_poly_z(&rng.poly_z(4, 50), &p);
            let modulus = {
                let mut m = PolyModP::from_poly_z(&rng.poly_z(4, 50), &p);
                if m.degree().unwrap_or(0) < 1 {
                    m = PolyModP::from_poly_z(&PolyZ::from_i64_slice(&[1, 0, 1]), &p);
                }
                m
            };
            let e = rng.next_u64() % 20;
            let by_pow = base.pow_mod(&BigUint::from_u64(e), &modulus);
            let mut by_mul = PolyModP::new(vec![BigUint::one()], &p).rem(&modulus);
            for _ in 0..e {
                by_mul = by_mul.mul(&base).rem(&modulus);
            }
            assert_eq!(by_pow, by_mul, "pow_mod vs repeated multiply");
        }
    }
}
