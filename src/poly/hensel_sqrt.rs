//! Newton lifting of a square root in `ℤ[x]/(f)` from a prime modulus to
//! prime powers.
//!
//! Given a monic `f ∈ ℤ[x]`, a `δ ∈ ℤ[x]/(f)`, and a `β₀` with
//! `β₀² ≡ δ (mod f, q)`, [`HenselSquareRoot`] produces `β` with
//! `β² ≡ δ (mod f, q^k)` for `k = 1, 2, 4, 8, …`. It is the p-adic Newton
//! iteration
//!
//! ```text
//! β ← β − (β² − δ) · (2β)⁻¹
//! ```
//!
//! whose convergence is Hensel's lemma (Hensel, *Neue Grundlagen der
//! Arithmetik*, J. reine angew. Math. 127 (1904), 51–84; the polynomial
//! form used here is Cohen, *A Course in Computational Algebraic Number
//! Theory*, §3.5.3). The precision-doubling schedule and the companion
//! Newton iteration for the reciprocal are Brent's (*Fast multiple-precision
//! evaluation of elementary functions*, J. ACM 23(2) (1976), 242–251,
//! §§3–4).
//!
//! # What makes it fast
//!
//! The naive rendering of the iteration in `(ℤ/q^{2k})[x]/(f)` costs four
//! polynomial products and four polynomial divisions per level, each
//! division a long division whose every step multiplies at full width and
//! reduces modulo `q^{2k}`. Three observations remove nearly all of it:
//!
//! - **Reduction by `f` is integer arithmetic.** `f` is monic with small
//!   coefficients, so `PolyZ::rem_monic` reduces a product by `f` over `ℤ`
//!   with `deg f` products of a wide number by a *small* one per step —
//!   linear time — and the coefficients are then reduced modulo `q^{2k}`
//!   once each, by Barrett with a reciprocal computed once per level.
//! - **The correction lives at half width.** `β_k² − δ` is divisible by
//!   `q^k`, so the correction is `q^k · ((β_k² − δ)/q^k · u mod q^k)`: the
//!   only full-width work is the squaring of `β_k`, which is itself a
//!   product of half-width operands, and the multiplication by `u` happens
//!   at half width. This is the p-adic shape of the last-step economy in
//!   Karp & Markstein, *High-precision division and square root*, ACM
//!   TOMS 23(4) (1997), 561–589.
//! - **The reciprocal lags a level.** The step from `q^k` to `q^{2k}` needs
//!   `u ≡ (2β)⁻¹` only modulo `q^k`, so `u` is refined to `q^{2k}` only
//!   when — and if — a further level is asked for. The final level, the
//!   widest and dearest, skips the refinement entirely.
//!
//! Every level checks nothing: the invariant is established once by
//! [`HenselSquareRoot::new`] at the prime and preserved algebraically. A
//! caller after an exact integer root (as the number field sieve is) squares
//! the [`symmetric_lift`](HenselSquareRoot::symmetric_lift) back over `ℤ`
//! and stops when it matches.

use super::{PolyMod, PolyZ};
use crate::bigint::{BarrettContext, BigInt, BigUint, Sign, NEWTON_DIVISION_THRESHOLD_LIMBS};

/// Coefficients narrower than this are multiplied on the calling thread
/// regardless of the worker budget: a thread costs more than the product.
const PARALLEL_PRODUCT_MIN_BITS: usize = 1 << 16;

/// Reduction modulo one fixed `q^k`, by Barrett once the modulus is wide
/// enough that a division would go through Newton's reciprocal — which
/// Barrett's `μ` is, computed once instead of once per coefficient.
struct Ring {
    modulus: BigUint,
    barrett: Option<BarrettContext>,
}

impl Ring {
    fn new(modulus: BigUint) -> Self {
        let barrett = (modulus.limbs().len() >= NEWTON_DIVISION_THRESHOLD_LIMBS)
            .then(|| BarrettContext::new(&modulus).ok())
            .flatten();
        Self { modulus, barrett }
    }

    fn reduce(&self, x: &BigUint) -> BigUint {
        match &self.barrett {
            Some(context) => context.reduce(x),
            None => x.rem(&self.modulus),
        }
    }

    /// `x mod m` in `[0, m)` for a signed `x`.
    fn reduce_signed(&self, x: &BigInt) -> BigUint {
        match x.sign() {
            Sign::Zero => BigUint::zero(),
            Sign::Positive => self.reduce(x.magnitude()),
            Sign::Negative => {
                let reduced = self.reduce(x.magnitude());
                if reduced.is_zero() {
                    reduced
                } else {
                    self.modulus.sub(&reduced)
                }
            }
        }
    }
}

/// One precision `q^{2^j}`: its reduction machinery and `δ` modulo it.
struct Level {
    ring: Ring,
    delta: Vec<BigUint>,
}

impl Level {
    /// The level at `modulus`, with `δ` reduced from its integer form —
    /// a division per coefficient, which is what
    /// [`HenselSquareRoot::with_precision_hint`] exists to avoid.
    fn from_integers(modulus: BigUint, delta: &[BigInt]) -> Self {
        let ring = Ring::new(modulus);
        let delta = delta.iter().map(|d| ring.reduce_signed(d)).collect();
        Self { ring, delta }
    }

    /// The level at `modulus`, with `δ` reduced from the level above — whose
    /// residues are below the square of this modulus, so Barrett applies.
    fn from_above(modulus: BigUint, above: &Self) -> Self {
        let ring = Ring::new(modulus);
        let delta = above.delta.iter().map(|d| ring.reduce(d)).collect();
        Self { ring, delta }
    }
}

/// A square root of `δ` in `ℤ[x]/(f)` modulo `q^k`, lifted one doubling of
/// `k` at a time. See the [module documentation](self).
pub struct HenselSquareRoot<'a> {
    f: &'a PolyZ,
    /// `δ` reduced by `f` over `ℤ`, padded to `deg f` coefficients: the
    /// integers every level's residues descend from.
    delta: Vec<BigInt>,
    /// `levels[j]` is the precision `q^{2^j}`; the root lives at
    /// `levels[level]`.
    levels: Vec<Level>,
    level: usize,
    /// `deg f` coefficients in `[0, q^{2^level})`.
    root: Vec<BigUint>,
    /// `(2β)⁻¹` at `levels[inverse_level]`, which is `level` right after
    /// construction and `level − 1` after every doubling: it is refined
    /// only on demand.
    inverse_level: usize,
    inverse: Vec<BigUint>,
    workers: usize,
}

impl<'a> HenselSquareRoot<'a> {
    /// Start a lift from `root`, a square root of `delta` in
    /// `(ℤ/q)[x]/(f)`, and `inverse ≡ (2·root)⁻¹` there. Both are checked;
    /// `None` when either congruence fails, since a wrong seed would lift
    /// to a wrong answer without complaint.
    ///
    /// `f` must be monic. `delta` need not be reduced by `f`; it is reduced
    /// over `ℤ` here.
    ///
    /// # Panics
    ///
    /// Panics if `f` is not monic, if `root` and `inverse` carry different
    /// moduli, or if that modulus is below 2.
    #[must_use]
    pub fn new(f: &'a PolyZ, delta: &PolyZ, root: &PolyMod, inverse: &PolyMod) -> Option<Self> {
        let degree = f.degree().expect("f is non-zero");
        assert!(
            f.leading_coefficient().is_one(),
            "the Hensel square root lift needs a monic f"
        );
        let prime = root.modulus().clone();
        assert!(
            prime >= BigUint::from_u64(2),
            "the modulus must be at least 2"
        );
        let f_mod = PolyMod::from_poly_z(f, &prime);
        let delta_reduced = delta.rem_monic(f);
        let delta_mod = PolyMod::from_poly_z(&delta_reduced, &prime);
        if root.mul(root).rem(&f_mod) != delta_mod {
            return None;
        }
        let two = PolyMod::new(vec![BigUint::from_u64(2)], &prime);
        let one = PolyMod::new(vec![BigUint::one()], &prime);
        if two.mul(root).mul(inverse).rem(&f_mod) != one {
            return None;
        }
        let mut delta = delta_reduced.coefficients().to_vec();
        delta.resize(degree, BigInt::zero());
        let root = padded(root.coefficients(), degree);
        let inverse = padded(inverse.coefficients(), degree);
        let base = Level::from_integers(prime, &delta);
        Some(Self {
            f,
            delta,
            levels: vec![base],
            level: 0,
            root,
            inverse_level: 0,
            inverse,
            workers: 1,
        })
    }

    /// Start a lift in a *field* `𝔽_q[x]/(f)`, computing the reciprocal of
    /// `2·root` by Fermat: `f` must be irreducible modulo the prime `q`, so
    /// that the multiplicative group has order `q^d − 1`. The root itself is
    /// still checked, and `None` is returned if it does not square to
    /// `delta` or if `2·root` is zero.
    ///
    /// # Panics
    ///
    /// As [`Self::new`].
    #[must_use]
    pub fn in_field(f: &'a PolyZ, delta: &PolyZ, root: &PolyMod) -> Option<Self> {
        let degree = f.degree().expect("f is non-zero");
        let prime = root.modulus().clone();
        let f_mod = PolyMod::from_poly_z(f, &prime);
        let two = PolyMod::new(vec![BigUint::from_u64(2)], &prime);
        let doubled = two.mul(root).rem(&f_mod);
        if doubled.is_zero() {
            return None;
        }
        let order = prime.pow_u64(degree as u64);
        let inverse = doubled.mod_pow(&order.sub(&BigUint::from_u64(2)), &f_mod);
        Self::new(f, delta, root, &inverse)
    }

    /// Spread the coefficient products of each level over up to `workers`
    /// threads. One (the default) keeps everything on the calling thread.
    #[must_use]
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.workers = workers.max(1);
        self
    }

    /// Announce the width the lift is expected to reach, so that `δ`'s
    /// residues can be prepared from the top down.
    ///
    /// Each level needs `δ mod q^{2^j}`. Taken from `δ` itself that is a
    /// long division of a wide integer by a narrow one, per coefficient
    /// per level, and at the widths the number field sieve reaches those
    /// divisions outweigh the products they sit beside. Taken from the
    /// level above instead — `δ mod q^{2^j} = (δ mod q^{2^{j+1}}) mod q^{2^j}`,
    /// with the input below the square of the modulus — each is one
    /// Barrett reduction. The hint says how far up to start; levels past
    /// it, if the lift goes on, fall back to the divisions.
    #[must_use]
    pub fn with_precision_hint(mut self, bits: usize) -> Self {
        let mut moduli = vec![self.levels[0].ring.modulus.clone()];
        while moduli.last().expect("non-empty").bits() < bits {
            let next = moduli.last().expect("non-empty").square();
            moduli.push(next);
        }
        let start = self.levels.len();
        if moduli.len() <= start {
            return self;
        }
        let top = moduli.len() - 1;
        let mut descending = vec![Level::from_integers(moduli[top].clone(), &self.delta)];
        for j in (start..top).rev() {
            let above = descending.last().expect("non-empty");
            descending.push(Level::from_above(moduli[j].clone(), above));
        }
        descending.reverse();
        self.levels.extend(descending);
        self
    }

    /// The current modulus `q^k`.
    #[must_use]
    pub fn modulus(&self) -> &BigUint {
        &self.levels[self.level].ring.modulus
    }

    /// The root modulo the current modulus.
    #[must_use]
    pub fn root(&self) -> PolyMod {
        PolyMod::new(self.root.clone(), self.modulus())
    }

    /// The root's coefficients, `deg f` of them in `[0, q^k)`.
    #[must_use]
    pub fn coefficients(&self) -> &[BigUint] {
        &self.root
    }

    /// The root with coefficients in `(−q^k/2, q^k/2]`: the integer
    /// polynomial it converges to, once `q^k` exceeds twice the largest
    /// coefficient of that polynomial.
    #[must_use]
    pub fn symmetric_lift(&self) -> PolyZ {
        self.root().symmetric_lift()
    }

    /// Make sure `levels[j]` exists, by squaring upward from the last one.
    fn ensure_level(&mut self, j: usize) {
        while self.levels.len() <= j {
            let modulus = self.levels.last().expect("non-empty").ring.modulus.square();
            let level = Level::from_integers(modulus, &self.delta);
            self.levels.push(level);
        }
    }

    /// Square the modulus: `q^k → q^{2k}`, with the root correct to the new
    /// precision.
    pub fn double(&mut self) {
        self.refine_inverse();
        self.ensure_level(self.level + 1);

        // β_k² − δ over ℤ, reduced by f, then modulo q^{2k}; each
        // coefficient is a multiple of q^k, and the quotient is what the
        // correction is built from.
        let squared = self.reduce_by_f(&self.square(&self.root));
        let previous = &self.levels[self.level];
        let next = &self.levels[self.level + 1];
        let error: Vec<BigUint> = squared
            .iter()
            .zip(&next.delta)
            .map(|(s, d)| {
                let residue = next
                    .ring
                    .reduce_signed(&s.sub(&BigInt::from_biguint(d.clone())));
                let (quotient, remainder) = residue.div_rem(&previous.ring.modulus);
                debug_assert!(
                    remainder.is_zero(),
                    "β² − δ must vanish modulo q^k: the seed was not a root"
                );
                quotient
            })
            .collect();

        // c = e · u mod (f, q^k), then β ← β − q^k · c mod q^{2k}.
        let correction = self.reduce_by_f(&self.multiply(&error, &self.inverse));
        for (b, c) in self.root.iter_mut().zip(&correction) {
            let c = previous.ring.reduce_signed(c);
            let step = previous.ring.modulus.mul(&c);
            *b = if *b >= step {
                b.sub(&step)
            } else {
                b.add(&next.ring.modulus).sub(&step)
            };
        }
        self.level += 1;
    }

    /// Bring `u ≡ (2β)⁻¹` from `q^{k/2}` up to the root's current `q^k`, by
    /// the same half-width Newton step: `2βu − 1` is a multiple of
    /// `q^{k/2}`, and `u ← u − q^{k/2} · ((2βu − 1)/q^{k/2} · u mod q^{k/2})`.
    fn refine_inverse(&mut self) {
        if self.inverse_level == self.level {
            return;
        }
        let doubled: Vec<BigUint> = self
            .root
            .iter()
            .map(|b| b.mul(&BigUint::from_u64(2)))
            .collect();
        let product = self.reduce_by_f(&self.multiply(&doubled, &self.inverse));
        let half = &self.levels[self.inverse_level];
        let full = &self.levels[self.level];
        let error: Vec<BigUint> = product
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let p = if i == 0 {
                    p.sub(&BigInt::one())
                } else {
                    p.clone()
                };
                let residue = full.ring.reduce_signed(&p);
                let (quotient, remainder) = residue.div_rem(&half.ring.modulus);
                debug_assert!(
                    remainder.is_zero(),
                    "2βu − 1 must vanish modulo the reciprocal's modulus"
                );
                quotient
            })
            .collect();
        let correction = self.reduce_by_f(&self.multiply(&error, &self.inverse));
        for (u, c) in self.inverse.iter_mut().zip(&correction) {
            let c = half.ring.reduce_signed(c);
            let step = half.ring.modulus.mul(&c);
            *u = if *u >= step {
                u.sub(&step)
            } else {
                u.add(&full.ring.modulus).sub(&step)
            };
        }
        self.inverse_level = self.level;
    }

    /// The product by `f` over `ℤ`, back to `deg f` signed coefficients.
    fn reduce_by_f(&self, convolution: &[BigUint]) -> Vec<BigInt> {
        let degree = self.root.len();
        let poly = PolyZ::new(
            convolution
                .iter()
                .cloned()
                .map(BigInt::from_biguint)
                .collect(),
        );
        let mut coefficients = poly.rem_monic(self.f).coefficients().to_vec();
        coefficients.resize(degree, BigInt::zero());
        coefficients
    }

    /// The convolution `a ⋆ b` over `ℕ`, `2d − 1` coefficients, its `d²`
    /// products spread over the worker budget.
    fn multiply(&self, a: &[BigUint], b: &[BigUint]) -> Vec<BigUint> {
        let degree = a.len();
        let pairs: Vec<(usize, usize)> = (0..degree)
            .flat_map(|i| (0..degree).map(move |j| (i, j)))
            .collect();
        let products = self.products(&pairs, |(i, j)| a[i].mul(&b[j]));
        let mut out = vec![BigUint::zero(); 2 * degree - 1];
        for ((i, j), p) in pairs.iter().zip(products) {
            out[i + j] = out[i + j].add(&p);
        }
        out
    }

    /// The convolution `a ⋆ a`, forming each cross product once.
    fn square(&self, a: &[BigUint]) -> Vec<BigUint> {
        let degree = a.len();
        let pairs: Vec<(usize, usize)> = (0..degree)
            .flat_map(|i| (i..degree).map(move |j| (i, j)))
            .collect();
        let products = self.products(&pairs, |(i, j)| {
            if i == j {
                a[i].square()
            } else {
                a[i].mul(&a[j]).mul(&BigUint::from_u64(2))
            }
        });
        let mut out = vec![BigUint::zero(); 2 * degree - 1];
        for ((i, j), p) in pairs.iter().zip(products) {
            out[i + j] = out[i + j].add(&p);
        }
        out
    }

    /// `task` over every pair, on the calling thread unless the budget and
    /// the width both justify threads.
    fn products<F>(&self, pairs: &[(usize, usize)], task: F) -> Vec<BigUint>
    where
        F: Fn((usize, usize)) -> BigUint + Sync,
    {
        let wide = self.modulus().bits() >= PARALLEL_PRODUCT_MIN_BITS;
        let threads = self.workers.min(pairs.len());
        if threads <= 1 || !wide {
            return pairs.iter().map(|&pair| task(pair)).collect();
        }
        let mut out: Vec<Option<BigUint>> = vec![None; pairs.len()];
        let chunk = pairs.len().div_ceil(threads);
        std::thread::scope(|scope| {
            let task = &task;
            let handles: Vec<_> = pairs
                .chunks(chunk)
                .map(|chunk| {
                    scope.spawn(move || chunk.iter().map(|&pair| task(pair)).collect::<Vec<_>>())
                })
                .collect();
            let mut position = 0;
            for handle in handles {
                for product in handle.join().expect("a coefficient product does not panic") {
                    out[position] = Some(product);
                    position += 1;
                }
            }
        });
        out.into_iter()
            .map(|p| p.expect("every product was filled"))
            .collect()
    }
}

fn padded(coefficients: &[BigUint], degree: usize) -> Vec<BigUint> {
    let mut out = coefficients.to_vec();
    out.resize(degree, BigUint::zero());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poly(coefficients: &[i64]) -> PolyZ {
        PolyZ::new(coefficients.iter().map(|&c| BigInt::from_i64(c)).collect())
    }

    /// The first prime at or above `from` modulo which `f` is irreducible,
    /// so that `𝔽_q[x]/(f)` is a field and `sqrt_in_field` applies.
    fn irreducible_prime(f: &PolyZ, from: u64) -> u64 {
        let degree = f.degree().expect("non-zero");
        (from..)
            .filter(|&q| crate::number_theory_impl::is_probable_prime_bpsw(&BigUint::from_u64(q)))
            .find(|&q| {
                let reduced = PolyMod::from_poly_z(f, &BigUint::from_u64(q));
                reduced.degree() == Some(degree) && reduced.is_irreducible()
            })
            .expect("the primes do not run out")
    }

    fn field_root(f: &PolyZ, delta: &PolyZ, prime: u64) -> PolyMod {
        let prime = BigUint::from_u64(prime);
        let f_mod = PolyMod::from_poly_z(f, &prime);
        let residue = PolyMod::from_poly_z(&delta.rem_monic(f), &prime);
        residue
            .sqrt_in_field(&f_mod, f.degree().expect("non-zero"))
            .expect("delta is a square by construction")
    }

    /// A lift from a small prime converges to the integer root whose residue
    /// it started from, and every level squares to δ modulo (f, q^k).
    #[test]
    fn the_lift_converges_to_the_integer_root() {
        // x² + 1 is irreducible modulo 7 (−1 is a non-residue).
        let f = poly(&[1, 0, 1]);
        let beta = poly(&[5, 3]);
        let delta = beta.mul(&beta).rem_monic(&f);
        let seed = field_root(&f, &delta, 7);
        let mut lift = HenselSquareRoot::in_field(&f, &delta, &seed).expect("a square");
        let mut converged = None;
        for _ in 0..6 {
            let modulus = lift.modulus().clone();
            let f_mod = PolyMod::from_poly_z(&f, &modulus);
            let root = lift.root();
            let delta_mod = PolyMod::from_poly_z(&delta, &modulus).rem(&f_mod);
            assert_eq!(root.mul(&root).rem(&f_mod), delta_mod, "at {modulus}");
            let candidate = lift.symmetric_lift();
            if candidate.mul(&candidate).rem_monic(&f) == delta {
                converged = Some(candidate);
                break;
            }
            lift.double();
        }
        let root = converged.expect("q^32 is far wider than the coefficients");
        assert!(root == beta || root.add(&beta).is_zero(), "got {root:?}");
    }

    /// The lazy reciprocal is refined correctly across several levels: the
    /// invariant `2βu ≡ 1` is what the level after next depends on.
    #[test]
    fn the_lagging_reciprocal_keeps_the_invariant() {
        let f = poly(&[7, -3, 11, 5, 1]);
        let beta = poly(&[-123_456_789, 987_654_321, -555_555_555, 42]);
        let delta = beta.mul(&beta).rem_monic(&f);
        let prime = irreducible_prime(&f, 1_000_003);
        let seed = field_root(&f, &delta, prime);
        let mut lift = HenselSquareRoot::in_field(&f, &delta, &seed).expect("a square");
        for _ in 0..3 {
            lift.double();
        }
        let modulus = lift.modulus().clone();
        assert_eq!(modulus, BigUint::from_u64(prime).pow_u64(8));
        let f_mod = PolyMod::from_poly_z(&f, &modulus);
        let root = lift.root();
        let delta_mod = PolyMod::from_poly_z(&delta, &modulus).rem(&f_mod);
        assert_eq!(root.mul(&root).rem(&f_mod), delta_mod);
        let candidate = lift.symmetric_lift();
        assert!(candidate == beta || candidate.add(&beta).is_zero());
    }

    /// Wide enough that the Barrett path and the threaded products both
    /// run: coefficients near 200 kbit take the modulus past the Newton
    /// division threshold before the lift is exact.
    #[test]
    fn the_wide_lift_uses_barrett_and_threads() {
        let f = poly(&[7, -3, 11, 5, 1]);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut wide = || {
            let mut value = BigUint::zero();
            for bit in 0..200_000usize {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                if seed & 1 == 1 {
                    value.set_bit(bit);
                }
            }
            value
        };
        let beta = PolyZ::new(vec![
            BigInt::from_parts(Sign::Negative, wide()),
            BigInt::from_biguint(wide()),
            BigInt::from_parts(Sign::Negative, wide()),
            BigInt::from_biguint(wide()),
        ]);
        let delta = beta.mul(&beta).rem_monic(&f);
        let seed = field_root(&f, &delta, irreducible_prime(&f, 1_000_003));
        let mut lift = HenselSquareRoot::in_field(&f, &delta, &seed)
            .expect("a square")
            .with_workers(4)
            .with_precision_hint(300_000);
        let mut converged = false;
        for _ in 0..16 {
            let candidate = lift.symmetric_lift();
            if candidate.mul(&candidate).rem_monic(&f) == delta {
                converged = candidate == beta || candidate.add(&beta).is_zero();
                break;
            }
            lift.double();
        }
        assert!(converged);
        assert!(
            lift.modulus().limbs().len() >= NEWTON_DIVISION_THRESHOLD_LIMBS,
            "the test must reach the Barrett width to mean anything"
        );
    }

    /// A seed that is not a root is refused rather than lifted.
    #[test]
    fn a_wrong_seed_is_refused() {
        let f = poly(&[1, 0, 1]);
        let beta = poly(&[5, 3]);
        let delta = beta.mul(&beta).rem_monic(&f);
        let prime = BigUint::from_u64(7);
        let wrong = PolyMod::new(vec![BigUint::from_u64(2), BigUint::from_u64(1)], &prime);
        assert!(HenselSquareRoot::in_field(&f, &delta, &wrong).is_none());
        let seed = field_root(&f, &delta, 7);
        let bad_inverse = PolyMod::new(vec![BigUint::one()], &prime);
        assert!(HenselSquareRoot::new(&f, &delta, &seed, &bad_inverse).is_none());
    }

    /// Timing probe at number-field-sieve width: a root with coefficients
    /// of half `RUMP_HENSEL_PROBE_BITS` (default 1,306,239, the c70 case),
    /// levels above 100 kbit reported. Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "timing probe for the lift at NFS width; run with --ignored"]
    fn lift_timing_probe() {
        use std::time::Instant;
        let bits: usize = std::env::var("RUMP_HENSEL_PROBE_BITS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1_306_239);
        let f = poly(&[7, -3, 11, 5, 1]);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut wide = || {
            let mut value = BigUint::zero();
            for bit in 0..(bits / 2 + 200) {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                if seed & 1 == 1 {
                    value.set_bit(bit);
                }
            }
            value
        };
        let beta = PolyZ::new((0..4).map(|_| BigInt::from_biguint(wide())).collect());
        let delta = beta.mul(&beta).rem_monic(&f);
        let seed = field_root(&f, &delta, irreducible_prime(&f, 1_000_003));
        for workers in [1usize, 4, 16] {
            let start = Instant::now();
            let mut lift = HenselSquareRoot::in_field(&f, &delta, &seed)
                .expect("a square")
                .with_workers(workers)
                .with_precision_hint(bits + 1);
            let setup = start.elapsed().as_secs_f64();
            let mut levels = 0;
            loop {
                let level = Instant::now();
                let candidate = lift.symmetric_lift();
                if candidate.mul(&candidate).rem_monic(&f) == delta {
                    break;
                }
                lift.double();
                levels += 1;
                if lift.modulus().bits() > 100_000 {
                    eprintln!(
                        "  level {} bits: {:.3}s",
                        lift.modulus().bits(),
                        level.elapsed().as_secs_f64()
                    );
                }
            }
            eprintln!(
                "hensel lift {:.3}s (setup {setup:.3}s, {levels} levels, {workers} workers)",
                start.elapsed().as_secs_f64()
            );
        }
    }
}
