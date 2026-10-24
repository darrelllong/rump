//! Newton lifting of a square root in `ℤ[x]/(f)` from a prime modulus to
//! prime powers.
//!
//! Given a monic `f ∈ ℤ[x]`, a `δ ∈ ℤ[x]/(f)`, and a `β₀` with
//! `β₀² ≡ δ (mod f, q)`, [`HenselSquareRoot`] produces `β` with
//! `β² ≡ δ (mod f, q^k)` for a climbing `k`, each at most twice the one
//! before. It is the p-adic Newton iteration
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
//!   once each, by Barrett (once the modulus is wide) with a reciprocal
//!   computed once per level.
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
//! - **The last precision is the one asked for.** A step from `q^k` may go
//!   to any `q^{k'}` with `k' ≤ 2k`, so a lift told its target `q^K` takes
//!   `K` halved, rounded up, down to 1 as its precisions, and its widest
//!   level is `q^K` and not the first `q^{2^j}` past it, which can be all
//!   but twice as wide.
//!
//! No level checks anything: the invariant is established once by
//! [`HenselSquareRoot::new`] at the prime and preserved algebraically.

use super::{PolyMod, PolyZ};
use crate::bigint::{BarrettContext, BigInt, BigUint, Sign, NEWTON_DIVISION_THRESHOLD_LIMBS};

/// A level whose modulus is narrower than this multiplies on the calling
/// thread regardless of the worker budget. The gate reads the modulus
/// width, which bounds every coefficient's: below it a thread costs more
/// than the products it would carry. `2¹⁶` is a policy; where the
/// crossover lies has not been measured, and a timing of the threaded
/// against the serial products across modulus widths would settle it.
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

    /// `(⌊x/m⌋, x mod m)`: by Barrett's estimate once the modulus is wide,
    /// where a division would build a reciprocal of its own each time.
    fn div_rem(&self, x: &BigUint) -> (BigUint, BigUint) {
        match &self.barrett {
            Some(context) => context.div_rem(x),
            None => x.div_rem(&self.modulus),
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

/// A precision a lift passes through: `q^exponent`, and the rise over the
/// precision before it, `q^{exponent − k}` for that one's `q^k`, when the
/// rise is not `q^k` itself.
struct Rung {
    exponent: usize,
    modulus: BigUint,
    rise: Option<BigUint>,
}

impl Rung {
    fn seed(prime: &BigUint) -> Self {
        Self {
            exponent: 1,
            modulus: prime.clone(),
            rise: None,
        }
    }
}

/// The rungs from `q^from`, which is `below`, to `q^top`: `top` halved and
/// rounded up while past `from`, climbing. Each exponent is then at most
/// twice the one before: the least past `from` halves to `from` or below.
fn rungs_to(prime: &BigUint, from: usize, below: &BigUint, top: usize) -> Vec<Rung> {
    let mut exponents = Vec::new();
    let mut exponent = top;
    while exponent > from {
        exponents.push(exponent);
        exponent = exponent.div_ceil(2);
    }
    let mut rungs: Vec<Rung> = Vec::with_capacity(exponents.len());
    let (mut k, mut modulus) = (from, below.clone());
    for &next in exponents.iter().rev() {
        let rise = next - k;
        let rung = if rise == k {
            Rung {
                exponent: next,
                modulus: modulus.square(),
                rise: None,
            }
        } else {
            let rise = if rise + 1 == k {
                divide_exact(&modulus, prime)
            } else {
                prime.pow_u64(rise as u64)
            };
            Rung {
                exponent: next,
                modulus: modulus.mul(&rise),
                rise: Some(rise),
            }
        };
        (k, modulus) = (next, rung.modulus.clone());
        rungs.push(rung);
    }
    rungs
}

/// The rungs from `q^from`, which is `below`, to the least power of `prime`
/// at least `bits` wide; none when `below` is. The exponent is first
/// estimated from `log₂ q` and then settled on the powers themselves.
fn rungs_reaching(prime: &BigUint, from: usize, below: &BigUint, bits: usize) -> Vec<Rung> {
    if below.bits() >= bits {
        return Vec::new();
    }
    let estimate = (bits - 1) as f64 / prime.to_f64_lossy().log2();
    let mut top = (estimate.ceil() as usize).max(from + 1);
    loop {
        let rungs = rungs_to(prime, from, below, top);
        let modulus = &rungs.last().expect("past from").modulus;
        if modulus.bits() < bits {
            top += 1;
        } else if top > from + 1 && divide_exact(modulus, prime).bits() >= bits {
            top -= 1;
        } else {
            return rungs;
        }
    }
}

/// `value / prime`, which divides it.
fn divide_exact(value: &BigUint, prime: &BigUint) -> BigUint {
    if let Some(small) = prime.to_u64() {
        let (quotient, remainder) = value.div_rem_u64(small);
        debug_assert_eq!(remainder, 0, "the prime divides its power");
        quotient
    } else {
        let (quotient, remainder) = value.div_rem(prime);
        debug_assert!(remainder.is_zero(), "the prime divides its power");
        quotient
    }
}

/// One precision `q^k`: its reduction machinery, `δ` modulo it, and the
/// ring of its rise over the level below when that is not the level below's
/// own (a step that does not double).
struct Level {
    exponent: usize,
    ring: Ring,
    rise: Option<Ring>,
    delta: Vec<BigUint>,
}

impl Level {
    /// The level at `rung`, with `δ` reduced from its integer form — a
    /// division per coefficient, which is what
    /// [`HenselSquareRoot::with_precision_hint`] exists to avoid.
    fn from_integers(rung: Rung, delta: &[BigInt]) -> Self {
        let ring = Ring::new(rung.modulus);
        let delta = delta.iter().map(|d| ring.reduce_signed(d)).collect();
        Self {
            exponent: rung.exponent,
            ring,
            rise: rung.rise.map(Ring::new),
            delta,
        }
    }
}

/// The levels at `rungs`, climbing: `δ` at the widest from `top`, given
/// that level's ring, and at each below reduced from the level above, whose
/// residues are below the square of its modulus, so Barrett applies. The
/// rings are independent of one another and made side by side, the widest
/// first; a level's coefficients are reduced side by side.
fn levels_at(
    mut rungs: Vec<Rung>,
    workers: usize,
    top: impl FnOnce(&Ring) -> Vec<BigUint>,
) -> Vec<Level> {
    rungs.reverse();
    let rings: Vec<(Ring, Option<Ring>)> =
        crate::parallel::map_ordered(&rungs, workers, |_, rung| {
            (
                Ring::new(rung.modulus.clone()),
                rung.rise.clone().map(Ring::new),
            )
        });
    let mut top = Some(top);
    let mut descending: Vec<Level> = Vec::with_capacity(rungs.len());
    for (rung, (ring, rise)) in rungs.iter().zip(rings) {
        let delta = match descending.last() {
            None => top.take().expect("one widest level")(&ring),
            Some(above) => {
                crate::parallel::map_ordered(&above.delta, workers, |_, d| ring.reduce(d))
            }
        };
        descending.push(Level {
            exponent: rung.exponent,
            ring,
            rise,
            delta,
        });
    }
    descending.reverse();
    descending
}

/// A square root of `δ` in `ℤ[x]/(f)` modulo `q^k`, lifted a level at a
/// time, `k` at most doubling, by the p-adic Newton iteration `β ← β − (β² − δ)·(2β)⁻¹`,
/// whose convergence is Hensel's lemma (Cohen, *A Course in Computational
/// Algebraic Number Theory*, §3.5.3).
///
/// The seed is checked once, by [`Self::new`] or [`Self::in_field`]; after
/// that no level checks anything, since each [`Self::double`] preserves
/// `β² ≡ δ (mod f, q^k)` algebraically. A caller after an exact root over
/// `ℤ`, as the number field sieve is, squares [`Self::symmetric_lift`] back
/// over `ℤ` and stops when it matches.
pub struct HenselSquareRoot<'a> {
    f: &'a PolyZ,
    /// `δ` reduced by `f` over `ℤ`, padded to `deg f` coefficients: the
    /// integers every level's residues descend from. `None` for a lift
    /// given `δ` modulo its last modulus, whose levels are all prepared.
    delta: Option<Vec<BigInt>>,
    /// `levels[j]` is a precision `q^k`, `levels[0]` the prime's, each `k`
    /// at most twice the one before; the root lives at `levels[level]`.
    levels: Vec<Level>,
    level: usize,
    /// `deg f` coefficients in `[0, q^k)` for `levels[level]`'s `q^k`.
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
    /// Panics if `f` is zero or not monic, if `root` and `inverse` carry different
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
        let mut delta = delta.rem_monic(f).coefficients().to_vec();
        delta.resize(degree, BigInt::zero());
        let base = Level::from_integers(Rung::seed(&prime), &delta);
        Self::seeded(f, Some(delta), vec![base], root, inverse)
    }

    /// The lift at `levels[0]`, its seed checked there against the level's
    /// own `δ`: `None` when `root² ≢ δ` or `2·root·inverse ≢ 1`.
    fn seeded(
        f: &'a PolyZ,
        delta: Option<Vec<BigInt>>,
        levels: Vec<Level>,
        root: &PolyMod,
        inverse: &PolyMod,
    ) -> Option<Self> {
        let degree = f.degree().expect("f is non-zero");
        let prime = root.modulus();
        let f_mod = PolyMod::from_poly_z(f, prime);
        let delta_mod = PolyMod::new(levels[0].delta.clone(), prime);
        if root.mul(root).rem(&f_mod) != delta_mod {
            return None;
        }
        let two = PolyMod::new(vec![BigUint::from_u64(2)], prime);
        let one = PolyMod::new(vec![BigUint::one()], prime);
        if two.mul(root).mul(inverse).rem(&f_mod) != one {
            return None;
        }
        Some(Self {
            f,
            delta,
            levels,
            level: 0,
            root: padded(root.coefficients(), degree),
            inverse_level: 0,
            inverse: padded(inverse.coefficients(), degree),
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
        Self::new(f, delta, root, &Self::field_inverse(f, root)?)
    }

    /// `(2·root)⁻¹` in the field `𝔽_q[x]/(f)` by Fermat, or `None` when
    /// `2·root` is zero there.
    fn field_inverse(f: &PolyZ, root: &PolyMod) -> Option<PolyMod> {
        let degree = f.degree().expect("f is non-zero");
        let prime = root.modulus();
        let f_mod = PolyMod::from_poly_z(f, prime);
        let two = PolyMod::new(vec![BigUint::from_u64(2)], prime);
        let doubled = two.mul(root).rem(&f_mod);
        if doubled.is_zero() {
            return None;
        }
        let order = prime.pow_u64(degree as u64);
        Some(doubled.mod_pow(&order.sub(&BigUint::from_u64(2)), &f_mod))
    }

    /// The modulus a lift from `prime` first has at or past `bits` wide:
    /// `prime^k` for the least such `k`, which is `prime` itself when that
    /// is wide enough. It is the modulus
    /// [`Self::with_precision_hint`] prepares `δ` at, and the one
    /// [`Self::in_field_bounded`] wants `δ` reduced by.
    ///
    /// # Panics
    ///
    /// Panics if `prime` is below 2, whose powers do not widen.
    #[must_use]
    pub fn modulus_reaching(prime: &BigUint, bits: usize) -> BigUint {
        assert!(
            *prime >= BigUint::from_u64(2),
            "the modulus must be at least 2"
        );
        rungs_reaching(prime, 1, prime, bits)
            .pop()
            .map_or_else(|| prime.clone(), |rung| rung.modulus)
    }

    /// [`Self::in_field`] for a lift that goes as far as `modulus` and no
    /// farther, given `δ` modulo `modulus` in place of `δ`.
    ///
    /// A lift's first act is to reduce `δ` by the modulus it will reach, a
    /// long division for each coefficient. One lift cannot avoid it. A
    /// caller lifting at many primes can: a
    /// [`remainder_tree`](crate::number_theory::remainder_tree) over the
    /// moduli reduces `δ` by all of them in a few divisions of balanced
    /// operands, and each lift is then given its residues.
    ///
    /// `modulus` is [`Self::modulus_reaching`]'s for `root`'s prime.
    /// `residues` are the coefficients of `δ`, reduced by `f` over `ℤ`,
    /// each modulo `modulus` and in `[0, modulus)`, lowest degree first;
    /// fewer than `deg f` are padded with zeros. `None` as
    /// [`Self::in_field`].
    ///
    /// # Panics
    ///
    /// As [`Self::new`]; if `modulus` is not a power of the prime; if there are more than `deg f` residues or one is not below
    /// `modulus`. [`Self::double`] panics past `modulus`, as does
    /// [`Self::with_precision_hint`] asked for more than its width.
    #[must_use]
    pub fn in_field_bounded(
        f: &'a PolyZ,
        residues: &[BigUint],
        modulus: &BigUint,
        root: &PolyMod,
    ) -> Option<Self> {
        let degree = f.degree().expect("f is non-zero");
        assert!(
            f.leading_coefficient().is_one(),
            "the Hensel square root lift needs a monic f"
        );
        let prime = root.modulus();
        assert!(
            *prime >= BigUint::from_u64(2),
            "the modulus must be at least 2"
        );
        assert!(
            residues.len() <= degree,
            "δ reduced by f has at most deg f coefficients"
        );
        assert!(
            residues.iter().all(|residue| residue < modulus),
            "a residue is below its modulus"
        );
        let mut rungs = rungs_reaching(prime, 1, prime, modulus.bits());
        rungs.insert(0, Rung::seed(prime));
        assert!(
            rungs.last().expect("the seed's").modulus == *modulus,
            "the modulus is a power of the prime"
        );
        let levels = levels_at(rungs, 1, |_| padded(residues, degree));
        let inverse = Self::field_inverse(f, root)?;
        Self::seeded(f, None, levels, root, &inverse)
    }

    /// Spread each level's coefficient products and reductions, and the
    /// making of the levels a hint plans, over up to `workers` threads. One
    /// (the default) keeps everything on the calling thread.
    #[must_use]
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.workers = workers.max(1);
        self
    }

    /// Announce the width the lift is expected to reach, so that `δ`'s
    /// residues can be prepared from the top down.
    ///
    /// The levels from the lift's last to the least `q^K` at least `bits`
    /// wide are planned from the top: `K` halved and rounded up, so that
    /// the widest is `q^K` itself. Each level needs `δ mod q^k`. Taken
    /// from `δ` itself that is a long division of a wide integer by a
    /// narrow one, per coefficient per level, and at wide precisions those
    /// divisions outweigh the products they sit beside. Taken from the
    /// level above instead — `δ mod q^k = (δ mod q^{k'}) mod q^k` for the
    /// next `k' ≤ 2k`, the input below the square of the modulus — each is
    /// one Barrett reduction. Levels past the hint, if the lift goes on,
    /// double and fall back to the divisions.
    ///
    /// # Panics
    ///
    /// Panics if the lift is [bounded](Self::in_field_bounded) and `bits`
    /// is past its modulus.
    #[must_use]
    pub fn with_precision_hint(mut self, bits: usize) -> Self {
        let last = self.levels.last().expect("non-empty");
        let rungs = rungs_reaching(
            &self.levels[0].ring.modulus,
            last.exponent,
            &last.ring.modulus,
            bits,
        );
        if rungs.is_empty() {
            return self;
        }
        let workers = self.workers;
        let integers = self.integers();
        let levels = levels_at(rungs, workers, |ring| {
            crate::parallel::map_ordered(integers, workers, |_, d| ring.reduce_signed(d))
        });
        self.levels.extend(levels);
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

    /// `δ` over `ℤ`, which a level past those prepared is reduced from.
    ///
    /// # Panics
    ///
    /// Panics if the lift is bounded: it was given `δ` modulo its last
    /// modulus, and has no more of `δ` than that.
    fn integers(&self) -> &[BigInt] {
        self.delta
            .as_deref()
            .expect("a bounded lift goes no farther than its modulus")
    }

    /// Make sure `levels[j]` exists, by squaring upward from the last one.
    fn ensure_level(&mut self, j: usize) {
        while self.levels.len() <= j {
            let last = self.levels.last().expect("non-empty");
            let rung = Rung {
                exponent: 2 * last.exponent,
                modulus: last.ring.modulus.square(),
                rise: None,
            };
            let level = Level::from_integers(rung, self.integers());
            self.levels.push(level);
        }
    }

    /// Lift to the next level: `q^k → q^{k'}`, `k'` the next planned
    /// precision or else `2k`, with the root correct to the new precision.
    ///
    /// # Panics
    ///
    /// Panics if the lift is [bounded](Self::in_field_bounded) and already
    /// at its modulus.
    pub fn double(&mut self) {
        self.refine_inverse();
        self.ensure_level(self.level + 1);

        // β_k² − δ over ℤ, reduced by f, then modulo q^{k'}; each
        // coefficient is a multiple of q^k, and the quotient is what the
        // correction is built from.
        let squared = self.reduce_by_f(&self.square(&self.root));
        let previous = &self.levels[self.level];
        let next = &self.levels[self.level + 1];
        let pairs: Vec<(&BigInt, &BigUint)> = squared.iter().zip(&next.delta).collect();
        let error = self.per_coefficient(&pairs, |&(s, d)| {
            let residue = next
                .ring
                .reduce_signed(&s.sub(&BigInt::from_biguint(d.clone())));
            let (quotient, remainder) = previous.ring.div_rem(&residue);
            debug_assert!(
                remainder.is_zero(),
                "β² − δ must vanish modulo q^k: the seed was not a root"
            );
            quotient
        });

        // c = e · u mod (f, q^{k'−k}), then β ← β − q^k · c mod q^{k'}.
        let correction = self.reduce_by_f(&self.multiply(&error, &self.inverse));
        let rise = next.rise.as_ref().unwrap_or(&previous.ring);
        let pairs: Vec<(&BigUint, &BigInt)> = self.root.iter().zip(&correction).collect();
        let root = self.per_coefficient(&pairs, |&(b, c)| {
            let step = previous.ring.modulus.mul(&rise.reduce_signed(c));
            if *b >= step {
                b.sub(&step)
            } else {
                b.add(&next.ring.modulus).sub(&step)
            }
        });
        self.root = root;
        self.level += 1;
    }

    /// Bring `u ≡ (2β)⁻¹` from the level below, `q^j`, up to the root's
    /// current `q^k`, `k ≤ 2j`, by the same half-width Newton step:
    /// `2βu − 1` is a multiple of `q^j`, and
    /// `u ← u − q^j · ((2βu − 1)/q^j · u mod q^{k−j})`.
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
        let indexed: Vec<(usize, &BigInt)> = product.iter().enumerate().collect();
        let error = self.per_coefficient(&indexed, |&(i, p)| {
            let p = if i == 0 {
                p.sub(&BigInt::one())
            } else {
                p.clone()
            };
            let residue = full.ring.reduce_signed(&p);
            let (quotient, remainder) = half.ring.div_rem(&residue);
            debug_assert!(
                remainder.is_zero(),
                "2βu − 1 must vanish modulo the reciprocal's modulus"
            );
            quotient
        });
        let correction = self.reduce_by_f(&self.multiply(&error, &self.inverse));
        let rise = full.rise.as_ref().unwrap_or(&half.ring);
        let pairs: Vec<(&BigUint, &BigInt)> = self.inverse.iter().zip(&correction).collect();
        let inverse = self.per_coefficient(&pairs, |&(u, c)| {
            let step = half.ring.modulus.mul(&rise.reduce_signed(c));
            if *u >= step {
                u.sub(&step)
            } else {
                u.add(&full.ring.modulus).sub(&step)
            }
        });
        self.inverse = inverse;
        self.inverse_level = self.level;
    }

    /// `convolution` reduced by `f` over `ℤ`, padded to `deg f` signed
    /// coefficients.
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
        self.per_coefficient(pairs, |&pair| task(pair))
    }

    /// `task` over `items`, a level's coefficients or their products, in
    /// order: on the calling thread unless the budget and the width both
    /// justify threads. Each item is a product or a reduction at the
    /// level's width, and they are independent.
    fn per_coefficient<T, R, F>(&self, items: &[T], task: F) -> Vec<R>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> R + Sync,
    {
        let wide = self.modulus().bits() >= PARALLEL_PRODUCT_MIN_BITS;
        let threads = if wide { self.workers } else { 1 };
        crate::parallel::map_ordered(items, threads, |_, item| task(item))
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

    /// Where the search for a seed prime starts: the smallest prime above
    /// 10⁶, so every field here has a modulus of about 20 bits and each
    /// doubling of the lift doubles that.
    const SEED_PRIME_SEARCH_START: u64 = 1_000_003;

    /// The prime that search finds for `x⁴ + 5x³ + 11x² − 3x + 7`, the
    /// polynomial the wide tests share; asserted where it is found, since
    /// [`PROBE_BITS_DEFAULT`] is derived from it.
    const WIDE_F_SEED_PRIME: u64 = 1_000_033;

    fn poly(coefficients: &[i64]) -> PolyZ {
        PolyZ::new(coefficients.iter().map(|&c| BigInt::from_i64(c)).collect())
    }

    /// `bits` random bits as an integer, one bit per step of Marsaglia's
    /// xorshift64 (*Xorshift RNGs*, J. Stat. Software 8(14), 2003; the
    /// 13/7/17 triple is his), so the same seed gives the same integer at
    /// every width.
    fn random_bits(seed: &mut u64, bits: usize) -> BigUint {
        let mut value = BigUint::zero();
        for bit in 0..bits {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            if *seed & 1 == 1 {
                value.set_bit(bit);
            }
        }
        value
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
        // The lift is exact once the modulus exceeds twice the largest
        // coefficient, 2·5: at q² = 49, after one doubling. Six reach q³²,
        // so the `expect` below cannot fire on a correct lift.
        const MAX_DOUBLINGS: usize = 6;
        let mut converged = None;
        for _ in 0..MAX_DOUBLINGS {
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
        let prime = irreducible_prime(&f, SEED_PRIME_SEARCH_START);
        let seed = field_root(&f, &delta, prime);
        let mut lift = HenselSquareRoot::in_field(&f, &delta, &seed).expect("a square");
        // The reciprocal is not refined on the first doubling, where it
        // already sits at the root's level, and is refined on every one
        // after; three is the least that refines it twice, the second time
        // from an already refined value. The modulus reached, q⁸, is about
        // 160 bits, past twice the 30-bit coefficients.
        const DOUBLINGS: usize = 3;
        for _ in 0..DOUBLINGS {
            lift.double();
        }
        let modulus = lift.modulus().clone();
        assert_eq!(modulus, BigUint::from_u64(prime).pow_u64(1 << DOUBLINGS));
        let f_mod = PolyMod::from_poly_z(&f, &modulus);
        let root = lift.root();
        let delta_mod = PolyMod::from_poly_z(&delta, &modulus).rem(&f_mod);
        assert_eq!(root.mul(&root).rem(&f_mod), delta_mod);
        let candidate = lift.symmetric_lift();
        assert!(candidate == beta || candidate.add(&beta).is_zero());
    }

    /// A lift told its target ends at that power of the prime, through
    /// steps that do not all double, with every level a root there; a
    /// bounded lift to it is the same lift; and past it the lift goes on to
    /// the integer root.
    #[test]
    fn a_planned_lift_ends_at_its_target() {
        // q⁶ and q¹³ are reached by steps of 2k − 1: 3 → 5 → 6 and 4 → 7 → 13.
        const TARGETS: u64 = 13;
        let f = poly(&[7, -3, 11, 5, 1]);
        let beta = poly(&[-123_456_789, 987_654_321, -555_555_555, 42]);
        let delta = beta.mul(&beta).rem_monic(&f);
        let prime = irreducible_prime(&f, SEED_PRIME_SEARCH_START);
        let seed = field_root(&f, &delta, prime);
        let prime = BigUint::from_u64(prime);
        for target in 1..=TARGETS {
            let modulus = prime.pow_u64(target);
            assert_eq!(
                HenselSquareRoot::modulus_reaching(&prime, modulus.bits()),
                modulus
            );
            let mut lift = HenselSquareRoot::in_field(&f, &delta, &seed)
                .expect("a square")
                .with_precision_hint(modulus.bits());
            let residues = PolyMod::from_poly_z(&delta, &modulus);
            let mut bounded =
                HenselSquareRoot::in_field_bounded(&f, residues.coefficients(), &modulus, &seed)
                    .expect("a square");
            while *lift.modulus() < modulus {
                lift.double();
                bounded.double();
                assert_eq!(bounded.modulus(), lift.modulus());
                assert_eq!(bounded.coefficients(), lift.coefficients());
                let f_mod = PolyMod::from_poly_z(&f, lift.modulus());
                let root = lift.root();
                let delta_mod = PolyMod::from_poly_z(&delta, lift.modulus()).rem(&f_mod);
                assert_eq!(
                    root.mul(&root).rem(&f_mod),
                    delta_mod,
                    "a root at every level"
                );
            }
            assert_eq!(*lift.modulus(), modulus, "the last level is the target");
            while lift.modulus().bits() <= 2 * 31 + 1 {
                lift.double();
            }
            let candidate = lift.symmetric_lift();
            assert!(candidate == beta || candidate.add(&beta).is_zero());
        }
    }

    /// A lift given `δ` modulo its last modulus is the lift given `δ`, level
    /// for level, negative coefficients of `δ` included.
    #[test]
    fn a_bounded_lift_is_the_lift_to_its_modulus() {
        let f = poly(&[7, -3, 11, 5, 1]);
        let beta = poly(&[-123_456_789, 987_654_321, -555_555_555, 42]);
        let delta = beta.mul(&beta).rem_monic(&f);
        assert!(
            delta
                .coefficients()
                .iter()
                .any(|c| c.sign() == Sign::Negative),
            "the case has no negative coefficient to reduce"
        );
        let prime = irreducible_prime(&f, SEED_PRIME_SEARCH_START);
        let seed = field_root(&f, &delta, prime);
        // Past twice the 30-bit coefficients, and reached at q⁸, 160 bits.
        const BITS: usize = 150;
        const DOUBLINGS: u64 = 3;
        let prime = BigUint::from_u64(prime);
        let modulus = HenselSquareRoot::modulus_reaching(&prime, BITS);
        assert_eq!(modulus, prime.pow_u64(1 << DOUBLINGS));
        assert_eq!(HenselSquareRoot::modulus_reaching(&prime, 1), prime);

        let residues = PolyMod::from_poly_z(&delta, &modulus);
        let mut bounded =
            HenselSquareRoot::in_field_bounded(&f, residues.coefficients(), &modulus, &seed)
                .expect("a square");
        let mut whole = HenselSquareRoot::in_field(&f, &delta, &seed).expect("a square");
        assert_eq!(bounded.coefficients(), whole.coefficients());
        for _ in 0..DOUBLINGS {
            bounded.double();
            whole.double();
            assert_eq!(bounded.modulus(), whole.modulus());
            assert_eq!(bounded.coefficients(), whole.coefficients());
        }
        assert_eq!(*bounded.modulus(), modulus);
        let candidate = bounded.symmetric_lift();
        assert!(candidate == beta || candidate.add(&beta).is_zero());

        // A seed that is no root is refused, as it is with `δ` in hand.
        let wrong = PolyMod::new(vec![BigUint::from_u64(2)], &prime);
        assert!(
            HenselSquareRoot::in_field_bounded(&f, residues.coefficients(), &modulus, &wrong)
                .is_none()
        );
    }

    /// The case of the bounded lift's refusals: `x² + 1` modulo 7, `δ` the
    /// square of `5 + 3x`, reduced modulo `modulus`.
    fn bounded_case(modulus: &BigUint) -> (PolyZ, Vec<BigUint>, PolyMod) {
        let f = poly(&[1, 0, 1]);
        let beta = poly(&[5, 3]);
        let delta = beta.mul(&beta).rem_monic(&f);
        let seed = field_root(&f, &delta, 7);
        let residues = PolyMod::from_poly_z(&delta, modulus);
        (f, residues.coefficients().to_vec(), seed)
    }

    #[test]
    #[should_panic(expected = "a bounded lift goes no farther than its modulus")]
    fn a_bounded_lift_does_not_pass_its_modulus() {
        let modulus = BigUint::from_u64(49);
        let (f, residues, seed) = bounded_case(&modulus);
        let mut lift =
            HenselSquareRoot::in_field_bounded(&f, &residues, &modulus, &seed).expect("a square");
        lift.double();
        assert_eq!(*lift.modulus(), modulus);
        lift.double();
    }

    #[test]
    #[should_panic(expected = "the modulus is a power of the prime")]
    fn a_bounded_lift_wants_a_power_of_its_prime() {
        // 300 lies between 7² and 7³.
        let modulus = BigUint::from_u64(300);
        let (f, residues, seed) = bounded_case(&modulus);
        let _ = HenselSquareRoot::in_field_bounded(&f, &residues, &modulus, &seed);
    }

    /// Wide enough that the Barrett path and the threaded products both
    /// run: coefficients near 200 kbit take the modulus past the Newton
    /// division threshold before the lift is exact.
    #[test]
    fn the_wide_lift_uses_barrett_and_threads() {
        // Width of the root's coefficients. The lift is exact once the
        // modulus exceeds twice the largest.
        const COEFFICIENT_BITS: usize = 200_000;
        // A coefficient's width and one bit for the factor of two: the
        // least power of the 20-bit seed prime this wide, q^10035 of
        // 200,015 bits and 3,126 limbs against the Newton threshold's
        // 3,072 (the assertion at the end checks), is the level the lift
        // ends on, so every level it takes is prepared from above and
        // none falls back to the division per coefficient the hint exists
        // to avoid.
        const HINT_BITS: usize = COEFFICIENT_BITS + 2;
        // Levels before the lift is called unconverged. It needs 14, from
        // 10,035 halved and rounded up to 1, and the loop checks before
        // each one, so 16 leaves one to spare; the assertion, not the
        // bound, reports a failure.
        const MAX_DOUBLINGS: usize = 16;
        // More than one thread, so the threaded path runs. A policy.
        const WORKERS: usize = 4;
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x9e37_79b9_7f4a_7c15;

        let f = poly(&[7, -3, 11, 5, 1]);
        let mut seed = SEED;
        let mut wide = || random_bits(&mut seed, COEFFICIENT_BITS);
        let beta = PolyZ::new(vec![
            BigInt::from_parts(Sign::Negative, wide()),
            BigInt::from_biguint(wide()),
            BigInt::from_parts(Sign::Negative, wide()),
            BigInt::from_biguint(wide()),
        ]);
        let delta = beta.mul(&beta).rem_monic(&f);
        let prime = irreducible_prime(&f, SEED_PRIME_SEARCH_START);
        assert_eq!(
            prime, WIDE_F_SEED_PRIME,
            "the level widths in these tests are derived from this prime"
        );
        let seed = field_root(&f, &delta, prime);
        let mut lift = HenselSquareRoot::in_field(&f, &delta, &seed)
            .expect("a square")
            .with_workers(WORKERS)
            .with_precision_hint(HINT_BITS);
        let mut converged = false;
        for _ in 0..MAX_DOUBLINGS {
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

    /// Timing probe at wide precision: a root with coefficients of half
    /// `RUMP_HENSEL_PROBE_BITS` (default [`PROBE_BITS_DEFAULT`]) bits, the
    /// wide levels reported. Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "timing probe for the lift at NFS width; run with --ignored"]
    fn lift_timing_probe() {
        use std::time::Instant;
        // The width factoring's c105 square root lifts at: 1.3 Mbit
        // coefficients, 20,410 limbs, the width bigint.rs's transform gate
        // is tuned for. It is exactly the width of the lift's last level
        // for the seed prime: ⌈2¹⁶ · log₂ 1,000,033⌉ = 1,306,239 bits.
        const PROBE_BITS_DEFAULT: usize = 1_306_239;
        // Bits added to half the probe width for the root's coefficients.
        // The lift ends on the first level whose modulus exceeds twice the
        // largest coefficient; the level below the target is 653,120 bits,
        // one wider than half the target, so a coefficient of exactly half
        // the width could end the lift a level early on a lucky draw. The
        // margin puts the coefficients past that level for certain.
        const COEFFICIENT_MARGIN_BITS: usize = 200;
        // Levels narrower than this are not reported: the wide ones are
        // what the probe is for, and the narrow ones would only pad the
        // output. A policy.
        const REPORT_MIN_BITS: usize = 100_000;
        // Worker budgets swept: serial, and two sizes of pool. A policy.
        const WORKER_COUNTS: [usize; 3] = [1, 4, 16];
        // Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x9e37_79b9_7f4a_7c15;

        let bits: usize = std::env::var("RUMP_HENSEL_PROBE_BITS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(PROBE_BITS_DEFAULT);
        let f = poly(&[7, -3, 11, 5, 1]);
        let mut seed = SEED;
        let mut wide = || random_bits(&mut seed, bits / 2 + COEFFICIENT_MARGIN_BITS);
        let beta = PolyZ::new((0..4).map(|_| BigInt::from_biguint(wide())).collect());
        let delta = beta.mul(&beta).rem_monic(&f);
        let seed = field_root(&f, &delta, irreducible_prime(&f, SEED_PRIME_SEARCH_START));
        for workers in WORKER_COUNTS {
            let start = Instant::now();
            let mut lift = HenselSquareRoot::in_field(&f, &delta, &seed)
                .expect("a square")
                .with_workers(workers)
                .with_precision_hint(bits);
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
                if lift.modulus().bits() > REPORT_MIN_BITS {
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
