//! Arithmetic in binary extension fields GF(2^m).
//!
//! Elements of GF(2^m) are represented as [`BigUint`] values whose bit
//! pattern encodes a polynomial over GF(2): bit `i` set means the coefficient
//! of `x^i` is 1. A [`Gf2m`](crate::finite_field::Gf2m) context holds the field modulus — an irreducible
//! polynomial of degree `m`, stored with the same bit-pattern convention —
//! and derives the degree from it, so the two can never disagree.
//!
//! Binary fields underlie the FIPS binary elliptic curves, and equally the
//! algebra of error-correcting codes, CRCs, and LFSR analysis; this module
//! supplies the arithmetic and takes no position on the application.
//!
//! ## The irreducibility contract
//!
//! [`Gf2m::new`] accepts any polynomial of degree at least one and does not
//! test it: irreducibility is the caller's obligation, cheap to discharge for
//! the fixed published moduli of the standard curves and testable with
//! [`Gf2m::is_irreducible`] when the polynomial arrives from elsewhere. What
//! the type then holds is `GF(2)[x]/(f)`, which is a field only when `f` is
//! irreducible; under a reducible `f` it is a ring with zero divisors, and
//! the field-theoretic guarantees fail one by one:
//!
//! - [`Gf2m::inverse`] returns `None` for the zero divisors, which are
//!   non-zero elements.
//! - Squaring is no longer injective, so [`Gf2m::sqrt`] returns a value that
//!   need not square back to its argument.
//! - The Frobenius sum that defines the trace need not land in GF(2), so
//!   [`Gf2m::trace`] has no meaningful value to report; [`Gf2m::half_trace`]
//!   need not solve its quadratic, and [`Gf2m::solve_quadratic`] answers
//!   `None` rather than a root it cannot verify.
//!
//! Every routine still terminates on such a ring — none loops, none indexes
//! out of range — and only [`Gf2m::inverse`] and [`Gf2m::solve_quadratic`]
//! have a defined answer there. Two break totality deliberately, in every
//! build: [`Gf2m::trace`] panics when the Frobenius sum leaves GF(2), because
//! the alternative is to report 0 and tell the caller a quadratic is solvable
//! when it is not, and [`Gf2m::half_trace`] panics on an even degree, where it
//! is not a solver at all.
//!
//! ## Algorithm notes
//!
//! - **Addition** is XOR (no reduction needed: XOR of two polynomials of
//!   degree < m has degree < m).
//! - **Multiplication** uses the left-to-right comb method with 4-bit windows
//!   (Algorithm 2.36 of Hankerson, Menezes, Vanstone — *Guide to ECC*),
//!   followed by polynomial reduction modulo the field polynomial.
//! - **Squaring** spreads each coefficient to twice its index (Algorithm 2.39
//!   of the same), the whole of `(Σ aᵢxⁱ)² = Σ aᵢx²ⁱ` in characteristic 2.
//! - **Inversion** uses the extended Euclidean algorithm for polynomials over
//!   GF(2) (Algorithm 2.48 of Hankerson, Menezes, Vanstone — *Guide to ECC*).
//! - **Trace** is the Frobenius sum Tr(c) = Σ_{i=0}^{m−1} c^{2^i} (IEEE Std
//!   1363-2000, Annex A.4.5), a GF(2)-linear map onto the prime subfield
//!   whose value decides solvability of z² + z = c.
//! - **Half-trace** computes HT(c) = Σ c^{2^{2i}} (i = 0...(m−1)/2), which
//!   is a root of z² + z = c for any c in GF(2^m) with Tr(c) = 0. This
//!   solves the quadratic needed for compressed-point decompression on
//!   binary curves with odd m. [`Gf2m::solve_quadratic`] wraps it with the
//!   trace test and carries an even-degree construction as well.

use crate::bigint::BigUint;

/// Bits per window of the comb multiplication, `w` in Hankerson, Menezes,
/// Vanstone — *Guide to ECC*, Algorithm 2.36, which takes `w = 4`. The
/// table holds `2^w` multiples of `b`, each a limb longer than `b`, and the
/// sweep over `a` makes `64 / w` passes; widening the window doubles the
/// table for every bit it saves from the passes, and four is where the
/// algorithm as published sets that balance.
const COMB_WINDOW_BITS: usize = 4;

/// Entries of the comb table: one multiple `u(x)·b(x)` per `w`-bit `u`.
const COMB_TABLE_ENTRIES: usize = 1 << COMB_WINDOW_BITS;

/// Selects one window from a limb.
const COMB_WINDOW_MASK: u64 = COMB_TABLE_ENTRIES as u64 - 1;

/// Windows in a limb: the passes the sweep makes.
const COMB_WINDOWS_PER_LIMB: usize = u64::BITS as usize / COMB_WINDOW_BITS;

/// A binary extension field GF(2^m), defined by its irreducible polynomial.
///
/// The degree and reduction taps are derived from the polynomial by
/// [`Gf2m::new`], the only constructor, so they cannot disagree with it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gf2m {
    poly: BigUint,
    degree: usize,
    // Bit positions of the polynomial below the leading term, ascending: the
    // reduction taps. x^m ≡ Σ x^t over the taps, since f(x) = x^m + Σ x^t is
    // zero in the quotient. Standard moduli are trinomials and pentanomials,
    // and reduction costs one shifted XOR per tap per word.
    taps: Vec<usize>,
}

impl Gf2m {
    /// Build the field GF(2^m) from its modulus polynomial, or `None` when
    /// the polynomial has degree below one (a constant defines no field).
    ///
    /// The degree is `poly.bits() - 1`; the reduction taps are read off the
    /// polynomial once, here.
    ///
    /// Irreducibility is *not* checked; see the module-level contract, and
    /// [`Self::is_irreducible`] for the test to run when the polynomial is not
    /// a published constant.
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

    /// The field's modulus polynomial, in the module's bit-pattern encoding
    /// (bit `i` is the coefficient of `xⁱ`).
    ///
    /// As a field element this value is zero — it is the polynomial the
    /// quotient divides by — so [`Self::inverse`] returns `None` for it.
    #[must_use]
    pub fn modulus(&self) -> &BigUint {
        &self.poly
    }

    /// The field degree `m`: the degree of the modulus polynomial, so field
    /// elements are exactly the bit patterns below `2^m`.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Add two field elements: XOR, with no reduction needed.
    ///
    /// In characteristic 2 this is also subtraction, so there is no separate
    /// `sub`. An associated function because addition does not depend on the
    /// field polynomial.
    #[inline]
    #[must_use]
    pub fn add(a: &BigUint, b: &BigUint) -> BigUint {
        let mut out = a.clone();
        out.bitxor_assign(b);
        out
    }

    /// Multiply two field elements modulo the field polynomial.
    ///
    /// Left-to-right comb multiplication with 4-bit windows (López & Dahab,
    /// *High-speed software multiplication in F_{2^m}*, INDOCRYPT 2000, LNCS
    /// 1977, 203–212; as Hankerson, Menezes, Vanstone — *Guide to ECC*,
    /// Algorithm 2.36, present it): precompute the sixteen products
    /// `u(x)·b(x)` for 4-bit `u`, then sweep `a` one window at a time,
    /// shifting the accumulator four bits between sweeps.
    ///
    /// A bit-serial shift-and-XOR pays one accumulator shift per bit of `a`;
    /// the comb costs at most sixteen passes of `na · (nb + 1)` word XORs and
    /// fifteen accumulator shifts, for operands of `na` and `nb` limbs. Still
    /// quadratic; the win is the constant.
    ///
    /// The double-width product is reduced tap-wise by `reduce_limbs`.
    /// Neither operand need be reduced on entry; the result always is.
    #[must_use]
    pub fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        if a.is_zero() || b.is_zero() {
            return BigUint::zero();
        }

        let a_limbs = a.limbs();
        let b_limbs = b.limbs();
        // One spare limb per entry: u(x) has degree below the window width,
        // so u(x)·b(x) exceeds b by fewer bits than that and can never
        // overflow the extra word. That headroom is what lets the doubling
        // below discard its final carry-out.
        let stride = b_limbs.len() + 1;

        // table[u] = u(x) · b(x) for every window pattern u, built by the
        // recurrence 2u ↦ (u·b) · x and 2u+1 ↦ (2u·b) + b, each entry from
        // one already written: one word pass per entry past the first two,
        // no multiplications. table[0] stays zero and is skipped at use;
        // table[1] is b itself.
        let mut table = vec![0u64; COMB_TABLE_ENTRIES * stride];
        table[stride..stride + b_limbs.len()].copy_from_slice(b_limbs);
        for u in 2..COMB_TABLE_ENTRIES {
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

        // Sized to hold a full `stride`-word table entry placed at the top
        // limb of `a`, with one word of slack. The accumulator never exceeds
        // the finished product: after window `k` it equals the product with
        // every remaining window shifted out, so it is bounded by
        // `deg(a) + deg(b) − w·k`. Hence the window-wide shift below can
        // never carry a bit off the top word, and dropping its final carry
        // is safe.
        let mut product = vec![0u64; a_limbs.len() + stride];
        // Windows most-significant first: absorb window `window` of every
        // limb of `a`, then shift the accumulator left one window to make
        // room for the next, which is Horner's rule in radix 2^w.
        for window in (0..COMB_WINDOWS_PER_LIMB).rev() {
            for (j, &limb) in a_limbs.iter().enumerate() {
                let u = ((limb >> (COMB_WINDOW_BITS * window)) & COMB_WINDOW_MASK) as usize;
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
                    let next = *limb >> (u64::BITS as usize - COMB_WINDOW_BITS);
                    *limb = (*limb << COMB_WINDOW_BITS) | carry;
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
    /// `(Σ aᵢxⁱ)² = Σ aᵢx²ⁱ`, because every cross term appears twice and
    /// cancels in characteristic 2. Squaring is a fixed rearrangement of bits:
    /// one lookup in a 256-entry byte table per input byte turns each 32-bit
    /// half into a 64-bit limb, then the doubled buffer is reduced tap-wise.
    /// Linear in the word count where [`Self::mul`] is quadratic, which is why
    /// [`Self::trace`], [`Self::sqrt`], [`Self::half_trace`], and
    /// [`Self::is_irreducible`] are built from squaring chains.
    ///
    /// `a` need not be reduced on entry; linearity makes the result the same
    /// for any representative of its class, and the result is reduced.
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

    /// Raise a field element to a non-negative integer power.
    ///
    /// Left-to-right binary square-and-multiply (Knuth, *TAOCP* vol. 2,
    /// §4.6.3), seeded with the base for the leading exponent bit: one
    /// squaring per remaining bit and one multiplication by the base per set
    /// bit among them.
    ///
    /// The exponent is not reduced modulo the group order `2^m − 1`, so a wide
    /// exponent costs its full bit length. `pow(a, 0)` is one for every `a`,
    /// including zero, matching [`mod_pow`](crate::modular::mod_pow). The base
    /// is reduced once on entry.
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

    /// Divide one field element by another: `a · b⁻¹`, or `None` when `b` is
    /// not a unit.
    ///
    /// Inversion then multiplication; [`Self::inverse`] dominates the cost.
    /// `None` means what it means there: `b` is zero, a non-canonical
    /// representative of zero, or a zero divisor under a reducible modulus.
    #[must_use]
    pub fn div(&self, a: &BigUint, b: &BigUint) -> Option<BigUint> {
        Some(self.mul(a, &self.inverse(b)?))
    }

    /// The unique square root of `a`.
    ///
    /// Squaring is the Frobenius automorphism of GF(2^m), so every element has
    /// exactly one square root. Since `a^{2^m} = a`, the root is
    /// `a^{2^{m−1}}`: `m − 1` applications of [`Self::square`], no
    /// multiplications.
    ///
    /// Under a reducible modulus squaring is not injective (in `GF(2)[x]/(x²)`
    /// both 0 and `x` square to 0), and the value returned need not square
    /// back to `a`.
    #[must_use]
    pub fn sqrt(&self, a: &BigUint) -> BigUint {
        let mut root = self.reduce(a.clone());
        for _ in 1..self.degree {
            root = self.square(&root);
        }
        root
    }

    /// Solve `z² + z = c` at any field degree, returning one root, or `None`
    /// when none exists.
    ///
    /// The map `z ↦ z² + z` is GF(2)-linear with kernel `{0, 1}`, so its image
    /// is the index-two subspace `Tr(c) = 0`: a solution exists exactly when
    /// the trace vanishes, and when it does there are exactly two roots, `z`
    /// and `z + 1`. This function returns one; add one for the other.
    ///
    /// Two constructions, chosen on the parity of `m`:
    ///
    /// - **Odd `m`** — [`Self::half_trace`], `HT(c) = Σ_{i=0}^{(m−1)/2}
    ///   c^{2^{2i}}`, which satisfies `HT(c)² + HT(c) = c + Tr(c)` and so
    ///   solves the equation whenever the trace is zero.
    /// - **Even `m`** — the construction of IEEE Std 1363-2000, Annex A.4.7:
    ///   with any `δ` of trace one, `z = Σ_{i=0}^{m−2} sᵢ δ^{2^i}` where
    ///   `sᵢ = Σ_{j=i+1}^{m−1} c^{2^j}`. Because `Tr(c) = 0`, `s₀ = c`, and
    ///   `sᵢ₊₁ = sᵢ + c^{2^{i+1}}`, so one squaring of `c` and one of `δ` per
    ///   term carries the sum over `m − 1` iterations.
    ///
    /// `δ` is found among the `m` monomials `x⁰ … x^{m−1}`: the trace is a
    /// non-zero GF(2)-linear functional, so it cannot vanish on a whole basis.
    /// The scan is bounded, so it terminates even when no `δ` exists.
    ///
    /// `None` covers three situations, and every `Some` is checked:
    ///
    /// - `Tr(c) = 1`: the equation has no root, the mathematical case.
    /// - The modulus is reducible and the Frobenius sum escapes GF(2), so no
    ///   trace is defined (`trace_bit` reports this); or it is reducible with
    ///   no trace-one monomial, so the even-degree search finds no `δ`.
    /// - The construction's output fails `z² + z = c`, which a reducible
    ///   modulus can cause on either branch. Both branches verify the root by
    ///   substitution, so a `Some` satisfies the equation whatever polynomial
    ///   was supplied.
    #[must_use]
    pub fn solve_quadratic(&self, c: &BigUint) -> Option<BigUint> {
        let c = self.reduce(c.clone());
        // A solution exists iff Tr(c) = 0. `trace_bit` is `None` when the
        // Frobenius sum escapes GF(2) under a reducible modulus.
        if self.trace_bit(&c)? != 0 {
            return None;
        }
        if self.degree % 2 == 1 {
            let z = self.half_trace(&c);
            // A reducible odd-degree modulus can violate the identity.
            return (Self::add(&self.square(&z), &z) == c).then_some(z);
        }

        // Even degree needs a trace-one δ; scan the m basis elements xⁱ.
        // `trace_bit` rather than `trace`, so a reducible ring gives `None`
        // instead of a panic.
        let mut delta = None;
        let mut basis = BigUint::one();
        for _ in 0..self.degree {
            if self.trace_bit(&basis) == Some(1) {
                delta = Some(basis);
                break;
            }
            basis = basis.add(&basis); // ×2: the next basis element
        }
        let delta = delta?;

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

        // A reducible modulus can pass the trace checks and still yield no
        // valid root.
        (Self::add(&self.square(&z), &z) == c).then_some(z)
    }

    /// The absolute trace `Tr(c) = Σ_{i=0}^{m−1} c^{2^i}`, always 0 or 1
    /// (IEEE Std 1363-2000, Annex A.4.5, "Trace").
    ///
    /// The sum is fixed by the Frobenius map, so it lies in GF(2); it costs
    /// `m − 1` squarings accumulated by XOR. `z² + z = c` has a solution
    /// exactly when `Tr(c) = 0`, the precondition of [`Self::half_trace`].
    /// `Tr(1) = m mod 2`. The argument is reduced first.
    ///
    /// Under a reducible modulus the sum need not land in GF(2);
    /// [`Self::solve_quadratic`] handles that case without panicking.
    ///
    /// # Panics
    ///
    /// When the Frobenius sum is neither 0 nor 1, in every build, which
    /// requires a reducible modulus. It panics rather than returning 0
    /// because 0 would claim `z² + z = c` is solvable.
    #[must_use]
    pub fn trace(&self, c: &BigUint) -> u8 {
        self.trace_bit(c)
            .expect("the Frobenius sum left GF(2): the field polynomial is reducible")
    }

    /// The trace as a prime-subfield bit, or `None` when the Frobenius sum is
    /// neither 0 nor 1.
    ///
    /// The sum is guaranteed to lie in GF(2) only for an irreducible modulus,
    /// which [`Gf2m::new`] does not check. `None` lets a caller such as
    /// [`Self::solve_quadratic`] recognise a reducible ring and bail.
    fn trace_bit(&self, c: &BigUint) -> Option<u8> {
        let mut power = self.reduce(c.clone());
        let mut acc = power.clone();
        for _ in 1..self.degree {
            power = self.square(&power);
            acc.bitxor_assign(&power);
        }
        if acc.is_zero() {
            Some(0)
        } else if acc.is_one() {
            Some(1)
        } else {
            None
        }
    }

    /// Test whether a GF(2) polynomial is irreducible (Rabin's test).
    ///
    /// [`Gf2m::new`] trusts its polynomial; run this when the polynomial is
    /// not a published constant. Rabin's criterion
    /// (*Probabilistic algorithms in finite fields*, 1980, here in its
    /// deterministic GF(2) form): `f` of degree `m` is irreducible iff
    /// `x^{2^m} ≡ x (mod f)` and, for every prime `q` dividing `m`,
    /// `gcd(x^{2^{m/q}} − x, f) = 1`. In characteristic 2 the subtraction is
    /// the XOR the code performs.
    ///
    /// Why those two conditions. `x^{2^k} − x` is the product of all monic
    /// irreducibles over GF(2) of degree dividing `k`. The first condition
    /// with `k = m` says every irreducible factor of `f` has degree dividing
    /// `m`; the gcd conditions rule out a factor of degree dividing some
    /// proper `m/q`. Together they leave only degree `m` itself, and `f` has
    /// degree `m`, so `f` is that single factor. Testing `m/q` for the prime
    /// divisors `q` suffices because every proper divisor of `m` divides some
    /// `m/q`.
    ///
    /// Mechanically, `x^{2^i} mod f` is `i` applications of [`Self::square`]
    /// to `x` — the Frobenius orbit. Sorting the checkpoints `m/q` ascending
    /// lets one forward pass of `m` squarings visit each in turn and finish at
    /// `x^{2^m}`, so the whole test costs `m` squarings plus one polynomial
    /// gcd per distinct prime divisor of `m`.
    ///
    /// Constants are not irreducible; degree 1 (`x` and `x + 1`) is, and
    /// returns early.
    ///
    /// # Panics
    ///
    /// Never: the internal `expect` is discharged by the degree test above it.
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
            // Euclid consumes a working copy of the modulus: one clone per
            // distinct prime divisor of m.
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
    /// `GF(2)[x]`, or `None` when the element is not a unit.
    ///
    /// Non-units are zero, any non-canonical representative of zero, and,
    /// under a reducible modulus, the zero divisors that share a factor with
    /// it.
    ///
    /// Algorithm 2.48 from Hankerson, Menezes, Vanstone — *Guide to ECC*: each
    /// step cancels the leading term of the higher-degree remainder by adding a
    /// shifted copy of the other (`u ^= v · x^{deg u − deg v}`), carrying only
    /// the cofactor of `a` alongside.
    ///
    /// Invariants, established by `u = a, v = poly, b = 1, c = 0`:
    ///
    /// - `u ≡ b · a` and `v ≡ c · a (mod poly)`, preserved because each
    ///   update is applied to `u`/`v` and `b`/`c` in lockstep.
    /// - `gcd(u, v) = gcd(a, poly)`, since adding `v · x^j` to `u` changes
    ///   neither side's common divisors.
    ///
    /// Termination: `u ^= v · x^j` strictly lowers `deg u`, and the swap keeps
    /// `deg u ≥ deg v`, so the loop is finite. It exits when `u = 1`, where
    /// the first invariant gives `b` as the inverse, or when `u = 0`, where
    /// `v` is a gcd other than 1 and `a` is not a unit.
    #[must_use]
    pub fn inverse(&self, a: &BigUint) -> Option<BigUint> {
        // Reduce first so the zero test catches non-canonical zeros such as
        // the field polynomial itself.
        let a = self.reduce(a.clone());
        if a.is_zero() {
            return None;
        }

        let mut u = a;
        let mut v = self.poly.clone();
        let mut b = BigUint::one();
        let mut c = BigUint::zero();

        // Loop until u = 1 (degree 0 polynomial over GF(2)).
        while !u.is_one() {
            // gcd(a, poly) = v ≠ 1: `a` is not a unit. A zero `u` would
            // never reach 1.
            if u.is_zero() {
                return None;
            }

            // Bit lengths, one more than the degrees; their difference is the
            // difference of the degrees either way, which is all that is used.
            let mut deg_u = u.bits(); // deg(u) + 1
            let mut deg_v = v.bits(); // deg(v) + 1

            // Ensure deg(u) >= deg(v). The cofactors and lengths travel with
            // their polynomials.
            if deg_u < deg_v {
                core::mem::swap(&mut u, &mut v);
                core::mem::swap(&mut b, &mut c);
                core::mem::swap(&mut deg_u, &mut deg_v);
            }

            // j = deg(u) - deg(v), non-negative after the swap above.
            let j = deg_u - deg_v;

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

    /// Compute the half-trace HT(c) = Σ_{i=0}^{(m−1)/2} c^{2^{2i}}, for odd
    /// field degree only.
    ///
    /// For any `c` with Tr(c) = 0, `z = HT(c)` solves `z² + z = c`, the
    /// quadratic behind compressed-point decompression on binary curves. In
    /// general `HT(c)² + HT(c) = c + Tr(c)`; the trace is not checked here,
    /// and for a trace-one argument the result solves `z² + z = c + 1`.
    /// [`Self::solve_quadratic`] tests the trace, works at every degree, and
    /// verifies its root.
    ///
    /// The loop squares twice and XORs, `(m − 1)/2` times, with no
    /// multiplications. The argument is reduced first.
    ///
    /// Why odd degree: squaring the sum shifts every exponent one Frobenius
    /// step, so `HT(c)² + HT(c) = Σ_j c^{2^j}` over the even and odd steps.
    /// For odd `m` that is `j = 0 … m`, and `c^{2^m} = c` gives `Tr(c) + c`.
    /// For even `m` it is `j = 0 … m−1`, which is `Tr(c)` alone and carries
    /// no information about `c`. Every FIPS 186-4 binary curve degree is odd.
    ///
    /// # Panics
    ///
    /// In every build, if the field degree is even. Use
    /// [`Self::solve_quadratic`] there.
    #[must_use]
    pub fn half_trace(&self, c: &BigUint) -> BigUint {
        assert!(
            self.degree % 2 == 1,
            "half_trace is defined for odd field degree; use solve_quadratic for even degrees"
        );
        // HT(c) = c^{2^0} + c^{2^2} + c^{2^4} + ... + c^{2^{degree-1}}
        // Starting from power = c, square twice per iteration to advance by
        // 2 exponent steps: c → c^4 → c^{16} → ...
        let c = self.reduce(c.clone());
        let mut t = c.clone(); // accumulator starts at c^{2^0}
        let mut power = c; // current term

        for _ in 0..(self.degree - 1) / 2 {
            // Advance power from c^{2^{2i}} to c^{2^{2(i+1)}} = c^{2^{2i+2}}.
            power = self.square(&self.square(&power));
            t.bitxor_assign(&power);
        }

        t
    }

    /// Reduce `a` modulo the field polynomial, returning the canonical
    /// representative of its class — the unique value of degree below `m`.
    ///
    /// An argument already in canonical form, the common case, is returned
    /// as is; only an unreduced representative pays for the fold.
    fn reduce(&self, a: BigUint) -> BigUint {
        if a.bits() <= self.degree {
            return a;
        }
        // `a` is owned, so the fold works on its limb buffer in place.
        let mut limbs = a.into_limbs();
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
    /// polynomial's leading term would have done. This is the tap identity
    /// `x^m ≡ Σ x^t` applied to 64 coefficients at once, sound because the
    /// identity is GF(2)-linear. Cost is one shifted XOR per tap per word:
    /// two for a trinomial, four for a pentanomial.
    ///
    /// Termination. A bit at position `p ≥ degree` is folded to `p − degree +
    /// t`, and every tap satisfies `t ≤ degree − 1`, so each folded bit lands
    /// strictly below the bit it came from — at most `p − 1`. The buffer's
    /// highest set bit therefore falls by at least one per pass, and the outer
    /// loop is finite. It is a loop rather than a single pass because a tap
    /// close to the degree re-raises bits above the degree inside the boundary
    /// word, which must then be folded again.
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
                // Wholly above the boundary: the entire word is excess, and
                // its bit 0 represents x^(top_word · 64), which the tap
                // identity sends to x^(top_word · 64 − degree + t).
                let w = buf[top_word];
                buf[top_word] = 0;
                (w, crate::bigint::bit_span(top_word, 64) - self.degree)
            } else {
                // The boundary word itself: only the bits at and above
                // `boundary_bit` are excess, and bit `boundary_bit` is x^degree
                // exactly, so the taps apply with no extra shift. A
                // `boundary_bit` of zero yields a mask of zero, which is
                // correct — the whole word is then excess.
                let w = buf[top_word] >> boundary_bit;
                buf[top_word] &= (1u64 << boundary_bit) - 1;
                (w, 0)
            };

            for &t in &self.taps {
                xor_shifted_word(buf, excess, base_shift + t);
            }
        }

        // Postcondition: no trailing zero words. Callers that pass the buffer
        // to `BigUint::from_limbs` get this twice; the function does not rely
        // on them.
        while buf.last() == Some(&0) {
            buf.pop();
        }
    }
}

/// Interleave a zero bit after every bit of `half` — the squaring map on
/// one 32-bit word, via an 8-bit spread table.
///
/// The table is built in a `const` block, so the 256 entries are materialized
/// at compile time and the runtime cost is four lookups and three shifted ORs
/// per 32-bit half.
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

/// Significant bits of a little-endian limb buffer, scanning from the top for
/// the first non-zero word.
///
/// Unlike [`BigUint::bits`] this tolerates trailing zero words, which is what
/// `Gf2m::reduce_limbs` needs: it clears the top word in place and must
/// re-measure a buffer that is momentarily denormalized. Zero-length and
/// all-zero buffers give 0.
fn limbs_bits(buf: &[u64]) -> usize {
    for (i, &limb) in buf.iter().enumerate().rev() {
        if limb != 0 {
            return crate::bigint::bit_span(i, 64) + (64 - limb.leading_zeros() as usize);
        }
    }
    0
}

/// XOR `word` into the buffer at the given bit offset, straddling the two
/// limbs the offset spans.
///
/// The `shift > 0` test is not an optimization: `word >> 64` overflows the
/// shift (a panic in a debug build), so the aligned case skips the high half.
/// The `high != 0` test keeps the write in bounds.
///
/// Bounds. The only caller, `Gf2m::reduce_limbs`, must guarantee
/// `index < buf.len()`, which it does because every offset it passes is
/// strictly below the bit position of the word being folded away. The write
/// to `index + 1` can reach one past that word, and is in bounds for a
/// different reason in each of the caller's two cases: folding a word wholly
/// above the boundary leaves `index + 1` at most the source word's own index,
/// while folding the boundary word itself leaves `high` zero whenever
/// `index + 1` would run off the end, because the excess there is narrower
/// than 64 bits by exactly the amount the shift could carry out. Both are
/// consequences of `t ≤ degree − 1` over the taps.
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
///
/// The inner loop is one Euclidean division of `a` by `b`, performed as
/// repeated cancellation of the leading term (`a ^= b · x^{deg a − deg b}`)
/// without ever forming the quotient, which is not needed. It terminates
/// because each cancellation strictly lowers `deg a`. The swap makes the
/// remainder the new divisor, ending with the gcd in `a` and zero in `b`.
///
/// A zero argument is handled by the same code without a special case:
/// `gcd(0, b) = b` falls out of the first swap. The result is monic by
/// construction — every non-zero polynomial over GF(2) is.
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

/// Distinct prime divisors of `n`, ascending, by trial division to `√n`.
///
/// `n` is a field degree (571 for the widest FIPS binary curve), so trial
/// division suffices. Each divisor found is divided out completely; whatever
/// survives above 1 is the single prime factor larger than `√n`.
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

    /// GF(2^163) under x^163 + x^7 + x^6 + x^3 + 1, the degree-163 field of
    /// FIPS 186-4 (curves B-163 and K-163); the first entry of the table in
    /// `irreducibility_known_answers`.
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
    fn inverse_returns_none_for_non_units() {
        // Non-units must give None, not loop. GF(2^4), x^4 + x + 1.
        let field = gf4();

        // The field polynomial is 0 in the field: a non-canonical zero.
        assert_eq!(field.inverse(field.modulus()), None);
        assert_eq!(field.div(&BigUint::one(), field.modulus()), None);

        // poly · x = x^5 + x^2 + x is also ≡ 0: an unreduced zero.
        assert_eq!(field.inverse(&BigUint::from_u64(0b10_0110)), None);

        // A reducible modulus x^2 has the zero-divisor x, no inverse.
        let reducible = Gf2m::new(BigUint::from_u64(0b100)).expect("degree 2");
        assert_eq!(reducible.inverse(&BigUint::from_u64(0b10)), None);
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
    fn half_trace_reduces_its_input() {
        // A non-canonical representative of a field element must give the
        // same half-trace as its reduced form. c XOR poly ≡ c, and carries
        // the degree-163 leading term.
        let field = gf163();
        let c = BigUint::from_u64(0b10); // x, already reduced
        let unreduced = Gf2m::add(&c, field.modulus()); // x XOR poly ≡ x
        assert_ne!(unreduced, c, "the representative differs bit-for-bit");
        assert!(
            unreduced.bits() > field.degree(),
            "the representative is genuinely unreduced"
        );
        assert_eq!(field.half_trace(&unreduced), field.half_trace(&c));
    }

    #[test]
    #[should_panic(expected = "odd field degree")]
    fn half_trace_panics_on_even_degree() {
        // half_trace is not a solver of z² + z = c on an even degree; calling
        // it there is a programming error and must panic in every build, not
        // return a value that fails its own equation.
        let field = gf4(); // GF(2^4), even degree
        let _ = field.half_trace(&BigUint::from_u64(0b10)); // x
    }

    /// splitmix64 (Steele, Lea & Flood, *Fast splittable pseudorandom
    /// number generators*, OOPSLA 2014).
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
        /// Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x5157_0163;
        /// Random elements tried; arbitrary.
        const SAMPLES: usize = 16;
        let field = gf163();
        let mut state = SEED;
        for _ in 0..SAMPLES {
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
        /// Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x0fe2_2163;
        /// Random elements tried; few, since each power is 163 squarings.
        const SAMPLES: usize = 4;
        let mut exponent = BigUint::zero();
        exponent.set_bit(163);
        let mut state = SEED;
        for _ in 0..SAMPLES {
            let a = random_element(&field, &mut state);
            assert_eq!(field.pow(&a, &exponent), a);
        }
    }

    #[test]
    fn div_inverts_mul() {
        /// Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xd1_4143;
        /// Random pairs tried; arbitrary.
        const SAMPLES: usize = 8;
        let field = gf163();
        let mut state = SEED;
        for _ in 0..SAMPLES {
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

        /// Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x7ace_0163;
        /// Random pairs tried; arbitrary.
        const SAMPLES: usize = 12;
        let mut state = SEED;
        for _ in 0..SAMPLES {
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

    /// An independent oracle: shift-and-XOR per set bit, reduced bit-serially
    /// with no shared kernel code.
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
        /// Degree of the dense modulus: one byte, so the search below is
        /// over every polynomial of that degree.
        const DENSE_DEGREE: usize = 8;
        /// The leading term and the tap one below it, x^8 + x^7: a tap gap
        /// of one, which re-raises bits above the degree inside the
        /// boundary word on every fold.
        const TOP_TAPS: u64 = (1 << DENSE_DEGREE) | (1 << (DENSE_DEGREE - 1));
        /// Terms the candidate must have before the top taps are forced:
        /// more than a pentanomial's five, so the modulus is denser than
        /// any standard one.
        const DENSE_TERMS: u32 = 6;
        let mut dense = None;
        for candidate in (1u64 << DENSE_DEGREE)..(1u64 << (DENSE_DEGREE + 1)) {
            let poly = BigUint::from_u64(candidate | TOP_TAPS);
            if candidate.count_ones() >= DENSE_TERMS && Gf2m::is_irreducible(&poly) {
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

        /// Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0xc0b_0236;
        /// Random pairs tried per field; arbitrary.
        const SAMPLES: usize = 24;
        let mut state = SEED;
        for field in &fields {
            for _ in 0..SAMPLES {
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
        /// Arbitrary, fixed so a failure reproduces.
        const SEED: u64 = 0x50_1363;
        /// Random elements tried in the odd-degree field; arbitrary.
        const ODD_SAMPLES: usize = 8;
        /// Random elements tried per even-degree field; arbitrary.
        const EVEN_SAMPLES: usize = 16;
        // Odd degree: must agree with the half-trace on trace-zero input.
        let odd = gf163();
        let mut state = SEED;
        for _ in 0..ODD_SAMPLES {
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
            for _ in 0..EVEN_SAMPLES {
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
                witness = witness.add(&BigUint::one());
            }
            assert_eq!(field.solve_quadratic(&witness), None);
        }
    }

    #[test]
    #[should_panic(expected = "reducible")]
    fn trace_panics_when_the_frobenius_sum_leaves_gf2() {
        // (x²+x+1)² is reducible, and there c = x sums to x²+x+1, outside
        // GF(2). Reporting 0 there would claim z² + z = c is solvable.
        let ring = Gf2m::new(BigUint::from_u64(0b1_0101)).expect("ring");
        let _ = ring.trace(&BigUint::from_u64(0b10));
    }

    #[test]
    fn solve_quadratic_terminates_on_reducible_rings() {
        // Reducible moduli have no well-defined trace-one element; the even-
        // degree branch must return None rather than loop forever.
        // x^2 (0b100) — the smallest reducible even-degree modulus.
        let ring = Gf2m::new(BigUint::from_u64(0b100)).expect("ring");
        assert_eq!(ring.solve_quadratic(&BigUint::zero()), None);
        assert_eq!(ring.solve_quadratic(&BigUint::one()), None);
        // (x^2 + x + 1)^2 = x^4 + x^2 + 1 (0b1_0101) — even degree, reducible.
        let ring = Gf2m::new(BigUint::from_u64(0b1_0101)).expect("ring");
        // Every residue of the degree-4 ring, all 2⁴ of them.
        for c in 0..(1u64 << ring.degree()) {
            // Each call returns (the test finishing proves no hang); any Some
            // is a genuinely verified root, never a fabricated one.
            if let Some(z) = ring.solve_quadratic(&BigUint::from_u64(c)) {
                assert_eq!(
                    Gf2m::add(&ring.square(&z), &z),
                    ring.reduce(BigUint::from_u64(c)),
                    "a returned root must satisfy z^2 + z = c"
                );
            }
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
