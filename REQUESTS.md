# REQUESTS

Primitives the `factoring` consumer (hosted alongside this on the Forgejo at
`sequoia`) wants from rump. Everything here is a general `BigUint`/`BigInt` operation, or
general algebra over one — nothing that knows it is looking for factors.
Trial division, Pollard's rho, the Knuth–Schroeppel multiplier, sieving,
smoothness testing against a factor base: those stay downstream and are not
asked for.

Each entry says what it is wanted for, and what the consumer does today
instead. Ordering is by how much the workaround costs, not by how much work
the primitive would be.

Written 2026-08-14 against rump 0.2.0, after implementing Pollard's rho and
the classical quadratic sieve; revised the same day against 0.2.1, while
starting the number field sieve.

Every entry sits in exactly one of four states, and the section it appears
under *is* that state:

- **Outstanding** — asked for, not yet in rump.
- **Landed in rump, consumer migration pending** — implemented here, but the
  consumer still runs its own copy. Two correctness surfaces until it does
  not, so a later fix lands in only one of them.
- **Fully migrated** — in rump, and the consumer's copy is deleted.
- **Deliberately downstream** — considered and left with the consumer, with
  the reason it stays there.

An entry moves when the *code* moves, not when the prose is next edited. This
file is the cross-repository ownership ledger; a second reviewer found it
carrying an outstanding list and the sentence "every entry this file has ever
carried is closed" at the same time, which is the failure this legend exists
to prevent.

---

## Outstanding

Every entry names the file and line where the consumer's implementation
sits, so a mover can read the working version rather than start from the
description. The consumer keeps its copy running until each lands, then
deletes it — none of these are urgent, and all of them are general algebra
that ended up downstream because it was written where it was first needed.

### Real roots and the real factorisation of a `PolyZ`

`PolyZ` already carries `resultant`, `discriminant`, and — through `PolyModP`
— roots modulo a prime. What it has no answer for is where a polynomial
crosses the *real* line, and that is the conspicuous gap in an otherwise
complete polynomial type.

Wanted, roughly:

```text
PolyZ::real_roots(&self) -> Vec<f64>
PolyZ::factor_real(&self) -> (f64, Vec<f64>, Vec<(f64, f64)>)
        // leading coefficient, real roots, conjugate pairs (p, q)
```

so that a caller can write `|f(x)| = |c| · ∏|x − rⱼ| · ∏((x − pₖ)² + qₖ²)`
without owning any of the numerics.

**What the consumer does today.** `src/gnfs/model.rs`, about 470 lines: complex
arithmetic, a Cauchy bound for the starting circle, Durand–Kerner to find all
roots at once, deflation, and a bisection refinement for the real ones. It is
correct and tested, and none of it knows it is looking for factors — the
module's *prose* is about sieve lines because that is what calls it, but every
function in it is ordinary polynomial numerics.

**What the workaround costs.** Nothing measurable. This is a boundary request
rather than a performance one, which is why it sits at the bottom of an
ordering by cost: the consumer is holding general algebra that the rule in this
file's header says belongs here. Worth doing when `poly.rs` is next open
anyway, not before.

Two things a mover should keep, because they were arrived at by being wrong
first:

- `f64` is ample for this and the consumer says so explicitly. The answer
  selects which integer positions to evaluate *exactly*; being a place or two
  out costs nothing.
- The real minima of `|f|` between consecutive real roots are **not** where a
  naive reading puts them. `|f|` does not rise and fall monotonically between
  them — a cubic with one real root has two critical points on the same side
  of it — so the roots of `f'` are wanted alongside the roots of `f`.

---

### Sparse `GF(2)` linear algebra: null space, Block Lanczos, singleton peel

**Consumer's code:** `src/qs/linalg.rs` (501 lines — `null_space` at :43,
`prune` at :151) and `src/qs/lanczos.rs` (546 lines — `dependencies` at :290).

This file previously recorded `GF(2)` linear algebra as deliberately *not*
requested, on the grounds that it was "sized for relation matrices". That was
the wrong test and the entry has been removed. Solving `Mx = 0` over `GF(2)`
for a large sparse `M` is not a factoring problem — it is the same computation
in index-calculus discrete logarithms, in coding theory, and anywhere a parity
system gets large. Only the *matrix* is factoring's; the solver is not.

Three pieces, usable separately:

- **Dense null space** by Gauss–Jordan over bit-packed rows. Cubic, and right
  below a few thousand columns.
- **Block Lanczos** over `GF(2)`, Montgomery's method (EUROCRYPT '95,
  equations 18–20 and figure 1), including the subspace selection `invert`
  and the cleanup for the `ker(A) ⊋ ker(M)` gap that the method leaves. This
  is the piece worth having: it is sparse-time where Gauss–Jordan is cubic,
  and it is fiddly enough that a second implementation would be a waste.
- **Singleton peel** — repeatedly drop rows holding a column no other row
  touches, cascading. Ordinary sparse preprocessing; the XOR-of-indices trick
  in `prune` keeps it linear.

The consumer would keep the *layout* — which column means which prime or
ideal — and hand over only the bits.

---

### The square root in `ℤ[x]/(f, q^k)` by Newton lifting

The last of six polynomial pieces that ended up downstream. The other
five landed 2026-08-16 and have moved to *Landed in rump, consumer
migration pending*, below.

Its two building blocks are delivered — `PolyModP::symmetric_lift` and
`PolyModP::with_modulus`, with a test showing they compose into the lift —
but the lift itself is not, because the consumer's version is inseparable
from its stopping rule. `lift_target_bits` and `MAX_LIFTS` decide when a
failing check becomes a verdict, and they are calibrated on the measured
`H(β)/H(δ)` ratio of number-field-sieve dependencies. A general version needs
the caller to name a target precision instead, which is a different signature
from the one the consumer uses; filing it that way is the next step rather
than a port. Note also that `with_modulus`'s two divisibility directions
behave differently — narrowing is the ring projection, widening is only a
section — which the consumer's `widen` relies on without saying so.

---

### Two-dimensional lattice reduction under a weighted norm

**Consumer's code:** `src/gnfs/lattice.rs:352` (`gauss_reduce`, with `norm_sq`
and `dot` beside it).

Lagrange–Gauss reduction of a two-dimensional basis, but under a *skewed* norm
`(x/√s)² + (y·√s)²` rather than the Euclidean one, in `i128` throughout. rump
has `lll_reduce` over `BigInt` for general dimension; the two-dimensional case
is exact, terminates in `O(log)` steps, and wants no bignums. The weight is
the general part — reduction under a diagonal form is what any anisotropic
lattice problem needs.

---

### A reusable batch-smoothness context

`smooth_parts` (`src/number_theory.rs`) already implements Bernstein's 2004
batch algorithm, and it is exactly the right tool for deciding which sieve
reports are worth full trial division. The consumer does not use it, and the
reason is an API one rather than an algorithmic one: the function rebuilds its
prime product `z` on every call.

```rust
let prime_values: Vec<BigUint> = primes.iter().map(|&p| BigUint::from_u64(p)).collect();
let z = product_tree(&prime_values).root()...
```

For a factor base to 20 000 that product is about 13 500 bits, built by a
product tree over ~1 100 primes. Paying for it once per run is nothing; paying
per batch decides how the caller may batch, and the natural batch here is one
block — about three reports. So the primitive as it stands can only be used in
one enormous batch at the end of a run, which is not how relation collection
works: the run stops as soon as it has enough.

Wanted, roughly:

```text
SmoothBase::new(primes: &[u64]) -> SmoothBase   // builds z once
SmoothBase::primes(&self) -> &[u64]
SmoothBase::smooth_parts(&self, values: &[BigUint]) -> Vec<BigUint>
```

with the existing free function kept as the one-shot convenience form,
implemented over the context so there is one algorithm and not two.

**What the consumer does today.** Nothing — it never calls `smooth_parts`.
`examine` (`src/qs/sieve.rs:581`) walks the whole factor base per report,
probing each prime for a root-class match. About 70% of reports are not smooth
(`conf/rep` is 0.299 at 40 digits, 0.473 at 46), and a report that is not
smooth never terminates early: `magnitude` never reaches one, so it pays the
entire base.

**What the workaround costs.** The 4 516 279 probes above, of which the
overwhelming majority are spent proving that values which are not smooth are
not smooth. A batched pre-filter would leave the base walk to be paid only by
the ~30% that go on to become relations.

Two notes for a mover:

- The consumer's need is a **predicate**, not a factorisation: it wants to know
  whether `smooth_part == |value|`, and only then does it want exponents. It is
  fine for the context to return smooth parts exactly as the free function
  does.
- The caller obligations already documented on `smooth_parts` — entries at
  least two, the panic on a smaller one — should move onto `SmoothBase::new`,
  where they can be checked once instead of per batch.

Both entries above were staged in the consumer as `REQUESTS-TO-RUMP.md` and
merged here 2026-08-16, measured on the quadratic sieve at 40 and 46 digits.
That staging file was deleted in the same change: it said so itself, and two
ledgers is the failure this file's legend exists to prevent.

---

## Landed in rump, consumer migration pending

**Delivered 2026-08-16: division by a fixed `u64` divisor.** `Reciprocal`
is in `src/bigint/reciprocal.rs`, exported from the crate root, documented
in `MANUAL.md` and `manual.tex` and cited in `CITATIONS.md`. It carries
`new`, `divisor`, `rem_u64`, `div_rem_u64` and `rem_euclid_i64`, with
`BigUint::rem_reciprocal` and `BigUint::div_rem_reciprocal` for multi-limb
dividends. Möller–Granlund Algorithm 4 over their Algorithm 2 reciprocal,
normalized internally so a 14-bit factor-base prime works as well as a
full-width divisor. One kernel serves both the word and the multi-limb
paths, because dividing by a word is Horner's recurrence whose every step
is a two-word-by-one-word division. Verified against the existing
hardware-division path — `div_rem_u64` / `rem_u64` as oracle — over
seventeen corner divisors plus thirty-two random ones, at widths from one
limb to sixty-four, and `rem_euclid_i64` against `i64::rem_euclid` wherever
the divisor fits a positive `i64`, `i64::MIN` included.

The consumer has not adopted it yet: the six sites listed below still call
`rem_euclid` and `div_rem_u64` directly.

The request as filed, kept for its measurements and its site list:

### Division by a fixed `u64` divisor, precomputed once

The sieve's inner loops divide by the same small divisor millions of times.
Every one of those divisors is a factor-base modulus: chosen when the base is
built and constant for the entire run. The hardware divider is 20–40 cycles
and does not care that the divisor has not changed; a precomputed reciprocal
turns each into a multiply and a shift.

This is the one classical integer-arithmetic primitive rump does not have.
`BarrettCtx` covers a fixed `BigUint` modulus (`src/bigint/barrett.rs`), and
`div_rem_u64` / `rem_u64` cover a word divisor used once. The gap is a word
divisor used *many* times.

Wanted, roughly:

```text
Reciprocal::new(divisor: u64) -> Reciprocal          // divisor >= 1
Reciprocal::divisor(&self) -> u64
Reciprocal::rem_u64(&self, value: u64) -> u64
Reciprocal::div_rem_u64(&self, value: u64) -> (u64, u64)
Reciprocal::rem_euclid_i64(&self, value: i64) -> u64 // non-negative residue

BigUint::rem_reciprocal(&self, r: &Reciprocal) -> u64
BigUint::div_rem_reciprocal(&self, r: &Reciprocal) -> (BigUint, u64)
```

Granlund–Montgomery (PLDI 1994) for the exact quotient form, Möller–Granlund
(IEEE ToC 2011) for the improved variant, Lemire–Kaser–Kurz (2019) for the
remainder-only "fastmod" that is enough where no quotient is wanted — which is
most of the calls below.

**What the consumer does today.** Ordinary hardware division, once per call:

| Site | Call |
|---|---|
| `src/qs/sieve.rs:504` | `(root as i64 - low).rem_euclid(modulus)` — the sieve walk's start, per root per block |
| `src/qs/sieve.rs:598` | `x.rem_euclid(prime as i64)` — the confirmation probe, per base prime per report |
| `src/qs/sieve.rs:612`, `955` | `magnitude.div_rem_u64(prime)` — the exponent ladder |
| `src/gnfs/sieve.rs:492` | `(target - low).rem_euclid(modulus)` — the same walk on the GNFS side |
| `src/gnfs/sieve.rs:706`, `850` | `value.div_rem_u64(prime)` |
| `src/gnfs/lattice.rs:489-542` | lattice `rem_euclid(p)`, several per special-`q` |

**What the workaround costs.** Counted, not estimated, on balanced semiprimes
with the quadratic sieve on twelve cores:

| | 40 digits | 46 digits |
|---|---|---|
| sieve-walk `rem_euclid` | 3 716 800 | 34 061 160 |
| confirmation probes | 4 516 279 | ~7.7 M |
| exponent-ladder divisions | 77 408 | — |

So roughly 8 million fixed-divisor divisions at 40 digits and 42 million at 46,
split about evenly between the two halves of the sieve at the smaller size and
dominated by the walk at the larger. Confirmation is 45–49% of sieve time, and
within `examine` the base walk is the bulk of it: evaluating `g(x)` exactly is
only 14%, and there are 58 probes for every division the probes find.

Three things a mover should keep, because the consumer got them wrong first or
would have:

- **`rem_euclid_i64` is the shape actually wanted**, not `%`. Sieve positions
  are signed and the residue must be non-negative. Every consumer that has to
  re-derive that from a truncating remainder gets a chance to be wrong.
- **The precompute must amortise, and here it does completely.** One
  `Reciprocal` per factor-base entry, built once when the base is built, reused
  across ~1 600 blocks and thousands of reports. A `new` that costs a division
  is fine; one that costs more than a few is still fine.
- **A remainder-only fast path is worth having separately.** The probe at
  `src/qs/sieve.rs:598` throws the quotient away, and that is the single
  hottest of these sites.

---

---


Implemented here and documented in `MANUAL.md` and `manual.tex`, cited in
`CITATIONS.md` — but the consumer still runs its own copy of each. Until
those are deleted there are two implementations of the same algebra, and a
later fix lands in only one of them. The second reviewer raised this as a
standing risk rather than a defect, and supplied the locations.

Retire, in one integration change against a versioned rump:

The consumer's copies carry *different names* from the rump API that replaces
them, so search by the name in the third column rather than by the first — a
search for the rump name finds nothing and reads as "already migrated", which
is the wrong answer. Line numbers are as of 2026-08-16 and will drift; the
names will not.

| rump API | consumer file | consumer's name for it |
|---|---|---|
| `PolyZ::roots_mod_prime_power` | `src/gnfs/select.rs:321` | `lifted_valuation` (the lifting/counting around it) |
| `PolyZ::balanced_base_expansion` | `src/gnfs/select.rs:393` | `balanced_expansion` |
| `PolyZ::rem_monic` | `src/gnfs/sqrt.rs:234` | `reduce` |
| `PolyZ::product_mod_monic` | `src/gnfs/sqrt.rs:186` | `algebraic_product` |
| `PolyZ::homogeneous_substitution` | `src/gnfs/lattice.rs:313` | `transformed_norm` |
| `PolyModP::with_modulus` | `src/gnfs/sqrt.rs:401` | `widen` |
| `PolyModP::symmetric_lift` | `src/gnfs/sqrt.rs:401-418` | (alongside `widen`) |

Delivered 2026-08-16 in `src/poly.rs`:

- **Roots modulo a prime power, by Hensel lifting.**
  `PolyZ::roots_mod_prime_power(prime, exponent, rng)`. The branching case is
  handled as described — `f'(r) ≢ 0 (mod p)` gives the unique lift
  `t ≡ −s·f'(r)⁻¹`, `f'(r) ≡ 0` gives all `p` lifts or none according to
  whether `p^{k+1} | f(r)`. Two generalizations beyond the consumer's version:
  a polynomial whose content is divisible by `p` is *answered* rather than
  refused (the roots of `f/pᵛ` at precision `e−v`, expanded back), and the
  width cap is on the candidate count rather than on the prime, with
  `MAX_ROOT_LEVEL` exported. The consumer's `LIFT_WIDTH` truncation is not
  reproduced: rump refuses rather than silently returning some of the roots.
  Verified against exhaustive search over every residue for nine `(p, e)`
  pairs, including forced-branching cubes and non-primitive polynomials.

- **Reduction modulo a monic polynomial.** `PolyZ::rem_monic`. Total, not an
  `Option` — a monic divisor always divides over ℤ — and it forms no quotient
  and performs no coefficient division, since the quotient coefficient is the
  remainder's leading coefficient.

- **Product tree over a quotient ring.** `PolyZ::product_mod_monic`. Pairing,
  reducing at every level.

- **Homogeneous substitution.** `PolyZ::homogeneous_substitution(a, b)`,
  evaluating `F(X,Y) = Yᵈ f(X/Y)` at a pair of polynomials, with both power
  ladders built once.

- **Balanced base-`m` expansion.** `PolyZ::balanced_base_expansion(n, base,
  degree)`. Digits below the top in `(−m/2, m/2]`, the top carrying the
  remainder so the identity is exact for every requested degree. The
  consumer's `Option` and its monic-degree check were selection *policy* and
  stayed downstream.

---

## Fully migrated

In rump, and the consumer's copy is deleted. This section is the
already-migrated tail; it is not a claim that the ledger is empty, which the
*Outstanding* section above settles.

**The `BigInt` signed ring.** Delivered after 0.2.1 (recorded under *Delivered
earlier* below); `src/gnfs/arith.rs` in the consumer is deleted.

**Implemented directly, 2026-08-16, by the consumer rather than requested.**
Four primitives the factoring crate had grown local copies of. Each is
arithmetic on numbers with nothing in it that knows about factoring, which is
the test this file applies, so they were written here rather than filed:

- `BigUint::digit_count(radix)`. Size-driven parameter tables ask how long a
  number is and throw the digits away; `to_str_radix(radix).len()` answers by
  producing the whole expansion, which is quadratic in the limbs. Logarithm
  from the limbs, boundary settled by comparison — the powers of the radix are
  the only place the floor is ambiguous, and they are exactly where a naive
  version is wrong. Tested against `to_str_radix` at every power of six radices.
- `BigInt::from_i128`. `BigUint::from_u128` and `BigInt::from_i64` both
  existed; the signed double word did not, so the consumer was building it by
  printing the value in decimal and parsing it back.
- `gcd_u64` made public. It already existed as `gcd`'s single-limb base case.
- `mod_inverse_u64`. The word-sized companion to `mod_inverse`, carrying its
  Bézout coefficients in `i128` because the intermediate is not bounded by
  `u64` even though the answer is.

Two callers in the consumer had independently grown the same extended Euclid
under two names, which is the usual sign that something belongs one level
down.

**Withdrawn 2026-08-15, same day it was raised: `BigInt::to_i64`.** Asked for
on behalf of the lattice sieve, which solved `i·P + j·Q = 0` over `BigInt` to
find where the rational form crosses zero on a row and needed that `i` as an
index. The caller is gone: `a − bm` is *linear* in `i`, so the row is a V about
that zero and the bar is `log₂|P| + log₂|i − i₀|` — one logarithm of `|P|` and
one ratio, both fixed per lattice, and no `BigInt` per position at all. The
exact solve was doing arithmetic to answer a question that had a closed form.

Recorded rather than deleted because the shape recurs: a request for a
narrowing conversion is often a sign the caller is computing something exactly
that it only needs to within a bit.

---

## Deliberately downstream

For the record, so the boundary stays where it is:

- **What the consumer owes on the two entries merged 2026-08-16**, carried over
  from the staging file so the boundary stays honest in both directions.
  Neither is rump's work: adopting `smooth_parts` at all once the
  `SmoothBase` context exists — the primitive has been available and unused,
  which is a downstream omission rather than an upstream gap — and deciding
  *where* the pre-filter sits in confirmation, then re-measuring the
  sieve/confirmation split afterwards, since that split moves with the bar,
  the floor, and the large-prime variation.
- Sieving, factor bases, smoothness bounds, the Knuth–Schroeppel multiplier,
  Brent's cycle detection, the *policy* of base-`m` polynomial selection, the
  bar and tolerance machinery — all of these know they are factoring, and all
  of them stay in the consumer.
- **Retracted 2026-08-16:** this list used to include "`GF(2)` linear algebra
  sized for relation matrices", "base-`m` polynomial selection" without the
  qualifier above, and "the algebraic square root". That applied the wrong
  test. The rule is whether the *code* knows it is looking for factors, not
  whether factoring is what happens to call it — and a `GF(2)` solver, a
  balanced expansion, and a Newton lift in a quotient ring do not. All three
  are now requested above.
- Elliptic curves. ECM is the obvious gap between rho and the sieves, but the
  curve arithmetic it needs is Montgomery-form `x`-only ladders chosen for
  factoring's failure mode — a curve operation that *fails* is the factor —
  not general-purpose EC. That belongs downstream too.
- GPU cofactorization. One constraint worth stating before it is ever asked
  for: rump is *scalar multiprecision* — limb vectors of dynamic length,
  branching algorithms, allocation. GPU cofactorization wants *fixed-width*
  arithmetic (64/128-bit Montgomery in registers, no allocation, no
  divergence), which is a different kernel entirely. If the consumer ever
  wants it, that is a new request with its own design, not a port of what is
  here. (Flagged 2026-08-15.)

---

## Delivered earlier — historical record

Closed entries from previous rounds, kept so the boundary that worked stays
visible. This section is a record of what shipped; it is **not** a statement
about the current ledger, which begins with *Outstanding* above.

**Post-0.2.1 — `BigInt::symmetric_remainder`.**

The last function in the consumer's `gnfs::arith`, and the only one that was
ever more than indirection. It is `modulo_positive`'s companion — the other
canonical representative, the one that is smallest in absolute value rather
than non-negative — and the consumer wanted it in two places at once: the
balanced base-`m` expansion, where it roughly halves every norm the sieve must
find smooth, and the lift of the algebraic square root out of `ℤ/q^k`, where
taking the non-negative representative instead produces a plausible wrong
answer rather than an error. With this delivered `gnfs::arith` is deleted;
`mul`, `div_rem` and `abs` had been indirection to `mul_ref`, `div_rem` and
`magnitude()` since 0.2.2, across fifty-three call sites.

**Post-0.2.1 — `BigInt` signed arithmetic (was Tier 1 #0, blocking for
GNFS).**

- `BigInt::mul_ref` is now `pub` — the existing crate-private product, made
  public unchanged.
- `BigInt::div_rem`, truncated toward zero (the C and Rust `/` convention;
  the remainder takes the dividend's sign), documented in the rustdoc with
  the `(-7)/2 = (-3, -1)` corner named, and differentially tested against
  `i128`. `modulo_positive` remains the floored remainder against an
  unsigned modulus.
- `BigInt::abs -> BigUint`.

This retires the consumer's `src/gnfs/arith.rs`: balanced base-`m`
expansion, symmetric lifting, and coefficient-bound comparisons all run on
rump's ring now.

**0.2.1 — the whole outstanding list.**

- `primes_below` (was #2). The consumer had written a sieve of Eratosthenes
  twice; both are gone.
- `sqrt_mod_prime_power` (was #3). The quadratic sieve's factor base carried
  its own Hensel lift for roots modulo `p^e`.
- `BigUint::div_rem_u64` and `to_u64` (was #1), and `ln_approx` /
  `to_f64_lossy` (was #4). The first removes an allocation from the sieve's
  hottest confirmation loop; the second lets parameter heuristics be written
  as the literature states them instead of off a decimal-digit proxy.
- `product_tree`, `remainder_tree`, `smooth_parts` (was #5) — batch smoothness
  by Bernstein's method.
- `PolyZ` and `PolyModP` (was #6), with `resultant`, `discriminant`,
  `squarefree_factorization`, `is_irreducible`, `factor`, and `roots`. This is
  what makes the number field sieve possible at all: `resultant` is the
  algebraic norm, and `roots` modulo a small prime *is* the algebraic factor
  base.
- `lll_reduce` and `lll_reduce_delta` (was #7).

**0.2.0 and 93b4d57 — the earlier round.**

- `MontgomeryCtx::add_mont` / `sub_mont`, `BigUint::mod_add` / `mod_sub`.
  Pollard's rho iterates `x ↦ x² + c` inside the Montgomery domain, and the
  `+ c` was a hand-written conditional subtraction downstream, correct only
  under a reduced-operand precondition living in a comment. Now one call:

  ```rust
  fn next(ctx: &MontgomeryCtx, y: &BigUint, c: &BigUint) -> BigUint {
      ctx.add_mont(&ctx.square_mont(y), c)
  }
  ```

- Decimal and radix conversion — `from_str_radix`, `to_str_radix`, `Display`,
  `FromStr`. Factoring is a decimal-facing activity and the consumer had a
  chunked converter of its own; it was deleted the day 0.2.0 landed.
- `is_probable_prime_bpsw`, `remove_factor`, `nth_root_floor`, `is_square`,
  `sqrt_rem`. All load-bearing in the factoring driver: BPSW terminates the
  recursion, `remove_factor` is trial division's inner loop, and the roots
  intercept perfect powers before the search wastes a run on them.
