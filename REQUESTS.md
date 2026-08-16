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

---

## Tier 1 — outstanding

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

## Tier 1 — cleared

Nothing else outstanding.

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

## Cleared

The `BigInt` signed ring was delivered after 0.2.1 (see below);
`src/gnfs/arith.rs` in the consumer can now be retired.

---

## Not requested, deliberately

For the record, so the boundary stays where it is:

- Sieving, factor bases, smoothness bounds, the Knuth–Schroeppel multiplier,
  Brent's cycle detection, `GF(2)` linear algebra sized for relation matrices,
  base-`m` polynomial selection, the algebraic square root — all of these know
  they are factoring, and all of them stay in the consumer.
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

## Delivered

Every entry this file has ever carried is closed. Kept as a record, so the
boundary that worked stays visible.

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
