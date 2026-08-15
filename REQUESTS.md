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

## Tier 1 — hand-rolled downstream today

These already exist in the consumer as private helpers. Each is small, each is
plainly rump's kind of thing, and each is a place where two crates can now
disagree about the same arithmetic.

### 0. `BigInt` signed arithmetic: `mul_ref`, `div_rem`, `abs`

**Blocking for GNFS.** `BigInt` is a `Sign` joined to a `BigUint`, and the
public surface can add, subtract, negate, compare, and scale by an *unsigned*
factor — but it cannot multiply two signed values, divide, or take an absolute
value:

```rust
let a = BigInt::from_i64(-6);
let b = BigInt::from_i64(7);
a.mul_ref(&b);   // error[E0624]: method `mul_ref` is private
a.div_rem(&b);   // error[E0599]: no method named `div_rem`
a.abs();         // error[E0599]: no method named `abs`
```

`mul_ref` already exists and is already correct — `PolyZ::mul` and
`PolyZ::scale` call it — it is simply not `pub`. Making it public costs
nothing and removes the largest of the three duplications.

`div_rem` and `abs` are the other two. The number field sieve needs them
constantly and in places where getting the sign convention wrong is a silent
wrong answer rather than a crash:

- **Balanced base-`m` expansion.** Polynomial selection writes `n` in base `m`
  with coefficients reduced into `(−m/2, m/2]`, which is a signed division per
  coefficient. Balanced rather than least-non-negative coefficients are the
  whole point: they roughly halve the size of the norms the sieve then has to
  find smooth.
- **Symmetric lifting.** The algebraic square root recovers a result modulo
  `q^k` and must map it back to the symmetric range `(−q^k/2, q^k/2]`. Same
  operation, and the step where an off-by-one in the sign convention produces
  a plausible wrong `β` rather than an error.
- **Coefficient bounds.** Deciding how far to Hensel-lift means comparing
  `|coefficient|` against a bound, which wants `abs` and the `Ord` that
  `BigInt` already has.

Suggested shapes, matching the `BigUint` originals:

```rust
pub fn mul_ref(&self, other: &Self) -> Self;          // just make the existing one pub
pub fn div_rem(&self, divisor: &Self) -> (Self, Self); // truncated, remainder takes the dividend's sign
pub fn abs(&self) -> BigUint;                          // or -> Self; either is usable
```

For `div_rem`, please document which convention it takes — truncated toward
zero (C, Rust `/`) or floored (Python). Either is fine; the consumer needs to
know which, because `modulo_positive` already exists and implies the floored
one is available somewhere.

Until these land the consumer carries `src/gnfs/arith.rs`, which reimplements
all three from `sign()` and `magnitude()` — the exact duplication this file
exists to retire.

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

---

## Delivered

rump 0.2.1 closed every other entry this file has ever carried. Kept as a
record, so the boundary that worked stays visible.

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
