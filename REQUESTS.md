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

## Tier 1

**`BigInt::to_i64` (and `to_i128`), the signed counterpart of
`BigUint::to_u64`.** Returning `None` on overflow, exactly as the unsigned one
does.

`BigUint::to_u64` exists and `BigInt` has none, so every place the consumer
needs a signed `BigInt` back as a machine integer goes through the magnitude
and reattaches the sign by hand:

```rust
fn to_i64(value: &BigInt) -> Option<i64> {
    let magnitude = value.magnitude().to_u64()?;
    let signed = i64::try_from(magnitude).ok()?;
    Some(if value.sign() == rump::Sign::Negative { -signed } else { signed })
}
```

Wanted for the lattice sieve, which solves `i·P + j·Q = 0` over `BigInt` to
find where the rational form crosses zero on a row, and then needs that `i`
as an index. Written 2026-08-15.

Note the asymmetry the hand-rolled version has to get right and which a
library version should decide deliberately: `i64::MIN` has magnitude `2^63`,
which `i64::try_from` rejects, so the snippet above returns `None` for a value
that does fit. The consumer does not care — its `i` are small — but a
primitive should not inherit that.

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
