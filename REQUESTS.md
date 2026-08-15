# REQUESTS

Primitives the [factoring](https://github.com/darrelllong/factoring) consumer
wants from rump. Everything here is a general `BigUint`/`BigInt` operation, or
general algebra over one — nothing that knows it is looking for factors.
Trial division, Pollard's rho, the Knuth–Schroeppel multiplier, sieving,
smoothness testing against a factor base: those stay downstream and are not
asked for.

Each entry says what it is wanted for, and what the consumer does today
instead. Ordering is by how much the workaround costs, not by how much work
the primitive would be.

Written 2026-08-14, against rump 0.2.0, after implementing Pollard's rho and
the classical quadratic sieve.

---

## Tier 1 — hand-rolled downstream today

These already exist in the consumer as private helpers. Each is small, each is
plainly rump's kind of thing, and each is a place where two crates can now
disagree about the same arithmetic.

### 1. `BigUint::div_rem_u64` (and `to_u64`)

The quadratic sieve's confirmation step divides a candidate by factor base
primes until it is one. The primes are `u64`; the candidate is not. Today:

```rust
let divisor = BigUint::from_u64(prime);      // heap allocation, per prime,
loop {                                        // per candidate
    let (quotient, remainder) = magnitude.div_rem(&divisor);
    if !remainder.is_zero() { break; }
    magnitude = quotient;
}
```

`rem_u64` exists and avoids the allocation, but only answers *whether* the
prime divides — recovering the quotient still needs the `BigUint` round trip.
A `div_rem_u64(&self, u64) -> (Self, u64)` would remove an allocation from
what is, measurably, the hottest non-sieve loop in the program. `remove_factor`
already does the ladder for the multi-precision case; the word-sized case is
the one the sieve actually wants.

Related, and smaller: `to_u64(&self) -> Option<u64>`, returning `None` when the
value does not fit. The consumer writes `value.low_u128() as u64` in a dozen
places, each of which is a silent truncation if the assumption behind it is
ever wrong. `low_u128` is the right primitive for range-pinned callers; this is
the one for callers who want the assumption checked.

### 2. `primes_below(bound) -> Vec<u64>`

A sieve of Eratosthenes. The consumer has now written it twice — once for
trial division, once for the factor base — and they are the same twenty lines.
Generating small primes is number theory, not factoring: `is_probable_prime`
already lives here, and this is the bulk companion to it.

A segmented or iterator form would be better than a `Vec` for large bounds,
but the `Vec` form is what is actually needed today.

### 3. `sqrt_mod` for prime powers

`sqrt_mod` handles a prime modulus. Sieving needs roots modulo `p^e`, so that a
value divisible by `p³` can be credited three times rather than once. The
consumer implements Hensel's lift itself:

```rust
// src/qs/base.rs — Newton's correction over the p-adics
fn hensel_lift(root: u64, kn: &BigUint, prime: u64, modulus: u64, next: u64) -> Option<u64>
```

The lift is generic — it has nothing to do with sieving — and the awkward
cases (`p = 2`, where the count of solutions goes 1, 2, 4 as `e` passes 1, 2,
3, and only when `a ≡ 1 mod 8`) are exactly the ones a library should own
rather than each caller. Suggested shape:

```rust
pub fn sqrt_mod_prime_power(a: &BigUint, p: &BigUint, e: u32) -> Vec<BigUint>
```

returning every solution, empty when there is none.

### 4. A floating-point size estimate

Parameter selection wants `ln n` as an `f64`: the quadratic sieve's smoothness
bound is `exp(½√(ln n · ln ln n))`, and every tuning heuristic in the
literature is written in those terms. The consumer currently sizes everything
off `to_str_radix(10).len()` and a lookup table, which is a decimal-digit proxy
for a natural logarithm.

`ln_approx(&self) -> f64`, or `to_f64_lossy(&self) -> f64` saturating to
infinity, would let the heuristics be written the way they are stated.

---

## Tier 2 — would change what is possible, not just how it is written

### 5. Batch smoothness testing (product and remainder trees)

Bernstein's *How to find smooth parts of integers*: to test many numbers for
smoothness over a fixed set of primes, form the product `P` of the primes by a
product tree, then reduce it against the candidates by a remainder tree, then
take gcds. It replaces per-candidate trial division with one pass whose cost is
near-linear in the total input size.

This is the standard way a modern sieve handles cofactors, and it is pure
`BigUint` — a product tree and a remainder tree, no notion of what the numbers
mean. Suggested shape:

```rust
pub fn product_tree(values: &[BigUint]) -> Vec<Vec<BigUint>>;
pub fn remainder_tree(tree: &[Vec<BigUint>], modulus: &BigUint) -> Vec<BigUint>;
pub fn smooth_parts(values: &[BigUint], primes: &[u64]) -> Vec<BigUint>;
```

The first two are the general primitives; the third is the convenience built
from them. The consumer would use all three.

---

## Tier 3 — the general number field sieve needs these

Not needed for the quadratic sieve, which is where the consumer is now.
Recorded because they are long-lead items and because the answer to "should
rump do polynomials?" is yes, and this is why.

### 6. Polynomials over `ℤ` and over `ℤ/mℤ`

GNFS is built on a degree-5 or -6 polynomial over `ℤ` and its behaviour modulo
small primes. Nothing in it is factoring-specific — it is the polynomial
algebra any computer-algebra layer provides, and rump already contains a
special case of it in `Gf2m`, which is polynomial arithmetic over `GF(2)` with
a fixed modulus and a bit-packed representation.

What GNFS actually reaches for:

- `Poly<BigInt>`: add, sub, mul, `div_rem`, pseudo-division, evaluation and
  Horner, derivative, content and primitive part.
- **Resultant** and **discriminant**. The norm `N(a − αb) = b^d f(a/b)` is the
  homogeneous form, and the resultant is how the algebraic side is computed at
  all. This is the single most important entry in this section.
- Over `𝔽_p`: squarefree decomposition, irreducibility testing, factorization
  (Cantor–Zassenhaus or Berlekamp), and **root finding** — the algebraic factor
  base is exactly the set of roots of `f` modulo each small prime.

A `Poly<T>` generic over a coefficient ring is the ambitious version; concrete
`PolyZ` and `PolyModP` types would be enough and would fit the crate's existing
style, which prefers a named type with a documented representation over a
tower of traits.

### 7. LLL lattice reduction

Polynomial selection searches a lattice, and the final square root step uses
reduction as well. `lll_reduce(basis: &mut [Vec<BigInt>])` over `ℤ`, with the
usual `δ = 3/4` and a settable parameter.

LLL is general — it is used for far more than factoring — and it is the kind of
algorithm that is worth having exactly once, implemented carefully, rather than
badly in each consumer.

---

## Not requested, deliberately

For the record, so the boundary stays where it is:

- Sieving, factor bases, smoothness bounds, the Knuth–Schroeppel multiplier,
  Brent's cycle detection, linear algebra over `GF(2)` sized for relation
  matrices — all of these know they are factoring, and all of them stay in the
  consumer.
- Elliptic curves. ECM is the obvious gap between rho and the sieve, but the
  curve arithmetic it needs is Montgomery-form `x`-only ladders chosen for
  factoring's failure mode (a curve operation that fails *is* the factor), not
  general-purpose EC. That belongs downstream too.

---

## Delivered

Kept as a record of what the list has already produced, so the boundary that
worked stays visible.

- **All of Tiers 1 and 2** (9ec0872) — items 1 through 5 above.
  `BigUint::div_rem_u64` / `to_u64`, `primes_below`,
  `sqrt_mod_prime_power` (every root mod `p^e`: quadratic Hensel for odd
  `p`, the mod-8 dyadic structure, valuation reduction for `p | a`),
  `BigUint::to_f64_lossy` / `ln_approx`, and Bernstein's `product_tree` /
  `remainder_tree` / `smooth_parts`. The signatures are as requested;
  `sqrt_mod_prime_power`'s returned set is `p^⌊v/2⌋` roots when `p^v ∥ a`,
  so beware calling it with `a` divisible by a large base prime.

- **`MontgomeryCtx::add_mont` / `sub_mont`, `BigUint::mod_add` / `mod_sub`**
  (93b4d57). Pollard's rho iterates `x ↦ x² + c` inside the Montgomery domain,
  and the `+ c` was a hand-written conditional subtraction downstream, correct
  only under a reduced-operand precondition living in a comment. It is now one
  call, with the precondition debug-checked where it belongs:

  ```rust
  fn next(ctx: &MontgomeryCtx, y: &BigUint, c: &BigUint) -> BigUint {
      ctx.add_mont(&ctx.square_mont(y), c)
  }
  ```

- **Decimal and radix conversion** — `from_str_radix`, `to_str_radix`,
  `Display`, `FromStr` (0.2.0). Factoring is a decimal-facing activity and the
  consumer had written a chunked converter of its own; it was deleted the day
  0.2.0 landed.

- **`is_probable_prime_bpsw`, `remove_factor`, `nth_root_floor`, `is_square`,
  `sqrt_rem`** (0.2.0). All four are load-bearing in the factoring driver:
  BPSW terminates the recursion, `remove_factor` is trial division's inner
  loop, and the roots are what intercept perfect powers before the search
  wastes a run on them.

- **All of Tier 3** — items 6 and 7 above. `PolyZ` (over ℤ): `add` / `sub` /
  `mul`, `evaluate` (Horner), `derivative`, `content` / `primitive_part`,
  `div_rem` (exact division over ℤ, `None` when the divisor's leading
  coefficient does not divide evenly — always defined for a monic divisor) and
  `pseudo_div_rem` (the ℤ-preserving form the resultant path uses),
  `resultant` and `discriminant` (fraction-free Bareiss over the Sylvester
  matrix). `PolyModP` (over 𝔽_p): `add` / `sub` / `mul`, `div_rem` / `rem` /
  `gcd` / `make_monic` / `pow_mod`, `squarefree_factorization`,
  `is_irreducible`, `factor` (squarefree → distinct-degree →
  Cantor–Zassenhaus), and `roots`. `lll_reduce` / `lll_reduce_delta` (integral
  LLL over ℤ, default δ = 3/4, settable). Every algorithm carries its
  primary-source citation; see CITATIONS.md.
