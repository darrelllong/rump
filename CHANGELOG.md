# Changelog

Releases are git tags; nothing here is published to crates.io. Entries record
what a consumer must change, not everything that moved.

## 0.3.0 — unreleased

### Breaking

- **`BarrettCtx::add_mod` and `sub_mod` are gone; the rest of the family is
  renamed to the crate's `mod_*` order.** `add_mod`/`sub_mod` were one-line
  forwarders to `BigUint::mod_add`/`mod_sub` — the same operation under the
  same two words in the opposite order, and `μ` plays no part in modular
  addition. Call `BigUint::mod_add(a, b, ctx.modulus())`. The operations that
  do use the context are now `mod_mul`, `mod_square`, and `mod_pow`, matching
  `BigUint::mod_mul` and the free `mod_pow` rather than inverting them.
- **No public function added in 0.3.0 panics on bad input; they return
  `Option`.** `Reciprocal::new`, `SmoothBase::new`, and
  `gauss_reduce_weighted` report a zero divisor, an entry below two, and a
  dependent basis / non-positive weight / out-of-range norm as `None`, which
  is what `BarrettCtx::new` and `MontgomeryCtx::new` already did. The
  pre-existing panicking surface is unchanged in this release.
- **`product_tree` and `remainder_tree` take and return a typed
  `ProductTree`.** Previously:

  ```text
  product_tree(&[BigUint]) -> Vec<Vec<BigUint>>
  remainder_tree(&[Vec<BigUint>], &BigUint) -> Vec<BigUint>
  ```

  Now:

  ```text
  product_tree(&[BigUint]) -> ProductTree
  remainder_tree(&ProductTree, &BigUint) -> Vec<BigUint>
  ```

  The change makes a structural precondition unrepresentable: a caller can no
  longer hand `remainder_tree` a `Vec<Vec<BigUint>>` of the wrong shape, and
  the function can rely on the layout `product_tree` established rather than
  re-deriving or trusting it. `ProductTree` is exported from the crate root.

  A caller that only pipes one into the other is unaffected apart from the
  type name. A caller that *inspected* the levels must go through
  `ProductTree`'s accessors.

  This is a source-breaking change to a public signature. It landed on `main`
  above the `v0.2.2` tag while the crate version still read `0.2.2`, which a
  second reviewer correctly flagged: the tagged `v0.2.2` and the `main` that
  followed it exported incompatible signatures under one version number. The
  version is bumped here so the break carries a number, rather than being
  corrected retroactively in the tagged release.

### Added

- **`Reciprocal`** — division by a `u64` divisor that does not change, with
  the reciprocal precomputed once (Möller & Granlund, IEEE ToC 60 (2011),
  Algorithm 4). `rem`, `div_rem`, `rem_euclid_i64`, and
  `BigUint::rem_reciprocal` / `div_rem_reciprocal` for multi-limb dividends.
  Worth reaching for at two limbs and above; measured *slower* than the
  hardware divide for word-sized dividends on Apple silicon, and the module
  documentation carries the numbers.
- **`SmoothBase`** — Bernstein batch smoothness with the primes' product built
  once, so the caller chooses the batch size rather than the setup cost
  choosing it. The free `smooth_parts` is now the one-shot form over this
  type, so there is one algorithm rather than two.
- **`gauss_reduce_weighted`** — exact two-dimensional Lagrange–Gauss reduction
  under a diagonal form `(w₀x)² + (w₁y)²`, in `i128`. For a skewed metric
  `(x/√s)² + (y√s)²` with rational `s = p/q`, pass `weights = [q, p]`.
- **A `compile_error!` gate on non-64-bit targets.** The crate has documented
  a 64-bit-only contract since 0.2.x — `bits()` scales a limb count by 64, and
  the `R²` and Karatsuba paths scale one by 128, products that overflow a
  32-bit `usize` for operands in the hundreds of megabytes. Until now that
  restriction was prose, and a 32-bit build compiled and then misindexed at run
  time. It now fails to build with a diagnostic explaining why. Verified in
  both directions: the gate fires under `--target i686-unknown-linux-gnu` and
  is silent under `aarch64-apple-darwin`.

### Documentation

- `README.md` no longer opens by calling the crate "pure, safe Rust" while
  listing two audited `unsafe` exceptions sixty lines later. The headline now
  states the exceptions, and the Properties entry says why the crate uses
  `deny(unsafe_code)` rather than `forbid` — `forbid` cannot be lifted by an
  inner `allow`, which is what the scrub helper and its test probe require.
- `REQUESTS.md` is a state machine rather than a diary. Every entry sits in
  exactly one of four states — outstanding; landed in rump with consumer
  migration pending; fully migrated; deliberately downstream — and the file no
  longer carries an outstanding list and the sentence "every entry this file
  has ever carried is closed" at the same time.

## 0.2.2 — tagged `v0.2.2` (`cf2d1dc`)

Answered the external review against v0.2.1: the `2¹⁰`-coupled trial-screen
identity, per-call allocation in the public `mul_mont`/`square_mont`, and the
incomplete `CITATIONS.md`. Added the public `BigInt` signed ring (`mul_ref`,
truncated `div_rem`, `abs`), `BigInt::symmetric_remainder`, and four
word-and-size primitives (`BigUint::digit_count`, `BigInt::from_i128`,
`gcd_u64`, `mod_inverse_u64`).
