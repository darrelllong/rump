# Changelog

Releases are git tags; nothing here is published to crates.io. Entries record
what a consumer must change, not everything that moved.

## 0.3.0 — unreleased

### Breaking

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
