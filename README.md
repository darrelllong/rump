# rump

**RU**st **M**ulti**P**recision: multiprecision integer arithmetic in Rust,
implemented directly from the literature, with no dependencies and two
audited `unsafe` exceptions (named under [Properties](#properties)) to an
otherwise `#![deny(unsafe_code)]` crate. Extracted from
[darrelllong/cryptography](https://github.com/darrelllong/cryptography) so the
arithmetic can serve consumers beyond cryptography, with the crate boundary
enforcing a clean API.

## What it provides

- **`BigUint`, `BigInt`** — unsigned and signed integers on little-endian
  `u64` limbs. Schoolbook (Knuth's Algorithm M), Karatsuba, and Toom–Cook
  three- and four-way multiplication; Knuth's Algorithm D division (*TAOCP*
  vol. 2, §4.3.1) with a Horner path for single-limb divisors.
- **`MontgomeryCtx`** — a public Montgomery domain (Montgomery 1985; the
  separated-operand-scanning shape from Koç, Acar & Kaliski, IEEE Micro 1996):
  encode once, compute in-domain (`mul_mont`, `square_mont`, their
  `_with_workspace` forms for loops that reuse one scratch buffer,
  `add_mont`, `sub_mont`, `pow`, `pow_encoded`), convert at the boundary.
  Fixed 4-bit window exponentiation. The `_with_workspace` forms remove the
  *scratch* allocation, not every allocation: each returns an owned `BigUint`
  and so allocates its result.
- **Number theory** — `gcd`, `lcm`, and `gcd_extended` (Bézout
  coefficients); the quadratic-residue symbols `jacobi` (binary reciprocity,
  HAC Algorithm 2.149), `legendre`, and `kronecker` (Cohen Algorithm
  1.4.10); `sqrt_mod` (the `p ≡ 3 (mod 4)` shortcut, the Tonelli–Shanks
  descent, and Cipolla's algorithm past a measured 2-adic depth,
  result verified by squaring); `mod_pow`, `mod_inverse`, and `crt_combine`
  (Garner, HAC Algorithm 14.71); fixed-base Miller-Rabin
  (`is_probable_prime`, `is_probable_prime_with_bases`), the reusable
  per-round primitive `miller_rabin_witness` for callers that bring their
  own witness schedule, and Baillie-PSW (`is_probable_prime_bpsw`, with
  the strong Lucas stage exposed as `is_strong_lucas_probable_prime`);
  batch inversion (`mod_inverse_batch`, Montgomery's trick);
  rational reconstruction (`rational_reconstruct`,
  `rational_reconstruct_bounded`) recovering the unique bounded fraction
  from its residue; `valuation`/`remove_factor` by a squared-power
  ladder; word-sized forms (`gcd_u64`, `mod_inverse_u64`) for callers
  holding machine words. The integer layer adds `sqrt_rem`/`sqrt_floor`
  (certified Newton), `nth_root_floor`, `is_square`, `is_perfect_power`,
  `popcount`, `trailing_zeros`, and `digit_count` (written length in any
  radix, without producing the digits).
- **`BarrettCtx`** — fixed-modulus reduction for a modulus of either
  parity (HAC Algorithm 14.42), the complement to the odd-modulus
  Montgomery domain, with `mul_mod`, `square_mod`, and `pow_mod` built on
  it.
- **`PolyZ`, `PolyModP`** — dense univariate polynomials over ℤ and 𝔽ₚ:
  exact and pseudo-division, resultant and discriminant (Bareiss),
  squarefree/distinct-degree/Cantor–Zassenhaus factorization,
  `is_irreducible`, and `roots`.
- **`lll_reduce`, `lll_reduce_delta`** — integral LLL lattice basis
  reduction (Cohen's Algorithm 2.6.3), exact integer Gram data throughout.

- **`Gf2m`** — binary extension fields GF(2^m): XOR addition, word-level
  comb multiplication (*Guide to ECC*, Algorithm 2.36) with tap-wise
  reduction, linear squaring (Algorithm 2.39), `pow`, `div`, extended-Euclidean
  inversion (Algorithm 2.48), the unique `sqrt`, `trace`, quadratic solving at every
  degree (`solve_quadratic`, with `half_trace` as the odd-degree
  primitive), and Rabin irreducibility testing. The degree is derived from
  the field polynomial, never supplied alongside it.
- **Sampling** — `random_below`, `random_nonzero_below`,
  `random_coprime_below`, and `random_probable_prime`, driven entirely by a
  caller-supplied `Rng` (one method: `fill_bytes`). rump chooses no entropy
  source; output quality is exactly source quality, so cryptographic callers
  must supply a CSPRNG.

The arithmetic and number theory are deterministic functions of their
inputs. Adversarially hardened primality testing lives with its consumer
(the cryptography crate), where the hash belongs.

## Properties

- `#![deny(unsafe_code)]`; the audited exceptions are a six-line
  volatile-write scrub helper and the test probe that verifies the scrub
  by reading the raw buffer tail back. `deny` rather than `forbid` because
  those two sites lift it with an inner `allow`, which `forbid` does not
  permit — so the guarantee is a default the crate's own code can override,
  not a boundary the compiler enforces.
- **Variable-time, for non-secret data.** Operations take data-dependent
  paths. Do not use this crate where timing must not leak secrets.
- **Not a secret-scrubbing or constant-time type, and does not pretend to be.**
  As cheap defense in depth every `BigUint` volatile-wipes its live limbs on
  drop and the exponentiation ladder wipes its workspaces on exit. That is the
  extent of it: spare capacity and buffers freed on reallocation are not wiped,
  the in-domain `mul_mont` / `square_mont` keep their scratch, and `Debug`
  prints every limb. Cryptographic memory hygiene and constant-time operation
  are out of scope; a consumer that handles key material adds them at that
  layer.

## Benchmarks

[PERFORMANCE.md](PERFORMANCE.md) is the full per-primitive report: pilot-bench
means with confidence intervals and variable-time extrema over random operands,
log–log scaling graphs, fitted complexity exponents, and a per-primitive
comparison against GMP on four hosts — Apple M4, AMD EPYC 7452, Raspberry
Pi 5, and Apple A18 Pro. Regenerate the data with
`scripts/bench_primitives.sh` (rump and, via
`pilot_gmp`, GMP through the same harness) and the document with
`scripts/build_performance.sh`.

`cargo run --release --bin bench_bigint` reports ns/op for the core kernels.
Headline vs GMP: `modpow` stays within **1.1–3.4×** (matched windowed
Montgomery); the Euclid family — `gcd`, `gcd_extended`, `mod_inverse` — is
**4–13×** and `jacobi` **2–12×**, down from **17–89×** on classical Euclid,
after switching to **Lehmer's gcd** and a **division-free binary Jacobi**.
`mul`/`sqr` climb schoolbook → Karatsuba → **Toom-3/Toom-4**; the **1.3–7.5×**
that remains at crypto sizes is GMP's assembly inner loops, not the algorithm
(on the Raspberry Pi, where that assembly edge shrinks, `mul` is only 1.3–1.9×).
Above ~131 kbit, `gcd` dispatches to **Half-GCD** (Möller, Math. Comp. 77
(2008); the algorithm behind GMP's `mpn_hgcd`) and goes subquadratic — see
PERFORMANCE.md's "GCD at scale". The same transform is carried through
the Bézout cofactors (`gcd_extended` and `mod_inverse`, above ~32 kbit)
and the Jacobi symbol (`jacobi_hgcd` — Möller's threading design, as in
GMP's `mpn_hgcd_jacobi`; Brent and Zimmermann's published subquadratic
symbol reaches the same complexity by the binary route).

## Manual

[MANUAL.md](MANUAL.md) documents every public API with a worked example.
Every code block in it is replicated in `tests/manual_examples.rs` and
asserted on `cargo test`, so the manual cannot drift from the code.

[manual.tex](manual.tex) (built copy: `manual.pdf`) is the LaTeX reference
manual — the same surface with the defining mathematics for each
primitive; `scripts/check_manual_tex.sh` extracts its listings and
executes them against the crate, and a rebuild is gated on that passing.

[CITATIONS.md](CITATIONS.md) is the primary-source reference list: every
non-schoolbook algorithm in the crate with the paper, book, or standard it
comes from.

## Testing

Differential suites check division against a bit-serial oracle, Montgomery
exponentiation against a division-based ladder, and the Jacobi symbol against
132 vectors computed by GMP's `mpz_jacobi` plus Euler's criterion — oracles
that share no code with the kernels they judge. The suites are
mutation-hardened: seeded defects in the quotient estimate, the REDC carry
chain, and the reciprocity logic are caught, and the survivors are proven
behavior-equivalent and documented in place.

## Naming

The repository and library are `rump`; the crates.io package is `rust-mp`
(the bare name is taken by an unrelated tool). Depend on `rust-mp` and write
`use rump::...`.

## License

BSD-2-Clause.
