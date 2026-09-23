# rump

**RU**st **M**ulti**P**recision: multiprecision integer arithmetic in Rust,
implemented directly from the literature, with no dependencies and, in the
default build, no `unsafe` at all — `#![forbid(unsafe_code)]`, which an inner
`allow` cannot lift (the opt-in `wipe` feature admits one audited volatile
scrub; see Properties). Extracted from
[darrelllong/cryptography](https://github.com/darrelllong/cryptography) so the
arithmetic can serve consumers beyond cryptography, with the crate boundary
enforcing a clean API.

Public names and ownership are governed by [`NAMES.md`](NAMES.md). Names not
present there are not added; the breaking API cut uses no compatibility shims
or duplicate public paths.

## What it provides

- **`BigUint`, `BigInt`** — unsigned and signed integers on little-endian
  `u64` limbs. Multiplication climbs a measured ladder: schoolbook (Knuth's
  Algorithm M), Karatsuba, Toom–Cook three- and four-way, and an exact
  two-prime NTT with CRT recovery, whose stages run on as many execution
  contexts as pay and never more than the machine reports; squaring has its
  own kernel at every rung. Division is Knuth's Algorithm D (*TAOCP* vol. 2,
  §4.3.1) with a Newton reciprocal above a measured width and a Horner path
  for single-limb divisors. Also `sqrt_rem`/`sqrt_floor` (certified Newton),
  `nth_root_floor`, `is_square`, `is_perfect_power`, radix conversion both
  ways, and `digit_count` without producing the digits.
- **`MontgomeryContext`, `BarrettContext`** — fixed-modulus reduction.
  The Montgomery domain (Montgomery 1985, in the separated-operand-scanning
  shape of Koç, Acar & Kaliski) encodes once and computes in-domain:
  `mul_mont`, `square_mont`, their `_with_workspace` forms for loops that
  reuse one scratch buffer, `add_mont`, `sub_mont`, windowed `pow`, and
  `gcd_with_modulus` on an encoded residue without decoding it. Barrett
  (HAC Algorithm 14.42) serves a modulus of either parity. `Montgomery64`
  and `Montgomery128` are the same domain at machine width, with no heap,
  for inner loops on one- and two-word moduli.
- **Number theory** — `gcd`, `lcm`, `gcd_extended`; Lehmer's algorithm
  below a measured crossover and subquadratic Half-GCD above it, the Jacobi
  symbol riding the same quotient sequence; `mod_inverse` and its batch form
  (Montgomery's trick); `mod_sqrt` (the `p ≡ 3 (mod 4)` shortcut,
  Tonelli–Shanks, and Cipolla past a measured 2-adic depth) and
  `mod_sqrt_prime_power`; `crt_combine` (Garner) and its balanced form;
  `rational_reconstruct`; `valuation`/`remove_factor`; product and
  remainder trees with `smooth_parts` (Bernstein's batch smoothness);
  `primes_below` and the segmented `primes_past`. Primality: fixed-base
  Miller–Rabin, Baillie–PSW with its strong Lucas stage exposed, the
  FIPS 186-4 Lucas test, `is_prime_u64` (a proof within a word, by the
  twelve bases of Sorenson & Webster), and the deterministic AKS proof.
  Word forms — `gcd_u64`, `gcd_u128`, `mod_inverse_u64`, `mod_inverse_u128`,
  `jacobi_u64`, `crt_combine_u64` — for callers holding machine words.
- **`PolyZ`, `PolyMod`** — dense univariate polynomials over ℤ and 𝔽ₚ:
  exact and pseudo-division, resultant and discriminant (Bareiss),
  squarefree, distinct-degree and Cantor–Zassenhaus factorization,
  `is_irreducible`, `roots`, square roots in 𝔽_{q^d} (`sqrt_in_field`),
  `HenselSquareRoot` (the p-adic Newton lift of a square root in ℤ[x]/(f)),
  and `real_roots`, which locates every real root by bisection on exact
  integer signs so a wide coefficient range cannot hide one.
- **Lattices** — integral LLL (`lll_reduce`, `lll_reduce_delta`,
  `lll_reduce_form`; Cohen's Algorithm 2.6.3, exact Gram data throughout),
  Lagrange–Gauss reduction under a diagonal form, `bareiss_determinant`,
  and certified enumeration: `short_vectors_form` and
  `closest_vectors_form` return an `Enumeration` that says whether the
  search was exhausted, stopped at its visit limit, or met a numerical
  limit, and every returned vector is rechecked exactly.
- **`gf2`** — linear algebra over GF(2) for sieve matrices: singleton
  pruning, structured Gaussian elimination (`filter_merge`), a dense null
  space, and Block Lanczos (Montgomery 1995) on a sparse matrix, its
  matrix products spread over retained workers.
- **`gfp`** — linear algebra over a large prime field GF(l): a sparse
  matrix whose entries are small integers, held with its `±1` entries apart
  so a row costs additions, and Wiedemann's algorithm for a kernel vector —
  the system an index-calculus discrete logarithm ends in.
- **`Gf2m`** — binary extension fields GF(2^m): XOR addition, word-level
  comb multiplication (*Guide to ECC*, Algorithm 2.36) with tap-wise
  reduction, linear squaring, `pow`, `div`, extended-Euclidean inversion,
  the unique `sqrt`, `trace`, quadratic solving at every degree, and Rabin
  irreducibility testing. The degree is derived from the field polynomial,
  never supplied alongside it.
- **Sampling** — `random_below`, `random_nonzero_below`,
  `random_coprime_below`, and `random_probable_prime`, driven entirely by a
  caller-supplied `RandomSource` (one method: `fill_bytes`). rump chooses no
  entropy source; output quality is exactly source quality, so cryptographic
  callers must supply a CSPRNG.

The arithmetic and number theory are deterministic functions of their
inputs. Adversarially hardened primality testing lives with its consumer
(the cryptography crate), where the hash belongs.

## Properties

- `#![forbid(unsafe_code)]`, with no exceptions, in the default build.
  `forbid` rather than `deny` deliberately: an inner `allow` cannot lift it,
  so the guarantee is enforced by the compiler against the crate's own code
  rather than being a default it could override. The opt-in `wipe` feature
  relaxes the attribute to `deny(unsafe_code)` because a volatile scrub has
  no safe expression; its two audited `unsafe` sites are the scrub helper
  and the raw read-back test that proves the shrink paths use it.
- **Variable-time, for non-secret data.** Operations take data-dependent
  paths. Do not use this crate where timing must not leak secrets.
- **Not a secret-scrubbing or constant-time type by default.** In the
  default build nothing is wiped: values live in ordinary heap buffers,
  freed memory keeps its contents, and `Debug` prints every limb. The
  opt-in **`wipe` feature** restores drop-time zeroization as cheap defense
  in depth: every `BigUint` volatile-wipes its live limbs on drop, the
  in-place shrink paths wipe the limbs they abandon, the exponentiation
  ladder and Montgomery workspaces wipe on exit, and the samplers wipe
  drawn bytes. Spare capacity and buffers freed by reallocation are still
  not wiped, and nothing becomes constant-time; a consumer needing more
  adds it at its own layer with a purpose-built representation.

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
`mul`/`sqr` climb schoolbook → Karatsuba → **Toom-3/Toom-4 → exact NTT**;
the NTT is hardware-aware, bounded by reported parallelism, and retains a
specialized one-buffer square. The **1.3–7.5×**
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
132 vectors recomputed with SageMath (`scripts/check_symbol_vectors.sage`) plus Euler's criterion — oracles
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
