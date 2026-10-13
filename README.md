# rump

**RU**st **M**ulti**P**recision: multiprecision arithmetic in Rust,
implemented from the literature, with no dependencies. The default build
forbids unsafe code; the opt-in `wipe` feature permits volatile scrubbing.

The crates.io package is `rust-mp`; the library name is `rump`. Public API
ownership is recorded in [NAMES.md](NAMES.md).

## What it provides

- **`BigUint`, `BigInt`** — unsigned and signed integers on little-endian
  `u64` limbs. Multiplication climbs a measured ladder: schoolbook (Knuth's
  Algorithm M), Karatsuba, Toom–Cook three- and four-way, and an exact
  two-prime NTT with CRT recovery. Its stages use a bounded number of workers;
  squaring has its own kernel at every rung. Division is Knuth's Algorithm D
  (*TAOCP* vol. 2,
  §4.3.1) with a Newton reciprocal above a measured width and a Horner path
  for single-limb divisors. Also `sqrt_rem`/`sqrt_floor` (certified Newton),
  `nth_root_floor`, `is_square`, `is_perfect_power`, radix conversion both
  ways, and `digit_count` without producing the digits.
- **`MontgomeryContext`, `BarrettContext`** — fixed-modulus reduction.
  The Montgomery domain (Montgomery 1985, in the separated-operand-scanning
  shape of Koç, Acar & Kaliski) encodes once and computes in-domain:
  `to_residue`, `mul_residue`, `square_residue`, and their `_with` forms
  reuse a `MontgomeryScratch`; `add_residue`, `sub_residue`, and
  `pow_residue` keep values in-domain. Residues are tied to their context,
  and mismatches return `ContextMismatch`. `gcd_with_modulus` works on a
  residue without decoding. Barrett (HAC Algorithm 14.42) accepts either
  modulus parity. `Montgomery64` and `Montgomery128` use fixed-width
  residues for allocation-free one- and two-word arithmetic.
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
  and `real_roots`, which uses exact integer sign evaluations to locate
  roots and returns floating-point approximations or a `RealRootError`.
- **Lattices** — integral LLL (`lll_reduce`, `lll_reduce_delta`,
  `lll_reduce_form`; Cohen's Algorithm 2.6.3, exact Gram data throughout),
  Lagrange–Gauss reduction under a diagonal form, `bareiss_determinant`,
  and certified enumeration: `short_vectors_form` and
  `closest_vectors_form` return an `Enumeration` that says whether the
  search was exhausted, stopped at its visit limit, or met a numerical
  limit, and every returned vector is rechecked exactly.
- **`gf2`** — linear algebra over GF(2) for sieve matrices: singleton
  pruning, structured Gaussian elimination (`filter_merge`), a dense null
  space, Block Lanczos (Montgomery 1995) on a sparse matrix, its matrix
  products spread over retained workers, and block Wiedemann (Coppersmith
  1994) on the same matrix, on one host for now.
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

- The default build uses `#![forbid(unsafe_code)]`. With `wipe`, the crate
  uses `deny(unsafe_code)` and allows the volatile scrub helper and its
  raw read-back test.
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

[PERFORMANCE.md](PERFORMANCE.md) is the per-primitive report: pilot-bench
means with confidence intervals and variable-time extrema over random
operands, fitted complexity exponents, log–log scaling graphs, and a
per-primitive comparison against GMP through the same harness on an NVIDIA
GB10 (one Cortex-X925 core), an Apple M4 Pro, an AMD EPYC 7452, a Raspberry
Pi 5 and an Apple A18 Pro. The GB10 results were measured with pilot-bench
f01eec4, the others with earlier pilot-bench builds. The tables are generated
from the committed data under `bench/`.

The GMP ratios compare whole implementations on particular machines. They
include differences in assembly, algorithms, thresholds, allocation, and
scheduling; they do not isolate the benefit of assembly. Source comments
record measurements behind the tuned crossovers.

Regenerate the data with `scripts/bench_primitives.sh` (rump and, via
`pilot_gmp`, GMP) and the document with `scripts/build_performance.sh`;
`cargo run --release --bin bench_bigint` prints ns/op for the core kernels.

## Manual

[MANUAL.md](MANUAL.md) describes the public API with worked examples.
Generated rustdoc contains the declarations and their current contracts.
Worked examples are also exercised in `tests/manual_examples.rs`. Changes
to the manual and those tests must be kept in sync.

[manual.tex](manual.tex) (built copy: `manual.pdf`) is the LaTeX reference
manual — the same surface with the defining mathematics for each
primitive; `scripts/check_manual_tex.sh` extracts its listings and
executes them against the crate, and a rebuild is gated on that passing.

[CITATIONS.md](CITATIONS.md) is the primary-source reference list: every
non-schoolbook algorithm in the crate with the paper, book, or standard it
comes from.

## Testing

Differential suites compare division with a bit-serial oracle, Montgomery
exponentiation with a division-based ladder, and Jacobi symbols with SageMath
vectors and Euler's criterion. These exercise different algorithms, though
some use the same underlying integer operations. Mutation tests and their
results are documented beside the relevant tests.

CI tests the default and `wipe` builds on Linux, macOS, and Windows on x86_64
and aarch64, with a separate i686 job. `scripts/release_gate.sh` runs local
format, lint, documentation, test, manual, packaging, MSRV, and cross-build
checks. It does not reproduce all of CI's operating-system jobs locally.

## Naming

The repository and library are `rump`; the crates.io package is `rust-mp`
(the bare name is taken by an unrelated tool). Depend on `rust-mp` and write
`use rump::...`.

## License

BSD-2-Clause.
