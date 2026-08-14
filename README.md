# rump

**RU**st **M**ulti**P**recision: multiprecision integer arithmetic in pure,
safe Rust, implemented directly from the literature. Extracted from
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
  encode once, multiply and exponentiate in-domain (`mul_mont`,
  `square_mont`, `pow`, `pow_encoded`), convert at the boundary. Fixed 4-bit
  window exponentiation.
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
  ladder. The integer layer adds `sqrt_rem`/`sqrt_floor` (certified
  Newton), `nth_root_floor`, `is_square`, `is_perfect_power`, `popcount`,
  and `trailing_zeros`.

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

- `#![deny(unsafe_code)]`; the sole audited exception is a six-line
  volatile-write scrub helper.
- Every `BigUint` wipes its limbs on drop, and the exponentiation ladder
  wipes its workspaces on exit — values do not linger in freed heap memory.
- **Variable-time.** Operations take data-dependent paths. Do not use this
  crate where timing must not leak secrets.

## Benchmarks

[PERFORMANCE.md](PERFORMANCE.md) is the full per-primitive report: pilot-bench
means with confidence intervals and variable-time extrema over random operands,
log–log scaling graphs, fitted complexity exponents, and a per-primitive
comparison against GMP on three hosts — Apple M4, AMD EPYC 7452, and Raspberry
Pi 5. Regenerate the data with `scripts/bench_primitives.sh` (rump and, via
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
PERFORMANCE.md's "GCD at scale". The standing items are carrying that
transform through the Bézout cofactors (`gcd_extended`, `mod_inverse`) and the
Jacobi symbol.

## Manual

[MANUAL.md](MANUAL.md) documents every public API with a worked example.
Every code block in it is replicated in `tests/manual_examples.rs` and
asserted on `cargo test`, so the manual cannot drift from the code.

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
