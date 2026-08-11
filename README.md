# rump

**R**ust **MU**lti**P**recision: multiprecision integer arithmetic in pure,
safe Rust, implemented directly from the literature. Extracted from
[darrelllong/cryptography](https://github.com/darrelllong/cryptography) so the
arithmetic can serve consumers beyond cryptography, with the crate boundary
enforcing a clean API.

## What it provides

- **`BigUint`, `BigInt`** — unsigned and signed integers on little-endian
  `u64` limbs. Schoolbook and Karatsuba multiplication with a dedicated
  squaring kernel; Knuth's Algorithm D division (*TAOCP* vol. 2, §4.3.1) with
  a Horner path for single-limb divisors.
- **`MontgomeryCtx`** — a public Montgomery domain (Montgomery 1985; the
  separated-operand-scanning shape from Koç, Acar & Kaliski, IEEE Micro 1996):
  encode once, multiply and exponentiate in-domain (`mul_mont`,
  `square_mont`, `pow`, `pow_encoded`), convert at the boundary. Fixed 4-bit
  window exponentiation.
- **Number theory** — `gcd`, `lcm`, and `gcd_extended` (Bézout
  coefficients); the quadratic-residue symbols `jacobi` (binary reciprocity,
  HAC Algorithm 2.149), `legendre`, and `kronecker` (Cohen Algorithm
  1.4.10); `sqrt_mod` (Tonelli–Shanks with the `p ≡ 3 (mod 4)` shortcut,
  result verified by squaring); `mod_pow`, `mod_inverse`, and `crt_combine`
  (Garner, HAC Algorithm 14.71); fixed-base Miller-Rabin
  (`is_probable_prime`, `is_probable_prime_with_bases`) and the reusable
  per-round primitive `miller_rabin_witness` for callers that bring their
  own witness schedule.

- **`Gf2m`** — binary extension fields GF(2^m): XOR addition, shift-and-XOR
  multiplication, binary extended-GCD inversion (Hankerson–Menezes–Vanstone,
  *Guide to ECC*, Algorithm 2.22), and the half-trace that solves
  `z² + z = c` for binary-curve point decompression. The degree is derived
  from the field polynomial, never supplied alongside it.
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

`cargo run --release --bin bench_bigint` reports ns/op for the core kernels.
`bash scripts/bench_gmp.sh` runs the same table against GMP for an
apples-to-apples comparison (requires libgmp). Measured against GMP 6.3.0,
modular exponentiation at RSA sizes runs within 1.6–2× of GMP's
assembly-backed kernels on both x86-64 and Apple Silicon.

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
