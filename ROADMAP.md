# ROADMAP

A proposed long-term scope for rump: a general computational number-theory
toolkit from which cryptographic constructions can be built — not
implementations of RSA, elliptic-curve protocols, signatures, padding, or key
formats. Proposed 2026-08-12; **to be triaged together before any of it is
built**. The *Assessment* notes are argument input for that triage, not
decisions. Legend for the *In rump today* notes: present, partial, absent.

## 1. Arbitrary-precision integers

Core: `add`, `sub`, `mul`, `square`, `div_rem`, `mod` (canonical in `[0, m)`),
`neg`, `abs`, `compare`, `sign`, `bit_length`, `test_bit`, `set_bit`,
`shift_left`, `shift_right`, `popcount`, `trailing_zeros`.

Integer functions: `sqrt_floor`, `sqrt_rem`, `nth_root_floor`, `is_square`,
`is_perfect_power`, `log_floor`.

Representations: `from_bytes` / `to_bytes` (endianness, signedness, length),
`from_string` / `to_string` (radix).

Multiplication backends: schoolbook, Karatsuba, Toom-3, Toom-4, FFT or NTT.

*In rump today:* core ops and byte I/O largely present; `sqrt_floor` present.
Absent: `sqrt_rem`, `nth_root_floor`, `is_square`, `is_perfect_power`,
`log_floor`, `popcount`, `trailing_zeros`, radix string I/O. Backends through
Toom-4 are implemented and threshold-tuned; FFT/NTT absent.

*Assessment:* FFT/NTT only pays above roughly 200 kbit even in GMP, higher in
pure Rust — worth building only if the toolkit routinely handles operands at
that scale (factorization and lattice work eventually would). The rest of
this section is inexpensive and uncontroversial.

## 2. Divisibility and Euclidean algorithms

`divides`, `gcd`, `binary_gcd`, `extended_gcd`, `lcm`, `coprime`, `gcd_many`,
`batch_gcd`, `valuation`, `remove_factor`.

*In rump today:* `gcd` (Lehmer → Half-GCD), `gcd_extended` (same), `lcm`
present. Absent: integer `binary_gcd` (the GF(2) polynomial form exists),
`gcd_many`, `batch_gcd`, `valuation`, `remove_factor`.

*Assessment:* `batch_gcd` (product/remainder trees) is genuinely valuable —
it is the classic shared-factor audit over key collections — and its tree
machinery is shared with §4 and §9. Low-risk section.

## 3. Modular integers

Basic: `mod_add`, `mod_sub`, `mod_neg`, `mod_mul`, `mod_square`, `mod_pow`,
`mod_inverse`, `mod_div`, `is_invertible`.

Variants: fixed-base exponentiation, multi-exponentiation, negative-exponent
handling, batch inversion, constant-time exponentiation, variable-time
exponentiation, Montgomery reduction, Barrett reduction, pseudo-Mersenne
reduction. An initialized `ModulusContext(m)` with `reduce`, `add`, `mul`,
`square`, `inverse`, `pow`.

*In rump today:* the basic set is present; `MontgomeryCtx` is the modulus
context for odd moduli (encode once, `mul_mont`/`square_mont`/`pow`). Absent:
Barrett (which would give a context for even moduli), pseudo-Mersenne,
multi-exponentiation, fixed-base tables, batch inversion.

*Assessment:* constant-time exponentiation contradicts rump's documented
contract ("variable-time; do not use where timing must not leak secrets").
Supporting it honestly means a deliberately separate constant-time surface
with its own tests and no shared variable-time paths — a real design
decision to make at triage, not a checkbox. Batch inversion (Montgomery's
trick, one inversion for n elements) is cheap to add and widely useful.

## 4. Congruences and CRT

`solve_linear_congruence`, `crt_pair`, `crt`, `generalized_crt` (non-coprime),
`crt_precompute` / `crt_reconstruct`, `symmetric_residue`,
`rational_reconstruction`, Garner reconstruction, product trees, remainder
trees.

*In rump today:* `crt_combine` (Garner, coprime moduli). The rest absent.

*Assessment:* `rational_reconstruction` is a direct corollary of the
extended-gcd engine just built (stop Euclid at the √m boundary) — near-free
now, and it unlocks exact linear algebra (§11). Good early candidate.

## 5. Exponentiation and multiplicative structure

`multiplicative_order`, `primitive_root`, `is_primitive_root`,
`carmichael_lambda`, `euler_phi`, `reduced_residue_system`, `powers_mod`,
`discrete_log_bsgs`, `discrete_log_pollard_rho`.

*In rump today:* absent (λ appears only implicitly, as `lcm(p−1, q−1)` in the
consumer crate).

*Assessment:* `multiplicative_order`, `primitive_root`, `euler_phi`, and
`carmichael_lambda` all require factoring their argument (of n, p−1, or λ's
prime-power parts), so this section is gated on §9 — sequence it after
factorization, or scope the functions to accept a supplied factorization.
The small-parameter discrete logs (BSGS, Pollard rho) are fine as stated:
testing tools, honest about their range.

## 6. Symbols and residuosity

`legendre_symbol`, `jacobi_symbol`, `kronecker_symbol`,
`is_quadratic_residue`, `quadratic_character`, `hilbert_symbol`.

*In rump today:* Legendre, Jacobi, Kronecker present (no factoring required).
Absent: Hilbert symbol, explicit residuosity wrappers.

*Assessment:* the standing performance item lands here too — the subquadratic
Jacobi (Brent–Zimmermann, ANTS 2010). Hilbert symbol needs p-adic valuations
(§2's `valuation`); modest.

## 7. Modular roots

`mod_sqrt` (Tonelli–Shanks, Cipolla, `p mod 4` / `p mod 8` special cases),
`mod_sqrt_prime_power`, `mod_sqrt_composite`, `mod_nth_roots`, Hensel
lifting, root verification; all roots returned where appropriate.

*In rump today:* `sqrt_mod` for odd primes (Tonelli–Shanks with the
`p ≡ 3 (mod 4)` shortcut, result verified by squaring; one root). The rest
absent.

*Assessment:* Cipolla is worth having alongside Tonelli–Shanks (it wins when
the 2-adic valuation is high — exactly the heavy tail our benchmarks
document). `mod_sqrt_composite` requires the factorization of the modulus by
definition; gate on §9 or take the factorization as an argument.

## 8. Prime operations

Sieving: `sieve`, `segmented_sieve`, `prime_iterator`, `small_prime_table`.
Testing: `trial_division`, `miller_rabin`, `strong_lucas_test`,
`baillie_psw`, `is_probable_prime`, `is_prime`, `next_prime`,
`previous_prime`. Proofs: Lucas, Pocklington, Proth, Pépin, ECPP, Pratt
certificates, `prove_prime`, `verify_prime_certificate`. Enumeration:
`prime_pi`, `nth_prime`, `primes_in_interval`.

*In rump today:* fixed-base Miller–Rabin (deterministic below 2⁶⁴ per
Sorenson–Webster), the 168-prime trial-division table,
`random_probable_prime`. The rest absent.

*Assessment:*
- **Baillie–PSW belongs near the top of any triage**: it is the standard
  probable-prime test (no known pseudoprime), cheap to add on the existing
  Miller–Rabin plus a strong Lucas test, and immediately upgrades
  `is_probable_prime`.
- `nth_prime` and `prime_pi` for large arguments require analytic counting
  (Meissel–Lehmer / Lagarias–Miller–Odlyzko) — a substantial project whose
  value to this toolkit is doubtful. Propose: sieve-ranged versions only
  (correct, honest about scope), drop the general forms.
- ECPP is a research-grade implementation (months, with class-group
  machinery). Pratt certificates, Pocklington, Proth, and Pépin are all
  feasible and give `prove_prime` for the forms that matter in practice;
  propose deferring ECPP indefinitely.

## 9. Factorization

Basic: trial division, wheel, Fermat, Pollard rho, Pollard p−1, Williams
p+1. Advanced: ECM, CFRAC, quadratic sieve, MPQS, NFS interface. Utilities:
`factor`, `factor_partial`, `is_smooth`, `smooth_part`, `divisors`,
`divisor_count`, `divisor_sum`, `squarefree_part`, `radical`; an explicit
`Factorization { sign, factors, unfactored_cofactor }` so partial results
cannot masquerade as complete.

*In rump today:* absent.

*Assessment:* the explicit `Factorization` type with an unfactored cofactor
is the best idea in this section — adopt it whatever else survives triage.
Basic methods through Pollard rho / p−1 and ECM are tractable and cover most
real use. Quadratic sieve is a serious but bounded project; NFS is not — an
*interface* to external tooling (CADO-NFS) is the honest scope there. CFRAC
is historically interesting but dominated by MPQS; propose dropping it.

## 10. Arithmetic functions

φ, λ, μ, Liouville, Mangoldt, radical, τ, σ_k, π(n), Fibonacci/Lucas,
Bernoulli numbers, factorials and binomials, primorials, rising/falling
factorials, CRT-friendly binomial computations.

*In rump today:* absent.

*Assessment:* everything factorization-dependent gates on §9. Bernoulli
numbers at large index are their own literature (Harvey's multimodular
algorithm); propose moderate-index scope. π(n) shares §8's analytic-counting
caveat.

## 11. Integer and rational matrices

Exact arithmetic, determinant, rank, kernel, Gaussian elimination over ℤ, ℚ,
F_p, Hermite and Smith normal forms, Bareiss, modular determinant, rational
reconstruction, characteristic polynomial, linear solving, sparse matrices,
Block Wiedemann, Block Lanczos.

*In rump today:* absent.

*Assessment:* dense exact linear algebra through HNF/SNF/Bareiss is a
well-bounded project and is what §14 (LLL) and algebraic experiments
actually need. Block Wiedemann/Lanczos exist to solve the giant sparse
systems of QS/NFS — build them only if §9's advanced methods survive triage
at a scale that needs them.

## 12. Polynomials

Arithmetic (`poly_add` … `poly_primitive_part`), fast algorithms (Karatsuba/
Toom, NTT, Newton inversion, multipoint evaluation, interpolation, product/
remainder/subproduct trees), factorization (square-free, distinct-degree,
equal-degree, Berlekamp, Cantor–Zassenhaus, integer factorization,
irreducibility).

*In rump today:* only GF(2)[x] internals (the `Gf2m` kernels and Rabin
irreducibility).

*Assessment:* large but well-trodden; polynomial arithmetic over F_p is also
the substrate for §13's extension fields and much of §8/§9. A natural second
pillar after the integer layer.

## 13. Finite fields

`PrimeField(p)` and `ExtensionField(base, irreducible)` with the full
operation set: arithmetic, inversion, exponentiation, Frobenius, trace, norm,
minimal polynomial, orders, roots, primitive elements, irreducible
generation, isomorphisms, bases. Binary fields additionally: carryless
multiplication, polynomial-basis reduction, squaring, half-trace, trace
solving.

*In rump today:* `Gf2m` covers the binary-field list; `MontgomeryCtx` is a
partial `PrimeField` context. General extension fields absent (gated on
§12).

## 14. Lattices

Gram–Schmidt (exact and floating), LLL, BKZ, CVP helpers, SVP enumeration,
Babai nearest-plane, integer kernel, determinant/volume, dual lattice.

*In rump today:* absent.

*Assessment:* LLL over exact rationals is feasible and delivers most of the
analytical value (Coppersmith-style experiments, knapsack attacks). BKZ done
credibly means floating-point orchestration and enumeration tuning at
research grade — propose LLL first and treat BKZ as its own later decision.

## 15. Continued fractions

Expansion, convergents, semiconvergents, best rational approximation, Pell
solver, rational reconstruction, quadratic-irrational expansions.

*In rump today:* absent (though the continued-fraction view of Euclid is
exactly what the Half-GCD work formalizes — the machinery is adjacent).

## 16. Combinatorial and low-level utilities

Factorials, binomials/multinomials, partitions, Cartesian products,
combination and Gray-code enumeration, bit vectors, Hamming weight/distance,
constant-weight enumeration, exact sampling from finite sets.

*In rump today:* absent beyond `random_below`-family sampling.

## Minimum useful first release (as proposed)

1. Arbitrary-precision integers — **largely present**
2. GCD and extended GCD — **present** (Lehmer + Half-GCD)
3. Modular contexts and exponentiation — **present** (Montgomery; Barrett absent)
4. Modular inverse and square roots — **present** (one root, odd primes)
5. Jacobi and Kronecker symbols — **present**
6. CRT and rational reconstruction — **partial** (Garner; reconstruction absent)
7. Sieving and primality testing — **partial** (Miller–Rabin; BPSW absent)
8. Basic factorization — absent
9. Polynomial arithmetic over ℤ and F_p — absent
10. Prime and extension fields — **partial** (GF(2^m) only)
11. Exact matrices and linear algebra — absent
12. LLL — absent

Triage still decides everything; on the current codebase, the shortest path
through this list runs 6 → 7 → 8 → 9 → 10 → 11 → 12, with
rational reconstruction and Baillie–PSW as the first two steps.
