# ROADMAP

A proposed long-term scope for rump: a general computational number-theory
toolkit from which cryptographic constructions can be built — not
implementations of RSA, elliptic-curve protocols, signatures, padding, or key
formats. Proposed 2026-08-12; **to be triaged together before any of it is
built**.

## Approved work queue (2026-08-13)

Items 1–9 of the priority table are approved and ordered; everything below
them remains subject to the joint triage.

1. **Half-GCD-threaded Jacobi** — completed 2026-08-13. The symbol state
   threads through the hgcd recursion (every applied quotient materializes
   in a Lehmer batch or a guarded division); crossover measured at 2048
   limbs, the same as gcd's. At 1 Mbit the ratio to GMP fell 41× → 11.9×
   and the symbol now costs what this crate's gcd costs, equal to within
   0.1% (PERFORMANCE.md, *GCD at scale*).
2. **In-place `add`/`sub`** — completed 2026-08-13. Three-operand
   `assign_add`/`assign_sub` on the unsigned type, sign-complete in-place
   `add_assign_ref`/`sub_assign_ref` on the signed type, and a
   buffer-reusing `clone_from`, every shrinking path scrubbing the limbs it
   abandons; the benchmark's `add`/`sub` rows now measure reused-output
   writes on both sides, and the ratios fell from 2–15× to 1.2–4.5×
   (PERFORMANCE.md).
3. **Baillie–PSW** — completed 2026-08-14. Strong Lucas with Selfridge's
   Method A (Montgomery-domain ladder) composed with one strong base-2
   Miller–Rabin round behind the shared trial-division screen
   (`is_probable_prime_bpsw`, `is_strong_lucas_probable_prime`). Verified
   against a sieve oracle sharing no code with the tests, both published
   pseudoprime tables below 10⁵, and structured adversarial cases; no
   composite passing both stages is known, none exists below 2⁶⁴.
4. **Rational reconstruction** — completed 2026-08-14. The extended
   Euclidean walk stopped at the first remainder within the numerator
   bound (von zur Gathen and Gerhard §5.10; Wang's technique),
   Lehmer-batched with single-step exactness at the stop line;
   `rational_reconstruct` (symmetric √((m−1)/2) bounds) and
   `rational_reconstruct_bounded` (explicit bounds, 2·N·D < m enforced).
   Verified against a naive uniqueness search exhaustively and by planted
   round trips to 4096 bits.
5. **Radix string I/O** — `from_string`/`to_string` by divide-and-conquer
   conversion.
6. **Integer predicates** — `is_square`, `is_perfect_power`,
   `nth_root_floor`, `sqrt_rem`, `popcount`, `trailing_zeros`, `valuation`,
   `remove_factor`.
7. **Batch inversion** — Montgomery's trick: one inversion for n elements.
8. **Cipolla's square root** — complements Tonelli–Shanks exactly where its
   measured heavy tail lives (high 2-adic valuation of p − 1).
9. **Barrett reduction context** — a modulus context for even moduli, where
   Montgomery cannot operate.

Each item lands with the established discipline: primary source, independent
oracle, differential verification, measured thresholds where dispatch is
involved, and documentation in the formal register.

Triage ordering rule (2026-08-15): everything gated on factorization — the
**Gated** entries below — is deferred to the end of the plan; the
factorization-free surface lands first.

Each section below is scrubbed by implementability. **Efficient** means softly
polynomial in the operand bit-length with a literature algorithm we can cite
and implement at craftsman level — the standard the rest of the crate holds.
**Gated** means efficient only given something expensive (almost always a
factorization), so the honest API takes that input as a parameter. **Out /
bounded** names the items that are intrinsically expensive — exponential or
subexponential by nature, enumeration-shaped, or research-grade engineering —
proposed for exclusion or for an explicitly bounded scope.

## 1. Arbitrary-precision integers

**Efficient — all of it.** Core ops (`add` … `trailing_zeros`), integer
functions (`sqrt_floor`, `sqrt_rem`, `nth_root_floor` and `is_square` /
`is_perfect_power` by Newton iteration, `log_floor`), radix string I/O, and
signed byte I/O. Already present: the core ops, byte I/O, `sqrt_floor`, and
the multiplication ladder through Toom-4.

**Bounded:** FFT/NTT multiplication is efficient but only pays above roughly
200 kbit; build it when the toolkit's own workloads (factorization, lattices)
routinely live there, not before.

## 2. Divisibility and Euclidean algorithms

**Efficient — all of it.** `divides`, `binary_gcd`, `coprime`, `valuation`,
`remove_factor`, `gcd_many`, and `batch_gcd` by product/remainder trees
(O(M(N)·log) over the whole input; the classic shared-factor audit). Already
present: `gcd`, `gcd_extended` (Lehmer → Half-GCD), `lcm`.

## 3. Modular integers

**Efficient — all of it** except one design question. The basic set is
present; add Barrett reduction (a context for even moduli, where Montgomery
cannot go), pseudo-Mersenne reduction, fixed-base tables, Straus/Pippenger
multi-exponentiation, and Montgomery's batch inversion (one inversion for n
elements — nearly free to add and widely useful).

**Design gate, not an efficiency gate:** constant-time exponentiation
contradicts the crate's documented variable-time contract. Supporting it
honestly means a separate constant-time surface with its own tests and no
shared variable-time paths — decide that deliberately at triage.

## 4. Congruences and CRT

**Efficient — all of it.** `solve_linear_congruence`, `crt_pair`, `crt`,
generalized CRT for non-coprime moduli, precompute/reconstruct contexts,
`symmetric_residue`, product/remainder trees, and **rational
reconstruction** — landed 2026-08-14 (queue item 4), the key that unlocks
exact linear algebra (§11). Present: `crt_combine` (Garner),
`rational_reconstruct`, `rational_reconstruct_bounded`.

## 5. Exponentiation and multiplicative structure

**Efficient:** `powers_mod`; `multiplicative_order` and `is_primitive_root`
*given* the factorization of the group order — take it as a parameter.

**Gated on factoring (§9):** `euler_phi`, `carmichael_lambda`,
`primitive_root` for general moduli.

**Out / bounded:** `discrete_log_bsgs` and `discrete_log_pollard_rho` are
exponential by nature — keep them only as explicitly bounded small-parameter
testing tools, documented as such. `reduced_residue_system` is enumeration
(output size φ(n)); at most an iterator, never a materialized list.

## 6. Symbols and residuosity

**Efficient — all of it.** Present: Legendre, Jacobi, Kronecker (no factoring
needed). Add: `is_quadratic_residue` / `quadratic_character` wrappers, and
the Hilbert symbol (needs §2's `valuation`). The subquadratic Jacobi
(queue item 1) landed 2026-08-13.

## 7. Modular roots

**Efficient (prime modulus):** Tonelli–Shanks (present), Cipolla — worth
having because it wins exactly where Tonelli–Shanks's heavy tail lives (high
2-adic valuation, documented in our benchmarks) — the mod-4/mod-8 special
cases, Hensel lifting to prime powers, nth roots mod p, root verification,
and both-roots return.

**Gated on factoring:** `mod_sqrt_composite` requires the modulus's
factorization by definition (it *is* integer factoring in disguise) — take
the factorization as a parameter.

## 8. Prime operations

**Efficient:** ranged sieving (`sieve`, `segmented_sieve`, `prime_iterator`,
`primes_in_interval`), `trial_division`, Miller–Rabin (present), strong
Lucas and Baillie–PSW (present — queue item 3, landed 2026-08-14),
`next_prime` / `previous_prime`
(expected polynomial), and certificates for structured forms: Pratt,
Pocklington, Proth, Pépin, with `verify_prime_certificate`.

**Out / bounded:** `prime_pi` and `nth_prime` for large arguments need
analytic counting (Meissel–Lehmer / LMO) — a major project of doubtful value
here; keep sieve-ranged versions only. General `prove_prime` means ECPP —
research-grade with class-group machinery; defer indefinitely and say so.

## 9. Factorization

**Nature of the section:** factoring is subexponential at best — nothing
here is "efficient" in this document's sense, and that is fine *if the API
says so*. The explicit `Factorization { sign, factors, unfactored_cofactor }`
type is the best idea in the proposal — partial results can never masquerade
as complete — and should be adopted regardless of what else survives.

**Tractable, honest tools:** trial/wheel division, Fermat, Pollard rho,
Pollard p−1, Williams p+1, ECM, `factor_partial`, `is_smooth`,
`smooth_part`, and the factorization-parameterized utilities (`divisors`,
`divisor_count`, `divisor_sum`, `squarefree_part`, `radical`).

**Large or out:** the quadratic sieve is a serious but bounded project —
triage-worthy; MPQS likewise. CFRAC is dominated by MPQS — drop it. NFS is
out as an implementation; an interface to external tooling (CADO-NFS) is the
honest scope.

## 10. Arithmetic functions

**Efficient:** Fibonacci/Lucas (matrix or fast-doubling), factorials,
binomials/multinomials, primorials, rising/falling factorials, and
CRT-friendly binomial computation.

**Gated on factoring:** φ, λ, μ, Liouville, Mangoldt, radical, τ, σ_k for
general n — parameterize on a `Factorization`.

**Out / bounded:** π(n) at scale (§8's caveat); Bernoulli numbers beyond
moderate index are their own literature (Harvey's multimodular method) —
moderate-index scope only.

## 11. Integer and rational matrices

**Efficient:** dense exact linear algebra — Bareiss, Gaussian elimination
over ℤ/ℚ/F_p, determinant (including modular+CRT), rank, kernel, HNF, SNF,
characteristic polynomial, solving, rational reconstruction of solutions.
Well-bounded, and exactly what §14 and algebraic experiments need.

**Bounded:** Block Wiedemann / Block Lanczos exist to solve the giant sparse
systems of QS/NFS; build them only if §9's advanced methods survive triage
at a scale that needs them.

## 12. Polynomials

**Efficient — all of it.** Arithmetic, fast algorithms (Karatsuba/Toom, NTT,
Newton inversion, multipoint evaluation, interpolation, subproduct trees),
and factorization over F_p — square-free, distinct-degree, equal-degree,
Cantor–Zassenhaus (randomized polynomial), Berlekamp — plus integer
polynomial factorization via LLL (van Hoeij). The substrate for §13 and much
of §8; the natural second pillar after the integer layer. Present: only the
GF(2)[x] internals behind `Gf2m`.

## 13. Finite fields

**Efficient — all of it.** `PrimeField(p)` (a completion of what
`MontgomeryCtx` starts), `ExtensionField` on §12's polynomial arithmetic,
Frobenius, trace, norm, minimal polynomials, roots, irreducible-polynomial
generation, isomorphisms, bases. Present: the binary-field list, complete,
in `Gf2m`.

## 14. Lattices

**Efficient:** exact and floating Gram–Schmidt, **LLL** (polynomial time;
delivers most of the analytical value — Coppersmith-style experiments,
knapsack attacks), Babai nearest-plane, integer kernel, determinant/volume,
dual lattice.

**Out / bounded:** SVP/CVP are NP-hard — enumeration helpers only, explicitly
dimension-bounded. BKZ done credibly is research-grade floating-point
orchestration; propose LLL first and treat BKZ as its own later decision.

## 15. Continued fractions

**Efficient — all of it.** Expansion, convergents, semiconvergents, best
rational approximation, rational reconstruction (shared with §4), quadratic
irrationals, and the Pell solver (polynomial in its output size — the
fundamental solution can be exponentially long, which is the answer's size,
not the algorithm's fault). Adjacent machinery: the continued-fraction view
of Euclid is exactly what the Half-GCD work formalizes.

## 16. Combinatorial and low-level utilities

**Efficient:** factorials, binomials, bit vectors, Hamming weight/distance,
exact sampling, Gray codes.

**Bounded:** partitions and combination/constant-weight enumeration have
exponentially many outputs — iterators only, never materialized.

## Minimum useful first release (as proposed, with current standing)

1. Arbitrary-precision integers — **largely present**
2. GCD and extended GCD — **present** (Lehmer + Half-GCD)
3. Modular contexts and exponentiation — **present** (Montgomery; Barrett absent)
4. Modular inverse and square roots — **present** (one root, odd primes)
5. Jacobi and Kronecker symbols — **present**
6. CRT and rational reconstruction — **present** (Garner; reconstruction landed 2026-08-14)
7. Sieving and primality testing — **partial** (Miller–Rabin and BPSW; ranged sieving absent)
8. Basic factorization — absent
9. Polynomial arithmetic over ℤ and F_p — absent
10. Prime and extension fields — **partial** (GF(2^m) only)
11. Exact matrices and linear algebra — absent
12. LLL — absent

Triage still decides everything; on the current codebase, the shortest path
through this list runs 6 → 7 → 8 → 9 → 10 → 11 → 12. Its two named first
steps — Baillie–PSW and rational reconstruction — both landed 2026-08-14.
