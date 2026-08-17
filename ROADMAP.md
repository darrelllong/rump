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
5. **Radix string I/O** — largely done: `from_str_radix`/`to_str_radix`
   exist with divide-and-conquer conversion above measured crossovers. What
   remains of this item is tuning at very large sizes, not the existence of
   radix I/O.
6. **Integer predicates** — completed 2026-08-14. `sqrt_rem` and
   `sqrt_floor` by Newton's certified iteration (150× the bisection they
   replace at 8 kbit), `nth_root_floor`, `is_square` (enumerated residue
   filters), `is_perfect_power`, `popcount`, `trailing_zeros`, `pow_u64`,
   and `valuation`/`remove_factor` by a squared-power ladder. Verified
   against machine-integer and Python oracles, 55,000+ cases in review.
7. **Batch inversion** — completed 2026-08-14. `mod_inverse_batch`:
   one inversion and 3(n−1) multiplications for n elements (Montgomery,
   Math. Comp. 48 (1987)); measured 1.5× at a batch of two, levelling
   near 3.4× — the ratio of one Lehmer inversion to three multiplications.
8. **Cipolla's square root** — completed 2026-08-14. Dispatched inside
   `sqrt_mod` when s² > 4·bits (crossings measured at s ≈ 70/93/124 for
   1024/2048/4096 bits by a both-engines probe); the descent's s² tail is
   capped at the crossover cost. Both engines' non-residue scans are
   bounded, so composite moduli — odd squares included — yield None.
9. **Barrett reduction context** — completed 2026-08-14. `BarrettCtx`
   (HAC 14.42/14.44) for either parity: reduce/mul_mod/square_mod/pow_mod.
   The second product is HAC Note 14.45(ii)'s exact half-product as of
   2026-08-17, taken up to a measured 32 kbit: against a plain division
   `reduce` reads 1.4× at 512 bits, 1.26× at 1024 and 1.31× at 8192, where
   the full-product version trailed a division by up to a third at 2–4 kbit.
   At 256, 2048 and 4096 bits the distribution straddles parity — which of
   the two wins depends on the modulus, reproducibly, up to a third of
   sampled pairs favouring division — so those widths are reported as
   parity rather than given a figure. The half-product is quadratic, so above 32 kbit the
   dispatched full product wins and is used instead. Note 14.45(i)'s
   approximate *high* half remains open, and would trade exactness for a
   wider correction bound.

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

## The volatile drop scrub: measured, and kept (2026-08-16)

The second review (`SECOND-REVIEWER.md`, finding 1) would block a release until
the volatile scrub, the production `unsafe`, and the raw read-back test probe
are all removed and the crate adopts `#![forbid(unsafe_code)]`. Two of its
three sub-claims are correct as stated and one is an empirical question it
raised without answering; the disposition is to keep the scrub, correct the
documentation, and record the cost.

**Correct, and fixed.** `README.md` opened by calling the crate "pure, safe
Rust" while its own Properties section, sixty lines down, listed two audited
`unsafe` exceptions. The headline now names them.

**Correct, and stated rather than fixed.** `deny(unsafe_code)` is a default an
inner `allow` can override, not a boundary the compiler enforces against the
crate's own code; `forbid` is the enforcing form. The crate cannot use `forbid`
while it keeps a volatile scrub, because a volatile store requires a raw
pointer. That is a genuine trade and it is now written down in README rather
than left for a reader to discover.

**The empirical claim, now measured.** The review argues the scrub "charges all
arithmetic consumers ... for volatile writes on every temporary" and that
"multiprecision algorithms manufacture many temporaries, so this is exactly
where an unconditional extra memory pass is least welcome". Directionally right,
but it is not a uniform tax: the cost tracks the number of `BigUint` drops per
unit of arithmetic, and that ratio varies by two orders of magnitude across the
crate's own workloads.

A/B against the same tree with `Drop for BigUint` emptied, aarch64 (M4), release
build. Reduction is by **minimum** over 60 readings per arm, taken as four
alternating A/B rounds of 15 — not a mean: an initial A/B/A run showed 14% drift
between two *identical* A arms, which is the size of the effect being measured,
so a mean over that data would report the machine. The p25 column is given
because a single minimum is one reading; where the two disagree, the number is
not trustworthy.

| Workload | Scrub cost (min) | (p25) |
|---|---|---|
| 2000 short-lived 512-bit temporaries | +16.8% | +18.8% |
| product tree + remainder tree, 256 values | +10.9% | +8.9% |
| `mod_inverse_batch`, 512 values | +5.0% | +1.3% |
| `gcd`, 4096 bits | +2.1% | +2.1% |
| `mod_pow`, 2048 bits | +1.4% | +1.0% |
| `PolyZ::mul`, degree 64 | −1.1% | +0.1% |

So: on compute-bound work — modular exponentiation, gcd, polynomial
multiplication, which is what the crate is mostly asked to do — the scrub is at
or below the noise floor. On allocation-dominated shapes it is 9–19%. The
`mod_inverse_batch` row is the one to distrust: minimum and p25 disagree by a
factor of four, so it is somewhere in 1–5% and this run cannot say where.

One caution on the first attempt, recorded because it nearly produced a wrong
number: an initial run reported the 2048-bit `mod_pow` arm 16–22% faster without
the scrub. That was warmup, not scrub cost — the readings within each A series
declined monotonically from 5.96 ms to 3.32 ms. Alternating the arms and taking
minima put the same figure at 1.4%.

The scrub stays. It is documented as defense in depth rather than a security
property, the crate already says cryptographic hygiene belongs at the consumer,
and the measured cost on the operations this crate exists to perform does not
justify removing a hygiene measure the parent crate audited. Revisit if a
consumer's profile is dominated by temporary churn.

## 1. Arbitrary-precision integers

**Efficient — all of it.** Core ops (`add` … `trailing_zeros`), integer
functions (`sqrt_floor`, `sqrt_rem`, `nth_root_floor` and `is_square` /
`is_perfect_power` by Newton iteration, `log_floor` — its use case served
since 2026-08-16 by `digit_count(radix)` = floor(log_r n) + 1, with
`bits()` the base-2 form; a bare floor-log remains unimplemented), radix
string I/O, and
signed byte I/O. Already present: the core ops, byte I/O, `sqrt_floor`, and
the multiplication ladder through Toom-4.

**Bounded:** FFT/NTT multiplication is efficient but only pays above roughly
200 kbit; build it when the toolkit's own workloads (factorization, lattices)
routinely live there, not before.

## 2. Divisibility and Euclidean algorithms

**Efficient — all of it.** `divides`, `binary_gcd`, `coprime`, `gcd_many`,
and `batch_gcd` by product/remainder trees
(O(M(N)·log) over the whole input; the classic shared-factor audit). Already
present: `gcd`, `gcd_extended` (Lehmer → Half-GCD), `lcm`, `valuation`,
`remove_factor`, `gcd_u64`, and the trees themselves (`product_tree`,
`remainder_tree`, public since 0.2.1).

## 3. Modular integers

**Efficient — all of it** except one design question. The basic set is
present, with Barrett reduction (queue item 9) and Montgomery's batch
inversion (queue item 7) landed 2026-08-14; still open: pseudo-Mersenne
reduction, fixed-base tables, Straus/Pippenger multi-exponentiation.

**Design gate, not an efficiency gate:** constant-time exponentiation
contradicts the crate's documented variable-time contract. Supporting it
honestly means a separate constant-time surface with its own tests and no
shared variable-time paths — decide that deliberately at triage.

## 4. Congruences and CRT

**Efficient — all of it.** `solve_linear_congruence`, `crt_pair`, `crt`,
generalized CRT for non-coprime moduli, precompute/reconstruct contexts,
`symmetric_residue`, and **rational
reconstruction** — landed 2026-08-14 (queue item 4), the key that unlocks
exact linear algebra (§11). Present: `crt_combine` (Garner),
`rational_reconstruct`, `rational_reconstruct_bounded`, and the
product/remainder trees (public since 0.2.1, listed under §2).

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

**Efficient (prime modulus):** Tonelli–Shanks and Cipolla (both present —
Cipolla landed 2026-08-14 as queue item 8, dispatched where the descent's
2-adic tail lives); the mod-8 special cases, Hensel lifting to prime
powers, and all-roots return landed with `sqrt_mod_prime_power` (0.2.1);
still open: nth roots mod p.

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
of §8; the natural second pillar after the integer layer. Present:
`PolyZ`/`PolyModP` with exact and pseudo-division, resultant, discriminant,
the full squarefree/distinct-degree/Cantor–Zassenhaus pipeline,
`is_irreducible`, and `roots` (0.2.1), alongside the GF(2)[x] internals
behind `Gf2m`; still open here: the asymptotic upgrades (Newton inversion,
multipoint evaluation, subproduct trees), Berlekamp, and van Hoeij.

## 13. Finite fields

**Efficient — all of it.** `PrimeField(p)` (a completion of what
`MontgomeryCtx` starts), `ExtensionField` on §12's polynomial arithmetic,
Frobenius, trace, norm, minimal polynomials, roots, irreducible-polynomial
generation, isomorphisms, bases. Present: the binary-field list, complete,
in `Gf2m`.

## 14. Lattices

**Efficient:** exact and floating Gram–Schmidt, **LLL** — present since
0.2.1 (`lll_reduce`, `lll_reduce_delta`, Cohen's integral Algorithm 2.6.3,
tested against an independent rational oracle); it delivers most of the
analytical value (Coppersmith-style experiments, knapsack attacks). Still
open here: Babai nearest-plane, integer kernel, determinant/volume, dual
lattice.

**Out / bounded:** SVP/CVP are NP-hard — enumeration helpers only, explicitly
dimension-bounded. BKZ done credibly is research-grade floating-point
orchestration; LLL shipped first (0.2.1, above) exactly as this section
once proposed, and BKZ remains its own later decision.

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
3. Modular contexts and exponentiation — **present** (Montgomery and Barrett)
4. Modular inverse and square roots — **present** (`sqrt_mod` one root mod an odd prime; `sqrt_mod_prime_power` all roots mod `p^e`, `p = 2` included)
5. Jacobi and Kronecker symbols — **present**
6. CRT and rational reconstruction — **present** (Garner; reconstruction landed 2026-08-14)
7. Sieving and primality testing — **partial** (Miller–Rabin, BPSW, and bulk
   `primes_below`; *segmented/ranged* sieving absent)
8. Basic factorization — absent (deliberately: it lives downstream, see
   REQUESTS.md's boundary)
9. Polynomial arithmetic over ℤ and F_p — **present** (0.2.1: `PolyZ`,
   `PolyModP`, resultant/discriminant, factorization, roots)
10. Prime and extension fields — **partial** (GF(2^m) only)
11. Exact matrices and linear algebra — absent
12. LLL — **present** (0.2.1: `lll_reduce`, `lll_reduce_delta`, integral form)

Triage still decides everything; of the shortest path this list once named
(6 → 7 → 8 → 9 → 10 → 11 → 12), items 6, 9, and 12 are done and 8 stays
downstream by design — what remains is the rest of 7 (segmented sieving)
and 10 → 11.
