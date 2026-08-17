# CITATIONS

Every non-schoolbook algorithm in rump, with the primary source cited in the
code that implements it. Each entry names the algorithm, where it lives, and
the reference as the source doc-comments state it. Schoolbook operations
(grade-school add/subtract, trial division, linear scans) carry no
citation and are not listed; see the note below on the one cited exception.

Verified against the source 2026-08-14. Rows added 2026-08-15 — the GF(2^m)
trace/quadratic/irreducibility/reduction/exponentiation rows, the randomness
rows, and the Montgomery-squaring / Barrett-exponentiation / extended-Euclid
/ modular-inverse rows found by auditing every doc-comment citation against
this table — are transcribed from the implementing doc-comments; their
physical-source verification is tracked on HANDOFF.md's citation-check
list where it is still owed. Rows verified 2026-08-14 may carry fuller
bibliographic detail (editions, page ranges, expanded titles) than the
terse in-code citation; that detail came from the verification pass, not
from the code. The convention throughout the crate:
name the algorithm, name its author, name the venue and year, and give the
section, algorithm number, or figure where the primary source is precise
about it. The inclusion criterion for classical results with no
bibliography: an eponymous algorithm the code implements (Horner) or an
eponymous theorem doing quantitative, load-bearing work (Lamé sizing a
buffer, Mertens bounding a tuning judgement) gets a row; terminology,
flavour references, and use-case asides (Hamming weight, Vandermonde,
"Sunzi's classic", Las Vegas) do not. Operations whose code carries no
citation (grade-school add/subtract, trial division, linear scans) are not
listed; the schoolbook multiply is listed, because its code does cite
Algorithm M.

---

## Arbitrary-precision integers (`src/bigint.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Schoolbook multiplication | `mul_schoolbook_ref` | Knuth, *TAOCP* vol. 2, §4.3.1, Algorithm M. |
| Single-limb division and radix accumulation | `div_rem_u64` / `rem_u64` (via `div_rem_limb`), the classical radix parse, the render base case in `to_digits_dc`, and `bigint_div_exact` (the Toom interpolation's exact division) | Horner's rule (classical), in base 2⁶⁴. |
| Division by an invariant divisor | `WordReciprocal` (`src/bigint/reciprocal.rs`); the two-word-by-one-word kernel is `WordReciprocal::div2by1`, and the reciprocal itself is built in `WordReciprocal::new` | Möller & Granlund, *Improved Division by Invariant Integers*, IEEE Transactions on Computers 60(2) (2011), 165–175 — Algorithm 4 (`div2by1`) over the reciprocal of Algorithm 2. Descends from Granlund & Montgomery, *Division by Invariant Integers using Multiplication*, PLDI 1994, 61–72. Transcribed; the algorithm and figure numbers are on the HANDOFF check list. |
| Long division, base 2⁶⁴ | `div_rem` (Knuth's Algorithm D) | Knuth, *The Art of Computer Programming*, vol. 2 (*Seminumerical Algorithms*), §4.3.1, Algorithm D; the q̂ ≤ q + 1 estimate bound is §4.3.1, exercise 20. Add-back and borrow mechanics after Warren, *Hacker's Delight*, 2nd ed., §9-2 (`divmnu`). |
| Karatsuba multiplication | `mul_karatsuba_ref` (dispatched from `mul_ref`) | Karatsuba & Ofman, *Multiplication of Multidigit Numbers on Automata*, Soviet Physics–Doklady 7 (1963); Knuth, *TAOCP* vol. 2, §4.3.3. |
| Integer squaring (schoolbook) | `sqr_schoolbook_ref` (behind `square_ref`) | *Handbook of Applied Cryptography*, Algorithm 14.16 (cross terms once, then double, then the diagonal). |
| Integer squaring (Karatsuba) | `sqr_karatsuba_ref` | the Karatsuba split with all three sub-products squares; Karatsuba & Ofman as above. |
| Toom-3 / Toom-4 multiplication | `mul_toom3_ref`, `mul_toom4_ref` | Knuth, *TAOCP* vol. 2, §4.3.3 (Toom–Cook, the generalization of Karatsuba); interpolation sequence after Bodrato, *Towards Optimal Toom–Cook Multiplication for Univariate and Multivariate Polynomials in Characteristic 2 and 0*, WAIFI 2007. |
| Integer square root (Newton) | `sqrt_rem` / `sqrt_floor` | Cohen, *A Course in Computational Algebraic Number Theory*, Algorithm 1.7.1. |
| nth root (Newton) | `nth_root_floor` | Cohen, Algorithm 1.7.1 for the square case (§1.7.1, Integer Square Roots); the general `k` is the crate's own generalization by the same Newton iteration, the AM–GM inequality on `k` terms giving the over-estimate. |
| Perfect-square residue filters | `is_square` | classical residue filter set (as in GMP's `mpz_perfect_square_p`); tables derived by enumeration. |
| Radix conversion (divide and conquer) | `from_str_radix` / `to_str_radix` | Knuth, *TAOCP* vol. 2, §4.4; Brent and Zimmermann, *Modern Computer Arithmetic*, §1.7. |
| Three-operand add/sub (buffer reuse) | `assign_add`, `assign_sub` | the shape of GMP's `mpz_add` (design analogue, not an algorithm source). |
| f64 / ln estimates | `to_f64_lossy`, `ln_approx` | top-64-bit mantissa extraction; no external algorithm. |

## Montgomery and Barrett modular reduction (`src/bigint.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Montgomery multiplication | `MontgomeryContext`, `mul_mont` | Montgomery, *Modular Multiplication Without Trial Division*, Math. Comp. 44 (1985), 519–521. Separated-operand-scanning shape from Koç, Acar & Kaliski, *Analyzing and Comparing Montgomery Multiplication Algorithms*, IEEE Micro 16(3) (1996), 26–33. |
| Montgomery squaring | `mont_sqr` (behind `square_mont`) | *Handbook of Applied Cryptography*, Algorithm 14.16 (cross terms once, then double, then the diagonal). |
| Word-level Montgomery constant `n₀′` | `montgomery_n0_inv` | Dussé & Kaliski, *A Cryptographic Library for the Motorola DSP56000*, EUROCRYPT '90 — the word-level variant of Montgomery reduction; the "where it was introduced" attribution is on the HANDOFF check list. |
| Low half-product | `mul_low_ref` (behind `BarrettContext::reduce`) | *Handbook of Applied Cryptography*, Note 14.45(ii): Barrett's second multiplication needs only the low `k+1` limbs, and forming only those is exact. |
| Barrett reduction | `BarrettContext`, `reduce` | Barrett, CRYPTO '86; *Handbook of Applied Cryptography*, Algorithm 14.42, correction bound Note 14.44 (half-product refinement Note 14.45(ii) and (iii), also Brent and Zimmermann, *MCA*, §2.4). |
| Barrett exponentiation | `BarrettContext::pow_mod` | left-to-right binary exponentiation, Knuth, *TAOCP* vol. 2, §4.6.3. |

## GCD and the Euclidean family (`src/number_theory.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Lehmer's gcd (double-digit) | `gcd_lehmer`, `gcd_extended_lehmer` | Knuth, *TAOCP* vol. 2, §4.5.2, Algorithm L, on aligned leading digits. |
| Quotient-batch buffer bound | `QuotientLog` | Lamé's theorem (classical): F₁₈₁ > 2¹²⁴ > F₁₈₀ caps a 124-bit window at 178 Euclidean steps, which sizes the fixed buffer. |
| Half-GCD (subquadratic) | `hgcd`, `gcd_via_hgcd`, `gcd_extended_via_hgcd` | Möller, *On Schönhage's algorithm and subquadratic integer gcd computation*, Math. Comp. 77 (2008), 589–607, Figure 4 (the algorithm behind GMP's `mpn_hgcd`); Equation 4, Lemmas 5–7, and §6.3 are cited at the specific steps they justify, and `HGCD_BASE_LIMBS` / `hgcd_base` are the analogues of GMP's `HGCD_THRESHOLD` / `hgcd2` loop. |
| Extended Euclid (Bézout cofactors) | `gcd_extended` | *Handbook of Applied Cryptography*, Algorithm 2.107; Knuth, *TAOCP* vol. 2, §4.5.2, Algorithm X. |
| Modular inverse | `mod_inverse`, `mod_inverse_u64` (the word-sized form, Bézout coefficients carried in `i128`) | *Handbook of Applied Cryptography*, Algorithm 2.142, reduced into `[0, n)`. |

## Symbols and residuosity (`src/number_theory.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Jacobi symbol, binary reciprocity | `jacobi_binary` | *Handbook of Applied Cryptography*, Algorithm 2.149. |
| Jacobi symbol, Lehmer-batched state | `jacobi_lehmer`, state machine | Möller's design (as shipped in GMP: `mpn_jacobi_n`, `gen-jacobitab.c`), after Schönhage's identities. |
| Jacobi symbol, HGCD-threaded (subquadratic) | `jacobi_hgcd` | Brent and Zimmermann, *An O(M(n) log n) algorithm for the Jacobi symbol*, ANTS-IX, LNCS 6197 (2010), 83–95; threading design Möller's (GMP's `mpn_hgcd_jacobi`); `JACOBI_HGCD_THRESHOLD_LIMBS` sits where GMP pins its `JACOBI_DC_THRESHOLD`. |
| Kronecker symbol | `kronecker` | Cohen, *A Course in Computational Algebraic Number Theory*, Algorithm 1.4.10. |

## Modular roots and exponentiation (`src/number_theory.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Tonelli–Shanks square root | `sqrt_mod_descent` | Cohen, Algorithm 1.5.1; the p ≡ 3 (mod 4) shortcut, *HAC* §3.5.1. |
| Cipolla's square root | `sqrt_mod_cipolla` | Cipolla, *Un metodo per la risoluzione della congruenza di secondo grado*, Rend. Accad. Sci. Fis. Mat. Napoli (3) 9 (1903), 153–163. |
| Prime-power square roots (Hensel lift) | `sqrt_mod_prime_power` | Hensel's lemma (Cohen, §3.5.3, "Factorization Modulo pᵉ: Hensel's Lemma"); dyadic and valuation cases per the standard structure of squares in ℤ/2^eℤ. |
| Windowed Montgomery exponentiation | `MontgomeryContext::pow` | fixed 4-bit window (left-to-right k-ary method), *Handbook of Applied Cryptography*, Algorithm 14.82; Knuth, *TAOCP* vol. 2, §4.6.3; the short-exponent binary square-and-multiply engine cites the same Knuth section. |

## Primality (`src/number_theory.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Miller–Rabin | `miller_rabin_witness`, `is_probable_prime` | *Handbook of Applied Cryptography*, Algorithm 4.24. Twelve-base determinism to ψ₁₂ = 318665857834031151167461 ≈ 3.19×10²³ (the 3.317×10²⁴ figure is ψ₁₃, thirteen bases): Sorenson & Webster, *Strong Pseudoprimes to Twelve Prime Bases*, Math. Comp. 86 (2017), 985–1003 (arXiv:1509.00864). |
| Strong Lucas (Selfridge Method A) | `is_strong_lucas_probable_prime` (`selfridge_discriminant` cites §6, the perfect-square exclusion; `strong_lucas_core` cites §5, the acceptance conditions) | Baillie and Wagstaff, *Lucas pseudoprimes*, Math. Comp. 35 (1980), 1391–1417; Crandall and Pomerance, Algorithm 3.6.9 (the book title *Prime Numbers* is the verification pass's addition). |
| Baillie–PSW | `is_probable_prime_bpsw` | Baillie and Wagstaff, Math. Comp. 35 (1980), 1391–1417; Pomerance, Selfridge and Wagstaff, *The pseudoprimes to 25·10⁹*, Math. Comp. 35 (1980), 1003–1026. Determinism below 2⁶⁴: Feitsma's base-2 Fermat-pseudoprime enumeration (verified by Galway). |
| Sieve of Eratosthenes | `primes_below` | classical (odd-only sieve). |
| Trial-division survival estimate | `SMALL_TRIAL_PRIMES` sizing note | Mertens' theorem (classical), for the 1/ln y survival fraction that bounds the value of extending the table. |

## Congruences, reconstruction, batch (`src/number_theory.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| CRT (Garner) | `crt_combine` | Incremental Garner recombination, *Handbook of Applied Cryptography*, Algorithm 14.71. |
| Rational reconstruction | `rational_reconstruct`, `_bounded` | von zur Gathen and Gerhard, *Modern Computer Algebra*, 3rd ed. (2013), §5.10; technique of Wang, *A p-adic algorithm for univariate partial fractions*, SYMSAC '81, 212–217 (rational-number statement: Wang, Guy and Davenport, SIGSAM Bulletin 16(2) (1982), 2–3); accelerated variants Collins and Encarnación, *Efficient rational number reconstruction*, J. Symbolic Comput. 20(3) (1995), 287–297. |
| Batch modular inversion | `mod_inverse_batch` | Montgomery, *Speeding the Pollard and elliptic curve methods of factorization*, Math. Comp. 48 (1987), 243–264 (simultaneous-inversion trick). |
| p-adic valuation / remove factor | `valuation`, `remove_factor` | squared-power ladder (shape of GMP's `mpz_remove`); `valuation` accepts any `p ≥ 2`, PARI's `valuation` convention. |
| Batch smoothness (product/remainder trees) | `product_tree`, `remainder_tree`, `smooth_parts`, and `SmoothnessBase` (the same algorithm with the primes' product retained across batches) | Bernstein, *How to find smooth parts of integers* (2004). |

## Polynomials (`src/poly.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Polynomial evaluation | `PolyZ::evaluate`, `PolyMod::evaluate` | Horner's rule (classical), in its original polynomial form: fold `acc ← acc·x + cᵢ` from the leading coefficient down. |
| Polynomial multiplication (Karatsuba) | `convolve_z`, `convolve_modp` (behind `PolyZ::mul`, `PolyMod::mul`) | Karatsuba & Ofman, Soviet Physics–Doklady 7 (1963), applied to the coefficient convolution; the mod-p form defers its reductions to one per output coefficient. |
| Polynomial squaring | `convolve_square_modp` (behind `PolyMod::pow_mod`) | the cross-terms-once square of HAC Algorithm 14.16 in the coefficient ring; in characteristic 2 it degenerates to the Frobenius spread `g(x)² = g(x²)`. |
| Pseudo-division over ℤ | `PolyZ::pseudo_div_rem` | Knuth, *TAOCP* vol. 2, §4.6.1, Algorithm R. |
| Resultant (Bareiss fraction-free determinant) | `PolyZ::resultant` | Bareiss, *Sylvester's identity and multistep integer-preserving Gaussian elimination*, Math. Comp. 22 (1968), 565–578; Cohen, *A Course in Computational Algebraic Number Theory*, §3.3.1. |
| Discriminant | `PolyZ::discriminant` | Cohen, *A Course in Computational Algebraic Number Theory*, §3.3.2 (disc = (−1)^(d(d−1)/2) res(f, f′)/lc). |
| Squarefree factorization over 𝔽ₚ | `PolyMod::squarefree_factorization` | Cohen, §3.4.2 (Squarefree Factorization); the characteristic-`p` residual is recovered through its `p`-th root. |
| Distinct-degree factorization | `PolyMod::distinct_degree` | Cohen, §3.4.3 (Distinct Degree Factorization); the Frobenius `x ↦ xᵖ` reduced modulo the running polynomial. |
| Equal-degree split (Cantor–Zassenhaus) | `PolyMod::equal_degree_split` (used by `factor` and `roots`; `is_irreducible` is deterministic and uses only the distinct-degree stage) | Cantor & Zassenhaus, *A new algorithm for factoring polynomials over finite fields*, Math. Comp. 36 (1981), 587–592; Cohen, §3.4.4 (Final Splitting). The `p = 2` case uses the trace map, the standard characteristic-2 instance. |
| Division by a monic polynomial | `PolyZ::rem_monic` | Knuth, *TAOCP* vol. 2, §4.6.1, Algorithm D specialized to a unit leading coefficient: the quotient coefficient is the remainder's leading coefficient, so no coefficient division occurs and the division cannot fail. |
| Product tree over a quotient ring | `PolyZ::product_mod_monic` | the product-tree shape of Bernstein, *Fast multiplication and its applications* (§12 in *Algorithmic Number Theory*, MSRI 44, 2008), carried into `ℤ[x]/(f)` — reduction after every level is legitimate because reduction modulo a monic polynomial is a ring homomorphism. |
| Balanced base-`m` expansion | `PolyZ::balanced_base_expansion` | the balanced (symmetric-digit) radix representation, Knuth, *TAOCP* vol. 2, §4.1; used in this form for number-field-sieve polynomial selection, e.g. Lenstra & Lenstra (eds.), *The Development of the Number Field Sieve*, LNM 1554 (1993), §2. |
| Homogeneous substitution | `PolyZ::homogeneous_substitution` | the homogenization `F(X,Y) = Yᵈ f(X/Y)` (classical projective construction); the algebraic-norm instance `bᵈ f(a/b)` is the sieve's, Lenstra & Lenstra (eds.), LNM 1554, §3. |
| Roots modulo a prime power (Hensel lifting, with branching) | `PolyZ::roots_mod_prime_power` | Cohen, §1.6 (Hensel's lemma) for the simple-root case; the branching case where `f′(r) ≡ 0 (mod p)` — all `p` lifts or none — is the general extension a root-finder needs and is not in the lemma as usually stated. |
| Symmetric (balanced) lift from ℤ/mℤ | `PolyMod::symmetric_lift` | the same balanced convention as the expansion above, applied coefficientwise; it is what makes a modular computation's answer recoverable over ℤ once the modulus exceeds twice the height. |

## Lattice reduction (`src/lattice.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| LLL reduction (integral) | `lll_reduce`, `lll_reduce_delta` | Lenstra, Lenstra & Lovász, *Factoring polynomials with rational coefficients*, Math. Ann. 261 (1982), 515–534; integral form Cohen, *A Course in Computational Algebraic Number Theory*, Algorithm 2.6.3. |
| Two-dimensional reduction under a diagonal form | `gauss_reduce_weighted` | Lagrange (1773) and Gauss, *Disquisitiones Arithmeticae* (1801), art. 171 — the two-dimensional case, which solves the shortest-vector problem exactly rather than approximately; the modern analysis is Vallée, *Gauss' algorithm revisited*, J. Algorithms 12 (1991), 556–572. Transcribed; the article and page references are on the HANDOFF check list. |

## GF(2^m) binary fields (`src/gf2m.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Left-to-right comb multiplication (4-bit windows) | `Gf2m::mul` | Hankerson, Menezes & Vanstone, *Guide to Elliptic Curve Cryptography*, Algorithm 2.36. |
| Squaring (bit-spread) | `Gf2m::square` | Hankerson, Menezes & Vanstone, *Guide to ECC*, Algorithm 2.39. |
| Inversion (extended Euclid over GF(2)[x]) | `Gf2m::inverse` | Hankerson, Menezes & Vanstone, *Guide to ECC*, Algorithm 2.48. |
| Square root (Frobenius inverse, `a^{2^{m−1}}`) | `Gf2m::sqrt` | the Frobenius map is a bijection in GF(2^m); its inverse is repeated squaring. |
| Tap-wise modular reduction | `reduce_limbs` | Hankerson, Menezes & Vanstone, *Guide to ECC*, §2.3.5 (fast reduction; Algorithms 2.41–2.45 for the specific NIST polynomials), generalized here to fold at the taps of any reduction polynomial. |
| Trace | `Gf2m::trace` | IEEE Std 1363-2000, Annex A.4.5 ("Trace"). |
| Half-trace (odd degree) | `Gf2m::half_trace` | the classical telescoping identity HT(c)² + HT(c) = c + Tr(c), derived in full in the doc-comment; no external source is claimed. |
| Quadratic solver | `Gf2m::solve_quadratic` | IEEE Std 1363-2000, Annex A.4.7 for the even-degree construction; odd degrees route through the half-trace. |
| Irreducibility (Rabin's test) | `Gf2m::is_irreducible` | Rabin, *Probabilistic algorithms in finite fields*, 1980, in its deterministic GF(2) form (venue on the HANDOFF check list). |
| Curve-degree facts (163/233/283/409/571 all odd, and the test moduli) | `Gf2m::half_trace` doc, test moduli | FIPS 186-4 (the binary-curve degrees; named as context, not as an algorithm source). |
| Exponentiation | `Gf2m::pow` | left-to-right binary square-and-multiply, Knuth, *TAOCP* vol. 2, §4.6.3 — the section already verified for the same method at `MontgomeryContext::pow`; its application *here* was added 2026-08-15 and is on the HANDOFF check list. |

## Randomness (`src/random.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| splitmix64 (test/dev RNG) | `SplitMix64` in the `random`, `number_theory`, `poly`, and `gf2m` test modules, `tests/bigint_division.rs`, `tests/bigint_montgomery.rs`, and `src/bin/bench_bigint.rs` | Steele, Lea & Flood, *Fast Splittable Pseudorandom Number Generators*, OOPSLA 2014. Not a CSPRNG; used only for deterministic test and harness draws. |
| Probable-prime generation | `random_probable_prime` | *Handbook of Applied Cryptography*, Algorithm 4.44 shape (draw, screen, Miller–Rabin). |
| Rejection-sampling range reduction | `random_below` | Knuth, *TAOCP* vol. 2, §3.4.1 (the unbiased range reduction discussed there). |
| Stall-guard prime-density bound | `random_probable_prime` guard | Rosser & Schoenfeld, *Approximate formulas for some functions of prime numbers*, Illinois J. Math. 6 (1962). The inequality `π(2x) − π(x) > (3/5)·x/ln x` is verified numerically to 10⁷; the corollary label and threshold are **unchecked against the paper** (HANDOFF check list). |

## Test oracles and vectors

External sources that pin test data or serve as oracles, rather than
algorithms the library implements.

| Source | Where it pins |
|---|---|
| OEIS A217255 (strong Lucas pseudoprimes) | the strong-Lucas pseudoprime list below 10⁵, `number_theory.rs` tests. |
| OEIS A001262 (strong base-2 pseudoprimes) | the strong base-2 pseudoprime list, `number_theory.rs` tests. |
| GMP 6.3.0 (`mpz_jacobi`, `mpz_kronecker`, via `scripts/bench_gmp.sh`) | `GMP_JACOBI_VECTORS`, `GMP_KRONECKER_VECTORS`. |
| GMP's shipped `jacobitab.h` (generated by Möller's `gen-jacobitab.c`) | the independently generated cross-check of the Lehmer–Jacobi state table. |
| CPython's integer formatter | the base-36 googol radix vector, `bigint.rs` tests. |
| Python bigints | the overshoot-by-two division family, `tests/bigint_division.rs`; the rational LLL oracle, `scripts/lll_oracle.py`. |
| Feitsma's base-2 Fermat-pseudoprime enumeration (verified by Galway) | BPSW determinism below 2⁶⁴ (also in the BPSW row above). |

---

## Notes on two verified-against-error citations

- **Cipolla 1903** — the page range is `153–163`; a widely-copied secondary
  reference (e.g. an IACR ePrint) gives `154–163`. The `153–163` here matches
  zbMATH record 2656312 (*Napoli Rend.* (3) 9 (1903)).
- **Mihăilescu's theorem** (Catalan's conjecture), cited in `is_perfect_power`
  test commentary: Mihăilescu, *Primary cyclotomic units and a proof of
  Catalan's conjecture*, J. reine angew. Math. 572 (2004), 167–195.

## Provenance, not citation

`scrub.rs` is verbatim from the parent crate's audited helper
(cryptography-rs, `src/ct.rs`), and the crate as a whole was extracted from
darrelllong/cryptography — internal lineage, recorded here so the audit
trail is one file, not an external source either implements.
