# CITATIONS

Every non-schoolbook algorithm in rump, with the primary source cited in the
code that implements it. Each entry names the algorithm, where it lives, and
the reference as the source doc-comments state it. Schoolbook operations
(grade-school add/subtract/multiply, trial division, linear scans) carry no
citation and are not listed.

Verified against the source 2026-08-14. The convention throughout the crate:
name the algorithm, name its author, name the venue and year, and give the
section, algorithm number, or figure where the primary source is precise
about it.

---

## Arbitrary-precision integers (`src/bigint.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Long division, base 2⁶⁴ | `div_rem` (Knuth's Algorithm D) | Knuth, *The Art of Computer Programming*, vol. 2 (*Seminumerical Algorithms*), §4.3.1, Algorithm D. Add-back and borrow mechanics after Warren, *Hacker's Delight*, 2nd ed., §9-2 (`divmnu`). |
| Karatsuba multiplication | `mul_ref` ladder | Knuth, *TAOCP* vol. 2, §4.3.3 (Karatsuba–Ofman). |
| Toom-3 / Toom-4 multiplication | `mul_toom3_ref`, `mul_toom4_ref` | Knuth, *TAOCP* vol. 2, §4.3.3 (Toom–Cook, the generalization of Karatsuba); interpolation sequence after Bodrato, *Towards Optimal Toom–Cook Multiplication for Univariate and Multivariate Polynomials in Characteristic 2 and 0*, WAIFI 2007. |
| Integer square root (Newton) | `sqrt_rem` / `sqrt_floor` | Cohen, *A Course in Computational Algebraic Number Theory*, Algorithm 1.7.1. |
| nth root (Newton) | `nth_root_floor` | Cohen, Algorithm 1.7.1 for the square case (§1.7.1, Integer Square Roots); the general `k` is the crate's own generalization by the same Newton iteration, the AM–GM inequality on `k` terms giving the over-estimate. |
| Perfect-square residue filters | `is_square` | classical residue filter set (as in GMP's `mpz_perfect_square_p`); tables derived by enumeration. |
| Radix conversion (divide and conquer) | `from_str_radix` / `to_str_radix` | Knuth, *TAOCP* vol. 2, §4.4; Brent and Zimmermann, *Modern Computer Arithmetic*, §1.7. |
| f64 / ln estimates | `to_f64_lossy`, `ln_approx` | top-64-bit mantissa extraction; no external algorithm. |

## Montgomery and Barrett modular reduction (`src/bigint.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Montgomery multiplication | `MontgomeryCtx`, `mul_mont` | Montgomery, *Modular Multiplication Without Trial Division*, Math. Comp. 44 (1985), 519–521. Separated-operand-scanning shape from Koç, Acar & Kaliski, *Analyzing and Comparing Montgomery Multiplication Algorithms*, IEEE Micro 16(3) (1996), 26–33. |
| Barrett reduction | `BarrettCtx`, `reduce` | Barrett, CRYPTO '86; *Handbook of Applied Cryptography*, Algorithm 14.42, correction bound Note 14.44 (half-product refinement Note 14.45, also Brent and Zimmermann, *MCA*, §2.4). |

## GCD and the Euclidean family (`src/number_theory.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Lehmer's gcd (double-digit) | `gcd_lehmer`, `gcd_extended_lehmer` | Knuth, *TAOCP* vol. 2, §4.5.2, Algorithm L, on aligned leading digits. |
| Half-GCD (subquadratic) | `hgcd`, `gcd_via_hgcd`, `gcd_extended_via_hgcd` | Möller, *On Schönhage's algorithm and subquadratic integer gcd computation*, Math. Comp. 77 (2008), 589–607, Figure 4 (the algorithm behind GMP's `mpn_hgcd`). |

## Symbols and residuosity (`src/number_theory.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Jacobi symbol, binary reciprocity | `jacobi_binary` | *Handbook of Applied Cryptography*, Algorithm 2.149. |
| Jacobi symbol, Lehmer-batched state | `jacobi_lehmer`, state machine | Möller's design (GMP's `gen-jacobitab.c` / `hgcd2_jacobi.c`), after Schönhage's identities. |
| Jacobi symbol, HGCD-threaded (subquadratic) | `jacobi_hgcd` | Brent and Zimmermann, *An O(M(n) log n) algorithm for the Jacobi symbol*, ANTS-IX, LNCS 6197 (2010), 83–95; threading design Möller's (GMP's `mpn_hgcd_jacobi`). |
| Kronecker symbol | `kronecker` | Cohen, *A Course in Computational Algebraic Number Theory*, Algorithm 1.4.10. |

## Modular roots and exponentiation (`src/number_theory.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Tonelli–Shanks square root | `sqrt_mod_descent` | Cohen, Algorithm 1.5.1; the p ≡ 3 (mod 4) shortcut, *HAC* §3.5.1. |
| Cipolla's square root | `sqrt_mod_cipolla` | Cipolla, *Un metodo per la risoluzione della congruenza di secondo grado*, Rend. Accad. Sci. Fis. Mat. Napoli (3) 9 (1903), 153–163. |
| Prime-power square roots (Hensel lift) | `sqrt_mod_prime_power` | Hensel's lemma (Cohen, §3.5.3, "Factorization Modulo pᵉ: Hensel's Lemma"); dyadic and valuation cases per the standard structure of squares in ℤ/2^eℤ. |
| Windowed Montgomery exponentiation | `MontgomeryCtx::pow` | fixed 4-bit window (k-ary method), Knuth, *TAOCP* vol. 2, §4.6.3; the binary square-and-multiply fallback cites the same section. |

## Primality (`src/number_theory.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Miller–Rabin | `miller_rabin_witness`, `is_probable_prime` | *Handbook of Applied Cryptography*, Algorithm 4.24. Twelve-base determinism to 3.317×10²⁴: Sorenson & Webster, *Strong Pseudoprimes to Twelve Prime Bases*, Math. Comp. 86 (2017), 985–1003 (arXiv:1509.00864). |
| Strong Lucas (Selfridge Method A) | `is_strong_lucas_probable_prime` | Baillie and Wagstaff, *Lucas pseudoprimes*, Math. Comp. 35 (1980), 1391–1417; Crandall and Pomerance, *Prime Numbers*, Algorithm 3.6.9. |
| Baillie–PSW | `is_probable_prime_bpsw` | Baillie and Wagstaff, Math. Comp. 35 (1980), 1391–1417; Pomerance, Selfridge and Wagstaff, *The pseudoprimes to 25·10⁹*, Math. Comp. 35 (1980), 1003–1026. Determinism below 2⁶⁴: Feitsma's base-2 Fermat-pseudoprime enumeration (verified by Galway). |
| Sieve of Eratosthenes | `primes_below` | classical (odd-only sieve). |

## Congruences, reconstruction, batch (`src/number_theory.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| CRT (Garner) | `crt_combine` | Incremental Garner recombination, *Handbook of Applied Cryptography*, Algorithm 14.71. |
| Rational reconstruction | `rational_reconstruct`, `_bounded` | von zur Gathen and Gerhard, *Modern Computer Algebra*, 3rd ed. (2013), §5.10; technique of Wang, *A p-adic algorithm for univariate partial fractions*, SYMSAC '81, 212–217 (rational-number statement: Wang, Guy and Davenport, SIGSAM Bulletin 16(2) (1982), 2–3); accelerated variants Collins and Encarnación, *Efficient rational number reconstruction*, J. Symbolic Comput. 20(3) (1995), 287–297. |
| Batch modular inversion | `mod_inverse_batch` | Montgomery, *Speeding the Pollard and elliptic curve methods of factorization*, Math. Comp. 48 (1987), 243–264 (simultaneous-inversion trick). |
| p-adic valuation / remove factor | `valuation`, `remove_factor` | squared-power ladder (shape of GMP's `mpz_remove`). |
| Batch smoothness (product/remainder trees) | `product_tree`, `remainder_tree`, `smooth_parts` | Bernstein, *How to find smooth parts of integers* (2004). |

## Polynomials (`src/poly.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Pseudo-division over ℤ | `PolyZ::pseudo_div_rem` | Knuth, *TAOCP* vol. 2, §4.6.1, Algorithm R. |
| Resultant (Bareiss fraction-free determinant) | `PolyZ::resultant` | Bareiss, *Sylvester's identity and multistep integer-preserving Gaussian elimination*, Math. Comp. 22 (1968), 565–578; Cohen, *A Course in Computational Algebraic Number Theory*, §3.3.1. |
| Discriminant | `PolyZ::discriminant` | Cohen, *A Course in Computational Algebraic Number Theory*, §3.3.2 (disc = (−1)^(d(d−1)/2) res(f, f′)/lc). |
| Squarefree factorization over 𝔽ₚ | `PolyModP::squarefree_factorization` | Cohen, §3.4.2 (Squarefree Factorization); the characteristic-`p` residual is recovered through its `p`-th root. |
| Distinct-degree factorization | `PolyModP::distinct_degree` | Cohen, §3.4.3 (Distinct Degree Factorization); the Frobenius `x ↦ xᵖ` reduced modulo the running polynomial. |
| Equal-degree split (Cantor–Zassenhaus) | `PolyModP::equal_degree_split` (used by `factor` and `roots`; `is_irreducible` is deterministic and uses only the distinct-degree stage) | Cantor & Zassenhaus, *A new algorithm for factoring polynomials over finite fields*, Math. Comp. 36 (1981), 587–592; Cohen, §3.4.4 (Final Splitting). The `p = 2` case uses the trace map, the standard characteristic-2 instance. |

## Lattice reduction (`src/lattice.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| LLL reduction (integral) | `lll_reduce`, `lll_reduce_delta` | Lenstra, Lenstra & Lovász, *Factoring polynomials with rational coefficients*, Math. Ann. 261 (1982), 515–534; integral form Cohen, *A Course in Computational Algebraic Number Theory*, §2.6.3 (The Integral LLL Algorithm). |

## GF(2^m) binary fields (`src/gf2m.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| Left-to-right comb multiplication (4-bit windows) | `Gf2m::mul` | Hankerson, Menezes & Vanstone, *Guide to Elliptic Curve Cryptography*, Algorithm 2.36. |
| Squaring (bit-spread) | `Gf2m::square` | Hankerson, Menezes & Vanstone, *Guide to ECC*, Algorithm 2.39. |
| Inversion (extended Euclid over GF(2)[x]) | `Gf2m::inverse` | Hankerson, Menezes & Vanstone, *Guide to ECC*, Algorithm 2.48. |
| Square root (Frobenius inverse, `a^{2^{m−1}}`) | `Gf2m::sqrt` | the Frobenius map is a bijection in GF(2^m); its inverse is repeated squaring. |

## Randomness (`src/random.rs`)

| Algorithm | Location | Reference |
|---|---|---|
| splitmix64 (test/dev RNG) | `SplitMix64` | Steele, Lea & Flood, *Fast Splittable Pseudorandom Number Generators*, OOPSLA 2014. Not a CSPRNG; used only for deterministic test draws. |
| Probable-prime generation | `random_probable_prime` | *Handbook of Applied Cryptography*, Algorithm 4.44 shape (draw, screen, Miller–Rabin). |

---

## Notes on two verified-against-error citations

- **Cipolla 1903** — the page range is `153–163`; a widely-copied secondary
  reference (e.g. an IACR ePrint) gives `154–163`. The `153–163` here matches
  zbMATH record 2656312 (*Napoli Rend.* (3) 9 (1903)).
- **Mihăilescu's theorem** (Catalan's conjecture), cited in `is_perfect_power`
  test commentary: Mihăilescu, *Primary cyclotomic units and a proof of
  Catalan's conjecture*, J. reine angew. Math. 572 (2004), 167–195.
