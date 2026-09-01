# REVIEW.md

Craftsman review of **rump** (`rust-mp` 0.2.2). Every file under `src/` and
`tests/` was read. This document is the only output. **No source was
changed.**

Tree: `main` @ `c0f0b1c` ("Cut 0.2.2"), clean except this file. The previous
review on this crate was written against 0.2.1 plus a dirty working tree and
is stale.

Severity: **bug** = wrong answer or a stated contract the code violates.
**gap** = a hole the next edit falls into, or an unforced cost on a named
hot path. **nit** = local quality.

---

## Remediation update — 2026-09-01

This document's verdict and issue list preserve the original 0.2.2 review;
they are not the state of the current 0.3.0 tree. The small-prime identity no
longer depends on `2¹⁰`, Montgomery sequences have caller-reusable
`MontgomeryScratch`, the citation ledger was completed, and the later
cross-repository review's correctness findings were fixed and regression-
tested. Rump also now supplies the general machines the factoring work
needed—GF(2) pruning and Block Lanczos, weighted lattice reduction, balanced
CRT, and exact AKS primality—without importing factoring policy into the
library.

The latest performance pass stayed at that ownership boundary. Block Lanczos
retains a bounded sparse-fold pool for one solver call, replaces dense 64×64
GF(2) set-bit walks with byte-sliced products, and fuses recurrence equation
(18). The recurrence remains serial. On Deepcore, successive balanced ABBA
sessions over Factoring's complete 16-input balanced-40 GNFS corpus measured
candidate/baseline ratios of 0.913999 and 0.951728, about 13% cumulative, with
identical canonical answers. The representative first balanced-55 input fell
from the previously recorded 63.11 s to 52.77 s at 256 workers. Direct scalar
oracles cover the fused operations; the one-worker/eight-worker dependency
fixture now exceeds the parallel threshold instead of testing two inline
paths.

The integer layer's exact NTT is now hardware-aware without importing a thread
runtime or a machine-specific CPU cap. Inputs are expanded in parallel into
disjoint bit-reversed segments, independent forward transforms share the
budget concurrently, and large inverses use DIF plus the dead operand buffer
for parallel permutation/normalization. Pointwise, residue, clear, and CRT
combination passes are parallel above a measured grain; only the carry chain
remains serial. The budget never exceeds `available_parallelism` and falls
back to one when detection fails. Exact-worker measurements through the full
2^26 ceiling select 4, 8, 16, 32, then 64 contexts; 128 and 256 were slower at
the ceiling. M4 release probes now measure 2.70x over identical serial NTT at
65,536 limbs and 2.97x at 131,072. On 256-context deepcore, 131,072 limbs fell
from 146.55 ms in the first parallel version to 97.70 ms, and 1,048,576 limbs
from 877.52 ms at its then-selected 16 workers to 351.68 ms at 32. A true NTT
square retains one transform buffer and one forward transform per prime.
Differential tests cover serial/parallel transform identity, DIT/DIF inverse
identity, segmented input identity, random and maximal-carry products and
squares, the one-coefficient transform, and deterministic worker dispatch.

---

## Verdict

This is not a toy multiprecision library. Knuth D, a public Montgomery
domain, Karatsuba / Toom-3 / Toom-4 / exact NTT with measured crossovers and an
unbalanced block decomposition, Lehmer + Half-GCD, binary / Lehmer / HGCD
Jacobi, Tonelli–Shanks / Cipolla, twelve-base Miller–Rabin to the *right*
bound, BPSW with Selfridge Method A, Cohen’s integral LLL, and a real
`PolyModP` factorizer sit behind default-build `forbid(unsafe_code)` and
`deny(missing_docs)`. The crate says it is variable-time and not a
secret-scrubbing type, and the code matches that claim. Differential
tests exist against oracles that share no kernel code. MANUAL.md cannot
drift.

The previous review’s blocking items are closed. `BigInt::mul_ref`,
`div_rem` (truncated toward zero, rustdoc names `(-7)/2 = (-3, -1)`), and
`abs` are public and tested against `i128`. REQUESTS.md Tier 1 is empty.
ψ₁₂ is stated correctly in Rust docs, CITATIONS, the manual, and
`random.rs`. `PolyModP` over a composite modulus is unspecified, *and*
there is a pin test so a later cleanup cannot call `x²+1` mod 15 a proof.

What remains is not “schoolbook where Knuth D should be.” It is the class
of thing Pike still marks next to production kernels: a primality screen
whose identity test is coupled to a magic `2¹⁰`, a Montgomery multiply
that allocates on every call after the comments spent a page on
workspace reuse, and a citations file the crate root calls complete
that is not.

rump is the arithmetic the factoring crate should lean on more, not
less. The signed ring is delivered. The consumer’s `gnfs/arith.rs` is
now a stale wrapper, not a missing primitive.

---

## Closed since the last review of this crate

Do not re-open these.

- **Public signed ring.** `BigInt::mul_ref` / `div_rem` / `abs` are
  `pub`. Truncation convention is in the rustdoc. Differentially tested
  against `i128`, including the `(-7, 2)` corner and a zero-divisor
  panic. This was REQUESTS Tier 1.
- **ψ₁₂, not ψ₁₃.** Twelve bases through 37. The bound is
  `318665857834031151167461`. The old `3.317×10²⁴` is named as ψ₁₃
  wherever it still appears, so it cannot be reread as the crate’s
  claim. `manual.tex` matches.
- **Composite `PolyModP`.** Documented. Pin test
  `composite_modulus_verdicts_are_unspecified_not_proofs`.
- **Selfridge `|D|` vs `i64`.** Termination is a theorem; the conversion
  is named as an empirical cap, a panic not a misclassification.
- **Karatsuba admission** is `long < 2·short`. The exact 2:1 split goes
  to unbalanced (above 256 limbs) or schoolbook (below). Tested at the
  boundary.
- **`into_limbs`** stays crate-private. Drop of an emptied `Vec` is
  sound.
- Sampler stall caps: rejection-count where acceptance ≥ ½; identical-
  candidate where it is not. `should_panic` tests exist.

---

## Issues

### 1 — gap — `small_prime_screen` identity is coupled to `2¹⁰`

[`src/number_theory.rs`](src/number_theory.rs):2822–2840.

A hit on the trial table is either the prime itself or a multiple. The
disambiguation is

```text
candidate.bits() <= 10 && candidate.rem_u64(1u64 << 10) == prime
```

That is correct *today*: the table tops out at 997, every table prime
has `bits() ≤ 10`, and for those the residue modulo 1024 *is* the
candidate. It is a trap. Extending `SMALL_TRIAL_PRIMES` past 1023 — the
obvious next edit, and the comment does not forbid it — makes
`is_probable_prime(1031)` return `false`. The prime divides itself,
`bits()` is 11, the identity test fails, and the screen reports
composite.

The allocation this avoids is `BigUint::from_u64(prime)`, or
`candidate.to_u64() == Some(prime)`, both free next to a `rem_u64` of a
multi-limb candidate.

```text
return Some(candidate.to_u64() == Some(prime));
```

One comparison, no bit-width invariant, no second remainder. The
`2¹⁰` story can leave the comment.

### 2 — gap — public `mul_mont` / `square_mont` allocate per call

[`src/bigint.rs`](src/bigint.rs):2322–2323, 2983–2995, 3016–3024.

The internal kernel is written to *reuse* a workspace: “threaded through
by the caller so a sequence of domain operations allocates once rather
than per multiply.” The public methods that the comments call the
innermost field-multiply do this:

```text
pub fn mul_mont(...) -> BigUint {
    let mut workspace = Vec::new();
    BigUint::montgomery_mul_odd_with_workspace(..., &mut workspace)
}
```

Every product allocates `2w+1 + 3w` limbs, runs REDC, drops the buffer.
`square_mont` is the same. `add_mont` / `sub_mont` do not, which is why
the contrast is visible.

This is the loop Pollard's rho lives in, and the loop every
`is_witness` squaring lives in. The crate already has the shape that
fixes it — `*_with_workspace` — and does not expose it. Either:

- publish `mul_mont_into` / a reusable scratch, or
- keep a `RefCell<Vec<u64>>` on `MontgomeryCtx` and reuse it, with the
  honesty that the context is then not `Sync` for concurrent
  in-domain work (it already is not a secret-scrubbing type).

Do not “fix” this by making `mul_mont` slower in some other way. The
kernel is fine. The boundary throws the workspace away.

`encode` / `decode` / `pow` allocate too; they are the edges, not the
inner loop. `pow` at least wipes before free. `mul_mont` does not wipe
and says so. That half is honest. The allocation is the unforced half.

### 3 — gap — crate root says `CITATIONS.md` collects every algorithm

[`src/lib.rs`](src/lib.rs):13–14. [CITATIONS.md](CITATIONS.md) is a real
table and a good one. It is not complete against that sentence, and
[HANDOFF.md](HANDOFF.md):222–227 already lists the holes:

- `Gf2m::trace` / `half_trace` / `solve_quadratic` (IEEE 1363 A.4.5 / A.4.7,
  named in the module docs)
- `Gf2m::is_irreducible` (Rabin 1980)
- `Gf2m::pow`
- tap-wise reduction (*Guide to ECC* §2.3.5)
- `random_below` (Knuth §3.4.1)

The crate-root claim is a contract. Either fill the rows or weaken the
sentence to “the non-schoolbook integer and number-theory kernels.”
HANDOFF also flags the Rosser–Schoenfeld “Corollary 3” label on the
prime-search density bound as agent-recalled: the *inequality* was
checked numerically to `10⁷`; the *label, constant, and threshold*
were not checked against the paper. That is the opposite of “verified
against the source 2026-08-14.” Name it as unchecked or check it.

### 4 — named, keep named

These are honest limitations. Do not “fix” them into a lie.

- **`miller_rabin_witness` on 2.** Even candidates, 2 included, return
  `true` (proven composite) because no Montgomery context exists. The
  wrappers reach 2 through the sieve. Direct callers must screen.
  Documented at :2779–2784.
- **`Gf2m` under a reducible modulus.** Not a field. `sqrt` need not
  invert. `trace` panics if the Frobenius sum leaves GF(2).
  `solve_quadratic` returns `None` rather than an unverified root.
  Documented at the module root; tests panic and terminate on purpose.
- **`random_probable_prime` cannot return 2.** HAC 4.44 samples odd
  candidates; `bits == 2` always yields 3. Documented.
- **`random_probable_prime` is MR, not BPSW.** Right for numbers this
  function drew. A caller who wants BPSW on generated primes calls it.
- **Selfridge `|D|` is `i64`.** Panic on overflow, never a wrong prime.
  Heuristic `|D|` is tiny.
- **`gcd_extended` / `mod_inverse` at HGCD scale** canonicalize after
  a driver that can leave the classical continued-fraction path.
  Bézout holds; the raw pair may differ by `(b/g, −a/g)`. Documented.
  README still wants the transform carried through the cofactors as a
  *performance* item, not a correctness one.
- **Barrett** uses full products. HAC 14.45 half-products are the
  named follow-up. The even-modulus capability is the value.
- **No dedicated integer squaring kernel.** `square_ref` is `mul_ref`.
  `square_mont` is the one that matters and exists.
- **`mul_mont` does not wipe scratch.** Named. Defense-in-depth, not
  CT. Do not grow `scrub.rs` to chase this.
- **Spare capacity is not wiped.** `normalize` `pop`s. `clone_from`
  *does* scrub the abandoned tail, which is the path that would
  otherwise leak. The crate-root paragraph already says this.
- **32-bit `usize` is unsupported.** Index arithmetic assumes 64-bit.
  Said at the crate root.
- **Variable-time.** Said everywhere it needs to be.

### 5 — nits

- **`is_witness` encode-after-pow.** `ctx.pow` already returns an
  ordinary residue; the next line encodes it to re-enter the domain for
  the squaring chain. Correct. One extra encode/decode pair per
  Miller–Rabin round. A `pow_encoded` that *stays* encoded would drop
  it. The HANDOFF already records a hang from confusing the two.
- **`REQUESTS.md` says the consumer can retire `gnfs/arith.rs`.** On
  rump’s side that is true. The factoring tree still has that file, and
  its module docs still claim `mul_ref` is private. That is the
  consumer lagging a delivered request, not a rump hole. Do not reopen
  Tier 1.
- **`xor_shifted_word` bounds** are a comment-only contract on
  `reduce_limbs`. Documented as fragile. Tests cover reduction. Leave
  it unless a second caller appears.
- **`Debug` prints every limb.** Named. Do not derive `Debug` off if
  a consumer starts treating it as a log format for secrets; that
  consumer is wrong.
- Review history in HANDOFF / comments. HANDOFF is the right place
  for it. Keep it out of rustdoc.

---

## File by file

### [Cargo.toml](Cargo.toml)

`rust-mp` 0.2.2, lib `rump`. No dependencies. MSRV 1.87 recorded
next to `is_multiple_of`. Fine.

### [src/lib.rs](src/lib.rs)

Extraction story, citations, variable-time, scrub limits, 64-bit only,
MSRV. This is how a crate root should read. Both denies. Re-exports
match README, and `BigInt`’s useful arithmetic is now on the type.
Issue 3 is the one over-claim.

### [src/scrub.rs](src/scrub.rs)

Six-line volatile write. Safety argument local. `T: Copy + Default`.
`compiler_fence(SeqCst)` after the loop. What it does not do is listed.
Do not widen this file.

### [src/bigint.rs](src/bigint.rs)

LE `u64` limbs, canonical (no leading zero). `Eq`/`Ord` ride that.
`clone_from` scrubs the abandoned tail.

**Multiplication.** Schoolbook → Karatsuba (`long < 2·short`, 32 limbs)
→ Toom-3 (128) → Toom-4 (3072) → exact two-prime NTT; lopsided operands use
block decomposition (`short ≥ 256` and `long ≥ 2·short`), each block re-entering
`mul`. The NTT splits limbs into base-2^16 digits, convolves modulo two proven
NTT primes, reconstructs by CRT, and carries back to base 2^64. Its measured
admission is worker-aware: 65,536 limbs serially, 32,768 with two useful
contexts, and 8,192 with four or more, plus padding gates for the radix-2
staircase. Scoped stage workers never exceed reported parallelism and produce
the same ordered transform as one worker. The selected transform targets grow
from 4 workers at 2^16 values to 64 at 2^24–2^26, always reduced to the
machine's reported availability; deepcore tests included 128 and 256 workers.
`square` pointwise-squares one
transform buffer instead of repeating a general product. The coefficient bound
is asserted, the 2^26 transform ceiling is explicit, and unsupported sizes fall
back to Toom. Independent schoolbook comparisons cover irregular shapes,
one-coefficient and partial-digit inputs, and maximal carries; deterministic
tests pin serial and parallel dispatch boundaries.

**Euclidean allocation.** Lehmer, extended GCD, inverse, Jacobi, rational
reconstruction, and HGCD's base case retain two transform buckets and recycle
the old operand/cofactor limb buffers into the next batch outputs. The previous
three fresh vectors per transformed output are gone. `abs_diff_bits` now scans
the borrow chain without materializing `|a-b|`; guarded HGCD division retains
its `2^s` threshold and adjusted-dividend buffer; matrix row steps mutate in
place. Deterministic before/after probes on M4 show roughly 18–35% lower Lehmer
time through 2,048 limbs and 10–27% lower HGCD time through 4,096 limbs, with
the gain diminishing once large matrix multiplication dominates.

**Division.** Single-limb Horner; multi-limb Knuth D with
normalization, two-limb estimate, third-limb correction, one add-back.
The `q_hat ≥ BASE` clamp is named as redundant and kept to match the
published algorithm. Tests include the estimate-correction and
add-back shapes, plus a bit-serial oracle in
`tests/bigint_division.rs`.

**Roots.** Newton, certified by the first non-decrease. `sqrt_floor`
skips the remainder square. `nth_root_floor` panics on `k == 0`.
`is_square` uses residue filters then one certified root. Filters
derived by enumeration, not transcription.

**Montgomery.** Context is a type. `encode` / `decode` / `mul_mont` /
`square_mont` / `add_mont` / `sub_mont` / `pow` / `pow_encoded`.
Reduced-operand contract debug-asserted. `pow` is right-to-left binary
at ≤ 64-bit exponents and 4-bit window above; rustdoc names both.
Issue 2 is the public-API allocation.

**Barrett.** Either parity. Full products. Named follow-up.

**`BigInt`.** Sign-magnitude, `from_parts` canonicalizes a contradictory
pair (documented as silent). `from_i64` uses `unsigned_abs` (`i64::MIN`
is total). Public ring is complete for what the NFS consumer asked.
`div_exact` / `div_exact_checked` stay `pub(crate)` and now call
`div_rem`. `modulo_positive` is the floored residue.

**`to_f64_lossy` / `ln_approx`.** Top-64 mantissa. `ln_approx` panics
on zero — the consumer already grew a guard. Do not make it return
`−∞`; the panic is the defined logarithm.

**Drop** wipes live limbs. Spare capacity and reallocations are out of
scope, said at the crate root.

### [src/number_theory.rs](src/number_theory.rs)

Lehmer on 124-bit windows; HGCD above threshold for `gcd` and Jacobi.
`gcd_extended` HGCD path canonicalizes. `product_tree` /
`remainder_tree` / `smooth_parts`: Bernstein. Rational reconstruction.
`mod_inverse_batch`: Montgomery’s trick. `valuation` / `remove_factor`:
squared ladder.

**Symbols.** Binary / Lehmer / HGCD Jacobi. Kronecker after Cohen.

**Roots.** `p ≡ 3 (mod 4)` shortcut, Tonelli–Shanks, Cipolla past a
measured 2-adic depth. Result verified by squaring. Composite moduli
yield `None` or a value that squares — the contract is verification,
not primality. `sqrt_mod_prime_power` is the QS Hensel the consumer
used to carry.

**Primality.** Trial to 997, then twelve bases. ψ₁₂ stated correctly,
and the misuse against untrusted input is named (`is_probable_prime`
is not `is_probable_prime_untrusted`). BPSW: one strong base-2 MR plus
strong Lucas, Selfridge A. Tests against a sieve oracle, published
pseudoprime tables, Mersenne neighbours, and random primes / products.
Issue 1 is the screen.

`miller_rabin_witness` is a compositeness primitive. `false` is not
“prime.” Named.

`primes_below`: odd-only sieve, exclusive bound. `usize::try_from` on
the length — on supported hosts that is “does not fit memory,” not a
wrap.

### [src/poly.rs](src/poly.rs)

`PolyZ`: content, primitive part, pseudo-division, exact `div_rem`
(`None` if not in `ℤ[x]`), resultant (Bareiss), discriminant.
In-place remainder window. Leading coefficients of a product cannot
cancel over an integral domain; the buffer is sized exactly.

`PolyModP`: modulus carried in the value; mismatched moduli
`assert_eq!` in **release**. `m < 2` rejected. Division-based
operations invert a leading coefficient and panic if it is not a
unit. Factorization (squarefree / DDF / Cantor–Zassenhaus) and
`roots` via `gcd(f, x^p − x)`. `is_irreducible` is deterministic DDF.
Composite-modulus pin test present. `factor` / `roots` panic if the
`Rng` yields no entropy — same posture as `random_*`.

### [src/gf2m.rs](src/gf2m.rs)

Degree and taps derived from the polynomial, never supplied beside it.
Comb mul, linear sqr, EEA inverse, Frobenius sqrt, trace, half-trace,
quadratic solve at every degree, Rabin irreducibility. Reducible-
modulus contract is the module’s first section and is tested
(`should_panic` on a Frobenius sum outside GF(2);
`solve_quadratic_terminates_on_reducible_rings`).

`xor_shifted_word` is the one bounds-fragile helper. The comment is
the proof. Do not give it a second caller.

### [src/lattice.rs](src/lattice.rs)

Cohen 2.6.3, exact integer Gram data, `δ = 3/4` default,
`lll_reduce_delta` with `δ ∈ (1/4, 1)` checked in `u128`. Empty basis
ok. Dependent basis panics on `d_k = 0`. Exact divisions via
`div_exact`. Lovász does **not** make GS norms monotone — documented
with `[1, 3], [3, 0]`. Independent oracle in `scripts/lll_oracle.py`.
This is a real LLL, not a float Gram–Schmidt toy.

### [src/random.rs](src/random.rs)

Rejection sampling, no modular bias. Stall guards that panic on a
pinned generator, sized so a working generator trips them at
`e⁻¹¹¹` or better. The one remaining gap — a cycling generator on
`random_coprime_below` against a sparse unit set — is stated, not
papered over. `random_probable_prime` is MR, cannot return 2, `bits
== 2` yields 3. Tests include `ZeroRng` / `ConstRng`.

### [src/bin/bench_bigint.rs](src/bin/bench_bigint.rs) /
[src/bin/pilot_mp.rs](src/bin/pilot_mp.rs)

Harnesses. Not the library contract.

### Tests

[tests/bigint_division.rs](tests/bigint_division.rs): vs a bit-serial
oracle. Mutation notes in-file.
[tests/bigint_montgomery.rs](tests/bigint_montgomery.rs): vs a
division-based ladder.
[tests/lehmer_differential.rs](tests/lehmer_differential.rs): Lehmer
vs classical.
[tests/manual_examples.rs](tests/manual_examples.rs): MANUAL.md cannot
drift.

In-crate: Jacobi vs GMP vectors, signed ring vs `i128`, Toom / Karatsuba
/ unbalanced vs schoolbook, Knuth D correction cases, BPSW tables,
composite `PolyModP`, LLL vs the Python oracle.

### Docs

[README.md](README.md) matches the surface. PERFORMANCE.md is a
generated report with hosts, CIs, and GMP ratios — edit the
*generator*, not the file ([HANDOFF.md](HANDOFF.md)).
[REQUESTS.md](REQUESTS.md) is a closed ledger. The consumer has not
yet deleted `gnfs/arith.rs`; that is not rump’s leftover.
[ROADMAP.md](ROADMAP.md): items 1–9 done; radix I/O item 5 is “tuning
at huge size,” said. Gated-on-factorization work is deferred.
[CITATIONS.md](CITATIONS.md): issue 3.
[HANDOFF.md](HANDOFF.md): the right place for war stories and the
citation-check list. Keep it.

---

## Suggested edits (for the author; not applied)

**Screen identity** — retire the `2¹⁰` invariant:

```text
if candidate.rem_u64(prime) == 0 {
    return Some(candidate.to_u64() == Some(prime));
}
```

**In-domain multiply** — one of:

```text
impl MontgomeryCtx {
    pub fn mul_mont_with_workspace(
        &self, lhs: &BigUint, rhs: &BigUint, workspace: &mut Vec<u64>,
    ) -> BigUint { /* existing kernel */ }
}
```

or a scratch `Vec` on the context. Measure rho and `is_witness` before
and after. If it does not move the needle, leave the allocation and
say so next to the comment that currently implies reuse.

**CITATIONS** — add the five Gf2m / `random_below` rows HANDOFF already
named, or shrink the crate-root sentence. Check Rosser & Schoenfeld
against the paper or mark the label unchecked.

---

## What not to do

- Do not port rump to the GPU. REQUESTS already says why.
- Do not make `into_limbs` public.
- Do not “fix” `miller_rabin_witness(2)` to return prime-ish without
  rewriting every caller that uses it as a compositeness predicate.
- Do not grow `scrub.rs`.
- Do not implement sieves, factor bases, or ECM curves here. Those
  know they are looking for factors.
- Do not add a thirteenth MR base to “make the old ψ₁₃ sentence true.”
  The sentence is gone.
- Do not reopen Tier 1. The ring is public. Nag the consumer.

rump at 0.2.2 is the craftsman integer crate the rest of the workspace
was written to assume. The remaining work is tightening the joints,
not replacing the kernels.
