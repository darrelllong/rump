# HANDOFF — picking rump up on another machine

State as of commit `e9e9f87` on `origin/main`, tag `v0.2.1`, crate version
`0.2.1` (unreleased work sits on `main` above the tag). Everything below is in
the repository unless marked otherwise.

```sh
git clone git@github.com:darrelllong/rump.git && cd rump
cargo test && cargo test --release      # 170 lib tests green + 9 ignored timing probes
cargo clippy --release --all-targets    # warning-clean
cargo fmt --check                       # clean
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --lib   # clean
```

If those five commands are green you have the same tree this note describes.

## What is not in git and must be carried across

- **`~/rump/REVIEW.md`** — the external reviewer's report. Three passes were
  against the published 0.1.1 tree; the fourth is against `5f05f9a`/`v0.2.1`
  and is the one that matters. Its verdict and the outstanding items are
  summarised under "The external review" below, but bring the file.
- **`~/rump-tier3.bundle`, `~/rump-perfdocs.bundle`** — `git bundle` mirrors of
  all branches and tags, made as belt-and-braces while work lived on an
  ephemeral scratchpad. `origin/main` is now ahead of both; they are only a
  fallback if GitHub is unreachable. Verify with `git bundle verify <file>`.
- **pilot-bench** — the benchmark driver, built separately at
  `$HOME/pilot-bench/build/cli/bench` (override with `PILOT_BENCH_CLI`). Only
  needed to re-measure; not needed to build or test.
- **libgmp** — only for the comparison column (`scripts/bench_gmp.sh`).

## Toolchain

Rust 1.87 or later (`rust-version` in Cargo.toml; the crate uses
`is_multiple_of`). **64-bit hosts only** — the limb indexing and the `R²` and
Karatsuba bit shifts assume a 64-bit `usize`, and this is stated in the crate
docs. Zero dependencies.

## Where things stand

Tier 3 for the factoring consumer is complete and delivered: `PolyZ` and
`PolyModP` (arithmetic, exact and pseudo-division, resultant, discriminant,
squarefree/distinct-degree/Cantor–Zassenhaus factorisation, roots) and integral
LLL. The last Tier 1 item — the public `BigInt` signed ring (`mul_ref`,
truncated `div_rem`, `abs`), the external reviewer's standing #1 — landed
2026-08-15, differentially tested against `i128` and documented in MANUAL.md
and the LaTeX manual. `REQUESTS.md` records the whole list as cleared.

The external reviewer's ship blockers are closed and re-verified on the real
tree. Since `v0.2.1` the tree has also had a full documentation pass and a
round of defect fixes; see "Outstanding work".

## The external review

The 0.2.1 pass **withdrew** the earlier "not production ready" verdict: every
hang and false-prime path was reproduced as fixed, with tests that fail if the
guards are reverted. Its two acceptance-gating items were then fixed here (hard
`assert_eq!` on mixed `PolyModP` moduli; `half_trace` panicking on even degree),
along with the Las Vegas retry cap. Its remaining non-blocking asks:

1. **Split `src/bigint.rs` (~4.8 kLOC) and `src/number_theory.rs` (~5.2 kLOC).**
   Not started. A behaviour-preserving plan is in "Outstanding work" below. The
   reviewer says it will keep listing this until done.
2. Point the Miller–Rabin and Tonelli inner loops at the Montgomery domain —
   **done** (`12cafc2`), and note the trap recorded below.

## The measurement story — read this before touching benchmarks

The variable-time figures were wrong for a long time and the cause is subtle.
Two distinct defects, both now fixed:

1. **The harness reported a mean that was not the sample mean.** It used
   pilot-bench's `readings_mean`, which is a changepoint-truncated "dominant
   segment" average meant to strip warmup from a steady-state process. Our
   heavy-tailed operations are i.i.d. mixtures — microsecond rejections
   punctured by rare enormous readings — which that detector reads as regime
   changes and discards, so the reported figure need not even lie inside the
   sample's own range (a 7168-bit `sqrt_mod` cell read 21.9 ms against its own
   0.12 ms p99). The reduction now computes the whole-sample mean, so that
   class of corruption is structurally impossible.
2. **The samples were far too small to mean anything.** Each reading needs a
   fresh random operand, and generating a multi-kilobit random prime costs far
   more than the operation being timed: a 180 s session at 7168 bits collected
   **four readings**. Its "mean" of 202 ms was the average of two fast Jacobi
   exits and two full descents; a different four would have said 0.13 ms or
   400 ms. That coin flip — not the hardware and not a kernel crossover — is
   the "discontinuity" the scaling figures showed, which is why running it on
   more architectures could only reproduce the noise. (Proof: n = 4 reproduces
   both the reported mean, 202.068 ms, and its 113.21 % CI exactly.)

Consequently: every row now publishes its reading count `n`; the reduction
refuses to report a mean below a 30-reading floor and writes
`insufficient-sample(n=…)` instead; and `scripts/check_bench_consistency.py`
enforces the invariant that a mean must lie inside the interval its own order
statistics allow. `scripts/build_performance.sh` runs that check, so the
document cannot be assembled from self-contradicting data.

**Re-measurement owed.** Four legacy rows (measured before the fix) are still
inconsistent and need re-running on their own hosts:

| Row | File | Host |
|---|---|---|
| `jacobi_8192` | `bench/gcd_scaling_hardy.md` | hardy (M4) |
| `sqrtmod_8192` | `bench/heavy_extended_hardy.md` | hardy (M4) |
| `add_256` | `bench/gmp_darby.md` | darby (Pi 5), GMP column |
| `divrem_256` | `bench/gmp_darby.md` | darby (Pi 5), GMP column |

`python3 scripts/check_bench_consistency.py bench/*.md` lists them. Note the
GMP column is affected too — the same harness produced it. Ideally the whole
fleet (hardy = M4, moore = EPYC 7452, darby = Pi 5, verne = A18 Pro) is
re-measured with the fixed harness; until then the wide variable-time cells
should be read as provisional. Re-measure one row with:

```sh
cargo build --release --bin pilot_mp
bash scripts/bench_primitives.sh sqrtmod_8192      # prints one Markdown row
```

Budget honestly: the widest heavy cells need a much larger session than the
120 s default (`PILOT_MP_HEAVY_SESSION`) to clear the reading floor, because
operand generation dominates. If a cell cannot clear it, leave it marked
`insufficient-sample` rather than publishing a number.

## Outstanding work, in the order I would do it

1. **Re-measure the four rows above** (fleet task; hardy is the machine that
   produced most of the affected data).
2. **Split the two large files** (reviewer item 1). Behaviour-preserving, and
   the safe shape is: keep `src/bigint.rs` and `src/number_theory.rs` as module
   roots and add sibling directories, moving one cohesive block at a time with
   `cargo build` after each move so visibility errors stay local. Suggested
   cuts: `bigint/montgomery.rs` (MontgomeryCtx, BarrettCtx, the REDC and
   mont_mul/mont_sqr slice kernels, `copy_padded`), `bigint/mul.rs`
   (schoolbook, Karatsuba, Toom-3, Toom-4 and the thresholds), `bigint/div.rs`
   (Algorithm D and the Horner path); `number_theory/gcd.rs` (Lehmer, Half-GCD,
   Bézout, inverses), `number_theory/symbols.rs` (Jacobi, Legendre, Kronecker),
   `number_theory/prime.rs` (Miller–Rabin, BPSW, the square roots, CRT). Leave
   the test modules in the roots initially — they reach many private items, and
   splitting code without moving tests is still the win. Expect no test edits;
   if a test needs changing, the move changed behaviour — stop.
3. **Cut 0.2.2**: bump `Cargo.toml` (and `Cargo.lock` via `cargo check`),
   commit, tag `v0.2.2`, push. **Do not publish to crates.io** — the owner
   holds that release explicitly; `v0.2.1` was git-only for the same reason.

## Deferred defects: resolved 2026-08-15

The list that used to live here (found in the documentation pass, deferred)
was worked through, with every fix adversarially reviewed and the review's
own findings folded back in. Summary of dispositions:

**Fixed as described** — `sdiv_step` (reference ordering, `rem + 2^s`
identity); the three polynomial division loops (in-place window subtraction,
degree tracked by hand, `rem` without quotient bookkeeping, correct quotient
sizing, monic `mod_inverse` skip, `scale` unit fast paths, dead `shift_up`
helpers deleted); `BarrettCtx::pow_mod` top-bit seeding; `sqrt_floor` /
`sqrt_rem` sharing a private `sqrt_newton`; `gf2m::inverse` bit lengths
travelling across the swap; `gf2m::reduce` via a new `pub(crate)`
`BigUint::into_limbs`; `squarefree_into` expect messages.

**Fixed differently than proposed, after measurement or a counterexample:**

- *Unbalanced multiplication.* The naive block decomposition (shift each
  product, add full-width) measured 2.3–11.5× **slower** than schoolbook —
  the recombination is `Θ(long²/k)` limb copies. The landed kernel
  accumulates in place (`add_into_at`) and dispatches only above its own
  measured crossover, `UNBALANCED_THRESHOLD_LIMBS = 256`
  (`unbalanced_crossover_timing` is the probe; numbers sit on the constant).
  Below it, lopsided shapes stay schoolbook, which wins there. The
  `long == 2·short` Karatsuba admission edge is fixed (strict `<`) with a
  table-driven boundary test.
- *Sampler stall caps.* A rejection-count cap is sound for `random_below` /
  `random_nonzero_below` (acceptance ≥ 1/2 always; capped at 256). It is
  **unsound** for `random_coprime_below` and `random_probable_prime` — a
  primorial `coprime_to` legitimately leaves 1 as the only unit below the
  bound, and a count cap fired at coin-flip odds on a working generator.
  Those two instead cap consecutive draws of the *same rejected candidate*
  (256), the one signature a working generator cannot produce; the prime
  search also skips the Miller–Rabin re-screen on a repeat. `should_panic`
  tests cover all four.
- *`shl_bits` scrub.* The "would be free" claim was wrong: measured 3–5% on
  every large balanced multiplication (the shift sits on Karatsuba/Toom
  recomposition). Not scrubbed; the decision and the number are in the
  comment.
- *`selfridge_discriminant`.* No cap added — a cap misclassifies or panics
  on legitimate input. Termination is now documented as a theorem
  (candidates `D ≡ 1 (mod 4)`, `|D| ≥ 5`, meet every class mod `n`; the
  character is non-principal after the square exclusion), with the `i64`
  conversion named plainly as the de facto cap whose unreachability is
  empirical.

**Still open, deliberately** — `Gf2m::sqrt` wrong-value-on-reducible-modulus
(documented; the signature cannot carry it); `xor_shifted_word` boundary
fragility (documented); `is_irreducible`'s per-checkpoint modulus clone
(inherent — Euclid consumes its working copy; now said at the call site).
`random_probable_prime` still cannot return 2 (documented; HAC 4.44 samples
odd candidates).

**Found by the adversarial review, beyond the original list** — the
twelve-base Miller–Rabin determinism bound was stated as ψ₁₃ = 3.317×10²⁴
everywhere; the true twelve-base bound is ψ₁₂ ≈ 3.19×10²³, and there is an
explicit composite between them that the crate certifies prime. Corrected in
`number_theory.rs`, `random.rs`, `CITATIONS.md`, and the manual — the *code*
was never wrong, the claim was. Also: `MontgomeryCtx::pow`'s rustdoc now
names both ladder engines (right-to-left binary at ≤ 64-bit exponents, 4-bit
window above); `PolyModP::div_rem`'s panic contract notes the
degenerate-degree early return; a pin test
(`composite_modulus_verdicts_are_unspecified_not_proofs`) nails the
`x² + 1 mod 15` unspecified verdict so it cannot later be sold as a proof.

**Citations to check against the physical sources** (added by others, left
verbatim rather than guessed at): Rosser & Schoenfeld, *Approximate formulas
for some functions of prime numbers*, Illinois J. Math. 6 (1962) — whether
"Corollary 3" is the right label for `π(2x) − π(x) > (3/5)·x/ln x` with
hypothesis `x ≥ 20½`, cited in `random.rs`'s prime-search stall guard and
the manual's §10 (the *inequality* is verified numerically to 10⁷; the
*label, constant, and threshold* were supplied by a reviewing agent from
memory, the exact provenance this list exists to catch); Rabin 1980's venue
for `is_irreducible`;
*Guide to ECC* §2.3.5 and Algorithms 2.41–2.45 for tap-wise reduction; Knuth
§3.4.1 for `random_below`; IEEE Std 1363-2000 Annex A.4.7 for the even-degree
quadratic solver; Knuth §4.6.3 as newly applied to `Gf2m::pow` (the section
is verified for the same method at `MontgomeryCtx::pow`, but this
application was recalled, not checked); whether Bodrato's optimised sequence really describes the
Toom interpolation as written (it reads as a first-principles Vandermonde
solve); and Dussé & Kaliski EUROCRYPT '90 as where the word-level `n₀'`
constant "was introduced". The once-missing `CITATIONS.md` rows were closed
2026-08-15 in two passes: a pattern sweep (which found `mont_sqr`
HAC 14.16, `BarrettCtx::pow_mod` §4.6.3, `gcd_extended` HAC 2.107 / Knuth
Algorithm X, `mod_inverse` HAC 2.142 — and *missed* Dussé & Kaliski,
because that citation has no algorithm number for a pattern to hit), then
an exhaustive line-by-line read of every comment in every source file,
which is the method to repeat next time. The read also surfaced: two
straggler `3.3·10²⁴` (ψ₁₃) claims in the BPSW doc and a test comment that
every pattern sweep had missed, now fixed; two mis-transcribed rows
(`hgcd2_jacobi.c`, which the code never names, and Cohen "§2.6.3" for the
code's "Algorithm 2.6.3"), now corrected; and the test-oracle provenance
(OEIS, GMP 6.3.0 vectors, `jacobitab.h`, CPython, Python), now collected
in its own table section. All new rows are transcriptions and are marked
as such in the file's header — the physical checks above still apply.

One caution from experience: an agent-reported "defect" claiming HAC Algorithm
14.82 is sliding-window was **wrong** — 14.82 is left-to-right k-ary (14.85 is
sliding-window), verified against the published chapter. The existing citation
is correct. Check before you correct.

## Working conventions

- **Documentation** is held to WHAT + WHY + HOW, for a human reader, in a formal
  technical register: name the algorithm, its author and venue; define the
  mechanism rather than gesturing at it. No informal or marketing register. The
  word "honest"/"honestly" as a self-label is banned. A `# Panics` section must
  describe a *reachable* panic; for an `expect` guarding an unreachable
  invariant, say so plainly.
- **Claims are verified, not asserted.** Performance numbers come from
  measurements or they come out. Citations are checked against sources, not
  recalled. Environmental claims (a host is unreachable, a cell is corrupt) are
  tested before being built upon.
- **Every non-trivial change gets an adversarial review** before commit
  (`advocatus-diaboli` / `ghost-of-beria` agents were used throughout).
- **Verify before trusting a refactor.** The Montgomery in-domain change looked
  right, passed review, and hung forever: `pow_encoded` takes an encoded base
  but returns an *ordinary* residue, so `while t != one_mont` never terminated.
  It was caught only because a benchmark hung. Run the suite; do not kill it
  early.
- Stage files explicitly (never `git add -A`) — concurrent work has been in
  flight in this repository more than once.
- The MANUAL's code blocks are mirrored verbatim in `tests/manual_examples.rs`
  and asserted on every `cargo test`, so the manual cannot drift. Change both.

## Map of the tree

| Path | What it is |
|---|---|
| `src/bigint.rs` | `BigUint`, `BigInt`, `Sign`, `MontgomeryCtx`, `BarrettCtx`, the multiply ladder, Algorithm D, radix I/O |
| `src/number_theory.rs` | gcd family, symbols, square roots, primality, CRT, reconstruction, trees |
| `src/gf2m.rs` | GF(2^m) |
| `src/poly.rs` | `PolyZ`, `PolyModP` |
| `src/lattice.rs` | integral LLL |
| `src/random.rs`, `src/scrub.rs` | sampling; the single audited `unsafe` |
| `scripts/bench_primitives.sh` | measurement — read its reduction before changing anything |
| `scripts/check_bench_consistency.py` | the self-consistency guard |
| `scripts/build_performance.sh` | assembles PERFORMANCE.md; **the prose lives here**, not in the generated file |
| `scripts/perf_analysis.py` | tables and SVGs |
| `scripts/lll_oracle.py` | independent rational LLL, the oracle `src/lattice.rs` tests against |
| `MANUAL.md` / `tests/manual_examples.rs` | documented API, mirrored as tests |
| `manual.tex` / `manual.pdf` | the LaTeX reference manual, with the defining equations per primitive; every listing is extracted and executed against the crate before a rebuild (`pdflatex manual.tex`, twice) |
| `CITATIONS.md` | primary sources |
| `REQUESTS.md` | the consumer's list, cleared |
| `ROADMAP.md` | proposed scope, pending triage |

`PERFORMANCE.md` is **generated**. Edit
`scripts/build_performance.sh` and regenerate; edits to the document are lost.
