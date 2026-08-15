# HANDOFF — picking rump up on another machine

State as of commit `e9e9f87` on `origin/main`, tag `v0.2.1`, crate version
`0.2.1` (unreleased work sits on `main` above the tag). Everything below is in
the repository unless marked otherwise.

```sh
git clone git@github.com:darrelllong/rump.git && cd rump
cargo test && cargo test --release      # 170 lib tests, all green
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
LLL. `REQUESTS.md` records the list as cleared.

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

## Deferred defects, found and verified but not fixed

These came out of a line-by-line documentation pass over every file. None is a
correctness bug in current use; each is real. Roughly in value order.

**Performance**

- `number_theory.rs` `sdiv_step`: recomputes a remainder that `div_rem` already
  returned. Since `hi − 2^s = q·lo + rem`, the value it rebuilds with a
  full-width multiply and subtract is just `rem + 2^s`. This is the most
  executed multiplication in the guarded-division path (every splice repair in
  `hgcd`, every near-boundary step in `hgcd_base`). It also clones both
  operands per call, only to order them.
- `poly.rs` `PolyModP::div_rem`: inverts the leading coefficient
  unconditionally. Every divisor in the factorisation pipeline is monic, and
  `number_theory::mod_inverse` has no `a == 1` shortcut, so each call pays a
  full Euclid loop. `make_monic` already short-circuits on `lc.is_one()`.
- `poly.rs` `PolyModP::rem` builds and discards the whole quotient;
  `div_rem` sizes the quotient `deg self + 1` instead of
  `deg self − deg divisor + 1` (the ℤ paths size it correctly). Both are on the
  hottest path in the file (`gcd` → `squarefree`/`distinct_degree`).
- `poly.rs`: all three division loops build the subtrahend at full remainder
  length (`shift_up` then `scale` then `sub`, three allocations per step) where
  an in-place subtract over the affected window would do; and `scale` /
  `pseudo_div_rem` have no `c = ±1` fast path, so a monic divisor multiplies by
  unity across the whole quotient every step.
- `bigint.rs` `BarrettCtx::pow_mod` does not seed the accumulator from the top
  set bit, so every call wastes a squaring, a multiply and two reductions. The
  Montgomery ladder in the same file deliberately does the opposite.
- `bigint.rs` `sqrt_floor` calls `sqrt_rem` and discards the remainder, paying a
  full-width squaring and subtraction to produce it; `sqrt_rem` also clones a
  full-width value that is dead afterwards.
- `bigint.rs`: no unbalanced-operand multiplication path, so a long × short
  product falls all the way to schoolbook. Related: at exactly
  `long == 2·short` the Karatsuba admission test accepts a case the kernel then
  rejects back to schoolbook.
- `gf2m.rs`: `inverse` recomputes `bits()` twice per iteration where the values
  are already in hand; `reduce` takes its argument by value and then copies the
  limb buffer (wants an `into_limbs` on `BigUint`); `is_irreducible` clones the
  modulus per checkpoint.

**Contract and robustness**

- `gf2m.rs` `Gf2m::sqrt` returns a wrong value, silently, on a reducible
  modulus — squaring is a bijection only in a field. It is the one public
  routine here that neither panics nor returns `None` in that case; documented,
  not fixed, because the signature cannot carry it.
- `number_theory.rs` `selfridge_discriminant` is the last unbounded loop in the
  crate. Termination rests on the perfect-square exclusion and the character
  argument; every other search in the file is capped. Reachable from
  `is_probable_prime_bpsw`.
- `random.rs`: three rejection-sampling loops whose termination is a property of
  the generator, not the code — a zero-filled `Rng` hangs `random_below`,
  `random_nonzero_below` and `random_probable_prime`. Inherent to rejection
  sampling and now documented at module level; a retry cap would match the
  posture taken elsewhere (`equal_degree_split` caps at 256 stalled draws).
  Also `random_probable_prime` can never return 2, since it forces the low bit.
- `poly.rs` `squarefree_into` carries an `expect` whose message asserts a bound
  that only holds under the prime-modulus precondition.
- `bigint.rs` `shl_bits` abandons the old limb buffer unscrubbed where every
  sibling path scrubs first. Consistent with the crate's stated scope (freed
  buffers are not wiped), but it is the one place the scrub would be free.
- `gf2m.rs` `xor_shifted_word` writes to `buf[index + 1]`, in bounds only
  because `high` is provably zero at the boundary. Correct today, documented,
  and fragile against any change to tap collection or buffer sizing.

**Citations to check against the physical sources** (added by others, left
verbatim rather than guessed at): Rabin 1980's venue for `is_irreducible`;
*Guide to ECC* §2.3.5 and Algorithms 2.41–2.45 for tap-wise reduction; Knuth
§3.4.1 for `random_below`; IEEE Std 1363-2000 Annex A.4.7 for the even-degree
quadratic solver; whether Bodrato's optimised sequence really describes the
Toom interpolation as written (it reads as a first-principles Vandermonde
solve); and Dussé & Kaliski EUROCRYPT '90 as where the word-level `n₀'`
constant "was introduced". `CITATIONS.md` is also missing rows for
`Gf2m::trace`/`half_trace`/`solve_quadratic`, `Gf2m::is_irreducible`,
`Gf2m::pow`, the tap-wise reduction, and `random_below`, though the crate root
claims the table is complete.

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
| `CITATIONS.md` | primary sources |
| `REQUESTS.md` | the consumer's list, cleared |
| `ROADMAP.md` | proposed scope, pending triage |

Note that `PERFORMANCE.md` is **generated**. Edit
`scripts/build_performance.sh` and regenerate; edits to the document are lost.
