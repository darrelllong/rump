# Rump audit

> **Motto:** better that, better algorithms
>
> **Creed:** Experiment is asking God for peer review.

## Reviewed state and method

2026-09-16 PDT / 2026-09-17 UTC. Frozen sibling checkouts on
`aarch64-apple-darwin`, rustc/Cargo 1.93.1; separate Rust 1.87 checks.

| Repository | HEAD at capture |
|---|---|
| cryptography | `aa865da77502306f544b7031f65eaf3f7b7960b2` |
| entropy | `de1bd2d061eea43fe1fca291edae346315b85983` |
| rump | `66651ab0c82a21929c52da92823fad930766fef9` |
| factoring | `d594060dc824a4f2c3a0800fdeb86c739dce7e71` |

This repository's reviewed-file manifest SHA-256 is
`a0cb6832502d7bd1961846626aa589a32cd07ed238101dd9a4bf54bccd3a4aff` (114 files).
The manifest covers tracked and nonignored untracked regular files, excluding
AUDIT.md and SUGGESTIONS.md: sort relative paths, emit `SHA256(file)`, two
spaces, path and newline, then SHA-256 the UTF-8 manifest. It identifies working
contents as well as commits. Tests used fresh build directories in the frozen
copies. Later edits require checking which evidence still applies.

This review combines source inspection, mathematical identities, the release
suites, and focused boundary experiments. Reproduced failures, inspected risks,
retained measurements and proposed experiments are distinguished below. Coverage
is stated explicitly; passing suites do not establish every input domain,
platform, timing property or statistical null law. This review changes the two
review documents only. Diagnostic code ran in separate scratch copies.

## Assessment

Exact lattice enumeration now preserves the anisotropic test metric and
reports how a search ends. The coefficient generator reproduces all nine
log-gamma coefficients bit for bit, and the 50,298-point reference sweep passes.
The new numerical counterexamples are in incomplete beta and at log-gamma's
upper representable boundary. The integer, modular, polynomial and GF(2)
regressions in the exercised suites pass.

## Findings

### R1 — High: incomplete beta returns an uncertified last iterate, including negative probabilities

**Reproduced; public API.** [src/number_theory.rs](src/number_theory.rs),
`regularized_incomplete_beta` and `beta_continued_fraction`.
Symmetry gives `I_(1/2)(a,a)=1/2` for every positive a. Fresh results:

| a = b | Computed result | Exact result |
|---:|---:|---:|
| 1,000,000 | 0.49999961848374941 | 0.5 |
| 100,000,000 | 0.20461178871784746 | 0.5 |
| 10,000,000,000 | −5.776747550350140 | 0.5 |
| 1,000,000,000,000 | −66.96555717858665 | 0.5 |

The continued fraction returns `h` after at most 300 iterations, whether or
not the convergence test succeeds. The normalization also subtracts large
log-gamma values. Thus neither a finite result nor a result inside [0,1]
certifies accuracy. Clamping would conceal the failure.

Return explicit nonconvergence/numerical failure and use a stable regime-specific
normalization and expansion. Validate symmetry, tails, transition regions and
monotonicity against independent values. The exact identity follows from
[DLMF 8.17.4](https://dlmf.nist.gov/8.17.E4).

`student_t_quantile` calls this function; factoring's polynomial race calls that
quantile. These examples have two large equal shapes, while the Student path
has one shape 1/2. A failure of the ordinary factoring race was **not** reproduced.
Qualify that consumer's actual degrees of freedom and tail probabilities
separately before changing the shared kernel.

Minimal numerical witness:

```rust
let p = rump::number_theory::regularized_incomplete_beta(0.5, 1e10, 1e10);
println!("{p:.17}"); // -5.77674755035013998; exact value is 0.5.
```

### R2 — Medium: log-gamma overflows for some representable finite answers

**Reproduced; public API.** `ln_gamma(2.557e305)` returns positive infinity.
A 70-decimal-digit reference for the exact f64 input gives
`1.7955951755681236831711000242607365e308`, which rounds to the finite f64
`1.7955951755681237e308`. The same problem occurs at `2.558e305` and `2.559e305`.

In `(z+0.5)*ln(t) - t`, the product overflows before subtracting t. An asymptotic
rearrangement such as `x*(ln(x)-1)` avoids that specific intermediate; its
rounding and correction terms still need analysis. The positive-real Stirling
expansion and remainder bounds are in
[DLMF §5.11](https://dlmf.nist.gov/5.11).

The shipped dense sweep stops below `1e305`, so its passing result does not
cover this boundary. The small-positive recurrence is sound on the exercised
cases: `ln_gamma(1e-310)=713.8013788281542`. No ordinary entropy or factoring
run reaching the upper counterexample was demonstrated.

### R3 — Medium: a failed feature query can be reported as a successful absence check

**Source inspection.** [scripts/consumer_matrix.sh](scripts/consumer_matrix.sh)
uses `if cargo tree ... | grep -q '"wipe"'; then FAIL; else PASS; fi` for the
factoring feature leg. With `pipefail`, a failed Cargo query still takes the
`else` branch and prints that wiping is absent. Other failed test legs set the
overall failure flag, but this particular leg is not reliable evidence of
absence when its query fails.

Capture and check Cargo's exit status before inspecting its output. Exercise
three cases: successful graph without wipe, successful graph with wipe, and
failed graph resolution. The latter two must fail the leg with distinct reasons.
The fresh feature query in this review succeeded and showed no wipe in factoring.

The local consumer script includes factoring and both entropy feature modes.
The push workflow still includes only cryptography and entropy default. Use the
full local matrix as an integration gate wherever all four repositories are
accessible; do not treat the smaller hosted job as equivalent coverage.

## Mathematical boundaries checked

| Boundary | Current evidence |
|---|---|
| Lattice completeness | Exact integral Gram construction and fraction-free Gram–Schmidt; outward floating enclosures for pruning; exact returned distances. Existing exhaustive and change-of-basis tests pass. |
| Anisotropic metric | Identity basis, `diag(1,2^1000)`, bound 1: short search returns exactly two nonzero vectors; closest search at zero returns three including zero. Both report `Exhausted`. |
| Search budgets | The API distinguishes `Exhausted`, `VisitLimit` and `NumericalLimit`; the low-budget regressions pass. Exhaustion certifies the requested nearest subset, not retention of every point when `limit` is smaller. |
| Arithmetic dispatch | The suite includes independent slow arithmetic at dispatch boundaries, division identities, Half-GCD/Toom coverage and Montgomery checks. |
| GF(2) composition | Filtered dependencies are expanded and checked on the original matrix in the passing suite. |
| Wiping | Both feature modes pass. The feature changes erasure behavior, not variable-time arithmetic into constant-time arithmetic. |

The coefficient script solves the nine interpolation equations at 60-digit
precision. All coefficients match. On its sampled domain, the exact-coefficient
absolute error near the zeros is `2.26e-16`, and the rounded-coefficient error is
`6.19e-16`; the latter's sampled relative error above 3 is `9.5e-16`. These are
sampled maxima, not supremum proofs over the real domain.

## Fresh verification

| Check | Result |
|---|---|
| `cargo test --offline --locked --release --all-targets` | 401 passed, 21 ignored |
| Same with `--features wipe` | 402 passed, 21 ignored |
| `cargo test --offline --locked --release --doc` | 7 passed |
| `cargo +1.87 check --offline --locked --all-targets` | Passed |
| `scripts/lanczos_coefficients.py --check` | Nine bit-identical coefficients; sampled error report |
| Generated `--sweep` file, then ignored `the_log_gamma_sweep_matches_high_precision` | 50,298 values checked; passed |
| Public numerical/lattice client | R1/R2 reproduced; anisotropic searches exhausted correctly |

All four consumer default release suites pass on the recorded combination;
entropy minimal, cryptography all features and rump wipe also pass. This is a
fresh local matrix, not a claim that the release scripts or remote hosts ran.
The other ignored tests, fresh large-width randomized campaigns, i686/Linux,
assembly/timing audits and remote benchmarks were not repeated.

## Cross-repository contracts

| Owner | Obligation at the boundary |
|---|---|
| [rump](../rump/AUDIT.md) | Exact arithmetic and matrix identities; numerical domains, error and convergence; explicit search completion. |
| [cryptography](../cryptography/AUDIT.md) | Scheme-specific validation, randomness requirements, confidentiality/authentication profiles, timing and secret handling. |
| [entropy](../entropy/AUDIT.md) | Explicit input view, statistic, null law, calibrated decision and complete report; a statistical pass is not a security claim. |
| [factoring](../factoring/AUDIT.md) | Exact relation identities and dependency expansion, verified proper divisors, measured selection cost; probable-prime leaves are not proofs. |

The links assume sibling checkouts. Cryptography enables rump's additive `wipe`
feature. Entropy default inherits it; entropy minimal and factoring alone do
not. The same rump version string can therefore describe different timing and
allocation costs. Record the resolved dependency revisions, lockfiles,
features, compiler and target alongside results.

Implementations start from papers, specifications and mathematical derivations.
State the equation, representation, hypotheses and invariant. Derive numerical
tables reproducibly and separate approximation error from floating evaluation.
Use published answer files and independently constructed oracles for checks;
another implementation's source is not an implementation reference. A round
trip alone cannot detect a shared error in its two halves.
