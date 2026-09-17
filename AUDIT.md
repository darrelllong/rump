# Rump audit — 2026-09-17

> **Motto:** better that, better algorithms
>
> **Creed:** Experiment is asking God for peer review.

## Scope and evidence

This review covers the captured sibling combination below on Apple M4 Pro,
`aarch64-apple-darwin`, rustc/Cargo 1.93.1, with separate Rust 1.87 checks.

| Repository | Captured HEAD |
|---|---|
| cryptography | `0242a217f1d79ab01bd43d4e5b79fc2a7be7a88f` |
| entropy | `63592e02ab50a494499a87c3abe0ab406ab01bf5` |
| rump | `ae7566b1b100239e1b511a9b05ff8229ea6613bd` |
| factoring | `732801274f7a27640b3616995b7503855a870e99` |

The reviewed-file manifest for this repository has SHA-256
`4e29e4ff3846e5c7fea88210bf13d6dfc9045bcf27a5d14ed3d7da44abd2cb77` (117 files).
[The manifest](review/2026-09-17/reviewed-files.sha256) contains sorted
`SHA256(file)  relative/path` lines; its own digest identifies the capture.
It covers tracked and nonignored regular files, excluding these two review
documents and the review artifacts added afterward. Entropy's final capture includes its new
seeding, sampling, thread-local and `CryptoRng` APIs through `63592e0`.

The review distinguishes reproduced results, source inspection, retained
measurements and proposed experiments. The files record current findings and
acceptance criteria; they do not implement the proposed changes. Implementation
references are papers, standards and mathematics. External libraries were called
through public APIs for comparison; their implementation source was not used.

## Assessment

The exercised integer, modular, polynomial, lattice and GF(2) suites pass in
both erasure modes. The probability API reports domain/nonconvergence errors,
and large equal-shape beta cases now return accurate central values. A fresh
small-shape counterexample nevertheless returns `Ok(1.0)` where the exact answer
is 0.5. Error-aware return types need stable normalization as well as a stopping
rule.

Rump should remain the exact arithmetic foundation. Its floating probability
functions have a better owner in entropy; its opaque modular arithmetic and
exact matrix/lattice support should serve the other projects without absorbing
their algorithm-selection policies.

## Findings

### R1 — High: incomplete beta's normalization underflows for tiny shapes

**Reproduced public-API failure.**
[src/number_theory.rs](src/number_theory.rs), `regularized_incomplete_beta` and
`incomplete_beta_tail`.

| a = b, x = 1/2 | Returned result | Exact result |
|---:|---|---:|
| 10^-310 | `Ok(1.0)` | 0.5 |
| 10^-200 | `Ok(1.0)` | 0.5 |
| 10^-160 | `Ok(0.500002783212028)` | 0.5 |
| 10^-100 | `Ok(0.5000000000000064)` | 0.5 |
| 10^6 | `Ok(0.5000000000000002)` | 0.5 |
| 10^10 | `Ok(0.49999999999999933)` | 0.5 |
| 10^12 | `Ok(0.5000000000000001)` | 0.5 |

The normalization evaluates `0.5 * ln(a*b/(a+b))`. At a=b=10^-200,
the intermediate product rounds to zero although the quotient's mathematical
value, a/2, is representable. The exponential becomes zero and the complement
branch returns one, which passes range validation. Finite output, successful
continued-fraction termination and a [0,1] check do not establish accuracy.

Symmetry gives `I_(1/2)(a,a)=1/2` for every a>0;
[DLMF 8.17.4](https://dlmf.nist.gov/8.17.E4) supplies the identity. Fix the
normalization and regime handling rather than special-casing just x=1/2.
Include tiny unequal shapes and nearby x values so that the general failure
cannot survive behind a symmetry shortcut.

Reproduce with:

```rust
let p = rump::number_theory::regularized_incomplete_beta(0.5, 1e-200, 1e-200);
assert_eq!(p, Ok(1.0));
```

The [retained client and output](review/2026-09-17/README.md) include this case.
The Student path has one beta shape 1/2, so this counterexample does not imply
failure of factoring's actual race. Eighteen Student cases, degrees of freedom
1, 2, 5, 10, 100 and 10,000 at probabilities 0.75, 0.975 and 0.999999, differed
from R's `qt` by at most 4.66e-10 relative. That is a sampled consumer check,
not a bound over the complete Student domain.

### R2 — Medium: arithmetic representation dominates some rho workloads

**Fresh consumer experiment; performance opportunity.** Factoring's generic rho
uses heap-backed opaque Montgomery residues even for one-word inputs, while it
already has a private native-word cofactor walker using `Montgomery64`. A matched
four-input experiment in [factoring's audit](../factoring/AUDIT.md) finds about
18–21× lower kernel wall time with the native path on this host. That is not a
whole-factorization speedup and does not establish a 128-bit or large-BigInt ratio.

Rump owns the native modular operations; factoring owns dispatch, walk budgets
and recovery. For larger inputs, inspect allocation and context ownership in
returned residues, as well as scratch reuse. Reusing multiplication scratch
does not eliminate every returned-value allocation or context reference count.
Measure before adding more abstraction.

The generic rho converts product residues out of the Montgomery domain before
GCD. For odd n and its invertible Montgomery scale R,
`gcd(R*d mod n,n)=gcd(d,n)`. A modulus-aware GCD operation on an opaque residue
can avoid decoding without exposing representation. It should validate the
domain and be exercised on 0, units, proper divisors and products equal to zero.

### R3 — Medium: floating statistics exceed the intended foundation boundary

**Source inspection.** `ln_gamma`, incomplete beta and Student quantiles sit
among integer number-theory routines. Entropy calls rump for log-gamma while
maintaining its own gamma tails and normal functions; factoring calls rump for
Student probabilities. This divides numerical error policy and testing across
owners without an arithmetic reason.

Move floating probability functions into entropy's statistics module/feature,
keeping exact BigInt, polynomial, finite-field, lattice and matrix operations
here. BigInt rejection sampling is naturally here because it depends on the
integer representation, provided bytes remain caller-supplied. No OS seeding,
thread-local RNG or factoring schedule belongs in rump.

### R4 — Qualification boundary: erasure and exactness have different contracts

The `wipe` feature explicitly changes limb/scratch erasure. It does not change
variable-time normalization, allocation, division or generic exponentiation into
constant-time arithmetic. Preserve separate timing claims in cryptography.

The exact lattice search reports `Exhausted`, `VisitLimit` and `NumericalLimit`;
a consumer must inspect the outcome before claiming completeness. GF(2) filtering
must preserve the composition of each filtered row and expand dependencies back
to original rows. These are supported by the exercised tests; no new exhaustive
large-width or adversarial campaign was performed here.

### R5 — Medium: Block Lanczos needs a self-contained equation-to-state record

**Source inspection.** [src/gf2.rs](src/gf2.rs)'s recurrence commentary names
Montgomery's equations (18)–(20) and Figure 1, but also relies on an external
implementation-oriented account of the indices. The review did not consult that
implementation. Replace that explanatory dependency with a complete mapping from
paper symbols to stored blocks, transposes, selected subspaces and update order.

Final dependency checks establish that returned vectors annihilate the original
matrix. They do not by themselves show that the recurrence retains the intended
subspace or avoids unnecessary fallback. Add independently derived small-matrix
step invariants and compare spans/ranks with dense elimination. This is a
mathematical traceability and completeness/performance issue; no incorrect
returned dependency was reproduced.

## Controls and boundaries checked

| Boundary | Evidence in this pass |
|---|---|
| Tiny beta and large central beta | R1 table from current public API |
| Upper finite log-gamma | `2.557e305`, `2.558e305`, `2.559e305` return finite values about 1.795595e308, 1.796298e308, 1.797002e308 |
| Student consumer region | 18 successful cases compared with independent R `qt` values |
| Arithmetic and matrix identities | Existing division/modular/polynomial/GF(2)/lattice regressions pass |
| Consumer feature query | `consumer_matrix.sh --self-test`: absent wipe→0; present wipe→1; failed query→2 |
| Factoring feature graph | Successful graph resolution; wipe absent |

## Fresh verification

| Check | Result |
|---|---|
| Release, offline/locked, all targets | 407 passed; 21 ignored |
| Same with `--features wipe` | 408 passed; 21 ignored |
| Release doctests | 7 passed |
| Rust 1.87 all-target check | Passed |
| Consumer feature-query self-test | Three cases passed |
| Four captured consumer/default combinations | Functional suites passed; see their individual records |

The current consumer script records revisions and lockfiles and distinguishes
query failure from an absent feature. The fresh local tests are not a release
preflight for factoring's separately pinned entropy revision. No full ignored
campaign, new coefficient sweep, Linux/i686 campaign or timing qualification was
run. [Validation records](review/2026-09-17/validation.json) retain the command
identities and log digests.

## Cross-repository ownership

Keep the four repositories, with a focused boundary refactor. The desired graph
is `cryptography → rump`, `entropy → cryptography` when crypto generators are
enabled, and `factoring → rump + entropy` with only the RNG/statistics features
it needs. Rump must not depend on either consumer.

| Owner | Keep here | Boundary change |
|---|---|---|
| rump | BigInt, modular arithmetic, primality, exact polynomial/finite-field/GF(2)/lattice support, caller-driven BigInt sampling | Move floating probability kernels out; retain reusable arithmetic without factoring policy or OS entropy |
| cryptography | Ciphers, hashes, authenticated schemes, DRBG mechanisms, cryptographic state evolution and erasure | Own Hash_DRBG, HMAC_DRBG and fast-key-erasure cores; entropy supplies their adapters |
| entropy | Noncryptographic PRNGs, OS seeding, sampling, stream views, thread-local access, probability functions and test batteries | Separate application RNG, statistics and batteries by features; make FFT/battery dependencies optional |
| factoring | Rho/ECM/QS/GNFS orchestration, relation/cofactor policy, polynomial selection and size/cost dispatch | Reuse native modular arithmetic; keep schedule, graph forecasting and algorithm selection here |

Generic exact algebra in rump is supporting mathematics, not a reason to move
QS/GNFS policy there. `ln_gamma`, incomplete beta and Student quantiles are
floating statistical functions; entropy already owns most probability kernels
and factoring already depends on entropy. Move them in a coordinated API release
with reference fixtures. A rump forwarding wrapper that calls entropy would
create a dependency cycle and is unsuitable.

Preserve the distinction between rump's quality-neutral `RandomSource`,
cryptography's byte-oriented `Csprng`, and entropy's generator/`CryptoRng`
interfaces. Add explicit adapters with documented security and byte-stream
contracts; never blanket-implement a cryptographic contract for every test RNG.
A marker describes a construction, not the entropy in a caller-supplied seed.

Cryptography enables rump's additive `wipe` feature. Entropy default inherits it;
entropy minimal and standalone factoring do not. Record the resolved graph in
benchmarks: compiling factoring alongside a consumer that enables wipe can change
its arithmetic costs. Separate processes/packages may be needed when measuring
that configuration. Optional features should remove unwanted dependencies, not
silently weaken a cryptographic build's erasure contract.

## Standard for accepting changes

Derive the formula and state its domain, representation and invariant. Retain
published known answers, independent mathematical identities and reproducible
coefficient/table generation. Test boundary strata and algorithm switches as
well as ordinary inputs. Source comments should explain the invariant, assumption
or non-obvious choice and cite the relevant paper section when useful.

Use paired measurements with fixed inputs, seeds, compiler, target, features and
sibling revisions. Record wall time, total process-tree CPU, memory and work
counters. Separate the cost of setup, steady-state work and teardown, then report
the complete operation too. Statistical acceptance, semantic security, exact
factorization and performance are separate claims with separate evidence.
