# Rump suggestions — 2026-09-17

> **Motto:** better that, better algorithms
>
> **Creed:** Experiment is asking God for peer review.

Current evidence and limitations are in [AUDIT.md](AUDIT.md). Each proposal below
has an acceptance experiment. Predicted improvements are not measured speedups.

## Priorities

| Order | Work | Acceptance evidence |
|---|---|---|
| 1 | Stable incomplete-beta normalization across small shapes | Exact identities and high-precision unequal-shape/tail fixtures |
| 2 | Coordinate floating-statistics ownership with entropy | One numerical owner, coherent errors, no dependency cycle |
| 3 | Support native and reusable modular operations for rho | Correct domain identities and measured consumer cost |
| 4 | Price sparse algebra and exact search on GNFS workloads | Full filter/solve/expand and search costs, with unchanged invariants |
| 5 | Maintain the complete consumer/feature matrix | Identified revisions, successful graph checks and release qualification |

## Fix the numerical mechanism, then move the module

Derive a log-beta normalization that avoids `a*b` underflow and overflow, with
appropriate small-parameter recurrences. A sum/difference of logarithms avoids
that product but still needs cancellation analysis for large or very unequal
shapes. Choose continued fractions, series and asymptotic branches according to
a declared accuracy target. Successful iteration is one condition, not the full
error certificate.

Use exact `I_(1/2)(a,a)=1/2`, `I_x(a,1)=x^a` and
`I_x(1,b)=1-(1-x)^b`, with independently stable reference evaluation. Sweep
logarithmic shape strata, x near both endpoints, central regions and every
algorithm switch. Include representable neighbors of underflow/overflow
boundaries. Evaluate the smaller tail directly or return a log tail where needed;
clamping probabilities cannot replace an error model. The mathematical starting
point is [DLMF §8.17](https://dlmf.nist.gov/8.17).

Keep the existing explicit error path through factoring: an unavailable Student
bound retains candidates or reports inability to decide. It must not remove an
arm as though a valid comparison had succeeded. Expand the sampled Student grid
to the actual race's degrees of freedom and corrected tail levels, with
high-precision reference values.

Coordinate the relocation to entropy with that crate's incomplete-gamma work.
A sound log-gamma alone cannot repair cancellation in downstream combinations.
Move the functions and reference-generation artifacts as one reviewed unit in a
coordinated API release. Do not create a rump→entropy forwarding dependency or
maintain two permanently diverging numerical kernels.

## Improve the arithmetic interface at measured boundaries

Let factoring dispatch to existing `Montgomery64` and `Montgomery128` operations
before asking the general BigUint representation to handle the same width.
Those decisions belong in factoring. Here, make native domain contracts and
boundary tests complete: odd/even modulus rules, near-word maxima, carry chains,
conversion and multiplication/square identities.

For generic residues, consider destination-buffer operations and a fused
square-plus-constant kernel only after allocation profiles identify the cost.
Retain the modulus/context identity check. Benchmark a complete rho batch,
including product accumulation, budget accounting, GCD and replay, rather than
only modular multiplication.

Expose an operation equivalent to `gcd_with_modulus` on an opaque Montgomery
residue if it saves consumer work. The proof is invertibility of R modulo odd n;
it permits GCD on the encoded residue. Verify the proof's hypotheses in the API
and preserve opaque representation. Test zero, multiples of each proper factor,
units, maximum-width odd moduli and residues from a different context.

## Keep exact algebra useful to the whole pipeline

Write the Block Lanczos recurrence directly against Montgomery's equations and
Figure 1: one table for paper symbol, dimensions, stored representation and update
point, plus the identities checked at each step. Replace implementation-oriented
provenance prose with that derivation and precise paper citations. Retain any
required attribution; an explanatory claim is not evidence of a mathematical
invariant. Use dense elimination as an independent small-matrix oracle for the
span/rank and verify selected-subspace identities before the final residual check.

GF(2) work should retain `filtered row = XOR(original rows in its composition)`
through every merge, compaction and dependency. Measure row/column counts,
nonzeros, composition lengths, memory traffic, solver work and expansion cost.
For GNFS, include the cost and height of products sent to square root. A smaller
matrix with longer/heavier compositions need not be a faster factorization.

Keep lattice enumeration completeness separate from returned-vector validity.
Use independent bounded small-dimensional enumeration and integral changes of
basis. Force both visit and numerical limits. A faster incomplete search cannot
replace an exhausted exact slice oracle under the same interface contract.

Profile multiplication, squaring, remainder-only division, repeated moduli and
prepared contexts on the operand sizes actually emitted by QS/GNFS and
cryptography. Include setup, allocation, wiping and destruction in a full-workload
comparison. Require evidence before revisiting a crossover or parallel filter;
retained benchmarks are baselines, not portable constants.

## Keep dependency and release evidence explicit

Run both rump feature modes, cryptography default/all features with required
OpenSSL checks, entropy application/minimal/statistical modes and factoring.
Keep the successful feature query as a separate assertion. Record exact sibling
revisions, lockfiles, compiler and target. A moving local integration combination
and a fixed release-manifest combination have different identities and need
separate results.

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
