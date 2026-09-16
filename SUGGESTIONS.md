# Rump suggestions

> **Motto:** better that, better algorithms
>
> **Creed:** Experiment is asking God for peer review.

2026-09-16. Evidence and reviewed source identity: [AUDIT.md](AUDIT.md).
These are proposed changes; no implementation or speedup is claimed here.
Implement from papers, specifications and independently derived mathematics.
Preserve the exact hypotheses, representation and invariants beside the algorithm.
An experiment records its source/dependency identities, features, input, seed,
measurement rule and acceptance criterion before its validation run.

## 1. Make lattice search numerically sound and explicit about completion

Addresses **R1/R3**. Keep exact integral LLL as the basis preparation. For
shortest/closest enumeration, derive an exact rational LDLᵀ decomposition of
the integral Gram matrix, or enclosing intervals with adaptive precision.
A branch may be pruned only when its certified lower norm bound exceeds the
radius. Use approximate centers for ordering candidates, not for irrevocable
exclusion without an error bound. Escalate precision when intervals overlap a
branch boundary; retain an exact fallback for small dimension.

Return candidates, visits, a numerical outcome and whether enumeration was
exhausted. Distinguish “best candidates found within budget” from “all candidates
within this bound.” Recheck every emitted norm/distance in exact arithmetic.
Do not call a valid form nonpositive merely because a floating approximation
loses a direction.

**Experiment:** enumerate small integral lattices independently over bounded
integer coefficient boxes. Compare full sets, not only the first norm. Include
boundary equality, nonorthogonal bases, targets outside the span, forms with
widely separated diagonal scales, and changes of integral basis with determinant
±1. The identity basis with `diag(1,2^1000)` must return its known candidates
without numerical failure. Force low visit budgets and check the incomplete
status. In factoring, compare shortlist quality and complete polynomial-search
cost; certified pruning is useful only if its cost is justified.

## 2. Give numerical primitives defined domains and reproducible coefficients

Addresses **R2/R4**. Use `ln Γ(x)=ln Γ(1+x)−ln x` for small positive x, avoiding
an overflowing intermediate Γ-like quotient. Define behavior at zero, negative
arguments, NaN and infinity. Publish an absolute-error target near log-gamma's
zeros and an appropriate scaled-error target elsewhere.

For the chosen Lanczos approximation, derive and generate the coefficients at
higher precision than f64, recording the mathematical construction, parameter,
rounding and residual checks. Separate approximation error, rounded-coefficient
error and evaluation error. A table is accepted because its construction and
errors are known, not because its decimals appear in another implementation.

**Experiment:** sweep logarithmically from the least positive subnormal through
normal positive values, with dense probes around 0.5, 1 and 2. Check the gamma
recurrence, `Γ(1/2)=sqrt(pi)`, integer factorial identities and independently
computed high-precision values. Exercise entropy's gamma/χ²/beta consumers on
both sides of their numerical branch boundaries. Extreme shapes are robustness
checks; calibrating ordinary statistical tails remains entropy's responsibility.
See [DLMF §5.5](https://dlmf.nist.gov/5.5) for the recurrence and reflection identities.

## 3. Reduce GF(2) work with certified composition tracking

The existing structured filter already carries row compositions. Keep the
invariant `filtered_row = XOR(original_rows named by composition)` through
merges, compaction and purging. A dependency must expand to a nonempty original
combination whose XOR is zero.

A row retirement with fill Δ changes the iterative-solve proxy `r*W` to
`(r−1)*(W+Δ)`, so it improves that proxy when `(r−1)*Δ < W`. Charge composition
length and expansion cost as well: a cheaper matrix can have a more expensive
proof of its relationship to the input.

For parallel filtering, derive a batch of eliminations whose incident row sets
are disjoint, so their updates commute. The method is developed in
[Bouillaguet–Zimmermann, §4](https://perso.lip6.fr/Charles.Bouillaguet/static/publis/merge.pdf).
Keep policy and workload selection in factoring, with general arithmetic here.

**Experiment:** replay the same sparse matrices through serial and parallel
filters; verify every expanded dependency against the untouched input. Record
filter time, reduced dimensions, nonzeros, composition lengths, solve time and
peak memory. Accept a change on filter-through-extraction cost, not its merge
rate alone. Preserve a dense route for small remainders.

## 4. Measure the arithmetic actually used by each consumer

Rump already has schoolbook, Karatsuba, Toom, exact NTT, reciprocal division,
Montgomery/Barrett reduction, Lehmer/Half-GCD and batch inversion. Start with
profiles before introducing another implementation of an existing operation.
Select algorithm crossovers by operand size **and shape**: balanced multiplication,
unbalanced multiplication, squaring, small remainders and repeated moduli differ.

**Experiment:** stratify at every dispatch boundary, with zero/one limbs,
all-one limbs, long carry/borrow chains, nearly equal operands and sparse
powers of two. Verify `n=q*d+r`, Bézout identities, residue ranges and exact
polynomial reconstruction using independent slow arithmetic. Benchmark both
feature modes, and measure reuse of Montgomery workspaces and prepared bases
in real callers. Report setup and allocation separately from loop arithmetic.
A consumer using cryptography must be timed with the unified `wipe` feature.

## 5. Close the consumer matrix

Add factoring to the existing downstream checks. Test the proposed rump with
cryptography, entropy default, entropy minimal, and factoring minimal; the
latter two need fresh target directories. Record exact sibling commits rather
than assuming matching version strings establish compatibility.

Keep a fixed combination for reproducible release evidence and a moving
combination for integration discovery. Run ignored arithmetic stress tests and
supported architecture/MSRV checks before release. The current audit's 3,500
integer probes are useful evidence, not a replacement for large-width coverage.

| Work | Owner and required consumer check |
|---|---|
| Lattice numerical repair and completion result | rump; factoring's polynomial search |
| Log-gamma and coefficient derivation | rump; entropy's tails and factoring's E′ model |
| Scheme-specific primality/timing policy | cryptography; generic rump APIs keep their stated contracts |
| Statistical calibration | entropy; cryptography reuses only a matching calibrated rule |
| Factorization parameter and merge policy | factoring; rump verifies the general algebra |
