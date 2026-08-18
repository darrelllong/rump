# Rump public-API performance audit

## Objective

Measure Rump as a general-purpose multiprecision and algebra library. The
audit covers the whole public surface: integers, conversions, modular
arithmetic, number theory, batch operations, polynomials, GF(2), GF(2^m),
lattices, and random sampling.

Factoring is not the benchmark and contributes no weighting to the result. It
may be run separately as an integration canary when a change specifically
claims to improve factoring, but it must not decide Rump's kernel thresholds
or support a general Rump performance claim.

For operations present in both releases, compare `v0.3.0` with `v0.2.2`.
For APIs introduced after `v0.2.2`, establish the first reproducible baseline
at `v0.3.0`. Do not summarize the audit as one overall speedup: any such number
would impose an arbitrary workload mix on a general-purpose library.

The production restriction remains absolute: pure portable safe Rust,
`#![forbid(unsafe_code)]`, no FFI, assembly, intrinsics, raw-pointer tricks,
target-specific representations, or public benchmark hooks. Benchmark-only
code must also be safe Rust. Host-side shell orchestration may use ordinary
portable process tools, but a result must not depend on a platform-only code
path in Rump.

## Required result

Produce all of the following:

1. a deterministic benchmark driver for every workload below;
2. a paired `v0.2.2`/`v0.3.0` runner using identical inputs;
3. raw Pilot output and a machine-readable manifest for every run;
4. a generated result document containing means, Pilot confidence intervals,
   candidate/baseline ratios, and classifications;
5. a list of regressions, improvements, inconclusive cells, and portable
   crossover findings; and
6. a rerunnable command sequence that does not rely on uncommitted sibling
   state.

Do not edit `PERFORMANCE.md` until the new run is complete and its internal
consistency checks pass. Do not hand-transcribe measured numbers.

## Statistical authority: Pilot Benchmark Framework

Pilot Benchmark Framework (`pilot-bench`, whose CLI executable is `bench`) is
the sole authority for confidence intervals, convergence, subsession sizing,
and autocorrelation handling in this audit. Do not calculate a replacement
normal interval or bootstrap interval in Python. Reduction scripts may read
Pilot's mean and CI, calculate the interval endpoints, calculate effect sizes,
classify a cell under the rules below, and render tables.

Run deciding measurements with:

```text
--preset strict
--confidence-level 0.99
--ci-perc 0.04
```

Thus Pilot must produce a 99% confidence interval whose total width is at most
4% of the mean. The ratio PI must be marked `must_satisfy=1`. Preserve Pilot's
chosen subsession size, subsession variance, autocorrelation coefficient,
sample count, raw readings, and session duration.

A session limit is a safety stop, not success. Pilot exit 13, a missing CI, a
CI wider than required, or too few subsession samples makes that cell
**inconclusive**. Do not use `|| true`, publish a mean from a stopped session,
or silently accept a nonconverged heavy operation. Increase the budget or
reduce the case into honest conditioned workloads.

The existing `scripts/bench_primitives.sh` computes its own IID normal CI and
therefore is not the deciding runner for this task. It may still supply raw
order statistics. Its generated prose also currently says the CI comes from
Pilot even though the displayed value is computed in Python; correct that
description when the new runner lands.

## One Pilot reading: paired ABBA comparison

Fresh clock/PID-seeded operands prevent exact reproduction and give the two
revisions different work. Replace that protocol for this audit with named,
deterministic corpora. Every case has a recorded corpus seed and case number.
Both revisions must receive the identical serialized inputs or generate the
identical inputs from the same specified algorithm and seed.

One target-program invocation by Pilot must run this balanced sequence:

```text
baseline, candidate, candidate, baseline
```

Each child reports an internally measured time, excluding input generation
and correctness checking. Average the two baseline readings, average the two
candidate readings, and print one CSV row:

```text
candidate_over_baseline,baseline_ns_per_op,candidate_ns_per_op,valid
```

ABBA places the average observation time of both revisions at the same point
and cancels first-order thermal or frequency drift. Each child must perform
the same declared warmup before its measured interval. The wrapper must fail,
rather than print a timing, if either child fails or the results disagree.

Drive the row with a Pilot PI specification equivalent to:

```text
candidate_over_baseline,,0,0,1:baseline,ns/op,1,0,0:candidate,ns/op,2,0,0:valid,,3,2,0
```

`candidate_over_baseline` is an ordinary arithmetic value (PI type 0), not a
throughput/harmonic-mean PI. The other columns are recorded for diagnosis;
the ratio is the PI whose CI must converge. The binary `valid` column must be
one in every reading, and any zero is a correctness failure, not a statistical
sample.

For APIs absent from `v0.2.2`, run the same arrangement using two independently
built copies of `v0.3.0`. That null comparison first proves that the harness
does not manufacture a directional result. Then run a single-revision Pilot
session to establish the absolute baseline.

The benchmark implementation and input generator must be byte-identical
between revision adapters except where the public API rename requires a small,
reviewed adapter. Build separate executables: do not link two versions of the
same `rump` library into one process.

## Timed-region contract

Each case must state what it includes. In particular, distinguish:

- constructor/setup cost from reuse cost;
- allocating operations from scratch/output-reusing operations;
- a one-shot call from a shared-context batch;
- input generation from the operation being measured; and
- a hot operand from a streaming working set.

Input generation, prime hunting, matrix construction, parsing of the corpus,
and result validation stay outside the timed region unless that operation is
itself the named benchmark. Returned values must be consumed with
`std::hint::black_box`.

Calibrate the inner repetition count so one reported child measurement lasts
at least 20 ms. Calibration is outside the timed result. An operation that
already takes longer than 20 ms runs once. Record repetitions and operations
per corpus traversal.

Every applicable primitive gets two modes:

- **hot**: repeat a small fixed operand set to measure the steady-state
  arithmetic kernel; and
- **streaming**: cycle through distinct operands in a working set at least
  twice the last-level cache available to the pinned benchmark CPU, with a
  portable minimum of 64 MiB.

Do not use a hot microbenchmark to justify a change that widens a structure or
adds data streamed beside the arithmetic. Conversely, do not use a streaming
test to claim the arithmetic kernel itself became slower without the hot row.

## Workload matrix

Use a covering design rather than the complete Cartesian product, but include
every stated input class and every dispatch boundary. The generator must emit
the selected case list before measurement so omissions are visible.

### Integers and conversion

Measure construction, byte import/export, binary/decimal/hex parse and format,
comparison, shifts, bit access, addition, subtraction, multiplication,
squaring, division/remainder, integer powers, and roots where public.

Required operand classes:

- dense uniformly scattered limbs;
- sparse values;
- all-ones and carry/borrow-heavy values;
- values immediately below and above powers of two;
- balanced widths and width ratios near 3:2, 2:1, and 8:1; and
- division with full-width, half-width, and word-width divisors, including
  near-equal operands, exact multiples, and small quotients.

Use ordinary widths 64, 256, 1024, 4096, 16384, and 65536 bits. Around every
schoolbook/Karatsuba/Toom, specialized-square, unbalanced-product, radix, and
division dispatch constant, also measure the immediately preceding, exact,
and immediately following limb or digit widths. Include the large Toom and
Half-GCD transitions even though they require a separate long-running group.

### Modular arithmetic and number theory

Measure separately:

- Barrett and Montgomery context construction;
- reduction, encoding, and decoding;
- allocating Montgomery operations and their scratch-reusing forms;
- modular multiplication, square, inverse, square root, and exponentiation;
- gcd, extended gcd, lcm, Jacobi/Kronecker/Legendre symbols;
- rational reconstruction and CRT;
- probable-prime tests, explicitly based Miller-Rabin, Lucas/BPSW; and
- valuation and factor removal.

Use word, 256-, 1024-, 2048-, 4096-, and 8192-bit moduli where the operation is
meaningful. Cover odd and even moduli on APIs that support both. Exponents are
0, 1, 17-bit `65537`, random 256-bit, and full-width. Measure them; do not
derive a full-width result by multiplying the 256-bit row. Window selection,
precomputation, leading zeros, and setup make that claimed linear conversion
too strong to serve as evidence.

Condition variable-time number-theory cases instead of mixing unlike exits:

- random coprime pairs, pairs with a planted common factor, and
  Fibonacci-like Euclidean cases;
- invertible and noninvertible residues;
- composites rejected by trial division, composites reaching Miller-Rabin,
  probable primes, and the documented pseudoprime regression corpus; and
- quadratic nonresidues, the `p mod 4 = 3` shortcut, and Tonelli-Shanks descent
  cases grouped by `v2(p - 1)`.

Each Pilot reading traverses the complete fixed corpus for its condition. It
does not draw a different mathematical mixture on every subprocess launch.

### Batch algorithms

Measure product trees, remainder trees, batch inversion, `SmoothnessBase`
construction and reuse, and batch smooth parts at batch sizes 1, 8, 64, 512,
and 4096. Cross operand widths 64, 256, and 1024 bits with small and large
bases through a covering design. Report setup and reuse separately.

Include `WordReciprocal` construction and repeated `rem`/`div_rem` at word
scale. This is a general primitive; its negative result inside one streamed QS
structure does not decide its usefulness elsewhere.

### Polynomials

Measure `PolyZ` and `PolyMod` construction, add/subtract, multiplication,
squaring, evaluation, division/remainder, gcd, resultant, discriminant,
factorization, roots, and the public monic/product/substitution/lift helpers.

Cover:

- dense and sparse coefficients;
- balanced and unbalanced degrees;
- coefficient widths 8, 64, and 256 bits, plus one 1024-bit case;
- degrees 2, 8, 32, 96, 128, 192, 256, and 512 where meaningful; and
- the exact neighborhoods of each polynomial multiplication and squaring
  cutoff.

For `real_roots`, use separate corpora for no real roots, distinct simple
roots, repeated roots, and tightly clustered roots. Validate multiplicities
and root intervals before accepting timing output.

### GF(2) linear algebra

Measure singleton pruning, dense null space, and Block Lanczos independently.
Use dense matrices for the dense solver and sparse matrices with row weights
3, 8, and 16 for Lanczos. Cover planted nullities 8 and 64, several rectangular
ratios, and singleton rates near 0%, 25%, and 75%.

For Block Lanczos, every result row must carry:

- total elapsed time;
- success/failure for the fixed solver seed;
- validity of every returned dependency; and
- rank of the recovered dependency span.

A faster `None`, invalid dependency, or lower recovered rank is not a speedup.
Run a fixed matrix across a recorded set of solver seeds. Use a Pilot binary PI
to report convergence rate with Pilot's binomial-proportion CI, and a separate
conditioned time PI for successful runs. Never fold failures into an
artificially fast time mean.

### GF(2^m)

Measure field validation/construction separately from add, multiply, square,
power, inverse, division, square root, trace, half-trace, quadratic solution,
and irreducibility testing. Include the existing degree-233 and degree-571
fields plus at least one smaller generic field. Use zero, one, sparse, and
dense field elements as distinct classes where fast paths differ.

### Lattices

Measure weighted two-dimensional Gauss reduction and LLL independently. Use
dimensions 2, 8, 16, and 32; coefficient widths 32, 128, and 512 bits; and
random full-rank, ill-conditioned, and nearly dependent bases. Validate the
reduced basis and transformation contract outside the timed region.

### Random sampling

Use a deterministic, safe-Rust `RandomSource` with a recorded byte stream.
Measure `random_below`, nonzero and coprime sampling, and probable-prime
generation. Include bounds immediately below a power of two and immediately
above half a power of two so both favorable and rejection-heavy paths appear.
Record bytes consumed as a non-timing metric; a faster sampler that changes
the distribution or consumes unboundedly more input is not accepted.

## Correctness during measurement

The ordinary and release gates must be green before performance collection.
The performance driver must also verify every corpus once per executable.
Unique results are compared exactly between revisions and, where practical,
against the existing independent oracle. Nonunique results are checked by
their mathematical invariant: dependency validity and rank, polynomial factor
product, root isolation/multiplicity, reduced-basis conditions, and so on.

Print a deterministic digest of canonicalized results. The paired wrapper
must compare digests before emitting `valid=1`. Correctness checking and digest
construction are outside the timed region.

## Host and build protocol

Run the deciding suite on at least:

- Apple M4;
- x86-64 Linux/EPYC; and
- AArch64 Linux, such as Raspberry Pi 5.

Use three clean worktrees: the external benchmark/orchestrator, `v0.2.2`, and
`v0.3.0`. Require clean trees and record the peeled commit of each tag. Build
both library revisions with the same exact `rustc`, release profile, dependency
lock, and environment. Record `rustc -Vv`, Cargo version, OS, kernel, CPU,
memory, power mode, and compiler flags.

Portable results use the ordinary release target without `target-cpu=native`.
A native-codegen run may be published as an additional observation but cannot
choose Rump's portable constants. Pin the process to one CPU where the host
provides a normal external facility, keep the machine on mains power, stop
unrelated heavy work, and let it reach a stable temperature. None of these
host controls may introduce a target-specific Rump implementation.

Run the complete matrix in three independent sessions on each host. Do not
pool them with a home-grown interval. Pilot owns the CI in each session, and a
classification is accepted only when all three sessions agree. A disagreement
is an inconclusive or host-sensitive result that must be reported as such.

## Classification rules fixed before the run

Let Pilot's 99% CI for `candidate_over_baseline` be `[low, high]`.

- **Pass:** `high <= 1.05`; the candidate is demonstrated to be no more than
  5% slower for that cell.
- **Regression:** `low > 1.05`; the candidate is demonstrated to be more than
  5% slower.
- **Improvement:** `high < 0.97`; the candidate is demonstrated to be at least
  3% faster.
- **Equivalent/no resolved change:** the interval contains 1.0 and
  `high <= 1.05`.
- **Inconclusive:** the interval crosses the 1.05 boundary, Pilot fails to
  converge, any session disagrees, or correctness/rank/convergence differs.

Do not call a statistically resolved 1% change important, and do not call an
unresolved 8% point estimate acceptable. The effect boundary and Pilot CI are
both part of the decision.

For an algorithm-dispatch change, measure a band on both sides of the proposed
cutoff. The selected engine must win by at least 10% throughout a useful
neighborhood on every required architecture. Choose one conservative portable
threshold from the intersection of the winning regions. Do not add
architecture aliases, target-dependent representations, unsafe dispatch, or
unportable tricks to rescue a threshold.

No family average can hide a failing cell. Family geometric means may be
shown as navigation aids only after every constituent row remains visible.

## Artifacts and audit trail

Store results under a versioned directory such as:

```text
bench/audit-v0.3.0/
  manifest.json
  cases.csv
  <host>/
    <session>/
      <case>/
        summary.csv
        readings.csv
        pilot-output.txt
```

The manifest records revisions, tags, toolchain, flags, host, corpus generator
version, seeds, case parameters, Pilot command line, and timestamps. The
generated Markdown report must link each table row back to its raw case
directory. Run the existing consistency checker and extend it to require:

- finite positive times and ratios;
- a Pilot-produced 99% CI of the requested width;
- means lying within their raw observed range;
- matching result digests;
- complete host/session coverage; and
- no result accepted from a stopped or failed Pilot session.

Commit the harness, generator, reducer, manifest, raw Pilot summaries, and
generated report separately from any production optimization. A subsequent
optimization commit must name the cases it intends to change and rerun those
cases plus every neighboring dispatch and streaming row before the complete
portable audit is refreshed.

## Definition of done

This task is complete only when:

1. all required cases exist and their corpus is reproducible;
2. the null `v0.3.0`-versus-`v0.3.0` comparison has no directional failures;
3. all common APIs have three agreeing Pilot sessions on every required host;
4. new APIs have an absolute `v0.3.0` baseline with Pilot CIs;
5. every correctness digest and invariant passes;
6. every regression and inconclusive cell is explicitly dispositioned rather
   than averaged away; and
7. the report can be regenerated solely from committed raw data.

