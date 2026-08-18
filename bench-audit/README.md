# Rump public-API performance audit harness

Implements `BENCH-TASK.md`. Pilot Benchmark Framework is the sole statistical
authority: nothing here computes a confidence interval.

## Layout

| Path | What it is |
|---|---|
| `corpus-gen/` | deterministic corpus generator; depends on nothing, so a corpus cannot vary with a Rump revision |
| `corpus.sha256` | checksums pinning the generated corpus; `setup.sh` verifies them |
| `adapter-v022/`, `adapter-v030/` | one executable per revision, path-depending on that tag's worktree |
| `abba/` | the paired wrapper: runs `baseline, candidate, candidate, baseline` and prints one CSV row |
| `scripts/setup.sh` | worktrees from tags, corpus, all executables |
| `scripts/run_case.sh` | one deciding cell: calibrate, run Pilot, capture artifacts |
| `cases.txt` | the paired matrix: every cell measured under both revisions |
| `cases-v030-only.txt` | APIs introduced in v0.3.0, measured for an absolute baseline only |

`adapter-*/src/shared.rs` and `src/main.rs` are byte-identical between the two
revisions apart from a version banner; only `src/cases.rs` differs, and only
where the 0.3.0 rename forces it. A *case name* is harness vocabulary and
means the same workload under both revisions even where the function it calls
was renamed — `mod_sqrt` is the case, and v0.2.2 reaches it through
`sqrt_mod`.

`cases-v030-only.txt` holds cells with no v0.2.2 counterpart: GF(2) linear
algebra, real-root isolation, word reciprocals, smoothness-base construction
and weighted 2D reduction. The paired wrapper never runs on them, because a
ratio against a function that does not exist would be a fabrication.

## What a digest can and cannot say

Every cell's results are digested, and the digests must agree across the two
revisions before any timing of that cell means anything. A digest proves the
two revisions *agree*; it cannot prove either one computes what the case name
claims. Three cases passed the cross-revision check while measuring nothing of
the kind:

- `mod_sqrt` was given a random odd modulus, which is essentially never prime,
  so every call took the rejection exit and no root was ever computed. It now
  uses stated primes in both congruence classes mod 4.
- `remainder_tree` was given `root + 2`, which makes every remainder equal
  to 2.
- `lattice_lll` digested its row length, so all three dimension-8 shapes
  agreed trivially.

`--emit results` prints the canonical results rather than their digest, which
is how each of those was caught. A new case should be read once through
`--emit results` before it is trusted.

## Rerunning from committed state alone

```sh
bench-audit/scripts/setup.sh
bench-audit/scripts/run_case.sh int_mul pair-1024-dense-32 bench/audit-v0.3.0/<host>/<session>/int_mul-1024
```

`setup.sh` creates the worktrees from the `v0.2.2` and `v0.3.0` tags, so no
uncommitted sibling state is involved.

## Timed-region contract

Outside the timed region: corpus parsing, operand construction, calibration,
context construction where the case measures reuse, correctness checking, and
digest construction. Inside: only the named operation, its result consumed with
`std::hint::black_box`.

Each case declares its own shape. `barrett_new` and `montgomery_new` measure
construction; `barrett_mod_mul` and `montgomery_mul_residue` build the context
outside the timed region and measure steady-state reuse.

## Correctness gates the timing

Every adapter can emit a deterministic FNV-1a digest of its canonicalized
results. `abba` collects a digest with every one of its four child timings and
refuses to print a row — exiting non-zero instead — if any disagree. Pilot
therefore cannot receive a sample in which the two revisions did different work.

## Measured feasibility of the mandated settings

`--preset strict --confidence-level 0.99 --ci-perc 0.04` is expensive for a
reason worth recording before anyone budgets the full matrix.

The CI width requirement is not the binding constraint. On `int_mul` at 1024
bits the ratio CI reached 0.019 against the 0.038 allowed within the first
minute. What binds is Pilot's autocorrelation handling combined with `strict`'s
200-subsession-sample floor: consecutive ABBA readings are serially correlated,
Pilot merges *N* readings into one subsession to remove that, and then needs
200 subsessions, so the requirement becomes `200 × N` readings.

Measured on the M4 Pro, `int_mul`/1024-bit, as the per-child measured interval
grows:

| interval per child | merge factor *N* | readings required |
|---|---|---|
| 20 ms (the task's floor) | 165 | 33 000 |
| 20 ms, early estimate | 12 | 2 400 |
| ~100 ms | 5 | 1 000 |

The merge factor is itself estimated from the accumulated data and moves as the
session runs, so these are observations rather than constants. The direction is
stable and has a clear cause: a longer measured interval averages more work per
reading and decorrelates consecutive readings.

`run_case.sh` therefore calibrates to the 20 ms floor and multiplies by
`RUMP_INTERVAL_MULT` (default 5). The floor is a minimum in the task, not a
maximum, so a longer interval is compliant.

Even so, a deciding cell is minutes rather than seconds, and the full matrix on
three hosts in three sessions each is a multi-day run. That is a property of the
requested statistics, not of this harness.
