# Parallel structured filtering on real sieve matrices

Parallel filtering after Bouillaguet & Zimmermann was proposed for rump,
*Parallel structured Gaussian elimination for the number field sieve*,
Mathematical Cryptology 1 (2021), §4, to be accepted on filter-through-
extraction cost rather than merge rate. It was implemented and measured; it
is **not** on main, because it does not pay.

## What was measured

A `Filter::merge_batched` that takes merges in batches of columns with fresh
plans whose member rows are pairwise disjoint — the XOR of two rows that do
not hold a column cannot hold it, so every plan in a batch stays exact —
forms plans and row updates in parallel, and applies the bookkeeping serially
in fill order with the cost rule rechecked. Every row remains the XOR of its
composition, checked at several worker counts; dropping the disjointness
check fails those checks, which is what makes the rule the load-bearing part
of the design rather than an optimisation detail.

The implementation is not kept. It was written to answer the question below
and the answer was no, so what remains is the design, the inputs, and the
numbers — enough to rebuild it if a future workload changes the verdict.

## Inputs

Dumped by factoring (scratch build of factoring `c7ce350` on rump `4218630`,
on vinge, aarch64, 2026-09-16) immediately before its `filter_merge` call,
with the `weight_cap` and `excess` it passed. factoring's own serial timings
from those runs:

| Matrix | rows × columns | cap, excess | filter | solve | extraction | whole run |
|---|---|---|---:|---:|---:|---:|
| GNFS c90 | 277 122 × 283 588 | 16, 128 | 4.57 s | 7.54 s (Lanczos) | 7.44 s (dependencies and square roots) | 224 s |
| GNFS c70 | 48 434 × 49 429 | 16, 128 | 0.63 s | 0.49 s | 0.54 s | 17.5 s |
| QS 70 | 10 630 × 10 606 | 16, 24 | 0.16 s | 0.25 s (dense) | 0.01 s | 18.3 s |
| QS 60 | 4 795 × 4 771 | 16, 24 | 0.055 s | 0.032 s | 0.004 s | 0.88 s |

## Replay

`filter_replay_from_file` on the branch, release build, on dennard (AMD EPYC
7452, rustc 1.95.0), pinned to cores 32–47 (`taskset -c 32-47`) with load
average about 33 on the other cores. The serial replay reproduces factoring's
filtered dimensions exactly (c90: 64 335 rows, 64 207 live columns, 5 363 116
nonzeros). Filter times are the best of the two serial runs and of the
batched runs at 2, 4, 8 and 16 workers:

| Matrix | serial filter | batched filter | serial cost | batched cost |
|---|---:|---:|---:|---:|
| GNFS c90 | 8.80 s (merge 7.35 s) | 7.94 s (merge 6.25 s, 8 workers) | 3.4504e11 | 3.4512e11 |
| GNFS c70 | 1.23 s (merge 1.14 s) | 0.95 s (merge 0.85 s, 8 workers) | 9.6215e9 | 9.6227e9 |
| QS 70 | 0.314 s | 0.242 s (16 workers) | 8.5870e8 | 8.5842e8 |
| QS 60 | 0.105 s | 0.086 s (8 workers) | 1.7868e8 | 1.7864e8 |

Cost is rows × nonzeros, the Block Lanczos proxy. Every dependency found was
expanded and checked to vanish on the input.

## Decision

The batched filter saves 10–23% of filtering time at an unchanged solver
cost. On GNFS c90 that is 0.86 s of dennard's 8.80 s serial replay; scaled to
factoring's 4.57 s filter on vinge, which was not replayed batched, it would
be under 1 s of a 224 s factorization. The serial bookkeeping, the purge
(1.2–1.3 s at c90) and the heap work stay serial, so more workers do not
help beyond 4–8. A public `threads` argument on `filter_merge` would be a
breaking change for that gain; it is not proposed.

## Also observed

Block Lanczos (`block_lanczos_dependencies_sparse`) on the serially filtered
matrices returned no dependencies for 4 of 40 seeds on GNFS c70 and 0 of 16
on GNFS c90 (90–102 dependencies otherwise); results are identical at 1, 4
and 16 threads, as documented. A caller must retry with a fresh seed on
`None`.
