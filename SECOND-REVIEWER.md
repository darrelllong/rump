# Second review of `rump`

## Scope and verdict

This is a hard review of the live repository, not only `HEAD`. The base commit
was `4d74efb`; the working tree also contained substantial uncommitted changes
to integer squaring, Barrett reduction, product trees, polynomial arithmetic,
documentation, and tests. I edited no code.

The arithmetic work is unusually well explained and well tested. I did not
find evidence of a wrong-answer defect in the exercised kernels. The ordinary
suite passed, and formatting, Clippy with warnings denied, and rustdoc with
warnings denied were clean on the final reviewed tree.

I would nevertheless block a release for two reasons:

1. the crate is advertised as pure safe Rust while production and test code
   deliberately use `unsafe`; and
2. the live `ProductTree` repair changes public function signatures without a
   corresponding compatibility plan or version change.

The first is also a division-of-labor problem: a variable-time arithmetic crate
for non-secret data is imposing partial cryptographic scrubbing, and its cost,
on every consumer.

## Findings

### 1. Must fix: the crate is not pure safe Rust, and the exception is in the wrong layer

`README.md` calls rump “pure, safe Rust.” `src/lib.rs:38-42` and
`src/lib.rs:70` instead establish a deny-with-exceptions policy. Production
`unsafe` is in `src/scrub.rs:53-58`, and a second exception is in the raw
read-back test at `src/bigint.rs:5332-5343`.

This is not merely wording. Every `BigUint` drop performs a volatile pass over
its live limbs (`src/bigint.rs:3753-3759`), and several shrinking and workspace
paths scrub as well. That creates three problems:

- It violates the project's stated pure-safe-Rust rule. `deny(unsafe_code)` is
  not an enforcement boundary when an inner item can replace it with `allow`;
  `forbid(unsafe_code)` is.
- It charges all arithmetic consumers, including factoring of public
  integers, for volatile writes on every temporary. Multiprecision algorithms
  manufacture many temporaries, so this is exactly where an unconditional
  extra memory pass is least welcome.
- It does not provide complete secret hygiene. The crate itself documents that
  reallocations, spare capacity, scratch retained by some APIs, copies, stack
  spills, swap, and core dumps remain. A partial guarantee is easy for a caller
  to overread as a security property.

The crate already says that it is variable-time, unsuitable for secrets, and
that cryptographic hygiene belongs at the consumer. Follow that boundary:
remove the volatile scrub policy and its raw-pointer test, delete the
production `unsafe`, and use `#![forbid(unsafe_code)]`. A consumer that handles
secrets needs a purpose-built constant-time and zeroizing representation; a
partial `Drop` pass on this general type is not a substitute.

This should be measured as well as enforced. Compare representative factoring,
modular exponentiation, polynomial multiplication, and allocation-heavy
number-theory workloads before and after removing the drop scrub. Kernel-only
benchmarks may understate the end-to-end gain.

### 2. High: `ProductTree` fixes an invariant by breaking the public API at version 0.2.2

The typed tree is a good correctness repair. A caller can no longer fabricate a
malformed `Vec<Vec<BigUint>>`, and `remainder_tree` can rely on the shape
established by `product_tree` (`src/number_theory.rs:1223-1347`).

However, the live change replaces:

```text
product_tree(...) -> Vec<Vec<BigUint>>
remainder_tree(&[Vec<BigUint>], ...)
```

with:

```text
product_tree(...) -> ProductTree
remainder_tree(&ProductTree, ...)
```

while `Cargo.toml` still says `0.2.2`. Return-type changes and parameter-type
changes are source-breaking. For a `0.x` crate, publish this as a new minor
line (for example `0.3.0`), or preserve compatibility under the old names and
introduce the typed surface under new names. Do not let a correctness
improvement become an unannounced downstream build break.

Also add a small API-compatibility/release checklist. The current test suite
proves arithmetic behavior, not whether public consumers still compile.

### 3. High: finish the ownership migration that the live polynomial work has started

The division rule in `REQUESTS.md` and `src/lib.rs` is the right one: general
arithmetic and algebra belong here; code that knows it is factoring belongs in
`factoring`. The live tree now implements several previously requested
operations, but the consumer still owns parallel implementations.

| Rump API now present | Consumer copy to retire |
|---|---|
| `PolyZ::balanced_base_expansion` | `factoring/src/gnfs/select.rs:393` |
| `PolyZ::roots_mod_prime_power` | lifting/counting machinery around `factoring/src/gnfs/select.rs:321` |
| `PolyZ::rem_monic` | `factoring/src/gnfs/sqrt.rs:234` |
| `PolyZ::product_mod_monic` | `factoring/src/gnfs/sqrt.rs:186` |
| `PolyZ::homogeneous_substitution` | `factoring/src/gnfs/lattice.rs:313` |
| `PolyModP::symmetric_lift` / `with_modulus` | `factoring/src/gnfs/sqrt.rs:401-418` |

Version and land the Rump API first, update the path consumer, then delete the
copies in the same integration change. Keeping both after the upstream API
exists creates two correctness surfaces and ensures later fixes land in only
one.

The remaining ownership moves are also correctly identified, but not yet
implemented:

- raw real-root and real-factorization numerics from
  `factoring/src/gnfs/model.rs`;
- dense null space, singleton peeling, and Block Lanczos from
  `factoring/src/qs/{linalg,lanczos}.rs`; and
- general two-dimensional weighted Gauss reduction from
  `factoring/src/gnfs/lattice.rs`.

Keep factoring-specific policy downstream: relation layout, factor bases,
smoothness bars, special-q scheduling, polynomial-selection scoring, lift
termination based on GNFS dependency sizes, and factor extraction. For the
algebraic square root, Rump should own reusable quotient-ring operations; the
consumer should own which dependencies and precision budget are worth trying.

### 4. High: the portability contract contradicts the project rule

`src/lib.rs:45-49` says that limb-count arithmetic assumes a 64-bit `usize` and
that 32-bit targets are unsupported. Elsewhere the crate is presented simply
as safe Rust. Pure Rust does not automatically mean portable Rust, but this
project explicitly rejects unportable shortcuts.

Remove unchecked `len * 64`, `len * 128`, and related index/shift assumptions
from public paths. Use checked size calculations and return or panic before an
unrepresentable allocation/index is formed. Add at least a 32-bit compile job
and, where practical, tests under a 32-bit target. If a target truly cannot be
supported, fail at compile time with a direct diagnostic instead of compiling
code whose documented failure mode is index overflow; that is still a
restriction, but it is an honest one rather than latent misbehavior.

### 5. Medium: the new performance thresholds are supported mainly by one machine

The new square and Barrett paths are thoughtful and have strong differential
tests. Their dispatch constants, however, are justified primarily by M4
measurements in `src/bigint.rs`. The 448-limb square handoff and 512-limb
Barrett half-product handoff depend on the relative costs of carries,
multiplication kernels, cache, and compiler output; those are not universal.

Before treating the constants as portable defaults, run the supplied probes on
the other architectures already named in `PERFORMANCE.md` (at minimum x86-64
and AArch64/Linux). Record both correctness boundaries and performance
crossovers. Prefer one conservative portable threshold unless the data shows a
large, stable reason for safe target-specific constants. No assembly, FFI, or
unsafe dispatch is needed.

### 6. Medium: checked-in documentation currently describes mutually exclusive states

`REQUESTS.md` has an “outstanding” section at lines 20-156, then says “Nothing
else outstanding” at line 162 and “Every entry this file has ever carried is
closed” at line 239. In the same live tree, several items in the outstanding
polynomial list have already been implemented.

Make the document a state machine rather than a diary:

- outstanding;
- landed in Rump, consumer migration pending;
- fully migrated; and
- deliberately downstream.

Each item should occupy exactly one state. This matters because the file is the
cross-repository ownership ledger, not incidental prose.

There is also a concrete threshold drift: `BigUint::square_ref` documents the
specialized range through 448 limbs (`src/bigint.rs:1474-1520`), while
`BarrettCtx::square_mod` still says 8 through 256
(`src/bigint.rs:3647-3651`). Generate such descriptions from shared constants
where possible, or keep one authoritative table and link to it.

Finally, `README.md` must not say “pure, safe Rust” while listing audited
unsafe exceptions later in the same file. Under the stated project rule, the
code should change; until then, the headline is false.

### 7. Medium: internal files have grown beyond reviewable ownership units

The public surface need not change, but the implementation should be split:

- `src/bigint.rs` is about 6,500 lines and owns unsigned and signed values,
  formatting, multiplication families, Montgomery, Barrett, scrub policy, and
  large in-module test/benchmark sections.
- `src/number_theory.rs` is about 5,700 lines and combines gcd/HGCD, symbols,
  modular roots, primality, CRT, rational reconstruction, and batch
  smoothness.
- `src/poly.rs` is about 4,000 lines and combines two coefficient rings,
  convolution dispatch, factorization, Hensel lifting, quotient-ring helpers,
  and extensive probes.

Split by invariant and algorithm family while keeping the current re-exported
API. Suggested internal boundaries are `bigint::{core,mul,montgomery,barrett,
convert}`, `number_theory::{euclid,symbols,roots,primality,batch,crt}`, and
`poly::{z,modp,convolution,factor,lift}`. The point is not shorter files by
itself; it is making each unsafe-free invariant and each performance dispatch
reviewable without loading several unrelated algorithms.

## What is strong

- The new dedicated squaring kernels are checked against both dispatched and
  schoolbook multiplication across handoff widths, odd splits, holes, and
  worst-case carry chains.
- The Barrett half-product is compared with a full-product truncation and the
  reduction branch is tested on both sides of its cutoff.
- Even-modulus exponentiation now has an independent division-based oracle;
  the previous self-comparison problem is gone.
- `ProductTree` makes a structural precondition unrepresentable.
- Polynomial dispatch now accounts for operand ratio and density rather than
  applying a balanced dense threshold to every shape.
- The new polynomial operations have direct algebraic property tests and
  exhaustive small Hensel comparisons.
- Public documentation generally states preconditions, failure behavior, and
  the reason an algorithm exists, not just its formula.

## Recommended order

1. Remove all production and test `unsafe`; enforce `forbid(unsafe_code)`.
2. Choose and document the compatibility/version plan for `ProductTree`.
3. Land a versioned Rump release and migrate/delete the duplicate factoring
   polynomial helpers.
4. Move generic GF(2), real-root, and weighted 2-D reduction code upstream.
5. Make sizing portable and add a 32-bit compile gate.
6. Reconcile `REQUESTS.md`, README, rustdoc, and threshold prose.
7. Re-measure the new dispatches on at least one non-M4 host and without the
   global volatile-drop cost.
8. Split the oversized internal modules without changing the public surface.

## Validation performed

- `cargo test --all-targets`: 196 unit tests, 51 integration/manual tests, and
  all binary targets passed; 12 timing probes were ignored by design.
- `cargo test --all-targets --no-run`: passed again on the final live tree.
- `cargo clippy --all-targets -- -D warnings`: passed.
- `cargo fmt --all -- --check`: passed.
- `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps`: passed.
- `git diff --check`: passed.

The ignored timing probes and very slow/external benchmark matrix were not run.
Performance recommendations above are therefore based on code inspection and
the repository's checked-in measurements, not a new cross-machine benchmark.
