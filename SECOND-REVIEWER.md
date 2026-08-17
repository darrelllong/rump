# Second review of `rump`

## API and naming recovery plan — 2026-08-16

The authoritative ledger is [`NAMES.md`](NAMES.md), paired with
[`../factoring/NAMES.md`](../factoring/NAMES.md). Where this review differs,
the ledgers win. In particular, the coordinated breaking cut uses no
deprecated aliases, forwarding wrappers, or duplicate public paths.

### Verdict

Rump's arithmetic is not the mess; its surface is. The crate root re-exports
unrelated integer types, modular contexts, primality tests, polynomial types,
lattice reduction, product trees, smoothness helpers, random sampling, and
GF(2^m) operations as one flat vocabulary. Inside that surface, at least four
naming grammars compete:

- allocating arithmetic uses `add_ref`, `sub_ref`, `mul_ref`, and
  `square_ref`;
- mutation uses both `add_assign_ref` and `assign_add`;
- modular arithmetic mixes `mod_mul`, `mod_pow`, `mod_inverse`, `sqrt_mod`,
  `mul_mont`, and `pow_encoded`;
- scratch reuse is exposed as raw `&mut Vec<u64>` through
  `_with_workspace` names.

The live uncommitted work has begun making some names more coherent—Barrett
operations are moving toward `mod_*`, `Reciprocal` word operations are losing
misleading `_u64` suffixes, and invalid constructors are becoming fallible.
Those directions are useful, but piecemeal public renames at version `0.2.2`
would repeat the `ProductTree` compatibility mistake. Settle the complete
grammar first, then land it as an intentional release train.

Freeze new public exports until that grammar is recorded. A naming commit must
not also change a kernel, cutoff, allocation strategy, or mathematical
contract. This section supersedes the ordering advice later in this report
where the two conflict; the earlier findings remain evidence for the plan.

### Public namespace: organize before moving files

Expose a stable facade first. Physical module splitting can follow without
another public break:

```rust
pub mod integer;        // parse errors and WordReciprocal
pub mod modular;        // modular functions and validated contexts
pub mod number_theory;  // gcd, symbols, primality, CRT, roots
pub mod polynomial;     // PolyZ and PolyMod
pub mod gf2;            // dense and sparse binary linear algebra
pub mod finite_field;   // Gf2m
pub mod lattice;        // LLL and weighted Gauss reduction
pub mod random;         // the byte-source trait and samplers

pub use bigint::{BigInt, BigUint, Sign};
```

Keep only the three core integer types at the root. The breaking commit removes
the old root paths as it adds their canonical module paths; it does not retain
both. Do not make the internal modules public wholesale: facade modules expose
only the supported contract.

The package is named `rust-mp`, the library target is `rump`, and the repository
is called Rump. That distinction is survivable only if it is stated once and
then left alone. Do not attempt a repository/package/crate rename in the same
release as the arithmetic cleanup. Reconsider branding only after the API is
stable; it has no bearing on method correctness.

### One naming grammar

Adopt these rules and apply them across `BigUint`, `BigInt`, contexts,
polynomials, documentation, and examples:

1. A borrowed argument is visible in the Rust signature; do not encode it in
   a `_ref` suffix.
2. The ordinary allocating operation is `add`, `sub`, `mul`, or `square`.
   Prefer the standard operator traits at call sites (`&a + &b`, `x += &y`).
3. A caller-supplied output is named `*_into`; scratch storage is `*_with_scratch`.
   Do not use `assign_add` for one and `add_assign_ref` for the other.
4. Modular operations consistently use the `mod_*` family: `mod_add`,
   `mod_sub`, `mod_mul`, `mod_square`, `mod_pow`, `mod_inverse`, and
   `mod_sqrt`. The live Barrett rename follows this rule; `sqrt_mod` does not.
5. `Result` reports an invalid public input or failed precondition. `Option`
   means a mathematically legitimate absence, such as no inverse or no root.
   If a nonzero value is the entire precondition, accept `NonZeroU64` rather
   than returning an unexplained `None`.
6. `_unchecked` is private. A safe public API must validate its invariant in
   every build or encode it in a type.

The migration table should start with:

| Current | Canonical target |
|---|---|
| `add_ref`, `sub_ref`, `mul_ref`, `square_ref` | `add`, `sub`, `mul`, `square` |
| `add_assign_ref`, `sub_assign_ref` | operator traits, or `add_assign` / `sub_assign` if an inherent method is still needed |
| `assign_add`, `assign_sub` | `add_into`, `sub_into` |
| `sqrt_mod` | `mod_sqrt` |
| `BarrettCtx` | `BarrettContext` |
| `MontgomeryCtx` | `MontgomeryContext` |
| `Rng` | `RandomSource` |
| `Reciprocal` | `WordReciprocal` |
| `SmoothBase` | `SmoothnessBase` |
| `PolyModP` | `PolyMod` unless construction begins proving that the modulus is prime |

`PolyZ` is a conventional mathematical name and can remain. `PolyModP` cannot:
its constructors accept an arbitrary modulus, including composite values, so
the `P` promises a prime field that the type does not establish. There are two
honest designs:

- rename the coefficient-ring type to `PolyMod`, and have field-only
  algorithms validate their preconditions; or
- introduce a validated `PrimeModulus`/`PrimeFieldContext` and allow a
  genuinely field-bound polynomial type to use a prime-field name.

The first is the smaller cleanup. Do not preserve a mathematically false name
for symmetry.

### Make modular domains types, not suffixes

The biggest API improvement is not a spelling change. A Montgomery-domain
value is currently an ordinary `BigUint`, so the compiler cannot prevent an
unencoded value, an out-of-range value, or a residue from another modulus from
reaching `mul_mont`. That forces names such as `mul_mont`, `one_mont`, and
`pow_encoded` to carry invariants the type system should carry.

The target should be an opaque context-bound `MontgomeryResidue`. Encoding
returns that type; decoding consumes or borrows it; residue operations are
simply `add`, `sub`, `mul`, `square`, and `pow`. A context mismatch must be a
checked error, never a debug-only assertion. The context may retain private
unchecked kernels after measurement, but callers must not see them.

Likewise, replace `&mut Vec<u64>` scratch parameters with an opaque
`MontgomeryScratch` or output-reusing residue operation. The public API should
state whether it reuses scratch and/or result storage. “Allocation-free” must
mean no allocation, not merely no scratch allocation.

Barrett values remain ordinary residues, so `BarrettContext::mod_mul` and
friends are appropriate. A context should expose only operations it
accelerates; deleting the live `add_mod`/`sub_mod` forwarding methods is the
right division because they added no Barrett behavior.

### Turn construction and batch helpers into object APIs

Several free-function/type pairs should become one invariant-bearing API:

```text
product_tree(values)                  -> ProductTree::new(values)
remainder_tree(&tree, modulus)        -> tree.remainders(modulus)
smooth_parts(values, primes)          -> SmoothnessBase::new(primes)?.smooth_parts(values)
```

Keep a convenience free function only when it adds a genuinely useful
one-shot path. Do not keep two equally prominent ways to perform the same
operation.

The live review corrected an important false premise in the first draft:
`is_probable_prime` is the fixed twelve-base Miller–Rabin schedule, while
`is_probable_prime_bpsw` is Baillie–PSW. They are distinct mathematical
contracts, not duplicate spellings, and both remain canonical under
`number_theory`. The explicit-base entry is
`miller_rabin_with_bases`, so the caller can see which guarantee changed.

### Division of labor with `factoring`

Rump owns exact, factoring-free mathematics. It does not own factoring search
policy merely because that policy contains arithmetic.

| Move to or keep in Rump | Keep in `factoring` |
|---|---|
| Big integers and modular contexts | Algorithm cascade and budgets |
| Polynomial-ring operations and validated field algorithms | Polynomial-selection score and lift-width policy |
| Product/remainder trees and batch smooth parts | Factor bases, ideals, relations, and smoothness acceptance |
| Weighted Gauss reduction | Q-lattice construction and special-q scheduling |
| Dense GF(2) null space, singleton pruning, Block Lanczos | Matrix layout and conversion of dependencies into congruences |
| Generic real-polynomial root finding | GNFS norm-model acceptance and sieve-region policy |

The polynomial primitives already added here should be consumed before more
variants are invented: balanced base expansion, prime-power roots, monic
remainder/product, homogeneous substitution, symmetric lift, and modulus
change. Stabilize their names and contracts in Rump, then have factoring switch
and delete each local copy in the same commit.

Bring the GF(2) solvers across under names that describe the operation, for
example `gf2::dense_null_space`, `gf2::prune_singletons`, and
`gf2::block_lanczos_dependencies`. They must not retain QS terminology; GNFS
already uses them. Move only the generic real-root solver, not factoring's
`NormModel` or its heuristic acceptance rules.

No helper moves merely because it might someday be reusable. Require a
factoring-free contract, an independent test oracle, and a credible second
consumer. Once a helper does move, however, leaving the consumer copy in place
is a defect.

### Release train

1. **Maintain the paired ledgers.** Every public name and ownership change
   lands there first in a documentation-only commit.
2. **Finish the current correctness work without opportunistic renames.** Do
   not mix reciprocal, Barrett, smoothness, or lattice behavior changes with
   the API sweep.
3. **Create the facade namespaces in the breaking commit.** Add each canonical
   path and remove its old root path in the same change. Add no forwarding
   alias or duplicate re-export.
4. **Release this as Rump `0.3.0`.** The existing `ProductTree` signature
   change and the planned naming changes are breaking. Version them honestly.
5. **Migrate factoring in four small batches:** polynomial primitives,
   weighted Gauss, GF(2), and real roots, against the recorded paired Rump
   revision. Each batch deletes the downstream copy and runs both gates.
6. **Add external compile fixtures.** Test only the canonical examples as
   downstream crates, not merely as unit tests inside Rump.
7. **Audit removed spellings.** Searches may find them only in release notes
   and history, never in live exports or forwarding functions.
8. **Split the large implementation files after the facade is stable.** Split
   by invariant (`integer`, multiplication, division, modular contexts;
   Euclid, symbols, roots, primality, batch; integer and modular polynomials),
   not by an arbitrary line target.

Each commit should pass formatting, Clippy and rustdoc with warnings denied,
ordinary tests, and `git diff --check`. Behavior-preserving moves also compare
the existing benchmark baselines. A cleanup commit does not get to claim a
speedup, and an optimization commit does not get to hide behind a rename.

The non-negotiable restriction remains: use `#![forbid(unsafe_code)]` and pure
safe Rust throughout. No assembly, FFI, intrinsics, raw-pointer tricks,
target-only representation, or unportable public API belongs in this plan.
Portability failures should be represented by checked sizes and clear errors,
not by target assumptions.

The coordinated consumer-side sequence and concrete factoring names are in
[`../factoring/SECOND-REVIEWER.md`](../factoring/SECOND-REVIEWER.md).

---

The original hard review follows unchanged.

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
