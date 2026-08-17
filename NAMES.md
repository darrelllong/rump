# NAMES

The naming authority for Rump. If this file disagrees with a doc comment,
manual, changelog, commit message, request, or review, this file wins.

Companion: [factoring NAMES](../factoring/NAMES.md) is the consumer half. A
cross-repository name is not agreed until it has the same canonical spelling
and owner in both ledgers.

## Enforcement

No public name is added or changed unless its ledger row is added or changed
first, in a documentation-only commit. Review asks one question before looking
at an implementation: “Where is this name in NAMES.md?”

There are no migration shims on main: no deprecated aliases, forwarding
wrappers, duplicate re-exports, temporary facade paths, v2 names, or
“new”/“old” pairs. Rump 0.3.0 and factoring 0.2.0 are the coordinated breaking
cut. The canonical name replaces the old name, and factoring changes in the
paired integration batch.

The two commits cannot be atomic across two Git repositories. Their recorded
sibling revisions are the atomic unit: old/old and new/new are supported;
old/new and new/old are not. Do not pollute either API to make a mixed checkout
compile. If an actual tagged external consumer ever needs maintenance, support
the old release on its release branch.

A distinct convenience is not a shim only when it has a distinct cost or
ownership contract documented in its ledger row. The one-shot smooth_parts and
reusable SmoothnessBase, for example, may both exist because one owns setup and
the other reuses it. Two spellings for the same operation may not.

## State legend

- **current** — the name in the tree today.
- **canonical** — the sole name it is going to.
- **done** — canonical name is in the tree; no alias remains.
- **pending** — canonical name agreed, not yet applied.
- **blocked** — waiting on the paired factoring change or a prerequisite type.
- **removed** — deliberately absent from the target API.

## Shared rules

These rules are identical in both repositories.

1. One concept has one owner, one public path, and one canonical name.
2. No compatibility shims on main. Breaking changes use the declared breaking
   releases and paired sibling revisions.
3. Full algorithm names at public boundaries. Domain-standard abbreviations
   such as gcd, LLL, GF(2), QS, and GNFS are allowed where they are the normal
   mathematical name; local contractions such as Ctx are not.
4. A suffix must name what differs. Bare “with” is allowed only when the
   function accepts a named options type.
5. Borrowing is visible in the Rust signature and is not repeated as “ref”.
6. Allocating operations use the mathematical verb (add, mul, square);
   mutation uses operator traits; caller-owned output uses an “into” suffix;
   reusable scratch uses a “with scratch” suffix.
7. Residue-ring operations use the mod prefix: mod_add, mod_sub, mod_mul,
   mod_square, mod_pow, mod_inverse, and mod_sqrt. A phrase such as
   roots_mod_prime_power may keep mathematical “roots modulo” order because it
   names the solution set rather than a ring operation.
8. A type encodes a simple precondition when it can. Otherwise invalid input
   returns Result; Option is reserved for legitimate mathematical absence.
9. Public names state the domain noun. Short local names are allowed only while
   they remain inside one module.
10. A rename commit is mechanical. It does not change a kernel, cutoff,
    representation, allocation strategy, or mathematical contract.

## Repository identity

| Current | Canonical | State | Notes |
|---|---|---|---|
| package rust-mp | rust-mp | done | crates.io package name |
| library crate rump | rump | done | Rust import name |
| repository Rump | Rump | done | no branding rename during the API cut |

## Public topology

The flat root is replaced at 0.3.0; it is not retained beside the new modules.
BigInt, BigUint, and Sign remain at the root because they are the crate's
primary values. Every other export has one module path:

| Canonical module | Canonical contents | State |
|---|---|---|
| crate root | BigInt, BigUint, Sign | done |
| integer | ParseBigIntError, WordReciprocal | done |
| modular | BarrettContext, MontgomeryContext, ModulusError, modular arithmetic | done |
| number_theory | gcd/lcm, symbols, primality, CRT, reconstruction, valuations, product trees, smoothness, SmoothnessBaseError | done |
| polynomial | PolyZ, PolyMod, polynomial limits and errors | done |
| finite_field | Gf2m | done |
| gf2 | dense null space, singleton pruning, Block Lanczos | blocked — consumer transfer |
| lattice | LLL, weighted Gauss reduction, ReductionError | done |
| random | RandomSource and random-value functions | done |

Private source files do not have to mirror this facade one-for-one. The public
path is the contract; physical splitting follows after names settle.

## Integer arithmetic

| Current | Canonical | State | Notes |
|---|---|---|---|
| BigUint::add_ref / BigInt::add_ref | add | done | borrowing remains visible in the signature |
| BigUint::sub_ref / BigInt::sub_ref | sub | done | |
| BigUint::mul_ref / BigInt::mul_ref | mul | done | |
| BigUint::square_ref | square | done | |
| BigInt::mul_biguint_ref | mul_biguint | done | |
| add_assign_ref / sub_assign_ref | removed; use += &rhs / -= &rhs | done | operator traits own receiver mutation |
| assign_add / assign_sub | add_into / sub_into | done | caller-owned reusable output |
| BigUint::modulo | rem | done | aligns with div_rem and rem_u64 |
| BigInt::modulo_positive | rem_euclid | done | non-negative residue |
| BigInt::symmetric_remainder | symmetric_rem | done | |
| Reciprocal | integer::WordReciprocal | done | fixed divisor is exactly one word |
| Reciprocal::new(NonZeroU64) | WordReciprocal::new(NonZeroU64) | done | total constructor |
| Reciprocal::rem / div_rem | WordReciprocal::rem / div_rem | done | leaf names are canonical |

Constructors, conversions, predicates, roots, shifts, and div_rem methods not
listed above retain their current leaf names under the canonical path.

## Modular arithmetic

| Current | Canonical | State | Notes |
|---|---|---|---|
| BarrettCtx | modular::BarrettContext | done | no local public contractions |
| MontgomeryCtx | modular::MontgomeryContext | done | |
| BarrettCtx mod_mul/mod_square/mod_pow | BarrettContext mod_mul/mod_square/mod_pow | done | mod order is settled |
| BarrettCtx add_mod/sub_mod | removed | done | redundant forwarders already deleted |
| free mod_pow / mod_inverse family | same leaf names under modular | done |
| sqrt_mod | modular::mod_sqrt | done | |
| sqrt_mod_prime_power | modular::mod_sqrt_prime_power | done | |
| BigUint mod_add/mod_sub/mod_mul | same leaf names | done | inherent operations stay on the value |
| BarrettContext::new returning Option | return Result<Self, ModulusError> | done | zero and one are rejected |
| MontgomeryContext::new returning Option | return Result<Self, ModulusError> | done | zero and even moduli are rejected |
| no public modular construction error | modular::ModulusError { Zero, One, Even } | done | shared factual variants; no context-dependent “below two” variant |
| raw BigUint Montgomery-domain values | opaque MontgomeryResidue | pending | context and reduction invariant belong in the type |
| mul_mont/square_mont/add_mont/sub_mont/one_mont/pow_encoded | residue mul/square/add/sub/one/pow | blocked — MontgomeryResidue |
| with_workspace methods taking Vec<u64> | opaque scratch or into operations | blocked — residue/output design |

ModulusError is a non-exhaustive Copy enum implementing Display and
std::error::Error. BarrettContext returns Zero or One; MontgomeryContext
returns Zero or Even. The variants describe the rejected value rather than
making a context-dependent assertion such as “below two.” Invalid modulus is
bad input, not a mathematical absence. No unchecked Montgomery operation is
public.

## Number theory and batching

| Current | Canonical | State | Notes |
|---|---|---|---|
| is_probable_prime | number_theory::is_probable_prime | done | fixed twelve-base Miller–Rabin default; not BPSW |
| is_probable_prime_bpsw | number_theory::is_probable_prime_bpsw | done | distinct Baillie–PSW contract, not an alias |
| is_probable_prime_with_bases | number_theory::miller_rabin_with_bases | done | algorithm is visible |
| miller_rabin_witness | same under number_theory | done | |
| is_strong_lucas_probable_prime | same under number_theory | done | |
| ProductTree | number_theory::ProductTree | done | typed invariant is canonical |
| product_tree / remainder_tree | same under number_theory | done | algorithmic one-shot pair |
| SmoothBase | number_theory::SmoothnessBase | done | reusable smoothness context |
| free smooth_parts | number_theory::smooth_parts | done | one-shot convenience through context |
| SmoothBase::new returning Option | SmoothnessBase::new returning Result<Self, SmoothnessBaseError> | done | entries below two are invalid input |
| no public smoothness construction error | number_theory::SmoothnessBaseError { index, value } | done | fields private; read through index() and value() |
| no error accessors | SmoothnessBaseError::index / value | done | exact rejected entry without caller rescanning |

All other current number-theory exports retain their leaf names and move only
to number_theory, except the modular functions assigned above.

SmoothnessBaseError is a non-exhaustive Copy struct implementing Display and
std::error::Error. It reports the first entry below two. Composites remain
valid, so neither the type nor its message calls the offending value
“non-prime.”

## Polynomials and fields

| Current | Canonical | State | Notes |
|---|---|---|---|
| PolyZ | polynomial::PolyZ | done | conventional mathematical name |
| PolyModP | polynomial::PolyMod | done | constructors do not prove prime modulus |
| PolyModP::pow_mod | PolyMod::mod_pow | done | residue-ring naming rule |
| PolyModP::with_modulus | PolyMod::change_modulus | done | “with” hid a nontrivial operation |
| MAX_ROOT_LEVEL | polynomial::MAX_ENUMERATED_ROOTS | done | names resource limit, not algorithm level |
| Gf2m | finite_field::Gf2m | done | conventional field notation |
| Rng | random::RandomSource | done | trait supplies bytes; it chooses no entropy source |
| factoring real-root solver | PolyZ::real_roots | blocked — consumer transfer | only generic root finding moves; NormModel stays downstream |

PolyZ balanced_base_expansion, rem_monic, product_mod_monic,
homogeneous_substitution, and roots_mod_prime_power retain their leaf names.
PolyMod::symmetric_lift also stays. Field-only operations on PolyMod must accept
or carry a validated prime modulus; the ring type must not make a false
prime-field promise.

## Lattice and GF(2)

| Current | Canonical | State | Notes |
|---|---|---|---|
| lll_reduce / lll_reduce_delta | same under lattice | done | LLL is standard |
| gauss_reduce_weighted | lattice::gauss_reduce_weighted | done | leaf order is canonical |
| weights: [i128; 2] | weights: [NonZeroU64; 2] | done | encodes positivity and removes no previously successful input |
| gauss_reduce_weighted returning Option | return Result<_, ReductionError> | done | invalid basis/range is bad input |
| no public lattice reduction error | lattice::ReductionError { DependentBasis, OutOfRange } | done | no weight variant after NonZeroU64 |
| factoring dense null space | gf2::dense_null_space | blocked — consumer transfer |
| factoring singleton peel | gf2::prune_singletons | blocked — consumer transfer |
| factoring Block Lanczos | gf2::block_lanczos_dependencies | blocked — consumer transfer |

ReductionError is a non-exhaustive Copy enum implementing Display and
std::error::Error. NonZeroU64 is sufficient for every successful call under
the documented i128 bound: a full-rank basis has a nonzero coordinate in each
dimension, and a weight above u64 already makes that weighted coordinate's
square unrepresentable. The signature therefore removes only inputs that
already returned OutOfRange, while making non-positive weights unrepresentable.

## Ownership

Rump owns exact integer, modular, polynomial, finite-field, GF(2), and lattice
primitives; generic algorithms whose inputs and outputs contain no factoring
concepts; validated arithmetic contexts; and reusable numerical root finding.

Factoring owns algorithm selection, factor bases, ideals, relations,
smoothness policy, sieve bars, retry and scheduling policy, polynomial scores,
special-q policy, matrix meaning, and factor extraction.

A helper moves only if it has a factoring-free contract, an independent test
oracle, and a second credible use. Once it moves, a downstream copy is a bug.
Transfer state is maintained in both this file and the ownership rows in the
[factoring ledger](../factoring/NAMES.md).

| Canonical Rump API | Factoring site | Transfer state |
|---|---|---|
| polynomial::PolyZ::balanced_base_expansion | gnfs/polynomial_selection.rs | Rump canonical; consumer transfer |
| polynomial::PolyZ::roots_mod_prime_power | gnfs/polynomial_selection.rs | Rump canonical; consumer transfer |
| polynomial::PolyZ monic remainder/product | gnfs/algebraic_square_root.rs | Rump canonical; consumer transfer |
| polynomial::PolyZ::homogeneous_substitution | gnfs/lattice.rs | Rump canonical; consumer transfer |
| polynomial::PolyMod symmetric_lift/change_modulus | gnfs/algebraic_square_root.rs | Rump canonical; consumer transfer |
| lattice::gauss_reduce_weighted | gnfs/lattice.rs | Rump canonical; consumer transfer |
| gf2 dense/sparse solvers | qs/linalg.rs and qs/lanczos.rs | not yet landed in Rump |
| polynomial::PolyZ::real_roots | gnfs/norm_model.rs | not yet landed in Rump |
| integer::WordReciprocal | six division sites | Rump canonical; consumer transfer |
| number_theory::SmoothnessBase | relation confirmation | Rump canonical; consumer transfer |

## Frozen during the cut

No new public export exists unless it is already canonical here. No new
algorithm enters either repository. Correctness repairs and ledgered migration
batches only.

The cut is complete when the canonical surface is the only surface, factoring
uses the paired Rump revision, both repositories' gates pass, and searches for
every removed spelling find only changelog/history references.

Pure portable safe Rust and forbid(unsafe_code) are non-negotiable in both
crates. No assembly, FFI, intrinsics, raw-pointer tricks, target-only layouts,
or target-specific public API may be used to implement this ledger.
