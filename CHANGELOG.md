# Changelog

Releases are git tags; nothing here is published to crates.io. Entries record
what a consumer must change, not everything that moved.

## Unreleased

### Added

- **`gfp`: linear algebra over a large prime field.** `gfp::SparseMatrix`
  holds a square matrix whose entries are small integers — the `±1` ones in
  their own lists, where a row costs an addition or a subtraction with no
  multiplication, and the tail of larger coefficients beside them — and
  `gfp::kernel_vector` solves `Mx = 0` by Wiedemann's algorithm over
  `gfp::Field`. This is the system an index-calculus discrete logarithm ends
  in, where `crate::gf2`'s packing does not apply: a vector entry is hundreds
  of bits, not one.

  The form here is scalar: one Krylov sequence, one recurrence from
  `gfp::minimal_polynomial`. It is quadratic in the matrix dimension and
  single-threaded, which at a hundred digits is hours and past about a
  hundred and ten stops being reasonable; a blocked form, which distributes,
  is the answer there and is not built.

  `kernel_vector` returns `gfp::Kernel`, which separates the two failures a
  caller must answer differently: `Inconclusive` is this draw, and the answer
  is another draw; `NoKernel` is the matrix, and the answer is different
  rows.

### Breaking

- **`number_theory::ln_gamma`, `regularized_incomplete_beta`,
  `student_t_quantile` and `NumericalError` are gone.** They are floating
  statistical kernels, not arithmetic, and they now live in `entropy::math`
  (entropy 0.6.0), unchanged in signature and behaviour and on the same
  reference fixtures. A caller depends on entropy and changes the path;
  entropy compiles that module with or without its default features, so it
  does not drag in a generator. rump keeps no forwarding wrapper: entropy
  depends on rump, so a wrapper would close a cycle. The reference generators
  `scripts/lanczos_coefficients.py` and `scripts/incomplete_beta_reference.py`
  and the fixtures under `tests/data/` moved with them.

- **`lattice::short_vectors_form` and `closest_vectors_form` return
  `lattice::Enumeration` and take `visit_limit: u64`.** Both panicked on
  valid forms whose scales differ widely — under `diag(1, 2¹⁰⁰⁰)` the
  common shift into doubles rounded the entry 1 to zero — and returned a
  truncated search indistinguishable from a complete one. The Gram–Schmidt
  data are now exact, pruning uses outward-rounded enclosures, and
  `Enumeration::outcome()` says whether the search was `Exhausted`, stopped
  at its `VisitLimit`, or met a `NumericalLimit`; only `Exhausted` certifies
  that nothing within the bound was missed. A negative bound is an empty,
  exhausted search for both functions. Callers add the visit limit (the old
  implicit cap was 50 000 000) and read `.into_vectors()`.

### Fixed

- **`number_theory::gcd`, `gcd_extended`, `mod_inverse` and `jacobi` above
  the Half-GCD crossover could answer wrongly.** The base case of the
  recursion applied Lehmer batches without checking where they landed, on
  the assumption that a batch reading a 124-bit window removes about that
  many bits. It bounds each quotient, not each remainder: a pair whose
  leading digits certify `A = 2·B + 2` drops the smaller element to two in
  one step, past the boundary the recursion's splice relies on. On such a
  pair at 131,072 bits `gcd` returned 2 where the true gcd had a thousand
  bits (the debug build stopped at an assertion instead). A batch is now
  committed only if its result keeps both elements above the boundary, the
  rule the guarded single divisions already obey; the pair is a test.

- **`poly::PolyZ::real_roots` decides every sign exactly, and its brackets
  hold every root.** Signs came from `f64` Horner evaluation and the search
  ran a fixed 200 bisections. Both fail at wide coefficient ranges: for
  `x² + 10³⁰⁰x − 10³⁰⁰` the value at the Cauchy bound is `+1` but evaluates
  to `−10³⁰⁰`, hiding the sign change that brackets the root near `−10³⁰⁰`,
  and 200 halvings of a `10³⁰⁰`-wide bracket leave it `10²³⁹` wide, so the
  root near `1` was returned as `1.56·10²³⁹`. That polynomial now returns
  `[-1e300, 1.0]`. Signs are the exact signs of the integer polynomial at
  dyadic points, the bound is rounded outward (adding to a rounded ratio does
  not move it: the float gap at `10²⁰` is 16384), and bisection runs until the
  bracket is one float wide. Callers that read the returned values keep
  reading them; the values are now correct.

- **`number_theory::dickman_rho` and `number_theory::semismooth_probability`.**
  Both existed only to rank a sieve's expected relation yield, and their one
  consumer was the factoring crate, which now owns them. Callers that want
  Dickman's function or the Bach–Peralta semismoothness estimate take them
  from factoring or carry their own.

### Changed

- **Large `BigUint` multiplication now dispatches to an exact NTT.** Four
  base-2^16 digits per limb are convolved under two 31-bit NTT primes and
  reconstructed by CRT, so recovery is deterministic and has no floating-point
  rounding. Radix-2 stages use scoped disjoint-slice workers bounded by
  `available_parallelism`; exact-worker curves through the full 2^26 ceiling
  select size targets from 4 through 64, not a fixed machine cap, and detection
  failure is serial. Hardware-aware M4
  crossovers are 65,536 limbs serially, 32,768 with two useful contexts, and
  8,192 with four or more, plus a measured padding gate because transform work
  doubles discontinuously. Input expansion writes directly into bit-reversed
  positions in parallel disjoint segments; independent forward transforms run
  concurrently within the same budget, and large inverses use DIF with the
  dead operand buffer as natural-order output. Linear passes parallelize above
  a measured grain. `BigUint::square` has a one-buffer NTT square rather than
  duplicating a general product's transform. Prime/root proofs, DIT/DIF
  round trips and worker-count independence, CRT boundaries, schoolbook
  differential products/squares, maximal carry chains, and public threshold
  dispatch are tested.

- **The Euclidean family reuses Lehmer/HGCD work buffers.** GCD, extended GCD,
  modular inverse, Jacobi, rational reconstruction, and the HGCD base loop keep
  their positive/negative transform buckets and recycle operand/cofactor limb
  vectors into subsequent outputs. Difference bit lengths no longer allocate,
  guarded HGCD division retains its threshold and adjusted dividend, and matrix
  row steps mutate in place. The classical-Euclid differential suite is
  unchanged; the deterministic M4 crossover probe improves Lehmer by 18–35%
  through 2,048 limbs and HGCD by 10–34% through 4,096 limbs.

- **Sparse GF(2) solves retain their bounded fold workers and use byte-sliced
  64-by-64 products.** `gf2::block_lanczos_dependencies` no longer creates a
  scoped thread set for both halves of every `MᵀM` application. Useful workers
  live for one solver call, remain bounded by the caller's `threads` value,
  and gather output ranges deterministically. Dense block words now multiply
  through eight byte lookups rather than an average of 32 dependent set-bit
  steps, and recurrence equation (18) is fused without three full-vector
  temporaries. The public result and random-source consumption are unchanged;
  scalar-equation oracles and a fixture that crosses the parallel threshold
  check bit-identical results at one and eight workers.

- **`BigUint::to_be_bytes` and `to_be_bytes_padded` allocate once and leave
  no second copy.** `to_be_bytes_padded` encoded unpadded and copied that
  vector into the padded one, dropping it unwiped; `to_be_bytes` encoded
  eight bytes per limb and drained the leading zeros, leaving up to seven
  low-order bytes of the value in the result's spare capacity. Both now
  write the limbs straight into one buffer of the final length, whose
  capacity equals its length. The output is unchanged; a consumer that wipes
  the returned bytes now reaches every heap copy these functions made.

### Added

- **`modular::MontgomeryContext::gcd_with_modulus`.** `gcd(v, n)` for the
  value a residue encodes, taken from the encoded limbs: for odd `n` the
  Montgomery radix is a unit, so encoding does not change the gcd. A caller
  that multiplies in the domain and tests for a factor, as Pollard's rho
  does, no longer decodes first. A foreign residue is `ContextMismatch`.

- **`modular::mod_inverse_u128` and `number_theory::crt_combine_u64`.** The
  double-word inverse is total over every `u128` modulus, including those past
  `2¹²⁷` where signed cofactors run out of bits; the word CRT combines two
  congruences into the residue below their product as a `u128`, with
  `crt_combine`'s `None` for zero or non-coprime moduli. Both replace
  factoring-local copies that were narrower than their types.

- **Little-endian bytes: `BigUint::from_le_bytes`, `to_le_bytes`, and
  `to_le_bytes_padded`.** The exact mirror of the big-endian trio: the empty
  slice decodes to zero, zero encodes as one `0x00`, the minimal encoding
  drops high zero bytes, and the padded form zero-fills the high end and
  panics when the value does not fit. Each writes its output directly, with
  no reversed intermediate. A consumer that reversed a big-endian encoding,
  and wiped the reversed copy, calls these instead.

- **`BigUint::mod_neg`.** One-shot modular negation on `mod_add`'s
  contract: any operand, non-zero modulus (panic otherwise), result in
  `[0, modulus)`. It replaces `BigUint::mod_sub(&BigUint::zero(), x, m)`.

- **`number_theory::is_lucas_probable_prime`.** The general Lucas
  probable-prime test of FIPS 186-4, Appendix C.3.3, step for step: a
  perfect square is composite, `D` is the first of 5, −7, 9, −11, … with
  Jacobi symbol −1 and a zero symbol means composite, and the candidate is
  accepted exactly when `U_0 = 0` at the end of the left-to-right ladder over
  the bits of `n + 1`. It is not an alias of
  `is_strong_lucas_probable_prime`, whose acceptance condition is stronger:
  323 passes this test and fails that one. One reading is documented rather
  than silent. A zero symbol from a `D` the candidate itself divides proves
  nothing, and only a prime reaches one, so it is passed over instead of
  reporting the primes 5 and 11 composite; for every candidate larger than
  the `|D|` its search reaches, the result is the standard's exactly. `0`,
  `1`, and even values, outside C.3.3's odd domain, get the true answer.
  Every integer below 10⁵ is checked against an independent sieve and
  OEIS A217120.

- **`number_theory::is_prime_aks`.** An exact, unconditional deterministic
  implementation of the Agrawal–Kayal–Saxena primality test. The polynomial
  stage uses a dedicated cyclic `X^r − 1` reduction in Rump's `PolyMod`
  engine, differentially checked against general monic polynomial reduction
  over both prime and composite coefficient moduli, and against a separate
  scalar repeated-multiplication oracle. An ignored parallel stress test checks
  arbitrary exhaustive ranges against an independent sieve while never using
  more workers than the machine reports. This is the proof-oriented AKS
  algorithm, not a replacement for the much faster probable-prime APIs.

- **`number_theory::crt_combine_balanced`.** It has the same exact validation
  and canonical result as `crt_combine`, but combines equal-width partial
  products through a balanced tree and can run independent pairs on a bounded
  number of scoped workers, never exceeding the machine's reported parallelism.
  This is the reusable machine needed by NFS
  coefficient reconstruction; Rump owns the CRT, while the consumer owns what
  its residues mean.

- **`wipe` cargo feature — the option to wipe is back.** 0.3.0 removed the
  drop-time scrub to reach `forbid(unsafe_code)`; that traded away a guarantee
  the parent cryptography crate depended on. The scrub returns as an opt-in
  feature: `BigUint` volatile-wipes its live limbs on drop, `clone_from` /
  `add_into` / `sub_into` and the shift/cancellation paths wipe the limbs
  their in-place shrinks strand in spare capacity, the exponentiation ladder
  wipes its table and temporaries on exit, the Montgomery workspaces and
  `MontgomeryScratch` wipe theirs, and the samplers wipe drawn byte buffers.
  The raw read-back test that proves the shrink paths scrub returns with it.
  Under the feature the crate attribute relaxes to `deny(unsafe_code)` with
  exactly those two audited `unsafe` sites; the default build is unchanged —
  `forbid(unsafe_code)`, nothing wiped. The old caveats are restated rather
  than improved: spare capacity and buffers freed by reallocation are not
  wiped, and nothing becomes constant-time.

## 0.3.0 — unreleased

### Breaking

- **`BarrettContext::add_mod` and `sub_mod` are gone; the rest of the family is
  renamed to the crate's `mod_*` order.** `add_mod`/`sub_mod` were one-line
  forwarders to `BigUint::mod_add`/`mod_sub` — the same operation under the
  same two words in the opposite order, and `μ` plays no part in modular
  addition. Call `BigUint::mod_add(a, b, ctx.modulus())`. The operations that
  do use the context are now `mod_mul`, `mod_square`, and `mod_pow`, matching
  `BigUint::mod_mul` and the free `mod_pow` rather than inverting them.
- **No public constructor added or touched in 0.3.0 panics on bad input.**
  Invalid input is a typed error; `Option` is reserved for legitimate
  mathematical absence.

  - `BarrettContext::new` and `MontgomeryContext::new` return
    `Result<Self, modular::ModulusError>`, whose `Zero`, `One` and `Even`
    variants describe the rejected value rather than the context that
    refused it. Barrett returns `Zero` or `One`; Montgomery returns `Zero`
    or `Even`, and still accepts a modulus of one, which is odd.
  - `SmoothnessBase::new` returns
    `Result<Self, number_theory::SmoothnessBaseError>`, reporting the first
    entry below two through `index()` and `value()`. A composite entry is
    still accepted deliberately, so neither the type nor its message calls
    the rejected value non-prime.
  - `gauss_reduce_weighted` returns
    `Result<_, lattice::ReductionError>` with `DependentBasis` and
    `OutOfRange`, and takes `weights: [NonZeroU64; 2]` — positivity moved
    into the type, so there is no weight variant to report.
  - `WordReciprocal::new` takes `NonZeroU64` and is total: non-zero is the
    entire precondition, so nothing is left for a return type to describe.

  All three error types are `#[non_exhaustive]`, `Copy`, and implement
  `Display` and `std::error::Error`. The pre-existing panicking surface is
  unchanged in this release.
- **`product_tree` and `remainder_tree` take and return a typed
  `ProductTree`.** Previously:

  ```text
  product_tree(&[BigUint]) -> Vec<Vec<BigUint>>
  remainder_tree(&[Vec<BigUint>], &BigUint) -> Vec<BigUint>
  ```

  Now:

  ```text
  product_tree(&[BigUint]) -> ProductTree
  remainder_tree(&ProductTree, &BigUint) -> Vec<BigUint>
  ```

  The change makes a structural precondition unrepresentable: a caller can no
  longer hand `remainder_tree` a `Vec<Vec<BigUint>>` of the wrong shape, and
  the function can rely on the layout `product_tree` established rather than
  re-deriving or trusting it. `ProductTree` is exported as `number_theory::ProductTree`.

  A caller that only pipes one into the other is unaffected apart from the
  type name. A caller that *inspected* the levels must go through
  `ProductTree`'s accessors.

  This is a source-breaking change to a public signature. It landed on `main`
  above the `v0.2.2` tag while the crate version still read `0.2.2`, which a
  second reviewer correctly flagged: the tagged `v0.2.2` and the `main` that
  followed it exported incompatible signatures under one version number. The
  version is bumped here so the break carries a number, rather than being
  corrected retroactively in the tagged release.

### Added

- **`WordReciprocal`** — division by a `u64` divisor that does not change, with
  the reciprocal precomputed once (Möller & Granlund, IEEE ToC 60 (2011),
  Algorithm 4). `rem`, `div_rem`, `rem_euclid_i64`, and
  `BigUint::rem_reciprocal` / `div_rem_reciprocal` for multi-limb dividends.
  Worth reaching for at two limbs and above; measured *slower* than the
  hardware divide for word-sized dividends on Apple silicon, and the module
  documentation carries the numbers.
- **`SmoothnessBase`** — Bernstein batch smoothness with the primes' product built
  once, so the caller chooses the batch size rather than the setup cost
  choosing it. The free `smooth_parts` is now the one-shot form over this
  type, so there is one algorithm rather than two.
- **`gauss_reduce_weighted`** — exact two-dimensional Lagrange–Gauss reduction
  under a diagonal form `(w₀x)² + (w₁y)²`, in `i128`. For a skewed metric
  `(x/√s)² + (y√s)²` with rational `s = p/q`, pass `weights = [q, p]`.
- **Portable: no target-width restriction.** Earlier 0.3.0 work added a
  `compile_error!` refusing non-64-bit targets, on the ground that `bits()`
  scales a limb count by 64 and the `R²` and Karatsuba paths by 128, products
  that overflow a 32-bit `usize` for operands in the hundreds of megabytes.
  The gate is gone and the arithmetic is checked instead: every limb-count to
  bit-index conversion goes through one checked multiplication that refuses an
  unrepresentable index rather than wrapping. 32-bit builds are supported and
  the release gate cross-checks `i686-unknown-linux-gnu`.
- **`#![forbid(unsafe_code)]`, no exceptions.** The volatile scrub helper, the
  `Drop` policy that called it, and the raw read-back test that verified it are
  removed. Nothing is wiped now: values live in ordinary heap buffers and
  freed memory keeps its contents. A consumer handling key material brings its
  own representation.

### Documentation

- `README.md` no longer opens by calling the crate "pure, safe Rust" while
  listing two audited `unsafe` exceptions sixty lines later. The headline now
  states the exceptions, and the Properties entry says why the crate uses
  `deny(unsafe_code)` rather than `forbid` — `forbid` cannot be lifted by an
  inner `allow`, which is what the scrub helper and its test probe require.
- `REQUESTS.md` is a state machine rather than a diary. Every entry sits in
  exactly one of four states — outstanding; landed in rump with consumer
  migration pending; fully migrated; deliberately downstream — and the file no
  longer carries an outstanding list and the sentence "every entry this file
  has ever carried is closed" at the same time.

## 0.2.2 — tagged `v0.2.2` (`cf2d1dc`)

Answered the external review against v0.2.1: the `2¹⁰`-coupled trial-screen
identity, per-call allocation in the public `mul_mont`/`square_mont`, and the
incomplete `CITATIONS.md`. Added the public `BigInt` signed ring (`mul_ref`,
truncated `div_rem`, `abs`), `BigInt::symmetric_remainder`, and four
word-and-size primitives (`BigUint::digit_count`, `BigInt::from_i128`,
`gcd_u64`, `mod_inverse_u64`).
