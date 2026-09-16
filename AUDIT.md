# Rump audit

> **Motto:** better that, better algorithms
>
> **Creed:** Experiment is asking God for peer review.

## Reviewed state

Reviewed 2026-09-16 on `aarch64-apple-darwin`, rustc/Cargo 1.93.1.
HEAD: `70d12839933c32569fdee5cb2bc1d5b4f3964f1a`. The reviewed tree includes concurrent comment/documentation edits.
Tests ran against frozen sibling copies, with fresh build directories.
This is a targeted mathematical and cross-repository review, supported by the
runs below; it is not an assertion that every line or parameter regime was examined.
Only AUDIT.md and SUGGESTIONS.md are changed by this review.
Concurrent edits continued after capture; the results below apply to this
recorded snapshot, not automatically to later working-tree contents.

| Sibling | Reviewed HEAD |
|---|---|
| cryptography | `601c97bbd9049a4a2ba02e3b37d113f00d063e66` |
| entropy | `b52a72ddc63dc5ae2a41df0deb6c780fc990079d` |
| factoring | `5271af623135efbf7116d6e32e38ab8ac3d48eed` |

Reviewed-file manifest SHA-256: `c26e97e3f15b7f4de4b57f2fa47358447a3f4bbea7816730e84805046ce449a0` (108 files).
The manifest includes tracked files and nonignored untracked regular files,
except AUDIT.md and SUGGESTIONS.md. Sort repository-relative paths; emit
`SHA256(file)`, two spaces, path and newline; hash that UTF-8 manifest.
Hashes identify the captured working contents, including dirty files, rather
than treating HEAD alone as their identity. A later source change requires
revalidation of the affected findings and results.

## Assessment

The tested integer and modular kernels satisfy their checked identities.
Two numerical boundary defects are reproduced: a finite log-gamma becomes
infinite, and lattice enumeration rejects a valid positive-definite problem.
The latter matters to factoring's polynomial search; the former is inherited
by entropy's public `lgamma` and is also relevant to factoring's distribution
models. Neither experiment establishes that an ordinary factoring run reaches
these extreme inputs.

## Findings

### R1 — High: floating conversion can destroy a valid lattice metric

**Reproduced.** [src/lattice.rs](src/lattice.rs), `short_vectors_form` and
`closest_vectors_form`. Take the identity basis of Z², the integral form
`diag(1, 2^1000)`, squared bound 1, and, for closest vectors, target zero.
The basis is already LLL-reduced. The nonzero short vectors are exactly
`(−1,0)` and `(1,0)`; the closest-vector result also includes zero.
Both functions panic on this input.

The Gram conversion shifts every integer right by `widest_bits − 900` before
converting it to f64. Here the shift is 101, so the positive entry 1 becomes
zero. The subsequent positivity assertion diagnoses a singular metric that
the conversion itself created. Exact verification of candidates cannot recover
candidates excluded before enumeration. A fixed relative slack of `1e-9`
also supplies no general bound on cancellation in floating Gram–Schmidt.

**Required result:** preserve valid small Gram directions, or return an
explicit numerical-limit outcome. A completeness claim needs rigorous pruning
bounds or an exact fallback. Recompute all returned norms exactly. Factoring's
`gnfs::kleinjung` calls `closest_vectors_form`; audit that caller's fallback
when the upstream outcome becomes explicit.

### R2 — Medium: log-gamma overflows before taking the logarithm

**Reproduced.** [src/number_theory.rs](src/number_theory.rs), `ln_gamma`.
At `x = 1e-310`, the result is positive infinity. The reflection path evaluates
`ln(π / sin(πx))`; the quotient overflows although its logarithm does not.
The recurrence `ln Γ(x) = ln Γ(1+x) − ln x` gives approximately
`713.8013788281542`, a finite f64. This identity follows directly from
[the gamma recurrence, DLMF §5.5(i)](https://dlmf.nist.gov/5.5).

**Required result:** evaluate the small-positive regime in logarithmic form,
with a stated domain and measured absolute error near zeros of log-gamma.
Relative “fifteen digits” language is inappropriate where the answer is zero.
Check subnormals, transition boundaries, infinities and invalid arguments.
Entropy's `math::lgamma` delegates directly to this function. The defect is
in that public numerical contract; corruption of a shipped χ² result was not
reproduced.

### R3 — Medium: bounded enumeration needs an explicit completion result

**Source inspection.** `short_vectors_form` stops after 50,000,000 visits and
returns the vectors collected so far, sorted and truncated. Its return type
cannot distinguish exhaustion from truncation. Both enumeration functions
document their visit caps, but neither returns completion or numerical status.
Sorting the candidates found does not prove they are the globally shortest.

Return completion status, visit count and numerical status alongside the
candidates. Keep every returned vector certified; reserve “complete” for an
exhausted, sound search. Also reconcile `short_vectors_form`'s documented
negative-bound panic with its early empty return. The visit-cap behavior was
inspected, not driven through 50 million iterations in this pass.

### R4 — Medium: coefficient derivation is not reproducible in-tree

**Source inspection.** `ln_gamma` embeds nine coefficients for `g = 7`.
[CITATIONS.md](CITATIONS.md) identifies Lanczos's paper and the parameter,
but the inspected tree supplies no generator or error analysis for this
particular coefficient set. Integer/half-integer tests cover useful values,
but cannot establish the full positive-real accuracy claim.

Keep a mathematical derivation and a deterministic high-precision coefficient
generator, with rounding rules and approximation error separated from floating
evaluation error. This is an evidence requirement for a specific numeric
table, not a claim that the table's ordinary values are wrong.

## Verification

| Fresh command or experiment | Result |
|---|---|
| `cargo test --offline --locked --release --all-targets` | 389 passed, 20 ignored |
| Same with `--features wipe` | 390 passed, 20 ignored |
| `cargo test --offline --locked --release --doc` | 7 passed |
| Same doctests with `--features wipe` | 7 passed |
| `mod_inverse_u128` against Python integer inversion | 2,500 cases, no mismatch |
| `crt_combine_u64` against integer CRT and both congruences | 1,000 cases, no mismatch |
| Lattice and small-positive log-gamma boundary probes | R1 and R2 reproduced |

The arithmetic probes used seed 20260916. Inverse cases covered modulus 1,
small moduli, 2^64, 2^127, 2^128−1 and 2^128−159, plus random 128-bit values;
zero modulus is outside the API's nonpanicking domain. The first 500 cases
cycle the moduli `[1,2,3,4,5,2^64,2^64−1,2^127,2^128−1,2^128−159]`; the
next 2,000 draw `m=max(1,rng.getrandbits(128))`, then a 128-bit value. The first
500 draw only the value. The same Python `random.Random(20260916)` then draws
1,000 CRT tuples in `(m1,m2,r1,r2)` order, each component 64 bits.
Expected inverses use `pow(a,-1,m)` when gcd is one; expected CRT uses
`a=r1 % m1; x=a+m1*((r2-a)*pow(m1,-1,m2) % m2)` for coprime nonzero moduli.
CRT inputs included unreduced residues and noncoprime moduli. Every returned CRT value additionally
satisfied both congruences and `0 ≤ x < m1*m2`.
The existing division tests check `n=q*d+r`, `r<d` and agreement with
bitwise long division. GF(2) tests expand filtered dependencies and check their
XOR against the original matrix. The 20 ignored tests, other architectures,
MSRV and performance sweeps were not rerun.

Minimal R1/R2 reproducer in a client depending on `rust-mp`:

```rust
use rump::{BigInt, BigUint, Sign};
let one = BigInt::from_i64(1);
let zero = BigInt::zero();
let basis = vec![vec![one.clone(), zero.clone()],
                 vec![zero.clone(), one.clone()]];
let mut power = BigUint::one();
power.shl_bits(1000);
let form = vec![vec![one.clone(), zero.clone()],
                vec![zero.clone(), BigInt::from_parts(Sign::Positive, power)]];
assert!(std::panic::catch_unwind(||
    rump::lattice::short_vectors_form(&basis, &form, &one, 10)
).is_err());
assert!(std::panic::catch_unwind(||
    rump::lattice::closest_vectors_form(&basis, &form, &[zero.clone(), zero], &one, 10)
).is_err());
assert!(rump::number_theory::ln_gamma(1e-310).is_infinite());
```

These assertions reproduce defects, so their expected outcomes must change
when the defects are repaired.

## Cross-repository contracts

| Owner | Contract and consumers |
|---|---|
| [rump](../rump/AUDIT.md) | Exact integer/field arithmetic and matrix identities; numerical approximations identify their domain and error. Used by all three companions. |
| [cryptography](../cryptography/AUDIT.md) | Scheme validation, entropy requirements, secret handling and timing properties. Enables rump's `wipe`; that feature does not make arithmetic constant-time. |
| [entropy](../entropy/AUDIT.md) | The statistic, input projection, null distribution and calibrated decision rule. A statistical PASS is not a security claim. |
| [factoring](../factoring/AUDIT.md) | Relation identities, matrix expansion and verified divisors. Uses entropy without default features; probable-prime leaves remain distinguished from proved primes. |

The links assume the repositories are sibling checkouts. Mathematical kernels
belong with their owner; consumers add their own preconditions and verify
results at the boundary. With cryptography enabled, Cargo unifies rump's
`wipe` feature across the dependency graph. Factoring alone was checked with
`cargo tree --offline --locked -e features -i rust-mp`: no `wipe` and no active
cryptography dependency.

Implementation work starts from papers, specifications and mathematical
derivations. A citation identifies the exact equation or algorithm, and the
implementation states its representation, hypotheses and invariants. Numerical
tables need a derivation or a precisely identified standard table. Published
answer files are test data; another implementation's source is not an
implementation reference. A passing round trip alone cannot validate two
functions that share the same mistake.
