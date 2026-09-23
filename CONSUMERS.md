# Consumers and hosts

Rump is released only when its consumers pass against it and it builds on
every supported host. Two scripts produce that evidence; this file records
the last combination they certified.

## Consumer matrix

`scripts/consumer_matrix.sh [--ignored] RUMP CRYPTOGRAPHY ENTROPY FACTORING`
clones each repository at the given revision from its sibling checkout and
runs, each in a fresh target directory:

| Leg | Command |
|---|---|
| cryptography | `CRYPTOGRAPHY_OPENSSL_REQUIRED=1 cargo test --release --no-fail-fast` |
| cryptography-all | the same with `--all-features` |
| entropy-default | `cargo test --release --no-fail-fast` |
| entropy-minimal | `cargo test --release --no-fail-fast --no-default-features` |
| factoring | `cargo test --release --no-fail-fast` (entropy without default features) |
| factoring-no-wipe | `cargo tree` succeeds and rump's `wipe` feature is absent from factoring's graph; `--self-test` checks this leg on a graph without `wipe`, one with it, and a manifest that fails to parse |
| cryptography-ignored | every ignored cryptography test (`--ignored` only) |
| rump-ignored | rump's ignored correctness tests: the AKS stress test and the Barrett correction search (`--ignored` only) |

rump's other ignored tests are timing probes that assert nothing. Entropy
defines two feature modes, default and no default features, and the matrix
runs both.

Last run, 2026-09-22, on the development Mac (aarch64-apple-darwin, rustc
1.93.1), without `--ignored`:

| Repository | Revision |
|---|---|
| rump | `db6a9dc` |
| cryptography | `8001bd5` |
| entropy | `b8975c1` (0.6.0, the revision factoring pins) |
| factoring | `66f6619` |

| Leg | Result |
|---|---|
| cryptography | 1736 passed, 0 failed, 23 ignored |
| cryptography-all | 1736 passed, 0 failed, 23 ignored |
| entropy-default | 519 passed, 0 failed, 2 ignored |
| entropy-minimal | 139 passed, 0 failed, 1 ignored |
| factoring | 320 passed, 0 failed, 3 ignored |
| factoring-no-wipe | wipe not enabled |

The entropy revision is pinned rather than taken from its checkout because
factoring's own manifest pins it: against entropy's tip of the day the
factoring leg fails to compile on an entropy API factoring has not moved to,
which is a fact about those two crates and says nothing about rump.

The ignored legs were last run on 2026-09-16 against rump
`4218630e0e3ca9bc2937a3d8ebc2b387e8cef1eb`, where cryptography's
`constant_time_eq_mask_timing_is_length_only` failed at ratio 4.55 while
other builds loaded the machine — run alone it passed 3 of 3, and
cryptography now compares the fastest of 101 samples — and the same tests
passed 19 of 19 on twilight (x86_64).

## Fixtures from consumers

`tests/data/index_calculus_62.txt` is a matrix from a real run of factoring's
discrete logarithm — 62 columns over `GF(524351)`, the rows it hands to
`gfp::SparseMatrix` after pruning, and the kernel vector that run verified.
It came from factoring `7c5a5df`, whose logarithms modulo `p = 1048703` were
checked by hand rather than against another program. rump's test reads it,
confirms the claimed vector is in the kernel, and requires the solver to land
on the same line.

A constructed matrix exercises arithmetic; this one carries the entry
distribution a factor base actually produces, which is what the module exists
for.

## Hosts

`scripts/host_builds.sh [--test] [REVISION] [HOST...]` fetches a pushed
revision from GitHub on each host and builds all targets in release mode with
default features and with `wipe`; `--test` also runs the suite. The Mac is
covered by `scripts/release_gate.sh`.

| Group | Host tested | Architecture |
|---|---|---|
| Mac | development machine | aarch64-apple-darwin |
| moore, dennard, twilight | moore, twilight | x86_64 AMD EPYC 7452 (twilight on Ubuntu 22.04) |
| baase, vinge, paris, knuth | knuth | aarch64 Cortex-X925 |
| darby | darby | aarch64 Cortex-A76 |
| dmz | dmz | x86_64 Intel i5-8259U |

Last run, 2026-09-22, `--test` at `db6a9dc`: moore, twilight, baase, knuth
(rustc 1.95.0) and dmz (rustc 1.93.1) each built both feature modes and
passed 429 tests with 0 failures. darby was measuring benchmarks and was not
built at that revision.

## Reported by consumers

Runs by the consumers' own sessions, not by these scripts; recorded with
their revisions so they can be repeated.

| Date | Consumer | Revision | rump | Where | Result |
|---|---|---|---|---|---|
| 2026-09-16 | cryptography | `aa865da` | `8df086e` | Mac, moore, baase, darby, dmz | builds and tests pass; OpenSSL BER cross-check passes against 3.0.13, 3.5.5, 3.5.7 and 3.6.4 |
| 2026-09-16 | entropy | `f980cf7` | `8df086e` | moore, baase, dmz, darby | builds and tests pass |
| 2026-09-16 | entropy | `de1bd2d` | `1ccb3ce` | Mac | default features: 466 passed, 0 failed |

cryptography's CI checks out rump main rather than a pinned revision, so a
change to rump's public contracts is run through `consumer_matrix.sh` against
the consumers' current revisions before it is pushed. entropy records the
rump revision each battery run was built against (`scripts/provenance.sh`).
