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
| entropy-default | `cargo test --release --no-fail-fast` |
| entropy-minimal | `cargo test --release --no-fail-fast --no-default-features` |
| factoring | `cargo test --release --no-fail-fast` (entropy without default features) |
| factoring-no-wipe | `cargo tree` succeeds and rump's `wipe` feature is absent from factoring's graph; `--self-test` checks this leg on a graph without `wipe`, one with it, and a manifest that fails to parse |
| cryptography-ignored | every ignored cryptography test (`--ignored` only) |
| rump-ignored | rump's ignored correctness tests: the AKS stress test, the Barrett correction search and the log-gamma sweep (`--ignored` only; needs `RUMP_LN_GAMMA_SWEEP`) |

rump's other ignored tests are timing probes that assert nothing.

Last run, 2026-09-16, on the development Mac (aarch64-apple-darwin, rustc
1.95.0), with `--ignored`:

| Repository | Revision |
|---|---|
| rump | `4218630e0e3ca9bc2937a3d8ebc2b387e8cef1eb` |
| cryptography | `40a9bdff34fa926e343eb9acf1a4d00aa362dbd9` |
| entropy | `7b574f1ff394219197aeb24e4ce137c91329fadf` |
| factoring | `cec57c8f9ba5508a87c77689562bde94b167219f` |

| Leg | Result |
|---|---|
| cryptography | 1677 passed, 0 failed, 19 ignored |
| entropy-default | 443 passed, 0 failed, 1 ignored |
| entropy-minimal | 329 passed, 0 failed, 1 ignored |
| factoring | 295 passed, 0 failed, 3 ignored |
| factoring-no-wipe | wipe not enabled |
| cryptography-ignored | 18 passed, 1 failed |
| rump-ignored | 2 passed, 1 failed |

Both failures are understood. cryptography's
`constant_time_eq_mask_timing_is_length_only` compared means of five short
samples and failed at ratio 4.55 while other builds loaded the machine; run
alone it passed 3 of 3, and cryptography's working tree now compares the
fastest of 101 samples. rump's log-gamma sweep failed at `x = 0.45` because
the bound documented at `4218630` was wrong below 1/2; `bdbe724` corrected the
bound and the test. The same cryptography ignored tests passed 19 of 19 on
twilight (x86_64) against rump `4218630`, and the three rump tests passed there
with the earlier sweep.

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

Last run, 2026-09-16, `--test` at `8df086eccfca543d026536081535d7095f7eaa3c`:
every host built both feature modes and passed 408 tests with 0 failures —
moore, twilight and knuth on rustc 1.95.0, darby and dmz on rustc 1.93.1.

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
