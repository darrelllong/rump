#!/usr/bin/env bash
# Test a rump revision against its consumers at recorded revisions.
#
#   scripts/consumer_matrix.sh [--ignored] [RUMP CRYPTOGRAPHY ENTROPY FACTORING]
#   scripts/consumer_matrix.sh --self-test
#
# Each argument is a revision in the sibling checkout of that name (../rump is
# this repository); without them every repository is taken at its HEAD. The
# siblings are cloned into a fresh directory, so uncommitted work never enters
# a run, and every leg builds in its own fresh target directory.
#
# Legs:
#   cryptography       cargo test --release --no-fail-fast, OpenSSL required
#   cryptography-all   the same with --all-features
#   entropy-default    cargo test --release --no-fail-fast
#   entropy-minimal    cargo test --release --no-fail-fast --no-default-features
#   factoring          cargo test --release --no-fail-fast (entropy minimal)
#   factoring-no-wipe  `cargo tree` must succeed, and its graph must not
#                      enable rump's `wipe`
# With --ignored, also:
#   cryptography-ignored  every ignored cryptography test
#   rump-ignored          rump's ignored correctness tests (the timing probes
#                         assert nothing and are not run); the log-gamma sweep
#                         needs RUMP_LN_GAMMA_SWEEP, from
#                         scripts/lanczos_coefficients.py --sweep FILE
#
# Prints the resolved revisions with each repository's Cargo.lock digest and
# one line per leg; exits nonzero if any leg fails or cannot run. Logs stay in
# the work directory, which is printed.
#
# --self-test exercises the feature check on its three outcomes: a graph
# without `wipe` (factoring at HEAD), a graph with it (cryptography at HEAD),
# and a failed query (a manifest that does not parse). It builds nothing.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The consumers are checked out beside the main rump checkout, which is not
# where a linked worktree lives; find it through the shared git directory.
SIBLINGS="$(dirname "$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")")"

# no_wipe DIRECTORY: 0 when `cargo tree` resolves the directory's graph and
# rump's `wipe` is absent from it, 1 when it is present, 2 when the query
# fails. The reason is printed either way.
no_wipe() {
    local graph status
    graph="$(cd "$1" && cargo tree -e features -i rust-mp 2>&1)"
    status=$?
    if [[ $status -ne 0 ]]; then
        echo "cargo tree failed (exit $status): $(printf '%s' "$graph" | tail -1)"
        return 2
    fi
    if printf '%s\n' "$graph" | grep -q '"wipe"'; then
        echo "rump's wipe feature is enabled"
        return 1
    fi
    echo "rump's wipe feature is not enabled"
    return 0
}

if [[ "${1:-}" == "--self-test" ]]; then
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/rump-consumers-selftest.XXXXXX")"
    git clone -q "$ROOT" "$WORK/rump"
    for repo in cryptography entropy factoring; do
        git clone -q "$SIBLINGS/$repo" "$WORK/$repo" || exit 2
    done
    mkdir -p "$WORK/broken" && printf '[package\nname = \n' >"$WORK/broken/Cargo.toml"
    bad=0
    expect() {
        local dir=$1 want=$2 reason got
        reason="$(no_wipe "$WORK/$dir")"
        got=$?
        if [[ $got -eq $want ]]; then
            printf '  PASS  %-12s status %d: %s\n' "$dir" "$got" "$reason"
        else
            printf '  FAIL  %-12s status %d, wanted %d: %s\n' "$dir" "$got" "$want" "$reason"
            bad=1
        fi
    }
    expect factoring 0
    expect cryptography 1
    expect broken 2
    rm -rf "$WORK"
    exit $bad
fi
IGNORED=0
if [[ "${1:-}" == "--ignored" ]]; then
    IGNORED=1
    shift
fi
if [[ $# -ne 0 && $# -ne 4 ]]; then
    sed -n '2,24p' "${BASH_SOURCE[0]}"
    exit 2
fi
REPOS=(rump cryptography entropy factoring)
REVS=("${1:-HEAD}" "${2:-HEAD}" "${3:-HEAD}" "${4:-HEAD}")

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rump-consumers.XXXXXX")"
echo "work directory: $WORK"
for i in 0 1 2 3; do
    repo=${REPOS[$i]}
    source_dir="$SIBLINGS/$repo"
    [[ $repo == rump ]] && source_dir="$ROOT"
    commit="$(git -C "$source_dir" rev-parse --verify "${REVS[$i]}^{commit}")" || exit 2
    git clone -q "$source_dir" "$WORK/$repo" && git -C "$WORK/$repo" checkout -q "$commit" || exit 2
    lock="$(shasum -a 256 "$WORK/$repo/Cargo.lock" 2>/dev/null | cut -c1-16)"
    printf '%-13s %s  Cargo.lock %s\n' "$repo" "$commit" "${lock:-absent}"
done
mkdir -p "$WORK/logs"

failed=0
# leg NAME DIRECTORY COMMAND...
leg() {
    local name=$1 dir=$2
    shift 2
    local log="$WORK/logs/$name.log"
    (cd "$WORK/$dir" && CARGO_TARGET_DIR="$WORK/target-$name" "$@") >"$log" 2>&1
    local status=$?
    local counts
    counts="$(awk '/^test result:/ {p += $4; f += $6; i += $8} END {printf "%d passed, %d failed, %d ignored", p, f, i}' "$log")"
    if [[ $status -eq 0 ]]; then
        printf '  PASS  %-22s %s\n' "$name" "$counts"
    else
        printf '  FAIL  %-22s %s (exit %d; %s)\n' "$name" "$counts" "$status" "$log"
        failed=1
    fi
}

leg cryptography cryptography env CRYPTOGRAPHY_OPENSSL_REQUIRED=1 cargo test --release --no-fail-fast
leg cryptography-all cryptography env CRYPTOGRAPHY_OPENSSL_REQUIRED=1 cargo test --release --no-fail-fast --all-features
leg entropy-default entropy cargo test --release --no-fail-fast
leg entropy-minimal entropy cargo test --release --no-fail-fast --no-default-features
leg factoring factoring cargo test --release --no-fail-fast
if reason="$(no_wipe "$WORK/factoring")"; then
    printf '  PASS  %-22s %s\n' "factoring-no-wipe" "$reason"
else
    printf '  FAIL  %-22s %s\n' "factoring-no-wipe" "$reason"
    failed=1
fi

if [[ $IGNORED -eq 1 ]]; then
    leg cryptography-ignored cryptography env CRYPTOGRAPHY_OPENSSL_REQUIRED=1 \
        cargo test --release --no-fail-fast -- --ignored
    if [[ -z "${RUMP_LN_GAMMA_SWEEP:-}" ]]; then
        printf '  FAIL  %-22s %s\n' "rump-ignored" "RUMP_LN_GAMMA_SWEEP is not set"
        failed=1
    else
        leg rump-ignored rump cargo test --release --lib -- --ignored \
            barrett_correction_search aks_stress_matches_an_independent_sieve \
            the_log_gamma_sweep_matches_high_precision
    fi
fi

exit $failed
