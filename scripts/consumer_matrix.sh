#!/usr/bin/env bash
# Test a rump revision against its consumers at recorded revisions.
#
#   scripts/consumer_matrix.sh [--ignored] [RUMP CRYPTOGRAPHY ENTROPY FACTORING]
#
# Each argument is a revision in the sibling checkout of that name (../rump is
# this repository); without them every repository is taken at its HEAD. The
# siblings are cloned into a fresh directory, so uncommitted work never enters
# a run, and every leg builds in its own fresh target directory.
#
# Legs:
#   cryptography       cargo test --release --no-fail-fast, OpenSSL required
#   entropy-default    cargo test --release --no-fail-fast
#   entropy-minimal    cargo test --release --no-fail-fast --no-default-features
#   factoring          cargo test --release --no-fail-fast (entropy minimal),
#                      and its feature graph must not enable rump's `wipe`
# With --ignored, also:
#   cryptography-ignored  every ignored cryptography test
#   rump-ignored          rump's ignored correctness tests (the timing probes
#                         assert nothing and are not run); the log-gamma sweep
#                         needs RUMP_LN_GAMMA_SWEEP, from
#                         scripts/lanczos_coefficients.py --sweep FILE
#
# Prints the resolved revisions and one line per leg; exits nonzero if any leg
# fails or cannot run. Logs stay in the work directory, which is printed.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIBLINGS="$(dirname "$ROOT")"
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
    printf '%-13s %s\n' "$repo" "$commit"
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
leg entropy-default entropy cargo test --release --no-fail-fast
leg entropy-minimal entropy cargo test --release --no-fail-fast --no-default-features
leg factoring factoring cargo test --release --no-fail-fast
if (cd "$WORK/factoring" && cargo tree -e features -i rust-mp 2>/dev/null) | grep -q '"wipe"'; then
    printf '  FAIL  %-22s %s\n' "factoring-no-wipe" "rump's wipe feature is enabled in factoring's graph"
    failed=1
else
    printf '  PASS  %-22s %s\n' "factoring-no-wipe" "rump's wipe feature is not enabled"
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
