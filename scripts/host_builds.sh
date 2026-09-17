#!/usr/bin/env bash
# Build a pushed rump revision on the supported hosts.
#
#   scripts/host_builds.sh [--test] [REVISION] [HOST...]
#
# REVISION (default HEAD) must already be on GitHub: each host clones it from
# there into a fresh directory under its local /tmp and builds all targets in
# release mode, with default features and with `wipe`. --test also runs the
# test suite with default features. The Mac is covered by
# scripts/release_gate.sh.
#
# Hosts default to one of each architecture group:
#   moore     x86_64 AMD EPYC (with dennard; twilight is the same CPU on
#             Ubuntu 22.04, so it is listed too)
#   twilight
#   knuth     aarch64 Cortex-X925 (with baase, vinge, paris)
#   darby     aarch64 Cortex-A76
#   dmz       x86_64 Intel
#
# Prints one line per host and build; exits nonzero if any fails.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST=0
if [[ "${1:-}" == "--test" ]]; then
    TEST=1
    shift
fi
REVISION="$(git -C "$ROOT" rev-parse --verify "${1:-HEAD}^{commit}")" || exit 2
[[ $# -gt 0 ]] && shift
HOSTS=("$@")
[[ ${#HOSTS[@]} -eq 0 ]] && HOSTS=(moore twilight knuth darby dmz)
URL="https://github.com/darrelllong/rump.git"

if ! git ls-remote "$URL" | grep -q "^$REVISION"; then
    # A revision that is not a branch tip can still be fetched by hash.
    echo "note: $REVISION is not a branch tip on GitHub; hosts fetch it by hash"
fi

LOGS="$(mktemp -d "${TMPDIR:-/tmp}/rump-hosts.XXXXXX")"
echo "revision $REVISION; logs in $LOGS"

for host in "${HOSTS[@]}"; do
    {
    (
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" bash -s -- "$REVISION" "$URL" "$TEST" <<'REMOTE'
set -uo pipefail
revision=$1 url=$2 test=$3
export PATH="$HOME/.cargo/bin:$PATH"
dir="/tmp/rump-build-$(id -un)-${revision:0:12}"
rm -rf "$dir"
git init -q "$dir" && cd "$dir" && git fetch -q --depth 1 "$url" "$revision" && git checkout -q FETCH_HEAD || { echo "FETCH FAILED"; exit 1; }
echo "RUSTC $(rustc --version)"
status=0
nice -n 10 cargo build -q --release --all-targets && echo "BUILD default ok" || { echo "BUILD default FAILED"; status=1; }
nice -n 10 cargo build -q --release --all-targets --features wipe && echo "BUILD wipe ok" || { echo "BUILD wipe FAILED"; status=1; }
if [[ $test == 1 ]]; then
    nice -n 10 cargo test -q --release 2>&1 | grep -E '^test result' | awk '{p += $4; f += $6} END {printf "TEST %d passed, %d failed\n", p, f}'
    [[ ${PIPESTATUS[0]} == 0 ]] || { echo "TEST FAILED"; status=1; }
fi
cd / && rm -rf "$dir"
exit $status
REMOTE
    ) >"$LOGS/$host.log" 2>&1
        echo $? >"$LOGS/$host.status"
    } &
done
wait

failed=0
for host in "${HOSTS[@]}"; do
    status="$(cat "$LOGS/$host.status" 2>/dev/null || echo 1)"
    rustc="$(grep -m1 '^RUSTC' "$LOGS/$host.log" | cut -d' ' -f2-)"
    summary="$(grep -E '^(BUILD|TEST|FETCH)' "$LOGS/$host.log" | tr '\n' ';' | sed 's/;$//')"
    if [[ $status == 0 ]]; then
        printf '  PASS  %-9s %s — %s\n' "$host" "$rustc" "$summary"
    else
        printf '  FAIL  %-9s %s — %s (%s)\n' "$host" "$rustc" "$summary" "$LOGS/$host.log"
        failed=1
    fi
done
exit $failed
