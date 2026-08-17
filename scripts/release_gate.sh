#!/usr/bin/env bash
# The release gate: every check that must pass before a Rump tag.
#
# Trust only the final exit status. Each leg prints its own PASS/FAIL line, but
# a leg can fail late or print reassuring output before dying, so the summary
# at the end is the answer and the exit code is the contract.
#
# Two legs need rustup toolchains rather than whatever `cargo` is on PATH:
# the 32-bit cross-check needs a target Homebrew's rust cannot install, and
# the MSRV leg needs a specific older compiler. Both are named explicitly for
# that reason. Override with RUMP_MSRV / RUMP_CROSS_TARGET if the pins move.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

MSRV="${RUMP_MSRV:-1.87}"
CROSS_TARGET="${RUMP_CROSS_TARGET:-i686-unknown-linux-gnu}"

failed=()
passed=()

run() {
    local name="$1"
    shift
    printf '\n=== %s ===\n' "$name"
    if "$@"; then
        passed+=("$name")
        printf '  PASS  %s\n' "$name"
    else
        failed+=("$name")
        printf '  FAIL  %s\n' "$name"
    fi
}

# `cargo fmt --all` is scoped to this package: Rump is standalone, with no
# workspace members and no path dependencies. Do not copy this line into a
# repository whose workspace reaches sibling crates — it will rewrite them.
run "fmt"            cargo fmt --all -- --check
run "clippy"         cargo clippy --all-targets -- -D warnings
run "rustdoc"        env RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
run "test"           cargo test --all-targets
run "doctests"       cargo test --doc
run "test-release"   cargo test --release --all-targets
run "manual.tex"     bash scripts/check_manual_tex.sh
run "package"        cargo package --allow-dirty --no-verify

# Cross and MSRV: named toolchains, not PATH.
cross_check() {
    local tc
    tc="$(rustup toolchain list | awk '/(default|active)/ {print $1; exit}')"
    local home="$HOME/.rustup/toolchains/$tc"
    RUSTC="$home/bin/rustc" "$home/bin/cargo" check --target "$CROSS_TARGET" --all-targets
}
msrv_check() {
    local home="$HOME/.rustup/toolchains/$MSRV-$(uname -m)-apple-darwin"
    [ -d "$home" ] || home="$(rustup which --toolchain "$MSRV" cargo 2>/dev/null | sed 's|/bin/cargo||')"
    [ -d "$home" ] || { echo "MSRV toolchain $MSRV not installed"; return 1; }
    RUSTC="$home/bin/rustc" "$home/bin/cargo" check --all-targets
}
run "cross-$CROSS_TARGET" cross_check
run "msrv-$MSRV"          msrv_check

# `git diff --check` reports whitespace errors in the working tree; HEAD names
# the committed state, which is what a tag will carry.
run "whitespace"     git diff --check HEAD

printf '\n===== release gate =====\n'
# `${arr[@]}` on an empty array trips `set -u`; the `+` expansions below make
# the empty case expand to nothing instead. An earlier version of this script
# died here with "unbound variable" *and still exited 0*, which is precisely
# the failure the "trust only the exit status" rule exists to catch.
for leg in ${passed[@]+"${passed[@]}"}; do printf '  pass  %s\n' "$leg"; done
for leg in ${failed[@]+"${failed[@]}"}; do printf '  FAIL  %s\n' "$leg"; done

if [ ${#failed[@]} -ne 0 ]; then
    printf '\n%d of %d legs failed.\n' "${#failed[@]}" "$(( ${#passed[@]} + ${#failed[@]} ))"
    exit 1
fi
printf '\nall %d legs green.\n' "${#passed[@]}"
