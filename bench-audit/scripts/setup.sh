#!/usr/bin/env bash
# Prepare the audit: worktrees from the two tags, corpus, and all executables.
#
# Reproducible from committed state alone: the worktrees come from tags, and the
# corpus is regenerated and checked against corpus.sha256, so a stale or edited
# corpus cannot be compared against a fresh one silently.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TREES="${RUMP_AUDIT_TREES:-$REPO/../rump-audit}"

mkdir -p "$TREES"
for tag in v0.2.2 v0.3.0; do
    dir="$TREES/$tag"
    if [ ! -d "$dir" ]; then
        git -C "$REPO" worktree add -q --detach "$dir" "$tag"
    fi
    printf '%-8s %s %s\n' "$tag" "$(git -C "$REPO" rev-parse "$tag^{commit}")" "$dir"
done

# A third, independent copy of v0.3.0 for the null comparison: two separately
# built executables of the same revision, which must show no directional result.
if [ ! -d "$TREES/v0.3.0-null" ]; then
    git -C "$REPO" worktree add -q --detach "$TREES/v0.3.0-null" v0.3.0
fi

( cd "$HERE/corpus-gen" && cargo build --release -q )
"$HERE/corpus-gen/target/release/corpus-gen" "$HERE/corpus"
( cd "$HERE/corpus" && shasum -a 256 -c "$HERE/corpus.sha256" >/dev/null ) \
    && echo "corpus matches corpus.sha256"

for crate in abba adapter-v022 adapter-v030; do
    ( cd "$HERE/$crate" && cargo build --release -q )
    echo "built $crate"
done
