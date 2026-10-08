#!/usr/bin/env bash
# Extract every code listing from manual.tex and execute it against the
# crate — the manual.tex counterpart of the MANUAL.md ↔ manual_examples.rs
# mirror. A rebuild of manual.pdf is gated on this passing.
#
# Mechanism: scripts/extract_manual_examples.py concatenates the listings
# into one main.rs; a throwaway crate in a temp directory depends on this
# repository by path and runs it. Every assertion in every listing must
# hold. Exit status is the run's.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Under Git Bash on Windows `pwd` is a POSIX-style path (/c/...) that cargo
# cannot resolve from inside a manifest; the mixed form (C:/...) is one both
# read.
if command -v cygpath >/dev/null 2>&1; then
    ROOT_DIR="$(cygpath -m "$ROOT_DIR")"
fi
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# python3 where there is one, else python where that is Python 3: the
# Windows runners name it so, and the Microsoft Store stub that answers to
# python3 on some Windows installations fails the probe.
PYTHON=""
for candidate in python3 python; do
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 7) else 1)' >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "check_manual_tex.sh: no Python 3 on PATH" >&2
    exit 1
fi

mkdir -p "$WORK/src"
cat > "$WORK/Cargo.toml" <<EOF
[package]
name = "manual-tex-check"
version = "0.0.0"
edition = "2021"

[dependencies]
rust-mp = { path = "$ROOT_DIR" }
EOF

"$PYTHON" "$ROOT_DIR/scripts/extract_manual_examples.py" "$ROOT_DIR/manual.tex" \
    > "$WORK/src/main.rs"

cargo run --quiet --release --manifest-path "$WORK/Cargo.toml"
echo "manual.tex listings: all compiled and passed"
