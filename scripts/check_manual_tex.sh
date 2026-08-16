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
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/src"
cat > "$WORK/Cargo.toml" <<EOF
[package]
name = "manual-tex-check"
version = "0.0.0"
edition = "2021"

[dependencies]
rust-mp = { path = "$ROOT_DIR" }
EOF

python3 "$ROOT_DIR/scripts/extract_manual_examples.py" "$ROOT_DIR/manual.tex" \
    > "$WORK/src/main.rs"

cargo run --quiet --release --manifest-path "$WORK/Cargo.toml"
echo "manual.tex listings: all compiled and passed"
