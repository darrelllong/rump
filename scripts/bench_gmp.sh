#!/usr/bin/env bash
# Build and run the GMP comparison benchmark (scripts/bench_gmp.c), the
# apples-to-apples counterpart of `cargo run --release --bin bench_bigint`.
# Any arguments are passed through as sizes in bits, matching bench_bigint.
#
# Requires libgmp: `brew install gmp` on macOS, libgmp-dev / gmp-devel on
# Linux.  Override the compiler with CC, or the GMP prefix with GMP_PREFIX.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT_DIR/scripts/bench_gmp.c"
OUT_DIR="$ROOT_DIR/target/bench_gmp"
OUT="$OUT_DIR/bench_gmp"
# pilot_gmp is the per-primitive GMP mirror of pilot_mp, built alongside so
# scripts/bench_primitives.sh can drive it (PILOT_MP_BIN=.../pilot_gmp) for
# the rump-vs-GMP comparison in PERFORMANCE.md.
PILOT_SRC="$ROOT_DIR/scripts/pilot_gmp.c"
PILOT_OUT="$OUT_DIR/pilot_gmp"
CC="${CC:-cc}"

CFLAGS=(-O2 -Wall -Wextra)
LDFLAGS=(-lgmp)

# Homebrew keeps GMP out of the default search path; Linux distributions
# install it where the compiler already looks.
GMP_PREFIX="${GMP_PREFIX:-}"
if [[ -z "$GMP_PREFIX" ]]; then
    for candidate in /opt/homebrew/opt/gmp /usr/local/opt/gmp; do
        if [[ -f "$candidate/include/gmp.h" ]]; then
            GMP_PREFIX="$candidate"
            break
        fi
    done
fi
if [[ -n "$GMP_PREFIX" ]]; then
    CFLAGS+=(-I"$GMP_PREFIX/include")
    LDFLAGS=(-L"$GMP_PREFIX/lib" "${LDFLAGS[@]}")
fi

mkdir -p "$OUT_DIR"
if [[ ! -x "$OUT" || "$SRC" -nt "$OUT" ]]; then
    "$CC" "${CFLAGS[@]}" -o "$OUT" "$SRC" "${LDFLAGS[@]}"
fi
if [[ ! -x "$PILOT_OUT" || "$PILOT_SRC" -nt "$PILOT_OUT" ]]; then
    "$CC" "${CFLAGS[@]}" -o "$PILOT_OUT" "$PILOT_SRC" "${LDFLAGS[@]}"
fi

# With no args, this is a build-only invocation (used to produce pilot_gmp);
# otherwise run the classic bigint mirror for backward compatibility.
if [[ $# -eq 0 ]]; then
    echo "built: $OUT"
    echo "built: $PILOT_OUT"
else
    exec "$OUT" "$@"
fi
