#!/usr/bin/env bash
# GCD & friends at scale: one pilot-bench row per operation and size, across a
# range far beyond the main tables' 256–4096 bits — where gcd's Half-GCD
# dispatch and its subquadratic curve become visible against the still-O(n²)
# gcdext / modinv / jacobi.
#
# Emits the same Markdown row format as bench_primitives.sh (whose single-op
# mode does the measuring), so perf_analysis.py renders these files with the
# same fit / compare / plot machinery:
#
#   scripts/bench_gcd_scaling.sh > bench/gcd_scaling_<host>.md
#   PILOT_MP_BIN=target/bench_gmp/pilot_gmp \
#     scripts/bench_gcd_scaling.sh > bench/gmp_gcd_scaling_<host>.md
#
# Every op runs to a full megabit. A single quadratic reading costs seconds at
# the top sizes, so those rows get a longer session cap: the same convergence
# discipline, given enough budget to gather a usable sample rather than
# truncating the table.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GCD_SIZES=(8192 16384 32768 65536 131072 262144 524288 1048576)
FRIEND_SIZES=(8192 16384 32768 65536 131072 262144 524288 1048576)
# Bit width above which a row gets the long session.
LONG_ABOVE=262144
LONG_SESSION=180

echo "# gcd & friends at scale"
echo
echo "One row per op and size via \`bench_primitives.sh <op>\` (fresh random"
echo "operands, pilot-bench convergence, same columns as the primitives"
echo "tables). Sizes are bit widths."
echo
echo "| Operation | mean ms/op | ±95% CI | min ns | p50 ns | p99 ns | max ns | max/min |"
echo "|---|---:|---:|---:|---:|---:|---:|---:|"
for size in "${GCD_SIZES[@]}"; do
    bash "$ROOT_DIR/scripts/bench_primitives.sh" "gcd_${size}"
done
for op in gcdext modinv jacobi; do
    for size in "${FRIEND_SIZES[@]}"; do
        if (( size > LONG_ABOVE )); then
            PILOT_MP_SESSION=$LONG_SESSION \
                bash "$ROOT_DIR/scripts/bench_primitives.sh" "${op}_${size}"
        else
            bash "$ROOT_DIR/scripts/bench_primitives.sh" "${op}_${size}"
        fi
    done
done
