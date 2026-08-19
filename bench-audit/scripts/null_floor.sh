#!/usr/bin/env bash
# The rig's resolution floor, measured by swapping the two arms.
#
# The null comparison runs v0.3.0 against a second, independently linked build
# of v0.3.0. Any difference it reports is the rig's, not the code's, because
# both sides are the same program.
#
# It is run in BOTH directions, and that is the point. A ratio on its own
# cannot say where a difference comes from; the pair can:
#
#   forward x reverse ~ 1      the effect follows the BINARY -- one image is
#                              genuinely faster than the other, from identical
#                              source, through code placement and alignment
#                              (Mytkowicz et al., ASPLOS '09).
#   forward x reverse ~ fwd^2  the effect follows the POSITION in the ABBA
#                              sequence, and the wrapper's drift cancellation
#                              is not doing its job.
#   both ~ 1                   the rig is clean for that workload.
#
# Measured here, the products cluster on 1.0, so it is the binary: for
# `int_add` at 256 bits one image runs 5.5% faster than a byte-identical
# twin. The effect tracks operation cost, as code placement should -- large
# where per-call overhead dominates, under 1% for `mod_pow`, `int_div_rem`
# and `nt_gcd`, where the arithmetic swamps it.
#
# The consequence for the audit is a per-workload floor: a paired result
# inside that band is not attributable to the revision, however tight its
# confidence interval. A tight interval around a layout artifact is still an
# artifact.
#
# Run on an IDLE machine. An earlier attempt ran alongside the paired matrix,
# where 32 cells were continuously executing one of the two binaries and never
# the other; that privileges one image's page cache and shared text pages, and
# it produced a one-sided bias in all ten cells that had nothing to say about
# the code.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASELIST="${1:?case list: lines of 'case corpus'}"
OUTROOT="${2:?output root}"
V="${3:-$HERE/adapter-v030/target/release/adapter-v030}"
N="${4:-$HERE/adapter-v030-null/target/release/adapter-v030-null}"

"$HERE/scripts/check_adapters.sh" || {
    echo "refusing to measure a floor with drifted adapter source" >&2
    exit 1
}

mkdir -p "$OUTROOT"
"$HERE/scripts/run_matrix.sh" "$CASELIST" "$OUTROOT/forward" "$V" "$N" > "$OUTROOT/forward.log" 2>&1 &
fwd=$!
"$HERE/scripts/run_matrix.sh" "$CASELIST" "$OUTROOT/reverse" "$N" "$V" > "$OUTROOT/reverse.log" 2>&1 &
rev=$!
wait $fwd $rev

echo "forward and reverse complete; reduce with scripts/null_floor_report.py"
