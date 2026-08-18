#!/usr/bin/env bash
# One single-arm cell: calibrate, run Pilot, capture every artifact.
#
# For the APIs v0.2.2 does not have. There is no baseline to pair against, so
# what is measured is an absolute cost and not a ratio, and the ABBA ordering
# that cancels drift in the paired cells has nothing to cancel between. These
# figures are reported as baselines for future comparison and never as a
# comparison themselves.
#
# Pilot is the sole statistical authority here as in the paired path: this
# script reads its mean and CI and computes no interval of its own. A session
# that stops on the limit (exit 13) is recorded as inconclusive.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${PILOT_BENCH_CLI:-$HOME/pilot-bench/build/cli/bench}"

CASE="${1:?case name}"
CORPUS="${2:?corpus name}"
OUTDIR="${3:?output dir}"
PROGRAM="${4:-$HERE/adapter-v030/target/release/adapter-v030}"

# The same mandated statistics as the paired path.
PRESET=(--preset strict --confidence-level 0.99 --ci-perc 0.04)
PI='ns_per_op,ns/op,0,0,1:valid,,1,2,0'
LIMIT="${RUMP_SESSION_LIMIT:-1800}"

TARGET_MULT="${RUMP_INTERVAL_MULT:-5}"
base_repeat="$("$PROGRAM" --case "$CASE" --corpus "$HERE/corpus/$CORPUS.txt" --emit calibrate)"
repeat=$(( base_repeat * TARGET_MULT ))

mkdir -p "$OUTDIR"
"$BENCH" run_program "${PRESET[@]}" -p "$PI" -o "$OUTDIR/pilot" -q -s "$LIMIT" -- \
    "$HERE/solo/target/release/solo" \
    --program "$PROGRAM" \
    --case "$CASE" --corpus "$HERE/corpus/$CORPUS.txt" --repeat "$repeat" \
    > "$OUTDIR/pilot-output.txt" 2>&1
pilot_exit=$?

{
    echo "case=$CASE"
    echo "corpus=$CORPUS"
    echo "repeat=$repeat"
    echo "calibrated_base_repeat=$base_repeat"
    echo "interval_multiplier=$TARGET_MULT"
    echo "session_limit=$LIMIT"
    echo "pilot_exit=$pilot_exit"
    echo "arm=single"
    echo "program=$PROGRAM"
    echo "confidence_level=0.99"
    echo "ci_perc=0.04"
    echo "preset=strict"
} > "$OUTDIR/case.env"

[ -f "$OUTDIR/pilot/pi_results.csv" ] && cp "$OUTDIR/pilot/pi_results.csv" "$OUTDIR/summary.csv"
[ -f "$OUTDIR/pilot/readings.csv" ] && cp "$OUTDIR/pilot/readings.csv" "$OUTDIR/readings.csv"

if [ "$pilot_exit" -eq 13 ]; then
    echo "INCONCLUSIVE $CASE/$CORPUS: Pilot stopped on the session limit" >&2
    exit 13
fi
exit "$pilot_exit"
