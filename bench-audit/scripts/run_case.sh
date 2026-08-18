#!/usr/bin/env bash
# One deciding cell: calibrate, run Pilot, capture every artifact, classify.
#
# Pilot is the sole statistical authority. This script reads its mean and CI and
# nothing else; it computes no interval of its own. A session that stops on the
# limit (exit 13) is recorded as inconclusive and its mean is not published.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${PILOT_BENCH_CLI:-$HOME/pilot-bench/build/cli/bench}"

CASE="${1:?case name}"
CORPUS="${2:?corpus name}"
OUTDIR="${3:?output dir}"
BASELINE="${4:-$HERE/adapter-v022/target/release/adapter-v022}"
CANDIDATE="${5:-$HERE/adapter-v030/target/release/adapter-v030}"

# The mandated statistics. `--ci-perc 0.04` with `--confidence-level 0.99` is a
# 99% interval no wider than 4% of the mean; `strict` sets the autocorrelation
# limit and the 200-subsession-sample floor.
PRESET=(--preset strict --confidence-level 0.99 --ci-perc 0.04)
PI='candidate_over_baseline,,0,0,1:baseline,ns/op,1,0,0:candidate,ns/op,2,0,0:valid,,3,2,0'
LIMIT="${RUMP_SESSION_LIMIT:-1800}"

# Calibration is outside the reported result. The task's floor is 20 ms per
# child; the multiplier lengthens the interval, which measurably reduces the
# autocorrelation Pilot has to merge away and so shortens the session.
TARGET_MULT="${RUMP_INTERVAL_MULT:-5}"
base_repeat="$("$CANDIDATE" --case "$CASE" --corpus "$HERE/corpus/$CORPUS.txt" --emit calibrate)"
repeat=$(( base_repeat * TARGET_MULT ))

mkdir -p "$OUTDIR"
"$BENCH" run_program "${PRESET[@]}" -p "$PI" -o "$OUTDIR/pilot" -q -s "$LIMIT" -- \
    "$HERE/abba/target/release/abba" \
    --baseline "$BASELINE" --candidate "$CANDIDATE" \
    --case "$CASE" --corpus "$HERE/corpus/$CORPUS.txt" --repeat "$repeat" \
    > "$OUTDIR/pilot-output.txt" 2>&1
pilot_exit=$?

# Record everything needed to judge the cell, including why it failed.
{
    echo "case=$CASE"
    echo "corpus=$CORPUS"
    echo "repeat=$repeat"
    echo "calibrated_base_repeat=$base_repeat"
    echo "interval_multiplier=$TARGET_MULT"
    echo "session_limit=$LIMIT"
    echo "pilot_exit=$pilot_exit"
    echo "baseline=$BASELINE"
    echo "candidate=$CANDIDATE"
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
