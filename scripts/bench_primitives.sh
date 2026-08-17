#!/usr/bin/env bash
# Microbenchmark every rump primitive with pilot-bench over random operands,
# and emit one Markdown report giving BOTH the average and the extrema from
# the same run.
#
# Each `pilot_mp <op>` invocation draws a fresh random operand; pilot-bench
# repeats it until the mean's confidence interval converges. Its saved
# `readings.csv` is then a random sample of the primitive's timing, from which
# this script reports:
#   * the mean (average cost over random inputs), and
#   * min / p50 / p99 / max ns/op and the max/min spread (the data-dependent
#     variable-time behaviour).
#
# Env knobs:
#   PILOT_BENCH_CLI  pilot-bench binary (default $HOME/pilot-bench/build/cli/bench)
#   PILOT_MP_BIN     pilot_mp binary   (default target/release/pilot_mp)
#   PILOT_PRESET     pilot-bench preset (default normal)
#   PILOT_MP_SESSION session-limit seconds per op (default 30)
#   PILOT_MP_HEAVY_SESSION  session for sqrtmod/isprime (default 120)
#
# Sample size is left to pilot-bench's own convergence: it collects readings
# until the mean's CI meets the preset. The per-op session limit (-s) is an
# upper bound on collection time: pilot stops cleanly and keeps its readings
# (exit 13) rather than being killed. Forcing a minimum via -m interacts badly
# with subsession sizing, so the budget is time, not a reading count — and the
# reduction therefore publishes the reading count `n` with every row and refuses
# to report a mean below a floor (see MIN_READINGS below).
#
# That floor matters most for the heavy-tailed primitives (mod_sqrt,
# is_probable_prime), whose cost depends on the number-theoretic structure of
# the random input: a non-residue rejects after one exponentiation while a high
# 2-adic valuation pays the whole Tonelli–Shanks descent, so the sample variance
# never stabilizes. They get a larger budget, but budget alone is not enough at
# the wide sizes: each reading needs a fresh random operand, and generating a
# multi-kilobit random prime costs far more than the operation being timed, so a
# 180 s session at 7168 bits collected four readings. A mean of four draws from a
# bimodal cost is noise wearing a number's clothes; the floor rejects it rather
# than publishing it.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${PILOT_BENCH_CLI:-$HOME/pilot-bench/build/cli/bench}"
MP="${PILOT_MP_BIN:-$ROOT_DIR/target/release/pilot_mp}"
PILOT_PRESET="${PILOT_PRESET:-normal}"
SESSION="${PILOT_MP_SESSION:-30}"
HEAVY_SESSION="${PILOT_MP_HEAVY_SESSION:-120}"

for bin in "$BENCH" "$MP"; do
    [[ -x "$bin" ]] || { echo "error: not executable: $bin" >&2; exit 1; }
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Print one Markdown row
#   `| op | mean_ms | ±95% CI | min | p50 | p99 | max | ratio | n |`
# by driving pilot-bench for collection and reducing its raw `readings.csv`.
# pilot-bench only runs the program repeatedly until its own convergence or the
# session limit; every reported figure — mean, CI, and the order statistics —
# is computed here from the full readings sample, not read from pilot's
# analytics (whose changepoint-based mean is wrong for a heavy-tailed workload;
# see the reduction below).
measure() {
    local op=$1
    local out="$WORK/$op"
    local session="$SESSION"
    case "$op" in
        sqrtmod_* | isprime_*) session="$HEAVY_SESSION" ;;
    esac
    rm -rf "$out"
    # Exit 13 = hit the session limit; that is a clean stop with readings
    # written, so it is not an error here.
    "$BENCH" run_program --preset "$PILOT_PRESET" -s "$session" -o "$out" \
        --pi "${op},ms/op,0,1,1" -- "$MP" "$op" >/dev/null 2>&1 || true
    python3 - "$op" "$out/readings.csv" <<'PY'
import sys, statistics
op, readings_path = sys.argv[1], sys.argv[2]

# Order statistics come from the raw readings sample.
xs = []
try:
    with open(readings_path) as f:
        next(f)  # header: piid,round,readings
        for line in f:
            parts = line.split(",")
            if len(parts) >= 3 and parts[2].strip():
                xs.append(float(parts[2]))  # ms/op
except FileNotFoundError:
    pass

if not xs:
    print(f"| {op} | no-readings | ? | ? | ? | ? | ? | ? | 0 |")
    sys.exit()

xs.sort()
ns = [x * 1e6 for x in xs]  # ms -> ns
q = lambda p: ns[min(len(ns) - 1, int((len(ns) - 1) * p))] if ns else float("nan")
# The reported mean is the sample mean of the full readings — the unbiased
# estimator of the average cost over random inputs, and by construction a value
# in [min, max], consistent with the quantiles below. pilot-bench's
# readings_mean is deliberately NOT used: it is a changepoint-based "dominant
# segment" mean (workload.cc, refresh_analytical_result) built to strip warmup
# from a warmup-then-steady-state process. Our heavy-tailed ops are i.i.d.
# mixtures — runs of microsecond rejections punctured by rare enormous readings
# — which that detector misreads as regime changes and drops, producing a
# figure that need not even lie within the sample's own range (one session
# reported 21.9 ms against its 0.12 ms p99 and 130 ms max; another 2.15 us
# against a 194 ms p99). There is no warmup to eliminate here — every reading
# is a fresh random operand — so the whole-sample mean is the correct and only
# estimator.
mean_ms = statistics.fmean(xs)
lo = ns[0] if ns else float("nan")
hi = ns[-1] if ns else float("nan")
ratio = hi / lo if ns and lo > 0 else float("inf")

# 95% CI half-width as a percent of the mean, normal (IID) approximation:
# each reading is a fresh independent operand, so there is no autocorrelation
# for pilot's subsession CI to correct — the plain standard error is right. A
# heavy tail makes this wide, which is the honest statement that a finite
# sample pins the mean of a heavy-tailed cost only loosely (and is what flags
# such a cell approximate below).
if len(xs) > 2 and mean_ms > 0:
    ci_pct = 100.0 * (1.96 * statistics.stdev(xs) / len(xs) ** 0.5) / mean_ms
else:
    ci_pct = float("nan")

# A statistic is only as good as the sample behind it, and for the expensive
# widths the sample can be tiny: one reading costs a fresh random operand, and
# generating a multi-kilobit random prime takes far longer than the operation
# being timed (at 7168 bits a 180 s session collected FOUR readings — its
# "mean" of 202 ms was just the average of those four, two fast Jacobi exits and
# two full descents, and a different four would have said 0.13 ms or 400 ms).
# That is not a heavy-tailed measurement, it is no measurement. Below the floor
# the cell says so instead of printing an authoritative-looking number; the
# order statistics stay for transparency, and downstream mean parsers skip the
# row because the column does not hold a number.
MIN_READINGS = 30
mean_str = f"{mean_ms:.6g}"
if len(xs) < MIN_READINGS:
    print(
        f"WARNING {op}: only {len(xs)} readings (floor {MIN_READINGS}); "
        f"the mean of so few draws from a bimodal cost is not a measurement — "
        f"raise the session budget for this width",
        file=sys.stderr,
    )
    mean_str = f"insufficient-sample(n={len(xs)})"
elif ci_pct == ci_pct and ci_pct > 10.0:  # first clause: not NaN
    # The CI never tightened below 10%: a heavy-tailed op that hit the session
    # cap. The mean stands but downstream tables treat it as approximate.
    mean_str = "~" + mean_str
ci_str = f"{ci_pct:.2f}%" if ci_pct == ci_pct else "?"
# The reading count is published with every figure: a mean, a CI, and a p99 are
# only interpretable against the sample that produced them, and for the
# expensive widths that sample can be single digits.
print(
    f"| {op} | {mean_str} | {ci_str} | {lo:.1f} | "
    f"{q(0.50):.1f} | {q(0.99):.1f} | {hi:.1f} | {ratio:.2f} | {len(xs)} |"
)
PY
}

# Single-op mode: `bench_primitives.sh <op>` re-measures one operation and
# prints just its Markdown row (no header) — handy for patching a stray reading.
if [[ $# -ge 1 ]]; then
    measure "$1"
    exit 0
fi

echo "# rump primitive microbenchmarks"
echo
echo "Fresh random operands per pilot-bench trial; preset \`${PILOT_PRESET}\`,"
echo "session cap ${SESSION}s per op. Mean is the average over random inputs,"
echo "with pilot-bench's own 95% confidence interval on it (\`±95% CI\`, as a"
echo "percent of the mean — its subsession CI, which accounts for"
echo "autocorrelation). The ns/op order statistics are the extrema of each"
echo "variable-time primitive, from the readings pilot-bench gathered"
echo "(\`max/min\` = 1.0 means data-independent). A \`~\` on the mean flags a"
echo "heavy-tailed op that hit the session cap with its CI still above 10%"
echo "(its extrema are the meaningful result); tail resolution tracks the"
echo "sample size."
echo
echo "| Operation | mean ms/op | ±95% CI | min ns | p50 ns | p99 ns | max ns | max/min | n |"
echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
while read -r op; do
    [[ -z "$op" ]] && continue
    measure "$op"
done < <("$MP" --list)
