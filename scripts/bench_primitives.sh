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
#
# The sample size is left to pilot-bench's own convergence: it collects
# readings until the mean's CI meets the preset, which for a high-variance
# primitive naturally gathers more samples (and hence better tails). Forcing
# a minimum via -m interacts badly with its subsession sizing, so it is not
# used.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${PILOT_BENCH_CLI:-$HOME/pilot-bench/build/cli/bench}"
MP="${PILOT_MP_BIN:-$ROOT_DIR/target/release/pilot_mp}"
PILOT_PRESET="${PILOT_PRESET:-normal}"

for bin in "$BENCH" "$MP"; do
    [[ -x "$bin" ]] || { echo "error: not executable: $bin" >&2; exit 1; }
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Print one Markdown row `| op | mean_ms | min | p50 | p99 | max | ratio |`
# by driving pilot-bench and reducing its readings.csv.
measure() {
    local op=$1 out="$WORK/$op"
    rm -rf "$out"
    "$BENCH" run_program --preset "$PILOT_PRESET" -o "$out" \
        --pi "${op},ms/op,0,1,1" -- "$MP" "$op" >/dev/null 2>&1 || true
    python3 - "$op" "$out/readings.csv" <<'PY'
import sys, statistics
op, path = sys.argv[1], sys.argv[2]
xs = []
try:
    with open(path) as f:
        next(f)  # header: piid,round,readings
        for line in f:
            parts = line.split(",")
            if len(parts) >= 3 and parts[2].strip():
                xs.append(float(parts[2]))  # ms/op
except FileNotFoundError:
    pass
if not xs:
    print(f"| {op} | ? | ? | ? | ? | ? | ? |")
    sys.exit()
xs.sort()
ns = [x * 1e6 for x in xs]  # ms -> ns
q = lambda p: ns[min(len(ns) - 1, int((len(ns) - 1) * p))]
mean_ms = statistics.fmean(xs)
lo, hi = ns[0], ns[-1]
ratio = hi / lo if lo > 0 else float("inf")
print(f"| {op} | {mean_ms:.6g} | {lo:.1f} | {q(0.50):.1f} | {q(0.99):.1f} | {hi:.1f} | {ratio:.2f} |")
PY
}

echo "# rump primitive microbenchmarks"
echo
echo "Fresh random operands per pilot-bench trial; preset \`${PILOT_PRESET}\`."
echo "Mean is the average over random inputs; the ns/op order statistics are"
echo "the extrema of each variable-time primitive, taken from the same pool of"
echo "readings pilot-bench gathered to converge the mean (\`max/min\` = 1.0"
echo "means data-independent). Tail resolution tracks that sample size."
echo
echo "| Operation | mean ms/op | min ns | p50 ns | p99 ns | max ns | max/min |"
echo "|---|---:|---:|---:|---:|---:|---:|"
while read -r op; do
    [[ -z "$op" ]] && continue
    measure "$op"
done < <("$MP" --list)
