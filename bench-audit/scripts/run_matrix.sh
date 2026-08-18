#!/usr/bin/env bash
# Run a case list in parallel, one physical core per cell.
#
# Why parallelism is sound here, and where it stops being sound:
#
# Each Pilot reading is an ABBA quadruple, so baseline and candidate run inside
# the same reading under the same contention. Whatever memory-bandwidth or
# thermal pressure the neighbours create is charged to both revisions equally,
# and the ratio — the PI whose CI must converge — is robust to it. The absolute
# ns/op columns are not, and are recorded for diagnosis rather than published as
# host throughput figures.
#
# The limits that keep it honest:
#   * one cell per *physical* core, never an SMT sibling pair, since two
#     hyperthreads on one core interleave and neither timing means anything;
#   * memory bound to the core's own NUMA node, so a cell does not measure a
#     cross-socket hop that a serial run would not have;
#   * streaming cases get their own low concurrency, because their working sets
#     are sized to exceed cache on purpose and packing them would have them
#     evict each other rather than the cache they were built to stress.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASELIST="${1:?case list: lines of 'case corpus'}"
OUTROOT="${2:?output root}"
BASELINE="${3:?baseline executable}"
CANDIDATE="${4:?candidate executable}"

# Physical cores only. On a 2-socket SMT-2 machine the first half of the CPU
# list is the physical cores and the second half their siblings.
mapfile -t CORES < <(
    lscpu -p=CPU,CORE,SOCKET | grep -v '^#' \
        | awk -F, '!seen[$2","$3]++ {print $1}'
)
CONCURRENCY="${RUMP_CONCURRENCY:-${#CORES[@]}}"
(( CONCURRENCY > ${#CORES[@]} )) && CONCURRENCY=${#CORES[@]}

echo "physical cores available: ${#CORES[@]}; concurrency: $CONCURRENCY"
mkdir -p "$OUTROOT"
: > "$OUTROOT/index.csv"
echo "case,corpus,core,pilot_exit,outdir" >> "$OUTROOT/index.csv"

core_for() {
    # Distinct core per slot, wrapping if the list is longer than the pool.
    echo "${CORES[$(( $1 % ${#CORES[@]} ))]}"
}

node_of() {
    local cpu="$1"
    for n in /sys/devices/system/node/node[0-9]*; do
        if grep -qw "$cpu" <(tr ',' '\n' < "$n/cpulist" 2>/dev/null | awk -F- '{if(NF==2){for(i=$1;i<=$2;i++)print i}else print $1}'); then
            basename "$n" | tr -dc '0-9'
            return
        fi
    done
    echo 0
}

# Concurrency for the streaming cells. Their working sets are sized to exceed
# cache on purpose; packing thirty-two of them onto one machine would have them
# evict each other, so each would measure its neighbours rather than the memory
# behaviour it was built to stress.
STREAM_CONCURRENCY="${RUMP_STREAM_CONCURRENCY:-2}"

# Which per-cell driver to use. `run_case.sh` is the paired path;
# `run_case_single.sh` measures one revision alone, for the APIs the baseline
# does not have. The single-arm driver takes the program as its fourth
# argument and ignores the fifth, so both are called the same way.
CASE_SCRIPT="${RUMP_CASE_SCRIPT:-run_case.sh}"

slot=0

run_list() {
    # $1: concurrency, then the "case corpus" lines on stdin.
    local limit="$1"
    (( limit < 1 )) && limit=1
    while read -r case corpus; do
        [ -z "${case:-}" ] && continue

        core="$(core_for "$slot")"
        node="$(node_of "$core")"
        outdir="$OUTROOT/$case-$corpus"
        (
            numactl --membind="$node" -- \
            taskset -c "$core" \
            "$HERE/scripts/$CASE_SCRIPT" "$case" "$corpus" "$outdir" "$BASELINE" "$CANDIDATE" \
                > "$outdir.log" 2>&1
            echo "$case,$corpus,$core,$?,$outdir" >> "$OUTROOT/index.csv"
        ) &
        slot=$(( slot + 1 ))

        # Throttle to the requested concurrency.
        while (( $(jobs -rp | wc -l) >= limit )); do
            sleep 1
        done
    done
    wait
}

# Two passes, because the streaming cells need the machine to themselves in a
# way the rest do not. Comments are stripped once, here, so neither pass has to.
cases_only() { grep -vE '^\s*(#|$)' "$CASELIST"; }

echo "pass 1: non-streaming cells at concurrency $CONCURRENCY"
cases_only | grep -v '^stream_' | run_list "$CONCURRENCY"

echo "pass 2: streaming cells at concurrency $STREAM_CONCURRENCY"
cases_only | grep '^stream_' | run_list "$STREAM_CONCURRENCY"

echo "matrix complete: $(( $(wc -l < "$OUTROOT/index.csv") - 1 )) cells"
