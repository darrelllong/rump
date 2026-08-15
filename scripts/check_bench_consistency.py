#!/usr/bin/env python3
"""Check that every benchmark row is arithmetically consistent with itself.

A row reports a mean alongside the order statistics of the same sample:

    | op_size | mean ms | ±CI | min ns | p50 ns | p99 ns | max ns | max/min | n |

Those numbers are not independent. Sort the sample; the reduction in
`bench_primitives.sh` reads the quantiles at

    i50 = int((n - 1) * 0.50)      p50 = x[i50]
    i99 = int((n - 1) * 0.99)      p99 = x[i99]

so `i50 + 1` readings are at most `p50`, the next `i99 - i50` are at most `p99`,
and the remaining `n - 1 - i99` are at most `max`.  Summing those caps bounds the
mean from above, and the symmetric argument (each reading at least the previous
quantile) bounds it from below:

    mean ≤ [(i50+1)·p50 + (i99-i50)·p99 + (n-1-i99)·max] / n
    mean ≥ [(i50+1)·min + (i99-i50)·p50 + (n-1-i99)·p99] / n

A mean outside that interval cannot have come from the sample its own quantiles
describe.  That is the signature of the defect this check exists to catch: the
harness once reported pilot-bench's `readings_mean`, a changepoint-truncated
"dominant segment" average that discards the heavy tail and so need not lie in
the sample's range at all (a 7168-bit sqrt_mod cell read 21.9 ms against its own
0.12 ms p99).  The reduction now reports the whole-sample mean, which satisfies
these bounds by construction, and this script keeps that true.

Why the bounds need `n`: at small `n` the quantile indices are coarse — with
four readings `p99` is simply the largest of four, and "the top 1%" is really the
top 25% — so the asymptotic form of the bound is wrong.  Rows that carry `n` are
checked exactly.  Rows without it predate the column and are checked against the
asymptotic form with a tolerance, and reported separately as unverifiable rather
than silently passed.

Exit status is 1 when a row that carries a reading count is inconsistent — the
current harness cannot produce one, so that is a live defect.  Rows predating the
count are reported but do not fail the run unless `--strict` is given, because
re-measuring them is a per-host task rather than a code fix.

    python3 scripts/check_bench_consistency.py [--strict] bench/*.md
"""

import re
import sys

# `| op_size | mean | ci | min | p50 | p99 | max | ratio |` with an optional
# trailing `| n |`.  A non-numeric mean (`insufficient-sample(n=4)`) is not a
# claim about the sample, so such rows are skipped rather than judged.
ROW = re.compile(
    r"\|\s*([a-z0-9_]+_\d+)\s*\|"
    r"\s*~?([0-9.eE+-]+)\s*\|"
    r"\s*[^|]*\|"
    r"\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
    r"\s*[0-9.]+\s*\|"
    r"(?:\s*(\d+)\s*\|)?"
)

# The reduction flags a cell approximate past this CI, and refuses to report a
# mean below this many readings; both are echoed here so the report can say why
# a row is weak rather than merely inconsistent.
MIN_READINGS = 30


def bounds(mean_ns, mn, p50, p99, mx, n):
    """Feasible interval for the sample mean given the order statistics."""
    if n is None:
        # Asymptotic form: half the mass at or below p50, 49% at or below p99,
        # 1% at or below max.  Correct only for large n, hence the caller's
        # tolerance.
        return (
            0.5 * mn + 0.49 * p50 + 0.01 * p99,
            0.5 * p50 + 0.49 * p99 + 0.01 * mx,
        )
    i50 = int((n - 1) * 0.50)
    i99 = int((n - 1) * 0.99)
    lo_count, mid_count, hi_count = i50 + 1, i99 - i50, n - 1 - i99
    lower = (lo_count * mn + mid_count * p50 + hi_count * p99) / n
    upper = (lo_count * p50 + mid_count * p99 + hi_count * mx) / n
    return lower, upper


def main(paths, strict=False):
    bad, weak, legacy = [], [], 0
    checked = 0
    for path in paths:
        try:
            lines = open(path).read().splitlines()
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for line in lines:
            m = ROW.match(line)
            if not m:
                continue
            op, mean_ms = m.group(1), float(m.group(2))
            mn, p50, p99, mx = (float(m.group(i)) for i in (3, 4, 5, 6))
            n = int(m.group(7)) if m.group(7) else None
            mean_ns = mean_ms * 1e6
            lower, upper = bounds(mean_ns, mn, p50, p99, mx, n)
            # 2% slack absorbs the printed figures' rounding (six significant
            # digits on the mean, one decimal on the quantiles); the asymptotic
            # branch needs more because its quantile positions are approximate.
            slack = 1.02 if n is not None else 1.10
            checked += 1
            if n is None:
                legacy += 1
            if mean_ns > upper * slack or mean_ns < lower / slack:
                bad.append((path, op, mean_ms, lower / 1e6, upper / 1e6, n))
            elif n is not None and n < MIN_READINGS:
                weak.append((path, op, n))

    print(f"checked {checked} rows ({legacy} without a reading count)")
    for path, op, n in weak:
        print(f"  weak    {path}:{op}: n={n} below the {MIN_READINGS}-reading floor")
    for path, op, mean_ms, lo, hi, n in bad:
        where = f"n={n}" if n is not None else "no n (legacy row)"
        print(
            f"  BAD     {path}:{op}: mean {mean_ms:.6g} ms outside the feasible "
            f"[{lo:.6g}, {hi:.6g}] ms implied by its own quantiles ({where})"
        )
    if not bad:
        print("all rows consistent")
        return 0
    fresh = [row for row in bad if row[5] is not None]
    print(f"\n{len(bad)} inconsistent row(s): a mean outside its own sample's range.")
    if fresh:
        # Data the current harness produced: the whole-sample mean satisfies the
        # bounds by construction, so an inconsistency here is a live defect.
        print(f"{len(fresh)} of them carry a reading count and so came from the "
              f"current harness — that is a defect, not stale data.")
        return 1
    # Only legacy rows, measured before the mean was fixed. They cannot be
    # verified exactly (no reading count) and re-measuring them is a per-host
    # task, so report without failing unless the caller demands strictness.
    print("All of them are legacy rows from the superseded harness; re-measure "
          "them on their hosts. Pass --strict to treat this as a failure.")
    return 1 if strict else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    strict = "--strict" in args
    paths = [a for a in args if a != "--strict"]
    if not paths:
        print("usage: check_bench_consistency.py [--strict] bench/*.md", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(paths, strict))
