#!/usr/bin/env python3
"""Reduce a bidirectional null run into a per-workload resolution floor.

Reads the `forward` and `reverse` arms written by null_floor.sh and reports,
per cell, both ratios and their product. The product is the diagnostic:

    product ~ 1        the difference follows the binary image, so it is code
                       placement between two builds of identical source
    product ~ fwd^2    the difference follows position in the ABBA sequence,
                       which would mean the wrapper is not cancelling drift

The floor for a workload is the larger of |forward - 1| and |reverse - 1|. A
paired result inside that band is not attributable to the revision no matter
how tight its interval: a narrow interval around an artifact is still an
artifact.
"""
import csv
import sys
from pathlib import Path


def pi_row(case_dir):
    """Pilot's piid-0 row: readings, mean, CI."""
    summary = case_dir / "summary.csv"
    if not summary.exists():
        return None
    with summary.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("piid") == "0":
                try:
                    return (
                        int(float(row["readings_num"])),
                        float(row["readings_mean"]),
                        float(row["readings_subsession_ci"]),
                    )
                except (KeyError, TypeError, ValueError):
                    return None
    return None


def arm(root, name):
    out = {}
    d = root / name
    if not d.is_dir():
        return out
    for case_dir in sorted(p for p in d.iterdir() if p.is_dir()):
        got = pi_row(case_dir)
        if got:
            out[case_dir.name] = got
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    fwd, rev = arm(root, "forward"), arm(root, "reverse")
    shared = sorted(set(fwd) & set(rev))
    if not shared:
        sys.exit("no cell has both a forward and a reverse arm")

    print(f"{'cell':<38} {'forward':>9} {'reverse':>9} {'product':>9} {'floor':>8}")
    floors = {}
    for cell in shared:
        fn, fm, fci = fwd[cell]
        rn, rm, rci = rev[cell]
        product = fm * rm
        floor = max(abs(fm - 1.0), abs(rm - 1.0))
        floors[cell] = floor
        print(f"{cell:<38} {fm:>9.5f} {rm:>9.5f} {product:>9.5f} {100*floor:>7.2f}%")

    worst = max(floors.values())
    median = sorted(floors.values())[len(floors) // 2]
    print()
    print(f"cells: {len(shared)}   median floor: {100*median:.2f}%   worst: {100*worst:.2f}%")
    print()
    print("A paired result inside its cell's floor is not attributable to the")
    print("revision. Where a cell has no floor of its own, the worst measured")
    print("floor is the honest bound to apply.")


if __name__ == "__main__":
    main()
