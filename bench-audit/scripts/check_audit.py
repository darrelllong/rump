#!/usr/bin/env python3
"""Consistency gate for the audit's raw data.

Extends the existing bench consistency discipline to the requirements
BENCH-TASK.md states. Exit non-zero on any violation, so the report cannot be
assembled from data that fails its own checks.

Enforced:
  * finite positive times and ratios;
  * a Pilot-produced 99% CI no wider than 4% of the mean;
  * means lying inside their own raw observed range;
  * matching result digests (the ABBA wrapper refuses to emit a row otherwise,
    so a published cell with valid != 1 is a harness failure);
  * complete host/session coverage against the declared case list;
  * no result accepted from a stopped or failed Pilot session;
  * no directional cell in the null comparison.

Usage:
    check_audit.py <reduced-dir> [--cases bench-audit/cases.txt]
                                 [--sessions N] [--strict-coverage]
"""
import csv
import sys
from pathlib import Path

REQUIRED_CI_FRACTION = 0.04


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    reduced = Path(sys.argv[1])
    cases_file = None
    sessions_required = None
    strict_coverage = "--strict-coverage" in sys.argv
    if "--cases" in sys.argv:
        cases_file = Path(sys.argv[sys.argv.index("--cases") + 1])
    if "--sessions" in sys.argv:
        sessions_required = int(sys.argv[sys.argv.index("--sessions") + 1])

    table = reduced / "cases.csv"
    if not table.exists():
        sys.exit(f"missing {table}")
    rows = list(csv.DictReader(table.open()))
    if not rows:
        sys.exit("cases.csv is empty")

    problems = []
    published = 0

    for r in rows:
        tag = f"{r['session']}/{r['arm']}/{r['case']}/{r['corpus']}"
        verdict = r["verdict"]
        exit_code = int(r["pilot_exit"])

        # A stopped or failed session must not be published as anything but
        # inconclusive. This is the check that keeps a session limit a safety
        # stop rather than a result.
        if exit_code != 0 and verdict != "inconclusive":
            problems.append(f"{tag}: Pilot exit {exit_code} but verdict {verdict}")
        if exit_code != 0:
            continue

        def number(key):
            try:
                return float(r[key])
            except (ValueError, KeyError):
                return None

        mean, low, high = number("mean"), number("low"), number("high")
        base, cand = number("baseline_ns"), number("candidate_ns")

        # The null arm compares v0.3.0 against an independently built v0.3.0,
        # so its true ratio is one. A directional verdict there means the rig
        # can manufacture a difference where none exists, which puts every
        # paired result in question — including, and especially, the ones that
        # came out the way anyone expected. The reducer says so in prose; this
        # is the check that stops the report being assembled anyway.
        if r["arm"] == "null" and verdict in ("regression", "improvement"):
            problems.append(
                f"{tag}: null comparison is directional ({verdict}); the rig "
                f"reports a difference between two builds of the same revision"
            )

        if verdict == "inconclusive":
            continue
        published += 1

        for name, value in (("mean", mean), ("baseline_ns", base), ("candidate_ns", cand)):
            if value is None or not (value > 0) or value != value or value in (
                float("inf"), float("-inf")
            ):
                problems.append(f"{tag}: {name} is not finite and positive ({value})")

        if mean is not None and low is not None and high is not None:
            width = high - low
            allowed = REQUIRED_CI_FRACTION * mean
            if width <= 0:
                problems.append(f"{tag}: non-positive CI width {width}")
            elif width > allowed * (1 + 1e-9):
                problems.append(
                    f"{tag}: CI width {width:.6g} exceeds {REQUIRED_CI_FRACTION:.0%} "
                    f"of mean ({allowed:.6g})"
                )

        # The mean must lie inside the range its own readings allow. This is the
        # invariant that caught a corrupted reduction in this repository before:
        # a mean outside its sample's range cannot describe that sample.
        readings = Path(r["dir"]) / "readings.csv"
        observed = reading_range(readings, "candidate_over_baseline")
        if observed is None:
            problems.append(f"{tag}: no raw readings to bound the mean")
        elif mean is not None and not (observed[0] - 1e-12 <= mean <= observed[1] + 1e-12):
            problems.append(
                f"{tag}: mean {mean:.6g} outside its observed range "
                f"[{observed[0]:.6g}, {observed[1]:.6g}]"
            )

        # Every reading must have carried valid=1; the wrapper cannot emit a row
        # otherwise, so a zero here means the harness itself is broken.
        digest_ok = validity(readings)
        if digest_ok is False:
            problems.append(f"{tag}: a reading reported valid=0 (digest mismatch)")

    if cases_file and cases_file.exists():
        declared = set()
        for line in cases_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            case, corpus = line.split()
            declared.add((case, corpus))
        measured = {(r["case"], r["corpus"]) for r in rows if r["arm"] == "paired"}
        missing = declared - measured
        if missing:
            message = f"{len(missing)} declared cell(s) absent from the results"
            sample = ", ".join(f"{c}/{p}" for c, p in sorted(missing)[:5])
            message += f" (e.g. {sample})"
            (problems if strict_coverage else problems).append(message)

    if sessions_required:
        per_cell = {}
        for r in rows:
            if r["arm"] != "paired":
                continue
            per_cell.setdefault((r["case"], r["corpus"]), set()).add(r["session"])
        short = {k: v for k, v in per_cell.items() if len(v) < sessions_required}
        if short:
            problems.append(
                f"{len(short)} cell(s) have fewer than {sessions_required} sessions; "
                "a classification is accepted only when every session agrees"
            )
        # Disagreement across sessions is itself an inconclusive result.
        verdicts = {}
        for r in rows:
            if r["arm"] != "paired":
                continue
            verdicts.setdefault((r["case"], r["corpus"]), set()).add(r["verdict"])
        split = {k: v for k, v in verdicts.items() if len(v) > 1}
        if split:
            problems.append(
                f"{len(split)} cell(s) disagree between sessions and must be reported "
                "as inconclusive or host-sensitive"
            )

    print(f"checked {len(rows)} cells ({published} published)")
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        print(f"{len(problems)} consistency violation(s)")
        return 1
    print("all consistency checks pass")
    return 0


def reading_range(path, column):
    if not path.exists():
        return None
    lo = hi = None
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return None
        try:
            idx = header.index(column)
        except ValueError:
            return None
        for row in reader:
            if len(row) <= idx:
                continue
            try:
                v = float(row[idx])
            except ValueError:
                continue
            lo = v if lo is None else min(lo, v)
            hi = v if hi is None else max(hi, v)
    return None if lo is None else (lo, hi)


def validity(path):
    """False if any reading reported valid=0; None if the column is absent."""
    if not path.exists():
        return None
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None or "valid" not in header:
            return None
        idx = header.index("valid")
        for row in reader:
            if len(row) > idx:
                try:
                    if float(row[idx]) != 1.0:
                        return False
                except ValueError:
                    continue
    return True


if __name__ == "__main__":
    sys.exit(main())
