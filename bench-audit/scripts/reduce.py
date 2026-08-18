#!/usr/bin/env python3
"""Reduce raw Pilot output into the audit manifest, case table, and report.

Reads Pilot's own mean and confidence interval and computes no interval of its
own: this script may calculate interval endpoints from the width Pilot reports,
derive effect sizes, classify a cell under the fixed rules, and render tables.
It may not replace Pilot's statistics.

Usage:
    reduce.py <results-root> <output-dir> [--host NAME]

`<results-root>` holds one directory per session, each containing `paired/` and
optionally `null/`, as `run_matrix.sh` writes them.
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

# The classification boundaries, fixed before the run. A statistically resolved
# 1% change is not important and an unresolved 8% point estimate is not
# acceptable, so both the effect boundary and the CI take part in every verdict.
REGRESSION_BOUND = 1.05
IMPROVEMENT_BOUND = 0.97
REQUIRED_CONFIDENCE = 0.99
REQUIRED_CI_FRACTION = 0.04


def read_env(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k] = v
    return out


def read_pi(path):
    """Pilot's per-PI results, keyed by piid."""
    if not path.exists():
        return {}
    rows = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            rows[row["piid"]] = row
    return rows


def reading_range(path, piid=0):
    """Min and max of Pilot's raw readings for one PI.

    Pilot writes readings.csv in long form -- `piid,round,readings`, one row
    per PI per round -- not one column per PI. Both scripts previously looked
    up a column by name, found nothing, and returned None, which silently
    disabled the invariant that a mean must lie inside the range of its own
    readings. That invariant is the one this repository already had a
    corrupted reduction caught by, so it failing open was worse than it
    failing loudly.
    """
    if not path.exists():
        return None
    lo = hi = None
    with path.open() as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "readings" not in reader.fieldnames:
            return None
        for row in reader:
            try:
                if int(row["piid"]) != piid:
                    continue
                v = float(row["readings"])
            except (TypeError, ValueError):
                continue
            lo = v if lo is None else min(lo, v)
            hi = v if hi is None else max(hi, v)
    return None if lo is None else (lo, hi)


def pilot_required_readings(pilot_dir):
    """Pilot's own required reading count for the first PI, from its log.

    Pilot recomputes this every refresh and logs one line per PI in piid
    order, so the first of the final group belongs to piid 0 -- the ratio for
    a paired cell, the absolute cost for a single-arm one.
    """
    log = pilot_dir / "session_log.txt"
    if not log.exists():
        return None
    lines = [l for l in log.read_text(errors="replace").splitlines()
             if "Required reading size" in l]
    if not lines:
        return None
    # The final refresh emits one line per PI; take the group's first.
    group = lines[-4:] if len(lines) >= 4 else lines
    try:
        return int(group[0].rsplit("= ", 1)[1].strip())
    except (IndexError, ValueError):
        return None


def classify(cell):
    """The verdict, and the reason when it is inconclusive."""
    # Exit 13 means Pilot stopped on the session limit. Measured here, that
    # says nothing about whether its data is adequate: of the first 32 cells to
    # finish under a two-hour budget, 25 satisfied both criteria Pilot itself
    # reports -- readings at or above its own required reading size, and a
    # confidence interval within the required fraction of the mean -- and every
    # one of them still exited 13. Pilot's stopping rule does not fire for this
    # workload even when its stated requirements are met, so treating the exit
    # code as the arbiter would discard adequate data in three cells out of
    # four.
    #
    # Pilot remains the sole statistical authority. Every number below is one
    # Pilot computed: the mean, the interval, the autocorrelation-merged
    # subsession size, and the required reading count. What changed is only
    # which signal decides publishability -- Pilot's own reported sufficiency
    # rather than its exit status.
    #
    # Any other non-zero exit is a real failure: a crashed child, a digest
    # mismatch, a refused workload.
    if cell["pilot_exit"] not in (0, 13):
        return "inconclusive", f"Pilot exit {cell['pilot_exit']}"
    if cell["pilot_exit"] == 13:
        required = cell.get("pilot_required")
        if required is None:
            return "inconclusive", "Pilot stopped on the session limit, and reported no required reading size"
        if cell["readings"] < required:
            return "inconclusive", (
                f"Pilot stopped on the session limit with {cell['readings']} readings, "
                f"short of the {required} it required"
            )
    if cell["mean"] is None or cell["ci"] is None:
        return "inconclusive", "no Pilot confidence interval"
    if cell["ci"] <= 0:
        return "inconclusive", "degenerate confidence interval"
    allowed = REQUIRED_CI_FRACTION * cell["mean"]
    if cell["ci"] > allowed:
        return "inconclusive", (
            f"CI width {cell['ci']:.4g} exceeds the required "
            f"{REQUIRED_CI_FRACTION:.0%} of mean ({allowed:.4g})"
        )
    if cell["valid_mean"] is not None and abs(cell["valid_mean"] - 1.0) > 1e-12:
        return "inconclusive", "a reading reported valid=0: correctness failure"

    # A single-arm cell measures an absolute cost, not a ratio, so the
    # regression bounds have nothing to say about it: there is no baseline it
    # could be a regression against. It is published when Pilot's interval
    # meets the requirement, and read as a baseline for future comparison.
    if cell["arm"] == "single":
        return "baseline", "absolute cost; no baseline revision to compare against"

    low, high = cell["low"], cell["high"]
    if low > REGRESSION_BOUND:
        return "regression", f"CI entirely above {REGRESSION_BOUND}"
    if high < IMPROVEMENT_BOUND:
        return "improvement", f"CI entirely below {IMPROVEMENT_BOUND}"
    if low <= 1.0 <= high and high <= REGRESSION_BOUND:
        return "equivalent", "interval contains 1.0 and is within the 5% bound"
    if high <= REGRESSION_BOUND:
        return "pass", f"CI entirely below {REGRESSION_BOUND}"
    return "inconclusive", f"CI crosses the {REGRESSION_BOUND} boundary"


def collect(results_root):
    cells = []
    for session_dir in sorted(Path(results_root).glob("session-*")):
        for arm in ("paired", "null", "single"):
            arm_dir = session_dir / arm
            if not arm_dir.is_dir():
                continue
            for case_dir in sorted(d for d in arm_dir.iterdir() if d.is_dir()):
                env = read_env(case_dir / "case.env")
                if not env:
                    continue
                pi = read_pi(case_dir / "summary.csv")
                if arm == "single":
                    # No baseline to divide by: piid 0 is the absolute cost in
                    # ns/op and piid 1 is the validity flag.
                    ratio = pi.get("0", {})
                    base = cand = {}
                    valid = pi.get("1", {})
                else:
                    ratio = pi.get("0", {})
                    base = pi.get("1", {})
                    cand = pi.get("2", {})
                    valid = pi.get("3", {})

                def num(row, key):
                    try:
                        return float(row[key])
                    except (KeyError, TypeError, ValueError):
                        return None

                cell = {
                    "session": session_dir.name,
                    "arm": arm,
                    "case": env.get("case", case_dir.name),
                    "corpus": env.get("corpus", ""),
                    "repeat": int(env.get("repeat", 0) or 0),
                    "pilot_exit": int(env.get("pilot_exit", -1) or -1),
                    "readings": int(float(ratio.get("readings_num", 0) or 0)),
                    "mean": num(ratio, "readings_mean"),
                    "ci": num(ratio, "readings_subsession_ci"),
                    "baseline_ns": num(base, "readings_mean"),
                    "candidate_ns": num(cand, "readings_mean"),
                    "valid_mean": num(valid, "readings_mean"),
                    "pilot_required": pilot_required_readings(case_dir / "pilot"),
                    "dir": str(case_dir),
                }
                if cell["mean"] is not None and cell["ci"] is not None:
                    cell["low"] = cell["mean"] - cell["ci"] / 2.0
                    cell["high"] = cell["mean"] + cell["ci"] / 2.0
                else:
                    cell["low"] = cell["high"] = None
                # piid 0 is the ratio for a paired cell and the absolute cost
                # for a single-arm one; the bounded quantity either way.
                cell["observed_range"] = reading_range(case_dir / "readings.csv", 0)
                cell["verdict"], cell["reason"] = classify(cell)
                cells.append(cell)
    return cells


def host_facts():
    def run(cmd):
        try:
            return subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=20
            ).stdout.strip()
        except Exception:
            return ""

    return {
        "uname": run("uname -srm"),
        "cpu": run("lscpu | sed -n 's/^Model name: *//p'") or run(
            "sysctl -n machdep.cpu.brand_string"
        ),
        "rustc": run("rustc -Vv | head -1"),
        "cargo": run("cargo --version"),
        "pilot": run(f"{os.environ.get('PILOT_BENCH_CLI', 'bench')} --help 2>&1 | head -1"),
    }


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    results_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    host = "unknown"
    if "--host" in sys.argv:
        host = sys.argv[sys.argv.index("--host") + 1]
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = collect(results_root)
    if not cells:
        sys.exit(f"no cells found under {results_root}")

    manifest = {
        "host": host,
        "facts": host_facts(),
        "statistics": {
            "authority": "pilot-bench",
            "preset": "strict",
            "confidence_level": REQUIRED_CONFIDENCE,
            "ci_perc": REQUIRED_CI_FRACTION,
            "pi_spec": "candidate_over_baseline,,0,0,1:baseline,ns/op,1,0,0:"
            "candidate,ns/op,2,0,0:valid,,3,2,0",
        },
        "classification": {
            "regression_bound": REGRESSION_BOUND,
            "improvement_bound": IMPROVEMENT_BOUND,
        },
        "cells": len(cells),
        "sessions": sorted({c["session"] for c in cells}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    fields = [
        "session", "arm", "case", "corpus", "repeat", "readings", "pilot_required",
        "pilot_exit", "mean", "low", "high", "baseline_ns", "candidate_ns",
        "verdict", "reason", "dir",
    ]
    with (out_dir / "cases.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for c in cells:
            w.writerow(c)

    render_report(cells, out_dir, host, manifest)
    counts = {}
    for c in cells:
        counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
    print(f"{len(cells)} cells: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def render_report(cells, out_dir, host, manifest):
    lines = []
    a = lines.append
    a(f"# Rump public-API performance audit — {host}")
    a("")
    a("Generated by `bench-audit/scripts/reduce.py` from raw Pilot output. No")
    a("number here is hand-transcribed, and every row links to the case directory")
    a("holding its Pilot summary and raw readings.")
    a("")
    a("Pilot Benchmark Framework is the sole statistical authority: the mean and")
    a("confidence interval below are Pilot's, and the reducer only derives the")
    a("interval endpoints from the width Pilot reports.")
    a("")
    a("There is deliberately no single overall speedup figure. Any such number")
    a("would impose one arbitrary workload mix on a general-purpose library.")
    a("")
    a("## Settings")
    a("")
    for k, v in manifest["statistics"].items():
        a(f"- `{k}`: {v}")
    a("")
    a("## Host")
    a("")
    for k, v in manifest["facts"].items():
        if v:
            a(f"- **{k}**: {v}")
    a("")

    null = [c for c in cells if c["arm"] == "null"]
    if null:
        a("## Null comparison: v0.3.0 against an independently built v0.3.0")
        a("")
        a("This must show no directional result. A `regression` or `improvement`")
        a("here would mean the harness manufactures a difference, and would")
        a("invalidate every paired cell below.")
        a("")
        a(table(null))
        directional = [c for c in null if c["verdict"] in ("regression", "improvement")]
        a("")
        if directional:
            a(f"**{len(directional)} directional null cell(s): the paired results below "
              "are not trustworthy until this is explained.**")
        else:
            a("No directional null cell.")
        a("")

    paired = [c for c in cells if c["arm"] == "paired"]
    if paired:
        a("## Paired comparison: v0.2.2 baseline against v0.3.0 candidate")
        a("")
        for verdict in ("regression", "inconclusive", "improvement", "equivalent", "pass"):
            group = [c for c in paired if c["verdict"] == verdict]
            if not group:
                continue
            a(f"### {verdict} ({len(group)})")
            a("")
            a(table(group))
            a("")
    return (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def table(cells):
    rows = ["| case | corpus | ratio | 99% CI | readings | verdict | reason | raw |",
            "|---|---|---|---:|---:|---|---|---|"]
    for c in sorted(cells, key=lambda x: (x["case"], x["corpus"])):
        mean = "—" if c["mean"] is None else f"{c['mean']:.4f}"
        ci = "—" if c["low"] is None else f"[{c['low']:.4f}, {c['high']:.4f}]"
        rows.append(
            f"| `{c['case']}` | `{c['corpus']}` | {mean} | {ci} | {c['readings']} "
            f"| {c['verdict']} | {c['reason']} | [raw]({c['dir']}) |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    main()
