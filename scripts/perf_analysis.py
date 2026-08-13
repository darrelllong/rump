#!/usr/bin/env python3
"""Turn the per-host `primitives_<host>.md` tables into the scaling graphs and
fitted-complexity table for PERFORMANCE.md.

Usage:
  perf_analysis.py fit   host=path [host=path ...]        -> markdown fit table
  perf_analysis.py plot  family out.svg host=path [...]    -> log-log scaling SVG

Each table row is
`| <op>_<size> | mean | ±95% CI | min | p50 | p99 | max | max/min |`, mean in
ms/op (a leading '~' marks a heavy-tailed op whose mean did not converge). The
±95% CI cell is pilot-bench's confidence interval on the mean; it is skipped
here. Sizes are bit widths (integers) or field degrees (gf2m).
"""
import math
import re
import sys

# ─── Parsing ────────────────────────────────────────────────────────────────

ROW = re.compile(
    r"\|\s*([a-z0-9_]+)_(\d+)\s*\|"  # op_size
    r"\s*~?([0-9.eE+-]+)\s*\|"       # mean (ms)
    r"\s*[^|]*\|"                    # ±95% CI (skipped, keeps group numbers)
    r"\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)"
)


def parse(path):
    """path -> {op: {size: {mean_ns, min, p50, p99, max, ratio, approx}}}."""
    out = {}
    for line in open(path):
        m = ROW.match(line)
        if not m:
            continue
        op, size = m.group(1), int(m.group(2))
        approx = "~" in line.split("|")[2]
        out.setdefault(op, {})[size] = dict(
            mean_ns=float(m.group(3)) * 1e6,  # ms -> ns
            min=float(m.group(4)),
            p50=float(m.group(5)),
            p99=float(m.group(6)),
            max=float(m.group(7)),
            ratio=float(m.group(8)),
            approx=approx,
        )
    return out


def load(args):
    """host=path tokens -> {host: parsed}."""
    hosts = {}
    for a in args:
        host, path = a.split("=", 1)
        hosts[host] = parse(path)
    return hosts


# ─── Power-law fit: mean = c * n^alpha over the integer sizes ────────────────


def fit_power(sizes, means):
    """Least-squares log-log fit; returns (alpha, c) for mean ≈ c·n^alpha."""
    xs = [math.log(s) for s in sizes]
    ys = [math.log(v) for v in means]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    alpha = sxy / sxx
    try:
        c = math.exp(my - alpha * mx)
    except OverflowError:
        # A degenerate fit (isprime's heavy-tailed mean) can blow the intercept
        # past the float range; the exponent is what the table reports anyway.
        c = float("inf")
    return alpha, c


def fmt_time(ns):
    """Three significant figures in the natural unit: ns, µs, ms, or s."""
    for bound, divisor, unit in (
        (1e3, 1.0, "ns"),
        (1e6, 1e3, "µs"),
        (1e9, 1e6, "ms"),
        (float("inf"), 1e9, "s"),
    ):
        if ns < bound:
            return f"{ns / divisor:.3g} {unit}"
    return f"{ns:.3g} ns"


def size_label(s):
    """Bit widths in kilobit form once they reach 8 kbit."""
    return f"{s // 1024}kb" if s % 1024 == 0 and s >= 8192 else f"{s}b"


INT_FAMILIES = {
    "arithmetic": ["add", "sub", "mul", "sqr"],
    "division": ["divrem", "modulo", "modmul"],
    "montgomery": ["montmul", "montsqr", "montpow_e65537", "montpow_rand", "montsetup"],
    "number-theory": ["gcd", "gcdext", "modinv", "jacobi", "modpow"],
    "variable-time": ["sqrtmod", "isprime"],
}

# Theoretical complexity per method (n = bit width), for the fit table.
THEORY = {
    "add": "O(n)",
    "sub": "O(n)",
    "mul": "schoolbook → Karatsuba → Toom-3/4",
    "sqr": "schoolbook → Karatsuba → Toom-3/4",
    "divrem": "O(n²) Algorithm D",
    "modulo": "O(n²)",
    "modmul": "O(n²) mul + reduce",
    "montmul": "O(n²)",
    "montsqr": "O(n²)",
    "montpow_e65537": "O(n²) (17-bit exponent)",
    "montpow_rand": "O(e·n²), e = 256",
    "montsetup": "O(n²) (one division)",
    "modpow": "O(e·n²), e = 256",
    "gcd": "Lehmer O(n²) → Half-GCD O(M(n)·log n)",
    "gcdext": "Lehmer O(n²) → Half-GCD O(M(n)·log n)",
    "modinv": "Lehmer O(n²) → Half-GCD O(M(n)·log n)",
    "jacobi": "O(n²) binary",
    "sqrtmod": "O(n³) Tonelli–Shanks (input-dependent)",
    "isprime": "O(k·n²) Miller–Rabin (input-dependent)",
}


def fit_table(hosts):
    # Only the exponent alpha is reported. The fit's intercept c is deliberately
    # omitted: it carries units of ns/bit^alpha (not ns) and so is meaningless to
    # read as a time or to compare across rows with different alpha. Actual times
    # are in the cost tables.
    order = [op for fam in INT_FAMILIES.values() for op in fam]
    ncols = " | ".join(f"{h} α" for h in hosts)
    print(f"| Method | Complexity | {ncols} |")
    print("|---|---|" + "---|" * len(hosts))
    for op in order:
        cells = []
        for data in hosts.values():
            d = data.get(op, {})
            sizes = sorted(d)
            if len(sizes) >= 2:
                alpha, _c = fit_power(sizes, [d[s]["mean_ns"] for s in sizes])
                cells.append(f"{alpha:.2f}")
            else:
                cells.append("–")
        if all(c == "–" for c in cells):
            continue
        print(f"| `{op}` | {THEORY.get(op, '?')} | " + " | ".join(cells) + " |")


# ─── Scaling SVG (log-log ns/op vs bit width) ───────────────────────────────

W, H = 720, 460
ML, MR, MT, MB = 70, 150, 40, 55
PALETTE = ["#A64E28", "#3D648A", "#2E7D4F", "#A97416", "#6D4C91", "#8A8378"]


# Families that exist only as graphs; fit/means iterate INT_FAMILIES alone,
# so entries here never duplicate table rows.
PLOT_FAMILIES = {"gcd-at-scale": ["gcd", "gcdext", "modinv", "jacobi"]}


def scaling_svg(family, out, hosts):
    ops = INT_FAMILIES.get(family) or PLOT_FAMILIES[family]
    # Collect (op, host) series over the integer sizes present.
    series = []
    all_x, all_y = [], []
    for oi, op in enumerate(ops):
        for hi, (host, data) in enumerate(hosts.items()):
            d = data.get(op, {})
            pts = sorted((s, d[s]["mean_ns"]) for s in d)
            if len(pts) < 2:
                continue
            series.append((op, host, hi, pts))
            for s, v in pts:
                all_x.append(s)
                all_y.append(v)
    if not series:
        return
    lx = lambda v: math.log10(v)
    xmin, xmax = lx(min(all_x)), lx(max(all_x))
    ymin, ymax = lx(min(all_y)), lx(max(all_y))
    ypad = 0.05 * (ymax - ymin)
    ymin, ymax = ymin - ypad, ymax + ypad
    px = lambda v: ML + (lx(v) - xmin) / (xmax - xmin) * (W - ML - MR)
    py = lambda v: H - MB - (lx(v) - ymin) / (ymax - ymin) * (H - MT - MB)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#FBF9F5"/>')
    s.append(f'<text x="{ML}" y="24" font-size="15" font-weight="600" fill="#211E19">'
             f'rump — {family} scaling (mean ns/op, log–log)</text>')
    # x grid: one line per size present, k-labeled once sizes reach kilobits.
    for sz in sorted(set(all_x)):
        x = px(sz)
        lbl = f"{sz // 1024}k" if sz % 1024 == 0 and sz >= 8192 else str(sz)
        s.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{H-MB}" stroke="#E4DED4"/>')
        s.append(f'<text x="{x:.1f}" y="{H-MB+18}" font-size="11" fill="#6F675C" '
                 f'text-anchor="middle">{lbl}</text>')
    s.append(f'<text x="{(ML+W-MR)/2:.0f}" y="{H-14}" font-size="12" fill="#6F675C" '
             f'text-anchor="middle">bit width</text>')
    # y decade grid.
    d0, d1 = math.floor(ymin), math.ceil(ymax)
    for dec in range(d0, d1 + 1):
        if dec < ymin or dec > ymax:
            continue
        y = py(10 ** dec)
        s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{W-MR}" y2="{y:.1f}" stroke="#E4DED4"/>')
        lbl = f"{10**dec:g} ns" if dec < 3 else f"{10**(dec-3):g} µs" if dec < 6 else f"{10**(dec-6):g} ms"
        s.append(f'<text x="{ML-8}" y="{y+4:.1f}" font-size="11" fill="#6F675C" '
                 f'text-anchor="end">{lbl}</text>')
    # series.
    dash = {0: "", 1: "5,3", 2: "1,3"}  # per host
    for oi, op in enumerate(ops):
        col = PALETTE[oi % len(PALETTE)]
        for (o, host, hi, pts) in series:
            if o != op:
                continue
            path = " ".join(f"{'M' if i==0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(pts))
            da = dash.get(hi, "")
            s.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2" '
                     f'{f"stroke-dasharray=\"{da}\"" if da else ""}/>')
            for x, y in pts:
                s.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.5" fill="{col}"/>')
    # legend: ops by colour, hosts by dash.
    ly = MT + 6
    for oi, op in enumerate(ops):
        col = PALETTE[oi % len(PALETTE)]
        s.append(f'<line x1="{W-MR+8}" y1="{ly}" x2="{W-MR+26}" y2="{ly}" stroke="{col}" stroke-width="2"/>')
        s.append(f'<text x="{W-MR+30}" y="{ly+4}" font-size="10" fill="#211E19">{op}</text>')
        ly += 16
    ly += 6
    for hi, host in enumerate(hosts):
        da = dash.get(hi, "")
        s.append(f'<line x1="{W-MR+8}" y1="{ly}" x2="{W-MR+26}" y2="{ly}" stroke="#211E19" '
                 f'stroke-width="2" {f"stroke-dasharray=\"{da}\"" if da else ""}/>')
        s.append(f'<text x="{W-MR+30}" y="{ly+4}" font-size="10" fill="#6F675C">{host}</text>')
        ly += 15
    s.append("</svg>")
    open(out, "w").write("\n".join(s))
    print(f"wrote {out}")


GMP_SIZES = [256, 1024, 2048, 4096]
# Ops with a genuine GMP mpz counterpart (mirrors pilot_gmp.c).
GMP_OPS = ["add", "sub", "mul", "sqr", "divrem", "modulo", "modmul", "modpow",
           "gcd", "gcdext", "modinv", "jacobi", "isprime"]


def compare_table(rump_path, gmp_path):
    """rump vs GMP: mean ns/op and rump/gmp ratio, at every size the data holds."""
    rump, gmp = parse(rump_path), parse(gmp_path)
    ops = [op for op in GMP_OPS if op in rump or op in gmp]
    sizes = sorted(
        {s for op in ops for s in rump.get(op, {})}
        | {s for op in ops for s in gmp.get(op, {})}
    )
    hdr = " | ".join(size_label(s) for s in sizes)
    print(f"| Method | {hdr} |")
    print("|---|" + "---|" * len(sizes))
    for op in ops:
        cells = []
        for s in sizes:
            r = rump.get(op, {}).get(s)
            g = gmp.get(op, {}).get(s)
            if r and g and g["mean_ns"] > 0:
                cells.append(
                    f"{fmt_time(r['mean_ns'])} / {fmt_time(g['mean_ns'])} / "
                    f"{r['mean_ns'] / g['mean_ns']:.1f}×"
                )
            else:
                cells.append("–")
        print(f"| `{op}` | " + " | ".join(cells) + " |")


def compare_by_size(rump_path, gmp_path):
    """The same comparison transposed: one row per size, one column per op."""
    rump, gmp = parse(rump_path), parse(gmp_path)
    ops = [op for op in GMP_OPS if op in rump or op in gmp]
    sizes = sorted(
        {s for op in ops for s in rump.get(op, {})}
        | {s for op in ops for s in gmp.get(op, {})}
    )
    print("| bits | " + " | ".join(f"`{op}`" for op in ops) + " |")
    print("|---|" + "---|" * len(ops))
    for s_ in sizes:
        cells = []
        for op in ops:
            r = rump.get(op, {}).get(s_)
            g = gmp.get(op, {}).get(s_)
            if r and g and g["mean_ns"] > 0:
                cells.append(
                    f"{fmt_time(r['mean_ns'])} / {fmt_time(g['mean_ns'])} / "
                    f"{r['mean_ns'] / g['mean_ns']:.1f}×"
                )
            else:
                cells.append("–")
        print(f"| {size_label(s_)} | " + " | ".join(cells) + " |")


FAMILY_TITLES = {
    "arithmetic": "Arithmetic",
    "division": "Division & reduction",
    "montgomery": "Montgomery domain",
    "number-theory": "Number theory",
    "variable-time": "Variable-time (input-dependent)",
}
SIZES = [256, 1024, 2048, 4096]


def means_table(hosts):
    """Per-family mean cost: one row per method and host, one column per size."""

    def cell(d, s):
        v = d.get(s)
        if not v:
            return "–"
        return fmt_time(v["mean_ns"]) + ("~" if v["approx"] else "")

    for fam, ops in INT_FAMILIES.items():
        fam_sizes = sorted(
            {s for data in hosts.values() for op in ops for s in data.get(op, {})}
        )
        print(f"\n**{FAMILY_TITLES[fam]}** — mean per operation\n")
        print("| Method | host | " + " | ".join(size_label(s) for s in fam_sizes) + " |")
        print("|---|---|" + "---:|" * len(fam_sizes))
        for op in ops:
            first = True
            for host, data in hosts.items():
                d = data.get(op, {})
                if not d:
                    continue
                name = f"`{op}`" if first else ""
                first = False
                row = [cell(d, s) for s in fam_sizes]
                print(f"| {name} | {host} | " + " | ".join(row) + " |")


def extrema_table(arm):
    """The variable-time view. Operations whose spread exceeds an order of
    magnitude get a full per-size table, operation-major, sizes ascending;
    every other row is summarized in one generated sentence, so the summary
    cannot drift from the data."""
    by_op = {}
    for op, sizes in arm.items():
        for sz, d in sizes.items():
            by_op.setdefault(op, []).append((sz, d))
    heavy = sorted(
        (op for op in by_op if max(d["ratio"] for _, d in by_op[op]) >= 10),
        key=lambda op: -max(d["ratio"] for _, d in by_op[op]),
    )
    print("| Operation | size | min | p50 | p99 | max | max/min |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for op in heavy:
        first = True
        for sz, d in sorted(by_op[op]):
            name = f"`{op}`" if first else ""
            first = False
            r = d["ratio"]
            spread = f"{r:,.0f}" if r >= 1000 else f"{r:.1f}"
            print(
                f"| {name} | {sz} | {fmt_time(d['min'])} | {fmt_time(d['p50'])} "
                f"| {fmt_time(d['p99'])} | {fmt_time(d['max'])} | {spread} |"
            )
    rest = [
        d["ratio"] for op in by_op if op not in heavy for _, d in by_op[op]
    ]
    if rest:
        print(
            f"\nThe remaining {len(rest)} rows — every other operation and size —"
            f" span **{min(rest):.1f}–{max(rest):.1f}×**: their cost is set by"
            f" operand width, not operand value."
        )


def main():
    mode = sys.argv[1]
    if mode == "fit":  # fit  <label>=path ...
        fit_table(load(sys.argv[2:]))
    elif mode == "means":  # means  <label>=path ...
        means_table(load(sys.argv[2:]))
    elif mode == "extrema":  # extrema  <label>=path  (single host)
        extrema_table(next(iter(load(sys.argv[2:]).values())))
    elif mode == "compare":  # compare [--by-size]  rump.md  gmp.md
        if sys.argv[2] == "--by-size":
            compare_by_size(sys.argv[3], sys.argv[4])
        else:
            compare_table(sys.argv[2], sys.argv[3])
    elif mode == "plot":  # plot  <family>  out.svg  <label>=path ...
        family, out = sys.argv[2], sys.argv[3]
        scaling_svg(family, out, load(sys.argv[4:]))
    else:
        sys.exit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
