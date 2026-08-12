#!/usr/bin/env python3
"""Turn the per-host `primitives_<host>.md` tables into the scaling graphs and
fitted-complexity table for PERFORMANCE.md.

Usage:
  perf_analysis.py fit   host=path [host=path ...]        -> markdown fit table
  perf_analysis.py plot  family out.svg host=path [...]    -> log-log scaling SVG

Each table row is `| <op>_<size> | mean | min | p50 | p99 | max | max/min |`,
mean in ms/op (a leading '~' marks a heavy-tailed op whose mean did not
converge). Sizes are bit widths (integers) or field degrees (gf2m).
"""
import math
import re
import sys

# ─── Parsing ────────────────────────────────────────────────────────────────

ROW = re.compile(
    r"\|\s*([a-z0-9_]+)_(\d+)\s*\|\s*~?([0-9.eE+-]+)\s*\|"
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
    c = math.exp(my - alpha * mx)
    return alpha, c


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
    "mul": "O(n^1.585) Karatsuba / O(n²) schoolbook",
    "sqr": "O(n^1.585) / O(n²)",
    "divrem": "O(n²) Algorithm D",
    "modulo": "O(n²)",
    "modmul": "O(n²) mul + reduce",
    "montmul": "O(n²)",
    "montsqr": "O(n²)",
    "montpow_e65537": "O(n²) (17-bit exponent)",
    "montpow_rand": "O(e·n²), e = 256",
    "montsetup": "O(n²) (one division)",
    "modpow": "O(e·n²), e = 256",
    "gcd": "O(n²)",
    "gcdext": "O(n²)",
    "modinv": "O(n²) extended Euclid",
    "jacobi": "O(n²)",
    "sqrtmod": "O(n³) Tonelli–Shanks (input-dependent)",
    "isprime": "O(k·n²) Miller–Rabin (input-dependent)",
}


def fit_table(hosts):
    order = [op for fam in INT_FAMILIES.values() for op in fam]
    ncols = " | ".join(f"{h} α | {h} c(ns)" for h in hosts)
    print(f"| Method | Complexity | {ncols} |")
    print("|---|---|" + "---|---|" * len(hosts))
    for op in order:
        cells = []
        for data in hosts.values():
            d = data.get(op, {})
            sizes = sorted(s for s in d if s in (256, 1024, 2048, 4096))
            if len(sizes) >= 2:
                alpha, c = fit_power(sizes, [d[s]["mean_ns"] for s in sizes])
                cells.append(f"{alpha:.2f} | {c:.2g}")
            else:
                cells.append("– | –")
        print(f"| `{op}` | {THEORY.get(op, '?')} | " + " | ".join(cells) + " |")


# ─── Scaling SVG (log-log ns/op vs bit width) ───────────────────────────────

W, H = 720, 460
ML, MR, MT, MB = 70, 150, 40, 55
PALETTE = ["#A64E28", "#3D648A", "#2E7D4F", "#A97416", "#6D4C91", "#8A8378"]


def scaling_svg(family, out, hosts):
    ops = INT_FAMILIES[family]
    # Collect (op, host) series over the integer sizes present.
    series = []
    all_x, all_y = [], []
    for oi, op in enumerate(ops):
        for hi, (host, data) in enumerate(hosts.items()):
            d = data.get(op, {})
            pts = sorted((s, d[s]["mean_ns"]) for s in d if s in (256, 1024, 2048, 4096))
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
    # x grid: the four sizes.
    for sz in sorted(set(all_x)):
        x = px(sz)
        s.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{H-MB}" stroke="#E4DED4"/>')
        s.append(f'<text x="{x:.1f}" y="{H-MB+18}" font-size="11" fill="#6F675C" '
                 f'text-anchor="middle">{sz}</text>')
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
    """rump vs GMP: mean ns/op and rump/gmp ratio at each size."""
    rump, gmp = parse(rump_path), parse(gmp_path)
    hdr = " | ".join(f"{s}b" for s in GMP_SIZES)
    print(f"| Method | {hdr} (rump ns / gmp ns / ×) |")
    print("|---|" + "---|" * len(GMP_SIZES))
    for op in GMP_OPS:
        cells = []
        for s in GMP_SIZES:
            r = rump.get(op, {}).get(s)
            g = gmp.get(op, {}).get(s)
            if r and g and g["mean_ns"] > 0:
                cells.append(f"{r['mean_ns']:.0f} / {g['mean_ns']:.0f} / {r['mean_ns']/g['mean_ns']:.1f}×")
            else:
                cells.append("–")
        print(f"| `{op}` | " + " | ".join(cells) + " |")


FAMILY_TITLES = {
    "arithmetic": "Arithmetic",
    "division": "Division & reduction",
    "montgomery": "Montgomery domain",
    "number-theory": "Number theory",
    "variable-time": "Variable-time (input-dependent)",
}
SIZES = [256, 1024, 2048, 4096]


def means_table(arm, x86):
    """Per-family mean ns/op, ARM and x86 side by side."""
    for fam, ops in INT_FAMILIES.items():
        print(f"\n**{FAMILY_TITLES[fam]}** — mean ns/op\n")
        print("| Method | " + " | ".join(f"{s}b (M1)" for s in SIZES)
              + " | " + " | ".join(f"{s}b (EPYC)" for s in SIZES) + " |")
        print("|---|" + "---:|" * (2 * len(SIZES)))
        for op in ops:
            a = arm.get(op, {})
            x = x86.get(op, {})

            def cell(d, s):
                v = d.get(s)
                return f"{v['mean_ns']:.0f}" + ("~" if v and v["approx"] else "") if v else "–"

            row = [cell(a, s) for s in SIZES] + [cell(x, s) for s in SIZES]
            print(f"| `{op}` | " + " | ".join(row) + " |")


def extrema_table(arm):
    """Every op with its ARM extrema, ranked by spread — the variable-time view."""
    rows = []
    for op, sizes in arm.items():
        for s, d in sizes.items():
            rows.append((d["ratio"], f"{op}_{s}", d))
    rows.sort(reverse=True)
    print("| Operation | min ns | p50 ns | p99 ns | max ns | max/min |")
    print("|---|---:|---:|---:|---:|---:|")
    for _, name, d in rows:
        print(f"| `{name}` | {d['min']:.0f} | {d['p50']:.0f} | {d['p99']:.0f} "
              f"| {d['max']:.0f} | {d['ratio']:.1f} |")


def main():
    mode = sys.argv[1]
    if mode == "report":
        # report armrump=.. x86rump=.. armgmp=.. x86gmp=..
        f = dict(a.split("=", 1) for a in sys.argv[2:])
        arm, x86 = parse(f["armrump"]), parse(f["x86rump"])
        print("## Fitted complexity\n")
        fit_table({"M1": arm, "EPYC": x86})
        print("\n## Cost by method")
        means_table(arm, x86)
        print("\n## vs GMP — M1\n")
        compare_table(f["armrump"], f["armgmp"])
        print("\n## vs GMP — EPYC\n")
        compare_table(f["x86rump"], f["x86gmp"])
        print("\n## Extrema (M1)\n")
        extrema_table(arm)
        return
    if mode == "fit":
        fit_table(load(sys.argv[2:]))
    elif mode == "plot":
        family, out = sys.argv[2], sys.argv[3]
        scaling_svg(family, out, load(sys.argv[4:]))
    elif mode == "compare":
        compare_table(sys.argv[2], sys.argv[3])
    else:
        sys.exit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
