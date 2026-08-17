#!/usr/bin/env python3
"""Extract every lstlisting block from manual.tex into one Rust program.

The LaTeX manual's code listings are drawn from MANUAL.md, whose blocks are
test-pinned; this extractor is the corresponding guard for manual.tex
itself. It emits a single `main.rs` to stdout that compiles against the
crate and executes every listing's assertions. Two blocks are skipped by
content, not by count: the Cargo.toml snippet and the `pub trait RandomSource`
declaration excerpt, neither of which is a runnable statement sequence.

No per-block scoping is applied: later blocks legitimately reuse bindings
from earlier ones in the same section (the GF(2^m) walkthrough), and Rust's
shadowing handles the rest.

Usage (from the repository root; `check_manual_tex.sh` wraps this):
    python3 scripts/extract_manual_examples.py manual.tex > /tmp/main.rs
"""
import re
import sys

tex = open(sys.argv[1]).read()
blocks = re.findall(r"\\begin\{lstlisting\}\n(.*?)\\end\{lstlisting\}", tex, re.S)

imports = None
body = []
skipped = 0
for b in blocks:
    if "rust-mp = " in b:  # the Cargo.toml snippet
        skipped += 1
        continue
    if b.strip().startswith("pub trait RandomSource"):  # the trait declaration excerpt
        skipped += 1
        continue
    if b.strip().startswith("use rump::"):
        imports = b
        continue
    body.append(b)

print(f"blocks: {len(blocks)}, skipped: {skipped}, code: {len(body)}", file=sys.stderr)
out = imports + "\n#[allow(unused_variables, unused_mut)]\nfn main() {\n"
for i, b in enumerate(body):
    out += f"    // ---- block {i} ----\n"
    for line in b.rstrip().splitlines():
        out += "    " + line + "\n"
print(out + "}\n")
