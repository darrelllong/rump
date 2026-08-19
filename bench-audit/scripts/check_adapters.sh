#!/usr/bin/env bash
# The adapter source invariants, enforced rather than described.
#
# Two rules, both of which have been broken silently:
#
#   * adapter-v030-null must be byte-identical to adapter-v030. It exists to
#     be a second, independently linked build of the *same* program, so that a
#     null comparison measures the rig and nothing else. When it drifted, the
#     null arm reported a 7.3% difference between what were, by then, two
#     different programs -- and that reads as measurement bias rather than as
#     the stale checkout it was.
#
#   * adapter-v022 and adapter-v030 must share shared.rs byte for byte, and
#     main.rs apart from its one-line banner. Only cases.rs may differ, and
#     only where the 0.3.0 rename forces it.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail=0

for f in main.rs shared.rs cases.rs; do
    if ! diff -q "$HERE/adapter-v030/src/$f" "$HERE/adapter-v030-null/src/$f" >/dev/null; then
        echo "FAIL  adapter-v030-null/src/$f differs from adapter-v030"
        diff "$HERE/adapter-v030/src/$f" "$HERE/adapter-v030-null/src/$f" | head -5
        fail=1
    fi
done

if ! diff -q "$HERE/adapter-v022/src/shared.rs" "$HERE/adapter-v030/src/shared.rs" >/dev/null; then
    echo "FAIL  shared.rs differs between adapter-v022 and adapter-v030"
    fail=1
fi

if ! diff <(tail -n +2 "$HERE/adapter-v022/src/main.rs") \
          <(tail -n +2 "$HERE/adapter-v030/src/main.rs") >/dev/null; then
    echo "FAIL  main.rs differs between adapter-v022 and adapter-v030 below the banner"
    fail=1
fi

if [ $fail -eq 0 ]; then
    echo "adapter source invariants hold"
fi
exit $fail
