# Feature modes, prepared contexts and scratch reuse on dennard

What SUGGESTIONS §4 asks to measure separately: the `wipe` feature against
the default build, a `MontgomeryScratch` kept across calls against one per
call, and a prepared context's setup against its use.

- Host: dennard, AMD EPYC 7452, Ubuntu 24.04.4, kernel 6.8.0-136; each
  benchmark pinned to core 7 (`taskset -c 7`) on an otherwise idle host.
- Toolchain: rustc 1.95.0; `pilot_mp` built `--release`, and again
  `--release --features wipe`.
- Source: rump 4218630 with the `barrettsetup`, `barrettmul`,
  `montmul_scratch` and `montsqr_scratch` rows added to `pilot_mp`.
- Measurement: `scripts/bench_primitives.sh <op>_<bits>` per row (pilot-bench
  0f3cb4f, preset `normal`, 30 s sessions, 120 s for `isprime_true`); columns
  as in the primitives tables.

## wipe against default

| Operation | default mean ms | wipe mean ms | wipe / default |
|---|---:|---:|---:|
| mul_256 | 5.60574e-05 | 6.91284e-05 | 1.233 |
| sqr_256 | 5.9618e-05 | 6.3478e-05 | 1.065 |
| divrem_256 | 0.000166131 | 0.000178374 | 1.074 |
| modmul_256 | 0.000295839 | 0.000318725 | 1.077 |
| barrettsetup_256 | 0.000304226 | 0.000346224 | 1.138 |
| barrettmul_256 | 0.000342161 | 0.000413811 | 1.209 |
| montsetup_256 | 0.000451371 | 0.00047626 | 1.055 |
| montmul_256 | 0.000131605 | 0.000155466 | 1.181 |
| montmul_scratch_256 | 0.000113383 | 0.000121778 | 1.074 |
| montsqr_256 | 0.000127605 | 0.0001459 | 1.143 |
| montsqr_scratch_256 | 0.000106498 | 0.000118805 | 1.116 |
| montpow_rand_256 | 0.0233651 | 0.0231419 | 0.990 |
| modpow_256 | 0.0238273 | 0.0237009 | 0.995 |
| gcd_256 | 0.00600663 | 0.00584849 | 0.974 |
| modinv_256 | 0.00961029 | 0.0100702 | 1.048 |
| isprime_true_256 | 0.314019 | 0.311567 | 0.992 |
| mul_2048 | 0.00206127 | 0.00210777 | 1.023 |
| sqr_2048 | 0.00143331 | 0.00149513 | 1.043 |
| divrem_2048 | 0.00104255 | 0.00107602 | 1.032 |
| modmul_2048 | 0.0051728 | 0.00526901 | 1.019 |
| barrettsetup_2048 | 0.00315397 | 0.00321156 | 1.018 |
| barrettmul_2048 | 0.00542519 | 0.00562125 | 1.036 |
| montsetup_2048 | 0.00464781 | 0.00466398 | 1.003 |
| montmul_2048 | 0.00267005 | 0.00278732 | 1.044 |
| montmul_scratch_2048 | 0.00260876 | 0.0026871 | 1.030 |
| montsqr_2048 | 0.00221427 | 0.00231659 | 1.046 |
| montsqr_scratch_2048 | 0.0021803 | 0.00224862 | 1.031 |
| montpow_rand_2048 | 0.723633 | 0.716757 | 0.990 |
| modpow_2048 | 0.7311 | 0.718968 | 0.983 |
| gcd_2048 | 0.0577335 | 0.0570647 | 0.988 |
| modinv_2048 | 0.0737538 | 0.0757583 | 1.027 |
| isprime_true_2048 | 67.9633 | 67.0663 | 0.987 |

The limb wipe costs 5–23% on 256-bit primitives, where a drop's volatile
writes are a visible share of the work, 0–5% at 2048 bits, and nothing
measurable on exponentiation, gcd or primality testing. A consumer that
enables `cryptography` gets `wipe` through Cargo's feature unification and
pays these figures.

## Reuse

- `montmul_scratch` against `montmul`: a kept scratch saves 14% at 256 bits
  and 2% at 2048 bits; `montsqr_scratch` 17% and 2%.
- `montpow_rand` (context prepared) against `modpow` (context built per
  call): equal within their intervals at both widths; setup is about 2% of a
  256-bit-exponent exponentiation at 256 bits and under 1% at 2048.
- `barrettmul` (context prepared, operands already reduced) against `modmul`
  (one product and one division): the prepared context is slower on this
  host, 0.342 µs against 0.278 µs at 256 bits and 5.43 µs against 5.17 µs at
  2048 bits, so for these odd moduli a Barrett context does not pay here at
  these widths; `barrettsetup` is one more division on top. The
  `barrettmul` rows were measured in a second session (load average 25–60 on
  other cores) after the first used unreduced operands, which charged it two
  extra reductions; `modmul` measured in that session within 7% of the first.

## default

| Operation | mean ms/op | ±95% CI | min ns | p50 ns | p99 ns | max ns | max/min | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mul_256 | 5.60574e-05 | 0.38% | 54.6 | 56.0 | 57.6 | 57.7 | 1.06 | 52 |
| sqr_256 | 5.9618e-05 | 0.46% | 57.4 | 59.5 | 62.0 | 63.8 | 1.11 | 53 |
| divrem_256 | 0.000166131 | 0.29% | 159.7 | 166.5 | 171.0 | 172.6 | 1.08 | 110 |
| modmul_256 | 0.000295839 | 0.30% | 280.8 | 295.4 | 307.1 | 309.1 | 1.10 | 170 |
| barrettsetup_256 | 0.000304226 | 0.40% | 295.2 | 304.0 | 311.2 | 315.3 | 1.07 | 50 |
| barrettmul_256 | 0.000342161 | 0.88% | 326.1 | 339.7 | 370.6 | 379.8 | 1.16 | 50 |
| montsetup_256 | 0.000451371 | 0.29% | 436.3 | 451.7 | 463.6 | 466.9 | 1.07 | 110 |
| montmul_256 | 0.000131605 | 0.49% | 127.5 | 131.2 | 136.6 | 136.9 | 1.07 | 50 |
| montmul_scratch_256 | 0.000113383 | 0.62% | 108.7 | 113.2 | 119.4 | 122.9 | 1.13 | 50 |
| montsqr_256 | 0.000127605 | 0.43% | 123.6 | 127.5 | 132.1 | 132.5 | 1.07 | 56 |
| montsqr_scratch_256 | 0.000106498 | 0.65% | 102.4 | 106.3 | 110.7 | 111.5 | 1.09 | 50 |
| montpow_rand_256 | 0.0233651 | 0.22% | 22412.3 | 23398.5 | 23845.1 | 23977.6 | 1.07 | 140 |
| modpow_256 | 0.0238273 | 0.30% | 22628.0 | 23870.6 | 24407.8 | 25249.2 | 1.12 | 113 |
| gcd_256 | 0.00600663 | 1.65% | 4962.9 | 5976.6 | 6956.8 | 6963.2 | 1.40 | 50 |
| modinv_256 | 0.00961029 | 1.57% | 8227.9 | 9566.3 | 11189.5 | 12323.9 | 1.50 | 81 |
| isprime_true_256 | 0.314019 | 0.38% | 298473.0 | 314080.0 | 324368.0 | 328558.0 | 1.10 | 80 |
| mul_2048 | 0.00206127 | 0.83% | 1941.8 | 2059.7 | 2180.2 | 2217.5 | 1.14 | 50 |
| sqr_2048 | 0.00143331 | 0.72% | 1274.8 | 1454.8 | 1483.4 | 1490.3 | 1.17 | 110 |
| divrem_2048 | 0.00104255 | 0.50% | 999.2 | 1043.0 | 1076.3 | 1081.2 | 1.08 | 50 |
| modmul_2048 | 0.0051728 | 0.68% | 4870.7 | 5171.2 | 5383.4 | 5421.3 | 1.11 | 50 |
| barrettsetup_2048 | 0.00315397 | 0.44% | 3029.4 | 3153.8 | 3241.0 | 3244.8 | 1.07 | 55 |
| barrettmul_2048 | 0.00542519 | 0.37% | 5165.2 | 5424.7 | 5627.3 | 5694.0 | 1.10 | 110 |
| montsetup_2048 | 0.00464781 | 0.41% | 4412.3 | 4650.2 | 4806.8 | 4818.6 | 1.09 | 80 |
| montmul_2048 | 0.00267005 | 0.38% | 2591.1 | 2675.2 | 2733.5 | 2772.6 | 1.07 | 50 |
| montmul_scratch_2048 | 0.00260876 | 0.35% | 2546.2 | 2606.4 | 2661.0 | 2692.1 | 1.06 | 51 |
| montsqr_2048 | 0.00221427 | 0.28% | 2148.8 | 2214.1 | 2277.0 | 2283.5 | 1.06 | 110 |
| montsqr_scratch_2048 | 0.0021803 | 0.45% | 2114.7 | 2175.8 | 2268.8 | 2271.0 | 1.07 | 50 |
| montpow_rand_2048 | 0.723633 | 0.27% | 705705.0 | 724393.0 | 741057.0 | 743868.0 | 1.05 | 80 |
| modpow_2048 | 0.7311 | 0.20% | 708573.0 | 731847.0 | 746410.0 | 759845.0 | 1.07 | 140 |
| gcd_2048 | 0.0577335 | 0.62% | 55315.7 | 57531.1 | 59556.5 | 59778.3 | 1.08 | 50 |
| modinv_2048 | 0.0737538 | 0.52% | 68681.7 | 73785.1 | 77697.1 | 79122.0 | 1.15 | 110 |
| isprime_true_2048 | 67.9633 | 0.13% | 67124000.0 | 68021400.0 | 68533100.0 | 68573000.0 | 1.02 | 55 |

## wipe

| Operation | mean ms/op | ±95% CI | min ns | p50 ns | p99 ns | max ns | max/min | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mul_256 | 6.91284e-05 | 0.27% | 66.8 | 68.9 | 70.9 | 71.0 | 1.06 | 110 |
| sqr_256 | 6.3478e-05 | 0.51% | 59.8 | 63.3 | 67.1 | 67.3 | 1.12 | 80 |
| divrem_256 | 0.000178374 | 0.54% | 172.7 | 177.7 | 185.4 | 188.4 | 1.09 | 51 |
| modmul_256 | 0.000318725 | 0.45% | 308.7 | 319.0 | 328.9 | 329.3 | 1.07 | 54 |
| barrettsetup_256 | 0.000346224 | 0.29% | 334.7 | 346.5 | 355.7 | 359.0 | 1.07 | 83 |
| barrettmul_256 | 0.000413811 | 0.72% | 396.2 | 412.5 | 444.4 | 455.2 | 1.15 | 50 |
| montsetup_256 | 0.00047626 | 0.35% | 459.3 | 475.6 | 491.5 | 498.0 | 1.08 | 80 |
| montmul_256 | 0.000155466 | 0.45% | 150.3 | 155.6 | 160.0 | 160.1 | 1.07 | 50 |
| montmul_scratch_256 | 0.000121778 | 0.30% | 116.6 | 122.0 | 126.3 | 127.1 | 1.09 | 110 |
| montsqr_256 | 0.0001459 | 0.43% | 137.3 | 145.7 | 152.6 | 154.2 | 1.12 | 84 |
| montsqr_scratch_256 | 0.000118805 | 0.41% | 113.3 | 118.8 | 124.8 | 125.0 | 1.10 | 80 |
| montpow_rand_256 | 0.0231419 | 0.32% | 22567.2 | 23132.2 | 23675.6 | 23724.5 | 1.05 | 50 |
| modpow_256 | 0.0237009 | 0.45% | 22822.6 | 23725.4 | 24354.1 | 24453.0 | 1.07 | 50 |
| gcd_256 | 0.00584849 | 1.37% | 4846.7 | 5855.5 | 6621.8 | 6700.3 | 1.38 | 80 |
| modinv_256 | 0.0100702 | 1.43% | 8985.1 | 9946.5 | 11059.8 | 11488.1 | 1.28 | 50 |
| isprime_true_256 | 0.311567 | 0.35% | 297790.0 | 311567.0 | 320112.0 | 322107.0 | 1.08 | 82 |
| mul_2048 | 0.00210777 | 0.73% | 1992.6 | 2096.5 | 2199.3 | 2199.6 | 1.10 | 50 |
| sqr_2048 | 0.00149513 | 0.48% | 1388.4 | 1508.3 | 1542.0 | 1553.6 | 1.12 | 110 |
| divrem_2048 | 0.00107602 | 0.33% | 1019.1 | 1074.3 | 1121.5 | 1137.0 | 1.12 | 140 |
| modmul_2048 | 0.00526901 | 0.56% | 4965.3 | 5281.9 | 5468.8 | 5496.5 | 1.11 | 80 |
| barrettsetup_2048 | 0.00321156 | 0.45% | 3037.1 | 3215.2 | 3336.8 | 3353.3 | 1.10 | 88 |
| barrettmul_2048 | 0.00562125 | 0.64% | 5380.9 | 5593.0 | 5877.2 | 5902.8 | 1.10 | 50 |
| montsetup_2048 | 0.00466398 | 0.52% | 4509.5 | 4658.2 | 4822.9 | 4862.6 | 1.08 | 50 |
| montmul_2048 | 0.00278732 | 0.37% | 2695.9 | 2791.3 | 2839.6 | 2847.7 | 1.06 | 51 |
| montmul_scratch_2048 | 0.0026871 | 0.32% | 2591.6 | 2697.5 | 2748.6 | 2763.8 | 1.07 | 80 |
| montsqr_2048 | 0.00231659 | 0.41% | 2241.8 | 2306.6 | 2394.1 | 2416.8 | 1.08 | 53 |
| montsqr_scratch_2048 | 0.00224862 | 0.27% | 2166.1 | 2248.1 | 2318.9 | 2339.7 | 1.08 | 142 |
| montpow_rand_2048 | 0.716757 | 0.26% | 695348.0 | 717635.0 | 731657.0 | 767757.0 | 1.10 | 110 |
| modpow_2048 | 0.718968 | 0.38% | 694785.0 | 720541.0 | 732248.0 | 733763.0 | 1.06 | 50 |
| gcd_2048 | 0.0570647 | 0.64% | 54361.2 | 56857.8 | 59671.7 | 60266.3 | 1.11 | 50 |
| modinv_2048 | 0.0757583 | 0.71% | 71301.9 | 75707.7 | 79557.8 | 81483.9 | 1.14 | 57 |
| isprime_true_2048 | 67.0663 | 0.11% | 66486600.0 | 67064700.0 | 67505000.0 | 67553400.0 | 1.02 | 52 |
