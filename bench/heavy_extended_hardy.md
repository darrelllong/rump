# Heavy-tailed operations, extended sizes (M4)

Verification record for the resolved isprime sampling defect and the
extended extrema range: 5120 to 8192 bits in 1024-bit steps, joining the
primitives table's 256-4096 rows in the extrema table. Sessions: 120 s
(isprime), 240 s (sqrtmod at 5120-6144), 360 s (sqrtmod at 7168-8192,
whose every reading must generate a fresh random prime modulus of the
row's width; the op-aware pool brought that hunt to seconds per reading,
which is what made these two rows measurable). Where pilot-bench's
subsession confidence interval did not converge within the session the
mean carries the `~` flag and the order statistics are the record.

An EPYC re-measurement corroborates the 7168-bit sqrtmod tail (spread
2173x, 321 us to 698 ms); its 8192-bit session drew only fast-path primes
(all readings ~0.39 ms, spread 1.05) and is recorded as inconclusive
rather than as a bound — the M4 row carries that size.

| Operation | mean ms/op | ±95% CI | min ns | p50 ns | p99 ns | max ns | max/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| isprime_5120 | ~5.06251 | 156.88% | 159.0 | 412.1 | 47034300.0 | 52267700.0 | 328766.96 |
| isprime_6144 | ~11.4923 | 234.42% | 198.3 | 509.2 | 82639800.0 | 984396000.0 | 4963074.25 |
| isprime_7168 | ~13.9094 | 172.02% | 218.8 | 589.5 | 130857000.0 | 1562000000.0 | 7140081.82 |
| isprime_8192 | ~20.1986 | 227.82% | 279.5 | 438.2 | 202646000.0 | 209883000.0 | 750826.37 |
| sqrtmod_5120 | ~57.6934 | 266.53% | 168057.0 | 176154.0 | 142061000.0 | 142449000.0 | 847.62 |
| sqrtmod_6144 | ~115.222 | 534.05% | 247443.0 | 81619700.0 | 246296000.0 | 249409000.0 | 1007.95 |
| sqrtmod_7168 | ~21.869 | — | 120316.0 | 122703.0 | 122845.0 | 130605000.0 | 1085.52 |
| sqrtmod_8192 | ~197.011 | — | 144805.0 | 145677.0 | 196751000.0 | 591002000.0 | 4081.36 |
