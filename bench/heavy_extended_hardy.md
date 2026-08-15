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
| sqrtmod_7168 | ~202.068 | 113.21% | 127667.0 | 130029.0 | 390633000.0 | 417382000.0 | 3269.30 |
| sqrtmod_8192 | ~197.011 | — | 144805.0 | 145677.0 | 196751000.0 | 591002000.0 | 4081.36 |
| sqrtmod_blum_5120 | ~54.6772 | 40.72% | 47499400.0 | 49035000.0 | 76781000.0 | 132083000.0 | 2.78 |
| sqrtmod_blum_6144 | 87.3766 | 1.57% | 82084300.0 | 86181700.0 | 92596400.0 | 94005700.0 | 1.15 |
| sqrtmod_blum_7168 | ~139.757 | 11.68% | 134392000.0 | 138113000.0 | 145517000.0 | 145645000.0 | 1.08 |
| sqrtmod_blum_8192 | 283.899 | 0.00% | 197574000.0 | 295283000.0 | 305040000.0 | 337698000.0 | 1.71 |
| sqrtmod_descent_5120 | 150.048 | 3.76% | 144465000.0 | 148122000.0 | 155624000.0 | 166806000.0 | 1.15 |
| sqrtmod_descent_6144 | 253.382 | 0.00% | 244868000.0 | 245634000.0 | 252923000.0 | 285753000.0 | 1.17 |
| sqrtmod_descent_7168 | ~484.933 | 35.83% | 419263000.0 | 483071000.0 | 484461000.0 | 552935000.0 | 1.32 |
| sqrtmod_descent_8192 | 683.301 | 0.00% | 683301000.0 | 683301000.0 | 683301000.0 | 683301000.0 | 1.00 |
