# Heavy-tailed operations, extended sizes (M4)

Verification record for the resolved isprime sampling defect and the
extended extrema range: 5120 to 8192 bits in 1024-bit steps, joining the
primitives table's 256-4096 rows in the extrema table. Sessions: 120 s
(isprime), 240 s (sqrtmod, whose every trial must generate a fresh random
prime modulus of the row's width). sqrtmod stops at 6144 bits: beyond it,
prime generation costs minutes per trial on this host and even 600 s
sessions yield single-reading cells, which this record does not accept.

| Operation | mean ms/op | ±95% CI | min ns | p50 ns | p99 ns | max ns | max/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| isprime_5120 | ~5.06251 | 156.88% | 159.0 | 412.1 | 47034300.0 | 52267700.0 | 328766.96 |
| isprime_6144 | ~11.4923 | 234.42% | 198.3 | 509.2 | 82639800.0 | 984396000.0 | 4963074.25 |
| isprime_7168 | ~13.9094 | 172.02% | 218.8 | 589.5 | 130857000.0 | 1562000000.0 | 7140081.82 |
| isprime_8192 | ~20.1986 | 227.82% | 279.5 | 438.2 | 202646000.0 | 209883000.0 | 750826.37 |
| sqrtmod_5120 | ~57.6934 | 266.53% | 168057.0 | 176154.0 | 142061000.0 | 142449000.0 | 847.62 |
| sqrtmod_6144 | ~115.222 | 534.05% | 247443.0 | 81619700.0 | 246296000.0 | 249409000.0 | 1007.95 |
