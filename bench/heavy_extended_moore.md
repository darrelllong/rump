# Heavy-tailed operations, extended sizes (EPYC)

Companion to the M4 record: isprime 5120-8192 and sqrtmod 5120-8192 on the
EPYC (sessions 300 s, 480 s for sqrtmod_8192). The sqrtmod_8192 cell here
carries the descent tail the M4's thin session missed on its first
attempt; the earlier inconclusive EPYC session (all readings on the fast
path) is superseded by this one.

| Operation | mean ms/op | ±95% CI | min ns | p50 ns | p99 ns | max ns | max/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| isprime_5120 | ~5.62891 | 283.73% | 1081.6 | 1143.5 | 84525300.0 | 1009210000.0 | 933088.63 |
| isprime_6144 | ~10.2137 | 194.75% | 1298.3 | 2609.9 | 148515000.0 | 1782170000.0 | 1372663.34 |
| isprime_7168 | ~10.3632 | 281.56% | 1516.0 | 1580.7 | 234358000.0 | 2807720000.0 | 1852106.92 |
| isprime_8192 | ~24.6679 | 194.77% | 1726.4 | 1812.3 | 346225000.0 | 4137470000.0 | 2396546.63 |
| sqrtmod_5120 | ~100.817 | 179.07% | 217464.0 | 83948500.0 | 255179000.0 | 264293000.0 | 1215.34 |
| sqrtmod_6144 | ~147.914 | 324.54% | 275245.0 | 281277.0 | 442696000.0 | 443670000.0 | 1611.91 |
| sqrtmod_8192 | 401.834 | 0.00% | 396556.0 | 413403.0 | 1031070000.0 | 1035370000.0 | 2610.90 |
