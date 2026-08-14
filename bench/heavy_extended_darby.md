# Heavy-tailed operations, extended sizes (Pi)

Companion to the M4 record: isprime 5120-8192 and sqrtmod 5120 on the
Pi 5 (sessions 300 s). sqrtmod beyond 5120 bits needs prime hunts this
host cannot complete often enough within a session to sample honestly;
the record stops where the data does.

| Operation | mean ms/op | ±95% CI | min ns | p50 ns | p99 ns | max ns | max/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| isprime_5120 | 16.4107 | 7.66% | 604.4 | 1051.3 | 194861000.0 | 2356750000.0 | 3899302.29 |
| isprime_6144 | ~58.2656 | 122.92% | 734.1 | 1084.4 | 334851000.0 | 384686000.0 | 524052.53 |
| isprime_7168 | ~90.2223 | 177.63% | 870.9 | 2916.3 | 529882000.0 | 6368320000.0 | 7312259.59 |
| isprime_8192 | ~96.5283 | 171.40% | 1034.7 | 3380.6 | 788685000.0 | 831577000.0 | 803712.29 |
| sqrtmod_5120 | 166.695 | 0.00% | 309235.0 | 311885.0 | 582424000.0 | 582883000.0 | 1884.92 |
