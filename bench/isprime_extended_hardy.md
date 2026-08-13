# isprime extended sizes (M4)

Verification record for the resolved isprime sampling defect: with the
op-aware pool and the 120 s session, the mean is strictly monotone in
size and the Miller–Rabin tail is present at every width. Rows 256–4096
are the primitives table's; 5120 and 6144 extend the check past the size
where the defective harness produced its non-monotonic artifact.

| Operation | mean ms/op | ±95% CI | min ns | p50 ns | p99 ns | max ns | max/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| isprime_256 | ~0.00491276 | 251.32% | 17.3 | 25.5 | 14720.5 | 1681580.0 | 97145.00 |
| isprime_1024 | ~0.0299412 | 227.08% | 37.2 | 80.8 | 411026.0 | 4826580.0 | 129600.45 |
| isprime_2048 | ~0.216704 | 192.49% | 64.9 | 143.7 | 2886540.0 | 34267100.0 | 528242.64 |
| isprime_4096 | ~1.4146 | 227.08% | 132.7 | 255.8 | 23864300.0 | 284412000.0 | 2143351.29 |
| isprime_5120 | ~5.06251 | 156.88% | 159.0 | 412.1 | 47034300.0 | 52267700.0 | 328766.96 |
| isprime_6144 | ~11.4923 | 234.42% | 198.3 | 509.2 | 82639800.0 | 984396000.0 | 4963074.25 |
