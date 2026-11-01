# BENCH T3 VLDG (double-buffer V) — MTP ON config prod — 28/10

date : 2026-10-28 14:12 UTC
reps : 2 · service restauré : OUI
SO prod = 4fd152f9 · SO vldg = 87f85442
ctx 114688 / 160000 · -b 2048 · MTP ON (spec-draft-n-max 3) · 64 gen
Seule diff entre les .so = le kernel T3 → B vs A = gain e2e MTP ON de vldg.
L'acceptance MTP doit rester identique (bit-exact).

## ctx 114688
| leg | rep | decode t/s | prefill t/s | acc | mean len | acc rate/pos |
|---|---|---|---|---|---|---|
| A-prod | 1 | 15.95 | 621.6 | 0.92 | 3.71 | (1.000,0.929,0.821) |
| B-vldg | 1 | 16.23 | 621.5 | 0.92 | 3.71 | (1.000,0.929,0.821) |
| A-prod | 2 | 14.86 | 621.6 | 0.92 | 3.71 | (1.000,0.929,0.821) |
| B-vldg | 2 | 15.72 | 621.6 | 0.92 | 3.71 | (1.000,0.929,0.821) |

Δ decode vs A-prod (médiane) : **+1.76%**  (A=15.95 → B=16.23 t/s)

## ctx 160000
| leg | rep | decode t/s | prefill t/s | acc | mean len | acc rate/pos |
|---|---|---|---|---|---|---|
| A-prod | 1 | 24.49 | 374.8 | 0.74138 | 3.15 | (0.938,0.854,0.708) |
| B-vldg | 1 | 24.82 | 375.3 | 0.74138 | 3.15 | (0.938,0.854,0.708) |
| A-prod | 2 | 24.72 | 375.0 | 0.74138 | 3.15 | (0.938,0.854,0.708) |
| B-vldg | 2 | 24.64 | 375.2 | 0.74138 | 3.15 | (0.938,0.854,0.708) |

Δ decode vs A-prod (médiane) : **+0.40%**  (A=24.72 → B=24.82 t/s)

## VRAM pic par leg (MiB)
| leg | max VRAM |
|---|---|
| A-prod | 32438 |
| B-vldg | 32438 |
