# BENCH DE REGIME CTX — 36k vs 90k, 1024 gen — MTP ON config prod — 24/10

date : 2026-10-24 01:39 UTC
reps : 2 · service restauré : OUI
.so prod fa3e75c2 · -b 2048 · MTP ON (spec-draft-n-max 3) · 1024 gen

| point | rep | prefill t/s | decode t/s | wall s | acc | mean len | acc rate/pos |
|---|---|---|---|---|---|---|---|
| P1-36k | 1 | 917.7 | 39.76 | 43.5 | 0.6776 | 3.03 | (0.917, 0.722, 0.514) |
| P1-36k | 2 | 923.4 | 41.49 | 43.0 | 0.6776 | 3.03 | (0.917, 0.722, 0.514) |
| P2-90k | 1 | 642.1 | 32.02 | 173.6 | 0.6281 | 2.88 | (0.834, 0.658, 0.492) |
| P2-90k | 2 | 642.5 | 31.95 | 174.0 | 0.6281 | 2.88 | (0.834, 0.658, 0.492) |

## médianes (toutes reps)
  - P1-36k: decode 41.49 t/s
  - P2-90k: decode 32.02 t/s
