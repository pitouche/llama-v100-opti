# BENCH MTP SAMPLING — régime 90k, 3 temperatures, 256 gen — MTP ON config prod — 24/10

date : 2026-10-24 04:51 UTC
reps : 2 · service restauré : OUI
.so prod fa3e75c2 · -b 2048 · MTP ON (spec-draft-n-max 3, backend-sampling) · 256 gen ignore_eos

Réf : prod acc 2.62-2.68 (0.717, 0.521, 0.385) ; filler temp0 (bench_ctx90k P2) 2.88 (0.834, 0.658, 0.492) ; clean temp0 3.94 (run9c).

| point | temp | top_k | rep | decode t/s | gen tok | acc | mean len | acc rate/pos |
|---|---|---|---|---|---|---|---|---|
| T0-greedy | 0.0 | 1 | 1 | 27.48 | 256 | 0.57706 | 2.73 | (0.779, 0.606, 0.462) |
| T0-greedy | 0.0 | 1 | 2 | 27.08 | 256 | 0.57706 | 2.73 | (0.779, 0.606, 0.462) |
| T07-topk20 | 0.7 | 20 | 1 | 40.82 | 256 | 0.57295 | 2.71 | (0.773, 0.591, 0.419) |
| T07-topk20 | 0.7 | 20 | 2 | 39.12 | 256 | 0.54483 | 2.63 | (0.781, 0.557, 0.403) |
| T10-topk40 | 1.0 | 40 | 1 | 35.55 | 256 | 0.4537 | 2.36 | (0.735, 0.533, 0.366) |
| T10-topk40 | 1.0 | 40 | 2 | 42.9 | 256 | 0.6374 | 2.9 | (0.792, 0.574, 0.422) |

## médianes (toutes reps)
  - T0-greedy: decode 27.48 t/s, mean_len 2.73
  - T07-topk20: decode 40.82 t/s, mean_len 2.71
  - T10-topk40: decode 42.9 t/s, mean_len 2.9
