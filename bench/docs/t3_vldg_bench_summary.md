# BENCH T3 VLDG (double-buffer V, GO microbench) — 28/10 — MTP OFF, -b 2048, interleave

date : 2026-10-28 13:33 UTC
reps : 2 · service restauré : OUI
SO prod = 4fd152f9 · SO vldg = 87f85442

Seule diff entre les .so = le kernel T3. B vs A = gain e2e vldg. Attente : decode B > A (croissant avec le ctx), prefill B ≈ A.

## ctx 114688
| leg | rep | decode t/s | prefill t/s |
|---|---|---|---|
| A-prod | 1 | 22.68 | 619.8 |
| B-vldg | 1 | 23.19 | 620.3 |
| A-prod | 2 | 22.67 | 620.3 |
| B-vldg | 2 | 23.19 | 620.1 |

Δ decode vs A-prod (médiane) : **+2.25%**  (A=22.68 t/s → B=23.19 t/s)
Δ prefill vs A-prod (médiane) : +0.00%  (A=620.3 t/s → B=620.3 t/s) — doit rester ~0 (T3 ne touche que le decode)

## ctx 196608
| leg | rep | decode t/s | prefill t/s |
|---|---|---|---|
| A-prod | 1 | 20.65 | 343.6 |
| B-vldg | 1 | 21.23 | 343.5 |
| A-prod | 2 | 20.64 | 343.9 |
| B-vldg | 2 | 21.22 | 343.4 |

Δ decode vs A-prod (médiane) : **+2.81%**  (A=20.65 t/s → B=21.23 t/s)
Δ prefill vs A-prod (médiane) : -0.12%  (A=343.9 t/s → B=343.5 t/s) — doit rester ~0 (T3 ne touche que le decode)

## VRAM pic par leg (MiB)
| leg | max VRAM |
|---|---|
| A-prod | 30988 |
| B-vldg | 30988 |
