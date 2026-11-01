# BENCH T3 (anyei #153) — 18/10 — MTP OFF, -b 2048, interleave

date : 2026-10-18 13:33 UTC
reps : 2 · service restauré : OUI

## ctx 114688
| leg | rep | decode t/s | prefill t/s |
|---|---|---|---|
| A-prod | 1 | 16.97 | 617.3 |
| B-t3on | 1 | 18.85 | 618.1 |
| C-t3off | 1 | 17.52 | 618.3 |
| A-prod | 2 | 17.68 | 618.5 |
| B-t3on | 2 | 19.99 | 618.5 |
| C-t3off | 2 | 17.39 | 618.3 |

Δ decode vs A-prod (médiane) :
  - B-t3on: +13.07%
  - C-t3off: -0.90%

## ctx 196608
| leg | rep | decode t/s | prefill t/s |
|---|---|---|---|
| A-prod | 1 | 14.13 | 342.6 |
| B-t3on | 1 | 17.82 | 342.9 |
| C-t3off | 1 | 14.13 | 342.7 |
| A-prod | 2 | 14.13 | 342.8 |
| B-t3on | 2 | 18.16 | 342.8 |
| C-t3off | 2 | 14.12 | 342.9 |

Δ decode vs A-prod (médiane) :
  - B-t3on: +28.52%
  - C-t3off: +0.00%

