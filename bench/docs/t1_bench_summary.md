# BENCH T1 (anyei #153 mmsmt) — 18/10 — MTP ON (config prod), -b 2048, interleave

date : 2026-10-18 20:22 UTC
reps : 2 · service restauré : OUI
config : -c 167000, --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling, KV q8_0, FA auto

## ctx 114688
| leg | rep | decode t/s | prefill t/s |
|---|---|---|---|
| A-t3 | 1 | 45.7 | 608.8 |
| B-t1t3 | 1 | 45.92 | 609.7 |
| C-smt0 | 1 | 45.68 | 609.7 |
| A-t3 | 2 | 45.76 | 609.8 |
| B-t1t3 | 2 | 45.9 | 609.9 |
| C-smt0 | 2 | 45.72 | 609.7 |

Δ decode vs A-t3 (médiane) :
  - B-t1t3: +0.35%
  - C-smt0: -0.09%

## ctx 160000
| leg | rep | decode t/s | prefill t/s |
|---|---|---|---|
| A-t3 | 1 | 40.89 | 369.4 |
| B-t1t3 | 1 | 36.19 | 369.8 |
| C-smt0 | 1 | 40.84 | 369.3 |
| A-t3 | 2 | 40.81 | 369.8 |
| B-t1t3 | 2 | 36.27 | 369.7 |
| C-smt0 | 2 | 40.88 | 369.4 |

Δ decode vs A-t3 (médiane) :
  - B-t1t3: -11.30%
  - C-smt0: -0.02%

