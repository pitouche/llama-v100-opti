# BENCH H2 (T1T3 avec/sans H2) — 18/10 — MTP ON (config prod), -b 2048, interleave

date : 2026-10-18 21:21 UTC
reps : 2 · service restauré : OUI
config : -c 167000, --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling, KV q8_0, FA auto
binaire : SO_T1T3 (T1 on, T3 on) — seul le gate H2 GGML_CUDA_VOLTA_Q8_FATTN_TC change

## ctx 114688
| leg | rep | decode t/s | prefill t/s |
|---|---|---|---|
| A-h2on | 1 | 45.92 | 609.6 |
| B-h2off | 1 | 53.94 | 609.8 |
| A-h2on | 2 | 45.88 | 610.2 |
| B-h2off | 2 | 53.92 | 609.7 |

Δ decode vs A-h2on (médiane) :
  - B-h2off: +17.47%  (= effet pur de H2 : positif => H2 utile)

## ctx 160000
| leg | rep | decode t/s | prefill t/s |
|---|---|---|---|
| A-h2on | 1 | 36.3 | 369.7 |
| B-h2off | 1 | 44.57 | 369.5 |
| A-h2on | 2 | 36.26 | 369.3 |
| B-h2off | 2 | 44.6 | 369.8 |

Δ decode vs A-h2on (médiane) :
  - B-h2off: +22.87%  (= effet pur de H2 : positif => H2 utile)

