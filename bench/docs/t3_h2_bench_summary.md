# BENCH H2 confirmation .so T3 PROD (d2df07b8, sans T1) — 18/10 — MTP ON (config prod), -b 2048, interleave

date : 2026-10-18 22:25 UTC
reps : 2 · service restauré : OUI
config : -c 167000, --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling, KV q8_0, FA auto
binaire : SO_T3 PROD (sans T1) — A = H2 ON (prod actuelle), B = H2 OFF (proposé)

## ctx 114688
| leg | rep | decode t/s | prefill t/s |
|---|---|---|---|
| A-h2on | 1 | 45.59 | 609.1 |
| B-h2off | 1 | 53.73 | 609.9 |
| A-h2on | 2 | 45.76 | 609.3 |
| B-h2off | 2 | 53.63 | 609.5 |

Δ decode vs A-h2on (médiane) :
  - B-h2off: +17.42%  (= gain de désactiver H2 sur le .so prod)

## ctx 160000
| leg | rep | decode t/s | prefill t/s |
|---|---|---|---|
| A-h2on | 1 | 40.88 | 369.5 |
| B-h2off | 1 | 47.24 | 369.8 |
| A-h2on | 2 | 40.89 | 369.0 |
| B-h2off | 2 | 47.22 | 369.3 |

Δ decode vs A-h2on (médiane) :
  - B-h2off: +15.53%  (= gain de désactiver H2 sur le .so prod)

