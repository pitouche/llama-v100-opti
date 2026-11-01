# A/B applicatif — lba (T3v2 launch_bounds/NT) vs prod d2df (T3), MTP OFF

date : 2026-10-22 22:58 UTC
service restauré : OUI
.so prod md5 : d2df07b8  ·  .so lba md5 : fa3e75c2

A-lba = candidat (GGML_BACKEND_PATH=lba) · B-prod = prod d2df. Même -b 2048, MTP OFF.
Δ = (A-lba - B-prod)/B-prod : positif = lba plus rapide.

## DECODE (t/s) par leg (single-token = T=1, cas amélioré par lba)
| contexte | rep | A-lba | B-prod | Δ |
|---|---|---|---|---|
| 32768 | 1 | 24.89 | 24.55 | +1.4% |
| 32768 | 2 | 24.88 | 24.57 | +1.3% |
| 65536 | 1 | 23.56 | 23.06 | +2.2% |
| 65536 | 2 | 23.55 | 23.04 | +2.2% |
| 98304 | 1 | 22.27 | 21.67 | +2.8% |
| 98304 | 2 | 22.27 | 21.68 | +2.7% |

## DECODE — médianes par leg
| contexte | A-lba | B-prod | Δ |
|---|---|---|---|
| 32768 | 24.89 | 24.57 | +1.3% |
| 65536 | 23.56 | 23.06 | +2.2% |
| 98304 | 22.27 | 21.68 | +2.7% |

## WALL (s par requête, prompt+64 gen)
| contexte | A-lba | B-prod | Δ s |
|---|---|---|---|
| 32768 | 36.5 | 36.4 | +0.1 |
| 65536 | 84.3 | 84.4 | -0.1 |
| 98304 | 146.5 | 146.6 | -0.1 |

