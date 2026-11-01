# H2 A/B — config lente Step A, MTP ON, .so 8035caf4 figé

date : 2026-10-29 00:06 UTC
legs : A-h2off (Q8_FATTN_TC=0) vs B-h2on (défaut ON) — tout le reste identique
config : -c 167000, -b 2048 -ub 2048, --mmproj, MTP ON (n-max 3, backend-sampling),
         KV q8_0, FA auto, shortlist 131072, DEFER_PROMPT=1, W4R4=1, FIT 1024, CVD=0
service restauré : OUI

## ctx 114688
| leg | rep | decode t/s | prefill t/s | VRAM |
|---|---|---|---|---|
| A-h2off | 1 | 15.7 | 621.7 | 32438 |
| B-h2on | 1 | 15.23 | 622.4 | 32438 |

Δ decode B-h2on vs A-h2off : **-3.0%** (15.70 → 15.23 t/s)

## ctx 160000
| leg | rep | decode t/s | prefill t/s | VRAM |
|---|---|---|---|---|
| A-h2off | 1 | 24.51 | 375.6 | 32438 |
| B-h2on | 1 | 23.01 | 375.7 | 32438 |

Δ decode B-h2on vs A-h2off : **-6.1%** (24.51 → 23.01 t/s)

