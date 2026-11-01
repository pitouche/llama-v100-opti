# SL/DEFER A/B — config lente, MTP ON, .so 8035caf4 figé

date : 2026-10-29 00:27 UTC
legs : A-full (shortlist+DEFER+H2off) vs B-noflags (sans shortlist/DEFER, H2off)
config : -c 167000, -b 2048 -ub 2048, --mmproj, MTP ON (n-max 3), KV q8_0, FA auto,
         W4R4=1, FIT 1024, CVD=0
service restauré : OUI

## ctx 114688
| leg | rep | decode t/s | prefill t/s | VRAM |
|---|---|---|---|---|
| A-full | 1 | 16.26 | 621.5 | 32438 |
| B-noflags | 1 | 56.4 | 608.9 | 32286 |

Δ decode B-noflags vs A-full : **+246.9%** (16.26 → 56.40 t/s)

## ctx 160000
| leg | rep | decode t/s | prefill t/s | VRAM |
|---|---|---|---|---|
| A-full | 1 | 25.28 | 375.6 | 32438 |
| B-noflags | 1 | 50.31 | 369.2 | 32286 |

Δ decode B-noflags vs A-full : **+99.0%** (25.28 → 50.31 t/s)

