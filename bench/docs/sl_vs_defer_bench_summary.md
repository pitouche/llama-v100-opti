# SL vs DEFER — config lente b2048, MTP ON, .so 8035caf4

date : 2026-10-29 00:44 UTC
legs : B-slonly (shortlist seul) vs C-deferonly (DEFER seul)
contrôles (sl_defer_ab) : full=16.26 (lent) | noflags=56.40 (rapide) @114k
service restauré : OUI

## ctx 114688
| leg | rep | decode t/s | prefill t/s | VRAM |
|---|---|---|---|---|
| B-slonly | 1 | 58.12 | 608.5 | 32286 |
| C-deferonly | 1 | 14.79 | 622.4 | 32438 |

  - B-slonly : 58.12 t/s (vs full 16.26 / noflags 56.40)
  - C-deferonly : 14.79 t/s (vs full 16.26 / noflags 56.40)

## ctx 160000
| leg | rep | decode t/s | prefill t/s | VRAM |
|---|---|---|---|---|
| B-slonly | 1 | 46.65 | 369.6 | 32286 |
| C-deferonly | 1 | 26.18 | 375.8 | 32438 |

  - B-slonly : 46.65 t/s (vs full 16.26 / noflags 56.40)
  - C-deferonly : 26.18 t/s (vs full 16.26 / noflags 56.40)

