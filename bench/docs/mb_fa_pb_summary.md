# Microbench FA T3 — levier A : parallel_blocks forcé (2/4 waves)

Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv=107351 (pad 256). 200 iters, médiane. MTP-OFF.
REF = prod fa3e75c2 (pb auto=60 @T=1-2) · PB120/PB240 = t3lba_pb (4863c97f) + env GGML_CUDA_FATTN_PB (kernel T3/lba identique au prod, seul le pb diffère).

| T | REF µs (pb60) | PB120 µs | PB240 µs | 120 vs REF | 240 vs REF |
|---:|---:|---:|---:|---:|---:|
| 1 | 482 | 496 | 520 | -2.8% | -7.9% |
| 2 | 521 | 543 | 581 | -4.1% | -11.4% |
| 3 | 662 | 694 | 769 | -4.8% | -16.1% |
| 4 | 668 | 710 | 797 | -6.3% | -19.3% |
| 5 | 793 | 841 | 966 | -6.1% | -21.8% |
| 6 | 1205 | 1206 | 1206 | -0.1% | -0.1% |
| 7 | 1218 | 1218 | 1218 | +0.0% | +0.0% |
| 8 | 1196 | 1196 | 1196 | +0.0% | +0.0% |

Lecture : « X vs REF » = (REF − X)/REF : positif = X plus rapide. pb forcé : @T=1-2 ntiles_dst=4 (pb 60/120/240 = 1/2/4 waves), @T=3-5 pb auto=40 (120/240 = 1/2 waves), @T=6-8 pb auto=20 (120/240 = 1/2 waves).