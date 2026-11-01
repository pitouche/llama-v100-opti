# Microbench kernel FA rows 1-8 — wmma stock / H2 BC=8 / T3

Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, mask f16=0, n_kv=107351 (pad 256).
.so prod T3 (d2df07b8), dispatch réel, 200 iters (médiane).

| T (rows) | stock (µs) | H2 (µs) | T3 (µs) | T3 vs stock | T3 vs H2 | H2 vs stock |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1194 | None | 585 | +51.0% | — | — |
| 2 | 1827 | None | 521 | +71.5% | — | — |
| 3 | 2217 | None | 662 | +70.1% | — | — |
| 4 | 2218 | 1542 | 668 | +69.9% | +56.7% | +30.5% |
| 5 | 2982 | None | 793 | +73.4% | — | — |
| 6 | 2993 | None | 1206 | +59.7% | — | — |
| 7 | 2988 | None | 1216 | +59.3% | — | — |
| 8 | 2986 | None | 1196 | +59.9% | — | — |

Lecture : « T3 vs X » = (X - T3)/X : positif = T3 plus rapide.
H2 n'est éligible que sur T=4 (rows==4) ; sur les autres T, le mode h2 est identique à stock.