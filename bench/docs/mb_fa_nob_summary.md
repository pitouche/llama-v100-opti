# Microbench FA T3 — sonde combinatoire : revert du levier B (minblocks NT=1 4 -> 3)

Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv=107351 (pad 256). 200 iters, médiane. MTP-OFF. Launcher identique (pb auto=60 @T=1).
REF = prod (4fd152f9 = B+C) · NOB = candidat (e19b6112 = C sans B, minblocks NT=1=3). Seule diff = launch_bounds NT=1 (4 vs 3) : 4 vs 3 blocs/SM.
Validation : maxrel < 1e-4 requis ; bit-identique = False.

| T | NT | REF (B+C) µs | NOB (sans B) µs | NOB vs REF | note |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 404 | 433 | -7.1% | **cas decode prod** |
| 2 | 2 | 467 | 467 | +0.0% | MTP verify |
| 3 | 2 | 620 | 622 | -0.2% | MTP verify |
| 4 | 2 | 625 | 626 | -0.2% | MTP verify |
| 5 | 2 | 701 | 701 | +0.0% | MTP verify |
| 6 | 3 | 1109 | 1110 | -0.1% | MTP verify |
| 7 | 3 | 1117 | 1117 | +0.0% | MTP verify |
| 8 | 3 | 1097 | 1097 | +0.0% | MTP verify |

Lecture : « NOB vs REF » = (REF − NOB)/REF : positif = NOB plus rapide (attendu négatif @T=1, le revert de B coûte ~ le gain de B). Le coût mesuré = la marge à battre par un kernel réécrit avec ILP/prefetch qui tient dans 3 blocs/SM (regs >= 129).