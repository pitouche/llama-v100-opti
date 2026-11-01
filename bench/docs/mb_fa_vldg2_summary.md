# Microbench FA T3 — validation 3 voies du rewrite vldg (re-run)

Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv=107351 (pad 256). 200 iters, médiane. MTP-OFF. Launcher identique (pb auto=60 @T=1).
REF = prod (4fd152f9) · CVEC = baseline source rebuildée (723c0016, SASS stream identique au prod) · VLDG = candidat double-buffer V (87f85442).
Validation : VLDG maxrel=0.00e+00 bit-identique=True ; CVEC maxrel=0.00e+00 (drift -use_fast_math attendu ~5e-5, tol 1e-4).

| T | NT | REF µs | CVEC µs | CVEC vs REF | VLDG µs | VLDG vs REF | VLDG vs CVEC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 406 | 402 | +0.8% | 360 | +11.1% | +10.4% |
| 2 | 2 | 467 | 466 | +0.2% | 399 | +14.5% | +14.3% |
| 3 | 2 | 622 | 620 | +0.3% | 508 | +18.3% | +18.0% |
| 4 | 2 | 626 | 625 | +0.2% | 515 | +17.7% | +17.5% |
| 5 | 2 | 702 | 700 | +0.3% | 595 | +15.3% | +15.1% |
| 6 | 3 | 1111 | 1109 | +0.2% | 891 | +19.8% | +19.7% |
| 7 | 3 | 1119 | 1117 | +0.2% | 898 | +19.8% | +19.6% |
| 8 | 3 | 1097 | 1097 | +0.0% | 881 | +19.7% | +19.7% |

Lecture : « X vs Y » = (Y − X)/Y : positif = X plus rapide. **Seuil GO : VLDG vs REF >= +2 % @T=1 sans régression T=2..8, et CVEC vs REF ~0** (pas d'artefact de build dans la référence).