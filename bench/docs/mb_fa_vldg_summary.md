# Microbench FA T3 — rewrite vldg (double-buffer V phase 2, ILP)

Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv=107351 (pad 256). 200 iters, médiane. MTP-OFF. Launcher identique (pb auto=60 @T=1).
REF = prod (4fd152f9 = B+C) · VLDG = candidat (87f85442 = phase 2 pipelinée : fenêtres V (4 mots 32-bit) du chunk ch+1 en vol pendant les mmas du chunk ch, chunk 0 chargé avant la boucle ; mêmes mots + même déquant que la baseline).
Validation : VLDG maxrel=0.00e+00, bit-identique=True (attendu True).

| T | NT | REF µs | VLDG µs | VLDG vs REF |
|---:|---:|---:|---:|---:|
| 1 | 1 | 406 | 359 | +11.4% |
| 2 | 2 | 468 | 399 | +14.7% |
| 3 | 2 | 620 | 508 | +18.1% |
| 4 | 2 | 626 | 516 | +17.5% |
| 5 | 2 | 702 | 595 | +15.3% |
| 6 | 3 | 1110 | 891 | +19.7% |
| 7 | 3 | 1118 | 898 | +19.7% |
| 8 | 3 | 1096 | 881 | +19.6% |

Lecture : « VLDG vs REF » = (REF − VLDG)/REF : positif = VLDG plus rapide. **Seuil GO : >= +2 % @T=1 sans régression T=2..8.**