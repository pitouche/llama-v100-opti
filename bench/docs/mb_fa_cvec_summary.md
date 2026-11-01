# Microbench FA T3 — levier C : load K q8_0 vectorisé 128-bit (cvec)

Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv=107351 (pad 256). 200 iters, médiane. MTP-OFF. Launcher identique (pb auto=60 @T=1).
REF = t3lba_lb4 (feaf05aa : 17 LDG.32/block-pair) · CVEC = (f89418fc : 5 LDG.128/block-pair).
Seule diff = lecture de la rangée K q8_0 (272 B = 17 uint4 alignés 16 B). T=1 = le cas decode prod ; T=2..8 : le load K est partagé par tile → gain attendu partout, relatif variable.

| T | NT | REF µs | CVEC µs | CVEC vs REF | note |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 451 | 402 | +10.7% | **cas decode prod** |
| 2 | 2 | 521 | 466 | +10.6% | MTP verify |
| 3 | 2 | 662 | 620 | +6.2% | MTP verify |
| 4 | 2 | 667 | 625 | +6.3% | MTP verify |
| 5 | 2 | 792 | 700 | +11.5% | MTP verify |
| 6 | 3 | 1204 | 1109 | +7.9% | MTP verify |
| 7 | 3 | 1216 | 1117 | +8.1% | MTP verify |
| 8 | 3 | 1194 | 1097 | +8.1% | MTP verify |

Lecture : « CVEC vs REF » = (REF − CVEC)/REF : positif = CVEC plus rapide. Porte e2e : ≥ +3 % @T=1. Validation numérique maxrel < 1e-4 requise.