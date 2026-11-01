# Microbench FA T3 — T3 v2 (levier D) : load V q8_0 vectorisé 128-bit

Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv=107351 (pad 256). 200 iters, médiane. MTP-OFF. Launcher identique (pb auto=60 @T=1).
REF = prod (4fd152f9) · V2 = candidat dual (0653dc50). Seule diff = lecture de la fenêtre V q8_0 (scale + quants de la lane) dans fattn_vs_v_row.
Validation : maxrel < 1e-4 requis ; bit-identique = True.

| T | NT | REF µs | V2 µs | V2 vs REF | note |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 404 | 825 | -104.0% | **cas decode prod** |
| 2 | 2 | 468 | 847 | -80.9% | MTP verify |
| 3 | 2 | 622 | 894 | -43.8% | MTP verify |
| 4 | 2 | 626 | 901 | -44.0% | MTP verify |
| 5 | 2 | 702 | 1009 | -43.6% | MTP verify |
| 6 | 3 | 1111 | 1631 | -46.8% | MTP verify |
| 7 | 3 | 1119 | 1638 | -46.4% | MTP verify |
| 8 | 3 | 1097 | 1630 | -48.6% | MTP verify |

Lecture : « V2 vs REF » = (REF − V2)/REF : positif = V2 plus rapide. Porte e2e : gain @T=1 ≥ +2 % pour déclencher l'A/B applicatif.