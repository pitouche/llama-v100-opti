# Microbench FA T3 — levier B : launch_bounds NT=1 minblocks=4 (124 regs)

Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv=107351 (pad 256). 200 iters, médiane. MTP-OFF. Launcher identique (pb auto=60 @T=1).
REF = prod fa3e75c2 (lba : NT=1 minblocks=3, 162 regs, 3 blocs/SM). LB4 = t3lba_lb4 (feaf05aa : NT=1 minblocks=4, 124 regs, 0 spill, 4 blocs/SM).
**Seul T=1 (NT=1) change** ; NT>=2 (T>=2) garde minblocks=2 dans les deux -> contrôle.

| T | NT | REF µs (162r) | LB4 µs (124r) | LB4 vs REF | note |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 482 | 450 | +6.8% | **levier B actif** |
| 2 | 2 | 521 | 520 | +0.2% | contrôle (NT≥2 identique) |
| 3 | 3 | 662 | 662 | +0.0% | contrôle (NT≥2 identique) |
| 4 | 3 | 667 | 668 | -0.2% | contrôle (NT≥2 identique) |
| 5 | 4 | 793 | 794 | -0.1% | contrôle (NT≥2 identique) |
| 6 | 3 | 1205 | 1207 | -0.2% | contrôle (NT≥2 identique) |
| 7 | 3 | 1218 | 1219 | -0.1% | contrôle (NT≥2 identique) |
| 8 | 3 | 1196 | 1196 | +0.0% | contrôle (NT≥2 identique) |

Lecture : « LB4 vs REF » = (REF − LB4)/REF : positif = LB4 plus rapide. Seul T=1 peut gagner ; T=2..8 doivent rester ~0 (sinon = autre effet en jeu).