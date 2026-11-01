# Microbench FA T3 — levier ILP/prefetch K : prefetch.global.L2 (CCTL.E.PF2) des rangées K de l'itération N+1

Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv=107351 (pad 256). 200 iters, médiane. MTP-OFF. Launcher identique (pb auto=60 @T=1).
REF = prod (4fd152f9 = B+C) · KPF = candidat (7a87cf78 = REF + 3 prefetch.global.L2/lane/l'itération, 0 reg, 0 smem, kernel prod exact REG:128 STACK:0) · NOB = rappel (e19b6112 = sans B, 3 CTA/SM).
Validation : KPF maxrel=0.00e+00, bit-identique=True (attendu True — cache only) ; NOB maxrel=5.03e-05 (drift -use_fast_math, tol 1e-4).

| T | NT | REF (B+C) µs | KPF µs | KPF vs REF | NOB µs | NOB vs REF | note |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 404 | 411 | -1.5% | 433 | -7.1% | **cas decode prod** |
| 2 | 2 | 468 | 472 | -0.9% | 466 | +0.4% | MTP verify |
| 3 | 2 | 622 | 614 | +1.2% | 620 | +0.2% | MTP verify |
| 4 | 2 | 626 | 618 | +1.2% | 625 | +0.2% | MTP verify |
| 5 | 2 | 702 | 695 | +1.0% | 701 | +0.2% | MTP verify |
| 6 | 3 | 1111 | 1111 | +0.0% | 1109 | +0.2% | MTP verify |
| 7 | 3 | 1119 | 1118 | +0.1% | 1117 | +0.2% | MTP verify |
| 8 | 3 | 1097 | 1102 | -0.5% | 1097 | +0.0% | MTP verify |

Lecture : « KPF vs REF » = (REF − KPF)/REF : positif = KPF plus rapide. **Seuil GO : >= +2 % @T=1 sans régression T=2..8.** NOB rappel = coût du revert du levier B (−7,1 % @T=1, mesuré 28/10).