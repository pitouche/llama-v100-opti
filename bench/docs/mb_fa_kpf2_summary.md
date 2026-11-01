# Microbench FA T3 — levier ILP/prefetch K : prefetch.global.L2 (CCTL.E.PF2)

Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv=107351 (pad 256). 200 iters, médiane. MTP-OFF. Launcher identique (pb auto=60 @T=1).
REF = prod (4fd152f9 = B+C) · KPF2 = candidat offset corrigé (1a8ba52e = prefetch des rangées K de l'itération SUIVANTE, 3 CCTL.E.PF2/lane, 0 reg, 0 smem, kernel prod REG:128 STACK:0) · KPF(v1) = 7a87cf78 (prefetch des rangées COURANTES — référence du lead time) · NOB = e19b6112 (sans B).
Validation : KPF2 maxrel=0.00e+00, KPF maxrel=0.00e+00, bit-identique=True (attendu True — cache only) ; NOB maxrel=5.03e-05 (drift -use_fast_math, tol 1e-4).

| T | NT | REF µs | KPF2 µs | KPF2 vs REF | KPF(v1) µs | KPF vs REF | NOB µs | NOB vs REF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 404 | 462 | -14.2% | 410 | -1.3% | 433 | -7.1% |
| 2 | 2 | 468 | 514 | -9.8% | 472 | -0.9% | 466 | +0.4% |
| 3 | 2 | 622 | 653 | -5.1% | 614 | +1.2% | 620 | +0.2% |
| 4 | 2 | 626 | 660 | -5.4% | 618 | +1.3% | 625 | +0.2% |
| 5 | 2 | 702 | 737 | -5.0% | 695 | +1.0% | 701 | +0.2% |
| 6 | 3 | 1111 | 1143 | -2.9% | 1111 | +0.0% | 1109 | +0.2% |
| 7 | 3 | 1117 | 1151 | -3.0% | 1118 | -0.1% | 1117 | +0.0% |
| 8 | 3 | 1097 | 1132 | -3.2% | 1102 | -0.5% | 1097 | +0.0% |

Lecture : « X vs REF » = (REF − X)/REF : positif = X plus rapide. **Seuil GO : >= +2 % @T=1 sans régression T=2..8.** KPF(v1) = prefetch sans lead time (lignes de l'itération courante, déjà en vol) ; KPF2 = prefetch avec lead time = 1 itération (lignes de l'itération suivante). NOB = coût du revert du levier B (−7,1 % @T=1, mesuré 28/10).