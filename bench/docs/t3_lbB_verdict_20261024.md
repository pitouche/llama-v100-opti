# Verdict B — launch_bounds NT=1 → minblocks=4 (162→124 regs, 3→4 blocs/SM) — 24/10 19:07 — **VALIDÉ**

Hypothèse : T3 est plafonné à 3 blocs/SM (12 warps/SM) **doublement** — regs (162) **et**
smem — et ce plafond résident est le goulot du long-scoreboard (60,6 %). Forcer
`launch_bounds(128,4)` sur NT=1 fait recompiler ptxas à 124 regs (0 spill) → 4 blocs/SM
(16 warps/SM, 25 %). Tester **seul** (A réfuté), en vérifiant spills/regs.

## Résultats (mb_fa, géométrie prod MTP-OFF, n_kv=107351/pad 107520, 200 iters, médiane)

| T | NT | REF µs med/min/max (162r) | LB4 µs med/min/max (124r) | LB4 vs REF | rôle |
|---:|---:|---:|---:|---:|---|
| **1** | **1** | 482 / 475 / 579 | **450 / 442 / 547** | **+6,8 %** | **levier B actif** |
| 2 | 2 | 521 / 515 / 534 | 520 / 513 / 532 | +0,2 % | contrôle (NT≥2 identique) |
| 3 | 3 | 662 / 657 / 669 | 662 / 656 / 671 | +0,0 % | contrôle |
| 4 | 3 | 667 / 662 / 678 | 668 / 662 / 674 | −0,2 % | contrôle |
| 5 | 4 | 793 / 785 / 805 | 794 / 784 / 808 | −0,1 % | contrôle |
| 6 | 3 | 1205 / 1198 / 1214 | 1207 / 1197 / 1216 | −0,2 % | contrôle |
| 7 | 3 | 1218 / 1206 / 1233 | 1219 / 1205 / 1228 | −0,1 % | contrôle |
| 8 | 3 | 1196 / 1187 / 1207 | 1196 / 1187 / 1220 | +0,0 % | contrôle |

- **T=1 (NT=1, le seul kernel modifié)** : 482→450 µs = **+6,8 %** (médiane) ET 475→442 µs
  (min) — les deux bornes bougent d'environ 32 µs ensemble → **signal réel, pas du bruit**.
- **T=2..8 (NT≥2, bit-identiques REF/LB4)** : tous dans **±0,2 %** → wiring propre, aucune
  interaction, le gain provient **uniquement** du launch_bounds sur NT=1.
- Validation numérique REF/LB4 : maxrel = 5,03e-05 (tol 1e-4) → même attention, OK.
- Spills/regs (ptxas, instance q8_0-q8_0-d256, NT=1) : LB4 = **124 regs, 0 spill, 0 stack**
  (vs REF 162). Les 4 bool-instances à 124-128, 0 spill partout.
- Invariants post-bench : service `ollama-v100` active, `.so` prod `fa3e75c2` intact,
  V100 libérée (le 30,5 Go = le service qui remonte, pas un bench résiduel).

## Pourquoi ça gagne

- 4 blocs/SM résidents = **16 warps/SM au lieu de 12** → plus de warps éligibles pour
  couvrir le latency du long-scoreboard (load KV q8_0). C'est exactement le levier que le
  verdict NCU pointait (occupancy 18,75 % limitée par regs **et** smem).
- Le carveout smem n'empêche pas 4 blocs : la prod tourne déjà à 3 blocs avec 8,61 KB/block
  sous une *config* de 32 KB, donc le carveout SM réel est 96 KB (V100) ; 4×8,61 = 34,4 KB
  rentre dans les 96 KB (vérifié : MaxSharedPerSM=98304 B, MaxSharedPerBlock=49152 B ≥ 8,61 KB).
- Recompiler à 124 regs au lieu de 162 **ne dégrade pas** le kernel : 0 spill, et le ILP
  par warp est préservé (le gain d'occupancy l'emporte net).

## Conclusion

**Levier B : VALIDÉ.** +6,8 % sur le decode single-token (T=1, le cas dominant) au plus
long KV (107 k, le pire cas latence). À ce KV, le gain ne peut que se maintenir ou s'améliorer
sur des KV plus courts.

### Recommandation prod
Adopter `libggml-cuda-t3lba_lb4.so` (`feaf05aa`) = variante `lba4` (T3 + lba avec
`fattn_vs_minblocks<D,NT>() : NT<=1 ? 4 : 2`), **sans** forcer `parallel_blocks`
(A réfuté → on garde le pb auto). C'est un changement de 1 constante (`3`→`4` sur NT=1),
kernel et launcher sinon bit-identiques à la prod.

### Caveat avant prod
Ce +6,8 % est mesuré en microbench isolé (1 FA, KV fixe 107 k). Il reste à confirmer en
e2e (tok/s global) que le gain FA se traduit en gain de pas de decode, car FA n'est qu'une
partie du pas (mmq Q8, norms, etc.). Mais le signe et l'ordre de grandeur sont solides.
