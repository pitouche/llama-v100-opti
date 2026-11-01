# Verdict A — parallel_blocks forcé (2/4 waves) — 24/10 18:23 — **RÉFUTÉ**

Hypothèse user : « davantage de vagues améliore le masquage de latence » (T3 latence-bound,
NCU : 60,6 % long-scoreboard, 1 wave = 240 slots). Testée en forçant pb=120 (2 waves) et
pb=240 (4 waves) @T=1-2 contre le pb auto=60 (1 wave) comme témoin.

## Résultats (microbench mb_fa, géométrie prod MTP-OFF, n_kv=107351/pad 107520, 200 iters médiane)

| T | REF µs (pb auto) | PB120 µs | PB240 µs | 120 vs REF | 240 vs REF |
|---:|---:|---:|---:|---:|---:|
| 1 | 482 | 496 | 520 | **−2,8 %** | **−7,9 %** |
| 2 | 521 | 543 | 581 | **−4,1 %** | **−11,4 %** |
| 3 | 662 | 694 | 769 | −4,8 % | −16,1 % |
| 4 | 668 | 710 | 797 | −6,3 % | −19,3 % |
| 5 | 793 | 841 | 966 | −6,1 % | −21,8 % |
| 6 | 1205 | 1206 | 1206 | 0,0 % | 0,0 % |
| 7 | 1218 | 1218 | 1218 | 0,0 % | 0,0 % |
| 8 | 1196 | 1196 | 1196 | 0,0 % | 0,0 % |

Validation numérique REF/PB120/PB240 : maxrel = 3,99e-05 (tol 1e-4) → même attention, OK.
Invariants post-bench : service `ollama-v100` active, `.so` prod `fa3e75c2` intact, V100 libérée.

## Lecture

- **pb forcé = strictement plus lent** à T=1-5 (jusqu'à −21,8 % pour pb240 @T=5), neutre à T=6-8.
- Le pb auto de la prod (60 @T=1-2, 40 @T=3-5, 20 @T=6-8) est déjà optimal : c'est la config 1 wave.
- Plus de KV-split (pb↑) = plus de blocks = plus de vagues + un `flash_attn_combine_results`
  plus gros (smem combine = pb·float2, nb rows = T·qh·kvh·pb). Le coût du combine + les tail
  effects dépassent le gain d'overlap : **les vagues additionnelles ne masquent PAS la latence,
  elles l'allongent.**
- T=6-8 neutres car le pb forcé (120/240) y est clampé sous la grille qui y est déjà
  saturée (grille 160/120 blocks, pas de 2ᵉ vague exploitable).

## Conclusion

**Levier A : RÉFUTÉ.** On ne force pas `parallel_blocks`. Le pb auto de la prod reste la
référence. Le mur n'est pas le dispatch de vagues : c'est bien l'occupancy résidente
(3 blocs/SM, 12 warps/SM) qui plafonne le masquage. → On passe au levier B, seul, qui
attaque directement ce plafond d'occupancy (4 blocs/SM via launch_bounds).
