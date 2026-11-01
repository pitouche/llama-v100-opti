# Protocole A/B — levier B : launch_bounds NT=1 → minblocks=4 — 24/10

**Contexte (NCU T=1, prod fa3e75c2)** : T3 est latence-bound par son occupancy résidente :
18,75 % (12 warps/SM), plafonné **doublement** — Block Limit Registers = 3 (162 regs) ET
Block Limit Shared Mem = 3 (smem 32,77 KB / 8,61 KB = 3,7 → 3 blocs). Le masquage du
long-scoreboard (60,6 %) ne peut pas monter au-delà de 12 warps/SM.

**Décision user** : B testé indépendamment (A réfuté, voir `t3_pb_A_verdict_20261024.md`),
en vérifiant spills/registres réellement utilisés.

## Variantes (2 lancements `mb_fa`, MTP-OFF, géométrie prod)

| variante | .so (MB_FA_SO) | NT=1 launch_bounds | regs NT=1 | spills | blocs/SM |
|---|---|---|---:|---:|---:|
| REF | prod `fa3e75c2` | minblocks=3 (lba) | 162 | 0 | 3 |
| LB4 | `h2lib/lib/libggml-cuda-t3lba_lb4.so` (`feaf05aa`) | minblocks=4 | **124** | **0** | 4 |

- **Seule différence** : `fattn_vs_minblocks<D,NT>()` retourne 4 au lieu de 3 pour NT=1.
  Kernel, smem, launcher, pb auto : identiques à la prod.
- **Vérification ptxas (build, instance q8_0-q8_0-d256)** : LB4 NT=1 = 124/128 regs,
  0 spill, 0 stack (vs 162/166 REF). Les 4 bool-instances (softcap×skip_masked) à 124-128.
- **Pourquoi ça passe côté smem** : la prod tourne à 3 blocs/SM alors que smem/block=8,61 KB
  (carveout 32 KB) → le carveout SM réel est déjà 96 KB (V100 : 4×8,61=34,4 KB < 96 KB).
  Le ceiling smem actuel vient des 32 KB de *config* ; 4 blocs = 34,4 KB restent dans les 96 KB
  du SM (vérifié : `MaxSharedPerSM=98304`, `MaxSharedPerBlock=49152` ≥ 8,61 KB).
- **Seul T=1 (NT=1) change** : NT≥2 (T≥2) garde minblocks=2 dans les deux variants →
  T=2..8 = contrôle (doit rester ~0 ; toute déviation = autre effet).
- pb auto identique dans les deux (launcher non patché, pas d'env GGML_CUDA_FATTN_PB).

## Grille de lecture (T=1 = le cas decode single-token, cible du levier)

- **Gain ≥ 3 % @T=1** (et T=2..8 neutres) → B validé : 4 blocs/SM (16 warps/SM, 25 %)
  masquent mieux le long-scoreboard ; on documente + on envisage prod (avec NCU de confirmation).
- **Neutre (±2 %)** → l'occupancy n'était pas le goulot dominant (ou les regs 124 vs 162
  dégradent le ILP par block) → B classé non-prioritaire.
- **Pire** → le recompile à 124 regs a cassé l'unrolling du KV loop → B rejeté.
- T=2..8 **doivent rester ~0** ; si ça bouge, c'est un bug de wiring (les kernels NT≥2
  sont bit-identiques entre REF et LB4 — vérifier le md5 des instances NT≥2 dans les 2 .so).

## Invariants post-bench
service `ollama-v100` active · `.so` prod `fa3e75c2` intact · V100 libérée · aucun bench MTP.

## Artefacts
- Script : `bench/mb_fa_lb4_bench.py` ; sortie `mb_fa_lb4_results.jsonl` + `mb_fa_lb4_summary.md`.
- Build : `build_t3v2.sh t3lba_lb4 '' '' fattn-mma-volta-small.lba4.cuh` (lba4.cuh = lba.cuh
  avec `NT<=1 ? 4 : 2`).
