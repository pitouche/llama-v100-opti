# Analyse — bench GENERATION (decode tg) vs longueur du contexte
Généré le 2026-10-02 22:02:27 par bench_gen.py

Setup : Qwen3.8-27B Q6_K + vision — **MTP DÉSACTIVÉ** (vanilla decode) — V100-PCIE-32GB — llama-server direct, KV q8_0, flash-attn, -c 204800, -b 512 -ub 512, 1 slot, context-shift, keep 4 (mêmes args que le run prefill, sans les 3 args --spec-*).
**Kernels V100 optimisés DÉSACTIVÉS** (env : VOLTA_Q6_W4R4=0, Q5_X4=0, Q8_FATTN_TC=0, GQA8_NCOLS2 unset, ALLREDUCE unset) — même binaire, même GGUF, mêmes args : la seule différence est le gating env des kernels Volta.

Métrique : `eval time` du log (DECODE) / tokens générés (tg).
Exclus du calcul : préfill (prompt eval time), restauration du checkpoint, chargement du modèle, réseau/API, temps avant 1er token.

## Tableau
| contexte cible | contexte réel (log) | restored | générés (tok) | gen time (s) | **tg (t/s)** | ms/tok | MTP acc (aux) | mean len (aux) |
|---|---|---|---|---|---|---|---|---|
| 2048 | 2083 | None | 300 | 11.76 | **26** | 39.21 | · | ·  |
| 16384 | 16198 | 1571 | 300 | 12.47 | **24** | 41.58 | · | ·  |
| 32768 | 32309 | 15686 | 300 | 13.52 | **22** | 45.07 | · | ·  |
| 65536 | 64339 | 31797 | 300 | 15.64 | **19** | 52.14 | · | ·  |
| 98304 | 96356 | 63827 | 300 | 18.02 | **17** | 60.08 | · | ·  |
| 131072 | 128437 | 95844 | 300 | 20.61 | **15** | 68.70 | · | ·  |
| 163840 | 160530 | 127925 | 300 | 23.21 | **13** | 77.36 | · | ·  |
| 196608 | 192646 | 160018 | 300 | 25.91 | **12** | 86.36 | · | ·  |

## Courbe — tg (t/s) vs contexte (n_past)
```
   26 |█████████████                 | contexte 2083
   24 |████████████                  | contexte 16198
   22 |███████████                   | contexte 32309
   19 |██████████                    | contexte 64339
   17 |████████                      | contexte 96356
   15 |███████                       | contexte 128437
   13 |██████                        | contexte 160530
   12 |██████                        | contexte 192646
```
(1 █ = 2 t/s)

## Réponse à la question

Qwen3.8-27B Q6_K sur cette V100 — génération : 2k → **26 t/s**, 16k → **24 t/s**, 32k → **22 t/s**, 64k → **19 t/s**, 96k → **17 t/s**, 128k → **15 t/s**, 160k → **13 t/s**, 192k → **12 t/s**.

## Méthode & limites
- 300 tokens générés par point (spéc : ≥256) ; un point avec gen < 256 (stop précoce/EOS) est marqué INVALID et rejoué 1x automatiquement.
- tg est calculé sur l'ensemble des tokens générés : le contexte croît de +300 tok pendant la génération (2k → 2,3k, etc.) — la vitesse moyenne du point en reflète l'évolution interne.
- Le contexte est rempli par la chaîne WARM (corpus B) : le préfill de chaque point est une diff restaurée (colonne 'restored'), EXCLU de la métrique. Le KV des tokens générés au point N est ignoré au point N+1 (restore sur le préfixe commun) → pas de contaminant.
- Texte : corpus warm + temp=0 top_k=1 + ignore_eos (déterministe) : mesure de bande passante du decode, pas de mesure de qualité. ignore_eos est obligatoire : sans, le modèle émet l'EOS après "OK-BENCH." et stoppe à ~57 tok (< 256). Après l'EOS, la suite générée est du texte dégénéré répéteur — sans incidence sur la bande passante mesurée.
- MTP DÉSACTIVÉ pour ce run : colonne acc = — (aucune ligne draft acceptance dans le log) ; c'est le run de comparaison vanilla.
- Les mesures de préfill restent dans bench_direct_results.jsonl / bench_direct_summary.md — rien n'est mélangé ici.
