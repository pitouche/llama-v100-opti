# Analyse — bench GENERATION (decode tg) vs longueur du contexte
Généré le 2026-10-11 19:32:46 par bench_gen.py

Setup : Qwen3.8-27B Q6_K + MTP draft=3 + vision — V100-PCIE-32GB — llama-server direct, KV q8_0, flash-attn, -c 204800, -b 512 -ub 512, 1 slot, context-shift, keep 4 (mêmes args que le run prefill).

Métrique : `eval time` du log (DECODE) / tokens générés (tg).
Exclus du calcul : préfill (prompt eval time), restauration du checkpoint, chargement du modèle, réseau/API, temps avant 1er token.

## Tableau
| contexte cible | contexte réel (log) | restored | générés (tok) | gen time (s) | **tg (t/s)** | ms/tok | MTP acc (aux) | mean len (aux) |
|---|---|---|---|---|---|---|---|---|
| 2048 | 2083 | None | 300 | 7.73 | **39** | 25.78 | 0.565 | 2.7  |
| 32768 | 32309 | 1571 | 300 | 7.03 | **43** | 23.43 | 0.768 | 3.3  |

## Courbe — tg (t/s) vs contexte (n_past)
```
   39 |███████████████████           | contexte 2083
   43 |█████████████████████         | contexte 32309
```
(1 █ = 2 t/s)

## Réponse à la question

Qwen3.8-27B Q6_K sur cette V100 — génération : 2k → **39 t/s**, 32k → **43 t/s**.

## Méthode & limites
- 300 tokens générés par point (spéc : ≥256) ; un point avec gen < 256 (stop précoce/EOS) est marqué INVALID et rejoué 1x automatiquement.
- tg est calculé sur l'ensemble des tokens générés : le contexte croît de +300 tok pendant la génération (2k → 2,3k, etc.) — la vitesse moyenne du point en reflète l'évolution interne.
- Le contexte est rempli par la chaîne WARM (corpus B) : le préfill de chaque point est une diff restaurée (colonne 'restored'), EXCLU de la métrique. Le KV des tokens générés au point N est ignoré au point N+1 (restore sur le préfixe commun) → pas de contaminant.
- Texte : corpus warm + temp=0 top_k=1 + ignore_eos (déterministe) : mesure de bande passante du decode, pas de mesure de qualité. ignore_eos est obligatoire : sans, le modèle émet l'EOS après "OK-BENCH." et stoppe à ~57 tok (< 256). Après l'EOS, la suite générée est du texte dégénéré répéteur — sans incidence sur la bande passante mesurée.
- MTP : acceptance/mean len rapportés en auxiliaire uniquement — AUCUNE conclusion MTP n'est tirée de ce bench (sur texte dégénéré l'acceptance est biaisée).
- Les mesures de préfill restent dans bench_direct_results.jsonl / bench_direct_summary.md — rien n'est mélangé ici.
