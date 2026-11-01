# Bench direct — synthèse (à laisser à l'assistant pour dépouillement)
Généré le 2026-10-02 09:48:07 par bench_direct.py

## A. COLD — coût d'un préfill complet
'préfill réel' = tokens réellement pré-évalués selon le log du runner (restore ≈ 0 vérifié). 't/s api' = prompt_tokens(total)/durée : biaisé à la hausse s'il reste du cache (artefact connu du 01/10).

| phase | C cible (tok) | prompt (api) | restore | préfill réel (tok) | préfill (s) | t/s log | ms/tok | t/s api | VRAM (GiB) |
|---|---|---|---|---|---|---|---|---|---|
| direct | 2048 | 2084 | 42 | 2042 | 2.51 | 813 | 1.23 | 744 | 31.4 |
| direct | 4096 | 4131 | 42 | 4089 | 4.83 | 846 | 1.18 | 807 | 31.4 |
| direct | 8192 | 8147 | 42 | 8105 | 9.64 | 841 | 1.19 | 820 | 31.4 |
| direct | 16384 | 16284 | 42 | 16242 | 19.96 | 814 | 1.23 | 802 | 31.4 |
| direct | 32768 | 32508 | 42 | 32466 | 43.39 | 748 | 1.34 | 744 | 31.4 |
| direct | 65536 | 64608 | 42 | 64566 | 101.05 | 639 | 1.56 | 637 | 31.4 |
| direct | 98304 | 97003 | 42 | 96961 | 175.09 | 554 | 1.81 | 553 | 31.4 |
| direct | 131072 | 129642 | 42 | 129600 | 265.25 | 489 | 2.05 | 488 | 31.4 |
| direct | 163840 | 162209 | 42 | 162167 | 369.26 | 439 | 2.28 | 439 | 31.4 |
| direct | 196608 | 194761 | 42 | 194719 | 489.74 | 398 | 2.52 | 397 | 31.4 |
| ollama | 16384 | 16284 | 42 | 16242 | 18.95 | 857 | 1.17 | 846 | 31.4 |
| ollama | 65536 | 64608 | 42 | 64566 | 97.66 | 661 | 1.51 | 659 | 31.4 |
| ollama | 131072 | 129642 | 42 | 129600 | 259.96 | 498 | 2.01 | 498 | 31.4 |
| ollama | 196608 | 194761 | 42 | 194719 | 482.17 | 404 | 2.48 | 403 | 31.4 |

## B. WARM — coût marginal : Δ prefilled quand C_prev est déjà en KV
C'est LA mesure pour la viabilité du contexte 200k conservé.

| phase | C_prev (tok) | C final (tok) | Δ attendu | restore (n_past) | diff réelle (tok) | diff (s) | t/s chaud | ms/tok | VRAM (GiB) |
|---|---|---|---|---|---|---|---|---|---|
| direct | · | 2048 | 2048 | 42 | 2041 | 2.57 | 793 | 1.26 | 31.4 |
| direct | 2048 | 4096 | 2048 | 1571 | 2533 | 3.14 | 806 | 1.24 | 31.4 |
| direct | 4096 | 8192 | 4096 | 3592 | 4582 | 5.68 | 806 | 1.24 | 31.4 |
| direct | 8192 | 16384 | 8192 | 7662 | 8536 | 11.16 | 765 | 1.31 | 31.4 |
| direct | 16384 | 32768 | 16384 | 15686 | 16623 | 24.30 | 684 | 1.46 | 31.4 |
| direct | 32768 | 65536 | 32768 | 31797 | 32542 | 59.45 | 547 | 1.83 | 31.4 |
| direct | 65536 | 98304 | 32768 | 63827 | 32529 | 74.24 | 438 | 2.28 | 31.4 |
| direct | 98304 | 131072 | 32768 | 95844 | 32593 | 89.07 | 366 | 2.73 | 31.4 |
| direct | 131072 | 163840 | 32768 | 127925 | 32605 | 103.68 | 314 | 3.18 | 31.4 |
| direct | 163840 | 196608 | 32768 | 160018 | 32628 | 118.42 | 276 | 3.63 | 31.4 |
| ollama | · | 32768 | 32768 | 42 | 32267 | 41.81 | 772 | 1.30 | 31.4 |
| ollama | 32768 | 98304 | 65536 | 31797 | 64559 | 129.56 | 498 | 2.01 | 31.4 |
| ollama | 98304 | 163840 | 65536 | 95844 | 64686 | 188.96 | 342 | 2.92 | 31.4 |

## C. OLLAMA vs DIRECT (points communs, t/s du log)

| point | direct | ollama | Δ ollama |
|---|---|---|---|
| cold 16384 | 814 | 857 | +5.3% |
| cold 65536 | 639 | 661 | +3.5% |
| warm 98304 | 438 | 498 | +13.7% |
| cold 131072 | 489 | 498 | +2.0% |
| warm 163840 | 314 | 342 | +8.8% |
| cold 196608 | 398 | 404 | +1.6% |

## D. MTP (draft) — acceptance par run

| run | phase | acceptance | mean len | VRAM (GiB) |
|---|---|---|---|---|
| cold_2048 | direct | 1.000 | 3.0 | 31.4 |
| cold_4096 | direct | 1.000 | 3.0 | 31.4 |
| cold_8192 | direct | 1.000 | 3.0 | 31.4 |
| cold_16384 | direct | 1.000 | 3.0 | 31.4 |
| cold_32768 | direct | 1.000 | 3.0 | 31.4 |
| cold_65536 | direct | 1.000 | 3.0 | 31.4 |
| cold_98304 | direct | 1.000 | 3.0 | 31.4 |
| cold_131072 | direct | 1.000 | 3.0 | 31.4 |
| cold_163840 | direct | 1.000 | 3.0 | 31.4 |
| cold_196608 | direct | 0.500 | 2.0 | 31.4 |
| warm_2048 | direct | 1.000 | 3.0 | 31.4 |
| warm_4096 | direct | 1.000 | 3.0 | 31.4 |
| warm_8192 | direct | 1.000 | 3.0 | 31.4 |
| warm_16384 | direct | 0.500 | 2.0 | 31.4 |
| warm_32768 | direct | 0.500 | 2.0 | 31.4 |
| warm_65536 | direct | 1.000 | 3.0 | 31.4 |
| warm_98304 | direct | 1.000 | 3.0 | 31.4 |
| warm_131072 | direct | 1.000 | 3.0 | 31.4 |
| warm_163840 | direct | 1.000 | 3.0 | 31.4 |
| warm_196608 | direct | 0.500 | 2.0 | 31.4 |
| ocold_16384 | ollama | 1.000 | 3.0 | 31.4 |
| ocold_65536 | ollama | 1.000 | 3.0 | 31.4 |
| ocold_131072 | ollama | 0.500 | 2.0 | 31.4 |
| ocold_196608 | ollama | 0.500 | 2.0 | 31.4 |
| owarm_32768 | ollama | 0.500 | 2.0 | 31.4 |
| owarm_98304 | ollama | 1.000 | 3.0 | 31.4 |
| owarm_163840 | ollama | 1.000 | 3.0 | 31.4 |

## E. Notes de méthode
- Cold : chaque point a un texte aléatoire dédié (seed par taille) → aucun préfixe commun entre points → le checkpoint/LCP du runner ne restaure rien (restored n_past ≈ 0 vérifié). Si un point COLD a restore > 200, il est invalide.
- Warm : même corpus croissant (B, distinct des corpus cold) → le runner restaure le checkpoint au préfixe commun et ne prefille que la diff ; la colonne 'diff réelle' vient du log (prompt eval time).
- C est la cible en TOKENS ; le remplissage est dimensionné par les ratios tok/mot mesurés (R_COLD=1.45, R_WARM=1.16, bord haut) → C réel légèrement sous la cible ; les colonnes prompt/diff réelle donnent les tokens exacts mesurés.
- Run direct = MÊME build de llama-server, MÊME GGUF, MÊMES args que le runner ollama (-c 204800, q8_0 KV, flash-attn auto, MTP draft=3, -b 512 -ub 512, 1 slot, context-shift, keep 4).
- VRAM : la pool KV est pré-allouée au contexte max au démarrage du server → la colonne est un contrôle (constante attendue), pas une mesure de croissance.
- Le premier point ollama inclut ~25 s de rechargement du modèle dans wall_s (pas dans les t/s du log).
- Lignes brutes : /home/guile/llama-experiments/bench/bench_direct_results.jsonl
