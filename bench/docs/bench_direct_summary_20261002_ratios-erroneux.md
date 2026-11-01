# Bench direct — synthèse (à laisser à l'assistant pour dépouillement)
Généré le 2026-10-02 07:01:39 par bench_direct.py

## A. COLD — coût d'un préfill complet
'préfill réel' = tokens réellement pré-évalués selon le log du runner (restore ≈ 0 vérifié). 't/s api' = prompt_tokens(total)/durée : biaisé à la hausse s'il reste du cache (artefact connu du 01/10).

| phase | C cible (tok) | prompt (api) | restore | préfill réel (tok) | préfill (s) | t/s log | ms/tok | t/s api | VRAM (GiB) |
|---|---|---|---|---|---|---|---|---|---|
| direct | 2048 | 2884 | 42 | 2842 | 3.54 | 804 | 1.25 | 755 | 31.4 |
| direct | 16384 | 22929 | 42 | 22887 | 28.73 | 797 | 1.25 | 789 | 31.4 |
| ollama | 16384 | 22929 | 42 | 22887 | 27.40 | 835 | 1.20 | 828 | 31.4 |

## B. WARM — coût marginal : Δ prefilled quand C_prev est déjà en KV
C'est LA mesure pour la viabilité du contexte 200k conservé.

| phase | C_prev (tok) | C final (tok) | Δ attendu | restore (n_past) | diff réelle (tok) | diff (s) | t/s chaud | ms/tok | VRAM (GiB) |
|---|---|---|---|---|---|---|---|---|---|
| direct | · | 4096 | 4096 | 42 | 4576 | 5.42 | 844 | 1.18 | 31.4 |
| direct | 4096 | 32768 | 28672 | 4106 | 32243 | 44.48 | 725 | 1.38 | 31.4 |
| ollama | · | 32768 | 32768 | 42 | 36307 | 46.84 | 775 | 1.29 | 31.4 |

## C. OLLAMA vs DIRECT (points communs, t/s du log)

| point | direct | ollama | Δ ollama |
|---|---|---|---|
| cold 16384 | 797 | 835 | +4.8% |

## D. MTP (draft) — acceptance par run

| run | phase | acceptance | mean len | VRAM (GiB) |
|---|---|---|---|---|
| cold_2048 | direct | 1.000 | 3.0 | 31.4 |
| cold_16384 | direct | 1.000 | 3.0 | 31.4 |
| warm_4096 | direct | 1.000 | 3.0 | 31.4 |
| warm_32768 | direct | 0.500 | 2.0 | 31.4 |
| ocold_16384 | ollama | 1.000 | 3.0 | 31.4 |
| owarm_32768 | ollama | 0.500 | 2.0 | 31.4 |

## E. Notes de méthode
- Cold : chaque point a un texte aléatoire dédié (seed par taille) → aucun préfixe commun entre points → le checkpoint/LCP du runner ne restaure rien (restored n_past ≈ 0 vérifié). Si un point COLD a restore > 200, il est invalide.
- Warm : même corpus croissant (B, distinct des corpus cold) → le runner restaure le checkpoint au préfixe commun et ne prefille que la diff ; la colonne 'diff réelle' vient du log (prompt eval time).
- C est la cible en TOKENS (remplissage ≈ 1.03 tok/mot) ; les colonnes prompt/diff réelle donnent les tokens exacts mesurés.
- Run direct = MÊME build de llama-server, MÊME GGUF, MÊMES args que le runner ollama (-c 204800, q8_0 KV, flash-attn auto, MTP draft=3, -b 512 -ub 512, 1 slot, context-shift, keep 4).
- VRAM : la pool KV est pré-allouée au contexte max au démarrage du server → la colonne est un contrôle (constante attendue), pas une mesure de croissance.
- Le premier point ollama inclut ~25 s de rechargement du modèle dans wall_s (pas dans les t/s du log).
- Lignes brutes : /home/guile/llama-experiments/bench/bench_direct_results.jsonl
