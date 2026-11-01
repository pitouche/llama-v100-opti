# Décision n_max MTP — balayage {2, 3, 4} — CLOSÉ (30/10)

**Expérience close. n_max=3 reste la config par défaut du script de bench et du
service prod. Aucune config à changer.**

## Contexte figé (identique pour les 3 runs)

- V100 (bench, service `ollama-v100`, port bench 44921), non-prod ; toute activité
  tok/s observée = nos benchs. Docs autoritaires : `supervisor_bench_README.md` +
  `GPU_UUID_NSYS_CUPTI_20261030.md`.
- Model `qwen3.8-27b-q6-mtp-vision`, .so `8035caf4`, ctx prompt **112060 tok**,
  b2048, KV q8_0, H2 OFF, flash-attn, `--spec-type draft-mtp
  --spec-draft-backend-sampling` (MTP ON).
- Seul changement entre runs : `--spec-draft-n-max` (line ~468 de
  `prof_prd_profile_20261030.py`, fonction `launch_server()`).
- Fenêtres : decode 160 tok MTP ON / 96 tok OFF ; même nsys flags (CUPTI 12.9,
  `--cuda-graph-trace=node`). Lancement via `supervisor_bench.py run … --detach`
  (~17 min, 4 fenêtres).
- **Run de référence figé** : `runs/prof-prd-20261030-114948/` (n_max=3), intact.

## Résultats (critère ultime : decode tok/s mesuré)

| n_max | run | decode tok/s | × vs OFF (21.6) | acceptance (per-pos) | mean len (tok/verify) | busy verify | wall/verify |
|---|---|---:|---:|---|---:|---:|---:|
| 2 | 135141 | 41.94 (23.8 ms/tok) | ×1.95 | 0.825 (99/120) — 0.883/0.767 | 2.65 | 53.1 ms | 69.3 ms |
| **3** | **114948** | **44.18 (22.6 ms/tok)** | **×2.05** | 0.711 (108/152) — 0.902/0.706/0.510 | **3.12** | 56.3 ms | 79.6 ms |
| 4 | 142401 | 33.86 (29.5 ms/tok) | ×1.57 | 0.701 (115/164) — 0.951/0.732/0.610/0.512 | 3.80 | 63.4 ms | 147.2 ms* |

\* décomposition n_max=4 : segmentation artefactuelle de `analyze_prd_profile.py`
(24 verifies détectés pour 155 tok → le « gap draft→verify 31.1 ms » et
wall/launch 24.5 ms sont **surestimés**, pas un coût physique). Le **mesuré**
fait foi : 33.86 t/s, −23 % vs n3.

## Pourquoi n3 est le point d'équilibre (le « knee »)

Courbe d'acceptance par position mesurée : **pos1 ≈ 0.90–0.95, pos2 ≈ 0.71–0.77,
pos3 ≈ 0.51–0.61, pos4 ≈ 0.51**.

- **n_max=2 (−5 %)** : passer de 4→3 positions de verify ne retire que
  **3.2 ms busy** (56.3→53.1) — le matvec/attention est batché, seule une
  fraction scale avec les positions — et 1 draft de moins (148→117, −21 %,
  −~2.3 ms). Le cycle ne tombe que de 13 % (79.6→69.3 ms) alors que
  E[tok/verify] tombe de 15 % (3.12→2.65). La perte d'espérance > le gain de
  cycle → perte nette. L'hypothèse initiale « −25 % du temps de verify »
  (NMAX2_RUN_CONFIG.md) était trop optimiste : un verify 3 pos coûte presque
  aussi cher qu'un 4 pos.
- **n_max=4 (−23 %)** : le 4ᵉ draft n'apporte que **~51 % d'acceptance**
  (identique au 3ᵉ de n3) mais coûte un draft supplémentaire séquentiel
  (2.3 ms) **et** élargit le verify à 5 positions (+7 ms busy, 56.3→63.4).
  Gain marginal déficitaire → effondrement du t/s.

## État final (vérifié 30/10 ~14:48)

- `prof_prd_profile_20261030.py` : `--spec-draft-n-max "3"` (line 468) —
  **restauré**, le script repart sur la référence.
- `ollama-v100` active ; V100 (UUID `GPU-b3c229d2-2803-0af9-2fb8-067708067c1b`,
  smi id=1) : runner prod seul, pas de résidu bench ; 1070 inchangée.
- `nsys status` propre, lock `/tmp/kolbo-supervisor-bench.lock` libre,
  pas de résidu process (nsys/sampler).
- Rapports : `prof_prd_analysis_20261030-{114948,135141,142401}.md`
  (run 114948 = réf figée, 135141 = n2, 142401 = n4).
- Artefacts nsys : `prof_prd_*_{135141,142401}.{nsys-rep,sqlite}` dans `bench/`
  + `runs/prof-prd-20261030-{135141,142401}/` (results.json, logs serveur,
  samplers, perf).

## Correctif outillage fait pendant l'expé

`analyze_prd_profile.py` portait la référence **hardcodée** dans le rapport
(« cohérent avec 44.18 tok/s mesurés » + section acceptance 0.711/108/152/
0.902-0.706-0.510), donc n'importe quel nouveau run affichait les chiffres de
114948. Corrigé : tok/s mesurés lus de `results.json`, acceptance parsée du
dernier `draft acceptance` du log serveur (per-pos générique 2/3/4 valeurs).
Les 3 rapports ont été régénérés et sont cohérents avec leurs données.

## Caveat pour la généralisation

L'acceptance est **workload-dépendante**. Le verdict « n3 optimum » est établi
sur CE prompt (ctx ~112k, style narratif). Sur un workload plus « facile »
(acceptance pos1/pos2 > 0.9), le knee pourrait se décaler vers n4 (la 4ᵉ pos
pourrait alors payer son draft) ; sur un workload plus dur, vers n2. À
re-valider par un petit balayage sur d'autres prompts si la question revient —
mais pour l'état prod actuel : **n_max=3, point final**.
