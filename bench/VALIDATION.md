# SECTION BENCH — protocole de re-validation des chiffres de la pile

> Ce fichier est l'autorité pour **re-valider les chiffres publiés** dans les 6
> repos `repos/<opti>/` (les 7, dont `h2` — la mission GO→NO-GO). Chaque repo a un § « Re-validation » ; ce fichier donne
> le protocole transversal (invariants + plan de re-measurement + orchestrateur).

## 0. Ce qu'on peut et ce qu'on ne peut pas re-valider

| Niveau | Outil | Coût | Valide |
|---|---|---|---|
| **L0 — datas brutes** | lecture de `results/*.jsonl` | 0 | que les deltas sont cohérents avec les bruts (pas de re-measurement) |
| **L1 — microbench** | `mb_fa*`, `mb_mmvq` (dlopen du `.so`, pas de rebuild) | ~5-15 min | le delta **kernel** (µs/it) sur la shape cible |
| **L2 — e2e A/B** | `ab_*.py`, `t3_bench.py`, … via superviseur | ~30 min à 1 h 30 | le delta **e2e t/s** (critère final des GO) |
| **L3 — run figé nsys** | `prof_prd_profile_20261030.py` + nsys CUPTI 12.9 | ~17 min/run | la décomposition kernel (diagnostic) + decode tok/s |

**Un chiffre est « validé » si L1 et L2 sont cohérents à ±3 %** (le delta
microbench ne se transfère jamais tel quel en e2e — c'est le ratio qui compte).
L0 suffit pour auditer la cohérence interne d'un repo sans GPU.

## 1. Invariants (tous niveaux, non négociables)

1. **Superviseur obligatoire** pour tout L2/L3 :
   `python3 bench/bench/supervisor_bench.py run <bench.py> --detach`
   — le bench stoppe/restore le service lui-même, le superviseur est le filet.
   (Autorité : `bench/bench/supervisor_bench_README.md`.)
2. **GPU par UUID** (jamais d'index en dur) — `bench/bench/gpu_uuid.py`.
   Sur cette machine la V100 peut être `cuda:0` **ou** `cuda:1` selon le boot ;
   le bench doit toujours résoudre l'UUID de la V100 → index.
3. **Interleave ≥ 2 reps, serveur frais par leg, port 8917, CVD=0 (V100),
   N_GEN 64, warmup 1024.** Ordre alterné des legs (A-B-A-B, pas A-A-B-B).
4. **md5 du `.so` vérifié avant et après chaque run** (le bench le fait ;
   vérifier dans le log). Le `.so` testé doit être celui désigné par le repo
   (table md5 dans chaque README + `sofiles/README.md`).
5. **MTP OFF pour mesurer un kernel FA** (gold standard — le MTP batché masque
   le delta kernel) ; **MTP ON pour config prod**. Toujours vérifier que la
   **draft acceptance est identique entre legs** (sinon delta confondu avec MTP).
6. **Thermique** : la V100 throte après ~35 min de charge ; un bench decode après
   une passe prefill chaude donne un **faux négatif** (fa16 a failli y passer —
   `fa16_thm_*.jsonl`). **Cooldown ≥ 60 s** entre les legs decode sensibles.
7. **NSYS_CUPTI_LIBRARY_PATH=big/nsys-cupti129** pour nsys (CUPTI 13.x
   inopérant sur Volta + driver 580). Le wrapper `bench/bench/nsys_cuda.sh` le met.

## 2. Plan de re-measurement par chiffre clé

| # | Chiffre publié | Repo | Niveau | Commande (depuis la racine du meta-repo) | Référence à battre |
|---|---|---|---|---|---|
| V1 | T3 +13,2 % @114 k, +29,2 % @196 k (MTP OFF) | t3-volta-small | L2 | `python3 bench/bench/supervisor_bench.py run repos/t3-volta-small/bench/t3_bench.py --detach` | 17,15→19,41 / 28,78→37,18 t/s (T3 vs `a4b6f440`) |
| V2 | T3 kernel FA ×2,0-2,2 (microbench) | t3-volta-small | L1 | `python3 repos/t3-volta-small/bench/mb_fa_bench.py --so sofiles/libggml-cuda-t3.so.d2df07b8 --rows 1 4 8 --ctx 98304` | `results/mb_fa_baseline_d2df_20261022.jsonl` (52,8 µs rows4) vs T3 (23,8 µs) |
| V3 | lba4 +6,8 % / cvec +10,7 % / vldg +13 % (µs rows1 @98 k) | t3-volta-small | L1 | `python3 repos/t3-volta-small/bench/mb_fa_t3v2_bench.py` (ou `mb_fa_vldg_bench.py`) | 14,55 → 13,56 → 13,00 → 12,65 µs |
| V4 | H2 OFF +15-17 % (MTP ON) | t3-volta-small | L2 | `t3_h2_bench.py` (toggle env `GGML_CUDA_VOLTA_Q8_FATTN_TC`) | .so T3 : H2 ON vs OFF |
| V5 | FA16 prefill −12,8/−15,1/−11,8 % @98/131/196 k | fa16-compact | L1 | `python3 repos/fa16-compact/bench/bench_fa16_ncu.py --so sofiles/libggml-cuda-dual.so.d4954502` | 16,6→14,4 / 21,8→18,5 / 25,9→22,8 µs/tok |
| V6 | FA16 e2e prod −10,5/−7,1 % @58/131 k | fa16-compact | L2 | `python3 bench/bench/supervisor_bench.py run repos/fa16-compact/bench/rollout_fa16.py --detach` | rollout 203406/203903 (76,5→68,5 s / 265,8→246,9 s) |
| V7 | FA16 decode sans régression (+1,67 % @98 k, cooldown) | fa16-compact | L1+L2 | `repos/fa16-compact/bench/bench_fa16_decode_mb.py` (après cooldown) | 58,96 vs 57,99 t/s |
| V8 | T1 neutre @114 k / −11,3 % @160 k (NO-GO) | t1-mmsmt | L2 | `python3 bench/bench/supervisor_bench.py run repos/t1-mmsmt/bench/t1_bench.py --detach` | A-t3 45,7/40,85 ; B-t1t3 45,9/36,2 |
| V9 | H3 −2,3 % microbench (NO-GO) | mmvq-clos | L1 | `python3 repos/mmvq-clos/bench/ab_mmvq2.py --so sofiles/libggml-cuda-dual.so.h3-20261029` | 181,1 vs 185,3 µs (q6_f17408x5120) |
| V10 | b2048 +9,5/+25,6 % decode @114/196 k | b2048 | L2 | `python3 bench/bench/supervisor_bench.py run repos/b2048/bench/ab_batch_t3.py --detach` | 21,02/18,18 vs 19,19/14,47 |
| V11 | n_max : 3 > 2 (−5 %) > 4 (−23 %) | mtp-nmax | L3 | `python3 bench/bench/supervisor_bench.py run repos/mtp-nmax/bench/prof_prd_profile_20261030.py --detach` ×3 (n=2,3,4) | 44,18 / 41,94 / 33,86 t/s ; run réf `results/runs/prof-prd-20261030-114948/` |
| V12 | H2 GO : +11-14 % @107 k (run9c3, 6 essais interleave) | h2 | L2 | `python3 bench/bench/supervisor_bench.py run repos/h2/bench/run9c3_bench.py --detach` | eval_ms REF 40,3/40,6/41,9 s vs H2 35,3/35,2/36,7 s (`repos/h2/nightrun8/ab_run9c3/`) ; seuils fixés AVANT (±5 % @107k) |

**L0 (audit sans GPU)** : pour chaque repo, recalculer les deltas des tables du
README depuis `results/*.jsonl` (les bruts sont tous là) — `validate_all.sh L0`
fait le parse de tous les jsonl ; le recalcul des deltas se fait à la main depuis
les tables des README (les bruts sont dans chaque repo).

## 3. Ordre d'exécution recommandé (si on re-valide tout)

1. **L0** sur les 7 repos (0 min, cohérence interne) — `validate_all.sh L0`.
2. **L1** sur la machine de référence : V2, V3, V5, V7, V9 (~1 h au total).
   Prerequis : `sofiles/` présent + microbenches (`mb_fa` etc. fournis compilés ;
   sinon `nvcc -arch=sm_70` — voir `build_t3.sh` dans le repo t3).
3. **L2** : V1 → V4 → V6 → V8 → V10 → V12 (chaque run via superviseur, ~4-5 h au
   total, à étaler sur la nuit). Le superviseur sérialise (1 bench à la fois, flock).
4. **L3** seulement si L2 montre un écart > 3 % : V11 (~1 h pour les 3 n_max).

`validate_all.sh` orchestre tout ça ; **ne jamais lancer deux benchs en
parallèle** (flock + la V100 n'est pas partageable).

## 4. Interprétation des résultats de re-validation

- **Delta reproduit à ±3 %** → chiffre validé, rien à faire.
- **Delta entre 3 % et 5 %** → à re-lancer une fois (bruit possible) ; si
  persistant, c'est un régression ou un drift de machine (driver, thermal, KV).
- **Delta > 5 % ou signe inversé** → le chiffre publié ne se tient plus sur cette
  machine/config : (a) vérifier le md5 du `.so` (drift de build), (b) vérifier le
  thermal (cooldown), (c) vérifier le shortlist MTP (l'acceptance bouge les
  chiffres MTP ON), (d) si tout est bon → le chiffre est daté, le mettre à jour
  dans le repo + `OPTI.md`.
- **Un microbench (L1) valide mais un e2e (L2) non** → le kernel est bon mais
  l'interaction (MTP, batch, thermal) a changé : c'est un changement de contexte,
  pas du kernel.

## 5. Prerequis machine (re-validation complète)

- La machine de référence (V100-PCIE-32GB + driver 580 + CUDA 12.6) ou équivalent
  strict (le `.so` est dual sm_70+sm_61 mais les chiffres sont mesurés sur sm_70).
- Le service `ollama-v100` + le modèle + le shortlist MTP (`qwen38-27b-exact-131072.i32`).
- `sofiles/` + `big/` (le sha256 manifeste doit passer : `sha256sum -c`).
- nsys 2024.1+ (ou équivalent) + CUPTI 12.9 (`big/nsys-cupti129/`).

**Note** : la re-validation **L2/L3 consomme le service** (stoppe/restore
`ollama-v100`) — ne pas la lancer pendant une fenêtre prod sensible ; le
superviseur + le `chat_watchdog` (cron 5 min) couvrent les fins de génération
entre runs.
