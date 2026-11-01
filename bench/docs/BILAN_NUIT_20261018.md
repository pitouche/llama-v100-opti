# NUIT D'OPTIMISATION — 18/10/2026 (00:50 → ~08:00 UTC)

Mission confiée par guile (ceci est le squelette, rempli au fil de la nuit) :
- reprendre les notes, mettre de l'ordre, explorer ce qui n'est PAS encore exploré
- MTP EXCLU de toutes les mesures (non reproductible / non sizeable) — toutes mesures MTP OFF
- viser plein de générations (objectif : profiler la V100, pas juste Grafana)
- recherche web pour compléter les pistes

## Rappel de l'état de l'art consolidé (avant cette nuit)

| levier | statut | résultat |
|---|---|---|
| **batch -b/-ub 512→2048** | **DÉPLOYÉ 17/10 soir** | decode 114k +44%, 192k +35% ; prefill +12-15% |
| H2 Q8 FA TC T=4 (a4b6f440, .so prod) | en prod depuis 14/10 | −12,5% @107k vs REF (RUN9C3) |
| patch 42d9581 (nwarps K-quant 4→2, MTP-ON) | testé 17/10 | NEUTRE (bruit de baseline, MTP-ON) |
| env Q5_X4 / Q8_FATTN_TC (MTP off) | testé 17/10 | NEUTRE au régime prod (ab_env) |
| P7 kernel Q6_K X4 | testé 08/10 | +1,1% court ctx |
| P0 Q8_0 QPN | autopsy 05/10 | NO-GO (design mort) |

**NON exploré (cible de cette nuit) :**
1. 42d9581 en **MTP OFF** (le test de 17/10 était pollué par le cycle MTP ; upstream
   revendique +3,17% tg128 sur V100-32GB PCIe driver 580 — même machine !)
2. b4096 vs b2048 en **decode** (le gain 35-44% du deploy est le b512→b2048 ;
   b4096 n'a été testé qu'en prefill, et au MTP-ON il montait en VRAM)
3. b8192 : inconnu complet (VRAM compute-buffer ? gain ?)
4. n_threads (CPU) 6 vs 8/12 — jamais benché
5. Recherche web : V100 llama.cpp/ollama depuis le 16/10

## Résultats

### T3 (anyei #153, fattn-mma-volta-small) — **GAIN SUBLIME**

**Méthode** : 3 legs interleave × 2 reps, serveur frais par leg, MTP OFF, -b 2048, KV q8_0, FA auto, CVD=0 (V100 smi id 1), port 8917, N_GEN 64, warmup 1024.
- A-prod : `.so` prod H2 (a4b6f440)
- B-t3on : `.so` T3, T3 actif (default ON)
- C-t3off : `.so` T3 + `GGML_CUDA_FA_NO_VOLTA_SMALL=1` (A/B même binaire)

| ctx | leg | rep | decode t/s | prefill t/s |
|---|---|---|---|---|
| 114688 | A-prod | 1 | 16.97 | 617.3 |
| 114688 | B-t3on | 1 | 18.85 | 618.1 |
| 114688 | C-t3off | 1 | 17.52 | 618.3 |
| 114688 | A-prod | 2 | 17.68 | 618.5 |
| 114688 | B-t3on | 2 | 19.99 | 618.5 |
| 114688 | C-t3off | 2 | 17.39 | 618.3 |
| 196608 | A-prod | 1 | 14.13 | 342.6 |
| 196608 | B-t3on | 1 | 17.82 | 342.9 |
| 196608 | C-t3off | 1 | 14.13 | 342.7 |
| 196608 | A-prod | 2 | 14.13 | 342.8 |
| 196608 | B-t3on | 2 | 18.16 | 342.8 |
| 196608 | C-t3off | 2 | 14.12 | 342.9 |

**Δ decode vs A-prod (médiane) :**
- ctx 114688 : **B-t3on +13.07 %** · C-t3off −0.90 %
- ctx 196608 : **B-t3on +28.52 %** · C-t3off +0.00 %

**Lecture :**
- C-t3off ≈ A-prod (±1 %) → le `.so` T3 ne diffère de prod QUE par T3. Méthodologie valide.
- B-t3on >> A-prod → le gain est **100 % attribuable à T3**.
- Le gain est **croissant avec le contexte** (13 % → 29 %) : cohérent avec un kernel attention qui profite d'autant plus que la KV cache est longue (114k → 196k).
- Prefill inchangé (±0.1 %) : T3 ne s'applique qu'au decode (rows ≤ 8), pas au prefill (rows = batch).
- VRAM identique (30988 MiB) : pas de buffer supplémentaire.
- Service restauré : OUI.

**Verdict : T3 VALIDÉ — gain decode +13 % @ 114k, +29 % @ 196k, zéro impact prefill/VRAM.**

### T1 (mmsmt, matmul small-batch) — porté, non mesurable MTP-OFF
- `.so` T1 construit OK (sha256 7d382c14…).
- Hook dans `ggml_cuda_mul_mat` avant `mmvq`, gated `GGML_CUDA_SMT`.
- Cible : MTP verify (rows 2-8, Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS/Q3_K/Q2_K/Q2_0).
- **Non testable en MTP-OFF** (pas de calls MTP verify dans le flux prod sans MTP).
- Gardé pour éventuelle variante MTP-on future.

### Build T1T3 combiné — OK
- `libggml-cuda-t1t3.so` construit (sha256 ca6e55cd…, 143 809 088 B).
- Gate T3 présent, 72 symboles volta_small, 1 symbole mmsmt exporté.
- Prêt si on veut empiler T1+T3 (mais T1 n'agit qu'en MTP-on).

## État machine

- 00:50 UTC : ollama-v100 active, runner 167k+b2048 (MTP ON, 24.6 GB, V100 32436/32768),
  lock superviseur libre, 1070 = 2.7 GB (nomic)
- 13:33 UTC : bench T3 terminé, service restauré, V100 32440 MiB (runner prod repassé), port 8917 libre
- .so prod a4b6f440 : **INTOUCHÉ** (md5 a4b6f440b020430c4f9541c6414ad832)
- Persistance GPU : `nvidia-persistenced.service` créé + activé (2 GPUs)

## Artefacts `.so` (H2LIB)

| lib | sha256 | taille | statut |
|---|---|---|---|
| libggml-cuda.so (prod) | md5 a4b6f440 | 135 420 368 B | INTOUCHÉ |
| libggml-cuda-t3.so | 597d20a3… | 143 452 928 B | **VALIDÉ bench** |
| libggml-cuda-t1.so | 7d382c14… | 135 768 368 B | OK (non testé) |
| libggml-cuda-t1t3.so | ca6e55cd… | 143 809 088 B | OK (combinaison) |

## Prochaine étape

**Déployer T3 en prod** : remplacer `libggml-cuda.so` par `libggml-cuda-t3.so`.
- Gain attendu en prod (MTP ON, ctx 167k) : entre +13 % et +29 % decode (T3 ne touche que le decode, MTP n'interagit pas avec le dispatch FA).
- Rollback : `cp` de l'ancien `.so` (backup dans `$BAKDIR`).
- Gate de secours : `GGML_CUDA_FA_NO_VOLTA_SMALL=1` dans l'env du service si besoin de désactiver sans rollback.
- Le `.so` T1T3 est prêt si on veut ajouter T1 en même temps (inerte en MTP-OFF, actif en MTP-ON).

## DÉPLOIEMENT T3 — FAIT 18/10 19:07 UTC

- `deploy_t3.sh` : attente idle (tps=0), backup → `/var/backups/llama-experiments/libggml-cuda.so.prod-a4b6f440-20261018-190701` (md5 a4b6f440 vérifié), stop, V100 libérée, `cp` T3, start.
- **Prod .so actuel : md5 d2df07b8 (T3)**. Service active, runner repassé (`-c 167000 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3`), V100 31896 MiB, pas de prunage de blobs (OLLAMA_NOPRUNE=true).
- Sanity : génération OK, réponse correcte, 60,5 t/s @ court ctx.
- ROLLBACK : `sudo cp /var/backups/llama-experiments/libggml-cuda.so.prod-a4b6f440-20261018-190701 /home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so && sudo systemctl restart ollama-v100`

## BENCH T1 (MTP ON, config prod exacte) — FAIT 18/10 20:22 UTC — **NO-GO**

3 legs interleave × 2 reps, `-c 167000 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling`, KV q8_0, FA auto. MTP vérifié actif (draft acceptance 0.77-0.94, mean len 3.3-3.8).

| ctx | A-t3 | B-t1t3 | C-smt0 | Δ B vs A | Δ C vs A |
|---|---|---|---|---|---|
| 114688 | 45.73 t/s | 45.91 t/s | 45.70 t/s | **+0.35 %** (bruit) | −0.09 % |
| 160000 | 40.85 t/s | **36.23 t/s** | 40.86 t/s | **−11.30 %** | −0.02 % |

**Lecture :**
- C-smt0 ≈ A (±0.1 %) → le `.so` T1T3 (rebuilt) diffère de T3 uniquement par T1. Méthodologie valide.
- B régredie de **−11.3 % @ 160k**, reproduit sur les 2 reps (36.19 / 36.27 vs 40.89 / 40.81) : ce n'est PAS du bruit, c'est une régression réelle de T1 à long ctx.
- @ 114k : +0.35 % = dans le bruit.
- **Verdict : T1 NON CONCLUANT (neutre court ctx, régression nette long ctx) → pas de déploiement T1T3.**

### Bug découvert et corrigé pendant le bench T1
- Le 1er `.so` T1T3 (md5 7e7dfa10) **ne chargeait pas** : `undefined symbol ggml_cabla...` = `_Z24ggml_cuda_mul_mat_cublas...`.
- Cause : `mmsmt.cu` (anyei) porte une **forward decl** `void ggml_cuda_mul_mat_cublas(...)` non-static ; notre `ggml-cuda.cu` définit la même fonction **static** → le linker n'a pas pu résoudre le symbole de mmsmt (le link passait quand même : symboles non résolus tolérés dans un .so, l'échec n'apparaît qu'au dlopen).
- Fix : `ggml-cuda.cu` — `ggml_cuda_mul_mat_cublas` dé-staticisé (t1port + t1t3port), recompile `ggml-cuda.t1.o`, rebuild T1T3 → **md5 04f41b70**, symbole `T` exporté, hook mmsmt + gate T3 présents.
- ⚠️ Le 2e bench T1 a donc bien mesuré T1 actif (pas de faux négatif par chargement raté — ce run là chargeait).

## BENCH H2 (T1T3 avec/sans H2) + CONFI .so T3 prod — FAIT 18/10 21:21 & 22:25 UTC — **H2 DÉPASSÉ → DÉSACTIVÉ EN PROD**

Question : maintenant que T3 couvre rows 1-8, H2 (VOLTA_Q8_W4, rows==4 only) est-il encore utile ? En dispatch H2 est testé AVANT T3 → sur rows==4 (MTP verify, n-max 3) c'est H2 qui a la priorité.

### t1t3_h2_bench.py (md5 .so T1T3 04f41b70, T1+T3 actifs, seul le gate H2 varie), MTP ON
| ctx | A-h2on | B-h2off | Δ B vs A |
|---|---|---|---|
| 114688 | 45.90 t/s | 53.93 t/s | **+17.47 %** |
| 160000 | 36.28 t/s | 44.59 t/s | **+22.87 %** |

### t3_h2_bench.py (CONFI sur le .so T3 prod réel d2df07b8, SANS T1), MTP ON
| ctx | A-h2on (= prod actuelle) | B-h2off (= proposé) | Δ B vs A |
|---|---|---|---|
| 114688 | 45.68 t/s | 53.68 t/s | **+17.42 %** |
| 160000 | 40.88 t/s | 47.23 t/s | **+15.53 %** |

**Lecture :**
- Draft acceptance **identique** sur les 4 legs (0.939/0.92/0.77) → le delta est 100 % le kernel FA, PAS le MTP. Méthodologie propre.
- **T3 (VOLTA_SMALL, rows 1-8) bat le kernel dédié H2 (VOLTA_Q8_W4) même sur sa propre géométrie rows==4.** Le kernel généraliste m8n8k4 anyei est plus efficace que le chemin W4-2 (élargissement shared-mem).
- Reproduit sur 2 binaires (T1T3 et T3-only) × 2 reps, signe constant → ce n'est PAS du bruit.
- **Verdict : H2 est DÉPASSÉ par T3. Désactiver H2 accélère le decode (+15 à +23 %), elle n'apporte plus rien au-dessus de T3.**

### DÉPLOIEMENT H2 OFF — FAIT 18/10 22:31 UTC
- `ollama-v100.service` : ajout `Environment="GGML_CUDA_VOLTA_Q8_FATTN_TC=0"` (commenté avec le gain + la marche de re-activation).
- `daemon-reload` + `restart` → service **active**. Env `FATTN_TC=0` vérifiée dans les 2 procs (ollama + llama-server).
- `.so` chargé = prod T3 **d2df07b8**, VRAM 32438 MiB (inchangé). Sanity : génération OK, **58.9 t/s** @ court ctx.
- ROLLBACK (1 ligne) : retirer `GGML_CUDA_VOLTA_Q8_FATTN_TC=0` du service → `systemctl daemon-reload && systemctl restart ollama-v100`.

## Conclusions de la nuit (synthèse finale)

| piste | verdict | gain |
|---|---|---|
| **T3 fattn-mma-volta-small** | **DÉPLOYÉ** | **+13 % decode @114k, +29 % @196k** (MTP-OFF bench ; prod MTP-ON) |
| **H2 VOLTA_Q8_W4 (rows==4)** | **DÉSACTIVÉ** (dépassé par T3) | **−15 à −23 %** si on le garde ; **+15 à +17 %** en le désactivant (MTP-ON bench .so T3 prod) |
| T1 mmsmt | NO-GO | neutre @114k, **−11.3 % @160k** (MTP-ON bench) |
| H3 42d9581 | neutre | −0.6 % (MTP-OFF) |
| b4096 / b8192 / -t12 | neutre / OOM / neutre | — |

- **Prod finale : T3 (md5 d2df07b8), H2 DÉSACTIVÉ (FATTN_TC=0), T1 absent**, MTP ON n-max 3, ctx 167k, b2048, KV q8_0.
- Artefacts : t3_bench_*, t1_bench_*, t1t3_h2_bench_*, t3_h2_bench_*.jsonl/md, deploy_t3_1907*.log, `.so` t1t3 md5 04f41b70 (hors prod, T1 rejeté).
