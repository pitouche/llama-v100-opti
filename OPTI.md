# OPTI.md — Optimisation prod V100 (Qwen3.8-27B q6-MTP) — synthèse des modifs appliquées

**Date : 19/10/2026 ~00:25 UTC** — mise à jour après le bench nmax (dernière piste restante).
**Machine** : 1× Tesla V100 32 GB (sm_70), CUDA 12.6, smi id **1** = `CUDA_VISIBLE_DEVICES=0`.
**Modèle prod** : `qwen3.8-27b-q6-mtp-vision`, blob `sha256-49330183…` (23 088 409 504 B), arch `qwen35`,
head 256, GQA 24/4, block_count 65, **1 seul module MTP** (`blk.64.nextn.*`).
**Serveur** : `ollama serve` → `lib/ollama/llama-server` (fork anyei/llama.cpp, build `ollama-v100`).

> Ce document est le **sommaire de référence** de tout ce qui a été modifié pour la prod V100 :
> déploiements, `.so`, gates/env, rollbacks, résultats bench, et état final.
> Le détail brut (logs, jsonl, gputimes) est dans `bench/` (pointeur en fin de doc).

---

## 0. État prod FINAL (à retenir)

| élément | valeur |
|---|---|
| Service | `ollama-v100` — **active** |
| `.so` chargé | `…/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so` — md5 **d2df07b8** = **build T3** |
| T3 (VOLTA_SMALL) | **ACTIF** (gate `GGML_CUDA_FA_NO_VOLTA_SMALL` non défini = ON) |
| H2 (VOLTA_Q8_W4, rows==4) | **DÉSACTIVÉ** par `GGML_CUDA_VOLTA_Q8_FATTN_TC=0` |
| T1 (mmsmt) | **ABSENT** du `.so` prod (rejeté) |
| MTP | ON — `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling` |
| Contexte / batch | `-c 167000`, `-b 2048 -ub 2048` |
| KV cache | `q8_0` (`--cache-type-k/v q8_0`) |
| FA | `auto` (decode rows≤8 → T3) |
| VRAM | ~**32 438 MiB / 32 768** |
| Sanity | génération OK, ~**58.9 t/s** court ctx |

**Gain decode cumulé vs prod initiale de la semaine** (batch 512, H2 seul, a4b6f440) :
- **T3** : +13 % @114 k, +29 % @196 k (bench MTP-OFF)
- **H2 OFF** : +17.4 % @114 k, +15.5 % @160 k (bench MTP-ON sur le `.so` T3 prod)
- (batch 512→2048 déployé la veille : +35 à +44 % decode)

---

## 1. Les trois kernels FA et le dispatch (pourquoi ça compte)

Sur V100 (sm_70), le kernel d'attention choisit par `ggml_cuda_get_best_fattn_kernel`
selon la géométrie (nb de rows = tokens de l'étape). Le dispatch teste **H2 AVANT T3** :

| route | géométrie | gate env | rôle |
|---|---|---|---|
| **H2** `VOLTA_Q8_W4` | **rows == 4**, head 256, KV q8_0/f16 | `GGML_CUDA_VOLTA_Q8_FATTN_TC` (ON par défaut, `=0` OFF) | kernel dédié "small" W4-2 |
| **T3** `VOLTA_SMALL` | **rows 1 à 8**, head 64/128/256, KV q8_0/f16 | `GGML_CUDA_FA_NO_VOLTA_SMALL` (`=1` OFF) | kernel m8n8k4 anyei (porté) |
| **T1** `mmsmt` | matmul small-batch (rows 2-8), quant 4/8 | `GGML_CUDA_SMT` (`=0` OFF) | hook mul_mat MTP-verify (rejeté) |

C'est le **dispatch H2→T3** qui a créé le bug de performance : sur rows==4 (le cas exact du
MTP verify avec n-max 3 → 4 tokens), **H2 prenait la main** avant T3. Or T3 bat H2 sur cette
même géométrie → garder H2 actif coûtait du decode. D'où H2 OFF.

Conditions exactes de la route T3 (portées dans `fattn.cu`) : Volta, pas désactivé par le gate,
`mask != nullptr`, `max_bias == 0`, pas de sinks, `Q->ne[1]` ∈ [1,8], head ∈ {64,128,256}, V==K,
KV q8_0 ou f16, `logit_softcap == 0` ou head != 64.

---

## 2. Patches sources portés (tree principal)

Tree : `/home/guile/llama-experiments/h2/cupti129-test/e2/`
Build dir de référence : `$H2LIB/build-clean/_deps/llama_cpp-build/ggml/src/ggml-cuda`
`H2LIB=/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib`

### 2.1 T3 (VALIDÉ, en prod)
```
t3port/ggml-cuda/fattn.cu
t3port/ggml-cuda/fattn-mma-volta-small.cuh
```
- Ajout de la route T3 dans `ggml_cuda_get_best_fattn_kernel` (conditions §1).
- **Fix porté** : `use_sparse=false` ajouté sur les 6 appels `launch_fattn` (sinon le kernel
  small partait en sparse et plantait / donnait faux résultats).
- Scripts : `build_t3.sh v3`.

### 2.2 T1 (rejeté, hors prod)
```
t1port/ggml-cuda/ggml-cuda.cu
```
- `#include "mmsmt.cuh"` + hook `ggml_cuda_should_use_mmsmt` dans `ggml_cuda_mul_mat` (avant `mmvq`).
- Gate `GGML_CUDA_SMT`. Cible : MTP verify (rows 2-8, quant 4/8).

### 2.3 Bug T1T3 — corrigé (rebuild)
- **Symptôme** : le `.so` T1T3 ne chargeait pas au `dlopen` →
  `undefined symbol ggml_cuda_mul_mat_cublas`.
- **Cause** : `mmsmt.cu` (anyei) porte une forward decl **non-static** de
  `ggml_cuda_mul_mat_cublas` ; notre `ggml-cuda.cu` la définissait **static** → symbole en
  suspens, toléré au link, fatale au `dlopen`.
- **Fix** : dé-staticisation de `ggml_cuda_mul_mat_cublas` (t1port + t1t3port), recompile
  `ggml-cuda.t1.o`, rebuild → `.so` T1T3 md5 **04f41b70**, symbole `T` exporté, hook + gate T3 intacts.
- ⚠️ Le 2ᵉ bench T1 a bien mesuré T1 **actif** (run qui chargeait), donc le verdict NO-GO est fiable.

---

## 3. Artefacts `.so`

| lib | md5 / sha256 | taille | contenu | statut |
|---|---|---|---|---|
| **libggml-cuda.so (PROD)** | md5 **d2df07b8** (sha256 597d20a3…) | 143 452 928 B | T3 + H2(gate) + gate T3, **SANS T1** | **EN PROD** |
| libggml-cuda-t1t3.so | md5 **04f41b70** | 143 809 088 B | T1 + T3 + H2(gate) | hors prod (T1 rejeté) |
| libggml-cuda-t1.so | sha256 7d382c14… | 135 768 368 B | T1 seul | conservé |
| **prod historique a4b6f440** | md5 a4b6f440 | 135 420 368 B | H2 seul (avant T3) | **backup rollback** |

Backup prod historique :
```
/var/backups/llama-experiments/libggml-cuda.so.prod-a4b6f440-20261018-190701
```

---

## 4. Gates / env dans `ollama-v100.service` (état courant)

```ini
Environment="LLAMA_ARG_SPEC_MTP_DEFER_PROMPT=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_NOPRUNE=true"
Environment="OLLAMA_CONTEXT_LENGTH=167000"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="LLAMA_ARG_FIT_TARGET=1024"
Environment="GGML_CUDA_VOLTA_Q6_W4R4=1"
# H2 (VOLTA_Q8_W4, rows==4) DÉSACTIVÉ : surpassé par T3 (bench 18/10 MTP ON .so T3 d2df07b8 :
# H2 OFF -> +17.4 % decode @114k, +15.5 % @160k). Re-activer = supprimer la ligne + reload.
Environment="GGML_CUDA_VOLTA_Q8_FATTN_TC=0"
Environment="GGML_CUDA_QWEN35_MTP_SHORTLIST=…/data/mtp-shortlists/qwen38-27b-exact-131072.i32"
Environment="LD_LIBRARY_PATH=…/cuda-12.6/…"
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_LIBRARY_PATH=/home/guile/llama-experiments/ollama-v100"
```
Le runner (config du modèle ollama) ajoute : `--spec-type draft-mtp --spec-draft-n-max 3
--spec-draft-backend-sampling`, `-c 167000 -b 2048 -ub 2048`, `--flash-attn auto`, KV q8_0, `--mmproj`.

Leviers par gate (récap) :
- **H2** : `GGML_CUDA_VOLTA_Q8_FATTN_TC` — ON défaut, `=0` OFF. **État prod : `0` (OFF).**
- **T3** : `GGML_CUDA_FA_NO_VOLTA_SMALL` — `=1` OFF. **État prod : non défini (T3 ON).**
- **T1** : `GGML_CUDA_SMT` — `=0` OFF. **État prod : T1 absent du `.so`.**

---

## 5. Résultats bench (tous : interleave 2 reps, serveur frais par leg, port 8917, CVD=0,
N_GEN 64, warmup 1024 ; config prod exacte sauf mention)

### 5.1 T3 — MTP OFF (gold standard, le gain FA principal)
`.so` : A-prod = a4b6f440 (H2 seul) · B-t3on = T3 actif · C-t3off = T3 + gate=1 (même binaire).

| ctx | A-prod | B-t3on | C-t3off | **Δ B vs A** |
|---:|---:|---:|---:|---:|
| 114 688 | 17.33 t/s | 19.42 t/s | 17.46 t/s | **+13.1 %** |
| 196 608 | 14.13 t/s | 17.99 t/s | 14.13 t/s | **+28.5 %** |

Lecture : C ≈ A (±1 %) → le `.so` T3 ne diffère de prod QUE par T3. Gain croissant avec le ctx
(kernel attention, KV longue). Prefill ±0.1 %, VRAM identique (30 988 MiB). **T3 VALIDÉ.**

### 5.2 T1 — MTP ON (config prod)
`.so` : A-t3 = prod · B-t1t3 = T1T3 (04f41b70) · C-smt0 = T1T3 + `GGML_CUDA_SMT=0`.

| ctx | A-t3 | B-t1t3 | C-smt0 | **Δ B vs A** |
|---:|---:|---:|---:|---:|
| 114 688 | 45.73 t/s | 45.91 t/s | 45.70 t/s | **+0.35 %** (bruit) |
| 160 000 | 40.85 t/s | **36.23 t/s** | 40.86 t/s | **−11.30 %** |

Lecture : C ≈ A → méthode propre. T1 neutre @114 k, **régression nette @160 k** (reproduite 2 reps).
**T1 REJETÉ** — pas d'empilement T1T3 en prod.

### 5.3 H2 — T1T3, MTP ON
Même `.so` T1T3 (04f41b70, T1+T3 actifs), seul le gate H2 varie.

| ctx | A-h2on | B-h2off | **Δ B vs A** |
|---:|---:|---:|---:|
| 114 688 | 45.90 t/s | 53.93 t/s | **+17.47 %** |
| 160 000 | 36.28 t/s | 44.59 t/s | **+22.87 %** |

### 5.4 H2 — CONFI sur le `.so` T3 prod réel (d2df07b8, SANS T1), MTP ON
| ctx | A-h2on (= prod avt) | B-h2off (= proposé) | **Δ B vs A** |
|---:|---:|---:|---:|
| 114 688 | 45.68 t/s | 53.68 t/s | **+17.42 %** |
| 160 000 | 40.88 t/s | 47.23 t/s | **+15.53 %** |

Lecture 5.3/5.4 : draft acceptance **identique** entre legs (0.77-0.94) → le delta est 100 % le
kernel FA, pas le MTP. T3 (VOLTA_SMALL) bat le kernel dédié H2 (VOLTA_Q8_W4) **même sur rows==4**.
Reproduit 2 binaires × 2 reps, signe constant. **H2 DÉSACTIVÉ.**

> **Mission H2 complète** (protocole de capture, CUPTI 12.9, A/B du GO run8-9c3,
> H2b, divergence textuelle, journal) : repo [`opti-h2`](https://github.com/pitouche/opti-h2)
> (`repos/h2/`), extrait du kit `v100-bench-kit` le 01/11/2026.

### 5.5 n-max — `--spec-draft-n-max` 3 vs 4, `.so` T3 prod, MTP ON  *(19/10, dernière piste)*
Le GGUF n'a que **1 module MTP** (single-head). Le code **ne clamp pas** n_max au nb de couches en
single-head (seul `chain_heads` le fait) → n_max=4 est légal. Test A/B (nmax_bench.py) :

| ctx | A-nmax3 (= prod) | B-nmax4 | **Δ B vs A** | mean acc len | acc rate/pos |
|---:|---:|---:|---:|---:|---|
| 114 688 | 45.72 t/s | 39.58 t/s | **−13.43 %** | 3.71 → 4.43 | 3 pos (0.97,0.90,0.84) → 4 pos (0.89,0.81,0.81,0.71) |
| 160 000 | 40.89 t/s | 36.25 t/s | **−11.35 %** | 3.71 → 4.20 | idem |

Lecture : le 4ᵉ draft n'est accepté que ~0.71, et chaque n_max supplémentaire **allonge le verify
target (27B)** et ajoute une passe de draft. Coût > bénéfice. **n-max=3 CONSERVÉ (prod), 4 rejeté.**
Note : n-max=2 reste un micro-levier non testé (courbe d'acceptance encore >0.89 à pos0/pos1), ROI
faible — à n'ouvrir que si on cherche le dernier 1-2 %.

### 5.6 Piste night bench (b4096 / b8192 / -t12) — close, neutre
- b4096 : decode −0.1/+0.0 % (neutre), prefill +2.7/+3.3 %, VRAM 32 144 (ok).
- b8192 : VRAM 32 477 (monte), pas de gain decode → non adopté.
- -t12 (CPU threads) : decode −0.5 % (neutre), on reste sur le défaut.

---

## 6. Conclusions / verdicts

| piste | verdict | gain decode | en prod ? |
|---|---|---|---|
| **T3 fattn-mma-volta-small** | **DÉPLOYÉ** | +13 % @114 k, +29 % @196 k (MTP-OFF) | **OUI** |
| **H2 OFF** (gate `FATTN_TC=0`) | **APPLIQUÉ** | +15 à +17 % (MTP-ON .so T3) | **OUI** |
| **T3 × n_kv sweep, kernel pur (22/10)** | T3 gagne sur **tous** les points (dispatch confirmé app) | kernel : +38→+52 % (T=1), +46→+72 % (T=2) | **OUI** |
| T1 mmsmt | **NO-GO** | neutre @114 k, −11.3 % @160 k | non |
| `--spec-draft-n-max` 4 | **NO-GO** | −11 à −13 % | non (3 gardé) |
| H3 42d9581 (nwarps K-quant) | neutre | −0.6 % (MTP-OFF) | non |
| b4096 / b8192 / -t12 | neutre / OOM-VRAM / neutre | ~0 | non |
| NVFP4 / quant exotiques (ik_llama) | non applicable rapide | — | non |
| **T3 v2 levier D — load V q8_0 vectorisé 128-bit (27/10)** | **NO-GO (confirmé binaire 28/10)** | tent.1 0653dc50 −104 % @T=1 (825 vs 404 µs) — attribution confirmée SASS : cubin 213 (kernel prod exact q8_0 D=256) diffère 065 vs cvec/prod, 128 LDG.128 + spills (STACK 112 B) ; v2.1 full-step ~−50 % @T=1 + crash occupancy (smem > 48 Ko) ; v2.2 warp-private crash IMA (32 rangées/warp vs 128 lues par la phase 2) ; prod == cvec sur les 4 cubins diff | non |

**Prod finale : T3 (d2df07b8) + H2 OFF + MTP ON n-max 3, ctx 167 k, b2048, KV q8_0.**

### 6.1 Microbench kernel rows 1-8 (21/10) — qui gagne où, en µs

Table définitive du « qui gagne » sur la géométrie prod exacte (D=256, 24 qh, 4 kvh, KV
q8_0, Q f32, mask f16=0, **n_kv=107 351**), mesurée **kernel pur** via dlopen du .so prod
T3 (d2df07b8) + `ggml_cuda_flash_attn_ext` direct (MÊME dispatch que la prod, 3 routes
séparées par env), 200 iters médiane : `bench/mb_fa_summary.md`.

| T (rows) | stock (µs) | H2 (µs) | T3 (µs) | T3 vs stock | T3 vs H2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1191 | — | **585** | **+50.9 %** | — |
| 2 | 1829 | — | **521** | **+71.5 %** | — |
| 3 | 2194 | — | **665** | **+69.7 %** | — |
| 4 | 2194 | 1532 | **669** | **+69.5 %** | **+56.3 %** |
| 5 | 2938 | — | **796** | **+72.9 %** | — |
| 6 | 2953 | — | **1207** | **+59.1 %** | — |
| 7 | 2957 | — | **1219** | **+58.8 %** | — |
| 8 | 2948 | — | **1197** | **+59.4 %** | — |

**Lecture :**
- **T3 gagne partout** (rows 1-8) : +51 à +73 % vs stock (wmma vec/tile), et bat H2 sur
  rows==4 (+56 %). H2 est éligible uniquement sur rows==4 ; sur les autres T le mode
  « h2 » retombait sur stock (confirmé : h2 et stock identiques au µs près).
- **Profil T3 non-linéaire** : creux à T=2 (521 µs), plateau T=1→5 (~520-800 µs), puis
  saut ×1.5 entre T=5→6 (796→1207 µs). Le saut = changement de géométrie m8n8k4 :
  T=1-5 → ntiles=1 (pack gqa 6), T=6-8 → ntiles=1 avec moins de parallélisme GQA
  (T·pack = 36-48 rows > 32 → le packing GQA se réduit, les CTAs diminuent) et/ou 2
  tuiles M. À creuser si on optimise T3 : c'est là que se joue le decode MTP (T=4
  dominant, mais T=1-2 fréquent au début de séquence / acceptance bas).
- **Validation numérique** (mêmes données, les 3 routes) : maxrel T3↔H2 = 1.5e-3
  (bf16 acc. H2), T3↔stock = 6.9e-3 (fp16 vs f32). Mêmes attentions → les µs sont
  comparables.

**Décision :** le levier kernel FA est déjà capturé (T3 déployé, H2 off). Les variantes
H2 restantes (BC 16/4, LONG_SPLITS 80/320) sont structurellement sans intérêt : H2 ne
bats pas T3 sur rows==4 et plafonne à LONG_SPLITS=160 dès ~2.5k contexte. Le seul gain
reste **d'optimiser T3 lui-même** (sweep géométrie : FATTN_VS_STEP/ntiles/NTHREADS, le
saut T=5→6), c'est la piste à retenir si on repart sur du kernel FA.

### 6.2 Microbench T3 sans MTP — T=1/2 × n_kv (22/10) — le cas dominant est couvert

Question : le microbench rows 1-8 n'avait mesuré que **n_kv=107 351**. Le cas decode
**sans MTP** (rows=1, et T=2 au début de flux) est le régime dominant — quel est le gain
le long du contexte, et T3 est-il bien dispatché partout ? (gate T3 : `K->ne[1] % 256 == 0`
; dans l'app, `get_n_kv()` pad **min 256** et par multiple de 256 → cohérent, confirmé par
`llama-kv-cache.cpp` : `n_pad_cur = max(n_pad, 256u)` ; les points mesurés sont des
multiples de 256 comme l'app, le n_kv brut 107 351 n'existe pas en app.)

`bench/t3_sweep_bench.py` + harnais `mb_fa` existant (dlopen .so prod d2df07b8, même
dispatch), 2 routes : **t3** (H2 off, comme prod) et **stock** (T3 off → vec/tile). H2 OFF
partout. Résultats : `bench/t3_sweep_summary.md` / `t3_sweep_results.jsonl`.

**T = 1** (decode classique — le cas dominant sans MTP) :

| n_kv | stock (µs) | T3 (µs) | T3 vs stock | T3 µs/100k keys |
|---:|---:|---:|---:|---:|
| 4 096 | 78 | 48 | +38.2 % | 1174 |
| 16 384 | 238 | 121 | +49.2 % | 737 |
| 32 768 | 406 | 207 | +49.0 % | 631 |
| 65 536 | 743 | 366 | +50.8 % | 558 |
| 107 520 | 1191 | 584 | +51.0 % | 543 |
| 131 072 | 1466 | 720 | +50.9 % | 549 |
| 155 648 | 1755 | 859 | +51.1 % | 552 |
| 176 128 | 1977 | 946 | +52.1 % | 537 |

**T = 2** :

| n_kv | stock (µs) | T3 (µs) | T3 vs stock | T3 µs/100k keys |
|---:|---:|---:|---:|---:|
| 4 096 | 97 | 52 | +46.4 % | 1274 |
| 16 384 | 283 | 122 | +56.9 % | 744 |
| 32 768 | 548 | 196 | +64.3 % | 597 |
| 65 536 | 1094 | 344 | +68.5 % | 525 |
| 107 520 | 1824 | 521 | +71.4 % | 485 |
| 131 072 | 2212 | 642 | +71.0 % | 490 |
| 155 648 | 2634 | 759 | +71.2 % | 488 |
| 176 128 | 3004 | 834 | +72.3 % | 473 |

**Lecture :**
- **T3 bien dispatché partout** : T3 < stock strictement sur les 32 points (autrement ce
  serait le fallback stock). La zone non explorée (T=1/2 × n_kv) est maintenant couverte
  de 4 k à 176 k — au-delà 176 128 = 172 k il n'y a plus de ctx (max prod 167 k).
- **Le gain croît avec le ctx** : T=1 +38 % → +52 %, T=2 +46 % → +72 %. C'est le régime
  attendu : le coût stock (vec T=1, tile T=2) croît plus vite que T3 (K-split sur les SMs,
  K/V lu une fois par kv-head via le packing GQA). À long ctx, **T3 = 2.0× (T=1) / 2.2×
  (T=2) stock sur le FA**.
- **Croissance sous-linéaire de T3** : 48 µs @4k → 946 µs @176k (T=1) ; le coût par 100k
  keys **baisse** de 1174 → ~540 µs (part fixe ~25 µs/launch qui pèse à court ctx), puis se
  stabilise ~540-550 µs/100k. T=2 a un coût/100k **plus bas** que T=1 (~475 vs ~545) —
  cohérent avec le « creux T=2 » vu en rows 1-8 (521 vs 585 µs @107k : 2×6=12 rows de Q
  packées dans la tile A vs 6, meilleur remplissage m8n8k4).
- **Ordre de grandeur** : à 107 k, T3 = 584 µs/couche/token (T=1) → 40 couches ≈ 23 ms
  de FA/token ; stock aurait coûté 47 ms. Le gain applicatif MTP-OFF mesuré (+13 % @114 k,
  §5.1) est cohérent avec ~24 ms/token d'écart sur le FA.
- **T3 loin de ses bornes** : à 107 k / T=1, borne mémoire ≈ 30 µs (48.75 Mo KV lues une
  fois, 1.6 To/s) et borne TC ≈ 42 µs (5.3 GFLOP/couche, 125 TFLOPS fp16) ; T3 = 584 µs
  ≈ **~14× le pic TC** (efficacité m8n8k4 ~7-10 %, tiles 8×8, syncs CTA). S'il y a un
  « T3 v2 », l'espace théorique est ×2-3 — mais c'est du rebuild, et le ROI app serait
  ~×2 sur 23 ms/token → ~+4-5 % decode long ctx (MTP-OFF ; avec MTP le FA rows=4 domine
  et le levier est le saut T=5→6, §6.1).

**Décision :** rien à changer en prod (T3 est déjà le .so déployé) — ce sweep **ferme la
zone non explorée** du cas sans MTP : T3 gagne de 4 k à 176 k, le gain kernel FA est
stabilisé. Le saut T=5→6 (§6.1) est hors sujet sans MTP (rows 6-8 n'existent pas en
decode pur) ; les seuls levers T3 restants sans MTP = overhead fixe court ctx (~25 µs)
et efficacité TC (~×2-3 théorique, rebuild).

### 6.3 Re-test b2048 sur T3 + courbe complète 16k → 196k (22/10) — le poids résiduel de b2048

**Origine de la piste** (détail : `b2048_REEVAL_datas.md` §0) : née le 17/10 par
élimination — 3ᵉ levier d'une session « quick wins sans rebuild » (après le patch
`42d9581` rejeté et les env VOLTA neutres), forcée par le scheduler ollama qui
choisissait `-b 512` tout seul (fit VRAM échoué à 200k → cascade 2048→1024→512, le
pire des 4) ; le prix = le ctx 200k→167k payé pour libérer le workspace b2048
(60 MB de marge) : le b2048 a toujours été acheté contre ~16,5 % du contexte.

Le levier b2048 (17/10 : ctx 200k→167k en échange de `-b 2048`) a été re-benché sur
le `.so` T3 prod (d2df07b8), MTP OFF, car T3 a changé les termes du trade-off. Deux runs,
même protocole (2-3 reps, ordre alterné, serveur frais par leg, port 8917, KV q8_0,
FA auto, H2 OFF, `GGML_CUDA_VOLTA_Q6_W4R4=1`) : `ab_batch_t3.py` (114 688 / 196 608, 3 reps)
et `ab_batch_t3_curve.py` (7 points 16k→196k, 2 reps). 40 mesures, 0 erreur, rc=0,
service restauré actif. Données : `b2048_REEVAL_datas.md` (§7 + §7.5),
`ab_batch_t3{,_curve}_results.jsonl` / `_summary.md`.

**Delta b2048-sur-T3 par contexte (médianes, Δ = b2048 vs b512, T3 constant) :**

| ctx | Δ decode | Δ prefill | Δ wall (s par requête) |
|---:|---:|---:|---:|
| 16 384 | 0 % | +19 % | −2.9 s (−14 %) |
| 32 768 | 0 % | +17 % | −5.7 s (−13 %) |
| 65 536 | 0 % | +14 % | −11.4 s (−12 %) |
| 98 304 | +17 % | +12 % | −18.3 s (−11 %) |
| 114 688 | +9.5 % (3 reps) | +11.5 % | −20.9 s (−10 %) |
| 131 072 | +44 % | +11 % | −25.8 s (−10 %) |
| 163 840 | +39 % | +10 % | −32.8 s (−9.5 %) |
| 196 608 | +34.5 % | +9 % | −39.9 s (−8.8 %) |

**Lecture :**
- **Le gain decode b2048 n'est pas une ligne droite : il est 0 % jusqu'à 65k, puis rupture
  brutale entre 65k et 131k (+44 %), plateau +35-45 % au-dessus.** Le mécanisme n'est pas
  le chunk de prefill (decode = rows 1) : c'est le **workspace CUDA** proportionnel à
  `-b` qui pilote le K-parallelism du FA sur KV longue. La jambe b512 s'effondre (13.5-14
  t/s @131-196k) pendant que b2048 descend en douceur (18-20) → ce n'est pas un gain
  actif, c'est **b512 qui s'écroule** au-delà du seuil (~98k).
- **Le prefill est le vrai levier constant : +9 à +19 %, décroissant avec le ctx**
  (plus gros sur les petites requêtes). T3 ne touche pas le prefill → 100 % batch.
- Le +9.5 % @114k (3 reps) est un point isolé sous le seuil de rupture — replier vers la
  courbe 98k→131k, ne pas le prendre au pied de la lettre.
- **Comparaison ancien .so (17/10)** : le gain decode long-ctx b2048 (+35-51 % @98-196k)
  existe déjà sur le .so sans T3 → **T3 n'a pas absorbé le gain b2048 au long ctx**.
  Le « effacement » perçu au re-test 2 points venait d'un seul point (114k) sous le seuil.

**Verdict : garder b2048.** Le prefill +10-19 % est payant dès 16k ; le decode long-ctx
+35 % vit au-delà du seuil ~98k ; le drop ctx 200k→167k n'ampute que le régime >167k qui,
sans b2048 (b512 forcé par le fit VRAM), se paierait le −35 % decode + −10 % prefill.
Le seul point faible structurel reste la **marge VRAM 60 MB** (piège 17/10 : ne rien
ajouter à ce runner sans redescendre le ctx).

---

## 7. Rollbacks (simples, testés conceptuellement)

**Re-activer H2 (1 ligne) :** retirer `GGML_CUDA_VOLTA_Q8_FATTN_TC=0` du service
→ `sudo systemctl daemon-reload && sudo systemctl restart ollama-v100`.

**Désactiver T3 sans rollback (gate de secours) :** ajouter `Environment="GGML_CUDA_FA_NO_VOLTA_SMALL=1"`
→ reload + restart. (H2 restant OFF, on retombe sur le wmma stock — plus lent que T3.)

**Rollback `.so` T3 → ancien prod H2 (complet) :**
```bash
sudo cp /var/backups/llama-experiments/libggml-cuda.so.prod-a4b6f440-20261018-190701 \
  /home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so
sudo systemctl restart ollama-v100
```
(Restaurer l'ancien `.so` = H2 seul, a4b6f440. Penser à ré-activer H2 = retirer le `FATTN_TC=0`.)

---

## 8. Méthodologie bench (reproductible)

- **Superviseur obligatoire** : `python3 supervisor_bench.py run <bench.py> [args] --detach`
  (README `bench/supervisor_bench_README.md` = autorité). Le superviseur **ne restaure JAMAIS le
  service** — le bench le fait via `restore_service()`. 1 superviseur à la fois
  (lock `/tmp/kolbo-supervisor-bench.lock`).
- Harnais commun : `bench/night_bench.py` (stop/start service, V100 libérée, serveur frais par leg,
  port **8917**, `CUDA_VISIBLE_DEVICES=0`, `gputimes_sampler.py`).
- **MTP OFF** pour mesurer T3/H3 (gold standard, pas de cycle MTP). **MTP ON** pour T1/H2/n-max
  (config prod exacte). Toujours vérifier que la **draft acceptance est identique entre legs**
  (sinon le delta est confondu avec le MTP). Toujours confirmer service restauré + port 8917 libre
  + V100 reprise par le runner.
- Interleave 2 reps, points ctx 114 688 + 160 000 (ou 196 608 MTP-OFF).

---

## 9. Pistes restantes (non actionnables « vite », gardées pour plus tard)

1. **`--spec-draft-n-max 2`** : micro-tweak non testé. Courbe d'acceptance encore élevée à pos0/pos1
   (0.97/0.90), baisse à 0.83 à pos2, 0.71 à pos3. Possibilité d'un 1-2 % si 2 bat 3. ROI faible.
2. **Commits MTP récents d'ik_llama** (ikawrakow/ik_llama.cpp, actif) :
   - #2678 « Avoid computing output for MTP in Prompt Processing » — éviter le calcul MTP au prefill.
   - #2687 « reduce MTP output buffer allocation ».
   Potentiel sur notre chemin decode/prefill MTP, MAIS = **port + rebuild lourd** d'un fork différent
   (ik_llama vs anyei) → hors périmètre « testable vite sans rebuild ».
3. **NVFP4 / quants exotiques** (IQ4_KS, IQ1_KT…) d'ik_llama : pas de bénéfice rapide sur le decode
   memory-bound V100, et quant du modèle déjà fixé (q6). Non applicable.
4. **FA ik_llama sur Volta** : leur doc `pascal-volta.md` confirme que le FA Volta stock = `wmma`
   pour head 256 / KV f16-q8_0, n_q 1-8 et >8. C'est exactement le chemin que **T3 a déjà dépassé**
   (T3 = m8n8k4 anyei > wmma stock). Rien de nouveau à exploiter.
5. **T3 v2 (rebuild)** : efficacité m8n8k4 ~7-10 % du pic TC (bornes : mémoire 30 µs / TC 42 µs
   vs 584 µs @107k T=1, §6.2). Espace théorique ×2-3, ROI app estimé +4-5 % decode long ctx
   MTP-OFF. Leviers : ILP/accumulateurs par thread, syncs CTA, FATTN_VS_STEP. Hors « testable
   vite » (rebuild .so + risque) — à garder si le FA redevient le 1ᵉʳ poste.
   **Mise à jour 27/10** : le levier « D » (vectorisation du load V q8_0 via staging smem,
   `LDG.128`) est **testé et écarté** — `bench/mb_fa_t3v2_verdict.md` (dilemme smem 48 Ko /
   128 positions lues en phase 2 ; dead-end). Reste : LDG.128 global sans staging, partage
   de fragments B.
   **Mise à jour 28/10 — tour complet des leviers ILP/partage B** :
   - Grille B × ILP résolue, **`bench/t3_grille_B_ILP_20261028.md`** (4 cellules mesurées).
   - Coût du revert du levier B : **−7,1 % @T=1** (no-B, e19b6112) — confirme le +6,8 % de B.
   - ILP cheap = prefetch L2 (`CCTL.E.PF2`, 0 reg/0 smem/bit-exact) : **v1 (lignes courantes)
     −1,5 % · v2 (itération suivante) −14,2 % @T=1 → NO-GO**. Cause prouvée SASS : l'asm
     `prefetch.global.L2` fait **tomber le hint `.CONSTANT` des 17 `LDG.E.128` du levier C**
     (confrontation prod/nob/kpf) ; le prefetch « 0-reg » n'est pas gratuit en cache.
   - **Partage de fragments B : NO-GO** — même mur registers/smem (pas de buffer possible à
     128 regs ; smem = mur du levier D). La **réécriture structurelle à < 128 regs** restait
     la seule piste.
   **Mise à jour 28/10 (après-midi) — rewrite « carte blanche » vldg : GO fort** :
   La réécriture structurelle a été faite **et elle tient** — `bench/t3_vldg_verdict_20261028.md`.
   **vldg** = double-buffer logiciel des fenêtres V en phase 2 (la fenêtre du chunk `ch+1`
   est chargée **pendant** les mmas du chunk `ch` ; chunk 0 chargé avant la boucle) :
   - **bit-identique** (maxrel 0.00e+00, mêmes octets / même déquant / mêmes mma),
   - **REG:128 STACK:0** → 4 CTA/SM (levier B intact), **17 LDG.E.128.CONSTANT intacts**
     (levier C intact — pas d'asm prefetch, cf. leçon KPF), 3792 instr = prod ;
   - le diff SASS prod↔vldg = 48 instr sur 3792 (réordonnancement local, aucune
     instruction ajoutée) — le gain vient du **réordonnancement logiciel** des loads V.
   - Microbench T=1..8 n_kv=107351 (2 runs, le 2ᵉ en 3 voies REF/CVEC/VLDG avec
     CVEC-vs-REF ≈ 0 → pas d'artefact de build) : **T=1 +11,1..11,4 %, T=6..8 +19,6 %**,
     gain croissant avec T. **Seuil GO (≥ +2 % @T=1) largement tenu.**
   - **A/B e2e MTP-OFF** (`t3_vldg_bench.py`, 2 reps interleave, serveurs frais/leg)
     : **decode +2.25 % @114 k (22.68→23.19), +2.81 % @196 k (20.65→21.23)** ;
     prefill ±0.12 %, VRAM pic identique (30 988 MiB) — gain e2e confirmé.
   - **A/B e2e MTP-ON** (`t3_vldg_mtp_bench.py`, config prod exacte, 114 k/160 k)
     : **decode +1.76 % @114 k (15.95→16.23), +0.40 % @160 k (24.49→24.82)** ;
     **acceptance MTP identique A vs B** (acc rate/pos au millième) → bit-exact
     confirmé en MTP ON (chemin f16 draft + q8_0 target). Signe constant, aucun
     point négatif. Gain dilué par l'overhead du draft model (comportement attendu).
   - **DÉPLOYÉ EN PROD le 28/10 15:27 UTC** (GO user « yes GO! ») : rebuild dual-arch
     sm_70+sm_61 (`v100/dual-arch`, `.cuh` vldg posé à la place du cvec, 7 objets
     rebuildés) → `v100/dual-arch/lib/libggml-cuda-dual.so` **8035caf4** — sa section
     sm_70 du kernel T3 est **byte-identique** au h2lib vldg 87f85442 validé
     (md5 streams bf53c498, 3792 instr / 17 LDG.E.128.CONSTANT / 512 HMMA intacts).
     Déploiement supervisé `bench/deploy_vldg.py` (stop → backup prod 4fd152f9 →
     swap → start → sanity gen OK → verify, rc=0 en 95 s). **Prod actuelle = 8035caf4**.
     Rollback : `/var/backups/llama-experiments/libggml-cuda.so.prod-4fd152f9-20261028-152612`
     + `sudo systemctl restart ollama-v100`. Post-deploy : service active, V100 0 %, 0
     erreur CUDA, réponse sanity cohérente.
   - **Reste (décision user)** : re-vérif post-deploy des régimes prod (2048/65k/98k +
     MTP ON config prod) dans la prochaine fenêtre bench ; l'anomalie MTP-ON préexistante
     (t/s bas + ordre ctx inversé, cf. verdict §6) est à investiguer séparément.     Le microbench/e2e GO ne déclenche aucun déploiement seul.
6. **Overhead fixe T3 court ctx** (~25 µs/launch, pèse 50 % @4k) : négligeable en app (quelques
   tokens de court ctx par session) — mention, pas un levier.

---

## 10. Pointeurs artefacts (détail brut)

Tous dans `/home/guile/llama-experiments/bench/` :
- Synthèse nuit + décisions : `BILAN_NUIT_20261018.md`
- T3 : `t3_bench_{summary.md,results.jsonl,server.log,gputimes.tsv}`, `deploy_t3.sh`, `deploy_t3_190701.log`
- T1 : `t1_bench_*`, `t1_h2_bench.py`
- H2 (T1T3) : `t1t3_h2_bench_*`
- H2 (confl .so T3 prod) : `t3_h2_bench_*`
- n-max (19/10) : `nmax_bench_{py,summary.md,results.jsonl,server.log,gputimes.tsv}`, `gguf_mtp_probe.py`
- Microbench kernel rows 1-8 (21/10) : `mb_fa.cu` (harnais dlopen .so prod), `mb_fa` (bin),
  `mb_fa_bench.py` (probe/full), `mb_fa_results.jsonl`, `mb_fa_summary.md`
- Microbench T3 sans MTP (22/10) : `t3_sweep_bench.py` (sweep T=1/2 × n_kv, réutilise `mb_fa`),
  `t3_sweep_results.jsonl`, `t3_sweep_summary.md`
- Re-test b2048 sur T3 (22/10) : `ab_batch_t3.py`, `ab_batch_t3_{results.jsonl,summary.md}` ;
  courbe complète 16k→196k : `ab_batch_t3_curve.py`, `ab_batch_t3_curve_{results.jsonl,summary.md}` ;
  consolidé décision : `b2048_REEVAL_datas.md`
- Night bench (b/t/H3) : `night_bench_*`, `night_bench_depeouille.md`
- Microbench T3 v2 levier D (27/10) : `mb_fa_t3v2_bench.py`, `mb_fa_t3v2_results.jsonl`,
  `mb_fa_t3v2_summary.md`, **`mb_fa_t3v2_verdict.md` (NO-GO)** ; artefacts des 2 designs
  rejetés dans `h2/cupti129-test/e2/t3v2/` (`.v21-fullstep-2691b94e.cuh`,
  `.v22-warp.cuh` 2db57e47, `.v22-warp-3501b4b7.cuh` 3501b4b7, baseline `.cvec.cuh` dbd4987c).
- Grille B × ILP complète (28/10) : **`t3_grille_B_ILP_20261028.md`** (no-B, KPF v1/v2,
  diagnostic SASS du hint `.CONSTANT`), `mb_fa_nob_{bench.py,results.jsonl,summary.md}`,
  `mb_fa_kpf_{bench.py,results.jsonl,summary.md}`, `mb_fa_kpf2_{bench.py,results.jsonl,summary.md}` ;
  variantes dans `h2/cupti129-test/e2/t3v2/` (`.nob.cuh` e21b528d, `.kpf.cuh` bdade32b,
  `.kpf2.cuh` 15923afa) et `.so` dans `h2/cupti129-test/e2/h2lib/lib/` (nob e19b6112, kpf
  7a87cf78, kpf2 1a8ba52e).
- Rewrite T3 « carte blanche » **vldg** (28/10 après-midi, **DÉPLOYÉ 15:27 UTC**) :
  **`t3_vldg_verdict_20261028.md`** (double-buffer V phase 2, bit-exact, REG:128, leviers
  B+C intacts, T=1 +11 % … T=8 +19,6 % ; e2e MTP-OFF +2.25/+2.81 % @114/196 k, MTP-ON
  +1.76/+0.40 % @114/160 k) ; `mb_fa_vldg_{bench.py,results.jsonl,summary.md}`
  et `mb_fa_vldg2_{bench.py,results.jsonl,summary.md}` (run 3 voies REF/CVEC/VLDG) ;
  `.cuh` = `h2/cupti129-test/e2/t3v2/fattn-mma-volta-small.vldg.cuh` ; `.so` =
  `h2/cupti129-test/e2/h2lib/lib/libggml-cuda-vldg.so` (87f85442, non-prod) + baseline
  rebuild `libggml-cuda-cvec.so` (723c0016) ; **prod déployée** =
  `v100/dual-arch/lib/libggml-cuda-dual.so` (**8035caf4**, dual sm_70+sm_61, sm_70 du
  kernel byte-identique au 87f85442) ; rollback =
  `/var/backups/llama-experiments/libggml-cuda.so.prod-4fd152f9-20261028-152612`
  (4fd152f9) + `sudo systemctl restart ollama-v100` ; déploiement : `bench/deploy_vldg.py`
- Superviseur : `supervisor_bench.py`, `supervisor_bench_README.md`, `supervisor_bench.log`
- Probes : `gguf_mtp_probe.py` (header GGUF → nb modules MTP)

Source fork : `/home/guile/llama-experiments/v100-repro/work/llama-volta-merged/`
`.so` builds : `$H2LIB/lib/` (§3).
