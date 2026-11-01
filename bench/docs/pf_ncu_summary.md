# PREFILL V100 — NCU FA + GEMM FFN (kernels prod) — 30/10–31/10

Microbenches offline `mb_fa_pf` (dlopen du `.so` prod) et `mb_gemm_pf` (dlopen du cuBLAS
prod) — **SANS rebuild, SANS toucher au service `ollama-v100`** (resté `active` avant/après,
V100 toujours par UUID). Kernels FA et GEMM **exactement** ceux de la trace prod :

- `flash_attn_ext_f16<256,256,32,2,false,false,false,false>` — 87.804 s / 49.1 % du prefill
- `void cutlass::Kernel2<cutlass_70_tensorop_s884gemm_f16_128x128_tn_align8>` — 64.633 s / 36.1 %

SM clock identique sur les 3 profils NCU (1.23 GHz) → les % de pipeline sont comparables.

---

## 1. FA prefill (T=2048, n_kv=114688, mask=0 → full attention)

Grid `(768,1,1)` block `(32,4,1)`=128 thr — **identique à la prod**.
Durée : **bare 194.06 ms** / **NCU 216.24 ms** (overhead ~11 %).

| Famille | Valeur |
|---|---|
| **Occupancy théorique = atteinte** | **6.25 %** (4 warps/SM sur 64) |
| Regs/thread | **255** (plafond Volta) |
| Smem dynamique/block | **67.58 KB** → **1 block/SM, BINDING** (regs : 2, smem : 1) |
| Waves/SM | 9.60 |
| **Tensor-core HMMA active** | **27.44 %** du pic |
| Compute (SM) / SM busy | 30.38 % / 31.30 % |
| Issue slots / IPC | 20.07 % / 0.80 |
| DRAM throughput | **4.66 %** (8.95 GB lu) |
| L2 hit rate | 94.45 % |
| L1/TEX hit rate | 31.29 % |
| Mem busy / max BW | 44.21 % / 38.28 % (41.65 GB/s) |
| Warps actifs/scheduler | 1.00 |
| **Warps éligibles/scheduler** | **0.20** (No Eligible 79.93 %) |
| Warp cycles/inst émise | 4.99 |

**Verdict FA : OCCUPANCY / LATENCY-bound.** La smem 67.58 KB bride à 1 block/SM →
1 warp/scheduler → la latence HMMA de Volta n'est pas masquée → tensor-core à 27 %,
SM à 31 %. **Pas du tout memory-BW-bound (DRAM 4.66 %)** : ni la BW, ni la quant KV ne
feront avancer le prefill FA. NCU (OPT) : *« theoretical occupancy (6.2 %) limited by the
required amount of shared memory »*, Est. Local Speedup **93.75 %** (gain potentiel énorme
SI on libère l'occupancy).

Pour y arriver : **re-tile du kernel FA** — smem < ~48 KB/block ET regs < 170 (3 blocks/SM)
ou < 128 (4). Template prod verrouillé `(256,256,32,2)` → ça touche le `.cu` ggml-cuda +
rebuild du `.so`, pas un flag.

---

## 1bis. Variantes de tile FA — A/B direct dans le .so prod (31/10, T=2048, n_kv=114688, mask=0)

Méthode : `mb_fa_tile.cu` dlopène le `libggml-cuda.so` prod et appelle directement les
instantiations exportées `ggml_cuda_flash_attn_ext_mma_f16_case<DKQ,DV,ncols1,ncols2>`
(SANS rebuild, SANS toucher au service). Les « variantes (128,128,32,2)/(128,256,32,2) »
demandées ne sont PAS des combos du template FA : DKQ/DV = head_length = 256,256 (fixes,
c'est la dim de tête du modèle) — les re-tiles réellement disponibles sans rebuild sont
les couples (ncols1, ncols2), ncols1 = tokens Q par CTA, ncols2 = têtes KV par CTA
(GQA=6 → ncols2 ∈ {1,2}).

### Durées bare (médiane, 5 iters, même process, même buffers)

| Variante | kernel | grid | bare | Δ vs 32x2 |
|---|---|---|---:|---:|
| **32x2** (baseline prod) | `flash_attn_ext_f16<256,256,32,2>` | 768 | **194.2 ms** | — |
| **16x2** (tile 32-col NInfer sm70) | `flash_attn_ext_f16<256,256,16,2>` | 1536 | **163.0 ms** | **−16.1 %** |
| 32x1 | `flash_attn_ext_f16<256,256,32,1>` | 1536 | 387.1 ms | +99.3 % |
| 64x1 | `flash_attn_ext_f16<256,256,64,1>` | 768 | 402.6 ms | +107.3 % |

Sorties finites et cohérentes pour 32x2/64x1/16x2 (écart float ~1e-3). **`16x1` est
mort sur sm_70** : l'instance existe (symbole exporté) mais le kernel
`flash_attn_ext_f16<256,256,16,1>` n'a PAS de code device 700 (fallback Ampere ncols16 =
config 256 thr sans device code Volta) → « no device code compatible with CUDA arch 700 »
+ unspecified launch failure au warmup (process bench tué, GPU/service non impactés).

### Métriques NCU (detailed, 1 pass, même setup que baseline `pf_ncu_fa`)

| Métrique | 32x2 (base) | 16x2 | 32x1 | 64x1 |
|---|---:|---:|---:|---:|
| Durée NCU | 216.32 ms | **182.29 ms** | 420.17 ms | 438.25 ms |
| grid / block | 768 / 128 | 1536 / 128 | 1536 / 128 | 768 / 128 |
| Regs/thread | 255 | **253** | 255 | 255 |
| Smem dynamique/block | 67.58 KB (1/SM) | **35.07 KB (2/SM)** | 36.35 KB (2/SM) | 67.58 KB (1/SM) |
| Occupancy atteinte | 6.25 % | **12.41 %** | 12.32 % | 6.25 % |
| HMMA cycles actifs | 27.43 % | **32.61 %** | 13.89 % | 13.61 % |
| Issue slots | 20.06 % | **28.17 %** | 16.43 % | 13.15 % |
| Warps active/sched (NCU warp) | 1.00 | **1.98** | — | — |
| Warps eligible/sched | 0.20 | **0.35** | — | — |
| Warp cycles/issued | 4.99 | 7.06 | — | — |
| One-or-more eligible | 20.07 % | 28.08 % | — | — |
| DRAM thr. / DRAM read | 4.28 % / 8.21 GB | 5.71 % / 9.28 GB | 1.73 % / 6.28 GB | 1.61 % / 6.28 GB |
| HMMA inst (identique partout) | 8.456 G | 8.456 G | 8.456 G | 8.456 G |

Lecture : même travail HMMA (8.456 G inst partout). Le 16x2 double l'occupancy
(smem 67.6→35.1 KB, 1→2 blocks/SM) et la latence HMMA est mieux masquée : HMMA 27.4→32.6 %
et durée −15.8 % sous NCU. Le 16x2 garde ncols2=2 → le config sm70 dédié
(nstages=1, nbatch_fa=32, « NInfer sm70 32-col ») s'applique, et le QK/VK par CTA diminue
(tête 256 sur 32 colonnes au lieu de 64) → moins de pression smem/combine, HMMA plus
ronds. Les variantes ncols2=1 (32x1, 64x1) perdent le double-buffering/pipelining GQA
(nstages=0, 1 tète/CTA, Q f16 non partagé) : 2× plus lent, écart à oublier.

### Verdict variantes
- **Retenir le tile (ncols1,ncols2) = (16,2)** : −16 % FA prefill sans toucher au code,
  mesuré sur le .so prod exact. FA = 49.1 % du prefill prod → ~8 % de gain prefill global.
- Les 253 regs restent le 2e limiteur (2 blocks/SM max) ; pour aller ≥3 blocks il faudrait
  un kernel ré-écrit (regs < 170) — hors périmètre d'un simple re-tile.
- Chemin prod : le dispatch `fattn.cu` choisit ncols1=32 (case 32/ncols2) ; le passage en
  16x2 exige soit le build `v100/dual-arch` (source déjà adaptée, config 32-col en
  nstages=1/Q_in_reg=false + kernel compact 48 KB en option via gate `use_volta_2cta` qui
  NE passe PAS au shape T=2048/n_kv=114688 : ntiles_dst=384 vs nsm=80 → efficacité 100 %
  à 1 wave mais 75 % < 75 % à 2 waves → gate fausse, et le config compact attend
  nbatch_V2=64/nbatch_combine=128 alors que le .so servi utilise le config
  nbatch_combine=128/nbatch_V2=128/nstages=2/Q_in_reg=true), soit un changement du choix
  de ncols1 dans `fattn.cu` + rebuild. **Pas de flag env existant** pour forcer 16x2 sur
  le .so servi actuel.
- Artefacts : `mb_fa_tile.cu`/`mb_fa_tile` · `mb_fa_tile_2048_bare.jsonl`/`_stderr.log` ·
  `pf_ncu_fa_t{32x2,64x1,16x2,32x1}.ncu-rep|.csv|_console.log|_stderr.log` ·
  `pf_fa_t16x2_warp.txt` (Scheduler/WarpState 16x2).

Caveat : 16x2 mesuré à n_kv=114688 (grid 1536 = 2 waves) ; le gain se maintient a fortiori
en causal (mêmes regs/smem), mais le dispatch prod réel (chunks -b 2048) doit être revu au
moment du rebuild.

## 2. GEMM FFN prefill (T=2048, cuBLAS prod, math mode prod `TF32_TENSOR_OP`)

Kernel exact prod reproduit : `cutlass_70_tensorop_s884gemm_f16_128x128_tn_align8`,
**même grid, même regs (236), même smem (32.77 KB dyn)** que la trace prod.

### FFN up : m=17408 k=5120 n=2048 (5120→17408)
- Grid local `(128,17,1)` = 2176 CTA = **identique à la prod** (17408/128=136 tiles m,
  2048/128=16 tiles n, regroupés 128×17 par cuBLAS — total CTAs égal, même travail).
- Durée : **bare 4.14–4.18 ms** / **NCU 4.45 ms** — prod avg grille `128x17` ≈ **4.41 ms**
  → le microbench **colle à la prod** (co-résidence service incluse, ~0 d'écart).

| Métrique | Valeur |
|---|---|
| **Tensor-core HMMA active** | **82.79 %** du pic |
| **Compute (SM)** | **81.14 %** |
| Shared pipeline (dont tensor) | **82.8 %** (highest-utilized) |
| DRAM throughput | 18.20 % (586 MB lu / 142 MB écrit) |
| Memory Throughput | 73.34 % (163 GB/s) |
| L2 hit rate | 89.89 % | L1/TEX hit rate | 1.43 % |
| Occupancy théo / atteinte | 12.50 % / **12.27 %** |
| Regs / smem dyn | 236 / 32.77 KB (limiteurs : 2 blocks regs, 2 blocks smem) |
| Waves/SM | 13.60 |
| Issue slots | 52.32 % |
| Warps actifs / éligibles par scheduler | 1.97 / 0.79 |

### FFN down : m=5120 k=17408 n=2048 (17408→5120)
- Grid local `(128,5,1)` = 640 CTA = **identique à la prod** (5120/128=40 × 16).
- Durée : **bare ~3.84–4.13 ms** / **NCU 4.13 ms** — prod avg grille `128x5` ≈ **2.79 ms**
  (la grille `128x5` de la trace regroupe probablement d'autres GEMM plus petits ; la
  correspondance n'est pas 1:1 au µs, mais le kernel/regs/smem sont identiques).

| Métrique | Valeur |
|---|---|
| **Tensor-core HMMA active** | **87.86 %** du pic |
| **Compute (SM)** | **87.69 %** |
| Shared pipeline (dont tensor) | **87.9 %** (highest-utilized) |
| DRAM throughput | 16.51 % (571 MB lu / 40 MB écrit) |
| Memory Throughput | 77.70 % (148 GB/s) |
| L2 hit rate | 89.80 % | L1/TEX hit rate | 1.83 % |
| Occupancy théo / atteinte | 12.50 % / 12.49 % |
| Regs / smem dyn | 236 / 32.77 KB |
| Waves/SM | 4.00 |
| Issue slots | 55.15 % |
| Warps actifs / éligibles par scheduler | 2.00 / 0.85 |

NCU sur les deux GEMM : **INF** « *utilizing greater than 80 % of the available compute or
memory performance… work will likely need to be shifted from the most utilized to another
unit* ». Seule OPT actionnable : accès shared uncoalescés → Est. Speedup **1.9 %** (up) /
**0.58 %** (down). **Verdict GEMM : TENSOR-BOUND.** cuBLAS/CUTLASS est déjà au plafond
tensor-core de la V100 sur ces shapes ; l'occupancy (12.5 %) est basse en absolu mais
suffisante — 2 blocks/SM × HMMA long pipeline = tensor pipe à 83–88 %.

---

## 3. Attention vs GEMM — la décision

| | **FA prefill** | **GEMM FFN up** | **GEMM FFN down** |
|---|---|---|---|
| Temps prod (kernel) | **87.804 s (49.1 %)** | grille `128x17` : 30.487 s | grille `128x5` : 19.259 s |
| Durée microbench NCU / bare | 216.24 / 194.06 ms | 4.45 / ~4.15 ms | 4.13 / ~3.84–4.13 ms |
| **Tensor-core HMMA** | **27.44 %** | **82.79 %** | **87.86 %** |
| Compute (SM) | 30.38 % | 81.14 % | 87.69 % |
| **Occupancy atteinte** | **6.25 %** | 12.27 % | 12.49 % |
| Regs / smem | 255 / 67.58 KB (1 blk/SM) | 236 / 32.77 KB (2 blk/SM) | 236 / 32.77 KB |
| DRAM | 4.66 % | 18.20 % | 16.51 % |
| L2 hit | 94.45 % | 89.89 % | 89.80 % |
| Warps éligibles/sched | 0.20 | 0.79 | 0.85 |
| Gains NCU estimés | **Local 93.75 %** (si occupancy libérée) | 1.9 % (shared) | 0.58 % (shared) |
| Verdict | **latency/occupancy-bound — marge énorme** | **tensor-bound — au plafond** | **tensor-bound — au plafond** |

### Conclusion ROI

1. **Le GEMM n'a (presque) rien à donner.** À 83–88 % du pic tensor-core, un GEMM custom
   doit battre CUTLAS 128x128 align8 sur Volta — c'est le meilleur kernel que cuBLAS sait
   générer pour sm_70. L'unique levier NCU vaut ~1–2 %. Ne pas réécrire un GEMM.
   (Le seul angle non-NCU restant : vérifier que la prod n'ajoute pas de
   convert f32→f16 / copies autour du GEMM — la matrice B sort en f32 ; mais c'est du
   « gain de copie », pas du « gain de GEMM ».)

2. **Le FA est le candidat unique.** C'est le kernel le plus long (49.1 %), il tourne à
   27 % de tensor-core et 6.25 % d'occupancy, et son seul vrai limiteur est la **smem
   (67.58 KB → 1 block/SM)**. Ré-tiler (bloc Q plus petit, moins de positions en vol,
   smem < 48 KB, regs < 170) pour passer à 3–4 blocks/SM = masquer la latence HMMA.
   Potentiel NCU : ~93 % de local speedup sur ce kernel → si on atteint 2×, c'est ~3–4 s
   en moins sur le prefill prod, soit **~8–10 % du prefill total**.

3. **Décision : on réécrit le tile FA, pas le GEMM.**
   - Effort : modifier le template `flash_attn_ext_f16` dans ggml-cuda (`(256,256,32,2)`
     → ex. `(128,128,32,2)` ou `(128,256,32,2)`), rebuild du `.so`, bench A/B offline
     (`mb_fa_pf` déjà prêt, baseline bare = 194 ms) + validation numérique (la sortie
     finie est déjà checkée par le harness).
   - Risque contenu : kernel isolé, pas de flag prod à changer, le microbench permet de
     valider avant tout deploy.
   - Le GEMM reste à surveiller uniquement si le FA ré-tilé atteint un plateau : dans ce
     cas, le 2e levier serait delta-net (7.6 s) et les convert/silu (1.9+1.9 s), pas le
     GEMM.

### Caveats
- FA mesuré en **full attention (mask=0)** : durée majorée ~2× vs causal, mais
  regs/occupancy/pipe = régime stationnaire. Le gain relatif d'occupancy est
  transposable au causal (même tile, même limiteur).
- Co-résidence service actif sur la V100 : DRAM/L2 partagés ; les durées bare colloquent
  avec la prod (4.15 ms vs 4.41 ms prod) donc l'impact est faible, mais un A/B
  FA-before/after FA-tile devra se faire à service idle ou en tenant compte du bruit.

## Artefacts
- `mb_fa_pf.cu` / `mb_fa_pf` (FA, dlopen `.so` prod) · `run_pf_ncu.sh`
- `pf_ncu_fa.txt|.csv|_bare.jsonl|_console.log` · `pf_fa_warp.txt` (Scheduler/WarpState)
- `mb_gemm_pf.cu` / `mb_gemm_pf` (GEMM, dlopen cuBLAS prod, math mode prod) · `run_pf_ncu_gemm.sh`
- `pf_ncu_gemm_up.txt|.csv|_bare.jsonl` · `pf_ncu_gemm_down.txt|.csv|_bare.jsonl`
- `pf_gemm_up_warp.txt` · `pf_gemm_down_warp.txt`

---

## 4. SÉQUIELLE FA16 — 31/10 (voir DECISION_FA16_20261031.md pour la décision complète)

Patch env `GGML_CUDA_VOLTA_FA_NCOLS1` (fattn.cu) + `GGML_CUDA_VOLTA_FA_COMPACT_TMAX`
(fattn-mma-f16.cuh), build dual `d4954502`. Résultats clés :

| cas (T=2048, n_kv=114688) | microbench us_med | NCU (dispatch réel) | Δ e2e prefill @114k |
|---|---:|---|---:|
| baseline 32x2 (prod) | 192 251 | 213.50 ms (768, 67.58 KB, 1/SM) | — |
| 16x2 (NCols1=16) | 163 571 | 183.05 ms (1536, 35.07 KB, 2/SM) | +7.1 % |
| **compact (TMAX=8192)** | **120 615** | **134.75 ms** (768, 49.15 KB, 2/SM) | **+19.4 %** |

- Kernel compact dispatché : `flash_attn_ext_f16_volta_compact<256,256,32,2>`,
  2 blocks/SM, NCU 134.75 ms vs 213.50 baseline (−37 % sur le kernel).
- E2E llama-server (KV q8_0, -b 2048, MTP off) : +5.2 % @16k → +19.4 % @114k → +20.6 % @131k.
- **Decode : AUCUNE régression** (microbench T=1 : 759.8 vs 767.0 µs ; thermal check e2e
  131k : 22.48 vs 22.45 t/s). L'écart decode apparent du run e2e initial (16.34 t/s pour C)
  est un artefact de l'ordre du run (C tournait 3e = GPU chaud), prouvé par le thermal check.
- **Décision : lever = TMAX=8192 sur le .so dual. 16x2 rejeté (dominé). Rollout = servir
  le .so dual + env, ou upstrem du gate.**
