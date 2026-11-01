# BILAN GLOBAL — 3 semaines d'expériences V100 (03/10 → 24/10/2026)

Modèle : qwen3.8-27b-q6-mtp-vision (23 088 MB, blob 49330183) sur V100 32 GB (HBM2 ~900 GB/s),
Ollama `ollama-v100` (port 11434). Backend CUDA = plugin `libggml-cuda.so` swappable.
Toutes les mesures : interleave A/B, reps ≥ 2, service restauré, superviseur (README V1).

---

## 1. Le chiffre global

**Decode long-ctx (114k) MTP OFF, même protocole, .so prod successif :**

| étape | .so / config | decode @114k | Δ |
|---|---|---:|---:|
| 17/10 avant batch | a4b6f440 (H2), b512 | 12.08 t/s | — |
| 17/10 soir | b2048 (ctx 167k) | 17.39 t/s | **+44 %** |
| 18/10 | T3 (d2df07b8) | 18.85–19.99 t/s | **+13 %** |
| 22/10 | lba (fa3e75c2 = T3 + launch_bounds) | +2.2–2.8 % | cumul **≈ ×1.70** |
| 24/10 19:34 | lba4 (feaf05aa = lba + launch_bounds NT=1 → 4 blocs/SM) | +6,8 % microbench @T=1 ; **+1,7 % e2e @98k** (20:21) | prod |
| 25/10 17:49 | dual-arch (b014d026 = feaf05aa + ELF sm_61, mmproj sur la 1070) | **neutre** (ab_dual : −0,8 %) | prod |
| 26/10 21:21 | cvec (4fd152f9 = dual + load K q8_0 vectorisé 128-bit) | +10,7 % microbench @T=1 ; **+3,5 % e2e @98k** | **prod actuelle** |

**@196k** : 10.46 → 14.13 (b2048, +35 %) → 17.8–18.2 (T3, **+28.5 %**).
**Prefill** : +12–15 % (b2048) ; T3/lba : zéro impact (rows ≤ 8 = decode uniquement).
**MTP ON (régime prod)** : le gain batch se lit aussi en live (t/s prod 26–30 → 37–45 pré-T3,
58.9 t/s court ctx post-déploiements 18/10). Le ×1.68 MTP-ON vs MTP-OFF est le levier MTP
lui-même (run9d 27.27 vs run9e 45.76 ms/tok @90k, 24/10).

VRAM identique à chaque étape (30 988–32 436 MiB), 0 eviction depuis le 18/10 00:16.

---

## 2. Timeline des 3 semaines

### Semaine 1 (03–05/10) — Observation et autopsies
- **03/10** : A/B MTP **shortlist** (lm_head 131072 rows) → neutre (45.9–52.9 t/s, bruit) —
  `bench/results_ab_optim.jsonl`. Backup shortlist. Métrique Grafana `ollama_mtp_mean_len`.
- **04/10** : nuit de bench ; **segfaults par `cp` in-place du .so en service** → règle de survie
  (toujours stop AVANT swap). OP2 histogramme mmvq → **scope OP2 réfuté** : Q6_K est déjà W4R4
  (42 % du GEMV decode), Q8_0 32 %, Q5_K 13 %, fusion SwiGLU inactive. Piège `GGML_BACKEND_DL`
  (fallback CPU silencieux) documenté.
- **05/10** : autopsy P0 Q8_0 QPN (mma m8n8k4 quadpair-split) → **NO-GO, design mort**
  (`autopsy-20261005/DESIGN-NOTE-P0-Q8_0-QPN.md`).

### Semaine 2 (08–16/10) — Mission Kolbo (H2) + infrastructure
- **08/10** : P7 kernel Q6_K X4 → +1,1 % court ctx, reproductible, bit-identique — **sous le
  seuil de rentabilité (2 ms/cycle) → non intégré** (prod known-good e4afbb4a).
- **08–13/10** : **CUPTI** v1 puis **v2** (shim LD_PRELOAD `libcupath_v2.so`, run6/run7) :
  96,11 % occupancy du decode, 0 overlap ; NCU compteur DRAM ; E1 : `ggml_q8v14partial`
  kernel-bound (occupancy 12,5 %, ~19 % du pic HBM). Autopsies run6 : mmvq 64,8 %, partial
  14 % (proportionnel au ctx), shortlist 833 GB/s (plafond).
- **14/10** : **H2 DÉPLOYÉE** (a4b6f440) — FA Q8 tensor-core rows==4, notre kernel :
  **−12,5 % @107k** (RUN9C3, +9 %). H2b (découplage partitionnement PC=16) : bit-identique,
  **pas de gain** → fermée.
- **15/10** : RUN8 @36,3k (G1–G5 PASS, pas d'avantage court ctx), RUN9B @41k indécis.
  **Embargo H2 levé.**
- **16/10** : **superviseur V1** (réinjection terminal_id) + **chat_watchdog** (cron 5 min,
  compaction+relance des erreurs de contexte) — toute la suite a tourné sous cette chaîne.
  **Publication** : repo GitHub `pitouche/v100-bench-kit` v1.0 (450 fichiers ; **v2.0** du 01/11 = mission H2 extraite dans `opti-h2`) + post Reddit
  r/LocalAIStack (chiffre canonique H2 −11,8 % @107k).

### Semaine 3 (17–24/10) — T3, déploiements, clôture des leviers
- **17/10** : 42d9581 rejeté (bruit), env Q5_X4/Q8_FATTN_TC neutres. **Levier batch trouvé par
  élimination** : le scheduler ollama retombait à b512 (fit VRAM 60 %) → **b512→b2048 : decode
  +35–44 %, prefill +12–15 %** → déployé 23:50 avec **ctx 200k→167k** (seul point qui passe la
  fit, 60 MB de marge). `BILAN_SESSION_20261017.md`.
- **18/10 (la nuit T3)** : port anyei #153 `fattn-mma-volta-small` (**T3**) → **VALIDÉ :
  +13,07 % @114k, +28,52 % @196k, C-t3off≈A (méthodologie propre), prefill/VRAM neutres →
  DÉPLOYÉ 19:07 (md5 d2df07b8)**. T1 (mmsmt) : NO-GO (−11,3 % @160k). **H2 battue par T3 même
  sur sa géométrie rows==4 → H2 DÉSACTIVÉE** (`FATTN_TC=0`) : **+15–17 % MTP-ON**. Bug d'export
  de symbole mmsmt corrigé (dé-staticisation `ggml_cuda_mul_mat_cublas`).
  `BILAN_NUIT_20261018.md`.
- **19/10** : n-max 3 vs 4 → **−11/−13 %** → n-max 3 gardé. `nmax_bench_summary.md`.
- **22/10** : sweep microbench FA (T3 vs stock : **+51 % @T=1, +71 % @T=2** dès 32k, ~540 µs
  par 100k keys). **STEP 256/512 : INVALID** (bug de couverture 50–75 % positions sautées,
  garbage) — `fa_step_INVALID_note.md`. **lba (t3v2 = launch_bounds/NT, T3v2)** : A/B
  +1,3/+2,2/+2,8 % @32/65/98k → **DÉPLOYÉ (fa3e75c2) 23:14** via `deploy_lba.py`.
  Réévaluation b2048 (`b2048_REEVAL_datas.md`), courbe batch T3.
- **23/10** : NCU mmvq en détail (Q6_K T=4 : DRAM 59 % du pic, issue 63,4 % ; knobs
  rows-per-block 1/2/8 tous pires ; q6x4 neutre). A/B applicatifs q5x4 / q8rpb4 (Q8_0 T=4)
  neutres. nsys MTP breakdown (échec CUPTI tables).
- **24/10** : réconciliation 32,6 vs 63 t/s (overhead one-shot MTP ~0,77 s + acceptance
  contenu) ; bench régime 36k/90k (**FA = seul terme O(ctx)**, −19 % sur le pas 36k→90k) ;
  **run9d CUPTI @90k MTP-ON** : mmvq 69,8 %, FA 15,9 % (kernel déjà T3), mapped_rows 4,3 % ;
  **mtp-accept CLOS** (warning pos_max = faux positif checkpoint N−4 ; écart bench↔prod =
  sampling × contenu, draft chain fixée top_k(10) sans température) ; **08:00 : plus de bench
  MTP (user)** ; **run9e MTP-OFF baseline** : 45,76 ms/tok, mmvq 73,6 % = **82 % du peak HBM2**
  → les 3 CLOS tiennent sans MTP. **Fin du tour : aucun levier kernel >5 % restant.**
  **12:23 : NCU FA/T3 (le trou identifié le matin)** : T3 **latence-bound / low-occupancy**,
  ni au mur DRAM (53 % du pic @T=1) ni compute-bound (SM 21 %) — réouverture légitime du
  levier FA. Voir §3 et `ncu_fa_t3_summary.md`.
  **17:30 : « multi-kvhead » (240→960 blocks) RÉFUTÉ** (`t3_mkvh_verdict_20261024.md`) —
  le grid (1,60,4) était mal lu : z=4 = les 4 kvhead (qheads GQA déjà packés), y=60 = split-KV.
  **18:23 : levier A (pb forcé 120/240 = 2/4 waves) RÉFUTÉ** (`t3_pb_A_verdict_20261024.md`) —
  −2,8 à −21,8 % (pire) @T=1-5, neutre @T=6-8 : le pb auto 1 wave est déjà optimal, le
  combine `flash_attn_combine_results` + tail effects pèsent plus que l'overlap.
  **19:07 : levier B (launch_bounds NT=1 → minblocks=4, 162→124 regs, 0 spill, 3→4 blocs/SM)
  VALIDÉ : +6,8 % @T=1** (482→450 µs, min 475→442), T=2..8 contrôles à ±0,2 %
  (`t3_lbB_verdict_20261024.md`, `feaf05aa`). **19:34 : B DÉPLOYÉ en prod**
  (`deploy_lba4.py` sous superviseur, rc=0) — prod = `feaf05aa`, backup `fa3e75c2` dans
  `/var/backups/llama-experiments/`, sanity gen MTP-ON OK, service active.
  **20:21 : B CONFIRMÉ en e2e MTP-OFF** (`ab_lb4.py`, lba4 vs lba, serveur frais/leg,
  GGML_BACKEND_PATH, .so prod intact) : decode **+1,1 % @65k, +1,7 % @98k** (croissant avec
  le ctx, reps stables à ±0,01 t/s), prefill/wall neutres. Gain MTP-ON : **non mesurable en
  e2e par construction** (le verify T=2..4 garde NT≥2 identique dans les deux variants ; le
  T=1 ne pèse qu'aux rejets totaux → gain < 1 % noyé dans la variance d'acceptance MTP) —
  le gain existe, l'e2e MTP-OFF est la confirmation appropriée, non-item non bloquant
  (`t3_lbB_e2e_verdict_20261024.md`).

### Semaine 4 (25–26/10) — mmproj sur la 1070 + levier C
- **25/10** : **dual-arch DÉPLOYÉ** (`b014d026` = feaf05aa + ELF sm_61, `deploy_mmproj_1070.py` :
  binaire ollama patché `6f6d33a9` + drop-in `OLLAMA_MMPROJ_DEVICE=cuda1`) — le mmproj vision
  (~1,2 GB) vit sur la GTX 1070, le mode de crash OOM `clip_image_batch_encode` de 21/10 est
  supprimé. A/B V100 (ab_dual) : **neutre (−0,8 %)**. Prod = `b014d026`.
- **26/10** : **levier C (load K q8_0 vectorisé 128-bit) VALIDÉ + DÉPLOYÉ** :
  microbench mb_fa (REF=feaf05aa vs cvec, n_kv=107 351) : **+10,7 % @T=1** (451→402 µs),
  +6,2 à +11,5 % @T=2..8, **bit-identique (maxrel=0)** — `mb_fa_cvec_summary.md`.
  E2E A/B MTP-OFF cette nuit (`ab_cvec.py`, cvec f89418fc vs prod b014d026, serveur
  frais/leg, 32k/65k/98k, 2 reps) : decode **+1,9/+2,6/+3,5 %**, prefill/wall neutres —
  `ab_cvec_summary.md`. Déploiement 21:21 sous superviseur (`deploy_cvec.py`) : .so prod
  = **4fd152f9** (rebuild dual-arch + patch C : 150 ELF sm_70 + 150 sm_61), backup
  `b014d026-20261026-212004`. Gain e2e croissant avec le ctx comme T3/lba/cvec : FA reste
  le seul terme O(ctx), mmvq (73,6 %) reste le mur HBM2. **Tour des leviers kernel T3 :
  A réfuté, B clos, C clos — il ne reste que D) KV q4_0 (hors périmètre, risque qualité).**

### Semaine 5 (27–28/10) — levier D clos, tour T3 v2 (A–D) terminé
- **27/10** : **levier D (load V q8_0 vectorisé 128-bit) NO-GO** après 3 designs
  (sans staging 0653dc50 : −104 % @T=1 ; staging full-step : ~−50 % + crash occupancy ;
  warp-private : crash IMA — 32 rangées/warp vs 128 lues) — `mb_fa_t3v2_verdict.md`.
- **28/10** : **D clos définitivement** — attribution binaire du −104 % confirmée
  (cubin 213 = kernel prod exact q8_0 D=256 NT=1 : 128 `LDG.128` + spills STACK 112 B vs 0,
  4537 vs 3841 lignes SASS ; le `.so` dual repli est fidèle à prod sur les 4 cubins diff).
  **Grille B × ILP complète** (`t3_grille_B_ILP_20261028.md`) : no-B (revert du levier B)
  **−7,1 % @T=1** ; prefetch L2 du K « 0 registre » (`CCTL.E.PF2`) **v1 −1,5 % / v2 −14,2 %**
  — cause prouvée SASS : l'asm prefetch **fait tomber le hint `.CONSTANT` des 17 `LDG.E.128`
  du levier C** (prod/nob = 17 CONSTANT, kpf = 0). **Partage fragments B : NO-GO** (mur
  register/smem, pas de place à 128 regs). **Tour T3 v2 (A–D) = CLOS** ; le seul reste =
  réécriture structurelle à <128 regs (projet, ROI incertain) ou KV q4_0 (hors périmètre).
- **28/10 (après-midi) — carte blanche : la réécriture structurelle est faite et elle
  PASSE** → **vldg GO fort** (`t3_vldg_verdict_20261028.md`) : double-buffer logiciel des
  fenêtres V en phase 2 (fenêtre du chunk `ch+1` en vol pendant les mmas du chunk `ch`) —
  **bit-identique** (maxrel 0), **REG:128 STACK:0** (4 CTA/SM, levier B intact), 17
  `LDG.E.128.CONSTANT` intacts (levier C intact), diff SASS = 48 instr sur 3792 (réordonnancement
  local, 0 instr ajoutée). Microbench T=1..8 n_kv=107351 (run 2 en 3 voies REF/CVEC/VLDG,
  CVEC-vs-REF ≈ 0) : **T=1 +11,1 % … T=6..8 +19,6 %** (gain croissant avec T). A/B e2e
  **MTP-OFF** : **decode +2.25 % @114 k, +2.81 % @196 k** (prefill ±0.12 %, VRAM identique).
  A/B e2e **MTP-ON** (config prod, 114 k/160 k) : **decode +1.76 % @114 k, +0.40 % @160 k**,
  **acceptance MTP identique A vs B** → bit-exact en MTP ON. Signe constant, 0 point négatif.
  ⚠️ Anomalie préexistante observée en MTP-ON (affecte la PROD, pas vldg) : t/s absolus
  15.9/24.5 vs 45.7/36.3 du 18/10, ordre des ctx inversé, MTP-ON < MTP-OFF @114 k —
  à investiguer séparément.
  **DÉPLOYÉ EN PROD le 28/10 15:27 UTC** (GO user) : rebuild dual-arch sm_70+sm_61
  (`.cuh` vldg posé dans le tree dual, 7 objets rebuildés) → `v100/dual-arch/lib/libggml-cuda-dual.so`
  **8035caf4**, sa section sm_70 du kernel **byte-identique** au h2lib 87f85442 validé
  (md5 streams bf53c498) ; déploiement supervisé `deploy_vldg.py` (stop→backup 4fd152f9→
  swap→start→sanity OK→verify, rc=0 en 95 s). **Prod actuelle = 8035caf4** ; rollback =
  `/var/backups/llama-experiments/libggml-cuda.so.prod-4fd152f9-20261028-152612` +
  `sudo systemctl restart ollama-v100`. Reste : re-vérif des régimes prod (2048/65k/98k
  + MTP ON) dans la prochaine fenêtre bench (risque faible, kernel sm_70 byte-identique
  à celui déjà A/B).

---

## 3. Statut de T3 (la question posée)

T3 = port du kernel **anyei #153 `fattn-mma-volta-small`** (attention tensor-core m8n8k4,
Q rows 1–8, KV q8_0/f16, D 64/128/256) dans le dispatch Volta (`BEST_FATTN_KERNEL_VOLTA_SMALL`),
gate `GGML_CUDA_FA_NO_VOLTA_SMALL`. C'est **le seul kernel FA pour le decode** (rows ≤ 8) —
casse le stock `q8v14partial` (~19 % du pic, occupancy 12,5 %).

- **Validé 18/10** (nuit) : +13,07 % @114k / +28,52 % @196k, gain croissant avec le ctx
  (cohérent : FA = seul terme O(ctx)), zéro impact prefill/VRAM.
- **Déployé 18/10 19:07** (d2df07b8) → a **rendu H2 obsolète** (H2 rows==4 battue par T3 sur sa
  propre géométrie → désactivée, +15–17 % supplémentaires en prod MTP-ON).
- **Amélioré 22/10** : version **lba** (launch_bounds `NT<=1?3:2` / NT, T3v2) → +1,3–2,8 % →
  **prod actuelle fa3e75c2** (T3 + lba + branch W4R4 mmvq). STEP≠128 tenté → INVALID (coverage).
- **Revalidé 24/10** (run9e MTP-OFF) : le decode MTP-OFF utilise `ext_mma_volta_small<256,NT=1,q8_0>`
  = T3 — 384 µs/appel @74k keys, 14,2 % du busy. Microbench : T3 = 543 µs/100k keys vs 1 191
  stock @107k (T=1) — le mur du decode reste mmvq (HBM2), T3 a juste fait chuter le seul terme
  qui grossissait avec le ctx.

**NCU FA/T3 (24/10 12:23, `ncu_fa_t3_summary.md`) — le CLOS est RÉQUALIFIÉ** : le kernel est
  **latence-bound par sa géométrie**, ni au mur DRAM ni compute-bound :
- @T=1, n_kv≈107k : **DRAM 53,1 % du pic** (464 GB/s sur ~220 Mo de KV lue 1×), SM 21,2 %,
  stall long-scoreboard = **60,6 % des cycles** entre issues, No Eligible 77,7 %.
- Occupancy théorique plafonnée à **18,75 %** (12 warps/SM) par 162 regs/thread + 32 Ko
  static shared → 3 blocks/SM (grid 240 @T=1, 1,5 wave max). Masquage latence DRAM insuffisant
  par construction. À T≥6 le dispatch passe en (1,20,8) et le temps double : la forme
  « 1 block = 1 (qhead, kvhead) » ne scale pas au-delà de T=2.
- → T3 reste le meilleur kernel FA connu (le CLOS « meilleur kernel » tient), **mais il a de la
  marge théorique** : le CLOS « mur mémoire » est **réfuté**.
- **Piste « T3 multi-kvhead » REFUTEE (24/10 après-midi, `t3_mkvh_verdict_20261024.md`) —
  le grid NCU (1,60,4) était mal lu** : z=4 = les 4 kvhead (les 6 qheads GQA de chaque kvhead
  sont **déjà packés** dans le block, pack=6 ≤ 8 — « K/V are read once per KV head »),
  y=60 = **split-KV** (parallel_blocks : la boucle d'efficacité de `launch_fattn` pousse
  jusqu'à 1 wave = 240 slots / 4 dst-tiles). Le ×4 kvhead **existe déjà** ; dé-packer les
  qheads ne ferait que ×6 la relecture du KV à blocks constants. Et 240→960 blocks ne
  donnerait PAS 40 % d'occ : le plafond est **3 blocks/SM par block** (162 regs/thr,
  32,77 Ko/32 Ko config), pas par le nb de blocks (960 = 4 waves × 12 warps/SM, idem).
  Le KV est déjà lu ~1× (240 blocks × 974 Ko ≈ 234 Mo ≈ 239 Mo mesurés).
  **Leviers réels restants (par effort/gain, A/B microbench .so de bench, MTP-OFF T=1) :**
  A) forcer parallel_blocks 120/240 (2–4 waves) — **RÉFUTÉ 24/10 18:23** (−2,8 à −21,8 % @T=1-5) ;
  B) `__launch_bounds__(128,4)` sur NT=1 → **VALIDÉ 24/10 19:07 : +6,8 % @T=1** (124 regs, 0 spill,
  4 blocs/SM, `feaf05aa`) ; C) vectorisation 128-bit du load K q8_0 (5,9/32 B par secteur
  NCU) — **VALIDÉ 26/10 : +10,7 % microbench @T=1, +1,9/+2,6/+3,5 % e2e @32/65/98k, DÉPLOYÉ
  (4fd152f9)** ; D) KV q4_0 (hors périmètre, risque qualité). Validation : harnais `mb_fa`
  + probe inter-kernel.

**Bilan T3 en une ligne : le plus gros gain kernel des 3 semaines, en prod depuis 18/10,
encore la meilleure version connue de ce kernel — NCU 24/10 : PAS au mur (latence-bound,
occ 18,75 %, 3 blocks/SM par regs+smem). Le multi-kvhead est réfuté (les 4 kvhead sont
déjà le grid, y=60 = split-KV). Levier A (pb>60) RÉFUTÉ (−2,8 à −21,8 %). Levier B
(launch_bounds NT=1 → 4 blocs/SM, 124 regs, 0 spill) VALIDÉ +6,8 % @T=1 (feaf05aa),
e2e +1,7 % @98k. Levier C (load K q8_0 vectorisé 128-bit, 5 LDG.128 au lieu de
17 LDG.32 par block-pair, 122-128 regs, 0 spill) VALIDÉ +10,7 % microbench @T=1
(451→402 µs, bit-identique maxrel=0), e2e +1,9/+2,6/+3,5 % @32/65/98k, DÉPLOYÉ
26/10 21:21 (4fd152f9, dual-arch pour la 1070). Reste D) KV q4_0 (hors périmètre).
Voir `t3_mkvh_verdict_20261024.md`, `t3_pb_A_verdict_20261024.md`, `t3_lbB_verdict_20261024.md`,
`mb_fa_cvec_summary.md`, `ab_cvec_summary.md`.**

---

## 4. Table de tous les leviers (3 semaines)

| levier | origine | résultat | statut |
|---|---|---|---|
| **batch b512→b2048** | scheduler ollama | decode +35–44 %, prefill +12–15 % | **DÉPLOYÉ 17/10** |
| **ctx 200k→167k** | contrainte VRAM fit 60 % | seul point qui passe, 0 eviction | **DÉPLOYÉ 17/10** |
| **H2** (FA Q8 TC rows==4) | kernel maison | −12,5 % @107k (RUN9C3) | déployé 14/10 → **DÉSACTIVÉ 18/10** (battue par T3) |
| **T3** (anyei #153) | port fork anyei | +13 % @114k, +29 % @196k | **DÉPLOYÉ 18/10** (d2df07b8) |
| **T3 lba** (launch_bounds) | T3v2 | +1,3–2,8 % | **DÉPLOYÉ 22/10** (fa3e75c2 = prod) |
| **MTP n-max 3** (vs 4) | bench 19/10 | 4 : −11/−13 % | **3 gardé** |
| **W4R4 Q6_K** | déjà en service | +1 % | en prod (env) |
| **MTP shortlist** | bench 03/10 | neutre | fermé |
| 42d9581 (nwarps K-quant) | upstream | neutre (2× : 17/10, 18/10) | fermé |
| b4096 / b8192 / -t12 | nuit 18/10 | neutre / OOM / neutre | fermés |
| env Q5_X4, Q8_FATTN_TC | 17/10 | neutres (H2 off rend ce dernier inerte) | fermés |
| **P7** (Q6_K X4) | kernel maison | +1,1 % < seuil rentabilité | non intégré (08/10) |
| **P0** (Q8_0 QPN) | autopsy 05/10 | design mort | NO-GO |
| **OP2** (fusion mmvq) | 04/10 | réfuté (déjà W4R4) | NO-GO |
| **T1** (mmsmt anyei) | port 18/10 | −11,3 % @160k | NO-GO |
| H2b (PC=16) | 14/10 | bit-identique, pas de gain | fermé |
| q6x4, q6rp1/2/8, q5x4, q8rpb4 | 23/10 | neutres / lents | **mmvq CLOS 24/10** |
| STEP 256/512 | 22/10 | bug de couverture (garbage) | **INVALID** |
| **FA long-ctx / T3 multi-kvhead** | run9d + NCU 24/10 | grid mal lu : les 4 kvhead sont déjà le grid (z=4), y=60 = split-KV | **RÉFUTÉ 24/10 17:30** (`t3_mkvh_verdict_20261024.md`) |
| **A : pb forcé 120/240 (2/4 waves)** | NCU 24/10 | −2,8 à −21,8 % @T=1-5 (pire), neutre @T=6-8 ; pb auto 1 wave déjà optimal | **RÉFUTÉ 24/10 18:23** (`t3_pb_A_verdict_20261024.md`) |
| **B : launch_bounds NT=1 → minblocks=4** | NCU 24/10 | 162→124 regs, 0 spill, 3→4 blocs/SM : +6,8 % microbench @T=1 ; e2e decode **+1,1/+1,7 %** @65/98k (croissant avec le ctx) | **VALIDÉ + DÉPLOYÉ 24/10 19:34 + CONFIRMÉ e2e 20:21** (`feaf05aa`, `t3_lbB_e2e_verdict_20261024.md`) |
| **dual-arch (mmproj 1070)** | crashs OOM mmproj 21/10 | V100 neutre (ab_dual −0,8 %) ; mmproj déporté sur la GTX 1070 (encode ~3-6 s) | **DÉPLOYÉ 25/10 17:49** (`b014d026` + binaire 6f6d33a9 + drop-in `OLLAMA_MMPROJ_DEVICE=cuda1`) |
| **C : load K q8_0 vectorisé 128-bit** | NCU 24/10 (loads 5,9/32 B/secteur, long-scoreboard 60,6 %) | 17 LDG.32 → 5 LDG.128/block-pair, 122-128 regs, 0 spill : **+10,7 % microbench @T=1** (451→402 µs, bit-identique maxrel=0) ; e2e decode **+1,9/+2,6/+3,5 %** @32/65/98k, prefill/wall neutres | **VALIDÉ + DÉPLOYÉ 26/10 21:21** (`4fd152f9`, `deploy_cvec.py`, `mb_fa_cvec_summary.md`, `ab_cvec_summary.md`) |
| **D : load V q8_0 vectorisé 128-bit (T3 v2)** | NCU 24/10 (même famille de loads, côté V) | tent.1 sans staging 0653dc50 : −104 % @T=1 (825 vs 404 µs, bit-identique) — attribution confirmée SASS 28/10 (cubin 213, kernel prod exact : 128 LDG.128 + spills STACK 112 B vs 0 ; prod==cvec sur les 4 cubins diff) ; staging full-step ~−50 % + crash occupancy ; warp-private crash IMA (32 rangées/warp vs 128 lues) | **NO-GO CLOS 27/10 (confirmé 28/10)** (`mb_fa_t3v2_verdict.md`) |
| **no-B : revert du levier B (coin grille)** | T3 v2 28/10 | minblocks NT=1 4→3 : 162-166 regs, 3 blocs/SM : **−7,1 % @T=1** (404→433 µs), T=2..8 ±0,2 % ; corrobore le +6,8 % propre de B | **MESURÉ 28/10** (e19b6112, `t3_grille_B_ILP_20261028.md`) |
| **KPF v1 : prefetch.global.L2 du K (lignes courantes)** | T3 v2 28/10 | 3 CCTL.E.PF2/lane, 0 reg/0 smem/bit-exact : **−1,5 % @T=1** ; cause SASS : l'asm prefetch fait **tomber le hint `.CONSTANT` des 17 LDG.E.128 du levier C** (prod/nob = 17 CONSTANT, kpf = 0) | **NO-GO 28/10** (7a87cf78) |
| **KPF v2 : prefetch L2 du K (itération suivante)** | T3 v2 28/10 | même bloc asm, offset +gridDim.y*STEP (lead 1 it) : **−14,2 % @T=1** (404→462 µs), −3 à −10 % T=2..8 ; perte de hint `.CONSTANT` (prouvé) + pression L2 sur working set K+V 55,7 MiB >> L2 6 MiB | **NO-GO 28/10** (1a8ba52e) |
| **Partage fragments B / ILP à 128 regs** | OPTI §9.5, T3 v2 28/10 | budget register : O[16]+S[8]+fenêtre K déjà = 128 regs ; toute forme cheap (shuffle/smem/asm) coûte regs, smem (mur levier D) ou le hint cache (KPF) ; le prefetch « 0-reg » n'est pas gratuit | **NO-GO 28/10** — reste la réécriture structurelle à <128 regs (projet, ROI incertain) |
| **mtp-accept** | 24/10 | pas de bug : sampling × contenu | **CLOS** (levier optionnel : temp du draft) |
| mapped_rows | 24/10 | 4,3 % (MTP) / absent MTP-OFF | low ROI, fermé |

---

## 5. État prod final (vérifié 26/10 21:28 UTC)

- `.so` : **4fd152f9** (T3 + lba4 + cvec = load K q8_0 vectorisé 128-bit, 122-128 regs,
  0 spill ; **dual-arch sm_70+sm_61** pour le mmproj sur la GTX 1070) — déployé 26/10 21:21
  via `deploy_cvec.py` (levier C validé). md5 vérifié, service `active`, sanity gen OK.
- Binaire ollama : **6f6d33a9** (patché mmproj-1070, passthrough `OLLAMA_MMPROJ_DEVICE`),
  drop-in systemd `mmproj-1070.conf` : `OLLAMA_MMPROJ_DEVICE=cuda1`, `OLLAMA_MMPROJ_FIT_MARGIN_MIB=1024`.
- Env service : `GGML_CUDA_VOLTA_Q6_W4R4=1`, `GGML_CUDA_VOLTA_Q8_FATTN_TC=0` (H2 off),
  `LLAMA_ARG_FIT_TARGET=1024`, `OLLAMA_NOPRUNE=true`.
- Runner : `-c 167000 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3
  --context-shift --flash-attn auto --cache-type-k/v q8_0`.
- V100 : ~29 386 MiB. GTX 1070 : ~3,9 GB (mmproj).
- Rollback en chaîne : 4fd152f9 → b014d026 (20261026-212004) → feaf05aa (fa05aa-20261025-174903)
  → fa3e75c2 → d2df07b8 → a4b6f440 → e4afbb4a (backups
  `/var/backups/llama-experiments/`). Rollback immédiat C :
  `sudo cp /var/backups/llama-experiments/libggml-cuda.so.prod-b014d026-20261026-212004
  /home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so &&
  sudo systemctl restart ollama-v100`.
- Instrumentation : CUPTI v2 (`h2/cupti129-test/`, `cupath_analyze_v2.py`), superviseur +
  watchdog en place, NCU 2024.1.1, nvidia-persistenced actif.

## 6. Open items (hors périmètre « levier kernel », qui est CLOS)

1. **Température du draft MTP** (chain fixée top_k(10) sans temp, `speculative.cpp:506-520`)
   — levier MTP, ROI faible estimé, en attente décision user (et MTP est suspendu depuis 24/10).
2. **Quantization plus agressive** (KV q4_0 libère ~2,5 GB ; modèle q4) — hors périmètre,
   risque qualité ; seul vrai levier « moins d'octets lus ». **Leviers kernel T3 : A réfuté,
   B et C clos/déployés (26/10), D NO-GO (27/10, confirmé binaire 28/10) — tour T3 v2 CLOS.**
   Partage fragments B + ILP/accumulateurs **NO-GO (28/10, grille B×ILP complète)** : le seul
   reste serait une réécriture structurelle du noyau à <128 regs (projet, ROI incertain).
   Voir `t3_grille_B_ILP_20261028.md`.
3. **Reprise MTP** — à la décision du user ; le tour MTP-OFF est clos, rien à refaire de ce côté.
4. **Publication** : le post Reddit (16/10) ne couvre que H2 (−11,8 % @107k) ; T3/lba et le
   batch 2048 post-date le post et le kit GitHub v1.0 — mise à jour possible si souhaité.
5. Surveiller la marge VRAM 60 MB (Grafana) ; si evictions → ctx 160k ou KV q4_0.

## 7. Leçons durables (pièges documentés sur la période)

- Jamais `cp` in-place du .so en service (segfault) ; toujours stop → swap → start.
- `GGML_BACKEND_DL` requis pour `ggml_backend_init` (sinon fallback CPU silencieux) ;
  vérifier `nm -D | grep ggml_backend_init` avant tout déploiement.
- Index GPU inversés (CUDA lib 0 = V100, nvidia-smi 1) ; CVD=0 pour la V100.
- CUPTI v2 : trace = fichier `<prefix>.<pid>` (+ `.names`/`.hdr`) ; `t_real_us` ≠ `gpu_s`
  (skew) ; G5 window FAIL by design si bracket = requête entière → phase9d / fenêtre glissante.
- `rc du pipeline ≠ validité de la mesure` (transparence du superviseur).
- SASS : extraire par **label mangled en colonne 0** (`_Z30flash…:`) + couper au label
  suivant — pas `.visible .entry` (absent dans la sortie `nvdisasm` utilisée). Un
  faux `cmp` sur `$i.txt.txt` a faussement conclu « AUCUN diff » le 28/10 ; toujours
  vérifier que les fichiers testés existent. Le cubin 207 = T3 D=256 **f16** ; le
  kernel prod q8_0 D=256 = **cubin 213** (prod == cvec, cf. `mb_fa_t3v2_verdict.md`).
  **Un `asm volatile("prefetch.global.L2")` sur le pointeur K fait tomber le hint
  `.CONSTANT` des `LDG.E.128` vectorisés** (`LDG.E.128.CONSTANT.SYS` → `LDG.E.128.SYS`)
  sur toute la boucle : le prefetch « 0 registre » n'est pas gratuit en cache
  (mesuré KPF v1 −1,5 % / v2 −14,2 % @T=1, cf. `t3_grille_B_ILP_20261028.md`).
- Pruning ollama : digests `sha256-` (tiret) vs `sha256:` → perte du blob params (incident
  17-18/10, défendu par `OLLAMA_NOPRUNE=true`).
- Gold standard interleave : leg C (candidat désactivé par env) ≈ A (prod) = condition de
  validité de TOUTE mesure A/B de .so — tenue sur chaque run de la période.
