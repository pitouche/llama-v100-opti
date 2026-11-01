# T3 v2 — rewrite « carte blanche » **vldg** (ILP phase 2 : double-buffer V) — DÉPLOYÉ EN PROD (28/10 15:27 UTC)

Contexte : après la clôture des leviers « rapides » du tour T3 v2 (levier D NO-GO,
grille B × ILP résolue, KPF v1/v2 NO-GO, partage B NO-GO — cf.
`t3_grille_B_ILP_20261028.md`), la seule piste restante était la **réécriture
structurelle** du noyau T3 pour créer un ILP réel **sous 128 regs** (4 CTA/SM,
levier B intact) et **sans casser le levier C** (les 17 `LDG.E.128.CONSTANT.SYS` du
K). Décision user : « carte blanche » sur le noyau T3.

**Verdict : GO fort.** Le rewrite **vldg** masque la latence des loads V (stall
dominant long-scoreboard 60,6 % NCU) par double-buffer logiciel des fenêtres V,
**bit-identique**, **REG:128 STACK:0**, levier C intact :

| T | NT | prod µs | vldg µs | vldg vs prod |
|---:|---:|---:|---:|---:|
| 1 | 1 | **406** | **359** | **+11,4 %** |
| 2 | 2 | 468 | 399 | +14,7 % |
| 3 | 2 | 620 | 508 | +18,1 % |
| 4 | 2 | 626 | 516 | +17,5 % |
| 5 | 2 | 702 | 595 | +15,3 % |
| 6 | 3 | 1110 | 891 | +19,7 % |
| 7 | 3 | 1118 | 898 | +19,7 % |
| 8 | 3 | 1096 | 881 | +19,6 % |

Seuil GO (≥ +2 % @T=1, pas de régression T=2..8) : **largement tenu**.

## 1. Pourquoi c'était bloquant, et pourquoi ça passe maintenant

- NCU @ n_kv ≈ 107k, T=1 : le kernel est **latence-bound** (long-scoreboard
  60,6 %, DRAM 53 % du pic, SM 21 %), pas bouché en bande passante.
- La forme « ILP » qui coûte des regs (double-buffer de rangées V) était interdite
  par le mur 65 536 regs/SM : >128 regs → 3 CTA/SM → −7 % (prix du revert B).
- Le prefetch « 0 registre » (`CCTL.E.PF2`) était mort : il casse le hint
  `.CONSTANT` des LDG K (provoqué par l'asm volatile).
- **L'angle vldg** : ne pas *ajouter* d'état, mais **déplacer** l'existant. La
  baseline charge les 4 mots de la fenêtre V (rangées pv et pv+16) **puis** déquant,
  puis mma — la latence du LDG est consommée **avant** les mmas qui en ont besoin.
  vldg charge la fenêtre du chunk `ch+1` **pendant** les mmas du chunk `ch` :
  les 4 mots de `ch+1` sont **en vol** pendant 4+ mmas. Mêmes octets, même déquant,
  même mma → **bit-exact** ; l'asm prefetch est absent → **levier C intact**.

## 2. Design (borne, bit-exact)

Source : `h2/cupti129-test/e2/t3v2/fattn-mma-volta-small.vldg.cuh`
(dérivée de `.cvec.cuh` dbd4987c, source baseline prod).

1. **`fattn_vs_vwin`** : 4 mots `uint32_t` (w0..w3) = les mots bruts 32-bit alignés
   de la fenêtre V du chunk (scale q8_0 + quants). `fattn_vs_v_win_load` les charge
   (mêmes `Vd[0]` / `Vq[0..2]` que la baseline, même garde anti-dépassement de fin
   de ligne) ; `fattn_vs_v_dequant` applique **l'arithmétique de déquant identique**
   (`__funnelshift_r`, `fattn_vs_dq8`, `bytes_lo/hi`).
2. **Phase 2 pipelinée** : la fenêtre V du **chunk 0** est chargée **avant** la boucle
   (en vol pendant l'épilogue phase 1 du step 0) ; à chaque itération `ch` :
   déquant de `wA`/`wB` (chunk `ch`), puis **chargement de `wA`/`wB` du chunk
   `ch+1`** (si présent) **avant** les 4 mmas du chunk `ch`.
3. Aucun `asm volatile`, aucun `prefetch`, aucun buffer smem, aucun registre ajouté
   au-delà de ce que ptxas allouait déjà (le double-buffer *remplace* la fenêtre V
   déjà lue, il ne s'ajoute pas à l'état vivant).

## 3. Validation build (kernel prod exact D=256 NT=1 q8_0 softcap=0 skip=0)

Build `build_t3v2.sh vldg` → `h2/cupti129-test/e2/h2lib/lib/libggml-cuda-vldg.so`
(md5 **87f85442**), 150 objets, prod inchangée (4fd152f9).

| | ptxas | SASS (cuobjdump, extraction par symbole) |
|---|---|---|
| **prod** (sm_70, du .so déployé) | 128 regs | 3792 instr, 17 LDG.E.128.CONSTANT, 24 LDG.E.CONSTANT, 512 HMMA |
| **cvec rebuild** (723c0016) | 128 regs | **stream de mnémoniques = prod, diff 0** (baseline reconstruite fidèle) |
| **vldg** (87f85442) | **128 regs, STACK 0, 0 spill** | **3792 instr, 17 LDG.E.128.CONSTANT, 24 LDG.E.CONSTANT, 512 HMMA** — **identique au prod** |

- **Levler B intact** : REG ≤ 128 → 4 CTA/SM (autour : NT1 b1b0 128, NT1 b1b1 127).
- **Levler C intact** : les 17 `LDG.E.128.CONSTANT.SYS` du K sont **toujours là**
  (pas d'asm prefetch, cf. leçon KPF).
- **Le diff SASS prod↔vldg est minuscule** : 48 instructions sur 3792 (stream de
  mnémoniques), tous des **déplacements locaux** (IMAD/SHF/IADD3/HMMA/LDG
  réordonnés dans la boucle V) — **aucune instruction ajoutée ni retirée**, même
  multiset (diff d'histogramme = 2 IMAD vs IMAD.SHL et un R10/R11, i.e. du
  ré-encoding). Le gain vient du **réordonnancement logiciel**, pas d'une
  instruction nouvelle.
- `prod vs cvec rebuild` = **diff 0** → la baseline non-prod reconstruite est
  fidèle au kernel prod exact : la référence du microbench est valide.

## 4. Microbench (2 runs superviseur `--detach`, n_kv=107351, 200 iters, médiane, MTP-OFF, pb auto=60 @T=1)

Harnais : `bench/mb_fa_vldg_bench.py` (2 voies REF/VLDG) puis
`bench/mb_fa_vldg2_bench.py` (**3 voies REF/CVEC/VLDG** = validation de
reproductibilité + contrôle d'artefact de build).

**Run 1** (`mb_fa_vldg_summary.md`) : bit-identique `maxrel=0.00e+00` ;
VLDG vs prod : T=1 +11,4 % … T=6..8 +19,6 %.

**Run 2 — 3 voies** (`mb_fa_vldg2_summary.md`) :
- **VLDG bit-identique** à la prod (`maxrel=0.00e+00`, `bit-identique=True`) ;
- **CVEC vs REF = +0,0…+0,8 %** (rebuild de la même source ≈ prod) → **pas
  d'artefact de build** dans la référence ;
- **VLDG vs REF** : T=1 **+11,1 %**, T=2..5 +14,5..18,3 %, T=6..8 +19,6..19,8 % ;
- **VLDG vs CVEC** : T=1 +10,4 %, jusqu'à +19,7 % → le gain est bien **vldg**,
  pas l'environnement.

Gain **croissant avec T** (plus de mmas par warp = plus de surface pour masquer
les loads V en vol). La répétabilité run1→run2 est excellente (±0,3 pp @T=1).

## 5. A/B e2e MTP-OFF (gold standard, 28/10 13:02–13:33 UTC)

Harnais : `bench/t3_vldg_bench.py` (dérivé de `night_bench.py`, interleave 2 reps,
serveur frais par leg, port 8917, CVD=0, `-b 2048`, KV q8_0, FA auto, N_GEN 64,
warmup 1024). 2 legs : A-prod `4fd152f9` vs B-vldg `87f85442`.

| ctx | A-prod decode | B-vldg decode | **Δ decode** | A-prod prefill | B-vldg prefill | Δ prefill |
|---:|---:|---:|---:|---:|---:|---:|
| 114 688 | 22.68 t/s | 23.19 t/s | **+2.25 %** | 620.3 | 620.3 | +0.00 % |
| 196 608 | 20.65 t/s | 21.23 t/s | **+2.81 %** | 343.9 | 343.5 | −0.12 % |

VRAM pic identique (30 988 MiB). Gain e2e MTP-OFF **confirmé**, cohérent (plus faible
que le microbench isolé, comme attendu — le kernel attention n'est pas 100 % du decode).
Service restauré, rc=0.

## 6. A/B e2e MTP-ON (config prod, 28/10 13:48–14:12 UTC)

Harnais : `bench/t3_vldg_mtp_bench.py` (dérivé de `night_bench.py` patché MTP ON :
`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling`,
`MAX_CTX=167000`, env prod complet : H2 OFF, shortlist 131072, DEFER_PROMPT=1,
Q6_W4R4=1, FIT_TARGET=1024). Points : 114 688 et 160 000. Mesure : decode t/s +
acceptance MTP.

| ctx | A-prod decode | B-vldg decode | **Δ decode** | acc | mean len | acc rate/pos |
|---:|---:|---:|---:|---:|---:|---|
| 114 688 | 15.95 t/s | 16.23 t/s | **+1.76 %** | 0.92 | 3.71 | (1.000, 0.929, 0.821) |
| 160 000 | 24.49 t/s | 24.82 t/s | **+0.40 %** | 0.741 | 3.15 | (0.938, 0.854, 0.708) |

**Acceptance MTP identique A vs B** (même acc rate/pos au millième près, mêmes
mean len) → **bit-exact confirmé en MTP ON** : le vldg est correct sur le chemin
f16 du draft (ctx_dft) ET sur le chemin q8_0 du target. VRAM pic identique
(32 438 MiB). Gain e2e MTP-ON **positif et constant en signe** (dilué par l'overhead
du draft model, comme attendu — le kernel FA est un composant du verify, pas le
seul). Service restauré, rc=0.

Note : les t/s absolus MTP-ON (15.9 / 24.5) sont nettement inférieurs aux bench
MTP-ON du 18/10 (45.7 @114 k / 36.3 @160 k), **et l'ordre des ctx est inversé**
(114 k = 15.95 < 160 k = 24.49, alors que le KV plus long devrait être plus lent).
Le prod **lui-même** (leg A, `.so` non modifié) est à ce niveau : ce n'est **pas** un
effet vldg. De plus MTP-ON (15.95) est aujourd'hui **plus lent que MTP-OFF** (22.68)
aux 114 k — alors que le 18/10 c'était +135 %. → **anomalie préexistante du régime
MTP-ON, à investiguer séparément** (affecte la prod, hors périmètre vldg) : suspect
du draft path (acceptation identique 0.92/3.71, donc pas le draft model ; coût par
spec-step ≈ 1.7× plus élevé à 114 k qu'à 160 k, inexplicable par le KV seul).
Le **delta A vs B** reste la mesure exploitable : interleave, même binaire sauf le
kernel T3, acceptance identique → le gain vldg est propre même dans ce régime dégradé.

## 7. Ce que ça change / ne change PAS

- **Change** : l'interleaving SASS de la boucle V (loads du chunk `ch+1` émis
  pendant les mmas du chunk `ch`) → la latence LDG V est masquée.
- **Ne change pas** : la géométrie (1 bloc = 1 (qhead,kvhead)), la partition, le
  smem, le launcher/pb auto, l'arithmétique (bit-exact), ni les leviers B et C.
- **N'ajoute pas** : d'instruction, de registre, de smem, d'asm.

## 8. Rebuild dual-arch + déploiement prod (28/10 15:00–15:28 UTC, après GO user « yes GO! »)

**Rebuild** — la prod `4fd152f9` était un build dual-arch (sm_70+sm_61) du 26/10 sur la
source cvec. Même recette : le `.cuh` vldg (md5 **f2781c5f**) a été posé sur la source
dual (`v100/dual-arch/src/ggml/src/ggml-cuda/fattn-mma-volta-small.cuh`, backup pré-vldg
`/tmp/fattn-mma-volta-small.dual-pre-vldg.cuh` = dbd4987c) — les autres fichiers T3 et les
6 template-instances étaient déjà identiques entre les deux trees. `build_dual.sh` →
**7 objets rebuildés** (dont les 2 fattn), `build ggml-cuda OK` →
`v100/dual-arch/lib/libggml-cuda-dual.so` md5 **8035caf4**, 300 ELF (150 sm_70 + 150 sm_61).

**Validation SASS du dual** (cuobjdump 12.8 — les 13.1 de la box refusent SM70) :
- kernel T3 exact (D=256 NT=1 q8_0 softcap=0 skip=0) dans le dual : **3792 instr, 17
  LDG.E.128.CONSTANT, 24 LDG.E.CONSTANT, 512 HMMA** — identique aux 4 références ;
- **la section sm_70 du kernel est byte-identique au h2lib vldg 87f85442** (le `.so` qui a
  passé microbench + e2e MTP-OFF/ON) : md5 des streams **bf53c498** à l'identique
  (`/tmp/dualvldg_sm70_only.sass` vs `/tmp/vldg_body.sass`, diff rc=0) ;
- la section sm_61 est présente (cubins 214) pour la GTX 1070 repli.
→ le `.so` déployé porte **exactement** le kernel validé, plus l'image sm_61.

**Déploiement supervisé** (`bench/deploy_vldg.py`, README superviseur, run
`15:26:10→15:27:45 UTC`, 95 s, rc=0) :
- préconditions OK : prod 4fd152f9, cand 8035caf4, ELF 150/150, binaire ollama 6f6d33a9 ;
- stop service → V100 libérée 0 MiB en 0 s → **backup prod**
  `/var/backups/llama-experiments/libggml-cuda.so.prod-4fd152f9-20261028-152612` (md5
  4fd152f9 vérifié) → **swap** → nouveau md5 prod **8035caf4** → service active →
  sanity gen OK → verify (md5, service, VRAM 28 844 MiB post-chargement, drop-in
  mmproj-1070 présent) → `ok=True`.
- Rollback : `sudo cp /var/backups/llama-experiments/libggml-cuda.so.prod-4fd152f9-20261028-152612 <prod path> && sudo systemctl restart ollama-v100`.

**Post-deploy** (~15:43 UTC) : réponse réelle cohérente (27×43=1161, Rayleigh), 0 erreur
CUDA/crash dans le journal, service active, prod **8035caf4**, V100 0 % / 3 947 MiB,
runner MTP ON opérationnel (`-c 128000 --spec-type draft-mtp …`), superviseur rc=0.

## 9. Garde-fous post-déploiement (vérifié 28/10 ~15:43 UTC)

- service `ollama-v100` = **active** ;
- prod `.so` md5 = **8035caf4** (vldg dual-arch) — **déployé** ;
- backup pré-déploiement = **4fd152f9** (cvec dual-arch) dans `/var/backups/…` (md5 vérifié) ;
- dual repli **d1e2d5de** (ancienne baseline cvec rebuild du tree dual, « repli ») :
  **écrasé par le rebuild vldg de 15:00** (build_dual.sh fait `cp` sans backup de
  l'ancien `.so` — aucun git sur le tree). **Pas de perte opérationnelle** : c'était
  un rebuild de la même source cvec `dbd4987c`, jamais en prod — l'état prod exact
  d'avant (le dual `4fd152f9` lui-même) est conservé en backup md5-vérifié et reste
  le point de rollback. Sources cvec `dbd4987c` et `.cuh` backup
  `/tmp/fattn-mma-volta-small.dual-pre-vldg.cuh` → reconstruisible à tout moment si
  besoin. ⚠️ À corriger dans `build_dual.sh` : faire un backup horodaté de
  `libggml-cuda-dual.so` avant `cp`.
- V100 util = **0 %**, VRAM 3 947 MiB (idle, runner déchargé après keep=4) ;
- superviseur `last-rc` = **0** ;
- 0 process résiduel bench/build (seuls les runners ollama habituels).

## 10. Artefacts

- Source : `h2/cupti129-test/e2/t3v2/fattn-mma-volta-small.vldg.cuh`
- `.so` : `h2/cupti129-test/e2/h2lib/lib/libggml-cuda-vldg.so` (**87f85442**)
  (non-prod, **pas** déployé)
- Baseline rebuild : `libggml-cuda-cvec.so` (**723c0016**)
- Build log : `/tmp/build_vldg.log` ; ptxas : `/tmp/t3v2-objs-vldg/`
- **Rebuild dual-arch** : `v100/dual-arch/lib/libggml-cuda-dual.so` (**8035caf4**, = prod
  déployée), log `v100/dual-arch/logs/build-dual.log` ; backup du `.cuh` dual pré-vldg :
  `/tmp/fattn-mma-volta-small.dual-pre-vldg.cuh` (dbd4987c)
- SASS : `/tmp/vldg_d256.sass`, extraits par fonction `/tmp/{cvec,vldg,prod}_nt1.sass`,
  streams `/tmp/{cvec,vldg,prod}_mnem.txt` ; dual : `/tmp/dualvldg.sass` (cuobjdump 12.8),
  kernel `/tmp/dualvldg_nt1.sass`, section sm_70 `/tmp/dualvldg_sm70_only.sass` (= `/tmp/vldg_body.sass`,
  md5 bf53c498), diffs `/tmp/diff_dual_vs_{h2lib,prod}.txt`
- Bench micro : `bench/mb_fa_vldg_{bench.py,results.jsonl,summary.md}`,
  `bench/mb_fa_vldg2_{bench.py,results.jsonl,summary.md}`
- Bench e2e MTP-OFF : `bench/t3_vldg_bench_{bench.py,results.jsonl,summary.md,server.log,gputimes.tsv}`
- Bench e2e MTP-ON : `bench/t3_vldg_mtp_bench_{bench.py,results.jsonl,summary.md,server.log,gputimes.tsv}`
- Déploiement : `bench/deploy_vldg.py`, log `bench/supervisor_bench.log`, backup prod
  `/var/backups/llama-experiments/libggml-cuda.so.prod-4fd152f9-20261028-152612`

## 11. État & prochaine étape (décision user)

Le noyau est **GO en microbench, e2e MTP-OFF et e2e MTP-ON** (signe constant, aucun
point négatif, bit-exact partout) et est **DEPLOYÉ EN PROD** le 28/10 15:27 UTC (dual-arch
`8035caf4`, sm_70 byte-identique au kernel validé).

Reste :
1. ~~A/B e2e MTP-OFF~~ ✅ (+2.25 % @114 k, +2.81 % @196 k)
2. ~~A/B e2e MTP-ON~~ ✅ (+1.76 % @114 k, +0.40 % @160 k, bit-exact, acc identique)
3. ~~rebuild prod compatible~~ ✅ (dual-arch sm_70+sm_61, 8035caf4)
4. ~~déploiement supervisé~~ ✅ (15:27 UTC, backup 4fd152f9 en /var/backups)
5. **re-vérification post-deploy des régimes prod** (2048/65k/98k + MTP ON config prod)
   — à faire dans la prochaine fenêtre bench : confirmer que la prod déployée reproduce
   les gains mesurés (A/B déjà faites avec le même kernel sm_70, donc risque faible, mais
   un contrôle en situation réelle reste prudent) ;
6. **anomalie MTP-ON préexistante** (cf. §6) : investiguer séparément, hors périmètre vldg.
