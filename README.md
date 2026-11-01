# llama-v100-opti — pile d'optimisation V100 (Qwen3.8-27B + MTP, fork Ollama/llama.cpp)

**Ce que c'est.** L'ensemble des optimisations appliquées à une **Tesla V100-PCIE-32GB
(sm_70, Volta)** servant **Qwen3.8-27B Q6_K_M + MTP (draft-mtp n_max=3), KV q8_0,
flash attention, ctx 167k** via un fork Ollama/llama.cpp. Fenêtre d'activité :
**3 → 31 octobre 2026**.

**Pourquoi plusieurs repos.** Chaque optimisation est un repo autonome
(`repos/<opti>/`) : son README tient la question, le verdict et les chiffres ;
`src/` porte les fichiers modifiés ; `results/` les données brutes (JSONL/CSV/TXT)
pour recalculer le delta sans re-lancer un run ; `bench/` les scripts pour le
re-lancer. La pile complète se lit dans cet ordre, et chaque brique se lit à
l'unité.

**Intégrité.** Les `.so` de chaque état de prod sont dans `sofiles/` (à côté du
meta-repo, hors git par poids, **sha256-manifesté**), et les traces nsys dans
`big/` (idem). Chaque chiffre d'un repo pointe un `.so` par md5 → re-measurable.

**Multi-repos — comment naviguer (9 repos publics, `github.com/pitouche`).**
1. **Ce meta-repo** — big picture : état prod, table GO/NO-GO, méthode, section
   bench (re-validation), et un pointeur par opti.
2. **7 repos opti** (`opti-<piste>`) — un par piste, self-contained : README
   (question/verdict/chiffres), `src/` (fichiers modifiés), `results/` (bruts),
   `bench/` (scripts), § Re-validation (commande exacte + référence à battre).
3. **`v100-bench-kit` v2.0** — le kit de la fenêtre 3-16/10 (historique honnête
   `HISTORIQUE.md` + harnais génériques : superviseur, auto_bench, CUPTI). La
   mission H2 en a été extraite le 01/11/2026 vers `opti-h2` (l'opti n° 7).
Ordre de lecture suggéré : ce README → `OPTI.md` (le détail) → le repo opti
qu'on veut contester → `bench/VALIDATION.md` pour re-mesurer.

---

## 1. État prod FINAL (à retenir)

| élément | valeur |
|---|---|
| Service | `ollama-v100` — active, port 11434 |
| `.so` chargé | `libggml-cuda-dual.so` **md5 `d4954502`** (dual sm_70+sm_61, T3+vldg+lba4+cvec+FA16) |
| T3 (VOLTA_SMALL, rows 1-8) | **ON** (porte anyei #153 + lba4 + cvec + vldg) |
| H2 (VOLTA_Q8_W4, rows==4) | **OFF** (`GGML_CUDA_VOLTA_Q8_FATTN_TC=0`) — surpassé par T3 |
| T1 (mmsmt, mul_mat MTP) | **rejeté** (NO-GO) |
| FA16 compact (prefill) | **ON** (`GGML_CUDA_VOLTA_FA_COMPACT_TMAX=8192`) |
| MTP | ON — `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling` |
| Contexte / batch | `-c 167000`, `-b 2048 -ub 2048` (b2048) |
| KV cache | `q8_0` |
| mmproj vision | sur GTX 1070 (`OLLAMA_MMPROJ_DEVICE=cuda1`, .so dual-arch) |
| VRAM | ~32 438 MiB / 32 768 |

**Décomposition du gain decode** (vs prod initiale de la semaine, batch 512, H2 seul,
a4b6f440) :

| étape | gain |
|---|---|
| b512→b2048 (ctx 200k→167k payé) | +35 à +44 % decode long ctx |
| T3 (port anyei) | +13 % @114 k, +29 % @196 k (MTP-OFF) |
| H2 OFF | +15 à +17 % (MTP-ON, .so T3) |
| lba4 (launch_bounds) | +6,8 % microbench T=1, +1,7 % e2e @98 k |
| cvec (load K 128-bit) | +10,7 % microbench T=1, +3,5 % e2e @98 k |
| vldg (double-buffer V) | +11 % microbench T=1, +2,8 % e2e @196 k |
| **cumul** | **≈ ×1.7 decode long ctx MTP-OFF ; 58,9 t/s court ctx MTP-ON** |

## 2. Table des verdicts (GO / NO-GO) — les 7 optis

| # | Repo | Piste | Verdict | Date | Chiffre clé | En prod ? | Re-validation |
|---|---|---|---|---|---|---|---|
| 1 | `repos/t3-volta-small/` | Kernel FA decode rows 1-8 (anyei #153, m8n8k4) + lba4 + cvec + vldg | **GO** | 22-28/10 | +13/+29 % e2e MTP-OFF ; ×2.0-2.2 kernel FA | **OUI** | V1-V3 |
| 2 | `repos/fa16-compact/` | Kernel FA prefill compact (48 KB smem, 2 blocks/SM) | **GO** | 29/10 | prefill +19-21 % @98-131 k ; +12,3 % e2e @58 k | **OUI** | V5-V7 |
| 3 | `repos/t1-mmsmt/` | Hook mul_mat MTP-verify (anyei mmsmt) | **NO-GO** | 17-18/10 | neutre @114 k, −11,3 % @160 k | non | V8 |
| 4 | `repos/mmvq-clos/` | GEMV Q6_K W4R4 (H3 hoist, W2R4 nwarps, PA pacc fp32) | **CLOS** (×3) | 29/10 | ≤ +2,3 % < seuil 5 % | non (mur HBM2) | V9 |
| 5 | `repos/b2048/` | `-b 2048` au lieu de 512 (workspace CUDA) | **GO** | 22/10 | +35-44 % decode >98 k ; prefill +10-19 % | **OUI** | V10 |
| 6 | `repos/mtp-nmax/` | `--spec-draft-n-max` 2/3/4 | **GO — n3 gardé** | 30/10 | n4 : −11 à −23 % ; n2 : −5 % | n3 = prod | V11 |
| 7 | `repos/h2/` | H2 FA Q8 tensor-core rows==4 (tuile BC 16→8, « mission Kolbo » 8-16/10) | **GO → NO-GO** (suprématie T3) | GO 15/10 → NO-GO 18/10 | GO : +11-14 % @107 k ; NO-GO : −15-17 % avec H2 ON | non — `GGML_CUDA_VOLTA_Q8_FATTN_TC=0` | V12 (GO) / V4 (NO-GO) |

(V# = ligne du plan de re-measurement, `bench/VALIDATION.md` §2.)

**H2** est le cas d'école de la pile : une **optimisation déployée (GO) puis retirée
(NO-GO)** — T3 le surpasse même sur rows==4 (sa géométrie dédiée). La mission entière
(protocoles, CUPTI, A/B, divergence textuelle) vit dans le repo `opti-h2` (`repos/h2/`) ;
le conflit de dispatch et le verdict sont racontés ici (§5.4) et dans `t3-volta-small`.

## 3. Arborescence

```
llama-v100-opti/
├── README.md            ← vous êtes ici (big picture + état prod + verdicts)
├── OPTI.md              ← document de référence détaillé (chaque déploiement, gates, rollbacks)
├── .gitignore
├── bench/
│   ├── VALIDATION.md    ← LA SECTION BENCH : protocole + re-validation de chaque chiffre
│   ├── validate_all.sh  ← orchestrateur de re-measurement (superviseur + lock)
│   ├── bench/           ← harnais communs (superviseur, night_bench, microbenches, A/B) + données brutes
│   └── docs/            ← bilans, notes de protocole, règles GPU/CUPTI
├── repos/               ← un repo git par optimisation (self-contained)
│   ├── t3-volta-small/  fa16-compact/  t1-mmsmt/
│   ├── mmvq-clos/       b2048/         mtp-nmax/       h2/
├── sofiles/             ← .so de chaque état prod (sha256-manifesté, hors git)
└── big/                 ← traces nsys (.sqlite) de la décision n_max (hors git)
```

**Règle de lecture** : le `.so` d'un repo est toujours désigné par md5 et
localisable dans `sofiles/` (ou `sofiles/README.md` si absent) ; les données
brutes sont toujours celles du repo (`results/`) ; le protocole de re-measurement
est `bench/VALIDATION.md`.

## 4. Méthode (invariants de tous les benchs)

- **Superviseur obligatoire** : `python3 bench/bench/supervisor_bench.py run <bench.py> --detach`
  (README `bench/bench/supervisor_bench_README.md` = autorité). Le bench stoppe/
  restaure le service lui-même ; le superviseur est le filet (reprise OWUI).
- **GPU par UUID, jamais d'index en dur** (numérotations inversées sur cette
  machine, peuvent bouger au reboot) — `bench/bench/gpu_uuid.py`.
- **Interleave ≥ 2 reps, serveur frais par leg, port 8917, CVD=0** (V100),
  N_GEN 64, warmup 1024 ; **md5 du `.so` vérifié avant/après chaque run**.
- **MTP OFF** pour mesurer un kernel FA (gold standard) ; **MTP ON** pour
  config prod ; toujours vérifier que la **draft acceptance est identique entre
  legs** (sinon le delta est confondu avec le MTP).
- **NSYS_CUPTI_LIBRARY_PATH=bench/nsys-cupti129** (CUPTI 13.x inopérant sur
  Volta + driver 580 — documenté dans `bench/docs/GPU_UUID_NSYS_CUPTI_20261030.md`).

## 5. Section bench — re-valider les chiffres

**Tout chiffre publié est re-measurable.** Le protocole transversal est
`bench/VALIDATION.md` (l'autorité : invariants + plan par chiffre) ;
l'orchestrateur est `bench/validate_all.sh` (flock 1-bench-à-la-fois, sorties
dans `bench/validation/<date>-<tag>/`, jamais par-dessus les bruts publiés).

| Niveau | Ce qu'il valide | Contenu | Coût |
|---|---|---|---|
| **L0** | cohérence des bruts (pas de re-measurement) | parse `results/*.jsonl` des 7 repos | 0 min, sans GPU |
| **L1** | le delta **kernel** (µs/it) | V2 V3 V5 V7 V9 — microbenches dlopen du `.so` | ~1 h |
| **L2** | le delta **e2e t/s** (critère des GO) | V1 V4 V6 V8 V10 V12 — A/B via superviseur | ~4 h (nuit) |
| **L3** | décomposition kernel (diagnostic) | V11 — runs figés nsys CUPTI 12.9 | seulement si L2 > 3 % d'écart |

**Un chiffre est validé si L1 et L2 sont cohérents à ±3 %.** Chaque repo opti
porte sa section « Re-validation » (commande exacte + référence à battre = V#)
; les bruts pour recalculer SANS re-lancer sont dans `results/` de chaque repo.
Le cas H2 y figure en double : V12 (re-jouer le **GO** run9c3) et V4
(re-jouer le **NO-GO** H2 ON/OFF sur le `.so` T3) — le seul GO de la pile qui a
ensuite été retiré.

## 6. Machine de référence & prérequis

| | |
|---|---|
| GPU | Tesla V100-PCIE-32GB (sm_70, HBM2 ~794 GB/s mesurés) + GTX 1070 (mmproj, hors tests) |
| OS / driver | Linux, NVIDIA 580.178.04, CUDA 12.6 (nvcc 12.6.85), gcc 13.4, cmake 3.31.6 |
| Service | `ollama-v100` (Ollama pin 16b4376a + fork llama.cpp b11081-v100-full) |
| Modèle | `qwen3.8-27b-q6-mtp-vision` GGUF 23 088 409 504 o, sha256 `49330183…` + mmproj |
| Shortlist MTP | `qwen38-27b-exact-131072.i32` (131 072 i32, couverture 99,2 %) — à fournir |

Chemin machine-dépendant → à adapter dans `bench/bench/night_bench.py` (tête de
fichier) et les env du service ; le kit `v100-bench-kit` documente le même
principe (`config/kit.env.example`).

## 7. Licences

- Code du kit (harnais, scripts, docs) : **MIT** (voir `LICENSE` du repo `v100-bench-kit` v2.0).
- llama.cpp : MIT (upstream b11081 + fork Volta) — copyright conservé.
- Ollama : MIT. CUPTI : licence NVIDIA (binaire `.so` à fournir, non distribué).
- Modèle : licence du propriétaire du GGUF (non distribué).
- Aucun secret dans les repos : tokens lus de fichiers 0600 hors repo.

## Licence & provenance

- **Code original de cette publication** (scripts bench, docs, résultats, et les
  fichiers `src/` des 7 repos opti) : **MIT (c) 2026 guile** — voir `LICENSE`
  dans chaque repo.
- **Provenance des `src/`** : ces fichiers sont des **fichiers complets dérivés**
  du fork Ollama/llama.cpp, lui-même dérivé de **ggerganov/llama.cpp
  (MIT, « The ggml authors »)** — les en-têtes d'inclusion et l'organisation
  (`common.cuh`, `fattn*.cuh`, `ggml-cuda.cu`) suivent l'upstream. En publiant
  sous MIT (compatible avec le MIT upstream), les obligations de l'upstream
  restent satisfaites. Les fichiers marqués
  `// autogenerated by generate_cu_files.py` sont du code généré de l'upstream.
- **Modèles** : Qwen3.8-27B (Q6_K_M + MTP) — licences des poids non reprises ici
  (les poids ne sont pas dans ces repos, uniquement les `.so`/scripts qui les
  servent).
- **Données** : les `results/` ne contiennent que des métriques, logs de serveur
  et traces de profilage ; aucun prompt utilisateur ni donnée de tiers.

## État de la publication (publiée le 2026-11-01)

**Les 8 repos sont en ligne sur github.com/pitouche** (publics, `main`).
Chaque repo est un **snapshot à un seul commit** figé à la publication
(2026-11-01) — l'historique d'évolution n'est pas conservé à dessein :

| repo | contenu | verdict |
|---|---|---|
| [llama-v100-opti](https://github.com/pitouche/llama-v100-opti) | **ce repo** — pile complète + section bench | — |
| [opti-t3-volta-small](https://github.com/pitouche/opti-t3-volta-small) | T3 FA decode (lba4/cvec/vldg) | GO prod |
| [opti-fa16-compact](https://github.com/pitouche/opti-fa16-compact) | FA16 prefill | GO prod |
| [opti-t1-mmsmt](https://github.com/pitouche/opti-t1-mmsmt) | hook mul_mat MTP | NO-GO |
| [opti-mmvq-clos](https://github.com/pitouche/opti-mmvq-clos) | GEMV lecture poids | CLOS |
| [opti-b2048](https://github.com/pitouche/opti-b2048) | `-b/-ub 2048` | GO prod |
| [opti-mtp-nmax](https://github.com/pitouche/opti-mtp-nmax) | `--spec-draft-n-max` 2/3/4 — n3 gardé | GO (n3) |
| [opti-h2](https://github.com/pitouche/opti-h2) | H2 FA Q8 TC rows==4 (archive mission Kolbo : GO 15/10 → NO-GO 18/10) | NO-GO |
| [v100-bench-kit](https://github.com/pitouche/v100-bench-kit) | kit 3-16/10 : HISTORIQUE.md + harnais génériques (superviseur, auto_bench, CUPTI) | contexte / outils |

- `sofiles/` et `big/` ne sont **pas** dans le git (poids, 2.2+0.7 GB) : ils sont
  à côté du meta-repo local, sha256-manifestés ; les `.so` de chaque état prod
  sont re-traçables par md5 depuis chaque repo.
- Bundles de reconstitution (transport hors réseau) : `../bundles/<name>.bundle`
  (vérifiables par `git bundle verify`).
- **Tags finaux** (figent l'état publié du 01/11/2026) : `v1.0-20261101` sur ce
  repo et les 7 repos opti (snapshot 1 commit) ; `v2.0-20261101` sur le kit
  (v1.0-20261016 = état d'avant l'extraction H2, conservé).
