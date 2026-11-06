# SWEEP Q6_K 170k — apport des optimisations S0→S6, 11 contextes (06/11/2026)

Campagne de **mesure uniquement** (pas de recherche d'opti) : l'apport de chaque
brique de la pile (S1…S6) à 11 tailles de contexte, en decode et prefill, MTP ON et
OFF. Confirme le signal du b2048/t3 (22/10) et livre les graphiques de la pile
complète.

- **Bruts** : `bench/data/sweep_q6_170k.jsonl` (158 records, 154 points valides)
- **Graphiques** : `bench/charts/*.png` + `bench/charts/sweep_q6_report.pdf`
- **Scripts** : `bench/bench/sweep_q6_170k.py` (campagne), `plot_q6_170k.py`
  (graphiques), `make_pdf.py` (rapport), `prompts_extended.py` (prompts)

## 1. Config figée

| | |
|---|---|
| Modèle | `qwen3.8-27b-q6-mtp` Q6_K (23 Go), blob prod `sha256-493301830a…` |
| mmproj | **AUCUN** (hors équation — anomalie vision gelée) |
| Serveur | binaire prod `/opt/PROD/lib/ollama/llama-server`, port 8917 |
| Contexte serveur | `-c 170000` (borne dure) — point haut = prompt 167 568 + 2048 gén |
| GPU | V100 seule (`--split-mode none --main-gpu 0`, CVD=0) |
| KV / FA | `q8_0/q8_0` / auto |
| Env commun | `Q6_W4R4=1`, `Q8_FATTN_TC=0` (H2 OFF), `FIT_TARGET=1024` |
| MTP ON | `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling` + `SPEC_MTP_DEFER_PROMPT=1` + shortlist |

Contextes : `16k 32k 48k 64k 80k 96k 112k 128k 144k 160k 163,6k` (11 points).

## 2. États (delta cumulé, S0 = référence « 0 opti »)

| État | .so | env / batch | Apporté |
|---|---|---|---|
| S0 | t3 `d2df07b8` | `FA_NO_VOLTA_SMALL=1`, b512 | base pre-T3 |
| S1 | t3 `d2df07b8` | `FA_NO_VOLTA_SMALL=1`, b2048 | +batch |
| S2 | t3 `d2df07b8` | —, b2048 | +T3 |
| S3 | dual `b014d026` | —, b2048 | +lba4 |
| S4 | cvec `723c0016` | —, b2048 | +cvec |
| S5 | vldg `87f85442` | —, b2048 | +vldg |
| S6 | dual `d4954502` (**PROD**) | `FA_COMPACT_TMAX=8192`, b2048 | +FA16 |

Chaque serveur de bench charge `GGML_BACKEND_PATH = sofiles/<state>.so` ; le `.so`
prod `/opt/PROD/…` n'est **jamais** touché.

## 3. Méthode de mesure (serveur frais par run)

- **R1** : 32 tok post-prefill froid · **R2** : 2048 tok (régime établi) ·
  **R3** : 32 tok chaud (diagnostique).
- **Régime établi** :
  - **OFF** = médiane des fenêtres `tg_3s` de R2.
  - **ON** = médiane des **6 dernières** fenêtres de R2 (« plateau ») — le
    warm-up MTP (les ~1 200 premiers tok générés sont lents, l'acceptance monte)
    est retiré. Voir §5, c'est lui qui creusait la 112k.
- **Gates** (rejet, pas de retry) : SM min ≥ 1 150 MHz, temp max < 93 °C,
  `prompt_tokens` (log serveur) dans ±0,5 % de la cible, n_gen R2 ≥ 2 000,
  ≥ 3 fenêtres `tg_3s` stables.
- **Anti-drift** : chaque run S6/32k/ON est un anchor ; Δ > 5 % vs 1er anchor →
  `drift_suspect`. Aucun drift sur les 154 runs (04:06 → 17:35).
- **Prompts** : canonique figé (`bench/prompts/canonical.json`, sha256
  `44b1737bb5…`), étendu ×3 pour les points > 107 340 (`prompts_extended.py`,
  stabilité re-tokenize vérifiée). Chaque run cite `prompt_sha256`.
- 4 re-mesures S6/off haut-contexte (faux positifs du gate SM) → validées :
  S6/off suit S5/off à −0,7…−1,0 % partout.

## 4. Résultats — decode MTP OFF (courbe de référence propre)

Gain par palier, t/s, 16k → 160k :

| palier | 16k | 32k | 64k | 96k | 112k | 128k | 144k | 160k |
|---|---|---|---|---|---|---|---|---|
| S0→S1 batch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **S1→S2 T3** | +4 | +6 | +13 | +18 | +20 | +22 | +24 | **+26** |
| S2→S3 lba4 | 0 | 0 | +2 | +4 | +4 | +6 | +6 | +7 |
| S3→S4 cvec | +1 | +2 | +3 | +4 | +4 | +4 | +5 | +5 |
| S4→S5 vldg | 0 | +1 | +1 | +1 | +2 | +2 | +2 | +2 |
| S5→S6 FA16 | −1 | −1 | −1 | −1 | −1 | −1 | −1 | −1 |
| **S0→S6 total** | +5 | +9 | +19 | +28 | +31 | +35 | +40 | **+43** |

- **Batch = levier purement prefill** (0 % en decode).
- **T3 = le cliff dominant en decode** : +4 % (16k) → +26 % (160k), s'ouvre avec le
  contexte. ≈ 2/3 du delta total S0→S6 à long contexte.
- lba4/cvec/vldg prolongent la montée ; **FA16 neutre en decode** (−1 %).
- **Synthèse : le gain s'ouvre avec le contexte — signal principal de la campagne.**

### Prefill MTP OFF
- batch S0→S1 : +9…+19 % (plus fort en bas d'axe) · FA16 S5→S6 : +4…+24 %.
- S2→S5 se superposent (T3/lba4/cvec/vldg n'y touchent pas).
- S0→S6 total prefill : +23 % (16k) → +35 % (160k). **Prefill = batch + FA16.**

## 5. Résultats — decode MTP ON (la « cicatrice » de 112k)

Multiplicateur ON(plateau)/OFF : **2,6–2,8× stable sur S2→S6**.

Le point chaud de la campagne : la courbe ON brute montrait un **creux à 112k**
(multiplicateur ×1,9, gain S0→S6 +18 % brut). **Ce creux est un artefact de la
manière dont on moyenne, pas une discontinuité de kernel.**

- **Cause** : le warm-up MTP. Les ~1 200 premiers tok générés sont lents (l'acceptance
  monte), et à 112k ils couvraient **11–14 des 15–26 fenêtres** de R2 → la médiane
  « toutes fenêtres » était tirée vers le bas.
- **En plateau** (6 dernières fenêtres) : le creux retombe — S6 42,9 → 52,5 t/s.
- **Indexé** (plateau × 3,4/(1+3A), ≈ t_step, acceptance égalisée à 0,8) : le 112k
  remonte, le gain S0→S6 passe de **+18 % brut à +44 % indexé**, et l'axe
  96k→144k devient lisse (+44…+56 %).

| ctx | gain S0→S6 ON brut | gain S0→S6 ON indexé | acc S0 → acc S6 |
|---|---|---|---|
| 96k | +33 % | +44 % | 0,81 → 0,79 |
| **112k** | **+18 %** (creux) | **+44 %** | 0,71 → **0,65** |
| 128k | +55 % | +56 % | 0,85 → 0,84 |
| 144k | +84 % (pic) | +43 % | 0,73 → 0,87 |

- **Vrai résidu à 112k** : l'acceptance y est **0,65** (vs 0,84 à 128k), **commun à
  tous les états** (S0…S6 à 0,647–0,65) → c'est un effet **contenu** du prompt, pas
  une opti. Petit, mais réel — la « cicatrice ».
- Le pic brut +84 % à 144k retombe à +43 % indexé : artefact du dénominateur S0.

### Méthode « plateau / indexé » (à retenir)
- **brut** = médiane toutes fenêtres R2 — inclut le warm-up MTP, à lire comme
  référence seulement.
- **plateau** = médiane des 6 dernières fenêtres — warm-up retiré, l'acceptance
  reste visible d'un ctx à l'autre.
- **indexé** = plateau × (1+3·0,8)/(1+3·A) — avec t/s = (1+3A)/t_step (validé à
  ±0,007 sur les 77 runs), ≈ **t_step**, vitesse du pas de vérification, indépendant
  de l'acceptance. **Courbe dérivée (égalisation), pas une mesure.**

Les 3 niveaux (brut / plateau / indexé) sont tous dans le PDF + le jsonl, avec les
chiffres : le lecteur choisit son niveau de confiance.

## 6. Réserves

- MTP ON sur **S0/S1** (7 points × 2 états) : plateau déjà bien aligné avec S2+,
  mais l'acceptance y reste instable pré-T3 (creux 0,58 à 32k) et les 112k n'ont
  que 20 fenêtres. À re-mesurer si on veut des chiffres ON fiables avant T3.
  Les courbes **OFF** restent la base solide.
- Point haut **163,6k*** : point unique sans répétition (limite dure `-c 170000`
  avec 2048 tok générés).
- Prefill mesuré **à froid** (1er passage) — sensible à la page cache ; les écarts
  inter-états y sont < 1 % hors batch/FA16.
- « plateau = 6 dernières fenêtres » est une heuristique pour sortir le warm-up ;
  le creux 112k retombe **déjà en plateau seul** (avant l'indexation), ce qui
  confirme que c'est le warm-up, pas la normalisation, qui fait le travail.

## 7. Re-validation

- **Re-lancer un point** : `python3 supervisor_bench.py run sweep_q6_170k.py S6 --detach`
  (idempotent — les runs valides sont sautés ; un seul state à la fois).
- **Recalculer les graphiques SANS re-lancer** (0 GPU) :
  `python3 bench/bench/plot_q6_170k.py && python3 bench/bench/make_pdf.py`.
- **L0 (cohérence des bruts)** : parse `bench/data/sweep_q6_170k.jsonl`,
  re-dérive plateau/indexé/gains et recoupe `bench/charts/`.
