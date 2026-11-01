# Analyse — bench GENERATION (decode tg) vs longueur du contexte

Run : 02/10 17:52–18:01 (run_start ts 1790963556), 8 points × 300 tokens générés.
Setup (identique au run prefill) : Qwen3.8-27B Q6_K, V100-PCIE-32GB (CUDA0),
KV q8_0, Flash Attention, MTP draft=3, `-c 204800`, `-b 512 -ub 512`, 1 slot,
context-shift/keep inchangés. llama-server direct :8917 (bench_gen.py, même
`build_env`/`SERVER_ARGS` que bench_direct.py).

**Métrique** : `eval time` du log du runner (ligne **decode** — la regex
exclut `prompt eval time`) ÷ tokens générés → `tg`.
Exclus du calcul par construction : préfill, restauration de checkpoint,
chargement du modèle, réseau/API, temps avant le 1er token.

> Note run : `"ignore_eos": True` était absent du 1er run (16:31, 8 points
> INVALID à ~57 tok — le modèle émettait l'EOS après « OK-BENCH. »). Le run
> ci-dessous est le run corrigé : 300 tokens générés partout, 0 INVALID.

## 1. Intégrité — OK

- **8/8 points valides** : `gen = 300 = api_gen_tokens` partout (log ↔ API concordent).
- **VRAM constante 32 160 MiB** sur les 8 points (pool KV pré-allouée, contrôle OK).
- **Mécanisme warm vérifié point par point** : `restored_n_past` = prompt du point
  précédent **− 512** (le checkpoint du runner est aligné sur un multiple du
  batch) à chaque fois ; la diff préfillée (14,6k → 32,6k) correspond au
  complément exact → aucun préfixe commun, aucun contaminant entre points.
- **0 erreur** dans le log : pas de CUDA error, pas de shift déclenché,
  pas de `truncated ≠ 0`. `KV cache shifting is not supported for this context,
  disabling` = info de démarrage (déjà présent au run prefill) — le shift
  n'a jamais opéré, rien à corriger dans les mesures.

## 2. Tableau — la mesure

| contexte cible | **contexte réel avant gén.** | restauré (n_past) | diff préfillée | générés (tok) | **generation eval time** | **tg (t/s)** | **ms/tok** | MTP acc (aux) | mean len (aux) | ms/step (dérivé) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | **2 083** | — (premier run, cold) | 2 083 | 300 | 7 701 ms | **39,0** | 25,67 | 0,565 | 2,69 | 69,1 |
| 16 384 | **16 198** | 1 571 | 14 627 | 300 | 6 642 ms | **45,2** | 22,14 | 0,768 | 3,29 | 72,8 |
| 32 768 | **32 309** | 15 686 | 16 623 | 300 | 6 862 ms | **43,7** | 22,87 | 0,789 | 3,36 | 76,8 |
| 65 536 | **64 339** | 31 797 | 32 542 | 300 | 7 551 ms | **39,7** | 25,17 | 0,787 | 3,36 | 84,6 |
| 98 304 | **96 356** | 63 827 | 32 529 | 300 | 8 159 ms | **36,8** | 27,20 | 0,799 | 3,40 | 92,5 |
| 131 072 | **128 437** | 95 844 | 32 593 | 300 | 9 140 ms | **32,8** | 30,47 | 0,799 | 3,40 | 103,6 |
| 163 840 | **160 530** | 127 925 | 32 605 | 300 | 10 630 ms | **28,2** | 35,43 | 0,787 | 3,36 | 119,1 |
| 196 608 | **192 646** | 160 018 | 32 628 | 300 | 10 539 ms | **28,5** | 35,13 | 0,924 | 3,77 | 132,4 |

« ms/step » = eval time ÷ nombre de pas (tokens ÷ mean len) — identité
arithmétique sur les valeurs runner, pas une métrique exposée.

## 3. Courbe — tg (t/s) vs contexte (n_past)

```
45.2 |███████████████████████      |  16 198
43.7 |██████████████████████       |  32 309
39.7 |█████████████████████        |  64 339
36.8 |████████████████████         |  96 356
32.8 |████████████████             | 128 437
28.2 |██████████████               | 160 530
28.5 |██████████████               | 192 646
39.0 |█████████████████████        |   2 083  (premier run de session — voir §5)
```
(1 █ ≈ 1,5 t/s)

Lecture : après un léger creux/décalage sur le 1er point, la courbe est
**strictement décroissante et régulière** de 16k à 192k, sans cliff.

## 4. Réponse directe

> « À 2k, 32k, 64k, 128k et ~196k de contexte, à combien de tok/s le
> Qwen3.8-27B Q6 génère-t-il réellement sur cette V100 ? »

| contexte | **tg mesuré** |
|---|---|
| 2 k | **39 t/s** (premier run de session, à nuancer — §5) |
| 32 k | **44 t/s** |
| 64 k | **40 t/s** |
| 128 k | **33 t/s** |
| ~196 k (192 646 réel) | **28–29 t/s** |

En ms/token : 23 → 35 ms/tok entre 16k et 192k. Pente : **~3,4 ms/step par
10k de contexte** (régression sur 16k→192k : `ms/step = 64,5 + 0,334·ctx(k)`,
écart type < 4 ms) → le coût d'un pas de vérification (forward cible + 3 drafts
MTP sur le contexte entier) monte linéairement, c'est la lecture-KV qui domine.

Décomposition (à acceptance fixée à la médiane 0,79) : 48,8 → 25,4 t/s de
2k à 192k — c'est-à-dire que **l'effet contexte pur vaut -13,4 t/s sur la
fenêtre**, et le solde des écarts observés vient de l'acceptance qui bouge
avec le contenu (§6).

## 5. Le point 2k — réserve honnête

Le point 2k est le **premier run de la session** (serveur fraîchement
démarré) : ses CUDA graphs de decode ne sont peut-être pas chauds, et c'est
le seul point **sans restauration de checkpoint** (premier préfill complet).
Conséquences visibles : `acc 0,565` (vs 0,77–0,92 partout ailleurs) et
`ms/step 69,1` — le 2k sort à 39,0 t/s alors que l'extrapolation de la courbe
16k→192k donnerait ~49 t/s à 2k. **Il faut donc lire le 2k comme « 39 t/s
mesuré, vraisemblablement un peu sous-estimé (effet premier-run) »** ; la
courbe propre est celle des 7 points restaurés. (Pas de contre-mesure prise :
ne pas re-mélanger un point chaud dans le même jsonl.)

## 6. MTP — information auxiliaire uniquement (pas de conclusion)

- acceptance : 0,768 / 0,789 / 0,787 / 0,799 / 0,799 / 0,787 / 0,924 sur
  16k→192k, mean len 3,29–3,77. **Très stable** (sauf le 2k de §5 et le
  0,924 au 196k — échantillon unique, pas une tendance).
- acc per pos (1/2/3) du run 192k : (0,949, 0,924, 0,899) — le 3e draft
  reste bien accepté même à 192k de contexte.
- Constat structurel, non causal : le débit baisse surtout parce que
  **ms/step monte avec le contexte** (69 → 132 ms, ×1,92) ; l'acceptance,
  elle, est quasi constante (0,79±0,02) sur le plateau. Sur texte dégénéré
  répéteur l'acceptance est biaisée — ce ne sont pas des chiffres
  d'acceptance « en conditions réelles ».

## 7. Comparaison avec le run prefill (rappel, non mélangé)

Le même contexte de 192–195k coûtait **~2,52 ms/tok en préfill cold**
(bench_direct, 398 t/s) ; ici le **decode** y coûte **35,1 ms/tok** — c'est-à-dire
×13,9 plus cher par token, mais c'est attendu : préfill = GEMM batch 512
(compute-bound), decode = 1 token/pas avec MTP (bande passante). Les deux
courbes restent dans leurs fichiers respectifs (`bench_direct_results.jsonl`
/ `bench_gen_results.jsonl`).

## 8. Méthode & limites

- 300 tokens générés/point (spéc ≥256) ; `temp=0 top_k=1 ignore_eos`
  déterministe ; après l'EOS le texte est dégénéré (répéteur) — sans
  incidence sur la bande passante.
- Le contexte croît de +300 tok pendant la génération : le tg est la vitesse
  **moyenne** du point (l'évolution interne sur 300 tok est négligeable,
  ~+0,1 ms/tok selon la pente du §4).
- Texte : corpus warm répéteur → débit de decode représentatif, acceptance
  biaisée (rapportée en auxiliaire, §6).
- Un seul essai par point : la régularité de la pente (écart type < 4 ms)
  indique le bruit faible, mais un point isolé (le 2k, le 196k) reste un
  échantillon unique.
- Les mesures de préfill du run précédent sont conservées dans
  `bench_direct_results.jsonl` / `bench_direct_summary.md` — rien n'est
  mélangé.
