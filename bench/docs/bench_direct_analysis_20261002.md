# Analyse — bench prefill V100 (02/10, run 09:48, ratios corrigés)

Setup : Qwen3.8-27B Q6_K (Unsloth) + MTP draft=3 + vision (mmproj) — V100-PCIE-32GB —
llama-server direct (port 8917) **et** Ollama (port 11434, même build de runner,
même GGUF, mêmes args : `-c 200k/204800, KV q8_0, flash-attn, MTP, -b 512 -ub 512,
1 slot, context-shift, keep 4`). Mesures issues du log du runner (`prompt eval
time` = tokens réellement pré-évalués), pas de l'API.

Sources brutes : `bench_direct_results.jsonl` · `bench_direct_server.log` ·
`bench_direct_summary.md` (tableaux complets).

---

## 1. Intégrité du run — validée

| contrôle | résultat |
|---|---|
| Points COLD avec `restore > 200` (préfill non complet) | **0** / 14 — tous `restore = 42` (residu du checkpoint minimal, pas de préfixe commun) |
| VRAM V100 (contrôle de fuite) | **31,4 GiB constante** sur les 27 runs mesurés (pool KV pré-allouée au contexte max) |
| Cible C vs tokens réellement préfillés | écart **-0,2 % à -1,5 %** (ratio R_COLD=1.45 corrigé) |
| Erreurs serveur (`no kernel image`, `CUDA error`, `ABORT`) | **0** — le server est bien resté sur la V100 (`CUDA0 (V100-PCIE-32GB)` dans le log) |

→ Tous les chiffres ci-dessous sont propres.

---

## 2. COURBE A — COLD : coût d'un préfill complet (llama.cpp direct)

t/s = tokens réellement pré-évalués / temps du log (position 0, KV vide).

```
t/s  | débit (1 █ = 25 t/s)                                    | taille réelle
     |                                                        |
 813 |████████████████████████████████                        |    2,042 tok
 846 |██████████████████████████████████                      |    4,089 tok
 841 |██████████████████████████████████                      |    8,105 tok
 814 |█████████████████████████████████                       |   16,242 tok
 748 |██████████████████████████████                          |   32,466 tok
 639 |██████████████████████████                              |   64,566 tok
 554 |██████████████████                                      |   96,961 tok
 489 |████████████████████                                    |  129,600 tok
 439 |██████████████████                                      |  162,167 tok
 398 |████████████████                                        |  194,719 tok
```

| tok préfillés | temps | t/s | ms/tok |
|---|---|---|---|
| 2 042 | 2,51 s | 813 | 1,23 |
| 4 089 | 4,83 s | **846** | 1,18 |
| 8 105 | 9,64 s | 841 | 1,19 |
| 16 242 | 19,96 s | 814 | 1,23 |
| 32 466 | 43,39 s | 748 | 1,34 |
| 64 566 | 101,05 s | 639 | 1,56 |
| 96 961 | 175,09 s | 554 | 1,81 |
| 129 600 | 265,25 s | 489 | 2,05 |
| 162 167 | 369,26 s | 439 | 2,28 |
| 194 719 | 489,74 s | 398 | 2,52 |

Lecture (mesure) :
- Plafond **~845 t/s** sous ~8 k tokens, léger creux à 16 k.
- Dégradation régulière au-delà : **398 t/s à 195 k**, soit **-53 %** de débit
  vs le pic, et **×2,14** sur le ms/tok (1,18 → 2,52).

---

## 3. COURBE B — WARM : coût marginal de la diff (llama.cpp direct)

Chaîne de préfixes croissants 2 k → 195 k (corpus B) : le runner restaure son
checkpoint au préfixe commun et ne prefille que la **diff**. À partir de 32 k,
la diff vaut ~32,5 k tokens constamment → on isole le seul variable : la
**position absolue** de la diff dans le KV (n_past).

```
t/s  | débit (1 █ = 25 t/s)          | position de la diff dans le KV
     |                               |
 547 |██████████████████████         |  65,536
 438 |██████████████████             |  98,304
 366 |███████████████                | 131,072
 314 |█████████████                  | 163,840
 276 |███████████                    | 196,608
```

| position (C final) | diff réelle | temps | t/s | ms/tok |
|---|---|---|---|---|
| 65 536 | 32 542 | 59,45 s | 547 | 1,83 |
| 98 304 | 32 529 | 74,24 s | 438 | 2,28 |
| 131 072 | 32 593 | 89,07 s | 366 | 2,73 |
| 163 840 | 32 605 | 103,68 s | 314 | 3,18 |
| **196 608** | **32 628** | **118,42 s** | **276** | **3,63** |

---

## 4. CONSTAT POSITIONNEL (résultat central du run)

**Le débit de préfill est gouverné par la position absolue dans le KV
(n_past), pas par la taille du chunk.** La même diff de ~32,5 k tokens :

| | @ pos 65 k | @ pos 98 k | @ pos 131 k | @ pos 160 k | @ pos 195 k |
|---|---|---|---|---|---|
| t/s | 547 | 438 | 366 | 314 | 276 |
| ms/tok | 1,83 | 2,28 | 2,73 | 3,18 | 3,63 |
| temps pour 32,5 k | 59,4 s | 74,2 s | 89,1 s | 103,7 s | 118,4 s |

→ de 65 k à 195 k : **×2 sur le temps** pour exactement la même quantité de
tokens ajoutés (1,83 → 3,63 ms/tok, +98 %).

**Comparaison au cold, à taille comparable :**
- Préfill **cold** complet de 194 719 tok = 489,7 s → moyenne **2,52 ms/tok** (398 t/s).
- Diff **warm** des 32 628 derniers tokens (position 160 k → 195 k) =
  118,4 s → **3,63 ms/tok**.
→ Le token marginal d'un contexte conservé est **+44 % plus cher** par token
que le préfill froid de la même zone. L'état (KV de 160 k déjà posé) coûte
plus cher par token qu'un préfill à froid.

Conséquence mesurable pour la viabilité du 200k conservé : ajouter un chunk de
32 k quand le contexte est déjà à 195 k coûte **118 s** ; quand il est à 64 k,
59 s. Le « contexte chaud » n'est pas gratuit : sa vitesse se dégrade avec sa
position, indépendamment de la taille de ce qu'on ajoute.

*(C'est une observation sur la machine : la même courbe cold/warm partage la
même pente « position → ms/tok ». Le run n'a pas séparé les causes — attention
K/V de plus en plus longue à traverser, KV q8_0, flash-attn auto, W4R4 —
celui-ci ne le permet pas.)*

---

## 5. Ollama vs Direct

**Cold** (mêmes textes, mêmes tailles, position 0) — Ollama au-dessus, marge
resserrée avec la taille :

| tok préfillés | direct | ollama | Δ |
|---|---|---|---|
| 16 242 | 814 | 857 | +5,3 % |
| 64 566 | 639 | 661 | +3,5 % |
| 129 600 | 489 | 498 | +2,0 % |
| 194 719 | 398 | 404 | +1,6 % |

→ À froid, Ollama ne déforme pas la courbe (écart +2 à +5 %, minime).

**Réserve sur le comparatif warm** : les diff Ollama font 65,5 k (3 pas :
32 k→98 k→160 k) contre 32,7 k (10 pas) côté direct. Mêmes positions,
tailles de chunk différentes → les lignes warm 98 304 / 163 840 du §C du
summary **ne sont pas isolées** (chunk 2× plus gros, même position) et ne
doivent pas être comparées telles quelles. Le seul comparatif warm clean
manque ; le §4 ci-dessus est valable pour les deux (même build de runner).

---

## 6. MTP (draft)

| acceptance | mean len | runs |
|---|---|---|
| 1,000 | 3,0 | 20 / 27 (direct 16, ollama 4) |
| 0,500 | 2,0 | 7 / 27 (cold_196608, warm_16k/32k/196k, ocold_131k/196k, owarm_32k) |

Le texte de remplissage (mots courts aléatoires sur vocabulaire restreint)
est très prédictif → le draft 3/3 est accepté la plupart du temps. L'acceptance
0,5 sur les gros points est un signal faible (texte, pas workload réel) et n'a
pas d'effet visible sur le préfill. **Pas de conclusion MTP exploitable
depuis ce run** — le bench mesure le préfill, pas la génération.

---

## 7. Récap en une ligne

Préfill V100 (Q6_K 27B, KV q8_0, flash-attn, MTP) : **~845 t/s sous 8 k,
398 t/s à 195 k** ; le coût marginal d'un contexte conservé est proportionnel
à sa **position absolue** — le dernier 32 k d'un contexte à 195 k coûte
**118 s (3,63 ms/tok)**, soit +44 %/token vs le préfill froid de la même zone.
Ollama ≈ direct (cold : +2 à +5 %).

---

## 8. Méthode & limites

- **Tokens réels** = `prompt eval time` du log du runner (pas l'usage API,
  pas la cible C). `restore` vérifié ligne par ligne (cold ≈ 42, warm =
  C_prev).
- **VRAM** = contrôle (constante 31,4 GiB, pool pré-allouée), pas une mesure
  de croissance.
- **Cold** : 1 seed par taille (aucun préfixe commun entre points). **Warm** :
  1 seul corpus croissant (préfixe partagé).
- **Premier point Ollama** : inclut ~25 s de rechargement dans `wall_s`
  (pas dans les t/s du log) — les points ollama ci-dessus sont donc purs.
- **Pas de génération mesurée** (N_PRED = 4, l'objet est le préfill).
- **Pas de séparation des causes** de la dégradation (longueur KV, quant,
  flash-attn, W4R4) : le run donne la courbe, pas sa décomposition.
- Runs bruts archivés : `bench_direct_results.jsonl` (ce run),
  `bench_direct_results_20261002_ratios-erroneux.jsonl` (run du matin,
  ratio 1.03 erroné — à ne pas mélanger).
