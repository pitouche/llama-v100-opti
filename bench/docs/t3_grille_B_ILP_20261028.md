# T3 — grille combinatoire B × ILP : analyse + sonde no-B (28/10)

Contexte : les deux leviers restants « partage fragments B » et « ILP/accumulateurs »
butent sur le même mur que le levier D : **le register file V100**. Cette note pose la
question à plat, produit le coin manquant de la grille (revert du levier B, sans
réécriture du kernel), et fixe le seuil à battre par un vrai variant ILP.

## 1. Le mur : register file 65 536 regs/SM = budget ZÉRO-SOMME entre occupancy et ILP

V100 (sm_70) : **65 536 registres/SM**, 128 threads/bloc. Le noyau T3 a un besoin
naturel de **162 regs/thread** (c'est ce que ptxas demande sans contrainte). Le levier B
(`launch_bounds(128,4)`) **force** ptxas à compresser à **128 regs** pour tenir 4 blocs/SM.

```
regs/thread   →   blocs/SM   →   threads/SM   →   warps/SM   →   occupancy
  ≤ 128        →     4        →    512        →    16        →   25 %   (PROD, levier B)
 129 – 170     →     3        →    384        →    12        →   18,75 % (no-B, naturel)
 171 – 226     →     2        →    256        →     8        →   12,5 %
```

C'est un mur dur : **plus de regs/thread = moins de threads résidents = moins de warps
pour masquer la latence**. ILP (plus d'ops mémoire indépendants en vol **par warp**)
consomme des regs ; occupancy (plus de warps **par SM**) en a besoin aussi. Les deux
se disputent le même 65 536. C'est exactement ce que NCU montre : le kernel est
**latence-bound** (60,6 % long-scoreboard, DRAM 53 %, SM 21 %) et plafonné par
l'occupancy, pas par la bande passante.

## 2. Réponse à « libérer des registres = hypothèse ou gain réel ? »

**Ni l'un ni l'autre, tel que formulé.** « Libérer des registres » n'est **pas un levier
de perf en soi** — c'est un **prérequis** (créer de la place). Deux faits :

1. **Il n'y a pas de registres « qui traînent ».** Les 128 regs du prod sont **tous vivants**
   (O[16], S[8], M/L[4], fenêtre K 5×uint4, fragments P, pointeurs/addresses, softmax) et
   ptxas les a **déjà minimisés** sous la contrainte B. On ne peut pas en « trouver » de
   morts sans réécriture.
2. **Recycler des regs = 0 gain seul.** Le gain n'apparaît **que** si la place librée est
   remplie par un ILP/prefetch qui masque réellement de la latence. Et réduire l'état
   vivant pour faire de la place, c'est souvent **moins** d'ILP par warp — l'inverse de
   l'objectif. D'où la vraie difficulté du chemin « réécriture pour tenir ILP à 4 blocs/SM
   (< 128 regs) » : c'est un projet, ROI **incertain**.

## 3. La grille 2×2 (B × ILP) — les 2 cellules « sans ILP » sont MESURÉES

Sonde **no-B** : revert du levier B seul (`minblocks NT=1 : 4 → 3`), **aucun** autre
changement (C intact, même smem/launcher/pb auto). Build `build_t3v2.sh` →
`libggml-cuda-nob.so` (md5 **e19b6112**), 150 objets, prod inchangée. ptxas NT=1 :
**162–166 regs, 0 spill** → 3 blocs/SM (cohérent avec le besoin naturel 162).

Microbench `mb_fa` T=1..8, n_kv=107 351, 200 iters, médiane, MTP-OFF
(`mb_fa_nob_bench.py`, REF = prod 4fd152f9 = B+C, NOB = e19b6112 = C sans B) :

| T | NT | prod (B+C) µs | no-B µs | no-B vs prod | note |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | **404** | **433** | **−7,1 %** | **cas decode prod** |
| 2 | 2 | 467 | 467 | +0,0 % | MTP verify |
| 3 | 2 | 620 | 622 | −0,2 % | MTP verify |
| 4 | 2 | 625 | 626 | −0,2 % | MTP verify |
| 5 | 2 | 701 | 701 | +0,0 % | MTP verify |
| 6 | 3 | 1109 | 1110 | −0,1 % | MTP verify |
| 7 | 3 | 1117 | 1117 | +0,0 % | MTP verify |
| 8 | 3 | 1097 | 1097 | +0,0 % | MTP verify |

- **Coût du revert de B = ~7 % @T=1** (404 → 433 µs). **Corroboré** par le mesure propre
  de `t3_lbB_verdict_20261024.md` (+6,8 % @T=1, mêmes recettes des deux côtés). Écart
  0,3 pp → le ~7 % est solide.
- **T=2..8 : ±0,2 %** — le levier B ne touche que NT=1, aucune interaction. Le no-B est
  un « revert de B » propre, pas une régression globale.
- Caveat build : NOB est buildé par la recette **h2lib** (`-use_fast_math`), pas la recette
  dual prod → attention **non bit-identique** (maxrel 5,03e-05, tol 1e-4). C'est un artefact
  de recette (ftz/fmad), **pas** un changement de calcul : le delta 7,1 % >> le drift 5e-5,
  et la mesure propre t3_lbB confirme.

### La grille

|  | **sans ILP** | **+ ILP / prefetch** |
|---|---|---|
| **avec B** (4 blk/SM, ≤128 regs) | prod = **404 µs** ✅ mesuré | ILP à < 128 regs → **réécriture dure, ROI incertain** |
| **sans B** (3 blk/SM, ≤170 regs) | no-B = **433 µs** ✅ mesuré (28/10) | **la cible cheap** : revert B + prefetch, doit battre 433 (et 404 pour gagner) |

## 4. Ce que ça implique pour les leviers ILP / partage B

- Le **partage de fragments B** et l'**ILP/prefetch** ont le même profil de regs : ils
  ajoutent de l'état → ptxas passe à **3 blocs/SM** (≤170) voire **2 blocs/SM** (>170).
  Ils partent donc **déjà à −7 %** (le prix du revert B) et doivent **récupérer > 7 %**
  d'ILP juste pour **egaler** le prod, et encore plus pour le battre.
- **Seuil de GO du prochain variant ILP (coin « sans B + prefetch », 3 blk/SM)** :
  battre **433 µs @T=1** (no-B) — et idéalement **404 µs** (prod) pour être un vrai gain.
  Si un variant ILP à 3 blk/SM ne bat même pas 433, l'ILP « cheap » est **mort** et il ne
  reste que la réécriture dure à 4 blk/SM (projet, ROI incertain) → NO-GO sauf volonté
  explicite de l'ouvrir.
- **Plafond d'optimisme** : le kernel est déjà à 53 % du pic DRAM et la latence masquée
  par 12 warps/SM (no-B). L'ILP ne peut récupérer qu'une partie du 60,6 % long-scoreboard ;
  le reste est structurel (géométrie 1 bloc = 1 (qhead,kvhead), cf. § NCU).

## 5. Résultat du variant ILP cheap : KPF — prefetch.global.L2 du K (0 registre, 0 smem)

Découverte clé de la nuit : **`prefetch.global.L2`** compile en **`CCTL.E.PF2`** sur sm_70
(vérifié SASS) — une instruction **sans destination registre** : 0 reg, 0 smem, bit-exact
(cache only), pas d'impact sur le launch_bounds. C'est la **seule** forme de prefetch
« ILP-like » qui ne casse **pas** le levier B. Deux variantes testées (3 `CCTL.E.PF2` par
lane par itération, 3 lignes de 128 B couvrant la rangée K de 272 B) :

- **KPF v1** (`.cuh` bdade32b, `.so` **7a87cf78**) : prefetch des rangées de l'itération
  **courante** (offset `k0 + 32*warp + lane`) — c.-à-d. les mêmes lignes que les LDG
  immédiats → **aucun lead time**. Kernel prod exact : REG:128 STACK:0, 0 spill — mais
  **17 `LDG.E.128` sans hint `.CONSTANT`** (cf. § cause exacte ci-dessous).
- **KPF v2** (`.cuh` 15923afa, `.so` **1a8ba52e**) : prefetch des rangées de l'itération
  **suivante** (offset `k0 + gridDim.y*STEP + 32*warp + lane`) — lead time = 1 itération
  complète. Même profil ress : REG:128 STACK:0, mêmes 17 LDG sans hint `.CONSTANT`.

Microbench `mb_fa` T=1..8, n_kv=107 351, 200 iters, médiane, MTP-OFF
(`mb_fa_kpf_bench.py` / `mb_fa_kpf2_bench.py`) ; validation **bit-identique**
(maxrel = 0.00e+00 des deux côtés, attendu — cache only) :

| T | NT | prod µs | KPF v1 µs | v1 vs prod | KPF v2 µs | v2 vs prod |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | **404** | 411 | **−1,5 %** | 462 | **−14,2 %** |
| 2 | 2 | 468 | 472 | −0,9 % | 514 | −9,8 % |
| 3 | 2 | 622 | 614 | +1,2 % | 653 | −5,1 % |
| 4 | 2 | 626 | 618 | +1,2 % | 660 | −5,4 % |
| 5 | 2 | 702 | 695 | +1,0 % | 737 | −5,0 % |
| 6 | 3 | 1111 | 1111 | +0,0 % | 1143 | −2,9 % |
| 7 | 3 | 1119 | 1118 | +0,1 % | 1151 | −3,0 % |
| 8 | 3 | 1097 | 1102 | −0,5 % | 1132 | −3,2 % |

**Verdict : NO-GO des deux variantes, et la cause est prouvée par SASS.**

**Cause exacte (confrontation SASS du kernel prod exact D=256 NT=1 q8_0)** :

| build | LDG.E.128.<b>CONSTANT</b>.SYS | LDG.E.128.SYS (sans) | CCTL.E.PF2 | @T=1 vs prod |
|---|---:|---:|---:|---|
| prod 4fd152f9 | **17** | 0 | 0 | baseline 404 µs |
| nob (h2lib, **sans asm**) | **17** | 0 | 0 | −7,1 % (revert B seul) |
| kpf v1 (asm prefetch) | 0 | **17** | 3 | −1,5 % |
| kpf v2 (asm prefetch) | 0 | **17** | 3 | −14,2 % |

La **seule** différence entre `nob` (sain) et `kpf*` est le bloc
`asm volatile("prefetch.global.L2 [%0];" :: "l"(kpf))`. Son effet collatéral :
**il fait tomber le hint `.CONSTANT` sur les 17 `LDG.E.128` du levier C**
(`LDG.E.128.CONSTANT.SYS` → `LDG.E.128.SYS`), et ce pour **toutes** les itérations —
le prefetch `volatile` sur le pointeur K casse à ptxas la preuve que la base K est en
lecture seule / traitable comme constante. **C'est le vrai coût caché du « 0-registre »
prefetch** : il est gratuit en regs mais coûte le chemin cache du levier C.

- **v1 = −1,5 % @T=1** : la perte de hint sur les LDG K courants est partiellement
  masquée (les lignes sont déjà chaudes), mais le prefetch des lignes déjà en vol n'ajoute
  rien ; on ne paie que la perte de hint + l'émission des 3 CCTL/lane.
- **v2 = −14,2 % @T=1** : même perte de hint **provenée**, **en plus** une pression L2
  supplémentaire très vraisemblable (non isolée séparément) : le prefetch pointe sur les
  lignes de l'itération suivante, working set K+V **55,7 MiB** vs L2 **6 MiB** (×9,3), zéro
  réutilisation entre itérations → les CCTL évacuent vraisemblablement les lignes chaudes
  (K/V courante, Q). Pire à T grand (plus de CTAs résidents, plus de prefetch concurrents) :
  −3 à −10 %.
- **Corollaire général** : un prefetch L2 explicite « gratuit en registres » n'est **pas**
  gratuit : il dégrade le scheduling de cache des LDG existants (perte du hint `.CONSTANT`).
  Et la latence long-scoreboard (60,6 %) est la dépendance **intra-warp** LDG→déquant→MMA,
  pas un défaut d'anticipation DRAM — à 53 % du pic DRAM avec 12–16 warps/SM, les LDG en
  cours suffisent à alimenter le pipeline.
- Le coin « sans B + prefetch » de la grille est **mort par inférence** (v2, même profil
  REG:128, est négatif avec B ; ajouter le revert de B ajouterait −7 %) : pas de build
  « nobk » nécessaire.

### Grille finale (les 4 cellules sont résolues)

|  | **sans ILP** | **+ ILP / prefetch K (0 reg)** |
|---|---|---|
| **avec B** (4 blk/SM, ≤128 regs) | prod = **404 µs** ✅ | KPF v1 = **411 µs (−1,5 %)** ❌ · v2 = **462 µs (−14,2 %)** ❌ |
| **sans B** (3 blk/SM, ≤170 regs) | no-B = **433 µs** ✅ (−7,1 %) | mort par inférence (v2 + revert B < v2) ❌ |

## 6. Conséquences pour les deux leviers restants

1. **ILP/accumulateurs (forme cheap)** : **NO-GO définitif**. Le seul canal 0-registe
   (prefetch L2) est négatif dans ses deux formes. La forme « double-buffer de rangées V
   en registres » (vraie ILP) ajoute de l'état vivant → >128 regs → perte du levier B
   (−7 %) + le même thrashing/pression que v2 si le buffer dépasse la fenêtre déjà lue.
   **Il ne reste que la réécriture structurelle à < 128 regs** (réduction d'état + ILP
   simultanés) : projet à ROI incertain, hors fenêtre « testable vite ».
2. **Partage de fragments B** : même mur — toute redistribution des fragments B par
   shuffle/smem coûte des regs (shuffle) ou du smem (et le smem est déjà le mur du
   levier D). À budget 128 regs, il n'y a pas de place pour un buffer de partage en plus
   des O[16] + S[8] + fenêtre K. **NO-GO** sans une réécriture qui **libère** d'abord de
   l'état (cf. §2 : pas de regs morts identifiables).
3. Le gain réel des deux leviers exigerait donc la même chose : une **réécriture du
   noyau** qui baisse l'état registre sous 128 pour y loger un ILP réel. C'est un projet
   (semaines), pas une sonde — à garder comme piste T3 v2 §9.5 de `OPTI.md`, **NO-GO
   comme « levier vite »**.

## 7. Invariants post-benchs (28/10 ~10:00 UTC)
service `ollama-v100` **active** · prod `4fd152f9` **intact** · dual repli `d1e2d5de` intact ·
V100 util **0 %** · rc superviseur **0** (« rien à reprendre ») · aucun process résiduel.

Artefacts KPF : `bench/mb_fa_kpf_bench.py`, `bench/mb_fa_kpf_results.jsonl`,
`bench/mb_fa_kpf_summary.md`, `bench/mb_fa_kpf2_bench.py`, `bench/mb_fa_kpf2_results.jsonl`,
`bench/mb_fa_kpf2_summary.md` ; `.cuh` = `h2/cupti129-test/e2/t3v2/fattn-mma-volta-small.kpf.cuh`
(bdade32b) et `.kpf2.cuh` (15923afa) ; `.so` = `h2/cupti129-test/e2/h2lib/lib/`
`libggml-cuda-kpf.so` (7a87cf78) et `libggml-cuda-kpf2.so` (1a8ba52e) ;
SASS kernel prod exact KPF v2 = `/tmp/prodk_kpf2.sass` (a7efa75a, 7604 lignes vs 3841
baseline — l'écart = le code d'adresse du prefetch, hors des boucles chaudes).

Artefacts no-B (ci-dessus §3-4) : `bench/mb_fa_nob_*` ; `.so` = `libggml-cuda-nob.so`
(e19b6112) ; `.cuh` = `fattn-mma-volta-small.nob.cuh` (e21b528d,
1 seule ligne diff vs prod : `minblocks NT<=1 ? 3 : 2`).
