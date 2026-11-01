# BILAN ENV VOLTA (Q5_X4 / Q8_FATTN_TC) — run INTERLEAVE (16:05, MTP off)

Méthode : le run INTERLEAVE est le seul qui fait foi — A/B/C testés cote à cote,
serveur frais par leg, DANS la même fenêtre thermique à chaque contexte.
(les runs séquentiels précédants étaient pollués par la derive thermique du run).

## Résultat
| contexte   | A-base | B +Q5_X4 | ΔQ5   | C +Q8_FATTN | ΔQ8   |
|-----------:|-------:|---------:|------:|------------:|------:|
| cold 16k   | 23.15  | 24.66    | +6.5% | 24.67       | +0.0% |
| warm 32k   | 23.08  | 23.05    | -0.1% | 23.08       | +0.1% |
| warm 65k   | 14.95  | 13.04    | -12.8%| 12.54       | -3.8% |
| warm 98k   | 11.48  | 11.55    | +0.6% | 11.28       | -2.3% |
| **warm 114k (PROD)** | 11.82 | 11.86 | **+0.3%** | 11.65   | -1.8% |
| warm 131k  | 10.66  | 11.04    | +3.6% | 11.23       | +1.7% |
| warm 163k  | 10.59  | 10.18    | -3.9% | 10.83       | +6.4% |
| warm 192k  | 10.19  | 10.10    | -0.9% | 10.06       | -0.4% |

## VERDICT
- **Q5_X4 : NEUTRE au regime prod (114k) = +0.3%, sous le bruit.**
  * le +6.5% a 16k cold = effet de premier prompt (cold), pas un gain durable
  * le -12.8% a 65k = un point isolé, a verifier mais hors regime prod
  * partout ailleurs : bruit (±3-4%)
- **Q8_FATTN_TC : NEUTRE** (tous les ΔQ8 dans ±2%, sauf outliers a 163k).

## CONCLUSION de la seance
Les leviers ENV testables sans rebuild (Q5_X4, Q8_FATTN_TC) **ne donnent rien au
regime de fonctionnement reel (~114k)**. Le patch 42d9581 non plus (MTP off).

La prod tourne deja proche du plafond de ce .so dans le regime reel. Les marges
reelles restantes (si on veut vraiment gagner) sont :
  1. le **prefill** long (195k = 398 t/s, degrade x2 par rapport au court) — piste
     identifiee dans les notes, "gains prefill jusqu'a 39.9% a contextes specifiques"
  2. un **rebuild** sur mesure (pas un env) pour le regime long contexte — plus couteux
  3. le **Q5_K** a 47% du pic : l'env Q5_X4 ne l'atteint pas, il faudrait verifier si
     le kernel Q5_X4 est reellement dispatche (ou si c'est un autre kernel qui domine)

## Methode (a garder)
- **INTERLEAVE** par contexte (A/B/C cote a cote, serveur frais par leg) = gold standard
- **MTP OFF** pour le decode pur (le cycle draft/verify bruitait)
- **jamais de verrouillage d'horloge** (firmware protege, interdit par guile)
- protocole stop/start systeme + superviseur, jamais d'unload par API

## Etat (16:15)
ollama-v100 active · model 24 GB GPU (100% GPU) · .so prod a4b6f440 INTACT · 0 test en vie
