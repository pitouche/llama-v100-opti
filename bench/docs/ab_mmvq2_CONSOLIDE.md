# A/B 42d9581 (nwarps draft K-quant 4→2) — SYNTHÈSE 2 RUNS (17/10)

Run1 = 11:02 (3 pts) · Run2 = 12:18 (7 pts, 16k→192k, paliers 32k)
Méthode : protocle SÛR stop/start (supervisor_bench.py --detach). 0 split CPU, 0 touch au .so prod.

## decode t/s (A=prod a4b6f440, B=42d9581)
| contexte   | A r1 | B r1 | Δ r1  | A r2 | B r2 | Δ r2  |
|-----------:|-----:|-----:|------:|-----:|-----:|------:|
| cold 16k   | 47.98| 48.42| +0.9% | 53.22| 53.07| -0.3% |
| warm 32k   | 45.65| 47.18| +3.4% | 54.50| 54.28| -0.4% |
| warm 65k   | 34.24| 37.97|+10.9% | 42.54| 38.96| -8.4% |
| warm 98k   |  —   |  —   |  —    | 29.69| 30.24| +1.9% |
| warm 131k  |  —   |  —   |  —    | 30.05| 30.07| +0.1% |
| warm 163k  |  —   |  —   |  —    | 26.36| 26.82| +1.7% |
| warm 192k  |  —   |  —   |  —    | 25.94| 26.59| +2.5% |

## Variance du baseline A (MÊME .so prod, entre r1 et r2)
| contexte   | A r1 | A r2 | écart |
|-----------:|-----:|-----:|------:|
| cold 16k   | 47.98| 53.22| +10.9% |
| warm 32k   | 45.65| 54.50| +19.4% |
| warm 65k   | 34.24| 42.54| +24.2% |

## VERDICT (honnête)
- Le patch **42d9581 est ~NEUTRE** : aucun gain decode fiable et reproductible.
- Le point +10.9% (r1, 65k) **n'est PAS reproduit** (r2 → -8.4%). L'écart A/B y est du bruit, pas du patch.
- La **variance du baseline (10–24% entre deux runs) est supérieure à la différence A/B**.
  C'est l'état horloge/thermique de la V100 au démarrage de chaque run qui domine la mesure.
- Seul signal faible et constant : B légèrement devant à LONG contexte (98k–192k, +1.7 à +2.5%),
  mais c'est sous le bruit de baseline → pas exploitable en l'état.
- **Qualité draft identique (acc/len) sur tous les points** → le patch est sûr, juste pas gagnant.

## Décision
Ne PAS déployer 42d9581 comme "gain". C'est un nul (pas une régression, pas un gain).

## Pour trancher un effet < 5% (si on veut vraiment mesurer)
La mesure A-puis-B en deux sessions est trop bruitée. Il faut l'une de ces trois choses :
1. **A/B/A/B alterné DANS la même session serveur** (même chauffe, mêmes horloges) — le gold standard ;
2. **verrouiller les horloges V100** (`nvidia-smi -lgc`) sur toute la durée des deux legs ;
3. **3 répétitions par config, médiane**, horloges verrouillées.

## État machine (vérifié 12:25 UTC)
- ollama-v100 : active · modèle 24 GB sur GPU (100% GPU, pas de split) · ctx 200000
- .so prod : a4b6f440 INTACT · 0 process de test · lock superviseur libéré
