# Config run n_max=2 — prof-prd (à coller dans le 2e chat)

## Contexte machine (invariable)

- Pas un environnement de production. 1 V100 (bench, service `ollama-v100`, port 11434)
  + 1 GTX 1070 (embed/reranker, 33695/11436). Toute activité tok/s observée = nos benchs.
- **Documents autoritaires** : `bench/supervisor_bench_README.md` +
  `bench/GPU_UUID_NSYS_CUPTI_20261030.md` (GPU par UUID — V100 = CUDA ordinal 0 mais
  `nvidia-smi --id=1` ; nsys exige `NSYS_CUPTI_LIBRARY_PATH=bench/nsys-cupti129` avec
  `libcupti.so.12.9` seule, md5 `9bc91feae571aa505069c0640e85c26c`, et
  `--cuda-graph-trace=node` ; `sudo -E` ignoré → `sudo env K=V…` ; `.rep` root-owned →
  `mv` + `chown guile`).
- Commandes de diagnostic directes ; sorties volumineuses : filtrer/limiter/dumper dans un
  fichier puis inspecter.

## Point de départ (run de référence, NE PAS ÉCRASER)

- Run : `runs/prof-prd-20261030-114948/` — **référence figée, garder intacte.**
- Résultats (ctx 112060 tok, b2048, KV q8_0, H2 OFF, model qwen3.8-27b-q6-mtp-vision) :

| config | phase | prompt tok/s | gén. tok/s | ms/tok |
|---|---|---:|---:|---:|
| MTP OFF (n_max=—) | prefill | 617.6 | 21.26 (16 tok) | 47.0 |
| MTP OFF | decode 96 tok | — | **21.62** | 46.3 |
| MTP ON **n_max=3** | prefill | 619.7 | 3.90 (16 tok, 1er cycle froid) | 256.3 |
| MTP ON **n_max=3** | decode 160 tok | — | **44.18** (×2.05) | 22.6 |

- Acceptance decode n_max=3 : **0.711** (108 acceptés / 152 générés, mean len 3.12),
  per-pos **0.902 / 0.706 / 0.510**.
- Décomposition steady-state (246 graph launches, 48 verifies ≈ 160 tok) :
  **wall/launch 15.5 ms** = busy GPU 12.4 ms + host gap 1.4 ms ;
  **wall/verify 79.6 ms** (≈5.1 launches/verify : 1 verify + ~3 drafts + 1 sampling)
  ; 3.33 tok/verify → 41.9 t/s (mesuré 44.18).
  verify 56.3 ms busy (matvec 39.5 ms = 70 %, flash_attn 9.1 ms = 16 %) ; draft 2.3 ms.
- Rapport : `bench/prof_prd_analysis_20261030-114948.md`
  (régénéré par `bench/analyze_prd_profile.py` qui suit **le run le plus récent**).

## Hypothèse à tester

La 3ᵉ position draft n'a que **51 %** d'acceptance. Si on passe `n_max=2` :
- le verify batch passe de 4→3 positions (−1 matvec/attention par position ≈ −25 % du
  temps de verify) et on ne lance plus le 3ᵉ draft (−2.3 ms/cycle) ;
- l'espérance tombe d'environ 3.12 → ~2.54 tok/verify (perte ≈ −0.32 ≈ −11 %).

## Le changement (UNIQUE, à faire avant le run)

`bench/prof_prd_profile_20261030.py`, fonction `launch_server()` (line ~468) :

```python
    spec_args = (["--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
                  "--spec-draft-backend-sampling"] if spec_on else [])
```

i.e. `"3"` → `"2"`. Rien d'autre : même model, même ctx, même b2048/q8_0, même
fenêtres (decode 160 tok MTP ON / 96 tok OFF), même nsys flags.

## Lancement

```bash
cd /home/guile/llama-experiments/bench
python3 supervisor_bench.py run prof_prd_profile_20261030.py --detach
```

- Le script génère **automatiquement un nouveau run** `prof-prd-<TS>` à la date du
  lancement (`TS = time.strftime(...)` line 78) → **aucun risque d'écraser 114948**
  (artefacts `prof_prd_*_<TS>.{nsys-rep,sqlite}` + `runs/prof-prd-<TS>/`).
- `--detach` obligatoire si lancé depuis un agent (le stop du service coupe le tir en
  cours ; le superviseur survit). Log : `bench/supervisor_bench.log`.
- Durée attendue ~17 min (4 fenêtres : off pf/dc, on pf/dc).

## Après le run

1. Vérifier : `systemctl is-active ollama-v100` = active ; pas de résidu compute app sur
   la V100 (sinon purger UNIQUEMENT les résidus bench sur la V100 résolue par UUID) ;
   `nsys status` propre ; lock `/tmp/kolbo-supervisor-bench.lock` relâché.
2. Analyser : `python3 bench/analyze_prd_profile.py` (prend auto le run le plus récent)
   → `bench/prof_prd_analysis_<TS>.md`.
3. **Restaurer ensuite la config** : remettre `--spec-draft-n-max 3` dans le script
   (sauf décision de garder 2), pour que le script reparte au run de référence.

## RÉSULTAT (run 20261030-164453, lancé 16:44:53, rc=0, ~17 min)

| métrique | n_max=3 (réf 114948) | **n_max=2 (164453)** | seuil |
|---|---:|---:|---:|
| decode mesuré | 44.18 t/s | **42.26 t/s** | > 44.2 pour gagner |
| ms/tok | 22.6 | 23.7 | — |
| wall/launch (steady) | 15.5 ms | **16.9 ms** | < 12.6 |
| wall/verify | 79.6 ms | **68.4 ms** | < 64.5 |
| busy verify/cycle | 56.3 ms | 53.1 ms | ~42–45 |
| tok/verify (espérance) | 3.12 | **2.76** | ~2.54 attendu |
| acceptance decode | 0.711 (108/152) | **0.825 (99/120)** | — |
| acc per-pos | 0.902 / 0.706 / 0.510 | 0.883 / 0.767 | — |
| launches/verify | ~5.1 | ~4.0 | — |

Lecture :
- **n_max=2 PERD sur ce workload** : 42.26 < 44.18 t/s (−4.4 %).
- Pourquoi : le gain de wall/verify (79.6 → 68.4 ms, −11.2 ms) est réel mais
  insuffisant face à la perte d'espérance (3.12 → 2.76 tok/verify) ; le ratio
  wall/launch **augmente** (15.5 → 16.9 ms) au lieu de passer sous 12.6 ms.
  La busy verify ne descend que 56.3 → 53.1 ms : à ctx 112k le verify reste
dominé par le stream des poids (matvec 37.0 ms/verify), pas par le nombre de
positions (4→3) — le matvec relu est le même, seul l'attention par position
baisse (9.1 → 9.0 ms, quasi nul).
- L'acceptance mesurée (0.825, per-pos 0.883/0.767) est **supérieure** à celle
du n_max=3 aux 2 premières positions (0.902/0.706) : la 2ᵉ position reste
bonne, c'est bien la 3ᵉ (51 %) qui coûtait cher — mais le coût était déjà
compensé par le token accepté.
- Le `draft → sampling : n=1, 56.7 ms` dans le rapport est un artefact de
transition signalé comme tel par l'analyseur, non un coût récurrent.

**Décision : garder `--spec-draft-n-max 3`** (config de référence). Script
restauré à `3` le 30/10 17:07. `n_max=2` n'est pas exclu sur d'autres
workloads (acceptance pos2 > 0.9, ctx plus court où le verify serait plus
instruction-bound), mais rien ne justifie de le tester ici.

Rapport complet : `bench/prof_prd_analysis_20261030-164453.md` ; run intact
`runs/prof-prd-20261030-164453/` ; référence 114948 inchangée
(results.json md5 ee292568a180496658df574e3151cdd4).

## CRITÈRE DE DÉCISION — wall/cycle à surveiller

Dans le rapport du nouveau run, section « MTP ON — décomposition par cycle » :

| métrique | référence n_max=3 | seuil n_max=2 pour GAGNER |
|---|---:|---:|
| **wall/launch** (steady-state) | **15.5 ms** | **< 12.6 ms** (−19 %) |
| wall/verify | 79.6 ms | < 64.5 ms |
| busy verify/cycle | 56.3 ms | < ~42–45 ms (batch 3 pos) |
| tok/verify (espérance) | 3.12 | ~2.54 attendu |
| **decode tok/s mesuré** | **44.18** | **> 44.2 t/s** (critère ultime) |

Dérivation du seuil : t/s = E / wall_cycle ; n_max=2 gagne si et seulement si
wall_cycle_2 / wall_cycle_3 < E2/E3 = 2.54/3.12 = **0.814** → wall/cycle < 15.5 × 0.814
≈ **12.6 ms**. (Si l'acceptance mesurée n_max=2 diffère, recalculer :
le plus simple : **comparer les tok/s mesurés** directement ; la décomposition sert à comprendre pourquoi.)

Lecture des résultats :
- tok/s > 44.2 → n_max=2 gagne sur CE workload (et le taux d'acceptance est
  workload-dépendant : à re-valider sur d'autres prompts avant de généraliser).
- tok/s < 44.2 → n_max=3 reste meilleur ici ; le rapport dira où le temps est parti
  (wall/verify vs gaps host) pour décider si un n_max=2 ne serait rentable que sur des
  workloads plus « faciles » (acceptance pos1/pos2 > 0.9).
