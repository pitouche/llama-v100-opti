# Sweep -b/-ub (512/1024/2048/4096) — synthèse + verdict (27/10)

Date : 2026-10-27. .so = prod 4fd152f9 (lecture seule, GGML_BACKEND_PATH). MTP OFF, -c 128000,
KV q8_0, FA auto, H2 OFF. Serveur frais par leg, 2 reps × 3 points (32k/65k/98k), 64 tok gen.
VRAM "prête" = nvidia-smi au moment où le serveur est up (inclut mmproj ~1.2 GB, constant →
les deltas sont valides). KV buffer constant = 4250 MiB (-c 128000, q8_0) dans TOUS les legs.

## Coût VRAM = le compute buffer (seule ligne qui bouge avec -b)
| -b/-ub | compute buffer | Δ vs 2048 | VRAM totale prête |
|---|---:|---:|---:|
| 512  |  705 MiB |  −615 MiB | 27 292 MiB |
| 1024 |  910 MiB |  −410 MiB | 27 498 MiB |
| 2048 | 1320 MiB |      —    | 27 908 MiB |
| 4096 | 2140 MiB |  +820 MiB | 28 730 MiB |

## PREFILL (t/s, médianes) — le batch aide, à rendement décroissant
| -b/-ub | 32k | 65k | 98k | Δ vs 2048 (98k) |
|---|---:|---:|---:|---:|
| 512  | 808.7 | 681.8 | 589.5 | **−10.4 %** |
| 1024 | 903.4 | 749.2 | 636.9 |  −3.2 % |
| 2048 | 940.1 | 775.6 | 658.0 | — |
| 4096 | 961.0 | 794.5 | 676.0 |  **+2.7 %** |

## DECODE (t/s, médianes) — insensible au batch, SUIF le cliff 512×long-ctx
| -b/-ub | 32k | 65k | 98k |
|---|---:|---:|---:|
| 512  | 24.98 | 23.92 | **14.6–19.8** ⚠ |
| 1024 | 24.96 | 23.92 | 22.97 |
| 2048 | 24.98 | 23.94 | 22.99 |
| 4096 | 24.97 | 23.93 | 22.98 |

### Le cliff 512 × long-contexte (re-qualifié, reproducible — `ab_ub_fix512.py`)
| ctx @ 512 | 65k | 80k | 98k |
|---|---:|---:|---:|
| decode t/s | 22.49 (normal) | 20.97 (début de dégrade) | **14.61 / 16.96** (cliff) |
| même ctx @ 2048 | — | — | 22.62 (normal) |

C'est par-token (pas un stall ponctuel : graphs reused=63 identique entre 512 et 2048, donc
pas de re-capture CUDA graph à chaque token). Mécanisme interne llama.cpp non identifié à
100 % (pas d'offload CPU, pas de context shift, pas de warn dans le log) — probable
interaction capture CUDA graph / scheduling du petit n_batch sur le FA long-ctx de ce modèle
hybride (Qwen3.5, full_attention_interval=4, couche récurrente, RS buffer 149 MiB).
**Impact décisionnel : ne pas descendre sous 1024 si on s'approche de 90k+ de contexte.**

## VERDICT — est-ce que 2048 vaut son coût VRAM ?

**Oui, 2048 est le point optimum. On le garde.**

- **2048 vs 512** : +615 MiB VRAM contre **+10 à +14 % de prefill**, decode identique — et
  2048 **évite le cliff decode** que 512 montre à 98k (14.6–17 vs 22.6 t/s). Excellent trade.
- **4096 vs 2048** : +820 MiB VRAM pour **seulement +2 à +2.7 % de prefill**, decode identique.
  **Pas rentable** — le rendement du batch est déjà épuigné à 2048.
- **1024** : milieu de route (−3.2 % prefill vs 2048, −410 MiB), mais commence à toucher le
  cliff decode en long ctx (à surveiller, pas mesuré directement). Pas le bon point d'ancrage.

Le gain du batch (2048) se lit donc **uniquement en prefill** (+10–14 % vs 512), jamais en
decode (le decode single-token ne dépend pas de -b, à l'exception du cliff petit-batch
long-ctx ci-dessus). Et son prix est borné : +615 MiB vs 512, +820 MiB vs 4096.

→ **Rien à changer en prod (-b/-ub 2048 déjà optimal). Seule nuance à retenir : ne pas
  repasser en -b 512 sur des sessions 90k+.**

Artefacts : `ab_ub_results.jsonl`, `ab_ub_summary.md` (sweep), `ab_ub_fix512_results.jsonl`,
`ab_ub_fix512_server.log` (re-qualification du cliff).
