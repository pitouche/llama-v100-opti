# Auto-bench 2026-10-01 — Qwen3.8-27B Q6_K (MTP+vision) sur V100 32GB
Version corrigée (20:55) après analyse des logs runner : le mécanisme de
réutilisation de contexte et plusieurs chiffres de prefill ont changé.

Setup: ollama custom (build v100) port 11434, runner -c 200000, MTP spec draft=3,
q8_0 KV, flash-attn, `--context-shift` + `--keep 4` en ligne de commande.
Model: qwen3.8-27b-q6-mtp-vision (24GB VRAM).
GTX 1070 (port 11435) = reranker, inactive pendant le bench.

## 0. CORRECTION — comment le runner réutilise le contexte (important)
Ce n'est **pas** un "context-shift qui compresse" le contexte (idée à jeter,
jamais observé dans les logs). Le mécanisme réel, vérifié dans le journal
(`runner_journal_bench.log`, 16 rechargements de runner sur la session) :

1. **Checkpoints de contexte** : le runner sauvegarde des checkpoints KV
   (taille ~149 MiB chacun) et, pour chaque nouvelle requête, **restaure le
   checkpoint au plus long préfixe commun** (LCP) avec le run précédent
   (`checking checkpoint with [x, x] against y` → `restored context checkpoint`,
   `selected slot by LCP similarity`, f_sim_best 0.98-1.000).
2. **Seule la diff est prefilled** : le reste est servi depuis la KV cache.
   Preuve directe (diff_check.jsonl, runner 200k, prompt ~28k) :
   | run | prompt | prompt_eval_count | temps prefill |
   |---|---|---|---|
   | A, prompt P froid | 27976 | 27976 | 35.0 s (799 t/s) |
   | B, prompt P identique | 27976 | 27976 | **0.09 s** (diff nulle) |
   | C, P + queue 8.4k | 36413 | 36413 | 13.9 s (**2613 t/s sur la diff**) |
   | D, idem C | 36413 | 36413 | **0.09 s** (diff nulle) |
   C'est propre et malin : une conversation qui s'allonge ne re-prefille que
   les nouveaux tokens.
3. **Artefact de mesure en découle** : l'API compte `prompt_eval_count` =
   prompt COMPLET mais `prompt_eval_duration` = temps de la DIFF seulement.
   Quand un checkpoint restaure le préfixe, les t/s de prefill rapportés
   sont gonflés (ex. 44529 "tok" préfillés en 28.9 s = seulement les 18.5k
   derniers en réalité → 1540 t/s fictifs au lieu de ~640 réels).
   La vision n'était pas affectée : images différentes → aucun checkpoint
   exploitable, `cached n_tokens` restait ~22 (template).

Conséquences : les préfill texte "1.45-1.55k t/s" du rapport initial sont
des artefacts ; les vrais chiffres sont dans la section 1. Et les runs qui
suivaient une conversation réelle (task 6226/6347, préfixe 86k restauré)
n'ont préfillé que 699 / 268 tokens.

## 1. Texte — prefill FROID (valeur réelle)
Points où rien n'était en cache (premier run, runner frais, ou préfixe différent) :
| prompt (tok) | prefill froid (t/s) |
|---|---|
| 73 | 153 (préchauffement graph CUDA) |
| 1676 | 780 |
| 3347 | 861 |
| 20022 | 837 |
| 48130 | 709 |
| 51030 | 703 |

**Plateau ~700-860 t/s** (~1.1-1.7 ms/tok), légèrement décroissant avec la
taille. Les "1.2-1.55k t/s" du rapport initial = mesure API avec checkpoint
du préfixe (artefact §0). Prefill chaud de la diff seule : ~2.6-3.6k t/s.

## 2. Vision — prompt eval vs taille image (chiffres VALIDES, runs froids)
Image synthétique 256→2048px, question courte, ctx texte ~40 tok :
| image (px) | tokens image | prefill (t/s) |
|---|---|---|
| 256 | 96 | 150 |
| 512 | 288 | 388 |
| 768 | 608 | 453 |
| 1024 | 1056 | 559 |
| 1344 | 1796 | 515 |
| 2048 | 4128 | 431 |

- Loi de tokenisation vision mesurée, parfaitement linéaire :
  **tokens image ≈ (largeur×hauteur)/1024 + 32** (1 token = 1024 px²).
  Pratique : 256px→96 tok, 1024px→1056, 2048px→4128.
- Prefill vision ~1/3 du texte froid (150-560 vs 700-860 t/s).
  Comparaison à contexte égal : texte 3.3k tok = 861 t/s, image 4.1k tok =
  431 t/s → la vision coûte ~2x plus cher en prefill par token.

## 3. Génération (MTP) vs taille de contexte
Acceptance relue dans le journal (plus fiable que mes souvenirs) :
| point | ctx final | gen (t/s) | acceptance MTP |
|---|---|---|---|
| sweep counting, run 1 | 1.6k | 69.2 | 1.000 |
| sweep counting, run 2 | 6.0k | 63.4 | 0.983 |
| sweep counting, run 3 | 11.9k | 50.4 | 1.000 |
| sweep counting, run 4 | 23.8k | 44.6 | 1.000 |
| sweep counting, run 5 | 40.0k | 39.6 | 1.000 |
| ctxfill 77-79k (KV chaud, counting 1..64) | 77-79k | 51.7 | ~0.5 |
| conversation réelle | 25.6k | 38.2 | ~0.5 |
| conversation réelle | 43.4k | 31.3 | ~0.5 |
| conversation réelle (contenu difficile) | 92.6k | 28.2 | 0.487 |

- **La baisse avec la taille du contexte est réelle et régulière** sur le
  même contenu (counting) : ~69 t/s à 1.6k → ~40 t/s à 40k, soit ~0.75 t/s
  par k tok de contexte (KV q8_0 + flash-attn : attention qui coûte de plus
  en plus cher, mais pas de chute brutale).
- L'axe dominant reste quand même le **contenu** : à contexte comparable,
  counting trivial (acceptance ~1.0) fait 40-70 t/s alors que du texte dense
  (acceptance ~0.5) fait 28-38 t/s.
- La ligne ctxfill 51.7 t/s à 77-79k n'est pas comparable au sweep : runner
  200k avec KV chaud vs runner 65k rechargé à chaque run du sweep.
- Référence vanilla (sans MTP) : 22-25 t/s → le MTP reste un gain ×1.4 à ×2.
- Les wall_s gonflés (129-260 s) venaient des 16 rechargements de runner
  (num_ctx 65k ↔ 200k) ; les t/s de gen eux sont purs.

## 4. Cross-check Grafana / Prometheus
- L'exporter (port 9444) + Prometheus + Grafana voient bien le run live :
  - ollama_generation_tokens_per_second[V100] : 26-69 t/s pendant les runs.
  - ollama_prompt_tokens_per_second[V100] : 128-947 t/s (même biais
    checkpoint que l'API : c'est la vitesse du delta, pas du prompt entier).
  - ollama_context_tokens[V100] : monte pendant le run, retombe à 0 après
    (comportement volontaire, IDLE_DECAY_S=60).
- Les gauges "live" sont calculées sur les deltas des lignes `cached
  n_tokens` : fidèles à l'activité réelle du run (donc cohérentes avec le
  §0), mais le premier échantillon de prefill peut être sous-estimé
  (base à plat). Les timings API font foi pour des chiffres "propres",
  à condition de vérifier s'il y a eu restauration de checkpoint.

## Conclusion (corrigée)
- Le runner réutilise le contexte par **checkpoints KV + LCP** : seule la
  diff est prefilled. Malin pour les conversations longues, mais ça fausse
  les t/s de prefill rapportés par l'API quand le préfixe est en cache.
- Prefill texte froid : plateau ~700-860 t/s jusqu'à 51k (pas 1.5k).
- Vision : ~1/3 du texte en prefill, 2048px ≈ 4k tok, loi (W×H)/1024+32.
- Gén MTP : baisse réelle avec la taille du contexte (~0.75 t/s par k tok
  sur même contenu), contenu/acceptance = axe dominant, gain MTP ×1.4-2.
- Monitoring Grafana OK pour du live ; chiffres propres = timings API +
  vérif des logs (checkpoint restauré ou non).

## Fichiers
- auto_bench_results.jsonl   (texte + vision, t/s API = biaisés par le cache)
- auto_bench_gen.jsonl       (sweep gen 1.6k→40k, runner rechargé par run)
- auto_bench_ctxfill.jsonl   (gen 77-79k, KV chaud)
- diff_check.jsonl           (preuve du mécanisme de diff : A/B/C/D + E froid 51k)
- runner_journal_bench.log   (journal du runner pendant la session, 18h-20h30)
- auto_bench_20261001.png    (figure corrigée : froid vs artefact)
- live_gauge_check.json      (cross-check gauges pendant un run)
- diff_check.py, plot_bench.py (scripts mis à jour)
