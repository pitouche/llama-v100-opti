# BILAN SESSION 17/10 — LEVIER BATCH : le gain reel, DEPLOYE en prod

## 1) Ce qui a ete teste (MTP OFF, .so prod a4b6f440, interleave = gold standard)

| test | resultat | verdict |
|---|---|---|
| patch 42d9581 (nwarps K-quant 4→2) | −1%…−25% (noise, rien a gagner) | **rejeté** — voir ab_mmvq2_CONSOLIDE.md |
| env Q5_X4 / Q8_FATTN_TC | ±2% au regime prod 114k | **neutres** — voir ab_env_BILAN.md |
| **-b/-ub 512→4096** | **prefill +14 a +22%, decode long-ctx +35 a +44%** | **GAGNANT — deployé** |

## 2) Le gain batch (ab_batch + ab_confirm, 12 mesures, 3 reps ordre alterne)

**Decode (t/s) MTP-off, variance inter-reps < 1% :**
| contexte | b512 | b2048 | Δ |
|---:|---:|---:|---:|
| **114688 (regime prod)** | 12.08 | 17.39 | **+44%** |
| 196608 | 10.46 | 14.13 | +35% |
| 16-32k | 23-25 | idem | 0% (stable, attendu) |

**Prefill (t/s) :**
| contexte | b512 | b2048 | b4096 |
|---:|---:|---:|---:|
| 114688 | 551 | 620 | 636 |
| 196608 | 420 | 464 | 478 |

Donnees brutes : `ab_batch_results.jsonl`, `ab_confirm_results.jsonl` (12 lignes valides).

## 3) POURQUOI b512 en prod (et b2048 au bench)
Le scheduler ollama (`server/sched.go:827 automaticGenerationBatch`) calcule -b tout seul :
ctx>32k demarre a 2048, puis `generationBatchFits()` demande VRAM prédite ≤ 60% de la dispo
pour un batch ≥ 2048 ; sinon cascade 2048→1024→512. A 200k de ctx + MTP + vision, la V100
(32 GB, runner a 32.35 GB) ne passait pas le fit → **retombe a 512, le pire des 4**.
Le bench 196k sans MTP passait le fit a 2048 (30.7 GB) → le gain etait la, mais le runner 200k
n'avait pas les ~0.8 GB de marge.

**Resolution (deploy 17/10 23:50, fait par guile en main) : descendre le ctx pour libérer la
marge.** Le 175k ne passait pas non plus (allocs CUDA), **167k passe avec 60 MB de free**.

## 4) ETAT PROD FINAL (vérifié 00:30 le 18/10)
- runner : `-c 167000 -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3
  --context-shift --flash-attn auto --cache-type-k/v q8_0` (stable, uptime > 13 min, 0 eviction)
- model : qwen3.8-27b-q6-mtp-vision:latest, **24.6 GB sur GPU (100% GPU, aucun split)**
- V100 : **32436 / 32768 MiB** — marge au poil de cul (≈330 MiB nvidia-smi / ~60 MB sur
  Grafana selon le moment). Buffers reels runner 167k+b2048 (log 00:16) : model 21013.71 +
  KV 5550.50 + RS 598.50 + compute CUDA 1626.33 + compute host 733.34.
- **0 eviction depuis le demarrage du runner 167k (00:16:45)**. Les 22 lignes evict/offload
  de la fenetre 23:30-00:30 sont toutes dans 23:48-23:57 = les essais rates 175k/170k avant
  de trouver 167k. Pas d'incident sur le runner en place.
- service ollama-v100 active, ctx API 167000, .so a4b6f440 INTACT, t/s live ~30
- note : `--context-shift` est passe dans les args → si le ctx depasse, c'est le shift qui
  gere maintenant (et non l'erreur). Surveiller Grafana si un user pousse > 167k.

**Gain attendu realtime (mesures MTP-off) : prefill ~+12%, decode long-ctx ~+35-44%.**
A confirmer au fil de l'eau sur Grafana (les t/s live pre-deploy etaient ~26-30 au regime 114k
MTP-on).

## 5) Pièges tombés cette session (a ne pas refaire)
1. **TPS extraction cassée** dans deploy_175k.sh : le grep non ancree capturait les lignes
   HELP/TYPE → `tps=\n\n32.45` ≠ "0.0" → le script attendait l'idle forever (fix : ancrer
   `^ollama_generation_tokens_per_second{gpu="V100"}`).
2. **Pas de sudo pour ecrire le manifest** ollama (fichiers root) → PermissionError.
   Les 2 scripts deploy_*.sh sont a corriger (sudo tee) si on s'en resert.
3. **VRAM : mon estimation 175k+b2048=32336 MiB "ca passe" etait fausse** — les allocs CUDA
   ont une granularite (la KV 175k=5811 MiB theoriques mais l'alloc passe a 170k pas a 175k,
   167k oui). Marge reelle : ~60 MB. N'AJOUTER AUCUN autre buffer sur ce runner sans
   redescendre le ctx.
4. Le run ab_confirm a plante sur un `NameError` dans le summary (apres ecriture du jsonl) →
   le superviseur a vu "service inactive" et a tire la reprise OWUI. Fix : `restore_service()`
   dans un `finally`, avant la gen du rapport.

## 6) Pour le PROCHAIN chat (pistes restantes, par ROI)
1. **Confirmer le gain en conditions reelles MTP-on** : comparer les t/s Grafana pre/post
   deploy (la mesure bench est MTP-off ; le cycle draft/verify change les ratios).
2. **Surveiller le 60 MB de free** : si evictions/appareils CUDA en production → redescendre
   ctx a 160k ou passer la KV en q4_0 (libere ~2.5 GB, qualite a verifier).
3. **MTP : verifier que le spec type tire son epingle** avec le nouveau batch (n_max=3,
   draft acc/len dans le journal prod).
4. Le Q5_K a 47% du pic reste une piste kernel (rebuild), pas un env — voir notes
   `v100-repro`.

## Fichiers de la session
- `ab_batch.py` / `ab_batch_results.jsonl` / `ab_batch_summary.md`
- `ab_confirm.py` / `ab_confirm_results.jsonl` (12 mesures)
- `deploy_175k.sh` (+ backup manifest/unit dans `deploy_175k_*/`), `deploy_batch2048.sh` (abandonné)
- `ab_mmvq2_CONSOLIDE.md`, `ab_env_BILAN.md` (les 2 negatifs)
