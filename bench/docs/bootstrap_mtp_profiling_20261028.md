# BOOTSTRAP — Profiling MTP (toutes phases) — chat neuf (28/10 16:20 UTC)

**Objectif du chat :** comprendre **où passe le temps du MTP**, phase par phase.
Pas d'optimisation ici — **mesure et attribution**. Sortie attendue : un tableau
"dans un spec-step MTP, X % = draft, Y % = verify, Z % = sampling/backend, W % = gaps",
par ctx (114k / 160k), en config prod exacte.

---

## 1. Pourquoi ce chat (le signal à expliquer)

Anomalie **préexistante**, affecte la **prod elle-même** (hors périmètre vldg, qui est
byte-identique et déjà A/B'd) :

| | 18/10 (BILAN_NUIT, MTP-ON) | 28/10 (t3_vldg_mtp_bench, leg A = prod) |
|---|---|---|
| @114 688 | **45.73 t/s** | **15.95 t/s** |
| @160 000 | **36.23 t/s** | **24.49 t/s** |

Signaux inquiétants :
1. **Chute globale** ~2.9× (45→16) sur le même type de config.
2. **Ordre des ctx inversé** : 114k (15.95) **plus lent** que 160k (24.49) — alors qu'un KV plus long devrait être plus lent.
3. **MTP-ON (15.95) < MTP-OFF (22.68) @114k** alors que le 18/10 le MTP-ON était +135 %.

→ Le MTP est **dégradé / anormal** en régime prod actuel. Le profilage doit localiser
la cause (draft ? verify ? sampling ? overhead spec-step ? un gap CPU/hôte ?).

Acceptance MTP mesurée 28/10 : acc 0.92 (mean len 3.71) @114k, 0.741 (3.15) @160k —
**identique A vs B** donc le draft model n'est pas le suspect du delta vldg, mais l'absolu
reste à comprendre.

---

## 2. Config MTP exacte (PROD à reproduire, ne pas dévier)

Runner prod (pgrep 28/10, port 42545) :
```
-c 128000 -np 1 --mmproj <blob> --spec-type draft-mtp --spec-draft-n-max 3 \
--spec-draft-backend-sampling --cache-type-k q8_0 --cache-type-v q8_0 \
--flash-attn auto -b 2048 -ub 2048 --split-mode none --main-gpu 0 --context-shift --keep 4
```
Harnais de référence : `bench/t3_vldg_mtp_bench.py` (déjà calé sur config prod) :
`MAX_CTX=167000`, env prod = `H2 OFF (FATTN_TC=0)`, `shortlist=131072`,
`DEFER_PROMPT=1`, `GGML_CUDA_VOLTA_Q6_W4R4=1`, `LLAMA_ARG_FIT_TARGET=1024`, CVD=0 (V100),
port 8917, KV q8_0, FA auto. Points de mesure : **114 688** et **160 000**.

> ⚠️ Le 18/10 (45.7/36.3) était **MTP-ON mais avant T3/vldg** (prod `.so` d2df07b8 = T3
> seul, H2 désactivé). Aujourd'hui prod = `8035caf4` (T3 + vldg, dual). Pour re-baser
> l'anomalie, comparer **toujours sur le même `.so`** ; l'absolu 28/10 (15.95/24.49) est
> le repère de travail, et le 18/10 le point historique à réconcilier.

---

## 3. Les phases MTP à disséquer (décomposition cible)

Un spec-step MTP (n_max=3) = 1 passe de draft + 1 passe de verify multi-token + sampling.
Ce qu'on veut mesurer, par phase, en temps GPU (et gap hôtes) :

1. **DRAFT** — inférence du draft model (chemin f16, ctx_dft), n_max-1 pas.
   - Quels kernels ? (FFN small-batch / mmsmt / GEMV) — c'est le périmètre historique T1.
   - Temps par pas de draft.
2. **VERIFY (target)** — 1 passe forward du target sur les n_max tokens candidats
   (rows 2-8, q8_0). C'est ici que vit le kernel attention T3 (déjà optimisé).
   - Décomposer : **attention (T3)** vs **FFN target** vs **KV load/store**.
3. **BACKEND-SAMPLING** — `--spec-draft-backend-sampling` : le sampling est fait côté
   backend (GPU). Kernels de sampling, coûts.
4. **GAPS / CPU** — temps hôte entre kernels, dispatch, sync, la boucle spec du runner.
   (C'est souvent là que se cache l'anomalie si le GPU est sous-occupé.)

→ Sortie : **attribution % par phase**, par ctx, + identification de la phase anormale.

---

## 4. Outils dispo (vérifié 28/10)

- **NCU 2024.1.1** : `/home/guile/ncu-2024.1.1/usr/bin/ncu` (ncu + ncu-ui). **Refus SM70 ?** →
  tester d'abord (les CUDA 13.1 de la box refusent SM70 pour nvdisasm/cuobjdump — NCU
  peut être pareil).
- **CUPTI infra** : `h2/cupti129-test/lib64/` (`libcupti.so.12`, `libcupti_static.a`,
  `libnvperf_target.so`) + shims `cupath_shim{,_v2,_v3}.c` + `libcupath{,_v2,_v3}.so` +
  `e1_bench.cu`, `kapp.cu`, `e2/traces/`. C'est l'infra qui a fait les profils CUPTI e1/e2.
  **Préférer cette voie** pour un trace kernel-par-kernel + timestamps (attribution
  des phases + des gaps), car elle tourne sur la vraie `llama-server`.
  Exemple de protocole déjà rodé : `h2/cupti129-test/run_cupath_llama.sh`
  (stop service → capture CUPTI → restore auto, trap EXIT).
- **gputimes** : `bench/*_gputimes.tsv` = `epoch_ms util_gpu util_mem mem_MiB power_W`
  (échantillonnage nvidia-smi ~1 s). Bon pour la macro-structure (phases visibles à
  l'œil), pas pour le détail kernel. Sampler : `bench/gputimes_sampler.py`
  (modifier la période si besoin — par défaut 1 s).
- **nsys** : **pas trouvé** dans la box (ni PATH ni /home/guile ni tools/cuda-12.6).
  À installer si requis (CUDA 12.6 toolkits dans `tools/cuda-12.6/` — vérifier si nsys-y).

> Méthode conseillée : **CUPTI trace** sur un run MTP-ON court (N_GEN ~64, 1 point ctx),
> avec `--log-verbosity 4` + timestamps par kernel → classifier les kernels en
> draft/verify/sampling par leur grid size / nom / contexte. Recouper avec **gputimes**
> pour les gaps hôtes. NCU en 2ᵉ passe sur le kernel dominant de la phase suspecte.

---

## 5. Contraintes / règles (inchangées)

- **Ne JAMAIS stopper/redémarrer le service** hors séquence bench (superviseur).
- **Bench via superviseur** : `python3 supervisor_bench.py run <bench.py> [args] --detach`
  (README = autorité : `bench/supervisor_bench_README.md`). Le fichier doit être **Python**.
- **Ne jamais `cp` in-place** un `.so` prod ; backup + swap. **Ne pas toucher la prod**
  pour du profiling (lire, tracer, mesurer — pas déployer).
- `wait` ≤ 300 s par commande. Sorties volumineuses → filtrer/limiter/fichier.
- V100 = CUDA dev 0 (CVD=0). nvidia-smi idx 1 = V100, idx 0 = GTX 1070 (inversé en smi).
- Un bench coupe le tir agent en cours → c'est géré par le superviseur/watchdog.

---

## 6. État du système (28/10 ~15:50 UTC, après déploiement vldg)

- Prod `.so` = **8035caf4** (vldg dual sm_70+sm_61, byte-identique au kernel validé).
- Rollback : `/var/backups/llama-experiments/libggml-cuda.so.prod-4fd152f9-20261028-152612`.
- Service `ollama-v100` **active** ; runner MTP ON opérationnel ; V100 idle 0 % / 3947 MiB.
- 0 erreur CUDA. 0 process résiduel bench.

---

## 7. Points d'entrée (fichiers)

- Verdict vldg (contient les chiffres MTP-ON 28/10 + l'anomalie §6) :
  `bench/t3_vldg_verdict_20261028.md`
- Harnais MTP-ON prêt : `bench/t3_vldg_mtp_bench.py` (+ results.jsonl/summary/gputimes).
- BILAN historique 18/10 (45.7/36.3, T1 NO-GO, H2 désactivé) : `bench/BILAN_NUIT_20261018.md`
- CUPTI infra : `h2/cupti129-test/` (shims, e1_bench, e2/, traces/, ncu_e1/).
- NCU : `/home/guile/ncu-2024.1.1/usr/bin/ncu`
- Mémoire MTP/OP2 : `V100/Qwen3.8-MTP/OP2-Q8_0-QPN` (contexte historique QPN, peu lié ici)
- Superviseur : `bench/supervisor_bench.py` + README.

---

## 8. Plan suggéré (à valider avec l'user)

1. **Étape A — Macro (gputimes + logs)** : 1 run MTP-ON @114k court, gputimes à ~100 ms
   (pas 1 s) + `--log-verbosity 4` → repérer la périodicité spec-step et les gaps hôtes.
   C'est l'indice le moins cher sur "MTP-ON < MTP-OFF" et "ordre ctx inversé".
2. **Étape B — CUPTI trace kernel-par-kernel** : classifier draft/verify/sampling,
   sommer le temps par classe, par ctx. → le tableau d'attribution %.
3. **Étape C — Driller la phase suspecte** (celle qui a bougé 18/10→28/10) : NCU dessus
   (si SM70 ok) ou CUPTI métriques (long-scoreboard, DRAM, SM occupancy).
4. **Étape D — Hypothèses testables** : e.g. coût par spec-step en f(ctx) (l'inversion
   ctx pointe soit sur un realloc/shift KV, soit sur un comportement du draft à ctx court,
   soit sur un overhead fixe mal amorti). Comparer MTP-ON vs MTP-OFF kernel-par-kernel
   pour voir **exactement** ce que le MTP ajoute.

**Première question à trancher** : re-baser l'absolu — est-ce que la chute 45→16 vient
d'un changement **`.so`** (d2df07b8→8035caf4) ou d'un changement **config/env** ?
→ le plus vite fait : re-lancer `t3_vldg_mtp_bench.py` leg A **avec le `.so` d2df07b8**
(backup dispo) pour isoler. (À confirmer avec l'user car c'est un 2ᵉ déploiement/rollback.)
