# ÉTAPE A — VERDICT (profiling MTP) — 28/10 ~20:45 UTC

Bench `mtp_prof_stepA.py` (run 28/10 18:20-18:33 UTC, `.so` prod figé `8035caf4`,
config prod exacte, service restauré OUI, rc=0). Analyse finale de `mtp_prof_stepA_gputimes.tsv`
+ `mtp_prof_stepA_server.log` + `supervisor_bench.log`. Script reproductible :
`mtp_prof_stepA_analyze.py`.

## Résultat macro (déjà dans `mtp_prof_stepA_summary.md`)

| leg | ctx | decode t/s | acc rate/pos | dur(b,g,a) ms | VRAM pic |
|---|---|---|---|---|---|
| mtp-on | 114688 | **15.57** | (1.000,0.929,0.821) | (0.012,243.5,0.025) | 32438 |
| mtp-on | 160000 | **24.26** | (0.938,0.854,0.708) | (0.019,473.1,0.045) | 32438 |
| mtp-off | 114688 | 22.92 | — | — | 29768 |
| mtp-off | 160000 | 21.68 | — | — | 29768 |

Reproduit l'anomalie du 28/10 mat (t3_vldg, legs A/B ~15.95/24.49 MTP-ON) :
**MTP-ON < MTP-OFF @114k (−32 %)** et **inversion ctx** (114k < 160k).

## Signal principal de l'Étape A (NOUVEAU — ce que les gputimes disent)

Fenêtres decode ANCRÉES sur horodatages superviseur + server.log (eval_ms), t0=18:20:31.745 UTC :

| fenêtre decode | n | gpu_med | %gpu≥95 | mem_med | W_med |
|---|---|---|---|---|---|
| L1 mtp-on 114k (15.57 t/s, 4.11 s) | 39 | **23 %** | 0 % | **6 %** | **61 W** |
| L2 mtp-on 160k (24.26 t/s, 2.64 s) | 15 | **23 %** | 0 % | **6 %** | **74 W** |
| L3 mtp-off 114k (22.92 t/s, 2.79 s) | 33 | **99 %** | 94 % | **88 %** | **224 W** |
| L4 mtp-off 160k (21.68 t/s, 2.95 s) | 37 | **98 %** | 97 % | **88 %** | **227 W** |

**→ Le decode MTP-ON sous-occupe le GPU de façon massive et quasi totale**
(util ~20-25 %, mem ~5-6 %, ~60 W, 0 % des échantillons ≥95 %), alors que le decode
MTP-OFF est entièrement GPU-bound (util ~98 %, mem ~88 %, ~225 W). Constant dans les 2 ctx.

C'est le localisation macro de l'anomalie : **le MTP fait passer le decode d'un régime
GPU-bound à un régime où le GPU attend** (gaps hôtes / dispatch / sync / boucle spec,
ou kernels trop courts + sous-alimentation). L'acceptance (0.92 @114k) est BONNE et
identique A/B → ce n'est PAS le draft model : c'est le **coût par spec-step / l'hôte**
qui domine, et il croît avec le ctx (dur(g) 243 ms @114k → 473 ms @160k) sans que l'acceptance
compense (→ inversion ctx : @114k l'overhead l'emporte sur le gain, @160k l'acceptance retombe
mais le step y est moins pénalisé).

## ⚠️ Correction d'instrumentation à retenir

Le plan de l'Étape A demandait gputimes à **~100 ms (100 Hz)** pour résoudre la
périodicité spec-step (~62 ms/step @114k). **Non obtenu : cadence effective = 13 Hz**
(dt_med=76 ms, dt_mean=77 ms, n=9809 / 756 s). Cause : `gputimes_sampler.py` a bien
reçu `PERIOD=0.1` (`mtp_prof_stepA.py:108`), mais chaque tour nvidia-smi (fork+exec)
coûte ~75 ms → le minimum pratique est ~13 Hz. **Conséquence : la périodicité fine du
spec-step n'est PAS résolue par ce run** ; la signalisation GPU-bas/GPU-haut reste valable
(decode = 2.6-4.1 s ≫ 76 ms) mais on ne peut pas y lire le tempo ~62 ms.

(Corollaire : les « longues plages GPU 100 % » de ce TSV sont le **prefill**, pas le
decode. Les fenêtres decode sont courtes et nettement identifiables par mem 32438→GPU-bas.)

## Ce que l'Étape A ne fait PAS (→ Étape B)

Elle **localise** (GPU sous-occupé en MTP-ON) mais n'**attribue pas** le temps à
draft / verify / backend-sampling / gaps. Pour ça il faut le trace kernel-par-kernel :
**Étape B CUPTI** — l'infra est prête et éprouvée : `h2/cupti129-test/run_cupath_llama.sh`
(+ `libcupath.so`, `cupath_shim_v3.c`) tourne sur la vraie `llama-server` prod,
LD_PRELOAD, stoppe/restaure le service (trap EXIT), capture le chemin de décodage
(verify MTP) avec timestamps. C'est l'outil désigné par le bootstrap §4.

## Décision (zone 0-10 h, tranché seul)

- **Étape A : terminée**, conclusion = « decode MTP-ON = GPU sous-occupé / host-bound,
  overhead spec-step croissant en ctx ; pas de résolution de périodicité à 13 Hz ».
- **Pas de nouveau bench macro** (ne rejoue rien) et **pas de swap de `.so`**
  (ré-basage `d2df07b8` = 2e séquence, seulement après que B ait montré sur quel `.so`
  la régression est née).
- **Suite immédiate = Étape B CUPTI** (kernel-par-kernel, 1 point ctx ~114k, N_GEN ~64)
  pour produire le tableau d'attribution % draft/verify/sampling/gaps par ctx — la
  sortie attendue du bootstrap. Infra prête ; à lancer via superviseur (le run coupe le
  tir en cours, géré).
