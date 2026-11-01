# RUN9E — Baseline MTP-OFF, CUPTI, régime frais @ 90k (24/10 08:57 UTC)

**La baseline MTP-OFF qu'on n'avait jamais eue.** Harnais identique run9d (CUPTI v2,
.so prod `fa3e75c2`, KV q8_0, FA auto, -b 2048, temp 0/top_k 1, filler) SANS les args
`--spec-type draft-mtp …` ni l'env MTP. REQ A ~90k 32 gen chauffe / REQ B ~90k 512 gen
FENÊTRE, KV fraîche. Artefacts : `runs/cupti9e-20261024-085706/` (trace, phases, server out).

> Piège de script (documenté, sans impact) : `os.listdir(OUT)` attend un dossier mais
> `CUPATH_TRACE` écrit un fichier → rc=1 en post-mortem après la restore. Trace intacte.

## Résultats wall (serveur)
| | MTP-OFF (9e) | MTP-ON (9d, réf) |
|---|---|---|
| prefill @90k | **734.35 t/s** | 735.9 t/s (identique → prefill indépendant MTP) |
| **decode @90k** | **45.76 ms/tok = 21.85 t/s** | 27.27 ms/tok = 36.61 t/s |
| REQ A (32 gen) | 46.07 ms/tok | — |

→ **MTP = ×1.68 au régime 90k** (45.76 → 27.27 ms/tok). Coherent night-bench 18/10
(17.7 t/s @114k / 14.1 t/s @196k, ctx plus longs).

## Architecture confirmée (server log) : HYBRIDE
- `n_layer=64` : **16 couches full-attention** (KV cache q8_0, 4352 MiB @ 131072 cells)
  + **48 couches linear-attention** (gated_delta_net, état récurrent 149.62 MiB, SANS KV).
- Decode MTP-OFF = 1 forward/token : 15.3 appels FA/tok (16 layers − 1 shared), 45.9 gdn/tok,
  439.7 quantize_q8_1/tok (K/V f16→q8_0 des 16 layers).
- Le « mur KV q8_0 90k » du verdict fa-ter tenait car seule la partie full-attn lit la KV.

## BREAKDOWN DU PAS DECODE MTP-OFF @90k (fenêtre eval 23.4 s / 512 tok, busy 21.7 s)
| famille | ms | %busy | xN/512 | ms/tok | note |
|---|---:|---:|---:|---:|---|
| **mmvq (mul_mat_vec_q)** | **15 979** | **73.6 %** | 225 123 | **31.21** | 1 passage forward complet/token |
| **flash_attn** | **3 079** | **14.2 %** | 15 660 | 6.02 | `ext_mma_volta_small<256,NT=1,q8_0>` = **déjà T3** |
| rms_norm (+add_rms) | 767 | 3.5 % | 149 258 | 1.50 | |
| quantize_q8_1 (K/V) | 553 | 2.9 % | 225 123 | 1.08 | 439.7 appels/tok |
| gated_delta_net | 251 | 1.2 % | 23 479 | 0.49 | 48 couches linear-attn, pas de KV |
| nary (silu/softplus) | 248 | 1.1 % | 91 523 | 0.48 | |
| cpy_scalar + concat + k_get_rows + ssm_conv + bcast | ~567 | 2.7 % | | 1.11 | |
| **busy total** | 21 711 | | | 42.40 | + 3.2 ms/tok overhead (wall 45.67) |

mmvq par type (T=1) : Q6_K nofus 7.66 + fus 10.96 · Q8_0 nofus 5.08 + fus 2.77 · Q5_K nofus 2.86 + fus 1.72 = 31.05 ms/tok.
FA detail : ext NT=1 3 008.6 ms (moy 384 µs @ n_kv≈74k) + combine 69.9 + NT=3 2.2 (warmup A).

## Mur HBM2 revalidé SANS MTP
- mmvq = 31.21 ms/tok vs poids ≈ 25.65 ms à 900 GB/s → **≈ 82 % du peak** (le solde =
  460 launches de kernel/token + fusion args). Même conclusion qu'en MTP-ON (27.27 ms/step
  ≈ mur) : **mmvq est au mur dans les deux régimes — le CLOS est robuste au régime.**
- FA = 14.2 % (vs 15.9 % MTP-ON), kernel déjà `ext_mma_volta_small` = **T3** (le microbench
  t3_sweep mesurait précisément T=1/T=2 : +50–72 % vs stock). Pas de lever kernel restant.
- quantize_q8_1 = 2.9 % < seuil 5 % → pas de microbench (règle d'or mmvq).

## A/B de la nuit 18/10 (MTP-OFF, 64 gen) — les DONNÉES EXISTAIENT (synthèse cassée)
| exp | prod | candidat | Δ |
|---|---|---|---|
| 42d9581 (H3) | 17.69 / 14.13 | 17.59 / 14.05 t/s (2×2 reps, 114k/196k) | **−0.57 %** (cohérent, neutre-lent) |
| batch 4096 | 17.69 / 14.13 | 17.68 / 14.12 | **neutre** (b8192 = OOM, VRAM 32 477) |
| threads 12 | 17.69 | 17.61 | **neutre** |

→ Les 3 angles testés sans MTP en 18/10 étaient déjà neutres/négatifs. Plus un A/B à refaire.

## VERDICT — le tour MTP-OFF est CLÔTURÉ
1. Pas de terme >5 % actionnable : mmvq (73.6 %) = mur HBM2 (clos, revalidé), FA (14.2 %) = déjà T3 (clos).
2. Les 3 « CLOS » établis en MTP-ON (mmvq, FA, mapped_rows→absent) **tiennent en MTP-OFF** ;
   mapped_rows n'existe pas sans MTP (variant du verify).
3. Le decode sans MTP est 1.68× plus lent que avec MTP — c'est le prix du pas supplémentaire/token,
   pas un gâchis : le pas est déjà à 82–94 % du mur.
4. Leviers restants dans tout le projet (les deux régimes) : **aucun au niveau kernel**.
   Seuls resteraient : quantization plus agressive (risque qualité, hors périmètre) et la
   température du draft MTP (levier MTP, en attente du user).

## Invariants post-bench (vérifié)
service `ollama-v100` active, .so prod `fa3e75c2` intact, V100 libérée par le bench, modèle servi.
