# fa-ter / FATTN_VS_STEP — levier INVALIDÉ (23/10, ~23:50 UTC)

## Verdict : PAS DE GAIN. STEP ≠ 128 est un bug de correction, pas un tuning.

## Contexte
- Cible : `flash_attn_ext_mma_volta_small<256,3,q8_0>` (84,3 ms, ~8,6 % du pas GPU warm, 309 µs/appel, ×272).
- Piste explorée : `FATTN_VS_STEP` (défaut 128), décrit dans le code comme "KV positions per CTA step",
  jamais sweepé. Hypothèse : STEP=256/512 → moins de tiles KV → split-KV plus gros → plus rapide.
- Microbench naïf (mb_fa, timings seuls) montrait −38 % (STEP=256) / −65 % (STEP=512) sur n_kv=36312,
  T=4 (l'instance prod NT=3). MAIS le champ `out` de mb_fa (4 floats) différait entre variantes →
  flag de correction.

## Pourquoi mb_fa ne pouvait pas trancher
mb_fa ne zéro PAS le padding K/V (n_kv→n_kv_pad) : les blocs q8_0 de padding sont des octets du pool
VMM, non initialisés, et le kernel lit jusqu'à k_max = ne[1] = n_kv_pad → sorties non déterministes
entre processus/.so. Le diff d'`out` ne permettait pas de distinguer (a) artefact de padding de
harnais, (b) reassociation FP, (c) vrai bug.

## Validation formelle (méthode réutilisable)
Artefacts (bench/) :
- **val_fa_step.cu** : harnais de validation. Mêmes tensors que prod (D=256, qh=24, kvh=4, T=4,
  Q f32, K/V q8_0, mask f16=0), seed=12345, **padding K/V explicite zéro**, dump du TENSOR DE
  SORTIE ENTIER (24576 floats) + dump des données host (`VAL_DUMP_DIR`). Build nvcc sm_70 g++-13.
  `MB_FA_SO=<.so>` pour pointer un candidat.
- **ref_fa.py** : référence CPU exacte (dequant q8_0 numpy, softmax fp64, layout [kh][p][d32]
  = layout du tensor nb11=8 blocs, offset hq = h*(T*D)+t*D).
  Usage : `python3 ref_fa.py hq.bin hkv_<n_kv>.bin <n_kv> <dump_gpu.bin> <tag>`

Résultats (T=4, split actif aux 3 tailles, même données host) :

| n_kv | variante | max_abs | max_rel | rmse |
|---|---|---:|---:|---:|
| 512 | prod (STEP=128) | 0.023 | 9.6 | 0.0035 |
| 512 | STEP=256 | 32.8 | 4232 | 4.98 |
| 512 | STEP=512 | 36.2 | 11125 | 6.64 |
| 1536 | prod | 0.028 | 12.3 | 0.0033 |
| 1536 | STEP=256 | 30.0 | 18672 | 4.41 |
| 1536 | STEP=512 | 40.9 | 6638 | 6.31 |
| 36312 | prod | 0.028 | 3.7 | 0.0032 |
| 36312 | STEP=256 | 21.6 | 12711 | 2.97 |
| 36312 | STEP=512 | 29.8 | 17521 | 4.46 |

→ **prod est correct** (rmse ~0.003 = bruit fp16 attendu vs ref fp64, même split-KV actif).
→ **STEP=256/512 produisent du garbage** (rmse 3–7, max_abs ~30).

## Racine (confirmée par lecture du code, fattn-mma-volta-small.ORIG.cu L354-378)
```
for (int k0 = blockIdx.y*FATTN_VS_STEP; k0 < k_max; k0 += gridDim.y*FATTN_VS_STEP) {
    const int pos = k0 + 32*warp + lane;   // 4 warps × 32 lanes = 128 positions par itération
```
La couverture par itération est **fixe à 128 positions** (4 warps × 32 lanes), indissociable de la
géométrie du kernel. `FATTN_VS_STEP` ne peut être que 128 : avec 256/512, les CTA sautent 128/384
positions sur 256/512 → **50 % / 75 % des positions KV ne sont jamais traitées** → l'attention est
calculée sur un sous-ensemble → sortie fausse. Le "gain" −65 % du microbench = simplement moins de
travail exécuté (bug). Les timings STEP=256/512 de mb_fa_step_sweep.jsonl sont sans valeur.

À retenir : `FATTN_VS_STEP` n'est PAS un knob de tuning libre ; le split-KV existe déjà via
`parallel_blocks` (heuristique d'efficacité de vagues, fattn-common.cuh L1280-1360).

## NCU — non utilisable sur cette V100 (à ne plus retravailler)
- ncu 2025.4.1 (cuda 13.1 et /opt/nvidia/nsight-compute) : `--list-chips` ne contient **pas GV100**
  → "Profiling is not supported on device 0" (même sous sudo, donc ce n'est pas juste
  ERR_NVGPUCTRPERM). Aucun ncu 12.x installé. → profiler ce kernel via ptxas + CUPTI + microbench.

## Conséquences
1. Le .so prod (fa3e75c2, STEP=128) est le bon ; NE PAS deployer fa_step256/fa_step512
   (libggml-cuda-fa_step256.so / fa_step512.so dans h2lib/lib = candidates INVALIDES, garder comme
   contre-exemple, pas de risque tant qu'ils ne sont pas copiés sur le .so prod — md5 prod revérifié
   fa3e75c2 après chaque build).
2. FA volta_small : pas de levier rapide (split-KV déjà optimal heuristiquement ; améliorer exigerait
   de réécrire la géométrie warp/positions du kernel, pour ~9 % du pas).
3. fa-ter CLOS. → passer à `older` : mapped_rows (4,4 %), quantize_q8_1 (2,5 %), gated_delta_net.
