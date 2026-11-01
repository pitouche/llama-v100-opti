# Verdict T3 v2 (levier D) — load V q8_0 vectorisé 128-bit — CLOSÉ SANS DÉPLOIEMENT (27/10, confirmation binaire 28/10)

## Question
Réduire le coût du load V q8_0 dans le kernel T3 (`fattn-mma-volta-small.cuh`, phase 2) :
remplacer les `LDG.32` dispersés par des `LDG.128` alignés (rangée V q8_0 D=256 = 272 B
= 17 uint4) via un staging smem, en restant bit-identique (mêmes octets, même funnelshift,
même d2).

## Méthode
- Baseline cvec = `fattn-mma-volta-small.cvec.cuh` (md5 **dbd4987c**, lecteur V global baseline).
- Deux designs de staging testés, build dual-arch via `v100/dual-arch/build_dual.sh`
  (source `v100/dual-arch/src/...`, sortie `lib/libggml-cuda-dual.so`), prod jamais touchée.
- Microbench `mb_fa t3` (dlopen du .so, géométrie prod D=256 KV q8_0, n_kv=107351, pb auto)
  + A/B formel via `supervisor_bench.py run mb_fa_t3v2_bench.py 200 107351 --detach`.

## Résultats

| design | source md5 | .so md5 | résultat |
|---|---|---|---|
| v2 tent. 1 (LDG.128 direct global, sans staging) | non conservée (`.cuh` temporaire, cf. `build_t3v2.sh` `CUD_CUH`) | 0653dc50 | **−104 % @T=1** (825 vs 404 µs, A/B formel `mb_fa_t3v2_summary.md`, bit-identique) — **attribution au build confirmée par SASS le 28/10, cf. § ci-dessous** |
| v2.1 full-step (128 rangées/step, co-op, 2 `__syncthreads`) | 2691b94e | d3cbb710 puis 205bd399 | **~−50 % @T=1** (613-617 vs 406 µs) ; crash `GGML_ASSERT(max_blocks_per_sm > 0)` dès T=2 (smem ~43 Ko → 0 CTA/SM dans `launch_fattn`) |
| v2.2 warp-private (32 rangées/warp, `__syncwarp`, 0 syncthreads) | 2db57e47 → 3501b4b7 (fix `const Vdst` + indexation `warp*544`) | 523e1444 → b40c1c99 | **crash `illegal memory access`** dès le premier run T=1, non résolu |
| baseline cvec (repli) | dbd4987c | 3504eb9a puis d1e2d5de | OK, référence prod (406-407 µs @T=1 re-mesurés 27/10) |

## Pourquoi ça ne passe pas (à retenir)

1. **Occupancy vs staging** : le staging d'un step complet (128 rangées × 272 B = 34.8 Ko,
   total ~43 Ko avec Qs/Ps/Os) fait chuter l'occupancy : −50 % sur T=1 même quand ça tient,
   et 0 CTA/SM (assert `launch_fattn`, `fattn-common.cuh:1285`) dès que la géométrie NT>1
   dépasse la borne smem de la V100 → crash au launch. Le kernel T3 est latence-bound :
   on ne peut pas lui enlever des CTAs pour gagner des LDG.128.
2. **Le staging warp-private est incompatible avec la phase 2** : `o0 = pv - k0` parcourt
   jusqu'à ~123 (D=256, NCH=16, oslice) alors que la tranche d'un warp ne couvre que
   32 positions → le design tel que conçu lit hors-bounds de sa propre tranche de Vbuf.
   Le crash IMA (rapporté en différé au premier `cudaFuncGetAttributes`, mais déclenché
   par le kernel) confirme un accès hors-bounds. Le corriger exige de restager 128 rangées
   par warp (×4) → retour au problème 1.
3. **Dilemme fondamental** : aligner la rangée (272 B, multiple de 16) ne couvre que
   32 rangées par 8.7 Ko ; couvrir les 128 positions lues en phase 2 coûte 34.8 Ko.
   Il n'existe pas de taille de staging à la fois alignée, suffisante ET ≤ ~13 Ko
   (la marge occupancy du kernel). Le levier D tel que spécifié est un dead-end.

## Conséquences
- **Prod inchangée** : `libggml-cuda.so` md5 **4fd152f9** (vérifié à chaque étape),
  service `ollama-v100` active, aucun déploiement, aucun e2e lancé.
- `.so` dual actuel = baseline cvec rebuild (d1e2d5de), source `dbd4987c` — en repli
  uniquement, pas en prod.
- Artefacts conservés dans `h2/cupti129-test/e2/t3v2/` :
  `fattn-mma-volta-small.v21-fullstep-2691b94e.cuh`, `fattn-mma-volta-small.v22-warp.cuh`
  (2db57e47), `fattn-mma-volta-small.v22-warp-3501b4b7.cuh` (== baseline après le
  rollback final — le build 3501b4b7 n'a jamais existé à l'identique, cf. § ci-dessous),
  `backups/fattn-mma-volta-small.cvec.cuh` (dbd4987c),
  `candidate-attempt1-0653dc50.so` (rejeté, −104 %).
- Note sur les md5 de session : la source `3501b4b7` est le code warp v2.2 corrigé
  (fix `const Vdst` + indexation `warp*544`) — elle a bien été compilée en `.so`
  **b40c1c99** (crash IMA dès T=1, non résolu). Elle n'est PAS une variante du baseline.
  En fin de session la source du tree dual a été restaurée à `dbd4987c` (rebuild d1e2d5de).

## Confirmation binaire du −104 % (tent. 1, investigation du 28/10)

Question de la nuit : le −104 % mesuré le 27/10 est-il l'effet réel du build 0653dc50,
ou un artefact (mauvais cubin comparé, drift de mesure, sm_61) ? Réponse : **effet réel,
confirmé au niveau SASS**.

- Le kernel prod exact (D=256, NT=1, K=V=q8_0, softcap=0, skip_masked=0, mangled
  `_Z30flash_attn_ext_mma_volta_smallILi256ELi1EL9ggml_type8ELS0_8ELb0ELb0EE...`) vit
  dans le **cubin 213** des 3 `.so` (prod, dual cvec, candidat). Attention : le
  cubin 207 contient des T3 D=256 **f16** — non pertinent (fausse piste de la nuit).
- Comparaison des 150 cubins sm_70 (0653dc50 vs d1e2d5de) : **146 identiques,
  4 DIFF = 209 (D=64 f16) / 211 (D=128 q8_0) / 213 (D=256 q8_0) / 215 (D=64 q8_0)**.
- Sur ces 4 cubins : **prod == cvec** (md5 SASS identiques : 209 5d7ed677, 211
  90dbab61, 213 59ef1fa2, 215 b1cd2d8f) et **cvec ≠ 065**. Le `.so` dual repli est
  donc fidèle à prod, et le candidat diffère réellement sur le kernel prod exact.
- Corps SASS du kernel prod (cubin 213) : md5 `0f87f327…` (065) vs `24d5f89c…` (cvec),
  **différents** ; 4537 vs 3841 lignes (diff unifié 8292 lignes). Le candidat a
  massivement vectorisé les loads : **128 `LDG.E.128` + spills `LDL`/`STL`**
  (STACK **112 B** en resource-usage vs 0 au baseline ; REG plafonné à 128,
  ptxas baseline NT=1 = 123 regs 0 spill). Le code généré est plus long ET déborde
  en local — exactement le profil d'une régression −104 %.
- La source de la tentative 1 n'est pas conservée (`backups/` ne contient que le
  baseline cvec), mais la signature binaire (128 LDG.128 + 112 B de spill) signe la
  vectorisation 128-bit du load V : l'attribution est solide, la forme est rejetée.
- Piège tooling documenté : un script de comparaison de la nuit (`/tmp/cmp_sass.sh`)
  testait `$i.txt.txt` au lieu de `$i.txt` → fausse conclusion « AUCUN diff » ; la
  re-comparaison propre ci-dessus (hash SASS des 150 cubins) fait foi. `nvdisasm`
  CUDA 12.8 (venv triton) : les fonctions apparaissent comme **labels en colonne 0**
  (`_Z30flash…:`), pas `.visible .entry` — extraire par label, couper au label suivant.

## Pistes alternatives si le FA redevient le 1er poste
- **LDG.128 global sans staging : testé (tent. 1, 0653dc50) → −104 %**. La lecture
  vectorisée directe ne suffit pas : le coût est dans la latence exposée / l'occupancy,
  pas seulement dans la largeur du LDG. Ne pas ré-essayer sans changer la structure.
- **Réduire le nombre de lecteurs V** (partage des fragments B entre lanes) plutôt que
  d'en cache — touche à la structure m8n8k4, plus risqué.
- **Accepter le ~7-10 % d'efficacité TC** : le ROI app estimé est +4-5 % decode long ctx
  (OPTI.md §9.5) ; le levier D est maintenant écarté, il reste le partage de fragments B
  et les leviers ILP/accumulateurs (§9.5).
