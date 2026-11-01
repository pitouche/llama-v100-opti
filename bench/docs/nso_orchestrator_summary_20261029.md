# Note de synthèse — l'ORCHESTRATEUR CUDA du decode steady (V100, MTP)

Date : 2026-10-29
Trace : `nightrun8/traces/cupath8_230436.667897` (CUPTI 12.9, shim `libcupath_v2.so`)
Analyseur : `bench/nso_orchestrator_cupath_analyze.py` · sortie brute : `/tmp/analyze8.txt`
Regime mesuré : task 95 — 64 tok / 1092.97 ms = **57.64 tok/s**, acc MTP = 3.91, `graphs reused = 28`.

---

## 1. Verdict nsys (30/10 : **route réouverte** — voir `bench/GPU_UUID_NSYS_CUPTI_20261030.md`)

> ⚠️ **CORRIGÉ le 30/10.** Le verdict « route morte » ci-dessous portait sur nsys **sans**
> `NSYS_CUPTI_LIBRARY_PATH`. Avec `NSYS_CUPTI_LIBRARY_PATH=bench/nsys-cupti129` (répertoire
> contenant **une seule** `libcupti.so.12.9`), nsys 2025.5.2 capture le CUDA complètement :
> bench nso b2048 du 30/10 → `bench/nso_b2048_20261030-015122.nsys-rep` (117 133 events CUPTI,
> 90 030 kernels, `--cuda-graph-trace=node`, config prod exacte H2 OFF). Les `.rep` du 29/10
> cités ci-dessous sont bien vides (0 CUDA) — c'est le comportement **par défaut** (CUPTI 13.1
> embarquée). La capture via le shim libcupath reste utile (traces « pures » sans nsys), mais
> n'est plus l'unique route.

`nsys` (Nsight Systems 2025.5.2, `/opt/nvidia/nsight-systems/2025.5.2`) sans CUPTI forcée ne capture **aucun**
événement CUDA sur ce setup :

- Le `.rep` du bench nso (`bench/nso_b2048_20261029-205411.nsys-rep` + `.sqlite`) contient
  seulement des événements **OSRT** / syscalls — **aucune table CUDA kernel/API**.
- Erreurs CUPTI dans `DIAGNOSTIC_EVENT` :
  ```
  Failed to enable CUPTI activity kind 1.
  EnableActivity(handle, kind) returned 2: CUPTI_ERROR_INVALID_DEVICE
  No CUDA events collected. Does the process use CUDA?
  Number of CUPTI events produced: 0, CUPTI buffers: 50.
  ActivityFlushAll returned 2: CUPTI_ERROR_INVALID_DEVICE
  ```
- Cause : nsys 2025.5.2 charge `libcupti.so.13.1`, le driver système est **580.178.04**
  → mismatch CUPTI 13.1 / driver 580 → `CUPTI_ERROR_INVALID_DEVICE`.
- Reproduit aussi sur un binaire CUDA synthétique (`/tmp/nsysanity/cudatest`, kernel simple en
  boucle sur V100) : `report1.nsys-rep`, `sanity3.nsys-rep`, `oneshot.nsys-rep`, `t13.nsys-rep`
  (avec nsys de CUDA 13.1) — tous sans kernels.

→ **Ne pas relancer de bench nsys pour les kernels/API CUDA.** La route exploitable est la
  capture CUPTI 12.9 via le shim `LD_PRELOAD libcupath_v2.so`.

## 2. Méthode retenue : traces CUPTI 12.9 du shim `libcupath`

- Répertoire : `/home/guile/llama-experiments/h2/cupti129-test`
- Shims : `libcupath.so`, `libcupath_v2.so`, `libcupath_v3.so` (+ `.c` sources)
- Format binaire **v2** : 56 octets / record, little-endian :
  ```
  0  u64 seq      8  u64 gpu_start (ns, domaine CUPTI ≈ CLOCK_REALTIME)
  16 u64 gpu_end  24 u64 dur_ns    32 u64 corr (correlationId CUPTI)
  40 u32 stream   44 u32 op        48 u64 aux  (RT: cbid ; GRAPHS[v3]: graphId)
  op : 0=KER 1=CPY 2=RT 3=RESET 4=MARK 5=GRAPHS (v3 seulement)
  ```
- Fichiers par préfixe `<prefix>` : `<prefix>` (binaire), `<prefix>.names`, `<prefix>.hdr`
  ```
  .hdr :   cupath_v2 pid=... t0_real_us=... t0_mono_ns=... driver=13000
  .names : 1 ligne par record KER/CPY/RT, ALIGNED 1:1 avec les records binaires op 0/1/2 :
           KER|CPY|RT seq=... gpu_s=... gpu_e=... dur_ns=... stream=... corr=... t_real_us=... name=...
  ```
- ⚠️ Le **cbid** des appels RT n'est **pas** dans `.names`, il est dans le binaire, champ `aux`
  (offset 48). Mapping cbid généré depuis
  `h2/cupti129-test/include/cupti_runtime_cbid.h` → `/tmp/cbid_map.json` (488 cbids).
  cbids clés : `211 cudaLaunchKernel · 214 _ptsz · 430/431 cudaLaunchKernelExC(_ptsz) ·
  311 cudaGraphLaunch · 131 cudaStreamSynchronize · 132 cudaStreamQuery · 135/137/138 event ·
  147 streamWaitEvent · 165 deviceSync · 41 cudaMemcpyAsync · 284 cudaLaunchHostFunc ·
  10/11 Get/PeeLastError · 4 GetDeviceProperties`.
- Trace principale : `nightrun8` (1 073 795 records : 295 085 KER / 4 554 CPY / 774 155 RT /
  1 RESET), bracket dans `nightrun8/traces/cupath8_bracket`.
  ⚠️ `cupath8_gen.json` est pollué (1 ligne de trace avant le JSON) — parser en cherchant le
  premier `{`.

### Config de nightrun8 (à connaître avant de citer)
`run8_capture.sh` : `GGML_CUDA_VOLTA_Q6_W4R4=1`, shortlist MTP, `LLAMA_ARG_SPEC_MTP_DEFER_PROMPT=1`,
`CUDA_VISIBLE_DEVICES=0`, `LLAMA_ARG_FIT_TARGET=1024`, **`-c 200000 -b 512 -ub 512`** (PAS
`-b 2048`), et **pas** de `GGML_CUDA_VOLTA_Q8_FATTN_TC=0` explicit. Le `.so` est le prod
(`8035caf4`). → nightrun8 est un **proxy steady** (57.64 tok/s ≈ prod), pas strictement
prod-exact b2048. `run9b_capture.sh` (b2048 + FATTN_TC=0, -c 131000) est le script correct mais
**aucune trace `.names` run9b n'a été trouvée** dans `nightrun8/`.

## 3. Qui orchestre les launches GPU

**`llama-server` / backend CUDA GGML** orchestre lui-même, en **mélange** de :
- **launchs unitaires** `cudaLaunchKernel` (≈ 9k/s), et
- **`cudaGraphLaunch`** (CUDA graphs de 3 tailles différentes).

Le decode steady n'est **pas** entièrement en unitaires. Par corrélation CUPTI (correlationId
partagée entre le RT et les KER qu'il déclenche), dans la fenêtre steady :

| mode | count | détail |
|---|---:|---|
| `cudaLaunchKernel` unitaires | 11 233 | corr unique (1 launch → 1 kernel) = **6 731 KER** |
| `cudaGraphLaunch` | **78** | 78 corr distinctes → 78 groupes multi-KER |

Les 78 graphes se répartissent en **3 tailles** :

| graph | nb lancements | KER/graph | KER total | GPU/instance (méd.) | rôle probable |
|---|---:|---:|---:|---:|---|
| **gros** | 14 | **2 229** | 31 206 | **57.4 ms** | **verify** MTP (1 par cycle ~67 ms) |
| moyen | 47 | **44** | 2 068 | **2.1 ms** | **draft** MTP (~2.4 ms de périodicité) |
| petit | 17 | **14** | 238 | 0.15 ms | sampling / petit bloc |

→ **14 × 2229 = 31 206 KER** sont exécutés via `cudaGraphLaunch` (le gros du verify),
**6 731 KER** sont lancés individuellement.

**Streams** : les KER tournent sur `stream 16` (37 893) et `stream 17` (2 350) ; les RT et les
graphLaunchs sont reportés sur `stream 0` (le shim ne passthrough le stream des nodes de graph,
ceci est un artefact de capture, pas 1 stream physique).

## 4. Cadence (fenêtre steady, 1.254 s)

- **32 100 KER/s**, **38 052 RT/s**, **8 960 launch/s** (unitaires).
- task 95 : 64 tok / 1093 ms ; **16 steps MTP** (14 verify + 2) → **~71.8 ms/step wall**,
  verify (graph 2229) ≈ **57 ms GPU/instance**.
- Périodicité des gros graphes (verify) : **médiane 67.7 ms** (min 67.5, max 140.5) → 1 par cycle MTP.
- Périodicité des graphes moyens (draft) : **médiane 2.37 ms**.

## 5. Coût host / API runtime

Fenêtre bracket (1.254 s) — union/somme des durées d'appels RT :

| API | count | /s | tot_us | %wall | p50 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| cudaGetLastError | 17 166 | 13 693 | 811 | 0.1% | 0.04 µs | 0.09 µs |
| cudaGetDevice | 16 531 | 13 186 | 941 | 0.1% | 0.05 µs | 0.12 µs |
| cudaLaunchKernel | 11 233 | 8 960 | 82 910 | 6.6% | 2.36 µs | 128 µs |
| **cudaStreamSynchronize** | 1 701 | 1 357 | **986 433** | **78.7%** | 1.96 µs | 2.16 ms |
| cudaMemcpyAsync | 947 | 755 | 28 874 | 2.3% | 2.82 µs | 303 µs |
| cudaGraphLaunch | 78 | 62 | 38 261 | 3.1% | 56.2 µs | 2.75 ms |

- **HOST-API busy (somme des durées RT) = 1 141.8 ms ≈ 91 % du wall 1.254 s** — mais c'est une
  **somme**, avec chevauchements. L'analyse union sur le span de génération est plus fiable.
- Sur le **span de génération réel** (1er→dernier KER = 1149 ms) :
  - host **dans** les API CUDA (union) = **1108 ms = 96.5 %**, dont **cudaStreamSynchronize
    = 981 ms (1 481 appels)** ;
  - **host non-CUDA (python/sampling/bookkeeping) ≈ 41 ms = 3.5 %**.
- ⚠️ Prudence : les durées d'API **synchrones** (streamSync, memcpy) **incluent du wait GPU**,
  pas du travail CPU actif. Le vrai overhead CPU non-CUDA est petit (~3.5 %).

## 6. Gaps GPU

- **Fenêtre bracket** : GPU busy (union KER+CPY) = 1104.2 / 1253.7 ms = **88.1 %** (idle 11.9 %).
- **Span de génération** (1149 ms) : GPU busy = **1072 / 1149 = 93.3 %**, idle 6.7 % ;
  gaps totaux = 77 ms, dont **gaps >500 µs = 76 → 43 ms** et micro-gaps <500 µs = 34 ms.
- Histogramme des gaps (fenêtre) :
  ```
  <10 µs      n=40 805  total= 26.9 ms   (micro-gaps entre kernels, le bruit de fond)
  10-50 µs    n=   281  total=  4.5 ms
  50-100 µs   n=   133  total=  9.0 ms
  100-500 µs  n=    40  total=  7.4 ms
  500-1000µs  n=     6  total=  5.1 ms
  >=1000 µs   n=    17  total= 31.3 ms
  ```
- **Gaps >500 µs = ~67-70 ms de périodicité** (alignés sur le cycle MTP, 14+ cycles) : ce sont
  les **pauses inter-cycles MTP** (draft→verify + sync + memcpy), dominés par
  `cudaStreamSynchronize` (250-390 µs) + `cudaGraphLaunch` (~25-30 µs) + `cudaMemcpyAsync`.
- **Gros gaps ponctuels** (début de fenêtre) :
  - **~5 ms × 2** : liés à **reconstruction/update/destroy de graph**
    (`cudaGraphDestroy` 432-522 µs + `cudaGraphExecUpdate` 284-306 µs + `cudaLaunchKernel`
    1.7-1.8 ms) — changement de bucket de batch.
  - **~1 ms × plusieurs** : pattern `k_get_rows`/`k_set_rows` + `cudaStreamSynchronize` +
    `cudaMemcpyAsync` (parfois `cudaMallocHost` ~1.2 ms).

## 7. Mix kernels (fenêtre steady, par durée GPU)

| kernel | count | %cnt | tot_us | %time | p50 |
|---|---:|---:|---:|---:|---:|
| `mul_mat_vec_q<type14>` (q6_K, W4R4) | 3 825 | 9.5% | 395 496 | **36.9%** | 128.7 µs |
| `mul_mat_vec_q<type8>` (q8) | 3 416 | 8.5% | 153 100 | 14.3% | 11.9 µs |
| `ggml_q8v14partial_kernel` (FA H2) | 272 | 0.7% | 150 038 | 14.0% | 551.7 µs |
| `mul_mat_vec_q<type13>` (q6_k) | 1 207 | 3.0% | 114 503 | 10.7% | 111.2 µs |
| `mmvq_cu_71` (mapped rows MTP) | 48 | 0.1% | 43 123 | 4.0% | 898.4 µs |
| `quantize_q8_1` | 8 884 | 22.1% | 23 903 | 2.2% | 2.7 µs |
| `ggml_q8v13reduce_kernel` | 272 | 0.7% | 23 234 | 2.2% | 85.3 µs |
| `flash_attn_ext_vec` | 48 | 0.1% | 20 328 | 1.9% | 422.0 µs |
| `mul_mat_vec_q<type14>` (autre) | 144 | 0.4% | 18 621 | 1.7% | 116.4 µs |
| `gated_delta_net_cuda_128x4_volta` | 816 | 2.0% | 17 917 | 1.7% | 22.1 µs |
| `add_rms_norm_mul_f32` | 2 159 | 5.4% | 15 829 | 1.5% | 7.3 µs |
| `cpy_scalar` | 3 584 | 8.9% | 15 325 | 1.4% | 3.8 µs |

Les **KER lancés individuellement** (corr unique, 6 731) sont surtout :
`quantize_q8_1` (1 499), `mul_mat_vec_q type14` (677), `cpy_scalar` (625),
`mul_mat_vec_q type8` (596), `unary_gated_op` (385), `add_rms_norm_mul_f32` (381),
`k_bin_bcast` (291), `rms_norm_scale` (288), `k_get_rows` (149), `concat_dim0_transposed` (144).
→ les opérations **KV/quant/draft** qui ne sont pas capturables en graph restent unitaires.

## 8. Limites / prudence

- **CUPTI ajoute un overhead** (≈ ×1.30 en wall, cf. Step D : 40.85 tok/s sous CUPTI vs
  53.45 sans) — les durées host/sync sont majorées.
- **nightrun8 n'est pas strictement prod-exact** : `-b 512` (pas 2048), pas de
  `GGML_CUDA_VOLTA_Q8_FATTN_TC=0` explicite (H2 peut être ON → q8v14partial visible à 14 %).
  Le steady (57.64 tok/s) est représentatif, mais la config exacte b2048/H2-OFF exige
  `run9b_capture.sh` (pas encore de trace associée).
- Les **p50/p99 de `/tmp/analyze8.txt` brut sont en ns** (oubli de /1000 à l'affichage) ;
  la table de la section 5 est déjà convertie en µs.
- `cupath8_gen.json` pollué (ligne de trace avant le JSON) → parser par premier `{`.
- `run7v2` (secours) n'a **pas de bracket** (RT=1 982 764, KER=698 499).
- Trace Step D (`/tmp/cupath_b2048_070601`) a des CPY mais **0 ligne RT** → insuffisante
  pour l'analyse orchestrateur API.

## 9. Réponse à la question centrale

**Qui orchestre les launches GPU, avec quel coût host, et quels gaps ?**

- **L'orchestrateur est `llama-server` + le backend CUDA GGML** (pas de processus tiers).
- Il combine **CUDA graphs** (verify = 2 229 ker, draft = 44 ker, sampling = 14 ker) et
  **launchs unitaires** (les op KV/quant/draft). Cadence ≈ 32k KER/s, 9k launch unitaires/s.
- **Coût host** : le host passe ~96.5 % du span de génération dans des appels CUDA, mais
  c'est dominé par des **appels bloquants** (`cudaStreamSynchronize` ≈ 981 ms de wait GPU).
  Le vrai travail CPU non-CUDA ≈ **3.5 %**.
- **Gaps** : GPU à ~93 % du span de génération. Les gaps restants (6.7 %) sont des
  micro-gaps (<10 µs, 27 ms) entre kernels + des pauses inter-cycles MTP (~67 ms,
  sync+graphLaunch+memcpy) + 2 rebuilds de graph (~5 ms) + quelques memcpy/rows (~1 ms).
