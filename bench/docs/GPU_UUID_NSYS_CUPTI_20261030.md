# GPU par UUID + nsys/CUPTI 12.9 — note de référence (état 30/10, valide tant que le driver ne change pas)

Date : 2026-10-30 — remplace/complète le verdict « Nsight/CUPTI : CLOS » de `h2/ETAT.md` (09/10).
**Cette note est la source de vérité pour : (a) l'identification des GPU, (b) le profiling CUDA via nsys.**

---

## 1. GPU : identification par UUID UNIQUEMENT (jamais d'index en dur)

### Vérité terrain 30/10 (état de la machine)

| GPU | UUID | ordinal CUDA (`CUDA_VISIBLE_DEVICES`) | index `nvidia-smi` (`--id`) |
|---|---|---:|---:|
| **Tesla V100-PCIE-32GB** — bench + prod `ollama-v100` | `GPU-b3c229d2-2803-0af9-2fb8-067708067c1b` | **0** | **1** |
| GTX 1070 — embed/reranker (services 33695, 11436) | `GPU-60540de1-34b4-cae2-030e-0a9ae61b36f0` | **1** | **0** |

⚠️ **Les deux numérotations sont INVERSÉES** : `nvidia-smi 0` = 1070, mais `CUDA_VISIBLE_DEVICES=0` = V100.
C'est la cause historique des erreurs « mauvaise carte » (crash `no kernel image` quand on confondait).
Les ordinaux CUDA peuvent **bouger au reboot** — d'où l'interdiction d'index en dur.

### Règles

1. **Jamais** `CUDA_VISIBLE_DEVICES=0` / `nvidia-smi --id=1` en dur dans un script bench/profiling.
2. Toujours résoudre par UUID :
   ```bash
   # shell
   CUDA_VISIBLE_DEVICES=$(python3 /home/guile/llama-experiments/bench/gpu_uuid.py --cvt-ordinal 2>/dev/null || echo 0)
   nvidia-smi --id=$(python3 /home/guile/llama-experiments/bench/gpu_uuid.py --smi-index 2>/dev/null || echo 1) ...
   ```
   ```python
   from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index   # bench/
   CVD = str(v100_cuda_ordinal())   # int -> CUDA_VISIBLE_DEVICES
   SMIDX = v100_smi_index()         # str -> nvidia-smi --id=
   ```
3. **Scripts volontairement multi-GPU** (tests 1070) : ne pas les patcher vers le V100.
   Pour cibler la 1070 : « l'unique GPU qui n'est PAS le V100 », résolu par UUID
   (pattern : `h2/test_nsystest.sh`, `h2/cuda_probe.sh`).
4. `bench_direct.py` détecte la V100 par nom via `cuDeviceGetName` (ctypes) — fonctionnel,
   mais c'est la seule brique qui n'utilise pas `gpu_uuid_select` ; unifier si on y touche.

### Helpers

- `bench/gpu_uuid.py` — CLI : `--cvt-ordinal` (int, pour CVD), `--smi-index` (str, pour `--id`), `--json`.
  **Abort (exit 3)** si le V100 n'est pas présent ou la résolution est ambiguë.
- `bench/gpu_uuid_select.py` — API Python non-bloquante (`v100_cuda_ordinal()`, `v100_smi_index()`,
  `v100_info()`), repli loggué (ordinal 0 / smi 1 = état 30/10) si le helper CLI est indisponible.
- `bench/probe_cuda_uuid.py` — sonde de vérification (table UUID↔ordinaux des deux mondes).

Migration effectuée le 30/10 : ~28 scripts `bench/*.py` + ~27 scripts shell (`h2/`, `run/`,
`h2/cupti129-test/**`) patchés, syntaxe vérifiée (`ast.parse` / `bash -n`).

---

## 2. nsys 2025.5.2 sur driver 580.178.04 : CUPTI mort par défaut, VIE avec CUPTI 12.9

### Le problème par défaut (reproduit 30/10, `kapp` sur V100)

nsys 2025.5.2 embarque `libcupti.so.13.1` et la charge par défaut →
`CUPTI_ERROR_INVALID_DEVICE` → **0 événement CUDA** (seuls des OSRT dans le `.rep`) :
`bench/nso_b2048_20261029-*.nsys-rep` (29/10), `/tmp/nsysanity/*`, `/tmp/verifyT1.nsys-rep` (30/10).

### LA solution : `NSYS_CUPTI_LIBRARY_PATH` vers un répertoire CONTENANT UNIQUEMENT une CUPTI 12.9

```bash
NSYS_CUPTI_LIBRARY_PATH=/home/guile/llama-experiments/bench/nsys-cupti129
#   → libcupti.so.12.9 (md5 9bc91feae571aa505069c0640e85c26c, soname libcupti.so.12)
```

La lib est un byte-for-byte de `libcupti.so.12.9` de nsys 2025.5.2 (md5 identique).

### ⚠️ Piège N°1 — c'est pour ça que `h2/ETAT.md` (09/10) avait conclu « 12.9 ne marche pas »

La matrice du 09/10 pointait `NSYS_CUPTI_LIBRARY_PATH` vers un répertoire contenant **plusieurs
versions** de CUPTI. Re-produit le 30/10 sur `kapp` :

| test | `NSYS_CUPTI_LIBRARY_PATH` | CUPTI réellement chargée | résultat |
|---|---|---|---|
| vanilla | *(rien)* | `libcupti.so.13.1` (embarquée) | **0 CUDA events** (SKIPPED) |
| T-B | dir avec **uniquement** 12.9 | `libcupti.so.12.9` | **20 kernels capturés** ✓ |
| T-C | dir avec **toutes les versions** (10.2…13.1) | `libcupti.so.13.1` | **0 CUDA events** |
| T-D | dir avec **uniquement** 11.8 | `libcupti.so.11.8` | **0 CUDA events** |

→ nsys 2025.5.2 **choisit la plus haute version** disponible dans le répertoire (et sa 13.1 embarquée
si rien n'y est). La matrice 09/10 testait donc toujours 13.1 en réalité : le verdict « CLOS »
portait sur la mauvaise variable. **Le répertoire doit contenir EXACTEMENT une seule CUPTI 12.9.**
Le 12.9 de nsys-2025.3.2 (md5 `0fef0da7…`) fonctionne aussi ; celui de nsys-2025.5.2 est retenu
(cohérence avec l'extracteur 2025.5.2).

### ⚠️ Piège N°2 — ne PAS mélanger nsys et le shim `libcupath` (LD_PRELOAD)

Le shim `h2/cupti129-test/libcupath_v2.so` (trace CUPTI 12.9 direct, format v2) et nsys sont
**mutuellement exclusifs** : double inscription CUPTI → les deux échouent (test T4 du 30/10).
Chaque route est complète pour sa question :
- **nsys + CUPTI 12.9** → décomposition kernels/API/graphs, timelines, gaps, gaps hôtes OSRT.
- **shim libcupath** → traces CUPTI « pures » sans nsys (nightrun8, Step C/D) — voir
  `bench/nso_orchestrator_summary_20261029.md`.

### Wrapper recommandé : `bench/nsys_cuda.sh`

```bash
bench/nsys_cuda.sh profile --trace=cuda,osrt --osrt-threshold=2000 -o /tmp/x -f true ./app
bench/nsys_cuda.sh launch --session-new=S --trace=cuda,osrt --cuda-graph-trace=node ./app
#   puis les contrôles par nsys direct : nsys start --session=S / nsys stop --session=S
```

Le wrapper : fixe `NSYS_CUPTI_LIBRARY_PATH` vers `bench/nsys-cupti129/`, résout le V100 par UUID
(si `CUDA_VISIBLE_DEVICES` non déjà défini), loggue une ligne de contrôle.
Pour **sudo** (nsys en root) : `sudo -E` est IGNORÉ ici (« preserving the entire environment is not
supported ») → passer l'env explicitement `sudo env K=V… nsys …` (pattern `nso_orchestrator_b2048.py`).

### CUDA graphs : `--cuda-graph-trace=node`

Sans ce flag, les kernels lancés via `cudaGraphLaunch` ne sont pas exposés en kernels individuels
(ils apparaissent en `CUDA_GRAPH_NODE_EVENTS`/graph trace uniquement). **Toujours l'ajouter pour
llama-server** (verify MTP = gros graphe ~2200 kernels, draft = 44, sampling = 14 — cf.
`nso_orchestrator_summary_20261029.md`).

### Validation d'un `.rep` (à faire systématiquement après capture)

```bash
# 1. la CUPTI chargée doit venir de bench/nsys-cupti129 (sinon re-run avec le wrapper)
strings <f>.nsys-rep | grep -m2 nsys-cupti129
# 2. exporter + vérifier la table kernels
python3 bench/inspect_nsdb.py <f>.nsys-rep     # liste tables + compteurs
#    OU : nsys stats --report cuda_gpu_kern_sum <f>.nsys-rep
# 3. CUPTI_ACTIVITY_KIND_KERNEL non vide (ou CUPTI_ACTIVITY_KIND_GRAPH_TRACE si graphs sans --node)
```

Preuve d'usage réel sur llama-server : `bench/nso_b2048_20261030-015122.nsys-rep` (30/10) —
117 133 events CUPTI, 90 030 kernels, `mul_mat_vec_q<type14>` 36 % du temps GPU,
`flash_attn_ext_mma_volta_small` 13.9 %, `cudaStreamSynchronize` 2254 ms/3578 appels,
90.0 % des kernels via `cudaGraphLaunch` (178 lancements). → **le bench nsys b2048 est exploitable.**

---

## 3. À retenir en 5 lignes

1. **UUID, jamais d'index** : V100 = `GPU-b3c229d2-…-067708067c1b` (CUDA ordinal **0**, smi index **1**
   à l'état 30/10 — c'est inversé, et ça peut bouger). Helpers : `bench/gpu_uuid.py` / `gpu_uuid_select.py`.
2. **nsys tel quel = 0 CUDA** sur driver 580 (CUPTI 13.1 embarquée, Volta non supportée ≥ 2025.4).
3. **`NSYS_CUPTI_LIBRARY_PATH=bench/nsys-cupti129`** (dir avec UNE SEULE CUPTI 12.9) = capture complète.
   Le 09/10 avait « testé 12.9 » dans un dir multi-versions → nsys re-chargeait 13.1 → faux négatif.
4. **`--cuda-graph-trace=node`** pour llama-server (sinon les kernels de graph ne sont pas individuels).
5. **Jamais** `LD_PRELOAD=libcupath*` + nsys ensemble (double inscription CUPTI → les deux échouent).
