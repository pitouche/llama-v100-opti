# DIAGNOSTIC RÉFÉRENCE — « Régression MTP » 45,7 → 15,95 t/s — CLOS (29/10)

> **Conclusion en une ligne :** il n'y a **pas de régression**. Le 15,95 t/s du 28/10 est
> un **artefact de mesure** créé par la combinaison (a) du harnais qui mesure les **64
> premiers tokens de decode après un prompt long et NEUF**, et (b) du flag
> **`DEFER_PROMPT=1`** (passé en prod) qui déplace le **catch-up MTP du prompt** de la
> phase prefill vers exactement cette fenêtre de decode. Le **steady-state réel** du MTP
> sur le `.so` prod actuel est **~54–56 t/s** @114k, **plus rapide** que le 45,7 du 18/10.
>
> **Ne pas rouvrir cette enquête** : chaque variable a été isolée par A/B sur le même
> `.so`, et le mécanisme est confirmé dans le code. Les chiffres ci-dessous sont la preuve.

---

## 1. Le signal initial (l'anomalie à expliquer)

| source | date | MTP | `.so` | shortlist | DEFER_PROMPT | @114k dec | @160k dec |
|---|---|---|---|---|---|---|---|
| `t1_bench` | 18/10 | ON | `d2df07b8` (T3 seul) | non | non | **45,73** | 36,23 |
| `t3_vldg_mtp_bench` (leg A-prod) | 28/10 | ON | `8035caf4` (T3+vldg) | oui | **oui** | **15,95** | 24,49 |

Trois signaux avaient alerté :
1. Chute apparente ~2,9× (45,7 → 15,95) « même type de config ».
2. **Ordre des ctx inversé** : 114k (15,95) **plus lent** que 160k (24,49).
3. **MTP-ON (15,95) < MTP-OFF (22,68)** @114k, alors que le 18/10 le MTP-ON était +135 %.

En réalité, les deux runs diffèrent sur **trois** variables : le `.so`, le shortlist, et
`DEFER_PROMPT`. L'isolation ci-dessous montre que **seul DEFER_PROMPT** explique le delta,
et que ce n'est qu'un artefact de la fenêtre mesurée.

---

## 2. Config exacte (toutes les A/B ci-dessous sont dans cette config)

```
.so  prod : 8035caf4  (figé, aucun swap pendant l'enquête)
config lente prod : -c 167000  -b 2048  -ub 2048  --mmproj <blob>
MTP : --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling
KV : q8_0/q8_0   flash-attn : auto   CVD=0 (V100)   port 8917
env base : GGML_CUDA_VOLTA_Q6_W4R4=1, LLAMA_ARG_FIT_TARGET=1024
env test : GGML_CUDA_VOLTA_Q8_FATTN_TC=0 (H2 off), shortlist 131072,
           LLAMA_ARG_SPEC_MTP_DEFER_PROMPT=1
points : 114 688 et 160 000   mesure : decode t/s (64 gen) + acceptance
```

> Points de mesure clés : @114 688 le prompt a **111 713 tokens NOUVEAUX** (checkpoint
> seulement à 41) ; @160 000 il n'a que **46 179 tokens NOUVEAUX** (checkpoint à 109 707).
> C'est **cette** différence (pas la longueur de contexte) qui pilote le catch-up.

---

## 3. La chaîne d'isolation (A/B, même `.so`, config lente)

| # | test | leg | @114k dec | @160k dec | verdict |
|---|---|---|---|---|---|
| 1 | H2 on/off | A-h2off | 15,70 | 24,51 | |
| | | B-h2on | 15,23 | 23,01 | **H2 innocent** (Δ −3 % / −6 %) |
| 2 | sl+DEFER vs aucun | A-full (sl+defer) | 16,26 | 25,28 | |
| | | B-noflags | **56,40** | **50,31** | l'un des deux est la cause (3,5×) |
| 3 | shortlist seul | B-slonly | **58,12** | 46,65 | **shortlist innocent** (≈ noflags) |
| | DEFER seul | C-deferonly | **14,79** | 26,18 | **DEFER_PROMPT = cause** |
| 4 | DEFER : prompt neuf vs checkpoint | A (111 713 nouveaux) | 14,72 | — | coût ∝ tokens **nouveaux** |
| | | B (4 nouveaux, checkpoint) | **53,85** | — | transitoire, pas steady-state |

Le **draft n'est pas en cause** : l'acceptation est **identique** dans tous les legs
(acc 0,92 / mean len 3,71 @114k, 0,74 / 3,15 @160k), avec ou sans shortlist/DEFER.

**Fichiers sources :** `h2_ab_bench_results.jsonl`, `sl_defer_ab_bench_results.jsonl`,
`sl_vs_defer_bench_results.jsonl`, `defer_steady_bench_results.jsonl`,
`mtp_prof_stepA_results.jsonl`, `t3_vldg_mtp_bench_results.jsonl`.

---

## 4. Le mécanisme (confirmé dans le code)

`DEFER_PROMPT=1` sert à **sortir le catch-up MTP du prompt du chemin critique du
prefill** (meilleur time-to-first-token). Coût : il le **repousse** sur le 1er token de
decode, **synchronement**.

- `tools/server/server-context.cpp` ~L4183–4197 : pendant le batch `all_prompt`, si
  `mtp_defer_prompt`, on appelle `prepare_deferred()` qui **capture les embeddings** au
  lieu de faire passer le head draft-MTP sur le prompt → le catch-up n'est **pas** payé
  dans le prefill.
- `tools/server/server-context.cpp` ~L4433 : dès `slot.stats.n_gen == 1` (1er token généré),
  on appelle `flush_deferred_before_last()` + `flush_deferred()` → **replay de TOUT le
  catch-up MTP différé** (re-`llama_decode` de tous les lots de prompt sauvegardés, cf.
  `common/speculative.cpp` `flush_deferred_count`).
- `common/speculative.cpp` `flush_deferred_count()` : rejoue chaque batch de prompt
  sauvegardé via `process_impl(replay, saved.h_nextn)` — d'où un coût **proportionnel au
  nombre de nouveaux tokens du prompt**, pas à la taille du KV.

**Conséquence sur le harnais :** le harnais mesure les 64 **premiers** tokens de decode
après le prompt. Avec `DEFER_PROMPT`, ces 64 tokens **contiennent** tout le catch-up
(~3,2 s pour 111 713 tokens nouveaux). Sans `DEFER_PROMPT`, ce catch-up s'était payé dans
le prefill, et la 1re fenêtre de decode était propre.

---

## 5. Preuve arithmétique (les chiffres s'emboîtent)

**Le gap (DEFER − sans-DEFER) est proportionnel aux tokens NOUVEAUX du prompt :**

| ctx | gap eval (ms) | tokens nouveaux | µs / nouveau token |
|---|---|---|---|
| 114 688 | 4346,5 − 1134,8 = **3 212** | 111 713 | **28,7** |
| 160 000 | 2638,2 − 1272,2 = **1 366** | 46 179 | **29,6** |

≈ **28–30 µs par nouveau token** (= ~51–52 ms par batch de 2048). C'est le replay du
**module MTP** (petit), pas du modèle 27B. La proportionnalité explique aussi
l'**inversion des ctx** : 114k a 2,4× plus de nouveaux tokens que 160k → catch-up 2,4×
plus gros → 114k « plus lent » que 160k, alors que le KV de 160k est plus long.

**Le temps TOTAL de requête est identique** (le catch-up est déplacé, pas supprimé) :

| | sans DEFER (B-noflags) | avec DEFER (defer_steady A) |
|---|---|---|
| prompt window (ms) | 183 468 | 179 750 |
| decode window (ms) | 1 135 | 4 346 |
| **total (ms)** | **184 603** | **184 097** |

→ sans DEFER, le prefill est +3 718 ms (catch-up dedans) ; avec DEFER, le decode-1re
fenêtre est +3 212 ms (catch-up dedans). Le total est **≈ équivalent**. `DEFER_PROMPT`
trade du **TTFT** (prefill plus court → 1er token plus tôt) contre une 1re fenêtre de
decode « sale ». En usage interactif c'est généralement un **gain** ; c'est juste une
mauvaise métrique de bench.

**Steady-state réel** (`defer_steady` REQ B, même prompt checkpointé, 4 tokens nouveaux,
MTP bien engagé : acc 0,83 / 65 drafts) : **53,85 t/s** @114k.

---

## 6. Ce que ça signifie (recommandations)

1. **Pas de régression binaire.** Le `.so` prod `8035caf4` est sain. Steady-state MTP-ON
   ≈ 54–56 t/s @114k, soit **+140 %** vs MTP-OFF (22,92) sur le même `.so`. Le 45,7 du
   18/10 était mesuré **sans** `DEFER_PROMPT` (1re fenêtre propre) et sur un `.so` plus
   ancien → **non comparable** directement au 15,95 contaminé.
2. **`DEFER_PROMPT=1` est bon pour la prod** (TTFT) mais **contamine toute métrique de
   « 1re fenêtre de decode » après un prompt neuf long**. Ne pas l'interpréter comme un
   débit.
3. **Le shortlist et H2 sont innocents** — ne pas les toucher.
4. **Le harnais doit être corrigé** (faite) pour ne plus confondre :
   - le **coût initial de catch-up** (transitoire, ∝ nouveaux tokens) ;
   - le **débit steady-state** après catch-up ;
   - le **temps total** de la requête.
   Voir §7 (correction) — les nouveaux champs de mesure : `catchup_ms`, `steady_tps`,
   `total_ms`, et une 2e requête sur le même prompt (checkpoint) qui isole le steady-state.

---

## 7. État système final (sain, rien à restaurer)

- `ollama-v100` **active** · `.so` prod **`8035caf4`** inchangé (aucun swap) ·
  V100 ~29 386 MiB (runner prod) · GTX 1070 ~3 947 MiB · lock superviseur **libre** ·
  dernier `rc = 0` (Step D CUPTI 29/10 07:10, service restauré).

---

## 8. Step C — OÙ PASSE LE TEMPS MTP (steady-state, harnais corrigé) — 29/10 03:32

Run `mtp_prof_stepC.py` (config lente prod exacte, 114 688, W1 = 64 gen prompt neuf,
W2 = 128 gen même prompt checkpointé, `rc=0`, service restauré, `.so` `8035caf4` figé).

| leg | W1 t/s (contaminée) | catchup | W2 steady t/s | W2 drafts | acc rate | mean len | t_cycle | t_draft | t_verify+ |
|---|---|---|---|---|---|---|---|---|---|
| **MTP-ON** | 15.87 | 2 835 ms | **53.45** | 37 | 0.9459 | 3.432 | 64.7 ms | **7.84 ms** | **56.9 ms** |
| **MTP-OFF** | 22.92 | ≈ 0 | **22.78** | — | — | — | — (1 tok/cycle) | — | 43.9 ms/tok |

**Décomposition d'un cycle MTP steady** (n_max = 3, 1 cycle = 1 draft + 1 verify rows=1..4) :

```
t_cycle  = 64.7 ms   (128 tok / 37 drafts, W2)
├─ t_draft    =  7.84 ms   (12 % — génération du draft, mesuré par le serveur : t_draft_us)
└─ t_verify+  = 56.9 ms   (88 % — verify 27B sur 4 rows + sampling + gaps hôtes)
```

Cross-check GPU (nvidia-smi 10 Hz sur la fenêtre W2) :

| fenêtre | util GPU médiane | min | power médiane |
|---|---|---|---|
| W2 MTP-ON (steady) | **96 %** | 0 | 223 W |
| W2 MTP-OFF (steady) | **99 %** | 52 | 221 W |
| W1 MTP-ON (catch-up) | 23 % | 0 | 64 W |

Le GPU est quasi saturé en steady (pas de throttle power) ; le t_verify+ de 56.9 ms
est donc dominé par le **travail réel du verify 27B sur 4 rows de contexte 114k**,
avec ~12 % de gaps hôtes (kernel médiane 3.9 µs → launch overhead sur ~2 400
kernels/cycle, cf. trace CUPTI §9). Le draft (module MTP petit) ne coûte que 12 %
du cycle : **l'optimisation MTP porte sur le verify, pas sur le draft**.

Note : `acc_rate`/`mean_acc_len` du jsonl initial étaient stalés (2 bugs harnais :
ratios calculés sur les cumuls, et snapshot relisant le log entre les legs en append).
Valeurs correctes ci-dessus = recalcul sur les deltas ; bugs corrigés dans
`night_bench.py` (`_spec_delta` ratios sur delta ; `_spec_snapshot(start, upto)`).

**Fichiers :** `mtp_prof_stepC_{results.jsonl,summary.md,server.log,gputimes.tsv}`.

---

## 9. Référentiel CUPTI (trace existante du 28/10 22:02) — à lire avec précaution

`/tmp/cupath_llama_220241.names` (89 Mo, 705 265 records KER/CPY) — MTP ON, config
**rapide** (`-b 512`, pas de mmproj), prompt **107 347 tokens déjà checkpointés**
(4 nouveaux → catch-up quasi nul) : decode 44.55 t/s, 36 drafts, mean acc len 3.58.

Cluster decode (analyse `analyse_cupath_decode3.py`, unités ns→ms) : 87 625 KER,
somme GPU-kernels 2 896 ms, kernel médiane 3.9 µs, p90 87.8 µs, max 1 773 µs.

Top familles (part du temps GPU-kernel decode) :

| % | famille | lecture |
|---|---|---|
| 29.9 | `ggml_q8v14partial` (592) | GEMV q8 — **verify** (rows 1..4) |
| 29.8 | `mul_mat_vec_q` ggml_type14 (8 326) | GEMV q6-k — **verify** (rows 1..4) |
| 11.5 | `mul_mat_vec_q` ggml_type8 (7 437) | GEMV q8 — **verify** |
| 8.6 | `mul_mat_vec_q` ggml_type13 (2 627) | GEMV q6_k — **verify** |
| 3.3 | `mul_mat_vec_q_mapped_rows` type8 (105) | **draft** (shortlist mapped rows) |
| 2.1 | `flash_attn_ext_mma_volta_small` (105) | attn **draft** (131k ctx) |
| 1.8 | `quantize_q8_1` (19 343) | KV q8_0 (verify + draft) |

Le draft ≈ **5.5 %** du temps GPU-kernel (mapped_rows + attn draft), le verify 27B
≈ **80 %+** — cohérent avec la décomposition serveur (t_draft 12 % du cycle wall,
le verify+ incluant les gaps hôtes et le sampling qui ne sont pas du temps GPU-kernel).

⚠️ Trace **b512 / non-mmproj / 4 tok nouveaux** : représentative des *familles* de
kernels et de l'attribution draft/verify, **pas** des durées wall de la config prod
b2048 (64.7 ms/cycle). Une trace CUPTI propre en b2048 reste à faire si on creuse
le verify (voir §10).

**Fichiers :** `h2/cupti129-test/{run_cupath_llama.sh,libcupath.so,analyse_cupath_decode3.py}`,
`/tmp/cupath_{llama_220241,gen.json,server.out}`.

---

## 10. Step D — CUPTI b2048 (trace FRAÎCHE, config prod exacte) — 29/10 07:10

Run `cupti_mtp_stepD_b2048.py 128` (suite directe Step C). Même séquence : warmup →
REQ A (prompt neuf 114k + 64 gen, catch-up DEFER payé hors fenêtre) → REQ B (même
prompt checkpointé, 128 gen = **STEADY**, c'est la fenêtre tracée). Config prod exacte
: `-b 2048 -ub 2048`, `--mmproj`, shortlist 131072, DEFER_PROMPT=1, `.so` 8035caf4,
KV q8_0, FA auto, c167000, CVD=0. Seul ajout : shim `libcupath.so`.

**`rc=0`, service restauré, `.so` figé.**

### Timings (REQ B steady, sous CUPTI)

| | value | note |
|---|---|---|
| wall REQ B | 3 108.65 ms / 128 tok | 24.48 ms/tok → **40.85 t/s** |
| (sans CUPTI, Step C) | 2 394.8 ms / 128 tok | 18.71 ms/tok → **53.45 t/s** |
| surcoût CUPTI | ×1.297 | 45 % de gaps hôtes (vs 4 % sans) |

Le CUPTI ralentit le run (overhead de flush des records kernel) mais **le mix de
kernels est identique** (même modèle, même config) → l'attribution par famille est
valide même si les durées absolues diffèrent.

### Attribution du temps GPU-kernel (REQ B steady, b2048)

90 030 KER, 2 109 CPY | somme GPU-kernels 2 994.6 ms | kernel médiane 3.9 µs

| famille | ms | count | % GPU-kernel | rôle |
|---|---|---|---|---|
| `ggml_q8v14partial` | 901.9 | 592 | **30.12 %** | GEMV q8 — **verify** (rows 1..4) |
| `mul_mat_vec_q` type14 (q6-k) | 864.3 | 8 325 | **28.86 %** | GEMV q6-k — **verify** |
| `mul_mat_vec_q` type8 (q8) | 336.1 | 7 436 | **11.22 %** | GEMV q8 — **verify** |
| `mul_mat_vec_q` type13 (q6_k) | 249.0 | 2 627 | **8.31 %** | GEMV q6_k — **verify** |
| `mul_mat_vec_q_mapped_rows` type8 | 97.9 | 109 | **3.27 %** | **draft** (shortlist mapped) |
| `flash_attn_ext_mma_volta_small` (n=1) | 65.2 | 109 | **2.18 %** | attn **draft** (131k ctx) |
| `quantize_q8_1` | 54.1 | 19 872 | 1.81 % | KV q8_0 (verify + draft) |
| `ggml_q8v13reduce` | 51.1 | 592 | 1.71 % | reduce q8 — verify |
| `mul_mat_vec_q` type14 (n=1) | 43.1 | 327 | 1.44 % | GEMV q6-k — **verify** |
| `gated_delta_net_cuda_128x4_volta` | 39.4 | 1 824 | 1.31 % | linear attention — verify |
| autres | ~446 | — | ~14.9 % | rms/norm/copy/sampling/etc. |

### Attribution agrégée draft vs verify

| catégorie | ms | % GPU-kernel |
|---|---|---|
| **verify GEMV** (q8v14 + mul_mat_vec_q type8/13/14) | 2 506.9 | **83.71 %** |
| other (norms, copies, sampling, KV quant, linear-attn) | 317.7 | 10.61 % |
| **draft** (mapped_rows + draft_attn) | 163.2 | **5.45 %** |
| verify_attn (flash_attn n=2) | 7.0 | 0.23 % |

### Conclusion Step D — où passe le temps MTP (définitif)

**Le verify 27B domine à 88 % du wall-clock (Step C) et 84 % du temps GPU-kernel
(Step D CUPTI).** Le draft MTP ne représente que ~5–12 % du cycle. L'attention est
négligeable (< 3 %) : le goulot est le **GEMV mémoire-bande passante** (lecture des
poids 27B pour 4 rows × 114k context). C'est le même bottleneck que le MTP-OFF
(22.78 t/s, 43.9 ms/tok) — le MTP accélère par le fait que le verify traite 4 rows
simultanément (1 réel + 3 draft) au lieu de 1, mais le coût par row est identique.

**Implications pour l'optimisation :**
1. Optimiser le draft MTP → gain max ~5 % du cycle. Pas prioritaire.
2. Optimiser le verify 27B → c'est **tout**. Le GEMV q8/q6-k est le target.
   - Réduire le nombre de rows verifyées (n_max 3 → 2) : −25 % de verify time,
     mais −~0.43 tok/cycle (mean acc len 3.43 → 2.43 estimé) → débit ≈ identique.
   - Améliorer le GEMV kernel (bande passante) → gain proportionnel.
   - Attention : pas de target ici (déjà < 3 %).
3. Les gaps hôtes (~12 % sans CUPTI, kernel médiane 3.9 µs → ~2 400 launches/cycle)
   sont secondaires mais non négligeables : graph capture ou kernel fusion pourraient
   réduire ce surcoût.

**Fichiers :** `/tmp/cupath_b2048_{070601.names,server.out,bracket,gen.json,gputimes.tsv}`,
`bench/cupti_mtp_stepD_b2048.py`.

---

## 11. Étapes suivantes du profiling (si on creuse)

1. ~~**Verify 27B rows=1..4** : trace CUPTI b2048~~ — **fait (Step D, §10)** :
   verify GEMV = 83.7 % du temps GPU-kernel, attention < 3 %, draft 5.5 %.
2. **Gaps hôtes ~12 %** (sans CUPTI) : kernel médiane 3.9 µs → ~2 400 launches/cycle ;
   vérifier si le graph capture réduit les gaps (la trace CUPTI b2048 n'avait pas de
   graph — les 45 % de gaps sous CUPTI viennent du flush des records).
3. **Cible GEMV verify** (§10) : q8v14partial (30 %) + mul_mat_vec_q q6-k (29 %) +
   type8 (11 %) + type13 (8 %) = ~78 % du GPU-kernel. Bande passante mémoire : c'est
   la lecture des poids 27B qui compte, pas les rows.
4. **Catch-up DEFER (~28.6 µs/tok nouveau)** : déjà caractérisé ; à mesurer seulement
   si on change le comportement de `DEFER_PROMPT`.

---

## 12. Artefacts / fichiers

```
/home/guile/llama-experiments/bench/mtp_defer_diagnostic_20261029.md   # cette note
bench/{h2_ab,sl_defer_ab,sl_vs_defer,defer_steady,mtp_prof_stepA,t3_vldg_mtp,mtp_prof_stepC}_*_results.jsonl
bench/{h2_ab,sl_defer_ab,sl_vs_defer,defer_steady,mtp_prof_stepC}_*_summary.md
bench/night_bench.py                                                   # harnais CORRIGÉ (run_point2 + _spec_delta/snapshot)
bench/mtp_prof_stepC.py                                                # bench Step C (W1/W2 + décomposition)
bench/cupti_mtp_stepD_b2048.py                                         # bench Step D (CUPTI b2048, steady)
/tmp/cupath_b2048_{070601.names,server.out,bracket,gen.json,gputimes.tsv}  # trace Step D
src/llama-volta-merged/tools/server/server-context.cpp  (~L4183, ~L4433)
src/llama-volta-merged/common/speculative.cpp           (flush_deferred_count ~L1715)
```
