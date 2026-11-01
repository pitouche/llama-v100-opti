# STEP D — CUPTI b2048 (trace fraîche, config prod exacte) — 29/10 07:10

date : 2026-10-29 07:06–07:10 UTC · `rc=0` · service restauré · `.so` prod `8035caf4` figé
config : `-c 167000 -b 2048 -ub 2048 --mmproj`, KV q8_0, FA auto, shortlist 131072,
DEFER_PROMPT=1, MTP n_max=3, CVD=0 · seul ajout : shim `libcupath.so`
séquence : warmup → REQ A (prompt neuf 114k + 64 gen, catch-up DEFER hors fenêtre)
→ REQ B (même prompt checkpointé, 128 gen = **STEADY**, fenêtre tracée)

## Timings REQ B (steady, sous CUPTI)

| | value |
|---|---|
| wall REQ B | 3 108.65 ms / 128 tok (24.48 ms/tok → 40.85 t/s) |
| sans CUPTI (Step C) | 2 394.8 ms / 128 tok (18.71 ms/tok → 53.45 t/s) |
| surcoût CUPTI | ×1.30 (overhead flush records ; mix de kernels inchangé) |
| drafts REQ B | 37 générés / 35 acceptés (acc 0.9459, mean len 3.43) |

## Attribution du temps GPU-kernel (90 030 KER, 2 994.6 ms)

| famille | ms | % | rôle |
|---|---|---|---|
| ggml_q8v14partial | 901.9 | 30.1 | verify GEMV q8 |
| mul_mat_vec_q type14 (q6-k) | 864.3 | 28.9 | verify GEMV |
| mul_mat_vec_q type8 (q8) | 336.1 | 11.2 | verify GEMV |
| mul_mat_vec_q type13 (q6_k) | 249.0 | 8.3 | verify GEMV |
| mul_mat_vec_q_mapped_rows (shortlist) | 97.9 | 3.3 | **draft** |
| flash_attn_ext_mma_volta_small (draft) | 65.2 | 2.2 | **draft** attn |
| quantize_q8_1 (KV) | 54.1 | 1.8 | verify+draft |
| ggml_q8v13reduce | 51.1 | 1.7 | verify |
| autres (norms/copy/sampling/lin-attn) | ~879 | ~29 | verify+overhead |

**Agrégé : verify GEMV = 83.7 % · draft = 5.5 % · verify attn = 0.2 % · other = 10.6 %**

## Conclusion

Le **verify 27B (GEMV bande passante)** est le goulot : 84 % du GPU-kernel, 88 % du
wall (Step C). Le draft MTP = 5.5 % → l'optimisation MTP doit viser le GEMV verify
(~78 % : q8v14 + q6-k + q8 + q6_k), pas le draft, pas l'attention.

artefacts : /tmp/cupath_b2048_{070601.names,070601,server.out,bracket,gen.json,gputimes.tsv}
analyse : `h2/cupti129-test/analyse_cupath_decode3.py /tmp/cupath_b2048_070601.names`
